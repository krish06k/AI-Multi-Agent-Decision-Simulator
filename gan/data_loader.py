"""
gan/data_loader.py
==================
Layer 1 — GAN Data Engine | Data Ingestion & Preprocessing

Fetches real OHLCV market data from Yahoo Finance, cleans and normalises it,
converts it into sliding-window sequences, and persists tensors for GAN training.

Flow:
    fetch_ohlcv() → clean() → add_features() → normalise() → to_sequences() → save()

Changelog v3 (50-company, pro-grade)
--------------------------------------
1.  Removed all debug print() statements.
2.  Removed unused `from matplotlib import ticker` import.
3.  yfinance MultiIndex column fix — flatten before selecting OHLCV.
4.  add_features() — full 12-feature pro set:
        returns, log_returns, volatility (10-bar), hl_range, oc_diff,
        volume_log (log1p), volume_z (z-scored), ema_ratio.
    Key changes vs v2:
      - volume_change removed — pct_change is noisy and explodes on zero-volume days.
      - volume_log (log1p) added — compresses the 7-decade range of equity volume.
      - volume_z (cross-ticker z-score) added — makes volume comparable across
        small-caps and mega-caps in the 50-company universe.
      - All return/vol columns clipped at ±5σ BEFORE normalisation so a single
        crash day (e.g. -20% Flash Crash) cannot dominate the scaler's range.
      - ema_ratio clipped at ±0.5 — prevents unstable readings in thin markets.
5.  normalise() — "smart" method is now the default.
      Price-level cols (Open/High/Low/Close) → MinMaxScaler [0,1].
      Volume → already log1p compressed, then MinMax.
      Return/signal cols → StandardScaler (z-score, fat tails preserved).
6.  per_ticker_normalise flag on build_dataset() — each ticker gets its own
    scaler so SPY (range $50–500) and a $2 penny stock don't share a scaler.
7.  to_sequences() — stride parameter exposed for sparse sampling on large corpora.
8.  validate_tensor() — NaN / Inf / shape checks with detailed stats.
9.  split() — temporal train/val/test split (no shuffle).
10. S&P-500 ticker lists (SP50_TICKERS, SP100_TICKERS) added as convenience
    constants for the 50-company multi-ticker training run.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_CFG_PATH = Path(__file__).resolve().parents[1] / "config" / "gan_config.yaml"


def _load_cfg() -> dict:
    with open(_CFG_PATH, "r") as f:
        return yaml.safe_load(f)


CFG = _load_cfg()

# ---------------------------------------------------------------------------
# Column constants
# ---------------------------------------------------------------------------
OHLCV_COLS: list[str] = ["Open", "High", "Low", "Close", "Volume"]

# ---------------------------------------------------------------------------
# Full 13-feature pro set (v3).
#
# Normalisation strategy (the "smart" method):
#   _PRICE_LEVEL_COLS  → MinMaxScaler [0, 1]  — bounded, always positive
#   _VOLUME_COLS       → MinMaxScaler [0, 1]  — already log1p-compressed
#   _RETURN_COLS       → StandardScaler       — zero-centred, fat tails preserved
#
# Why this matters for a 50-company corpus
# -----------------------------------------
# SPY trades ~80M shares/day. A micro-cap may trade 50k.
# Raw Volume with MinMax across 50 tickers collapses micro-cap volume to ~0.
# volume_log (log1p) compresses the 7-decade range.
# volume_z (per-ticker z-score) makes intraday spikes comparable across all names.
# Together they let momentum agents and panic agents read volume correctly
# across the full universe.
# ---------------------------------------------------------------------------
FEATURE_COLS: list[str] = [
    # ── Price levels (MinMax scaled) ──────────────────────────────────────
    "Open", "High", "Low", "Close",
    # ── Volume (log-compressed then MinMax) ───────────────────────────────
    "volume_log",       # log1p(Volume) — compresses 7-decade range
    "volume_z",         # per-ticker z-score of Volume — cross-ticker comparable
    # ── Returns / movement (Z-score, clipped ±5σ before scaling) ─────────
    "returns",          # pct_change(Close) — tick-to-tick % move
    "log_returns",      # log(Close/Close_prev) — primary GAN signal
    "volatility",       # 10-bar rolling std of returns — regime signal
    # ── Intraday structure (Z-score, clipped) ─────────────────────────────
    "hl_range",         # (High-Low)/Close — normalised intraday range
    "oc_diff",          # (Close-Open)/Close — candle body direction
    # ── Trend context (Z-score, clipped ±0.5) ─────────────────────────────
    "ema_ratio",        # Close/EMA(20)-1 — price position vs medium trend
]

_PRICE_LEVEL_COLS = {"Open", "High", "Low", "Close"}
_VOLUME_COLS      = {"volume_log", "volume_z"}
_RETURN_COLS      = {
    "returns", "log_returns", "volatility",
    "hl_range", "oc_diff", "ema_ratio",
}

# Clipping bounds applied BEFORE normalisation (prevents scaler domination)
_CLIP_BOUNDS: dict[str, tuple[float, float]] = {
    "returns":     (-0.20, 0.20),    # ±20% — covers even circuit-breaker days
    "log_returns": (-0.22, 0.22),    # log-scale equivalent
    "volatility":  (0.0,   0.15),    # 0 → 15% annualised daily vol
    "hl_range":    (0.0,   0.25),    # 0 → 25% intraday range
    "oc_diff":     (-0.15, 0.15),    # ±15% open-to-close
    "ema_ratio":   (-0.50, 0.50),    # ±50% deviation from EMA(20)
    "volume_z":    (-4.0,  4.0),     # ±4σ volume spike
}

NormMethod = Literal["minmax", "zscore", "smart"]

# ---------------------------------------------------------------------------
# Convenience ticker lists for multi-company training
# ---------------------------------------------------------------------------

# 50 large-cap US equities covering all 11 GICS sectors — good regime diversity
SP50_TICKERS: list[str] = [
    # Technology
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "CRM", "AMD", "INTC",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "USB", "PNC", "COF",
    # Healthcare
    "UNH", "JNJ", "LLY", "ABBV", "PFE", "MRK", "TMO", "DHR", "AMGN", "GILD",
    # Consumer
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT", "COST", "WMT", "LOW",
    # Industrials / Energy / Utilities / Materials / Real Estate
    "CAT", "BA", "XOM", "CVX", "NEE", "DUK", "LIN", "APD", "PLD", "AMT",
]

# Broader 100-company universe for later experimentation
SP100_TICKERS: list[str] = SP50_TICKERS + [
    "V", "MA", "PYPL", "SQ", "ADBE", "NOW", "SNPS", "KLAC", "LRCX", "MU",
    "QCOM", "TXN", "AMAT", "MRVL", "NXPI", "STX", "WDC", "HPQ", "DELL", "CSCO",
    "GE", "HON", "LMT", "RTX", "NOC", "GD", "MMM", "EMR", "ITW", "PH",
    "OXY", "PSX", "VLO", "MPC", "HAL", "SLB", "BKR", "COP", "EOG", "PXD",
    "T", "VZ", "CMCSA", "DIS", "NFLX", "PARA", "WBD", "FOX", "FOXA", "CHTR",
]
SP500_TOP_50 = SP50_TICKERS

# ---------------------------------------------------------------------------
# 1. Fetch
# ---------------------------------------------------------------------------

def fetch_ohlcv(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance.

    Parameters
    ----------
    ticker   : Yahoo Finance ticker symbol, e.g. "SPY" or "AAPL".
    start    : ISO date string "YYYY-MM-DD".
    end      : ISO date string "YYYY-MM-DD".
    interval : Bar size — "1d", "1h", "5m", etc.

    Returns
    -------
    pd.DataFrame with columns [Open, High, Low, Close, Volume].

    Raises
    ------
    ValueError if download returns empty DataFrame or required columns missing.
    """
    logger.info("Fetching %s  %s → %s  [interval=%s]", ticker, start, end, interval)
    df: pd.DataFrame = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise ValueError(
            f"yfinance returned no data for ticker='{ticker}' "
            f"between {start} and {end}."
        )

    # FIX: yfinance ≥0.2 returns MultiIndex columns e.g. ("Close", "SPY").
    # Flatten to plain string column names before selecting.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    missing = [c for c in OHLCV_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"yfinance response missing expected columns: {missing}. "
            f"Got: {list(df.columns)}"
        )

    df = df[OHLCV_COLS].copy()
    logger.info("Downloaded %d rows for %s", len(df), ticker)
    return df


# ---------------------------------------------------------------------------
# 2. Clean
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitise a raw OHLCV DataFrame.

    Steps
    -----
    1. Validate all OHLCV columns are present.
    2. Drop rows where ALL values are NaN.
    3. Forward-fill remaining gaps (≤5 consecutive bars).
    4. Drop residual NaN rows.
    5. Price sanity: Open > 0, High ≥ Low.
    6. Remove duplicate index entries.
    7. Sort chronologically.
    """
    missing = [c for c in OHLCV_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    original_len = len(df)
    df = df.dropna(how="all")
    df = df.ffill(limit=5)
    df = df.dropna()

    # Price sanity — handle both Series and DataFrame comparisons safely
    bad_open  = df["Open"] <= 0
    bad_range = df["High"] < df["Low"]
    invalid   = bad_open | bad_range
    if isinstance(invalid, pd.DataFrame):
        invalid = invalid.any(axis=1)
    n_bad = int(invalid.sum())
    if n_bad:
        logger.warning("Dropping %d rows with invalid OHLC relationships.", n_bad)
        df = df[~invalid]

    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    logger.info(
        "Cleaned: %d → %d rows (dropped %d).",
        original_len, len(df), original_len - len(df),
    )
    return df


# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the full 13-feature pro set from a cleaned OHLCV DataFrame.

    Feature map (v3)
    ----------------
    volume_log    : log1p(Volume)
                    Compresses the 7-decade dynamic range of equity volume.
                    SPY's 80M/day and a micro-cap's 50k/day both become
                    meaningful numbers instead of one dominating the scaler.

    volume_z      : (Volume - mean) / std  (per-ticker)
                    Makes volume spikes comparable across all 50 tickers.
                    A 3σ volume spike means the same thing for AAPL and a $3 stock.

    returns       : Close.pct_change()
                    Tick-to-tick percentage move.  Clipped ±20%.

    log_returns   : log(Close / Close_prev)
                    The GAN's primary learning signal. Stationary, roughly
                    normal with fat tails. Clipped ±22%.

    volatility    : returns.rolling(10).std()
                    10-bar realised volatility.  The regime signal: calm
                    periods read ~0.005, stressed periods read ~0.02+.
                    Clipped [0, 0.15].

    hl_range      : (High - Low) / Close
                    Normalised intraday trading range. Correlated with vol
                    but captures intraday structure independently.
                    Clipped [0, 0.25].

    oc_diff       : (Close - Open) / Close
                    Candle body direction and size. +ve = bullish bar.
                    Clipped ±15%.

    ema_ratio     : Close / EMA(20) - 1
                    Price position relative to the 20-bar exponential MA.
                    Regime signal: extended > +5% = overbought,
                    < -5% = oversold. Clipped ±50%.

    Clipping note
    -------------
    All return/vol features are clipped BEFORE normalisation.
    This prevents a single extreme event (Flash Crash, COVID crash) from
    dominating the StandardScaler's std estimate and compressing all
    normal-regime data into a tiny range. The GAN then learns the typical
    market dynamics rather than being optimised around rare outliers.

    Parameters
    ----------
    df : Cleaned OHLCV DataFrame (output of clean()).

    Returns
    -------
    DataFrame with 12 columns: 4 price + 2 volume + 6 return/signal features.
    Columns align exactly with FEATURE_COLS.
    """
    df = df.copy()
    close  = df["Close"]
    volume = df["Volume"]

    # ── Volume features ────────────────────────────────────────────────────
    # log1p compresses range without losing zero-volume information
    df["volume_log"] = np.log1p(volume).astype(np.float32)

    # Per-ticker z-score — each ticker normalised independently
    vol_mean = volume.mean()
    vol_std  = volume.std() + 1e-8   # guard against zero-std tickers
    df["volume_z"] = ((volume - vol_mean) / vol_std).astype(np.float32)

    # ── Return features ────────────────────────────────────────────────────
    df["returns"]     = close.pct_change().fillna(0).astype(np.float32)
    df["log_returns"] = np.log(close / close.shift(1)).fillna(0).astype(np.float32)

    # ── Volatility — regime signal ─────────────────────────────────────────
    df["volatility"] = (
        df["returns"].rolling(10, min_periods=1).std().fillna(0).astype(np.float32)
    )

    # ── Intraday structure ─────────────────────────────────────────────────
    df["hl_range"] = ((df["High"] - df["Low"]) / close).astype(np.float32)
    df["oc_diff"]  = ((close - df["Open"]) / close).astype(np.float32)

    # ── Trend context ──────────────────────────────────────────────────────
    ema20 = close.ewm(span=20, adjust=False).mean()
    df["ema_ratio"] = ((close / ema20) - 1).astype(np.float32)

    # ── Apply clipping BEFORE normalisation ───────────────────────────────
    # This is the critical step: prevents extreme events from compressing
    # the scaler's range and killing the GAN's ability to learn normal regimes.
    for col, (lo, hi) in _CLIP_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi).astype(np.float32)

    # ── Final NaN sweep ────────────────────────────────────────────────────
    # Forward-fill then zero-fill any residual NaNs from rolling windows
    return_feature_list = list(_RETURN_COLS) + list(_VOLUME_COLS)
    existing = [c for c in return_feature_list if c in df.columns]
    df[existing] = df[existing].ffill(limit=5).fillna(0)

    logger.info(
        "add_features() | shape %s | new cols: %s",
        df.shape,
        [c for c in FEATURE_COLS if c not in ("Open", "High", "Low", "Close")],
    )
    return df


# ---------------------------------------------------------------------------
# 4. Normalise
# ---------------------------------------------------------------------------

def normalise(
    df: pd.DataFrame,
    method: NormMethod = "smart",
    feature_cols: list[str] | None = None,
) -> Tuple[np.ndarray, object]:
    """
    Normalise an OHLCV(+features) DataFrame column-wise.

    Parameters
    ----------
    df           : DataFrame containing the columns in feature_cols.
    method       : Normalisation strategy:

                   "minmax" — every column to [0, 1].
                              Simple but compresses tail events on return cols.

                   "zscore" — every column zero-mean unit-variance.
                              Better than minmax for returns, but z-scoring raw
                              price levels leaks the global price trend into
                              every sequence (bad for regime conditioning).

                   "smart"  — RECOMMENDED FOR FINANCE.
                              Price cols (Open/High/Low/Close) → MinMaxScaler.
                              Volume cols (volume_log, volume_z) → MinMaxScaler.
                              Return/vol/signal cols → StandardScaler.
                              Combines the best of both: price levels are bounded,
                              return distributions are zero-centred with fat tails
                              visible (not compressed by a [0,1] cap).

    feature_cols : Columns to include. Defaults to FEATURE_COLS if all present,
                   else falls back to OHLCV_COLS.

    Returns
    -------
    arr    : np.ndarray float32, shape (N, len(feature_cols)).
    scaler : "minmax"/"zscore" → single sklearn scaler.
             "smart" → dict:
               {"price":        MinMaxScaler (for Open/High/Low/Close),
                "volume":       MinMaxScaler (for volume_log, volume_z),
                "return":       StandardScaler (for all signal cols),
                "feature_cols": list[str],
                "price_mask":   bool array len(feature_cols),
                "volume_mask":  bool array,
                "return_mask":  bool array}
    """
    if feature_cols is None:
        feature_cols = (
            FEATURE_COLS
            if all(c in df.columns for c in FEATURE_COLS)
            else OHLCV_COLS
        )

    arr = df[feature_cols].values.astype(np.float32)

    if method == "minmax":
        scaler = MinMaxScaler(feature_range=(0, 1))
        arr_out = scaler.fit_transform(arr).astype(np.float32)

    elif method == "zscore":
        scaler = StandardScaler()
        arr_out = scaler.fit_transform(arr).astype(np.float32)

    elif method == "smart":
        price_mask  = np.array([c in _PRICE_LEVEL_COLS for c in feature_cols], dtype=bool)
        volume_mask = np.array([c in _VOLUME_COLS       for c in feature_cols], dtype=bool)
        return_mask = np.array([c in _RETURN_COLS       for c in feature_cols], dtype=bool)

        arr_out = arr.copy()
        price_scaler  = MinMaxScaler(feature_range=(0, 1))
        volume_scaler = MinMaxScaler(feature_range=(0, 1))
        return_scaler = StandardScaler()

        if price_mask.any():
            arr_out[:, price_mask] = (
                price_scaler.fit_transform(arr[:, price_mask]).astype(np.float32)
            )
        if volume_mask.any():
            arr_out[:, volume_mask] = (
                volume_scaler.fit_transform(arr[:, volume_mask]).astype(np.float32)
            )
        if return_mask.any():
            arr_out[:, return_mask] = (
                return_scaler.fit_transform(arr[:, return_mask]).astype(np.float32)
            )

        scaler = {
            "price":        price_scaler,
            "volume":       volume_scaler,
            "return":       return_scaler,
            "feature_cols": feature_cols,
            "price_mask":   price_mask,
            "volume_mask":  volume_mask,
            "return_mask":  return_mask,
        }

    else:
        raise ValueError(
            f"Unknown normalisation method: '{method}'. "
            f"Choose 'minmax', 'zscore', or 'smart'."
        )

    logger.info(
        "normalise('%s') | shape %s | price=%d  volume=%d  return=%d",
        method,
        arr_out.shape,
        int(np.array([c in _PRICE_LEVEL_COLS for c in feature_cols]).sum()),
        int(np.array([c in _VOLUME_COLS       for c in feature_cols]).sum()),
        int(np.array([c in _RETURN_COLS       for c in feature_cols]).sum()),
    )
    return arr_out, scaler


# ---------------------------------------------------------------------------
# 5. Sequence windowing
# ---------------------------------------------------------------------------

def to_sequences(
    arr: np.ndarray,
    seq_len: int | None = None,
    stride: int = 1,
    label_horizon: int = 0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Slide a window over a 2-D normalised array to produce overlapping 3-D sequences.

    Parameters
    ----------
    arr           : (N, F) float32 normalised array.
    seq_len       : Window length. Defaults to gan_config.yaml seq_len (30).
    stride        : Step between consecutive windows.
                    stride=1  → maximum overlap (3493 sequences from 3522 rows).
                    stride=5  → sparse sampling (reduces correlation between
                                training batches — useful for 50-company runs
                                to avoid memory issues).
    label_horizon : If > 0, also returns a label tensor of the next N Close
                    values after each window. Useful for conditional training.
                    Close is assumed at column index 3 (OHLCV order).

    Returns
    -------
    tensor : (num_sequences, seq_len, F) float32.
    labels : (num_sequences, label_horizon) float32 — only if label_horizon > 0.
    """
    seq_len = seq_len or CFG.get("seq_len", 30)

    if len(arr) < seq_len + label_horizon:
        raise ValueError(
            f"Array length {len(arr)} is shorter than "
            f"seq_len ({seq_len}) + label_horizon ({label_horizon})."
        )

    end_idx = len(arr) - label_horizon if label_horizon > 0 else len(arr)
    seqs, labels_list = [], []

    for i in range(0, end_idx - seq_len + 1, stride):
        seqs.append(arr[i : i + seq_len])
        if label_horizon > 0:
            label_start = i + seq_len
            labels_list.append(arr[label_start : label_start + label_horizon, 3])

    tensor = torch.tensor(np.stack(seqs), dtype=torch.float32)
    logger.info(
        "to_sequences() | %d seqs | shape %s | stride=%d | density=%.1f%%",
        len(seqs), tuple(tensor.shape), stride,
        len(seqs) / max(len(arr), 1) * 100,
    )

    if label_horizon > 0:
        label_tensor = torch.tensor(np.stack(labels_list), dtype=torch.float32)
        logger.info("Label tensor: %s", tuple(label_tensor.shape))
        return tensor, label_tensor

    return tensor


# ---------------------------------------------------------------------------
# 6. Validation
# ---------------------------------------------------------------------------

def validate_tensor(tensor: torch.Tensor, name: str = "tensor") -> None:
    """
    Assert the sequence tensor has no NaN, Inf, or unexpected shape.

    Raises
    ------
    ValueError on any failed check.
    """
    if tensor.ndim != 3:
        raise ValueError(
            f"{name}: expected 3-D tensor (N, seq_len, F), "
            f"got shape {tuple(tensor.shape)}"
        )
    n_nan = int(torch.isnan(tensor).sum().item())
    n_inf = int(torch.isinf(tensor).sum().item())
    if n_nan > 0:
        raise ValueError(f"{name}: contains {n_nan} NaN values.")
    if n_inf > 0:
        raise ValueError(f"{name}: contains {n_inf} Inf values.")

    # Per-feature stats to catch scale imbalance
    feat_min  = tensor.min(dim=0).values.min(dim=0).values
    feat_max  = tensor.max(dim=0).values.max(dim=0).values
    feat_std  = tensor.std(dim=0).mean(dim=0)
    f = tensor.shape[-1]
    logger.info(
        "%s validated OK | shape=%s | "
        "global [min=%.3f, max=%.3f] | per-feature std range [%.3f, %.3f]",
        name, tuple(tensor.shape),
        tensor.min().item(), tensor.max().item(),
        feat_std.min().item(), feat_std.max().item(),
    )
    # Warn if any feature std is very different from others (scale imbalance)
    std_ratio = feat_std.max().item() / (feat_std.min().item() + 1e-8)
    if std_ratio > 10.0:
        logger.warning(
            "%s: feature std ratio = %.1f — possible scale imbalance. "
            "Check clipping bounds or consider 'smart' normalisation.",
            name, std_ratio,
        )


# ---------------------------------------------------------------------------
# 7. Temporal split
# ---------------------------------------------------------------------------

def split(
    tensor: torch.Tensor,
    train: float = 0.70,
    val: float   = 0.15,
    test: float  = 0.15,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split a sequence tensor into train / val / test in temporal order.

    Never shuffles — preserves the time ordering so there is no information
    leakage from future data into training.

    Parameters
    ----------
    tensor : (N, seq_len, F) tensor.
    train  : Training fraction (default 70%).
    val    : Validation fraction (default 15%).
    test   : Test fraction (default 15%).

    Returns
    -------
    train_t, val_t, test_t — sequential, non-overlapping slices.
    """
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError(f"train + val + test must equal 1.0, got {train+val+test}.")

    n       = len(tensor)
    n_train = int(n * train)
    n_val   = int(n * val)

    train_t = tensor[:n_train]
    val_t   = tensor[n_train : n_train + n_val]
    test_t  = tensor[n_train + n_val :]

    logger.info(
        "split() | train=%d  val=%d  test=%d  (total=%d)",
        len(train_t), len(val_t), len(test_t), n,
    )
    return train_t, val_t, test_t


# ---------------------------------------------------------------------------
# 8. Persist
# ---------------------------------------------------------------------------

def save(
    tensor: torch.Tensor,
    scaler: object,
    path: str | Path,
) -> None:
    """
    Save sequence tensor and scaler to disk.

    Files written
    -------------
    <path>.pt           — PyTorch tensor.
    <path>_scaler.pkl   — Fitted scaler (sklearn object or smart-scaler dict).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tensor_path = path.with_suffix(".pt")
    scaler_path = path.parent / (path.stem + "_scaler.pkl")

    torch.save(tensor, tensor_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    logger.info("Saved tensor  → %s  shape=%s", tensor_path, tuple(tensor.shape))
    logger.info("Saved scaler  → %s", scaler_path)


def load(path: str | Path) -> Tuple[torch.Tensor, object]:
    """Load a previously saved tensor + scaler pair from disk."""
    path = Path(path)
    tensor_path = path.with_suffix(".pt")
    scaler_path = path.parent / (path.stem + "_scaler.pkl")

    if not tensor_path.exists():
        raise FileNotFoundError(f"Tensor not found: {tensor_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    tensor = torch.load(tensor_path, map_location="cpu")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    logger.info("Loaded tensor %s from %s", tuple(tensor.shape), tensor_path)
    return tensor, scaler


# ---------------------------------------------------------------------------
# 9. Per-ticker normalise helper (used internally by build_dataset)
# ---------------------------------------------------------------------------

def _normalise_and_sequence(
    df: pd.DataFrame,
    method: NormMethod,
    seq_len: int,
    stride: int,
    label_horizon: int,
    use_features: bool,
) -> Tuple[torch.Tensor, object, Optional[torch.Tensor]]:
    """
    Internal pipeline step: features → normalise → sequences for one ticker.

    Returns (seq_tensor, scaler, label_tensor_or_None).
    """
    if use_features:
        df = add_features(df)

    arr, scaler = normalise(df, method=method)
    result = to_sequences(arr, seq_len=seq_len, stride=stride,
                          label_horizon=label_horizon)

    if label_horizon > 0:
        seq_t, lbl_t = result
        return seq_t, scaler, lbl_t
    return result, scaler, None


# ---------------------------------------------------------------------------
# 10. build_dataset — 50-company multi-ticker pipeline
# ---------------------------------------------------------------------------

def build_dataset(
    ticker: str | list[str],
    start: str,
    end: str,
    seq_len: int | None = None,
    method: NormMethod = "smart",
    stride: int = 1,
    use_features: bool = True,
    label_horizon: int = 0,
    per_ticker_normalise: bool = True,
    save_path: str | Path | None = None,
    validate: bool = True,
    skip_errors: bool = False,
):
    """
    End-to-end pipeline: fetch → clean → features → normalise → sequence.

    Supports single tickers and large multi-ticker corpora (50+ companies).

    Parameters
    ----------
    ticker               : Ticker string or list of tickers.
                           Use SP50_TICKERS for the 50-company universe.
    start, end           : Date range "YYYY-MM-DD".
    seq_len              : Window length (default: gan_config.yaml seq_len).
    method               : "minmax" | "zscore" | "smart" (recommended).
    stride               : Sliding window step. Use stride=5 for 50-company
                           runs to reduce memory and inter-sequence correlation.
    use_features         : Run add_features() (True by default).
    label_horizon        : If > 0, return next-N-close label tensor too.
    per_ticker_normalise : If True (default), fit a SEPARATE scaler per ticker.
                           Critical for a 50-company universe — without this,
                           SPY ($500) and a $2 penny stock share one scaler and
                           the penny stock is effectively invisible after scaling.
                           If False, one global scaler is fitted on the first
                           ticker and applied to all others (use only when all
                           tickers are in the same price range).
    save_path            : Optional base path for save(). No extension needed.
    validate             : Run validate_tensor() on the final tensor.
    skip_errors          : If True, log and skip tickers that fail fetch/clean
                           instead of raising. Useful for large batch runs where
                           a few delisted tickers shouldn't abort the whole job.

    Returns
    -------
    tensor  : (N_total, seq_len, F) float32 — all tickers concatenated.
    scaler  : Scaler for the LAST successfully processed ticker.
              For per_ticker_normalise=True, save the per-ticker scalers
              separately if you need to denormalise specific tickers.
    labels  : (N_total, label_horizon) — only if label_horizon > 0.
    scalers : dict[ticker, scaler] — only returned if per_ticker_normalise=True.
              Contains a separate fitted scaler for every ticker.
    """
    seq_len = seq_len or CFG.get("seq_len", 30)
    tickers = [ticker] if isinstance(ticker, str) else list(ticker)

    all_tensors:  list[torch.Tensor] = []
    all_labels:   list[torch.Tensor] = []
    scaler        = None
    ticker_scalers: dict[str, object] = {}
    global_scaler_fitted = False

    for t in tickers:
        try:
            df = fetch_ohlcv(t, start, end)
            df = clean(df)
        except Exception as exc:
            if skip_errors:
                logger.warning("Skipping %s: %s", t, exc)
                continue
            raise

        try:
            if per_ticker_normalise:
                # Each ticker gets its own scaler — correct for multi-name universe
                seq_t, scaler, lbl_t = _normalise_and_sequence(
                    df, method, seq_len, stride, label_horizon, use_features
                )
                ticker_scalers[t] = scaler
            else:
                # Single global scaler — only correct when all tickers have similar ranges
                if use_features:
                    df = add_features(df)
                arr_raw = df[FEATURE_COLS if all(c in df.columns for c in FEATURE_COLS)
                             else OHLCV_COLS].values.astype(np.float32)
                if not global_scaler_fitted:
                    arr, scaler = normalise(df, method=method)
                    global_scaler_fitted = True
                else:
                    # Reuse the already-fitted scaler from the first ticker
                    if isinstance(scaler, dict):
                        arr = arr_raw.copy()
                        if scaler["price_mask"].any():
                            arr[:, scaler["price_mask"]] = scaler["price"].transform(
                                arr_raw[:, scaler["price_mask"]]
                            ).astype(np.float32)
                        if scaler["volume_mask"].any():
                            arr[:, scaler["volume_mask"]] = scaler["volume"].transform(
                                arr_raw[:, scaler["volume_mask"]]
                            ).astype(np.float32)
                        if scaler["return_mask"].any():
                            arr[:, scaler["return_mask"]] = scaler["return"].transform(
                                arr_raw[:, scaler["return_mask"]]
                            ).astype(np.float32)
                    else:
                        arr = scaler.transform(arr_raw).astype(np.float32)

                result = to_sequences(arr, seq_len=seq_len, stride=stride,
                                      label_horizon=label_horizon)
                if label_horizon > 0:
                    seq_t, lbl_t = result
                else:
                    seq_t, lbl_t = result, None

            all_tensors.append(seq_t)
            if label_horizon > 0 and lbl_t is not None:
                all_labels.append(lbl_t)

        except Exception as exc:
            if skip_errors:
                logger.warning("Skipping %s during feature/normalise: %s", t, exc)
                continue
            raise

    if not all_tensors:
        raise RuntimeError("No tickers were successfully processed.")

    tensor = torch.cat(all_tensors, dim=0)

    if validate:
        validate_tensor(
            tensor,
            name=f"build_dataset({len(all_tensors)}/{len(tickers)} tickers)",
        )

    if save_path is not None:
        save(tensor, scaler, save_path)
        if per_ticker_normalise:
            scaler_dir = Path(save_path).parent
            for t, sc in ticker_scalers.items():
                sc_path = scaler_dir / f"{t}_scaler.pkl"
                with open(sc_path, "wb") as f:
                    pickle.dump(sc, f)
            logger.info(
                "Saved %d per-ticker scalers to %s",
                len(ticker_scalers), scaler_dir,
            )

    logger.info(
        "build_dataset() complete | %d/%d tickers | tensor %s",
        len(all_tensors), len(tickers), tuple(tensor.shape),
    )

    if label_horizon > 0:
        labels = torch.cat(all_labels, dim=0)
        if per_ticker_normalise:
            return tensor, scaler, labels, ticker_scalers
        return tensor, scaler, labels

    if per_ticker_normalise:
        return tensor, scaler, ticker_scalers

    return tensor, scaler


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch and prepare OHLCV data.")
    parser.add_argument(
        "--ticker", default="SPY",
        help="Ticker or comma-separated list. Use 'SP50' for the built-in 50-company universe.",
    )
    parser.add_argument("--start",     default="2010-01-01")
    parser.add_argument("--end",       default="2024-01-01")
    parser.add_argument("--seq_len",   type=int,  default=None)
    parser.add_argument("--method",    choices=["minmax", "zscore", "smart"], default="smart")
    parser.add_argument("--stride",    type=int,  default=1,
                        help="Use stride=5 for 50-company runs to reduce memory.")
    parser.add_argument("--features",  action="store_true", default=True)
    parser.add_argument("--no_features", dest="features", action="store_false")
    parser.add_argument("--label_horizon", type=int, default=0)
    parser.add_argument("--save_path", default="data/raw/SPY")
    parser.add_argument("--split",     action="store_true", default=False)
    parser.add_argument("--per_ticker_normalise", action="store_true", default=True)
    parser.add_argument("--skip_errors", action="store_true", default=False,
                        help="Skip failed tickers instead of raising.")
    args = parser.parse_args()

    if args.ticker.upper() == "SP50":
        tickers = SP50_TICKERS
    else:
        tickers = [t.strip() for t in args.ticker.split(",")]

    result = build_dataset(
        ticker=tickers if len(tickers) > 1 else tickers[0],
        start=args.start,
        end=args.end,
        seq_len=args.seq_len,
        method=args.method,
        stride=args.stride,
        use_features=args.features,
        label_horizon=args.label_horizon,
        per_ticker_normalise=args.per_ticker_normalise,
        save_path=args.save_path,
        skip_errors=args.skip_errors,
    )

    # Unpack regardless of return shape
    if args.label_horizon > 0:
        if args.per_ticker_normalise:
            tensor, scaler, labels, ticker_scalers = result
        else:
            tensor, scaler, labels = result
        print(f"\nDataset ready:")
        print(f"  Sequences : {tuple(tensor.shape)}")
        print(f"  Labels    : {tuple(labels.shape)}")
    else:
        if args.per_ticker_normalise:
            tensor, scaler, ticker_scalers = result
            print(f"\nPer-ticker scalers: {len(ticker_scalers)} fitted")
        else:
            tensor, scaler = result
        print(f"\nDataset ready — tensor shape: {tuple(tensor.shape)}")

    if args.split:
        tr, va, te = split(tensor)
        print(f"\nTemporal split:")
        print(f"  Train : {len(tr):,} sequences")
        print(f"  Val   : {len(va):,} sequences")
        print(f"  Test  : {len(te):,} sequences")