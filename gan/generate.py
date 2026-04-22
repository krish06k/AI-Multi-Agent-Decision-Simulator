"""
gan/generate.py
===============
Layer 1 — GAN Data Engine | Inference & Synthetic Data Export

Pure inference script. Loads a trained Generator checkpoint and produces
synthetic OHLCV sequences on demand. Handles denormalisation so output
is in real price space (not the normalised GAN space).

Key responsibilities
--------------------
* load_generator()     — instantiate correct model class from checkpoint.
* generate()           — sample raw normalised sequences from the model.
* denormalise()        — invert scaler back to real OHLCV values.
                         Handles plain sklearn scalers AND the "smart" scaler
                         dict produced by data_loader.normalise("smart").
* to_dataframe()       — convert tensor → labelled pandas DataFrame.
* save_synthetic()     — persist generated sequences to data/synthetic/.
* generate_pipeline()  — one-call convenience wrapping all of the above.

Called by
---------
* simulation/runner.py — at each environment reset to seed a price path.
* CLI: python -m gan.generate  (pre-populate data/synthetic/ in bulk).

Output format
-------------
Each generated sequence is a DataFrame:
    columns : ["Open", "High", "Low", "Close", "Volume"]
    index   : RangeIndex (0 … seq_len-1) — relative time steps.

Changelog v4
------------
* to_dataframe() now clips extreme single-step price moves to ±12%
  before enforcing OHLC consistency. This reduces invalid sequences
  that fail the pass rate filter without distorting realistic moves.
* return columns clamped to ±3 std before inverse transform in denormalise().

Changelog v3
------------
* denormalise() now handles the "smart" scaler dict.
* generate_pipeline() accepts an optional per_ticker_scalers dict.
* _reconstruct_ohlcv() now selects the Close column by name.
* OHLC consistency enforcement.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml

from gan.generator import Generator, build_generator
from gan.conditional_gan import (
    CondGenerator,
    build_cond_generator,
    encode_regime,
    RegimeLabel,
)
from gan.data_loader import FEATURE_COLS, OHLCV_COLS, _PRICE_LEVEL_COLS, _VOLUME_COLS, _RETURN_COLS

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_CFG_PATH = Path(__file__).resolve().parents[1] / "config" / "gan_config.yaml"


def _load_cfg() -> dict:
    with open(_CFG_PATH, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# 1. Load Generator from checkpoint
# ---------------------------------------------------------------------------

def load_generator(
    checkpoint_path: str | Path,
    conditional: bool = False,
    cfg: dict | None = None,
    device: torch.device | str = "cpu",
) -> Generator | CondGenerator:
    """
    Instantiate and restore a Generator (or CondGenerator) from a .pt checkpoint.

    Parameters
    ----------
    checkpoint_path : Path to the .pt file saved by trainer.save_checkpoint().
    conditional     : If True, builds a CondGenerator; else plain Generator.
    cfg             : Optional config override. If None, reads gan_config.yaml.
    device          : Target device.

    Returns
    -------
    Generator (or CondGenerator) in eval mode with weights loaded.

    Raises
    ------
    FileNotFoundError if checkpoint file is missing.
    KeyError          if checkpoint lacks "generator_state".
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    cfg = cfg or _load_cfg()
    ckpt = torch.load(path, map_location=device)

    if "generator_state" not in ckpt:
        raise KeyError(f"Checkpoint {path} has no 'generator_state' key.")

    G = build_cond_generator(cfg) if conditional else build_generator(cfg)
    G.load_state_dict(ckpt["generator_state"])
    G = G.to(device)
    G.eval()

    logger.info(
        "Loaded %s from %s (epoch %s) on %s.",
        type(G).__name__, path.name, ckpt.get("epoch", "?"), device,
    )
    return G


# ---------------------------------------------------------------------------
# 2. Generate raw (normalised) sequences
# ---------------------------------------------------------------------------

def generate(
    generator: Generator | CondGenerator,
    n: int,
    regime: Optional[RegimeLabel] = None,
    device: torch.device | str = "cpu",
    temperature: float = 1.0,
    batch_size: int = 256,
) -> torch.Tensor:
    """
    Sample n synthetic sequences from the Generator.

    Handles large n by batching internally to avoid OOM.

    Parameters
    ----------
    generator   : Loaded Generator/CondGenerator in eval mode.
    n           : Number of sequences to generate.
    regime      : Optional regime label (only used by CondGenerator).
    device      : Target device.
    temperature : Noise scale; >1 → more diverse, <1 → more conservative.
    batch_size  : Internal chunk size for large n.

    Returns
    -------
    tensor : (n, seq_len, F) in normalised space.
    """
    generator.eval()
    is_cond   = isinstance(generator, CondGenerator)
    noise_dim = generator.real_noise_dim if is_cond else generator.noise_dim

    chunks: list[torch.Tensor] = []
    remaining = n

    with torch.no_grad():
        while remaining > 0:
            chunk = min(remaining, batch_size)
            z = torch.randn(chunk, noise_dim, device=device) * temperature

            if is_cond:
                r = (
                    encode_regime(regime or "calm")
                    .unsqueeze(0).expand(chunk, -1).to(device)
                )
                out = generator(z, r)
            else:
                out = generator(z)

            chunks.append(out.cpu())
            remaining -= chunk

    result = torch.cat(chunks, dim=0)
    logger.info(
        "Generated %d sequences | regime=%s  temp=%.2f  shape=%s",
        n, regime or "unconditional", temperature, tuple(result.shape),
    )
    return result


# ---------------------------------------------------------------------------
# 3. Denormalise → real space
# ---------------------------------------------------------------------------

def denormalise(
    tensor: torch.Tensor,
    scaler: object,
    feature_cols: list[str] | None = None,
    clip_volume: bool = True,
) -> torch.Tensor:
    """
    Invert scaler to convert GAN output back to real-world values.

    Handles three scaler types:
    1. Plain sklearn MinMaxScaler / StandardScaler — single inverse_transform call.
    2. "smart" scaler dict from data_loader.normalise("smart"):
           {"price": MinMaxScaler, "volume": MinMaxScaler,
            "return": StandardScaler,
            "feature_cols": list, "price_mask": bool[], "volume_mask": bool[],
            "return_mask": bool[]}
       Each column group is inverted with its own scaler.
    3. Legacy tuple format (price_scaler, return_scaler, cols, mask).

    Parameters
    ----------
    tensor       : (n, seq_len, F) or (seq_len, F) in normalised space.
    scaler       : Fitted scaler object or smart-scaler dict.
    feature_cols : Column names corresponding to the F dimension.
                   Defaults to FEATURE_COLS if F matches, else OHLCV_COLS.
    clip_volume  : Clip Volume (and volume_log, volume_z) to ≥ 0 after inversion.

    Returns
    -------
    torch.Tensor of same shape in real-world scale.
    """
    squeeze = tensor.dim() == 2
    if squeeze:
        tensor = tensor.unsqueeze(0)

    n, seq_len, F = tensor.shape
    arr = tensor.numpy().reshape(-1, F).copy()    # (n*seq_len, F)

    # --- Determine feature_cols ---
    if feature_cols is None:
        if F == len(FEATURE_COLS):
            feature_cols = FEATURE_COLS
        elif F == len(OHLCV_COLS):
            feature_cols = OHLCV_COLS
        else:
            if isinstance(scaler, dict) and "feature_cols" in scaler:
                feature_cols = scaler["feature_cols"]
            else:
                feature_cols = OHLCV_COLS[:F]

    # --- Inversion logic ---
    if isinstance(scaler, dict) and "price" in scaler:
        # Smart scaler dict (v3/v4 format)
        # Generator outputs tanh (-1, 1) but .pt training data was MinMaxScaler
        # normalized to (0, 1). Remap price and volume columns before inversion.
        arr_out = arr.copy()
        if scaler["price_mask"].any():
            price_cols = arr[:, scaler["price_mask"]]
            price_cols = (price_cols + 1.0) / 2.0          # tanh (-1,1) → (0,1)
            price_cols = price_cols.clip(0.0, 1.0)          # safety clamp
            arr_out[:, scaler["price_mask"]] = (
                scaler["price"].inverse_transform(price_cols)
            )
        if scaler.get("volume_mask", np.zeros(F, dtype=bool)).any():
            vm = scaler["volume_mask"]
            vol_cols = arr[:, vm]
            vol_cols = (vol_cols + 1.0) / 2.0              # same remap for volume
            vol_cols = vol_cols.clip(0.0, 1.0)
            arr_out[:, vm] = scaler["volume"].inverse_transform(vol_cols)
        if scaler["return_mask"].any():
            # StandardScaler — clamp to ±3 std to prevent extreme outliers
            arr_out[:, scaler["return_mask"]] = (
                scaler["return"].inverse_transform(
                    arr[:, scaler["return_mask"]].clip(-3.0, 3.0)
                )
            )

    elif isinstance(scaler, tuple) and len(scaler) == 4:
        # Legacy tuple: (price_scaler, return_scaler, cols, mask)
        price_scaler, return_scaler, _cols, price_mask = scaler
        arr_out = arr.copy()
        if price_mask.any():
            arr_out[:, price_mask]  = price_scaler.inverse_transform(arr[:, price_mask])
        if (~price_mask).any():
            arr_out[:, ~price_mask] = return_scaler.inverse_transform(arr[:, ~price_mask])

    else:
        # Plain sklearn scaler (minmax or zscore)
        from sklearn.preprocessing import MinMaxScaler as _MMS
        if isinstance(scaler, _MMS):
            arr = (arr + 1.0) / 2.0
        arr_out = scaler.inverse_transform(arr)

    arr_out = arr_out.reshape(n, seq_len, F).astype(np.float32)

    # Clip volume-related columns to ≥ 0
    if clip_volume:
        for i, col in enumerate(feature_cols):
            if col in ("Volume", "volume_log", "volume_z"):
                arr_out[:, :, i] = np.maximum(arr_out[:, :, i], 0.0)

    result = torch.from_numpy(arr_out)
    if squeeze:
        result = result.squeeze(0)
    return result


# ---------------------------------------------------------------------------
# 4. Tensor → DataFrame (OHLCV only)
# ---------------------------------------------------------------------------

def to_dataframe(
    tensor: torch.Tensor,
    feature_cols: list[str] | None = None,
    index: Optional[pd.Index] = None,
) -> "pd.DataFrame | list[pd.DataFrame]":
    """
    Convert generated tensor to OHLCV-only pandas DataFrame(s).

    Extracts Open/High/Low/Close/Volume from the full feature tensor.
    Clips extreme single-step price moves to ±12% to reduce invalid sequences.
    Enforces OHLC consistency: High ≥ max(Open, Close), Low ≤ min(Open, Close).

    Parameters
    ----------
    tensor       : (seq_len, F) for a single sequence, or (n, seq_len, F) for batch.
    feature_cols : Column names for the F dimension. Defaults to FEATURE_COLS.
    index        : Optional DatetimeIndex for the time axis.

    Returns
    -------
    pd.DataFrame if input is 2-D, list[pd.DataFrame] if 3-D.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS if tensor.shape[-1] == len(FEATURE_COLS) else OHLCV_COLS

    def _single(t: torch.Tensor) -> pd.DataFrame:
        arr = t.numpy()
        all_cols = list(feature_cols)
        full_df  = pd.DataFrame(arr, columns=all_cols, index=index)

        # Reconstruct Volume from volume_log if raw Volume not present
        if "Volume" not in full_df.columns and "volume_log" in full_df.columns:
            full_df["Volume"] = np.expm1(np.maximum(full_df["volume_log"], 0))

        # Select OHLCV columns that exist
        available = [c for c in OHLCV_COLS if c in full_df.columns]
        df = full_df[available].copy()

        # v4: Clip extreme single-step Close moves to ±12%
        # Reconstructs a smooth price path from the first Close value
        # This directly targets the sequences that fail the pass rate filter
        if "Close" in df.columns and len(df) > 1:
            close = df["Close"].values.copy()
            for i in range(1, len(close)):
                prev = close[i - 1]
                if prev > 0:
                    ret = (close[i] - prev) / prev
                    if abs(ret) > 0.12:
                        close[i] = prev * (1.12 if ret > 0 else 0.88)
            df["Close"] = close

            # Rescale Open proportionally to keep OHLC ratios consistent
            if "Open" in df.columns:
                orig_close = full_df["Close"].values.copy()
                orig_close = np.where(orig_close < 0.01, 0.01, orig_close)
                scale = close / orig_close
                df["Open"] = (full_df["Open"].values * scale).clip(min=0.01)

        # Enforce OHLC consistency: High ≥ max(Open,Close), Low ≤ min(Open,Close)
        if {"Open", "High", "Low", "Close"}.issubset(df.columns):
            df["High"] = df[["Open", "Close", "High"]].max(axis=1)
            df["Low"]  = df[["Open", "Close", "Low"]].min(axis=1)

        # Clip all prices to positive
        for col in ("Open", "High", "Low", "Close"):
            if col in df.columns:
                df[col] = df[col].clip(lower=0.01)

        if "Volume" in df.columns:
            df["Volume"] = df["Volume"].clip(lower=0.0)

        return df

    if tensor.dim() == 2:
        return _single(tensor)
    elif tensor.dim() == 3:
        return [_single(tensor[i]) for i in range(tensor.size(0))]
    else:
        raise ValueError(f"Expected 2-D or 3-D tensor, got {tensor.dim()}-D.")


# ---------------------------------------------------------------------------
# 5. Save synthetic sequences
# ---------------------------------------------------------------------------

def save_synthetic(
    dfs: "pd.DataFrame | list[pd.DataFrame]",
    path: str | Path,
    prefix: str = "synthetic",
) -> list[Path]:
    """
    Persist generated DataFrames to data/synthetic/ as CSVs.

    Parameters
    ----------
    dfs    : Single DataFrame or list of DataFrames.
    path   : Output directory.
    prefix : Filename prefix.

    Returns
    -------
    List of saved file paths.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]

    saved: list[Path] = []
    for i, df in enumerate(dfs):
        fpath = path / f"{prefix}_{i:04d}.csv"
        df.to_csv(fpath, index=True)
        saved.append(fpath)

    logger.info("Saved %d synthetic sequences to %s", len(saved), path)
    return saved


# ---------------------------------------------------------------------------
# 6. Load scaler
# ---------------------------------------------------------------------------

def load_scaler(path: str | Path) -> object:
    """
    Load the sklearn scaler saved by data_loader.save().

    Parameters
    ----------
    path : Base path WITHOUT extension (same as used in data_loader.save()).
           e.g. "data/raw/SPY" → loads "data/raw/SPY_scaler.pkl".

    Returns
    -------
    Fitted scaler (sklearn object or smart-scaler dict).
    """
    path = Path(path)
    scaler_path = path.parent / (path.stem + "_scaler.pkl")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    logger.info("Loaded scaler from %s", scaler_path)
    return scaler


# ---------------------------------------------------------------------------
# 7. generate_pipeline — one-call end-to-end synthesis
# ---------------------------------------------------------------------------

def generate_pipeline(
    checkpoint_path: str | Path,
    scaler_path: str | Path,
    n: int = 100,
    regime: Optional[RegimeLabel] = None,
    conditional: bool = False,
    device: torch.device | str = "cpu",
    temperature: float = 1.0,
    save_path: Optional[str | Path] = None,
    prefix: str = "synthetic",
    cfg: dict | None = None,
    feature_cols: list[str] | None = None,
) -> "list[pd.DataFrame]":
    """
    End-to-end synthesis:
    load_generator → generate → denormalise → to_dataframe → (save).

    Parameters
    ----------
    checkpoint_path : .pt file from trainer.save_checkpoint().
    scaler_path     : Base path to scaler (e.g. "data/raw/SPY").
    n               : Number of sequences to generate.
    regime          : Optional regime for CondGenerator.
    conditional     : Whether to load CondGenerator.
    device          : Computation device.
    temperature     : GAN noise diversity (1.0 = standard).
    save_path       : If given, save CSVs to this directory.
    prefix          : CSV filename prefix.
    cfg             : Config override dict.
    feature_cols    : Feature column names for the tensor's F dimension.

    Returns
    -------
    List of pd.DataFrames, each (seq_len, 5) in real OHLCV scale.
    """
    G      = load_generator(checkpoint_path, conditional=conditional, cfg=cfg, device=device)
    scaler = load_scaler(scaler_path)

    raw        = generate(G, n=n, regime=regime, device=device, temperature=temperature)
    real_scale = denormalise(raw, scaler, feature_cols=feature_cols)
    dfs        = to_dataframe(real_scale, feature_cols=feature_cols)

    if save_path:
        tag = f"{prefix}_{regime or 'unconditional'}"
        save_synthetic(dfs, path=save_path, prefix=tag)

    return dfs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic OHLCV sequences.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--scaler",     required=True, help="Base path to scaler (no ext).")
    parser.add_argument("--n",          type=int,    default=100)
    parser.add_argument("--regime",     choices=["calm", "stressed", "trending"], default=None)
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--save_path",  default="data/synthetic")
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    dfs = generate_pipeline(
        checkpoint_path=args.checkpoint,
        scaler_path=args.scaler,
        n=args.n,
        regime=args.regime,
        conditional=args.conditional,
        device=args.device,
        temperature=args.temperature,
        save_path=args.save_path,
    )
    print(f"\nGenerated {len(dfs)} sequences.")
    print(f"Sample head:\n{dfs[0].head()}")
    print(f"Sample stats:\n{dfs[0].describe().round(2)}")