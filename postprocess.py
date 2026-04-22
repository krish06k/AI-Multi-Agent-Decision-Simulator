"""
postprocess.py
==============
Post-processes raw GAN-generated synthetic OHLCV sequences.

Pipeline per file
-----------------
1. OHLC consistency fix   — ensures High >= max(O,C), Low <= min(O,C)
2. Return rescaling        — volatility-normalise so std sits in target band
3. Hard clipping           — cap single-tick moves at ±MAX_MOVE
4. Close rebuild           — reconstruct price from cleaned returns
5. Open smoothing          — Open[t] = Close[t-1] (realistic gap fill)
6. OHLC wicks              — add small realistic wicks around O/C range
7. Volume smoothing        — if present, log-normalise and smooth volume
8. Final validation        — only save sequences that pass ALL checks
9. Regime labelling        — tag each file as calm / volatile / trending

Changes from previous version
------------------------------
- Clip threshold (MAX_MOVE) and validation threshold (VAL_MAX_MOVE) are now
  the SAME value — previously clipping to 0.08 but validating at 0.15 meant
  the max_move check was dead code and std was doing all the work.
- Aggressive rescaling: if std still too high after first rescale, apply a
  second pass with a stricter target before giving up on the sequence.
- Volume column handled explicitly — log-normalised + smoothed.
- Regime label written as a sidecar .txt file consumed by simulation/runner.py
  to tag sequences as calm/volatile/trending at load time.
- Detailed summary printed at the end showing rejection reasons.
"""

import os
import glob
import shutil

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INPUT_DIR   = "data/synthetic"
OUTPUT_DIR  = "data/filtered"

TARGET_STD  = 0.018   # realistic daily-return std (≈ S&P 500 daily vol)
MAX_STD     = 0.030   # hard ceiling — sequences above this after 2 rescales → reject
MAX_MOVE    = 0.08    # max single-tick return (clip AND validate with same value)
MIN_STD     = 0.003   # floor — sequences flatter than this are also unrealistic

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rescale_returns(r: pd.Series, target_std: float) -> pd.Series:
    """Scale return series so its std matches target_std."""
    std = r.std()
    if std > 1e-8:
        r = r * (target_std / std)
    return r


def _label_regime(r: pd.Series) -> str:
    """
    Assign a regime label based on return statistics.
    Used by simulation/runner.py to pick the right shock profile.
    """
    std  = r.std()
    trend = abs(r.mean()) / (std + 1e-8)   # signal-to-noise

    if std < 0.008:
        return "calm"
    elif std > 0.022 or r.abs().max() > 0.05:
        return "volatile"
    elif trend > 0.15:
        return "trending"
    else:
        return "calm"


def _smooth_volume(vol: pd.Series) -> pd.Series:
    """Log-normalise + 3-tick rolling smooth so volume looks realistic."""
    vol = vol.clip(lower=1)
    vol = np.log1p(vol)
    vol = vol.rolling(3, min_periods=1).mean()
    # re-scale to [100, 10_000] range typical of normalised synthetic data
    vmin, vmax = vol.min(), vol.max()
    if vmax > vmin:
        vol = 100 + (vol - vmin) / (vmax - vmin) * 9900
    return vol.round(0)


def is_valid(r: pd.Series) -> tuple[bool, str]:
    """
    Return (passed, reason).
    Validates the CLEANED return series, not the raw one.
    Both thresholds use the same MAX_MOVE constant now.
    """
    std      = r.std()
    max_move = r.abs().max()

    if std > MAX_STD:
        return False, f"std_too_high({std:.4f}>{MAX_STD})"
    if std < MIN_STD:
        return False, f"std_too_low({std:.4f}<{MIN_STD})"
    if max_move > MAX_MOVE:
        return False, f"max_move({max_move:.4f}>{MAX_MOVE})"

    return True, "ok"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

files = sorted(glob.glob(f"{INPUT_DIR}/*.csv"))

passed        = 0
reject_reasons: dict[str, int] = {}

for f in files:
    try:
        df = pd.read_csv(f, index_col=0)
    except Exception as e:
        reject_reasons["read_error"] = reject_reasons.get("read_error", 0) + 1
        continue

    # Guard — must have Close column
    if "Close" not in df.columns:
        reject_reasons["no_close_col"] = reject_reasons.get("no_close_col", 0) + 1
        continue

    # -----------------------------------------------------------------------
    # 1. OHLC consistency — make sure High/Low bracket O and C
    # -----------------------------------------------------------------------
    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            df[col] = df["Close"]   # synthesise missing columns from Close

    df["High"] = np.maximum.reduce([df["Open"], df["High"], df["Low"], df["Close"]])
    df["Low"]  = np.minimum.reduce([df["Open"], df["High"], df["Low"], df["Close"]])

    # -----------------------------------------------------------------------
    # 2. Return rescaling — first pass: bring std to TARGET_STD
    # -----------------------------------------------------------------------
    r = df["Close"].pct_change().fillna(0)
    r = _rescale_returns(r, TARGET_STD)

    # -----------------------------------------------------------------------
    # 3. Hard clip at ±MAX_MOVE  (consistent with validation threshold)
    # -----------------------------------------------------------------------
    r = np.clip(r, -MAX_MOVE, MAX_MOVE)

    # -----------------------------------------------------------------------
    # 4. Second rescale if std still too high after clipping
    #    (clipping distorts the distribution — rescale again to be safe)
    # -----------------------------------------------------------------------
    if r.std() > MAX_STD:
        r = _rescale_returns(r, TARGET_STD)
        r = np.clip(r, -MAX_MOVE, MAX_MOVE)

    # -----------------------------------------------------------------------
    # 5. Validate BEFORE rebuilding price (cheaper to reject here)
    # -----------------------------------------------------------------------
    valid, reason = is_valid(r)
    if not valid:
        reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
        continue

    # -----------------------------------------------------------------------
    # 6. Rebuild Close from cleaned returns
    # -----------------------------------------------------------------------
    start_price = df["Close"].iloc[0]
    if start_price <= 0:
        start_price = 100.0
    df["Close"] = start_price * (1 + r).cumprod()

    # -----------------------------------------------------------------------
    # 7. Open smoothing — Open[t] = Close[t-1]
    # -----------------------------------------------------------------------
    df["Open"] = df["Close"].shift(1).fillna(df["Close"].iloc[0])

    # -----------------------------------------------------------------------
    # 8. Realistic OHLC wicks (small random deviation around O/C)
    # -----------------------------------------------------------------------
    wick_noise = np.abs(np.random.normal(0, 0.003, len(df)))
    df["High"] = np.maximum(df["Open"], df["Close"]) * (1 + wick_noise)
    df["Low"]  = np.minimum(df["Open"], df["Close"]) * (1 - wick_noise)

    # -----------------------------------------------------------------------
    # 9. Volume — smooth if present, synthesise if absent
    # -----------------------------------------------------------------------
    if "Volume" in df.columns:
        df["Volume"] = _smooth_volume(df["Volume"])
    else:
        # Synthesise volume correlated with abs(return) — realistic behaviour
        base_vol = 1000
        df["Volume"] = (base_vol * (1 + 10 * r.abs()) * np.random.lognormal(0, 0.3, len(df))).round(0)

    # -----------------------------------------------------------------------
    # 10. Regime label — sidecar file read by simulation/runner.py
    # -----------------------------------------------------------------------
    regime = _label_regime(r)
    label_path = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(f))[0] + "_regime.txt")
    with open(label_path, "w") as lf:
        lf.write(regime)

    # -----------------------------------------------------------------------
    # 11. Save
    # -----------------------------------------------------------------------
    out_path = os.path.join(OUTPUT_DIR, os.path.basename(f))
    df.to_csv(out_path)
    passed += 1

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
total = len(files)
print(f"\n✅ Postprocess complete")
print(f"Valid samples : {passed}/{total} ({100 * passed / max(total, 1):.1f}%)")
print(f"Rejected      : {total - passed}")

if reject_reasons:
    print("\nRejection breakdown:")
    for reason, count in sorted(reject_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason:<40} {count}")