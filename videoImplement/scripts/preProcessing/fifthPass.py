

from __future__ import annotations

import numpy as np
import pandas as pd
from scripts.others.util import dprint


def _infer_dt(t: np.ndarray) -> float:
    if t.size < 2:
        return float("nan")
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(np.median(dt)) if dt.size else float("nan")


def _nearest_bool_by_time(t_src: np.ndarray, b_src: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """
    Sample boolean array b_src at times t_grid by nearest-neighbour lookup on t_src.
    (Used to propagate is_bad_data without "interpolating" booleans.)
    """
    idx = np.searchsorted(t_src, t_grid, side="left")
    idx = np.clip(idx, 0, len(t_src) - 1)
    idx_left = np.clip(idx - 1, 0, len(t_src) - 1)
    choose_left = np.abs(t_src[idx_left] - t_grid) <= np.abs(t_src[idx] - t_grid)
    idx_final = np.where(choose_left, idx_left, idx)
    return b_src[idx_final].astype(bool)


def averagePLRGraphs(df1: pd.DataFrame, df2: pd.DataFrame, px_to_mm: float, fps: int | None = None) -> pd.DataFrame | None:
    dprint("Pass 5: averaging two PLR graphs")

    if df1 is None or df2 is None:
        dprint("Error: one of the DataFrames is None")
        return None

    if "timestamp" not in df1.columns or "timestamp" not in df2.columns:
        dprint("Error: missing 'timestamp' column (needed for alignment)")
        return None
    if "diameter_mm" not in df1.columns or "diameter_mm" not in df2.columns:
        dprint("Error: missing 'diameter_mm' column")
        return None

    t1 = pd.to_numeric(df1["timestamp"], errors="coerce").to_numpy(float)
    t2 = pd.to_numeric(df2["timestamp"], errors="coerce").to_numpy(float)
    d1 = pd.to_numeric(df1["diameter_mm"], errors="coerce").to_numpy(float)
    d2 = pd.to_numeric(df2["diameter_mm"], errors="coerce").to_numpy(float)

  
    if np.isnan(d1).any() or np.isnan(d2).any():
        dprint("Warning: NaNs detected in one/both inputs. Averaging may contain gaps.")

    if len(df1) == len(df2) and np.allclose(t1, t2, atol=1e-6, rtol=0):
        out = df1.copy()
        out["diameter_mm"] = (d1 + d2) / 2.0
        out["diameter"] = out["diameter_mm"] * float(px_to_mm)

        b1 = df1["is_bad_data"].astype(bool).to_numpy() if "is_bad_data" in df1.columns else np.zeros(len(out), dtype=bool)
        b2 = df2["is_bad_data"].astype(bool).to_numpy() if "is_bad_data" in df2.columns else np.zeros(len(out), dtype=bool)
        out["is_bad_data"] = (b1 | b2)

        if "confidence" in df1.columns and "confidence" in df2.columns:
            c1 = pd.to_numeric(df1["confidence"], errors="coerce").to_numpy(float)
            c2 = pd.to_numeric(df2["confidence"], errors="coerce").to_numpy(float)
            out["confidence"] = np.nanmin(np.vstack([c1, c2]), axis=0)
        return out

    dt1 = _infer_dt(t1)
    dt2 = _infer_dt(t2)
    if not np.isfinite(dt1) and fps is not None and fps > 0:
        dt1 = 1.0 / float(fps)
    if not np.isfinite(dt2) and fps is not None and fps > 0:
        dt2 = 1.0 / float(fps)

    if not np.isfinite(dt1) or not np.isfinite(dt2):
        dprint("Error: could not infer dt from timestamps (and fps not provided).")
        return None

    dt_common = max(dt1, dt2)  
    start = max(float(np.nanmin(t1)), float(np.nanmin(t2)))
    end = min(float(np.nanmax(t1)), float(np.nanmax(t2)))

    if end <= start + dt_common:
        dprint(f"Error: insufficient overlap to average (overlap={end-start:.3f}s).")
        return None

    grid = np.arange(start, end + dt_common / 2.0, dt_common)

    d1i = np.interp(grid, t1, d1)
    d2i = np.interp(grid, t2, d2)
    davg = (d1i + d2i) / 2.0

    b1 = df1["is_bad_data"].astype(bool).to_numpy() if "is_bad_data" in df1.columns else np.zeros(len(df1), dtype=bool)
    b2 = df2["is_bad_data"].astype(bool).to_numpy() if "is_bad_data" in df2.columns else np.zeros(len(df2), dtype=bool)
    bavg = _nearest_bool_by_time(t1, b1, grid) | _nearest_bool_by_time(t2, b2, grid)

    if "confidence" in df1.columns and "confidence" in df2.columns:
        c1 = pd.to_numeric(df1["confidence"], errors="coerce").to_numpy(float)
        c2 = pd.to_numeric(df2["confidence"], errors="coerce").to_numpy(float)
        c1i = np.interp(grid, t1, c1)
        c2i = np.interp(grid, t2, c2)
        cavg = np.minimum(c1i, c2i)
    else:
        cavg = np.full_like(grid, 1.0, dtype=float)

    out = pd.DataFrame({
        "frame_id": np.arange(len(grid), dtype=int),
        "timestamp": grid,
        "diameter_mm": davg,
        "diameter": davg * float(px_to_mm),
        "confidence": cavg,
        "is_bad_data": bavg.astype(bool),
    })

    dprint(
        f"Pass 5 aligned average: df1_len={len(df1)}, df2_len={len(df2)}, "
        f"out_len={len(out)}, overlap=[{start:.3f}s, {end:.3f}s]"
    )
    return out
