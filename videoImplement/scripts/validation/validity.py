
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Safe non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "frame_id" not in df.columns:
        df["frame_id"] = np.arange(len(df))
    if "is_bad_data" not in df.columns:
        df["is_bad_data"] = False
    # Pass 4 bookkeeping (present in the main pipeline). Keep it if available;
    # otherwise default to False so downstream metrics behave sensibly.
    if "is_interpolated" not in df.columns:
        df["is_interpolated"] = False
    return df


def _align_by_frame(raw: Optional[pd.DataFrame], interp: pd.DataFrame, smooth: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Align on frame_id to avoid accidental off-by-one issues."""
    interp = _ensure_cols(interp)
    smooth = _ensure_cols(smooth)

    if raw is None:
        # align interp & smooth
        m = pd.merge(interp, smooth, on="frame_id", suffixes=("_interp", "_smooth"), how="inner")
        interp2 = m[[c for c in m.columns if c.endswith("_interp") or c == "frame_id"]].copy()
        smooth2 = m[[c for c in m.columns if c.endswith("_smooth") or c == "frame_id"]].copy()
        interp2.columns = [c.replace("_interp", "") for c in interp2.columns]
        smooth2.columns = [c.replace("_smooth", "") for c in smooth2.columns]
        return None, interp2, smooth2

    raw = _ensure_cols(raw)

    m = raw[["frame_id", "timestamp", "diameter_mm", "confidence", "is_bad_data"]].merge(
        interp[["frame_id", "diameter_mm", "is_bad_data", "is_interpolated"]],
        on="frame_id",
        suffixes=("_raw", "_interp"),
        how="inner",
    ).merge(
        smooth[["frame_id", "diameter_mm"]],
        on="frame_id",
        how="inner",
    )

    raw2 = m[["frame_id", "timestamp", "diameter_mm_raw", "confidence", "is_bad_data_raw"]].copy()
    raw2.columns = ["frame_id", "timestamp", "diameter_mm", "confidence", "is_bad_data"]

    interp2 = m[["frame_id", "timestamp", "diameter_mm_interp", "is_bad_data_interp", "is_interpolated"]].copy()
    interp2.columns = ["frame_id", "timestamp", "diameter_mm", "is_bad_data", "is_interpolated"]

    smooth2 = m[["frame_id", "timestamp", "diameter_mm"]].copy()
    smooth2["is_bad_data"] = False

    return raw2, interp2, smooth2


def _infer_fps_from_ts(ts: np.ndarray) -> float:
    if len(ts) < 3:
        return float("nan")
    dt = np.diff(ts)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return float("nan")
    return float(1.0 / np.median(dt))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    a2, b2 = a[mask], b[mask]
    if np.std(a2) == 0 or np.std(b2) == 0:
        return float("nan")
    return float(np.corrcoef(a2, b2)[0, 1])


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(a[mask] - b[mask])))


def _welch_psd(x: np.ndarray, fs: float, nperseg: int = 256, noverlap: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Simple Welch PSD (no SciPy dependency)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 8 or not np.isfinite(fs) or fs <= 0:
        return np.array([]), np.array([])

    if noverlap is None:
        noverlap = nperseg // 2

    nperseg = int(min(nperseg, len(x)))
    if nperseg < 8:
        return np.array([]), np.array([])

    step = nperseg - noverlap
    if step <= 0:
        step = nperseg

    win = np.hanning(nperseg)
    scale = np.sum(win ** 2)

    psds = []
    for start in range(0, len(x) - nperseg + 1, step):
        seg = x[start:start + nperseg]
        seg = seg - np.mean(seg)
        segw = seg * win
        fft = np.fft.rfft(segw)
        pxx = (np.abs(fft) ** 2) / (scale * fs)
        psds.append(pxx)

    if not psds:
        return np.array([]), np.array([])

    pxx = np.mean(np.vstack(psds), axis=0)
    f = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return f, pxx


def _max_interp_gap(interp_mask: np.ndarray, fs: float) -> Tuple[int, float]:
    """Return (max_run_frames, max_run_seconds)."""
    if interp_mask.size == 0:
        return 0, float("nan")
    max_run = 0
    cur = 0
    for v in interp_mask.astype(bool):
        if v:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    if not np.isfinite(fs) or fs <= 0:
        return int(max_run), float("nan")
    return int(max_run), float(max_run / fs)



def _distortion_summary(diff: np.ndarray) -> Dict[str, float]:
    """Summarise smoothing distortion (smooth - interp).

    Returns mean (bias), SD, 95th percentile of absolute distortion, and max absolute distortion.
    """
    diff = np.asarray(diff, dtype=float)
    mask = np.isfinite(diff)
    if mask.sum() == 0:
        return {
            "mean_mm": float("nan"),
            "sd_mm": float("nan"),
            "p95_abs_mm": float("nan"),
            "max_abs_mm": float("nan"),
        }

    d = diff[mask]
    return {
        "mean_mm": float(np.mean(d)),
        "sd_mm": float(np.std(d, ddof=0)),
        "p95_abs_mm": float(np.percentile(np.abs(d), 95)),
        "max_abs_mm": float(np.max(np.abs(d))),
    }


def _detect_blink_pits(y: np.ndarray, fs: float, min_drop_mm: float = 0.6, max_dur_s: float = 0.25) -> int:
    """Heuristic: count short, sharp downward 'pits' typical of partial occlusion.

    This is NOT a medical metric; it is a sanity indicator.
    """
    y = np.asarray(y, dtype=float)
    if len(y) < 10 or not np.isfinite(fs) or fs <= 0:
        return 0

    # robust baseline using rolling median
    win = int(max(5, round(0.5 * fs)))  # 0.5s window
    if win % 2 == 0:
        win += 1

    # rolling median via pandas (simple and robust)
    med = pd.Series(y).rolling(window=win, center=True, min_periods=1).median().to_numpy()
    drop = med - y

    pit = drop >= min_drop_mm
    # count contiguous runs shorter than max_dur_s
    max_len = int(round(max_dur_s * fs))
    count = 0
    cur = 0
    for v in pit.astype(bool):
        if v:
            cur += 1
        else:
            if 1 <= cur <= max_len:
                count += 1
            cur = 0
    if 1 <= cur <= max_len:
        count += 1
    return int(count)


def _event_check(ts: np.ndarray, y: np.ndarray, onset_s: float, pre_s: float = 2.5, post_s: float = 3.0) -> Dict[str, float]:
    """Very light PLR plausibility check around an assumed stimulus onset.

    Returns baseline, min, amplitude, latency10 (10% amplitude crossing).
    This is used as *supporting evidence* only.
    """
    ts = np.asarray(ts, dtype=float)
    y = np.asarray(y, dtype=float)
    out: Dict[str, float] = {
        "onset_s": float(onset_s),
        "baseline_mm": float("nan"),
        "min_mm": float("nan"),
        "amplitude_mm": float("nan"),
        "relative_amplitude": float("nan"),
        "time_to_min_s": float("nan"),
        "latency10_s": float("nan"),
    }

    if len(ts) < 5:
        return out

    pre_mask = (ts >= (onset_s - pre_s)) & (ts < (onset_s - 0.1))
    post_mask = (ts >= onset_s) & (ts <= (onset_s + post_s))

    if pre_mask.sum() < 5 or post_mask.sum() < 5:
        return out

    base = np.nanmedian(y[pre_mask])
    y_post = y[post_mask]
    ts_post = ts[post_mask]
    if not np.isfinite(base):
        return out

    min_val = np.nanmin(y_post)
    i_min = int(np.nanargmin(y_post))
    t_min = float(ts_post[i_min] - onset_s)
    amp = float(base - min_val)

    out["baseline_mm"] = float(base)
    out["min_mm"] = float(min_val)
    out["amplitude_mm"] = float(amp)
    out["relative_amplitude"] = float(amp / base) if base > 0 else float("nan")
    out["time_to_min_s"] = float(t_min)

    if amp > 0.05:  # avoid nonsense when no response
        thresh = base - 0.10 * amp
        crossed = np.where(y_post <= thresh)[0]
        if crossed.size > 0:
            out["latency10_s"] = float(ts_post[crossed[0]] - onset_s)

    return out


def run_validity_report(
    out_dir: str,
    raw_df: Optional[pd.DataFrame],
    interpolated_df: pd.DataFrame,
    smoothed_df: pd.DataFrame,
    fps_hint: Optional[float] = None,
    confidence_thresh: float = 0.75,
    assumed_onsets_s: Optional[List[float]] = None,
    trial_label: Optional[str] = None,
    hf_cut_hz: float = 4.0,
    calibration_truth_mm: float = 8.0,
    pipeline_params: Optional[Dict[str, float]] = None,
) -> Dict:
    """Compute metrics + save CSV/JSON/plots into out_dir."""

    os.makedirs(out_dir, exist_ok=True)

    raw_aligned, interp_aligned, smooth_aligned = _align_by_frame(raw_df, interpolated_df, smoothed_df)

    ts = interp_aligned["timestamp"].to_numpy() if "timestamp" in interp_aligned.columns else smooth_aligned.get("timestamp", pd.Series(np.arange(len(smooth_aligned)))).to_numpy()
    y_interp = interp_aligned["diameter_mm"].to_numpy(dtype=float)
    y_smooth = smooth_aligned["diameter_mm"].to_numpy(dtype=float)

    fps = float(fps_hint) if fps_hint is not None else _infer_fps_from_ts(ts)

    # Define "measured" frames (used to prove Pass1-4 does not change kept measurements)
    measured_mask = None
    # Points filled by interpolation (Pass 4), i.e. repaired samples.
    interp_points_mask = interp_aligned["is_interpolated"].astype(bool).to_numpy() if "is_interpolated" in interp_aligned.columns else None

    if raw_aligned is not None:
        y_raw = raw_aligned["diameter_mm"].to_numpy(dtype=float)
        conf = raw_aligned["confidence"].to_numpy(dtype=float) if "confidence" in raw_aligned.columns else np.full_like(y_raw, np.nan)
        bad = raw_aligned["is_bad_data"].to_numpy(dtype=bool) if "is_bad_data" in raw_aligned.columns else np.zeros_like(y_raw, dtype=bool)

        measured_mask = np.isfinite(y_raw) & (conf >= confidence_thresh) & (~bad)

        # Fallback: if the pipeline didn't provide an explicit is_interpolated mask,
        # approximate repaired points as those not considered "measured" but present in interp.
        if interp_points_mask is None:
            interp_points_mask = (~measured_mask) & np.isfinite(y_interp)

        # "Pass 1–4 does not modify retained measurements" check
        r_raw_interp_measured = _corr(y_raw[measured_mask], y_interp[measured_mask])
        rmse_raw_interp_measured = _rmse(y_raw[measured_mask], y_interp[measured_mask])
    else:
        r_raw_interp_measured = float("nan")
        rmse_raw_interp_measured = float("nan")
        measured_mask = np.isfinite(y_interp)
        if interp_points_mask is None:
            interp_points_mask = np.zeros_like(y_interp, dtype=bool)

    # Smoothing distortion
    r_interp_smooth = _corr(y_interp, y_smooth)
    rmse_interp_smooth = _rmse(y_interp, y_smooth)
    mae_interp_smooth = _mae(y_interp, y_smooth)

    # Normalized RMSE w.r.t. interp signal range
    y_range = float(np.nanmax(y_interp) - np.nanmin(y_interp)) if np.isfinite(y_interp).any() else float("nan")
    nrmse_interp_smooth = float(rmse_interp_smooth / y_range) if (np.isfinite(rmse_interp_smooth) and np.isfinite(y_range) and y_range > 0) else float("nan")


    # Smoothing distortion summary stats
    diff_smooth_minus_interp = y_smooth - y_interp
    dist = _distortion_summary(diff_smooth_minus_interp)

    # Global waveform preservation (very simple sanity indicators)
    global_min_interp_mm = float(np.nanmin(y_interp)) if np.isfinite(y_interp).any() else float("nan")
    global_min_smooth_mm = float(np.nanmin(y_smooth)) if np.isfinite(y_smooth).any() else float("nan")
    delta_global_min_mm = float(global_min_smooth_mm - global_min_interp_mm) if (np.isfinite(global_min_smooth_mm) and np.isfinite(global_min_interp_mm)) else float("nan")

    # --- Data quality metrics (make the conceptual distinction explicit) ---
    # 1) "Flagged before repair" = frames the pipeline marked as bad (Pass 1–3) *before* interpolation.
    flagged_before_repair = np.zeros(len(y_interp), dtype=bool)
    if "is_bad_data" in interp_aligned.columns:
        flagged_before_repair |= interp_aligned["is_bad_data"].astype(bool).to_numpy()
    flagged_before_repair |= ~np.isfinite(y_interp)

    flagged_pct_before_repair = float(100.0 * flagged_before_repair.sum() / len(flagged_before_repair)) if len(flagged_before_repair) else float("nan")

    # 2) "Flagged remaining after repair" = frames still invalid after interpolation/smoothing.
    flagged_remaining = np.zeros(len(y_smooth), dtype=bool)
    if "is_bad_data" in smooth_aligned.columns:
        flagged_remaining |= smooth_aligned["is_bad_data"].astype(bool).to_numpy()
    flagged_remaining |= ~np.isfinite(y_smooth)

    flagged_pct_remaining_final = float(100.0 * flagged_remaining.sum() / len(flagged_remaining)) if len(flagged_remaining) else float("nan")

    # Interpolation (repair) burden
    interpolated_filled_pct = float(100.0 * interp_points_mask.sum() / len(interp_points_mask)) if len(interp_points_mask) else float("nan")
    max_gap_frames, max_gap_s = _max_interp_gap(interp_points_mask, fs=fps)

    # PSD comparisons
    f_i, p_i = _welch_psd(y_interp, fs=fps)
    f_s, p_s = _welch_psd(y_smooth, fs=fps)

    # compute high frequency power ratios (>3 Hz) if possible
    def band_power(f: np.ndarray, p: np.ndarray, fmin: float, fmax: Optional[float] = None) -> float:
        if f.size == 0 or p.size == 0:
            return float("nan")
        m = f >= fmin
        if fmax is not None:
            m &= (f <= fmax)
        if m.sum() == 0:
            return float("nan")
        return float(np.trapz(p[m], f[m]))

    total_i = band_power(f_i, p_i, 0.0, None)
    total_s = band_power(f_s, p_s, 0.0, None)
    hf_i = band_power(f_i, p_i, float(hf_cut_hz), None)
    hf_s = band_power(f_s, p_s, float(hf_cut_hz), None)

    hf_ratio = float(hf_s / hf_i) if (np.isfinite(hf_s) and np.isfinite(hf_i) and hf_i > 0) else float("nan")
    hf_reduction_pct = float(100.0 * (1.0 - hf_ratio)) if np.isfinite(hf_ratio) else float("nan")
    total_ratio = float(total_s / total_i) if (np.isfinite(total_s) and np.isfinite(total_i) and total_i > 0) else float("nan")

    # Blink pit count (heuristic)
    pit_count = _detect_blink_pits(y_interp, fs=fps) if np.isfinite(fps) else 0

    # Optional event plausibility checks (supporting evidence only)
    events: List[Dict[str, float]] = []
    if assumed_onsets_s:
        for onset in assumed_onsets_s:
            events.append(_event_check(ts, y_smooth, onset_s=float(onset)))

    # Optional calibration metrics (only when the folder name looks like a calibration run)
    is_calibration = False
    tl = (trial_label or os.path.basename(os.path.abspath(out_dir))).lower()
    if 'calibration' in tl:
        is_calibration = True

    calib_stats: Optional[Dict[str, float]] = None
    if is_calibration and np.isfinite(calibration_truth_mm):
        cal_mask = np.isfinite(y_smooth)
        if interp_points_mask is not None:
            # exclude filled-in points for calibration stability
            cal_mask &= (~interp_points_mask)
        if measured_mask is not None:
            cal_mask &= measured_mask

        if cal_mask.sum() >= 3:
            err = y_smooth[cal_mask] - float(calibration_truth_mm)
            mean_mm = float(np.mean(y_smooth[cal_mask]))
            sd_mm = float(np.std(y_smooth[cal_mask], ddof=0))
            mae_mm = float(np.mean(np.abs(err)))
            rmse_mm = float(np.sqrt(np.mean(err ** 2)))
            bias_mm = float(np.mean(err))
            lo95 = float(mean_mm - 1.96 * sd_mm)
            hi95 = float(mean_mm + 1.96 * sd_mm)
            calib_stats = {
                'truth_mm': float(calibration_truth_mm),
                'n_samples': float(cal_mask.sum()),
                'mean_mm': mean_mm,
                'sd_mm': sd_mm,
                'bias_mm': bias_mm,
                'mae_mm': mae_mm,
                'rmse_mm': rmse_mm,
                'limits95_low_mm': lo95,
                'limits95_high_mm': hi95,
            }


    report: Dict = {
        "trial_label": trial_label or os.path.basename(os.path.abspath(out_dir)),
        "n_frames": int(len(y_interp)),
        "duration_s": float(ts[-1] - ts[0]) if len(ts) > 1 else float("nan"),
        "fps_est": float(fps),
        "confidence_thresh": float(confidence_thresh),
        # NEW: clearer naming
        "flagged_pct_before_repair": float(flagged_pct_before_repair),
        "flagged_pct_remaining_final": float(flagged_pct_remaining_final),
        "interpolated_filled_pct": float(interpolated_filled_pct),
        "max_interpolated_filled_gap_frames": int(max_gap_frames),
        "max_interpolated_filled_gap_s": float(max_gap_s),
        # Backwards-compatible aliases (deprecated)
        "flagged_pct_final": float(flagged_pct_remaining_final),
        "interpolated_pct": float(interpolated_filled_pct),
        "max_interpolated_gap_frames": int(max_gap_frames),
        "max_interpolated_gap_s": float(max_gap_s),
        "r_raw_vs_interp_on_measured": float(r_raw_interp_measured),
        "rmse_raw_vs_interp_on_measured_mm": float(rmse_raw_interp_measured),
        "r_interp_vs_smooth": float(r_interp_smooth),
        "rmse_interp_vs_smooth_mm": float(rmse_interp_smooth),
        "mae_interp_vs_smooth_mm": float(mae_interp_smooth),
        "nrmse_interp_vs_smooth": float(nrmse_interp_smooth),
        "smooth_minus_interp_mean_mm": float(dist['mean_mm']),
        "smooth_minus_interp_sd_mm": float(dist['sd_mm']),
        "smooth_minus_interp_p95_abs_mm": float(dist['p95_abs_mm']),
        "smooth_minus_interp_max_abs_mm": float(dist['max_abs_mm']),
        "global_min_interp_mm": float(global_min_interp_mm),
        "global_min_smooth_mm": float(global_min_smooth_mm),
        "delta_global_min_mm": float(delta_global_min_mm),
        "psd_highfreq_cut_hz": float(hf_cut_hz),
        "psd_highfreq_power_reduction_pct": float(hf_reduction_pct),
        "pipeline_params": pipeline_params or {},
        "calibration_stats": calib_stats,
        "psd_total_power_ratio_smooth_over_interp": float(total_ratio),
        "psd_highfreq_power_ratio": float(hf_ratio),
        "psd_highfreq_power_ratio_gt3Hz": float(hf_ratio),
        "blink_pit_count": int(pit_count),
        "assumed_event_checks": events,
    }

    # Save JSON
    json_path = os.path.join(out_dir, "validity_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save CSV (flattened)
    flat = report.copy()
    # store events compactly
    if events:
        for i, ev in enumerate(events, start=1):
            for k, v in ev.items():
                flat[f"event{i}_{k}"] = v

    # flatten pipeline params (JSON stays structured)
    params = report.get('pipeline_params') or {}
    if isinstance(params, dict):
        for k, v in params.items():
            flat[f"param_{k}"] = v

    # flatten calibration stats (if present)
    cstats = report.get('calibration_stats')
    if isinstance(cstats, dict):
        for k, v in cstats.items():
            flat[f"calib_{k}"] = v

    flat.pop("assumed_event_checks", None)
    flat.pop("pipeline_params", None)
    # keep calibration_stats in JSON only; CSV has flattened calib_* columns
    # but remove the dict itself to avoid stringifying it
    if 'calibration_stats' in flat and isinstance(flat['calibration_stats'], dict):
        flat.pop('calibration_stats', None)

    csv_path = os.path.join(out_dir, "validity_report.csv")
    pd.DataFrame([flat]).to_csv(csv_path, index=False)

    # Plots
    _plot_overlay(out_dir, raw_aligned, interp_aligned, smooth_aligned, interp_points_mask)
    _plot_residuals(out_dir, ts, y_interp, y_smooth)
    _plot_psd(out_dir, f_i, p_i, f_s, p_s)

    return report


def _plot_overlay(out_dir: str, raw_df: Optional[pd.DataFrame], interp_df: pd.DataFrame, smooth_df: pd.DataFrame, interp_points_mask: np.ndarray) -> None:
    plt.figure(figsize=(12, 6))

    if "timestamp" in interp_df.columns:
        t = interp_df["timestamp"].to_numpy(dtype=float)
    else:
        t = np.arange(len(interp_df), dtype=float)

    # Raw points
    if raw_df is not None and "timestamp" in raw_df.columns:
        tr = raw_df["timestamp"].to_numpy(dtype=float)
        yr = raw_df["diameter_mm"].to_numpy(dtype=float)
        plt.scatter(tr, yr, s=8, alpha=0.35, label="Raw (video measurement)")

    # Interpolated + smoothed
    yi = interp_df["diameter_mm"].to_numpy(dtype=float)
    ys = smooth_df["diameter_mm"].to_numpy(dtype=float)
    plt.plot(t, yi, linewidth=1.5, label="Pass 4 (interpolated)")
    plt.plot(t, ys, linewidth=2.0, label="Pass 6 (smoothed)")

    # highlight interpolated points
    if interp_points_mask is not None and interp_points_mask.any() and len(interp_points_mask) == len(t):
        plt.scatter(t[interp_points_mask], yi[interp_points_mask], s=10, alpha=0.5, label="Interpolated points")

    plt.xlabel("Time (s)")
    plt.ylabel("Pupil diameter (mm)")
    plt.title("Validity overlay: raw vs interpolated vs smoothed")
    plt.legend(loc="best")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "validity_overlay.png")
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_residuals(out_dir: str, t: np.ndarray, y_interp: np.ndarray, y_smooth: np.ndarray) -> None:
    diff = y_smooth - y_interp
    mask = np.isfinite(diff) & np.isfinite(t)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t[mask], diff[mask])
    plt.axhline(0, linewidth=1)
    plt.ylabel("Smooth − Interp (mm)")
    plt.title("Smoothing distortion over time")

    plt.subplot(2, 1, 2)
    d = diff[mask]
    if d.size:
        plt.hist(d, bins=60)
    plt.xlabel("Smooth − Interp (mm)")
    plt.ylabel("Count")
    plt.title("Distribution of smoothing distortion")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "validity_residuals.png")
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_psd(out_dir: str, f_i: np.ndarray, p_i: np.ndarray, f_s: np.ndarray, p_s: np.ndarray) -> None:
    plt.figure(figsize=(12, 5))
    if f_i.size and p_i.size:
        plt.semilogy(f_i, p_i, label="Pass 4 (interpolated)")
    if f_s.size and p_s.size:
        plt.semilogy(f_s, p_s, label="Pass 6 (smoothed)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power spectral density")
    plt.title("PSD comparison (Welch)")
    plt.legend(loc="best")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "validity_psd.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
