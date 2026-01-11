

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from scripts.others.util import dprint
import scripts.others.graph as graph

from scripts.preProcessing.firstPass import confidenceFilter
from scripts.preProcessing.secondPass import removeSusBio
from scripts.preProcessing.thirdPass import madFilter
from scripts.preProcessing.fourthPassLinear import interpolateData
from scripts.preProcessing.fifthPass import averagePLRGraphs
from scripts.preProcessing.sixthPass import savgolSmoothing


# ---- Shared configuration ----------------------------------------------------

_BASE_RES = (1920, 1080)
_BASE_PX_TO_MM = 30.0  # pixels per mm at 1920x1080, based on your calibration


def parse_resolution(res_str: str) -> Tuple[int, int]:
    m = re.match(r"^\s*(\d+)\s*x\s*(\d+)\s*$", res_str)
    if not m:
        raise ValueError(f"Invalid resolution format: {res_str} (expected like 1920x1080)")
    return int(m.group(1)), int(m.group(2))


def px_to_mm_from_resolution(width: int, height: int) -> float:
    """Estimate px/mm by scaling from the 1920x1080 calibration.

    Assumes the same camera + lens + working distance.
    """
    _, base_h = _BASE_RES
    return _BASE_PX_TO_MM * (height / base_h)


def infer_fps_from_timestamps(df: pd.DataFrame) -> int:
    """Infer fps from the median timestamp difference."""
    if "timestamp" not in df.columns:
        raise ValueError("Cannot infer fps: 'timestamp' column missing.")

    dt = pd.to_numeric(df["timestamp"], errors="coerce").diff().dropna()
    if dt.empty:
        raise ValueError("Cannot infer fps: not enough timestamp samples.")

    median_dt = float(dt.median())
    if median_dt <= 0:
        raise ValueError("Cannot infer fps: invalid timestamp deltas.")

    return int(round(1.0 / median_dt))


@dataclass
class ProcessingConfig:
    fps: float
    resolution: Tuple[int, int]
    px_to_mm: Optional[float] = None
    confidence_thresh: float = 0.75

    # Pass parameters
    max_gap_ms: int = 400
    savgol_window_ms: int = 150

    def __post_init__(self) -> None:
        if self.px_to_mm is None:
            self.px_to_mm = px_to_mm_from_resolution(self.resolution[0], self.resolution[1])


# ---- Trial folder parsing ----------------------------------------------------

TRIAL_RE = re.compile(
    r"^PLR_(?P<user>.+)_(?P<eye>[LR])_(?P<res>\d+x\d+)_(?P<fps>\d+)_(?P<trial>\d+)$"
)


@dataclass(frozen=True)
class TrialMeta:
    user: str
    eye: str
    res: str
    fps: int
    trial: int
    dirname: str


def _parse_trial_dirname(dirname: str) -> Optional[TrialMeta]:
    m = TRIAL_RE.match(dirname)
    if not m:
        return None
    return TrialMeta(
        user=m.group("user"),
        eye=m.group("eye"),
        res=m.group("res"),
        fps=int(m.group("fps")),
        trial=int(m.group("trial")),
        dirname=dirname,
    )


# ---- Helpers ----------------------------------------------------------------


def _ensure_flag_column(df: pd.DataFrame) -> pd.DataFrame:
    if "is_bad_data" not in df.columns:
        df["is_bad_data"] = False
    return df


def _flagged_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    if "is_bad_data" in df.columns:
        mask |= df["is_bad_data"].astype(bool)
    if "diameter_mm" in df.columns:
        mask |= df["diameter_mm"].isna()
    return mask


def _pct(x: int, total: int) -> float:
    return (100.0 * x / total) if total > 0 else 0.0


def _report_stage(stage: str, df: pd.DataFrame, prev_flagged: Optional[pd.Series]) -> pd.Series:
    total = len(df)
    cur_flagged = _flagged_mask(df)
    cur_cnt = int(cur_flagged.sum())

    if prev_flagged is None:
        new_cnt = cur_cnt
    else:
        new_cnt = int((cur_flagged & ~prev_flagged).sum())

    dprint(
        f"{stage}: flagged {cur_cnt}/{total} ({_pct(cur_cnt, total):.1f}%), "
        f"newly flagged in this pass: {new_cnt} ({_pct(new_cnt, total):.1f}%)"
    )
    return cur_flagged


def _quality_summary(total_frames: int, flagged_total: int, interpolated: int) -> None:
    flagged_pct = _pct(flagged_total, total_frames)
    interp_pct = _pct(interpolated, total_frames)

    dprint(
        f"Quality summary: flagged={flagged_total}/{total_frames} ({flagged_pct:.1f}%), "
        f"interpolated={interpolated}/{total_frames} ({interp_pct:.1f}%)."
    )

    if flagged_pct > 25.0 or interp_pct > 25.0:
        dprint("WARNING: >25% flagged or interpolated. Consider re-recording.")
    else:
        dprint("OK: <=25% flagged and interpolated.")


def _load_raw_csv(trial_dir: str) -> pd.DataFrame:
    raw_csv = os.path.join(trial_dir, "raw.csv")
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"Could not find raw.csv in: {trial_dir}")
    return _ensure_flag_column(pd.read_csv(raw_csv))


# ---- Core pipeline -----------------------------------------------------------


def run_passes_1_to_4(
    df: pd.DataFrame,
    config: ProcessingConfig,
) -> Tuple[pd.DataFrame, Dict[str, int], pd.Series]:
    """Run Pass 1–4 and return:
    - output DataFrame after interpolation
    - counts dict: total / interpolated / flagged_after_pass4
    - flagged mask after pass4
    """

    total = len(df)
    prev_flagged: Optional[pd.Series] = None

    prev_flagged = _report_stage("Raw (before Pass 1)", df, prev_flagged)

    df = confidenceFilter(df, confidence_thresh=config.confidence_thresh)
    prev_flagged = _report_stage("Pass 1 (confidence filter)", df, prev_flagged)

    df = removeSusBio(df, fps=config.fps)
    prev_flagged = _report_stage("Pass 2 (biology/blink checks)", df, prev_flagged)

    df = madFilter(df)
    prev_flagged = _report_stage("Pass 3 (MAD outlier filter)", df, prev_flagged)

    pre_nan = df["diameter_mm"].isna()
    df = interpolateData(df, fps=config.fps, max_gap_ms=config.max_gap_ms)
    post_nan = df["diameter_mm"].isna()
    interpolated_cnt = int((pre_nan & ~post_nan).sum())

    prev_flagged = _report_stage("Pass 4 (linear interpolation)", df, prev_flagged)

    counts = {
        "total": total,
        "interpolated": interpolated_cnt,
        "flagged_after_pass4": int(_flagged_mask(df).sum()),
    }
    return df, counts, prev_flagged


def run_pass_6(df: pd.DataFrame, config: ProcessingConfig) -> Tuple[pd.DataFrame, pd.Series]:
    prev_flagged = _flagged_mask(df)
    df = savgolSmoothing(df, fps=config.fps, target_window_ms=config.savgol_window_ms)
    prev_flagged = _report_stage("Pass 6 (Savitzky–Golay smoothing)", df, prev_flagged)
    return df, prev_flagged


def process_single_trial(
    trial_dir: str,
    config: ProcessingConfig,
    show_plot: bool,
    recompute_mm: bool,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Process one trial directory and write outputs into the same folder."""

    raw_df = _load_raw_csv(trial_dir)

    if recompute_mm:
        if config.px_to_mm is None:
            raise ValueError("recompute_mm requested but config.px_to_mm is None")
        raw_df["diameter_mm"] = pd.to_numeric(raw_df["diameter"], errors="coerce") / float(config.px_to_mm)

    df_after_4, counts, _ = run_passes_1_to_4(raw_df.copy(), config)
    out_csv_p4 = os.path.join(trial_dir, "processed_interpolated.csv")
    df_after_4.to_csv(out_csv_p4, index=False)
    dprint(f"Saved Pass 4 output: {out_csv_p4}")

    df_final, _ = run_pass_6(df_after_4.copy(), config)

    out_csv = os.path.join(trial_dir, "processed.csv")
    df_final.to_csv(out_csv, index=False)
    dprint(f"Saved processed CSV: {out_csv}")

    out_plot = os.path.join(trial_dir, "processedPlot.png")
    graph.plotResults(df_final, savePath=out_plot, showPlot=show_plot, showMm=True)
    dprint(f"Saved plot: {out_plot}")

    _quality_summary(
        total_frames=counts["total"],
        flagged_total=int(_flagged_mask(df_final).sum()),
        interpolated=counts["interpolated"],
    )

    return df_final, counts


def _find_trials_in_parent(parent_dir: str) -> List[Tuple[str, TrialMeta]]:
    trials: List[Tuple[str, TrialMeta]] = []
    for name in os.listdir(parent_dir):
        full = os.path.join(parent_dir, name)
        if not os.path.isdir(full):
            continue
        meta = _parse_trial_dirname(name)
        if not meta:
            continue
        if not os.path.exists(os.path.join(full, "raw.csv")):
            continue
        trials.append((full, meta))
    return trials


def process_group_average(
    parent_dir: str,
    group_key: Tuple[str, str, str, int],
    trial_items: List[Tuple[str, TrialMeta]],
    show_plot: bool,
    recompute_mm: bool,
    max_gap_ms: int,
    savgol_window_ms: int,
    confidence_thresh: float,
) -> None:
    """Process the latest 2 trials in this group and write one averaged output folder."""

    user, eye, res, fps = group_key

    trial_items = sorted(trial_items, key=lambda x: x[1].trial)
    chosen = trial_items[-2:] if len(trial_items) >= 2 else trial_items
    if not chosen:
        return

    dprint(f"\n=== Group: user={user}, eye={eye}, res={res}, fps={fps} | trials={[m.trial for _, m in chosen]} ===")

    width, height = parse_resolution(res)
    px_to_mm = px_to_mm_from_resolution(width, height)
    config = ProcessingConfig(
        fps=fps,
        resolution=(width, height),
        px_to_mm=px_to_mm,
        confidence_thresh=confidence_thresh,
        max_gap_ms=max_gap_ms,
        savgol_window_ms=savgol_window_ms,
    )

    dfs_after_interp: List[pd.DataFrame] = []
    counts_list: List[Dict[str, int]] = []

    for trial_dir, meta in chosen:
        dprint(f"\n--- Processing trial folder: {meta.dirname} ---")
        df = _load_raw_csv(trial_dir)

        if recompute_mm:
            df["diameter_mm"] = pd.to_numeric(df["diameter"], errors="coerce") / float(config.px_to_mm)

        df_after_4, counts, _ = run_passes_1_to_4(df, config)

        out_csv = os.path.join(trial_dir, "processed_interpolated.csv")
        df_after_4.to_csv(out_csv, index=False)
        dprint(f"Saved Pass 4 output: {out_csv}")

        df_final, _ = run_pass_6(df_after_4.copy(), config)

        out_csv_final = os.path.join(trial_dir, "processed.csv")
        df_final.to_csv(out_csv_final, index=False)

        out_plot_final = os.path.join(trial_dir, "processedPlot.png")
        graph.plotResults(df_final, savePath=out_plot_final, showPlot=False, showMm=True)

        _quality_summary(
            total_frames=counts["total"],
            flagged_total=int(_flagged_mask(df_final).sum()),
            interpolated=counts["interpolated"],
        )

        dfs_after_interp.append(df_after_4)
        counts_list.append(counts)

    averaged_df: pd.DataFrame
    if len(dfs_after_interp) == 2:
        has_nan_1 = bool(dfs_after_interp[0]["diameter_mm"].isna().any())
        has_nan_2 = bool(dfs_after_interp[1]["diameter_mm"].isna().any())

        if (not has_nan_1) and (not has_nan_2):
            averaged_df = averagePLRGraphs(dfs_after_interp[0], dfs_after_interp[1], px_to_mm=config.px_to_mm, fps=fps)
            if averaged_df is None:
                dprint("Pass 5 skipped: averaging failed. Using first trial as output.")
                averaged_df = dfs_after_interp[0].copy()
        else:
            dprint("Pass 5 skipped: one or both trials still contain NaNs after interpolation.")
            averaged_df = dfs_after_interp[0].copy()
    else:
        averaged_df = dfs_after_interp[0].copy()

    prev_p5 = _flagged_mask(averaged_df)
    _report_stage("Pass 5 (averaging)", averaged_df, prev_p5)

    averaged_after_5 = averaged_df.copy()
    averaged_df, _ = run_pass_6(averaged_df, config)

    out_dirname = f"PLR_{user}_{eye}_{res}_{fps}_AVG"
    out_dir = os.path.join(parent_dir, out_dirname)
    os.makedirs(out_dir, exist_ok=True)

    out_csv_p5 = os.path.join(out_dir, "processed_avg_interpolated.csv")
    averaged_after_5.to_csv(out_csv_p5, index=False)

    out_csv = os.path.join(out_dir, "processed_avg.csv")
    averaged_df.to_csv(out_csv, index=False)

    out_plot = os.path.join(out_dir, "processedAvgPlot.png")
    graph.plotResults(averaged_df, savePath=out_plot, showPlot=show_plot, showMm=True)

    dprint(f"Saved averaged CSV: {out_csv}")
    dprint(f"Saved averaged plot: {out_plot}")

    total = len(averaged_df)
    flagged_total = int(_flagged_mask(averaged_df).sum())
    interpolated_total = sum(c["interpolated"] for c in counts_list) // max(1, len(counts_list))
    _quality_summary(total, flagged_total, interpolated_total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        required=True,
        help=(
            "Either: (a) a single trial directory containing raw.csv, or "
            "(b) a parent directory containing multiple PLR_* trial folders."
        ),
    )
    parser.add_argument("--fps", type=int, default=None, help="Override fps for single-trial mode.")
    parser.add_argument(
        "--resolution",
        type=str,
        default=None,
        help="Override resolution for single-trial mode (e.g., 1920x1080).",
    )
    parser.add_argument(
        "--recompute_mm",
        action="store_true",
        help="Recompute diameter_mm from diameter pixels using px_to_mm derived from resolution.",
    )
    parser.add_argument("--max_gap_ms", type=int, default=400)
    parser.add_argument("--savgol_window_ms", type=int, default=150)

    parser.add_argument(
        "--no_show_plot",
        action="store_true",
        help="Disable interactive matplotlib plot windows (still saves PNG).",
    )
    parser.add_argument(
        "--show_all_plots",
        action="store_true",
        help="Batch mode only: show plots for each processed trial (not recommended).",
    )

    args = parser.parse_args()

    data_path = args.data
    show_plot = not args.no_show_plot

    # Single-trial mode
    if os.path.exists(os.path.join(data_path, "raw.csv")):
        folder_name = os.path.basename(os.path.abspath(data_path))
        meta = _parse_trial_dirname(folder_name)

        fps = args.fps
        if fps is None:
            if meta is not None:
                fps = meta.fps
            else:
                df_tmp = pd.read_csv(os.path.join(data_path, "raw.csv"))
                fps = infer_fps_from_timestamps(df_tmp)

        if args.resolution is not None:
            width, height = parse_resolution(args.resolution)
        elif meta is not None:
            width, height = parse_resolution(meta.res)
        else:
            raise ValueError("Resolution not provided and could not be inferred. Use --resolution 1920x1080.")

        config = ProcessingConfig(
            fps=float(fps),
            resolution=(width, height),
            px_to_mm=px_to_mm_from_resolution(width, height),
            confidence_thresh=0.75,
            max_gap_ms=args.max_gap_ms,
            savgol_window_ms=args.savgol_window_ms,
        )

        process_single_trial(
            trial_dir=data_path,
            config=config,
            show_plot=show_plot,
            recompute_mm=args.recompute_mm,
        )
        return

    # Batch mode
    trials = _find_trials_in_parent(data_path)
    if not trials:
        raise ValueError(f"No trial folders found under: {data_path}")

    groups: Dict[Tuple[str, str, str, int], List[Tuple[str, TrialMeta]]] = {}
    for trial_dir, meta in trials:
        key = (meta.user, meta.eye, meta.res, meta.fps)
        groups.setdefault(key, []).append((trial_dir, meta))

    dprint(f"Found {len(trials)} trial folders forming {len(groups)} groups.")

    for key, items in sorted(groups.items(), key=lambda kv: kv[0]):
        process_group_average(
            parent_dir=data_path,
            group_key=key,
            trial_items=items,
            show_plot=show_plot,
            recompute_mm=args.recompute_mm,
            max_gap_ms=args.max_gap_ms,
            savgol_window_ms=args.savgol_window_ms,
            confidence_thresh=0.75,
        )

        if args.show_all_plots and show_plot:
            for trial_dir, meta in items[-2:]:
                p = os.path.join(trial_dir, "processed.csv")
                if os.path.exists(p):
                    df = pd.read_csv(p)
                    graph.plotResults(df, savePath=None, showPlot=True, showMm=True, title=f"{meta.dirname}")


if __name__ == "__main__":
    main()
