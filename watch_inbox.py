
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import select
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
import pandas as pd


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".h264"}
RESERVED_SUBDIRS = {"_archive", "_processing"}


def wait_until_stable(path: Path, stable_secs: float = 1.0, timeout_secs: float = 300.0) -> bool:
    """Return True when file size has been unchanged for `stable_secs` seconds."""
    start = time.time()
    last_size = -1
    stable_since = None

    while True:
        if not path.exists():
            return False

        try:
            size = path.stat().st_size
        except OSError:
            size = -1

        now = time.time()

        if size == last_size and size > 0:
            if stable_since is None:
                stable_since = now
            if (now - stable_since) >= stable_secs:
                return True
        else:
            stable_since = None
            last_size = size

        if (now - start) > timeout_secs:
            return False

        time.sleep(0.2)


def _is_candidate_video(p: Path, inbox_root: Path) -> bool:
    if not p.is_file():
        return False
    if p.parent != inbox_root:
        return False
    if p.suffix.lower() not in VIDEO_EXTS:
        return False
    if p.name.endswith(".part"):
        return False
    if p.name in RESERVED_SUBDIRS:
        return False
    return True


def _run_main_raw_only(python_exe: Path, repo_dir: Path, video_path: Path) -> None:
    """Run main.py raw-extraction only (no preprocess) for ONE video."""
    vi_dir = repo_dir / "videoImplement"
    subprocess.run(
        [str(python_exe), "main.py", "--input", str(video_path), "--no_raw_plot", "--no_preprocess"],
        cwd=str(vi_dir),
        check=True,
    )


def _run_process_single_trial(
    python_exe: Path,
    repo_dir: Path,
    trial_dir: Path,
    show_plots: bool,
) -> None:
    """Run process.py in single-trial mode against `trial_dir` (contains raw.csv)."""
    vi_dir = repo_dir / "videoImplement"
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "MacOSX")

    cmd = [
        str(python_exe),
        "process.py",
        "--data",
        str(trial_dir),
        "--resolution",
        "1920x1080",
    ]
    if not show_plots:
        cmd.append("--no_show_plot")

    subprocess.run(cmd, cwd=str(vi_dir), check=True, env=env)


def _safe_unique_dir(parent: Path, name: str) -> Path:
    """Return a unique subdir path under parent (avoid collisions)."""
    cand = parent / name
    if not cand.exists():
        return cand
    i = 2
    while True:
        cand2 = parent / f"{name}_{i}"
        if not cand2.exists():
            return cand2
        i += 1


def _build_session_average_raw(trial_dirs: List[Path], out_average_dir: Path) -> Path:
    """Create ONE averaged raw.csv (across all trials) into out_average_dir.

    Output folder layout:
      out_average_dir/raw.csv

    Averaging method (robust + simple):
      - Align by frame_id.
      - Truncate to the shortest trial (min max frame_id).
      - Treat is_bad_data==True as missing (NaN) before averaging.
      - Average diameter_mm ignoring NaNs.
    """
    if not trial_dirs:
        raise ValueError("No trials available to average.")

    out_average_dir.mkdir(parents=True, exist_ok=True)
    raw_out = out_average_dir / "raw.csv"

    dfs = []
    max_frames = []
    for td in trial_dirs:
        raw_csv = td / "raw.csv"
        if not raw_csv.exists():
            raise FileNotFoundError(f"Missing raw.csv in trial: {td}")
        df = pd.read_csv(raw_csv)
        if "frame_id" not in df.columns:
            df = df.reset_index().rename(columns={"index": "frame_id"})
        if "is_bad_data" not in df.columns:
            df["is_bad_data"] = False
        if "diameter_mm" not in df.columns:
            raise ValueError(f"raw.csv missing diameter_mm: {raw_csv}")

        df = df.sort_values("frame_id").reset_index(drop=True)
        max_frames.append(int(df["frame_id"].max()))
        dfs.append(df)

    min_max_frame = int(min(max_frames))
    frame_ids = np.arange(0, min_max_frame + 1, dtype=int)

    df0 = dfs[0].set_index("frame_id").reindex(frame_ids)
    timestamps = df0["timestamp"].to_numpy() if "timestamp" in df0.columns else frame_ids.astype(float)

 
    px_to_mm_est = 30.0  
    try:
        if 'diameter' in dfs[0].columns and 'diameter_mm' in dfs[0].columns:
            r = pd.to_numeric(dfs[0]['diameter'], errors='coerce') / pd.to_numeric(dfs[0]['diameter_mm'], errors='coerce')
            r = r.replace([np.inf, -np.inf], np.nan).dropna()
            if not r.empty:
                v = float(r.median())
                if np.isfinite(v) and v > 0:
                    px_to_mm_est = v
    except Exception:
        pass

    mm_stack = []
    conf_stack = []
    for df in dfs:
        dfi = df.set_index("frame_id").reindex(frame_ids)
        mm = dfi["diameter_mm"].astype(float)
        bad = dfi["is_bad_data"].astype(bool)
        mm = mm.mask(bad)
        mm_stack.append(mm.to_numpy())

        if "confidence" in dfi.columns:
            conf = dfi["confidence"].astype(float)
            conf = conf.mask(bad)
            conf_stack.append(conf.to_numpy())

    mm_stack = np.vstack(mm_stack) 
    mean_mm = np.nanmean(mm_stack, axis=0)
    is_bad = np.isnan(mean_mm)

    mean_px = mean_mm * float(px_to_mm_est)

    if conf_stack:
        conf_stack = np.vstack(conf_stack)
        mean_conf = np.nanmean(conf_stack, axis=0)
    else:
        mean_conf = np.full_like(mean_mm, fill_value=np.nan, dtype=float)

    out_df = pd.DataFrame(
        {
            "frame_id": frame_ids,
            "timestamp": timestamps,
            "diameter": mean_px,  
            "confidence": mean_conf,
            "is_bad_data": is_bad,
            "diameter_mm": mean_mm,
        }
    )
    out_df.to_csv(raw_out, index=False)
    return out_average_dir


@dataclass
class Config:
    inbox: Path
    archive: Path
    repo: Path
    python: Path
    poll_interval: float = 0.5


@dataclass
class Session:
    root: Path
    trial_root: Path
    average_dir: Path
    trials: List[Path]
    seen_files: Set[str]


def _create_new_session(repo_dir: Path) -> Session:
    vi_dir = repo_dir / "videoImplement"
    sessions_root = vi_dir / "sessions"
    sessions_root.mkdir(parents=True, exist_ok=True)

    sid = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    session_root = sessions_root / sid
    trial_root = session_root / "trial"
    avg_dir = session_root / "average"

    trial_root.mkdir(parents=True, exist_ok=True)
    avg_dir.mkdir(parents=True, exist_ok=True)

    return Session(root=session_root, trial_root=trial_root, average_dir=avg_dir, trials=[], seen_files=set())


def _ingest_one_video(cfg: Config, session: Session, inbox_video: Path) -> None:
    """Run main.py -> move raw folder into session/trial/<stem> -> archive video."""
    processing_dir = cfg.inbox / "_processing"
    processing_dir.mkdir(parents=True, exist_ok=True)

    processing_video = processing_dir / inbox_video.name
    shutil.move(str(inbox_video), str(processing_video))

    _run_main_raw_only(cfg.python, cfg.repo, processing_video)

    vi_dir = cfg.repo / "videoImplement"
    data_root = vi_dir / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    produced_trial = (data_root / processing_video.stem).resolve()
    if not produced_trial.exists():
        raise FileNotFoundError(f"Expected trial folder not found: {produced_trial}")

    dest_trial = _safe_unique_dir(session.trial_root, processing_video.stem)
    shutil.move(str(produced_trial), str(dest_trial))
    session.trials.append(dest_trial)

    cfg.archive.mkdir(parents=True, exist_ok=True)
    shutil.move(str(processing_video), str(cfg.archive / processing_video.name))


def _process_session(cfg: Config, session: Session) -> None:
    """Process all trials + create ONE average + process it (with interactive plots)."""
    if not session.trials:
        print("[session] No trials captured; nothing to process.")
        return

    print(f"[session] Processing {len(session.trials)} trial(s)...")
    for td in session.trials:
        print(f"[trial] process.py (no windows): {td.name}")
        _run_process_single_trial(cfg.python, cfg.repo, td, show_plots=False)


    if session.average_dir.exists():
        for child in session.average_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)

    _build_session_average_raw(session.trials, session.average_dir)
    print(f"[average] Built averaged raw.csv from {len(session.trials)} trial(s).")

    print("[average] process.py (WITH interactive windows). Close plots to finish.")
    _run_process_single_trial(cfg.python, cfg.repo, session.average_dir, show_plots=True)

    print(f"[session] Done. Session folder: {session.root}")


def _print_help() -> None:
    print("\nCommands:")
    print("  status   - show how many trials have been captured this session")
    print("  process  - process all trials + one session average, then exit")
    print("  quit     - exit without processing")
    print("  help     - show this help\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inbox",
        default=os.path.expanduser("~/plr_inbox"),
        help="Mac folder where Pi uploads videos (default: ~/plr_inbox)",
    )
    parser.add_argument(
        "--archive",
        default=os.path.expanduser("~/plr_inbox/_archive"),
        help="Where to move source videos after main.py (default: ~/plr_inbox/_archive)",
    )
    parser.add_argument(
        "--repo",
        default=str(Path(__file__).resolve().parent),
        help="Path to the EyeSSEF repo root (default: this script's folder)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to run the pipeline (default: current python)",
    )
    parser.add_argument(
        "--poll_interval",
        type=float,
        default=0.5,
        help="Polling interval in seconds (default 0.5)",
    )
    args = parser.parse_args()

    inbox = Path(args.inbox).expanduser().resolve()
    archive = Path(args.archive).expanduser().resolve()
    repo = Path(args.repo).expanduser().resolve()
    py = Path(args.python).expanduser().resolve()

    if not inbox.exists():
        raise FileNotFoundError(f"Inbox folder not found: {inbox}")
    archive.mkdir(parents=True, exist_ok=True)

    cfg = Config(inbox=inbox, archive=archive, repo=repo, python=py, poll_interval=args.poll_interval)
    session = _create_new_session(repo)

    print(f"[session] Started: {session.root}")
    print(f"[watch] Inbox: {cfg.inbox}")
    print("[watch] This session will only run main.py on incoming videos.")
    print("[watch] Type `process` to process ALL trials + ONE average, then exit.")
    _print_help()

    while True:
        for p in sorted(cfg.inbox.iterdir()):
            if not _is_candidate_video(p, cfg.inbox):
                continue
            if p.name in session.seen_files:
                continue

            if not wait_until_stable(p):
                continue

            print(f"[watch] New video: {p.name}")
            try:
                _ingest_one_video(cfg, session, p)
                session.seen_files.add(p.name)
                print(f"[watch] main.py done -> trials captured: {len(session.trials)}")
            except subprocess.CalledProcessError as e:
                print(f"[watch] ERROR: main.py failed for {p.name}: {e}")
            except Exception as e:
                print(f"[watch] ERROR: failed ingest for {p.name}: {e}")

            break

        try:
            rlist, _, _ = select.select([sys.stdin], [], [], cfg.poll_interval)
        except Exception:
            time.sleep(cfg.poll_interval)
            continue

        if rlist:
            cmd = sys.stdin.readline().strip().lower()
            if not cmd:
                continue

            if cmd in {"help", "h", "?"}:
                _print_help()
            elif cmd == "status":
                print(f"[session] trials captured: {len(session.trials)}")
                if session.trials:
                    print("[session] trial names:")
                    for td in session.trials:
                        print(f"  - {td.name}")
            elif cmd == "process":
                _process_session(cfg, session)
                print("[session] Exiting (start a new session by running watch_inbox.py again).")
                return
            elif cmd in {"quit", "exit", "q"}:
                print("[watch] Exiting without processing.")
                return
            else:
                print(f"[watch] Unknown command: {cmd}")
                print("Type `help` for commands.")


if __name__ == "__main__":
    main()
