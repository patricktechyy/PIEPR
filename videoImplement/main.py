

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from typing import List, Optional, Tuple

import cv2
import pandas as pd

import scripts.detection.ppDetect as ppDetect
import scripts.others.graph as graph
import scripts.others.util as util


DEFAULT_VIDEO_PATH = "../eyeVids/tuna/PLR_Tuna_R_1920x1080_30_4.mp4"
CONFIDENCE_THRESH = 0.75

_BASE_RES = (1920, 1080)
_BASE_PX_TO_MM = 30.0  

TRIAL_VIDEO_RE = re.compile(
    r"^PLR_(?P<user>.+)_(?P<eye>[LR])_(?P<res>\d+x\d+)_(?P<fps>\d+)_(?P<trial>\d+)$"
)


def parse_resolution(res_str: str) -> Tuple[int, int]:
    m = re.match(r"^\s*(\d+)\s*x\s*(\d+)\s*$", res_str)
    if not m:
        raise ValueError(f"Invalid resolution format: {res_str} (expected like 1920x1080)")
    return int(m.group(1)), int(m.group(2))


def px_to_mm_from_resolution(width: int, height: int) -> float:
    """Estimate px/mm by scaling from the 1920x1080 calibration.

    This assumes the same camera + lens + working distance.
    """
    _, base_h = _BASE_RES
    return _BASE_PX_TO_MM * (height / base_h)


def reset_folder(folder: str) -> str:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)
    return folder


def video_to_images(video_path: str, out_dir: str) -> Tuple[float, int]:
    """Decode a video into BMP frames named frame0.bmp, frame1.bmp, ..."""
    util.dprint(f"Decoding frames from: {video_path} -> {out_dir}")

    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(out_dir, f"frame{frame_idx}.bmp")
        cv2.imwrite(out_path, frame)
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    util.dprint(f"Decoded {frame_idx} frames.")
    return fps, frame_idx


def _sorted_frame_paths(frames_dir: str) -> List[str]:
    """Return frame paths sorted by the numeric index in 'frame<idx>.bmp'."""
    frame_re = re.compile(r"^frame(\d+)\.bmp$", re.IGNORECASE)
    items = []
    for name in os.listdir(frames_dir):
        m = frame_re.match(name)
        if not m:
            continue
        items.append((int(m.group(1)), os.path.join(frames_dir, name)))
    items.sort(key=lambda x: x[0])
    return [p for _, p in items]


def pupil_detection_in_folder(frames_dir: str, preview_title: str) -> Tuple[List[float], List[float]]:
    util.dprint(f"Running pupil detection on frames in: {frames_dir}")

    confidences: List[float] = []
    diameters_px: List[float] = []

    frame_paths = _sorted_frame_paths(frames_dir)
    for p in frame_paths:
        img_with_pupil, outline_confidence, pupil_diameter = ppDetect.detect(p)
        confidences.append(float(outline_confidence))
        diameters_px.append(float(pupil_diameter))

        cv2.imshow(preview_title, img_with_pupil)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    return confidences, diameters_px


def calculate_timestamps(fps: float, total_frames: int) -> List[float]:
    if fps <= 0:
        raise ValueError("FPS must be > 0 to compute timestamps.")
    dt = 1.0 / float(fps)
    return [i * dt for i in range(total_frames)]


def save_raw_csv(
    frame_ids: List[int],
    timestamps: List[float],
    diameters_px: List[float],
    confidences: List[float],
    out_csv: str,
    px_to_mm: float,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "frame_id": frame_ids,
            "timestamp": timestamps,
            "diameter": diameters_px,
            "confidence": confidences,
        }
    )

    df["is_bad_data"] = df["confidence"] < float(CONFIDENCE_THRESH)
    df["diameter_mm"] = pd.to_numeric(df["diameter"], errors="coerce") / float(px_to_mm)

    df.to_csv(out_csv, index=False)
    util.dprint(f"Saved raw.csv: {out_csv}")
    return df


def _get_video_props(video_path: str) -> Tuple[float, int, int]:
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return fps, width, height


def _find_videos(input_path: str, recursive: bool) -> List[str]:
    exts = {".mp4", ".mov", ".avi", ".mkv", ".h264"}

    if os.path.isfile(input_path):
        return [input_path]

    if not os.path.isdir(input_path):
        return []

    videos: List[str] = []
    if recursive:
        for root, _, files in os.walk(input_path):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    videos.append(os.path.join(root, f))
    else:
        for f in os.listdir(input_path):
            p = os.path.join(input_path, f)
            if os.path.isfile(p) and os.path.splitext(f)[1].lower() in exts:
                videos.append(p)

    videos.sort()
    return videos


def generate_report(video_path: str, show_raw_plot: bool = True) -> str:
    """Run detection on one video and write raw.csv to videoImplement/data/<stem>/"""
    util.dprint(f"Processing video: {video_path}")

    reset_folder("videos")
    reset_folder("frames")

    stem = os.path.splitext(os.path.basename(video_path))[0]

    fps_override: Optional[int] = None
    res_override: Optional[str] = None
    m = TRIAL_VIDEO_RE.match(stem)
    if m:
        fps_override = int(m.group("fps"))
        res_override = m.group("res")

    meta_fps, meta_w, meta_h = _get_video_props(video_path)

    fps = float(fps_override) if fps_override is not None else float(meta_fps)

    if res_override is not None:
        w, h = parse_resolution(res_override)
    else:
        w, h = int(meta_w), int(meta_h)

    px_to_mm = px_to_mm_from_resolution(w, h)

    _, total_frames = video_to_images(video_path, "frames")
    conf, diam = pupil_detection_in_folder("frames", preview_title=f"Pupil detection: {stem}")

    if total_frames != len(diam):
        n = min(total_frames, len(diam), len(conf))
        total_frames = n
        diam = diam[:n]
        conf = conf[:n]

    timestamps = calculate_timestamps(fps, total_frames)

    out_dir = reset_folder(os.path.join("data", stem))
    out_csv = os.path.join(out_dir, "raw.csv")

    df = save_raw_csv(
        frame_ids=list(range(total_frames)),
        timestamps=timestamps,
        diameters_px=diam,
        confidences=conf,
        out_csv=out_csv,
        px_to_mm=px_to_mm,
    )

    graph.plotResults(
        df,
        savePath=os.path.join(out_dir, "rawPlot.png"),
        showPlot=show_raw_plot,
        showMm=True,
    )

    return out_dir


def _run_preprocessing(data_dir: str) -> None:
    process_py = os.path.join(os.path.dirname(__file__), "process.py")
    cmd = [sys.executable, process_py, "--data", data_dir]
    util.dprint("Running preprocessing: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=None,
        help="A single video file, or a folder containing videos. If omitted, uses DEFAULT_VIDEO_PATH.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for videos recursively when --input is a folder.",
    )
    parser.add_argument(
        "--no_raw_plot",
        action="store_true",
        help="Do not open interactive matplotlib windows for the raw plot.",
    )
    parser.add_argument(
        "--no_preprocess",
        action="store_true",
        help="Only generate raw.csv; do not run process.py afterwards.",
    )
    args = parser.parse_args()

    input_path = args.input or DEFAULT_VIDEO_PATH
    videos = _find_videos(input_path, recursive=args.recursive)

    if os.path.isdir(input_path) and not videos and not args.no_preprocess:
        _run_preprocessing(input_path)
        raise SystemExit(0)

    if not videos:
        raise SystemExit(f"No videos found at: {input_path}")

    show_raw = (len(videos) == 1) and (not args.no_raw_plot)

    for v in videos:
        generate_report(v, show_raw_plot=show_raw)

    if not args.no_preprocess:
        data_parent = os.path.join(os.path.dirname(__file__), "data")
        _run_preprocessing(data_parent)


if __name__ == "__main__":
    main()
