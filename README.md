# PIEPR

PIEPR (PIEPR — Pupil Image Extraction, Processing, & Report) is a lightweight, pipeline to extract a **raw pupil diameter time-series** from a recorded trial video, then applying a standard set of preprocessing steps (confidence filtering, blink/biological checks, outlier removal, interpolation, and Savitzky–Golay smoothing) in order to produce a clean **pupillary light reflex (PLR)** graph.

This repository will only focus on **offline processing**:
- Input: recorded eye videos (e.g., `.mp4`)
- Output: `raw.csv` + `processed.csv` + plots (`.png`)

It also includes an optional **"inbox watcher" workflow** to automatically process new trial videos uploaded from a Raspberry Pi (which records these trial videos).

---

## Quick start

### 1) Install
In order to install, please follow the full installation guide:
- **docs/INSTALL.md**

### 2) Manually run the pipeline on one video
From the repo root:
```bash
python -m venv .venv
# to activate your venv (see docs/INSTALL.md)

python videoImplement/main.py --input path/to/your_video.mp4 --no_raw_plot
```
Outputs will be created under:
```
videoImplement/data/<video_stem>/
  raw.csv
  rawPlot.png
  processed_interpolated.csv
  processed.csv
  processedPlot.png
```

---

## Automated workflow (Pi → macOS inbox → session processing)
Note: this will only work for macOS

If your Raspberry Pi is uploading trial videos to a folder on your Mac system (e.g., via `rsync`), you can run:

```bash
python watch_inbox.py --inbox ~/plr_inbox
```

This will run in **session mode**, meaning:
- each incoming video is converted to a trial folder (for raw extraction)
- when you type `process`, it processes all captured trials and generates **one** session average

Full setup steps for the automated workflow:
- **AUTO_TRANSFER_MAC.md** 

---

## Documentation (very important)

- **docs/INSTALL.md** — install Python + dependencies (including PyPupilEXT)
- **docs/USAGE.md** — how to run `main.py`, `process.py`, batch mode, and naming conventions
- **docs/OUTPUTS.md** — what files are produced and how to interpret them
---

## Core scripts

- `videoImplement/main.py`
  - Decodes video → frames
  - Runs pupil detection per frame
  - Results in `raw.csv` and `rawPlot.png`

- `videoImplement/process.py`
  - Runs preprocessing passes
  - Results in `processed.csv` and `processedPlot.png`
  - In automated workflow mode, also writes one averaged output folder per recording setting

---

## Video naming

Please name the trial videos like the following example:

```
PLR_<User>_<EyeSide L/R>_<Resolution>_<FPS>_<TrialIndex>.mp4

Example:
PLR_Patrick_R_1920x1080_30_2.mp4
```

This allows `main.py` / `process.py` to infer the correct FPS and resolution.

---

## Third‑party dependency note (PyPupilEXT)

This project uses **PyPupilEXT** (`pypupilext`) for pupil ellipse fitting (PuReST). PyPupilEXT is distributed as platform-specific wheels via its GitHub releases and is licensed under **GPLv3**.

Installation instructions are also included in **docs/INSTALL.md**.
