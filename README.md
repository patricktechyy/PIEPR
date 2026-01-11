# PIEPR

PIEPR (PIEPR - PLR + Infrared + Extraction, Processing & Reporting) is a lightweight pipeline to extract a **raw pupil diameter time-series** from a recorded video of the Pupil, then applying a set of preprocessing steps (confidence threshold filtering, biologically-feasible checks, blink detection, outlier removal using MAD, linear interpolation, and Savitzky–Golay smoothing) in order to produce a clean **pupillary light reflex (PLR)** graph.

This repository will only focus on **offline processing**:
- Input: recorded eye videos (e.g., `.mp4`)
- Output: `raw.csv` + `processed.csv` + plots (`.png`)

It also includes an optional **"inbox watcher" workflow** to automatically process new trial videos uploaded from a Raspberry Pi (which records these trial videos using the camera modules) via SSH on the same local network.
If you have a Raspberry Pi and camera setup (with proper stabilization and optionally, light stimuli), you can download the .zip of our Pi Code and run it on the Pi using `bash plr.sh`. 

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
Note: this will only work for macOS (our apologies, a version for Windows may be developed in future iterations).

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

- **docs/INSTALL.md** - install Python + dependencies (including PyPupilEXT)
- **docs/USAGE.md** - how to run `main.py`, `process.py`, batch mode, and naming conventions
- **docs/OUTPUTS.md** - what files are produced and how to interpret them
---

## Core scripts

- `videoImplement/main.py`
  - Decodes video → frames
  - Runs pupil detection per frame
  - Results in `raw.csv` and `rawPlot.png`

- `videoImplement/process.py`
  - Runs preprocessing passes (the 6 mentioned above)
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
(If you are recording on a Raspberry Pi using our code, then the file naming should have already been automated)

---

## Third‑party dependency note (PyPupilEXT)

This project uses **PyPupilEXT** (`pypupilext`) for pupil ellipse fitting (PuReST). PyPupilEXT is distributed as platform-specific wheels via its GitHub releases and is licensed under **GPLv3**.
Many thanks to Zandi et al. for developing PupilEXT (and creating a Python port of it) and Santini et al. for his development of the PuReST algorithm.

Installation instructions are also included in **docs/INSTALL.md**.
