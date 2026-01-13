# Folder and Structures

There's 2 way of running PIEPR
---
## 1) Manual mode (from `videoImplement/main.py`)

When running:

```bash
python videoImplement/main.py --input <video>
```

PIEPR will create one folder per video under:

```
videoImplement/data/<video_stem>/
```

Inside that folder it will contain:

| File | What it is |
|---|---|
| `raw.csv` | This file will contain the pupil estimate (pixels + mm), timestamps, as well as the confidence of the estimate |
| `rawPlot.png` | This is an image of the plot of the raw pupil diameter |
| `processed_interpolated.csv` | Output after Pass 1–4 (flagging + interpolation) |
| `processed.csv` | The final result after Pass 6 (Savitzky–Golay smoothing) |
| `processedPlot.png` | Image of the plot of the final processed result |
| `pass_quality.csv` | Per-pass flagged counts (% flagged + newly flagged). Useful for your Results table |
| `pass_quality.json` | Same as pass_quality.csv, plus interpolation gap stats + reason breakdown |
| `validity_report.csv` | One-row numeric validity summary for reporting / spreadsheet use |
| `validity_report.json` | Full validity summary (same content as CSV, but structured) |
| `validity_overlay.png` | Visual overlay of raw vs interpolated vs smoothed curves |
| `validity_residuals.png` | Residuals between interpolated and smoothed curves (smoothing distortion) |
| `validity_psd.png` | Frequency-domain (PSD) comparison of interpolated vs smoothed signals |

---

## 2) Processing (`videoImplement/process.py`)

When running:

```bash
python videoImplement/process.py --data videoImplement/data
```

`process.py` will group trials by:

- user
- eye side (L/R)
- resolution
- fps

Then it will process the **latest two** trials in each group and produces the following output folder:

```
videoImplement/data/PLR_<user>_<eye>_<res>_<fps>_AVG/
  processed_avg_interpolated.csv
  processed_avg.csv
  processedAvgPlot.png
```

---

## 3) Inbox watcher sessions (`watch_inbox.py`, most recommended way of running) 

Session mode outputs go under:

```
videoImplement/sessions/session_<YYYYMMDD_HHMMSS>/
  trial/
    <video_stem_1>/...
    <video_stem_2>/...
    ...
  average/
    raw.csv
    processed_interpolated.csv
    processed.csv
    processedPlot.png
```

Notes:
- `trial/` contains per-trial results
- `average/` contains **one** averaged result across *all* trials in that session
- Validity artifacts (`validity_*`) are saved in each processed folder and average folder when `process.py` runs.
