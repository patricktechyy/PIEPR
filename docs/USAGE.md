# Usage

There's 2 way of running PIEPR

1) **Manual**: you can run `videoImplement/main.py` and `videoImplement/process.py` yourself.
2) **Inbox watcher**: `watch_inbox.py` monitors an “inbox” folder and creates a session automatically.

---

## A. Manual processing

### 1) Process one video (raw + processed)

From the repo's root:

```bash
python videoImplement/main.py --input path/to/video.mp4 --no_raw_plot
```

This will:
- extract frames from the video
- run the pupil detection
- Outputs `raw.csv` + `rawPlot.png`
- run preprocessing and outputs `processed.csv` + `processedPlot.png`

Output folder:

```
videoImplement/data/<video_stem>/
  raw.csv
  rawPlot.png
  processed_interpolated.csv
  processed.csv
  processedPlot.png
```

### 2) Process a whole folder of trial videos

```bash
python videoImplement/main.py --input path/to/folder --recursive --no_raw_plot
```

### 3) Only generate raw.csv (skip preprocessing)

```bash
python videoImplement/main.py --input path/to/video.mp4 --no_raw_plot --no_preprocess
```

---

## B. Running `process.py`

### 1) Single-trial mode (this only works with folder that already contains raw.csv)

```bash
python videoImplement/process.py --data videoImplement/data/PLR_Patrick_R_1920x1080_30_2 --resolution 1920x1080 --fps 30
```

### 2) Batch mode (many trial folders)

```bash
python videoImplement/process.py --data videoImplement/data
```

---

## C. How to name your videos (very recommended)

Name trial videos like:

```
PLR_<User>_<EyeSide L/R>_<Resolution>_<FPS>_<TrialIndex>.mp4

Example:
PLR_Patrick_R_1920x1080_30_2.mp4
```

In order to:
- `main.py` reliably infer fps/res.
- `process.py` group and average trials correctly.

---

## D. Automated workflow

### 1) Ensure that these inbox folders are created

```bash
mkdir -p ~/plr_inbox
mkdir -p ~/plr_inbox/_processing
mkdir -p ~/plr_inbox/_archive
```

### 2) Run the watcher

From the repo root:

```bash
python watch_inbox.py --inbox ~/plr_inbox
```

### 3) Collect trial videos

Copy/upload videos into `~/plr_inbox/`.

The watcher will automatically:
- wait until file size is stable
- run `main.py` raw extraction
- move trial results into `videoImplement/sessions/session_<timestamp>/trial/`
- archive videos into `~/plr_inbox/_archive/`

### 4) Process the session

Type the following and enter:

```
process
```

This will:
- process all trials
- build exactly **one** session average in:
  `videoImplement/sessions/session_<timestamp>/average/raw.csv`
- process the average (results in the plots)
