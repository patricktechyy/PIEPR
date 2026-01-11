# Auto-transfer system from Pi to macOS + Auto-processing system for macOS

This repo includes two sides which are:

1) **Pi's side**: `plr_autosend.sh` watches the Pi recording folder and pushes new videos to your Mac inbox folder.
2) **Mac's side**: `watch_inbox.py` watches the inbox and proceeds to run the PIEPR pipeline based on **sessions**.

---

## Part A — Raspberry Pi (autosend)

### 1) Install dependencies required on Pi
```bash
sudo apt-get update
sudo apt-get install -y inotify-tools rsync openssh-client
```

### 2) Edit `plr_autosend.sh`
Open the script and check these three lines:
- `SRC_DIR=...`  (this should be the Pi's recording directory, e.g. `/home/<pi_user>/PLR_Video`)
- `MAC_USER=...`
- `MAC_HOST=...` (this should be the Mac IP on the same network, e.g. `172.20.10.4`)
- `MAC_INBOX=...` (this should be the Mac inbox folder, e.g. `/Users/<mac_user>/plr_inbox`)


### 3) Run the autosend on Pi
```bash
chmod +x plr_autosend.sh
./plr_autosend.sh
```

---

## Part B — Mac (autorun pipeline)

### 1) Create inbox folder
```bash
mkdir -p ~/plr_inbox
mkdir -p ~/plr_inbox/_processing
mkdir -p ~/plr_inbox/_archive
```

### 2) Install watcher dependency
```bash
python3 -m pip install -r requirements_inbox.txt
```

### 3) Run the Mac watcher (from repo root)
```bash
python3 watch_inbox.py --inbox ~/plr_inbox --repo .
```

What happens in **session mode**:
- Pi uploads `*.part` then renames to the final `*.mp4`
- watcher ignores `.part`, waits until file size is stable
- for every new video in the inbox root, watcher:
  - moves the video into `~/plr_inbox/_processing/`
  - runs **raw extraction only**:
    - `videoImplement/main.py --input <video> --no_raw_plot --no_preprocess`
  - archives the source video into `~/plr_inbox/_archive/`
  - saves the raw trial folder into:
    - `videoImplement/sessions/session_<timestamp>/trial/<video_stem>/`

When done collecting videos for that session, type and enter the following:

```
process
```

Then the watcher will:
- run `videoImplement/process.py` on **every** trial folder in `trial/` (no plot windows)
- create exactly **one** averaged signal across **all** trials in:
  - `videoImplement/sessions/session_<timestamp>/average/raw.csv`
- run `videoImplement/process.py` on `average/` and **show** the interactive plots
- exit (start a new session by running `watch_inbox.py` again)

