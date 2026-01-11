# Installation

This guide is recommended for those who would start from the absolute zero (no python, etc)

## Supported platforms

- **Windows 10/11**
- **macOS** (Intel or Apple Silicon, recommended due to automated workflow)
- **Linux** (Ubuntu/Debian recommended)

## 0) Install Python

PIEPR works best with **Python 3.10** because PyPupilEXT primarily ships for **Python 3.10**

### Windows
1. Download Python 3.10 from python.org.
2. During install, remember to tick **“Add Python to PATH”** (this helps simplify our process).
3. Open *Command Prompt* and verify that you have the correct Python version:
   ```bash
   python --version
   ```

### macOS
- Install with Homebrew and verify Python's version:
  ```bash
  brew install python@3.10
  python3.10 --version
  ```

### Ubuntu / Debian
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip
python3 --version
```

---

## 1) Get this repository

```bash
git clone https://github.com/patricktechyy/PIEPR.git
cd PIEPR
```
---
## 2) Create a virtual environment (very recommended)

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

---

## 3) Install base Python dependencies

From the repo's root:

```bash
pip install -r requirements.txt
```

This will install the requirements which are:
- OpenCV (`opencv-python`) for video/frame handling
- NumPy / Pandas for data work
- SciPy for Savitzky–Golay smoothing
- Matplotlib for plots

---

## 4) Install PyPupilEXT (required)

PIEPR uses **PyPupilEXT** (`pypupilext`) to fit an ellipse and estimate pupil diameter (PuReST).

### Installing the wheel from PyPupilEXT releases
1. Go to the **PyPupilEXT GitHub Releases** page.
2. Download the correct `.whl` file for your:
   - operating system
   - CPU architecture
   - Python version (recommended: **cp310**)
3. Install it:
   ```bash
   pip install /path/to/PyPupilEXT-*.whl
   ```

---

## 5) Quick installation test

Run:

```bash
python -c "import cv2, numpy, pandas, scipy, matplotlib; import pypupilext; print('OK')"
```

If this prints `OK`, you’re ready.

---

## Raspberry Pi notes
- PyPupilEXT may not have a ready-made Raspberry Pi wheel. In that case you must build it from source (and that can take a very long time :c).

Hence the easiest approach (which is our approach) is:
1) record on the Pi
2) transfer videos to a laptop/desktop
3) process on the laptop/desktop

It's even easier with our automated workflow :D
