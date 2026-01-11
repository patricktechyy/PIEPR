#!/bin/bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$HOME/PLR_Video"
mkdir -p "$BASE_DIR"

# cache sudo once so user isn't spammed with password prompts
sudo -v || exit 1

# -------- helpers --------
prompt_default () {
  local prompt="$1"
  local def="$2"
  local var
  read -p "$prompt [$def]: " var
  if [ -z "$var" ]; then
    echo "$def"
  else
    echo "$var"
  fi
}

valid_roi () {
  # expects x,y,w,h all in 0..1-ish numeric form (not strict range-check)
  [[ "$1" =~ ^[0-9]*\.?[0-9]+,[0-9]*\.?[0-9]+,[0-9]*\.?[0-9]+,[0-9]*\.?[0-9]+$ ]]
}

cleanup() {
  # stop fixation daemon if running
  if [ -n "${focus_pid:-}" ] && kill -0 "$focus_pid" 2>/dev/null; then
    kill -TERM "$focus_pid" 2>/dev/null
    wait "$focus_pid" 2>/dev/null
  fi

  # stop IR daemon if running
  if [ -n "${ir_pid:-}" ] && kill -0 "$ir_pid" 2>/dev/null; then
    kill -TERM "$ir_pid" 2>/dev/null
    wait "$ir_pid" 2>/dev/null
  fi

  # enforce IR off state
  if [ "${side:-R}" = "L" ]; then
    sudo python3 "$SCRIPT_DIR/left_IR_off.py" >/dev/null 2>&1
  else
    sudo python3 "$SCRIPT_DIR/right_IR_off.py" >/dev/null 2>&1
  fi
}
trap cleanup EXIT

echo ""
echo "The values in [ ] are the default values if no input is detected."
echo "If any information is entered wrongly by accident, press Ctrl + C to discard this trial."
echo ""
id="$(prompt_default "Enter Patient Name" "TEST")"
id="${id// /_}"
id="$(echo "$id" | tr -cd 'A-Za-z0-9_-')"

side="$(prompt_default "Eye side (L/R)" "R")"
side="${side^^}"
if [ "$side" != "L" ] && [ "$side" != "R" ]; then
  echo "Invalid side, defaulting to R."
  side="R"
fi

mode="$(prompt_default "Mode (1080p30 / 720p60)" "1080p30")"
case "$mode" in
  1080p30) width=1920; height=1080; fps=30 ;;
  720p60)  width=1280; height=720;  fps=60 ;;
  *) echo "Invalid mode, defaulting to 1080p30."; width=1920; height=1080; fps=30 ;;
esac
echo "Using: ${width}x${height} @ ${fps}fps"

lp="$(prompt_default "Lens position (recommended 2.86)" "2.86")"
focal_len=$(awk -v lp="$lp" 'BEGIN { printf "%d", (100.0/lp)+0.5 }')
echo "Lens position: $lp. Focal length = ${focal_len}cm."
echo ""

DA="$(prompt_default "5 min Dark Adaptation timer? (Y/N)" "Y")"
DA="${DA^^}"
if [ "$DA" = "Y" ]; then
  echo "Starting Dark Adaptation (5 min). The room should be dark."
  sudo python3 "$SCRIPT_DIR/DA_timer.py"
  echo "Dark Adaptation done."
else
  echo "Dark Adaptation skipped."
fi

# Start fixation (background daemon)
sudo python3 "$SCRIPT_DIR/focus.py" >/dev/null 2>&1 &
focus_pid=$!

# Start IR illumination (background)
if [ "$side" = "L" ]; then
  sudo python3 "$SCRIPT_DIR/left_IR_on.py" >/dev/null 2>&1 &
else
  sudo python3 "$SCRIPT_DIR/right_IR_on.py" >/dev/null 2>&1 &
fi
ir_pid=$!

sleep 1

# ROI loop
roi="0.4,0.6,0.15,0.15"
echo ""
echo "Default Region-of-Interest (ROI) as x,y,a,b: $roi"

echo "ROI Adjustment Guide: Adjust values 0-1 based on the live preview so that the subject's Eye is centred."
echo "Increase/Decrease 'x' to shift the Eye image Left/Right"
echo "Increase/Decrease 'y' to shift the Eye image Up/Down"
echo "Do NOT change a, b (zoom) unless calibration was done specifically for that zoom level"
echo ''

while true; do
  echo ""
  echo "Current ROI: $roi"
  echo "Enter new ROI as x,y,0.15,0.15; or type OK to proceed."
  echo ""

  # suppress most camera logs:
  # - LIBCAMERA_LOG_LEVELS reduces libcamera logging
  # - -v 0 reduces rpicam verbosity (supported on rpicam-apps) :contentReference[oaicite:1]{index=1}
  LIBCAMERA_LOG_LEVELS="*:ERROR" rpicam-still -t 0 \
    --width "$width" --height "$height" --framerate "$fps" \
    --roi "$roi" --autofocus-mode manual --lens-position "$lp" \
    -v 0 >/dev/null 2>&1 &
  pid=$!

  read -p "ROI or OK: " input
  kill "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null

  input="${input^^}"
  if [ "$input" = "OK" ]; then
    break
  fi

  if valid_roi "$input"; then
    roi="$input"
  else
    echo "Invalid ROI format. Example: 0.40,0.60,0.15,0.15"
  fi
done

# auto index
index=1
for file in "$BASE_DIR"/PLR_"$id"_"$side"_*; do
  if [[ $file =~ _([0-9]+)\.mp4$ ]]; then
    trial_num="${BASH_REMATCH[1]}"
    if (( trial_num >= index )); then
      index=$((trial_num + 1))
    fi
  fi
done

video_file="$BASE_DIR/PLR_${id}_${side}_${width}x${height}_${fps}_${index}.mp4"

echo ""
echo "Recording..."
echo "Output: $video_file"
echo ""

# Record: -n disables preview window; -v 0 reduces logs (if supported) :contentReference[oaicite:2]{index=2}
# If you want to keep logs for debugging, redirect stderr to a log file instead of /dev/null
LIBCAMERA_LOG_LEVELS="*:ERROR" rpicam-vid -t 125000 \
  --codec libav --bitrate 20000000 -o "$video_file" \
  --width "$width" --height "$height" --framerate "$fps" \
  --roi "$roi" --autofocus-mode manual --lens-position "$lp" \
  -n --denoise cdn_off -v 0 >/dev/null 2>&1 &

vid_pid=$!

sleep 0.5 #camera delay
echo "Recording Started"

sudo python3 "$SCRIPT_DIR/led.py"

wait "$vid_pid"
echo "Recording Finished"
sudo python3 "$SCRIPT_DIR/rec_finish.py"

sudo python3 "$SCRIPT_DIR/end.py"
echo "Session Complete!"
echo "MP4: $video_file"
