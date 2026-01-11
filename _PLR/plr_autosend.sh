#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="/home/anton/PLR_Video"

MAC_USER="ptrckhanzel"
MAC_HOST="172.254.249.195"   # <-- put ur mac ip here bro
MAC_INBOX="/Users/ptrckhanzel/plr_inbox" # <-- put ur mac directory for inbox here lil bro

echo "[autosend] watching: $SRC_DIR -> ${MAC_USER}@${MAC_HOST}:${MAC_INBOX}"

inotifywait -m -e close_write,moved_to --format '%f' "$SRC_DIR" | while read -r fname; do
  case "$fname" in
    *.mp4|*.mov) ;;
    *) continue ;;
  esac

  local_path="$SRC_DIR/$fname"
  [ -s "$local_path" ] || continue

  # Upload as .part first so your Mac watcher doesn't process a half-uploaded file
  echo "[autosend] uploading: $fname"
  rsync -av --partial --progress "$local_path" "${MAC_USER}@${MAC_HOST}:${MAC_INBOX}/${fname}.part"

  echo "[autosend] finalize: rename .part -> real file"
  ssh "${MAC_USER}@${MAC_HOST}" "mv -f '${MAC_INBOX}/${fname}.part' '${MAC_INBOX}/${fname}'"

  echo "[autosend] done"

done
