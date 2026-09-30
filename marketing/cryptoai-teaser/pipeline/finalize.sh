#!/usr/bin/env bash
# Mux rendered picture with the score, loudness-normalised for social platforms (-14 LUFS).
# Usage: finalize.sh <video_in> <score.wav> <out.mp4>
set -euo pipefail
ffmpeg -y -loglevel error -i "$1" -i "$2" -map 0:v -map 1:a -c:v copy \
  -af "loudnorm=I=-14:TP=-1.5:LRA=11" -ar 48000 -c:a aac -b:a 320k \
  -shortest -movflags +faststart "$3"
