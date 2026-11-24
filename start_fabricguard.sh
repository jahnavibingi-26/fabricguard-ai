#!/bin/bash
source ~/yolo-venv/bin/activate
cd ~/fabric_project/pipeline
LOGS=../logs_live
if [ "$1" = "review" ]; then
  python dashboard.py --logs $LOGS --port 8001 --read-only
else
  python dashboard.py --logs $LOGS --port 8001 &
  DASH=$!
  trap "kill $DASH 2>/dev/null" EXIT
  sleep 2
  xdg-open http://localhost:8001 >/dev/null 2>&1 &
  python live.py --source 2 --out $LOGS --width 640 --height 480 --conf 0.5 --llama
fi
