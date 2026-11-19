"""Score each model on val and test. python evaluation/evaluate.py --data data.yaml"""
import argparse
import pandas as pd
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
a = p.parse_args()

runs = {"n_640": ("models/n_640_best.pt", 640), "n_1024": ("models/n_1024_best.pt", 1024)}
rows = []
for name, (w, sz) in runs.items():
    for split in ("val", "test"):
        m = YOLO(w).val(data=a.data, split=split, imgsz=sz, verbose=False, plots=False)
        rows.append({"run": name, "split": split, "P": m.box.mp, "R": m.box.mr,
                     "mAP50": m.box.map50, "mAP50-95": m.box.map})
print(pd.DataFrame(rows).round(3).to_string(index=False))
