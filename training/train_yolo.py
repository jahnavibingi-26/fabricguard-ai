"""Train YOLOv8n on GC10-DET (YOLO format).
python training/train_yolo.py --data data.yaml --imgsz 1024 --epochs 100 --batch 8 --name n_1024
python training/train_yolo.py --data data.yaml --imgsz 640 --leaky --name n640_leaky   # DPU-friendly activation
"""
import argparse
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--imgsz", type=int, default=1024)
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--name", default="n_1024")
p.add_argument("--project", default="runs")
p.add_argument("--leaky", action="store_true", help="swap SiLU for LeakyReLU (for the KV260 DPU)")
a = p.parse_args()

if a.leaky:
    import torch.nn as nn
    from ultralytics.nn.modules import conv
    conv.Conv.default_act = nn.LeakyReLU(0.1, inplace=True)
    model = YOLO("yolov8n.yaml").load("yolov8n.pt")
else:
    model = YOLO("yolov8n.pt")

model.train(data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, patience=30,
            flipud=0.5, device=0, project=a.project, name=a.name, workers=2)
