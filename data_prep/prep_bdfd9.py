#!/usr/bin/env python3
"""Clean BDFD9 for training.

Output: every image rotated to one orientation (default landscape), letterboxed to a fixed
size (landscape 640x480 or portrait 480x640),
saved as JPG, with box-only YOLO labels (class cx cy w h, normalized). Polygon labels
(labels_seg/) are ignored. Splits are kept exactly as the dataset authors made them.

Usage:
  python3 prep_bdfd9.py --diagnose                 # decide whether line-mark classes need swapping
  python3 prep_bdfd9.py --swap-linemarks           # build (landscape default; --orientation portrait available)
"""
import argparse
import collections
import os
import random
import shutil
import statistics

from PIL import Image, ImageDraw, ImageOps

ROT90_CW = getattr(Image, "Transpose", Image).ROTATE_270   # PIL rotates counter-clockwise
LANCZOS = getattr(Image, "Resampling", Image).LANCZOS
NAMES = ["DropStitch", "Hole", "Horizontal LineMark", "Knot", "Oil Spot",
         "Setup", "Slub", "Vertical LineMark", "Yarn Dust"]
HLINE, VLINE = 2, 7
SPLITS = ["train", "valid", "test"]


def read_boxes(path):
    """Return [(cls, cx, cy, w, h)] from lines with exactly 5 fields; count skipped lines."""
    rows, skipped = [], 0
    for line in open(path):
        p = line.split()
        if not p:
            continue
        if len(p) == 5:
            rows.append((int(p[0]), *map(float, p[1:])))
        else:
            skipped += 1
    return rows, skipped


def list_images(src, split):
    d = os.path.join(src, split, "images")
    return sorted(f for f in os.listdir(d) if f.lower().endswith((".webp", ".jpg", ".jpeg", ".png")))


def diagnose(src):
    """Median pixel w/h ratio of line-mark boxes, by image orientation."""
    stats = collections.defaultdict(list)
    for split in SPLITS:
        for f in list_images(src, split):
            stem = os.path.splitext(f)[0]
            with Image.open(os.path.join(src, split, "images", f)) as im:
                W, H = im.size
            orient = "landscape" if W > H else "portrait" if H > W else "square"
            boxes, _ = read_boxes(os.path.join(src, split, "labels", stem + ".txt"))
            for c, cx, cy, w, h in boxes:
                if c in (HLINE, VLINE):
                    stats[(NAMES[c], orient)].append((w * W) / max(h * H, 1e-9))
    print("Median box width/height (pixels). >1 means wide, <1 means tall.\n")
    for k in sorted(stats):
        print(f"  {k[0]:22s} {k[1]:10s} n={len(stats[k]):3d}  median w/h = {statistics.median(stats[k]):.2f}")
    print("\nHow to read it:")
    print(" - Horizontal is WIDE (>1) and Vertical is TALL (<1) in BOTH orientations")
    print("   -> classes follow the image frame -> use --swap-linemarks when rotating.")
    print(" - The ratios FLIP between landscape and portrait")
    print("   -> classes follow the fabric -> do NOT use --swap-linemarks.")


def transform(im, boxes, W_t, H_t, swap, want_landscape):
    im = ImageOps.exif_transpose(im).convert("RGB")
    rotated = (im.height > im.width) if want_landscape else (im.width > im.height)
    if rotated:
        im = im.transpose(ROT90_CW)
        new = []
        for c, cx, cy, w, h in boxes:
            if swap and c in (HLINE, VLINE):
                c = VLINE if c == HLINE else HLINE
            new.append((c, 1.0 - cy, cx, h, w))        # 90 degrees clockwise
        boxes = new
    s = min(W_t / im.width, H_t / im.height)
    nw, nh = max(1, round(im.width * s)), max(1, round(im.height * s))
    ox, oy = (W_t - nw) // 2, (H_t - nh) // 2
    canvas = Image.new("RGB", (W_t, H_t), (114, 114, 114))
    canvas.paste(im.resize((nw, nh), LANCZOS), (ox, oy))
    out = []
    for c, cx, cy, w, h in boxes:
        x1, y1 = (cx - w / 2) * nw + ox, (cy - h / 2) * nh + oy
        x2, y2 = (cx + w / 2) * nw + ox, (cy + h / 2) * nh + oy
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W_t, x2), min(H_t, y2)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        out.append((c, (x1 + x2) / 2 / W_t, (y1 + y2) / 2 / H_t, (x2 - x1) / W_t, (y2 - y1) / H_t))
    return canvas, out, rotated


def build(src, dst, W_t, H_t, swap, n_preview):
    want_landscape = W_t >= H_t
    random.seed(0)
    if os.path.exists(dst):
        raise SystemExit(f"{dst} already exists. Delete it or choose another --dst.")
    before, after = collections.Counter(), collections.Counter()
    n_rot = n_skip = n_drop = 0
    for split in SPLITS:
        os.makedirs(f"{dst}/{split}/images")
        os.makedirs(f"{dst}/{split}/labels")
        files = list_images(src, split)
        preview = set(random.sample(files, min(n_preview, len(files))))
        for f in files:
            stem = os.path.splitext(f)[0]
            boxes, skipped = read_boxes(os.path.join(src, split, "labels", stem + ".txt"))
            n_skip += skipped
            for b in boxes:
                before[b[0]] += 1
            with Image.open(os.path.join(src, split, "images", f)) as im:
                canvas, out, rotated = transform(im, boxes, W_t, H_t, swap, want_landscape)
            n_rot += rotated
            n_drop += len(boxes) - len(out)
            canvas.save(f"{dst}/{split}/images/{stem}.jpg", quality=95)
            with open(f"{dst}/{split}/labels/{stem}.txt", "w") as fh:
                for c, cx, cy, w, h in out:
                    fh.write(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    after[c] += 1
            if f in preview:
                os.makedirs(f"{dst}/_preview", exist_ok=True)
                d = ImageDraw.Draw(canvas)
                for c, cx, cy, w, h in out:
                    d.rectangle([(cx - w / 2) * W_t, (cy - h / 2) * H_t,
                                 (cx + w / 2) * W_t, (cy + h / 2) * H_t], outline=(255, 0, 0), width=2)
                    d.text(((cx - w / 2) * W_t + 3, (cy - h / 2) * H_t + 3), NAMES[c], fill=(255, 255, 0))
                canvas.save(f"{dst}/_preview/{split}_{stem}.jpg")
    with open(f"{dst}/data.yaml", "w") as fh:
        fh.write("train: train/images\nval: valid/images\ntest: test/images\n\nnc: 9\nnames:\n")
        fh.writelines(f"  {i}: {n}\n" for i, n in enumerate(NAMES))
    for fn in ("README.md", "dataset_metadata.csv", "classes.txt"):
        if os.path.exists(os.path.join(src, fn)):
            shutil.copy(os.path.join(src, fn), dst)
    print(f"Done. Output: {dst}  (size {W_t}x{H_t}, rotated: {n_rot} images)")
    print(f"Skipped non-box label lines: {n_skip}   boxes dropped after clipping: {n_drop}")
    print(f"{'class':22s}{'before':>8s}{'after':>8s}")
    for i, n in enumerate(NAMES):
        print(f"{n:22s}{before[i]:8d}{after[i]:8d}")
    print("Note: with --swap-linemarks the two LineMark rows may differ before/after; the totals must match.")
    print(f"Open {dst}/_preview and check the red boxes sit on the defects.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.expanduser("~/Downloads/BDFD9_v3"))
    ap.add_argument("--dst", default=os.path.expanduser("~/Downloads/bdfd9_clean"))
    ap.add_argument("--orientation", choices=["landscape", "portrait"], default="landscape")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--swap-linemarks", action="store_true")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--preview", type=int, default=8, help="preview images per split")
    a = ap.parse_args()
    if a.diagnose:
        diagnose(a.src)
    else:
        land = a.orientation == "landscape"
        W = a.width or (640 if land else 480)
        H = a.height or (480 if land else 640)
        build(a.src, a.dst, W, H, a.swap_linemarks, a.preview)
