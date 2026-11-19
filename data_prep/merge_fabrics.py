#!/usr/bin/env python3
"""Merge cleaned BDFD9 (9 classes) + FabriD (4 classes, polygon labels) into one 12-class dataset.

- FabriD polygons -> bounding boxes (min/max of the polygon points).
- FabriD images: rotated to landscape and letterboxed to 640x480 (same as cleaned BDFD9).
- Leakage control for FabriD: train images are dropped if they share a base file name with a
  valid/test image or look near-identical (dHash); valid images near-identical to a test image are dropped.
- Defect-free FabriD images are kept with empty label files (background images).
- Writes extra per-source test sets so results can be reported separately.

Run from the folder that contains prep_bdfd9.py:
  python3 merge_fabrics.py
"""
import argparse
import collections
import csv
import os
import shutil

from PIL import Image

from prep_bdfd9 import transform   # same rotate + letterbox + box transform as the BDFD9 step

LANCZOS = getattr(Image, "Resampling", Image).LANCZOS
NAMES = ["DropStitch", "Hole", "Horizontal LineMark", "Knot", "Oil Spot", "Setup", "Slub",
         "Vertical LineMark", "Yarn Dust", "Needle Mark", "Missing Yarn", "Stain"]
# FabriD ids (from its data.yaml): 0 Needle Mark, 1 hole, 2 missing yarn, 3 stain
FABRID_MAP = {0: 9, 1: 1, 2: 10, 3: 11}
SPLITS = ["train", "valid", "test"]


def dhash(im, n=8):
    g = im.convert("L").resize((n + 1, n), LANCZOS)
    px = g.tobytes()
    bits = 0
    for r in range(n):
        for c in range(n):
            bits = (bits << 1) | int(px[r * (n + 1) + c] > px[r * (n + 1) + c + 1])
    return bits


def ham(a, b):
    return bin(a ^ b).count("1")


def read_fabrid_boxes(path):
    """Polygon (or box) rows -> [(mapped_cls, cx, cy, w, h)] normalized."""
    boxes = []
    for line in open(path):
        t = line.split()
        if len(t) < 5:
            continue
        cls, v = int(t[0]), list(map(float, t[1:]))
        if len(v) == 4:                      # already a box
            cx, cy, w, h = v
        else:                                # polygon x1 y1 x2 y2 ...
            xs, ys = v[0::2], v[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        boxes.append((FABRID_MAP[cls], cx, cy, w, h))
    return boxes


def load_fabrid(root):
    data = {}
    for sp in SPLITS:
        idir = f"{root}/{sp}/images"
        items = []
        for f in sorted(os.listdir(idir)):
            stem = os.path.splitext(f)[0]
            with Image.open(f"{idir}/{f}") as im:
                h = dhash(im)
            items.append({"file": f, "stem": stem, "base": f.split(".rf.")[0], "hash": h,
                          "img": f"{idir}/{f}", "lbl": f"{root}/{sp}/labels/{stem}.txt"})
        data[sp] = items
    return data


def filter_leakage(data, thr):
    dropped = collections.Counter()
    test_b = {x["base"] for x in data["test"]}
    test_h = [x["hash"] for x in data["test"]]
    valid_b = {x["base"] for x in data["valid"]}
    valid_h = [x["hash"] for x in data["valid"]]
    keep_valid = []
    for x in data["valid"]:
        if x["base"] in test_b:
            dropped["valid: same name as test"] += 1
        elif any(ham(x["hash"], h) <= thr for h in test_h):
            dropped["valid: near-duplicate of test"] += 1
        else:
            keep_valid.append(x)
    data["valid"] = keep_valid
    keep_train = []
    for x in data["train"]:
        if x["base"] in test_b or x["base"] in valid_b:
            dropped["train: same name as valid/test"] += 1
        elif any(ham(x["hash"], h) <= thr for h in test_h + valid_h):
            dropped["train: near-duplicate of valid/test"] += 1
        else:
            keep_train.append(x)
    data["train"] = keep_train
    return dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdfd9", default="~/Downloads/bdfd9_clean")
    ap.add_argument("--fabrid", default="~/Downloads/Febric Defect Detection and Segm.yolov")
    ap.add_argument("--dst", default="~/Downloads/fabric_merged")
    ap.add_argument("--dhash-threshold", type=int, default=4, help="max Hamming distance (of 64 bits) counted as near-duplicate")
    a = ap.parse_args()
    bd, fd, dst = (os.path.expanduser(p) for p in (a.bdfd9, a.fabrid, a.dst))
    if os.path.exists(dst):
        raise SystemExit(f"{dst} already exists. Delete it or choose another --dst.")

    for sp in SPLITS + ["test_bdfd9", "test_fabrid"]:
        os.makedirs(f"{dst}/{sp}/images")
        os.makedirs(f"{dst}/{sp}/labels")

    manifest = []
    cls_count = {sp: collections.Counter() for sp in SPLITS}
    n_img = collections.Counter()
    n_empty = collections.Counter()

    def write(split, name, source, canvas_or_path, boxes, extra=None):
        targets = [split] + ([f"test_{source}"] if split == "test" else [])
        for t in targets:
            if isinstance(canvas_or_path, str):
                shutil.copy(canvas_or_path, f"{dst}/{t}/images/{name}.jpg")
            else:
                canvas_or_path.save(f"{dst}/{t}/images/{name}.jpg", quality=95)
            with open(f"{dst}/{t}/labels/{name}.txt", "w") as fh:
                for c, cx, cy, w, h in boxes:
                    fh.write(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        for c, *_ in boxes:
            cls_count[split][c] += 1
        n_img[(split, source)] += 1
        n_empty[(split, source)] += (len(boxes) == 0)
        manifest.append([name, source, split, extra or ""])

    # BDFD9 (already cleaned: landscape 640x480, 9 classes)
    for sp in SPLITS:
        for f in sorted(os.listdir(f"{bd}/{sp}/images")):
            stem = os.path.splitext(f)[0]
            boxes = []
            for line in open(f"{bd}/{sp}/labels/{stem}.txt"):
                t = line.split()
                if len(t) == 5:
                    boxes.append((int(t[0]), *map(float, t[1:])))
            write(sp, f"bd_{stem}", "bdfd9", f"{bd}/{sp}/images/{f}", boxes)

    # FabriD
    data = load_fabrid(fd)
    before = {sp: len(data[sp]) for sp in SPLITS}
    dropped = filter_leakage(data, a.dhash_threshold)
    n_rot = 0
    for sp in SPLITS:
        for i, x in enumerate(data[sp]):
            boxes = read_fabrid_boxes(x["lbl"])
            with Image.open(x["img"]) as im:
                canvas, out, rot = transform(im, boxes, 640, 480, False, True)
            n_rot += rot
            write(sp, f"fd_{sp}_{i:05d}", "fabrid", canvas, out, x["file"])

    # yaml files
    names_block = "nc: 12\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES))
    open(f"{dst}/data.yaml", "w").write("train: train/images\nval: valid/images\ntest: test/images\n\n" + names_block)
    for src in ("bdfd9", "fabrid"):
        open(f"{dst}/data_test_{src}.yaml", "w").write(
            f"train: train/images\nval: test_{src}/images\ntest: test_{src}/images\n\n" + names_block)
    with open(f"{dst}/manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "source", "split", "original_file"])
        w.writerows(manifest)

    # summary
    print(f"Done. Output: {dst}\n")
    print("FabriD images before leakage filter:", before)
    print("Dropped for leakage:", dict(dropped) if dropped else "none")
    print(f"FabriD images rotated to landscape: {n_rot}\n")
    print(f"{'split':8s}{'bdfd9':>8s}{'fabrid':>8s}{'total':>8s}{'empty(bd/fd)':>16s}")
    for sp in SPLITS:
        b, f_ = n_img[(sp, 'bdfd9')], n_img[(sp, 'fabrid')]
        print(f"{sp:8s}{b:8d}{f_:8d}{b + f_:8d}{n_empty[(sp, 'bdfd9')]:>8d}/{n_empty[(sp, 'fabrid')]:<7d}")
    print(f"{'ALL':8s}{sum(n_img[(s, 'bdfd9')] for s in SPLITS):8d}{sum(n_img[(s, 'fabrid')] for s in SPLITS):8d}"
          f"{sum(n_img.values()):8d}\n")
    print(f"{'class':22s}{'train':>8s}{'valid':>8s}{'test':>8s}")
    for i, n in enumerate(NAMES):
        print(f"{n:22s}{cls_count['train'][i]:8d}{cls_count['valid'][i]:8d}{cls_count['test'][i]:8d}")


if __name__ == "__main__":
    main()
