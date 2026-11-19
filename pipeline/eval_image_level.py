"""Image-level evaluation of the whole FabricGuard pipeline (detector + severity rules).

Ground truth per image: defect image = its YOLO label file is non-empty; clean image = empty label file.
System call per image: PASS (no detection) / WARNING / REJECT.

  miss         = defect image called PASS
  false alarm  = clean image called WARNING or REJECT
  clean_reject = clean image called REJECT (the costly one: it stops the line)

Sweep knobs:
  min_conf      detections below this are ignored
  rej_default   confidence needed for REJECT, most classes
  rej_weak      confidence needed for REJECT for the weak classes (Hole, Needle Mark, Missing Yarn, Stain)

Pick settings on the VALIDATION set, then report the TEST set once:
    python eval_image_level.py --data ~/Downloads/fabric_merged/valid --out ../evaluation/image_level_valid.csv
    python eval_image_level.py --data ~/Downloads/fabric_merged/test  --focus 0.25,0.5,0.7
"""
import argparse
import csv
import os

from PIL import Image

import infer
import severity as sev

MIN_CONFS = [0.15, 0.20, 0.25]
REJECT_DEFAULTS = [0.50, 0.60]
REJECT_WEAKS = [0.50, 0.70, 0.80]


def collect(det, img_dir, lab_dir):
    """Run the detector once at the lowest confidence; the sweep then re-filters offline."""
    items = []
    for f in sorted(os.listdir(img_dir)):
        stem, ext = os.path.splitext(f)
        if ext.lower() not in infer.IMG_EXT:
            continue
        lab = os.path.join(lab_dir, stem + ".txt")
        gt = []
        if os.path.exists(lab):
            for line in open(lab):
                p = line.split()
                if p:
                    k = int(float(p[0]))
                    gt.append(det.names[k] if k < len(det.names) else str(k))
        img = Image.open(os.path.join(img_dir, f)).convert("RGB")
        dets, _ = det(img)
        items.append({"name": f, "defect": bool(gt), "gt": sorted(set(gt)),
                      "src": "BDFD9" if f.startswith("bd_") else "FabriD", "size": img.size,
                      "dets": [(det.names[c], s, b) for c, s, b in dets]})
    return items


def verdicts(items, min_conf, rej_default, rej_weak):
    old = sev.MIN_CONF, sev.REJECT_MIN_CONF, sev.REJECT_MIN_CONF_BY_CLASS
    sev.MIN_CONF, sev.REJECT_MIN_CONF = min_conf, rej_default
    sev.REJECT_MIN_CONF_BY_CLASS = {c: rej_weak for c in sev.WEAK_CLASSES}
    try:
        out = []
        for it in items:
            w, h = it["size"]
            recs = [r for r in (sev.build_record(n, s, b, w, h) for n, s, b in it["dets"]) if r]
            out.append(sev.frame_verdict(recs)[0])
        return out
    finally:
        sev.MIN_CONF, sev.REJECT_MIN_CONF, sev.REJECT_MIN_CONF_BY_CLASS = old


def evaluate(items, min_conf, rej_default, rej_weak):
    v = verdicts(items, min_conf, rej_default, rej_weak)
    n_def = sum(i["defect"] for i in items)
    n_clean = len(items) - n_def
    misses = sum(1 for i, x in zip(items, v) if i["defect"] and x == "PASS")
    fa = sum(1 for i, x in zip(items, v) if not i["defect"] and x != "PASS")
    return {"min_conf": min_conf, "rej_default": rej_default, "rej_weak": rej_weak,
            "misses": misses, "miss_pct": round(100 * misses / max(n_def, 1), 1),
            "false_alarms": fa, "false_alarm_pct": round(100 * fa / max(n_clean, 1), 1),
            "clean_reject": sum(1 for i, x in zip(items, v) if not i["defect"] and x == "REJECT"),
            "defect_reject": sum(1 for i, x in zip(items, v) if i["defect"] and x == "REJECT"),
            "defect_warning": sum(1 for i, x in zip(items, v) if i["defect"] and x == "WARNING")}


def breakdown(items, min_conf, rej_default, rej_weak):
    """Misses by defect class and by source, at one chosen setting."""
    v = verdicts(items, min_conf, rej_default, rej_weak)
    per_class, per_src = {}, {}
    for it, x in zip(items, v):
        if not it["defect"]:
            continue
        s = per_src.setdefault(it["src"], [0, 0]); s[0] += 1; s[1] += (x == "PASS")
        for c in it["gt"]:
            k = per_class.setdefault(c, [0, 0]); k[0] += 1; k[1] += (x == "PASS")
    print(f"\nMisses by source at min_conf={min_conf}, reject={rej_default}/{rej_weak}:")
    for s, (n, m) in sorted(per_src.items()):
        print(f"  {s:8} defect images {n:4}  missed {m:4}  ({100 * m / n:.1f}%)")
    print("Misses by defect class (an image counts for every class it contains):")
    for c, (n, m) in sorted(per_class.items(), key=lambda kv: -kv[1][1] / kv[1][0]):
        print(f"  {c:22} images {n:4}  missed {m:4}  ({100 * m / n:.1f}%)")
    return per_class, per_src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/Downloads/fabric_merged/test"))
    ap.add_argument("--model", default=os.path.join(infer.HERE, "..", "models", "merged_best.onnx"))
    ap.add_argument("--out", default=os.path.join(infer.HERE, "..", "evaluation", "image_level_test.csv"))
    ap.add_argument("--focus", default="0.25,0.5,0.7", help="min_conf,rej_default,rej_weak for the miss breakdown")
    a = ap.parse_args()

    det = infer.Detector(a.model, conf=min(MIN_CONFS))
    items = collect(det, os.path.join(a.data, "images"), os.path.join(a.data, "labels"))
    print(f"{a.data}\n{len(items)} images: {sum(i['defect'] for i in items)} with defects, "
          f"{sum(not i['defect'] for i in items)} clean\n")

    rows = [evaluate(items, mc, rd, rw) for mc in MIN_CONFS for rd in REJECT_DEFAULTS for rw in REJECT_WEAKS]
    cols = ["min_conf", "rej_default", "rej_weak", "misses", "miss_pct", "false_alarms", "false_alarm_pct",
            "clean_reject", "defect_reject", "defect_warning"]
    print("  ".join(f"{c:>13}" for c in cols))
    for r in rows:
        print("  ".join(f"{r[c]:>13}" for c in cols))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows([{c: r[c] for c in cols} for r in rows])
    print(f"\nSaved {a.out}")

    mc, rd, rw = [float(x) for x in a.focus.split(",")]
    breakdown(items, mc, rd, rw)


if __name__ == "__main__":
    main()
