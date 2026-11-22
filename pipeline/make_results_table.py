"""Merge the infer.py logs with the ground-truth labels into ONE per-image table for the report.

    python make_results_table.py --logs ../logs_test_full --data ~/Downloads/fabric_merged/test \
        --out ../evaluation/test_results_all.csv

Columns: image, source, truth, gt_classes, verdict, outcome, n_defects, top_class, top_conf, top_severity,
top_class_in_gt, advice_source, advice_cached, llama_seconds, vision_ms, fixed_advice, llama_advice
Outcome: detected (defect image not PASS), MISS (defect image called PASS),
         correct PASS (clean image PASS), FALSE ALARM (clean image not PASS).
"""
import argparse
import csv
import json
import os

import yaml


def read_jsonl(path):
    out = []
    if os.path.exists(path):
        for line in open(path):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--data", default=os.path.expanduser("~/Downloads/fabric_merged/test"))
    ap.add_argument("--out", default="../evaluation/test_results_all.csv")
    a = ap.parse_args()

    names = yaml.safe_load(open(os.path.join(os.path.dirname(a.data.rstrip("/")), "data.yaml")))["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    insp = {e["image"]: e for e in read_jsonl(os.path.join(a.logs, "inspections.jsonl")) if e.get("type") == "inspection"}
    llama = {e["image"]: e for e in read_jsonl(os.path.join(a.logs, "llama.jsonl"))}

    rows = []
    for img, e in sorted(insp.items()):
        lab = os.path.join(a.data, "labels", os.path.splitext(img)[0] + ".txt")
        gt = []
        if os.path.exists(lab):
            for line in open(lab):
                p = line.split()
                if p:
                    gt.append(names[int(float(p[0]))])
        gt = sorted(set(gt))
        defect = bool(gt)
        v = e["verdict"]
        outcome = ("detected" if v != "PASS" else "MISS") if defect else ("correct PASS" if v == "PASS" else "FALSE ALARM")
        dets = sorted(e["detections"], key=lambda d: -d["conf"])
        top = dets[0] if dets else None
        L = llama.get(img)
        fixed = e.get("advice_fixed")
        rows.append({
            "image": img, "source": "BDFD9" if img.startswith("bd_") else "FabriD",
            "truth": "defect" if defect else "clean", "gt_classes": "; ".join(gt),
            "verdict": v, "outcome": outcome, "n_defects": e["n_defects"],
            "top_class": top["class"] if top else "", "top_conf": top["conf"] if top else "",
            "top_severity": e.get("top_severity") or "",
            "top_class_in_gt": ("yes" if top["class"] in gt else "no") if (top and defect) else "",
            "advice_source": (L["source"] if L else ("fixed only" if fixed else "")),
            "advice_cached": (L.get("cached") if L else ""), "llama_seconds": (L.get("seconds") if L else ""),
            "vision_ms": e["latency_ms"]["total_ms"],
            "fixed_advice": (f"Cause: {fixed['cause']}. Action: {fixed['action']}." if fixed else ""),
            "llama_advice": (L["text"] if L and L.get("source") == "llama" else ""),
        })

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    cnt = lambda k, val: sum(1 for r in rows if r[k] == val)
    print(f"{n} images -> {a.out}\n")
    print("VISION (image level)")
    print(f"  verdicts: PASS {cnt('verdict','PASS')}  WARNING {cnt('verdict','WARNING')}  REJECT {cnt('verdict','REJECT')}")
    nd, nc = cnt("truth", "defect"), cnt("truth", "clean")
    print(f"  defect images {nd}: detected {cnt('outcome','detected')}, MISSED {cnt('outcome','MISS')} ({100*cnt('outcome','MISS')/max(nd,1):.1f}%)")
    print(f"  clean images  {nc}: correct PASS {cnt('outcome','correct PASS')}, FALSE ALARM {cnt('outcome','FALSE ALARM')} ({100*cnt('outcome','FALSE ALARM')/max(nc,1):.1f}%)")
    for s in ("BDFD9", "FabriD"):
        sub = [r for r in rows if r["source"] == s and r["truth"] == "defect"]
        if sub:
            print(f"  {s}: {sum(r['outcome']=='MISS' for r in sub)}/{len(sub)} defect images missed")
    det_rows = [r for r in rows if r["truth"] == "defect" and r["top_class"]]
    if det_rows:
        ok = sum(r["top_class_in_gt"] == "yes" for r in det_rows)
        print(f"  top detected class is one of the labelled classes: {ok}/{len(det_rows)} ({100*ok/len(det_rows):.1f}%) of detected defect images")
    ms = [r["vision_ms"] for r in rows]
    print(f"  vision latency mean {sum(ms)/n:.1f} ms (laptop; Llama jobs may compete for CPU during this run)")
    alerts = [r for r in rows if r["verdict"] != "PASS"]
    print("\nADVISORY")
    print(f"  alerts (WARNING or REJECT): {len(alerts)}; with fixed advice: {sum(bool(r['fixed_advice']) for r in alerts)}")
    llama_rows = [r for r in alerts if r["advice_source"] in ("llama", "fallback")]
    if llama_rows:
        nl = sum(r["advice_source"] == "llama" for r in llama_rows)
        nf = len(llama_rows) - nl
        fresh = [float(r["llama_seconds"]) for r in llama_rows if r["advice_cached"] is False and r["llama_seconds"] != ""]
        print(f"  Llama advice shown: {nl}/{len(llama_rows)} ({100*nl/len(llama_rows):.1f}%), FALLBACK to fixed text: {nf}")
        print(f"  reused from an identical earlier alert (cache): {sum(r['advice_cached'] is True for r in llama_rows)}; "
              f"fresh Llama runs: {len(fresh)}" + (f", mean {sum(fresh)/len(fresh):.1f} s each" if fresh else ""))
    else:
        print("  no Llama results found (was infer.py run with --llama?)")


if __name__ == "__main__":
    main()
