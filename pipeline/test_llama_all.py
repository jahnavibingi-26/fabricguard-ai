#!/usr/bin/env python3
"""Run the Llama advisory (ExecuTorch llama_main) on all 12 classes x 3 variants = 36 prompts.

Reports the shape-check pass rate, timing (laptop or board) and saves every reply next to the
fixed fallback text. Uses the same severity.py functions as the pipeline.

  python test_llama_all.py
  python test_llama_all.py --limit 3          # quick smoke test
"""
import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import severity as sev

NAMES = ["DropStitch", "Hole", "Horizontal LineMark", "Knot", "Oil Spot", "Setup", "Slub",
         "Vertical LineMark", "Yarn Dust", "Needle Mark", "Missing Yarn", "Stain"]
# label, confidence, box area share of the frame, box centre (x, y) in a 640x480 frame
VARIANTS = [("small/low-conf", 0.30, 0.01, (100, 100)),
            ("medium", 0.55, 0.03, (320, 240)),
            ("large/high-conf", 0.85, 0.25, (400, 300))]
W, H = 640, 480
ASSIST = "<|start_header_id|>assistant<|end_header_id|>"


def make_box(area, cx, cy):
    side = (area * W * H) ** 0.5
    return (max(0, cx - side / 2), max(0, cy - side / 2), min(W, cx + side / 2), min(H, cy + side / 2))


def parse(stdout, stderr):
    """Reply = text after the last assistant header in stdout; stats = PyTorchObserver JSON line."""
    tail = stdout.rsplit(ASSIST, 1)[-1]
    reply = re.split(r"<\|eot_id\|>|PyTorchObserver", tail)[0].strip()
    m = re.search(r"PyTorchObserver (\{.*\})", stdout + "\n" + stderr)
    return reply, (json.loads(m.group(1)) if m else {})


def shape_ok(text, rec):
    fn = getattr(sev, "check_llama_output", None)
    if fn:
        try:
            return fn(text, rec).strip() == text.strip()      # the check returns the fallback when it rejects
        except Exception:
            pass
    t = text.strip()
    return t.startswith("Cause:") and "Action:" in t and len(t.split()) <= 70


def words(s):
    return set(re.findall(r"[a-z']+", s.lower()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", default="~/executorch/cmake-out/examples/models/llama/llama_main")
    ap.add_argument("--pte", default="~/fabric_project/llm/llama32_1b_8da4w.pte")
    ap.add_argument("--tokenizer", default="~/llama-3.2-1b/original/tokenizer.model")
    ap.add_argument("--out", default="../evaluation/llama_12class.csv")
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts (0 = all)")
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()
    runner, pte, tok = (os.path.expanduser(p) for p in (a.runner, a.pte, a.tokenizer))

    rows = []
    jobs = [(n, v) for n in NAMES for v in VARIANTS]
    if a.limit:
        jobs = jobs[:a.limit]
    for i, (name, (label, conf, area, (cx, cy))) in enumerate(jobs, 1):
        rec = sev.build_record(name, conf, make_box(area, cx, cy), W, H)
        if rec is None:
            print(f"[{i}/{len(jobs)}] {name} / {label}: no record built (below confidence filter?)")
            continue
        prompt = sev.build_prompt(rec)
        try:
            seq = sev.suggested_seq_len(prompt)
        except Exception:
            seq = 400
        cmd = [runner, f"--model_path={pte}", f"--tokenizer_path={tok}", "--temperature=0",
               f"--seq_len={seq}", f"--prompt={prompt}"]
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
        wall = time.time() - t0
        reply, st = parse(p.stdout, p.stderr)
        ok = p.returncode == 0 and bool(reply) and shape_ok(reply, rec)
        fixed = sev.fallback_alert(rec)
        rw = words(reply)
        sevlvl = rec["severity"]
        row = {
            "class": name, "variant": label, "severity": sevlvl,
            "prompt_tokens": st.get("prompt_tokens"), "generated_tokens": st.get("generated_tokens"),
            "prefill_tps": round(st.get("prefill_token_per_sec", 0), 1),
            "decode_tps": round(st.get("decode_token_per_sec", 0), 1),
            "load_ms": st.get("model_load_end_ms", 0) - st.get("model_load_start_ms", 0),
            "first_token_ms": st.get("first_token_ms", 0) - st.get("inference_start_ms", 0),
            "wall_s": round(wall, 2), "passes": ok,
            "word_overlap_with_fixed": round(len(rw & words(fixed)) / max(1, len(rw)), 2),
            "reply": reply, "fixed": fixed,
        }
        rows.append(row)
        print(f"[{i}/{len(jobs)}] {name:20s} {label:16s} {sevlvl:8s} pass={ok} wall={wall:.1f}s")
        if not ok:
            print(f"     reply: {reply[:200]!r}  (rc={p.returncode})")

    if not rows:
        raise SystemExit("No results.")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_ok = sum(r["passes"] for r in rows)
    def mean(k):
        v = [r[k] for r in rows if r[k]]
        return round(statistics.mean(v), 2) if v else 0
    print("\n===== SUMMARY =====")
    print(f"prompts: {len(rows)}   passed shape check: {n_ok} ({100 * n_ok / len(rows):.0f}%)")
    print("by severity:", {s: f"{sum(r['passes'] for r in rows if r['severity'] == s)}/{sum(1 for r in rows if r['severity'] == s)}"
                           for s in sorted({r['severity'] for r in rows})})
    bad = sorted({r["class"] for r in rows if not r["passes"]})
    print("classes with a failure:", bad or "none")
    print(f"mean prompt tokens {mean('prompt_tokens')}   mean generated {mean('generated_tokens')}")
    print(f"mean prefill {mean('prefill_tps')} tok/s   mean decode {mean('decode_tps')} tok/s")
    print(f"mean model load {mean('load_ms')} ms   mean first token {mean('first_token_ms')} ms   mean wall {mean('wall_s')} s")
    print(f"mean word overlap with fixed advice {mean('word_overlap_with_fixed')} (rough groundedness proxy, not a proof)")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
