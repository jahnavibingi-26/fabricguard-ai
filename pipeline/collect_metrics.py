"""Collect the report numbers on THIS machine: vision speed, memory, power, Llama speed.

Run it once on the laptop (--label laptop) and once on the KV260 (--label board). Only the board run counts as
an on-device result; keep the laptop run as a reference column.

    python collect_metrics.py --label laptop --images ~/Downloads/fabric_merged/test/images --n 100
    python collect_metrics.py --label board  --images ../samples --n 30 --llama-runs 3 \
        --runner ../bin/llama_main --pte ../llm/llama32_1b_8da4w.pte --tokenizer ../llm/tokenizer.model

Power (optional): give a file that reports power, for example
    --power-file /sys/class/hwmon/hwmonX/power1_input --power-scale 1e-6      (microwatts -> watts)
Find candidates on the board with:  for f in /sys/class/hwmon/hwmon*; do echo $f $(cat $f/name); done
No power file? Measure with a USB power meter and write the number in the report.

Writes ../evaluation/metrics_<label>.json and prints a table to copy into the report.
"""
import argparse
import json
import os
import re
import resource
import statistics
import subprocess
import threading
import time

from PIL import Image

import infer
import severity as sev

LLAMA_CLASSES = ["Hole", "Stain", "Knot", "DropStitch", "Needle Mark", "Missing Yarn"]


def pct(values, q):
    s = sorted(values)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


def mem_total_mb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemTotal"):
            return round(int(line.split()[1]) / 1024)
    return None


def own_peak_mb():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM"):
            return round(int(line.split()[1]) / 1024, 1)
    return None


class PowerSampler:
    """Samples a power file in a thread. Returns watts."""

    def __init__(self, path, scale):
        self.path, self.scale, self.vals, self.run = path, scale, [], False

    def _loop(self):
        while self.run:
            try:
                self.vals.append(float(open(self.path).read().strip()) * self.scale)
            except Exception:
                pass
            time.sleep(0.2)

    def __enter__(self):
        if self.path:
            self.run = True
            self.t = threading.Thread(target=self._loop, daemon=True)
            self.t.start()
        return self

    def __exit__(self, *a):
        self.run = False

    def summary(self):
        return {"mean_w": round(statistics.mean(self.vals), 2), "max_w": round(max(self.vals), 2),
                "samples": len(self.vals)} if self.vals else None


def bench_vision(det, files, n, power_path, power_scale):
    files = files[:n]
    det(Image.new("RGB", (det.W, det.H), (128, 128, 128)))          # warm-up, not timed
    totals, parts = [], {"pre_ms": [], "infer_ms": [], "post_ms": []}
    with PowerSampler(power_path, power_scale) as ps:
        for f in files:
            img = Image.open(f).convert("RGB")
            t0 = time.perf_counter()
            dets, lat = det(img)
            for cid, score, box in dets:                              # include the severity rules in the timing
                sev.build_record(det.names[cid], score, box, img.size[0], img.size[1])
            totals.append((time.perf_counter() - t0) * 1e3)
            for k in parts:
                parts[k].append(lat[k])
    mean = statistics.mean(totals)
    return {"images": len(totals), "mean_ms": round(mean, 1), "median_ms": round(statistics.median(totals), 1),
            "p95_ms": round(pct(totals, 0.95), 1), "fps": round(1000 / mean, 2),
            "mean_pre_ms": round(statistics.mean(parts["pre_ms"]), 1),
            "mean_model_ms": round(statistics.mean(parts["infer_ms"]), 1),
            "mean_post_ms": round(statistics.mean(parts["post_ms"]), 1),
            "peak_rss_mb": own_peak_mb(), "power": ps.summary()}


def run_llama(a, rec):
    prompt = sev.build_prompt(rec)
    cmd = [a.runner, f"--model_path={a.pte}", f"--tokenizer_path={a.tokenizer}", "--temperature=0",
           f"--seq_len={sev.suggested_seq_len(prompt)}", "--prompt=" + prompt]
    t0 = time.time()
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
    except subprocess.TimeoutExpired:
        return {"class": rec["class"], "ok": False, "error": "timeout"}
    wall = time.time() - t0
    txt = out.stdout + "\n" + out.stderr
    m = re.search(r"PyTorchObserver (\{.*?\})", txt)
    j = json.loads(m.group(1)) if m else {}
    marker = "<|start_header_id|>assistant<|end_header_id|>"
    reply = re.sub(r"<\|.*?\|>", "", out.stdout.split(marker)[-1]).strip() if marker in out.stdout else ""
    reply = re.sub(r"PyTorchObserver.*", "", reply, flags=re.S).strip()
    ok, _ = sev.check_llama_output(reply)
    res = {"class": rec["class"], "ok": ok, "wall_s": round(wall, 2)}
    if j:
        res.update({
            "load_s": round((j["model_load_end_ms"] - j["model_load_start_ms"]) / 1000, 2),
            "ttft_s": round((j["first_token_ms"] - j["inference_start_ms"]) / 1000, 2),
            "gen_s": round((j["inference_end_ms"] - j["first_token_ms"]) / 1000, 2),
            "prefill_tps": round(j["prefill_token_per_sec"], 1), "decode_tps": round(j["decode_token_per_sec"], 1),
            "prompt_tokens": j["prompt_tokens"], "generated_tokens": j["generated_tokens"]})
    return res


def bench_llama(a):
    if not (os.path.exists(a.runner) and os.path.exists(a.pte) and os.path.exists(a.tokenizer)):
        print("  Llama skipped: runner, .pte or tokenizer not found (use --runner --pte --tokenizer).")
        return None
    recs = []
    for i in range(a.llama_runs):
        cls = LLAMA_CLASSES[i % len(LLAMA_CLASSES)]
        recs.append(sev.build_record(cls, 0.9, (100, 100, 200, 180), 640, 480))
    runs = []
    with PowerSampler(a.power_file, a.power_scale) as ps:
        for i, r in enumerate(recs, 1):
            res = run_llama(a, r)
            print(f"  Llama run {i}/{len(recs)} {r['class']}: " +
                  (f"wall {res['wall_s']} s, first token {res.get('ttft_s')} s, decode {res.get('decode_tps')} tok/s, "
                   f"format ok={res['ok']}" if "wall_s" in res else res.get("error", "failed")))
            runs.append(res)
    good = [r for r in runs if "ttft_s" in r]
    if not good:
        return {"runs": runs}
    mean = lambda k: round(statistics.mean(r[k] for r in good), 2)
    return {"runs": runs, "n": len(good), "format_ok": sum(r["ok"] for r in runs),
            "mean_load_s": mean("load_s"), "mean_ttft_s": mean("ttft_s"), "mean_gen_s": mean("gen_s"),
            "mean_prefill_tps": mean("prefill_tps"), "mean_decode_tps": mean("decode_tps"),
            "mean_prompt_tokens": mean("prompt_tokens"), "mean_generated_tokens": mean("generated_tokens"),
            "mean_wall_s": mean("wall_s"), "max_wall_s": max(r["wall_s"] for r in good),
            "peak_rss_mb_llama_process": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1),
            "power": ps.summary()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="laptop or board")
    ap.add_argument("--images", default=os.path.join(infer.HERE, "..", "samples"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model", default=os.path.join(infer.HERE, "..", "models", "merged_best.onnx"))
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--runner", default=os.path.expanduser("~/executorch/cmake-out/examples/models/llama/llama_main"))
    ap.add_argument("--pte", default=os.path.join(infer.HERE, "..", "llm", "llama32_1b_8da4w.pte"))
    ap.add_argument("--tokenizer", default=os.path.expanduser("~/llama-3.2-1b/original/tokenizer.model"))
    ap.add_argument("--llama-runs", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--power-file", default="")
    ap.add_argument("--power-scale", type=float, default=1e-6)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    out_path = a.out or os.path.join(infer.HERE, "..", "evaluation", f"metrics_{a.label}.json")
    files = infer.list_images(os.path.expanduser(a.images))
    if not files:
        raise SystemExit(f"No images found in {a.images}")
    result = {"label": a.label, "prompt_version": sev.PROMPT_VERSION, "mem_total_mb": mem_total_mb(),
              "threads": a.threads or "default", "model": os.path.basename(a.model)}

    idle = None
    if a.power_file:
        print("Measuring idle power for 5 s ...")
        with PowerSampler(a.power_file, a.power_scale) as ps:
            time.sleep(5)
        idle = ps.summary()
        result["idle_power"] = idle

    print(f"VISION: {min(a.n, len(files))} images from {a.images}")
    det = infer.Detector(a.model, conf=sev.MIN_CONF, threads=a.threads or None)
    result["vision"] = bench_vision(det, files, a.n, a.power_file, a.power_scale)
    print(f"LLAMA: {a.llama_runs} runs")
    result["llama"] = bench_llama(a)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    json.dump(result, open(out_path, "w"), indent=1)

    v, L = result["vision"], result["llama"]
    print(f"\n===== REPORT TABLE ({a.label}) =====")
    print(f"Vision latency per image : mean {v['mean_ms']} ms, median {v['median_ms']} ms, p95 {v['p95_ms']} ms "
          f"(pre {v['mean_pre_ms']} + model {v['mean_model_ms']} + post {v['mean_post_ms']})")
    print(f"Vision FPS               : {v['fps']}")
    print(f"Memory                   : total {result['mem_total_mb']} MB, vision process peak {v['peak_rss_mb']} MB"
          + (f", llama_main peak {L['peak_rss_mb_llama_process']} MB" if L and 'peak_rss_mb_llama_process' in L else ""))
    pw = lambda d: f"mean {d['mean_w']} W, max {d['max_w']} W" if d else "not measured"
    print(f"Power                    : idle {pw(idle)} | vision {pw(v['power'])} | Llama {pw(L['power']) if L and 'power' in L else 'not measured'}")
    if L and "mean_ttft_s" in L:
        print(f"Llama time to first token: {L['mean_ttft_s']} s (model load {L['mean_load_s']} s on top, reloaded on every call)")
        print(f"Llama tokens per second  : prefill {L['mean_prefill_tps']}, decode {L['mean_decode_tps']} "
              f"({L['mean_prompt_tokens']} prompt tokens, {L['mean_generated_tokens']} generated)")
        print(f"Llama total response time: mean {L['mean_wall_s']} s, max {L['max_wall_s']} s per alert "
              f"(format ok {L['format_ok']}/{len(L['runs'])})")
        print(f"Frame to Llama advice    : about {round(v['mean_ms'] / 1000 + L['mean_wall_s'], 1)} s "
              f"(the fixed advice is on screen after {v['mean_ms']} ms)")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
