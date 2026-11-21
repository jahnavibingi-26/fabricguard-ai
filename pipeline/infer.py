"""FabricGuard AI - CPU inference pipeline (vision + severity + async Llama advice + logging).

Only needs: numpy, Pillow, onnxruntime (no torch, no ultralytics), so it also runs on the board.

Examples (from ~/fabric_project/pipeline, inside ~/yolo-venv):
    python infer.py --source ~/Downloads/fabric_merged/test/images --limit 20
    python infer.py --source ~/Downloads/fabric_merged/test/images --limit 20 --llama

Outputs in ../logs/:
    inspections.jsonl   one line per image (PASS included): verdict, detections, latency
    llama.jsonl         one line per Llama advice (text, seconds, source = llama | fallback)
    annotated/          images with boxes and a PASS / WARNING / REJECT banner
"""
import argparse
import ast
import json
import os
import queue
import re
import statistics
import subprocess
import threading
import time
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw

import severity as sev

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NMS_IOU = 0.45

DEFAULT_NAMES = ["DropStitch", "Hole", "Horizontal LineMark", "Knot", "Oil Spot", "Setup", "Slub",
                 "Vertical LineMark", "Yarn Dust", "Needle Mark", "Missing Yarn", "Stain"]
TIER_COLOR = {"LOW": (80, 200, 120), "MEDIUM": (240, 200, 40), "HIGH": (255, 140, 0), "CRITICAL": (230, 30, 30)}
VERDICT_COLOR = {"PASS": (40, 160, 70), "WARNING": (230, 170, 0), "REJECT": (210, 30, 30)}


# ------------------------------------------------------------------ pre / post processing
def letterbox(img, W, H):
    """Resize keeping aspect ratio, pad with grey to (W, H). Returns canvas, scale, pad_x, pad_y."""
    w0, h0 = img.size
    r = min(W / w0, H / h0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    canvas = Image.new("RGB", (W, H), (114, 114, 114))
    px, py = (W - nw) // 2, (H - nh) // 2
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
    return canvas, r, px, py


def to_tensor(canvas):
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None])


def nms(boxes, scores, classes, iou_thr):
    """Class-aware NMS in numpy. boxes are xyxy. Returns indices to keep."""
    shifted = boxes + classes[:, None].astype(np.float32) * 4096.0   # separate classes
    x1, y1, x2, y2 = shifted.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_thr]
    return keep


def decode(out, conf, r, px, py, w0, h0, iou=NMS_IOU):
    """YOLOv8 ONNX output (1, 4+nc, N) -> list of (class_id, score, [x1, y1, x2, y2]) in original pixels."""
    p = out[0].T
    cls_scores = p[:, 4:]
    cls = cls_scores.argmax(1)
    score = cls_scores[np.arange(len(p)), cls]
    keep = score >= conf
    p, cls, score = p[keep], cls[keep], score[keep]
    if len(p) == 0:
        return []
    cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    idx = nms(boxes, score, cls, iou)
    dets = []
    for i in idx:
        x1, y1, x2, y2 = boxes[i]
        x1, x2 = np.clip([(x1 - px) / r, (x2 - px) / r], 0, w0)
        y1, y2 = np.clip([(y1 - py) / r, (y2 - py) / r], 0, h0)
        dets.append((int(cls[i]), float(score[i]), [float(x1), float(y1), float(x2), float(y2)]))
    return dets


class Detector:
    def __init__(self, model_path, conf=sev.MIN_CONF, threads=None):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.in_name = inp.name
        self.H, self.W = int(inp.shape[2]), int(inp.shape[3])
        self.conf = conf
        self.names = self._names()

    def _names(self):
        try:
            d = ast.literal_eval(self.sess.get_modelmeta().custom_metadata_map["names"])
            return [d[i] for i in range(len(d))]
        except Exception:
            return DEFAULT_NAMES

    def __call__(self, img):
        t0 = time.perf_counter()
        canvas, r, px, py = letterbox(img, self.W, self.H)
        x = to_tensor(canvas)
        t1 = time.perf_counter()
        out = self.sess.run(None, {self.in_name: x})[0]
        t2 = time.perf_counter()
        dets = decode(out, self.conf, r, px, py, img.size[0], img.size[1])
        t3 = time.perf_counter()
        return dets, {"pre_ms": (t1 - t0) * 1e3, "infer_ms": (t2 - t1) * 1e3, "post_ms": (t3 - t2) * 1e3}


# ------------------------------------------------------------------ Llama (separate pipeline, background thread)
class LlamaWorker:
    """Runs llama_main one alert at a time in a background thread so the vision loop never waits."""

    def __init__(self, runner, model, tokenizer, log_path, timeout=300):
        self.runner, self.model, self.tokenizer = runner, model, tokenizer
        self.log_path, self.timeout = log_path, timeout
        self.cache = {}
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, image_name, rec):
        self.q.put((image_name, rec))

    def close(self):
        self.q.put(None)
        self.thread.join()

    def _run(self, rec):
        prompt = sev.build_prompt(rec)
        cmd = [self.runner, f"--model_path={self.model}", f"--tokenizer_path={self.tokenizer}",
               "--temperature=0", f"--seq_len={sev.suggested_seq_len(prompt)}", "--prompt=" + prompt]
        t0 = time.time()
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
            stdout = out.stdout
        except subprocess.TimeoutExpired:
            return None, time.time() - t0, {}
        marker = "<|start_header_id|>assistant<|end_header_id|>"
        reply = re.sub(r"<\|.*?\|>", "", stdout.split(marker)[-1]).strip() if marker in stdout else ""
        stats = {}
        m = re.search(r"PyTorchObserver (\{.*?\})", stdout + out.stderr)
        if m:
            try:
                j = json.loads(m.group(1))
                stats = {k: j[k] for k in ("prefill_token_per_sec", "decode_token_per_sec",
                                           "prompt_tokens", "generated_tokens") if k in j}
                stats["first_token_s"] = round((j["first_token_ms"] - j["inference_start_ms"]) / 1000, 2)
            except Exception:
                pass
        reply = re.sub(r"PyTorchObserver.*", "", reply, flags=re.S).strip()
        return reply, time.time() - t0, stats

    def _loop(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            image_name, rec = item
            key = (rec["class"], rec["severity"], rec["location"])
            computed = None
            if key in self.cache:
                text, computed, source, stats = self.cache[key]
                secs, cached = 0.0, True
            else:
                reply, secs, stats = self._run(rec)
                ok, clean = sev.check_llama_output(reply)
                text, source, cached = (clean, "llama", False) if ok else (sev.fallback_alert(rec), "fallback", False)
                self.cache[key] = (text, secs, source, stats)
            line = {"type": "llama", "time": datetime.now().isoformat(timespec="seconds"), "image": image_name,
                    "class": rec["class"], "severity": rec["severity"], "source": source,
                    "prompt_version": sev.PROMPT_VERSION, "seconds": round(secs, 2), "cached": cached, "computed_seconds": round(computed, 2) if computed else None,
                    "text": text, **stats}
            with open(self.log_path, "a") as f:
                f.write(json.dumps(line) + "\n")
            print(f"   [llama:{source}{' cached' if cached else ''} {secs:.1f}s] {image_name}: {text}")


# ------------------------------------------------------------------ drawing
def annotate(img, records, verdict, out_path):
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    for r in records:
        x1, y1, x2, y2 = r["box"]
        col = TIER_COLOR[r["severity"]]
        d.rectangle([x1, y1, x2, y2], outline=col, width=3)
        label = f"{r['class']} {r['conf']:.2f} {r['severity']}"
        tw = d.textlength(label)
        d.rectangle([x1, max(0, y1 - 12), x1 + tw + 4, max(12, y1)], fill=col)
        d.text((x1 + 2, max(0, y1 - 12)), label, fill=(0, 0, 0))
    d.rectangle([0, 0, im.size[0], 18], fill=VERDICT_COLOR[verdict])
    d.text((6, 3), f"{verdict}  ({len(records)} defect{'s' if len(records) != 1 else ''})", fill=(255, 255, 255))
    im.save(out_path, quality=90)


# ------------------------------------------------------------------ main
def list_images(src):
    if os.path.isdir(src):
        return sorted(os.path.join(src, f) for f in os.listdir(src) if os.path.splitext(f)[1].lower() in IMG_EXT)
    return [src]


def main():
    ap = argparse.ArgumentParser(description="FabricGuard AI inference")
    ap.add_argument("--source", required=True, help="image file or folder")
    ap.add_argument("--model", default=os.path.join(HERE, "..", "models", "merged_best.onnx"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "logs"))
    ap.add_argument("--conf", type=float, default=sev.MIN_CONF)
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (0 = all)")
    ap.add_argument("--threads", type=int, default=0, help="onnxruntime CPU threads (0 = default)")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to wait after each image (simulates a conveyor for the demo)")
    ap.add_argument("--llama", action="store_true", help="also run Llama advice in the background")
    ap.add_argument("--runner", default=os.path.expanduser("~/executorch/cmake-out/examples/models/llama/llama_main"))
    ap.add_argument("--pte", default=os.path.join(HERE, "..", "llm", "llama32_1b_8da4w.pte"))
    ap.add_argument("--tokenizer", default=os.path.expanduser("~/llama-3.2-1b/original/tokenizer.model"))
    a = ap.parse_args()

    os.makedirs(os.path.join(a.out, "annotated"), exist_ok=True)
    insp_log = os.path.join(a.out, "inspections.jsonl")
    llama_log = os.path.join(a.out, "llama.jsonl")

    det = Detector(a.model, conf=a.conf, threads=a.threads or None)
    print(f"Model input {det.W}x{det.H}, {len(det.names)} classes, conf >= {a.conf}")
    det(Image.new("RGB", (det.W, det.H), (128, 128, 128)))      # warm-up, not timed

    worker = LlamaWorker(a.runner, a.pte, a.tokenizer, llama_log) if a.llama else None
    files = list_images(os.path.expanduser(a.source))
    if a.limit:
        files = files[:a.limit]
    totals, counts = [], {"PASS": 0, "WARNING": 0, "REJECT": 0}
    t_start = time.perf_counter()

    for path in files:
        name = os.path.basename(path)
        img = Image.open(path).convert("RGB")
        t0 = time.perf_counter()
        dets, lat = det(img)
        recs = []
        for cid, score, box in dets:
            rec = sev.build_record(det.names[cid], score, box, img.size[0], img.size[1])
            if rec:
                rec["box"] = [round(v, 1) for v in box]
                recs.append(rec)
        verdict, tier, top = sev.frame_verdict(recs)
        lat["total_ms"] = (time.perf_counter() - t0) * 1e3
        totals.append(lat["total_ms"])
        counts[verdict] += 1

        entry = {"type": "inspection", "time": datetime.now().isoformat(timespec="seconds"), "image": name,
                 "verdict": verdict, "top_severity": tier, "n_defects": len(recs),
                 "detections": [{k: r[k] for k in ("class", "conf", "severity", "area_pct", "location", "box")} for r in recs],
                 "latency_ms": {k: round(v, 1) for k, v in lat.items()},
                 "advice_fixed": ({"cause": top["cause"], "action": top["action"], "disposition": top["disposition"]}
                                  if top else None)}
        with open(insp_log, "a") as f:
            f.write(json.dumps(entry) + "\n")
        annotate(img, recs, verdict, os.path.join(a.out, "annotated", os.path.splitext(name)[0] + ".jpg"))

        print(f"{name}: {verdict:7} defects={len(recs)} {lat['total_ms']:.0f} ms"
              + (f" | {top['class']} {top['severity']} -> {top['disposition']}" if top else ""))
        if top:
            print(f"   advice (fixed): {sev.fallback_alert(top)}")
            if worker:
                worker.submit(name, top)
        if a.delay:
            time.sleep(a.delay)

    wall = time.perf_counter() - t_start
    if worker:
        print("Waiting for Llama jobs to finish...")
        worker.close()
    if totals:
        s = sorted(totals)
        print(f"\n{len(totals)} images | PASS {counts['PASS']} WARNING {counts['WARNING']} REJECT {counts['REJECT']}")
        print(f"Vision latency per image: mean {statistics.mean(totals):.1f} ms, median {statistics.median(totals):.1f} ms, "
              f"p95 {s[min(len(s) - 1, int(0.95 * len(s)))]:.1f} ms  ->  about {1000 / statistics.mean(totals):.1f} FPS")
        print(f"Logs: {insp_log}" + (f" and {llama_log}" if worker else ""))


if __name__ == "__main__":
    main()
