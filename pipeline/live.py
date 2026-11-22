"""FabricGuard AI - live inspection from a USB / laptop camera or a video file.

camera or video -> latest frame -> YOLO (ONNX Runtime) -> severity rules -> PASS / WARNING / REJECT
                                                         -> logs + live view (latest.jpg) + background Llama advice

Needs OpenCV for capture:  pip install opencv-python-headless
Examples (from ~/fabric_project/pipeline):
    python live.py --source 0 --out ../logs_live --llama                 # first camera (laptop or USB)
    python live.py --source ~/videos/fabric.mp4 --out ../logs_live --loop # your own video file, repeated
    python dashboard.py --logs ../logs_live --port 8000                   # second terminal, shows the live view

Real-time behaviour: a reader thread keeps only the NEWEST frame. If inspection is slower than the camera
(likely on the KV260 CPU), frames are dropped instead of building a delay; dropped frames are reported.
Every processed frame is logged (PASS included). An alert EVENT (annotated image, Llama advice) is created when
a defect class first appears, gets worse (WARNING -> REJECT), or repeats after --cooldown seconds.
"""
import argparse
import json
import os
import statistics
import threading
import time
from datetime import datetime

from PIL import Image

import infer
import severity as sev

VRANK = {"PASS": 0, "WARNING": 1, "REJECT": 2}


class FrameSource:
    """Reads a camera (index) or a video file in a thread and keeps only the newest frame."""

    def __init__(self, source, loop=False, width=0, height=0):
        import cv2
        self.cv2 = cv2
        self.is_cam = str(source).isdigit()
        self.cap = cv2.VideoCapture(int(source) if self.is_cam else os.path.expanduser(str(source)))
        if not self.cap.isOpened():
            raise SystemExit(f"Cannot open source: {source}  (for a camera try --source 0, 1, 2; list them with: ls /dev/video*)")
        if width and height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 0
        self.interval = 0.0 if self.is_cam else (1.0 / fps if 0 < fps < 240 else 1 / 25)
        self.loop, self.lock = loop, threading.Lock()
        self.slot, self.idx, self.ended, self.stop = None, 0, False, False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        nxt, fails = time.perf_counter(), 0
        while not self.stop:
            ok, frame = self.cap.read()
            if not ok:
                fails += 1
                if self.loop and not self.is_cam and fails < 3:
                    self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.ended = True
                return
            fails = 0
            with self.lock:
                self.idx += 1
                self.slot = (self.idx, frame)
            if self.interval:                      # play a video file at its own speed
                nxt += self.interval
                wait = nxt - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                else:
                    nxt = time.perf_counter()

    def newest(self, after):
        with self.lock:
            if self.slot and self.slot[0] > after:
                return self.slot
        return None

    def frames_read(self):
        with self.lock:
            return self.idx

    def close(self):
        self.stop = True
        try:
            self.cap.release()
        except Exception:
            pass


def to_pil(frame_bgr, max_w=960):
    img = Image.fromarray(frame_bgr[:, :, ::-1])      # BGR -> RGB
    if img.size[0] > max_w:
        img = img.resize((max_w, round(img.size[1] * max_w / img.size[0])), Image.BILINEAR)
    return img


def inspect(det, img):
    """One frame -> (records, verdict, tier, top record, latency dict)."""
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
    return recs, verdict, tier, top, lat


def is_event(last_alert, top, verdict, now, cooldown):
    """New class, worse verdict, or the cooldown has passed -> create an alert event."""
    if not top:
        return False
    prev = last_alert.get(top["class"])
    if prev is None or VRANK[verdict] > VRANK[prev[1]] or now - prev[0] >= cooldown:
        last_alert[top["class"]] = (now, verdict)
        return True
    return False


def main():
    ap = argparse.ArgumentParser(description="FabricGuard live inspection")
    ap.add_argument("--source", default="0", help="camera index (0, 1, ...) or a video file")
    ap.add_argument("--out", default=os.path.join(infer.HERE, "..", "logs_live"))
    ap.add_argument("--model", default=os.path.join(infer.HERE, "..", "models", "merged_best.onnx"))
    ap.add_argument("--conf", type=float, default=sev.MIN_CONF)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--width", type=int, default=0, help="request camera width (for example 640)")
    ap.add_argument("--height", type=int, default=0, help="request camera height (for example 480)")
    ap.add_argument("--loop", action="store_true", help="repeat a video file")
    ap.add_argument("--max-seconds", type=float, default=0, help="stop after this many seconds (0 = until Ctrl+C or end of video)")
    ap.add_argument("--cooldown", type=float, default=5.0, help="seconds before the same defect class raises a new alert")
    ap.add_argument("--live-every", type=float, default=0.2, help="seconds between live-view image updates")
    ap.add_argument("--llama", action="store_true", help="background Llama advice for alert events")
    ap.add_argument("--runner", default=os.path.expanduser("~/executorch/cmake-out/examples/models/llama/llama_main"))
    ap.add_argument("--pte", default=os.path.join(infer.HERE, "..", "llm", "llama32_1b_8da4w.pte"))
    ap.add_argument("--tokenizer", default=os.path.expanduser("~/llama-3.2-1b/original/tokenizer.model"))
    a = ap.parse_args()

    os.makedirs(os.path.join(a.out, "annotated"), exist_ok=True)
    insp_log, llama_log = os.path.join(a.out, "inspections.jsonl"), os.path.join(a.out, "llama.jsonl")
    live_tmp, live_file = os.path.join(a.out, "latest_tmp.jpg"), os.path.join(a.out, "latest.jpg")

    det = infer.Detector(a.model, conf=a.conf, threads=a.threads or None)
    det(Image.new("RGB", (det.W, det.H), (128, 128, 128)))          # warm-up, not timed
    worker = infer.LlamaWorker(a.runner, a.pte, a.tokenizer, llama_log) if a.llama else None
    src = FrameSource(a.source, loop=a.loop, width=a.width, height=a.height)
    print(f"Live inspection on {a.source} (model input {det.W}x{det.H}). Ctrl+C to stop. Logs: {os.path.abspath(a.out)}")

    last_idx, last_live, last_alert = 0, 0.0, {}
    totals, counts, events = [], {"PASS": 0, "WARNING": 0, "REJECT": 0}, 0
    t_start = time.perf_counter()
    try:
        while True:
            if a.max_seconds and time.perf_counter() - t_start > a.max_seconds:
                break
            item = src.newest(last_idx)
            if item is None:
                if src.ended:
                    break
                time.sleep(0.005)
                continue
            idx, frame = item
            last_idx = idx
            img = to_pil(frame)
            recs, verdict, tier, top, lat = inspect(det, img)
            totals.append(lat["total_ms"])
            counts[verdict] += 1
            now = time.perf_counter()
            name = f"frame_{idx:06d}.jpg"
            entry = {"type": "inspection", "live": True, "frame": idx,
                     "time": datetime.now().isoformat(timespec="seconds"), "image": name,
                     "verdict": verdict, "top_severity": tier, "n_defects": len(recs),
                     "detections": [{k: r[k] for k in ("class", "conf", "severity", "area_pct", "location", "box")} for r in recs],
                     "latency_ms": {k: round(v, 1) for k, v in lat.items()},
                     "advice_fixed": ({"cause": top["cause"], "action": top["action"], "disposition": top["disposition"]}
                                      if top else None)}
            with open(insp_log, "a") as f:
                f.write(json.dumps(entry) + "\n")

            is_ev = is_event(last_alert, top, verdict, now, a.cooldown)
            if is_ev or verdict != "PASS":
                ann_dir = os.path.join(a.out, "annotated")
                infer.annotate(img, recs, verdict, os.path.join(ann_dir, name))
                if idx % 100 == 0:                                   # keep only the newest 600 images
                    for old_f in sorted(os.listdir(ann_dir))[:-600]:
                        try:
                            os.remove(os.path.join(ann_dir, old_f))
                        except OSError:
                            pass
            if is_ev:
                events += 1
                print(f"[frame {idx}] {verdict:7} {top['class']} {top['severity']} conf {top['conf']} -> {top['disposition']}")
                if worker:
                    worker.submit(name, top)
            if now - last_live >= a.live_every:
                infer.annotate(img, recs, verdict, live_tmp)
                os.replace(live_tmp, live_file)                      # atomic: the dashboard never reads a half-written file
                last_live = now
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        src.close()

    wall = time.perf_counter() - t_start
    read = src.frames_read()
    if worker:
        print("Waiting for Llama jobs to finish...")
        worker.close()
    if totals:
        s = sorted(totals)
        print(f"\nFrames read {read}, inspected {len(totals)}, dropped {read - len(totals)} (inspection slower than the source)")
        print(f"Frames: PASS {counts['PASS']} WARNING {counts['WARNING']} REJECT {counts['REJECT']} | alert events {events}")
        print(f"Inspection latency: mean {statistics.mean(totals):.1f} ms, p95 {s[min(len(s) - 1, int(0.95 * len(s)))]:.1f} ms; "
              f"end-to-end rate {len(totals) / wall:.1f} inspected frames per second over {wall:.0f} s")


if __name__ == "__main__":
    main()
