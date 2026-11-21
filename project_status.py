#!/usr/bin/env python3
"""Read-only status report for ~/fabric_project. Changes nothing.

  source ~/yolo-venv/bin/activate
  python3 project_status.py
"""
import glob
import os
import re
import subprocess

H = os.path.expanduser("~")
P = f"{H}/fabric_project"


def sh(cmd, cwd=None):
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def mb(path):
    return f"{os.path.getsize(os.path.realpath(path)) / 1e6:,.1f} MB"


def line(label, path, note=""):
    exists = os.path.exists(path)
    size = mb(path) if exists and os.path.isfile(os.path.realpath(path)) else ""
    print(f"  [{'OK' if exists else '--'}] {label:34s} {size:>10s}  {note if exists else 'NOT FOUND: ' + path}")
    return exists


print("\n=== 1. FILES ===")
line("Final model .pt", f"{P}/models/merged_best.pt")
line("Final model .onnx", f"{P}/models/merged_best.onnx")
for f in ("severity.py", "infer.py", "eval_image_level.py", "test_llama.py", "test_llama_all.py"):
    line(f"pipeline/{f}", f"{P}/pipeline/{f}")
line("data_prep/prep_bdfd9.py", f"{P}/data_prep/prep_bdfd9.py")
line("data_prep/merge_fabrics.py", f"{P}/data_prep/merge_fabrics.py")
line("Merged dataset (data.yaml)", f"{H}/Downloads/fabric_merged/data.yaml")
line("Llama .pte", f"{P}/llm/llama32_1b_8da4w.pte")
line("Llama tokenizer", f"{H}/llama-3.2-1b/original/tokenizer.model")
runner = f"{H}/executorch/cmake-out/examples/models/llama/llama_main"
line("llama_main (laptop build)", runner)
print("  -- results saved:")
for f in ("image_level_valid.csv", "image_level_test.csv", "split_summary.csv", "split_per_class.csv",
          "llama_12class.csv"):
    line(f"evaluation/{f}", f"{P}/evaluation/{f}")
line("README.md", f"{P}/README.md")

print("\n=== 2. SEVERITY SETTINGS IN severity.py ===")
sev = f"{P}/pipeline/severity.py"
if os.path.exists(sev):
    for l in open(sev):
        if re.match(r"^(MIN_CONF|REJECT_MIN_CONF|REJECT_MIN_CONF_BY_CLASS|WEAK_CLASSES)\b", l):
            print("  " + l.rstrip())
    print("  expected: MIN_CONF = 0.15, REJECT_MIN_CONF = 0.5, weak classes (Hole, Needle Mark, Missing Yarn, Stain) = 0.7")

print("\n=== 3. ONNX MODEL ===")
onnx = f"{P}/models/merged_best.onnx"
if os.path.exists(onnx):
    try:
        import onnxruntime as ort
        s = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
        i, o = s.get_inputs()[0], s.get_outputs()[0]
        print(f"  loads OK. input {i.name} {i.shape}, output {o.name} {o.shape}")
        print("  expected input [1, 3, 480, 640] (height 480, width 640)")
    except Exception as e:
        print("  could not load:", e)

print("\n=== 4. LLAMA RUNNER ARCHITECTURE (the Arm requirement) ===")
cands = [runner] + glob.glob(f"{H}/executorch/**/llama_main*", recursive=True)
seen = set()
for c in cands:
    if c in seen or not os.path.isfile(c):
        continue
    seen.add(c)
    print(f"  {c}\n     -> {sh(f'file -b {c!r}')[:110]}")
arm = any("aarch64" in sh(f"file -b {c!r}").lower() or "arm" in sh(f"file -b {c!r}").lower() for c in seen)
print(f"  aarch64/Arm runner found: {'YES' if arm else 'NO - the KV260 needs one built for aarch64'}")

print("\n=== 5. GIT ===")
if os.path.isdir(f"{P}/.git"):
    print("  branch :", sh("git branch --show-current", P) or "(none)")
    print("  remote :", sh("git remote -v", P).splitlines()[0] if sh("git remote -v", P) else "NONE (not connected to GitHub)")
    print("  commits:", sh("git log --oneline | head -5", P) or "none")
    unpushed = sh("git log @{u}..HEAD --oneline 2>&1", P)
    print("  unpushed:", "no upstream set (nothing pushed yet)" if "fatal" in unpushed or "no upstream" in unpushed
          else (unpushed or "none"))
    st = sh("git status --short | head -15", P)
    print("  uncommitted changes:", st if st else "none")
    tracked = sh("git ls-files", P).splitlines()
    print(f"  tracked files: {len(tracked)} (a clean repo should be roughly 120-140)")
    bad = sorted({t.split('/')[0] for t in tracked if t.split('/')[0] in ("logs", "logs_clean", "logs_clean_015",
                                                                           "results", "data_clean_test")})
    print("  folders that should NOT be tracked but are:", bad or "none")
    big = sorted(((os.path.getsize(f"{P}/{t}"), t) for t in tracked if os.path.isfile(f"{P}/{t}")), reverse=True)[:3]
    print("  biggest tracked files:", [(t, f"{s / 1e6:.1f} MB") for s, t in big])
    pte = [t for t in tracked if t.endswith((".pte", ".pth", ".zip"))]
    print("  .pte/.pth/.zip tracked:", pte or "none (good)")
else:
    print("  no git repo found at", P)

print("\n=== 6. NOT DONE (checked from the files above) ===")
todo = []
if not arm:
    todo.append("aarch64 llama_main (cross-build on laptop) and Llama test on the board")
todo.append("on-device measurements (FPS, latency, tokens/s, RAM): no board_* results files" if not glob.glob(f"{P}/evaluation/board*") else None)
if not os.path.exists(f"{P}/README.md"):
    todo.append("README.md (two pipelines documented separately, dataset credits)")
if not os.path.exists(f"{P}/evaluation/llama_12class.csv"):
    todo.append("Llama 12-class shape-check run (test_llama_all.py)")
for t in filter(None, todo):
    print("  - " + t)
print("  - operator screen and demo video (not checked by this script)")
print("  - technical report (not checked by this script)\n")
