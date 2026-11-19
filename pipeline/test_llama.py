"""Run one FabricGuard advisory prompt through llama_main (ExecuTorch) and time it.

Run from ~/fabric_project/pipeline (needs severity.py next to it):
    python test_llama.py
"""
import os
import re
import subprocess
import time

from severity import build_record, build_prompt, check_llama_output, fallback_alert, suggested_seq_len, est_tokens

RUNNER = os.path.expanduser("~/executorch/cmake-out/examples/models/llama/llama_main")
MODEL = os.path.expanduser("~/fabric_project/llm/llama32_1b_8da4w.pte")
TOKENIZER = os.path.expanduser("~/llama-3.2-1b/original/tokenizer.model")

for path in (RUNNER, MODEL, TOKENIZER):
    if not os.path.exists(path):
        raise SystemExit(f"Missing file: {path}")

rec = build_record("Hole", 0.81, (100, 120, 180, 200), 640, 480)
prompt = build_prompt(rec)
seq_len = suggested_seq_len(prompt)
print(f"Prompt chars: {len(prompt)} | est tokens: {est_tokens(prompt)} | seq_len: {seq_len}")

cmd = [RUNNER, f"--model_path={MODEL}", f"--tokenizer_path={TOKENIZER}",
       "--temperature=0", f"--seq_len={seq_len}", "--prompt=" + prompt]

t0 = time.time()
out = subprocess.run(cmd, capture_output=True, text=True)
wall = time.time() - t0

combined = out.stdout + "\n" + out.stderr
print("\n----- last part of stdout -----")
print(out.stdout[-1800:])
print("\n----- last part of stderr (stats are usually here) -----")
print(out.stderr[-1200:])

# try to pull out the reply: text after the prompt's final assistant header
reply = ""
marker = "<|start_header_id|>assistant<|end_header_id|>"
if marker in out.stdout:
    reply = out.stdout.split(marker)[-1]
reply = re.sub(r"<\|.*?\|>", "", reply).strip()

ok, clean = check_llama_output(reply)
print("\n----- result -----")
print("Reply            :", clean if clean else "(could not extract)")
print("Passes shape check:", ok)
print("Fallback text    :", fallback_alert(rec))
print("Wall seconds     :", round(wall, 2))
print("Return code      :", out.returncode)
