#!/usr/bin/env bash
# Build ~/fabric_project/board_bundle : everything to copy to the KV260 (except llama_main for ARM, built later).
#
#   bash make_board_bundle.sh            # full bundle, includes the 915 MB .pte (takes a minute)
#   bash make_board_bundle.sh --no-pte   # skip the .pte (copy it to the board separately)
#
# It checks that your pipeline files are the LATEST versions before copying them.
set -u
P="$HOME/fabric_project"
B="$P/board_bundle"
DATA="$HOME/Downloads/fabric_merged/test"
TOK="$HOME/llama-3.2-1b/original/tokenizer.model"
PTE="$P/llm/llama32_1b_8da4w.pte"
SKIP_PTE=0
[ "${1:-}" = "--no-pte" ] && SKIP_PTE=1
FAIL=0
ok()   { echo "  OK       $1"; }
miss() { echo "  MISSING  $1"; FAIL=1; }

echo "== Checking source files in $P"
chk() {  # chk <file> <text that must be inside> <label>
  if [ ! -f "$1" ]; then miss "$1 (file not found)"; return; fi
  if grep -qF -- "$2" "$1"; then ok "$3"; else miss "$3  <- $1 is an OLD version"; fi
}
chk "$P/pipeline/severity.py"  'PROMPT_VERSION = "v3"'                       "severity.py (prompt v3)"
chk "$P/pipeline/severity.py"  'MIN_CONF = 0.15'                             "severity.py (validation-chosen thresholds)"
chk "$P/pipeline/infer.py"     'prompt v3 uses only class + location'        "infer.py (latest)"
chk "$P/pipeline/dashboard.py" 'id="livecard"'                               "dashboard.py (live panel)"
chk "$P/pipeline/live.py"      'class FrameSource'                           "live.py (camera / video)"
[ -f "$P/pipeline/test_llama.py" ] && ok "test_llama.py" || miss "test_llama.py"
[ -f "$P/pipeline/collect_metrics.py" ] && ok "collect_metrics.py" || miss "collect_metrics.py"
[ -f "$P/models/merged_best.onnx" ] && ok "models/merged_best.onnx" || miss "models/merged_best.onnx"
[ -f "$TOK" ] && ok "tokenizer.model" || miss "tokenizer.model ($TOK)"
if [ $SKIP_PTE -eq 0 ]; then [ -e "$PTE" ] && ok "llama .pte" || miss "llama .pte ($PTE)"; fi
[ -d "$DATA/images" ] && ok "test images folder" || miss "test images folder ($DATA)"

if [ $FAIL -ne 0 ]; then
  echo
  echo "Fix the MISSING items above, then run this script again. Nothing was copied."
  exit 1
fi

echo
echo "== Building $B"
rm -rf "$B"
mkdir -p "$B"/models "$B"/pipeline "$B"/samples "$B"/llm "$B"/bin "$B"/logs

cp "$P/models/merged_best.onnx" "$B/models/"
for f in infer.py severity.py dashboard.py live.py test_llama.py collect_metrics.py; do cp "$P/pipeline/$f" "$B/pipeline/"; done
cp "$TOK" "$B/llm/"
if [ $SKIP_PTE -eq 0 ]; then cp -L "$PTE" "$B/llm/llama32_1b_8da4w.pte"; fi

# sample images: 10 BDFD9 defect, 10 FabriD defect, 10 clean FabriD (chosen from the test split)
pick() {  # pick <prefix> <defect|clean> <count>
  local n=0 f b img
  for f in "$DATA"/labels/${1}_*.txt; do
    [ "$n" -ge "$3" ] && break
    if [ "$2" = "defect" ]; then grep -q . "$f" || continue; else grep -q . "$f" && continue; fi
    b=$(basename "$f" .txt)
    img=$(ls "$DATA"/images/"$b".* 2>/dev/null | head -1)
    [ -n "$img" ] && cp "$img" "$B/samples/" && n=$((n + 1))
  done
  echo "  samples: $n $1 $2 images"
}
pick bd defect 10
pick fd defect 10
pick fd clean 10

cat > "$B/requirements_board.txt" <<'EOF'
numpy
Pillow
onnxruntime
# only for live.py (camera / video):
opencv-python-headless
EOF

cat > "$B/board_check.sh" <<'EOF'
#!/usr/bin/env bash
# Run ON THE KV260:  bash board_check.sh   and paste the output.
echo "== OS";      cat /etc/os-release | head -4
echo "== CPU";     uname -m; nproc; grep -m1 "model name" /proc/cpuinfo
echo "== Python";  python3 --version
echo "== Memory";  free -h
echo "== Disk";    df -h / | tail -2
echo "== Network"; hostname -I
echo "== Cameras"; ls /dev/video* 2>/dev/null || echo "no /dev/video*"
echo "== Python packages"
python3 - <<'PY'
for m in ("numpy", "PIL", "onnxruntime", "cv2"):
    try:
        mod = __import__(m); print(f"  {m}: ok", getattr(mod, "__version__", ""))
    except Exception as e:
        print(f"  {m}: MISSING ({type(e).__name__})")
PY
echo "== Llama runner"; ls -l bin/llama_main 2>/dev/null || echo "  bin/llama_main not present yet (ARM build still to do)"
EOF

cat > "$B/BOARD_README.txt" <<'EOF'
FabricGuard AI - files for the KV260
====================================
models/merged_best.onnx   YOLOv8n, 12 classes, input 640x480 (vision pipeline, ONNX Runtime)
pipeline/                 infer.py (folder of images), live.py (camera / video), dashboard.py, severity.py, test_llama.py, collect_metrics.py
llm/                      llama32_1b_8da4w.pte + tokenizer.model (advisory pipeline, ExecuTorch)
bin/                      put the ARM build of llama_main here (not built yet)
samples/                  30 test images: 10 BDFD9 defect, 10 FabriD defect, 10 clean
SHA256SUMS                check the copy:  sha256sum -c SHA256SUMS

On the board, in this folder:
 1. bash board_check.sh                      (paste the output to Claude)
 2. python3 -m venv venv && . venv/bin/activate && pip install -r requirements_board.txt
 3. cd pipeline && python3 infer.py --source ../samples --out ../logs/vision --model ../models/merged_best.onnx
      -> prints ms per image and FPS (the on-device vision numbers)
 4. python3 dashboard.py --logs ../logs/vision --port 8000     (open http://<board-ip>:8000 from the laptop)
 5. Llama: needs bin/llama_main built for ARM, then:
      python3 infer.py --source ../samples --out ../logs/llama --llama \
          --runner ../bin/llama_main --pte ../llm/llama32_1b_8da4w.pte --tokenizer ../llm/tokenizer.model
 6. Camera: python3 live.py --source 0 --out ../logs/live --width 640 --height 480
 7. All report numbers in one run (vision latency/FPS, memory, power, Llama speed):
      python3 collect_metrics.py --label board --images ../samples --n 30 --llama-runs 3 \
          --runner ../bin/llama_main --pte ../llm/llama32_1b_8da4w.pte --tokenizer ../llm/tokenizer.model
      (power: add --power-file <hwmon file> --power-scale 1e-6 if the board exposes one; see the script header)
EOF

# checksums last, so they cover everything
( cd "$B" && find . -type f ! -name SHA256SUMS -print0 | xargs -0 sha256sum > SHA256SUMS )

echo
echo "== Done"
du -sh "$B"
echo "  files: $(find "$B" -type f | wc -l)  (samples: $(ls "$B"/samples | wc -l))"
ls -lh "$B/models" "$B/llm"
echo
echo "Copy to the board (replace USER and IP):  scp -r $B USER@IP:~/fabricguard"
echo "or copy the folder to a USB drive. Then on the board: cd fabricguard && sha256sum -c SHA256SUMS"
