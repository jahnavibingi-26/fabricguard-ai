"""FabricGuard AI - severity rules, advice table and Llama prompt helpers (v1, fabric).

Pipeline role (vision side feeds the advisory side):
  YOLO detections -> build_record() -> frame_verdict() -> PASS / WARNING / REJECT
  record -> build_prompt() -> Llama 3.2 1B (ExecuTorch) -> check_llama_output()
  if the Llama text fails the shape check, fallback_alert() is shown instead.

All thresholds and the ADVICE text are fixed, expert-style assumptions that
have NOT been validated by a textile engineer. State this in the report.
"""

# ---------------------------------------------------------------- config
MIN_CONF = 0.15              # detections below this are ignored (chosen on the validation set)
REJECT_MIN_CONF = 0.50       # below this, a detection can never exceed MEDIUM (-> WARNING)
AREA_BUMP_1 = 0.05           # box area share of image -> +1 severity level
AREA_BUMP_2 = 0.20           # box area share of image -> +2 severity levels in total
LINE_SPAN_BUMP = 0.50        # a line mark spanning >= half the frame -> +1 (set to 2.0 to disable)
FRAME_COUNT_ESCALATE = 4     # this many defects in one frame -> frame severity +1
MAX_WORDS = 70               # Llama reply shape check

# Per-class REJECT confidence. Classes not listed use REJECT_MIN_CONF. Weak classes (test AP50 below
# about 0.55) can be given a stricter value so a doubtful detection becomes WARNING, not REJECT.
WEAK_CLASSES = ("Hole", "Needle Mark", "Missing Yarn", "Stain")
REJECT_MIN_CONF_BY_CLASS = {c: 0.70 for c in WEAK_CLASSES}   # chosen on the validation set

TIERS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

# Base severity per class (index into TIERS). Order = data.yaml class order.
BASE_TIER = {
    "DropStitch": 2,            # structural: loop missing, can run into a ladder
    "Hole": 2,                  # structural: fabric broken (CRITICAL only if large)
    "Horizontal LineMark": 1,
    "Knot": 1,
    "Oil Spot": 1,
    "Setup": 1,
    "Slub": 0,
    "Vertical LineMark": 1,
    "Yarn Dust": 0,
    "Needle Mark": 1,
    "Missing Yarn": 2,          # structural: yarn end absent
    "Stain": 1,
}

# {class: (likely cause, recommended action)} - fixed text, unvalidated.
ADVICE = {
    "DropStitch": ("a needle failed to form a loop, often a worn or bent needle or a yarn tension fault",
                   "stop the machine, replace the damaged needle, check yarn tension and mark the roll section"),
    "Hole": ("yarn breakage, snagging or needle damage has torn the fabric",
             "mark and cut out the section, then check the yarn path and needles for damage"),
    "Horizontal LineMark": ("uneven yarn feed or tension variation across one course",
                            "check the yarn feeder and tension, then inspect the next metre of fabric"),
    "Vertical LineMark": ("a damaged or misaligned needle or sinker is leaving a line along the fabric length",
                          "inspect needles and sinkers at the line position and replace the damaged one"),
    "Knot": ("a yarn joint or knot has passed through the machine",
             "mark the knot, check the yarn package splices and the yarn guides"),
    "Oil Spot": ("oil or lubricant from machine parts has contaminated the fabric",
                 "check lubrication points for leaks, clean the area and isolate the stained section"),
    "Setup": ("a machine restart or adjustment has left setup marks",
              "review recent machine stops and settings, and mark the affected section"),
    "Slub": ("a thick or uneven section of yarn or a fibre lump has been knitted in",
             "check the yarn lot quality and clean the yarn guides"),
    "Yarn Dust": ("lint or fibre dust has settled on the fabric",
                  "clean the machine and surrounding area and check the airflow and take-up area"),
    "Needle Mark": ("worn or over-tight needles are marking the fabric",
                    "inspect needle condition and fabric tension, and replace worn needles"),
    "Missing Yarn": ("a yarn end broke or was not fed, leaving a gap in the fabric",
                     "check the yarn feed and break sensors and re-thread the missing end"),
    "Stain": ("contamination from handling, dye or dirt has marked the fabric",
              "find the source, clean the handling points and isolate the stained section"),
}

DISPOSITION = {
    "LOW": "Log and watch the next metres",
    "MEDIUM": "Flag for operator check",
    "HIGH": "Stop line and hold this roll section",
    "CRITICAL": "Stop line and reject this roll section",
}

VERDICT_OF_TIER = {"LOW": "WARNING", "MEDIUM": "WARNING", "HIGH": "REJECT", "CRITICAL": "REJECT"}


# ---------------------------------------------------------------- records
def _norm(name):
    return "".join(ch for ch in name.lower() if ch.isalnum())

_CLASS_LOOKUP = {_norm(k): k for k in BASE_TIER}


def canonical_class(name):
    """Map a model class name to the table key (case/space/underscore tolerant)."""
    return _CLASS_LOOKUP.get(_norm(name))


def location_text(cx, cy, w, h):
    col = "left" if cx < w / 3 else "right" if cx > 2 * w / 3 else "centre"
    row = "upper" if cy < h / 3 else "lower" if cy > 2 * h / 3 else "middle"
    text = f"{row} {col}" if col != "centre" or row != "middle" else "centre"
    if cx < 0.1 * w or cx > 0.9 * w:
        text += " (near the selvedge)"
    return text


def build_record(cls_name, conf, xyxy, img_w, img_h):
    """One detection -> record dict, or None if filtered out / unknown class."""
    cls = canonical_class(cls_name)
    if cls is None or conf < MIN_CONF:
        return None
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    bw, bh = max(x2 - x1, 0.0), max(y2 - y1, 0.0)
    area_share = (bw * bh) / float(img_w * img_h)

    level = BASE_TIER[cls]
    if cls in ("Horizontal LineMark", "Vertical LineMark"):
        # line boxes are long and thin by nature, so box area would double-count: use span only
        span = bw / img_w if cls == "Horizontal LineMark" else bh / img_h
        level += int(span >= LINE_SPAN_BUMP)
    else:
        level += (area_share >= AREA_BUMP_1) + (area_share >= AREA_BUMP_2)
    level = min(level, len(TIERS) - 1)
    if conf < REJECT_MIN_CONF_BY_CLASS.get(cls, REJECT_MIN_CONF):
        level = min(level, 1)          # low confidence never goes straight to REJECT
    tier = TIERS[level]
    return {
        "class": cls,
        "conf": round(float(conf), 2),
        "area_pct": round(100 * area_share, 1),
        "location": location_text((x1 + x2) / 2, (y1 + y2) / 2, img_w, img_h),
        "severity": tier,
        "cause": ADVICE[cls][0],
        "action": ADVICE[cls][1],
        "disposition": DISPOSITION[tier],
    }


def frame_verdict(records):
    """List of records -> (verdict, top_tier, top_record). No detections = PASS."""
    if not records:
        return "PASS", None, None
    top = max(records, key=lambda r: (TIERS.index(r["severity"]), r["conf"]))
    level = TIERS.index(top["severity"])
    if len(records) >= FRAME_COUNT_ESCALATE:
        level = min(level + 1, len(TIERS) - 1)
    tier = TIERS[level]
    return VERDICT_OF_TIER[tier], tier, top


# ---------------------------------------------------------------- Llama side
PROMPT_VERSION = "v3"   # v1: 33/36 shape checks, v2: 31/36 (all failures were refusals on HIGH/CRITICAL prompts);
                        # v3: severity and area are NOT sent to Llama (the rules and the screen already show them)

SYSTEM_MSG = ("You are a fabric quality assistant. Use only the facts given. "
              "Reply in exactly two parts: 'Cause:' then 'Action:'. Maximum 60 words.")

_EXAMPLE_USER = ("Defect: Oil Spot. Location: upper left. "
                 "Known cause: oil or lubricant from machine parts has contaminated the fabric. "
                 "Known action: check lubrication points for leaks, clean the area and isolate the stained section.")
_EXAMPLE_REPLY = ("Cause: Oil or lubricant from a machine part has reached the fabric near the upper left. "
                  "Action: Check the lubrication points for leaks, clean the area and set the stained "
                  "section aside for the operator to check.")


def _turn(role, text):
    return f"<|start_header_id|>{role}<|end_header_id|>\n\n{text}<|eot_id|>"


def build_prompt(rec):
    """Llama 3 chat prompt with ONE worked example. No <|begin_of_text|>: the runner adds BOS."""
    user = (f"Defect: {rec['class']}. Location: {rec['location']}. "
            f"Known cause: {rec['cause']}. Known action: {rec['action']}.")
    return (_turn("system", SYSTEM_MSG) + _turn("user", _EXAMPLE_USER) + _turn("assistant", _EXAMPLE_REPLY)
            + _turn("user", user) + "<|start_header_id|>assistant<|end_header_id|>\n\n")


def est_tokens(prompt):
    """Rough token estimate (about 3.5 characters per token) for choosing --seq_len."""
    return int(len(prompt) / 3.5) + 1


def suggested_seq_len(prompt, reply_tokens=80, cap=512):
    """--seq_len is prompt + reply TOTAL. Always confirm with the runner's own token count."""
    return min(est_tokens(prompt) + reply_tokens, cap)


def fallback_alert(rec):
    return f"Cause: {rec['cause'].capitalize()}. Action: {rec['action'].capitalize()}."


def check_llama_output(text):
    """Accept only 'Cause: ... Action: ...' of at most MAX_WORDS words. Returns (ok, clean_text)."""
    t = (text or "").strip()
    ok = t.startswith("Cause:") and "Action:" in t and len(t.split()) <= MAX_WORDS
    return ok, t


# ---------------------------------------------------------------- demo
if __name__ == "__main__":
    W, H = 640, 480
    demo = [("Hole", 0.81, (100, 120, 180, 200)),
            ("Stain", 0.42, (400, 300, 470, 360)),
            ("Horizontal LineMark", 0.66, (20, 240, 600, 260))]
    recs = [r for r in (build_record(c, p, b, W, H) for c, p, b in demo) if r]
    verdict, tier, top = frame_verdict(recs)
    print("VERDICT:", verdict, "| top tier:", tier)
    for r in recs:
        print(" -", r["class"], r["severity"], r["conf"], r["area_pct"], "%", r["location"])
    p = build_prompt(top)
    print("\nPrompt chars:", len(p), "| est tokens:", est_tokens(p), "| seq_len:", suggested_seq_len(p))
    print("Fallback:", fallback_alert(top))
    print("Shape check on good text:", check_llama_output(_EXAMPLE_REPLY)[0])
    print("Shape check on refusal  :", check_llama_output("I can't help with that.")[0])
    print("No detections ->", frame_verdict([])[0])
