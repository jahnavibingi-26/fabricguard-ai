"""Severity rules + structured record + Llama prompt builder.

Input: detections from YOLO as dicts {cls, conf, xyxy} in pixel coords.
Output: an inspection record (JSON-serialisable) that is logged and fed to Llama.
Llama never sees pixels, only this record.
"""
from dataclasses import dataclass, asdict

# Tier 3 = structural/functional risk, 2 = material integrity or process, 1 = mostly cosmetic.
# Tiers are an engineering assumption: state this in the report and tune with domain input.
CLASS_TIER = {
    "crease": 3, "waist_folding": 3, "crescent_gap": 3,
    "inclusion": 2, "rolled_pit": 2, "punching_hole": 2, "welding_line": 2,
    "water_spot": 1, "oil_spot": 1, "silk_spot": 1,
}

# Minimum confidence to raise an alert. Tune on the val set (trade misses vs false alarms).
MIN_CONF = {c: 0.30 for c in CLASS_TIER}
MIN_CONF.update({"rolled_pit": 0.25, "crease": 0.25, "inclusion": 0.25})  # weak classes: favour recall

LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

# Fixed, expert-style advice per class: (cause, action). This is the guaranteed advice
# and the fallback if Llama fails. ASSUMPTION: wording is engineering common sense, not
# validated by a steel-mill expert; state this in the report.
ADVICE = {
    "punching_hole": ("Unwanted holes, likely from punching die wear or strip misalignment.",
                      "Inspect the punching die and strip alignment."),
    "welding_line": ("Weld seam visible in the strip, likely from a joining weld in the coil.",
                     "Check weld quality and mark the seam for removal or downgrade."),
    "crescent_gap": ("Crescent-shaped gap, likely from shear cutter clearance or wear.",
                     "Check shear blade clearance and condition."),
    "water_spot": ("Water stains, likely from cooling or storage.",
                   "Check the drying section and storage conditions."),
    "oil_spot": ("Oil contamination, likely from a lubricant leak.",
                 "Check for lubricant leaks and inspect the cleaning section."),
    "silk_spot": ("Silk-like surface marks, likely from roll condition or rolling parameters.",
                  "Inspect the roll surface and rolling parameters."),
    "inclusion": ("Foreign material embedded in the surface, likely from upstream contamination.",
                  "Flag the coil section and check upstream cleanliness."),
    "rolled_pit": ("Pits pressed into the surface, likely from roll damage or debris on the rolls.",
                   "Inspect the rolls for damage or debris."),
    "crease": ("Fold line in the strip, likely from uneven strip tension or coiling.",
               "Check tension and coiling settings."),
    "waist_folding": ("Waist-shaped fold, likely from tension or leveling problems.",
                      "Check tension and leveler settings."),
}

# Disposition by overall severity (deterministic, never produced by Llama).
DISPOSITION = {
    "CRITICAL": "Stop the line and hold the coil.",
    "HIGH": "Hold the affected section for inspection.",
    "MEDIUM": "Flag the section for review.",
    "LOW": "Log and monitor.",
}


@dataclass
class Defect:
    cls: str
    conf: float
    area_pct: float
    location: str
    severity: str


def location_label(cx, cy, w, h):
    col = "left" if cx < w / 3 else "right" if cx > 2 * w / 3 else "center"
    row = "top" if cy < h / 3 else "bottom" if cy > 2 * h / 3 else "middle"
    return f"{row}-{col}"


def _severity(cls, conf, area_pct, n_defects):
    score = CLASS_TIER[cls]            # 1..3
    if area_pct >= 5.0:
        score += 2
    elif area_pct >= 1.0:
        score += 1
    if conf >= 0.7:
        score += 1
    if n_defects >= 3:
        score += 1
    # map score (1..7) to level
    if score >= 6:
        return "CRITICAL"
    if score >= 5:
        return "HIGH"
    if score >= 3:
        return "MEDIUM"
    return "LOW"


def build_record(detections, img_w, img_h, image_id="", ts=""):
    kept = [d for d in detections if d["conf"] >= MIN_CONF[d["cls"]]]
    defects = []
    for d in kept:
        x1, y1, x2, y2 = d["xyxy"]
        area_pct = 100.0 * max(0, x2 - x1) * max(0, y2 - y1) / (img_w * img_h)
        loc = location_label((x1 + x2) / 2, (y1 + y2) / 2, img_w, img_h)
        defects.append(Defect(d["cls"], round(d["conf"], 2), round(area_pct, 2), loc,
                              _severity(d["cls"], d["conf"], area_pct, len(kept))))
    defects.sort(key=lambda x: (-LEVELS.index(x.severity), -x.conf))
    overall = defects[0].severity if defects else "OK"
    return {
        "image_id": image_id, "timestamp": ts,
        "status": "DEFECT" if defects else "PASS",
        "overall_severity": overall,
        "defect_count": len(defects),
        "defects": [asdict(x) for x in defects],
    }


def cache_key(record):
    """Advice is cached per (top class, overall severity) so Llama runs rarely."""
    if not record["defects"]:
        return None
    return (record["defects"][0]["cls"], record["overall_severity"])


SYSTEM = ("You write short alerts for a steel strip inspection line. Use only the facts given. "
          "Reply in exactly two sentences: 'Cause:' then 'Action:'. Mention severity and location in the first.")

_EX_USER = ("Defect: water_spot. Severity: LOW. Location: top-left. "
            "Cause: Water stains, likely from cooling or storage. "
            "Action: Check the drying section and storage conditions. Write the alert.")
_EX_ASSIST = ("Cause: Low-severity water stains at top-left, likely from cooling or storage. "
              "Action: Check the drying section and storage conditions.")


def _turn(role, text):
    return f"<|start_header_id|>{role}<|end_header_id|>\n\n{text}<|eot_id|>"


def build_prompt(record):
    """Llama 3 chat format, one worked example. No <|begin_of_text|>: runner adds BOS."""
    top = record["defects"][0]
    cause, action = ADVICE[top["cls"]]
    others = len(record["defects"]) - 1
    extra = f" Other defects in frame: {others}." if others else ""
    user = (f"Defect: {top['cls']}. Severity: {record['overall_severity']}. "
            f"Location: {top['location']}.{extra} Cause: {cause} Action: {action} Write the alert.")
    return (_turn("system", SYSTEM) + _turn("user", _EX_USER) + _turn("assistant", _EX_ASSIST)
            + _turn("user", user)
            + "<|start_header_id|>assistant<|end_header_id|>\n\n")


def fallback_alert(record):
    """Deterministic alert from the fixed table. Used if Llama is slow, fails or looks wrong."""
    top = record["defects"][0]
    cause, action = ADVICE[top["cls"]]
    return (f"Cause: {record['overall_severity'].capitalize()}-severity {top['cls'].replace('_', ' ')} "
            f"at {top['location']}. {cause} Action: {action}")


def check_llama_output(text, record):
    """Accept Llama text only if it has the expected shape; else return the fallback."""
    t = text.replace("<|eot_id|>", "").strip()
    ok = t.startswith("Cause:") and "Action:" in t and len(t.split()) <= 70
    return t if ok else fallback_alert(record)


if __name__ == "__main__":
    demo = [{"cls": "crease", "conf": 0.89, "xyxy": (300, 100, 900, 700)},
            {"cls": "water_spot", "conf": 0.41, "xyxy": (1500, 800, 1600, 900)}]
    rec = build_record(demo, 2048, 1000, "demo.jpg")
    print(rec)
    print(build_prompt(rec))
    print("FALLBACK:", fallback_alert(rec))
    print("DISPOSITION:", DISPOSITION[rec["overall_severity"]])
