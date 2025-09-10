# backend/symptom_rules.py
# ------------------------------------------------------------
# Rule-based scorer (transparent first stage of the pipeline)
#
# What this file does
#   • Looks for specific breast-related terms and patterns in free text.
#   • Adds weights for matched patterns (LOW/MEDIUM/HIGH signals).
#   • Applies a few "always HIGH" overrides for classic red flags.
#   • Returns:
#       - score (int)
#       - matches (list[str])  → short labels describing what we recognised
#
# How the score maps to risk
#   >= 999  → HIGH   (sentinel used for any override match)
#   >=   5  → HIGH
#      3–4  → MODERATE
#      0–2  → LOW
#
# Negation
#   Light guards skip patterns like "no discharge" or "not inverted".
# ------------------------------------------------------------

import re
from typing import List, Tuple

# Words that count as “breast context” (used by some patterns)
BREAST_CTX = r"(breast|boob|chest|nipple|areola|underarm|armpit|axilla|axillary)"

# Light negation guards (skip a match if clearly negated in the same sentence)
NEG_NO_DISCHARGE = re.compile(
    r"\b(no|not|never|without)\b.*\b(discharge|leak(?:ing)?|fluid)\b", re.I
)
NEG_NO_INVERT = re.compile(
    r"\b(no|not|never|without)\b.*\b(invert(?:ed|ing)?|retract(?:ed|ing)?|"
    r"pulled\s*in|turned\s*in(?:ward|wards)?)\b",
    re.I,
)

# ------------------------------------------------------------
# NHS-aligned patterns  →  (regex, weight, label)
# Notes:
#   • HIGH: strong red flags (also covered by overrides below).
#   • MODERATE: visible/structural changes that are concerning.
#   • LOW: non-specific symptoms that still deserve sensible advice.
# ------------------------------------------------------------
RAW_PATTERNS: List[Tuple[str, int, str]] = [
    # ===== HIGH red flags (pattern-level; overrides also catch these) =====

    # Skin changes: dimpling / peau d’orange / marked redness
    (
        rf"(?=.*\b(breast|skin|nipple)\b)"
        rf"(?=.*\b(dimpl(?:e|ing)|pucker(?:ing)?|peau\s*d'?orange|orange\s*peel|"
        rf"golf\s*ball|pitted|bumpy|textur(?:e|ed)|dent(?:ed)?|red(?:ness)?)\b)",
        4,
        "skin changes (dimpling/orange-peel/redness)",
    ),

    # ===== MODERATE (concerning but not auto-HIGH) =====

    # Nipple inversion / retraction (generic; “new” is handled by HIGH override)
    (
        r"(?=.*\bnipples?\b)(?=.*\b("
        r"invert(?:ed|ing)?|"
        r"retract(?:ed|ing)?|"
        r"turn(?:ed|ing)\s*in(?:ward|wards)?|"
        r"pull(?:ed|ing)\s*in|"
        r"point(?:ing|ed)?\s*in(?:ward|wards)?|"
        r"gone\s*in|going\s*in"
        r")\b)",
        3,
        "nipple inversion/retraction",
    ),

    # Clear/watery discharge with one-sided or spontaneous feature
    (
        rf"(?=.*\b(nipple|breast)\b)(?=.*\b(discharge|leak(?:ing)?|fluid)\b)"
        rf"(?=.*\b(clear|watery)\b)"
        rf"(?=.*\b(one\s*(side|breast|nipple)|single|unilateral|without\s*(touch|press|squeez)|"
        rf"spontaneous|happen(?:s|ing)?\s*without)\b)",
        3,
        "nipple discharge with concerning features",
    ),

    # Change in breast size/shape (requires explicit breast context)
    (
        rf"(?=.*\b{BREAST_CTX}\b)(?=.*\bchange(?:s)?\b)(?=.*\b(size|shape)\b)",
        3,
        "change in breast size/shape",
    ),

    # ===== LOW (non-specific) =====

    # Non-bloody nipple discharge
    (
        rf"(?=.*\b(nipple|breast)\b)(?=.*\b(discharge|leak(?:ing)?|fluid)\b)"
        rf"(?!.*\b(blood|bloody|red)\b)",
        2,
        "non-bloody nipple discharge",
    ),

    # Underarm / armpit pain
    (
        rf"(?=.*\b(underarm|armpit|axilla|axillary)\b)(?=.*\bpain\b)",
        2,
        "underarm/armpit pain",
    ),

    # Persistent breast pain/tenderness (needs breast context)
    (
        rf"(?=.*\b{BREAST_CTX}\b)(?=.*\b(persistent|constant|ongoing|continuous)\b)"
        rf"(?=.*\b(pain|ache|tender(ness)?)\b)",
        2,
        "persistent breast pain/tenderness",
    ),

    # Very general low signals
    (r"\b(unexplained|unintentional)\b.*\b(weight\s*loss|loss\s*of\s*appetite)\b", 1, "unexplained weight loss/appetite"),
    (r"\bfatigue|tired(ness)?\b", 1, "fatigue/tiredness"),
    (r"\btender(ness)?\b", 1, "tenderness"),
    (r"\bache\b", 1, "ache"),
    (r"\bitch(?:y|ing|iness)\b", 1, "itchiness"),
    (rf"(?=.*\b{BREAST_CTX}\b)\bpain\b", 1, "pain"),
]

# Compile once for speed
PATTERNS: List[Tuple[re.Pattern, int, str]] = [
    (re.compile(p, re.I), w, label) for (p, w, label) in RAW_PATTERNS
]

# ------------------------------------------------------------
# Always-HIGH overrides
# If any of these matches, we short-circuit to score=999 (HIGH).
# ------------------------------------------------------------
OVERRIDES_RAW: List[Tuple[str, str]] = [
    # Bloody nipple discharge → HIGH
    (
        rf"(?=.*\b(nipple|breast)\b)(?=.*\b(discharge|leak(?:ing)?|fluid)\b)(?=.*\b(blood|bloody|red)\b)",
        "bloody nipple discharge (override)",
    ),

    # NEW nipple inversion/retraction → HIGH
    (
        r"(?=.*\bnipples?\b)"
        r"(?=.*\b(new|recent|recently|sudden(?:ly)?|just|started)\b)"
        r"(?=.*\b("
        r"invert(?:ed|ing)?|"
        r"retract(?:ed|ing)?|"
        r"turn(?:ed|ing)\s*in(?:ward|wards)?|"
        r"pull(?:ed|ing)\s*in|"
        r"point(?:ing|ed)?\s*in(?:ward|wards)?|"
        r"gone\s*in|going\s*in"
        r")\b)",
        "new nipple inversion/retraction (override)",
    ),

    # Inversion + any discharge together → HIGH
    (
        rf"(?=.*\bnipple\b)(?=.*\b(invert(?:ed|ing)?|pulled\s*in|retract(?:ed|ing)?|"
        rf"turned\s*in(?:ward|wards)?|gone\s*in)\b)"
        rf"(?=.*\b(discharge|leak(?:ing)?|fluid)\b)",
        "nipple discharge + inversion (override)",
    ),

    # Lump/swelling in breast/chest/armpit → HIGH
    (
        rf"(?=.*\b(lump|swelling)\b)(?=.*\b(breast|chest|armpit|underarm|axilla|axillary)\b)",
        "lump/swelling in breast/chest/armpit (override)",
    ),

    # Strong skin-change synonyms → HIGH
    (
        rf"(?=.*\b(breast|skin|nipple)\b)"
        rf"(?=.*\b(dimpl(?:e|ing)|pucker(?:ing)?|peau\s*d'?orange|orange\s*peel|"
        rf"golf\s*ball|pitted|bumpy|textur(?:e|ed)|dent(?:ed)?)\b)",
        "skin changes (override)",
    ),
]

HIGH_OVERRIDES: List[Tuple[re.Pattern, str]] = [
    (re.compile(p, re.I), label) for (p, label) in OVERRIDES_RAW
]

# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------
def rule_based_score(text: str) -> tuple[int, list[str]]:
    """
    Score free text and list the matched rule labels.

    Returns:
      (score, matches)
        - score: int total; 999 means an override set risk to HIGH
        - matches: list of short strings describing what we recognised
    """
    if not text:
        return 0, []
    t = re.sub(r"\s+", " ", text.strip().lower())

    # 1) Check always-HIGH overrides first (no negation here)
    for pat, label in HIGH_OVERRIDES:
        if pat.search(t):
            return 999, [label]

    # 2) Accumulate weights from pattern matches (with light negation guards)
    score = 0
    matches: list[str] = []
    for pat, w, label in PATTERNS:
        if pat.search(t):
            # Skip if clearly negated for the two ambiguous families
            if "discharge" in label and NEG_NO_DISCHARGE.search(t):
                continue
            if "inversion" in label and NEG_NO_INVERT.search(t):
                continue
            score += w
            matches.append(label)

    # De-duplicate labels while preserving order
    matches = list(dict.fromkeys(matches))
    return score, matches


def classify_risk(score: int) -> str:
    """
    Map score → risk band (used by app.py).
      >= 999 → HIGH  (override sentinel)
      >=   5 → HIGH
        3–4 → MODERATE
        0–2 → LOW
    """
    if score >= 999:
        return "HIGH"
    if score >= 5:
        return "HIGH"
    if score >= 3:
        return "MODERATE"
    return "LOW"
