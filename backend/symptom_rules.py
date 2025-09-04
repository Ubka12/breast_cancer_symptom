# backend/symptom_rules.py
import re
from typing import List, Tuple

# Common context terms seen in your stack
BREAST_CTX = r"(breast|boob|chest|nipple|areola|underarm|armpit|axilla|axillary)"

# Simple negation guards (keep lightweight)
NEG_NO_DISCHARGE = re.compile(r"\b(no|not|never|without)\b.*\b(discharge|leak(?:ing)?|fluid)\b", re.I)
NEG_NO_INVERT    = re.compile(r"\b(no|not|never|without)\b.*\b(invert(?:ed|ing)?|retract(?:ed|ing)?|pulled\s*in|turned\s*in(?:ward|wards)?)\b", re.I)

# ------- NHS-aligned patterns (regex, weight, label) -------
RAW_PATTERNS: List[Tuple[str, int, str]] = [
    # ---- HIGH red flags (also enforced via overrides below) ----
    # Skin changes consistent with peau d’orange / dimpling / marked redness
    (rf"(?=.*\b(breast|skin|nipple)\b)(?=.*\b(dimpl(?:e|ing)|pucker(?:ing)?|peau\s*d'?orange|orange\s*peel|golf\s*ball|pitted|bumpy|textur(?:e|ed)|dent(?:ed)?|red(?:ness)?)\b)",
     4, "skin changes (dimpling/orange-peel/redness)"),

    # ---- MODERATE (visible/structural changes without “new”) ----
   # Nipple inversion/retraction (generic pattern)
(r"(?=.*\bnipples?\b)(?=.*\b("
 r"invert(?:ed|ing)?|"
 r"retract(?:ed|ing)?|"
 r"turn(?:ed|ing)\s*in(?:ward|wards)?|"
 r"pull(?:ed|ing)\s*in|"
 r"point(?:ing|ed)?\s*in(?:ward|wards)?|"
 r"gone\s*in|going\s*in"
 r")\b)", 3, "nipple inversion/retraction"),

    # Clear discharge with concerning features (one-sided or spontaneous)
    (rf"(?=.*\b(nipple|breast)\b)(?=.*\b(discharge|leak(?:ing)?|fluid)\b)"
     rf"(?=.*\b(clear|watery)\b)"
     rf"(?=.*\b(one\s*(side|breast|nipple)|single|unilateral|without\s*(touch|press|squeez)|spontaneous|happen(?:s|ing)?\s*without)\b)",
     3, "nipple discharge with concerning features"),

    # Size/shape change — now requires breast context
    (rf"(?=.*\b{BREAST_CTX}\b)(?=.*\bchange(?:s)?\b)(?=.*\b(size|shape)\b)", 3, "change in breast size/shape"),

    # ---- LOW (non-specific) ----
    (rf"(?=.*\b(nipple|breast)\b)(?=.*\b(discharge|leak(?:ing)?|fluid)\b)"
     rf"(?!.*\b(blood|bloody|red)\b)", 2, "non-bloody nipple discharge"),

    # Underarm/chest pain (location-specific low)
    (rf"(?=.*\b(underarm|armpit|axilla|axillary)\b)(?=.*\bpain\b)", 2, "underarm/armpit pain"),

    # Persistent pain — now requires breast context to avoid “persistent headache” etc.
    (rf"(?=.*\b{BREAST_CTX}\b)(?=.*\b(persistent|constant|ongoing|continuous)\b)(?=.*\b(pain|ache|tender(ness)?)\b)",
     2, "persistent breast pain/tenderness"),

    # Very general low-signals (kept, but weight=1)
    (r"\b(unexplained|unintentional)\b.*\b(weight\s*loss|loss\s*of\s*appetite)\b", 1, "unexplained weight loss/appetite"),
    (r"\bfatigue|tired(ness)?\b", 1, "fatigue/tiredness"),
    (r"\btender(ness)?\b", 1, "tenderness"),
    (r"\bache\b", 1, "ache"),
    (r"\bitch(?:y|ing|iness)\b", 1, "itchiness"),
    (rf"(?=.*\b{BREAST_CTX}\b)\bpain\b", 1, "pain"),
]

# Pre-compile the patterns once for speed
PATTERNS: List[Tuple[re.Pattern, int, str]] = [
    (re.compile(p, re.I), w, label) for (p, w, label) in RAW_PATTERNS
]

# ------- HIGH “always-HIGH” overrides -------
OVERRIDES_RAW: List[Tuple[str, str]] = [
    # Bloody nipple discharge – always HIGH
    (rf"(?=.*\b(nipple|breast)\b)(?=.*\b(discharge|leak(?:ing)?|fluid)\b)(?=.*\b(blood|bloody|red)\b)",
     "bloody nipple discharge (override)"),
# NEW nipple inversion → always HIGH
(r"(?=.*\bnipples?\b)"
 r"(?=.*\b(new|recent|recently|sudden(?:ly)?|just|started)\b)"
 r"(?=.*\b("
 r"invert(?:ed|ing)?|"
 r"retract(?:ed|ing)?|"
 r"turn(?:ed|ing)\s*in(?:ward|wards)?|"
 r"pull(?:ed|ing)\s*in|"
 r"point(?:ing|ed)?\s*in(?:ward|wards)?|"
 r"gone\s*in|going\s*in"
 r")\b)",
 "new nipple inversion/retraction (override)"),


    # Inversion + any discharge together → HIGH
    (rf"(?=.*\bnipple\b)(?=.*\b(invert(?:ed|ing)?|pulled\s*in|retract(?:ed|ing)?|turned\s*in(?:ward|wards)?|gone\s*in)\b)"
     rf"(?=.*\b(discharge|leak(?:ing)?|fluid)\b)",
     "nipple discharge + inversion (override)"),

    # Lump/swelling in breast/chest/armpit → HIGH
    (rf"(?=.*\b(lump|swelling)\b)(?=.*\b(breast|chest|armpit|underarm|axilla|axillary)\b)",
     "lump/swelling in breast/chest/armpit (override)"),

    # Skin changes (strong synonyms) → HIGH
    (rf"(?=.*\b(breast|skin|nipple)\b)(?=.*\b(dimpl(?:e|ing)|pucker(?:ing)?|peau\s*d'?orange|orange\s*peel|golf\s*ball|pitted|bumpy|textur(?:e|ed)|dent(?:ed)?)\b)",
     "skin changes (override)"),
]
HIGH_OVERRIDES: List[Tuple[re.Pattern, str]] = [
    (re.compile(p, re.I), label) for (p, label) in OVERRIDES_RAW
]

def rule_based_score(text: str) -> tuple[int, list[str]]:
    """
    Return (score, matches). Score 999 if any HIGH override hits.
    Lightweight negation guards applied for inversion/discharge.
    """
    if not text:
        return 0, []
    t = re.sub(r"\s+", " ", text.strip().lower())

    # 1) HIGH overrides first (don’t apply negation guards here: explicit red-flags)
    for pat, label in HIGH_OVERRIDES:
        if pat.search(t):
            return 999, [label]

    # 2) Otherwise accumulate weights with light negation filters
    score = 0
    matches: list[str] = []
    for pat, w, label in PATTERNS:
        if pat.search(t):
            # basic negation skip for the two ambiguous families
            if "discharge" in label and NEG_NO_DISCHARGE.search(t):
                continue
            if "inversion" in label and NEG_NO_INVERT.search(t):
                continue
            score += w
            matches.append(label)

    # De-dup while preserving order
    matches = list(dict.fromkeys(matches))
    return score, matches

def classify_risk(score: int) -> str:
    """
    Policy:
      >=999 → HIGH (sentinel when any high-override present)
      >=5   → HIGH
      3–4   → MODERATE
      0–2   → LOW
    """
    if score >= 999:
        return "HIGH"
    if score >= 5:
        return "HIGH"
    if score >= 3:
        return "MODERATE"
    return "LOW"
