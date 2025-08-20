import re
from typing import List, Tuple

# ------- NHS-aligned patterns (regex, weight, label) -------
PATTERNS: List[Tuple[str, int, str]] = [
    # ---- HIGH red flags (also enforced via overrides below) ----
    # Skin changes consistent with peau d’orange / dimpling / marked redness
    (r"(?=.*\b(breast|skin|nipple)\b)(?=.*\b(dimpl(?:e|ing)|pucker(?:ing)?|peau\s*d'?orange|orange\s*peel|golf\s*ball|pitted|bumpy|textur(?:e|ed)|dent(?:ed)?|red(?:ness)?)\b)",
     4, "skin changes (dimpling/orange-peel/redness)"),

    # ---- MEDIUM (visible/structural changes without “new”) ----
    # Inversion without “new/recent”
    (r"(?=.*\bnipple\b)(?=.*\b(invert(?:ed|ing)?|pulled\s*in|retract(?:ed|ing)?|turned\s*in(?:ward|wards)?|pointing\s*in|gone\s*in)\b)",
     3, "nipple inversion/retraction"),

    # Clear discharge with concerning features (one-sided or spontaneous)
    (r"(?=.*\bnipple\b|\bbreast\b)(?=.*\b(discharge|leak(?:ing)?|fluid)\b)"
     r"(?=.*\b(clear|watery)\b)"
     r"(?=.*\b(one\s*(side|breast|nipple)|single|unilateral|without\s*(touch|press|squeez)|spontaneous|happen(?:s|ing)?\s*without)\b)",
     3, "nipple discharge with concerning features"),

    # Size/shape change (non-specific)
    (r"(?=.*\bchange(?:s)?\b)(?=.*\b(size|shape)\b)", 3, "change in size/shape"),

    # ---- LOW (non-specific) ----
    (r"(?=.*\bnipple\b|\bbreast\b)(?=.*\b(discharge|leak(?:ing)?|fluid)\b)"
     r"(?!.*\b(blood|bloody|red)\b)", 2, "non-bloody nipple discharge"),

    (r"(?=.*\b(underarm|armpit)\b)(?=.*\bpain\b)", 2, "underarm/armpit pain"),
    (r"(?=.*\b(persistent|constant|ongoing|continuous)\b)(?=.*\b(pain|ache|tender(ness)?)\b)",
     2, "persistent pain/tenderness"),

    (r"\b(unexplained|unintentional)\b.*\b(weight\s*loss|loss\s*of\s*appetite)\b", 1, "unexplained weight loss/appetite"),
    (r"\bfatigue|tired(ness)?\b", 1, "fatigue/tiredness"),
    (r"\btender(ness)?\b", 1, "tenderness"),
    (r"\bache\b", 1, "ache"),
    (r"\bitch(?:y|ing|iness)\b", 1, "itchiness"),
    (r"\bpain\b", 1, "pain"),
]

# ------- HIGH “always-high” overrides -------
HIGH_OVERRIDES: List[Tuple[str, str]] = [
    # Bloody nipple discharge – always HIGH
    (r"(?=.*\bnipple\b|\bbreast\b)(?=.*\b(discharge|leak(?:ing)?|fluid)\b)(?=.*\b(blood|bloody|red)\b)",
     "bloody nipple discharge (override)"),

    # NEW inversion → HIGH
    (r"(?=.*\bnipple\b)"
     r"(?=.*\b(new|recent|recently|sudden(?:ly)?|just|turned\b)\b)"
     r"(?=.*\b(invert(?:ed|ing)?|pulled\s*in|retract(?:ed|ing)?|turned\s*in(?:ward|wards)?|pointing\s*in|gone\s*in)\b)",
     "new nipple inversion/retraction (override)"),

    # Inversion + any discharge together → HIGH
    (r"(?=.*\bnipple\b)(?=.*\b(invert(?:ed|ing)?|pulled\s*in|retract(?:ed|ing)?|turned\s*in(?:ward|wards)?|gone\s*in)\b)"
     r"(?=.*\b(discharge|leak(?:ing)?|fluid)\b)",
     "nipple discharge + inversion (override)"),

    # Lump/swelling in breast/chest/armpit → HIGH
    (r"(?=.*\b(lump|swelling)\b)(?=.*\b(breast|chest|armpit|underarm)\b)",
     "lump/swelling in breast/chest/armpit (override)"),

    # Skin changes (strong synonyms) → HIGH
    (r"(?=.*\b(breast|skin|nipple)\b)(?=.*\b(dimpl(?:e|ing)|pucker(?:ing)?|peau\s*d'?orange|orange\s*peel|golf\s*ball|pitted|bumpy|textur(?:e|ed)|dent(?:ed)?)\b)",
     "skin changes (override)"),
]


def rule_based_score(text: str) -> tuple[int, list[str]]:
    """
    Return (score, matches). Score 999 if any HIGH override hits.
    """
    if not text:
        return 0, []
    t = re.sub(r"\s+", " ", text.strip().lower())

    # 1) HIGH overrides first
    for pat, label in HIGH_OVERRIDES:
        if re.search(pat, t, flags=re.IGNORECASE):
            return 999, [label]

    # 2) Otherwise accumulate weights
    score = 0
    matches: list[str] = []
    for pat, w, label in PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            score += w
            matches.append(label)

    # De-dup while preserving order
    matches = list(dict.fromkeys(matches))
    return score, matches


def classify_risk(score: int) -> str:
    """
    Policy:
      >=999 → HIGH
      >=4   → HIGH
       =3   → MEDIUM
      1–2   → LOW
       0    → LOW
    """
    if score >= 999:
        return "HIGH"
    if score >= 4:
        return "HIGH"
    if score == 3:
        return "MEDIUM"
    if score >= 1:
        return "LOW"
    return "LOW"
