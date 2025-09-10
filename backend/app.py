# backend/app.py
# ------------------------------------------------------------
# Breast symptom checker API
# Flow: Rules  ➜  SBERT (if confident)  ➜  Paraphrase+retry  ➜  LLM fallback
# - Input: expects JSON {"symptoms": "<free text>"} from the UI
# - Output: JSON with risk, advice, method, and evidence fields
# ------------------------------------------------------------

from __future__ import annotations

import os
import re
import logging
from typing import Any, Dict, List, Tuple

from flask import Flask, request, jsonify, render_template, abort
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

# ============================
# Settings & logging
# ============================
# Basic console logging; level can be set with APP_LOG_LEVEL=DEBUG/INFO/...
APP_LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, APP_LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Simple input limits so we only handle a short symptom description
MAX_INPUT_CHARS = 800
# We treat very short inputs as "not enough detail" unless they pass either check:
MIN_WORDS = int(os.getenv("MIN_WORDS", "2"))
MIN_CHARS = int(os.getenv("MIN_CHARS", "6"))

# SBERT decision threshold τ (tau). If similarity ≥ τ and context is present, accept SBERT.
BERT_CONFIDENCE_THRESHOLD = float(os.getenv("SBERT_TAU", "0.75"))

# Words that indicate we are indeed talking about breast context
BREAST_CONTEXT = ("breast", "nipple", "boob", "areola", "underarm", "armpit", "chest")

# ============================
# Small helpers
# ============================
def sanitize_user_text(text: str) -> str:
    """Tidy up user text: strip control chars, collapse spaces, clamp length."""
    if not text:
        return ""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)  # remove control chars
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    return text.strip()[:MAX_INPUT_CHARS]

def _parse_origins(val: str) -> List[str]:
    """Turn a comma-separated env string into a list of allowed origins."""
    return [o.strip() for o in (val or "").split(",") if o.strip()]

def _terms_from_matches(matches) -> str:
    """Make a short, readable list from rule/SBERT matches for explanations."""
    out = []
    for m in matches or []:
        if isinstance(m, dict):
            out.append(str(m.get("term") or m.get("keyword") or m.get("pattern") or m))
        else:
            out.append(str(m))
    return ", ".join(out)

def _safe_clip(text: str, limit: int = 600) -> str:
    """Clip arbitrary text for prompts/logs."""
    text = (text or "").strip()
    return text[:limit]

# ============================
# Flask app & CORS
# ============================
app = Flask(__name__, static_folder="static", template_folder="templates")

# Allow only local dev origins for the /check API
ALLOWED_ORIGINS = _parse_origins(
    os.getenv(
        "ALLOWED_ORIGINS",
        "http://127.0.0.1:5500,http://localhost:5500,"
        "http://127.0.0.1:5173,http://localhost:5173"
    )
)
CORS(
    app,
    resources={
        r"/check": {
            "origins": ALLOWED_ORIGINS,
            "methods": ["POST", "OPTIONS"],
            "allow_headers": ["Content-Type"],
        }
    },
)
# Hard cap: we only accept a few KB per request
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024

# ============================
# Optional .env loader (no-op if missing)
# ============================
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# ============================
# Imports (work both as package or flat files)
# ============================
try:
    from .symptom_rules import rule_based_score, classify_risk  # type: ignore
except Exception:
    from symptom_rules import rule_based_score, classify_risk  # type: ignore

# ============================
# SBERT similarity (optional; we degrade gracefully if unavailable)
# ============================
_bert_available = True
try:
    try:
        from .bert_symptom_checker import bert_symptom_score  # type: ignore
    except Exception:
        from bert_symptom_checker import bert_symptom_score  # type: ignore
except Exception as e:
    _bert_available = False
    logger.warning("SBERT module unavailable: %s", e)

    # If SBERT isn't installed, provide a stub that always returns LOW.
    def bert_symptom_score(text: str) -> Dict[str, Any]:
        return {
            "risk": "LOW",
            "matched_reference": "",
            "similarity_score": 0.0,
            "note": "BERT/SBERT module unavailable (install sentence-transformers/torch).",
        }

# ============================
# Local text2text (T5-small) to paraphrase/explain only
# This does not set risk. It helps normalise wording and produce a short explanation.
# ============================
_local_t5_ok = True
_t5_pipe = None
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline as hf_pipeline  # type: ignore
    _t5_tok = AutoTokenizer.from_pretrained("t5-small")
    _t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    _t5_pipe = hf_pipeline("text2text-generation", model=_t5_model, tokenizer=_t5_tok, device=-1)  # CPU-only
except Exception as e:
    logger.warning("T5-small unavailable (paraphrase/explain will fall back): %s", e)
    _local_t5_ok = False
    _t5_pipe = None

def llm_paraphrase(user_text: str) -> str | None:
    """Rewrite the user's wording more clearly (no new info, no risk)."""
    if not _local_t5_ok or not user_text:
        return None
    prompt = (
        "Rewrite the following text in concise, clear UK English, focusing ONLY on breast symptoms. "
        "Do not add or remove symptoms. Do not assign risk or diagnosis.\n\n"
        f"Text: {_safe_clip(user_text)}"
    )
    try:
        out = _t5_pipe(prompt, max_length=48, num_beams=4, do_sample=False)
        return (out[0]["generated_text"] or "").strip()
    except Exception as e:
        logger.warning("T5 paraphrase failed: %s", e)
        return None

def _basic_explain_text(risk: str, matches=None, matched_reference=None, similarity=None) -> str:
    """Plain fallback explanation using the evidence we have."""
    if matches:
        terms = _terms_from_matches(matches)[:120]
        return f"We recognised: {terms}. Based on these, we showed {risk.title()} advice."
    if matched_reference:
        sim_txt = f" (similarity {similarity:.2f})" if similarity is not None else ""
        return f"Your text was similar to known breast symptom descriptions{sim_txt}, so we showed {risk.title()} advice."
    return f"We could not match clear red-flag terms; we showed {risk.title()} advice for safety."

def llm_explain(risk: str, matches=None, matched_reference=None, similarity=None) -> str | None:
    """Short, readable explanation (1–2 lines). Uses only given evidence."""
    if not _local_t5_ok:
        return None
    bits = []
    if matches:
        bits.append("matched terms: " + _terms_from_matches(matches)[:200])
    if matched_reference:
        if similarity is not None:
            bits.append(f"similar to known description (cosine {similarity:.2f})")
        else:
            bits.append("similar to known description")
    evidence = "; ".join(bits) if bits else "no strong match"
    prompt = (
        "Explain for a layperson in UK English, in 1–2 short sentences, why the advice level was shown. "
        "Use ONLY the evidence provided. Do not add new symptoms or diagnoses. Do not assign risk yourself. "
        f"Advice level: {risk}. Evidence: {evidence}"
    )
    try:
        out = _t5_pipe(prompt, max_length=64, num_beams=4, do_sample=False)
        return (out[0]["generated_text"] or "").strip()
    except Exception as e:
        logger.warning("T5 explain failed: %s", e)
        return None

# ============================
# Local zero-shot classifier (final safety net)
# If used, this produces LOW/MEDIUM/HIGH from a small NLI model.
# It is only reached if Rules and SBERT don't apply.
# ============================
_zsc_available = True
_zsc = None
try:
    from transformers import pipeline as zsc_pipeline  # type: ignore
except Exception:
    _zsc_available = False
    zsc_pipeline = None  # type: ignore

def get_zero_shot():
    """Load a compact NLI model on first use."""
    global _zsc
    if _zsc is None and _zsc_available and zsc_pipeline is not None:
        _zsc = zsc_pipeline(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli",
            device=-1,  # CPU
        )
    return _zsc

# ============================
# Advice text per risk band
# ============================
def make_advice(risk: str) -> str:
    r = (risk or "").upper()
    if r == "HIGH":
        return (
            "Some symptoms you described are considered red-flag signs. "
            "Please contact your GP or call NHS 111 this week."
        )
    elif r in ("MEDIUM", "MODERATE"):
        return (
            "Your symptoms may need a GP check-up. "
            "Please seek medical advice if they persist or change."
        )
    else:
        return "Your symptoms are less likely to be serious, but contact your GP or NHS 111 if you're unsure."

# ============================
# Error handling (catch-all)
# ============================
@app.errorhandler(Exception)
def _any_error(e):
    if isinstance(e, HTTPException):
        return e
    logger.exception("Unhandled error: %s", e)
    return jsonify(error="internal_error"), 500

# ============================
# Simple pages & health checks
# ============================
@app.route("/healthz")
def healthz():
    return jsonify(status="ok"), 200

@app.route("/health")
def health():
    return jsonify(status="ok"), 200

@app.route("/")
def home():
    try:
        return render_template("index.html")
    except Exception:
        return "OK", 200

@app.route("/about")
def about():
    try:
        return render_template("about.html")
    except Exception:
        return "About", 200

@app.route("/disclaimer")
def disclaimer():
    try:
        return render_template("disclaimer.html")
    except Exception:
        return "Disclaimer", 200

@app.route("/selfcheck")
def selfcheck():
    try:
        return render_template("selfcheck.html")
    except Exception:
        return "Self-check", 200

@app.route("/support")
def support():
    try:
        return render_template("support.html")
    except Exception:
        return "Support", 200

@app.route("/symptoms")
def symptoms():
    try:
        return render_template("symptoms.html")
    except Exception:
        return "Symptoms", 200

# Legacy direct-file routes
@app.route("/index.html")
def index_html():
    return home()

@app.route("/<page>.html")
def legacy_html(page):
    allowed = {"index", "about", "symptoms", "selfcheck", "support", "disclaimer"}
    if page in allowed:
        try:
            return render_template(f"{page}.html")
        except Exception:
            return page, 200
    return abort(404)

# ============================
# Final fallback: quick local classifier
# (Used only if Rules and SBERT do not produce a decision)
# ============================
def local_llm_symptom_score(text: str):
    """
    Zero-shot fallback:
      - Predicts HIGH / MEDIUM / LOW from short text.
      - If no breast words are present, we are stricter about raising risk.
    """
    zsc = get_zero_shot()
    if not zsc:
        return {"risk": "LOW", "advice": make_advice("LOW"), "llm_raw": "zero-shot unavailable"}

    labels = ["HIGH", "MEDIUM", "LOW"]
    hyp = "This description indicates a {} level of breast symptom urgency."
    result = zsc(text, candidate_labels=labels, hypothesis_template=hyp, multi_label=False)
    ordered = list(zip(result["labels"], result["scores"]))
    ordered.sort(key=lambda x: x[1], reverse=True)
    top_label, top_score = ordered[0]

    has_context = any(w in text.lower() for w in BREAST_CONTEXT)
    if not has_context and top_score < 0.80:
        top_label = "LOW"

    return {"risk": top_label, "advice": make_advice(top_label), "llm_raw": f"zero-shot {ordered}"}

# ============================
# /check API: main decision flow
# ============================
@app.route("/check", methods=["POST"])
def check_symptoms():
    # Expect JSON {"symptoms": "..."} from the UI
    data = request.get_json(silent=True) or {}
    raw_text = data.get("symptoms") or ""
    text = sanitize_user_text(raw_text)

    # Log what we received (handy during testing)
    logger.info("[DEBUG] raw=%r | sanitized=%r | len=%d", raw_text, text, len(text))
    logger.info("[DEBUG] _bert_available=%s | τ=%.2f", _bert_available, BERT_CONFIDENCE_THRESHOLD)

    # Guard: if too short, ask the user for more words
    if (len(text.split()) < MIN_WORDS) and (len(text) < MIN_CHARS):
        logger.debug("Empty/too short input; returning none path.")
        return jsonify(
            {"risk": "LOW", "advice": "Please describe your symptoms using a few words.", "method": "none"}
        ), 200

    # ---- Stage 1: Rule-based (transparent & fast) ----
    score, matches = rule_based_score(text)
    if score > 0:
        risk = classify_risk(score)
        advice = make_advice(risk)
        explanation = llm_explain(risk, matches=matches) or _basic_explain_text(risk, matches=matches)
        logger.info("Rule-based used. score=%s risk=%s", score, risk)
        return jsonify(
            {
                "risk": risk,
                "advice": advice,
                "method": "rule-based",
                "matched_rules": matches,
                "score": score,
                "explanation": explanation,
            }
        ), 200

    # ---- Stage 2: SBERT (if installed, confident, and breast context present) ----
    similarity = 0.0
    bert_risk = "LOW"
    matched_ref = ""
    has_context = any(w in text.lower() for w in BREAST_CONTEXT)

    if _bert_available:
        try:
            bert_result = bert_symptom_score(text) or {}
            logger.info("[DEBUG] SBERT result: %s", bert_result)
            similarity = float(bert_result.get("similarity_score") or 0.0)
            bert_risk = str(bert_result.get("risk") or "LOW").upper()
            matched_ref = str(bert_result.get("matched_reference") or "")
        except Exception as e:
            logger.exception("SBERT error: %s", e)
            similarity = 0.0
            bert_risk = "LOW"
            matched_ref = ""

    if _bert_available and similarity >= BERT_CONFIDENCE_THRESHOLD and has_context:
        advice = make_advice(bert_risk)
        explanation = (
            llm_explain(bert_risk, matched_reference=matched_ref, similarity=similarity)
            or _basic_explain_text(bert_risk, matched_reference=matched_ref, similarity=similarity)
        )
        logger.info("SBERT used. similarity=%.3f risk=%s ref=%s", similarity, bert_risk, matched_ref)
        return jsonify(
            {
                "risk": bert_risk,
                "advice": advice,
                "method": "bert",
                "matched_reference": matched_ref,
                "similarity_score": similarity,
                "explanation": explanation,
            }
        ), 200
    else:
        logger.info(
            "SBERT skipped/failed. available=%s similarity=%.3f context=%s ref=%s",
            _bert_available,
            similarity,
            has_context,
            matched_ref,
        )

    # ---- Stage 3: Paraphrase then retry Rules/SBERT (normalises wording only) ----
    para = llm_paraphrase(text)
    if para and para.lower() != text.lower():
        # 3a) Rules on paraphrased text
        score2, matches2 = rule_based_score(para)
        if score2 > 0:
            risk2 = classify_risk(score2)
            advice2 = make_advice(risk2)
            explanation2 = llm_explain(risk2, matches=matches2) or _basic_explain_text(risk2, matches=matches2)
            logger.info("Paraphrase→Rule path used. score=%s risk=%s", score2, risk2)
            return jsonify(
                {
                    "risk": risk2,
                    "advice": advice2,
                    "method": "rule-based",
                    "matched_rules": matches2,
                    "score": score2,
                    "paraphrased": True,
                    "paraphrase": para,
                    "explanation": explanation2,
                }
            ), 200

        # 3b) SBERT on paraphrased text
        similarity2 = 0.0
        bert_risk2 = "LOW"
        matched_ref2 = ""
        has_context2 = any(w in para.lower() for w in BREAST_CONTEXT)

        if _bert_available:
            try:
                bert_result2 = bert_symptom_score(para) or {}
                logger.info("[DEBUG] SBERT result (para): %s", bert_result2)
                similarity2 = float(bert_result2.get("similarity_score") or 0.0)
                bert_risk2 = str(bert_result2.get("risk") or "LOW").upper()
                matched_ref2 = str(bert_result2.get("matched_reference") or "")
            except Exception as e:
                logger.exception("SBERT error (paraphrase path): %s", e)
                similarity2 = 0.0
                bert_risk2 = "LOW"
                matched_ref2 = ""

        if _bert_available and similarity2 >= BERT_CONFIDENCE_THRESHOLD and has_context2:
            advice2 = make_advice(bert_risk2)
            explanation2 = (
                llm_explain(bert_risk2, matched_reference=matched_ref2, similarity=similarity2)
                or _basic_explain_text(bert_risk2, matched_reference=matched_ref2, similarity=similarity2)
            )
            logger.info("Paraphrase→SBERT path used. similarity=%.3f risk=%s", similarity2, bert_risk2)
            return jsonify(
                {
                    "risk": bert_risk2,
                    "advice": advice2,
                    "method": "bert",
                    "matched_reference": matched_ref2,
                    "similarity_score": similarity2,
                    "paraphrased": True,
                    "paraphrase": para,
                    "explanation": explanation2,
                }
            ), 200

    # ---- Stage 4: Final fallback (local zero-shot classifier) ----
    llm_result = local_llm_symptom_score(text)
    logger.info("Local zero-shot fallback used. risk=%s", llm_result.get("risk"))
    return jsonify(
        {
            "risk": llm_result.get("risk", "LOW"),
            "advice": llm_result.get("advice", ""),
            "method": "llm",  # keep 'llm' so your UI stays consistent
            "llm_raw": llm_result.get("llm_raw", ""),
            "explanation": _basic_explain_text(llm_result.get("risk", "LOW")),
        }
    ), 200

# ============================
# Entrypoint for local dev
# ============================
if __name__ == "__main__":
    logger.info("Allowed CORS origins: %s", ALLOWED_ORIGINS)
    logger.info(
        "SBERT available=%s | τ=%.2f | MIN_WORDS=%d | MIN_CHARS=%d",
        _bert_available, BERT_CONFIDENCE_THRESHOLD, MIN_WORDS, MIN_CHARS
    )
    app.run(host="127.0.0.1", port=8000, debug=False, use_reloader=False)
