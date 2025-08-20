
# backend/app.py

import os
import logging
from flask import Flask, request, jsonify, render_template, abort
from flask_cors import CORS

# ----------------------------
# Logging early (used by helpers)
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------
# Optional .env support
# ----------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
# Optional kill switch so you can force-disable LLMs without touching keys
if os.getenv("DISABLE_LLM", "0") not in ("0", "", None):
    _OPENAI_OK = False

# ----------------------------
# Create app
# ----------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# ----------------------------
# Import local modules (both module and script modes)
# ----------------------------
try:
    from .symptom_rules import rule_based_score, classify_risk  # type: ignore
except Exception:
    from symptom_rules import rule_based_score, classify_risk

# SBERT similarity module
_bert_available = True
try:
    try:
        from .bert_symptom_checker import bert_symptom_score  # type: ignore
    except Exception:
        from bert_symptom_checker import bert_symptom_score
except Exception:
    _bert_available = False

    def bert_symptom_score(text: str):
        # graceful stub if module not available
        return {
            "risk": "LOW",
            "matched_reference": "",
            "similarity_score": 0.0,
            "note": "BERT/SBERT module unavailable (install sentence-transformers/torch).",
        }

# ----------------------------
# Local FREE text2text (T5-small) for paraphrase & explain — no risk assignment
# ----------------------------
_local_t5_ok = True
_t5_pipe = None
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline as hf_pipeline
    _t5_tok = AutoTokenizer.from_pretrained("t5-small")
    _t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    _t5_pipe = hf_pipeline("text2text-generation", model=_t5_model, tokenizer=_t5_tok, device=-1)  # CPU
except Exception as e:
    logger.warning("T5-small unavailable (paraphrase/explain will fall back): %s", e)
    _local_t5_ok = False
    _t5_pipe = None

def _safe_clip(text: str, limit: int = 600) -> str:
    text = (text or "").strip()
    return text[:limit]

def llm_paraphrase(user_text: str) -> str | None:
    """
    Free, local paraphrase using T5-small. Deterministic (no sampling).
    Guardrails: rewrite only; do NOT add/remove symptoms; do NOT assign risk.
    """
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

def _terms_from_matches(matches) -> str:
    """
    Robustly extract human-readable terms from matches which may be a list of
    strings OR dicts ({'term': '...'}). Prevents AttributeError: 'str' has no 'get'.
    """
    out = []
    for m in matches or []:
        if isinstance(m, dict):
            out.append(str(m.get("term") or m.get("keyword") or m.get("pattern") or m))
        else:
            out.append(str(m))
    return ", ".join(out)

def _basic_explain_text(risk: str, matches=None, matched_reference=None, similarity=None) -> str:
    """Non-LLM fallback explanation in 1–2 plain sentences."""
    if matches:
        terms = _terms_from_matches(matches)[:120]
        return f"We recognised: {terms}. Based on these, we showed {risk.title()} advice."
    if matched_reference:
        sim_txt = f" (similarity {similarity:.2f})" if similarity is not None else ""
        return f"Your text was similar to known breast symptom descriptions{sim_txt}, so we showed {risk.title()} advice."
    return f"We could not match clear red-flag terms; we showed {risk.title()} advice for safety."

def llm_explain(risk: str, matches=None, matched_reference=None, similarity=None) -> str | None:
    """
    Free, local explanation via T5-small. Uses ONLY evidence we provide; no new facts.
    """
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

# ----------------------------
# FREE local zero-shot fallback (DistilBERT-MNLI)
# ----------------------------
_zsc_available = True
_zsc = None  # lazy-loaded

try:
    from transformers import pipeline as zsc_pipeline
except Exception:
    _zsc_available = False
    zsc_pipeline = None  # type: ignore

def get_zero_shot():
    """Lazy-load a small, fast NLI model for zero-shot classification."""
    global _zsc
    if _zsc is None and _zsc_available and zsc_pipeline is not None:
        _zsc = zsc_pipeline(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli",
            device=-1,  # CPU
        )
    return _zsc

# ----------------------------
# Constants
# ----------------------------
BERT_CONFIDENCE_THRESHOLD = 0.55  # conservative to avoid spurious matches
BREAST_CONTEXT = ("breast", "nipple", "boob", "areola", "underarm", "armpit", "chest")

# ----------------------------
# Frontend pages
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/disclaimer")
def disclaimer():
    return render_template("disclaimer.html")

@app.route("/selfcheck")
def selfcheck():
    return render_template("selfcheck.html")

@app.route("/support")
def support():
    return render_template("support.html")

@app.route("/symptoms")
def symptoms():
    return render_template("symptoms.html")

# Legacy direct-file routes (so /about.html etc. still work)
@app.route("/index.html")
def index_html():
    return render_template("index.html")

@app.route("/<page>.html")
def legacy_html(page):
    allowed = {"index", "about", "symptoms", "selfcheck", "support", "disclaimer"}
    if page in allowed:
        return render_template(f"{page}.html")
    return abort(404)

# ----------------------------
# Advice text
# ----------------------------
def make_advice(risk: str) -> str:
    if risk == "HIGH":
        return (
            "Some symptoms you described are considered red-flag signs. "
            "Please contact your GP or call NHS 111 this week."
        )
    elif risk == "MEDIUM":
        return (
            "Your symptoms may need a GP check-up. "
            "Please seek medical advice if they persist or change."
        )
    else:
        return (
            "Your symptoms are less likely to be serious, but contact your GP or NHS 111 if you're unsure."
        )

# ----------------------------
# Local zero-shot risk (final fallback)
# ----------------------------
def local_llm_symptom_score(text: str):
    """
    Free, local fallback:
      - Zero-shot classifier chooses: HIGH / MEDIUM / LOW
      - Guardrail: if no breast-context word, require higher confidence for HIGH/MEDIUM.
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

# ----------------------------
# API
# ----------------------------
@app.route("/check", methods=["POST"])
def check_symptoms():
    data = request.json or {}
    text = (data.get("symptoms") or "").strip()

    if not text or len(text) < 3:
        logger.debug("Empty/too short input; returning none path.")
        return jsonify(
            {"risk": "LOW", "advice": "Please describe your symptoms using a few words.", "method": "none"}
        ), 200

    # 1) Rule-based first (transparent & fast)
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

    # 2) SBERT fallback (only if available, confident, and breast context present)
    bert_result = bert_symptom_score(text)
    similarity = float(bert_result.get("similarity_score", 0.0) or 0.0)
    bert_risk = (bert_result.get("risk") or "LOW").upper()
    has_context = any(w in text.lower() for w in BREAST_CONTEXT)

    if _bert_available and similarity >= BERT_CONFIDENCE_THRESHOLD and has_context:
        advice = make_advice(bert_risk)
        explanation = llm_explain(
            bert_risk, matched_reference=bert_result.get("matched_reference", ""), similarity=similarity
        ) or _basic_explain_text(
            bert_risk, matched_reference=bert_result.get("matched_reference", ""), similarity=similarity
        )
        logger.info("SBERT used. similarity=%.3f risk=%s", similarity, bert_risk)
        return jsonify(
            {
                "risk": bert_risk,
                "advice": advice,
                "method": "bert",
                "matched_reference": bert_result.get("matched_reference", ""),
                "similarity_score": similarity,
                "explanation": explanation,
            }
        ), 200

    # 3) LLM paraphrase (only to normalise text), then re-run Rules + SBERT
    para = llm_paraphrase(text)
    if para and para.lower() != text.lower():
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

        bert_result2 = bert_symptom_score(para)
        similarity2 = float(bert_result2.get("similarity_score", 0.0) or 0.0)
        bert_risk2 = (bert_result2.get("risk") or "LOW").upper()
        has_context2 = any(w in para.lower() for w in BREAST_CONTEXT)

        if _bert_available and similarity2 >= BERT_CONFIDENCE_THRESHOLD and has_context2:
            advice2 = make_advice(bert_risk2)
            explanation2 = llm_explain(
                bert_risk2, matched_reference=bert_result2.get("matched_reference", ""), similarity=similarity2
            ) or _basic_explain_text(
                bert_risk2, matched_reference=bert_result2.get("matched_reference", ""), similarity=similarity2
            )
            logger.info("Paraphrase→SBERT path used. similarity=%.3f risk=%s", similarity2, bert_risk2)
            return jsonify(
                {
                    "risk": bert_risk2,
                    "advice": advice2,
                    "method": "bert",
                    "matched_reference": bert_result2.get("matched_reference", ""),
                    "similarity_score": similarity2,
                    "paraphrased": True,
                    "paraphrase": para,
                    "explanation": explanation2,
                }
            ), 200

    # 4) Final fallback — local zero-shot classification
    llm_result = local_llm_symptom_score(text)
    logger.info("Local zero-shot fallback used. risk=%s", llm_result.get("risk"))
    return jsonify(
        {
            "risk": llm_result.get("risk", "LOW"),
            "advice": llm_result.get("advice", ""),
            "method": "llm",  # keep 'llm' so your UI doesn't change
            "llm_raw": llm_result.get("llm_raw", ""),
            "explanation": _basic_explain_text(llm_result.get("risk", "LOW")),
        }
    ), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False, use_reloader=False)
