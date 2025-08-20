import os
import openai

# Load API key from environment; assumes you've done: 
# in PowerShell for session:  $env:OPENAI_API_KEY="your_key_here"
# or permanently (then reopen shell): setx OPENAI_API_KEY "your_key_here"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

def call_llm_fallback(symptom_text: str) -> dict:
    """
    Ask an LLM to classify risk (HIGH/MEDIUM/LOW) and give advice based on free-form symptom text.
    Returns a dict with keys: risk, advice, raw (model output for inspection).
    """
    if not OPENAI_API_KEY:
        return {
            "risk": "LOW",
            "advice": "Could not reach LLM fallback (API key missing). Please try again later.",
            "raw": "OPENAI_API_KEY not set"
        }

    prompt = (
        "You are a clinical support assistant. Given the patient's symptom description below, "
        "assign one of: HIGH, MEDIUM, or LOW risk for possible serious breast pathology. "
        "Then give a concise actionable advice sentence. "
        "Respond in JSON with fields: risk, advice, reasoning.\n\n"
        f"Symptom description: \"{symptom_text}\"\n"
    )

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=250
        )
        content = resp["choices"][0]["message"]["content"].strip()

        # Naive parsing: try to extract risk and advice from JSON-like or free text.
        # Expecting model to output something like:
        # {"risk": "MEDIUM", "advice": "...", "reasoning": "..."}
        import json
        try:
            parsed = json.loads(content)
            risk = parsed.get("risk", "LOW").upper()
            advice = parsed.get("advice", "")
            raw = content
        except json.JSONDecodeError:
            # Fallback: heuristic extraction
            risk = "LOW"
            if "HIGH" in content.upper():
                risk = "HIGH"
            elif "MEDIUM" in content.upper():
                risk = "MEDIUM"
            advice = content
            raw = content

        return {
            "risk": risk if risk in ("HIGH", "MEDIUM", "LOW") else "LOW",
            "advice": advice,
            "raw": raw
        }
    except Exception as e:
        return {
            "risk": "LOW",
            "advice": "LLM fallback failed; using safe low-risk advice. Please try again later.",
            "raw": f"LLM error: {str(e)}"
        }
