// backend/static/main.js
// Handles the form submit, calls /check, and renders the result.

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("symptom-form");
  const resultDiv = document.getElementById("result");
  const textarea = document.getElementById("symptoms");
  if (!form || !resultDiv) return;

  // Escape HTML so user text can't break the page
  const esc = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (m) => (
      { "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;" }[m]
    ));

  // Display-friendly names for backend method codes
  const niceMethod = (m) => {
    const key = String(m || "").toLowerCase();
    if (key === "rule-based") return "Recognised terms";
    if (key === "bert")       return "Similar to known examples";
    if (key === "llm")        return "General safety advice";
    return "Automated check";
  };

  // Choose colour/style based on risk
  const riskMeta = (r) => {
    const R = String(r || "LOW").toUpperCase();
    return {
      HIGH:   { label: "HIGH",   cls: "is-high"   },
      MEDIUM: { label: "MEDIUM", cls: "is-medium" },
      LOW:    { label: "LOW",    cls: "is-low"    },
    }[R] || { label: R, cls: "is-low" };
  };

  // Turn a rule match (which may be an object) into a short label
  function ruleLabel(x) {
    if (typeof x === "string") return x;
    if (x && typeof x === "object") {
      return x.term || x.keyword || x.pattern || x.rule || x.name || JSON.stringify(x);
    }
    return "";
  }

  // Pick the best evidence field from the response and normalise to text labels
  function toEvidenceLabels(data) {
    let raw = [];
    if (Array.isArray(data.noticed)) raw = data.noticed;                 // optional convenience field
    else if (Array.isArray(data.matched_rules)) raw = data.matched_rules; // rule-based path
    else if (Array.isArray(data.matches)) raw = data.matches;             // alternate name
    let labels = raw.map(ruleLabel).filter(Boolean);

    // If BERT path had no labels, fall back to the matched reference text
    if (!labels.length && String(data.method).toLowerCase() === "bert" && data.matched_reference) {
      labels = [String(data.matched_reference)];
    }
    return labels;
  }

  // Build the HTML for the result panel
  function renderResult(data) {
    const risk    = riskMeta(data.risk);
    const method  = String(data.method || "");
    const advice  = String(data.advice || "");
    const sim     = typeof data.similarity_score === "number" ? data.similarity_score : null;
    const labels  = toEvidenceLabels(data); // ← fixes the “[object Object]” issue

    resultDiv.innerHTML = `
      <article class="result-panel ${risk.cls}" aria-live="polite" aria-atomic="true">
        <div class="rp-head">
          <span class="rp-chip">Risk: ${esc(risk.label)}</span>
          <div class="rp-meta">
            <span>Method: ${esc(niceMethod(method))}</span>
            <span class="rp-method-raw">(${esc(method.toUpperCase())})</span>
          </div>
        </div>

        <div class="rp-body">
          <p class="rp-next">${esc(advice)}</p>

          ${labels.length ? `
            <section class="rp-evidence">
              <h4>What we noticed</h4>
              <div class="pill-row">
                ${labels.map(t => `<span class="pill">${esc(t)}</span>`).join("")}
              </div>
            </section>` : ""}

          ${sim !== null ? `<p class="rp-footnote">Similarity to known examples: ${sim.toFixed(3)}</p>` : ""}
        </div>
      </article>
    `;
  }

  // Submit the form, call /check, and show the result
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const symptoms = (textarea?.value || "").trim();
    resultDiv.innerHTML = "";

    if (!symptoms) {
      resultDiv.innerHTML = `<div class="advice-message">Please describe your symptoms to get advice.</div>`;
      textarea?.focus();
      return;
    }

    const submitBtn = form.querySelector('button[type="submit"]');
    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.dataset._label = submitBtn.textContent || "Check Symptoms";
      submitBtn.textContent = "Checking…";
    }
    resultDiv.innerHTML = `<p aria-live="polite">Checking…</p>`;

    try {
      const res = await fetch("/check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symptoms })
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      renderResult(data);
    } catch (err) {
      console.error(err);
      resultDiv.innerHTML = `
        <div class="advice-message">
          Sorry, we couldn’t connect to the checker service. Please try again.
        </div>`;
    } finally {
      if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.textContent = submitBtn.dataset._label || "Check Symptoms";
      }
    }
  });
});
