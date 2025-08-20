// backend/static/main.js
document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("symptom-form");
  const resultDiv = document.getElementById("result");
  const textarea = document.getElementById("symptoms");
  if (!form || !resultDiv) return;

  // ---- helpers ----
  const esc = (s) =>
    String(s ?? "").replace(/[&<>"']/g, m => (
      {"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[m]
    ));

  const niceMethod = (m) => {
    const key = String(m || "").toLowerCase();
    if (key === "rule-based") return "Recognised terms";
    if (key === "bert")       return "Similar to known examples";
    if (key === "llm")        return "General safety advice";
    return "Automated check";
  };

  const riskMeta = (r) => {
    const R = String(r || "LOW").toUpperCase();
    return {
      HIGH:   { label: "HIGH",   cls: "is-high"   },
      MEDIUM: { label: "MEDIUM", cls: "is-medium" },
      LOW:    { label: "LOW",    cls: "is-low"    },
    }[R] || { label: R, cls: "is-low" };
  };

  // ---- renderer ----
  function renderResult(data) {
    const risk    = riskMeta(data.risk);
    const method  = String(data.method || "");
    const advice  = String(data.advice || "");
    const matched = Array.isArray(data.matched_rules) ? data.matched_rules : [];
    const sim     = typeof data.similarity_score === "number" ? data.similarity_score : null;

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

          ${matched.length ? `
            <section class="rp-evidence">
              <h4>What we noticed</h4>
              <div class="pill-row">
                ${matched.map(m => `<span class="pill">${esc(m)}</span>`).join("")}
              </div>
            </section>` : ""}

          ${sim !== null ? `<p class="rp-footnote">Similarity to known examples: ${sim.toFixed(3)}</p>` : ""}
        </div>
      </article>
    `;
  }

  // ---- submit handler ----
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
