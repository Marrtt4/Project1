export function renderStatus(snap) {
  const statusEl = document.getElementById("status-text");
  const errorEl = document.getElementById("error-text");

  if (errorEl) {
    errorEl.textContent = snap.model_error || snap.error || "";
    errorEl.style.display = errorEl.textContent ? "block" : "none";
  }

  if (!statusEl) return;

  if (snap.loading_model) {
    statusEl.textContent = "Loading DirectML model into GPU...";
  } else if (snap.status === "queued") {
    statusEl.textContent = "Queued...";
  } else if (snap.status === "generating") {
    statusEl.textContent = `Generating: ${snap.accepted}/${snap.count} accepted (Attempt ${snap.attempts}/${snap.max_attempts}, avg ${snap.avg_seconds}s)`;
  } else {
    statusEl.textContent = snap.message || (snap.model_loaded ? "Model ready." : "Idle.");
  }
}

export function renderLog(logEntries = []) {
  const logContainer = document.getElementById("attempt-log");
  if (!logContainer) return;

  logContainer.innerHTML = logEntries
    .map((entry) => {
      const badge = entry.ok ? "PASS" : "FAIL";
      const reasons = entry.reasons?.length ? ` — ${entry.reasons.join(", ")}` : "";
      return `<div>[Attempt ${entry.attempt}] <strong>${badge}</strong>${reasons}</div>`;
    })
    .join("");
}

export function renderGallery(images = []) {
  const gallery = document.getElementById("gallery");
  if (!gallery) return;

  gallery.innerHTML = images
    .map((img) => {
      const warning = img.passed
        ? ""
        : `<span class="badge-warn">Below quality bar: ${img.note || ""}</span>`;
      return `
        <div class="image-card">
          <img src="${img.url}" alt="${img.name}" loading="lazy" />
          <div class="image-meta">${img.name} ${warning}</div>
        </div>
      `;
    })
    .join("");
}
