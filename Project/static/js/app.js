import { getFormPayload, setFormDisabled } from "./form.js";
import { triggerModelLoad, startBrandJob, cancelBrandJob } from "./brand-generator.js";
import { renderStatus, renderLog, renderGallery } from "./results.js";

const ACTIVE_STATUSES = new Set(["queued", "loading_model", "generating"]);
let pollTimer = null;

async function pollStatus() {
  try {
    const res = await fetch("/api/brand/status");
    const snap = await res.json();

    renderStatus(snap);
    renderLog(snap.log);
    renderGallery(snap.images);

    const isBusy = ACTIVE_STATUSES.has(snap.status) || snap.loading_model;
    setFormDisabled(isBusy);

    if (isBusy && !pollTimer) {
      pollTimer = setInterval(pollStatus, 1500);
    } else if (!isBusy && pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  } catch (err) {
    console.error("Status poll failed:", err);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("load-model-btn")?.addEventListener("click", async () => {
    await triggerModelLoad();
    pollStatus();
  });

  document.getElementById("generate-btn")?.addEventListener("click", async (e) => {
    e.preventDefault();
    try {
      const payload = getFormPayload();
      await startBrandJob(payload);
      pollStatus();
    } catch (err) {
      renderStatus({ error: err.message });
    }
  });

  document.getElementById("cancel-btn")?.addEventListener("click", async () => {
    await cancelBrandJob();
  });

  // Initial sync on page load
  pollStatus();
});
