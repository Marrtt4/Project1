export function getFormPayload() {
  const prompt = document.getElementById("prompt-input")?.value.trim() || "";
  const count = parseInt(document.getElementById("count-input")?.value || "4", 10);
  const aspect = document.getElementById("aspect-select")?.value || "landscape";
  const strictness = document.getElementById("strictness-select")?.value || "normal";

  // Collects any <input type="color" class="palette-color"> values
  const palette = Array.from(document.querySelectorAll(".palette-color"))
    .map((el) => el.value.trim())
    .filter((hex) => /^#[0-9a-fA-F]{6}$/.test(hex));

  return { prompt, count, aspect, strictness, palette };
}

export function setFormDisabled(disabled) {
  const startBtn = document.getElementById("generate-btn");
  const cancelBtn = document.getElementById("cancel-btn");

  if (startBtn) startBtn.disabled = disabled;
  if (cancelBtn) cancelBtn.disabled = !disabled;
}
