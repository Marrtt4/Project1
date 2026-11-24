import importlib.util
import queue
import re
import sys
import threading
import time
import traceback
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

import quality_check

BASE_DIR = Path(__file__).resolve().parent
GENERATED_DIR = BASE_DIR / "generated"
GENERATED_DIR.mkdir(exist_ok=True)
REJECT_DIR = GENERATED_DIR / "rejected"   # rejected renders are kept here so you can look at them
REJECT_DIR.mkdir(exist_ok=True)
HARD = ("blank", "too dark", "mostly pure", "washed", "noisy")  # never used as a fallback

MODULE_NAME = "amd_text_to_image"
MODULE_PATH = BASE_DIR / "amd_text_to_image.py"

ASPECTS = {"landscape": (768, 432), "square": (512, 512), "portrait": (432, 768)}
BLOCKED = re.compile(r"\b(nsfw|nude|naked|porn\w*|erotic\w*|gore|sexual\w*|explicit)\b", re.I)

app = Flask(__name__)
state = {"model_loaded": False, "loading_model": False, "model_error": ""}
state_lock = threading.Lock()
cancel_event = threading.Event()
jobs = queue.Queue()  # Flask threads put jobs here; the MAIN thread runs them.
_render = None

ACTIVE = ("queued", "loading_model", "generating")
brand_job = {
    "id": None, "status": "idle", "prompt": "", "aspect": "landscape", "strictness": "normal",
    "count": 0, "accepted": 0, "attempts": 0, "max_attempts": 0, "avg_seconds": 0.0,
    "log": [], "images": [], "message": "", "error": "",
}


# ---------------------------------------------------------------------------
# GPU side: everything here runs on the MAIN thread only (DirectML hangs if the
# model is created in one thread and used from another).
# ---------------------------------------------------------------------------
def load_model():
    global _render
    if _render is not None:
        return
    with state_lock:
        state["loading_model"] = True
        state["model_error"] = ""
    try:
        if not MODULE_PATH.exists():
            raise FileNotFoundError(f"Cannot find {MODULE_PATH}. Keep it next to app.py.")
        sys.modules.pop(MODULE_NAME, None)
        spec = importlib.util.spec_from_file_location(MODULE_NAME, str(MODULE_PATH))
        module = importlib.util.module_from_spec(spec)
        sys.modules[MODULE_NAME] = module
        spec.loader.exec_module(module)
        _render = getattr(module, "render")
        with state_lock:
            state["model_loaded"] = True
        print("Model ready.", flush=True)
    except BaseException as exc:  # noqa: BLE001
        sys.modules.pop(MODULE_NAME, None)
        traceback.print_exc()
        with state_lock:
            state["model_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        with state_lock:
            state["loading_model"] = False


def set_job(**fields):
    with state_lock:
        brand_job.update(fields)


def run_brand(job_id, prompt, count, aspect, strictness, palette, max_attempts):
    """Render -> CPU check -> keep or redo, until `count` good images exist."""
    width, height = ASPECTS[aspect]
    accepted, sigs, durations, rejects = [], [], [], []
    attempts = 0
    noisy_streak = 0
    try:
        if cancel_event.is_set():
            set_job(status="cancelled", message="Cancelled before it started.")
            return
        if _render is None:
            set_job(status="loading_model")
            load_model()
        if _render is None:
            raise RuntimeError(state["model_error"] or "Model is not loaded.")

        set_job(status="generating")
        while len(accepted) < count and attempts < max_attempts:
            if cancel_event.is_set():
                set_job(status="cancelled", message="Cancelled.")
                return
            attempts += 1
            started = time.time()
            image, flagged = _render(prompt, width, height, raw=True)

            if flagged:
                ok, reasons, metrics, sig = False, ["safety filter flagged the image"], {}, None
            else:
                ok, reasons, metrics, sig = quality_check.check(image, strictness, palette, sigs)

            print(f"[attempt {attempts}] {'PASS' if ok else 'FAIL'} {reasons} {metrics}", flush=True)

            noisy_streak = noisy_streak + 1 if any(r.startswith("noisy") for r in reasons) else 0
            if noisy_streak >= 3 and not accepted:
                image.save(REJECT_DIR / f"{job_id}_attempt{attempts}.png")
                set_job(status="error", attempts=attempts, error=(
                    "The image generator is outputting static/garbled images (3 in a row), so this is "
                    "a generator problem, not the quality check. Open generated/rejected to see them, "
                    "then run: python amd_text_to_image.py \"a red apple on a wooden table\""))
                return

            if ok:
                name = f"brand_{job_id}_{len(accepted) + 1}.png"
                image.save(GENERATED_DIR / name)
                accepted.append({"name": name, "url": f"/generated/{name}",
                                 "metrics": metrics, "passed": True})
                sigs.append(sig)
            elif not flagged:
                image.save(REJECT_DIR / f"{job_id}_attempt{attempts}.png")
                rejects.append((len(reasons), -metrics.get("contrast", 0), image, reasons, metrics))

            durations.append(time.time() - started)
            with state_lock:
                brand_job["log"].append({
                    "attempt": attempts, "ok": ok, "reasons": reasons, "metrics": metrics,
                    "slot": len(accepted) if ok else len(accepted) + 1,
                })
                brand_job["log"] = brand_job["log"][-200:]
                brand_job["accepted"] = len(accepted)
                brand_job["attempts"] = attempts
                brand_job["avg_seconds"] = round(sum(durations) / len(durations), 1)

        passed = len(accepted)
        filled = 0
        if passed < count:
            # Don't hand back an empty gallery: top up with the best near-misses
            # (never blank/black/washed-out or safety-flagged ones), clearly labelled.
            usable = sorted((r for r in rejects if not any(x.startswith(HARD) for x in r[3])),
                            key=lambda r: (r[0], r[1]))
            for _, _, image, reasons, metrics in usable[: count - passed]:
                name = f"brand_{job_id}_{len(accepted) + 1}.png"
                image.save(GENERATED_DIR / name)
                accepted.append({"name": name, "url": f"/generated/{name}", "metrics": metrics,
                                 "passed": False, "note": "; ".join(reasons)})
                filled += 1

        if passed == count:
            msg = f"All {count} image(s) passed the CPU quality check ({attempts} render(s))."
        elif filled:
            msg = (f"{passed} of {count} passed the quality check after {attempts} attempts. "
                   f"{filled} near-miss image(s) were added and are marked 'Below quality bar'. "
                   f"Rejected renders are in generated/rejected.")
        else:
            msg = (f"0 usable images after {attempts} attempts. See the log for the reasons "
                   f"and generated/rejected for the discarded renders.")
        set_job(status="done", images=accepted, message=msg)
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        set_job(status="error", error=f"{type(exc).__name__}: {exc}")


def worker_loop():
    """Main thread: wait for jobs and run them one at a time."""
    while True:
        try:
            kind, args = jobs.get(timeout=0.5)  # timeout lets Ctrl+C work on Windows
        except queue.Empty:
            continue
        if kind == "load":
            load_model()
        elif kind == "brand":
            run_brand(*args)


# ---------------------------------------------------------------------------
# Web side (background thread, never touches the GPU)
# ---------------------------------------------------------------------------
def clean_prompt(text):
    text = re.sub(r"[\x00-\x1f]+", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()[:500]


def to_int(value, default, low, high):
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        number = default
    return max(low, min(number, high))


@app.get("/")
def home():
    name = "ckm_brand_atelier.html"
    if (BASE_DIR / "templates" / name).exists():
        return render_template(name)
    if (BASE_DIR / name).exists():  # file was left next to app.py
        return send_from_directory(BASE_DIR, name)
    return f"Cannot find {name}. Put it in a 'templates' folder next to app.py.", 500


@app.get("/generated/<path:name>")
def generated(name):
    return send_from_directory(GENERATED_DIR, name)


@app.get("/api/brand/status")
def brand_status():
    with state_lock:
        snap = dict(brand_job)
        snap["log"] = list(brand_job["log"])
        snap.update(state)
    return jsonify(snap)


@app.post("/api/load")
def load_api():
    if _render is not None:
        return jsonify({"ok": True, "message": "Model already loaded."})
    if state["loading_model"]:
        return jsonify({"ok": True, "message": "Model is already loading."})
    jobs.put(("load", ()))
    return jsonify({"ok": True, "message": "Model loading started."})


@app.post("/api/brand/start")
def brand_start():
    data = request.get_json(silent=True) or {}
    prompt = clean_prompt(data.get("prompt", ""))
    if not prompt:
        return jsonify({"error": "The prompt is empty. Build a prompt first."}), 400
    if BLOCKED.search(prompt):
        return jsonify({"error": "The prompt contains terms the studio does not allow."}), 400

    count = to_int(data.get("count"), 4, 1, 8)
    aspect = data.get("aspect") if data.get("aspect") in ASPECTS else "landscape"
    strictness = data.get("strictness") if data.get("strictness") in quality_check.LEVELS else "normal"
    palette = [c for c in (data.get("palette") or [])
               if isinstance(c, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", c)][:6]
    max_attempts = min(count * 4, 32)

    with state_lock:
        if brand_job["status"] in ACTIVE:
            return jsonify({"error": "The studio is already generating a batch."}), 409
        job_id = str(int(time.time() * 1000))
        brand_job.update({
            "id": job_id, "status": "queued", "prompt": prompt, "aspect": aspect,
            "strictness": strictness, "count": count, "accepted": 0, "attempts": 0,
            "max_attempts": max_attempts, "avg_seconds": 0.0, "log": [], "images": [],
            "message": "", "error": "",
        })
    cancel_event.clear()
    jobs.put(("brand", (job_id, prompt, count, aspect, strictness, palette, max_attempts)))
    return jsonify({"ok": True, "id": job_id})


@app.post("/api/brand/cancel")
def brand_cancel():
    cancel_event.set()
    return jsonify({"ok": True, "message": "Stop requested. The current image will finish first."})


@app.errorhandler(Exception)
def on_error(exc):
    from werkzeug.exceptions import HTTPException
    traceback.print_exc()
    code = exc.code if isinstance(exc, HTTPException) else 500
    return jsonify({"error": f"{type(exc).__name__}: {exc}"}), code


def serve():
    app.run(host="127.0.0.1", port=7860, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    # Web server in a background thread, so the page works while the model loads.
    threading.Thread(target=serve, daemon=True).start()
    print("Website: http://127.0.0.1:7860", flush=True)
    # Load the model on the MAIN thread, the same way the standalone script does.
    load_model()
    try:
        worker_loop()
    except KeyboardInterrupt:
        print("Stopping...")
