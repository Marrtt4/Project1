import importlib.util
import queue
import random
import sys
import threading
import time
import traceback
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
GENERATED_DIR = BASE_DIR / "generated"
GENERATED_DIR.mkdir(exist_ok=True)

MODULE_NAME = "amd_text_to_image"
MODULE_PATH = BASE_DIR / "amd_text_to_image.py"

app = Flask(__name__)
state = {
    "model_loaded": False,
    "loading_model": False,
    "model_error": "",
    "generating": False,
    "single_running": False,
    "auto_running": False,
    "last_prompt": "",
    "last_error": "",
    "images_made": 0,
}
state_lock = threading.Lock()
stop_event = threading.Event()
jobs = queue.Queue()  # Flask threads put jobs here; the MAIN thread runs them.
_generator = None

SCENES = [
    "luxury perfume packaging on travertine in soft window light",
    "editorial fashion campaign with sculptural shadows",
    "minimal skincare bottle on brushed stone",
    "premium coffee packaging in a warm modern studio",
    "art book and embossed stationery on linen",
    "high-end candle packaging in burgundy and cream",
]


# ---------------------------------------------------------------------------
# Everything below that touches the GPU runs on the main thread only,
# exactly like the standalone cows script (DirectML hangs when the model is
# created in one thread and used from others).
# ---------------------------------------------------------------------------
def load_model():
    global _generator
    if _generator is not None:
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
        _generator = getattr(module, "generate")
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


def run_one(prompt):
    """Generate one image. Never raises; problems land in state['last_error']."""
    with state_lock:
        state["generating"] = True
        state["last_prompt"] = prompt
        state["last_error"] = ""
    try:
        load_model()
        if _generator is None:
            raise RuntimeError(state["model_error"] or "Model is not loaded.")
        _generator(prompt)
        with state_lock:
            state["images_made"] += 1
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        with state_lock:
            state["last_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        with state_lock:
            state["generating"] = False


def single_job(prompt):
    try:
        run_one(prompt)
    finally:
        with state_lock:
            state["single_running"] = False


def auto_loop(minutes, max_images, base_prompt):
    deadline = time.time() + minutes * 60
    made = 0
    try:
        while time.time() < deadline and not stop_event.is_set():
            if max_images and made >= max_images:
                break
            scene = random.choice(SCENES)
            prompt = f"{base_prompt}, {scene}" if base_prompt else scene
            run_one(prompt)
            made += 1
            if state["last_error"]:
                break
    finally:
        with state_lock:
            state["auto_running"] = False


def worker_loop():
    """Main thread: wait for jobs and run them one at a time."""
    while True:
        try:
            kind, args = jobs.get(timeout=0.5)  # timeout lets Ctrl+C work on Windows
        except queue.Empty:
            continue
        if kind == "load":
            load_model()
        elif kind == "single":
            single_job(*args)
        elif kind == "auto":
            auto_loop(*args)


# ---------------------------------------------------------------------------
# Web side (runs in a background thread, never touches the GPU)
# ---------------------------------------------------------------------------
def latest_images():
    files = sorted(GENERATED_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"name": p.name, "url": f"/generated/{p.name}", "created": int(p.stat().st_mtime)} for p in files[:60]]


def to_int(value, default, low, high):
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        number = default
    return max(low, min(number, high))


def busy():
    return state["generating"] or state["auto_running"] or state["single_running"]


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/generated/<path:name>")
def generated(name):
    return send_from_directory(GENERATED_DIR, name)


@app.get("/api/gallery")
def gallery():
    return jsonify(latest_images())


@app.get("/api/status")
def status():
    with state_lock:
        snapshot = dict(state)
    snapshot["gallery_count"] = len(list(GENERATED_DIR.glob("*.png")))
    return jsonify(snapshot)


@app.post("/api/load")
def load_api():
    if _generator is not None:
        return jsonify({"ok": True, "message": "Model already loaded."})
    if state["loading_model"]:
        return jsonify({"ok": True, "message": "Model is already loading."})
    jobs.put(("load", ()))
    return jsonify({"ok": True, "message": "Model loading started."})


@app.post("/api/generate")
def generate_api():
    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        return jsonify({"error": "Enter a prompt first."}), 400
    with state_lock:
        if busy():
            return jsonify({"error": "The studio is already generating."}), 409
        state["single_running"] = True
    jobs.put(("single", (prompt,)))
    return jsonify({"ok": True, "message": "Generation started."})


@app.post("/api/auto/start")
def auto_start():
    data = request.get_json(silent=True) or {}
    minutes = to_int(data.get("minutes"), 30, 1, 240)
    max_images = to_int(data.get("max_images"), 8, 0, 100)
    base_prompt = str(data.get("base_prompt", "")).strip()
    with state_lock:
        if busy():
            return jsonify({"error": "A generation job is already active."}), 409
        state["auto_running"] = True
        state["last_error"] = ""
    stop_event.clear()
    jobs.put(("auto", (minutes, max_images, base_prompt)))
    return jsonify({"ok": True, "message": "Auto studio started."})


@app.post("/api/auto/stop")
def auto_stop():
    stop_event.set()
    return jsonify({"ok": True, "message": "Stop requested. The current image will finish first."})


def serve():
    app.run(host="127.0.0.1", port=7860, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    # Web server in a background thread, so the page works while the model loads.
    threading.Thread(target=serve, daemon=True).start()
    print("Website: http://127.0.0.1:7860", flush=True)
    # Load the model on the MAIN thread, the same way the cows script does.
    load_model()
    try:
        worker_loop()
    except KeyboardInterrupt:
        print("Stopping...")
