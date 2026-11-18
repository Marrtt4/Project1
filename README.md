# Aurelia AMD Studio

A luxury brand-design website integrated with your local AMD DirectML image generator.

## Windows setup

1. Put this project in a normal local folder.
2. Open PowerShell in the folder.
3. Create/activate the same Python environment that already runs `amd_text_to_image.py`.
4. Install Flask: `pip install flask`
5. Ensure the existing dependencies are installed: torch, torch-directml, diffusers, transformers, safetensors, pillow, numpy.
6. Run: `python app.py`
7. Open `http://127.0.0.1:7860` in your browser.

The first request loads the model. Images are saved in `generated/` and appear automatically in the website gallery. Set `DML_DEVICE_INDEX` before launch if you need to force a DirectML adapter.

## Troubleshooting

- Any generation error now shows in the red text under the Stop button AND prints a full traceback in the PowerShell window running `python app.py`. Send that traceback if something still fails.
- The first image takes a while: the model downloads and loads on the first request (the GPU box shows "Loading model").
- `torch-directml` needs a matching `torch` and a supported Python (3.10 or 3.11 is the safest). If pip reports conflicts, install `torch-directml` first and let it choose the torch version.
