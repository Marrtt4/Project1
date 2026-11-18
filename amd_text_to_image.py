"""
AMD DirectML text-to-image generator for Windows.
Uses the AMD Radeon GPU through torch-directml. No upscaling.
"""
import copy
import os
import re
import sys
import time

import numpy as np
import torch
import torch_directml
from diffusers import AutoPipelineForText2Image
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from PIL import Image
from transformers import CLIPImageProcessor

# ---------------- Settings ----------------
QUALITY = "fast"                 # SD-Turbo: fits 8 GB VRAM. "high" = SDXL-Turbo (needs more than 8 GB)
REALISTIC_STYLE = True
FILM_GRAIN = 0.0                 # 0 = off
STEPS = 4
GEN_WIDTH, GEN_HEIGHT = 768, 432    # final image size (same as your working setup)
SEED = None
OUT_DIR = "generated"
# ------------------------------------------

SD_MODEL = "stabilityai/sd-turbo"
SDXL_MODEL = "stabilityai/sdxl-turbo"
MODEL = SDXL_MODEL if QUALITY == "high" else SD_MODEL
IS_SDXL = MODEL == SDXL_MODEL
PHOTO_PREFIX = "RAW photo, "
PHOTO_SUFFIX = (
    ", color photograph, photorealistic, natural lighting, shot on 35mm lens, "
    "DSLR, sharp focus, highly detailed, realistic textures"
)
NON_PHOTO_WORDS = re.compile(
    r"\b(sketch|drawing|illustration|cartoon|anime|painting|watercolor|pixel art|"
    r"vector|comic|line art|3d render|cgi)\b",
    re.IGNORECASE,
)

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUT_DIR, exist_ok=True)

print("Starting AMD DirectML...")


def pick_dml_device():
    """Pick the discrete Radeon (RX 7600S) instead of the laptop's integrated GPU.

    Set DML_DEVICE_INDEX=<number> to force a specific adapter.
    """
    count = torch_directml.device_count()
    names = [torch_directml.device_name(i) for i in range(count)]
    for i, name in enumerate(names):
        print(f"  DirectML adapter {i}: {name}")
    forced = os.environ.get("DML_DEVICE_INDEX")
    if forced is not None:
        return torch_directml.device(int(forced))
    for i, name in enumerate(names):
        if "7600" in name:
            return torch_directml.device(i)
    return torch_directml.device()


DML_DEVICE = pick_dml_device()
print(f"DirectML device: {DML_DEVICE}")
print(f"Using {'SDXL-Turbo' if IS_SDXL else 'SD-Turbo'} on the AMD GPU.")

# Keep the safety checker on CPU for compatibility.
print("Loading safety checker on CPU...")
safety_checker = StableDiffusionSafetyChecker.from_pretrained(
    "CompVis/stable-diffusion-safety-checker"
)
feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load half-precision weights and move the image model to DirectML.
print("Loading image model onto AMD GPU...")
extra = {} if IS_SDXL else {"safety_checker": None, "requires_safety_checker": False}
try:
    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        **extra,
    )
except Exception:
    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        use_safetensors=True,
        **extra,
    )
# The final latent -> image decode needs a lot of video memory, so it runs on the CPU.
# IMPORTANT: make the CPU copy of the VAE *before* moving the pipeline to the GPU, and
# leave the pipeline's own VAE on the GPU. Diffusers decides which device to put the
# prompt tokens on by looking at the first model in the pipeline (the VAE). If the VAE
# sits on the CPU while the text encoder is on DirectML, the tokens end up on the wrong
# device and you get "unbox expects Dml at::Tensor as inputs".
cpu_vae = copy.deepcopy(pipe.vae).to("cpu", dtype=torch.float32).eval()
pipe = pipe.to(DML_DEVICE)
pipe.set_progress_bar_config(leave=True)

def decode_on_cpu(latents):
    """Turn the GPU's latent result into a PIL image using the CPU."""
    vae = cpu_vae.float()
    cfg = vae.config
    # Copy to CPU first, then force float32 through numpy. DirectML's .to(device, dtype)
    # can leave the tensor in half precision, which breaks the float32 CPU VAE.
    latents = latents.detach().to("cpu")
    latents = torch.from_numpy(np.ascontiguousarray(latents.numpy()).astype(np.float32))
    mean = getattr(cfg, "latents_mean", None)
    std = getattr(cfg, "latents_std", None)
    if mean is not None and std is not None:
        mean = torch.tensor(mean).view(1, -1, 1, 1)
        std = torch.tensor(std).view(1, -1, 1, 1)
        latents = latents * std / cfg.scaling_factor + mean
    else:
        latents = latents / cfg.scaling_factor
    with torch.no_grad():
        decoded = vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(decoded, output_type="pil")[0]


def safety_filter(image):
    """Return (checked image, was_flagged). Safety checking stays on CPU."""
    rgb = image.convert("RGB")
    arr = np.array(rgb).astype(np.float32) / 255.0
    clip_input = feature_extractor(rgb, return_tensors="pt").pixel_values
    with torch.no_grad():
        checked, flags = safety_checker(images=arr[None], clip_input=clip_input)
    if flags[0]:
        replacement = (np.clip(checked[0], 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(replacement), True
    return image, False


def add_grain(image, strength):
    if strength <= 0:
        return image
    array = np.array(image).astype(np.float32)
    noise = np.random.default_rng().normal(0, strength, array.shape[:2])[..., None]
    return Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8))


def build_prompt(prompt):
    if REALISTIC_STYLE and not NON_PHOTO_WORDS.search(prompt):
        return PHOTO_PREFIX + prompt + PHOTO_SUFFIX
    return prompt


def generate(prompt):
    # A CPU generator is intentional. Diffusers creates the noise on CPU and transfers it.
    generator = None
    if SEED is not None:
        generator = torch.Generator(device="cpu").manual_seed(SEED)

    start = time.time()
    with torch.no_grad():
        result = pipe(
            prompt=build_prompt(prompt),
            num_inference_steps=STEPS,
            guidance_scale=0.0,
            width=GEN_WIDTH,
            height=GEN_HEIGHT,
            generator=generator,
            output_type="latent",
        )

    print("Decoding image on CPU...")
    image, flagged = safety_filter(decode_on_cpu(result.images))
    if flagged:
        print("Safety filter flagged this image and replaced it.")

    slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:40] or "image"
    path = os.path.join(OUT_DIR, f"{slug}_{int(time.time())}.png")
    add_grain(image, FILM_GRAIN).save(path)
    print(f"Saved {path} ({time.time() - start:.1f}s)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        generate(" ".join(sys.argv[1:]))
    else:
        print("Describe the image you want. Press Enter on an empty line to quit.")
        while True:
            text = input("\nPrompt> ").strip()
            if not text:
                break
            generate(text)
