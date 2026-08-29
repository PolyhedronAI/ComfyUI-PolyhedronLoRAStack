"""Guard v541 -- Load Model: fused switch+loader, GGUF routing, name output kept."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _fail(m): print("FAIL: " + m); sys.exit(1)
def _read(*p): return open(os.path.join(ROOT, *p), encoding="utf-8").read()

def main():
    py = _read("nodes", "ph_basics.py"); js = _read("web", "js", "ph_basics.js"); init = _read("__init__.py")
    if 'RETURN_TYPES = ("MODEL", "STRING", "STRING")' not in py:
        _fail("must emit MODEL + model_name + info")
    if '"unet_gguf"' not in py or '"diffusion_models_gguf"' not in py:
        _fail("merged list must include the GGUF folders")
    if 'endswith(".gguf")' not in py:
        _fail("gguf suffix routing missing")
    if 'NODE_CLASS_MAPPINGS.get("UnetLoaderGGUF")' not in py:
        _fail("runtime delegation to the registered GGUF loader missing")
    if "ComfyUI-GGUF" not in py or "is not installed" not in py:
        _fail("clear missing-GGUF-pack error missing")
    if "comfy.sd.load_diffusion_model" not in py:
        _fail("core safetensors path missing")
    if "output_vae=False, output_clip=False" not in py:
        _fail("checkpoint path must load MODEL only")
    if "fp8_optimizations" not in py or '"fp8_e5m2"' not in py:
        _fail("weight_dtype options incomplete")
    if "slot {int(select)} is empty" not in py:
        _fail("empty-slot fail-loud missing")
    if '"ULSLoadModel"' not in js:
        _fail("status line not wired for ULSLoadModel")
    if "[PLS] ph_basics.js v542 loaded" not in js:
        _fail("banner not bumped with the file touch")
    for needle in ("ULSLoadModel", "\u2b21 Polyhedron Load Model"):
        if needle not in init: _fail(f"registration incomplete: {needle!r}")
    print("PASS: v541 load model -- gguf/unet/checkpoint routing, MODEL+name out, fail-loud")
    sys.exit(0)

if __name__ == "__main__":
    main()
