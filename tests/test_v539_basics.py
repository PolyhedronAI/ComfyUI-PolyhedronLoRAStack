"""Guard v539 -- Load CLIP / Load VAE: native types, live type list, wan intelligence."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _fail(m): print("FAIL: " + m); sys.exit(1)
def _read(*p): return open(os.path.join(ROOT, *p), encoding="utf-8").read()

def main():
    py = _read("nodes", "ph_basics.py"); js = _read("web", "js", "ph_basics.js"); init = _read("__init__.py")
    # VAE: native return, precision override, family table, model cross-check
    if 'RETURN_TYPES = ("VAE", "STRING")' not in py: _fail("VAE must return the NATIVE VAE type + info")
    body = py.split('"""', 2)[-1]  # module docstring documents the decision; the CODE must be clean
    if "WANVAE" in body: _fail("wrapper-only WANVAE type must not appear in code")
    if '"precision": (["auto", "bf16", "fp16", "fp32"]' not in py: _fail("precision override missing")
    for ch in ("4:", "16:", "48:"):
        if ch not in py: _fail(f"family table entry {ch} missing")
    if "TI2V-5B" not in py or "wan_2.1_vae" not in py: _fail("wan guidance text missing")
    if "_model_latent_channels" not in py or "latent_format" not in py: _fail("model cross-check missing")
    if "raise ValueError" not in py: _fail("mismatch must fail loudly before sampling")
    # CLIP: live type list + conservative auto
    # v716 REHUNG: the pull moved into the shared _core_type_list() helper when
    # the node learned to read DualCLIPLoader's list too, so the old literal
    # "_core_nodes.CLIPLoader.INPUT_TYPES()" no longer appears. The INVARIANT is
    # unchanged and is what gets pinned here: the list comes from the RUNNING
    # ComfyUI, both loaders are consulted, and a frozen snapshot backs each up.
    if "import nodes as _core_nodes" not in py: _fail("live type list pull missing")
    if '.INPUT_TYPES()["required"]["type"][0]' not in py: _fail("live type list pull missing")
    if '_core_type_list("CLIPLoader"' not in py: _fail("single-encoder type list no longer pulled live")
    if '_core_type_list("DualCLIPLoader"' not in py: _fail("dual-encoder type list no longer pulled live")
    if "_CLIP_TYPE_FALLBACK" not in py: _fail("frozen fallback snapshot missing")
    if "_CLIP_TYPE_DUAL_FALLBACK" not in py: _fail("frozen dual fallback snapshot missing")
    if '("umt5", "wan")' not in py or '("qwen_2.5_vl", "qwen_image")' not in py: _fail("auto patterns changed")
    if "set the type explicitly" not in py: _fail("auto must error instead of guessing")
    if 'comfy.sd.load_clip' not in py or 'get_folder_paths("embeddings")' not in py: _fail("core load_clip call shape changed")
    # shared ui channel + JS
    if '_UI_KEY = "pls_basics"' not in py or 'pls_basics' not in js: _fail("pls_basics ui channel mismatch")
    if "[PLS] ph_basics.js v542 loaded" not in js: _fail("ph_basics.js banner missing/stale")
    # seed regression (v538 surface intact)
    for needle in ('"control_after_generate": True', "0xffffffffffffffff", '"pls_seed"'):
        if needle not in py: _fail(f"v538 seed surface regressed: {needle!r}")
    for needle in ("ULSLoadCLIP", "ULSLoadVAE", "⬡ Polyhedron Load CLIP", "⬡ Polyhedron Load VAE"):
        if needle not in init: _fail(f"registration incomplete: {needle!r}")
    print("PASS: v539 basics -- native VAE + wan intelligence, live CLIP types, seed intact")
    sys.exit(0)

if __name__ == "__main__":
    main()
