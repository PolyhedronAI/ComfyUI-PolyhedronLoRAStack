# -*- coding: ascii -*-
"""Guard v1063 -- the LoRA stack's SEQ loader is the pack's own, and the pack
as a whole calls no Core node classes (Frank, 07.10.2026: the go for the three
numbers; Core is GPL, the pack MIT -- the comfy.* model library only).

  D1  static (always): uls_stack_node imports no `nodes` / `comfy_extras`; both
      stack classes build _StackLoraLoader. PACK-WIDE: no file in nodes/ (vendor
      aside) or __init__.py imports a core node module -- the one exception is
      ph_basics._load_gguf (UnetLoaderGGUF, the third-party ComfyUI-GGUF pack,
      an open question), and the scan also catches importlib / __import__ /
      sys.modules lookups.
  D2  parity with the core's LoraLoader on a real ModelPatcher (a small
      diffusion model) and a real comfy.sd.CLIP (CLIP-L, random weights):
      three LoRA files (UNet + text encoder with alpha, UNet only with
      metadata, text encoder only), strengths (0.8, 0.5) / (1.0, 0.0) /
      (-0.7, 1.3): the patched weights are bit for bit the core's, the
      metadata attachment is the same; strengths (0, 0) hand back the very
      same objects.
  D3  a three-LoRA SEQ stack (_apply_seq) == three chained core LoraLoaders,
      bit for bit, model and clip.
  D4  the cache: a second run of the stack reads no file (the core's loader
      re-reads all three -- it keeps only the last one); both timed, said.

Script-style: exit 0 = pass.
"""
import ast
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NODES = ROOT / "nodes"
FAILS = []


def _need(c, m):
    print(("  ok    " if c else "  FAIL  ") + m)
    if not c:
        FAILS.append(m)


def _core_root():
    env = os.environ.get("PLS_CORE_ROOT")
    for c in ([env] if env else []) + [str(ROOT.parent.parent)]:
        if c and os.path.isfile(os.path.join(c, "comfy", "sd.py")) and os.path.isfile(os.path.join(c, "nodes.py")):
            return c
    return None


CORE_MODS = ("nodes", "comfy_extras")


def node_imports(path):
    """(function name or '<module>', line, module) for every reference to a
    core node module in the CODE of a file: import / from-import, importlib /
    __import__ with a literal name, sys.modules[...] / .get(...)."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    owner = {}

    def visit(node, fn):
        for ch in ast.iter_child_nodes(node):
            here = ch.name if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn
            owner[id(ch)] = here
            visit(ch, here)

    visit(tree, "<module>")
    out = []

    def hit(node, mod):
        if isinstance(mod, str) and mod.split(".")[0] in CORE_MODS:
            out.append((owner.get(id(node), "<module>"), node.lineno, mod))

    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                hit(n, a.name)
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            hit(n, n.module)
        elif isinstance(n, ast.Call) and n.args and isinstance(n.args[0], ast.Constant):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name in ("import_module", "__import__") or (
                    name == "get" and isinstance(f, ast.Attribute) and isinstance(f.value, ast.Attribute)
                    and f.value.attr == "modules"):
                hit(n, n.args[0].value)
        elif isinstance(n, ast.Subscript) and isinstance(n.value, ast.Attribute) and n.value.attr == "modules" \
                and isinstance(n.slice, ast.Constant):
            hit(n, n.slice.value)
    return out


# ------------------------------------------------------------------ D1
print("D1 static")
stack_src = (NODES / "uls_stack_node.py").read_text(encoding="utf-8")
_need(not node_imports(NODES / "uls_stack_node.py"), "D1 uls_stack_node imports no core node module")
_need(stack_src.count("self._loader = _StackLoraLoader()") == 2 and "LoraLoader as" not in stack_src,
      "D1 both stack classes build the pack's own SEQ loader")
files = sorted(p for p in NODES.rglob("*.py") if "vendor" not in p.parts) + [ROOT / "__init__.py"]
found = []
for p in files:
    for fn, line, mod in node_imports(p):
        found.append((p.relative_to(ROOT).as_posix(), fn, line, mod))
allowed = [f for f in found if f[0] == "nodes/ph_basics.py" and f[1] == "_load_gguf"]
rest = [f for f in found if f not in allowed]
_need(not rest, "D1 PACK-WIDE: no core node module in %d files (code, not prose)%s"
      % (len(files), "" if not rest else " -- %r" % rest))
_need(len(allowed) == 1 and "UnetLoaderGGUF" in (NODES / "ph_basics.py").read_text(encoding="utf-8"),
      "D1 the one exception: ph_basics._load_gguf (UnetLoaderGGUF, third party) %r" % allowed)
probe = ("import nodes\nfrom comfy_extras.nodes_audio import x\ndef f():\n    import importlib\n"
         "    importlib.import_module('comfy_extras.nodes_flux')\n    sys.modules.get('nodes')\n"
         "    sys.modules['nodes']\n    __import__('nodes')\n")
tmpf = Path(tempfile.mkdtemp(prefix="pls1063s_")) / "p.py"
tmpf.write_text(probe, encoding="utf-8")
_need(len(node_imports(tmpf)) == 6, "D1 the scanner sees all six spellings (%d)" % len(node_imports(tmpf)))

CORE = _core_root()
if not CORE:
    print("  skip  D2-D4 (no core reachable -- set PLS_CORE_ROOT)")
else:
    sys.path.insert(0, CORE)
    import comfy.cli_args  # noqa: E402
    comfy.cli_args.args.cpu = True
    import importlib  # noqa: E402
    import types  # noqa: E402
    import torch  # noqa: E402
    import torch.nn as nn  # noqa: E402
    import safetensors.torch as sft  # noqa: E402
    import folder_paths  # noqa: E402
    import comfy.utils  # noqa: E402
    import comfy.sd  # noqa: E402
    import comfy.sd1_clip  # noqa: E402
    import comfy.model_patcher  # noqa: E402
    import comfy.supported_models_base as smb  # noqa: E402
    import nodes as core_nodes  # noqa: E402  (the measuring stick)

    # the pack as ComfyUI loads it (relative imports inside uls_stack_node)
    pkg_name = "pls1063pack"
    spec = importlib.util.spec_from_file_location(pkg_name, str(ROOT / "__init__.py"),
                                                  submodule_search_locations=[str(ROOT)])
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(ROOT)]
    sys.modules[pkg_name] = pkg
    sub = types.ModuleType(pkg_name + ".nodes")
    sub.__path__ = [str(NODES)]
    sys.modules[pkg_name + ".nodes"] = sub
    S = importlib.import_module(pkg_name + ".nodes.uls_stack_node")

    # -------------------------------------------------------------- models
    D = 24

    class _DM(nn.Module):
        def __init__(s):
            super().__init__()
            s.blocks = nn.ModuleList([nn.Linear(D, D), nn.Linear(D, D)])
            s.out = nn.Linear(D, 8)

    class _Cfg(object):
        unet_config = {"image_model": "guard"}

    class _Base(nn.Module):
        def __init__(s):
            super().__init__()
            s.diffusion_model = _DM()
            s.model_config = _Cfg()

    torch.manual_seed(0)
    base = _Base()
    model = comfy.model_patcher.ModelPatcher(base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    clip = comfy.sd.CLIP(smb.ClipTarget(comfy.sd1_clip.SD1Tokenizer, comfy.sd1_clip.SD1ClipModel))
    gen = torch.Generator().manual_seed(1)
    csd = {}
    for k, v in clip.cond_stage_model.state_dict().items():
        csd[k] = v if not v.is_floating_point() else (
            torch.ones_like(v) if ("norm" in k and k.endswith("weight")) else torch.randn(v.shape, generator=gen) * 0.02)
    clip.cond_stage_model.load_state_dict(csd, strict=False)

    loras = Path(tempfile.mkdtemp(prefix="pls1063_"))
    folder_paths.folder_names_and_paths["loras"] = ([str(loras)], {".safetensors"})
    getattr(folder_paths, "filename_list_cache", {}).pop("loras", None)
    R = 4

    def r(*sh):
        return torch.randn(*sh, generator=gen) * 0.05

    def unet_part(sd, alpha=None):
        for name, o, i in (("blocks_0", D, D), ("blocks_1", D, D), ("out", 8, D)):
            sd["lora_unet_%s.lora_down.weight" % name] = r(R, i)
            sd["lora_unet_%s.lora_up.weight" % name] = r(o, R)
            if alpha is not None:
                sd["lora_unet_%s.alpha" % name] = torch.tensor(float(alpha))

    def te_part(sd, alpha=None):
        for layer in (0, 3):
            for proj in ("q_proj", "v_proj"):
                k = "lora_te_text_model_encoder_layers_%d_self_attn_%s" % (layer, proj)
                sd[k + ".lora_down.weight"] = r(R, 768)
                sd[k + ".lora_up.weight"] = r(768, R)
                if alpha is not None:
                    sd[k + ".alpha"] = torch.tensor(float(alpha))

    a, b, c = {}, {}, {}
    unet_part(a, alpha=2.0)
    te_part(a, alpha=2.0)
    unet_part(b)
    te_part(c, alpha=1.0)
    sft.save_file(a, str(loras / "both.safetensors"))
    sft.save_file(b, str(loras / "unet_meta.safetensors"), metadata={"ss_network_alpha": "4", "guard": "v1063"})
    sft.save_file(c, str(loras / "te.safetensors"))

    def patched(mp):
        """The weights this patcher would run with (patch, read, unpatch)."""
        mp.patch_model()
        try:
            return {k: v.detach().clone() for k, v in mp.model.state_dict().items()}
        finally:
            mp.unpatch_model()

    def same_sd(x, y):
        return x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x)

    def attach(mp):
        try:
            return mp.get_attachment("lora_metadata")
        except Exception:
            return getattr(mp, "attachments", {}).get("lora_metadata")

    base_m, base_c = patched(model), patched(clip.patcher)

    # -------------------------------------------------------------- D2
    print("D2 one LoRA, parity with the core's LoraLoader")
    ok_all = True
    for name in ("both.safetensors", "unet_meta.safetensors", "te.safetensors"):
        for sm, sc in ((0.8, 0.5), (1.0, 0.0), (-0.7, 1.3)):
            om, oc = S._StackLoraLoader().load_lora(model, clip, name, sm, sc)
            cm, cc = core_nodes.LoraLoader().load_lora(model, clip, name, sm, sc)
            pm_o, pm_c = patched(om), patched(cm)
            pc_o, pc_c = patched(oc.patcher), patched(cc.patcher)
            same = same_sd(pm_o, pm_c) and same_sd(pc_o, pc_c) and attach(om) == attach(cm) \
                and attach(oc.patcher) == attach(cc.patcher)
            moved = (not same_sd(pm_o, base_m)) if (sm and "te" != name[:2]) else True
            if not (same and moved):
                print("  info  %s %s %s: same %s moved %s" % (name, sm, sc, same, moved))
            ok_all &= same and moved
    _need(ok_all, "D2 3 files x 3 strength pairs: patched model + clip weights and the metadata == the core's")
    _need(attach(S._StackLoraLoader().load_lora(model, clip, "unet_meta.safetensors", 1.0, 1.0)[0]) == {
        "ss_network_alpha": "4", "guard": "v1063"}, "D2 the file's metadata rides along (as with the core)")
    om, oc = S._StackLoraLoader().load_lora(model, clip, "both.safetensors", 0, 0)
    _need(om is model and oc is clip, "D2 strengths (0, 0): the very same model and clip back")

    # -------------------------------------------------------------- D3
    print("D3 a three-LoRA SEQ stack")
    names = ["both.safetensors", "unet_meta.safetensors", "te.safetensors"]
    ws, wcs = [0.9, -0.4, 0.0], [0.6, 0.0, 1.1]
    m1, c1, errs = S._apply_seq(S._StackLoraLoader(), model, clip, names, ws, wcs)
    m2, c2 = model, clip
    for n, w, wc in zip(names, ws, wcs):
        if abs(w) < 1e-6 and abs(wc) < 1e-6:
            continue
        m2, c2 = core_nodes.LoraLoader().load_lora(m2, c2, n, w, wc)
    _need(not errs and same_sd(patched(m1), patched(m2)) and same_sd(patched(c1.patcher), patched(c2.patcher)),
          "D3 _apply_seq with the pack's loader == three chained core LoraLoaders (model + clip)")

    # -------------------------------------------------------------- D4
    print("D4 cache")
    S._TD_CACHE.clear()
    S._TD_CACHE_BYTES = 0
    real = comfy.utils.load_torch_file
    reads = []

    def counting(*a, **k):
        reads.append(os.path.basename(str(a[0])))
        return real(*a, **k)

    comfy.utils.load_torch_file = counting
    try:
        ours, core = S._StackLoraLoader(), core_nodes.LoraLoader()
        times = {}
        for who, ld in (("ours", ours), ("core", core)):
            for run in (1, 2):
                reads.clear()
                t0 = time.perf_counter()
                m, c = model, clip
                for n in names:
                    m, c = ld.load_lora(m, c, n, 0.7, 0.7)
                times[(who, run)] = (time.perf_counter() - t0, len(reads))
    finally:
        comfy.utils.load_torch_file = real
    print("  info  file reads / time: ours run1 %d %.3fs, run2 %d %.3fs | core run1 %d %.3fs, run2 %d %.3fs" % (
        times[("ours", 1)][1], times[("ours", 1)][0], times[("ours", 2)][1], times[("ours", 2)][0],
        times[("core", 1)][1], times[("core", 1)][0], times[("core", 2)][1], times[("core", 2)][0]))
    _need(times[("ours", 1)][1] == 3 and times[("ours", 2)][1] == 0,
          "D4 the pack's loader: 3 reads on the first run, 0 on the second")
    _need(times[("core", 2)][1] == 3, "D4 (measured: the core's loader re-reads all 3 on every run)")

print("test_v1063_stack_lora_loader: %s" % ("PASS" if not FAILS else "FAIL (%d)" % len(FAILS)))
sys.exit(1 if FAILS else 0)
