"""Guard v886 -- Polyhedron Load Upscale Model: a drop-in that says more.

WHAT THIS PINS (the load path is EXECUTED against stand-ins, not read):

  L1  THE DROP-IN IS A DROP-IN. Slot 0 of RETURN_TYPES is UPSCALE_MODEL and
      the object handed back is the spandrel descriptor UNCHANGED -- the same
      identity core's loader passes on. Core's three quirks are honoured,
      each proven by RUNNING the load: safe_load=True, the 'module.' prefix
      replacement for those SwinIR exports, .eval(), and a patcher built on
      out.model with core's load/offload devices.

  L2  THE SIZE IS IN THE LIST, AND COMES BACK OFF IT. The combo decorates
      with ' \u00b7 <n> MB' (the v828 house form, shared with every other
      Polyhedron loader) and load() strips it before touching the folder --
      driven end to end with a real file on disk, so a stale decoration
      cannot reach get_full_path_or_raise.

  L3  VALIDATE_INPUTS EXISTS AND WAVES EVERYTHING THROUGH. Without it core's
      server-side membership check kills a run whose saved value carries a
      size that has since changed (the v823/v828 wound). Pinned by CALLING
      it with a value the current list does not offer.

  L4  THE FAILURE NAMES THE FILE. A state dict spandrel reads as something
      other than an ImageModelDescriptor must raise with the FILENAME in the
      message -- core's own text names no file, which is no help with two
      loaders on one canvas.

  L5  ONE SOURCE FOR THE CARD. The info readout goes through
      ph_power_upscale._model_card; it is IMPORTED, never re-implemented, and
      a failure inside it must not stop the node from loading a model.

  L6  NO SECOND OWNER OF DTYPE. This loader must not grow a precision /
      dtype widget: _esrgan_pass already decides half-precision from the
      descriptor's own supports_half, and two owners drift.

  L7  BOOKKEEPING: registered with a \u2b21 display name, present in BOTH
      regenerated baselines, and its widget row is exactly `model_name`.

  L8  MUTATIONS: hole each promise, demand red.

Script-style: exit 0 = pass.
"""
import importlib.util
import os
import re
import sys
import tempfile
import types

NAME = "v886"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MOD = os.path.join(ROOT, "nodes", "ph_upscale_loader.py")
INIT = os.path.join(ROOT, "__init__.py")

_checks = 0


def _fail(msg):
    print("[%s] FAIL -- %s" % (NAME, msg))
    sys.exit(1)


def _need(cond, msg):
    global _checks
    _checks += 1
    if not cond:
        _fail(msg)


def _read(p):
    return open(p, encoding="utf-8").read()


# ---------------------------------------------------------------------------
# stand-ins: a fake models/upscale_models tree, a fake comfy, a fake spandrel
# ---------------------------------------------------------------------------
TMP = tempfile.mkdtemp()
UPS = os.path.join(TMP, "upscale_models")
os.makedirs(UPS)
# 3 MiB so the house formatter says "3 MB" and nothing about KB
with open(os.path.join(UPS, "ESRGAN_4x.pth"), "wb") as fh:
    fh.write(b"\0" * (3 * (1 << 20)))

fp = types.ModuleType("folder_paths")
fp.models_dir = TMP
fp.get_filename_list = lambda f: (sorted(os.listdir(UPS))
                                  if f == "upscale_models" else [])


def _full(f, n):
    p = os.path.join(UPS, n)
    return p if f == "upscale_models" and os.path.isfile(p) else None


def _raise_full(f, n):
    p = _full(f, n)
    if p is None:
        raise FileNotFoundError("no such upscale model: %r" % (n,))
    return p


fp.get_full_path = _full
fp.get_full_path_or_raise = _raise_full
fp.get_folder_paths = lambda f: [UPS]
sys.modules["folder_paths"] = fp

_loaded = {}          # what the fake comfy recorded


class _Patcher:
    def __init__(self, model, load_device=None, offload_device=None):
        self.model = model
        self.load_device = load_device
        self.offload_device = offload_device


def _install_comfy(with_core_patcher=True):
    for n in ("comfy", "comfy.utils", "comfy.model_patcher",
              "comfy.model_management", "comfy.sd", "comfy.samplers",
              "comfy.sample", "comfy.cli_args"):
        sys.modules.setdefault(n, types.ModuleType(n))
    sys.modules["comfy.samplers"].KSampler = types.SimpleNamespace(
        SAMPLERS=["euler"], SCHEDULERS=["simple"])
    sys.modules["comfy.cli_args"].args = types.SimpleNamespace(
        preview_method=None)
    sys.modules["comfy.cli_args"].LatentPreviewMethod = object
    sys.modules["comfy"].sample = sys.modules["comfy.sample"]
    sys.modules["comfy"].samplers = sys.modules["comfy.samplers"]
    sys.modules["comfy"].sd = sys.modules["comfy.sd"]
    cu = sys.modules["comfy.utils"]

    def _load_torch_file(path, safe_load=False):
        _loaded["path"] = path
        _loaded["safe_load"] = safe_load
        return dict(_loaded.get("sd", {"weight": 1}))

    def _prefix_replace(sd, mapping):
        _loaded["prefix"] = mapping
        out = {}
        for k, v in sd.items():
            for a, b in mapping.items():
                if k.startswith(a):
                    k = b + k[len(a):]
            out[k] = v
        return out

    cu.load_torch_file = _load_torch_file
    cu.state_dict_prefix_replace = _prefix_replace
    mp = sys.modules["comfy.model_patcher"]
    for attr in ("CoreModelPatcher", "ModelPatcher"):
        if hasattr(mp, attr):
            delattr(mp, attr)
    if with_core_patcher:
        mp.CoreModelPatcher = _Patcher
    mp.ModelPatcher = _Patcher
    mm = sys.modules["comfy.model_management"]
    mm.get_torch_device = lambda: "cuda:0"
    mm.unet_offload_device = lambda: "cpu"
    sys.modules["comfy"].utils = cu
    sys.modules["comfy"].model_patcher = mp
    sys.modules["comfy"].model_management = mm


class _Desc:
    """Stand-in spandrel ImageModelDescriptor."""

    def __init__(self):
        self.model = object()
        self.scale = 4
        self.supports_half = True
        self.architecture = types.SimpleNamespace(name="ESRGAN")
        self.evaled = False

    def eval(self):
        self.evaled = True
        return self


class _NotAnImage:
    def eval(self):
        return self


def _install_spandrel(returns):
    sp = types.ModuleType("spandrel")

    class ModelLoader:
        def load_from_state_dict(self, sd):
            _loaded["sd_seen"] = dict(sd)
            return returns

    sp.ModelLoader = ModelLoader
    sp.ImageModelDescriptor = _Desc
    sys.modules["spandrel"] = sp


def _load(src=None, siblings=("ph_basics", "ph_power_upscale")):
    """Import the loader with its siblings reachable as real modules.

    ph_power_upscale is loaded for REAL (not stubbed): the info card must come
    from the one source, and a guard that stubs the thing it claims to pin
    proves only that its own stub works."""
    pkg = types.ModuleType("plspack")
    pkg.__path__ = [os.path.join(ROOT, "nodes")]
    sys.modules["plspack"] = pkg
    for stem in siblings:
        sys.modules.pop("plspack." + stem, None)
        spec = importlib.util.spec_from_file_location(
            "plspack." + stem, os.path.join(ROOT, "nodes", stem + ".py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules["plspack." + stem] = m
        spec.loader.exec_module(m)
    if src is None:
        spec = importlib.util.spec_from_file_location(
            "plspack.ph_upscale_loader", MOD)
        m = importlib.util.module_from_spec(spec)
        sys.modules["plspack.ph_upscale_loader"] = m
        spec.loader.exec_module(m)
        return m
    m = types.ModuleType("plspack.ph_upscale_loader_mut")
    m.__package__ = "plspack"
    m.__file__ = MOD
    exec(compile(src, MOD, "exec"), m.__dict__)
    return m


# ---------------------------------------------------------------------------
# L1 / L2 / L5 -- the load path, EXECUTED
# ---------------------------------------------------------------------------
def l1l2l5(src=None):
    _install_comfy()
    desc = _Desc()
    _install_spandrel(desc)
    _loaded.clear()
    _loaded["sd"] = {"module.layers.0.residual_group.blocks.0.norm1.weight": 1,
                     "module.other": 2}
    mod = _load(src)
    node = mod.ULSLoadUpscaleModel()

    _need(mod.ULSLoadUpscaleModel.RETURN_TYPES[0] == "UPSCALE_MODEL",
          "slot 0 must stay UPSCALE_MODEL or this is not a drop-in")

    listed = mod.ULSLoadUpscaleModel.INPUT_TYPES()["required"]["model_name"][0]
    _need(len(listed) == 1 and re.match(r"^ESRGAN_4x\.pth \u00b7 3 ?MB$", listed[0]),
          "the combo must decorate with the house size form "
          "'<name> \u00b7 <n> MB' (got %r)" % (listed,))

    out, info = node.load(listed[0])          # the DECORATED value, as saved
    _need(out is desc,
          "the descriptor must be handed on UNCHANGED - core passes the same "
          "object identity, and the consumers fingerprint it")
    _need(desc.evaled, "core calls .eval() on the descriptor; so must we")
    _need(_loaded.get("safe_load") is True,
          "load_torch_file must be called with safe_load=True (core's own)")
    _need(_loaded.get("path", "").endswith("ESRGAN_4x.pth"),
          "the size decoration must be STRIPPED before the folder lookup - "
          "otherwise a file whose size changed on disk stops resolving")
    _need(_loaded.get("prefix") == {"module.": ""},
          "the 'module.' prefix replacement for SwinIR exports is core's "
          "quirk and must be carried over")
    _need("module.other" not in _loaded.get("sd_seen", {}),
          "the prefix replacement must actually reach spandrel")
    _need(isinstance(getattr(desc, "patcher", None), _Patcher)
          and desc.patcher.model is desc.model
          and desc.patcher.load_device == "cuda:0"
          and desc.patcher.offload_device == "cpu",
          "the patcher must wrap out.model with core's load/offload devices")

    _need("ESRGAN_4x.pth" in info and "ESRGAN" in info and "x4" in info,
          "the info readout must name the FILE and what was loaded, through "
          "the REAL _model_card (got %r)" % (info,))
    _need("MB" in info, "the info readout must carry the size too")

    # L5c -- the card source is unreachable: the model must still load, and
    # the degradation must SPEAK. The first draft returned a bare class name
    # in silence; a silent degradation is the failure mode this house keeps
    # paying for, so it is pinned here rather than tolerated.
    import io as _io
    sys.modules.pop("plspack.ph_power_upscale", None)
    broken = types.ModuleType("plspack.ph_power_upscale")
    sys.modules["plspack.ph_power_upscale"] = broken   # has no _model_card
    d3 = _Desc()
    _install_spandrel(d3)
    _loaded.clear()
    buf, old_out = _io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        out3, info3 = node.load("ESRGAN_4x.pth")
    finally:
        sys.stdout = old_out
    _need(out3 is d3,
          "a missing card source must never stop the model from loading")
    _need("unavailable" in buf.getvalue(),
          "a degraded readout must SAY it is degraded (v552) - got %r"
          % (buf.getvalue(),))

    # older core generation: no CoreModelPatcher
    sys.modules.pop("plspack.ph_power_upscale", None)
    _install_comfy(with_core_patcher=False)
    d2 = _Desc()
    _install_spandrel(d2)
    _loaded.clear()
    node.load("ESRGAN_4x.pth")
    _need(isinstance(getattr(d2, "patcher", None), _Patcher),
          "a core without CoreModelPatcher must still get a patcher - the "
          "v882 law: a measurement holds only for the build it was taken on")
    return mod


# ---------------------------------------------------------------------------
# L3 -- VALIDATE_INPUTS waves a stale decoration through
# ---------------------------------------------------------------------------
def l3(mod):
    v = getattr(mod.ULSLoadUpscaleModel, "VALIDATE_INPUTS", None)
    _need(callable(v),
          "VALIDATE_INPUTS is missing - core's membership check would kill "
          "every run whose saved value carries a size that changed since")
    _need(v(model_name="ESRGAN_4x.pth \u00b7 999 MB") is True,
          "VALIDATE_INPUTS must accept a value the current list no longer "
          "offers; load() is where an unknown file fails, by name")


# ---------------------------------------------------------------------------
# L4 -- the failure names the file
# ---------------------------------------------------------------------------
def l4(src=None):
    _install_comfy()
    _install_spandrel(_NotAnImage())
    _loaded.clear()
    mod = _load(src)
    try:
        mod.ULSLoadUpscaleModel().load("ESRGAN_4x.pth")
    except Exception as exc:
        _need("ESRGAN_4x.pth" in str(exc),
              "the rejection must name the FILE - core's message names none, "
              "which is no help with two loaders on one canvas (got %r)"
              % (str(exc),))
        return
    _fail("a non-image model was ACCEPTED - core rejects it and so must we")


# ---------------------------------------------------------------------------
# L5b / L6 / L7 -- one source, no second dtype owner, bookkeeping
# ---------------------------------------------------------------------------
def l5b_l6_l7(src=None):
    py = src if src is not None else _read(MOD)
    init = _read(INIT)

    _need("_model_card" in py and "import" in py.split("_model_card")[0][-400:],
          "the info card must be IMPORTED from ph_power_upscale, not "
          "re-implemented - two places computing one sentence drift")
    _need("def _model_card" not in py,
          "a local copy of _model_card is exactly the drift this pins against")

    it = py[py.index("def INPUT_TYPES"):py.index("def VALIDATE_INPUTS")]
    names = re.findall(r'"([a-z_0-9]+)":\s*\(', it)
    _need(names == ["model_name"],
          "this loader takes ONE widget. A precision/dtype widget would be a "
          "second owner of a decision _esrgan_pass already makes from the "
          "descriptor's supports_half (found: %s)" % (", ".join(names),))
    _need(not re.search(r"\b(fp16|bf16|dtype|precision)\b", it),
          "no dtype vocabulary belongs in this node's inputs")

    _need('NODE_CLASS_MAPPINGS["ULSLoadUpscaleModel"]' in init,
          "the node is not registered")
    _need(re.search(r'NODE_DISPLAY_NAME_MAPPINGS\["ULSLoadUpscaleModel"\]\s*=\s*"\u2b21',
                    init),
          "the display name must carry the house mark \u2b21")

    # v889 FIX, declared: these two lines hard-coded the baseline FILENAME, so
    # the guard broke the moment a later cut regenerated the baselines -- which
    # is exactly the v580 wound that test_v577/#678/#691 were already amended
    # for. A guard that must be hand-edited on a schedule will one day be
    # hand-edited wrong. Resolve the newest by version instead; the PROMISE
    # (this node_id is in the baselines with exactly one widget row) is
    # unchanged and still checked below.
    import glob as _glob

    def _newest(pattern):
        hits = sorted(_glob.glob(os.path.join(ROOT, pattern)))
        assert hits, "no baseline matching " + pattern
        return _read(hits[-1])

    ids = _newest("NODE_IDS_baseline_v*.txt")
    wid = _newest("WIDGET_ORDER_baseline_v*.txt")
    _need(re.search(r"^ULSLoadUpscaleModel$", ids, re.M),
          "the new node_id must be in the regenerated NODE_IDS baseline - a "
          "new node is a declared act")
    _need(re.search(r"^ULSLoadUpscaleModel\tmodel_name$", wid, re.M),
          "the widget baseline must carry exactly one slot for it")


# ---------------------------------------------------------------------------
# L8 -- mutations
# ---------------------------------------------------------------------------
def _mut(fn, old, new, label):
    src = _read(MOD)
    if old not in src:
        _fail("MUTATION ANCHOR MISSING (%s): %r" % (label, old[:70]))
    try:
        fn(src.replace(old, new, 1))
    except SystemExit as e:
        if e.code:
            return True
        _fail("mutation '%s' exited 0 - not caught" % label)
    except Exception:
        return True
    _fail("mutation '%s' left the guard GREEN - the promise is not pinned"
          % label)


def l8():
    n = 0
    n += bool(_mut(l1l2l5,
                   "        model_name = _strip_size(model_name)",
                   "        model_name = str(model_name)",
                   "the size decoration is not stripped"))
    n += bool(_mut(l1l2l5,
                   'sd = comfy.utils.load_torch_file(model_path, safe_load=True)',
                   'sd = comfy.utils.load_torch_file(model_path)',
                   "safe_load dropped"))
    n += bool(_mut(l1l2l5,
                   'if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:',
                   'if False and "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:',
                   "the SwinIR prefix quirk dropped"))
    n += bool(_mut(l1l2l5,
                   "        out = ModelLoader().load_from_state_dict(sd).eval()",
                   "        out = ModelLoader().load_from_state_dict(sd)",
                   ".eval() dropped"))
    n += bool(_mut(l4,
                   '"[PLS] Load Upscale Model: \'%s\' is not a single-image upscale "',
                   '"[PLS] Load Upscale Model: not a single-image upscale "',
                   "the rejection stops naming the file"))
    n += bool(_mut(l5b_l6_l7,
                   '                    "tooltip": "Pixel upscale model (models/upscale_models). "',
                   '                }),\n                "precision": (["auto", "fp16"], {\n'
                   '                    "tooltip": "Pixel upscale model (models/upscale_models). "',
                   "a second dtype owner appears"))
    return n


def main():
    mod = l1l2l5()
    l3(mod)
    l4()
    l5b_l6_l7()
    muts = l8()
    print("PASS: %s -- %d promises pinned, %d mutations caught (drop-in "
          "identity, core's three quirks, size in and off the list, named "
          "rejection, one card source, one dtype owner, both baselines)"
          % (NAME, _checks, muts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
