"""Guard #103 -- Empty Latent model families + core delegation (v679).

DRIVEN, not read:
  * TYPE_SPEC families pinned: sd = 4ch/8 (the classic class -- the old
    image alias put SDXL at 16 channels, wrong without a wired VAE),
    flux2 present as loud fallback row, image class unchanged (16/8).
  * TYPE_ORDER append-only: the six original labels keep their exact
    positions and spellings (combo serializes the label VALUE).
  * aliases: flux2 -> flux2, sdxl/sd15/sd21 -> sd, flux/sd3/qwen stay
    in the 16ch image class; unknown input still degrades to image.
  * plan_latent_shape for the new families analytic.
  * generate() seam with a stub delegate: TRUTH ORDER proven -- wired
    VAE probe beats the delegate, delegate beats the spec row, spec
    carries when the host has no core class (loud), noise overlay is
    applied ON the delegate's latent.

MUTATIONS (wound in a COPY, catch proven): M1 sd row silently becomes
16ch, M2 the delegate ignores the wired-VAE priority, M3 an original
combo label gets renamed.
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MATH_PY = os.path.join(ROOT, "nodes", "uls_latent_math.py")
NODE_PY = os.path.join(ROOT, "nodes", "ph_empty_latent.py")


def _fail(msg):
    print("[test_v679_latent_families] FAIL --", msg)
    sys.exit(1)


def _need(ok, msg):
    if not ok:
        _fail(msg)


def _import_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_node(path, name):
    """Import the node module with a minimal comfy stub (the sandbox has
    no ComfyUI) and the nodes/ dir on sys.path for its siblings."""
    import types
    nodes_dir = os.path.dirname(path)
    added = nodes_dir not in sys.path
    if added:
        sys.path.insert(0, nodes_dir)
    stubs = {}
    for mn in ("comfy", "comfy.model_management", "torch"):
        if mn not in sys.modules:
            stubs[mn] = types.ModuleType(mn)
            sys.modules[mn] = stubs[mn]
    if "comfy.model_management" in stubs:
        sys.modules["comfy.model_management"].intermediate_device = \
            lambda: "cpu"
        sys.modules["comfy"].model_management = \
            sys.modules["comfy.model_management"]
    try:
        return _import_from(path, name)
    finally:
        for mn in stubs:
            del sys.modules[mn]
        if added:
            sys.path.remove(nodes_dir)


M = _import_from(MATH_PY, "uls_latent_math_guard")


def run_registry():
    _need(M.TYPE_SPEC["sd"] == (4, 8, False, 0),
          "sd family must be the classic 4ch/8 class")
    _need(M.TYPE_SPEC["image"] == (16, 8, False, 0),
          "image class must stay 16ch/8 (SD3/Flux.1/Qwen)")
    _need("flux2" in M.TYPE_SPEC and M.CORE_DELEGATES.get("flux2")
          == "EmptyFlux2LatentImage",
          "flux2 must exist with its core delegate")
    # v872: this was an EXACT tail match (TYPE_ORDER[6:] == [sd, flux2]), which
    # made the file's own documented action -- "new families join at the end" --
    # fail the guard. Pinned as a PREFIX instead: every family that existed
    # before keeps its exact index, and anything new may only follow. That is
    # STRICTER for the existing eight (a rename or a reorder still trips it) and
    # correctly permits the sanctioned append.
    _need(M.TYPE_ORDER[:8] == ["image", "wan", "hunyuan", "mochi",
                               "ltxv", "cosmos", "sd", "flux2"],
          "TYPE_ORDER must be append-only (labels serialize by value)")
    _need(len(M.TYPE_ORDER) == len(set(M.TYPE_ORDER)),
          "TYPE_ORDER must not repeat a family")
    _need(all(k in M.TYPE_SPEC for k in M.TYPE_ORDER)
          and all(k in M.TYPE_LABELS for k in M.TYPE_ORDER),
          "every family in TYPE_ORDER needs a TYPE_SPEC row and a label")
    for k, lbl in (("image", "Image"), ("wan", "WAN video"),
                   ("hunyuan", "Hunyuan video"), ("mochi", "Mochi video"),
                   ("ltxv", "LTXV video"), ("cosmos", "Cosmos video")):
        _need(M.TYPE_LABELS[k] == lbl,
              "original label '%s' renamed -- saved graphs break" % k)
    _need(M.canonical_type("flux2") == "flux2"
          and M.canonical_type("Flux2 image") == "flux2"
          and M.canonical_type("sdxl") == "sd"
          and M.canonical_type("sd15") == "sd"
          and M.canonical_type("flux") == "image"
          and M.canonical_type("sd3") == "image"
          and M.canonical_type("qwen") == "image"
          and M.canonical_type("voellig-unbekannt") == "image",
          "alias resolution drifted")
    _need(M.plan_latent_shape("sd", 1024, 1024, 1, 2) == (2, 4, 128, 128)
          and M.plan_latent_shape("flux2", 512, 512, 1, 1) == (1, 128, 32, 32),
          "new-family shapes drifted")
    # v680: pinned against the FIELD measurement -- a 1440x1440 Flux2
    # decode carried a (1, 128, 90, 90) latent.
    _need(M.TYPE_SPEC["flux2"] == (128, 16, False, 0)
          and M.plan_latent_shape("flux2", 1440, 1440, 1, 1)
          == (1, 128, 90, 90),
          "flux2 geometry must match the measured field truth")


class _T:
    """Minimal tensor stand-in: shape + chainable to()."""
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.device = "cpu"
        self.dtype = "f32"

    def to(self, *a, **k):
        return self


def run_seam():
    # Drive generate() with stubbed torch-level pieces: fake noise maker
    # + fake comfy device + stub delegate. We import the node module and
    # patch its two soft deps.
    N = _load_node(NODE_PY, "ph_empty_latent_guard")

    class _Noise:
        @staticmethod
        def make_noise(kind, shape, seed, strength):
            return _T(shape)

    class _MM:
        @staticmethod
        def intermediate_device():
            return "cpu"

    class _Comfy:
        model_management = _MM()

    N.uls_noise = _Noise()
    N.comfy = _Comfy()
    node = N.ULSEmptyLatent() if hasattr(N, "ULSEmptyLatent") else None
    _need(node is not None, "node class name drifted")

    calls = []

    def core(name, w, h, b):
        calls.append((name, w, h, b))
        return {"samples": _T((b, 99, h // 16, w // 16))}

    # (a) delegate path: no VAE -> core wins, noise overlays ON its shape
    p, n, lat, _lw, _lh, _pw, _ph = node.generate("P", "N", "Flux2 image", 512, 512, 1, 1,
                              "fractal", 7, 1.0, _core=core)
    _need(calls == [("EmptyFlux2LatentImage", 512, 512, 1)],
          "flux2 without VAE must borrow the core empty node")
    _need(lat["samples"].shape == (1, 99, 32, 32),
          "noise must be laid over the DELEGATE'S geometry")
    # (b) spec fallback: delegate says 'host has no such node' -> loud row
    calls.clear()
    p, n, lat, _lw, _lh, _pw, _ph = node.generate("P", "N", "Flux2 image", 512, 512, 1, 1,
                              "zeros", 0, 1.0, _core=lambda *a: None)
    _need(lat["samples"].shape == (1, 128, 32, 32),
          "spec fallback must carry the MEASURED flux2 geometry when the "
          "host lacks the core class")

    # (c) VAE probe beats the delegate
    class _VAE:
        latent_channels = 32
        downscale_ratio = 16
    calls.clear()
    p, n, lat, _lw, _lh, _pw, _ph = node.generate("P", "N", "Flux2 image", 512, 512, 1, 1,
                              "zeros", 0, 1.0, vae=_VAE(), _core=core)
    _need(calls == [] and lat["samples"].shape == (1, 32, 32, 32),
          "a wired VAE must beat the core delegate (truth order)")


def _wounded(old, new, tag, path=MATH_PY):
    src = open(path, encoding="utf-8").read()
    if src.count(old) != 1:
        _fail("mutation %s: pattern not unique" % tag)
    out = os.path.join(HERE, "_wound_lat_%s.py" % tag)
    open(out, "w", encoding="utf-8").write(src.replace(old, new))
    return out


def run_mutations():
    m1 = _wounded('"sd":      (4,   8, False, 0),', '"sd":      (16,  8, False, 0),', "M1")
    m3 = _wounded('"image":   "Image",', '"image":   "Image latent",', "M3")
    caught = 0
    for path, probe in ((m1, "sdch"), (m3, "label")):
        try:
            W = _import_from(path, "uls_latent_math_wound_" + probe)
            if probe == "sdch":
                ok = W.TYPE_SPEC["sd"][0] == 4
            else:
                ok = W.TYPE_LABELS["image"] == "Image"
        except Exception:
            ok = False
        if not ok:
            caught += 1
        else:
            print("[test_v679_latent_families] mutation %s NOT caught" % probe)
        os.remove(path)
    # M2 wounds the NODE: delegate stops respecting the wired VAE
    m2 = _wounded("if delegate and vae_ch is None and vae_sdiv is None:",
                  "if delegate:", "M2", path=NODE_PY)
    try:
        W = _load_node(m2, "ph_empty_latent_wound_vae")

        class _Noise:
            @staticmethod
            def make_noise(kind, shape, seed, strength):
                return _T(shape)

        class _MM:
            @staticmethod
            def intermediate_device():
                return "cpu"

        class _Comfy:
            model_management = _MM()

        W.uls_noise = _Noise()
        W.comfy = _Comfy()

        class _VAE:
            latent_channels = 32
            downscale_ratio = 16
        _, _, lat = W.ULSEmptyLatent().generate(
            "P", "N", "Flux2 image", 512, 512, 1, 1, "zeros", 0, 1.0,
            vae=_VAE(), _core=lambda n, w, h, b:
                {"samples": _T((b, 99, h // 16, w // 16))})
        ok = lat["samples"].shape == (1, 32, 32, 32)
    except Exception:
        ok = False
    os.remove(m2)
    if not ok:
        caught += 1
    else:
        print("[test_v679_latent_families] mutation vae-priority NOT caught")
    _need(caught == 3, "mutation coverage incomplete")


def check_v688_outputs():
    """v688 -- the four size outputs. Appended (never re-ordered), and taken
    from the TENSOR, not from the widgets: the widgets are a request, the
    tensor is what survived the VAE probe / core delegate / spec fallback.
    Reporting the request would re-introduce the v679/v680 mismatch where a
    delegate snapped 1448 down to the /16 grid and the reported number lied."""
    py = open(os.path.join(ROOT, "nodes", "ph_empty_latent.py"),
              encoding="utf-8").read()
    if 'RETURN_NAMES = ("positive", "negative", "latent",' not in py:
        _fail("v688: the first three outputs must keep their slots")
    for name in ("latent_width", "latent_height", "width", "height"):
        if '"%s"' % name not in py:
            _fail("v688: missing output %s" % name)
    i_lat = py.index('"latent_width"')
    i_px = py.index('"width", "height")')
    if not (i_lat < i_px):
        _fail("v688: latent sizes come first -- that order is canon now")
    # driven: the helper must read the tensor, not echo the widgets
    ns = {}
    src = py[py.index("def _size_outputs("):py.index("class ULSEmptyLatent")]
    exec(src, ns)

    class _T:
        shape = (1, 128, 90, 90)

    out = ns["_size_outputs"](_T(), 1448, 1448)
    if out != (90, 90, 1448, 1448):
        _fail("v688: latent sizes must come from the tensor, got %r" % (out,))

    class _Bad:
        @property
        def shape(self):
            raise RuntimeError("no shape")

    if ns["_size_outputs"](_Bad(), 64, 64) != (0, 0, 64, 64):
        _fail("v688: a shapeless latent must not take the node down")
    js = open(os.path.join(ROOT, "web", "js", "ph_empty_latent.js"),
              encoding="utf-8").read()
    if '"control_after_generate"' not in js.split("LEGACY_NOISE")[1][:400]:
        _fail("v688: control_after_generate must be hidden with the rest")


def main():
    run_registry()
    run_seam()
    run_mutations()
    print("[test_v679_latent_families] PASS -- family registry pinned "
          "(sd=4ch, append-only labels, alias table), new shapes "
          "analytic, truth order VAE > core delegate > spec DRIVEN "
          "through the seam, 3/3 mutations caught")


if __name__ == "__main__":
    main()