"""Guard v885 -- the previews: input door, node progress slot, honest silence.

WHAT THIS PINS (driven, not read, wherever a thing can be RUN):

  P1  THE ONE-SHOT CLOCK PREVIEW (_RunClock.offer_preview / push), EXECUTED
      against a recording ProgressBar. Measured at Core
      (comfy/utils.py::ProgressBar.update_absolute): a non-None preview
      BYPASSES Core's throttle, so attaching one to every push floods the
      socket. The promise is therefore: an offered image rides EXACTLY ONE
      push and the slot is empty again afterwards. Also: no offer -> None in
      the slot (plain progress stays throttled by Core), and the bar's value
      and total are unchanged by the whole mechanism.

  P2  THE INPUT DOOR (_emit_input_preview), EXECUTED with a stubbed
      PromptServer: it fires BEFORE any pass, tags stage="input", needs no
      model / no VAE / no latent_rgb_factors, carries the source geometry,
      hands the SAME frame to the clock, and DISARMS with one spoken line on
      failure (never raising into the render).

  P3  "Off" STILL MEANS OFF, and says so. The v550 promise (zero overhead,
      the probe is never built) is not quietly traded for a nicer picture:
      with process_preview="Off" neither probe is built AND no input frame is
      emitted -- but the run states it once, because a silent Off is
      indistinguishable from a broken preview.

  P4  THE DEFAULT MOVED, THE HEAL DID NOT. INPUT_TYPES defaults to
      "latent2rgb" (a new node shows its work) while _healPreV550 keeps
      filling "Off" (a pre-v550 save ran with no process view at all and must
      stay bit-for-bit what it was). The v851 lesson: on an appended widget
      the DEFAULT is the most consequential number of the cut -- and here the
      two defaults are deliberately DIFFERENT, so both are pinned.

  P5  THE SAMPLER STOPPED SWALLOWING. The preview failure in the sampling
      callback speaks once per run instead of `except: pass`, and a joint
      (video+audio) latent in a TAE mode is told WHY it falls back -- pinned
      by EXECUTING the previewer against a joint latent, not by grepping.

  P6  MUTATIONS: each promise above must go red when its code is holed.

Script-style: exit 0 = pass.
"""
import base64
import importlib.util
import io
import os
import re
import sys
import types

NAME = "v885"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PU = os.path.join(ROOT, "nodes", "ph_power_upscale.py")
RC = os.path.join(ROOT, "nodes", "ph_runclock.py")
SA = os.path.join(ROOT, "nodes", "uls_sampler.py")
JS = os.path.join(ROOT, "web", "js", "ph_power_upscale.js")

_checks = 0


def _fail(msg):
    print("[%s] FAIL -- %s" % (NAME, msg))
    sys.exit(1)


def _need(cond, msg):
    global _checks
    _checks += 1
    if not cond:
        _fail(msg)


def _read(path):
    return open(path, encoding="utf-8").read()


# ---------------------------------------------------------------------------
# stubs: enough comfy to import the two modules, nothing more
# ---------------------------------------------------------------------------
import torch  # noqa: E402  (after the constants, before the loads)
from PIL import Image  # noqa: E402

fp = types.ModuleType("folder_paths")
fp.models_dir = "/tmp"
fp.get_filename_list = lambda f: []
fp.get_full_path = lambda f, n: None
fp.get_folder_paths = lambda f: ["/tmp"]
fp.get_temp_directory = lambda: "/tmp"
sys.modules.setdefault("folder_paths", fp)

for _n in ("comfy", "comfy.samplers", "comfy.sample", "comfy.utils",
           "comfy.model_management", "comfy.cli_args"):
    sys.modules.setdefault(_n, types.ModuleType(_n))


class _LPM:
    NoPreviews = "none"


sys.modules["comfy.cli_args"].args = types.SimpleNamespace(preview_method="l2rgb")
sys.modules["comfy.cli_args"].LatentPreviewMethod = _LPM
sys.modules["comfy"].samplers = sys.modules["comfy.samplers"]
sys.modules["comfy"].sample = sys.modules["comfy.sample"]
sys.modules["comfy"].utils = sys.modules["comfy.utils"]
sys.modules["comfy.samplers"].KSampler = types.SimpleNamespace(
    SAMPLERS=["euler"], SCHEDULERS=["simple"])


def _load(path, modname, src=None):
    pkg = types.ModuleType("plspack")
    pkg.__path__ = [os.path.join(ROOT, "nodes")]
    sys.modules["plspack"] = pkg
    full = "plspack." + modname
    if src is None:
        spec = importlib.util.spec_from_file_location(full, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[full] = m
        spec.loader.exec_module(m)
        return m
    m = types.ModuleType(full + "_mut")
    m.__package__ = "plspack"
    m.__file__ = path
    exec(compile(src, path, "exec"), m.__dict__)
    return m


class _Bar:
    """Records what the clock actually hands Core."""

    def __init__(self):
        self.calls = []

    def update_absolute(self, value, total=None, preview=None):
        self.calls.append((value, total, preview))


class _Server:
    """Stand-in PromptServer: records the events a probe sends."""

    def __init__(self):
        self.sent = []
        self.last_node_id = 7

    def send_sync(self, event, payload):
        self.sent.append((event, payload))


def _install_server(srv):
    m = types.ModuleType("server")
    m.PromptServer = types.SimpleNamespace(instance=srv)
    sys.modules["server"] = m


# ---------------------------------------------------------------------------
# P1 -- the one-shot preview, EXECUTED
# ---------------------------------------------------------------------------
def p1(rc_src=None):
    rc = _load(RC, "ph_runclock", rc_src)
    bar = _Bar()
    clock = rc._RunClock(bar)
    clock.post("pix:final", 4, 1000.0)

    clock.measure("pix:final", 0.5)          # no offer -> empty slot
    _need(bar.calls and bar.calls[-1][2] is None,
          "a push with nothing offered must carry NO preview - otherwise every "
          "progress update bypasses Core's throttle and floods the socket")

    img = ("JPEG", Image.new("RGB", (4, 4)), 512)
    clock.offer_preview(img)
    clock.measure("pix:final", 0.5)
    _need(bar.calls[-1][2] is img,
          "an offered preview must reach the very next push (the node's own "
          "progress slot is the whole point of the offer)")

    clock.measure("pix:final", 0.5)
    _need(bar.calls[-1][2] is None,
          "the preview slot must be ONE-SHOT: a second push must not re-send "
          "the same image, or the offer's rate stops being the producer's")

    vals = [c[0] for c in bar.calls]
    _need(all(isinstance(v, int) for v in vals) and vals == sorted(vals),
          "the bar's own value must be untouched by the preview channel")


# ---------------------------------------------------------------------------
# P2 -- the input door, EXECUTED
# ---------------------------------------------------------------------------
def _pu(src=None):
    return _load(PU, "ph_power_upscale", src)


def p2(pu=None):
    pu = pu or _pu()
    srv = _Server()
    _install_server(srv)
    rc = _load(RC, "ph_runclock")
    bar = _Bar()
    clock = rc._RunClock(bar)

    frames = torch.rand(9, 768, 1344, 3)      # [B, H, W, C], already RGB
    pu._emit_input_preview(frames, clock=clock)

    _need(len(srv.sent) == 1, "the input door must send exactly one frame")
    ev, d = srv.sent[0]
    _need(ev == pu._PROBE_EVENT,
          "the input frame must ride the SAME event as the other two doors, "
          "or the pane cannot receive it")
    _need(d["stage"] == "input",
          "the input frame must tag its own stage - the pane must never guess")
    _need(d["canvas"] == [1344, 768] and d["rect"] == [0, 0, 1344, 768],
          "the input frame must carry the SOURCE geometry (it is the whole "
          "canvas by construction)")
    _need(d["steps"] == 9, "the input frame must state the source's length")
    _need(d["jpeg"] and base64.b64decode(d["jpeg"])[:2] == b"\xff\xd8",
          "the input frame must carry a real JPEG")
    _need(bar.calls == [] and clock._preview is not None,
          "the input frame must be OFFERED to the clock (consumed by the next "
          "push), not pushed itself - the clock owns the bar")

    # no model, no vae, no latent maths anywhere in this door. Measured on the
    # CODE, not the prose: the docstring is allowed - is REQUIRED - to name
    # what the door deliberately does not do, and a whole-text grep would then
    # fail on its own explanation. (The v565 guard learned this the hard way;
    # this one inherits the lesson instead of repeating it.)
    # v1127: the docstring is cut out by its line span from the syntax tree -- Python 3.13
    # dedents __doc__, so a text replace of __doc__ no longer finds it in the source.
    body = pu.__dict__["_emit_input_preview"].__doc__ or ""
    src = _read(PU)
    door = src[src.index("def _emit_input_preview"):src.index("def _make_tile_probe")]
    import ast as _ast
    fn = next(n for n in _ast.walk(_ast.parse(src)) if isinstance(n, _ast.FunctionDef)
              and n.name == "_emit_input_preview")
    ds = fn.body[0] if (fn.body and isinstance(fn.body[0], _ast.Expr)
                        and isinstance(getattr(fn.body[0], "value", None), _ast.Constant)
                        and isinstance(fn.body[0].value.value, str)) else None
    lines = src.splitlines()
    doc_text = "\n".join(lines[ds.lineno - 1:ds.end_lineno]) if ds is not None else ""
    code = door.replace(doc_text, "") if doc_text else door   # docstring out
    code = "\n".join(ln for ln in code.splitlines()
                     if not ln.strip().startswith("#"))  # comments out
    _need("einsum" not in code and "latent_rgb_factors" not in code
          and not re.search(r"\bmodel\b", code) and not re.search(r"\bvae\b", code),
          "the input door must need no model and no latent maths - the frames "
          "are finished RGB (the v565 law, one door further)")
    _need("input view unavailable" in src,
          "the input door must state its own failure (v552) - a silent door is "
          "the one place a broken chain can hide")
    _need(isinstance(body, str) and body.strip(),
          "the input door must document WHY it exists")

    # it disarms instead of raising: a frame it cannot read must not end a run
    srv2 = _Server()
    _install_server(srv2)
    pu._emit_input_preview("not a tensor", clock=None)   # must not raise
    _need(srv2.sent == [],
          "a broken input frame must send nothing - and, above all, not raise")


# ---------------------------------------------------------------------------
# P3 / P4 -- Off is still Off and says so; default moved, heal did not
# ---------------------------------------------------------------------------
def p3p4(src=None):
    pu_src = src if src is not None else _read(PU)
    js = _read(JS)

    _need('str(process_preview) != "Off"' in pu_src,
          "Off must still mean the probes are NEVER built (the v550 promise)")
    up = pu_src[pu_src.index("def upscale("):]
    off_blk = up[up.index('if str(process_preview) == "Off":'):]
    off_blk = off_blk[:off_blk.index("cur = frames")]
    _need("process view is Off" in off_blk,
          "an Off run must SAY it once - a silent Off is indistinguishable "
          "from a broken preview, and that is exactly what cost a field run")
    _need("_emit_input_preview" in off_blk.split("else:", 1)[1],
          "the input frame belongs on the NOT-Off side of that branch - Off "
          "means off, including this door")

    req = pu_src[pu_src.index('"required"'):pu_src.index('"optional"')]
    blk = req[req.index('"process_preview"'):]
    blk = blk[:blk.index('"mute_staging_logs"')]
    m = re.search(r'"default":\s*"([^"]+)"', blk)
    _need(m and m.group(1) == "latent2rgb",
          "the process_preview DEFAULT must be latent2rgb (v885) - a NEW node "
          "shows its work")
    _need('"Off"' in blk and '"vae (sharp)"' in blk,
          "all three process_preview values must survive the default change")

    sig = re.search(r"process_preview=\"([^\"]+)\"", pu_src)
    _need(sig and sig.group(1) == "latent2rgb",
          "the function's own keyword default must MATCH the widget default - "
          "two places holding two answers is the drift this house punishes")

    heal = re.search(r"function _healPreV550\(wv\) \{[\s\S]*?\n\}", js)
    _need(heal and '"Off"' in heal.group(0),
          "the HEAL must keep filling Off: a save from before v550 ran with no "
          "process view at all and must stay bit-for-bit what it was. The "
          "default moved; history did not.")


# ---------------------------------------------------------------------------
# P5 -- the sampler stopped swallowing, EXECUTED
# ---------------------------------------------------------------------------
class _Nested:
    def __init__(self, tensors):
        self.tensors = list(tensors)
        self.is_nested = True

    def unbind(self):
        return self.tensors


class _H3:
    """MiniMax H3's latent format as MEASURED at Core: the AV format inherits
    the VIDEO factors (24 rows); its latent_channels=32 is only the max over
    the two streams and must not be believed."""
    latent_channels = 32
    latent_rgb_factors = [[0.1, 0.1, 0.1]] * 24
    latent_rgb_factors_bias = [0.05, -0.02, -0.07]


def p5(src=None):
    sa = _load(SA, "uls_sampler", src)
    joint = _Nested([torch.randn(1, 24, 3, 32, 48), torch.randn(1, 32, 2, 60)])

    pv = sa._AnimatedVideoPreviewer(_H3, preview_mode=sa._AnimatedVideoPreviewer.MODE_TAE_STD)
    _need(pv.use_tae, "the fixture must start in a TAE mode or it proves nothing")
    buf, old = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        frames = pv.frames_b64(joint)
    finally:
        sys.stdout = old
    said = buf.getvalue()
    _need(frames, "a joint latent must still PRODUCE a preview - the video "
                  "half is perfectly drawable")
    _need("JOINT" in said and "latent2rgb" in said,
          "a TAE mode on a joint latent must name the REASON it falls back, "
          "before the attempt - not blame the decoder afterwards")
    _need(not pv.use_tae, "the fallback must stick for the rest of the run")

    buf2, old = io.StringIO(), sys.stdout
    sys.stdout = buf2
    try:
        pv.frames_b64(joint)
    finally:
        sys.stdout = old
    _need("JOINT" not in buf2.getvalue(),
          "the notice must fire ONCE per run, not once per step")

    # And the callback's swallowed failure now SPEAKS -- EXECUTED, not grepped.
    # (The first version of this check read the file from disk while the
    # mutation lived in a string: it was blind to its own mutation and stayed
    # green. Caught here, before it ever shipped. A text pin inside a guard
    # that mutates SOURCE must read the mutated source, or better: stop being
    # a text pin.)
    class _Boom:
        ok = True
        mode = "x"

        def frames_b64(self, x0):
            raise RuntimeError("decoder exploded")

    sys.modules["comfy.utils"].ProgressBar = lambda total: types.SimpleNamespace(
        update_absolute=lambda *a, **k: None)
    cb = sa._make_preview_callback(None, 1, node_id=3, previewer=_Boom())
    buf3, old_out = io.StringIO(), sys.stdout
    sys.stdout = buf3
    try:
        cb(0, torch.randn(1, 4, 8, 8), None, 1)
        cb(0, torch.randn(1, 4, 8, 8), None, 1)   # second failure, same run
    finally:
        sys.stdout = old_out
    spoke = buf3.getvalue()
    _need("live preview failed" in spoke,
          "a failing live preview must say so - `except: pass` left an empty "
          "node and not one character anywhere")
    _need(spoke.count("live preview failed") == 1,
          "and it must say so ONCE per run, not once per step")


# ---------------------------------------------------------------------------
# P6 -- mutations: hole each promise, demand red
# ---------------------------------------------------------------------------
def _mut(fn, path, old, new, label, loader_arg=True):
    src = _read(path)
    if old not in src:
        _fail("MUTATION ANCHOR MISSING (%s): %r" % (label, old[:70]))
    mutated = src.replace(old, new, 1)
    try:
        if loader_arg:
            fn(mutated)
        else:
            fn(mutated)
    except SystemExit as e:
        if e.code:
            return True
        _fail("mutation '%s' exited 0 - the guard did not catch it" % label)
    except Exception:
        return True
    _fail("mutation '%s' left the guard GREEN - the promise is not pinned"
          % label)


def p6():
    n = 0
    # M1: the preview is not cleared -> it rides every push (throttle defeated)
    n += bool(_mut(p1, RC,
                   "img, self._preview = self._preview, None",
                   "img = self._preview",
                   "clock preview never cleared"))
    # M2: the input door forgets its stage tag
    n += bool(_mut(lambda s: p2(_pu(s)), PU,
                   '"node": str(node_id), "stage": "input",',
                   '"node": str(node_id), "stage": "pixel",',
                   "input frame mislabels its stage"))
    # M3: the input door stops offering to the clock
    n += bool(_mut(lambda s: p2(_pu(s)), PU,
                   '        _offer(clock, pil)\n        from server import PromptServer\n'
                   '        PromptServer.instance.send_sync(_PROBE_EVENT, {\n'
                   '            "node": str(node_id), "stage": "input",',
                   '        from server import PromptServer\n'
                   '        PromptServer.instance.send_sync(_PROBE_EVENT, {\n'
                   '            "node": str(node_id), "stage": "input",',
                   "input frame never reaches the node's own slot"))
    # M4: the default is moved back -> a new node is mute again
    n += bool(_mut(p3p4, PU,
                   '"process_preview": (["Off", "latent2rgb", "vae (sharp)"],\n'
                   '                                    {"default": "latent2rgb",',
                   '"process_preview": (["Off", "latent2rgb", "vae (sharp)"],\n'
                   '                                    {"default": "Off",',
                   "default rolled back to Off"))
    # M5: Off goes silent again
    n += bool(_mut(p3p4, PU,
                   '            print("[PLS] Power Upscale: process view is Off',
                   '            print("[PLS] Power Upscale: quiet run',
                   "Off stops saying it is Off"))
    # M6: the joint notice is removed -> the TAE path blames the decoder again
    n += bool(_mut(p5, SA,
                   "if ragged and self.use_tae and not self._said_joint:",
                   "if False and ragged and self.use_tae and not self._said_joint:",
                   "joint latent gets no reason for its fallback"))
    # M7: the sampler swallows again
    n += bool(_mut(p5, SA,
                   '                print(f"[PLS SAMPLER] live preview failed "',
                   '                print(f"[PLS SAMPLER] quiet "',
                   "sampler preview failure goes silent"))
    return n


def main():
    p1()
    pu = _pu()
    p2(pu)
    p3p4()
    p5()
    muts = p6()
    print("PASS: %s -- %d promises pinned, %d mutations caught "
          "(one-shot clock preview, input door, Off stays Off and says so, "
          "default moved while the heal did not, sampler speaks)"
          % (NAME, _checks, muts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
