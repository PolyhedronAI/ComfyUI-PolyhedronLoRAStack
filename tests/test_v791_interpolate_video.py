"""Guard -- Polyhedron Interpolate video pin (v791).

Field wound behind this cut: Interpolate hung its source_fps on the fps
output of a Save that sat behind a DE-SELECTED ULSAnySwitchInv branch;
the unselected output emits an ExecutionBlocker, one blocked input
disables the whole node -- silently. The fix at the root: a video pin,
so the frame rate travels INSIDE the wired object and no fps wire needs
to exist at all.

DRIVEN, not read (everything through the n=1 passthrough, so no RIFE
engine is ever loaded):
  * canon: NO required pins any more; optional pins exactly
    [frames, video] in this order (frames stayed pin #0, saved graphs
    keep their slot); widget canon identical to the baseline row;
    signature order pinned including the trailing video (v788 lesson).
  * _frames_from_video: frames + float(rate) out of a components
    object (Fraction included), rate<=0 collapses to 0.0, an empty or
    unreadable video is a LOUD error.
  * seam: neither pin wired -> loud; BOTH wired -> loud (a silent
    precedence would hide which source ran); video-only substitutes
    the video's own rate for source_fps and SAYS so in interp_info;
    a rate-less video falls back to the widget and says that too;
    the classic frames-only path is byte-for-byte unaffected (no note).

MUTATIONS (wound in a COPY, landing asserted via count==1, catch proven
by execution): M1 both-wired check removed, M2 rate substitution
dropped, M3 empty-video check weakened, M4 passthrough loses the note.
"""

import importlib.util
import io
import os
import shutil
import sys
import tempfile
import types
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_SEQ = [0]


def _fail(msg):
    print("[test_v791_interpolate_video] FAIL --", msg)
    sys.exit(1)


def _need(ok, msg):
    if not ok:
        _fail(msg)


def _read(*parts):
    with io.open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def _pkg(src=None):
    """Import ph_interpolate as a real package member (v786 pattern) so
    its relative ph_runclock import works; comfy stubs registered."""
    if src is None:
        src = _read("nodes", "ph_interpolate.py")
    _SEQ[0] += 1
    name = "phv791_%d" % _SEQ[0]
    tmp = tempfile.mkdtemp(prefix="pls791_")
    sys.modules.setdefault("folder_paths", types.ModuleType("folder_paths"))
    comfy = sys.modules.setdefault("comfy", types.ModuleType("comfy"))
    mm = types.ModuleType("comfy.model_management")
    mm.get_torch_device = lambda: "cpu"
    comfy.model_management = mm
    sys.modules["comfy.model_management"] = mm
    pkg = types.ModuleType(name)
    pkg.__path__ = [tmp]
    sys.modules[name] = pkg
    shutil.copy(os.path.join(ROOT, "nodes", "ph_runclock.py"),
                os.path.join(tmp, "ph_runclock.py"))
    ip_path = os.path.join(tmp, "ph_interpolate.py")
    with io.open(ip_path, "w", encoding="utf-8") as fh:
        fh.write(src)
    for mod_name, path in (
            ("ph_runclock", os.path.join(tmp, "ph_runclock.py")),
            ("ph_interpolate", ip_path)):
        full = name + "." + mod_name
        spec = importlib.util.spec_from_file_location(full, path)
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = name
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name + ".ph_interpolate"]


class _Vid:
    def __init__(self, images, rate):
        self._i, self._r = images, rate

    def get_components(self):
        return types.SimpleNamespace(images=self._i, frame_rate=self._r,
                                     audio=None)


class _Broken:
    def get_components(self):
        raise OSError("container unreadable")


def _one_frame():
    import torch
    return torch.rand(1, 24, 32, 3)


def _call(mod, **over):
    kw = dict(source_fps=16.0)
    kw.update(over)
    return mod.ULSInterpolate().interpolate(**kw)


def run_canon(mod):
    import inspect
    it = mod.ULSInterpolate.INPUT_TYPES()
    req_pins = [k for k, v in it["required"].items()
                if isinstance(v, tuple) and v
                and v[0] in ("IMAGE", "VIDEO", "MASK", "LATENT")]
    _need(req_pins == [], "required must carry no pins, has %r" % req_pins)
    opt = list(it.get("optional", {}).keys())
    _need(opt == ["frames", "video"],
          "optional pin order drifted: %r (frames must stay pin #0)" % opt)
    _need(it["optional"]["frames"][0] == "IMAGE"
          and it["optional"]["video"][0] == "VIDEO",
          "pin types drifted")
    widgets = list(it["required"].keys())
    row = None
    import glob
    newest = sorted(glob.glob(os.path.join(
        ROOT, "WIDGET_ORDER_baseline_*.txt")))[-1]
    for line in _read(os.path.basename(newest)).splitlines():
        if line.startswith("ULSInterpolate\t"):
            row = line.split("\t", 1)[1]
    _need(row == ",".join(widgets),
          "widget canon differs from baseline: %r vs %r"
          % (row, ",".join(widgets)))
    params = list(inspect.signature(
        mod.ULSInterpolate.interpolate).parameters)
    _need(params[:3] == ["self", "frames", "ckpt_name"]
          and params[-1] == "video",
          "signature order drifted (v788 lesson): %r" % params)
    print("[test_v791_interpolate_video] canon OK")


def run_video_helper(mod):
    frames = _one_frame()
    got, fr = mod._frames_from_video(_Vid(frames, Fraction(20, 1)))
    _need(got is frames and fr == 20.0,
          "helper lost frames or Fraction rate: %r" % fr)
    _, fr0 = mod._frames_from_video(_Vid(frames, 0))
    _need(fr0 == 0.0, "rate<=0 must collapse to 0.0")
    _, frn = mod._frames_from_video(_Vid(frames, None))
    _need(frn == 0.0, "unreadable rate must collapse to 0.0")
    import torch
    for bad in (_Vid(torch.zeros(0, 8, 8, 3), 20), _Vid(None, 20),
                _Broken()):
        try:
            mod._frames_from_video(bad)
            _fail("helper accepted an empty/unreadable video silently")
        except RuntimeError:
            pass
    print("[test_v791_interpolate_video] video helper OK")


def run_seam(mod):
    frames = _one_frame()
    for bad in (dict(), dict(frames=frames, video=_Vid(frames, 20))):
        try:
            _call(mod, **bad)
            _fail("interpolate accepted %s silently"
                  % ("neither pin" if not bad else "both pins"))
        except RuntimeError:
            pass
    # RE-GROUNDED IN v887 (declared). These three calls used to unpack EXACTLY
    # four values. That pinned the ARITY of the return, which was never this
    # guard's promise -- its subject is the video-INPUT seam (which pin wins,
    # whose rate is used, and that the node says so). v887 APPENDS a `video`
    # output, a lawful act, and the fixed-arity unpack called it a violation.
    # Read by INDEX instead, and pin the promise that actually protects a
    # saved workflow: the historic outputs keep their slots and anything new
    # lands behind them (links store an origin SLOT INDEX -- the output-side
    # twin of #577).
    hist = ("frames", "frame_count", "fps", "interp_info")
    _need(tuple(mod.ULSInterpolate.RETURN_NAMES)[:len(hist)] == hist,
          "the historic outputs must keep their slots and order; a new output "
          "is APPENDED, never inserted (got %r)"
          % (mod.ULSInterpolate.RETURN_NAMES,))
    _need(len(mod.ULSInterpolate.RETURN_TYPES)
          == len(mod.ULSInterpolate.RETURN_NAMES),
          "RETURN_TYPES and RETURN_NAMES drifted in length")

    res = _call(mod, video=_Vid(frames, Fraction(20, 1)))
    n, fps, info = res[1], res[2], res[3]
    _need(n == 1 and fps == 20.0,
          "video rate not substituted: fps %r" % fps)
    _need("read from the wired video" in info,
          "substitution must be said in interp_info: %s" % info)
    res = _call(mod, video=_Vid(frames, 0))
    fps, info = res[2], res[3]
    _need(fps == 16.0 and "names no frame rate" in info,
          "rate-less fallback broken: fps %r / %s" % (fps, info))
    res = _call(mod, frames=frames)
    fps, info = res[2], res[3]
    _need(fps == 16.0 and "wired video" not in info,
          "classic frames path changed: %s" % info)
    print("[test_v791_interpolate_video] seam OK")


def _expect_catch(tag, src, needle, replacement, runner):
    _need(src.count(needle) == 1,
          "%s did not LAND (needle count %d)" % (tag, src.count(needle)))
    mod = _pkg(src.replace(needle, replacement))
    try:
        runner(mod)
    except SystemExit:
        print("[test_v791_interpolate_video] %s caught" % tag)
        return
    _fail("%s survived -- the guard cannot see this wound" % tag)


def run_mutations():
    src = _read("nodes", "ph_interpolate.py")
    _expect_catch("M1 both-wired check removed", src,
                  "if frames is not None and video is not None:",
                  "if False:", run_seam)
    _expect_catch("M2 rate substitution dropped", src,
                  "source_fps = float(vfps)",
                  "source_fps = float(source_fps)", run_seam)
    _expect_catch("M3 empty-video check weakened", src,
                  "if imgs is None or int(imgs.shape[0]) == 0:",
                  "if imgs is None and False:", run_video_helper)
    _expect_catch("M4 passthrough loses the note", src,
                  "            if fps_note:\n"
                  "                info += \" | \" + fps_note",
                  "            pass", run_seam)


def main():
    mod = _pkg()
    run_canon(mod)
    run_video_helper(mod)
    run_seam(mod)
    run_mutations()
    print("[test_v791_interpolate_video] PASS")


if __name__ == "__main__":
    main()
