"""Guard v887 -- Interpolate carries the sound; Save stops overriding in silence.

THE BUG THIS EXISTS FOR: the Interpolate node had no audio path at all. With
no route THROUGH the node, the sound had to travel AROUND it, and every route
around it lands in a trap. This pins the route through, and the one trap that
was silent.

WHAT THIS PINS (arithmetic and wiring are EXECUTED, not read):

  A1  THE SYNC LAW, computed against an independent restatement of the
      specification: source frame k lands at output index k*multiplier, and at
      out_fps = src*mult its timestamp is exactly k/src -- its ORIGINAL
      second. Driven over a sweep of frame counts and multipliers, so the
      claim "interpolation cannot drift" is a measurement, not a slogan.

  A2  THE TAIL IS THE ONLY DIFFERENCE, and it is (1 - 1/mult)/src seconds --
      because the node emits mult*(n-1)+1 frames, not mult*n. _sync_note must
      report exactly that number, checked against the arithmetic.

  A3  THE VIDEO OUTPUT CARRIES THREE THINGS: the interpolated frames, the NEW
      rate, and the ORIGINAL audio -- byte-identical object, not a copy.
      Driven through interpolate() on the passthrough path (no model needed)
      with a stand-in VIDEO API.

  A4  A BARE FRAME BATCH CARRIES NO SOUND, and says so rather than inventing
      one. Also: no VIDEO API -> the video output is None and the FRAMES
      output is untouched. A convenience may never cost a result.

  A5  APPEND-ONLY ON THE OUTPUT SIDE. `video` sits LAST; the four historic
      outputs keep their slot indices. Links store an origin slot INDEX, so
      an insert re-numbers every wire in every saved workflow -- the
      output-side twin of #577.

  A6  THE SAVE STOPS OVERRIDING IN SILENCE. With a VIDEO and an IMAGE both
      wired the video wins (unchanged behaviour -- this cut does not move
      anyone's output), but it now SAYS which one won and what was dropped.
      Pinned by running _save_video's resolution branch.

  A7  MUTATIONS: hole each promise, demand red.

Script-style: exit 0 = pass.
"""
import importlib.util
import io
import os
import sys
import types
from fractions import Fraction

import torch

NAME = "v887"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INT = os.path.join(ROOT, "nodes", "ph_interpolate.py")
SAVE = os.path.join(ROOT, "nodes", "ph_save.py")

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
# stand-ins
# ---------------------------------------------------------------------------
fp = types.ModuleType("folder_paths")
fp.models_dir = "/tmp"
fp.get_filename_list = lambda f: []
fp.get_full_path = lambda f, n: None
fp.get_folder_paths = lambda f: ["/tmp"]
fp.get_temp_directory = lambda: "/tmp"
fp.get_output_directory = lambda: "/tmp"
sys.modules.setdefault("folder_paths", fp)

for _n in ("comfy", "comfy.utils", "comfy.model_management", "comfy.sd",
           "comfy.samplers", "comfy.sample", "comfy.cli_args"):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules["comfy.model_management"].get_torch_device = lambda: torch.device("cpu")
sys.modules["comfy"].utils = sys.modules["comfy.utils"]
sys.modules["comfy"].model_management = sys.modules["comfy.model_management"]


class _Comps:
    def __init__(self, images, frame_rate, audio):
        self.images = images
        self.frame_rate = frame_rate
        self.audio = audio


class _Vid:
    """Stand-in for a ComfyUI VIDEO object."""

    def __init__(self, images, frame_rate, audio=None):
        self._c = _Comps(images, frame_rate, audio)

    def get_components(self):
        return self._c


_BUILT = []


class _FakeVideoFromComponents:
    def __init__(self, comps):
        self.comps = comps
        _BUILT.append(comps)

    def get_components(self):
        return self.comps


def _install_video_api(mod, present=True):
    """Point the module's VIDEO API at recording stand-ins (or remove it)."""
    mod._HAS_VIDEO_API = present
    if present:
        mod.VideoFromComponents = _FakeVideoFromComponents
        mod.VideoComponents = _Comps
        mod.Fraction = Fraction
    else:
        mod.VideoFromComponents = mod.VideoComponents = None


def _load(path, name, src=None):
    pkg = types.ModuleType("plspack")
    pkg.__path__ = [os.path.join(ROOT, "nodes")]
    sys.modules["plspack"] = pkg
    full = "plspack." + name
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


def _interp(src=None):
    return _load(INT, "ph_interpolate", src)


def _frame(n=1):
    return torch.rand(n, 16, 16, 3)


# ---------------------------------------------------------------------------
# A1 / A2 -- the arithmetic, EXECUTED against a restatement of the spec
# ---------------------------------------------------------------------------
def a1a2(src=None):
    mod = _interp(src)
    for n in (2, 5, 17, 81):
        for mult in (2, 3, 4):
            out_count = mult * (n - 1) + 1
            _need(mod._timeline(n, mult)[1] == out_count,
                  "the timeline law changed: %d frames x%d" % (n, mult))
            src_fps, out_fps = 16.0, 16.0 * mult
            # A1: every source frame keeps its own second -- computed from the
            # SPEC (slot = k*mult), not from the node's own words.
            for k in range(n):
                slot = k * mult
                _need(abs(slot / out_fps - k / src_fps) < 1e-12,
                      "source frame %d drifts: %.9f vs %.9f"
                      % (k, slot / out_fps, k / src_fps))
            # A2: the whole difference is the tail
            want = (1.0 - 1.0 / mult) / src_fps
            got = n / src_fps - out_count / out_fps
            _need(abs(got - want) < 1e-9,
                  "the tail is not (1-1/mult)/src: %.9f vs %.9f" % (got, want))
            note = mod._sync_note(n, src_fps, out_count, out_fps)
            _need("duration" in note and "ms" in note,
                  "the duration arithmetic must be reported: %r" % note)
            ms = -1000.0 * want
            _need(("%+.0f ms" % ms) in note,
                  "_sync_note must report the computed tail %+.0f ms (got %r)"
                  % (ms, note))
    _need(mod._sync_note(5, 0, 5, 16) == "",
          "an unusable rate must produce no claim at all")
    return mod


# ---------------------------------------------------------------------------
# A3 / A4 / A5 -- the video output, EXECUTED through interpolate()
# ---------------------------------------------------------------------------
def a3a4a5(src=None):
    mod = _interp(src)
    _install_video_api(mod)

    names = tuple(mod.ULSInterpolate.RETURN_NAMES)
    _need(names[:4] == ("frames", "frame_count", "fps", "interp_info"),
          "the four historic outputs must keep their slots - a link stores an "
          "origin slot INDEX (got %r)" % (names,))
    # v890 RE-GROUNDING, declared: this line pinned "video is LAST", which was
    # the snapshot of 26.08., not the promise. The promise -- stated by this
    # very guard one line up -- is that links store an origin slot INDEX, so
    # outputs are APPEND-ONLY: the historic four keep 0..3, video keeps slot 4
    # forever, and anything new goes BEHIND it. Pinning "last" would break at
    # the first legitimate append (v890's audio) -- the test_v842/S6 latency,
    # one generation later. Pinning the SLOT is the stronger form.
    _need(len(names) > 4 and names[4] == "video"
          and mod.ULSInterpolate.RETURN_TYPES[4] == "VIDEO",
          "the video output must keep SLOT 4 - v887 workflows store that index")
    _need(len(mod.ULSInterpolate.RETURN_TYPES) == len(names),
          "RETURN_TYPES / RETURN_NAMES drifted in length")

    # A3 -- one frame in: the passthrough path, no model, but the same return
    audio = {"waveform": torch.zeros(1, 2, 800), "sample_rate": 16000}
    _BUILT.clear()
    imgs = _frame(1)
    res = mod.ULSInterpolate().interpolate(
        video=_Vid(imgs, Fraction(20, 1), audio))
    # v890 re-grounding: == 5 pinned the ARITY -- the very wound this guard's
    # sibling test_v791 was cured of in v887 itself. Reading slot 4 by index is
    # the promise; the tuple may grow behind it (it did: audio, slot 5).
    _need(len(res) >= 5, "the return must carry the video slot")
    vid = res[4]
    _need(vid is not None, "a wired video must produce a video output")
    comps = vid.get_components()
    _need(comps.audio is audio,
          "the ORIGINAL audio object must ride through untouched - a copy is "
          "a chance to lose it")
    _need(float(comps.frame_rate) == 20.0,
          "the video output must carry the rate the frames are meant for")
    _need(comps.images is res[0],
          "the video must carry the frames this node actually returned")

    # A3b -- THE MAIN PATH, really run. The passthrough above cannot separate
    # source_fps from out_fps (they are equal there), so a call site handing
    # the OLD rate to the video builder would sail straight through it. That
    # blind spot showed up in this guard's own mutation round, which is what
    # mutations are for. Here a stand-in RIFE runs a real 3-frame x2 job on
    # the CPU: source 20, output 40, and the two can finally disagree.
    class _Rife:
        def __call__(self, f0, f1, ts, scales, fast_mode, ensemble):
            return (f0 + f1) * 0.5

    mod._load_model = lambda *a, **k: _Rife()
    imgs3 = _frame(3)
    res_m = mod.ULSInterpolate().interpolate(
        video=_Vid(imgs3, Fraction(20, 1), audio), multiplier=2)
    _need(res_m[1] == 5 and abs(res_m[2] - 40.0) < 1e-9,
          "the main path must give 2*(3-1)+1 = 5 frames at 40 fps (got %r/%r)"
          % (res_m[1], res_m[2]))
    cm = res_m[4].get_components()
    _need(abs(float(cm.frame_rate) - 40.0) < 1e-9,
          "the video output must carry the NEW rate (%r) - carrying the "
          "source rate is the sync bug this whole cut is about"
          % (cm.frame_rate,))
    _need(cm.audio is audio,
          "the original audio must ride through the main path too")
    _need("duration" in res_m[3],
          "interp_info must carry the duration arithmetic on the main path")

    # A4 -- a bare frame batch has no sound, and nothing is invented
    res2 = mod.ULSInterpolate().interpolate(frames=_frame(1))
    _need(res2[4] is not None,
          "a frame batch must still get a video output (the RATE alone is "
          "worth carrying)")
    _need(res2[4].get_components().audio is None,
          "a bare frame batch carries no sound - none may be invented")

    # A4b -- no VIDEO API: the convenience goes, the result stays
    _install_video_api(mod, present=False)
    res3 = mod.ULSInterpolate().interpolate(frames=_frame(1))
    _need(res3[4] is None, "without the VIDEO API the video output is None")
    _need(res3[0] is not None and int(res3[0].shape[0]) == 1,
          "the frames output must survive a missing VIDEO API - a convenience "
          "may never cost a result")
    _install_video_api(mod)
    return mod


# ---------------------------------------------------------------------------
# A6 -- the Save says which input won
# ---------------------------------------------------------------------------
def a6(src=None):
    save = _load(SAVE, "ph_save", src)
    # The VIDEO branch guards on this first; without it the resolution never
    # runs and the guard would pass on an unreachable code path.
    save._HAS_VIDEO_API = True
    node = save.ULSSave()
    frames_v = torch.rand(4, 8, 8, 3)
    frames_i = torch.rand(9, 8, 8, 3)
    buf, old = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        try:
            node._save_video(
                image=frames_i, video=_Vid(frames_v, 16.0), audio=None,
                mask=None, video_preset="__no_such_preset__", quality=80,
                frame_rate=99.0, autoplay=False, pingpong=False, loop_count=0,
                trim_to_audio=False, filename_prefix="x", save_output=False,
                save_metadata=False, prompt=None, extra_pnginfo=None)
        except Exception:
            pass          # the write itself is out of scope; the LINE is not
    finally:
        sys.stdout = old
    said = buf.getvalue()
    _need("BOTH a video and an image are wired" in said,
          "a wired image that is silently discarded is the one shape of this "
          "node that can throw a stage's work away without a word (said: %r)"
          % (said[:200],))
    _need("VIDEO wins" in said and "9 image frame" in said,
          "the line must name the winner AND what was dropped (said: %r)"
          % (said[:300],))

    # and the ordinary single-input case stays quiet
    buf2, old = io.StringIO(), sys.stdout
    sys.stdout = buf2
    try:
        try:
            node._save_video(
                image=None, video=_Vid(frames_v, 16.0), audio=None, mask=None,
                video_preset="__no_such_preset__", quality=80, frame_rate=99.0,
                autoplay=False, pingpong=False, loop_count=0,
                trim_to_audio=False, filename_prefix="x", save_output=False,
                save_metadata=False, prompt=None, extra_pnginfo=None)
        except Exception:
            pass
    finally:
        sys.stdout = old
    _need("BOTH a video and an image" not in buf2.getvalue(),
          "a single wired input must NOT trigger the warning - a line that "
          "fires always is a line nobody reads")


# ---------------------------------------------------------------------------
# A7 -- mutations
# ---------------------------------------------------------------------------
def _mut(fn, path, old, new, label):
    src = _read(path)
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


def a7():
    n = 0
    n += bool(_mut(a1a2, INT,
                   'return ("duration %.3fs -> %.3fs (%+.0f ms; every source frame keeps its "',
                   'return ("duration %.3fs -> %.3fs (%+.0f s; every source frame keeps its "',
                   "the tail is reported in the wrong unit"))
    n += bool(_mut(a1a2, INT,
                   "    d_out = n_out / float(out_fps)",
                   "    d_out = n_out / float(src_fps)",
                   "the output duration is computed at the wrong rate"))
    # v890: the anchor follows the grown output tuple; the promise it drives
    # is unchanged (video owns slot 4, links store the index).
    n += bool(_mut(a3a4a5, INT,
                   '    RETURN_NAMES = ("frames", "frame_count", "fps", "interp_info", "video",\n'
                   '                    "audio")',
                   '    RETURN_NAMES = ("frames", "frame_count", "fps", "interp_info", "audio",\n'
                   '                    "video")',
                   "the video output is pushed off slot 4"))
    n += bool(_mut(a3a4a5, INT,
                   "        return video.get_components().audio",
                   "        return None",
                   "the audio is dropped on the way through"))
    n += bool(_mut(a3a4a5, INT,
                   "    out_video = _build_video(out, audio, out_fps)",
                   "    out_video = _build_video(out, audio, source_fps)",
                   "the video output carries the OLD rate"))
    n += bool(_mut(a6, SAVE,
                   '                print("[PLS] Save: BOTH a video and an image are wired. The "',
                   '                print("[PLS] Save: proceeding. The "',
                   "the Save goes silent about the override again"))
    return n


def main():
    a1a2()
    a3a4a5()
    a6()
    muts = a7()
    print("PASS: %s -- %d promises pinned, %d mutations caught (the sync law "
          "computed from the spec, the tail measured, audio+rate ride in one "
          "object, append-only outputs, the Save names its winner)"
          % (NAME, _checks, muts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
