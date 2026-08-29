"""v871 -- ULSVAE splits a joint AV latent the way Core does, and says so.

Core puts the split rule in the NODES:
    nodes.VAEDecode              -> latent.unbind()[0]    (video)
    nodes_audio.vae_decode_audio -> latent.unbind()[-1]   (audio)

ph_vae mirrors both. This guard drives the three helpers against a fake that
MIRRORS Core's comfy/nested_tensor.py, and pins the structural promises that no
unit test can see: the LATENT passthrough must stay WHOLE, the audio maths must
stay a 1:1 mirror, and AUDIO must be APPENDED so old workflows keep their wires.

Runs without torch and without comfy.
"""
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = (ROOT / "nodes" / "ph_vae.py").read_text(encoding="utf-8")
FAILED = []


def _fail(m):
    FAILED.append(m)
    print("FAIL: {}".format(m))


def _ok(m):
    print("ok  : {}".format(m))


WANT = ("_is_joint_latent", "_video_latent", "_audio_latent")
ns = {}
found = {}
tree = ast.parse(SRC)
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in WANT:
        found[node.name] = ast.get_source_segment(SRC, node)
for name in WANT:
    if name not in found:
        _fail("helper {} is missing from ph_vae.py".format(name))
if len(found) == len(WANT):
    exec(compile("\n\n".join(found[n] for n in WANT), "<v871>", "exec"), ns)
    _ok("three split helpers lifted and compiled without torch")


class FakeTensor(object):
    def __init__(self, tag):
        self.tag = tag
        self.is_nested = False


class FakeJoint(object):
    """Mirror of Core comfy/nested_tensor.py: .tensors + .unbind(), is_nested
    True. Deliberately NO numel()/detach() -- see test_v870."""

    def __init__(self, tensors):
        self.tensors = list(tensors)
        self.is_nested = True

    def unbind(self):
        return self.tensors


video, audio = FakeTensor("video"), FakeTensor("audio")
joint = FakeJoint([video, audio])

# --- PRECONDITION PIN: the fixture must actually BE joint ------------------
if not getattr(joint, "is_nested", False) or len(joint.unbind()) != 2:
    _fail("fixture drifted: the joint fixture must carry TWO parts and answer "
          "is_nested -- otherwise every probe below observes nothing")
else:
    _ok("fixture pins its own precondition (two parts, is_nested True)")

if not FAILED:
    is_joint, vid, aud = ns["_is_joint_latent"], ns["_video_latent"], ns["_audio_latent"]

    if not (is_joint(joint) is True and is_joint(video) is False
            and is_joint(None) is False):
        _fail("_is_joint_latent must answer True/False/False and never raise")
    else:
        _ok("_is_joint_latent: joint True, plain False, None False")

    if vid(joint) is not video:
        _fail("_video_latent must return unbind()[0] -- Core's video half")
    elif vid(video) is not video:
        _fail("_video_latent must pass a plain latent through untouched")
    else:
        _ok("_video_latent: joint -> first part, plain -> itself (Core's rule)")

    if aud(joint) is not audio:
        _fail("_audio_latent must return unbind()[-1] -- Core's audio half")
    else:
        _ok("_audio_latent: joint -> last part (Core's rule)")

    if aud(video) is not None:
        _fail("_audio_latent MUST return None for a plain video latent -- "
              "otherwise a video latent gets fed to an audio VAE")
    elif aud(FakeJoint([video])) is not None:
        _fail("_audio_latent must return None when there is only ONE part")
    elif aud(None) is not None:
        _fail("_audio_latent must return None for None")
    else:
        _ok("_audio_latent: no audio half -> None, never a wrong-half decode")

# --- STRUCTURE -------------------------------------------------------------
run_src = ""
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == "ULSVAE":
        for m in node.body:
            if isinstance(m, ast.FunctionDef) and m.name == "run":
                run_src = ast.get_source_segment(SRC, m) or ""

if "lat_dec = _video_latent(lat_in)" not in run_src:
    _fail("run() must derive a separate lat_dec for the decode lane")
elif "_decode_lane(vae, lat_dec" not in run_src:
    _fail("the decode lane must receive the VIDEO half (lat_dec)")
elif "_decode_need(vae, lat_dec" not in run_src:
    _fail("the memory verdict must be computed on the VIDEO half too")
else:
    _ok("decode lane and its budget both run on the video half")

# THE passthrough promise: the LATENT output must stay WHOLE.
if "_video_latent(samples" in run_src or 'lat_in = _video_latent' in run_src:
    _fail("lat_in itself was split -- the LATENT output would silently drop "
          "the audio half and a downstream audio decode would get video data")
elif 'out_latent = samples if isinstance(samples, dict)' not in run_src:
    _fail("the LATENT passthrough no longer forwards the original samples dict")
else:
    _ok("LATENT output still carries the WHOLE joint latent (passthrough safe)")

if "_audio_latent(lat_in)" not in run_src:
    _fail("run() never asks for the audio half")
elif "audio_vae" not in run_src:
    _fail("run() does not take an audio_vae")
else:
    _ok("run() takes audio_vae and asks for the audio half")

# AUDIO must be APPENDED, never inserted: old workflows keep their slots.
for pat, why in ((('"LATENT", "IMAGE", "AUDIO"'), "RETURN_TYPES order"),
                 (('"latent", "image", "audio"'), "RETURN_NAMES order")):
    if pat not in SRC:
        _fail("{}: AUDIO must be APPENDED last, or saved workflows lose their "
              "wires".format(why))
    else:
        _ok("{}: AUDIO appended last".format(why))

lane = ""
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_decode_audio_lane":
        lane = ast.get_source_segment(SRC, node) or ""
if not lane:
    _fail("_decode_audio_lane is missing")
else:
    # 1:1 mirror of Core's vae_decode_audio maths
    for frag, why in (("* 5.0", "the five-sigma factor"),
                      ("std[std < 1.0] = 1.0", "the floor at 1.0"),
                      ("movedim(-1, 1)", "Core's channel move")):
        if frag not in lane:
            _fail("audio maths drifted from Core: {} is gone".format(why))
    if all(f in lane for f in ("* 5.0", "std[std < 1.0] = 1.0", "movedim(-1, 1)")):
        _ok("audio maths is a 1:1 mirror of Core's vae_decode_audio")
    if "decode_tiled" in lane:
        _fail("the audio lane must NOT tile: tile_x/tile_y are spatial and this "
              "node's tile widgets describe image tiles")
    else:
        _ok("audio lane does not tile (stated choice, not an oversight)")
    if "audio_sample_rate_output" not in lane or "44100" not in lane:
        _fail("the sample-rate fallback chain does not mirror Core")
    else:
        _ok("sample-rate chain mirrors Core and names which link fired")

print("\n{}: {} failure(s)".format(pathlib.Path(__file__).name, len(FAILED)))
sys.exit(1 if FAILED else 0)
