# -*- coding: ascii -*-
"""Guard v1130 -- Control: Control Fit, Control Map, Apply Control, exact pose / depth from the 3D chain.

(Frank, 14.10.: "bitte die beste Loesung waehlen ... alle noetigen Nodes und Node-Anpassungen ... sowohl fuer
H3 als auch Qwen Edit ... evtl. auch mal ein Stock-Video reinziehen als Vorlage".)

  F  Control Fit: frame grids (17n+5, 8n+1, 4n+1), re-timing 30 -> 24 fps by nearest frame, start offset,
     hold / loop / cut, sizes in steps of 32 by fit / fill / stretch, a mask frame for frame, one image
     repeated; the chunked resize equals the whole-batch resize.
  M  Control Map: canny / gray / scribble / soft edges on a test picture (edges where the picture has
     them, nothing elsewhere); pose and depth call the models with the frames (stubbed here: no
     weights in a test); keypoints mode draws through the pack's own drawer (v1131).
  K  3D: Pose Animation's skeleton moves and turns with the mesh; Mesh Ring Render projects it to OpenPose
     keypoints EXACTLY (a wrist keypoint = the projected wrist joint, to 1e-4 px); S shows the figure's
     right shoulder on the image's left, N the opposite; the nose shows in S, not in N; an unnamed skeleton
     (SAM 3D Body style bone_000 ...) gives the same keypoints as the same skeleton with Mixamo names;
     depth: background 0, one scale per direction (max 1), figure in [0.2, 1].
  T  facing (correction): a walk along +X is turned to S; the same clip with a drift of 1 mm, or with an
     idle-like sway of 5 % of the height, is NOT turned (threshold 10 %; it was spun by up to 180 degrees).
  A  Apply Control: family check (a video control on an image model is refused), strength 0 / nothing
     connected passes the model through; with the Core's model library: the video patch gets the frames,
     mask, source, strength and sigma range; the image patch gives every batch item its own control and
     repeats it over a CFG batch, refuses a batch it cannot divide.
     The maps run a few frames at a time (same result as the whole clip at once).
  N  nodes: Apply Control's pick takes one frame of control and mask; missing model files answer with the folder to put them in; ASCII; registration.
Script-style: exit 0 = pass.
"""
import math
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
NAME = "test_v1130_control"
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


OUT = tempfile.mkdtemp(prefix="v1130_")
fp = types.ModuleType("folder_paths")
fp.get_temp_directory = lambda: OUT
fp.get_output_directory = lambda: OUT
fp.get_input_directory = lambda: OUT
fp.get_filename_list = lambda kind: []
sys.modules.setdefault("folder_paths", fp)
sys.path.insert(0, str(ROOT / "nodes"))
sys.path.insert(0, str(ROOT / "tests"))

CORE = None
for c in [os.environ.get("PLS_CORE_ROOT"), str(ROOT.parent.parent)]:
    if c and os.path.isfile(os.path.join(c, "comfy_extras", "nodes_sdpose.py")):
        CORE = c
        break
CORE_OK = CORE_CONTROL = False
if CORE:
    argv = sys.argv
    try:
        sys.argv = [argv[0], "--cpu"]
        sys.path.insert(0, CORE)
        import comfy.options
        comfy.options.enable_args_parsing()
        import comfy_extras.nodes_sdpose   # noqa: F401 (the drawer is used through ph_control)
        CORE_OK = True
        try:
            import comfy.ldm.minimax.controlnet   # noqa: F401
            import comfy.ldm.qwen_image21.model   # noqa: F401
            from comfy_extras.nodes_model_patch import QwenImage21FunControlPatch   # noqa: F401
            CORE_CONTROL = True
        except Exception:
            CORE_CONTROL = False
    except Exception as e:
        print("  note  Core not importable here (%s) -- the Core parts are skipped" % type(e).__name__)
    finally:
        sys.argv = argv
print("  core  %s%s" % (CORE or "none", "" if CORE_CONTROL else " (no Fun ControlNet code in this Core: A-core skipped)"))

comfy_api = types.ModuleType("comfy_api")
latest = types.ModuleType("comfy_api.latest")


class _Mesh:
    def __init__(self, vertices, faces, uvs=None, vertex_colors=None, texture=None, **kw):
        self.vertices, self.faces, self.uvs, self.vertex_colors, self.texture = vertices, faces, uvs, vertex_colors, texture


latest.Types = types.SimpleNamespace(MESH=_Mesh, File3D=lambda p: ("File3D", p))
comfy_api.latest = latest
try:
    import comfy_api.latest  # noqa: F401
except Exception:
    sys.modules["comfy_api"] = comfy_api
    sys.modules["comfy_api.latest"] = latest

import ph_control as C             # noqa: E402
try:                               # public build: the 3D chain (Pose Animation, Mesh Ring Render) is not part of it
    import ph_glb_skin as G        # noqa: E402
    import ph_mesh_anim as MA      # noqa: E402
    import ph_mesh_ring as MR      # noqa: E402
    _HAS_3D = True
except ImportError:
    _HAS_3D = False

# F ------------------------------------------------------------------------------ Control Fit
_need([C.snap_count(n, C.GRIDS[0]) for n in (4, 5, 21, 22, 141, 150)] == [5, 5, 5, 22, 141, 141],
      "F  17n+5: 4/5/21/22/141/150 -> 5/5/5/22/141/141")
_need(C.snap_count(30, "8n+1") == 25 and C.snap_count(30, "4n+1") == 29 and C.snap_count(30, "any") == 30, "F  8n+1, 4n+1, any")
clip = torch.arange(90, dtype=torch.float32).view(90, 1, 1, 1).expand(90, 36, 64, 3) / 90.0     # frame k = value k/90
out, _, rep = C.fit(clip, 30.0, 24.0, 0.0, 0, C.GRIDS[0], "hold last frame", 0, 0, "fill (crop)")
src = [int(round(float(v) * 90)) for v in out[:, 0, 0, 0]]
want = [int(math.floor(k * 30 / 24.0 + 1e-6)) for k in range(len(src))]
_need(rep["frames"] == 56 and src == want, "F  3 s at 30 fps -> 72 frames at 24 fps, snapped to 56, nearest source frame each (%s ...)" % src[:6])
out, _, rep = C.fit(clip, 30.0, 24.0, 1.0, 0, "any", "hold last frame", 0, 0, "fill (crop)")
_need(int(round(float(out[0, 0, 0, 0]) * 90)) == 30 and rep["frames"] == 48, "F  start 1 s: first frame = source 30, 48 frames left")
for end, n_expect, last in (("hold last frame", 141, 89), ("cut", 56, None)):
    out, _, rep = C.fit(clip, 30.0, 24.0, 0.0, 141, C.GRIDS[0], end, 0, 0, "fill (crop)")
    ok = rep["frames"] == n_expect and (last is None or int(round(float(out[-1, 0, 0, 0]) * 90)) == last)
    _need(ok, "F  141 frames wanted from 3 s, %s -> %d frames" % (end, rep["frames"]))
out, _, rep = C.fit(clip, 30.0, 24.0, 0.0, 141, C.GRIDS[0], "loop", 0, 0, "fill (crop)")
_need(rep["frames"] == 141 and int(round(float(out[90, 0, 0, 0]) * 90)) < 90 and "loop" in rep["note"], "F  loop wraps around")
img = torch.rand(1, 100, 150, 3)
for mode, (W, H) in (("fit (pad)", (704, 1280)), ("fill (crop)", (704, 1280)), ("stretch", (704, 1280))):
    o, _, r = C.fit(img, 0, 24.0, 0.0, 1, "any", "hold last frame", 704, 1280, mode)
    _need(tuple(o.shape) == (1, 1280, 704, 3), "F  %s -> 704 x 1280" % mode)
o, _, r = C.fit(img, 0, 24.0, 0.0, 1, "any", "hold last frame", 0, 0, "fit (pad)")
_need(o.shape[2] % 32 == 0 and o.shape[1] % 32 == 0 and (o.shape[2], o.shape[1]) == (160, 96), "F  0 x 0 keeps the source, rounded to 32: %dx%d" % (o.shape[2], o.shape[1]))
o, _, r = C.fit(img, 0, 24.0, 0.0, 0, "any", "hold last frame", 0, 640, "fit (pad)")
_need((o.shape[2], o.shape[1]) == (960, 640), "F  height only -> width follows the aspect: %dx%d" % (o.shape[2], o.shape[1]))
o, _, r = C.fit(img, 0, 24.0, 0.0, 22, C.GRIDS[0], "hold last frame", 64, 64, "fit (pad)")
_need(o.shape[0] == 22 and "repeated" in r["note"], "F  one image -> 22 frames (repeated)")
mk = torch.zeros(90, 36, 64)
mk[45:] = 1.0
o, m_o, r = C.fit(clip, 30.0, 24.0, 0.0, 0, "any", "hold last frame", 64, 32, "stretch", mask=mk)
_need(m_o is not None and m_o.shape[0] == o.shape[0] and float(m_o[35].mean()) == 0 and float(m_o[37].mean()) == 1,
      "F  the mask follows frame for frame (output 36 = source 45 is the first white)")
ring = torch.cat([torch.full((10, 8, 8, 3), k / 10.0) for k in range(8)])            # 8 directions x 10 frames
o, _, r = C.fit(ring, 24.0, 24.0, 0.0, 22, C.GRIDS[0], "loop", 0, 0, "stretch", 8, block=10, block_index=3)
_need(o.shape[0] == 22 and torch.allclose(o, torch.full_like(o, 0.3)) and r["block"][:2] == (3, 8),
      "F  block 3 of 10 frames = direction E only; loop stays inside the block")
o, _, r = C.fit(ring, 24.0, 24.0, 0.0, 1, "any", "hold last frame", 0, 0, "stretch", 8, block=1, block_index=25)
_need(o.shape[0] == 1 and abs(float(o.mean()) - 0.2) < 1e-6, "F  block 1 picks one image of a batch (index 25)")
try:
    C.fit(ring, 24.0, 24.0, 0.0, 0, "any", "cut", 0, 0, "stretch", 8, block=10, block_index=8)
    _need(False, "F  a block past the end is refused")
except ValueError as e:
    _need("block 8" in str(e), "F  block past the end: %s" % str(e)[:60])
node = C.ULSControlFit()
res = node.fit(clip, 24.0, 0.0, 0.0, 22, C.GRIDS[0], "hold last frame", 64, 32, "fill (crop)", 32, fps=30.0)
_need(res[2] == 22 and res[3] == 64 and res[4] == 32 and abs(res[5] - 22 / 24.0) < 1e-9 and "30.00 fps" in res[6],
      "F  node: the fps input is used, outputs count / size / seconds")

# M ------------------------------------------------------------------------------ Control Map
pic = torch.zeros(1, 64, 64, 3)
pic[:, 16:48, 16:48] = 1.0
cm = C.canny_map(pic, 0.2, 0.5)
edge = cm[0, ..., 0]
yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
dist = torch.minimum(torch.minimum((yy - 15.5).abs(), (yy - 47.5).abs()), torch.minimum((xx - 15.5).abs(), (xx - 47.5).abs()))
inside_band = ((yy >= 13) & (yy <= 50) & (xx >= 13) & (xx <= 50)) & (dist <= 3)
_need(float(edge[~inside_band].max()) == 0 and int((edge > 0).sum()) > 80,
      "M  canny: %d edge pixels, all within 3 px of the square's border" % int((edge > 0).sum()))
se = C.soft_edges(pic)[0, ..., 0]
_need(float(se[31, 31]) < 0.05 and float(se[16, 32]) > 0.5, "M  soft edges: strong on the border, ~0 inside")
sc = C.scribble(pic, 0.1, 0.3, 2)[0, ..., 0]
_need(set(np.unique(sc.numpy()).tolist()) <= {0.0, 1.0} and float(sc.sum()) > float((edge > 0).sum()),
      "M  scribble: binary and thicker than canny")
_clip = torch.rand(19, 24, 32, 3)
_need(float((C.by_chunks(C.soft_edges, _clip, chunk=4) - C.soft_edges(_clip)).abs().max()) < 1e-5,
      "M  maps in chunks of frames = the whole clip at once (19 frames, chunk 4)")
gr = C.to_gray(torch.tensor([[[[1.0, 0.0, 0.0]]]]))
_need(abs(float(gr[0, 0, 0, 0]) - 0.299) < 1e-6 and gr.shape[-1] == 3, "M  gray: luminance in 3 channels")
calls = {}
orig_sd, orig_da = C.sdpose_keypoints, C.da3_depth


def _sd(x, name, batch=16):
    calls["sd"] = (tuple(x.shape), name)
    return [{"canvas_width": 64, "canvas_height": 64, "people": []} for _ in range(x.shape[0])]


C.sdpose_keypoints, C.da3_depth = _sd, lambda x, name, res: (calls.__setitem__("da", (tuple(x.shape), name, res)) or x * 0.5)
orig_draw = C.draw_keypoints
C.draw_keypoints = lambda kp, *a: torch.zeros(len(kp), 64, 64, 3)
mp = C.ULSControlMap()
# AMENDED v1217: Control Map shows its frames (preview, default on) and then returns {"ui", "result"} -- the result
# is read here with the preview off (the preview itself is pinned in test_v1217_line_preview L5)
r = mp.make("pose (SDPose)", "sdpose_wholebody_fp16.safetensors", "(none)", True, False, 4, 0.2, 0.5, 504, images=pic.repeat(3, 1, 1, 1),
            preview=False)
_need(calls.get("sd") == ((3, 64, 64, 3), "sdpose_wholebody_fp16.safetensors") and "no person found" in r[2],
      "M  pose: the frames go to SDPose; no person -> the info says the control is black")
r = mp.make("depth (Depth Anything 3)", "(none)", "da3_mono_large.safetensors", True, False, 4, 0.2, 0.5, 518, images=pic, preview=False)
_need(calls.get("da") == ((1, 64, 64, 3), "da3_mono_large.safetensors", 518), "M  depth: the frames and the resolution go to Depth Anything 3")
for mode, mdl in (("pose (SDPose)", ("(none)", "x")), ("depth (Depth Anything 3)", ("x", "(none)"))):
    try:
        mp.make(mode, mdl[0], mdl[1], True, False, 4, 0.2, 0.5, 504, images=pic)
        _need(False, "M  %s without a model is refused" % mode)
    except ValueError as e:
        _need("models/" in str(e), "M  %s without a model: %s" % (mode, str(e)[:80]))
C.sdpose_keypoints, C.da3_depth, C.draw_keypoints = orig_sd, orig_da, orig_draw

# K ------------------------------------------------------------------------------ skeleton -> keypoints, depth
if not _HAS_3D:
    print("  skip  K / T: the 3D chain is not part of this build")
else:
    rig = G.load(str(ROOT / "tests" / "data" / "v1126_walk.glb"))
    seq, times, rep = G.pose_sequence(rig, 0, 4, None, None, True, "remove travel", "auto")
    _need(len(rep["joints"]) == 4 and rep["joints"][0].shape == (len(rig["nodes"]), 3), "K  pose_sequence returns the joints of every frame")
    hips = rig["nodes"].index("bone_000")
    moved = np.abs(rep["joints"][1][hips] - G.joint_positions(rig, 0, times[1])[hips])
    _need(float(moved[1]) < 1e-9 and float(np.hypot(moved[0], moved[2])) > 0.05, "K  the skeleton is moved on the spot like the mesh (height kept)")
    res = MA.ULSPoseAnimation().pose(rig, 0, 4, 0.0, 0.0, True, MA.IN_PLACE[0], MA.FACING[0])
    skel = res[3]
    _need(len(skel["frames"]) == 4 and np.allclose(skel["frames"][2], rep["joints"][2]), "K  Pose Animation outputs the skeleton")
    mesh = res[0]
    cams, names = MR.ring_cameras("8 (S, SE, E, NE, N, NW, W, SW)", 0.0)
    out = MR.ULSMeshRingRender().render(mesh, cams, 96, 128, "clay", 0.5, 1, "#000000", None, skel)["result"]
    imgs, masks, _n, info, depth, kps = out
    _need(len(kps) == 32 and depth.shape == (32, 128, 96, 3) and "18/18 body points" in info and "shape" in info,
          "K  32 views: depth + keypoints; the unnamed skeleton read by its shape (%s)" % info.split("; ")[-1][:70])
    ms = [MR.mesh_item(mesh, b) for b in range(4)]
    centre, half = MR.framing([x["v"] for x in ms])
    jm = C.infer_humanoid(skel["parent"], skel["frames"][0])
    for vi, (cam_i, fr) in enumerate([(0, 0), (2, 1), (4, 3)]):
        k = kps[cam_i * 4 + fr]["people"][0]["pose_keypoints_2d"]
        wr = torch.tensor(skel["frames"][fr][jm["R hand"]][None], dtype=torch.float32)
        sx, sy, _z, _b = MR.project(wr, cams[cam_i], centre, half, 96, 128)
        if vi == 0:
            _need(abs(k[12] - float(sx[0])) < 1e-4 and abs(k[13] - float(sy[0])) < 1e-4, "K  R wrist keypoint = the projected wrist joint (S, frame 1)")
    s_kp = kps[0]["people"][0]["pose_keypoints_2d"]
    n_kp = kps[16]["people"][0]["pose_keypoints_2d"]
    _need(s_kp[6] < s_kp[15] and n_kp[6] > n_kp[15], "K  S: her right shoulder on the image's left; N: on the right")
    _need(s_kp[2] == 1.0 and n_kp[2] == 0.0, "K  the nose shows in S, not in N")
    named = dict(skel)
    inv = {v: k for k, v in jm.items() if isinstance(k, str)}
    mix = {"hips": "Hips", "R thigh": "RightUpLeg", "R shin": "RightLeg", "R foot": "RightFoot", "L thigh": "LeftUpLeg",
           "L shin": "LeftLeg", "L foot": "LeftFoot", "R upperarm": "RightArm", "R forearm": "RightForeArm", "R hand": "RightHand",
           "L upperarm": "LeftArm", "L forearm": "LeftForeArm", "L hand": "LeftHand", "head": "Head", "head top": "HeadTop_End",
           "R toe": "RightToeBase", "L toe": "LeftToeBase"}
    named["names"] = ["mixamorig:" + mix[inv[i]] if i in inv and inv[i] in mix else n for i, n in enumerate(skel["names"])]
    kn, note = MR.skeleton_keypoints(named, cams, 4, centre, half, 96, 128)
    same = all(np.allclose(a["people"][0]["pose_keypoints_2d"], b["people"][0]["pose_keypoints_2d"], atol=1e-6)
               for a, b in zip(kn, kps))
    _need("names" in note and same, "K  the same skeleton with Mixamo names gives the same keypoints (read by names)")
    d = depth[..., 0]
    mk = masks
    bg_zero = float(d[mk < 0.01].max()) == 0.0
    per_cam = [float(d[i * 4:(i + 1) * 4].max()) for i in range(8)]
    fig = d[mk > 0.99]
    _need(bg_zero and all(abs(x - 1.0) < 0.02 for x in per_cam) and float(fig.min()) >= 0.19,
          "K  depth: background 0, max 1 in every direction, figure >= 0.2")
    try:
        MR.skeleton_keypoints({"names": skel["names"], "parent": skel["parent"], "frames": skel["frames"][:2]}, cams, 4,
                              centre, half, 96, 128)
        _need(False, "K  a skeleton with the wrong frame count is refused")
    except ValueError as e:
        _need("frame" in str(e), "K  wrong frame count: %s" % str(e)[:70])

    if True:                      # the drawer is the pack's own since v1131: no Core needed
        drawn = C.draw_keypoints(kps[:2], True, False, False, 3)
        on = (drawn[0].sum(-1) > 0)
        ys_, xs_ = torch.nonzero(on, as_tuple=True)
        _need(tuple(drawn.shape) == (2, 128, 96, 3) and int(on.sum()) > 200 and abs(float(xs_.float().mean()) - 48) < 16,
              "K  the pack's DWPose-style drawer draws the projected skeleton (%d px, centred)" % int(on.sum()))

    # T ------------------------------------------------------------------------------ facing (correction)
    import copy                        # noqa: E402
    side = copy.deepcopy(rig)
    for ch in side["anims"][0]["channels"]:
        if ch["path"] == "translation" and ch["node"] == hips:
            v = ch["values"].copy()
            ch["values"][:, 0], ch["values"][:, 2] = v[:, 2], v[:, 0]          # walk along +X instead of +Z
    _, _, r1 = G.pose_sequence(side, 0, 4, None, None, True, "remove travel", "auto")
    tiny = copy.deepcopy(side)
    for ch in tiny["anims"][0]["channels"]:
        if ch["path"] == "translation" and ch["node"] == hips:
            base = ch["values"][0].copy()
            ch["values"][:, [0, 2]] = base[[0, 2]] + (ch["values"][:, [0, 2]] - base[[0, 2]]) * 0.0009
    _, _, r2 = G.pose_sequence(tiny, 0, 4, None, None, True, "remove travel", "auto")
    _h = float(np.ptp(rig["V"][:, 1]))
    _sc = 0.05 * _h / float(np.hypot(r1["travel"][0], r1["travel"][2]))
    sway = copy.deepcopy(side)
    for ch in sway["anims"][0]["channels"]:
        if ch["path"] == "translation" and ch["node"] == hips:
            base = ch["values"][0].copy()
            ch["values"][:, [0, 2]] = base[[0, 2]] + (ch["values"][:, [0, 2]] - base[[0, 2]]) * _sc
    _, _, r3 = G.pose_sequence(sway, 0, 4, None, None, True, "remove travel", "auto")
    _need(r3["turn"] == 0.0, "T  an idle-like sway of 5 %% of the height is not turned (turn %.1f)" % r3["turn"])
    _need(abs(abs(r1["turn"]) - 90.0) < 1.0, "T  a walk along +X is turned onto S (%.1f deg)" % r1["turn"])
    _need(r2["turn"] == 0.0 and np.hypot(r2["travel"][0], r2["travel"][2]) < r2["min_travel"],
          "T  the same clip drifting %.1f mm is not turned (threshold %.0f mm)" % (1000 * np.hypot(r2["travel"][0], r2["travel"][2]),
                                                                                 1000 * r2["min_travel"]))

# A ------------------------------------------------------------------------------ Apply Control


def _cls(mod, name):
    return type(name, (), {"__module__": mod})


class _MS:
    def percent_to_sigma(self, p):
        return 1.0 - p


class _Model:
    def __init__(self, fam_mod):
        self.dm = _cls(fam_mod, "Net")()
        self.wrappers, self.patches = [], {}

    def get_model_object(self, k):
        return self.dm if k == "diffusion_model" else _MS()

    def clone(self):
        c = _Model.__new__(_Model)
        c.dm, c.wrappers, c.patches = self.dm, list(self.wrappers), dict(self.patches)
        return c


vid_patch = types.SimpleNamespace(model=_cls("comfy.ldm.minimax.controlnet", "MiniMaxH3FunControl")())
img_patch = types.SimpleNamespace(model=_cls("comfy.ldm.qwen_image21.model", "QwenImage21FunControl")())
_need(C.patch_family(vid_patch) == "video" and C.patch_family(img_patch) == "image", "A  the family is read from the patch")
try:
    C.apply(_Model("comfy.ldm.qwen_image21.model"), vid_patch, None, 1.0, 0.0, 1.0, torch.zeros(5, 32, 32, 3))
    _need(False, "A  a video control on an image model is refused")
except ValueError as e:
    _need("family" in str(e), "A  wrong family: %s" % str(e)[:80])
m0 = _Model("comfy.ldm.minimax.model")
for args, why in (((0.0, 0.0, 1.0, torch.zeros(5, 32, 32, 3)), "strength 0"), ((1.0, 0.0, 1.0, None), "no control")):
    mm, rp = C.apply(m0, vid_patch, None, *args)
    _need(mm is m0 and "skipped" in rp, "A  %s passes the model through" % why)
if CORE_CONTROL:
    CC = C._core()
    got = {}

    class _Rec:
        def __init__(self, *a):
            got["args"] = a

        def register(self, m):
            got["reg"] = m

    keep = CC.VideoControlPatch
    CC.VideoControlPatch = _Rec
    try:
        ctl, mk5, srcv = torch.rand(22, 32, 32, 3), torch.ones(22, 32, 32), torch.rand(22, 32, 32, 3)
        mm, rp = C.apply(m0, vid_patch, "VAE", 0.7, 0.1, 0.6, ctl, mk5, srcv)
        a = got["args"]
        _need(a[0] is vid_patch and a[1] == "VAE" and tuple(a[2].shape) == (22, 3, 32, 32) and a[3] is mk5
              and tuple(a[4].shape) == (22, 3, 32, 32) and a[5] == 0.7 and abs(a[6] - 0.9) < 1e-9 and abs(a[7] - 0.4) < 1e-9
              and got["reg"] is mm and mm is not m0,
              "A  video: the patch gets frames (channels first), mask, source, strength 0.7, sigma 0.9-0.4")
    finally:
        CC.VideoControlPatch = keep
    import comfy.latent_formats
    LC = comfy.latent_formats.QwenImage21().latent_channels

    class _VAE:
        def spacial_compression_encode(self):
            return 8

        def encode(self, x):
            return torch.full((x.shape[0], LC, x.shape[1] // 8, x.shape[2] // 8), float(x.mean()))

    class _Ctl:
        def init_stream(self, x, control, prefix_len):
            got["stream_control"] = control.shape
            return torch.zeros_like(x)

        def step(self, index, c, *a):
            return c, torch.zeros_like(c)

    BP = C._image_patch_class()
    imgs4 = torch.stack([torch.full((64, 64, 3), v) for v in (0.1, 0.4, 0.7, 1.0)])
    p = BP(types.SimpleNamespace(model=_Ctl()), _VAE(), imgs4, 1.0)
    p.layers = [0, 4]
    p.prepare(8, 8)
    means = [float(p.control[i, :LC].mean()) for i in range(4)]
    _need(p.control.shape[0] == 4 and means == sorted(means) and len(set(round(x, 4) for x in means)) == 4,
          "A  image batch: 4 control images -> 4 different control rows")
    p.active = True
    p.pristine = torch.zeros(8, 70, 32)
    x = {"img": torch.zeros(8, 70, 32)}
    p.after_block(0, {"img": x["img"], "prefix_len": 6, "mod": None, "pe": None, "attn_fn": None, "transformer_options": {}}, x)
    _need(p.control.shape[0] == 8 and torch.equal(p.control[:4], p.control[4:]), "A  ... repeated over a CFG batch of 8")
    p2 = BP(types.SimpleNamespace(model=_Ctl()), _VAE(), imgs4[:3], 1.0)
    p2.layers = [0, 4]
    p2.prepare(8, 8)
    p2.active = True
    p2.pristine = torch.zeros(4, 70, 32)
    try:
        p2.after_block(0, {"img": torch.zeros(4, 70, 32), "prefix_len": 6, "mod": None, "pe": None, "attn_fn": None,
                           "transformer_options": {}}, {"img": torch.zeros(4, 70, 32)})
        _need(False, "A  3 control images for a batch of 4 are refused")
    except ValueError as e:
        _need("batch" in str(e), "A  3 images for a batch of 4: %s" % str(e)[:70])
else:
    print("  skip  A-core: this Core has no Fun ControlNet code (checked in the core_master phase)")

# N ------------------------------------------------------------------------------ nodes, registration, ASCII
try:
    C.ULSApplyControl().apply(m0, None, "(none)", 1.0, 0.0, 1.0)
    _need(False, "N  Apply Control without a control file is refused")
except ValueError as e:
    _need("model_patches" in str(e), "N  no control file: %s" % str(e)[:70])
try:
    C.ULSApplyControl().apply(m0, None, "x", 1.0, 0.8, 0.2, control_patch=vid_patch)
    _need(False, "N  end before start is refused")
except ValueError as e:
    _need("before" in str(e), "N  end before start: %s" % str(e)[:60])
_x = torch.rand(11, 20, 30, 3)
_ix = [0, 3, 3, 10, 2, 0, 10, 10, 5]
for _mode in ("fit (pad)", "fill (crop)", "stretch"):
    _a = C.resize_picked(_x, _ix, 64, 32, _mode, chunk=3)
    _b = C.resize_frames(_x[torch.as_tensor(_ix)], 64, 32, _mode, 0.0)
    _need(_a.shape == _b.shape and float((_a - _b).abs().max()) < 1e-6, "F  chunked resize = whole-batch resize (%s)" % _mode)
_c8 = torch.arange(8.0).view(8, 1, 1, 1).expand(8, 4, 6, 3).contiguous()
_k = C.pick_frame(3, _c8, torch.ones(8, 4, 6), None)
_need(_k[0].shape == (1, 4, 6, 3) and float(_k[0].max()) == 3.0 and _k[1].shape == (1, 4, 6) and _k[2] is None,
      "N  pick 3 takes frame 3 of control and mask")
_need(C.pick_frame(-1, _c8)[0] is _c8, "N  pick -1 leaves the control whole")
try:
    C.pick_frame(8, _c8)
    _need(False, "N  pick past the end is refused")
except ValueError as e:
    _need("only 8" in str(e), "N  pick past the end: %s" % str(e)[:60])
_need(list(C.ULSApplyControl.INPUT_TYPES()["required"])[-1] == "pick", "N  pick is the last widget of Apply Control")
init = (ROOT / "__init__.py").read_text(encoding="utf-8")
_need(all(k in init for k in ("ULSControlFit", "ULSControlMap", "ULSApplyControl")), "N  the three nodes are registered")
for f in [x for x in ("nodes/ph_control.py", "tests/test_v1130_control.py", "nodes/ph_mesh_ring.py") if (ROOT / x).exists()]:
    _need(all(b < 128 for b in (ROOT / f).read_bytes()), "N  %s is pure ASCII" % f)

if FAILS:
    print("[%s] FAIL -- %d" % (NAME, len(FAILS)))
    sys.exit(1)
print("[%s] PASS -- fit, maps, exact 3D pose + depth, facing fix, apply (video + image batch)" % NAME)
