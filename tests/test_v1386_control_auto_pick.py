"""Guard v1386 -- Apply Control with nothing picked takes the one fitting control file on disk.

Frank's first led walk (24.11.): control_model stood on '(none)' as the workflow came; the node stopped with "no
Fun ControlNet Union file in models/model_patches" whether or not one was there.

  F   control_family_of(): qwen -> image, minimax / h3 -> video (name or door label), else None
  P   pick_control(): exactly one fitting file -> it; a file of unknown family fits every model; a file of the
      other family does not; none -> an error that says so (and lists what IS there); several -> an error that
      lists them; a model of unknown family takes any single file
  D   controls_on_disk(): the door's control files only when installed (labelled), hand-placed files as listed
  J   the node's line (JS): nothing picked while the set's file is on disk -> 'Use it', never a Download
  N   node: '(none)' / '' / '(put ...)' -> the auto pick and a word in the info; a picked file is left alone; the
      old blanket message is gone
  Z   ASCII; version stamps agree

Script-style: exit 0 = pass.
"""
import os
import re
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILS = []


def check(c, m):
    if not c:
        FAILS.append(m)


def raises(fn, words, m):
    try:
        fn()
        check(False, m + " (no error)")
    except ValueError as e:
        check(all(w in str(e) for w in words), m + ": %s" % e)


fp = types.ModuleType("folder_paths")
fp.get_temp_directory = fp.get_output_directory = fp.get_input_directory = lambda: "/tmp"
fp.get_filename_list = lambda kind: []
sys.modules.setdefault("folder_paths", fp)
sys.path.insert(0, os.path.join(ROOT, "nodes"))
import ph_control as C  # noqa: E402

H3 = "minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors"
QW = "qwen_image_2.1_fun_controlnet_union_int8_convrot.safetensors"

# F
check(C.control_family_of(H3) == "video" and C.control_family_of(QW) == "image" and C.control_family_of("my_control.safetensors") is None,
      "F the family off the name")
check(C.control_family_of("x.safetensors", "H3 Fun ControlNet Union 2.0 int8") == "video" and C.control_family_of("sub\\\\MiniMax_union.safetensors") == "video",
      "F the door label counts, a subfolder does not hide the name")

# P
check(C.pick_control("video", [(H3, "video")]) == H3, "P the one fitting file")
check(C.pick_control("video", [(H3, "video"), (QW, "image")]) == H3 and C.pick_control("image", [(H3, "video"), (QW, "image")]) == QW,
      "P the other family's file does not count")
check(C.pick_control("video", [("own.safetensors", None)]) == "own.safetensors", "P a file of unknown family fits")
check(C.pick_control(None, [(QW, "image")]) == QW, "P a model of unknown family takes the single file")
raises(lambda: C.pick_control("video", []), ["no control file picked", "for a video model", "holds none", "Download"], "P nothing on disk")
raises(lambda: C.pick_control("video", [(QW, "image")]), ["no Fun ControlNet Union for a video model", "there: " + QW], "P only the other family on disk: named")
raises(lambda: C.pick_control("video", [(H3, "video"), ("b_h3.safetensors", "video")]), ["2 on disk fit a video model", "pick one in control_model", H3, "b_h3.safetensors"],
       "P several fit: listed")
raises(lambda: C.pick_control("video", [(H3, "video"), ("own.safetensors", None)]), ["2 on disk fit"], "P a known and an unknown one: no guess")
raises(lambda: C.pick_control("MiniMax H3 Image", [(H3, "video")]), ["for a MiniMax H3 Image model"], "P a video control on the image model is not taken")

# D
WW = C._ww()
reg = [n for n, sp in WW.REGISTRY.items() if sp.get("set") == "control"]
check(H3 in reg and QW in reg, "D the door knows the control files")
real = WW.local_path
try:
    WW.local_path = lambda n: ("/m/" + n) if n == H3 else None
    got = C.controls_on_disk(["own.safetensors", "sub\\" + H3])
    check([g[1] for g in got] == ["video", None] and WW.strip(got[0][0]) == H3 and got[1][0] == "own.safetensors",
          "D installed door files (labelled) + hand-placed ones; a door file listed again is not doubled: %s" % got)
    WW.local_path = lambda n: None
    check(C.controls_on_disk([]) == [], "D nothing installed, nothing listed: empty")
finally:
    WW.local_path = real

# N
node = C.ULSApplyControl()
calls = []
keep = (C.controls_on_disk, C.load_patch, C.apply, C.describe_patch, C.model_family)
try:
    C.controls_on_disk = lambda files=None: [(H3 + " \u00b7 4.2 GB", "video")]
    C.load_patch = lambda name: calls.append(name) or "PATCH"
    C.apply = lambda *a, **k: ("MODEL+", {"family": "video", "frames": 141, "size": (704, 1280), "inpaint": False})
    C.describe_patch = lambda p: "video Fun ControlNet Union 2.x"
    C.model_family = lambda m: "video"
    for none in ("(none)", "", "(put a file into models/model_patches)"):
        calls.clear()
        m, info = node.apply("MODEL", "VAE", none, 1.0, 0.0, 0.75)
        check(m == "MODEL+" and calls and WW.strip(calls[0]) == H3 and "control_model was not picked: took the one fitting file on disk, " + H3 in info,
              "N %r: the auto pick, named in the info: %s" % (none, info))
    calls.clear()
    m, info = node.apply("MODEL", "VAE", "mine.safetensors", 1.0, 0.0, 0.75)
    check(calls == ["mine.safetensors"] and "was not picked" not in info, "N a picked file is left alone")
    calls.clear()
    m, info = node.apply("MODEL", "VAE", "(none)", 1.0, 0.0, 0.75, -1, None, None, None, "WIRED")
    check(calls == [] and "was not picked" not in info, "N a wired patch needs no pick")
    C.controls_on_disk = lambda files=None: []
    raises(lambda: node.apply("MODEL", "VAE", "(none)", 1.0, 0.0, 0.75), ["no control file picked", "Download"], "N nothing on disk")
finally:
    C.controls_on_disk, C.load_patch, C.apply, C.describe_patch, C.model_family = keep
src = open(os.path.join(ROOT, "nodes", "ph_control.py"), encoding="ascii").read()
check("no Fun ControlNet Union file in models/model_patches." not in src, "N the old blanket message is gone")

# J
js = open(os.path.join(ROOT, "web", "js", "ph_world_weights.js"), encoding="utf-8").read()
i_use, i_dl = js.find('if (st.kind === "none" && m.installed) {'), js.find('line.show(text, "#e0a030", "Download " + m.size')
check(0 < i_use < i_dl and '"Use it"' in js[i_use:i_dl] and "install(" not in js[i_use:i_dl] and "return;" in js[i_use:i_dl],
      "J nothing picked + the set's file on disk: 'Use it' picks it, no Download is offered")

# Z
for rel in ("nodes/ph_control.py", "tests/test_v1386_control_auto_pick.py"):
    with open(os.path.join(ROOT, rel), "rb") as fh:
        check(all(ch < 128 for ch in fh.read()), "Z  ASCII-clean: " + rel)
_pv = re.search(r'PLUGIN_VERSION = "v(\d+)"', open(os.path.join(ROOT, "web", "js", "uls_compat.js")).read())
_pp = re.search(r'^version = "4\.(\d+)\.0"', open(os.path.join(ROOT, "pyproject.toml")).read(), re.M)
# PUBLIC CUT (v409, declared): the public build counts its own way -- vNNN = Major x 100 + Minor (test_v365 V).
check(_pv and _pp and int(_pv.group(1)) >= 409 and int(_pv.group(1)) - 400 == int(_pp.group(1)), "Z  version stamps agree")

if FAILS:
    print("FAIL test_v1386_control_auto_pick:")
    for f_ in FAILS:
        print("  - " + f_)
    sys.exit(1)
print("ok test_v1386_control_auto_pick")
