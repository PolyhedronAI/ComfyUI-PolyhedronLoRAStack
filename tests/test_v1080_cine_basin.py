"""Guard v1080 -- the Cine Timeline as a basin: "new render -> new track" is ON by
default, the first run of a place lands in V1, every later run in a track of its
own; and the clips stay where the Save writes them (no second folder by default).

WHY (Frank 10.10.): "ein erneuter Durchlauf laedt keine zweite Video/Audio-Ebene
rein, sondern ueberschreibt die erste ... das soll ein Sammelbecken werden"; and the
clip_prefix cable diverted his final Save into output/<project>/ -- "das gefaellt
mir nicht".

WHAT THIS PINS:

  B1  cine_edit.clean_edit keeps an explicit auto_track False (the OFF state), keeps
      True, and leaves a missing key missing (= on).
  B2  ph_cine_timeline.edit_plan honours a pinned take when the key is missing or
      True, ignores it only when auto_track is False (driven against two real clips).
  B3  JS: autoOn() is exported and reads "not explicitly false"; _setAuto(off) stores
      false (never deletes the key); the checkbox default is on; _autoTrack1 pins the
      FIRST run of a place to V1 and returns before adding a track (text pins on the
      run path, the harness drives autoOn).
  B4  nothing in the timeline's run path copies or moves a clip: no shutil.copy /
      move / os.rename on a video in ph_cine_timeline.py or cine_assemble.register;
      register() stores a PATH relative to output/ (driven on a temp tree).
  B5  the texts say it: the project tooltip and the node DESCRIPTION name the stack
      BY PATH and call clip_prefix optional; the render lines no longer promise
      "-> clip <prefix>".

Script-style: exit 0 = pass.
"""
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

NAME = "v1080"
ROOT = Path(__file__).resolve().parent.parent


def _fail(m):
    print("[%s] FAIL -- %s" % (NAME, m))
    sys.exit(1)


def _need(c, m):
    if not c:
        _fail(m)


# ---- B1 -----------------------------------------------------------------
pkg = types.ModuleType("plsn1080")
pkg.__path__ = [str(ROOT / "nodes")]
sys.modules["plsn1080"] = pkg
E = importlib.import_module("plsn1080.cine_edit")
_need(E.clean_edit({"auto_track": False}).get("auto_track") is False, "B1 explicit False is kept")
_need(E.clean_edit({"auto_track": 1}).get("auto_track") is True, "B1 truthy becomes True")
_need("auto_track" not in E.clean_edit({}), "B1 a missing key stays missing (= on)")

# ---- B2 (needs ffmpeg + folder_paths stand-in) ---------------------------
ff = shutil.which("ffmpeg")
TMP = Path(tempfile.mkdtemp(prefix="v1080_"))
(TMP / "output" / "fx").mkdir(parents=True)
(TMP / "input").mkdir()
(TMP / "temp").mkdir()
fp = types.ModuleType("folder_paths")
fp.get_output_directory = lambda: str(TMP / "output")
fp.get_input_directory = lambda: str(TMP / "input")
fp.get_temp_directory = lambda: str(TMP / "temp")
fp.models_dir = str(TMP / "models")
fp.get_filename_list = lambda f: []
fp.get_folder_paths = lambda f: [str(TMP / "models" / f)]
sys.modules["folder_paths"] = fp
if ff:
    TL = importlib.import_module("plsn1080.ph_cine_timeline")
    T = sys.modules["plsn1080.cine_timeline"]
    for i, n in enumerate((48, 72)):
        f = TMP / "output" / "fx" / ("whole_0000%d_.mp4" % (i + 1))
        subprocess.run([ff, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=64x36:r=24", "-frames:v", str(n),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(f)], capture_output=True)
        os.utime(f, (time.time() - 100 + i * 50, time.time() - 100 + i * 50))
    model = T.load(json.dumps({"seconds": 2.0, "project": "fx", "render": "whole",
                               "shots": [{"at": 0, "cut": "the camera cuts to", "text": "x"}]}))
    pin = {"takes": {"0": "fx/whole_00001_.mp4"}}
    a = TL.edit_plan(model, dict(pin), strips=False)["clips"][0]
    b = TL.edit_plan(model, dict(pin, auto_track=True), strips=False)["clips"][0]
    c = TL.edit_plan(model, dict(pin, auto_track=False), strips=False)["clips"][0]
    _need(a["name"] == "whole_00001_.mp4" and b["name"] == "whole_00001_.mp4",
          "B2 missing key / True: V1 keeps its pinned take (got %s / %s)" % (a["name"], b["name"]))
    _need(c["name"] == "whole_00002_.mp4", "B2 False: V1 follows the newest run (got %s)" % c["name"])
    # B4 register stores a path, the file stays where it is
    A = sys.modules["plsn1080.cine_assemble"]
    src = TMP / "output" / "2026-10-10" / "PH_final_00001_.mp4"
    src.parent.mkdir(parents=True)
    shutil.copy(TMP / "output" / "fx" / "whole_00001_.mp4", src)
    before = sorted(os.listdir(TMP / "output" / "fx"))
    rel = A.register(str(TMP / "output" / "fx"), 0, str(src), primary=True)
    _need(rel == "2026-10-10/PH_final_00001_.mp4" and src.is_file(), "B4 register keeps the file where it is: %r" % rel)
    after = sorted(os.listdir(TMP / "output" / "fx"))
    _need(set(after) - set(before) == {"stack.json"}, "B4 only the stack LIST joined the project folder, got %r" % (after,))
    takes = A.scene_takes(str(TMP / "output" / "fx"), 1)
    _need(takes.get(0) and os.path.realpath(takes[0][0]) == os.path.realpath(str(src)),
          "B4 the registered (primary) clip leads the whole-render takes: %r" % (takes.get(0),))
else:
    print("[%s] note: ffmpeg missing -- B2/B4 driven parts skipped" % NAME)
shutil.rmtree(TMP, ignore_errors=True)

# ---- B3 JS ---------------------------------------------------------------
js = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")
_need('export function autoOn(e) { return !(e && e[AUTO_KEY] === false); }' in js, "B3 autoOn exported, 'not explicitly false'")
i = js.find("function _setAuto(node, mode)")   # AMENDED v1198: three modes
_need(i > 0 and '} else { e[AUTO_KEY] = false; delete e.takes; }' in js[i:i + 700], "B3 off stores an explicit false")
# v1163 (Frank 18.10.: "nicht new Track, sondern Replace"): the checkbox became an explicit choice
# "new render -> new track | replace" -- the same two modes, the same default (a missing key = on = new track)
_need('export const AUTO_CHOICES = [["track", "new track"], ["render", "one track per render"], ["replace", "replace"]];' in js
      and 'autoChk.addEventListener("change", () => _setAuto(node, autoChk.value));' in js,
      "B3 the choice: new track (first, the default) | replace -> _setAuto")
_need('if (st.autoChk) st.autoChk.value = autoMode(editOf(node));' in js, "B3 the choice mirrors autoOn")
j = js.find("async function _autoTrack1(node, rel, scene, nFiles, sweep)")   # AMENDED v1198: + the sweep
body = js[j:j + 3000]   # AMENDED v1198: the render branch made the body longer
_need("if (!autoOn(e)) return;" in body, "B3 _autoTrack1 gates on autoOn")
k1, k2 = body.find("if (!(e.takes && e.takes[key]))"), body.find("_renderTrack(node, rel")
_need(0 < k1 < k2 and "e.takes[key] = rel;" in body[k1:k2] and "return;" in body[k1:k2],
      "B3 the first run of a place is pinned to V1 and no track is added")
# drive autoOn
tmpd = Path(tempfile.mkdtemp(prefix="v1080js_"))
(tmpd / "h.mjs").write_text('const AUTO_KEY = "auto_track";\n' + js[js.find("export function autoOn"):js.find("\n", js.find("export function autoOn"))].replace("export ", "")
                            + '\nconsole.log(JSON.stringify([autoOn({}), autoOn({auto_track: true}), autoOn({auto_track: false}), autoOn(null)]));\n', encoding="utf-8")
r = subprocess.run(["node", str(tmpd / "h.mjs")], capture_output=True, text=True, timeout=60)
_need(r.returncode == 0 and json.loads(r.stdout.strip()) == [True, True, False, True], "B3 autoOn driven: %r %r" % (r.stdout, r.stderr[-200:]))
shutil.rmtree(tmpd, ignore_errors=True)

# ---- B4 static: no copy of clips in the timeline's run path ----------------
py = (ROOT / "nodes" / "ph_cine_timeline.py").read_text(encoding="utf-8")
_need(not re.search(r"shutil\.(copy|copy2|copyfile|move)\(", py), "B4 ph_cine_timeline.py copies or moves no file")
asm = (ROOT / "nodes" / "cine_assemble.py").read_text(encoding="utf-8")
reg = asm[asm.find("def register("):asm.find("def read_primary(")]
_need("shutil" not in reg and "os.rename" not in reg and "relpath" in reg, "B4 register() stores a relative path, moves nothing")

# ---- B5 texts --------------------------------------------------------------
_need("BY PATH" in py and "optional" in py and "no copy, no second folder" in py, "B5 the node DESCRIPTION says by path / optional / no copy")
_need("stays where your Save node writes" in js or "stay where your Save node writes" in js, "B5 the project tooltip says the clips stay where the Save writes")
_need('"-> clip %s"' not in py and "-> clip " not in py, "B5 the render lines no longer promise '-> clip <prefix>'")
print("[%s] PASS -- basin by default (first run -> V1, then tracks), off is explicit, "
      "clips stay where the Save writes them, the texts say so" % NAME)
