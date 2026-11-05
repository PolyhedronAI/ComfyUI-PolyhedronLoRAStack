"""v1288 -- Cine Timeline: a V1 pin that is not a take of this project no longer blocks the basin.

Frank 05.11.: a workflow copied for a new project (hippie 60) kept V1 pinned to a clip of another project
(2026-10-29/PH_100548_char_s17_whole_00001_.mp4). The server ignores such a pin (V1 shows the newest run of the
project), but the basin ("new render -> new track") took any pin as valid: the first run of the new project went on a
track AND showed on V1, and every later run moved V1 along instead of V1 keeping its clip.

  P1  pinValid: no pin -> false; no plan loaded -> true (trust, never overwrite a valid pin early); the pin among the
      place's takes -> true; a pin of another project / another place -> false; whole mode (k 0) and scene keys
  P2  the auto-track path pins this run to V1 when the old pin is stale, and says so
  P3  "delete all tracks" still leaves V1 and its pin alone (the v1198 promise "V1 stays as it is")
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JS = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")
_fails = []


def _need(cond, msg):
    if not cond:
        _fails.append(msg)
    print(("  ok    " if cond else "  FAIL  ") + msg)


# ---- P2 the auto-track path
at = JS[JS.find("async function _autoTrack1("):JS.find("// v1198: one track per render -- the render's track")]
_need("const stale = !!(e.takes && e.takes[key]) && !pinValid(node._plsTl.plan, key, e.takes[key]);" in at
      and "if (!(e.takes && e.takes[key]) || stale) {" in at and "e.takes[key] = rel;" in at
      and '" -- the old pin was not a take of this project"' in at, "P2 a stale pin: this run is pinned to V1, the state line says why")
_need(at.find("const stale") < at.find("if (!(e.takes && e.takes[key]) || stale)") and at.count("_saveEdit(node, e);") >= 1,
      "P2 the check sits before the pin decision")

# ---- P3 delete all tracks keeps V1's pin
ct = JS[JS.find("export function clearTracks(e) {"):JS.find("function _clearTracks(node) {")]
_need("o.takes" not in ct and "delete o.render_tracks;" in ct, "P3 clearTracks leaves the V1 pins alone")

# ---- P1 (node)
node = shutil.which("node")
if node:
    T = Path(tempfile.mkdtemp(prefix="v1288js_"))
    body = (JS.replace('import { app } from "../../scripts/app.js";', "const app = { registerExtension() {} };")
              .replace('import { api } from "../../scripts/api.js";', "const api = {};")
              .replace('import { setHidden, refit } from "./ph_widget_vis.js";', "const setHidden = () => {}, refit = () => {};"))
    (T / "m.mjs").write_text(body, encoding="utf-8")
    (T / "h.mjs").write_text(r"""
import * as T from "./m.mjs";
const D = { clips: [{ k: 0, takes: ["2026-11-05/HIPPIE/PH_1_a.mp4", "2026-11-05/HIPPIE/PH_0_a.mp4"] }, { k: 2, takes: ["d/s2.mp4"] }, { k: 3, takes: [] }] };
const o = {
  none: T.pinValid(D, "0", ""), noplan: T.pinValid(null, "0", "x.mp4"), noclips: T.pinValid({}, "0", "x.mp4"),
  own: T.pinValid(D, "0", "2026-11-05/HIPPIE/PH_0_a.mp4"), foreign: T.pinValid(D, "0", "2026-10-29/PH_100548_char_s17_whole_00001_.mp4"),
  scene: T.pinValid(D, "2", "d/s2.mp4"), otherplace: T.pinValid(D, "2", "2026-11-05/HIPPIE/PH_0_a.mp4"),
  empty: T.pinValid(D, "3", "d/s3.mp4"), missingplace: T.pinValid(D, "5", "d/s5.mp4"),
};
console.log(JSON.stringify(o));
""", encoding="utf-8")
    r = subprocess.run([node, str(T / "h.mjs")], capture_output=True, text=True, timeout=60)
    try:
        J = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        J = None
        print(r.stdout[-400:], r.stderr[-800:])
    if J:
        print("        " + json.dumps(J))
        _need(J["none"] is False, "P1 no pin -> not valid")
        _need(J["noplan"] is True and J["noclips"] is True, "P1 no plan loaded -> the pin is trusted (as before)")
        _need(J["own"] is True and J["scene"] is True, "P1 a take of this place -> valid (whole and scene keys)")
        _need(J["foreign"] is False and J["otherplace"] is False, "P1 a clip of another project / another place -> stale")
        _need(J["empty"] is False and J["missingplace"] is False, "P1 a place without takes -> stale")
    else:
        _need(False, "P1 node.js run failed")
    shutil.rmtree(T, ignore_errors=True)
else:
    _need(False, "P1 node.js missing")

print("v1288 foreign pin: %s" % ("PASS" if not _fails else "FAIL (%d)" % len(_fails)))
sys.exit(1 if _fails else 0)
