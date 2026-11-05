"""v1287 -- Cine Timeline: every track-head button seen, "delete all tracks" next to "+ new track".

Frank 05.11. (screenshot): "at the back of the timeline buttons are still hidden under + new track; and we need a
button that deletes all tracks together -- they pile up -- right next to + new track; the node must shrink after".

Measured in the frontend (1.49.6, classic + Nodes 2.0, 4 empty tracks): the head column was 216 px, V1 needs 254 px,
an empty track 279 px -- the size pick (S/M/L/XL) and the file... button were cut. 284 px: nothing cut, also at the
node's floor width and after the workflow was saved and opened again; after "delete all tracks" the node is 1356 ->
1035 px high (the stack's rows go, fieldH -> refit).

  T1  GUT_W = 284, and the widest head (an empty picture track) fits it: padding + every head button + the chip +
      the size pick (39 px measured) + the gaps -- a button added later without room trips this
  T2  the button sits in the new-track head next to "+ new track", calls the v1198 step, and left the toolbar
  T3  the v1198 step itself is unchanged: asks first, Ctrl+Z wording, the active track back to V1
  T4  the button is dimmed while V1 is the only track
  J   (node.js) the node's height follows: fieldH with 4 tracks minus fieldH after clearTracks = exactly the four
      tracks' rows (picture + gap + sound + gap each); after clearing only the new-track row is left of the stack
"""
import json
import re
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


# ---- T1 the head column holds every button
m = re.search(r"export const GUT_W = (\d+), NEWROW_H = 22;", JS)
gut_w = int(m.group(1)) if m else 0
_need(gut_w == 284, "T1 GUT_W = 284 (was 216)")
hb_w = re.search(r'const HB = "[^"]*width:(\d+)px;', JS)
chip_w = re.search(r'function _chip\(text, act\) \{\s*const c = _el\("span", "flex:0 0 auto;width:(\d+)px;', JS)
gut_css = re.search(r'const GUT = "flex:0 0 " \+ GUT_W \+ "px;width:" \+ GUT_W \+ "px;box-sizing:border-box;display:flex;align-items:center;gap:(\d+)px;padding:0 (\d+)px 0 (\d+)px;', JS)
vh = JS[JS.find("function _vHead("):JS.find("function _aHead(")]
n_hb = vh.count("_hb(")
SEL_W = 39   # the size pick, measured in the frontend (font 10 px, "XL" the widest option)
if hb_w and chip_w and gut_css:
    gap, pr, pl = int(gut_css.group(1)), int(gut_css.group(2)), int(gut_css.group(3))
    items = n_hb + 2                                    # every _hb button + the chip + the size pick
    need = pl + pr + n_hb * int(hb_w.group(1)) + int(chip_w.group(1)) + SEL_W + gap * (items - 1)
    print("        widest head: %d buttons, needs %d px, column %d px" % (n_hb, need, gut_w))
    _need(n_hb == 8 and need <= gut_w, "T1 the widest head (lock .. x, size, file...) fits the column (%d <= %d)" % (need, gut_w))
else:
    _need(False, "T1 head geometry not found (HB / chip / GUT)")
_need("if (this.size && this.size[0] < 680 + GUT_W) this.setSize([680 + GUT_W, this.size[1]]);" in JS,
      "T1 a new node is wider by the (wider) heads")

# ---- T2 the button next to + new track
fld = JS[JS.find('const addBtn = _el("button"'):JS.find("const anchorsRow = _row(")]
_need('"\\u2715 delete all tracks"' in fld and 'clearBtn.dataset.pls = "cleartracks";' in fld
      and 'clearBtn.addEventListener("click", (e) => { e.stopPropagation(); _clearTracks(node); });' in fld
      and 'clearBtn.addEventListener("pointerdown", (e) => e.stopPropagation());' in fld,
      "T2 the button: label, tag, the v1198 step, the click stays off the canvas")
_need("newHead.append(addBtn, clearBtn);" in fld, "T2 it sits right next to + new track")
_need("sep(), autoLab, restLab, takesBtn);" in JS and JS.count('dataset.pls = "cleartracks"') == 1 and 'mk("\\u2715 tracks"' not in JS,
      "T2 one button only: it left the toolbar")
_need("tc, tl, newLane, addBtn, clearBtn, delBtn," in JS, "T2 the state holds it (for the dimming)")

# ---- T3 the step itself
ct = JS[JS.find("function _clearTracks(node) {"):JS.find("function _autoTrack(")]
_need('    if (!window.confirm("Remove all " + n + " track(s) above V1?' in ct and "Ctrl+Z brings the tracks back" in ct and "node._plsTl.active = BASE_ID;" in ct
      and "_saveEdit(node, clearTracks(e));" in ct, "T3 asks first, one undo step, V1 active again")

# ---- T4 dimmed without tracks
_need('if (st.clearBtn) { const none = !others.length; st.clearBtn.style.opacity = none ? "0.35" : "1";' in JS,
      "T4 dimmed while V1 is the only track")

# ---- J (node): the height follows the stack
node = shutil.which("node")
if node:
    T = Path(tempfile.mkdtemp(prefix="v1287js_"))
    body = (JS.replace('import { app } from "../../scripts/app.js";', "const app = { registerExtension() {} };")
              .replace('import { api } from "../../scripts/api.js";', "const api = {};")
              .replace('import { setHidden, refit } from "./ph_widget_vis.js";', "const setHidden = () => {}, refit = () => {};"))
    (T / "m.mjs").write_text(body, encoding="utf-8")
    (T / "h.mjs").write_text(r"""
import * as T from "./m.mjs";
const lay = (id) => ({ id, src: { kind: "empty" }, pieces: [], video: true, audio: true, lock: false });
const e = { layers: [lay("L4"), lay("L3"), lay("L2"), lay("L1"), { id: "base", src: { kind: "base" } }] };
const n = { properties: { [T.EDIT_KEY]: e } };
const h4 = T.fieldH(n, {}), x4 = T.stackExtra(n);
let rows = 0; for (const id of ["L1", "L2", "L3", "L4"]) { const q = T.layerHeights(n, id); rows += q.clip + T.LANE_GAP + q.audio + T.LANE_GAP; }
n.properties[T.EDIT_KEY] = T.clearTracks(e);
const h0 = T.fieldH(n, {}), x0 = T.stackExtra(n);
console.log(JSON.stringify({ h4, h0, rows, x0, x4, row: T.NEWROW_H + T.LANE_GAP, left: T.stackLayers(n.properties[T.EDIT_KEY]).map((x) => x.id) }));
""", encoding="utf-8")
    r = subprocess.run([node, str(T / "h.mjs")], capture_output=True, text=True, timeout=60)
    try:
        J = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        J = None
        print(r.stdout[-400:], r.stderr[-800:])
    if J:
        print("        fieldH %d -> %d, four tracks' rows %d" % (J["h4"], J["h0"], J["rows"]))
        _need(J["rows"] > 0 and J["h4"] - J["h0"] == J["rows"], "J the node shrinks by exactly the cleared tracks' rows")
        _need(J["x0"] == J["row"] and J["left"] == ["base"], "J after clearing only the new-track row is left (V1 stays)")
    else:
        _need(False, "J node.js run failed")
    shutil.rmtree(T, ignore_errors=True)
else:
    _need(False, "J node.js missing")

print("v1287 track heads + delete all tracks: %s" % ("PASS" if not _fails else "FAIL (%d)" % len(_fails)))
sys.exit(1 if _fails else 0)
