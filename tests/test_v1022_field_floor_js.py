# -*- coding: ascii -*-
"""Guard v1022 -- one text-field floor for both renderers (ph_widget_vis.js).

P1 on 26.09. (1.49.6 and 1.53.6): MiniMax Keyframes and MiniMax Reference
carry two text fields since v1017 (prompt + tags). Classic gave each 38 px
(one line), Nodes 2.0 gives every textarea 64 px -> class F5 on both nodes.
v1022 puts a 64 px floor on the classic field through the DOM widget's own
layout hook; after it P1 reports NO finding on either node, on either
frontend (classic N 328 / 2.0 332 and 388 / 392).

  F1  fieldFloor asks for 64 + the measured 12 px element margin; a larger
      earlier getMinHeight still wins; non-textarea widgets are left alone;
      a second install does not stack.
  F2  while the field's input is WIRED it keeps one 24 px row (min AND max)
      for its socket dot -- no field-sized gap (v1021: ~50 px), and the dot
      does not land on the next row (it did at 0 px); unwired it is back.
      Both input spellings are recognised (1.49.6 name, 1.53.6 widget.name).
  F3  growToFloor grows a node from an old save to its minimum and never
      shrinks one the user made taller; the width is never touched (v531).
  S   Keyframes installs it for prompt + tags in onNodeCreated AND
      onConfigure and grows the node; Reference installs it in onNodeCreated
      (its tidy() refits on every change).

Driven in node.js. Script-style: exit 0 = pass.
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "js"
FAILS = []


def _fail(m):
    FAILS.append(m)
    print("  FAIL  " + m)


def _ok(m):
    print("  ok    " + m)


# ------------------------------------------------------------------ S static
kf = (WEB / "ph_minimax_keyframes.js").read_text(encoding="utf-8")
mr = (WEB / "ph_minimax_ref.js").read_text(encoding="utf-8")
imp_kf = re.search(r'import \{[^}]*\bfieldFloor\b[^}]*\bgrowToFloor\b[^}]*\} from "\./ph_widget_vis\.js"', kf)
n_kf = kf.count('fieldFloor(this, ["prompt", "tags"])')
if imp_kf and n_kf == 2 and kf.count("growToFloor(this)") == 2:
    _ok("S Keyframes: floor for prompt + tags in onNodeCreated and onConfigure, node grows to it")
else:
    _fail("S Keyframes: import %s, installs %d, grows %d" % (bool(imp_kf), n_kf, kf.count("growToFloor(this)")))
imp_mr = re.search(r'import \{[^}]*\bfieldFloor\b[^}]*\} from "\./ph_widget_vis\.js"', mr)
m = re.search(r"onNodeCreated = function \(\) \{(.{0,300}?)setTimeout\(\(\) => tidy\(this\), 0\);", mr, re.S)
if imp_mr and m and 'fieldFloor(this, ["prompt", "tags"])' in m.group(1):
    _ok("S Reference: floor for prompt + tags in onNodeCreated, before the first tidy()")
else:
    _fail("S Reference: import %s, install before tidy %s" % (bool(imp_mr), bool(m)))

if shutil.which("node") is None:
    print("  note  node.js not found -- F1-F3 SKIPPED")
else:
    tmp = Path(tempfile.mkdtemp(prefix="v1022js_"))
    shutil.copy(WEB / "ph_widget_vis.js", tmp / "m.mjs")
    (tmp / "h.mjs").write_text(r"""
import { fieldFloor, growToFloor, fieldWired, FIELD_MIN_PX, DOM_FIELD_MARGIN, WIRED_ROW_PX } from "./m.mjs";
const out = {};
const ta = (name, opts) => ({ name, element: { tagName: "TEXTAREA" }, options: opts || {} });
function mk() {
  const node = { widgets: [ta("prompt"), ta("tags", { getMinHeight: () => 200 }),
                           { name: "frame_fit", element: null, options: {} },
                           { name: "view", element: { tagName: "DIV" }, options: {} }],
                 inputs: [], size: [400, 100] };
  return node;
}
const n = mk(); fieldFloor(n, ["prompt", "tags", "frame_fit", "view"]); fieldFloor(n, ["prompt"]);
const W = (k) => n.widgets.find((w) => w.name === k).options;
out.consts = [FIELD_MIN_PX, DOM_FIELD_MARGIN, WIRED_ROW_PX];
out.f1 = W("prompt").getMinHeight() === 76 && W("tags").getMinHeight() === 200
      && W("prompt").getMaxHeight() === undefined
      && W("frame_fit").getMinHeight === undefined && W("view").getMinHeight === undefined
      && W("prompt")._plsFloor === true;
// double install must not wrap the wrapper
const n2 = mk(); fieldFloor(n2, ["prompt"]); const g1 = n2.widgets[0].options.getMinHeight;
fieldFloor(n2, ["prompt"]); out.f1b = n2.widgets[0].options.getMinHeight === g1;
// F2 wired, both spellings
n.inputs = [{ name: "prompt", link: 7 }];
const a = [W("prompt").getMinHeight(), W("prompt").getMaxHeight(), W("tags").getMinHeight()];
n.inputs = [{ name: "x", widget: { name: "prompt" }, link: 3 }];
const b = [W("prompt").getMinHeight(), W("prompt").getMaxHeight()];
n.inputs = [{ name: "prompt", link: null }];
const c = [W("prompt").getMinHeight(), W("prompt").getMaxHeight()];
out.f2 = JSON.stringify([a, b, c]) === JSON.stringify([[24, 24, 200], [24, 24], [76, null]])
      && fieldWired({ inputs: [{ name: "tags", link: 1 }] }, "tags") && !fieldWired({}, "tags");
out.f2raw = [a, b, c];
// F3 grow-only, width untouched
const mkN = (h, minH) => ({ size: [333, h], computeSize: () => [100, minH],
  setSize(s) { this.size = s; }, setDirtyCanvas() {} });
const small = mkN(246, 298); growToFloor(small);
const tall = mkN(500, 298); growToFloor(tall);
out.f3 = small.size[0] === 333 && small.size[1] === 298 && tall.size[0] === 333 && tall.size[1] === 500;
console.log(JSON.stringify(out));
""", encoding="utf-8")
    r = subprocess.run(["node", str(tmp / "h.mjs")], capture_output=True, text=True, timeout=60)
    shutil.rmtree(tmp, ignore_errors=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if r.returncode != 0 or not line:
        _fail("F harness failed: %s" % r.stderr.strip()[-300:])
    else:
        d = json.loads(line[-1])
        if d["consts"] == [64, 12, 24]:
            _ok("F1 floor 64 px + 12 px margin, wired row 24 px (the measured numbers)")
        else:
            _fail("F1 constants %r" % d["consts"])
        (_ok if d["f1"] else _fail)("F1 prompt asks 76; a larger earlier floor (200) wins; non-textareas untouched")
        (_ok if d["f1b"] else _fail)("F1 a second install does not stack")
        (_ok if d["f2"] else _fail)("F2 wired (either spelling): one 24 px row, min AND max; unwired: back to 76 %r"
                                    % (d["f2raw"],))
        (_ok if d["f3"] else _fail)("F3 grow-only to the floor, a taller node keeps its size, width never touched")

print()
if FAILS:
    print("test_v1022_field_floor_js: %d FAILURE(S)" % len(FAILS))
    sys.exit(1)
print("test_v1022_field_floor_js: PASS (F1-F3, S)")
