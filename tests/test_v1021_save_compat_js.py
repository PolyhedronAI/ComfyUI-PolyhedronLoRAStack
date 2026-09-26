# -*- coding: ascii -*-
"""Guard v1021 -- the canon save path on every frontend (ph_save_compat.js).

Measured 26.09. on Core master's frontend 1.53.6: graph.serialize() no longer
calls node.serialize(); it writes widgets_values in LIVE row order itself and
then calls node.onSerialize UNBOUND. Load CLIP and CLIP Text
Encode show their rows in a display order, so a save -> reload there
poured every value into the wrong row. v1021 adds a per-instance onSerialize
closure that rebuilds widgets_values in canon order -- and does nothing on
1.49.6, where the serialize() swing already did it.

  J1  serialiseValues mirrors 1.53.6: serialize === false skipped, objects
      deep-copied, undefined -> null.
  J2  1.53.6 path: onSerialize called UNBOUND on a node in display order
      writes the CANON array and leaves the rows in display order.
  J3  1.49.6 path: a node already in canon (the swing ran) is left exactly
      as the base wrote it -- the old save stays byte for byte.
  J4  an onSerialize that was there before still runs, with the node as
      `this`; a second install does not stack; `extra` runs.
  J5  a throw inside the swing never breaks the save.
  S   both nodes install it in onNodeCreated with THEIR display flag
      and THEIR swing helpers, and import it.

Driven in node.js. Script-style: exit 0 = pass.
"""
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


# ---------------------------------------------------------------- S static
# public cut v382: Reference is not in this repo -- the two display-order
# nodes here are Load CLIP and CLIP Text Encode
WANT = {
    "ph_basics.js": ("n._plsDisplayed", "_toCanon(n, spec)", "_toDisplay(n, spec)"),
    "ph_clip_encode.js": ("n._plsDisplayed", "_canonOrder(n)", "_reorderWidgetsToDisplay(n)"),
}
for f, needles in WANT.items():
    s = (WEB / f).read_text(encoding="utf-8")
    imp = 'import { saveInCanon } from "./ph_save_compat.js";' in s
    m = re.search(r"onNodeCreated = function \(\) \{(.{0,1600}?)saveInCanon\((.{0,400}?)\}\);", s, re.S)
    body = m.group(2) if m else ""
    if imp and m and all(n in body for n in needles):
        _ok("S %s installs saveInCanon in onNodeCreated with %s" % (f, needles[0]))
    else:
        _fail("S %s: import %s, install %s, needles %s" % (f, imp, bool(m), [n for n in needles if n not in body]))

if shutil.which("node") is None:
    print("  note  node.js not found -- J1-J5 SKIPPED")
else:
    tmp = Path(tempfile.mkdtemp(prefix="v1021js_"))
    shutil.copy(WEB / "ph_save_compat.js", tmp / "m.mjs")
    (tmp / "h.mjs").write_text(r"""
import { serialiseValues, saveInCanon } from "./m.mjs";
const out = {};
const w = (name, value, extra) => Object.assign({ name, value }, extra || {});
// J1
const obj = { a: [1, 2] };
const v = serialiseValues([w("a", 1), w("btn", 0, { serialize: false }), w("b", undefined), w("c", obj)]);
out.j1 = JSON.stringify(v) === JSON.stringify([1, null, { a: [1, 2] }]) && v[2] !== obj;
// a node with CANON [x, y, z] shown as DISPLAY [z, x, y]
const CANON = ["x", "y", "z"], DISPLAY = ["z", "x", "y"];
function mk() {
  const by = { x: w("x", 1), y: w("y", "two"), z: w("z", true) };
  const n = { widgets: DISPLAY.map((k) => by[k]), disp: true, calls: 0 };
  n.swing = (order, flag) => { n.widgets = order.map((k) => n.widgets.find((q) => q.name === k)); n.disp = flag; };
  return n;
}
const inst = (n, extra) => saveInCanon(n, (q) => q.disp, (q, fn) => {
  q.calls++; q.swing(CANON, false); try { return fn(); } finally { q.swing(DISPLAY, true); } }, extra);
// J2: 1.53.6 -- display order at save time, onSerialize called UNBOUND
const a = mk(); inst(a);
const oa = { widgets_values: [true, 1, "two"] };          // what 1.53.6 wrote (live order)
const f = a.onSerialize; f(oa);
out.j2 = JSON.stringify(oa.widgets_values) === JSON.stringify([1, "two", true])
      && a.widgets.map((q) => q.name).join() === DISPLAY.join() && a.disp === true;
// J3: 1.49.6 -- the swing already ran, node is in canon while onSerialize runs
const b = mk(); inst(b); b.swing(CANON, false);
const ob = { widgets_values: [1, "two", true, null] };    // base output, trailing hole kept
b.onSerialize(ob);
out.j3 = JSON.stringify(ob.widgets_values) === JSON.stringify([1, "two", true, null]) && b.calls === 0;
// J4
const c = mk(); let seen = null;
c.onSerialize = function (o) { seen = this; o.prev = 1; };
inst(c, (q, o) => { o.extra = 2; }); inst(c);            // second install must not stack
const oc = {}; const g = c.onSerialize; g(oc);
out.j4 = seen === c && oc.prev === 1 && oc.extra === 2 && c.calls === 1;
// J5
const d = mk(); saveInCanon(d, () => true, () => { throw new Error("boom"); });
const od = { widgets_values: [9] }; let threw = false;
try { d.onSerialize(od); } catch (e) { threw = true; }
out.j5 = !threw && JSON.stringify(od.widgets_values) === "[9]";
console.log(JSON.stringify(out));
""", encoding="utf-8")
    r = subprocess.run(["node", str(tmp / "h.mjs")], capture_output=True, text=True, timeout=60)
    shutil.rmtree(tmp, ignore_errors=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if r.returncode != 0 or not line:
        _fail("J harness failed: %s" % r.stderr.strip()[-300:])
    else:
        import json
        d = json.loads(line[-1])
        labels = {
            "j1": "J1 serialiseValues: serialize:false skipped, objects copied, undefined -> null",
            "j2": "J2 1.53.6: unbound onSerialize writes CANON [1,'two',true], rows back in display order",
            "j3": "J3 1.49.6: a node already in canon is left exactly as the base wrote it",
            "j4": "J4 the earlier onSerialize runs with the node as this; no double install; extra runs",
            "j5": "J5 a throw in the swing never breaks the save",
        }
        for k, lab in labels.items():
            (_ok if d.get(k) else _fail)(lab)

print()
if FAILS:
    print("test_v1021_save_compat_js: %d FAILURE(S)" % len(FAILS))
    sys.exit(1)
print("test_v1021_save_compat_js: PASS (J1-J5, S)")
