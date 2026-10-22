"""Guard v888 -- an inactive row is GONE, not greyed. And it comes back.

Frank, 26.08.: "Kann man diese ausgegrauten Eintraege auch komplett
verstecken ...? so sieht es unfertig aus." Four nodes changed from
grey-in-place to hide-and-refit. The whole risk of that change sits in two
places, so both are EXECUTED in node, against the real extracted functions:

  V1  THE MECHANICS (ph_widget_vis.js): a hidden widget carries the TYPE
      marker (LiteGraph skips by TYPE -- `hidden` alone leaves canvas rows
      closing the gap while DOM elements stay put, the v753 overlap), reports
      zero height, and on SHOW gets its exact former type back and its
      computeSize DELETED when it never had one (writing `undefined` back
      leaves a row zero-height forever). setHidden is idempotent and reports
      whether the layout really moved.

  V2  THE LOAD-BEARING PROMISE: the widget STAYS IN node.widgets, in its
      slot, with its value. ComfyUI serialises widgets_values BY INDEX --
      removing a row would renumber every saved workflow (#577). Driven by
      hiding a middle row and comparing the whole values array before/after,
      and again after a round trip.

  V3  THE SAMPLER, really run: in Single the DUAL_ONLY rows are gone from
      the layout and present in the array; switching to High + Low brings
      them back with their ORIGINAL types and labels -- no INACTIVE_MARK
      baked in.

  V4  THE POWER UPSCALE, really run: same, and the refit is called ONCE per
      pass (not once per widget) and ONLY when something actually moved --
      a mode switch that changes nothing must not resize the node.

  V5  WIDTH IS NEVER TOUCHED (the v531 law): refit sets height only, so a
      node the user widened stays that width.

  V6  EVERY MIGRATED FILE imports the ONE module -- no fifth copy of the
      hide helper (the v755 lesson, whose announced consolidation this is).

  V7  MUTATIONS: hole each promise, demand red.

Script-style: exit 0 = pass.
"""
import os
import re
import subprocess
import sys
import tempfile

NAME = "v888"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JSDIR = os.path.join(ROOT, "web", "js")
VIS = os.path.join(JSDIR, "ph_widget_vis.js")

MIGRATED = ("uls_sampler.js", "ph_power_upscale.js",
            "ph_empty_latent.js", "ph_save.js")

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


def _grab(src, pattern, what):
    m = re.search(pattern, src, re.S)
    if not m:
        _fail("could not extract %s" % what)
    return m.group(0)


def _run(js):
    """Run a harness in node. Returns (rc, stdout+stderr)."""
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(js)
        path = fh.name
    try:
        p = subprocess.run(["node", path], capture_output=True, text=True,
                           timeout=60)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    finally:
        os.unlink(path)


# The module is plain ESM with no ComfyUI imports -- it can be imported as is.
VIS_SRC = _read(VIS)

# A LiteGraph-ish node: widgets in an array, computeSize summing the rows that
# are actually laid out. Written from the LAYOUT RULE, not from our code: a row
# whose type carries the marker contributes its computeSize height.
NODE_STUB = """
const ROW_H = 20;
function mkNode(names) {
    const node = {
        size: [400, 0],
        widgets: names.map((n, i) => ({ name: n, type: "number", value: i })),
        setDirtyCanvas() {},
        computeSize() {
            let h = 30;
            for (const w of this.widgets) {
                h += w.computeSize ? w.computeSize()[1] : ROW_H;
            }
            // A NATURAL width, deliberately NOT the current one: LiteGraph's
            // computeSize reports what the node needs, and a refit that read
            // the width from it would collapse a node the user widened. With
            // the stub echoing size[0] that mistake would be invisible.
            return [210, h];
        },
        setSize(s) { this.sizeCalls.push(s.slice()); this.size = s.slice(); },
        sizeCalls: [],
    };
    node.size = [400, node.computeSize()[1]];
    node.sizeCalls = [];
    return node;
}
function laidOut(node) {
    return node.widgets.filter((w) => !String(w.type).startsWith("pls-hidden-"))
                       .map((w) => w.name);
}
function values(node) { return node.widgets.map((w) => w.value); }
function ok(c, m) { if (!c) { console.error("ASSERT " + m); process.exit(1); } }
"""


# ---------------------------------------------------------------------------
# V1 / V2 / V5 -- the module itself
# ---------------------------------------------------------------------------
def v1v2v5(vis_src=None):
    src = vis_src if vis_src is not None else VIS_SRC
    js = src.replace("export function", "function").replace("export const", "const")
    js += NODE_STUB + """
const node = mkNode(["a", "b", "c", "d"]);
const before = values(node).join(",");
const h0 = node.computeSize()[1];

// V1: hide the MIDDLE row
ok(setHidden(node.widgets[1], true) === true, "setHidden must report the move");
ok(String(node.widgets[1].type).startsWith("pls-hidden-"),
   "a hidden widget must carry the TYPE marker - LiteGraph skips by type");
ok(node.widgets[1].computeSize()[1] < 0 || node.widgets[1].computeSize()[1] === 0,
   "a hidden widget must claim no height");
ok(laidOut(node).join(",") === "a,c,d", "the row is still laid out: " + laidOut(node));
ok(node.computeSize()[1] < h0, "the node must get shorter");
ok(setHidden(node.widgets[1], true) === false, "setHidden must be idempotent");

// V2: the array, the slot and the value are UNTOUCHED
ok(node.widgets.length === 4, "a hidden widget must STAY in node.widgets");
ok(node.widgets[1].name === "b", "it must keep its SLOT");
ok(values(node).join(",") === before,
   "widgets_values must be untouched - ComfyUI serialises BY INDEX (#577)");

// V1b: show it again -- exact type back, computeSize DELETED
ok(setHidden(node.widgets[1], false) === true, "show must report the move");
ok(node.widgets[1].type === "number", "the exact former type must come back");
ok(!Object.prototype.hasOwnProperty.call(node.widgets[1], "computeSize"),
   "computeSize must be DELETED on show, never set to undefined");
ok(node.computeSize()[1] === h0, "the node must return to its former height");
ok(values(node).join(",") === before, "a round trip must not touch the values");

// a widget that HAD a computeSize keeps it across a round trip
const w = node.widgets[2];
const own = () => [0, 33];
w.computeSize = own;
setHidden(w, true); setHidden(w, false);
ok(w.computeSize === own, "an own computeSize must survive the round trip");

// V5: refit touches HEIGHT only
node.size = [777, node.computeSize()[1] + 50];
node.sizeCalls = [];
setHidden(node.widgets[0], true);
refit(node);
ok(node.sizeCalls.length === 1, "refit must set the size once");
ok(node.sizeCalls[0][0] === 777,
   "refit must NOT touch the width (the v531 law) - got " + node.sizeCalls[0][0]);
console.log("V1/V2/V5 ok");
"""
    rc, out = _run(js)
    if rc:
        _fail("module harness: %s" % out.strip().splitlines()[-1:] or out)
    _need("V1/V2/V5 ok" in out, "module harness did not complete: %s" % out)


# ---------------------------------------------------------------------------
# V3 / V4 -- the two nodes Frank photographed, really run
# ---------------------------------------------------------------------------
def _node_harness(js_path, apply_fn_name, list_const, extra_consts,
                  drive, vis_src=None, node_src=None):
    src = node_src if node_src is not None else _read(js_path)
    vis = (vis_src if vis_src is not None else VIS_SRC) \
        .replace("export function", "function").replace("export const", "const")
    parts = [vis, NODE_STUB]
    for const in [list_const] + list(extra_consts):
        parts.append(_grab(src, r"const %s = \[[\s\S]*?\];" % const, const))
    parts.append(_grab(src, r"function _findWidget\(node, name\) \{[\s\S]*?\n\}",
                       "_findWidget"))
    # _applyModeState reads the link state; lift the REAL reader when the file
    # has one (the sampler does), never stub it -- a stub would be a second
    # implementation, and two implementations are two truths.
    m_link = re.search(r"function _inputLinked\(node, name\) \{[\s\S]*?\n\}", src)
    if m_link:
        parts.append(m_link.group(0))
    # v1212 AMENDED (declared): the Power Upscale's _applyModeState also reads
    # whether a curve is wired and labels denoise -- lift the REAL helpers when
    # the file has them (same rule as _inputLinked above: never a stub)
    for pat in (r"function _curveWired\(node\) \{[\s\S]*?\n\}",
                r"function _setSuffix\(w, suffix\) \{[\s\S]*?\n\}",
                r"async function _curveLabels\(node, wired, dual\) \{[\s\S]*?\n\}"):
        m_c = re.search(pat, src)
        if m_c:
            parts.append(m_c.group(0))
    parts.append(_grab(src, r"(?:let _visMoved = false;\n\n)?function _setDisabled\(w, disabled\) \{[\s\S]*?\n\}",
                       "_setDisabled"))
    m = re.search(r"function _visChanged\(_node\) \{[\s\S]*?\n\}", src)
    if m:
        parts.append(m.group(0))
    parts.append(_grab(src, r"function %s\(node\) \{[\s\S]*?\n\}" % apply_fn_name,
                       apply_fn_name))
    parts.append(drive)
    return _run("\n".join(parts))


def v3(vis_src=None, node_src=None):
    drive = """
const DUAL = DUAL_ONLY.slice();
const node = mkNode(["dual_moe"].concat(DUAL).concat(SINGLE_ONLY));
node.widgets[0].type = "toggle"; node.widgets[0].value = false;   // Single
node.inputs = [];
// handoff_mode must name the parity path, or the v544 rule legitimately keeps
// scheduler_low / sigma_shift_low hidden even in High + Low. That is the node
// telling the truth, not a fault -- the fixture has to state the mode it means.
const hm = node.widgets.find((w) => w.name === "handoff_mode");
if (hm) hm.value = "Wan MoE parity";
const before = values(node).join(",");
const tall = node.computeSize()[1];

_applyModeState(node);
for (const n of DUAL) {
    ok(!laidOut(node).includes(n), "Single must HIDE " + n + ": " + laidOut(node));
    ok(node.widgets.some((w) => w.name === n), n + " left node.widgets");
}
ok(values(node).join(",") === before, "hiding must not touch widgets_values");
ok(node.computeSize()[1] < tall, "the node must get shorter in Single");

// back to High + Low: every row returns, with its ORIGINAL type and label
node.widgets[0].value = true;
_applyModeState(node);
for (const n of DUAL) {
    const w = node.widgets.find((x) => x.name === n);
    ok(laidOut(node).includes(n), "High+Low must SHOW " + n);
    ok(w.type === "number", n + " came back with type " + w.type);
    ok(!String(w.label || "").includes("\\u2298"),
       n + " came back wearing the old INACTIVE_MARK: " + w.label);
}
// SINGLE_ONLY rows are hidden in High + Low by construction, so the node is
// NOT back to the Single height -- it is back to the High+Low height. The
// promise worth pinning is that nothing is stuck hidden: every row is either
// laid out or hidden by a rule that names it.
const stillHidden = node.widgets.filter((w) =>
    String(w.type).startsWith("pls-hidden-")).map((w) => w.name);
for (const n of stillHidden) {
    ok(SINGLE_ONLY.includes(n),
       n + " is stuck hidden with no rule to hide it: " + stillHidden);
}
console.log("V3 ok");
"""
    rc, out = _node_harness(os.path.join(JSDIR, "uls_sampler.js"),
                            "_applyModeState", "DUAL_ONLY",
                            ["SINGLE_ONLY", "SIGMA_INERT"], drive,
                            vis_src, node_src)
    if rc:
        _fail("sampler harness: %s" % out.strip())
    _need("V3 ok" in out, "sampler harness did not complete: %s" % out)


def v4(vis_src=None, node_src=None):
    drive = """
const node = mkNode(["dual_moe"].concat(DUAL_ONLY));
node.widgets[0].type = "toggle"; node.widgets[0].value = false;
const before = values(node).join(",");

node.sizeCalls = [];
_applyModeState(node);
for (const n of DUAL_ONLY) ok(!laidOut(node).includes(n), "Single must hide " + n);
ok(values(node).join(",") === before, "hiding must not touch widgets_values");
ok(node.sizeCalls.length === 1,
   "ONE refit per pass, not one per widget - got " + node.sizeCalls.length);

// a pass that changes NOTHING must not resize the node at all
node.sizeCalls = [];
_applyModeState(node);
ok(node.sizeCalls.length === 0,
   "a no-op pass must not refit - got " + node.sizeCalls.length);
console.log("V4 ok");
"""
    rc, out = _node_harness(os.path.join(JSDIR, "ph_power_upscale.js"),
                            "_applyModeState", "DUAL_ONLY", ["RESTORE_ONLY", "CURVE_UNREAD"], drive,   # AMENDED v1199: the restore dials' list; v1212: the curve's
                            vis_src, node_src)
    if rc:
        _fail("upscale harness: %s" % out.strip())
    _need("V4 ok" in out, "upscale harness did not complete: %s" % out)


# ---------------------------------------------------------------------------
# V6 -- one module, no fifth copy
# ---------------------------------------------------------------------------
def v6():
    for name in MIGRATED:
        src = _read(os.path.join(JSDIR, name))
        _need('from "./ph_widget_vis.js"' in src,
              "%s must import the ONE visibility module - a private copy is "
              "the drift the v755 note announced this cut would end" % name)
        _need("setHidden" in src,
              "%s imports the module but never hides anything" % name)
        _need(not re.search(r"function\s+setHidden\s*\(", src),
              "%s defines its OWN setHidden - that is the fifth copy" % name)
    _need('export function setHidden' in VIS_SRC and 'export function refit' in VIS_SRC,
          "the module must export both halves")


# ---------------------------------------------------------------------------
# V7 -- mutations
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


def v7():
    n = 0
    # the marker goes, only `hidden` remains -> LiteGraph still lays the row out
    n += bool(_mut(v1v2v5, VIS,
                   "        w.type = HIDDEN_PREFIX + w.type;",
                   "        w.hidden = true;",
                   "hide by flag instead of by TYPE"))
    # computeSize written back as undefined instead of deleted
    n += bool(_mut(v1v2v5, VIS,
                   "        else delete w.computeSize;",
                   "        else w.computeSize = undefined;",
                   "computeSize undefined instead of deleted"))
    # refit touches the width
    n += bool(_mut(v1v2v5, VIS,
                   "        node.setSize([node.size[0], node.computeSize()[1]]);",
                   "        node.setSize([node.computeSize()[0], node.computeSize()[1]]);",
                   "refit touches the width"))
    # the widget is REMOVED from the array -> serialisation renumbers
    n += bool(_mut(v1v2v5, VIS,
                   "        w.type = HIDDEN_PREFIX + w.type;\n        w.computeSize = () => [0, -4];",
                   "        w.type = HIDDEN_PREFIX + w.type;\n        w.computeSize = () => [0, ROW_H];",
                   "a hidden row still claims its height"))
    # the sampler bakes the old marker into the label again
    n += bool(_mut(lambda s: v3(node_src=s),
                   os.path.join(JSDIR, "uls_sampler.js"),
                   "    w.label = w._uls_baseLabel;",
                   "    w.label = disabled ? INACTIVE_MARK + w._uls_baseLabel : w._uls_baseLabel;",
                   "INACTIVE_MARK bakes into a row that comes back"))
    # the upscale refits per widget instead of per pass
    n += bool(_mut(lambda s: v4(node_src=s),
                   os.path.join(JSDIR, "ph_power_upscale.js"),
                   # v1212 AMENDED (declared): the DUAL_ONLY row now reads `off` (dual OR a wired curve)
                   "        if (_setDisabled(_findWidget(node, name), off)) moved = true;",
                   "        if (_setDisabled(_findWidget(node, name), off)) refit(node);",
                   "refit once per widget instead of once per pass"))
    return n


def main():
    v1v2v5()
    v3()
    v4()
    v6()
    muts = v7()
    print("PASS: %s -- %d promises pinned, %d mutations caught (hide by type, "
          "slot and value kept, both nodes driven through a real mode switch, "
          "one refit per pass, height only, one module)"
          % (NAME, _checks, muts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
