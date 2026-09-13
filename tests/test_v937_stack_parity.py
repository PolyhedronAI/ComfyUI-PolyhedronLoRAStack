#!/usr/bin/env python3
"""v937 -- S2: the Stack's Nodes 2.0 panel reaches parity THROUGH SHARED CODE.

The four open points of the concept (section 4): the trigger button, the
stack-order number in the group pill, the conflict warnings, the weight-header
explainer. The painted Stack kept the first two as inline blocks in its
onMouseDown and the explainer as a literal in its draw code. They moved
VERBATIM into named pieces of uls_node.js (insertRowTrigger,
openStackOrderInput, WEIGHT_HDR_TIP_LINES; checkConflicts already was one),
the painted path calls them, and the panel CALLS the same pieces. A second
copy would be a second truth and drift the first time either is fixed.

  R  the REAL render() of uls_stack_dom.js, driven in Node.js against
     recording stubs: which shared piece a click calls, with which row; the
     three badge states and where there is none; the badge click stays out of
     the group dialog and redraws through its onChange; the warnings; the tip
  S  static (code only): the painted path CALLS the shared pieces; every
     moved text exists ONCE in the tree; the order input tells its caller after
     every state change; the panel rebuilds nothing
Both renderers are driven in a real browser by tools/browser_probe_stack.py
(sections H and I).
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOM = os.path.join(ROOT, "web", "js", "uls_stack_dom.js")
NODE = os.path.join(ROOT, "web", "js", "uls_node.js")

HEAD = r"""
function mkEl(tag) {
  const el = { tagName: String(tag).toUpperCase(), className: "", title: "", children: [],
    _text: "", style: {}, type: "", checked: false,
    appendChild(c) { this.children.push(c); c.parentNode = this; return c; } };
  Object.defineProperty(el, "textContent", { get() { return el._text; },
    set(v) { el._text = v; if (v === "") el.children = []; } });
  return el;
}
globalThis.document = { createElement: mkEl, head: mkEl("head") };
const calls = [];
const TIP = ["Weight / CLIP Strength", "Click: model weight.  Shift+Click: set a per-LoRA", "x"];
let CONFLICTS = [];
let SPEC = null;
const registerVueView = (cls, spec) => { SPEC = spec; };
const openLoraSelect = () => {}, showWeightInput = () => {}, loadLoraList = () => Promise.resolve();
const newRow = () => ({ enabled: true, name: "None", wLow: 1, wHigh: 1, group: "\u2014" });
const getLoraList = () => [], applyNorm = (x) => x || "auto", applyNext = (x) => x, openPreviewOverlay = () => {};
const showGroupModePopup = () => calls.push(["groupPopup"]);
const insertRowTrigger = (row, e) => calls.push(["trigger", row.name]);
const openStackOrderInput = (node, row, e, cb) => calls.push(["order", row.group, typeof cb, cb]);
const checkConflicts = (rows) => { calls.push(["conflicts", rows.length]); return CONFLICTS; };
const WEIGHT_HDR_TIP_LINES = TIP;
// v949: the shared palette is imported too
const GROUP_COLORS = { subject: "#ff6b9d", style: "#8b6fe8", "\u2014": "#404050" };
const APPLY_INFO = { auto: { label: "Auto", color: "#9a9aaa" }, bypass: { label: "Bypass", color: "#7af0c0" }, patch: { label: "Baked", color: "#f0c87a" } };
"""

TAIL = r"""
function walk(el, f) { f(el); for (const c of el.children || []) walk(c, f); }
function find(root, cls) { const out = []; walk(root, e => { if ((" " + e.className + " ").includes(" " + cls + " ")) out.push(e); }); return out; }
const node = { _uls: { flatMode: false, apply: "auto", groupOrder: { subject: 3 }, rows: [
  { enabled: true, name: "a.safetensors", group: "subject", wLow: 1, wHigh: 1 },
  { enabled: true, name: "b.safetensors", group: "style", wLow: 1, wHigh: 1 },
  { enabled: true, name: "c.safetensors", group: "\u2014", wLow: 1, wHigh: 1 } ] },
  _ulsSync() {}, setDirtyCanvas() {} };
CONFLICTS = [{ row: 1, level: "warn", msg: "ROWMSG" }, { row: -1, level: "warn", msg: "GLOBALMSG" },
             { row: -1, level: "info", msg: "INFOMSG" }];
const root = mkEl("div");
const out = {};
SPEC.render(node, root);
out.trig = find(root, "uls-dom-trig").length;
const t1 = find(root, "uls-dom-trig")[1];
if (t1 && typeof t1.onclick === "function") t1.onclick({});
out.trig_call = calls.filter(c => c[0] === "trigger").map(c => c[1]);
const badges = find(root, "uls-dom-obadge");
out.badges = badges.map(b => [b.className, b.textContent]);
let stopped = false;
calls.length = 0;
badges[1].onclick({ stopPropagation() { stopped = true; } });
out.badge_stop = stopped;
out.badge_calls = calls.map(c => c.slice(0, 3));
node._uls.groupOrder.style = 7;
const oc = calls.find(c => c[0] === "order");
if (oc && typeof oc[3] === "function") oc[3]();        // the onChange the panel passed
out.after_cb = find(root, "uls-dom-obadge").map(b => b.textContent);
node._uls._orderConflictGroup = "style"; SPEC.render(node, root);
out.conflict = find(root, "uls-dom-obadge").map(b => [b.className, b.textContent]);
node._uls._orderConflictGroup = null;
const rw = find(root, "uls-dom-rwarn");
out.rwarn = rw.map(w => w.title);
// WHERE it sits, not only that it exists (mutation round: a warning moved to
// the neighbouring row passed a titles-only check)
out.rwarn_rows = find(root, "uls-dom-row").map(r => find(r, "uls-dom-rwarn").map(w => w.title));
out.gwarn = find(root, "uls-dom-gwarn").map(w => [w.className, w.textContent]);
out.conf_calls = calls.filter(c => c[0] === "conflicts").map(c => c[1]);
out.tip = (find(root, "uls-dom-whdr")[0] || {}).title;
node._uls.flatMode = true; SPEC.render(node, root);
out.flat_badges = find(root, "uls-dom-obadge").length;
console.error(JSON.stringify(out));
"""

failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def code_only(src):
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    src = re.sub(r"^\s*//.*$", "", src, flags=re.M)
    return re.sub(r"\s//.*$", "", src, flags=re.M)


def body_of(src, head):
    """The brace-balanced body that follows the first `head` in src."""
    i = src.index(head)
    j = src.index("{", i)
    depth = 0
    for k in range(j, len(src)):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                return src[j:k + 1]
    return src[j:]


def main():
    print("v937: Stack panel parity through shared code")
    if shutil.which("node") is None:
        print("  FAIL node (Node.js) not found -- this guard needs it")
        return 1
    dom = open(DOM, encoding="utf-8").read()
    node = open(NODE, encoding="utf-8").read()

    # ---- R: the real render(), driven -----------------------------------
    body = dom.split('} from "./uls_node.js";', 1)[1]
    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, "h.mjs")
        open(path, "w", encoding="utf-8").write(HEAD + body + TAIL)
        p = subprocess.run(["node", path], capture_output=True, text=True, timeout=60)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    lines = [ln for ln in p.stderr.splitlines() if ln.startswith("{")]
    if p.returncode != 0 or not lines:
        check(False, "harness ran: " + (p.stderr or p.stdout)[-500:])
        return 1
    r = json.loads(lines[-1])
    check(r["trig"] == 3, "R  one \u21b5 per row (%s)" % r["trig"])
    check(r["trig_call"] == ["b.safetensors"],
          "R  \u21b5 calls the SHARED insertRowTrigger with ITS row")
    check(r["badges"] == [["uls-dom-obadge set", "3"], ["uls-dom-obadge", ""]],
          "R  badge: gold with the number / dashed when unset / none for the "
          "no-group row (%s)" % r["badges"])
    check(r["badge_stop"] and r["badge_calls"] == [["order", "style", "function"]],
          "R  badge click calls the SHARED openStackOrderInput with an onChange, "
          "and never the group dialog")
    check(r["after_cb"] == ["3", "7"], "R  the onChange redraws the panel (%s)" % r["after_cb"])
    check(r["conflict"][1] == ["uls-dom-obadge conflict", "!"],
          "R  a flashing conflict shows the red '!'")
    check(r["rwarn"] == ["ROWMSG"] and r["rwarn_rows"] == [[], ["ROWMSG"], []],
          "R  the row warning sits on ITS row, message as tooltip (%s)" % r["rwarn_rows"])
    check(r["gwarn"] == [["uls-dom-gwarn warn", "GLOBALMSG"], ["uls-dom-gwarn", "INFOMSG"]],
          "R  global warnings under the list, warn vs info (%s)" % r["gwarn"])
    check(r["conf_calls"] and all(n == 3 for n in r["conf_calls"]),
          "R  the warnings come from the SHARED checkConflicts over the node's rows")
    check(r["tip"] == "Weight / CLIP Strength\nClick: model weight.  Shift+Click: set a per-LoRA\nx",
          "R  the weight-header explainer is the SHARED lines, joined")
    check(r["flat_badges"] == 0, "R  flat mode: no order badge (as in the painted pill)")

    # ---- S: static --------------------------------------------------------
    nc = code_only(node)
    stack_md = body_of(nc, "nodeType.prototype.onMouseDown = function")
    check(re.search(r"\binsertRowTrigger\(row, e\)", stack_md) is not None,
          "S  the painted Stack CALLS insertRowTrigger(row, e)")
    check(re.search(r"\bopenStackOrderInput\(this, row, e\)", stack_md) is not None,
          "S  the painted Stack CALLS openStackOrderInput(this, row, e)")
    check("deriveTriggers(" not in stack_md and "Stack Order" not in stack_md,
          "S  the inline copies are gone from the painted onMouseDown")
    check(node.count("Stack Order  (1\u20138,  0 = clear)") == 1,
          "S  the order input exists ONCE in the tree")
    check(node.count('"Click: model weight.  Shift+Click: set a per-LoRA",') == 1
          and nc.count("const lines = WEIGHT_HDR_TIP_LINES") == 2,
          "S  the explainer text exists ONCE; both painted headers (Stack, Engine) use it")
    oi = body_of(nc, "function openStackOrderInput(")
    syncs = [ln for ln in oi.split("\n") if "node._ulsSync()" in ln]
    check(len(syncs) == 4 and all("onChange?.()" in ln for ln in syncs),
          "S  the order input tells its caller after EVERY state change (%d)" % len(syncs))
    check("this." not in oi, "S  the moved order input no longer depends on `this`")
    m = re.search(r"export\s*\{([^}]*)\}", node)
    block = re.sub(r"//[^\n]*", "", m.group(1) if m else "")   # comments are not names
    exported = {x.strip() for x in block.replace("\n", " ").split(",")}
    for name in ("insertRowTrigger", "openStackOrderInput", "checkConflicts",
                 "WEIGHT_HDR_TIP_LINES"):
        check(name in exported, "S  uls_node.js exports %s" % name)
    dc = code_only(dom)
    for bad in ("deriveTriggers", "insertTriggerAtCursor", "Stack Order",
                "Two style LoRAs", "Total weight sum", "Shift+Click"):
        check(bad not in dc, "S  the panel rebuilds nothing: no %r in its code" % bad)
    return 1 if failures else 0


if __name__ == "__main__":
    rc = main()
    print("v937: %s" % ("FAIL (%d)" % len(failures) if failures else "PASS"))
    sys.exit(rc)
