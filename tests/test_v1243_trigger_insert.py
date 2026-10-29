#!/usr/bin/env python3
"""
test_v1243_trigger_insert -- the LoRA Stack's trigger insert keeps the prompt field's colour and
writes the plain trigger.

Frank, 28.10.: "Wenn ich Trigger aus dem LoRA Stack hinzufuege -- die haben btw. immer noch Klammern
und ein Gewicht -- dann wird das Schwarz ... wenn ich den Tab wechsle ... ist es wieder gruen."

F  THE FLASH: insertTriggerAtCursor flashed the field by writing the background SHORTHAND and put it
   back from el.style.background. The CTE colours its fields through backgroundColor, and a shorthand
   read with only background-color set is "" -- so the restore wiped the tint (measured in Chromium:
   backgroundColor "" afterwards). v1243 flashes with box-shadow and never touches the background.
W  THE TEXT: every insert wrote "(trigger:1.00)". Encoders that do not parse weight brackets (a
   Qwen-based text encoder, e.g. MiniMax H3) keep them as literal text. v1243: plain trigger by
   default; Shift = "(trigger:weight)" with the ROW's weight (two decimals), also in the choice popup.

DRIVEN -- the REAL insertTriggerAtCursor, triggerInsertText and insertRowTrigger lifted from
uls_node.js and run in node against a stand-in element whose style object follows the browser's
shorthand rule (writing `background` resets backgroundColor; reading it gives "" while only
backgroundColor is set) and a manual timer queue:
  F1 tinted field, insert, flash over   -> backgroundColor unchanged, boxShadow back to its old value
  F2 second insert inside the flash      -> after both flashes the old boxShadow, not the flash glow
  F3 during the flash                    -> a glow is shown (feedback still there)
  W1 click                               -> "fe8rn" inserted, no brackets
  W2 Shift+click on a row with weight .35 -> "(fe8rn:0.35)"
  W3 several triggers                    -> the popup is opened with the row's weight "0.35"
STATIC -- the popup's mousedown goes through triggerInsertText(..., ev.shiftKey); its header names
          both forms; the Nodes 2.0 button tooltip names Shift.

MUTATIONS (each must turn the run red; --mutate runs them):
  M1 the old background flash restored        -> F1
  M2 triggerInsertText always weighted         -> W1
  M3 insertRowTrigger ignores Shift            -> W2
  M4 the weight fixed to 1.00 again            -> W2 / W3
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NODE_JS = os.path.join(ROOT, "web", "js", "uls_node.js")
DOM_JS = os.path.join(ROOT, "web", "js", "uls_stack_dom.js")
NAME = "test_v1243_trigger_insert"

HEAD = r"""
let timers = [];
globalThis.setTimeout = (f, ms) => { const t = { f, ms, dead: false }; timers.push(t); return t; };
globalThis.clearTimeout = (t) => { if (t) t.dead = true; };
const runTimers = () => { const q = timers; timers = []; for (const t of q) if (!t.dead) t.f(); };
class Style {
  constructor() { this._bgc = ""; this._bgshort = null; this.boxShadow = ""; }
  get backgroundColor() { return this._bgc; }
  set backgroundColor(v) { this._bgc = v; this._bgshort = null; }
  // browser rule: the shorthand serialises to "" unless it was set as a whole
  get background() { return this._bgshort === null ? "" : this._bgshort; }
  set background(v) { this._bgshort = v; this._bgc = ""; }
}
class HTMLTextAreaElement { constructor() { this.value = ""; this.isConnected = true; this.style = new Style();
  this.selectionStart = 0; this.selectionEnd = 0; } dispatchEvent() { return true; } }
class HTMLInputElement {}
globalThis.window = { HTMLTextAreaElement, HTMLInputElement };
globalThis.Event = class { constructor(t) { this.type = t; } };
let _lastFocusedTextarea = null;
let _lastCursorPos = 0;
"""

TAIL = r"""
const out = {};
// F1 / F3
const ta = new HTMLTextAreaElement(); ta.value = "PROMPT"; ta.style.backgroundColor = "#2f4f2f";
ta.style.boxShadow = "OLD"; _lastFocusedTextarea = ta; _lastCursorPos = 6;
insertTriggerAtCursor("fe8rn");
out.during = ta.style.boxShadow; out.bgDuring = ta.style.backgroundColor;
runTimers();
out.f1_bg = ta.style.backgroundColor; out.f1_shadow = ta.style.boxShadow; out.f1_val = ta.value;
// F2: two inserts inside one flash
insertTriggerAtCursor("a"); insertTriggerAtCursor("b"); runTimers(); runTimers();
out.f2_shadow = ta.style.boxShadow; out.f2_bg = ta.style.backgroundColor;
// W: insertRowTrigger with stubs
const ta2 = new HTMLTextAreaElement(); ta2.value = ""; _lastFocusedTextarea = ta2; _lastCursorPos = 0;
const metaCache = new Map(); let TRIG = ["fe8rn"]; let popup = null;
const deriveTriggers = () => TRIG; const fetchMeta = () => Promise.resolve({});
const showInsertToast = () => {}; const openTriggerSelectPopup = (t, w) => { popup = { t, w }; };
const row = { name: "x/fe8rn.safetensors", weight: 0.35 };
metaCache.set(row.name, {});
__ROWTRIGGER__
insertRowTrigger(row, { clientX: 0, clientY: 0, shiftKey: false }); runTimers();
out.w1 = ta2.value;
ta2.value = ""; _lastCursorPos = 0;
insertRowTrigger(row, { clientX: 0, clientY: 0, shiftKey: true }); runTimers();
out.w2 = ta2.value;
TRIG = ["a", "b"];
insertRowTrigger(row, { clientX: 0, clientY: 0, shiftKey: false });
out.w3 = popup ? popup.w : null;
console.error(JSON.stringify(out));
"""


def body_of(src, head):
    i = src.index(head)
    j = src.index("{", i)
    depth = 0
    for k in range(j, len(src)):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                return src[i:k + 1]
    return src[i:]


def run(src_node, src_dom):
    fails = []

    def need(c, m):
        print(("  ok   " if c else "  FAIL ") + m)
        if not c:
            fails.append(m)

    ins = body_of(src_node, "function insertTriggerAtCursor(")
    try:
        tit = body_of(src_node, "function triggerInsertText(")
    except ValueError:
        tit = ""
    rt = body_of(src_node, "function insertRowTrigger(")
    js = HEAD + ins + "\n" + tit + "\n" + TAIL.replace("__ROWTRIGGER__", rt)
    d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, "h.mjs")
        open(p, "w", encoding="utf-8").write(js)
        r = subprocess.run(["node", p], capture_output=True, text=True, timeout=60)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    lines = [ln for ln in r.stderr.splitlines() if ln.startswith("{")]
    if r.returncode != 0 or not lines:
        need(False, "harness ran: " + (r.stderr or r.stdout)[-400:])
        return fails
    o = json.loads(lines[-1])
    need(o["f1_bg"] == "#2f4f2f" and o["bgDuring"] == "#2f4f2f",
         "F1 the field keeps its tint through the flash (%r / %r)" % (o["bgDuring"], o["f1_bg"]))
    need(o["f1_shadow"] == "OLD", "F1 the box shadow is back to its old value (%r)" % o["f1_shadow"])
    need(o["f1_val"] == "PROMPT fe8rn", "F1 the text is inserted (%r)" % o["f1_val"])
    need(o["f2_shadow"] == "OLD" and o["f2_bg"] == "#2f4f2f",
         "F2 two inserts in one flash end on the old shadow and tint (%r)" % o["f2_shadow"])
    need(bool(o["during"]) and o["during"] != "OLD", "F3 a glow is shown during the flash")
    need(o["w1"] == "fe8rn", "W1 click inserts the plain trigger (%r)" % o["w1"])
    need(o["w2"] == "(fe8rn:0.35)", "W2 Shift+click inserts (trigger:row weight) (%r)" % o["w2"])
    need(o["w3"] == "0.35", "W3 the choice popup gets the row's weight (%r)" % o["w3"])
    pop = body_of(src_node, "function openTriggerSelectPopup(")
    need("triggerInsertText(trigger, weight, ev.shiftKey)" in pop, "S1 the popup inserts through triggerInsertText with Shift")
    need("Shift+click" in pop and "plain" in pop, "S2 the popup header names plain and Shift+click")
    need("Shift+click" in src_dom, "S3 the Nodes 2.0 trigger button tooltip names Shift+click")
    need("el.style.background =" not in ins, "S4 insertTriggerAtCursor never writes the background shorthand")
    for rel in ("web/js/uls_node.js", "web/js/uls_stack_dom.js", "tests/" + NAME + ".py"):
        b = open(os.path.join(ROOT, rel), "rb").read()
        if rel.endswith(".py"):
            need(all(x < 128 for x in b), "S5 %s is pure ASCII" % rel)
    return fails


MUTATIONS = [
    ("M1 old background flash", lambda s: s.replace(
        'el.style.boxShadow = "inset 0 0 0 2px rgba(139, 111, 232, 0.85), 0 0 8px rgba(139, 111, 232, 0.55)";',
        'const origBg = el.style.background; el.style.background = "rgba(139, 111, 232, 0.25)"; '
        'setTimeout(() => { el.style.background = origBg; }, 400);', 1)),
    ("M2 always weighted", lambda s: s.replace("if (!weighted) return String(trigger);", "", 1)),
    ("M3 Shift ignored", lambda s: s.replace("triggerInsertText(triggers[0], wStr, !!(e && e.shiftKey))",
                                             "triggerInsertText(triggers[0], wStr, false)", 1)),
    ("M4 weight fixed", lambda s: s.replace("const wStr = (Number.isFinite(w) ? w : 1).toFixed(2);",
                                            'const wStr = "1.00";', 1)),
]


def main():
    print(NAME)
    if shutil.which("node") is None:
        print("  FAIL node (Node.js) not found -- this guard needs it")
        return 1
    src = open(NODE_JS, encoding="utf-8").read()
    dom = open(DOM_JS, encoding="utf-8").read()
    fails = run(src, dom)
    if "--mutate" in sys.argv:
        print("mutations:")
        for name, f in MUTATIONS:
            m = f(src)
            if m == src:
                print("  FAIL %s did not apply" % name)
                fails.append(name)
                continue
            got = run(m, dom)
            caught = bool(got)
            print(("  ok   " if caught else "  FAIL ") + "%s caught (%d red)" % (name, len(got)))
            if not caught:
                fails.append(name + " not caught")
    if fails:
        print("[%s] FAIL -- %d" % (NAME, len(fails)))
        return 1
    print("[%s] PASS -- tint kept through the flash, plain trigger, Shift = weighted with the row's weight" % NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
