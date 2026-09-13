#!/usr/bin/env python3
"""v939 -- three repairs: the trigger target, Power Upscale's click wall, the
Engine's CLIP display.

T  THE TRIGGER BUTTON (painted Stack, both renderers; pre-existing, measured
   10.09.): the last focused TEXTAREA *or text INPUT* was the target, so right
   after the weight / order input -- a text input that is removed on Enter --
   the target was a removed element, and insertTriggerAtCursor always called
   the TEXTAREA value setter: "Illegal invocation", nothing inserted, no
   message. Second facet, found by the browser probe: selectionchange stored
   the cursor of ANY focused input, so typing "3" moved the prompt's insert
   point to character 1. Driven here on the REAL listeners and the REAL
   insertTriggerAtCursor lifted from uls_node.js, against stand-in DOM classes.
P  POWER UPSCALE (S0 click-wall audit): its result and process panes were
   hidden by the inner box's display only; the frontend's container stayed
   clickable -- 320 px wide from the node's bottom to the screen's. The WIDGET
   now follows its pane; the result widget is released only once it has a
   height (a shown DOM widget of height 0 falls back to full canvas height
   after a renderer round trip). Static here; in a real browser the audit
   (tools/browser_clickwall_audit.py) reads 58 of 58 clean.
E  THE ENGINE VIEW: a decoupled CLIP strength ("1.00 / 0.80") wrapped in the
   fixed-width weight box.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE = os.path.join(ROOT, "web", "js", "uls_node.js")
PU = os.path.join(ROOT, "web", "js", "ph_power_upscale.js")
ENG = os.path.join(ROOT, "web", "js", "uls_engine_dom.js")

HEAD = r"""
const handlers = {};
class HTMLTextAreaElement { constructor() { this._v = ""; this.tagName = "TEXTAREA"; this.isConnected = true; this.style = {};
  this.selectionStart = 0; this.selectionEnd = 0; this._inside = null; }
  closest(sel) { return this._inside; } dispatchEvent() { return true; } focus() {} }
Object.defineProperty(HTMLTextAreaElement.prototype, "value", {
  get() { return this._v; },
  set(v) { if (!(this instanceof HTMLTextAreaElement)) throw new TypeError("Illegal invocation"); this._v = v; } });
class HTMLInputElement { constructor() { this._v = ""; this.tagName = "INPUT"; this.type = "text"; this.isConnected = true; this.style = {};
  this.selectionStart = 0; this.selectionEnd = 0; this._inside = null; }
  closest(sel) { return this._inside; } dispatchEvent() { return true; } focus() {} }
Object.defineProperty(HTMLInputElement.prototype, "value", {
  get() { return this._v; },
  set(v) { if (!(this instanceof HTMLInputElement)) throw new TypeError("Illegal invocation"); this._v = v; } });
globalThis.window = { HTMLTextAreaElement, HTMLInputElement };
globalThis.Event = class { constructor(t, o) { this.type = t; } };
globalThis.InputEvent = globalThis.Event;
globalThis.document = { activeElement: null,
  addEventListener(t, f) { handlers[t] = f; } };
"""

TAIL = r"""
const out = {};
const focus = (el) => { document.activeElement = el; handlers.focusin({ target: el }); };
const sel = (el, pos) => { el.selectionStart = pos; document.activeElement = el; handlers.selectionchange(); };
const blur = (el) => { handlers.focusout && handlers.focusout({ target: el }); document.activeElement = null; };
// prompt field, cursor at 11
const ta = new HTMLTextAreaElement(); ta._v = "PROMPT_MARK tail";
focus(ta); sel(ta, 11); blur(ta);
// our own order input: focused, typed "3" (cursor 1), removed on Enter
const own = new HTMLInputElement(); own._inside = { id: "uls-weight-input" };
focus(own); sel(own, 1); blur(own); own.isConnected = false;
let threw = null;
try { out.ret = insertTriggerAtCursor("(x:1.00)"); } catch (e) { threw = String(e); }
out.threw = threw; out.value = ta._v;
// a prompt that has left the page is no target
const ta2 = new HTMLTextAreaElement(); ta2._v = "Q";
focus(ta2); sel(ta2, 1); ta2.isConnected = false;
threw = null;
try { out.ret_detached = insertTriggerAtCursor("(y:1.00)"); } catch (e) { threw = String(e); }
out.threw_detached = threw; out.value_detached = ta2._v;
// a foreign text INPUT as prompt: its OWN setter, no throw
const inp = new HTMLInputElement(); inp._v = "abc";
focus(inp); sel(inp, 3);
threw = null;
try { out.ret_input = insertTriggerAtCursor("(z:1.00)"); } catch (e) { threw = String(e); }
out.threw_input = threw; out.value_input = inp._v;
console.error(JSON.stringify(out));
"""

failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


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


def code_only(src):
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    src = re.sub(r"^\s*//.*$", "", src, flags=re.M)
    return re.sub(r"\s//.*$", "", src, flags=re.M)


def main():
    print("v939: trigger target, Power Upscale click wall, Engine CLIP display")
    node = open(NODE, encoding="utf-8").read()
    pu = open(PU, encoding="utf-8").read()
    eng = open(ENG, encoding="utf-8").read()

    # ---- T: the real listeners + insertTriggerAtCursor, driven ------------
    a = node.index("let _lastFocusedTextarea = null;")
    b = node.index("function insertTriggerAtCursor(")
    listeners = node[a:b]
    ins = body_of(node, "function insertTriggerAtCursor(")
    if shutil.which("node") is None:
        check(False, "node (Node.js) not found -- this guard needs it")
        return 1
    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, "h.mjs")
        open(path, "w", encoding="utf-8").write(HEAD + listeners + "\n" + ins + "\n" + TAIL)
        p = subprocess.run(["node", path], capture_output=True, text=True, timeout=60)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    lines = [ln for ln in p.stderr.splitlines() if ln.startswith("{")]
    if p.returncode != 0 or not lines:
        check(False, "harness ran: " + (p.stderr or p.stdout)[-500:])
        return 1
    r = json.loads(lines[-1])
    for k in ("ret", "value", "ret_detached", "value_detached", "ret_input", "value_input"):
        r.setdefault(k, "(missing)")
    check(r["threw"] is None, "T  after our own input, no exception (%s)" % r["threw"])
    check(r["ret"] is True and r["value"] == "PROMPT_MARK (x:1.00) tail",
          "T  the trigger lands in the PROMPT, at ITS cursor (%r)" % r["value"])
    check(r["threw_detached"] is None and r["ret_detached"] is False and r["value_detached"] == "Q",
          "T  a prompt that has left the page is refused, not written into")
    check(r["threw_input"] is None and r["ret_input"] is True and r["value_input"] == "abc (z:1.00)",
          "T  a text INPUT prompt is written with its OWN setter (%r)" % r["value_input"])

    # ---- P: Power Upscale -------------------------------------------------
    pc = code_only(pu)
    check(re.search(r'addDOMWidget\("pls_pu_result"[^;]*;\s*w\.computeSize[^;]*;\s*node\._pvW = w;\s*w\.hidden = true;', pc, re.S)
          is not None, "P  the result widget starts hidden, like its box")
    check(re.search(r'addDOMWidget\("pls_pu_process"[^;]*;\s*w\.computeSize[^;]*;\s*node\._procW = w;\s*w\.hidden = true;', pc, re.S)
          is not None, "P  the process widget starts hidden, like its box")
    hide_pv = body_of(pc, "function _pvHide(")
    hide_pr = body_of(pc, "function _procHide(")
    check('node._pvBox.style.display = "none";' in hide_pv and "node._pvW.hidden = true" in hide_pv,
          "P  hiding the result pane hides its widget")
    check('node._procBox.style.display = "none";' in hide_pr and "node._procW.hidden = true" in hide_pr,
          "P  hiding the process pane hides its widget")
    apply_pr = body_of(pc, "function _procApply(")
    check(re.search(r'_procBox\.style\.display = "block";\s*if \(node\._procW\) node\._procW\.hidden = false;', apply_pr)
          is not None, "P  revealing the process pane releases its widget")
    apply_pv = body_of(pc, "function _pvApply(")
    onload = body_of(apply_pv, "node._pvFrames[0].onload = () =>")
    check("node._pvW.hidden = false" not in apply_pv.replace(onload, ""),
          "P  the result widget is NOT released when its box is merely shown")
    check("node._pvPrevH > 0) node._pvW.hidden = false" in onload,
          "P  ... but once the frame has loaded and the pane HAS a height")
    check(pc.count("style.display = \"none\"") >= 2
          and all(k in pc for k in ("_pvW.hidden = true", "_procW.hidden = true")),
          "P  every place that hides a pane's box also hides its widget")

    # ---- E: Engine CLIP display ------------------------------------------
    check(re.search(r"\.uls-dom-w\.uls-eng-w\s*\{[^}]*white-space:\s*nowrap", eng) is not None
          and 'w.className = "uls-dom-w uls-eng-w"' in eng,
          "E  the Engine's weight box keeps \"1.00 / 0.80\" on one line")
    return 1 if failures else 0


if __name__ == "__main__":
    rc = main()
    print("v939: %s" % ("FAIL (%d)" % len(failures) if failures else "PASS"))
    sys.exit(rc)
