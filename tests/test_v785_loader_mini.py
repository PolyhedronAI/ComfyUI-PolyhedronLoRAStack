#!/usr/bin/env python3
# -*- coding: ascii -*-
"""v785 -- the ultra-short Media Loader (Frank's request): one toggle, and
only the image and the CONNECTED outputs stay.

THE PROMISES, DRIVEN:

[1] _miniSlots hides exactly the unconnected outputs when on, hides
    nothing when off, NEVER hides a connected one, and never reorders --
    the slot indices are law (the links live on them).
[2] _miniWidgets applies the house hide form (ph_clip_encode v556) to the
    five widgets and RESTORES the originals on the way back -- including
    a widget's own computeSize, and including "had none at all".
[3] The toggle round-trip on a fake node: on -> class set, widgets hidden,
    slots hidden, state persisted in its OWN properties key
    (ph_media_mini); off -> everything back.
[4] The wiring: the button sits in the Selection header (leftmost), the
    restore path applies the state on reload, and onConnectionsChange
    keeps the mini slots live.

Declared, not hidden: whether the frontend HONOURS `output.hidden` is a
field probe -- ignoring it is harmless (slots simply stay visible), and
nothing here ever touches slot order or indices.

Every mutation below was verified to LAND before its catch was believed.
"""

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

FAILED = []
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fail(msg):
    FAILED.append(msg)
    print("  X " + msg)


def _ok(msg):
    print("  \u2713 " + msg)


def _read(*parts):
    with io.open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def _fn_slice(src, sig):
    i = src.index(sig)
    j = src.index("\n}", i)
    return src[i:j + 2]


def _meth_slice(src, sig):
    """A class-method slice: from the signature to the first line that
    closes at 4-space indent."""
    i = src.index(sig)
    j = src.index("\n    }", i)
    return src[i:j + 6]


DRIVER = r"""
const MIN_NODE_H = 120;
%MACHINERY%

function mkWidget(name, withCS) {
    const w = { name, hidden: false, element: { style: { display: "" } } };
    if (withCS) w.computeSize = () => [200, 20];
    return w;
}
function mkOut(links) { return { links, hidden: false }; }

const out = {};

// [1] slots
{
    const outs = [mkOut([1]), mkOut(null), mkOut([]), mkOut([2, 3])];
    const hid = _miniSlots(outs, true);
    out.slotsOnHidden = outs.map((o) => o.hidden);
    out.slotsOnCount = hid;
    _miniSlots(outs, false);
    out.slotsOffHidden = outs.map((o) => o.hidden);
}

// [2] widgets round-trip
{
    const ws = [mkWidget("frame_load_cap", true),
                mkWidget("force_fps", false),
                mkWidget("media_ref", true)];       // NOT in the list
    const origCS = ws[0].computeSize;
    _miniWidgets(ws, MINI_WIDGETS, true);
    out.wHidden = ws.map((w) => !!w.hidden);
    out.wCS0 = ws[0].computeSize()[1];              // [0,-4] while hidden
    out.wRefTouched = ws[2].hidden || ws[2].computeSize === undefined;
    _miniWidgets(ws, MINI_WIDGETS, false);
    out.wRestored = ws[0].computeSize === origCS
        && !Object.prototype.hasOwnProperty.call(ws[1], "computeSize")
        && ws.every((w) => !w.hidden)
        && ws[0].element.style.display === "";
}

// [3] the toggle round-trip on a fake node
{
    const node = {
        properties: {},
        widgets: [mkWidget("frame_load_cap", true), mkWidget("on_empty", false)],
        outputs: [mkOut([1]), mkOut(null)],
        size: [400, 900],
        setSize(sz) { this.size = sz.slice(); },
        setDirtyCanvas() {},
    };
    const ui = {
        node,
        root: { _cls: new Set(),
                classList: { toggle(c, on) {
                    if (on) this._s.add(c); else this._s.delete(c);
                }, _s: null } },
        miniBtn: { textContent: "", title: "" },
        _minNodeHeight() { return 300; },
    };
    ui.root.classList._s = ui.root._cls;
    // graft the real methods onto the fake ui
    ui._applyMini = _applyMini;
    ui._toggleMini = _toggleMini;
    Object.defineProperty(ui, "mini", { get: _miniGet, set: _miniSet });
    ui._toggleMini();
    out.tOnKey = node.properties.ph_media_mini;
    out.tOnClass = ui.root._cls.has("ph-media-mini");
    out.tOnWidget = node.widgets[0].hidden;
    out.tOnSlots = node.outputs.map((o) => o.hidden);
    out.tOnHeight = node.size[1];
    out.tOnBtn = ui.miniBtn.textContent;
    ui._toggleMini();
    out.tOffKey = node.properties.ph_media_mini;
    out.tOffClass = ui.root._cls.has("ph-media-mini");
    out.tOffWidget = node.widgets[0].hidden;
    out.tOffSlots = node.outputs.map((o) => o.hidden);
    out.tOffHeight = node.size[1];
}
console.log(JSON.stringify(out));
"""


def _lift(js):
    apply_m = _meth_slice(js, "    _applyMini() {")
    toggle_m = _meth_slice(js, "    _toggleMini() {")
    # method -> free function on `this` via function-with-this trick
    apply_f = ("function _applyMini() {"
               + apply_m[apply_m.index("{") + 1:])
    toggle_f = ("function _toggleMini() {"
                + toggle_m[toggle_m.index("{") + 1:])
    # the REAL get/set bodies ride along -- never a rebuilt copy, or the
    # key-drift mutation could not land (found the honest way: the first
    # draft rebuilt them and M4 was blind).
    gi = js.index("    get mini()")
    g_body = js[js.index("{", gi):js.index("}\n", gi) + 1]
    si = js.index("    set mini(v)")
    s_body = js[js.index("{", si):js.index("}\n", si) + 1]
    parts = [
        js[js.index("const MINI_WIDGETS"):js.index("function _viewSwap(")],
        apply_f,
        toggle_f,
        "function _miniGet() " + g_body,
        "function _miniSet(v) " + s_body,
    ]
    return "\n".join(parts)


def _drive(js):
    tmp = tempfile.mkdtemp(prefix="pls785_")
    drv = os.path.join(tmp, "d.mjs")
    with io.open(drv, "w", encoding="utf-8") as fh:
        fh.write(DRIVER.replace("%MACHINERY%", _lift(js)))
    r = subprocess.run(["node", drv], capture_output=True, text=True)
    shutil.rmtree(tmp, ignore_errors=True)
    if r.returncode != 0:
        return None, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1]), None


def _judge(o):
    bad = []
    if o.get("slotsOnHidden") != [False, True, True, False]:
        bad.append("mini does not hide exactly the unconnected outputs "
                   "(%r)" % (o.get("slotsOnHidden"),))
    if o.get("slotsOnCount") != 2:
        bad.append("the hidden count is wrong (%r)" % (o.get("slotsOnCount"),))
    if o.get("slotsOffHidden") != [False, False, False, False]:
        bad.append("off does not clear every hidden flag (%r)"
                   % (o.get("slotsOffHidden"),))
    if o.get("wHidden") != [True, True, False]:
        bad.append("the five widgets are not hidden (or media_ref was "
                   "touched): %r" % (o.get("wHidden"),))
    if o.get("wCS0") != -4:
        bad.append("a hidden widget does not compute [0,-4] (%r)"
                   % (o.get("wCS0"),))
    if o.get("wRefTouched"):
        bad.append("a widget OUTSIDE the list was touched")
    if not o.get("wRestored"):
        bad.append("the restore does not put the originals back "
                   "(computeSize / had-none / display)")
    if o.get("tOnKey") is not True or o.get("tOffKey") is not False:
        bad.append("the state does not live in ph_media_mini (%r/%r)"
                   % (o.get("tOnKey"), o.get("tOffKey")))
    if not o.get("tOnClass") or o.get("tOffClass"):
        bad.append("the root class does not follow the toggle")
    if o.get("tOnWidget") is not True or o.get("tOffWidget") is not False:
        bad.append("the widgets do not follow the toggle")
    if o.get("tOnSlots") != [False, True] or o.get("tOffSlots") != [False, False]:
        bad.append("the slots do not follow the toggle (%r/%r)"
                   % (o.get("tOnSlots"), o.get("tOffSlots")))
    if o.get("tOnHeight") != 120:
        bad.append("the node does not get SHORT on the way in (%r)"
                   % (o.get("tOnHeight"),))
    if o.get("tOffHeight") != 900:
        bad.append("the node does not get its height back on the way out "
                   "(%r)" % (o.get("tOffHeight"),))
    if not str(o.get("tOnBtn", "")).endswith("Full"):
        bad.append("the button label does not flip (%r)" % (o.get("tOnBtn"),))
    return bad


def check_driven():
    print("[1..3] the machinery, driven")
    js = _read("web", "js", "ph_media_loader.js")
    o, err = _drive(js)
    if o is None:
        _fail("driver failed: %s" % (err,))
        return
    bad = _judge(o)
    for b in bad:
        _fail(b)
    if not bad:
        _ok("slots, widget round-trip, toggle, persistence key, height "
            "and label all hold")

    muts = [
        ("M1 mini hides CONNECTED outputs too",
         "        o.hidden = !!on && !linked;",
         "        o.hidden = !!on;"),
        ("M2 off leaves the flags standing",
         "        o.hidden = !!on && !linked;",
         "        if (on && !linked) o.hidden = true;"),
        ("M3 the restore forgets the original computeSize",
         "            if (w._plsMiniHadCS) w.computeSize = w._plsMiniCS;\n"
         "            else delete w.computeSize;\n", ""),
        ("M4 the state key drifts",
         "this.node.properties.ph_media_mini = !!v; }",
         "this.node.properties.ph_media_min = !!v; }"),
        ("M5 the class no longer follows",
         '        this.root.classList.toggle("ph-media-mini", on);\n', ""),
        ("M6 the node stays long on the way in",
         "                this.node.setSize([w, MIN_NODE_H]);",
         "                void w;"),
    ]
    for name, old, new in muts:
        if js.count(old) != 1:
            _fail(name + ": anchor not unique (%d)" % js.count(old))
            continue
        o2, err2 = _drive(js.replace(old, new, 1))
        caught = o2 is None or bool(_judge(o2))
        if caught:
            _ok(name + " -> caught")
        else:
            _fail(name + " -> SURVIVED")


def check_wiring():
    print("[4] the wiring")
    js = _read("web", "js", "ph_media_loader.js")
    tpl = js[js.index('<div class="ph-media-prev-lbl">'):]
    tpl = tpl[:tpl.index("</div>")]
    if '<button class="ph-mini-toggle"' not in tpl:
        _fail("the toggle button is not in the Selection header")
    elif tpl.index("ph-mini-toggle") > tpl.index("<span>Selection</span>"):
        _fail("the toggle button is not LEFTMOST in the header")
    else:
        _ok("the button sits leftmost in the Selection header")
    rest = _meth_slice(js, "    restore() {")
    if "_applyMini" not in rest:
        _fail("restore() does not apply the mini state -- a reload would "
              "come back full-size with the key still set")
    else:
        _ok("the restore path applies the state")
    i = js.index("nodeType.prototype.onConnectionsChange = function ()")
    hook = js[i:i + 400]
    if "_miniSlots(this.outputs, true)" not in hook \
            or "this._mlUI?.mini" not in hook:
        _fail("onConnectionsChange does not keep the mini slots live")
    else:
        _ok("plug/unplug keeps the mini view honest")
    css = js[js.index(".ph-media.ph-media-mini"):]
    css = css[:css.index(".ph-mini-toggle")]
    for cls in (".ph-media-bar", ".ph-batch-status", ".ph-media-grid",
                ".ph-media-foot", ".ph-media-prev-vtrim",
                ".ph-media-prev-cap", ".ph-media-prev-audio",
                ".ph-solo-toggle"):
        if cls not in css:
            _fail("the mini CSS does not hide %s" % cls)
            return
    _ok("the mini CSS hides every pane but the preview")
    for frag in ("o.splice", "outputs.sort", "outputs.splice"):
        if frag in js[js.index("function _miniSlots"):
                      js.index("function _viewSwap")]:
            _fail("_miniSlots touches slot order -- the indices are law")
            return
    _ok("slot order and indices are never touched")


def main():
    print("test_v785_loader_mini -- the ultra-short Media Loader")
    check_driven()
    check_wiring()
    if FAILED:
        print("\nFAILED (%d):" % len(FAILED))
        for f in FAILED:
            print("  - " + f)
        sys.exit(1)
    print("\nPASS -- one toggle, only the image and the connected outputs "
          "stay, and the way back restores every original.")


if __name__ == "__main__":
    main()
