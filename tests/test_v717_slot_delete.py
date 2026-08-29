"""Guard v717 -- the remove button: clears from the back, never touches the canon.

Promises, in order of what they would cost if they broke:

  1. THE BUTTON IS THE LAST ENTRY IN node.widgets, and it is serialize:false.
     Both matter, and the reason is measured rather than folklore. Read out of
     comfyui-frontend-package 1.47.10 on 2026-07-23, the two directions do not
     mirror each other:

         serialise: widgets_values[INDEX in node.widgets] = value,
                    skipping serialize===false -- a skipped widget leaves a HOLE
         configure: a RUNNING counter over widgets with serialize !== false,
                    which does NOT advance for the skipped ones

     So a serialize:false widget in the MIDDLE shifts every value behind it on
     load. Behind the button there must be nothing to shift, and the button must
     not claim a slot in the first place. Either belt alone would do; both are
     worn, which also makes it irrelevant whether configure() runs before or
     after the button is appended.

  2. Removal clears the LAST FILLED slot and clears nothing else. No value ever
     moves between canon positions -- shifting is what silently renumbers saved
     workflows.

  3. The canon stays whole: clearing sets a value to the placeholder, it never
     removes a widget.

  4. The floor is respected per node: Load CLIP keeps slot 1 (its list has no
     placeholder, so an empty slot 1 is not even expressible), Load Model can be
     emptied completely (its list does have one, and the button must not be
     cleverer than the dropdown).

  5. Repeated removal walks down one slot at a time and then stops at the floor.

  6. A hole made through the dropdown is tidied from the back, not jumped over.

The JS is lifted from source and driven in real node -- run it, do not read it.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fail(m):
    print("FAIL: " + m)
    sys.exit(1)


def _read(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


def _lift(src, signature):
    i = src.find(signature)
    if i < 0:
        _fail(f"could not lift {signature!r} -- signature changed?")
    end = src.find("\n}", i)
    if end < 0:
        _fail(f"could not find the end of {signature!r}")
    return src[i:end + 2]


def _run_js(js):
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(js)
        path = fh.name
    try:
        proc = subprocess.run(["node", path], capture_output=True, text=True,
                              timeout=60)
    finally:
        os.unlink(path)
    if proc.returncode != 0:
        _fail("node run failed: " + (proc.stderr or "")[-800:])
    return proc.stdout.strip()


MODEL_NAMES = ["model_1", "model_2", "model_3", "model_4", "model_5", "model_6"]
CLIP_NAMES = ["clip_name", "clip_name_2", "clip_name_3", "clip_name_4"]
MODEL_PH = "\u2014 select model \u2014"
CLIP_PH = "\u2014 none \u2014"


def _harness(js):
    return "\n".join([
        _lift(js, "function _hide(w, hidden)"),
        _lift(js, "function _filled(value, placeholder)"),
        _lift(js, "function visibleCount(values, placeholder)"),
        _lift(js, "function updateSelectMax(node, values, placeholder)"),
        _lift(js, "function lastFilledSlot(values, placeholder)"),
        _lift(js, "function removeLastSlot(node, spec)"),
        _lift(js, "function applySlots(node, spec)"),
        _lift(js, "function addRemoveButton(node, spec)"),
        """
const HIDDEN_PREFIX = "pls-hidden-";
const REMOVE_BTN = "pls_remove_slot";
// v718: the button paints itself in amber and its label is a bare glyph.
const REMOVE_GLYPH = "\u2715";
const _drawRemove = () => {};
function _w(node, name) {
    return (node.widgets || []).find((w) => w.name === name);
}
function mkNode(names, withSelect) {
    const widgets = names.map((n) => ({ name: n, type: "combo", value: null }));
    if (withSelect) widgets.push({ name: "select", type: "number", value: 1,
                                   options: { max: 6 } });
    return {
        widgets,
        size: [340, 200],
        addWidget(type, name, value, cb) {
            const w = { name, type, value, callback: cb };
            this.widgets.push(w);
            return w;
        },
        computeSize() {
            let h = 0;
            for (const w of this.widgets) {
                h += (typeof w.computeSize === "function")
                    ? w.computeSize(this.size[0])[1] : 20;
            }
            return [140, h];
        },
        setSize(s) { this.size = [s[0], s[1]]; },
        setDirtyCanvas() {},
    };
}
function fill(node, spec, n) {
    for (let i = 0; i < spec.names.length; i++) {
        const w = _w(node, spec.names[i]);
        if (w) w.value = (i < n) ? ("file_" + i) : spec.placeholder;
    }
}
function press(node, spec) {
    const b = _w(node, REMOVE_BTN);
    if (!b || !b.callback) throw new Error("no remove button to press");
    b.callback();
}
function state(node, spec) {
    return {
        values: spec.names.map((n) => _w(node, n).value),
        canon: node.widgets.map((w) => w.name),
        lastWidget: node.widgets[node.widgets.length - 1].name,
        btnHidden: String(_w(node, REMOVE_BTN).type).startsWith(HIDDEN_PREFIX),
        btnLabel: _w(node, REMOVE_BTN).label,
        btnSerialize: _w(node, REMOVE_BTN).serialize,
        width: node.size[0],
    };
}
""",
    ])


def check_button_position(js):
    """Promise 1 -- last in node.widgets, and serialize:false."""
    base = _harness(js)
    script = base + """
const out = {};
for (const [key, spec] of [["model", { names: %s, placeholder: %s,
                                       hasSelect: true, floor: 0 }],
                           ["clip", { names: %s, placeholder: %s,
                                      hasSelect: false, floor: 1,
                                      typeGateFrom: 3 }]]) {
    const node = mkNode(spec.names, spec.hasSelect);
    if (key === "clip") node.widgets.push({ name: "type", type: "combo",
                                            value: "flux" });
    addRemoveButton(node, spec);
    addRemoveButton(node, spec);          // must be idempotent
    fill(node, spec, 2);
    applySlots(node, spec);
    const s = state(node, spec);
    out[key] = { lastWidget: s.lastWidget, canon: s.canon,
                 serialize: s.btnSerialize,
                 optSerialize: _w(node, REMOVE_BTN).options
                     ? _w(node, REMOVE_BTN).options.serialize : undefined,
                 count: node.widgets.filter((w) => w.name === REMOVE_BTN).length };
}
console.log(JSON.stringify(out));
""" % (json.dumps(MODEL_NAMES), json.dumps(MODEL_PH),
       json.dumps(CLIP_NAMES), json.dumps(CLIP_PH))
    res = json.loads(_run_js(script))
    for key, got in res.items():
        if got["lastWidget"] != "pls_remove_slot":
            _fail(f"{key}: the remove button must be the LAST widget, but "
                  f"{got['lastWidget']!r} is -- a widget behind it would be "
                  f"shifted on load (serialise writes by index and skips "
                  f"serialize:false, configure counts and does not)")
        # .get(): an undefined value vanishes from the JSON entirely, and a
        # missing flag must be REPORTED, not raise a KeyError in the guard.
        if got.get("serialize") is not False:
            _fail(f"{key}: the remove button must carry serialize:false, got "
                  f"{got.get('serialize')!r}")
        if got.get("optSerialize") is not False:
            _fail(f"{key}: options.serialize must be false too -- both "
                  f"spellings are read in the wild")
        if got["count"] != 1:
            _fail(f"{key}: addRemoveButton is not idempotent, {got['count']} "
                  f"buttons ended up on the node")
        for name in (MODEL_NAMES if key == "model" else CLIP_NAMES):
            if name not in got["canon"]:
                _fail(f"{key}: canon lost {name!r} -- {got['canon']}")


def check_removal(js):
    """Promises 2, 3, 5 -- clear from the back, one at a time, canon intact."""
    base = _harness(js)
    script = base + """
const spec = { names: %s, placeholder: %s, hasSelect: true, floor: 0 };
const node = mkNode(spec.names, true);
addRemoveButton(node, spec);
fill(node, spec, 4);
_w(node, "select").value = 4;
applySlots(node, spec);
const steps = [state(node, spec)];
for (let i = 0; i < 6; i++) { press(node, spec); steps.push(state(node, spec)); }
console.log(JSON.stringify(steps));
""" % (json.dumps(MODEL_NAMES), json.dumps(MODEL_PH))
    steps = json.loads(_run_js(script))

    for i, st in enumerate(steps):
        filled = [v for v in st["values"] if v != MODEL_PH]
        want = max(0, 4 - i)
        if len(filled) != want:
            _fail(f"after {i} press(es) {want} slot(s) should remain filled, "
                  f"got {len(filled)} ({st['values']})")
        # promise 2: the survivors must be the FIRST ones, untouched
        for j, v in enumerate(st["values"]):
            expect = f"file_{j}" if j < want else MODEL_PH
            if v != expect:
                _fail(f"after {i} press(es) slot {j + 1} holds {v!r}, expected "
                      f"{expect!r} -- removal must clear from the BACK and move "
                      f"nothing between canon positions")
        # promise 3
        if st["canon"] != MODEL_NAMES + ["select", "pls_remove_slot"]:
            _fail(f"CANON BROKEN after {i} press(es): {st['canon']}")
        if st["width"] != 340:
            _fail(f"press {i} changed the node WIDTH ({st['width']})")

    # promise 5: pressing past empty must be a no-op, not an error or a wrap
    if [v for v in steps[-1]["values"] if v != MODEL_PH]:
        _fail("Load Model should be emptiable completely (its slot 1 has a "
              "placeholder, so the dropdown can do it too)")
    if not steps[-1]["btnHidden"]:
        _fail("with nothing left to clear the button must hide itself")

    # v718 REHUNG. The old promise was "the label names the slot it clears".
    # Frank found it lying on screen: with slot 1 filled and slot 2 empty it read
    # "remove slot 1" while the row that visibly disappeared was ROW 2. Both
    # readings of "slot" are legitimate and no number can serve both, so the
    # label is now a bare glyph that claims nothing. What the label may NOT do is
    # go back to interpolating a number.
    for i, st in enumerate(steps):
        if st["btnLabel"] != "\u2715":
            _fail(f"after {i} press(es) the label is {st['btnLabel']!r}; it must "
                  f"stay the bare glyph. A slot number cannot be right for both "
                  f"the slot being cleared and the row that disappears.")


def check_floor(js):
    """Promise 4 -- Load CLIP keeps slot 1, and says so by hiding the button."""
    base = _harness(js)
    script = base + """
const spec = { names: %s, placeholder: %s, hasSelect: false, floor: 1,
               typeGateFrom: 3 };
const node = mkNode(spec.names, false);
node.widgets.push({ name: "type", type: "combo", value: "flux" });
addRemoveButton(node, spec);
fill(node, spec, 4);
applySlots(node, spec);
const steps = [state(node, spec)];
for (let i = 0; i < 5; i++) { press(node, spec); steps.push(state(node, spec)); }
steps.forEach((s, i) => { s.typeHidden = null; });
const t = _w(node, "type");
console.log(JSON.stringify({ steps, typeValue: t.value,
                             typeHidden: String(t.type).startsWith(HIDDEN_PREFIX) }));
""" % (json.dumps(CLIP_NAMES), json.dumps(CLIP_PH))
    res = json.loads(_run_js(script))
    steps = res["steps"]
    for i, st in enumerate(steps):
        filled = [v for v in st["values"] if v != CLIP_PH]
        want = max(1, 4 - i)
        if len(filled) != want:
            _fail(f"Load CLIP after {i} press(es): {want} encoder(s) should "
                  f"remain, got {len(filled)} ({st['values']})")
    if steps[-1]["values"][0] == CLIP_PH:
        _fail("slot 1 of Load CLIP must never be cleared -- its list has no "
              "placeholder, so an empty slot 1 is not expressible at all")
    if not steps[-1]["btnHidden"]:
        _fail("at the floor the button must hide itself instead of sitting "
              "there doing nothing")
    # the type field must have come back on the way down past three
    if res["typeHidden"]:
        _fail("after removing down to one encoder the `type` field must be "
              "visible again")
    if res["typeValue"] != "flux":
        _fail(f"the `type` value must survive the whole descent, got "
              f"{res['typeValue']!r}")


def check_hole(js):
    """Promise 6 -- a hole is tidied from the back, and NOTHING shifts into it.

    The hole has to be chosen with care. With slots 1 and 3 filled, clearing 3
    leaves 1 alone and a compacting bug would have nothing left to move -- the
    test would pass while the code shifts values. So: fill 1, 3 and 4, leaving
    the hole at 2 WITH a filled slot behind it. Clearing 4 must leave slot 2
    empty and slot 3 exactly where it was; a compacting implementation would
    pull slot 3's value forward into slot 2 and is caught here.
    """
    base = _harness(js)
    script = base + """
const spec = { names: %s, placeholder: %s, hasSelect: true, floor: 0 };
const node = mkNode(spec.names, true);
addRemoveButton(node, spec);
fill(node, spec, 4);
_w(node, "model_2").value = spec.placeholder;    // hole at 2, 3 and 4 behind it
applySlots(node, spec);
const before = state(node, spec);
press(node, spec);
const after = state(node, spec);
console.log(JSON.stringify({ before, after }));
""" % (json.dumps(MODEL_NAMES), json.dumps(MODEL_PH))
    res = json.loads(_run_js(script))
    # v718: the label no longer carries a number, so which slot was targeted is
    # read from the RESULT below instead of from the caption -- a stronger check
    # anyway, since it measures the behaviour rather than the advertisement.
    after = res["after"]["values"]
    if after[0] != "file_0":
        _fail(f"slot 1 must be untouched by the removal, got {after[0]!r}")
    if after[3] != MODEL_PH:
        _fail(f"slot 4 should have been cleared, got {after[3]!r}")
    if after[1] != MODEL_PH:
        _fail(f"slot 2 must STAY EMPTY -- nothing may be compacted into the "
              f"hole; moving a value between canon positions is exactly what "
              f"silently renumbers saved workflows. Got {after[1]!r}")
    if after[2] != "file_2":
        _fail(f"slot 3 must keep its own value, got {after[2]!r} -- a value "
              f"moved out of its canon position")


def check_source_pins(js):
    """The floors are read from the source, never restated here."""
    m = re.search(r"ULSLoadCLIP:\s*\{.*?floor:\s*(\d+)", js, re.S)
    if not m:
        _fail("Load CLIP has no floor declared -- slot 1 could be cleared")
    if int(m.group(1)) != 1:
        _fail(f"Load CLIP floor is {m.group(1)}, must be 1: its clip_name list "
              f"has no placeholder entry, so an empty slot 1 cannot round-trip")
    m = re.search(r"ULSLoadModel:\s*\{.*?floor:\s*(\d+)", js, re.S)
    if not m:
        _fail("Load Model has no floor declared")
    if int(m.group(1)) != 1:
        _fail(f"Load Model floor is {m.group(1)}, must be 1 since v719: with "
              f"only slot 1 filled the button has nothing to offer but "
              f"emptying the node, so it hides. Slot 1 stays clearable through "
              f"the dropdown that owns the placeholder")


def main():
    js = _read("web", "js", "ph_basics.js")
    check_source_pins(js)
    check_button_position(js)
    check_removal(js)
    check_floor(js)
    check_hole(js)
    print("PASS: v717 remove button -- last in the canon and serialize:false, "
          "clears from the back one slot at a time, floors honoured, holes "
          "tidied without shifting anything")
    sys.exit(0)


if __name__ == "__main__":
    main()
