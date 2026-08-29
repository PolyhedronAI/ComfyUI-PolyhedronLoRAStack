"""Guard v716 -- progressive slots (Load Model / Load CLIP) + encoder recipes.

The load-bearing promises, in order of how much they would cost if they broke:

  1. THE WIDGETS STAY IN THE CANON. Hiding must never remove a widget from
     node.widgets, because widgets_values is positional -- a widget that
     disappears shifts every value behind it and silently corrupts every saved
     workflow on the next load. This is the one that must never regress.
  2. Slot k+1 appears once k is filled; trailing empties collapse to one spare.
  3. Showing a hidden widget restores it fully -- the v556 bug wrote
     `computeSize = undefined` back and left the widget zero-height forever.
  4. `select` on Load Model never exceeds the last filled slot.
  5. Placeholder strings are IDENTICAL in JS and Python, or the two sides
     disagree about which slots count as filled.
  6. The encoder recipe check refuses provably-wrong sets and, just as
     importantly, does NOT refuse sets it merely cannot identify.

The JS is lifted out of the source by text slice and driven in real node, per
the house doctrine: run the code, do not read it. Signatures must not be
changed by that lift (see the v621 doctrine note).
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nodes"))


def _fail(m):
    print("FAIL: " + m)
    sys.exit(1)


def _read(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


def _lift(src, signature):
    """Hoist one function body out of the source, up to the first column-0 '}'."""
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
        proc = subprocess.run([_node(), path], capture_output=True, text=True,
                              timeout=60)
    finally:
        os.unlink(path)
    if proc.returncode != 0:
        _fail("node run failed: " + (proc.stderr or "")[-800:])
    return proc.stdout.strip()


def _node():
    return "node"


def _harness(js_src):
    """The lifted layout machinery plus a minimal LiteGraph-ish node."""
    return "\n".join([
        _lift(js_src, "function _hide(w, hidden)"),
        _lift(js_src, "function _filled(value, placeholder)"),
        _lift(js_src, "function visibleCount(values, placeholder)"),
        _lift(js_src, "function updateSelectMax(node, values, placeholder)"),
        _lift(js_src, "function applySlots(node, spec)"),
        """
const HIDDEN_PREFIX = "pls-hidden-";
// v717: applySlots also labels/hides the remove button, so the lifted function
// needs the constant. This harness deliberately does NOT add the button itself
// -- the removal behaviour is tests/test_v717_slot_delete.py's subject, and
// _w() returning undefined here is the correct "no button on this node" case.
const REMOVE_BTN = "pls_remove_slot";
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
        computeSize() {
            let h = 0;
            for (const w of this.widgets) {
                h += (typeof w.computeSize === "function")
                    ? w.computeSize(this.size[0])[1] : 20;
            }
            return [140, h];       // narrow on purpose: width must survive
        },
        setSize(s) { this.size = [s[0], s[1]]; },
        setDirtyCanvas() {},
    };
}
function visibleNames(node) {
    return node.widgets
        .filter((w) => !String(w.type || "").startsWith(HIDDEN_PREFIX)
                       && w.name !== "select")
        .map((w) => w.name);
}
""",
    ])


MODEL_NAMES = ["model_1", "model_2", "model_3", "model_4", "model_5", "model_6"]
CLIP_NAMES = ["clip_name", "clip_name_2", "clip_name_3", "clip_name_4"]
MODEL_PH = "\u2014 select model \u2014"
CLIP_PH = "\u2014 none \u2014"


def check_js(js_src):
    base = _harness(js_src)

    # -- promise 2 + 1: progression, and the canon stays whole ---------------
    script = base + """
const spec = { names: %s, placeholder: %s, hasSelect: true };
const out = [];
const node = mkNode(spec.names, true);
const steps = [
    [],                       // nothing filled
    ["a"],                    // slot 1
    ["a", "b"],               // slots 1+2
    ["a", "b", "c"],
    ["a", null, "c"],         // hole in the middle
    ["a", "b", "c", "d", "e", "f"],   // all six
];
for (const values of steps) {
    // BY NAME, and tolerant of a missing widget: a mutation that REMOVES a
    // widget must reach the canon assertion below and be reported as a canon
    // break, not kill the harness on an index that no longer exists.
    for (let i = 0; i < spec.names.length; i++) {
        const w = _w(node, spec.names[i]);
        if (!w) continue;
        w.value = (values[i] === undefined || values[i] === null)
            ? spec.placeholder : values[i];
    }
    applySlots(node, spec);
    out.push({ filled: values.filter((v) => v).length,
               visible: visibleNames(node),
               canon: node.widgets.map((w) => w.name),
               selectMax: _w(node, "select").options.max,
               width: node.size[0] });
}
console.log(JSON.stringify(out));
""" % (json.dumps(MODEL_NAMES), json.dumps(MODEL_PH))
    res = json.loads(_run_js(script))

    expect_visible = [1, 2, 3, 4, 4, 6]
    for step, (got, want) in enumerate(zip(res, expect_visible)):
        if len(got["visible"]) != want:
            _fail(f"progression step {step}: {want} slot(s) should be visible, "
                  f"got {len(got['visible'])} ({got['visible']})")
        if got["visible"] != MODEL_NAMES[:want]:
            _fail(f"progression step {step}: visible slots must be the FIRST "
                  f"{want} in order, got {got['visible']}")
        # promise 1 -- the one that must never regress
        if got["canon"] != MODEL_NAMES + ["select"]:
            _fail(f"CANON BROKEN at step {step}: node.widgets is {got['canon']} "
                  f"-- hiding must never remove a widget (widgets_values is "
                  f"positional; saved workflows would shift)")
        if got["width"] != 340:
            _fail(f"step {step}: relayout changed the node WIDTH "
                  f"({got['width']}) -- height only")

    # -- promise 4: select max follows the last filled slot ------------------
    expect_max = [1, 1, 2, 3, 3, 6]
    for step, (got, want) in enumerate(zip(res, expect_max)):
        if got["selectMax"] != want:
            _fail(f"select max at step {step}: expected {want}, got "
                  f"{got['selectMax']}")

    # -- promise 3: show restores the widget completely ----------------------
    script = base + """
const out = {};
// (a) a widget that had NO own computeSize must not keep one after show
const w1 = { name: "x", type: "combo", value: 1 };
_hide(w1, true); _hide(w1, false);
out.ownCS = Object.prototype.hasOwnProperty.call(w1, "computeSize");
out.type = w1.type;
out.heightAfter = (typeof w1.computeSize === "function")
    ? w1.computeSize(100)[1] : "no-own-computeSize";
// (b) a widget that HAD one must get exactly it back
const orig = (width) => [width, 42];
const w2 = { name: "y", type: "combo", value: 1, computeSize: orig };
_hide(w2, true); _hide(w2, false);
out.restored = (w2.computeSize === orig);
// (c) idempotent -- repeated hides must not stack the prefix
const w3 = { name: "z", type: "combo", value: 1 };
_hide(w3, true); _hide(w3, true); _hide(w3, true);
out.stacked = w3.type;
_hide(w3, false);
out.unstacked = w3.type;
console.log(JSON.stringify(out));
"""
    res = json.loads(_run_js(script))
    if res["ownCS"]:
        _fail("v556 bug is back: showing a widget that never had its own "
              "computeSize left one behind -- it stays zero-height forever")
    if res["type"] != "combo":
        _fail(f"show did not restore the widget type (got {res['type']!r})")
    if not res["restored"]:
        _fail("show did not restore the widget's ORIGINAL computeSize")
    if res["stacked"] != "pls-hidden-combo":
        _fail(f"hide is not idempotent -- type stacked to {res['stacked']!r}")
    if res["unstacked"] != "combo":
        _fail(f"show after repeated hides gave {res['unstacked']!r}")

    # -- Load CLIP: slot 1 has no placeholder, so slot 2 is always offered ---
    script = base + """
const spec = { names: %s, placeholder: %s, hasSelect: false };
const node = mkNode(spec.names, false);
node.widgets[0].value = "umt5.safetensors";
for (let i = 1; i < spec.names.length; i++) node.widgets[i].value = spec.placeholder;
applySlots(node, spec);
const one = visibleNames(node);
node.widgets[1].value = "clip_l.safetensors";
applySlots(node, spec);
const two = visibleNames(node);
console.log(JSON.stringify({ one, two, canon: node.widgets.map((w) => w.name) }));
""" % (json.dumps(CLIP_NAMES), json.dumps(CLIP_PH))
    res = json.loads(_run_js(script))
    if res["one"] != CLIP_NAMES[:2]:
        _fail(f"Load CLIP with one encoder should offer exactly one spare, got "
              f"{res['one']}")
    if res["two"] != CLIP_NAMES[:3]:
        _fail(f"Load CLIP with two encoders should offer the third, got "
              f"{res['two']}")
    if res["canon"] != CLIP_NAMES:
        _fail(f"CANON BROKEN on Load CLIP: {res['canon']}")

    # -- the `type` gate: hidden from three encoders, and it COMES BACK -------
    # Measured: core ignores clip_type at 3 and 4 files. A visible-but-inert
    # control is worse than an absent one, so it is hidden -- but hiding is only
    # acceptable if the value and the canon position survive and the field
    # returns when the count drops.
    #
    # THE THRESHOLD IS LIFTED FROM THE SOURCE, not restated here. Writing 3 into
    # the harness would make this check answer a question about the harness
    # instead of about the node -- a mutation moving the gate to 2 sailed
    # straight through the first version of this test for exactly that reason.
    import re
    m = re.search(r"typeGateFrom:\s*(\d+)", js_src)
    if not m:
        _fail("typeGateFrom is gone from ph_basics.js -- the `type` field would "
              "stay visible at 3 and 4 encoders where core ignores it")
    gate = int(m.group(1))
    if gate != 3:
        _fail(f"the `type` gate sits at {gate} encoders, but core only stops "
              f"consulting clip_type from THREE upwards (measured in "
              f"comfy/sd.py load_text_encoder_state_dicts) -- at two the widget "
              f"still decides the family and must stay visible")
    script = base + """
const spec = { names: %s, placeholder: %s, hasSelect: false, typeGateFrom: %d };
const node = mkNode(spec.names, false);
node.widgets.push({ name: "type", type: "combo", value: "flux" });
const seen = [];
const fills = [1, 2, 3, 4, 2];
for (const n of fills) {
    for (let i = 0; i < spec.names.length; i++) {
        _w(node, spec.names[i]).value = (i < n) ? ("f" + i) : spec.placeholder;
    }
    applySlots(node, spec);
    const t = _w(node, "type");
    seen.push({ filled: n,
                typeHidden: String(t.type).startsWith(HIDDEN_PREFIX),
                typeValue: t.value,
                canon: node.widgets.map((w) => w.name) });
}
console.log(JSON.stringify(seen));
""" % (json.dumps(CLIP_NAMES), json.dumps(CLIP_PH), gate)
    res = json.loads(_run_js(script))
    expect_hidden = [False, False, True, True, False]
    for step, (got, want) in enumerate(zip(res, expect_hidden)):
        if got["typeHidden"] != want:
            state = "hidden" if want else "visible"
            _fail(f"with {got['filled']} encoder(s) the `type` field must be "
                  f"{state} (core ignores it from three upwards), got "
                  f"hidden={got['typeHidden']}")
        if got["typeValue"] != "flux":
            _fail(f"the `type` VALUE must survive the gate, got "
                  f"{got['typeValue']!r} at step {step}")
        if got["canon"] != CLIP_NAMES + ["type"]:
            _fail(f"CANON BROKEN by the type gate at step {step}: {got['canon']}")


def check_parity(js_src, py_src):
    """Promise 5 -- the placeholders must be the same string on both sides."""
    for label, literal in (("model", '"\\u2014 select model \\u2014"'),
                           ("clip", '"\\u2014 none \\u2014"')):
        if literal not in js_src:
            _fail(f"{label} placeholder literal missing from ph_basics.js")
    if '_MODEL_PLACEHOLDER = "\\u2014 select model \\u2014"' not in py_src:
        _fail("python model placeholder changed -- JS/PY parity broken")
    if '_CLIP_PLACEHOLDER = "\\u2014 none \\u2014"' not in py_src:
        _fail("python clip placeholder changed -- JS/PY parity broken")
    # the slot names the JS drives must be the ones python actually defines
    if 'f"clip_name_{i}"' not in py_src:
        _fail("python no longer defines the appended clip_name_N slots")
    if "clip_name_4" not in js_src or "model_6" not in js_src:
        _fail("JS slot table no longer covers the full slot range")


def check_recipes():
    """Promise 6 -- refuse the provably wrong, never refuse mere ignorance."""
    import ph_te_detect as D

    def ident(kind):
        return {"kind": kind, "source": "local", "is_long": False,
                "long_ctx": None, "spiece": False}

    unknown = {"kind": None, "source": "none", "is_long": False,
               "long_ctx": None, "spiece": False}

    ok, _ = D.check_recipe("flux", [ident("clip_l"), ident("t5xxl")])
    if not ok:
        _fail("flux clip_l+t5xxl must pass")
    ok, _ = D.check_recipe("flux", [ident("t5xxl"), ident("clip_l")])
    if not ok:
        _fail("ORDER MUST NOT MATTER -- core identifies by state dict, measured "
              "2026-07-23; the check must not invent an ordering rule")
    ok, msg = D.check_recipe("flux", [ident("t5xxl"), ident("t5xxl")])
    if ok:
        _fail("flux with two t5xxl and no clip_l must be refused")
    if "clip_l" not in msg:
        _fail(f"the refusal must name what is missing, got: {msg}")
    ok, _ = D.check_recipe("sdxl", [ident("clip_l"), ident("clip_g")])
    if not ok:
        _fail("sdxl clip_l+clip_g must pass")
    ok, _ = D.check_recipe("hidream", [ident("clip_l"), ident("clip_g")])
    if ok:
        _fail("hidream needs at least one of t5xxl/llama3_8 -- must be refused")
    ok, msg = D.check_recipe("flux", [ident("clip_l"), unknown])
    if not ok:
        _fail("an UNIDENTIFIED slot must never fail the check -- GGUF and "
              "future encoders have to keep working")
    if not msg:
        _fail("skipping the check must say so rather than pass silently")
    ok, _ = D.check_recipe("wan", [ident("t5xxl")])
    if not ok:
        _fail("a type with no recipe entry must pass unchecked")

    # the count outranks the type -- measured in comfy/sd.py
    if D.forced_type_for_count(3) != "sd3":
        _fail("three encoders must force sd3")
    if D.forced_type_for_count(4) != "hidream":
        _fail("four encoders must force hidream")
    if D.forced_type_for_count(2) is not None:
        _fail("at two encoders the type widget still decides")


def check_detect():
    """The header reader must identify from keys and shapes alone."""
    import struct

    import ph_te_detect as D

    def mk(path, tensors):
        header = {k: {"dtype": "F16", "shape": v, "data_offsets": [0, 2]}
                  for k, v in tensors.items()}
        blob = json.dumps(header).encode("utf-8")
        with open(path, "wb") as fh:
            fh.write(struct.pack("<Q", len(blob)))
            fh.write(blob)
            fh.write(b"\x00\x00")

    cases = {
        "clip_l": ({"text_model.encoder.layers.0.mlp.fc1.weight": [3072, 768],
                    "text_model.embeddings.position_embedding.weight": [77, 768]},
                   "clip_l", False),
        "long_clip_l": ({"text_model.encoder.layers.0.mlp.fc1.weight": [3072, 768],
                         "text_model.embeddings.position_embedding.weight": [248, 768]},
                        "clip_l", True),
        # a clip_g state dict CONTAINS layer 0 too -- order of the rules decides
        "clip_g": ({"text_model.encoder.layers.30.mlp.fc1.weight": [5120, 1280],
                    "text_model.encoder.layers.0.mlp.fc1.weight": [5120, 1280]},
                   "clip_g", False),
        "t5xxl": ({"encoder.block.23.layer.1.DenseReluDense.wi_1.weight":
                   [10240, 4096]}, "t5xxl", False),
        "t5xl": ({"encoder.block.23.layer.1.DenseReluDense.wi_1.weight":
                  [5120, 2048]}, "t5xl", False),
    }
    tmp = tempfile.mkdtemp()
    for name, (tensors, want, want_long) in cases.items():
        path = os.path.join(tmp, name + ".safetensors")
        mk(path, tensors)
        got = D.identify_file(path)
        if got["kind"] != want:
            _fail(f"{name} identified as {got['kind']!r}, expected {want!r}")
        if got["is_long"] != want_long:
            _fail(f"{name}: long-context flag is {got['is_long']}, expected "
                  f"{want_long}")

    # a file that is not safetensors must degrade to unknown, never raise
    junk = os.path.join(tmp, "x.gguf")
    with open(junk, "wb") as fh:
        fh.write(b"GGUF\x00\x00\x00\x00" + b"\xff" * 64)
    if D.identify_file(junk)["kind"] is not None:
        _fail("a non-safetensors file must identify as unknown")
    if D.identify_file(os.path.join(tmp, "nope.safetensors"))["kind"] is not None:
        _fail("a missing file must identify as unknown, not raise")


def main():
    js_src = _read("web", "js", "ph_basics.js")
    py_src = _read("nodes", "ph_basics.py")
    check_js(js_src)
    check_parity(js_src, py_src)
    check_detect()
    check_recipes()
    print("PASS: v716 progressive slots -- canon whole, progression correct, "
          "select clamped, placeholders in parity, headers identified, recipes "
          "enforced without punishing ignorance")
    sys.exit(0)


if __name__ == "__main__":
    main()
