"""Guard v718 -- Load CLIP display order: the disk never learns the screen moved.

`type` (and `device`) sit at the TOP of Load CLIP on screen, while the canon on
disk is unchanged: clip_name, type, device, clip_name_2..4. This is the most
dangerous edit class in the tree. v584 and v603 both shipped a re-ordered widget
row and every saved graph loaded shifted by one slot -- a prompt landed inside
`separator`, a boolean landed in a prompt box.

So this guard does NOT check that a permutation function returns a permuted
list. It reproduces the frontend's ACTUAL serialise and configure algorithms,
transcribed from comfyui-frontend-package 1.47.10 (read 2026-07-23), and drives
a full round trip through them:

    serialise:  widgets_values[INDEX in node.widgets] = value,
                skipping serialize===false (leaving a hole at that index)
    configure:  a RUNNING counter over widgets with serialize !== false,
                which does NOT advance for skipped ones

Those two are not mirror images of each other, which is exactly why the remove
button must ride at the back and why this has to be tested against the real
asymmetry rather than against a tidy assumption about it.

Promises:
  1. A graph saved by an OLD version (canon order on disk) loads correctly --
     every value in its own widget.
  2. A graph saved by THIS version is byte-identical in widgets_values to what
     the old version would have written. The file never learns about the screen.
  3. save -> load -> save is a fixed point.
  4. The permutation happens AFTER the load, never during it.
  5. The remove button stays LAST in both orders and never claims a slot.
  6. The button label carries no slot number.
"""
import json
import os
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


def _lift_const(src, name):
    i = src.find(f"const {name} = ")
    if i < 0:
        _fail(f"constant {name} is gone from ph_basics.js")
    end = src.find("];", i)
    if end < 0:
        _fail(f"could not read {name}")
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


CLIP_PH = "\u2014 none \u2014"
# The canon, restated here ON PURPOSE: this is the contract with every workflow
# already on disk. If ph_basics.py's INPUT_TYPES order ever changes, this guard
# must fail rather than quietly follow along.
EXPECTED_CANON = ["clip_name", "type", "device",
                  "clip_name_2", "clip_name_3", "clip_name_4"]


def _harness(js):
    return "\n".join([
        _lift_const(js, "CLIP_CANON"),
        _lift_const(js, "CLIP_DISPLAY"),
        _lift(js, "function _reorder(node, order)"),
        _lift(js, "function _toDisplay(node, spec)"),
        _lift(js, "function _toCanon(node, spec)"),
        _lift(js, "function _serializeInCanon(node, spec, base, args)"),
        """
const spec = { canon: CLIP_CANON, display: CLIP_DISPLAY };

/*
 * TRANSCRIBED from comfyui-frontend-package 1.47.10, minified source:
 *
 *   save:  if(t&&this.serialize_widgets){e.widgets_values=[];
 *          for(let[n,r]of t.entries()){if(r.serialize===!1)continue;
 *          let t=r?.value;e.widgets_values[n]=...}}
 *
 *   load:  if(e.widgets_values){let t=0;for(let n of this.widgets??[])
 *          if(n.serialize!==!1){if(t>=e.widgets_values.length)break;
 *          n.value=e.widgets_values[t++]}}
 *
 * Note the asymmetry: save indexes by position and leaves holes, load counts
 * and does not advance over skipped widgets.
 */
function frontendSerialize(node) {
    const out = [];
    for (const [i, w] of node.widgets.entries()) {
        if (w.serialize === false) continue;
        out[i] = w.value ?? null;
    }
    // JSON turns the sparse tail/holes into nulls, as the real save does
    return JSON.parse(JSON.stringify(out));
}
function frontendConfigure(node, widgetsValues) {
    if (!widgetsValues) return;
    let i = 0;
    for (const w of node.widgets) {
        if (w.serialize !== false) {
            if (i >= widgetsValues.length) break;
            w.value = widgetsValues[i++];
        }
    }
}

function mkNode(withButton) {
    const widgets = CLIP_CANON.map((n) => ({ name: n, type: "combo",
                                             value: null }));
    const node = { widgets, size: [340, 200], _plsDisplayed: false };
    if (withButton) {
        widgets.push({ name: "pls_remove_slot", type: "button", value: null,
                       serialize: false, label: "\\u2715" });
    }
    return node;
}
function names(node) { return node.widgets.map((w) => w.name); }
function valuesByName(node) {
    const o = {};
    for (const w of node.widgets) o[w.name] = w.value;
    return o;
}
""",
    ])


def check_canon_contract(js):
    """The canon must still be what every saved workflow expects.

    Read from ph_basics.py rather than trusted: if INPUT_TYPES is ever
    re-ordered, this must fail loudly instead of quietly following along. The
    display order lives in the JS; the canon lives on disk and is forever.
    """
    import re

    py = _read("nodes", "ph_basics.py")
    section = py[py.index("class ULSLoadCLIP"):]
    section = section[:section.index("class ULSLoadVAE")]

    # declaration lines only: a key at the start of its own line inside
    # INPUT_TYPES. Matching anywhere would hit the same word inside a tooltip.
    found = []
    for m in re.finditer(r'^\s+"([a-z_0-9]+)":\s*[\(\[a-z_]', section,
                         re.MULTILINE):
        name = m.group(1)
        if name in EXPECTED_CANON and name not in [n for _, n in found]:
            found.append((m.start(), name))
    got = [n for _, n in sorted(found)]

    if got != EXPECTED_CANON:
        _fail(f"the CANON in ph_basics.py is now {got}, expected "
              f"{EXPECTED_CANON}. Re-ordering INPUT_TYPES shifts every saved "
              f"workflow by a slot -- that is the v584/v603 wound. Display "
              f"order belongs in the JS, never here.")


def check_round_trip(js):
    """Promises 1, 2, 3 -- the disk is unchanged by the display move."""
    base = _harness(js)
    script = base + """
const out = {};
// values a workflow saved by an OLDER version would hold, in CANON order
const onDisk = ["umt5.safetensors", "flux", "cpu",
                "clip_l.safetensors", "%s", "%s"];

// (1) load it the way the frontend does -- rows are still CANON at this point
const node = mkNode(true);
frontendConfigure(node, onDisk);
out.afterLoad = valuesByName(node);
out.orderAtLoad = names(node);

// permutation happens AFTER the load
_toDisplay(node, spec);
out.orderOnScreen = names(node);
out.valuesAfterPermute = valuesByName(node);

// (2) now save -- through the REAL save path, lifted from the source. Do not
// rebuild the dance here: a harness that reimplements it tests itself.
out.savedWidgetsValues = _serializeInCanon(node, spec,
                                           function () { return frontendSerialize(this); },
                                           []);
out.orderAfterSave = names(node);

// (3) fixed point: load what we just saved into a fresh node, save again
const node2 = mkNode(true);
frontendConfigure(node2, out.savedWidgetsValues);
_toDisplay(node2, spec);
out.savedAgain = _serializeInCanon(node2, spec,
                                   function () { return frontendSerialize(this); },
                                   []);

// what a node that NEVER permutes would write, for comparison
const plain = mkNode(true);
frontendConfigure(plain, onDisk);
out.plainSave = frontendSerialize(plain);

console.log(JSON.stringify(out));
""" % (CLIP_PH, CLIP_PH)
    res = json.loads(_run_js(script))

    on_disk = ["umt5.safetensors", "flux", "cpu",
               "clip_l.safetensors", CLIP_PH, CLIP_PH]
    want = dict(zip(EXPECTED_CANON, on_disk))

    if res["orderAtLoad"][:len(EXPECTED_CANON)] != EXPECTED_CANON:
        _fail(f"promise 4 BROKEN: at load time the rows must still be in CANON "
              f"order, but they are {res['orderAtLoad']} -- permuting during "
              f"the load is exactly the v603 wound")
    for name, value in want.items():
        if res["afterLoad"].get(name) != value:
            _fail(f"promise 1 BROKEN: after loading an old workflow, {name!r} "
                  f"holds {res['afterLoad'].get(name)!r} instead of {value!r} "
                  f"-- values shifted between widgets")
    if res["valuesAfterPermute"] != res["afterLoad"]:
        _fail("the permutation moved VALUES, not just rows -- it must only "
              "change the order the widgets are drawn in")
    if res["orderOnScreen"][0] != "type":
        _fail(f"`type` must sit at the top on screen, got "
              f"{res['orderOnScreen'][0]!r}")
    if res["orderOnScreen"][-1] != "pls_remove_slot":
        _fail(f"promise 5 BROKEN: the remove button must stay LAST in display "
              f"order too, got {res['orderOnScreen'][-1]!r}")
    if res["savedWidgetsValues"] != res["plainSave"]:
        _fail(f"promise 2 BROKEN: this version writes {res['savedWidgetsValues']} "
              f"where an unpermuted node writes {res['plainSave']} -- the file "
              f"on disk must never learn that the display moved")
    if res["savedWidgetsValues"] != on_disk:
        _fail(f"promise 2 BROKEN: saved widgets_values is "
              f"{res['savedWidgetsValues']}, expected the canon order {on_disk}")
    if res["savedAgain"] != res["savedWidgetsValues"]:
        _fail(f"promise 3 BROKEN: save -> load -> save is not a fixed point "
              f"({res['savedAgain']} vs {res['savedWidgetsValues']})")
    if res["orderAfterSave"] != res["orderOnScreen"]:
        _fail("after saving, the node must be back in DISPLAY order on screen "
              f"({res['orderAfterSave']})")


def check_button_and_label(js):
    """Promises 5, 6 -- button last in canon order too, and no slot number."""
    base = _harness(js)
    script = base + """
const node = mkNode(true);
_toCanon(node, spec);               // no-op: already canon
const canonOrder = names(node);
_toDisplay(node, spec);
_toCanon(node, spec);
console.log(JSON.stringify({ canonOrder, afterRound: names(node) }));
"""
    res = json.loads(_run_js(script))
    if res["afterRound"][-1] != "pls_remove_slot":
        _fail(f"the remove button must stay last after a display/canon round "
              f"trip, got {res['afterRound'][-1]!r}")
    if res["afterRound"][:len(EXPECTED_CANON)] != EXPECTED_CANON:
        _fail(f"canon order not restored: {res['afterRound']}")

    if "remove slot ${" in js or "remove slot ' + " in js:
        _fail("the button label still interpolates a slot number -- with slot 1 "
              "filled and slot 2 empty it said 'remove slot 1' while ROW 2 "
              "disappeared; no number can be right for both readings")
    if "REMOVE_GLYPH" not in js:
        _fail("the bare glyph label is gone")
    if "#ff8c00" not in js:
        _fail("the amber tone is gone from the remove button")
    if "w.draw = _drawRemove" not in js:
        _fail("the button no longer paints itself -- the amber X is gone")


def check_serialize_override(js):
    """The save path must hook serialize(), not onSerialize()."""
    if "nodeType.prototype.serialize = function" not in js:
        _fail("serialize() is not overridden -- without it the display order "
              "reaches the disk")
    # An ASSIGNMENT, not the word: the file explains in prose why onSerialize is
    # the wrong hook, and a guard that punishes its own documentation is a guard
    # that will be edited to shut up.
    if "onSerialize =" in js or "onSerialize=" in js:
        _fail("onSerialize must not be hooked: it is a callback the base method "
              "chooses to invoke, and this frontend has been observed declining "
              "to invoke a base method. serialize() is the method itself.")
    if "} finally {" not in js:
        _fail("the canon/display swap must be in a try/finally -- a throw in "
              "the base serialize would otherwise leave the node parked in "
              "canon order on screen")


def main():
    js = _read("web", "js", "ph_basics.js")
    check_canon_contract(js)
    check_round_trip(js)
    check_button_and_label(js)
    check_serialize_override(js)
    print("PASS: v718 display order -- type on top on screen, canon untouched "
          "on disk, round trip is a fixed point, button last and unnumbered")
    sys.exit(0)


if __name__ == "__main__":
    main()
