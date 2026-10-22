"""v965 -- the Core jump (Core master / frontend 1.53.6) and the Keyframes field.

Three promises, each measured 19.09.2026 before it was written:

  H  web/js/uls_vue_hidden.js -- on a frontend whose widgets derive `hidden`
     FROM options.hidden through an accessor (1.53.6: BaseWidget
     `get hidden(){return this._state.options.hidden}`), the switch point
     must NOT define a getter on options.hidden (that recursed until
     RangeError and no workflow loaded). It writes options.hidden as a value
     for the forms the frontend cannot see (a "pls-hidden-" type, type
     "hidden", a zero-height row) and leaves the rest to the frontend. On a
     frontend without the accessor (1.49.6) the v936 getter mirror stays.
     Driven in node with two widget shapes: a plain object (no accessor) and
     a class with the 1.53.6 accessor pair.
  P  nodes/uls_pdd_apply.py PDDHead -- Core master's FinalLayer.forward opens
     with `n = self.video_out.weight.shape[0] // self.video_out.out_features`
     (commit 2504e68d). The stand-in must expose `weight` as ONE block
     ([out, in]) so n == 1 and Core takes the path that calls our forward.
     Driven with torch against the real class.
  K  nodes/ph_minimax_keyframes.py + web/js/ph_minimax_keyframes.js -- the
     prompt that ran is handed back over ui.pls_kf and shown in a read-only
     DOM field that is visible exactly while the `prompt` input is linked;
     prompt_text is output slot 3 (appended). JS driven in node with a stub
     node; backend pinned by test_v964.

Mutations for H and K; P is pinned by shape.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HID = os.path.join(ROOT, "web", "js", "uls_vue_hidden.js")
KFJ = os.path.join(ROOT, "web", "js", "ph_minimax_keyframes.js")
PDD = os.path.join(ROOT, "nodes", "uls_pdd_apply.py")
FAILED = []


def _fail(m):
    FAILED.append(m)
    print("FAIL: " + m)


def _ok(m):
    print("ok  : " + m)


def _lift_js(src, name):
    """The full text of one top-level function/export function by name."""
    m = re.search(r"(?:export )?function %s\s*\(" % re.escape(name), src)
    if not m:
        raise RuntimeError("function %s not found" % name)
    i = src.index("{", m.end() - 1)
    depth, j = 0, i
    while j < len(src):
        c = src[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    return src[m.start():j + 1].replace("export ", "")


def _run_node(js):
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False, encoding="utf-8") as fh:
        fh.write(js)
        path = fh.name
    try:
        p = subprocess.run(["node", path], capture_output=True, text=True, timeout=60)
    finally:
        os.unlink(path)
    if p.returncode != 0:
        return None, p.stderr[-800:]
    return json.loads(p.stdout.strip().splitlines()[-1]), ""


# ===================================================================== H
hid_src = open(HID, encoding="utf-8").read()

H_STUBS = """
const app = { graph: null, canvas: null, registerExtension() {} };
const HIDDEN_TYPE_PREFIX = "pls-hidden-";
"""


def h_build(src):
    fns = "\n".join(_lift_js(src, n) for n in
                    ("zeroHeight", "classicHidden", "hiddenIsAccessor", "mirrorMode",
                     "syncValue", "mirrorWidget"))
    return H_STUBS + fns + """
// --- the 1.53.6 widget shape: hidden is an ACCESSOR over options.hidden ---
class BaseWidget153 {
    constructor(o) { this._state = { options: o || {} }; }
    get options() { return this._state.options; }
    set options(v) { this._state.options = v || {}; }
    get hidden() { return this._state.options.hidden; }
    set hidden(v) { this._state.options.hidden = v; }
}
const out = {};
// h1: plain object (1.49.6 shape) keeps the getter mirror
const plain = { type: "number", options: {} };
out.h1_mode = mirrorMode(plain);
mirrorWidget(plain);
const d1 = Object.getOwnPropertyDescriptor(plain.options, "hidden");
out.h1 = out.h1_mode === "getter" && !!(d1 && typeof d1.get === "function");
// h2: accessor shape -> value mode, NO getter defined on options
const acc = new BaseWidget153({}); acc.type = "number";
out.h2_mode = mirrorMode(acc);
mirrorWidget(acc);
const d2 = Object.getOwnPropertyDescriptor(acc.options, "hidden");
out.h2 = out.h2_mode === "value" && !(d2 && typeof d2.get === "function");
// h3: a "pls-hidden-" widget on the accessor shape ends up hidden through options.hidden
const accHid = new BaseWidget153({}); accHid.type = "pls-hidden-number";
mirrorWidget(accHid);
out.h3 = accHid.options.hidden === true && accHid.hidden === true;
// h4: no recursion -- reading hidden after the mirror must not blow the stack
let blew = false;
try { for (let i = 0; i < 2000; i++) { void accHid.hidden; void classicHidden(accHid); } }
catch (e) { blew = true; }
out.h4 = !blew;
// h5: a visible widget on the accessor shape stays visible (nothing written)
const accVis = new BaseWidget153({}); accVis.type = "number";
mirrorWidget(accVis);
out.h5 = accVis.options.hidden !== true && accVis.hidden !== true;
// h6: the frontend hiding through options.hidden reads back as hidden classically
const accFe = new BaseWidget153({ hidden: true }); accFe.type = "number";
mirrorWidget(accFe);
out.h6 = classicHidden(accFe) === true;
// h7: mirrorWidget is idempotent in value mode (first call true, second false)
const accTwice = new BaseWidget153({}); accTwice.type = "pls-hidden-x";
out.h7 = mirrorWidget(accTwice) === true && mirrorWidget(accTwice) === false;
console.log(JSON.stringify(out));
"""


res, err = _run_node(h_build(hid_src))
if res is None:
    _fail("H harness did not run: " + err)
else:
    for k in ("h1", "h2", "h3", "h4", "h5", "h6", "h7"):
        (_ok if res.get(k) else _fail)("H %s (%s)" % (k, {
            "h1": "plain widget: getter mirror as before",
            "h2": "accessor widget: value mode, no getter on options",
            "h3": "accessor + pls-hidden- type -> options.hidden = true",
            "h4": "no recursion reading hidden after the mirror",
            "h5": "visible accessor widget untouched",
            "h6": "frontend-hidden reads back as classic hidden",
            "h7": "value mode idempotent"}[k]))

H_MUTATIONS = [
    ("M1 the accessor is never detected (getter mirror everywhere -> recursion)",
     "        if (d) return typeof d.get === \"function\";\n", "        if (d) return false;\n", "h4"),
    ("M2 value mode never writes the verdict",
     "    if (classicHidden(w) && o.hidden !== true) {\n        o.hidden = true;\n        return true;\n    }\n",
     "    if (false) {\n        o.hidden = true;\n        return true;\n    }\n", "h3"),
    ("M3 value mode defines the getter after all",
     "        syncValue(w);\n        return first;\n    }\n",
     "        syncValue(w);\n        Object.defineProperty(w.options, \"hidden\", { configurable: true, get() { return classicHidden(w); } });\n        return first;\n    }\n", "h2"),
]
caught = 0
for name, a, b, key in H_MUTATIONS:
    if hid_src.count(a) != 1:
        _fail("H mutation anchor broken: " + name)
        continue
    r, e = _run_node(h_build(hid_src.replace(a, b, 1)))
    hit = r is None or not r.get(key)
    caught += hit
    print("  %s %s" % ("caught:" if hit else "MISSED:", name))
    if not hit:
        _fail("H mutation survived: " + name)
print("  H mutations caught %d/%d" % (caught, len(H_MUTATIONS)))

# ===================================================================== P
try:
    import torch
    sys.path.insert(0, os.path.join(ROOT, "nodes"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("uls_pdd_apply_v965", PDD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    Head = mod._head_cls()
    n_blocks, out_f, in_f = 8, 12, 7
    fused = (torch.randn(n_blocks, out_f, in_f), torch.randn(n_blocks, out_f))

    class _Holder(object):
        sigma = None

    head = Head(torch.nn.Linear(in_f, out_f), fused, _Holder(), None, "video")
    w = head.weight
    ok = tuple(w.shape) == (out_f, in_f) and int(w.shape[0]) // head.out_features == 1 \
        and torch.equal(w, fused[0][0]) and torch.equal(head.bias, fused[1][0])
    (_ok if ok else _fail)("P PDDHead.weight is ONE block [out, in] -> Core master computes n == 1 "
                           "and calls our forward (shape %s)" % (tuple(w.shape),))
    head2 = Head(torch.nn.Linear(in_f, out_f), (fused[0], None), _Holder(), None, "audio")
    (_ok if head2.bias is None else _fail)("P a bias-less bank reports bias None")
except Exception as exc:  # loud, never vacuum-green
    _fail("P could not drive PDDHead (%r) -- torch is required" % (exc,))

# the Core-side reason, pinned where it lives when Core is reachable
core_model = None
for up in (os.path.dirname(ROOT), os.path.dirname(os.path.dirname(ROOT))):
    cand = os.path.join(up, "comfy", "ldm", "minimax", "model.py")
    if os.path.exists(cand):
        core_model = cand
        break
if core_model:
    msrc = open(core_model, encoding="utf-8", errors="replace").read()
    if "self.video_out.weight.shape[0] // self.video_out.out_features" in msrc:
        print("note: this Core computes n from video_out.weight (master rule) -- "
              "the weight property is load-bearing here.")
    else:
        print("note: this Core does not read video_out.weight in FinalLayer "
              "(7a131a3a rule) -- the weight property is dormant here.")

# ===================================================================== K
kf_src = open(KFJ, encoding="utf-8").read()
static = [
    ("no canvas painting (a DOM field only -- Vue wraps it itself)",
     "onDrawForeground" not in kf_src and "onDrawBackground" not in kf_src),
    ("the field never serialises", "serialize: false" in kf_src and "w.serialize = false" in kf_src),
    ("hidden field costs no height", "[0, -4]" in kf_src),
    ("filled from the backend's pls_kf payload", "message.pls_kf" in kf_src),
    ("visibility follows the prompt LINK", 'i.name === "prompt"' in kf_src and "link != null" in kf_src),
    ("refit through the ONE module, width never touched (v897)",
     # v1022: the import also brings fieldFloor/growToFloor -- what is pinned is
     # that refit comes from the ONE module, not the exact list
     re.search(r'import \{ refit\b[^}]*\} from "\./ph_widget_vis\.js"', kf_src) is not None
     and "setSize(node.computeSize())" not in kf_src),
    ("read-only", "el.readOnly = true" in kf_src),
]
for label, good in static:
    (_ok if good else _fail)("K static: " + label)


def k_build(src):
    fns = "\n".join(_lift_js(src, n) for n in ("_w", "_promptLinked", "_makeField", "_sync", "_fill"))
    return """
const FIELD = "pls_kf_view"; const FIELD_H = 132;
const document = { createElement() { return { style: {}, classList: { add() {} }, value: "" }; } };
// ph_widget_vis.refit stand-in: HEIGHT only, exactly like the real one (test_v897 pins that)
function refit(node) { node.setSize([node.size ? node.size[0] : 300, node.computeSize()[1]]); }
function mkNode(linked) {
    const n = { widgets: [], inputs: [{ name: "prompt", link: linked ? 7 : null }], _sizes: 0,
        addDOMWidget(name, type, el, opts) { const w = { name, type, element: el, options: opts || {} }; this.widgets.push(w); return w; },
        size: [777, 200], computeSize() { return [300, 200]; }, setSize(sz) { this._sizes++; this.size = sz; }, setDirtyCanvas() {} };
    return n;
}
""" + fns + """
const out = {};
// k1: unlinked -> field exists, hidden, zero height
const a = mkNode(false); _makeField(a); _sync(a);
const wa = _w(a, FIELD);
out.k1 = !!wa && wa.hidden === true && wa.computeSize(300)[1] === -4 && wa.serialize === false;
// k2: linked -> shown, full height, node refit once
const b = mkNode(true); _makeField(b); _sync(b);
const wb = _w(b, FIELD);
out.k2 = !!wb && wb.hidden === false && wb.computeSize(300)[1] === FIELD_H && b._sizes === 1 && b.size[0] === 777;
// k3: filled with info + prompt
_fill(b, { prompt: "the prompt", info: "line one" });
out.k3 = wb.element.value === "line one\\n\\nthe prompt";
// k4: idempotent field creation
_makeField(b); out.k4 = b.widgets.filter(w => w.name === FIELD).length === 1;
// k5: link removed -> hidden again
b.inputs[0].link = null; _sync(b); out.k5 = wb.hidden === true;
console.log(JSON.stringify(out));
"""


res, err = _run_node(k_build(kf_src))
if res is None:
    _fail("K harness did not run: " + err)
else:
    for k in ("k1", "k2", "k3", "k4", "k5"):
        (_ok if res.get(k) else _fail)("K " + k)

K_MUTATIONS = [
    ("M1 the field ignores the link and stays hidden",
     "    w.hidden = !on;\n", "    w.hidden = true;\n", "k2"),
    ("M2 the report is dropped from the field",
     '    w.element.value = (info ? info + "\\n\\n" : "") + (prompt || "(empty prompt)");\n',
     '    w.element.value = prompt;\n', "k3"),
    ("M3 a hidden field still reserves its height",
     "        w.computeSize = (width) => (w.hidden ? [0, -4] : [width, FIELD_H]);\n",
     "        w.computeSize = (width) => [width, FIELD_H];\n", "k1"),
]
caught = 0
for name, a, b, key in K_MUTATIONS:
    if kf_src.count(a) != 1:
        _fail("K mutation anchor broken: " + name)
        continue
    r, e = _run_node(k_build(kf_src.replace(a, b, 1)))
    hit = r is None or not r.get(key)
    caught += hit
    print("  %s %s" % ("caught:" if hit else "MISSED:", name))
    if not hit:
        _fail("K mutation survived: " + name)
print("  K mutations caught %d/%d" % (caught, len(K_MUTATIONS)))

if FAILED:
    print("\n[test_v965_core_sprung] FAIL -- %d" % len(FAILED))
    sys.exit(1)
print("\n[test_v965_core_sprung] PASS -- H switch point (both frontends), P PDDHead.weight, K keyframes field")
