#!/usr/bin/env python3
"""v974 -- the Sigma List shows what it does.

v973 made a preset override the widgets at RUN time, which was right, and left
the node lying on screen, which was not: `shift` read 1.00 while the run used
12.0, and the only witness was a console line. A preset you cannot see is half
a preset -- the same trap wearing a different hat.

Picking a preset now WRITES its values into the widgets and greys them out;
back to `custom` hands them back. The v973 backend override stays as the belt
to these braces, for workflows saved before this cut.

The values are SERVED from the one table (/pls/sigma_presets). A copy in the
JS would be a second place for one truth, and this tree has paid for that
lesson more than once.

  U1  the route is registered, and its handler reads SIGMA_PRESETS -- it does
      not carry a table of its own
  U2  the JS holds NO sigma grid of its own: no long decimal run anywhere in
      the file (the anti-drift promise, checked against the real numbers)
  U3  the JS parses as a module
  U4  RUN: picking a preset fills all three widgets from the served table and
      disables them (the grid compared as numbers -- JSON spells 1.0 as 1)
  U5  RUN: switching back to custom restores exactly what the user had typed
      and re-enables the widgets
  U6  RUN: a preset name this build does not know leaves the widgets editable
      instead of freezing empty fields
  U7  RUN: switching preset -> preset -> custom still restores the USER's
      values, not the first preset's (the snapshot is taken once)
  U8  an unreachable table leaves every widget editable -- the node never
      freezes a field it cannot fill
  U9  ADDED v975 after a field find: the grid must be visible in the DOM text
      area, not merely present in widget.value. v974 set only the model value
      and shipped an empty, greyed-out field -- the harness could not see it
      because its fake widget had no inputEl. The harness now carries one

MUTATION PROBE (run 20.09.2026, each mutation applied and reverted):
  1. setDisabled no-op'd                          -> U4 caught
  2. rememberUser overwrites on every switch      -> U7 caught
  3. restoreUser deleted                          -> U5 caught
  4. unknown preset treated as a preset           -> U6 caught
  5. the hyperflow grid pasted into the JS        -> U2 caught
  6. fetch failure path freezes the widgets       -> U8 caught
  7. setValue writes widget.value only (the v974 bug) -> U9 caught
"""
import os, re, sys, json, shutil, subprocess, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JS = os.path.join(ROOT, "web", "js", "ph_sigma_list.js")
ROUTES = os.path.join(ROOT, "nodes", "ph_sigma_routes.py")   # public build: the sigma routes live in their own module (v384)
sys.path.insert(0, ROOT)

failures = []
ran = []


def check(cond, msg):
    ran.append(msg)
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def _read(p):
    return open(p, encoding="utf-8").read()


NODE = shutil.which("node")
js_src = _read(JS)
routes_src = _read(ROUTES)

# --- U1 one table, served ---------------------------------------------------
check('("GET",  "/pls/sigma_presets", handle_sigma_presets_get)' in routes_src
      or '"/pls/sigma_presets"' in routes_src,
      "U1 the route /pls/sigma_presets is registered")
m = re.search(r"async def handle_sigma_presets_get.*?(?=\nasync def |\ndef )",
              routes_src, re.S)
check(m is not None, "U1 the handler exists")
if m:
    body = m.group(0)
    check("SIGMA_PRESETS" in body,
          "U1 the handler reads SIGMA_PRESETS instead of listing grids itself")
    check(not re.search(r"\d+\.\d{4,}", body),
          "U1 the handler carries no sigma numbers of its own")

# --- U2 the JS carries no grid ---------------------------------------------
from nodes.wan_sigma_schedule import SIGMA_PRESETS
numbers = []
for name, entry in SIGMA_PRESETS.items():
    if entry[0]:
        numbers += [repr(v) for v in entry[0]]
leaked = [n for n in set(numbers) if len(n) > 4 and n in js_src]
check(not leaked, "U2 no preset number appears in the JS (leaked: %s)" % leaked)
check(not re.search(r"\d+\.\d{4,}", js_src),
      "U2 the JS holds no long-decimal run at all")
check("/pls/sigma_presets" in js_src, "U2 the JS asks the route for the table")

# --- U3..U8 the JS, RUN in a harness ---------------------------------------
if NODE is None:
    check(False, "U3 node is available to run the JS harness")
else:
    tmp = tempfile.mkdtemp(prefix="v974_")
    scripts = os.path.join(tmp, "scripts")
    os.makedirs(scripts, exist_ok=True)
    # The ONLY import is core's app.js, which is not ours to ship -- stub it,
    # and keep the real ph_sigma_list.js byte-for-byte beside it.
    with open(os.path.join(scripts, "app.js"), "w", encoding="utf-8") as f:
        f.write("export const app = { _ext: null,"
                " registerExtension(e) { this._ext = e; } };\n")
    work = os.path.join(tmp, "web", "js")
    os.makedirs(work, exist_ok=True)
    shutil.copyfile(JS, os.path.join(work, "ph_sigma_list.js"))
    # ph_sigma_list.js imports refit from its sibling. A harness that strips or
    # stubs an import measures a file the pack does not ship -- the standing
    # merkposten since v737. Put the REAL sibling beside it.
    shutil.copyfile(os.path.join(ROOT, "web", "js", "ph_widget_vis.js"),
                    os.path.join(work, "ph_widget_vis.js"))
    # v984: the shared sigma plot module is a real sibling too.
    shutil.copyfile(os.path.join(ROOT, "web", "js", "ph_sigma_plot.js"),
                    os.path.join(work, "ph_sigma_plot.js"))

    served = {}
    for name, entry in SIGMA_PRESETS.items():
        grid, shift, zero, note = entry
        served[name] = {"grid": list(grid) if grid else None, "shift": shift,
                        "enforce_terminal_zero": zero, "note": note}

    harness = r"""
import { app } from "./scripts/app.js";
const SERVED = %s;
const MODE = process.argv[2] || "ok";

globalThis.fetch = async (url) => {
    if (String(url).includes("sigma_preview")) {      // v976 curve preview
        return { ok: true, json: async () => ({ ok: true, sigmas: [1, 0.5, 0],
                 steps: 2, shift: 1, source: "text", note: "" }) };
    }
    if (MODE === "down") return { ok: false, json: async () => ({}) };
    return { ok: true, json: async () => ({ ok: true, presets: SERVED }) };
};

await import("./web/js/ph_sigma_list.js");
const ext = app._ext;
const nodeType = function () {};
// v976 reserves a strip via computeSize and draws into it: the harness has to
// carry the node fields that needs, or it measures a node ComfyUI never makes.
nodeType.prototype = { computeSize: () => [300, 200] };
await ext.beforeRegisterNodeDef(nodeType, { name: "ULSSigmaList" });

function mkNode() {
    const node = Object.create(nodeType.prototype);
    node.widgets = [
        // A REAL multiline widget: the model value and what the eye sees are
        // two different things. v974 set only the first and shipped an empty
        // field. The harness now carries both, so that class of bug fails here.
        { name: "sigmas_text", value: "MY, OWN, LIST",
          inputEl: { value: "MY, OWN, LIST", readOnly: false, style: {} } },
        { name: "shift", value: 3.5 },
        { name: "enforce_terminal_zero", value: false },
        { name: "preset", value: "custom", callback: null },
    ];
    node.properties = {};
    node.size = [300, 200];
    node.flags = {};
    node.setDirtyCanvas = () => {};
    nodeType.prototype.onNodeCreated.call(node);
    return node;
}
const W = (n, name) => n.widgets.find((w) => w.name === name);
const settle = () => new Promise((r) => setTimeout(r, 30));

const node = mkNode();
await settle();
const out = { steps: [] };
const snap = (tag) => out.steps.push({
    tag,
    text: W(node, "sigmas_text").value,
    seen: W(node, "sigmas_text").inputEl.value,
    shift: W(node, "shift").value,
    zero: W(node, "enforce_terminal_zero").value,
    disabled: ["sigmas_text", "shift", "enforce_terminal_zero"].map(
        (n) => !!W(node, n).disabled),
});

snap("initial_custom");
const pw = W(node, "preset");
pw.value = "hyperflow_8step_h3_raw"; await pw.callback(pw.value); await settle();
snap("preset_hyperflow");
pw.value = "ays_sdxl_10step"; await pw.callback(pw.value); await settle();
snap("preset_ays");
pw.value = "custom"; await pw.callback(pw.value); await settle();
snap("back_to_custom");
pw.value = "from_a_later_build"; await pw.callback(pw.value); await settle();
snap("unknown_preset");
console.log(JSON.stringify(out));
""" % json.dumps(served)

    hp = os.path.join(tmp, "harness.mjs")
    with open(hp, "w", encoding="utf-8") as f:
        f.write(harness)

    r = subprocess.run([NODE, "--check", os.path.join(work, "ph_sigma_list.js")],
                       capture_output=True, text=True)
    check(r.returncode == 0, "U3 the JS parses as a module (%s)" % (r.stderr.strip()[:60]))

    def run(mode):
        rr = subprocess.run([NODE, hp, mode], capture_output=True, text=True, cwd=tmp)
        if rr.returncode != 0:
            print(rr.stderr[-900:])
            return None
        return json.loads(rr.stdout.strip().splitlines()[-1])

    got = run("ok")
    check(got is not None, "U3 the harness runs the real file")
    if got:
        st = {s["tag"]: s for s in got["steps"]}
        HF = SIGMA_PRESETS["hyperflow_8step_h3_raw"]

        s = st["preset_hyperflow"]
        # Compared as NUMBERS, not as a string: JSON turns 1.0 into 1, so the
        # field reads "1, 0.931506, ..." -- the same grid, spelled the way
        # JavaScript spells it. The promise is the grid, not the notation.
        shown = [float(x) for x in re.split(r"[,;\s]+", s["text"]) if x]
        check(len(shown) == len(HF[0])
              and all(abs(a - b) < 1e-9 for a, b in zip(shown, HF[0])),
              "U4 the grid is written into the text widget")
        check(abs(float(s["shift"]) - HF[1]) < 1e-9,
              "U4 the preset's shift is written into the shift widget")
        check(bool(s["zero"]) is bool(HF[2]),
              "U4 the preset's terminal-zero rule is written into its widget")
        check(all(s["disabled"]), "U4 all three widgets are greyed out under a preset")

        # U9 -- the field the EYE reads, not just the model value. Field find
        # on the day v974 shipped: the grid was in widget.value and the textarea
        # was empty. A promise must be pinned where it can break.
        s = st["preset_hyperflow"]
        seen = [float(x) for x in re.split(r"[,;\s]+", s["seen"]) if x]
        check(len(seen) == len(HF[0])
              and all(abs(a - b) < 1e-9 for a, b in zip(seen, HF[0])),
              "U9 the grid is VISIBLE in the text area, not just in widget.value")

        s = st["back_to_custom"]
        check(s["seen"] == "MY, OWN, LIST",
              "U9 back on custom the visible text is the user's own again")
        check(s["text"] == "MY, OWN, LIST" and abs(float(s["shift"]) - 3.5) < 1e-9
              and bool(s["zero"]) is False,
              "U5 back on custom the user's own values return, exactly")
        check(not any(s["disabled"]), "U5 the widgets are editable again")

        s = st["unknown_preset"]
        check(not any(s["disabled"]),
              "U6 an unknown preset name leaves the widgets editable")

        check(st["back_to_custom"]["text"] == "MY, OWN, LIST",
              "U7 preset -> preset -> custom restores the USER's list, not the first preset's")

    got = run("down")
    check(got is not None, "U8 the harness runs with the table unreachable")
    if got:
        st = {s["tag"]: s for s in got["steps"]}
        check(not any(st["preset_hyperflow"]["disabled"]),
              "U8 an unreachable table leaves the widgets editable, never frozen empty")
    shutil.rmtree(tmp, ignore_errors=True)

print()
if failures:
    print("v974 sigma list ui: FAIL (%d)" % len(failures))
    for f in failures:
        print("   - " + f)
    sys.exit(1)
print("%d checks, 0 failed" % len(ran))
print("v974 sigma list ui: PASS")
