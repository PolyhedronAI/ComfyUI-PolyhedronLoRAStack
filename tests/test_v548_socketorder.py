"""Guard v548 -- Power Upscale: input-socket reorder survives wired widget inputs.

The v547 mechanic bailed SILENTLY (strict length-9 guard) whenever a widget was
wired as an input (e.g. 'seed' fed by the Seed node), because such widgets occupy
an entry in node.inputs - caught live 2026-07-11: widgets reordered, sockets did
not. This guard extracts INPUT_DISPLAY_ORDER and _reorderInputsToDisplay VERBATIM
from web/js/ph_power_upscale.js and runs them in node against that exact case:
10 inputs (the 9 sockets in raw backend order + a wired 'seed'), a populated
links map. Measured, not believed:
  - the nine sockets land in display order, the seed input SURVIVES at the tail
  - every link's target_slot is repaired to the new index (full array)
  - a second run is a no-op that still reports true (idempotency)
  - an incomplete socket set no-ops LOUDLY (console.warn), never silently
Structural pins: per-file load banner (v531 doctrine; the banner carries the
FILE's last-touched version - update it in the cut that next touches this file),
loadedGraphNode hook, onConfigure call site.
Script-style: exit 0 = pass. Requires node (same dependency as GATE-2).
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JS_PATH = os.path.join(ROOT, "web", "js", "ph_power_upscale.js")


def _fail(msg):
    print("[test_v548_socketorder] FAIL: " + msg)
    sys.exit(1)


def main():
    js = open(JS_PATH, encoding="utf-8").read()

    # ---- structural pins --------------------------------------------------
    # The EXACT banner version is pinned by the file's newest guard (v549+);
    # here we pin existence + format only (v531 doctrine).
    if not re.search(r"\[PLS\] ph_power_upscale\.js v\d+ loaded", js):
        _fail("per-file load banner missing (v531 doctrine: Firefox caches each "
              "JS file individually; the missing banner hid the v547 no-op)")
    if "loadedGraphNode(" not in js:
        _fail("loadedGraphNode hook missing (second, idempotent re-apply point)")
    if "_reorderInputsToDisplay(this)" not in js:
        _fail("onConfigure no longer re-applies the input display order")
    if "[PLS v548] PU socket reorder skipped" not in js:
        _fail("loud-skip warn missing - a SILENT no-op is what cost the v547 "
              "live round")

    m_order = re.search(r"const INPUT_DISPLAY_ORDER = \[[\s\S]*?\];", js)
    if not m_order:
        _fail("INPUT_DISPLAY_ORDER not found")
    m_fn = re.search(r"function _reorderInputsToDisplay\(node\) \{[\s\S]*?\n\}", js)
    if not m_fn:
        _fail("_reorderInputsToDisplay not found")

    # ---- behavioural harness: the exact live failure case ------------------
    harness = m_order.group(0) + "\n" + m_fn.group(0) + "\n" + r"""
const warned = [];
console.warn = (m) => warned.push(String(m));

function mk(name, link, widget) {
    const o = { name: name, link: (link === undefined ? null : link) };
    if (widget) o.widget = { name: name };
    return o;
}
// Raw {...required, ...optional} backend order + a WIRED seed widget-input:
// exactly the node that killed the v547 reorder live.
const inputs = [
    mk("model", 1), mk("positive", 2), mk("negative", 3), mk("vae", null),
    mk("upscale_model", 4), mk("image", 5), mk("video", null),
    mk("model_low", 6), mk("upscale_model_low", 7), mk("seed", 8, true),
];
const links = new Map();
for (const [lid, slot] of [[1,0],[2,1],[3,2],[4,4],[5,5],[6,7],[7,8],[8,9]]) {
    links.set(lid, { id: lid, target_slot: slot });
}
let dirty = 0;
const node = { inputs: inputs, graph: { links: links },
               setDirtyCanvas: function () { dirty++; } };

if (_reorderInputsToDisplay(node) !== true) {
    console.error("FAIL: returned false on the wired-seed node (the v547 bug)");
    process.exit(1);
}
const names = node.inputs.map((i) => i.name);
const want = ["image","video","model","model_low","positive","negative","vae",
              "upscale_model","upscale_model_low","seed"];
if (JSON.stringify(names) !== JSON.stringify(want)) {
    console.error("FAIL order: " + JSON.stringify(names));
    process.exit(1);
}
for (let i = 0; i < node.inputs.length; i++) {
    const lid = node.inputs[i].link;
    if (lid == null) continue;
    if (links.get(lid).target_slot !== i) {
        console.error("FAIL target_slot: link " + lid + " points at "
            + links.get(lid).target_slot + ", slot is " + i);
        process.exit(1);
    }
}
if (links.get(8).target_slot !== 9) {
    console.error("FAIL: the wired seed link was not repaired to the tail slot");
    process.exit(1);
}
if (warned.length !== 0) {
    console.error("FAIL: the happy path must not warn");
    process.exit(1);
}
if (_reorderInputsToDisplay(node) !== true) {
    console.error("FAIL idempotency: second run must still report true");
    process.exit(1);
}
// Incomplete socket set -> loud no-op, order untouched.
const broken = { inputs: [mk("model", null), mk("positive", null)],
                 graph: { links: new Map() } };
if (_reorderInputsToDisplay(broken) !== false) {
    console.error("FAIL: incomplete socket set must no-op (return false)");
    process.exit(1);
}
if (warned.length === 0 || warned[0].indexOf("[PLS v548]") !== 0) {
    console.error("FAIL: the skip must WARN with the [PLS v548] marker");
    process.exit(1);
}
if (broken.inputs.map((i) => i.name).join(",") !== "model,positive") {
    console.error("FAIL: a skipped reorder must leave the inputs untouched");
    process.exit(1);
}
console.log("OK");
"""

    tmp = tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                      encoding="utf-8")
    try:
        tmp.write(harness)
        tmp.close()
        proc = subprocess.run(["node", tmp.name], capture_output=True, text=True)
    finally:
        os.unlink(tmp.name)

    if proc.returncode != 0 or "OK" not in proc.stdout:
        _fail("node harness failed:\n" + proc.stdout + proc.stderr)

    print("PASS: v548 socket reorder -- wired-seed case survives, target_slot "
          "repaired, idempotent, loud skip")
    sys.exit(0)


if __name__ == "__main__":
    main()
