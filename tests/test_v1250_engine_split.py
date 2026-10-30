"""test_v1250_engine_split -- the LoRA Engine's panel follows engine_config, the rows that run.

Frank 30.10. ("hier sind btw. keine Acceleratoren geladen"): the Ritter walk workflows carried eight rows in the
hidden widget engine_config (two enabled -- the render used them) and ONE empty row in `_engine`, the panel's own
copy. The panel showed empty; its first touch would have written the empty rows into the widget. v1250: at load the
two are compared and the widget wins, said in the console.

  E1 driven (node): a split node -> the panel's rows are the widget's rows, mode/apply taken over (driven)
  E1b driven: same names, flipped enabled flags -> still corrected
  E2 driven: an agreeing pair is left alone (no warn, rows untouched)
  E3 driven: no engine_config at all -> `_engine` rules as before
  E4 the correction is said in the console with the enabled names (source pin)
  E5 pure ASCII
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NAME = "test_v1250_engine_split"
FAILS = []


def need(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def main():
    print(NAME)
    js = open(os.path.join(ROOT, "web", "js", "uls_node.js"), encoding="utf-8").read()
    i = js.index("// v1250: the rows live twice")
    j = js.index("};", js.index("} catch {}", i))
    block = js[i:j]
    harness = r"""
const o_in = JSON.parse(process.argv[2]);
const warns = [];
const console = { warn: (m) => warns.push(String(m)), log: () => {} };
function newEngineRow() { return { enabled: true, name: "None", weight: 1, wClip: undefined }; }
function ensurePreview() {}
function applyNorm(v) { return ["auto", "patch", "baked"].includes(v) ? v : "auto"; }
const node = {
    id: 5,
    widgets: [{ name: "engine_config", value: o_in.widget }],
    _uls: { rows: o_in.panel.rows, mode: o_in.panel.mode || "SEQ", apply: "auto", dareVariant: "channel" },
    _engineResize() { this.resized = true; },
};
const o = { widgets_values: [o_in.widget] };
(function () {
""" + block.replace("this.", "node.") + r"""
})();
process.stdout.write(JSON.stringify({ rows: node._uls.rows, mode: node._uls.mode, apply: node._uls.apply, warns, resized: !!node.resized }));
"""
    hp = os.path.join("/tmp", "v1250_h.js")
    open(hp, "w").write(harness)

    def drive(widget, panel):
        r = subprocess.run(["node", hp, json.dumps({"widget": widget, "panel": panel})], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-600:])
        return json.loads(r.stdout)

    eight = {"rows": [{"enabled": i in (1, 6), "name": "Minimax_H3\\acc_%d.safetensors" % i, "weight": 1} for i in range(8)],
             "mode": "SEQ", "dare_variant": "channel", "apply": "patch"}
    d = drive(json.dumps(eight), {"rows": [{"enabled": True, "name": "None", "weight": 1}]})
    need(len(d["rows"]) == 8 and [r["name"] for r in d["rows"] if r["enabled"]] == ["Minimax_H3\\acc_1.safetensors", "Minimax_H3\\acc_6.safetensors"]
         and d["apply"] == "patch" and d["resized"], "E1 split node -> the panel carries the widget's eight rows (two enabled), apply taken over")
    need(len(d["warns"]) == 1 and "acc_1.safetensors, Minimax_H3\\acc_6.safetensors" in d["warns"][0], "E4 said in the console with the enabled names")
    flipped = [dict(r, enabled=not r["enabled"]) for r in eight["rows"]]
    d1b = drive(json.dumps(eight), {"rows": flipped})
    need([r["enabled"] for r in d1b["rows"]] == [r["enabled"] for r in eight["rows"]] and len(d1b["warns"]) == 1,
         "E1b same names, flipped enabled flags -> still corrected (the flag is part of the comparison)")
    d2 = drive(json.dumps(eight), {"rows": [dict(r) for r in eight["rows"]]})
    need(d2["warns"] == [] and not d2["resized"] and len(d2["rows"]) == 8, "E2 an agreeing pair is left alone")
    d3 = drive("", {"rows": [{"enabled": True, "name": "panel_only.safetensors", "weight": 1}]})
    need(d3["warns"] == [] and d3["rows"][0]["name"] == "panel_only.safetensors", "E3 no engine_config -> `_engine` rules")
    raw = open(os.path.abspath(__file__), "rb").read()
    need(all(c < 128 for c in raw), "E5 pure ASCII")
    if FAILS:
        print("[%s] FAIL -- %d" % (NAME, len(FAILS)))
        return 1
    print("[%s] PASS -- the panel follows the rows that run" % NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
