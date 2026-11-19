"""test_v1345_stack_split -- the LoRA Stack's panel follows uls_config, the rows that run (twin of v1250).

Frank 18.11.: the p-h3-image LoRA never loaded and the Hippie round-2 weights never arrived. Tool-written workflows
carried the intended rows in the hidden widget uls_config, while `_uls` (the panel's copy) kept the old ones; at
load the panel restored `_uls` and its next sync overwrote the widget. v1345: at load the two are compared and the
widget wins, said in the console and in a toast.

  S1 driven (node): a split node -> the panel's rows are the widget's rows (weights into wLow/wHigh), settings taken
  S1b driven: same names, only a weight differs -> corrected
  S1c driven: same rows, only `enabled` flipped -> corrected
  S2 driven: an agreeing pair (as _ulsSync writes it: weight = wLow) is left alone -- no warn, no toast
  S3 driven: no saved uls_config (layout file / fresh node) -> nothing compared, panel untouched
  S4 driven: no `_uls` saved -> nothing compared (the widget alone already rules the backend)
  S5 driven: legacy group names compare equal after migration (no false alarm)
  S6 console + toast name the enabled rows (driven)
  S7 pure ASCII-free check not required (file predates the rule); the new block is ASCII
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NAME = "test_v1345_stack_split"
FAILS = []


def need(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def main():
    print(NAME)
    js = open(os.path.join(ROOT, "web", "js", "uls_node.js"), encoding="utf-8").read()
    i = js.index("// v1345: the Stack's twin of the Engine's v1250 split guard")
    j = js.index("// Always re-fetch group modes from backend", i)
    block = js[i:j]
    harness = r"""
const o_in = JSON.parse(process.argv[2]);
const warns = [], toasts = [];
const console = { warn: (m) => warns.push(String(m)), log: () => {} };
const app = { extensionManager: { toast: { add: (t) => toasts.push(t) } } };
function newRow() { return { enabled: true, name: "None", wHigh: 1, wLow: 1, group: "subject" }; }
function ensurePreview() {}
function applyNorm(v) { return ["auto", "patch", "baked"].includes(v) ? v : "auto"; }
const node = {
    id: 4,
    widgets: [{ name: "uls_config", value: "" }],
    _uls: { rows: o_in.panel, mult: 1, flatMode: false, groupOrder: {}, groupDare: {}, groupTrim: {}, apply: "auto" },
    _ulsResize() { this.resized = true; },
};
const o = {};
if (o_in.widget !== null) o.widgets_values = [o_in.widget];
if (o_in.saved_uls) o._uls = "{}";
(function () {
""" + block.replace("this.", "node.") + r"""
})();
process.stdout.write(JSON.stringify({ rows: node._uls.rows, mult: node._uls.mult, trim: node._uls.groupTrim,
                                      apply: node._uls.apply, warns, toasts, resized: !!node.resized }));
"""
    hp = os.path.join("/tmp", "v1345_h.js")
    open(hp, "w").write(harness)

    def drive(widget, panel, saved_uls=True):
        r = subprocess.run(["node", hp, json.dumps({"widget": widget, "panel": panel, "saved_uls": saved_uls})],
                           capture_output=True, text=True)
        if r.returncode:
            print(r.stderr[-800:])
            return None
        return json.loads(r.stdout)

    def prow(name, on, w, group="subject"):
        return {"enabled": on, "name": name, "wHigh": w, "wLow": w, "weight": w, "group": group}

    panel = [prow("a.safetensors", True, 0.4), prow("b.safetensors", True, 0.7), prow("c.safetensors", False, 0.2)]
    want = {"rows": [{"enabled": True, "name": "p-h3.safetensors", "weight": 1.0, "group": "subject"},
                     {"enabled": True, "name": "a.safetensors", "weight": 0.2, "wHigh": 0.2, "wLow": 0.2, "group": "subject"},
                     {"enabled": False, "name": "b.safetensors", "weight": 0.7, "group": "lighting"}],
            "mult": 0.8, "group_trim": {"scene": True}, "apply": "patch"}
    d = drive(json.dumps(want), panel)
    need(d is not None and [r["name"] for r in d["rows"]] == ["p-h3.safetensors", "a.safetensors", "b.safetensors"]
         and [r["enabled"] for r in d["rows"]] == [True, True, False]
         and [(r["wLow"], r["wHigh"]) for r in d["rows"]] == [(1.0, 1.0), (0.2, 0.2), (0.7, 0.7)]
         and d["rows"][2]["group"] == "scene", "S1 split node -> panel carries the widget's rows, weights into wLow/wHigh, "
                                                 "legacy group 'lighting' migrated to 'scene'")
    need(d is not None and d["mult"] == 0.8 and d["trim"] == {"scene": True} and d["apply"] == "patch" and d["resized"],
         "S1 settings taken over (mult, group_trim, apply), panel resized")
    w2 = json.dumps({"rows": [dict(prow("a.safetensors", True, 0.2)), prow("b.safetensors", True, 0.7),
                              prow("c.safetensors", False, 0.2)]})
    d = drive(w2, panel)
    need(d is not None and d["rows"][0]["wLow"] == 0.2 and len(d["warns"]) == 1, "S1b only a weight differs -> corrected")
    w3 = json.dumps({"rows": [prow("a.safetensors", True, 0.4), prow("b.safetensors", False, 0.7),
                              prow("c.safetensors", False, 0.2)]})
    d = drive(w3, panel)
    need(d is not None and d["rows"][1]["enabled"] is False and len(d["warns"]) == 1, "S1c only `enabled` flipped -> corrected")
    synced = json.dumps({"rows": [{"enabled": r["enabled"], "name": r["name"], "weight": r["wLow"], "wHigh": r["wHigh"],
                                   "wLow": r["wLow"], "group": r["group"]} for r in panel], "mult": 1})
    d = drive(synced, panel)
    need(d is not None and not d["warns"] and not d["toasts"] and not d["resized"], "S2 agreeing pair left alone")
    d = drive(None, panel)
    need(d is not None and not d["warns"] and d["rows"][0]["wLow"] == 0.4, "S3 no saved uls_config -> nothing compared")
    d = drive(json.dumps(want), panel, saved_uls=False)
    need(d is not None and not d["warns"] and d["rows"][0]["name"] == "a.safetensors", "S4 no saved _uls -> nothing compared")
    legacy = json.dumps({"rows": [dict(prow("a.safetensors", True, 0.4), group="character"), prow("b.safetensors", True, 0.7),
                                  prow("c.safetensors", False, 0.2)]})
    d = drive(legacy, panel)
    need(d is not None and not d["warns"], "S5 legacy group name equals its migration (no false alarm)")
    d = drive(json.dumps(want), panel)
    need(d is not None and len(d["warns"]) == 1 and "p-h3.safetensors 1" in d["warns"][0] and "a.safetensors 0.2" in d["warns"][0]
         and "b.safetensors" not in d["warns"][0] and len(d["toasts"]) == 1 and d["toasts"][0]["severity"] == "warn"
         and "p-h3.safetensors" in d["toasts"][0]["detail"], "S6 console + toast name the enabled rows")
    need(all(ord(ch) < 128 for ch in block), "S7 the v1345 block is ASCII")
    print("%s: %s" % (NAME, "PASS" if not FAILS else "FAIL %d" % len(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
