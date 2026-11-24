/*
 * ph_model_relocate.js (v1326)
 *
 * A workflow stores a model as the name its list offered on the day it was
 * saved. Sorting the models folder into sub folders changes that name
 * ("model.safetensors" -> "Family/model.safetensors"), and the widget then
 * shows a value its list no longer offers.
 *
 * On load, every list widget of a Polyhedron node whose stored value is a
 * model file that is no longer offered is moved onto the ONE entry with the
 * same file name -- so the node shows the file it will use. No match leaves
 * the value alone; two or more matches leave it alone too (picking one would
 * be the silent-wrong-model bug). Mirrors nodes/ph_model_path.py, which does
 * the same for a queued prompt.
 *
 * v1347: no bubble any more (Frank, 20.11.: "entweder laeuft es, dann ist es doch egal").
 * A working move and an unresolved file are both said in the console only: the node
 * shows the file it will use, and a file that cannot be resolved stops the run with
 * Core's own message -- a bubble on load also fired for slots that never run.
 */

import { app } from "../../scripts/app.js";

// the size decoration of the Load nodes' lists -- same pattern as ph_basics.js / ph_basics.py
const SIZE_SUFFIX = /\s\u00b7\s[\d.,]+\s?(KB|MB|GB|TB)$/;
const MODEL_FILE = /\.(safetensors|sft|gguf|ckpt|pth|pt|bin|onnx)$/i;

export function stripSize(v) {
    return String(v == null ? "" : v).replace(SIZE_SUFFIX, "").trimEnd();
}

export function fileName(v) {
    const parts = stripSize(v).trim().split(/[\\/]+/);
    return parts[parts.length - 1].toLowerCase();
}

/*
 * The entry of `options` a stored `value` belongs to now, or undefined.
 *   - offered as it stands, or offered once the sizes are stripped: that entry
 *   - else exactly one entry with the same file name: that entry
 *   - else undefined (nothing, or several -- never a guess)
 */
export function relocated(options, value) {
    if (!Array.isArray(options) || !options.length) return undefined;
    if (options.includes(value)) return value;
    const bare = stripSize(value);
    if (!MODEL_FILE.test(bare)) return undefined;
    const same = options.find((o) => stripSize(o) === bare);
    if (same !== undefined) return same;
    const want = fileName(value);
    const seen = new Set();
    const hits = [];
    for (const o of options) {
        if (typeof o !== "string" || fileName(o) !== want) continue;
        const key = stripSize(o).split(/[\\/]+/).filter(Boolean).join("/").toLowerCase();
        if (seen.has(key)) continue;
        seen.add(key);
        hits.push(o);
    }
    return hits.length === 1 ? hits[0] : undefined;
}

// v1347: how many offered entries carry the stored value's file name (0 = missing, 2+ = ambiguous)
export function sameNameCount(options, value) {
    if (!Array.isArray(options)) return 0;
    const want = fileName(value);
    const seen = new Set();
    for (const o of options) {
        if (typeof o !== "string" || fileName(o) !== want) continue;
        seen.add(stripSize(o).split(/[\\/]+/).filter(Boolean).join("/").toLowerCase());
    }
    return seen.size;
}

function isOurs(node) {
    const cls = String(node?.comfyClass || node?.type || "");
    const mod = String(node?.constructor?.nodeData?.python_module || "");
    return cls.startsWith("ULS") || mod.includes("PolyhedronLoRAStack");
}

// Returns the moves made on this node: [{ from, to }]. v1347: a stored model file that stays
// unresolved goes into `missing` (when given) as { value, count } -- count 0 = not found, 2+ = ambiguous.
export function relocateNode(node, missing) {
    const moves = [];
    for (const w of node?.widgets || []) {
        const opts = w && w.options && w.options.values;
        if (!Array.isArray(opts) || typeof w.value !== "string") continue;
        if (opts.includes(w.value)) continue;
        const hit = relocated(opts, w.value);
        if (hit === undefined) {
            if (Array.isArray(missing) && MODEL_FILE.test(stripSize(w.value)))
                missing.push({ value: stripSize(w.value), count: sameNameCount(opts, w.value) });
            continue;
        }
        if (hit === w.value) continue;
        if (stripSize(hit) !== stripSize(w.value)) moves.push({ from: stripSize(w.value), to: stripSize(hit) });
        w.value = hit;
    }
    return moves;
}

app.registerExtension({
    name: "polyhedron.modelRelocate",
    loadedGraphNode(node) {
        try {
            if (!isOurs(node)) return;
            const missing = [];
            const moves = relocateNode(node, missing);
            // v1347: console only. A working move needs no word on screen; an unresolved file stops the run, and
            // Core names it there ("Value not in list") -- a bubble on load also fired for slots that never run.
            for (const m of moves) console.info("[PLS] model moved: " + m.from + " -> " + m.to);
            for (const m of missing) console.info("[PLS] model not found: " + m.value + (m.count > 1 ? " (" + m.count + " files of this name)" : "") + " -- only matters if this widget is used");
        } catch (e) { /* never break the ui */ }
    },
});
