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
 * What moved is said once per load, top right, in the house bubble.
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

function isOurs(node) {
    const cls = String(node?.comfyClass || node?.type || "");
    const mod = String(node?.constructor?.nodeData?.python_module || "");
    return cls.startsWith("ULS") || mod.includes("PolyhedronLoRAStack");
}

// Returns the moves made on this node: [{ from, to }]
export function relocateNode(node) {
    const moves = [];
    for (const w of node?.widgets || []) {
        const opts = w && w.options && w.options.values;
        if (!Array.isArray(opts) || typeof w.value !== "string") continue;
        if (opts.includes(w.value)) continue;
        const hit = relocated(opts, w.value);
        if (hit === undefined || hit === w.value) continue;
        if (stripSize(hit) !== stripSize(w.value)) moves.push({ from: stripSize(w.value), to: stripSize(hit) });
        w.value = hit;
    }
    return moves;
}

let _css = false;
let _pending = [];
let _timer = null;

function ensureCSS() {
    if (_css) return;
    _css = true;
    const st = document.createElement("style");
    st.textContent = ""
        + ".ph-relocate-toast { position:fixed; top:16px; right:16px; z-index:10000;"
        + " max-width:460px; padding:10px 12px; border-radius:8px; background:#1d1d1f;"
        + " border:1px solid #3a3a3d; box-shadow:0 6px 24px rgba(0,0,0,.45);"
        + " font:12px/1.35 system-ui,sans-serif; color:#ddd; }"
        + ".ph-relocate-toast h4 { margin:0 0 4px; font-size:12px; color:#ff8c00; }"
        + ".ph-relocate-toast p { margin:0 0 2px; color:#bbb; word-break:break-all; }";
    document.head.appendChild(st);
}

function flush() {
    _timer = null;
    const moves = _pending;
    _pending = [];
    if (!moves.length) return;
    try {
        ensureCSS();
        const box = document.createElement("div");
        box.className = "ph-relocate-toast";
        box.setAttribute("data-ph-overlay", "1");
        const h = document.createElement("h4");
        h.textContent = "\u2b21 " + moves.length + (moves.length === 1 ? " model found at a new place" : " models found at a new place");
        box.appendChild(h);
        for (const m of moves.slice(0, 6)) {
            const p = document.createElement("p");
            p.textContent = m.from + "  \u2192  " + m.to;
            box.appendChild(p);
        }
        if (moves.length > 6) {
            const p = document.createElement("p");
            p.textContent = "... and " + (moves.length - 6) + " more";
            box.appendChild(p);
        }
        document.body.appendChild(box);
        setTimeout(() => { try { box.remove(); } catch (e) { } }, 12000);
    } catch (e) { /* never break the ui */ }
}

app.registerExtension({
    name: "polyhedron.modelRelocate",
    loadedGraphNode(node) {
        try {
            if (!isOurs(node)) return;
            const moves = relocateNode(node);
            if (!moves.length) return;
            for (const m of moves) console.info("[PLS] model moved: " + m.from + " -> " + m.to);
            _pending.push(...moves);
            if (_timer) clearTimeout(_timer);
            _timer = setTimeout(flush, 400);
        } catch (e) { /* never break the ui */ }
    },
});
