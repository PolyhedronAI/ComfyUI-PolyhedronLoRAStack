/*
 * Polyhedron Interpolate -- combo normalisation (v829).
 *
 * The ckpt_name entries carry decoration since v829 ("\u25c8 rife426.pth
 * \u00b7 23 MB", the cutout's v752 form). A graph saved before that holds
 * the bare filename, which the list no longer offers -- the widget would
 * sit on a value with no matching entry. This snaps a stored value onto
 * the entry the list offers RIGHT NOW, comparing both sides stripped.
 *
 * The strip mirrors the backend's _strip_deco exactly: one trailing
 * " \u00b7 <n> <unit>" and one leading diamond, nothing broader -- the
 * v828 rule (filenames may legitimately contain dashes and middots).
 */
import { app } from "../../../scripts/app.js";

const SIZE_SUFFIX = /\s\u00b7\s[\d.,]+\s?(KB|MB|GB|TB)$/;

function stripDeco(v) {
    let out = String(v == null ? "" : v).replace(SIZE_SUFFIX, "").trimEnd();
    if (out.startsWith("\u25c8 ")) out = out.slice(2);
    return out;
}

function normalize(node) {
    for (const w of node.widgets || []) {
        if (!w || w.name !== "ckpt_name") continue;
        const opts = w.options && w.options.values;
        if (!Array.isArray(opts) || !opts.length) continue;
        if (opts.includes(w.value)) continue;
        const bare = stripDeco(w.value);
        const hit = opts.find((o) => stripDeco(o) === bare);
        if (hit !== undefined) w.value = hit;
    }
}

app.registerExtension({
    name: "polyhedron.interpolate",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "ULSInterpolate") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            // The setTimeout(0) lands AFTER LiteGraph poured
            // widgets_values in -- the ph_basics.js v828 timing.
            setTimeout(() => {
                try { normalize(this); } catch (e) { /* never break the ui */ }
            }, 0);
        };
    },
});

console.log("[PLS] ph_interpolate.js v829 loaded");
