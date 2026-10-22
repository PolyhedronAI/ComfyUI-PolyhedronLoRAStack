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
import { modelStatus, noticeMissing, stripModelLabel, nodeActive, watchMode } from "./ph_model_door.js";

/*
 * v1059: a run no longer downloads a RIFE checkpoint. When the picked one is
 * not on disk, the house notice offers it (ph_model_door.js) -- once per file
 * and session, on load and on every pick.
 */
async function checkCkpt(node) {
    const w = (node.widgets || []).find((x) => x && x.name === "ckpt_name");
    if (!w || !nodeActive(node)) return;            // v1204: bypassed / muted -- nothing is needed
    const name = stripModelLabel(w.value);
    let models = [];
    try { models = await modelStatus("rife"); } catch (e) { return; }
    const m = models.find((x) => x.name === name);
    if (m && !m.installed) noticeMissing(m, node.title || "Polyhedron Interpolate", null, node);
}

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
                try {
                    const w = (this.widgets || []).find((x) => x && x.name === "ckpt_name");
                    if (w && !w._plsDoorBound) {
                        const orig = w.callback;
                        const node = this;
                        w.callback = function () {
                            const r = orig ? orig.apply(this, arguments) : undefined;
                            checkCkpt(node);
                            return r;
                        };
                        w._plsDoorBound = true;
                    }
                    checkCkpt(this);
                    watchMode(this, (n) => checkCkpt(n));     // v1204: re-checked when the node comes back
                } catch (e) { /* never break the ui */ }
            }, 0);
        };
    },
});

console.log("[PLS] ph_interpolate.js v1059 loaded");
