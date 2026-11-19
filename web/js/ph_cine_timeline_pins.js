/**
 * ph_cine_timeline_pins.js -- v1051: the Cine Timeline's image pins grow
 * with what is wired, the MiniMax Reference way (ph_minimax_ref.js).
 *
 * The backend declares image_1..image_9 (H3 takes up to 9 picture
 * references). A fresh node shows image_1 only; every wired pin stays, plus
 * one spare after the last wired one. Pins are added/removed, never while
 * wired, and kept in display order with every link's target_slot repaired.
 * An old save with image_1..3 heals on load.
 */
import { app } from "../../scripts/app.js";

const NODE = "ULSCineTimeline";
export const N_IMAGES = 9;
const IMAGES = Array.from({ length: N_IMAGES }, (_, i) => "image_" + (i + 1));
export const DISPLAY_ORDER = ["timeline", "refs"].concat(IMAGES, ["storyboard"]);
const MANAGED = new Set(IMAGES);

function pin(node, name) { return (node.inputs || []).find((i) => i && i.name === name) || null; }
function wired(node, name) { const p = pin(node, name); return !!(p && p.link != null); }

/* the rule, one place: which image pins should exist */
export function wantedImages(node) {
    let last = 0;
    IMAGES.forEach((n, i) => { if (wired(node, n)) last = i + 1; });
    const want = new Set(IMAGES.slice(0, Math.min(N_IMAGES, last + 1)));
    for (const p of node.inputs || []) if (p && p.link != null && MANAGED.has(p.name)) want.add(p.name);
    return want;
}

function reorder(node) {
    const ins = node.inputs;
    if (!Array.isArray(ins) || !ins.length) return false;
    const rank = new Map(DISPLAY_ORDER.map((n, i) => [n, i]));
    const known = [], rest = [];
    for (const p of ins) (p && rank.has(p.name) ? known : rest).push(p);
    known.sort((a, b) => rank.get(a.name) - rank.get(b.name));
    const next = known.concat(rest);
    if (next.every((p, i) => p === ins[i])) return false;
    // v1340: the links learn their NEW slot BEFORE the array is swapped -- the live frontend re-reads each
    // pin's link from target_slot when node.inputs is assigned (measured 17.11. on the Image Reference, v1339).
    const links = node.graph && node.graph.links;
    if (links) {
        // ALL ids first: in the live frontend a pin's link is read through the links' target_slot, so reading
        // it after a slot was rewritten in this loop returns another pin's wire (measured 17.11., v1340)
        const ids = next.map((p) => (p ? p.link : null));
        for (let i = 0; i < next.length; i++) {
            const lid = ids[i];
            if (lid == null) continue;
            const l = (typeof links.get === "function") ? links.get(lid) : links[lid];
            if (l) l.target_slot = i;
        }
    }
    node.inputs = next;
    return true;
}

export function tidy(node) {
    if (!node || node._plsTpBusy) return;
    node._plsTpBusy = true;
    try {
        if (!node.inputs) node.inputs = [];
        const want = wantedImages(node);
        for (let i = node.inputs.length - 1; i >= 0; i--) {
            const p = node.inputs[i];
            if (p && MANAGED.has(p.name) && p.link == null && !want.has(p.name)) node.removeInput(i);
        }
        for (const name of IMAGES) if (want.has(name) && !pin(node, name)) node.addInput(name, "IMAGE");
        reorder(node);
        if (typeof node.setDirtyCanvas === "function") node.setDirtyCanvas(true, true);
    } finally {
        node._plsTpBusy = false;
    }
}

const LG_INPUT = (typeof LiteGraph !== "undefined" && LiteGraph.INPUT) || 1;

app.registerExtension({
    name: "polyhedron.cine_timeline_pins",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            setTimeout(() => tidy(this), 0);
            return r;
        };
        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type) {
            onConnectionsChange?.apply(this, arguments);
            if (type === LG_INPUT) tidy(this);
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            requestAnimationFrame(() => requestAnimationFrame(() => tidy(this)));
        };
    },
});
