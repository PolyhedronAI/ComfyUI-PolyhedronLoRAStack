/*
 * ph_image_ref.js -- v1078
 *
 * Frontend for Polyhedron Image Reference (ULSImageReference): the node grows
 * with what is wired, the ph_minimax_ref.js way -- every WIRED image pin
 * stays, plus exactly ONE spare after the last wired one (up to N_IMAGES).
 * clip, vae and latent are always there. A hole in the middle stays visible,
 * its number is in use.
 *
 * Pins are added/removed for real (addInput/removeInput); a WIRED pin is never
 * removed. After every change the pins go back into DISPLAY_ORDER and every
 * incoming link's target_slot is repaired (the v547 pattern), because LiteGraph
 * appends new pins at the END and configure() restores the SAVED order.
 *
 * Names and counts MUST match nodes/ph_image_ref.py (N_IMAGES, IMAGE_KEYS);
 * tests/test_v1078_image_ref.py guards that parity and drives tidy() against a
 * fake node.
 */

import { app } from "../../scripts/app.js";
import { refit, fieldFloor } from "./ph_widget_vis.js";

console.info("[PLS] ph_image_ref.js v1339 loaded");

const NODE = "ULSImageReference";
export const N_IMAGES = 9;
const FIXED = ["clip", "vae", "latent"];

function range(n) { return Array.from({ length: n }, (_, i) => i + 1); }

export const DISPLAY_ORDER = [].concat(FIXED, range(N_IMAGES).map((n) => "image_" + n));

export const PIN_TYPE = (() => {
    const t = { clip: "CLIP", vae: "VAE", latent: "LATENT" };
    for (const n of range(N_IMAGES)) t["image_" + n] = "IMAGE";
    return t;
})();

function pin(node, name) {
    return (node.inputs || []).find((i) => i && i.name === name) || null;
}

function wired(node, name) {
    const p = pin(node, name);
    return !!(p && p.link != null);
}

/* The rule, one place only. Returns the set of pin names that should exist. */
export function wantedPins(node) {
    const want = new Set(FIXED);
    let last = 0;
    for (const n of range(N_IMAGES)) if (wired(node, "image_" + n)) last = n;
    for (const n of range(Math.min(N_IMAGES, last + 1))) want.add("image_" + n);
    for (const p of node.inputs || []) {
        if (p && p.link != null && PIN_TYPE[p.name]) want.add(p.name);   // never drop a wired pin
    }
    return want;
}

function reorder(node) {
    const ins = node.inputs;
    if (!Array.isArray(ins) || !ins.length) return false;
    const rank = new Map(DISPLAY_ORDER.map((n, i) => [n, i]));
    const known = [];
    const rest = [];
    for (const p of ins) (p && rank.has(p.name) ? known : rest).push(p);
    known.sort((a, b) => rank.get(a.name) - rank.get(b.name));
    const next = known.concat(rest);
    let changed = false;
    for (let i = 0; i < ins.length; i++) if (ins[i] !== next[i]) { changed = true; break; }
    if (!changed) return false;
    // v1339: the links learn their NEW slot BEFORE the array is swapped. Measured in the live frontend
    // (17.11.): assigning node.inputs re-reads every pin's link from the links' target_slot -- swapped
    // first, a wired widget socket (prompt) handed its link to the pin that now sat on its old slot
    // (inpaint_mask), and the prompt lost its wire.
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

/* v1339: the check after the swap, by NAME. A pin whose saved link is no longer on it gets it back and the
   link points at that slot. Nothing is ever set to null here: in the live frontend writing a pin's link
   goes through a setter, and null DISCONNECTS -- a wrong null would cut the wire this guard exists for. */
export function healLinks(node, byName) {
    const links = node.graph && node.graph.links;
    const get = (lid) => (links ? ((typeof links.get === "function") ? links.get(lid) : links[lid]) : null);
    let healed = 0;
    for (let i = 0; i < node.inputs.length; i++) {
        const p = node.inputs[i];
        if (!p || !byName.has(p.name)) continue;
        const lid = byName.get(p.name);
        if (lid == null) continue;
        const l = get(lid);
        if (!l) continue;                       // the wire is gone for real (removed by the user)
        if (p.link !== lid) { p.link = lid; healed++; }
        if (l.target_slot !== i) l.target_slot = i;
    }
    return healed;
}

export function tidy(node) {
    if (!node || node._plsIrBusy) return;
    node._plsIrBusy = true;
    try {
        if (!node.inputs) node.inputs = [];
        // v1339: every input's link, BY NAME, before pins come and go. Measured in the live frontend (17.11.):
        // with a widget socket wired (prompt <- CLIP Text Encode positive_text), adding image_2 left the prompt's
        // link on the slot that inpaint_mask then held -- the prompt lost its wire and inpaint_mask carried a
        // STRING. The pin list may be spliced by the frontend; the names are the truth.
        const byName = new Map(node.inputs.filter(Boolean).map((p) => [p.name, p.link]));
        const want = wantedPins(node);
        for (let i = node.inputs.length - 1; i >= 0; i--) {
            const p = node.inputs[i];
            if (!p || !PIN_TYPE[p.name] || FIXED.includes(p.name)) continue;
            if (p.link == null && !want.has(p.name)) node.removeInput(i);
        }
        for (const name of DISPLAY_ORDER) {
            if (want.has(name) && !pin(node, name)) node.addInput(name, PIN_TYPE[name]);
        }
        reorder(node);
        healLinks(node, byName);
        refit(node);
        if (typeof node.setDirtyCanvas === "function") node.setDirtyCanvas(true, true);
    } finally {
        node._plsIrBusy = false;
    }
}

const LG_INPUT = (typeof LiteGraph !== "undefined" && LiteGraph.INPUT) || 1;

app.registerExtension({
    name: "polyhedron.image_ref",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            fieldFloor(this, ["prompt", "negative_prompt"]);
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
