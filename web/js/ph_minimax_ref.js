/*
 * ph_minimax_ref.js -- v1016
 *
 * Frontend for Polyhedron MiniMax Reference (ULSMiniMaxReference).
 *
 * v1016 gave the node every input MiniMax H3 takes: 9 images, 3 videos, the
 * 3 soundtracks that belong to them, 3 standalone audios and the audio VAE.
 * Shown all at once that is 19 pins and 9 sliders on one box. This file makes
 * the node grow with what is wired, the way Core's Autogrow does:
 *
 *   * images, videos, audios: every WIRED slot stays, plus exactly ONE spare
 *     after the last wired one (up to the group's maximum). A hole in the
 *     middle stays visible -- its number is in use.
 *   * video_audio_n appears only next to a wired video_n (it is that video's
 *     soundtrack and means nothing alone).
 *   * megapixels_n is visible only while image_n is wired.
 *   * clip, vae, audio_vae, latent are always there.
 *
 * PINS vs WIDGETS, the two laws this file keeps:
 *   - pins are added/removed for real (addInput/removeInput, ph_switch.js);
 *     a WIRED pin is never removed, whatever the rule says.
 *   - widgets are only HIDDEN (ph_widget_vis.js) -- they serialise by index
 *     (#577), removing one would shift every saved value behind it.
 *
 * ORDER: LiteGraph appends a new pin at the END, and configure() restores the
 * SAVED order. Both would scatter the groups, so after every change the pins
 * are put back into DISPLAY_ORDER and every incoming link's target_slot is
 * repaired (the v547 Power Upscale pattern). Pins this file does not manage
 * (a converted widget such as `prompt`) keep their relative order after them.
 *
 * Old workflows (<= v1015) were saved with image_1..3 only; the first tidy
 * adds the spares and audio_vae, so they heal on load without a re-create.
 *
 * Names and counts MUST match nodes/ph_minimax_ref.py (N_IMAGES / N_VIDEOS /
 * N_AUDIOS and the optional keys); tests/test_v1016_minimax_ref_js.py guards
 * that parity and drives tidy() against a fake node.
 */

import { app } from "../../scripts/app.js";
import { setHidden, refit, fieldFloor } from "./ph_widget_vis.js";

console.info("[PLS] ph_minimax_ref.js v1016 loaded");

const NODE = "ULSMiniMaxReference";
export const N_IMAGES = 9;
export const N_VIDEOS = 3;
export const N_AUDIOS = 3;

const FIXED = ["clip", "vae", "audio_vae", "latent", "refs", "first_frame"];   // v1019: + refs (Reference Board), v1049: + first_frame (scene carry)

function range(n) { return Array.from({ length: n }, (_, i) => i + 1); }

export const DISPLAY_ORDER = [].concat(
    FIXED,
    range(N_IMAGES).map((n) => "image_" + n),
    ...range(N_VIDEOS).map((n) => ["video_" + n, "video_audio_" + n]),
    range(N_AUDIOS).map((n) => "audio_" + n),
);

export const PIN_TYPE = (() => {
    const t = { clip: "CLIP", vae: "VAE", audio_vae: "VAE", latent: "LATENT", refs: "PLS_REFS", first_frame: "IMAGE" };
    for (const n of range(N_IMAGES)) t["image_" + n] = "IMAGE";
    for (const n of range(N_VIDEOS)) {
        t["video_" + n] = "VIDEO";
        t["video_audio_" + n] = "AUDIO";
    }
    for (const n of range(N_AUDIOS)) t["audio_" + n] = "AUDIO";
    return t;
})();

function pin(node, name) {
    return (node.inputs || []).find((i) => i && i.name === name) || null;
}

function wired(node, name) {
    const p = pin(node, name);
    return !!(p && p.link != null);
}

function lastWired(node, prefix, max) {
    let last = 0;
    for (const n of range(max)) if (wired(node, prefix + n)) last = n;
    return last;
}

/* The rule, one place only. Returns the set of pin names that should exist. */
export function wantedPins(node) {
    const want = new Set(FIXED);
    const li = lastWired(node, "image_", N_IMAGES);
    for (const n of range(Math.min(N_IMAGES, li + 1))) want.add("image_" + n);
    const lv = lastWired(node, "video_", N_VIDEOS);
    for (const n of range(Math.min(N_VIDEOS, lv + 1))) want.add("video_" + n);
    for (const n of range(N_VIDEOS)) {
        if (wired(node, "video_" + n) || wired(node, "video_audio_" + n)) {
            want.add("video_audio_" + n);
        }
    }
    const la = lastWired(node, "audio_", N_AUDIOS);
    for (const n of range(Math.min(N_AUDIOS, la + 1))) want.add("audio_" + n);
    // never drop a wired pin, whatever the rule above says
    for (const p of node.inputs || []) {
        if (p && p.link != null && PIN_TYPE[p.name]) want.add(p.name);
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
    node.inputs = next;
    const links = node.graph && node.graph.links;
    if (links) {
        for (let i = 0; i < node.inputs.length; i++) {
            const lid = node.inputs[i] ? node.inputs[i].link : null;
            if (lid == null) continue;
            const l = (typeof links.get === "function") ? links.get(lid) : links[lid];
            if (l) l.target_slot = i;
        }
    }
    return true;
}

export function tidy(node) {
    if (!node || node._plsMrBusy) return;
    node._plsMrBusy = true;
    try {
        if (!node.inputs) node.inputs = [];
        const want = wantedPins(node);
        // remove managed, unwanted, UNWIRED pins (from the back, indices stay valid)
        for (let i = node.inputs.length - 1; i >= 0; i--) {
            const p = node.inputs[i];
            if (!p || !PIN_TYPE[p.name] || FIXED.includes(p.name)) continue;
            if (p.link == null && !want.has(p.name)) node.removeInput(i);
        }
        // add the wanted ones that are missing (old saves, fresh spares)
        for (const name of DISPLAY_ORDER) {
            if (want.has(name) && !pin(node, name)) node.addInput(name, PIN_TYPE[name]);
        }
        reorder(node);
        // megapixels_n follows image_n
        for (const n of range(N_IMAGES)) {
            const w = (node.widgets || []).find((x) => x && x.name === "megapixels_" + n);
            if (w) setHidden(w, !wired(node, "image_" + n));
        }
        refit(node);
        if (typeof node.setDirtyCanvas === "function") node.setDirtyCanvas(true, true);
    } finally {
        node._plsMrBusy = false;
    }
}

const LG_INPUT = (typeof LiteGraph !== "undefined" && LiteGraph.INPUT) || 1;

app.registerExtension({
    name: "polyhedron.minimax_ref",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            fieldFloor(this, ["prompt", "tags"]);          // v1022: 64 px on both renderers
            setTimeout(() => tidy(this), 0);
            return r;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type) {
            onConnectionsChange?.apply(this, arguments);
            if (type === LG_INPUT) tidy(this);
        };

        // links restore asynchronously after a load -- the proven double-rAF
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            requestAnimationFrame(() => requestAnimationFrame(() => tidy(this)));
        };
    },
});
