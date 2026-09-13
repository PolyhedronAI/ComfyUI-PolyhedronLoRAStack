/**
 * Polyhedron Sampler — live VIDEO preview (frontend)
 * ─────────────────────────────────────────────────────────────────────────
 * Animates the in-progress generation inside the sampler node. The backend
 * (nodes/uls_sampler.py) decodes EVERY frame of the predicted latent — ComfyUI
 * core decodes only the first — and streams them, rate-limited, as a
 * "polyhedron.live_preview" event carrying a list of base64 JPEG frames + an fps.
 * Here we load those frames and loop them through the node's native image area
 * (node.imgs), so the node shows a moving preview while it samples — the same
 * idea as the WAN video sampler, but self-contained (no VideoHelperSuite, no
 * external previewer, no extra model).
 *
 * Lifecycle: frames stream in during sampling and loop at fps; a new run
 * (execution_start) clears the previous preview; when the prompt finishes
 * (execution_success / _error) the loop stops on the last shown frame. node.imgs
 * is runtime-only (never serialised), so nothing persists across reloads.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
// v949: under Nodes 2.0 the node image area (node.imgs) is never painted -- the
// Vue node shows previews from the frontend's output store, fed by the core
// "b_preview_with_metadata" event. So in Vue mode each frame is ALSO handed in
// through that event, as a Blob -- the official door, reactivity included.
import { vueMode } from "./uls_vue_views.js";

const NODE_TYPE = "ULSSampler";
const EVENT = "polyhedron.live_preview";

function _stopAnim(node) {
    if (node._ulsAnimTimer) {
        clearInterval(node._ulsAnimTimer);
        node._ulsAnimTimer = null;
    }
}

function _startAnim(node) {
    if (node._ulsAnimTimer) return;
    const fps = Math.min(30, Math.max(1, node._ulsFps || 12));
    let i = 0;
    node._ulsAnimTimer = setInterval(() => {
        const frames = node._ulsFrames;
        if (!frames || frames.length === 0) return;
        const img = frames[i % frames.length];
        i++;
        // skip frames that haven't decoded yet so we never blit a blank image
        if (!img || !img.complete || img.naturalWidth === 0) return;
        node.imgs = [img];
        if (vueMode()) {
            const blob = frames[(i - 1) % frames.length]._ulsBlob;
            if (blob) _pushVuePreview(node, blob);
        }
        if (node.setSizeForImage) {
            try { node.setSizeForImage(true); } catch (err) { /* keep going */ }
        }
        node.setDirtyCanvas(true, false);
    }, 1000 / fps);
}

// v949: the same frame through the frontend's own preview door (Nodes 2.0).
function _pushVuePreview(node, blob) {
    try {
        api.dispatchEvent(new CustomEvent("b_preview_with_metadata", {
            detail: { blob, displayNodeId: String(node.id), jobId: undefined },
        }));
    } catch (err) { /* a preview must never break the run */ }
}

function _b64ToBlob(b64) {
    try {
        const bin = atob(b64);
        const u8 = new Uint8Array(bin.length);
        for (let k = 0; k < bin.length; k++) u8[k] = bin.charCodeAt(k);
        return new Blob([u8], { type: "image/jpeg" });
    } catch (err) { return null; }
}

function _forEachSampler(fn) {
    const nodes = (app.graph && app.graph._nodes) || [];
    for (const node of nodes) {
        if (node && node.type === NODE_TYPE) fn(node);
    }
}

app.registerExtension({
    name: "polyhedron.sampler.livepreview",
    async setup() {
        // Incoming preview frames for a specific sampler node.
        api.addEventListener(EVENT, (e) => {
            const d = (e && e.detail) || {};
            if (!d.node || !Array.isArray(d.frames) || d.frames.length === 0) return;
            const node = app.graph.getNodeById(Number(d.node));
            if (!node || node.type !== NODE_TYPE) return;
            const imgs = [];
            for (const b64 of d.frames) {
                const img = new Image();
                img.src = "data:image/jpeg;base64," + b64;
                img._ulsBlob = _b64ToBlob(b64);   // v949: for the Nodes 2.0 door
                imgs.push(img);
            }
            node._ulsFrames = imgs;
            node._ulsFps = d.fps || 12;
            _startAnim(node);
        });

        // A new run starts: drop any stale preview on all sampler nodes.
        api.addEventListener("execution_start", () => {
            _forEachSampler((node) => {
                _stopAnim(node);
                node._ulsFrames = null;
            });
        });

        // Prompt finished (or errored): stop looping, freeze on the last frame.
        api.addEventListener("execution_success", () => _forEachSampler(_stopAnim));
        api.addEventListener("execution_error", () => _forEachSampler(_stopAnim));
    },
});
