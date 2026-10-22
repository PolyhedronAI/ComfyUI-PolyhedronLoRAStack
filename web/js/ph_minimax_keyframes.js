// ph_minimax_keyframes.js -- v965: the prompt that RAN, shown inside the node.
//
// When `prompt` is wired (CLIP Text Encode positive_text, a timeline, ...)
// the frontend hides the prompt box and the node shows an empty area (field,
// 19.09.2026: "abgeschnitten und komisch"). This file adds ONE read-only
// field (the ph_clip_encode.js EXT-field pattern): hidden while the prompt
// box is visible, shown while the `prompt` input is linked, filled after a
// run from the backend's `pls_kf` payload with the resolved prompt and the
// keyframe report. Fixed height with scroll -- it never pushes the other
// widgets around (the v600 lesson). Nothing is painted on the canvas; the
// field is a DOM widget, which Core's Vue renderer wraps itself.
import { app } from "../../scripts/app.js";
import { refit, fieldFloor, growToFloor } from "./ph_widget_vis.js";   // height only -- width is never touched (v888/v897)

const NODE = "ULSMiniMaxKeyframes";
const FIELD = "pls_kf_view";
const FIELD_H = 132;

function _w(node, name) {
    return (node.widgets || []).find((w) => w && w.name === name);
}

function _promptLinked(node) {
    const inp = (node.inputs || []).find((i) => i && i.name === "prompt");
    return !!(inp && inp.link != null);
}

function _makeField(node) {
    if (_w(node, FIELD)) return _w(node, FIELD);
    if (typeof node.addDOMWidget !== "function") return null;
    let w = null;
    try {
        const el = document.createElement("textarea");
        el.readOnly = true;
        el.classList.add("comfy-multiline-input");
        el.style.width = "100%";
        el.style.boxSizing = "border-box";
        el.style.height = FIELD_H + "px";
        el.style.overflowY = "auto";
        el.style.fontSize = "11px";
        el.placeholder = "prompt arrives over the wire -- it shows here after the run";
        w = node.addDOMWidget(FIELD, "customtext", el, { serialize: false });
        w.serialize = false;
        w._plsKfView = true;
        w.hidden = true;
        el.style.display = "none";
        w.computeSize = (width) => (w.hidden ? [0, -4] : [width, FIELD_H]);
    } catch (e) { /* a display field must never break node creation */ }
    return w;
}

function _sync(node) {
    const w = _w(node, FIELD);
    if (!w) return;
    const on = _promptLinked(node);
    const was = !!w.hidden;
    w.hidden = !on;
    if (w.element) w.element.style.display = on ? "" : "none";
    if (was !== !on) refit(node);   // height follows the field; the dragged width stays
}

function _fill(node, d) {
    const w = _w(node, FIELD);
    if (!w || !w.element) return;
    const prompt = String(d.prompt || "");
    const info = String(d.info || "");
    w.element.value = (info ? info + "\n\n" : "") + (prompt || "(empty prompt)");
}

app.registerExtension({
    name: "Polyhedron.MiniMaxKeyframes",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE) return;

        const _created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = _created ? _created.apply(this, arguments) : undefined;
            _makeField(this);
            fieldFloor(this, ["prompt", "tags"]);          // v1022: 64 px on both renderers
            setTimeout(() => { _sync(this); growToFloor(this); }, 0);
            return r;
        };

        const _conf = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = _conf ? _conf.apply(this, arguments) : undefined;
            _makeField(this);
            fieldFloor(this, ["prompt", "tags"]);
            setTimeout(() => { _sync(this); growToFloor(this); }, 0);
            return r;
        };

        const _conn = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const r = _conn ? _conn.apply(this, arguments) : undefined;
            try { _sync(this); } catch (e) { /* never throw from a link event */ }
            return r;
        };

        const _exec = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = _exec ? _exec.apply(this, arguments) : undefined;
            try {
                const d = message && message.pls_kf && message.pls_kf[0];
                if (d) { _makeField(this); _sync(this); _fill(this, d); }
            } catch (e) { /* a preview must never break a run */ }
            return r;
        };
    },
});
