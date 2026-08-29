/*
 * ph_seed.js -- v538
 *
 * Frontend for ⬡ Polyhedron Seed (ULSSeed).
 *
 *   1. 🎲 Roll -- crypto 53-bit random seed (exact in JS numbers), written
 *      straight into the widget and control_after_generate pinned to
 *      'fixed'. Roll-and-KEEP: unlike 'randomize' you see what you locked
 *      before the run.
 *   2. ↺ Reuse last -- restores the seed that actually ran (backend ui
 *      channel "pls_seed") and pins to 'fixed'. Survives reload/restart via
 *      node.properties.pls_last_used (serialized with the workflow).
 *   3. Status line "last used: N" at the node bottom.
 *
 * No timers, no DOM widgets, no animation -- plain widgets + one canvas line.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { attachHydration } from "./ph_widget_hydrate.js";
import { impressionCanvas } from "./ph_noise_field.js";
import { refit } from "./ph_widget_vis.js";

console.info("[PLS] ph_seed.js v538 loaded");

const NODE = "ULSSeed";
const STATUS_H = 16;

function widgetByName(node, name) {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
}

function pinFixed(node) {
    const c = widgetByName(node, "control_after_generate");
    if (c) c.value = "fixed";
}

/* 21 high bits * 2^32 + 32 low bits = uniform 53-bit, every value exact. */
function roll53() {
    const a = new Uint32Array(2);
    crypto.getRandomValues(a);
    return (a[0] & 0x1fffff) * 0x100000000 + a[1];
}

function setUsed(node, used) {
    node._plsUsed = used;
    node.properties = node.properties || {};
    node.properties.pls_last_used = used;
    node.setDirtyCanvas(true, true);
}


// ── v687: the REAL noise field, drawn in the node ───────────────────────────
// v686 drew a javascript IMITATION of each noise character. It could never
// match the run: the browser made value noise, python makes torch.randn plus a
// frequency filter. So the preview now asks the BACKEND for the actual field
// (/uls/noise/preview runs uls_noise.make_noise, the same function the sampler
// uses) and blits the PNG. What you see is what gets denoised.
//
// The geometry is LATENT, not pixels -- that is the grid noise really lives on
// (1440px Flux2 at /16 is 90x90). preview_width/height say which grid to draw
// and change nothing about the run.
//
// The drag/resize machinery is taken from the Empty Latent's preview, including
// the two traps it paid for:
//   v521 -- computeSize must reserve exactly the box height draw() LAST
//           produced, never recompute the fit from computeSize's own `width`
//           (ComfyUI passes a different width there and the box overflowed).
//   v539 -- onResize never fires on configure(), so the drag budget has to be
//           persisted in node.properties or every reload snaps back to default.
const PREV_MIN_H = 120;      // floor for the height budget
const PREV_MAX_H = 420;      // ceiling for the height budget
const PREV_DEF_H = 172;      // budget on drop
const BOX_MIN = 64;          // smallest the box is ever drawn
const BOX_MAX_W = 320;
const BOX_MAX_H = 320;
const PREV_GAP = 12;         // gap between box and readout column
const PREV_MARGIN = 12;      // left inset
const TOP_PAD = 8;
const PREV_BOTTOM = 8;
const TEXT_COL_W = 150;      // reserved for the readout right of the box
const MIN_NODE_W = 320;      // v519 law: floor the width or the rows telescope

function _fitBox(width, desiredH, aspect) {
    const a = (aspect && aspect > 0) ? aspect : 1;
    const availW = Math.max(BOX_MIN, width - 2 * PREV_MARGIN - PREV_GAP - TEXT_COL_W);
    const budget = Math.max(PREV_MIN_H, Math.min(PREV_MAX_H, desiredH || PREV_DEF_H));
    const availH = Math.max(BOX_MIN, budget - TOP_PAD - PREV_BOTTOM);
    let boxH = availH;
    let boxW = boxH * a;
    if (boxW > availW)    { boxW = availW;    boxH = boxW / a; }
    if (boxW > BOX_MAX_W) { boxW = BOX_MAX_W; boxH = boxW / a; }
    if (boxH > BOX_MAX_H) { boxH = BOX_MAX_H; boxW = boxH * a; }
    return { w: Math.max(BOX_MIN, boxW), h: Math.max(BOX_MIN, boxH) };
}

/** Is this widget currently converted to an input AND wired? Then its widget
 *  value is stale by definition -- the real one travels over the cable and is
 *  only ever seen by the backend. */
function _wiredInput(node, name) {
    if (!node || !Array.isArray(node.inputs)) return false;
    const slot = node.inputs.find((i) => i && i.name === name);
    return !!(slot && slot.link != null);
}

function _noiseOf(node) {
    const g = (n, d) => {
        const w = widgetByName(node, n);
        return (w && w.value != null) ? w.value : d;
    };
    // v689: with the size wired, use what the LAST RUN reported (backend ui
    // channel). Before the first run there is nothing to know -- say so in the
    // readout instead of drawing a confident 64x64 that is simply wrong.
    const wiredW = _wiredInput(node, "preview_width");
    const wiredH = _wiredInput(node, "preview_height");
    const rep = node._plsPrevDims;
    if ((wiredW || wiredH) && !rep) {
        return {
            type: String(g("noise_type", "gaussian")),
            strength: Number(g("noise_strength", 1.0)),
            character: Number(g("noise_character", 1.0)),
            seed: Math.max(0, Math.floor(Number(g("seed", 0)) || 0)),
            w: 0, h: 0, pending: true,
        };
    }
    return {
        type: String(g("noise_type", "gaussian")),
        strength: Number(g("noise_strength", 1.0)),
        character: Number(g("noise_character", 1.0)),
        // v689: NO >>> 0 here. That coerces to uint32, and a 53-bit seed like
        // 1125899906842624 is an exact multiple of 2^32 -> it collapsed to 0,
        // so the preview drew the field for seed 0 while the RUN used the real
        // one. Small scrubbed values survived the truncation, which is why it
        // stayed hidden.
        seed: Math.max(0, Math.floor(Number(g("seed", 0)) || 0)),
        w: Math.max(8, Math.min(512, Number((wiredW && rep) ? rep.w
                                                              : g("preview_width", 64)) | 0)),
        h: Math.max(8, Math.min(512, Number((wiredH && rep) ? rep.h
                                                            : g("preview_height", 64)) | 0)),
        pending: false,
    };
}

// Fetch the field, at most one request in flight, and only when something the
// image depends on actually changed. A scrubbed seed fires a request per pixel
// otherwise, and every one of them runs a real generator on the server.
function _requestField(node, widget, st) {
    const key = st.type + ":" + st.seed + ":" + st.strength.toFixed(3) +
                ":" + st.character.toFixed(2) + ":" + st.w + "x" + st.h;
    if (widget._key === key) return;
    widget._key = key;
    if (widget._timer) clearTimeout(widget._timer);
    widget._timer = setTimeout(() => {
        widget._timer = 0;
        const url = "/uls/noise/preview?type=" + encodeURIComponent(st.type) +
                    "&seed=" + st.seed + "&strength=" + st.strength +
                    "&character=" + st.character +
                    "&w=" + st.w + "&h=" + st.h;
        const img = new Image();
        img.onload = () => {
            if (widget._key !== key) return;      // a newer request won
            widget._img = img;
            widget._imgKey = key;
            widget._failed = false;
            node.setDirtyCanvas(true, false);
        };
        img.onerror = () => {
            if (widget._key !== key) return;
            widget._failed = true;
            widget._impHot = false;   // v690: do not leave an impression on
                                      // screen labelled as one forever -- if
                                      // the route is down, say so instead.
            node.setDirtyCanvas(true, false);
        };
        img.src = api.apiURL ? api.apiURL(url) : url;
    }, 90);
}

function _addNoisePreview(node) {
    // v689 FIELD BUG -- "the noise changes even outside the box, as soon as I
    // move the mouse". Every state flag below is reached through the CLOSURE
    // variable `widget`, never through `this`.
    //
    // WHY: ComfyUI does not guarantee that a widget's mouse()/draw() is invoked
    // with the widget as `this`. When pointerdown set `this._dragging` on one
    // object and pointerup cleared it on another, the flag stayed true forever
    // and every later mouse move over the node kept scrubbing the seed. The
    // Empty Latent's preview never had this bug because it always used the
    // closure -- v687 rewrote it with `this` and re-introduced it.
    const widget = {
        type: "uls_seed_noise_preview",
        name: "noise_preview",
        value: null,
        serialize: false,           // never in widgets_values -> clean round-trip
        _dragging: false,
        _lastX: 0,
        _h: PREV_DEF_H,             // height budget; changed only via onResize
        _boxH: null,                // last-drawn box height, cached for computeSize
        _img: null,
        _key: null,
        // v521: reserve exactly what draw() produced -- never recompute here.
        computeSize(width) {
            const bh = (widget._boxH != null) ? widget._boxH
                                            : (PREV_DEF_H - TOP_PAD - PREV_BOTTOM);
            return [width, bh + TOP_PAD + PREV_BOTTOM];
        },
        draw(ctx, n, widgetWidth, posY) {
            n._plsPrevTop = posY;
            const st = _noiseOf(n);
            // v690: no fetch while the mouse is down. The debounce in
            // _requestField RESTARTS on every state change, so a continuous
            // scrub never let it fire and the box showed "building field..."
            // for the whole gesture. During the drag we draw the local
            // impression instead (and keep drawing it until the real field
            // that was requested on release has actually arrived, so there is
            // no blank flash in between).
            const dragging = widget._dragging === true;
            if (!st.pending && !dragging) _requestField(n, widget, st);
            const haveExact = !!(widget._img && widget._imgKey === widget._key);
            const showImpression = !st.pending && !haveExact &&
                                   (dragging || widget._impHot === true);
            const box = _fitBox(widgetWidth, widget._h, st.w / st.h);
            const boxW = box.w, boxH = box.h;
            widget._boxH = boxH;
            const x = PREV_MARGIN;
            const y = posY + TOP_PAD;
            ctx.save();
            try {
                ctx.fillStyle = "#111318";
                ctx.strokeStyle = "#2b2f3a";
                ctx.lineWidth = 1;
                ctx.beginPath(); ctx.rect(x, y, boxW, boxH); ctx.fill(); ctx.stroke();
                if (haveExact) {
                    widget._impHot = false;              // the real field won; stop standing in
                    ctx.imageSmoothingEnabled = false;   // latent grids are tiny; keep them crisp
                    ctx.drawImage(widget._img, x + 1, y + 1,
                                  Math.max(1, boxW - 2), Math.max(1, boxH - 2));
                } else if (showImpression) {
                    // Same grid as the fetched field (st.w x st.h), so the
                    // handover on release does not jump in grain.
                    const cv = impressionCanvas(widget, st.type, st.seed, st.w, st.h, st.character);
                    ctx.imageSmoothingEnabled = false;
                    ctx.drawImage(cv, x + 1, y + 1,
                                  Math.max(1, boxW - 2), Math.max(1, boxH - 2));
                } else {
                    ctx.fillStyle = "#6b7280";
                    ctx.font = "10px monospace";
                    ctx.textAlign = "center";
                    ctx.fillText(st.pending ? "size is wired -- run once"
                                            : (widget._failed ? "preview unavailable"
                                                              : "building field..."),
                                 x + boxW / 2, y + boxH / 2);
                    ctx.textAlign = "left";
                }
                const tx = x + boxW + PREV_GAP;
                let ty = y + 12;
                ctx.font = "11px monospace";
                ctx.fillStyle = "#c9d1e0";
                ctx.fillText("noise: " + st.type, tx, ty); ty += 17;
                ctx.fillText("strength: " + st.strength.toFixed(2), tx, ty); ty += 17;
                ctx.fillStyle = "#8b93a7";
                ctx.fillText(st.pending ? "latent: from cable"
                                        : ("latent " + st.w + "x" + st.h), tx, ty); ty += 15;
                // The box carries two different pictures. Say which one, always
                // -- exactly two lines either way, so the block never changes
                // height mid-gesture. An unlabelled impression would be a lie
                // in the same class as inventing a wired size.
                if (showImpression) {
                    ctx.fillStyle = "#c08a4a";
                    ctx.fillText("impression while", tx, ty); ty += 13;
                    ctx.fillText("scrubbing", tx, ty); ty += 18;
                } else {
                    ctx.fillStyle = "#7c8497";
                    ctx.fillText("the real field,", tx, ty); ty += 13;
                    ctx.fillText("not an impression", tx, ty); ty += 18;
                }
                if (st.type === "zeros") {
                    ctx.fillStyle = "#6b7280";
                    ctx.fillText("(no noise at all)", tx, ty);
                } else {
                    ctx.fillStyle = "#c9d1e0";
                    ctx.fillText("seed: " + st.seed, tx, ty); ty += 16;
                    ctx.fillStyle = "#7c8497";
                    ctx.fillText("drag = scrub seed", tx, ty); ty += 13;
                    ctx.fillText("click = re-roll", tx, ty);
                }
            } catch (e) { /* never break the canvas */ }
            ctx.restore();
        },
        mouse(event, pos, n) {
            const sdW = widgetByName(n, "seed");
            if (!sdW) return false;
            const et = event.type;
            if (et === "pointerdown" || et === "mousedown") {
                widget._dragging = true;
                widget._impHot = true;   // stand in until the real field lands
                widget._lastX = pos[0];
                widget._moved = 0;
                return true;
            }
            if (et === "pointermove" || et === "mousemove") {
                if (!widget._dragging) return false;
                const dx = pos[0] - widget._lastX;
                widget._lastX = pos[0];
                widget._moved += Math.abs(dx);
                if (Math.abs(dx) >= 1) {
                    let v = Number(sdW.value || 0) + Math.round(dx);
                    if (v < 0) v = 0;
                    sdW.value = v;
                    pinFixed(n);
                    n.setDirtyCanvas(true, true);
                }
                return true;
            }
            if (et === "pointerup" || et === "mouseup") {
                const wasDrag = widget._dragging;
                widget._dragging = false;
                if (wasDrag && widget._moved < 3) {        // a click, not a drag
                    sdW.value = roll53();
                    pinFixed(n);
                    n.setDirtyCanvas(true, true);
                }
                return true;
            }
            return false;
        },
    };
    node.addCustomWidget(widget);

    // Grow/shrink on a lower-edge drag, and floor the width so the rows and the
    // preview composite can never be squeezed shut (v519 mechanic: LiteGraph
    // passes `size` by reference during a drag, so mutating it constrains the
    // node in place).
    const _prevResize = node.onResize;
    node.onResize = function (size) {
        if (size && size.length >= 1 && size[0] < MIN_NODE_W) size[0] = MIN_NODE_W;
        const r = _prevResize ? _prevResize.apply(this, arguments) : undefined;
        const top = this._plsPrevTop;
        if (top != null && size && size.length >= 2) {
            // Height floor, same mechanic as the width one: the rows above plus
            // the preview's own minimum. Without it the node can be dragged shut
            // over its own widgets.
            const floor = top + PREV_MIN_H;
            if (size[1] < floor) size[1] = floor;
            const desired = size[1] - top;
            widget._h = Math.max(PREV_MIN_H, Math.min(PREV_MAX_H, desired));
            this.properties = this.properties || {};
            this.properties.pls_preview_h = widget._h;   // v539: onResize never fires on configure()
        }
        return r;
    };
    return widget;
}

app.registerExtension({
    name: "polyhedron.seed",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE) return;

        // v686: a graph saved before v685 has no value for the appended
        // noise_type / noise_strength rows -- they load as undefined and the
        // prompt dies in ComfyUI's type validation before our python runs.
        // Wire the shared hydration FIRST, so its onNodeCreated snapshot is
        // taken with the definition's defaults still in place.
        attachHydration(nodeType);

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            const baseCS = this.computeSize.bind(this);
            this.computeSize = (out) => {
                const s = baseCS(out);
                s[1] += STATUS_H;
                return s;
            };
            this.addWidget("button", "\ud83c\udfb2 Roll", null, () => {
                const w = widgetByName(this, "seed");
                if (!w) return;
                w.value = roll53();
                pinFixed(this);
                this.setDirtyCanvas(true, true);
            });
            this.addWidget("button", "\u21ba Reuse last", null, () => {
                if (this._plsUsed == null) return;
                const w = widgetByName(this, "seed");
                if (!w) return;
                w.value = this._plsUsed;
                pinFixed(this);
                this.setDirtyCanvas(true, true);
            });
            _addNoisePreview(this);
            // v897: height only. `setSize(computeSize())` is a HARD RESET to
            // LiteGraph's computed MINIMUM -- it throws away a width the user
            // dragged out. ph_switch.js:39 records what that cost when tidy() did
            // it; v888 pinned the law and gave it one home. These were the last
            // four callers still doing the reset.
            refit(this);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            const v = this.properties ? this.properties.pls_last_used : null;
            if (typeof v === "number") this._plsUsed = v;
            // v539 law, inherited with the widget: onResize never fires on
            // configure(), so without this every reload snaps the preview back
            // to its default height and leaves dead space in a sized node.
            const ph = this.properties ? this.properties.pls_preview_h : null;
            if (typeof ph === "number") {
                const pw = (this.widgets || []).find((w) => w.type === "uls_seed_noise_preview");
                if (pw) pw._h = Math.max(PREV_MIN_H, Math.min(PREV_MAX_H, ph));
            }
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const p = message ? message.pls_seed : null;
            const rec = Array.isArray(p) ? p[0] : p;
            if (rec && rec.used != null) setUsed(this, Number(rec.used));
            if (rec && rec.pw != null && rec.ph != null) {
                this._plsPrevDims = { w: Number(rec.pw), h: Number(rec.ph) };
                this.setDirtyCanvas(true, false);
            }
        };

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            onDrawForeground?.apply(this, arguments);
            if ((this.flags && this.flags.collapsed) || this._plsUsed == null) return;
            ctx.save();
            ctx.font = "11px Arial";
            ctx.fillStyle = "#9a9a9a";
            ctx.textAlign = "left";
            ctx.fillText("last used: " + this._plsUsed, 8, this.size[1] - 5);
            ctx.restore();
        };
    },
});
