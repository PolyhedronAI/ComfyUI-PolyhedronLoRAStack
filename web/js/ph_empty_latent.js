/*
 * ph_empty_latent.js — UI for ⬡ Polyhedron Empty Latent (ULSEmptyLatent).
 *
 * Three frontend concerns, all layered on the Python-defined widgets without
 * disturbing serialisation:
 *
 *  1. TYPE/NOISE GREYING (the v515 Power-Upscale pattern): `length` is disabled
 *     for the Image type (video types use it); `noise_seed` and `noise_strength`
 *     are disabled when `noise_type` is "zeros" (they do nothing there). Derived
 *     on create + configure + widget callbacks; never serialised.
 *
 *  2. SIZE PRESETS (persistent, user-editable): a combo lists USER presets first,
 *     then built-ins (no dead placeholder). Its value is sticky so the ◀ ▶ arrows
 *     cycle through and APPLY each preset; a plain edit of width/height shows
 *     "Custom". Save/Delete POST to the backend (user presets stored in the
 *     ComfyUI user dir, surviving updates).
 *
 *  3. LIVE NOISE PREVIEW + MOUSE-SCRUB SEED: a canvas widget renders the CHARACTER
 *     of the selected noise type at the current seed, sized to the latent's ASPECT
 *     RATIO (the exact partition going into the sampler) and GROWING with the node
 *     up to a max (expanding from its centre). Dragging it scrubs `noise_seed`, a
 *     plain click re-rolls it — written to the REAL noise_seed widget, so it
 *     serialises into the PNG and the exact tensor regenerates in Python.
 *
 * REPRODUCIBILITY RULE honoured here: every widget this file adds sets
 * `serialize = false`. widgets_values carries ONLY the Python widgets.
 *
 * KEEP-IN-SYNC (asserted by the v517 guard on both sides):
 *   LATENT_TYPES / VIDEO_TYPES / TYPE_GEOM mirror uls_latent_math.
 *   SEEDLESS_TYPES mirrors uls_noise.SEEDLESS_TYPES ("zeros").
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { setHidden, refit } from "./ph_widget_vis.js";

const NODE_TYPE = "ULSEmptyLatent";

// latent_type combo labels — mirror uls_latent_math.LATENT_TYPE_LABELS (order too).
// v898: "MiniMax H3 AV (video+audio)" was MISSING here. Python has carried it
// since the H3 line went in; this list did not. The cost was invisible and
// exact: TYPE_GEOM[type] came back undefined, the type fell out of VIDEO_TYPES,
// isVideo went false -- and since v888 an inactive row is HIDDEN rather than
// greyed, so `length` DISAPPEARED in the one mode where it is the only way to
// name a frame count. Frank saw a node that could be told seconds and nothing
// else. The mirror is now driven against Python by test_v898.
const LATENT_TYPES = ["Image", "WAN video", "Hunyuan video", "Mochi video", "LTXV video", "Cosmos video", "SD/SDXL image", "Flux2 image", "MiniMax H3 AV (video+audio)"];
// Per-type latent geometry — mirrors uls_latent_math.TYPE_SPEC (spatial_div, temporal_div, video).
const TYPE_GEOM = {
    "Image":         { sdiv: 8,  tdiv: 0, video: false },
    "WAN video":     { sdiv: 8,  tdiv: 4, video: true },
    "Hunyuan video": { sdiv: 8,  tdiv: 4, video: true },
    "Mochi video":   { sdiv: 8,  tdiv: 6, video: true },
    "LTXV video":    { sdiv: 32, tdiv: 8, video: true },
    "Cosmos video":  { sdiv: 8,  tdiv: 8, video: true },
    "SD/SDXL image": { sdiv: 8,  tdiv: 0, video: false },
    "Flux2 image":   { sdiv: 16, tdiv: 0, video: false },
    // tdiv 0 is not "no time": H3's time axis is not a divisor at all, it is
    // the 17k+5 grid below. TYPE_SPEC says the same with its own 0.
    "MiniMax H3 AV (video+audio)": { sdiv: 16, tdiv: 0, video: true },
};
const MINIMAX_TYPE = "MiniMax H3 AV (video+audio)";
const MINIMAX_FPS = 24;                     // mirrors uls_latent_math.MINIMAX_FPS

/*
 * H3's frame grid, mirrored from uls_latent_math.minimax_align_frames -- which
 * itself mirrors Core's align_frame_count in comfy_extras/nodes_minimax_h3.py:
 *
 *     frame_count = align_frame_count(max(5, length))     # while n % 17 != 5: n++
 *
 * So FIVE is the model's floor and the grid is 17k+5. A request for a single
 * frame cannot be honoured by this latent type -- it comes back as 5. That is
 * not ours to fix, but it IS ours to say out loud, which is what the pill does.
 * The guard drives this function against the Python one over a range, because
 * two places computing the same thing drift.
 */
function minimaxAlign(n) {
    // The max() states the floor; the loop ENFORCES it, because the first
    // grid point at or above any small number IS five. Core writes it the
    // same way, and it is worth keeping for the same reason: the rule should
    // be readable, not inferred from a modulus. (Measured while mutating this
    // guard: removing the max() alone changes no output -- so the loop is what
    // a mutation has to break, and M4 breaks it.)
    let v = Math.max(5, Math.floor(Number(n) || 0));
    while (v % 17 !== 5) v += 1;
    return v;
}

function minimaxFramesFromSeconds(seconds) {
    return minimaxAlign(Math.round((Number(seconds) || 0) * MINIMAX_FPS));
}
// Types that emit a 5D (video) latent — derived from TYPE_GEOM (mirrors Python).
const VIDEO_TYPES = LATENT_TYPES.filter((t) => TYPE_GEOM[t] && TYPE_GEOM[t].video);
// Old model_family values map onto the new type labels (presets / robustness).
const FAMILY_TO_TYPE = {
    flux: "Image", flux2: "Image", sd3: "Image", sdxl: "Image", sd: "Image",
    qwen: "Image", image: "Image",
    wan: "WAN video", hunyuan: "Hunyuan video", hunyuan_video: "Hunyuan video",
    mochi: "Mochi video", ltxv: "LTXV video", ltx: "LTXV video", cosmos: "Cosmos video",
};
// Noise types whose seed/strength are inert — mirrors uls_noise.SEEDLESS_TYPES.
const SEEDLESS_TYPES = ["zeros"];

const INACTIVE_MARK = "⊘ ";
const CUSTOM_LABEL = "— Custom —";

// Preview sizing (v521 -- the Media-Loader mechanic). The block is the LAST widget.
// The noise box is CONTAIN-fitted into the available area (box column width x the
// dragged height budget widget._h), keeping the latent aspect, bounded by a MIN and
// a MAX size. computeSize returns EXACTLY the fitted box height + pads, so the node
// HUGS the box: no dead space is ever left below it. Dragging the lower edge up
// shrinks the box (proportionally, down to the min); dragging it down or widening the
// node grows the box (up to the max / the width limit). Only an explicit resize
// changes widget._h -> no node.size feedback into layout.
//
// *** v519 LESSON -- the cause of the v517/v518 Delete-button overlap: never size the
// box from draw()'s `height` argument. ComfyUI passes the fixed NODE_WIDGET_HEIGHT
// (~20) there, NOT the computeSize height. We size from widget._h (our own reserved
// budget), which is authoritative and feedback-free. ***
const PREVIEW_MIN_H = 120;   // floor for the height budget (smallest box + pads)
const PREVIEW_MAX_H = 420;   // ceiling on the height budget (lower-edge drag)
const PREVIEW_DEF_H = 172;   // default height budget on drop
const BOX_MIN = 64;          // smallest the noise box is ever drawn (MIN size)
const BOX_MAX_W = 320;       // maximum box width  (MAX size / "maximale Breite")
const BOX_MAX_H = 320;       // maximum box height (MAX size)
const PREV_GAP = 12;         // gap between the box and the readout column
const PREV_MARGIN = 12;      // left/right inset of the box (left = anchor edge)
const TOP_PAD = 8;           // gap between the Delete button and the box top
const PREV_BOTTOM = 8;       // gap below the box -- the node hugs box + TOP_PAD +
                             // PREV_BOTTOM, so this is the only space under the box
const TEXT_COL_W = 132;      // reserved for the readout to the right of the box
const MIN_NODE_W = 320;      // v519: floor the node width so the widget rows and the
                             // preview box + readout can't be squeezed until fields
                             // telescope (canonical mechanic shared with the
                             // Media-Loader / Cockpit / Stack nodes)

function _findWidget(node, name) {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
}

// ── Take a widget off the node (v888; was greying-in-place since v515) ───────
// The row keeps its slot in node.widgets -- values serialise BY INDEX (#577) --
// so only the pixels go. See web/js/ph_widget_vis.js.
function _setDisabled(w, disabled) {
    if (!w) return false;
    if (w._uls_baseLabel === undefined) w._uls_baseLabel = (w.label != null) ? w.label : w.name;
    w.disabled = disabled;
    w.label = w._uls_baseLabel;
    return setHidden(w, disabled);
}

// ── Derive greying from latent_type + noise_type ─────────────────────────────
function _applyState(node) {
    const tW = _findWidget(node, "latent_type");
    const type = tW ? String(tW.value) : "Image";
    const isVideo = VIDEO_TYPES.includes(type);

    let moved = false;
    // length lives only in the video types
    if (_setDisabled(_findWidget(node, "length"), !isVideo)) moved = true;

    // noise seed/strength are inert for seedless types (zeros)
    const ntW = _findWidget(node, "noise_type");
    const seedless = ntW ? SEEDLESS_TYPES.includes(String(ntW.value)) : false;
    if (_setDisabled(_findWidget(node, "noise_seed"), seedless)) moved = true;
    if (_setDisabled(_findWidget(node, "noise_strength"), seedless)) moved = true;

    if (moved) refit(node);   // v888: one refit per pass
    if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
}

// ════════════════════════════════════════════════════════════════════════════
// Per-type noise-character renderer (structural preview only). Resolution-STABLE:
// every sample is a pure function of its integer pixel position + seed, so the
// field is rendered at the box's own pixel size (1:1, no upscaling -> no blocks)
// and does not shimmer when the box is resized.
// ════════════════════════════════════════════════════════════════════════════
// Integer position hash -> [0,1). Deterministic per (x, y, seed).


// Standard normal N(0,1) at a pixel via Box-Muller on two independent hashes.


// Smooth value noise with a fixed pixel period (smoothstep-interpolated lattice).


// Render an NW×NH grayscale field for the given noise type + seed (values 0..1),
// per pixel from the position hashes above (resolution-stable). White types stay
// crisp per-pixel; the colored types are smooth value-noise octaves.


// Field grid == the box's own pixel size, quantised to 8px steps (so a resize
// rebuilds only every few px) and capped, then rendered 1:1 -> crisp, no blocks.


// Build (and cache) an offscreen canvas of the field at the box's pixel size.


/*
 * v898 -- WHAT ACTUALLY GOVERNS THE FRAME COUNT, as one readable line.
 *
 * This replaces _latentReadout, which had been dead code since v686 moved the
 * preview onto the Seed node: defined, never called, still carrying the old
 * divisor arithmetic. Reviving it as the pill is cheaper than writing a second
 * one and leaves no second truth behind.
 *
 * The rule it makes visible lives in nodes/ph_empty_latent.py::_minimax_av:
 *
 *     if duration_seconds and duration_seconds > 0:  frames_in = from_seconds()
 *     else:                                          frames_in = length
 *
 * Two fields, one silent winner, and a grid that snaps the answer afterwards.
 * Every part of that was invisible on the node. Returns null for any type where
 * the question does not arise -- an honest pill says nothing when it has
 * nothing to say.
 */
function _framePill(node) {
    const tW = _findWidget(node, "latent_type");
    if (!tW || String(tW.value) !== MINIMAX_TYPE) return null;

    const dW = _findWidget(node, "duration_seconds");
    const lW = _findWidget(node, "length");
    const secs = dW ? Number(dW.value) || 0 : 0;
    const len = lW ? Math.max(1, lW.value | 0) : 1;

    const bySeconds = secs > 0;
    const asked = bySeconds ? Math.round(secs * MINIMAX_FPS) : len;
    // v899: one frame is a still, and the model supports it -- the five was
    // never the model's floor, it belongs to Core's AUDIO+VIDEO node. Mirrors
    // the `image_mode` branch in ph_empty_latent._minimax_av.
    const imageMode = (asked === 1);
    const frames = imageMode ? 1 : minimaxAlign(asked);
    const lead = bySeconds
        ? "duration " + (Math.round(secs * 100) / 100) + " s"
        : "length " + len;

    let note = "";
    if (imageMode) {
        note = " · image mode, one latent frame";
    } else if (frames !== asked) {
        // Say WHICH of the two reasons moved the number. Below five it is the
        // floor of Core's VIDEO node -- and the way past it is length 1, which
        // the note names: a dead end without an exit is worse than no note.
        note = (asked < 5)
            ? " · video floor is 5, use 1 for a still"
            : " · snapped to the 17k+5 grid";
    }
    return {
        text: lead + " \u2192 " + frames + (frames === 1 ? " frame" : " frames"),
        note: note,
        seconds: Math.round((frames / MINIMAX_FPS) * 1000) / 1000,
        bySeconds: bySeconds,
        imageMode: imageMode,
        frames: frames,
    };
}

// ── The custom noise-preview widget (draw + mouse-scrub) ─────────────────────
// Contain-fit the noise box into the available area (box column width x the height
// budget), keeping the latent aspect, bounded by BOX_MIN and BOX_MAX_W/H. Shared by
// computeSize (so the node hugs the box) and draw (so it renders the same box). `width`
// is the widget width LiteGraph passes to both; `desiredH` is widget._h (drag budget).
function _fitBox(width, desiredH, aspect) {
    const a = (aspect && aspect > 0) ? aspect : 1;
    const availW = Math.max(BOX_MIN, width - 2 * PREV_MARGIN - PREV_GAP - TEXT_COL_W);
    const budget = Math.max(PREVIEW_MIN_H, Math.min(PREVIEW_MAX_H, desiredH || PREVIEW_DEF_H));
    const availH = Math.max(BOX_MIN, budget - TOP_PAD - PREV_BOTTOM);
    // largest rectangle with the latent aspect fitting (availW x availH), then capped
    let boxH = availH;
    let boxW = boxH * a;
    if (boxW > availW)    { boxW = availW;    boxH = boxW / a; }
    if (boxW > BOX_MAX_W) { boxW = BOX_MAX_W; boxH = boxW / a; }
    if (boxH > BOX_MAX_H) { boxH = BOX_MAX_H; boxW = boxH * a; }
    boxW = Math.max(BOX_MIN, boxW);
    boxH = Math.max(BOX_MIN, boxH);
    return { w: boxW, h: boxH };
}


// v685 -- hide the legacy noise controls without touching the canon.
// The house law (v585): PYTHON = CANON, never re-ordered and never shortened,
// because widgets_values loads BY POSITION. The frontend may hide a row; it
// may never make one disappear from the array. type="hidden" + a zero
// computeSize does exactly that: LiteGraph skips drawing it, keeps it in
// node.widgets, and therefore keeps serialising its value.
// v688: control_after_generate belongs to this list too. It is not ours --
// ComfyUI attaches it automatically to any seed-shaped INT, in this case the
// (now hidden) noise_seed -- which is why it was left standing alone between
// batch_size and preset. Same treatment as the rest: hidden, never removed. It
// sits in widgets_values right behind noise_seed, so dropping it would shift
// preset and everything after it in every saved graph.
const LEGACY_NOISE = ["noise_type", "noise_seed", "noise_strength",
                      "control_after_generate"];

function _hideLegacyNoise(node) {
    if (!node || !Array.isArray(node.widgets)) return 0;
    let n = 0;
    for (const w of node.widgets) {
        if (!LEGACY_NOISE.includes(w.name)) continue;
        w.type = "hidden";
        w.computeSize = () => [0, -4];   // -4 cancels LiteGraph's row spacing
        n++;
    }
    return n;
}

// ════════════════════════════════════════════════════════════════════════════
// Size-preset combo + Save/Delete buttons
// ════════════════════════════════════════════════════════════════════════════
function _presetLabel(p) {
    let s = p.name;
    if (p.aspect) s += "  (" + p.aspect + ")";
    return s;
}

// User presets FIRST, then built-ins (the requested ordering).
function _orderedPresets(presets) {
    const user = presets.filter((p) => !p.builtin);
    const builtin = presets.filter((p) => p.builtin);
    return user.concat(builtin);
}

function _presetTypeLabel(p) {
    if (p.family) {
        const key = String(p.family).toLowerCase();
        return FAMILY_TO_TYPE[key] || "Image";
    }
    return null;   // no family -> leave the current type untouched (size-only preset)
}

// Does the current node state match this preset? (used to show the right label)
function _matchesPreset(node, p) {
    const wv = _findWidget(node, "width"), hv = _findWidget(node, "height");
    if (!wv || !hv || wv.value !== p.width || hv.value !== p.height) return false;
    const tl = _presetTypeLabel(p);
    if (tl) {
        const tW = _findWidget(node, "latent_type");
        if (!tW || String(tW.value) !== tl) return false;
        if (p.length != null) {
            const lv = _findWidget(node, "length");
            if (!lv || lv.value !== p.length) return false;
        }
    }
    return true;
}

function _applyPreset(node, p) {
    if (!p) return;
    const set = (name, val) => {
        const w = _findWidget(node, name);
        if (w && val != null) { w.value = val; if (w.callback) w.callback(w.value); }
    };
    const tl = _presetTypeLabel(p);
    if (tl) set("latent_type", tl);
    set("width", p.width);
    set("height", p.height);
    if (tl && p.length != null) set("length", p.length);
    _applyState(node);
    if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
}

// Reflect the current node state in the combo label (a real preset, or Custom).
function _syncPresetCombo(node, comboW) {
    if (!comboW) return;
    const presets = node._uls_presets || [];
    const hit = presets.find((p) => _matchesPreset(node, p));
    comboW.value = hit ? _presetLabel(hit) : CUSTOM_LABEL;
    if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
}

async function _fetchPresets() {
    try {
        const r = await api.fetchApi("/pls/latent_presets");
        const j = await r.json();
        return Array.isArray(j.presets) ? j.presets : [];
    } catch (e) {
        console.warn("[PLS] latent presets fetch failed:", e);
        return [];
    }
}

function _refreshPresetCombo(node, comboW, presets) {
    const ordered = _orderedPresets(presets);
    node._uls_presets = ordered;
    // values = user presets first, then built-ins (no dead placeholder)
    comboW.options.values = ordered.map(_presetLabel);
    _syncPresetCombo(node, comboW);   // show the matching preset, or Custom
    if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
}

function _addPresetControls(node) {
    // combo (non-serialising; the concrete w/h/length widgets carry the values).
    // Sticky value + real preset entries -> the ◀ ▶ arrows cycle and apply.
    const comboW = node.addWidget("combo", "preset", CUSTOM_LABEL, (v) => {
        if (v === CUSTOM_LABEL) return;                 // Custom is a state, not an action
        const presets = node._uls_presets || [];
        const p = presets.find((q) => _presetLabel(q) === v);
        if (p) _applyPreset(node, p);
    }, { values: [] });
    comboW.serialize = false;

    const saveW = node.addWidget("button", "＋ save preset", null, async () => {
        const tW = _findWidget(node, "latent_type");
        const type = tW ? String(tW.value) : "Image";
        const isVideo = VIDEO_TYPES.includes(type);
        const wv = _findWidget(node, "width"), hv = _findWidget(node, "height");
        const lv = _findWidget(node, "length");
        const def = (wv && hv) ? `${wv.value}x${hv.value}` : "preset";
        const name = window.prompt("Preset name:", def);
        if (!name) return;
        const body = {
            action: "save",
            preset: {
                name: name,
                width: wv ? wv.value : 1024,
                height: hv ? hv.value : 1024,
            },
        };
        if (isVideo) {
            // store the canonical key so it round-trips across future type renames
            const keyMap = { "WAN video": "wan", "Hunyuan video": "hunyuan", "Mochi video": "mochi",
                             "LTXV video": "ltxv", "Cosmos video": "cosmos" };
            body.preset.family = keyMap[type] || "wan";
            if (lv) body.preset.length = lv.value;
        }
        try {
            const r = await api.fetchApi("/pls/latent_presets", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            const j = await r.json();
            if (j.ok) _refreshPresetCombo(node, comboW, j.presets);
            else window.alert("Save failed: " + (j.error || "unknown"));
        } catch (e) { window.alert("Save failed: " + e); }
    });
    saveW.serialize = false;

    const delW = node.addWidget("button", "🗑 delete preset…", null, async () => {
        const presets = node._uls_presets || [];
        const userNames = presets.filter((p) => !p.builtin).map((p) => p.name);
        if (userNames.length === 0) { window.alert("No user presets to delete (built-ins are fixed)."); return; }
        const name = window.prompt("Delete which user preset?\n\n" + userNames.join("\n"), userNames[0]);
        if (!name) return;
        try {
            const r = await api.fetchApi("/pls/latent_presets", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ action: "delete", name: name }),
            });
            const j = await r.json();
            if (j.ok) _refreshPresetCombo(node, comboW, j.presets);
            else window.alert("Delete failed: " + (j.error || "unknown"));
        } catch (e) { window.alert("Delete failed: " + e); }
    });
    delW.serialize = false;

    // when width/height/length/type change by hand, keep the combo label honest
    for (const nm of ["latent_type", "width", "height", "length"]) {
        const w = _findWidget(node, nm);
        if (w) {
            const _cb = w.callback;
            w.callback = function () {
                const rr = _cb ? _cb.apply(this, arguments) : undefined;
                _syncPresetCombo(node, comboW);
                return rr;
            };
        }
    }

    // populate asynchronously
    _fetchPresets().then((presets) => _refreshPresetCombo(node, comboW, presets));
    return comboW;
}

// ════════════════════════════════════════════════════════════════════════════

// ── The pill: one line saying which field governs, and what comes out ────────
// A CUSTOM widget with serialize:false -- it never enters widgets_values (#577),
// so it cannot renumber a saved graph. Its height is declared INSIDE computeSize
// (the v891 lesson): a row painted in onDrawForeground gets covered by DOM
// widgets, and a row whose height is not in computeSize overlaps its neighbour.
// Zero height when it has nothing to say, so every non-H3 type is untouched.
const PILL_NAME = "$ph_frame_pill";
const PILL_TYPE = "uls_frame_pill";
const PILL_H = 20;

function makePillWidget() {
    const w = {
        type: PILL_TYPE,
        name: PILL_NAME,
        value: null,
        serialize: false,
        computeSize(width) {
            return [width, (w._node && _framePill(w._node)) ? PILL_H : 0];
        },
        draw(ctx, node, width, y) {
            w._node = node;
            const info = _framePill(node);
            if (!info) return;
            ctx.save();
            try {
                const x = 14;
                ctx.font = "11px sans-serif";
                ctx.textBaseline = "middle";
                const mid = y + PILL_H / 2;
                // The leading half is the answer; the note is the reason, in a
                // quieter colour, so the eye reads the number first.
                ctx.fillStyle = "#ddd";
                ctx.fillText(info.text, x, mid);
                if (info.note) {
                    const wi = ctx.measureText(info.text).width;
                    ctx.fillStyle = "#8a8a8a";
                    ctx.fillText(info.note, x + wi, mid);
                }
                const right = info.seconds + " s";
                ctx.fillStyle = "#8a8a8a";
                ctx.textAlign = "right";
                ctx.fillText(right, width - 14, mid);
            } catch (e) {
                /* a readout must never break the canvas */
            }
            ctx.restore();
        },
    };
    return w;
}

function ensurePill(node) {
    if (typeof node.addCustomWidget !== "function") return false;
    if ((node.widgets || []).some((x) => x && x.name === PILL_NAME)) return false;
    const w = node.addCustomWidget(makePillWidget());
    if (w) w._node = node;
    return true;
}

app.registerExtension({
    name: "polyhedron.empty_latent",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const _onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = _onCreated ? _onCreated.apply(this, arguments) : undefined;
            const self = this;

            ensurePill(self);

            // v898: the pill reads three widgets, so all three must repaint it.
            // length and duration_seconds do not change the greying, which is
            // why they were not in the list below.
            for (const nm of ["latent_type", "length", "duration_seconds"]) {
                const pw = _findWidget(self, nm);
                if (pw) {
                    const _pcb = pw.callback;
                    pw.callback = function () {
                        const rr = _pcb ? _pcb.apply(this, arguments) : undefined;
                        if (self.setDirtyCanvas) self.setDirtyCanvas(true, true);
                        refit(self);
                        return rr;
                    };
                }
            }

            // re-derive greying when type or noise_type change
            for (const nm of ["latent_type", "noise_type"]) {
                const w = _findWidget(self, nm);
                if (w) {
                    const _cb = w.callback;
                    w.callback = function () {
                        const rr = _cb ? _cb.apply(this, arguments) : undefined;
                        _applyState(self);
                        return rr;
                    };
                }
            }

            _addPresetControls(self);
            // v685: the noise preview and its three widgets MOVED to the
            // Polyhedron Seed node, where noise actually reaches the model.
            // What stays here is the CANON: widgets_values is a positional
            // array, so removing noise_type / noise_seed / noise_strength from
            // Python would shift every value after them in every saved graph
            // (preset landing in batch_size, and so on). They are hidden, not
            // removed -- zero size, never drawn, still serialised, values
            // untouched. _hideLegacyNoise is called from the same setTimeout
            // the rest of the setup uses.
            _hideLegacyNoise(self);
            _applyState(self);

            // v687: the noise preview lives on the Polyhedron Seed node now (it
            // shows the REAL field from uls_noise.make_noise), so nothing here
            // reserves block height any more.
            // v519: enforce the minimum node width on drop -- onResize only fires during
            // a drag, so a freshly created node needs the floor applied here too.
            if (self.size) self.size[0] = Math.max(self.size[0] || 0, MIN_NODE_W);
            return r;
        };

        const _onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = _onConfigure ? _onConfigure.apply(this, arguments) : undefined;
            _applyState(this);
            // v687: the preview height budget moved with the preview itself to
            // the Polyhedron Seed node (guard v539 followed it). An old graph
            // may still carry uls_preview_h in its properties -- harmless, and
            // deliberately not read here: restoring a height for a widget that
            // no longer exists is how dead code survives.
            // v519: a graph saved before the width floor may hold a sub-minimum width
            // that would telescope the fields -> clamp it up on load, too.
            if (this.size && this.size[0] < MIN_NODE_W) this.size[0] = MIN_NODE_W;
            return r;
        };
    },
});
