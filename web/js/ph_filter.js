/*
 * ph_filter.js -- in-node preview pane for the "Polyhedron Filter" node.
 *
 * Machinery:
 *   - one DOM widget (non-canon, serialize:false -> rides below the canon
 *     widgets, HANDOVER 4) holding a header row + a <canvas>;
 *   - the backend sends a downscaled UNGRADED proxy of the incoming frame
 *     over {"ui":{"ph_filter":[{filename, subfolder, type, width, height}]}};
 *     the canvas shows it aspect-fitted (letterboxed);
 *   - LIVE GRADING: _gradeRGB mirrors the backend pipeline (_grade_np in
 *     nodes/ph_filter.py -- the ground truth) op for op; every grading
 *     widget change recomputes the graded proxy, coalesced to one pass per
 *     animation frame. The preview is an 8-bit approximation of the run;
 *   - a draggable BEFORE/AFTER divider splits the canvas: left pane is the
 *     original proxy, right pane is the graded proxy;
 *   - "A/B" press-and-hold shows the full original;
 *   - the ui entry is stored in node.properties.ph_filter_preview, so a
 *     page reload restores the preview as long as ComfyUI's temp file
 *     lives (a ComfyUI restart clears temp -> run once to repopulate).
 *
 * House rules honored: no ResizeObserver, no document-level listeners
 * (pointer capture keeps every event on the canvas itself), the DOM widget
 * reserves height through computeSize only (feedback-free, ph_save.js
 * pattern), and nothing here serializes into widgets_values.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

console.info("[PLS] ph_filter.js loaded");

// Mirror of the python INPUT_TYPES defaults -- the Reset button writes these.
// The guard pins this map against the python source; edit both or neither.
const CANON_DEFAULTS = {
    exposure: 0.0, temperature: 0.0, tint: 0.0, contrast: 0.0, gamma: 1.0,
    shadows: 0.0, highlights: 0.0, saturation: 0.0, vibrance: 0.0, hue_shift: 0.0,
    lut_name: "none", lut_strength: 1.0, sharpen_amount: 0.0, sharpen_radius: 1.0,
    preset: "none",
};

const PREVIEW_MIN_H = 140;   // floor for the reserved preview height (px)
const PREVIEW_MAX_H = 900;   // ceiling for the reserved preview height (px)
const PREVIEW_DEF_H = 260;   // reserved height before any preview loads
const HEADER_H      = 52;    // button row + two full-width hint lines (px)
const PANE_PAD      = 8;     // inner padding around the canvas (px)
const DIVIDER_GRAB  = 14;    // pointer distance (px) that grabs the divider

function _clampFrac(f) {
    if (!(f >= 0)) return 0;
    if (f > 1) return 1;
    return f;
}

function _fitRect(iw, ih, cw, ch) {
    // Aspect-fit (letterbox) of an iw x ih image inside a cw x ch canvas.
    // Integer output; centered. Pure math -- guard-driven.
    if (iw <= 0 || ih <= 0 || cw <= 0 || ch <= 0) return { x: 0, y: 0, w: 0, h: 0 };
    const s = Math.min(cw / iw, ch / ih);
    const w = Math.max(1, Math.floor(iw * s));
    const h = Math.max(1, Math.floor(ih * s));
    return { x: Math.floor((cw - w) / 2), y: Math.floor((ch - h) / 2), w: w, h: h };
}

function _paneHeight(iw, ih, nodeW) {
    // Reserved pane height for an iw x ih proxy at the CURRENT node width.
    // Follows the media aspect, clamped to [PREVIEW_MIN_H, PREVIEW_MAX_H].
    // Pure math -- guard-driven. Feeds computeSize only; geometry writes
    // must preserve the node's width (the full setSize(computeSize()) reset
    // is the re-run collapse wound and is banned in this file).
    const ar = (iw > 0 && ih > 0) ? iw / ih : 1;
    const innerW = Math.max(50, nodeW - 2 * PANE_PAD);
    let hh = Math.round(innerW / ar) + HEADER_H + 2 * PANE_PAD;
    if (hh < PREVIEW_MIN_H) hh = PREVIEW_MIN_H;
    if (hh > PREVIEW_MAX_H) hh = PREVIEW_MAX_H;
    return hh;
}

function _gradeRGB(r, g, b, p) {
    // OP-FOR-OP mirror of _grade_np in nodes/ph_filter.py -- the backend is
    // the ground truth, this is the live preview of it. The parity guard
    // drives both on the same input; any edit here must land there too.
    // p: {exposure, temperature, tint, contrast, gamma, shadows, highlights,
    //     saturation, vibrance, hue_shift}
    const ev = Math.pow(2, p.exposure);
    r *= ev; g *= ev; b *= ev;

    r *= 1 + 0.25 * p.temperature;
    b *= 1 - 0.25 * p.temperature;
    g *= 1 - 0.25 * p.tint;

    const cf = 1 + p.contrast;
    r = 0.5 + (r - 0.5) * cf;
    g = 0.5 + (g - 0.5) * cf;
    b = 0.5 + (b - 0.5) * cf;

    r = Math.pow(Math.max(r, 0), p.gamma);
    g = Math.pow(Math.max(g, 0), p.gamma);
    b = Math.pow(Math.max(b, 0), p.gamma);

    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    const shLift = p.shadows * 0.25 * (1 - luma) * (1 - luma);
    const hiLift = p.highlights * 0.25 * luma * luma;
    r += shLift + hiLift; g += shLift + hiLift; b += shLift + hiLift;

    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    const sf = 1 + p.saturation;
    r = luma + (r - luma) * sf;
    g = luma + (g - luma) * sf;
    b = luma + (b - luma) * sf;

    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let rng = Math.max(r, g, b) - Math.min(r, g, b);
    rng = rng < 0 ? 0 : (rng > 1 ? 1 : rng);
    const vf = 1 + p.vibrance * (1 - rng);
    r = luma + (r - luma) * vf;
    g = luma + (g - luma) * vf;
    b = luma + (b - luma) * vf;

    if (p.hue_shift !== 0) {
        const a = p.hue_shift * Math.PI / 180;
        const c = Math.cos(a), s = Math.sin(a);
        const nr = (0.213 + 0.787 * c - 0.213 * s) * r + (0.715 - 0.715 * c - 0.715 * s) * g + (0.072 - 0.072 * c + 0.928 * s) * b;
        const ng = (0.213 - 0.213 * c + 0.143 * s) * r + (0.715 + 0.285 * c + 0.140 * s) * g + (0.072 - 0.072 * c - 0.283 * s) * b;
        const nb = (0.213 - 0.213 * c - 0.787 * s) * r + (0.715 - 0.715 * c + 0.715 * s) * g + (0.072 + 0.928 * c + 0.072 * s) * b;
        r = nr; g = ng; b = nb;
    }

    r = r < 0 ? 0 : (r > 1 ? 1 : r);
    g = g < 0 ? 0 : (g > 1 ? 1 : g);
    b = b < 0 ? 0 : (b > 1 ? 1 : b);
    return [r, g, b];
}

function _parseCube(text) {
    // OP-FOR-OP mirror of _parse_cube in nodes/ph_filter.py. Returns
    // {size, data (Float32Array, [b][g][r] majority, r fastest), dmin, dmax}
    // or throws on malformed input.
    let size = 0;
    let dmin = [0, 0, 0], dmax = [1, 1, 1];
    const rows = [];
    for (const raw of text.split(/\r?\n/)) {
        const line = raw.trim();
        if (!line || line.startsWith("#")) continue;
        const up = line.toUpperCase();
        if (up.startsWith("TITLE")) continue;
        if (up.startsWith("LUT_3D_SIZE")) { size = parseInt(line.split(/\s+/)[1], 10); continue; }
        if (up.startsWith("LUT_1D_SIZE")) throw new Error("1D LUTs are not supported");
        if (up.startsWith("DOMAIN_MIN")) { dmin = line.split(/\s+/).slice(1, 4).map(Number); continue; }
        if (up.startsWith("DOMAIN_MAX")) { dmax = line.split(/\s+/).slice(1, 4).map(Number); continue; }
        const parts = line.split(/\s+/);
        if (parts.length === 3) rows.push([Number(parts[0]), Number(parts[1]), Number(parts[2])]);
    }
    if (!(size >= 2)) throw new Error("missing or invalid LUT_3D_SIZE");
    if (rows.length !== size * size * size)
        throw new Error("expected " + size * size * size + " entries, got " + rows.length);
    const data = new Float32Array(rows.length * 3);
    for (let i = 0; i < rows.length; i++) {
        data[i * 3] = rows[i][0];
        data[i * 3 + 1] = rows[i][1];
        data[i * 3 + 2] = rows[i][2];
    }
    return { size: size, data: data, dmin: dmin, dmax: dmax };
}

function _lutRGB(r, g, b, lut, strength) {
    // OP-FOR-OP mirror of _apply_lut_np in nodes/ph_filter.py: trilinear
    // 3D-LUT lookup blended with the input by strength, clamped to 0..1.
    const n = lut.size, d = lut.data;
    const cr = _lutCoord(r, lut.dmin[0], lut.dmax[0], n);
    const cg = _lutCoord(g, lut.dmin[1], lut.dmax[1], n);
    const cb = _lutCoord(b, lut.dmin[2], lut.dmax[2], n);
    const r0 = Math.floor(cr), g0 = Math.floor(cg), b0 = Math.floor(cb);
    const r1 = Math.min(r0 + 1, n - 1), g1 = Math.min(g0 + 1, n - 1), b1 = Math.min(b0 + 1, n - 1);
    const fr = cr - r0, fg = cg - g0, fb = cb - b0;
    const at = (bi, gi, ri, ch) => d[((bi * n + gi) * n + ri) * 3 + ch];
    const out = [0, 0, 0];
    const src = [r, g, b];
    for (let ch = 0; ch < 3; ch++) {
        const lo = (1 - fg) * ((1 - fr) * at(b0, g0, r0, ch) + fr * at(b0, g0, r1, ch))
                 + fg * ((1 - fr) * at(b0, g1, r0, ch) + fr * at(b0, g1, r1, ch));
        const hi = (1 - fg) * ((1 - fr) * at(b1, g0, r0, ch) + fr * at(b1, g0, r1, ch))
                 + fg * ((1 - fr) * at(b1, g1, r0, ch) + fr * at(b1, g1, r1, ch));
        let v = src[ch] * (1 - strength) + ((1 - fb) * lo + fb * hi) * strength;
        out[ch] = v < 0 ? 0 : (v > 1 ? 1 : v);
    }
    return out;
}

function _lutCoord(v, lo, hi, n) {
    const span = Math.max(hi - lo, 1e-6);
    let t = (v - lo) / span;
    t = t < 0 ? 0 : (t > 1 ? 1 : t);
    return t * (n - 1);
}

function _gaussKernel(radius) {
    // OP-FOR-OP mirror of _gauss_kernel in nodes/ph_filter.py: sigma equals
    // the radius, half-width ceil(3 sigma), normalized weights.
    const sigma = Math.max(radius, 0.1);
    const half = Math.max(1, Math.ceil(3 * sigma));
    const w = new Array(2 * half + 1);
    let sum = 0;
    for (let i = -half; i <= half; i++) {
        const v = Math.exp(-(i * i) / (2 * sigma * sigma));
        w[i + half] = v;
        sum += v;
    }
    for (let i = 0; i < w.length; i++) w[i] /= sum;
    return { half: half, w: w };
}

function _sharpenBuf(buf, width, height, amount, radius) {
    // OP-FOR-OP mirror of _sharpen_np in nodes/ph_filter.py: unsharp mask
    // out = x + amount * (x - gaussian_blur(x)), separable blur with
    // replicate (edge-clamp) borders, clamped to 0..1. buf: Float32Array
    // [r,g,b, r,g,b, ...] row-major; returns a NEW Float32Array.
    if (!(amount > 0)) return buf;
    const k = _gaussKernel(radius);
    const half = k.half, w = k.w;
    const n = width * height;
    const tmp = new Float32Array(n * 3);
    const blur = new Float32Array(n * 3);
    // horizontal pass (replicate edges)
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let r = 0, g = 0, b = 0;
            for (let i = -half; i <= half; i++) {
                let xx = x + i;
                xx = xx < 0 ? 0 : (xx >= width ? width - 1 : xx);
                const j = (y * width + xx) * 3, wt = w[i + half];
                r += wt * buf[j]; g += wt * buf[j + 1]; b += wt * buf[j + 2];
            }
            const o = (y * width + x) * 3;
            tmp[o] = r; tmp[o + 1] = g; tmp[o + 2] = b;
        }
    }
    // vertical pass (replicate edges)
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let r = 0, g = 0, b = 0;
            for (let i = -half; i <= half; i++) {
                let yy = y + i;
                yy = yy < 0 ? 0 : (yy >= height ? height - 1 : yy);
                const j = (yy * width + x) * 3, wt = w[i + half];
                r += wt * tmp[j]; g += wt * tmp[j + 1]; b += wt * tmp[j + 2];
            }
            const o = (y * width + x) * 3;
            blur[o] = r; blur[o + 1] = g; blur[o + 2] = b;
        }
    }
    const out = new Float32Array(n * 3);
    for (let i = 0; i < out.length; i++) {
        let v = buf[i] + amount * (buf[i] - blur[i]);
        out[i] = v < 0 ? 0 : (v > 1 ? 1 : v);
    }
    return out;
}

function viewURL(item) {
    const p = new URLSearchParams({
        filename: item.filename || "",
        subfolder: item.subfolder || "",
        type: item.type || "temp",
    });
    return `/view?${p.toString()}&r=${Date.now()}`;   // cache-bust each run
}

const GRADE_WIDGETS = ["exposure", "temperature", "tint", "contrast", "gamma",
                       "shadows", "highlights", "saturation", "vibrance", "hue_shift"];
const LIVE_WIDGETS = GRADE_WIDGETS.concat(["lut_name", "lut_strength",
                                           "sharpen_amount", "sharpen_radius"]);

function _pfParams(node) {
    const p = {};
    for (const name of GRADE_WIDGETS) {
        const w = (node.widgets || []).find((x) => x.name === name);
        p[name] = w ? Number(w.value) : (name === "gamma" ? 1 : 0);
    }
    return p;
}

function _pfLutState(node) {
    const wName = (node.widgets || []).find((x) => x.name === "lut_name");
    const wStr = (node.widgets || []).find((x) => x.name === "lut_strength");
    return {
        name: wName ? String(wName.value) : "none",
        strength: wStr ? Number(wStr.value) : 1,
    };
}

function _pfEnsureLut(node) {
    // Fetch + parse the selected .cube once per name (served by the pack's
    // /uls/filter/lut route -- the SAME file the backend grades with). Until
    // it arrives -- or if it fails -- the preview grades without the LUT and
    // _pfDraw says so instead of silently diverging from the run.
    const want = _pfLutState(node).name;
    if (want === "none") {
        node._pfLut = null;
        node._pfLutName = "none";
        node._pfLutErr = false;
        return;
    }
    if (node._pfLutName === want && (node._pfLut || node._pfLutErr)) return;
    node._pfLutName = want;
    node._pfLut = null;
    node._pfLutErr = false;
    api.fetchApi("/uls/filter/lut?name=" + encodeURIComponent(want))
        .then((res) => {
            if (!res.ok) throw new Error("HTTP " + res.status);
            return res.text();
        })
        .then((text) => {
            if (node._pfLutName !== want) return;   // stale response
            node._pfLut = _parseCube(text);
            _pfSchedule(node);
        })
        .catch((e) => {
            if (node._pfLutName !== want) return;
            node._pfLut = null;
            node._pfLutErr = true;
            console.warn("[PLS] Polyhedron Filter: LUT preview unavailable:", want, e);
            _pfSchedule(node);
        });
}

function _pfRecompute(node) {
    // Grade the cached source pixels with the mirrored pipeline into the
    // offscreen graded canvas. Runs over the 768px proxy (sub-megapixel);
    // scheduled through _pfSchedule so slider drags coalesce per frame.
    const src = node._pfSrcData;
    if (!src) return;
    const p = _pfParams(node);
    const lutState = _pfLutState(node);
    const lut = (lutState.name !== "none" && lutState.strength > 0 && node._pfLut)
        ? node._pfLut : null;
    const wSharpA = (node.widgets || []).find((x) => x.name === "sharpen_amount");
    const wSharpR = (node.widgets || []).find((x) => x.name === "sharpen_radius");
    const sharpA = wSharpA ? Number(wSharpA.value) : 0;
    const sharpR = wSharpR ? Number(wSharpR.value) : 1;
    if (!node._pfGraded) {
        node._pfGraded = document.createElement("canvas");
    }
    const gc = node._pfGraded;
    gc.width = src.width;
    gc.height = src.height;
    const outData = new ImageData(src.width, src.height);
    const a = src.data, o = outData.data;
    const npx = src.width * src.height;
    let fbuf = new Float32Array(npx * 3);
    for (let i = 0, f = 0; i < a.length; i += 4, f += 3) {
        let rgb = _gradeRGB(a[i] / 255, a[i + 1] / 255, a[i + 2] / 255, p);
        if (lut) rgb = _lutRGB(rgb[0], rgb[1], rgb[2], lut, lutState.strength);
        fbuf[f] = rgb[0]; fbuf[f + 1] = rgb[1]; fbuf[f + 2] = rgb[2];
    }
    if (sharpA > 0) fbuf = _sharpenBuf(fbuf, src.width, src.height, sharpA, sharpR);
    for (let i = 0, f = 0; i < o.length; i += 4, f += 3) {
        o[i] = Math.round(fbuf[f] * 255);
        o[i + 1] = Math.round(fbuf[f + 1] * 255);
        o[i + 2] = Math.round(fbuf[f + 2] * 255);
        o[i + 3] = a[i + 3];
    }
    gc.getContext("2d").putImageData(outData, 0, 0);
    _pfDraw(node);
}

function _pfSchedule(node) {
    // Coalesce recomputes to one per animation frame (render throttling
    // only -- no layout is measured here).
    if (node._pfPending) return;
    node._pfPending = true;
    requestAnimationFrame(() => {
        node._pfPending = false;
        _pfRecompute(node);
    });
}

const TITLE_HINT = "click-drag values \u00b7 click to type";

function _pfTitleLayout(nodeW, titleH) {
    // Geometry of the painted title controls (node-local coords; the title
    // strip spans y in [-titleH, 0)). Pure math -- guard-driven.
    // Returns {reset:[x,y,w,h]}; the scrub hint is drawn left of the chip
    // when the width allows (the fields themselves carry the full sentence
    // in their tooltips, so hiding the hint on narrow nodes loses nothing).
    const chipW = 44, chipH = 16, margin = 26;
    const rx = nodeW - margin - chipW;
    const ry = -titleH + Math.floor((titleH - chipH) / 2);
    return { reset: [rx, ry, chipW, chipH] };
}

function _pfHit(rect, pos) {
    return pos[0] >= rect[0] && pos[0] <= rect[0] + rect[2]
        && pos[1] >= rect[1] && pos[1] <= rect[1] + rect[3];
}

function _pfReset(node) {
    // Every canon control back to its default (the guard pins
    // CANON_DEFAULTS against the python INPUT_TYPES defaults).
    for (const w of node.widgets || []) {
        if (!(w.name in CANON_DEFAULTS)) continue;
        w.value = CANON_DEFAULTS[w.name];
        if (w.callback) w.callback(w.value);
    }
    _pfEnsureLut(node);
    _pfSchedule(node);
    node.setDirtyCanvas(true, true);
}

const PRESET_KEYS = Object.keys(CANON_DEFAULTS).filter((k) => k !== "preset");

function _pfApplyParams(node, params) {
    // Write a sanitized preset onto the widgets (values only -- the canon
    // stays the single source; the sliders visibly jump, then stay freely
    // adjustable). Unknown keys were already dropped server-side.
    for (const w of node.widgets || []) {
        if (!(w.name in params) || w.name === "preset") continue;
        w.value = params[w.name];
        if (w.callback) w.callback(w.value);
    }
    _pfEnsureLut(node);
    _pfSchedule(node);
    node.setDirtyCanvas(true, true);
}

function _pfLoadPreset(node, file) {
    if (!file || file === "none") return;
    api.fetchApi("/uls/filter/preset?name=" + encodeURIComponent(file))
        .then((res) => {
            if (!res.ok) throw new Error("HTTP " + res.status);
            return res.json();
        })
        .then((data) => _pfApplyParams(node, data.params || {}))
        .catch((e) => console.warn("[PLS] Polyhedron Filter: preset load failed:", file, e));
}

function _pfSavePreset(node, name) {
    const params = {};
    for (const w of node.widgets || []) {
        if (PRESET_KEYS.includes(w.name)) params[w.name] = w.value;
    }
    api.fetchApi("/uls/filter/preset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name, params: params }),
    })
        .then((res) => {
            if (!res.ok) throw new Error("HTTP " + res.status);
            return res.json();
        })
        .then((data) => {
            // Make the new file selectable NOW (the server-side combo list
            // refreshes on the next node-definition reload anyway).
            const w = (node.widgets || []).find((x) => x.name === "preset");
            if (w && w.options && Array.isArray(w.options.values)
                && !w.options.values.includes(data.file)) {
                w.options.values.push(data.file);
            }
            if (w) w.value = data.file;
            node.setDirtyCanvas(true, true);
            console.info("[PLS] Polyhedron Filter: preset saved:", data.file);
        })
        .catch((e) => console.warn("[PLS] Polyhedron Filter: preset save failed:", e));
}

function _pfDrawTitleControls(node, ctx) {
    const titleH = (typeof LiteGraph !== "undefined" && LiteGraph.NODE_TITLE_HEIGHT) || 30;
    const lay = _pfTitleLayout(node.size[0], titleH);
    node._pfTitleHits = lay;

    // Reset chip
    const [rx, ry, rw, rh] = lay.reset;
    ctx.save();
    ctx.fillStyle = "#333";
    ctx.strokeStyle = "#555";
    ctx.lineWidth = 1;
    ctx.fillRect(rx, ry, rw, rh);
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.fillStyle = "#ff8c00";
    ctx.font = "normal 10px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("Reset", rx + rw / 2, ry + rh / 2 + 0.5);

    // scrub hint, painted RIGHT AT THE FIELDS' top edge (the point of the
    // hint is the fields-are-sliders connection): small grey text left of
    // the Reset chip, drawn only when it fits next to the node title. The
    // same sentence lives in every numeric field's own tooltip and in the
    // node description, so nothing is lost on narrow nodes.
    ctx.font = "normal 10px sans-serif";
    ctx.textAlign = "right";
    const hintW = ctx.measureText(TITLE_HINT).width;
    if (rx - 10 - hintW > 150) {   // keep clear of the node title text
        ctx.fillStyle = "#888";
        ctx.fillText(TITLE_HINT, rx - 10, ry + rh / 2 + 0.5);
    }
    ctx.restore();
}

function _pfDraw(node) {
    const cv = node._pfCanvas;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    const cw = cv.width, ch = cv.height;
    ctx.clearRect(0, 0, cw, ch);
    ctx.fillStyle = "#181818";
    ctx.fillRect(0, 0, cw, ch);

    const img = node._pfImg;
    if (!img) {
        ctx.fillStyle = "#888";
        ctx.font = "12px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(node._pfNote || "run once to load the preview", cw / 2, ch / 2);
        return;
    }

    const r = _fitRect(img.naturalWidth, img.naturalHeight, cw, ch);

    if (node._pfShowOrig) {
        // A/B hold: full original, no divider.
        ctx.drawImage(img, r.x, r.y, r.w, r.h);
        _pfBadge(ctx, r.x + 6, r.y + 6, "original");
        return;
    }

    // Right side first: the GRADED proxy (mirrored JS pipeline output);
    // falls back to the source until the first recompute lands.
    const graded = (node._pfGraded && node._pfGraded.width > 0) ? node._pfGraded : img;
    ctx.drawImage(graded, r.x, r.y, r.w, r.h);

    // Left of the divider: the ORIGINAL proxy, clipped.
    const frac = _clampFrac(node._pfFrac);
    const splitX = r.x + Math.round(r.w * frac);
    ctx.save();
    ctx.beginPath();
    ctx.rect(r.x, r.y, splitX - r.x, r.h);
    ctx.clip();
    ctx.drawImage(img, r.x, r.y, r.w, r.h);
    ctx.restore();

    // Divider line + handle.
    ctx.strokeStyle = "#ff8c00";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(splitX, r.y);
    ctx.lineTo(splitX, r.y + r.h);
    ctx.stroke();
    ctx.fillStyle = "#ff8c00";
    ctx.beginPath();
    ctx.arc(splitX, r.y + r.h / 2, 6, 0, Math.PI * 2);
    ctx.fill();

    _pfBadge(ctx, r.x + 6, r.y + 6, "in");
    _pfBadge(ctx, r.x + r.w - 40, r.y + 6, "out");

    // Honesty badge: a LUT is selected but the preview could not load it --
    // the right side is then grading WITHOUT the LUT, unlike the run.
    const ls = _pfLutState(node);
    if (ls.name !== "none" && ls.strength > 0 && !node._pfLut) {
        _pfBadge(ctx, r.x + Math.floor(r.w / 2) - 55, r.y + 6,
                 node._pfLutErr ? "LUT not in preview" : "loading LUT...");
    }
}

function _pfBadge(ctx, x, y, text) {
    ctx.save();
    ctx.font = "10px sans-serif";
    ctx.textAlign = "left";
    const w = ctx.measureText(text).width + 10;
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillRect(x, y, w, 15);
    ctx.fillStyle = "#ff8c00";
    ctx.fillText(text, x + 5, y + 11);
    ctx.restore();
}

function _pfSyncCanvasSize(node) {
    // Match the canvas backing store to the box's CSS size so drawing is
    // crisp; called from draw paths, never from a resize observer.
    const cv = node._pfCanvas;
    if (!cv) return;
    const bw = Math.max(50, Math.floor(cv.clientWidth || 0));
    const bh = Math.max(50, Math.floor(cv.clientHeight || 0));
    if (bw && bh && (cv.width !== bw || cv.height !== bh)) {
        cv.width = bw;
        cv.height = bh;
    }
}

function _pfLoad(node, item) {
    if (!item || !item.filename) {
        node._pfImg = null;
        node._pfNote = "preview unavailable -- run once";
        _pfDraw(node);
        return;
    }
    const img = new Image();
    img.onload = () => {
        node._pfImg = img;
        node._pfNote = null;
        // Cache the source pixels once per load; every recompute grades
        // from this untouched copy (same-origin /view -> readable canvas).
        try {
            const sc = document.createElement("canvas");
            sc.width = img.naturalWidth;
            sc.height = img.naturalHeight;
            const sctx = sc.getContext("2d");
            sctx.drawImage(img, 0, 0);
            node._pfSrcData = sctx.getImageData(0, 0, sc.width, sc.height);
        } catch (e) {
            node._pfSrcData = null;   // preview still shows, just ungraded
        }
        node._pfGraded = null;
        // Reserve a pane height that follows the media aspect at the current
        // node width (feedback-free: computeSize only READS _pfPrevH).
        // v629: adjust the HEIGHT only and keep the user's node width --
        // setSize(computeSize()) reset the whole geometry to LiteGraph's
        // minimum on every run (the re-run collapse seen on screen).
        node._pfPrevH = _paneHeight(img.naturalWidth, img.naturalHeight,
                                    node.size ? node.size[0] : 300);
        node.setSize([node.size[0], node.computeSize()[1]]);
        _pfSyncCanvasSize(node);
        _pfDraw(node);
        _pfSchedule(node);
        node.setDirtyCanvas(true, true);
    };
    img.onerror = () => {
        node._pfImg = null;
        node._pfNote = "preview expired -- run once to refresh";
        _pfDraw(node);
    };
    img.src = viewURL(item);
}

function _pfRestore(node) {
    const item = node.properties && node.properties.ph_filter_preview;
    if (item) _pfLoad(node, item);
    else _pfDraw(node);
}

app.registerExtension({
    name: "polyhedron.filter",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "ULSFilter") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            const node = this;

            const box = document.createElement("div");
            box.style.cssText =
                "display:flex;flex-direction:column;width:100%;height:100%;" +
                `padding:${PANE_PAD}px;box-sizing:border-box;gap:4px;`;

            const header = document.createElement("div");
            header.style.cssText =
                "display:flex;align-items:center;gap:8px;flex:0 0 auto;";
            const abBtn = document.createElement("button");
            abBtn.textContent = "A/B";
            abBtn.title = "hold to show the original";
            abBtn.style.cssText =
                "font-size:11px;padding:1px 10px;background:#333;color:#ff8c00;" +
                "border:1px solid #555;border-radius:4px;cursor:pointer;" +
                "white-space:nowrap;flex:0 0 auto;";
            const saveBtn = document.createElement("button");
            saveBtn.textContent = "Save preset";
            saveBtn.title = "store the current look as a preset in the pack's presets/ folder";
            saveBtn.style.cssText = abBtn.style.cssText;
            // Row 1: the buttons. The info sign sits UNCOLORED inline at
            // the start of the tip sentence below (screen wish, final cut).
            header.appendChild(abBtn);
            header.appendChild(saveBtn);

            const tips = document.createElement("div");
            tips.style.cssText =
                "width:100%;font-size:10px;color:#888;line-height:1.35;" +
                "flex:0 0 auto;";
            const tipScrub = document.createElement("div");
            tipScrub.textContent =
                "\u{1F6C8} Click-drag a value: scrubs it live. Click once: type it.";
            const tipDivider = document.createElement("div");
            tipDivider.textContent =
                "Drag the divider \u00b7 hold A/B for the original.";
            tips.appendChild(tipScrub);
            tips.appendChild(tipDivider);

            const cv = document.createElement("canvas");
            cv.style.cssText =
                "flex:1 1 auto;width:100%;min-height:60px;border-radius:4px;" +
                "touch-action:none;cursor:ew-resize;";

            box.appendChild(header);
            box.appendChild(tips);
            box.appendChild(cv);

            node._pfCanvas = cv;
            node._pfImg = null;
            node._pfNote = null;
            node._pfFrac = 0.5;
            node._pfShowOrig = false;
            node._pfPrevH = PREVIEW_DEF_H;
            node._pfSrcData = null;
            node._pfGraded = null;
            node._pfPending = false;
            node._pfLut = null;
            node._pfLutName = "none";
            node._pfLutErr = false;

            // Live preview: every grading + LUT widget schedules a recompute
            // on change. The original callback is preserved (canon widgets
            // keep their normal behavior; this only ADDS the preview
            // reaction). lut_name changes additionally (re)fetch the file.
            for (const w of node.widgets || []) {
                if (!LIVE_WIDGETS.includes(w.name)) continue;
                const prev = w.callback;
                w.callback = function () {
                    const ret = prev ? prev.apply(this, arguments) : undefined;
                    _pfEnsureLut(node);
                    _pfSchedule(node);
                    return ret;
                };
            }

            // Reset lives as a PAINTED chip in the title bar (top right,
            // above every field -- non-canon DOM widgets must ride BELOW the
            // canon, so "above the fields" means drawing into the title and
            // hit-testing the clicks ourselves; same technique as the
            // pack's canvas UIs). Logic: _pfReset below.

            // Save preset: name via the graph canvas prompt (falls back to
            // the browser prompt), then POST to /uls/filter/preset.
            saveBtn.addEventListener("click", (ev) => {
                const doSave = (v) => { if (v && String(v).trim()) _pfSavePreset(node, String(v).trim()); };
                if (app.canvas && typeof app.canvas.prompt === "function") {
                    app.canvas.prompt("Preset name", "", doSave, ev);
                } else {
                    doSave(window.prompt("Preset name"));
                }
                ev.preventDefault();
                ev.stopPropagation();
            });

            // Preset dropdown: selecting a file loads + applies it (the
            // sliders visibly jump to the preset's values).
            {
                const w = (node.widgets || []).find((x) => x.name === "preset");
                if (w) {
                    const prev = w.callback;
                    w.callback = function (v) {
                        const ret = prev ? prev.apply(this, arguments) : undefined;
                        _pfLoadPreset(node, v !== undefined ? v : w.value);
                        return ret;
                    };
                }
            }

            const pw = node.addDOMWidget("ph_filter_preview", "div", box,
                { serialize: false, hideOnZoom: false });
            if (pw) {
                pw.serialize = false;
                pw.computeSize = (width) => [width, node._pfPrevH];
            }

            // --- pointer machinery: everything stays ON the canvas --------
            // (setPointerCapture keeps move/up delivered here; no document
            // listeners exist in this file -- the v624 leak class stays out.)
            let dragging = false;
            const fracFromEvent = (ev) => {
                const rect = cv.getBoundingClientRect();
                const img = node._pfImg;
                if (!img) return node._pfFrac;
                const r = _fitRect(img.naturalWidth, img.naturalHeight,
                                   cv.width, cv.height);
                if (r.w <= 0) return node._pfFrac;
                const x = (ev.clientX - rect.left) * (cv.width / rect.width);
                return _clampFrac((x - r.x) / r.w);
            };
            cv.addEventListener("pointerdown", (ev) => {
                dragging = true;
                cv.setPointerCapture(ev.pointerId);
                node._pfFrac = fracFromEvent(ev);
                _pfDraw(node);
                ev.preventDefault();
                ev.stopPropagation();
            });
            cv.addEventListener("pointermove", (ev) => {
                if (!dragging) return;
                node._pfFrac = fracFromEvent(ev);
                _pfDraw(node);
            });
            const endDrag = (ev) => {
                if (!dragging) return;
                dragging = false;
                try { cv.releasePointerCapture(ev.pointerId); } catch (e) { /* released */ }
            };
            cv.addEventListener("pointerup", endDrag);
            cv.addEventListener("pointercancel", endDrag);

            abBtn.addEventListener("pointerdown", (ev) => {
                node._pfShowOrig = true;
                abBtn.setPointerCapture(ev.pointerId);
                _pfDraw(node);
                ev.preventDefault();
                ev.stopPropagation();
            });
            const abUp = (ev) => {
                if (!node._pfShowOrig) return;
                node._pfShowOrig = false;
                try { abBtn.releasePointerCapture(ev.pointerId); } catch (e) { /* released */ }
                _pfDraw(node);
            };
            abBtn.addEventListener("pointerup", abUp);
            abBtn.addEventListener("pointercancel", abUp);

            // Late init (house rule: the setTimeout(0) of onNodeCreated runs
            // reliably in this frontend) -- restore a persisted preview and
            // size the canvas once the DOM is attached.
            setTimeout(() => {
                _pfSyncCanvasSize(node);
                _pfEnsureLut(node);
                _pfRestore(node);
            }, 0);
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const item = message?.ph_filter?.[0];
            if (!item) return;
            if (item.filename) {
                this.properties = this.properties || {};
                this.properties.ph_filter_preview = item;
            }
            _pfLoad(this, item);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            onConfigure?.apply(this, arguments);
            // Fires WITH info in this frontend (measured on the CLIP node's
            // height persistence); the setTimeout(0) restore above is the
            // belt for the path where it would not.
            const node = this;
            setTimeout(() => {
                _pfSyncCanvasSize(node);
                _pfEnsureLut(node);
                _pfRestore(node);
            }, 0);
        };

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            onDrawForeground?.apply(this, arguments);
            if (this.flags?.collapsed) return;
            _pfDrawTitleControls(this, ctx);
            // Keep the canvas backing store in step with its CSS box; cheap
            // no-op when nothing changed (draw-path sizing, no observers).
            const cv = this._pfCanvas;
            if (cv && (cv.width !== Math.floor(cv.clientWidth || 0) ||
                       cv.height !== Math.floor(cv.clientHeight || 0))) {
                _pfSyncCanvasSize(this);
                _pfDraw(this);
            }
        };

        const onMouseDown = nodeType.prototype.onMouseDown;
        nodeType.prototype.onMouseDown = function (e, pos, canvas) {
            const hits = this._pfTitleHits;
            if (hits && !this.flags?.collapsed) {
                if (_pfHit(hits.reset, pos)) {
                    _pfReset(this);
                    return true;   // consumed: no node drag from the chip
                }
            }
            return onMouseDown ? onMouseDown.apply(this, arguments) : undefined;
        };
    },
});
