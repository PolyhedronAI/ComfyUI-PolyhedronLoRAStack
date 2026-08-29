/*
 * ph_noise_field.js -- v690
 *
 * A LOCAL, synchronous noise-field renderer. One job: draw something the
 * instant the mouse moves.
 *
 * WHY THIS EXISTS
 * ---------------
 * The seed node's preview shows the REAL field, fetched from
 * /uls/noise/preview (v687). That is the right default -- it is what the
 * sampler will denoise. But a fetch cannot keep up with a scrub: the request
 * is debounced by 90ms and the debounce RESTARTS on every mouse move, so a
 * continuous drag fired no request at all and the box sat on
 * "building field..." for the whole gesture. Dragging the canvas to hunt for
 * a variant -- the reason the control exists -- had stopped working.
 *
 * So during a drag we draw this instead: a per-pixel field computed in the
 * browser from position hashes. It is an IMPRESSION of the noise character,
 * NOT the tensor the run will use. The two are the same family, different
 * bytes, and no amount of tuning would make them agree -- the real one is
 * torch.randn plus a frequency filter in python. The readout says which of
 * the two is on screen; see ph_seed.js. Never let that label drift.
 *
 * Ported from the pre-v685 Empty Latent preview (last full copy: v528
 * ph_empty_latent.js), where the same code made the drag feel like a dial.
 * The Empty Latent no longer has a preview at all; nothing there was touched.
 *
 * RESOLUTION-STABLE: every sample is a pure function of its integer pixel
 * position plus the seed, so the field is rendered 1:1 at the latent grid and
 * does not shimmer when the node is resized.
 */

// Integer position hash -> [0,1). Deterministic per (x, y, seed).
function _hash2(x, y, seed) {
    let h = (Math.imul(x | 0, 374761393) + Math.imul(y | 0, 668265263) + Math.imul(seed | 0, 2246822519)) >>> 0;
    h = Math.imul(h ^ (h >>> 13), 1274126177) >>> 0;
    h ^= h >>> 16;
    return (h >>> 0) / 4294967296;
}

// Standard normal N(0,1) at a pixel via Box-Muller on two independent hashes.
function _normal(x, y, seed) {
    const u1 = Math.max(1e-7, _hash2(x, y, seed));
    const u2 = _hash2(x, y, (seed ^ 0x68bc21eb) | 0);
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(6.28318530718 * u2);
}

// Smooth value noise with a fixed pixel period (smoothstep-interpolated lattice).
function _valueNoise(x, y, period, seed) {
    const p = period > 1 ? period : 1;
    const gx = Math.floor(x / p), gy = Math.floor(y / p);
    let fx = x / p - gx, fy = y / p - gy;
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy);
    const v00 = _hash2(gx, gy, seed), v10 = _hash2(gx + 1, gy, seed);
    const v01 = _hash2(gx, gy + 1, seed), v11 = _hash2(gx + 1, gy + 1, seed);
    const a = v00 + (v10 - v00) * fx, b = v01 + (v11 - v01) * fx;
    return a + (b - a) * fy;
}

// The seed widget holds a 53-bit value; the hashes are 32-bit. Folding the
// high bits in keeps distant seeds distinct -- a bare `| 0` would map every
// multiple of 2^32 onto the same field, which is exactly the class of bug
// v689 fixed on the fetch path. Scrubbing by +-1 must always change the
// picture, or the drag looks broken even though it works.
function _seed32(seed) {
    const s = Math.max(0, Math.floor(Number(seed) || 0));
    const lo = s >>> 0;
    const hi = Math.floor(s / 4294967296) >>> 0;
    return (lo ^ Math.imul(hi, 2654435761)) | 0;
}

// Render an NW x NH grayscale field for the given noise type + seed (0..1).
// White types stay crisp per pixel; the colored ones are value-noise octaves.
function _renderField(type, seed, NW, NH) {
    const s = _seed32(seed);
    const n = NW * NH;
    const f = new Float32Array(n);
    if (type === "gaussian") {
        // true normal distribution -> clusters around mid-grey
        for (let y = 0; y < NH; y++) for (let x = 0; x < NW; x++)
            f[y * NW + x] = 0.5 + _normal(x, y, s) / 6;
    } else if (type === "blue") {
        // high-pass: per-pixel white minus its 3x3 mean, recentred
        const w = new Float32Array(n);
        for (let y = 0; y < NH; y++) for (let x = 0; x < NW; x++) w[y * NW + x] = _hash2(x, y, s);
        for (let y = 0; y < NH; y++) for (let x = 0; x < NW; x++) {
            let sum = 0, c = 0;
            for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
                const yy = y + dy, xx = x + dx;
                if (yy >= 0 && yy < NH && xx >= 0 && xx < NW) { sum += w[yy * NW + xx]; c++; }
            }
            f[y * NW + x] = 0.5 + (w[y * NW + x] - sum / c);
        }
    } else if (type === "brown") {
        // low frequency: one large-period smooth field (soft cloudy blobs)
        for (let y = 0; y < NH; y++) for (let x = 0; x < NW; x++) f[y * NW + x] = _valueNoise(x, y, 48, s);
    } else if (type === "pink") {
        // 1/f: a few octaves, amplitude falling faster than fractal
        for (let y = 0; y < NH; y++) for (let x = 0; x < NW; x++) {
            let v = 0, amp = 1, tot = 0, p = 48;
            for (let o = 0; o < 4; o++) { v += amp * _valueNoise(x, y, p, (s + o * 101) | 0); tot += amp; amp *= 0.6; p = p > 2 ? (p >> 1) : 2; }
            f[y * NW + x] = v / tot;
        }
    } else if (type === "fractal") {
        // multi-octave value noise (fBm), amplitude halving per octave
        for (let y = 0; y < NH; y++) for (let x = 0; x < NW; x++) {
            let v = 0, amp = 1, tot = 0, p = 64;
            for (let o = 0; o < 6; o++) { v += amp * _valueNoise(x, y, p, (s + o * 263) | 0); tot += amp; amp *= 0.5; p = p > 2 ? (p >> 1) : 2; }
            f[y * NW + x] = v / tot;
        }
    } else if (type === "offset") {
        // v833: spatially this IS white -- the DC term is per CHANNEL and a
        // single grayscale field cannot show it (display normalisation would
        // cancel a constant anyway). Honest impression: the gaussian field.
        for (let y = 0; y < NH; y++) for (let x = 0; x < NW; x++)
            f[y * NW + x] = 0.5 + _normal(x, y, s) / 6;
    } else if (type === "pyramid") {
        // v833: full-res white BASE plus decaying smooth octaves (0.7) --
        // near-white with a mild low-frequency lift, unlike fractal's clouds.
        for (let y = 0; y < NH; y++) for (let x = 0; x < NW; x++) {
            let v = _normal(x, y, s) / 3 + 0.5, amp = 1, tot = 1, pp = 32;
            for (let o = 1; o <= 4; o++) { amp *= 0.7; v += amp * (_valueNoise(x, y, pp, (s + o * 173) | 0) - 0.5); tot += amp; pp = pp > 2 ? (pp >> 1) : 2; }
            f[y * NW + x] = 0.5 + (v - 0.5) / tot;
        }
    } else {   // zeros and anything unknown
        for (let i = 0; i < n; i++) f[i] = 0.5;
    }
    // Normalise to full 0..1 for display contrast, exactly as the fetched
    // preview does server-side -- otherwise the handover on release would
    // jump in brightness.
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < n; i++) { if (f[i] < mn) mn = f[i]; if (f[i] > mx) mx = f[i]; }
    const rng = (mx - mn) || 1;
    for (let i = 0; i < n; i++) f[i] = (f[i] - mn) / rng;
    return f;
}

// Grid guard: the latent grid is what we draw, but a pathological wired size
// must not turn a mouse move into a million-pixel loop.
const GRID_MAX = 256;
const GRID_MIN = 8;

function _clampGrid(v) {
    const n = Math.round(Number(v) || GRID_MIN);
    return Math.max(GRID_MIN, Math.min(GRID_MAX, n));
}

/**
 * Build (and cache) an offscreen canvas of the impression field.
 *
 * `store` is any object that may hold the cache across calls -- the preview
 * widget itself. It is passed IN rather than reached through `this`: the
 * caller lives inside a LiteGraph widget, where `this` is not guaranteed to
 * be the widget (that mix-up is what stuck the drag in v687/v688).
 *
 * Returns a canvas of exactly gridW x gridH pixels, to be drawn with
 * imageSmoothingEnabled = false so the latent grid stays crisp.
 */
export function impressionCanvas(store, type, seed, gridW, gridH, character) {
    const NW = _clampGrid(gridW), NH = _clampGrid(gridH);
    const t = String(type || "gaussian");
    // v833: the character dial, clamped like the backend clamps it.
    let ch = Number(character);
    if (!isFinite(ch)) ch = 1.0;
    ch = Math.max(0, Math.min(1, ch));
    const key = t + ":" + Math.max(0, Math.floor(Number(seed) || 0)) + ":" + NW + "x" + NH + ":" + ch.toFixed(2);
    if (store && store._impKey === key && store._impCanvas) return store._impCanvas;
    const cv = (store && store._impCanvas) || document.createElement("canvas");
    cv.width = NW; cv.height = NH;
    const cx = cv.getContext("2d");
    const img = cx.createImageData(NW, NH);
    if (t === "zeros") {
        for (let i = 0; i < NW * NH; i++) {
            img.data[i * 4] = 24; img.data[i * 4 + 1] = 26; img.data[i * 4 + 2] = 30;
            img.data[i * 4 + 3] = 255;
        }
    } else {
        let f = _renderField(t, seed, NW, NH);
        // v833: cross-fade the impression with white below character 1.0 --
        // same recipe family as the backend (sqrt keeps the variance flat).
        // offset scales its DC instead, which one grayscale channel cannot
        // show; gaussian ignores the dial -- both skip the mix like the
        // backend does.
        if (ch < 1.0 && t !== "gaussian" && t !== "offset") {
            const w = _renderField("gaussian", seed, NW, NH);
            const a = Math.sqrt(1 - ch * ch);
            const g2 = new Float32Array(NW * NH);
            for (let i = 0; i < NW * NH; i++) g2[i] = a * (w[i] - 0.5) + ch * (f[i] - 0.5) + 0.5;
            let mn = Infinity, mx = -Infinity;
            for (let i = 0; i < NW * NH; i++) { if (g2[i] < mn) mn = g2[i]; if (g2[i] > mx) mx = g2[i]; }
            const rng = (mx - mn) || 1;
            for (let i = 0; i < NW * NH; i++) g2[i] = (g2[i] - mn) / rng;
            f = g2;
        }
        for (let i = 0; i < NW * NH; i++) {
            const v = Math.max(0, Math.min(255, Math.round(f[i] * 255)));
            img.data[i * 4] = v; img.data[i * 4 + 1] = v; img.data[i * 4 + 2] = v;
            img.data[i * 4 + 3] = 255;
        }
    }
    cx.putImageData(img, 0, 0);
    if (store) { store._impCanvas = cv; store._impKey = key; }
    return cv;
}
