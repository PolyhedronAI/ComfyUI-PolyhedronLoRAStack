/**
 * ph_clip_encode.js  (v556)
 *
 * The node's own eyes: segment visibility, a live word/char counter while you
 * type, and a read-only preview of the FINAL composed prompt (including any
 * external text pulled in from Florence2 & friends).
 *
 * Honest split, by design:
 *   - words / chars  : counted HERE, instantly, on every keystroke.
 *   - tokens         : counted in the BACKEND with the very tokenizer behind
 *                      ULSTokenCounter, and shipped back on the pls_cte ui
 *                      channel after each run. The browser has no tokenizer,
 *                      so it never pretends to have one.
 *
 * Preview mechanics follow the proven house laws: DOM widget with
 * serialize:false + hideOnZoom:false (v542), stored-height computeSize with a
 * height-only setSize (v531 - a run must never shrink the width).
 */
import { app } from "../../scripts/app.js";

console.info("[PLS] ph_clip_encode.js v560 loaded");

const NODE_TYPE = "ULSCLIPTextEncode";
const MAX_SEGMENTS = 6;
// v600: the pane WAS a 84-420px wall of composed prompt. It is a status bar
// now -- the text itself left through the new `full_text` output. The node
// exists to EDIT pos/neg; a preview that shoves the editing fields off screen
// has inverted its own purpose.
//
// v715: the band is no longer a CONSTANT height, and its height is no longer
// reserved by _refit alone. Two field findings, one root cause each:
//
//  (a) THE BAND COULD BE PUSHED UNDER THE neg_1 FIELD. The reservation lived
//      ONLY in _refit (total = frame + BAR_H + fields) while
//      nodeType.prototype.computeSize was NOT overridden -- so LiteGraph itself
//      did not know the band existed. Any height change that does not run
//      through _refit, above all Frank dragging the node's bottom edge, set
//      size[1] freely and size[1] - BAR_H then landed inside the last field.
//      neg_1 is a real DOM textarea above the canvas, so it simply covered the
//      band. ph_reference.js has no such bug because it adds its band height
//      inside computeSize (line 199ff) -- LiteGraph's own minimum then accounts
//      for it on EVERY path. That is the pattern adopted here.
//
//  (b) THE TEXT RAN OFF THE RIGHT EDGE. It was one fillText with no width
//      measurement at all. It now WRAPS at its own " . " separators, which is
//      what Frank asked for -- better than ph_reference's shrink-to-7px, which
//      stops being readable.
//
// The rule learned the hard way in the Mask Editor (v707-v714) is what shapes
// this: the reservation and the painting must come from ONE function, never
// from two numbers that are supposed to agree. _barLines() is that function.
// Declared HERE, above _barLines, not next to the paint block further down --
// _barLines reads them, and a const used before its declaration only survives
// because nothing calls the function during module evaluation. Not a trap worth
// leaving lying around.
const BAR_PAD_X = 10;          // horizontal padding inside the footer box
const BAR_FONT = "12px monospace";
const BAR_ROW_H = 15;          // one text row inside the band
const BAR_PAD_Y = 6;           // air above and below the rows
const BAR_SEP = " \u00b7 ";    // the counter's own separator = the break point

// One offscreen context for text measurement. Created once, never attached.
let _measCtx;
function _meas() {
    if (_measCtx !== undefined) return _measCtx;
    try {
        _measCtx = document.createElement("canvas").getContext("2d") || null;
    } catch (e) { _measCtx = null; }
    return _measCtx;
}

// The counter text broken into rows that fit the node's width. Cached per
// (text, width) because computeSize is called often and this is the only
// place the row count is decided.
function _barLines(node) {
    const txt = _counterText(node);
    const w = (node && node.size) ? Math.round(node.size[0]) : 300;
    const key = w + "|" + txt;
    if (node._cteBarKey === key && node._cteBarLines) return node._cteBarLines;
    const avail = Math.max(20, w - BAR_PAD_X * 2);
    const ctx = _meas();
    let lines;
    if (!ctx) {
        lines = [txt];                       // no measurement possible -- one row
    } else {
        ctx.font = BAR_FONT;
        if (ctx.measureText(txt).width <= avail) {
            lines = [txt];
        } else {
            lines = [];
            let cur = "";
            for (const part of txt.split(BAR_SEP)) {
                const cand = cur ? cur + BAR_SEP + part : part;
                if (cur && ctx.measureText(cand).width > avail) {
                    lines.push(cur);
                    cur = part;
                } else {
                    cur = cand;
                }
            }
            if (cur) lines.push(cur);
        }
    }
    node._cteBarKey = key;
    node._cteBarLines = lines;
    return lines;
}

// THE band height. Everything -- computeSize, _refit and the paint -- asks
// this, so the reserved strip and the painted strip cannot disagree.
function _barHeight(node) {
    return _barLines(node).length * BAR_ROW_H + BAR_PAD_Y * 2;
}

function _w(node, name) {
    return (node.widgets || []).find((w) => w.name === name);
}

/**
 * v557 - hide a widget WITHOUT dropping its value (it stays serialised).
 *
 * The v556 version set `computeSize` + `hidden` but NOT `type`, and LiteGraph
 * skips a widget in the LAYOUT pass by its TYPE marker. The canvas widgets
 * then closed the gap while the textarea (a DOM element, positioned from the
 * layout) stayed where it was -> the overlap Frank measured. Second bug: on
 * show, v556 wrote `computeSize = undefined` back whenever the original had
 * none, which left the widget zero-height FOREVER. Both fixed here; the
 * marker prefix is the community-proven mechanism.
 */
const HIDDEN_PREFIX = "pls-hidden-";

function _hide(w, hidden) {
    if (!w) return;
    const isHidden = String(w.type || "").startsWith(HIDDEN_PREFIX);
    if (hidden === isHidden) return;   // idempotent - never stack the swaps
    if (hidden) {
        w._pls_type = w.type;
        w._pls_hadCS = Object.prototype.hasOwnProperty.call(w, "computeSize");
        w._pls_cs = w.computeSize;
        w.type = HIDDEN_PREFIX + w.type;
        w.computeSize = () => [0, -4];
        w.hidden = true;
        if (w.element) { w.element.style.display = "none"; w.element.hidden = true; }
    } else {
        w.type = w._pls_type !== undefined
            ? w._pls_type : String(w.type).slice(HIDDEN_PREFIX.length);
        if (w._pls_hadCS) w.computeSize = w._pls_cs;
        else delete w.computeSize;   // v556 bug: NEVER assign undefined back
        w.hidden = false;
        if (w.element) { w.element.style.display = ""; w.element.hidden = false; }
    }
}

// v560: the classic ComfyUI conditioning colours - positive green, negative
// the brown/maroon tone. Applied to the TEXTAREAS (a DOM widget can be styled
// directly), so one node still reads as two clearly separate intents.
const POS_TINT = "#2f4f2f", POS_EDGE = "#4a7a4a";
const NEG_TINT = "#4a2f2f", NEG_EDGE = "#7a4a4a";

function _tint(w, bg, edge) {
    if (!w || !w.element) return;
    const el = w.element;
    try {
        el.style.backgroundColor = bg;
        el.style.borderLeft = "4px solid " + edge;
        el.style.borderRadius = "4px";
        el.style.resize = "none";   // fields auto-fit; the browser resize grip both fights
                                    // _refit and is the faint dashed mark at the field foot
    } catch (e) { /* styling must never break the node */ }
}

function _applyTints(node) {
    for (let i = 1; i <= MAX_SEGMENTS; i++) {
        _tint(_w(node, `pos_${i}`), POS_TINT, POS_EDGE);
    }
    _tint(_w(node, "neg_1"), NEG_TINT, NEG_EDGE);
}

function _applyVisibility(node) {
    const n = Math.max(1, Math.min(Number(_w(node, "segments")?.value) || 1,
                                   MAX_SEGMENTS));
    for (let i = 1; i <= MAX_SEGMENTS; i++) _hide(_w(node, `pos_${i}`), i > n);
    _hide(_w(node, "neg_1"), !_w(node, "use_negative")?.value);
    // Let the DOM settle BEFORE measuring - a textarea reports its height only
    // after the browser laid it out; measuring too early is what produced the
    // squeezed node. Height only (v531): a re-layout must never shrink the width.
    requestAnimationFrame(() => {
        try {
            _refit(node);                 // fit the node to the now-visible fields
            const fit = node.size[1];
            node.setDirtyCanvas(true, true);
            if (!node._plsVisLogged) {
                node._plsVisLogged = true;
                console.info(`[PLS] CTE layout: ${n} segment(s), neg ` +
                             `${_w(node, "use_negative")?.value ? "on" : "off"}` +
                             ` -> fit ${Math.round(fit)}px`);
            }
        } catch (e) { /* never break the ui */ }
    });
}

// v557: serialisation heal - a v556 save lacks comment_markers (13 -> 14).
const LEN_PRE_V557 = 13;
function _healPreV557(wv) {
    if (!Array.isArray(wv) || wv.length !== LEN_PRE_V557) return wv;
    const out = wv.slice();
    out.push("//");   // comment_markers -> 13 (the historic default)
    return out;
}


/**
 * v604 -- CANON vs DISPLAY. The law the tree already knew, and I broke anyway.
 *
 * v585 (measured 2026-07-13, the hard way): the live frontend serialises
 * widgets_values in WIDGET ORDER. v584 moved one widget into the display middle
 * and every saved graph loaded shifted by a slot -- the seed landed in cfg_low.
 * The law that came out of it: **the canon is APPEND-ONLY**. Re-sorting is legal
 * only once the SAVE PATH is normalised through the canon mapping.
 *
 * v603 re-sorted INPUT_TYPES itself, tried to catch the fallout with a heal, and
 * shipped. Frank's node came back with his prompt inside `separator` and the
 * string "true" in a prompt box. The heal was correct, ran green in its guard,
 * and was fighting the wrong enemy: it patched the SYMPTOM (values landing in the
 * wrong slot) instead of the CAUSE (the canon had moved out from under them).
 *
 * v604 does what ph_power_upscale learned to do at v546/v588:
 *
 *   PYTHON  = CANON. Never re-ordered. What is on disk, forever.
 *   JS      = DISPLAY. Permuted freely -- the filters sit above the prompts.
 *   load    : canon -> display   (configure)
 *   save    : display -> canon   (onSerialize)
 *
 * The file on disk never learns that the display moved. Every graph Frank ever
 * saved keeps loading, because nothing it depends on has changed.
 */
const CANON = ["segments", "pos_1", "pos_2", "pos_3", "pos_4", "pos_5", "pos_6",
               "use_negative", "neg_1", "separator", "strip_comments",
               "strip_newlines", "external_mode", "comment_markers"];
const DISPLAY = ["external_mode", "strip_comments", "strip_newlines", "separator",
                 "comment_markers", "segments",
                 "pos_1", "pos_2", "pos_3", "pos_4", "pos_5", "pos_6",
                 "use_negative", "neg_1"];

const C2D = DISPLAY.map((n) => CANON.indexOf(n));   // display[i] = canon[C2D[i]]
const D2C = CANON.map((n) => DISPLAY.indexOf(n));   // canon[i]   = display[D2C[i]]

function _canonToDisplay(wv) {
    if (!Array.isArray(wv) || wv.length !== CANON.length) return wv;
    return C2D.map((j) => wv[j]);
}
function _displayToCanon(wv) {
    if (!Array.isArray(wv) || wv.length !== DISPLAY.length) return wv;
    return D2C.map((j) => wv[j]);
}

/** Permute the widget ROW ORDER on screen. The values are untouched; only the
 *  order they are drawn in changes. Serialisation is corrected separately -- see
 *  onSerialize -- which is the entire difference between this and v603. */
/**
 * v606 -- configure() IS NOT CALLED, AND I BUILT ON IT THREE TIMES.
 *
 * PROOF, from Frank's screen: the rows ARE permuted (the filters sit on top --
 * onNodeCreated fired) and the VALUES are not (his prompt is in `separator` --
 * configure() did not). Two hooks, one file, one fires. That was already my
 * suspicion in v603. I never chased it down, and I shipped on top of it three
 * times.
 *
 * So: STOP DEPENDING ON IT.
 *
 * The rows are permuted AFTER the load, in the onConfigure timeout that is PROVEN
 * to run (it is the one that folds the segments and paints the tints -- both of
 * which work). While LiteGraph is pouring widgets_values into node.widgets, the
 * widgets are still in CANON order, so the values land where they belong. Nothing
 * needs mapping. Nothing CAN go wrong there, because nothing happens there.
 *
 * And serialize() swings the rows back to canon for the length of one call. It has
 * to be serialize and not onSerialize: onSerialize is a callback the base method
 * chooses to invoke, and I have now watched this frontend decline to invoke a base
 * method. serialize() is the method itself -- there is no saving a graph without it.
 */
function _canonOrder(node) {
    if (!node.widgets || !node._plsDisplayed) return;
    const by = new Map(node.widgets.map((w) => [w.name, w]));
    const rows = CANON.map((n) => by.get(n)).filter(Boolean);
    if (rows.length !== CANON.length) return;
    const extras = node.widgets.filter((w) => !CANON.includes(w.name));
    node.widgets = rows.concat(extras);
    node._plsDisplayed = false;
}

/**
 * THE RESCUE, and it reads the VALUES rather than the file.
 *
 * A graph poisoned by v603/v604/v605 has display order on disk. A healthy one has
 * canon. I cannot tell them apart by asking configure() -- configure() does not
 * answer. So ask the WIDGETS what they are holding, after the load, and let the
 * values say what they are:
 *
 *   `separator` must be one of four words. `strip_comments` must be a boolean.
 *   `external_mode` must be one of three words. A prompt is none of those things.
 *
 * If the filters are holding prose, the array came in shifted, and the shift is
 * exactly the canon->display permutation. Undo it. If they are holding what they
 * should, do NOTHING -- a rescue that cannot tell the wounded from the well is
 * just a second wound.
 */
const SEPARATORS = ["comma", "newline", "space", "none"];
const MODES = ["append", "prepend", "replace"];

function _looksScrambled(node) {
    const v = (n) => _w(node, n)?.value;
    const sepOK = SEPARATORS.includes(v("separator"));
    const modeOK = MODES.includes(v("external_mode"));
    const scOK = typeof v("strip_comments") === "boolean";
    const snOK = typeof v("strip_newlines") === "boolean";
    // Two independent witnesses must BOTH say "wrong" before we touch a thing.
    return (!sepOK && !modeOK) || (!sepOK && !scOK) || (!modeOK && !snOK);
}

function _rescueScrambled(node) {
    if (!node.widgets || !_looksScrambled(node)) return false;
    // The widgets currently hold the CANON array, poured onto DISPLAY rows (or the
    // reverse). Either way the fix is the same permutation, applied to the values.
    const cur = DISPLAY.map((n) => _w(node, n)?.value);
    const fixed = _canonToDisplay(cur);
    DISPLAY.forEach((n, i) => { const w = _w(node, n); if (w) w.value = fixed[i]; });
    console.warn("[PLS] CTE: the widget values came in shifted (configure() did not run in this "
                 + "frontend). Values re-seated by name. Save once and it stays fixed.");
    return true;
}

function _reorderWidgetsToDisplay(node) {
    if (!node.widgets || node._plsDisplayed) return;
    const by = new Map(node.widgets.map((w) => [w.name, w]));
    const rows = DISPLAY.map((n) => by.get(n)).filter(Boolean);
    if (rows.length !== DISPLAY.length) return;   // a widget is missing - do not guess

    // Anything not IN the canon keeps its own order and rides at the BACK. Since v613
    // the wordcount is PAINTED (onDrawForeground), not a DOM widget, so today `extras`
    // is empty and node.widgets == the 14 canon rows in display order. The concat is
    // kept anyway: it is the safe, general rule (widgets_values loads BY POSITION against
    // the canon, so any future non-canon widget must ride last, not shift a value).
    const extras = node.widgets.filter((w) => !CANON.includes(w.name));
    node.widgets = rows.concat(extras);
    node._plsDisplayed = true;
}


// ==========================================================================
// v613 -- AUTO-FIT TO CONTENT, WITH A STOP. Each visible prompt field grows to show
// ALL its text and STOPS there -- no whitespace past the text, no manual dragging.
//
// This is what Frank asked for from the start. v610 had the right idea but two loop
// bugs: it measured "chrome" via node.computeSize() (which INCLUDED the growing field
// heights, so the target inflated every pass), and a ResizeObserver on the bar re-fired
// setSize. v613 has NEITHER: chrome is derived by SUBTRACTING the current field heights
// from computeSize (self-consistent, cannot accumulate), there is NO ResizeObserver and
// NO onResize hook, a re-entry guard makes doubly sure, and _refit runs only from real
// events (typing, segment/neg change, load). One setSize per refit, no feedback.
// ==========================================================================
const FIELD_MIN_H = 48;     // a field is at least this tall (~2 lines) even when empty
const FIELD_MAX_H = 2000;   // a monstrous paste scrolls past this instead of a mile-high node
const FIELD_DEF_H = 64;     // height before the first measurement

const FIELD_NAMES = ["pos_1", "pos_2", "pos_3", "pos_4", "pos_5", "pos_6", "neg_1"];

function _visibleFields(node) {
    return FIELD_NAMES
        .map((nm) => _w(node, nm))
        .filter((w) => w && w.element && !w.hidden
                       && !String(w.type || "").startsWith(HIDDEN_PREFIX));
}

// the height this field needs to show ALL its text, measured from scrollHeight. The
// height='auto' toggle is synchronous and fires no event, so it cannot start a loop.
function _contentH(w) {
    const el = w && w.element;
    if (!el) return FIELD_DEF_H;
    const prev = el.style.height;
    el.style.height = "auto";
    const raw = Math.ceil(el.scrollHeight);
    el.style.height = prev;
    return Math.max(FIELD_MIN_H, Math.min(FIELD_MAX_H, raw));
}

// THE refit. Size every visible field to its content and the node to their sum -- so
// the text is fully shown and growth STOPS exactly there. No feedback: chrome is
// computeSize MINUS the current field heights (stable, cannot accumulate), one setSize,
// re-entry guarded.
function _refit(node) {
    if (node._plsRefitting) return;
    node._plsRefitting = true;
    try {
        const fields = _visibleFields(node);
        if (!fields.length) return;
        // the LAST visible positive field carries a little air below it (NEG_GAP), so the
        // use_negative toggle does not butt against the prompt box. It rides on the field's
        // reserved height -- provably honoured (v613 auto-fit) -- so the air lands above the
        // toggle whatever the frontend does with the toggle's own row.
        let lastPos = null;
        for (const w of fields) if (w.name && w.name.indexOf("pos_") === 0) lastPos = w;
        let cur = 0;
        for (const w of fields) cur += (typeof w._plsH === "number" ? w._plsH : FIELD_DEF_H);
        const frame = Math.max(0, node.computeSize()[1] - cur);   // node minus the fields
        // v715: NO "+ BAR_H" here any more. computeSize now includes the band
        // (see the override below), so `frame` already carries it -- adding it
        // again would grow the node by one band on EVERY refit, and refits are
        // frequent. This is the one line where the v715 change can go wrong,
        // and the guard drives _refit twice to prove the height is stable.
        let total = frame;
        for (const w of fields) {
            const el = w.element;
            const pad = (w === lastPos) ? NEG_GAP : 0;
            // v622: the stored _plsH is the source of truth. Trust a fresh measurement ONLY
            // when the field is actually laid out (a real offsetParent AND a non-zero width).
            // An unlaid-out textarea -- exactly the RUN/RELOAD-before-layout moment -- reports
            // a collapsed scrollHeight; measuring then overwrites the good height and squeezes
            // the field (the v619 regression that v620's defer alone did not fully close). When
            // the field is not laid out, KEEP _plsH instead of re-measuring it.
            const laidOut = !!el && (el.offsetParent !== null) && (el.clientWidth > 0);
            let h;
            if (laidOut) {
                h = _contentH(w);              // laid out -- measure and update the stored truth
                w._plsH = h + pad;
            } else {
                h = (typeof w._plsH === "number" ? w._plsH : FIELD_DEF_H) - pad;  // keep stored
            }
            if (el) {
                el.style.height = h + "px";
                el.style.overflowY = (h >= FIELD_MAX_H) ? "auto" : "hidden";
            }
            total += h + pad;
        }
        if (Math.abs(node.size[1] - total) > 1) {
            node.setSize([node.size[0], total]);   // height only (v531)
        }
        node.setDirtyCanvas(true, true);
    } finally {
        node._plsRefitting = false;
    }
}

// v620: defer a refit to AFTER the browser lays the textareas out. A textarea reports its
// height (scrollHeight) only once laid out -- the v560 _applyVisibility rule, pinned by
// test_v557. A refit fired SYNCHRONOUSLY from a PROGRAMMATIC path (load / run / wiring change)
// runs before that layout, measures the field as empty, and collapses it (the v619 regression:
// _syncExternal + both onConfigure hooks measured too early, so every RUN and RELOAD squeezed
// the fields; a live INPUT event escaped it because the element is laid out when the user types).
// A live input event may still refit inline; only the programmatic callers defer through here.
// No loop: rAF is a one-shot schedule (not a ResizeObserver), and _refit's _plsRefitting
// re-entry guard still holds.
function _refitNextFrame(node) {
    const raf = (typeof requestAnimationFrame === "function")
        ? requestAnimationFrame
        : (fn) => setTimeout(fn, 0);
    raf(() => { try { _refit(node); } catch (e) { /* never break the ui */ } });
}

// give a field its computeSize so LiteGraph reserves exactly the fitted height. Set
// once, in onNodeCreated, BEFORE the first _applyVisibility (which snapshots computeSize
// via _hide's hasOwnProperty). No styling beyond the tint -- the field stays native.
// v623: capture/restore of fitted field heights across a save+reload. _plsH is v622's
// runtime "remembered height"; serialize() stashes it in the workflow via _capture, and
// onConfigure re-applies it via _restore. A browser reload (F5) wipes all JS state, so the
// runtime cache alone cannot survive one -- the height has to live in the saved file, the
// only thing a reload keeps. Both are named top-level fns so the guard can drive them
// directly (test_v623_persist_height).
function _captureFieldHeights(node) {
    const h = {};
    for (let i = 1; i <= MAX_SEGMENTS; i++) {
        const w = _w(node, `pos_${i}`);
        if (w && typeof w._plsH === "number") h[`pos_${i}`] = w._plsH;
    }
    const nw = _w(node, "neg_1");
    if (nw && typeof nw._plsH === "number") h["neg_1"] = nw._plsH;
    return h;
}

// Re-apply saved heights. Unconditional assignment: _wireField may have reset _plsH to
// FIELD_DEF_H already this load, and its own `typeof !== "number"` guard then leaves the
// number we write here alone. Mutating `if (!h) return;` proves this is load-bearing.
function _restoreFieldHeights(node, info) {
    const h = info && info.pls_field_heights;
    if (!h) return;
    for (let i = 1; i <= MAX_SEGMENTS; i++) {
        const w = _w(node, `pos_${i}`);
        if (w && typeof h[`pos_${i}`] === "number") w._plsH = h[`pos_${i}`];
    }
    const nw = _w(node, "neg_1");
    if (nw && typeof h["neg_1"] === "number") nw._plsH = h["neg_1"];
}

function _wireField(w) {
    if (!w || w._plsWired) return;
    w._plsWired = true;
    if (typeof w._plsH !== "number") w._plsH = FIELD_DEF_H;
    w.computeSize = (width) => [width, (typeof w._plsH === "number" ? w._plsH : FIELD_DEF_H)];
}

// ---------------------------------------------------------------------------
// v619: EXT_POS / EXT_NEG -- read-only display fields for the RESOLVED external text
// (pos_external / neg_external). NOT canon, never serialized -- they ride last and are
// filled from the backend after a run. Shown only when the matching input PIN is wired.
// FIXED height with scroll (NOT auto-fit) so a 400-word caption can never push the editing
// fields off screen (the v600 lesson). One robust path: static, Join Strings, or Florence2
// all resolve on the backend, which sends the text back over pls_cte.
// ---------------------------------------------------------------------------
const EXT_H = 96;   // fixed height of an external display field (~5 lines, scroll past)
const EXT_FIELDS = [["pls_ext_pos", "pos_external", "pos", POS_TINT, POS_EDGE],
                    ["pls_ext_neg", "neg_external", "neg", NEG_TINT, NEG_EDGE]];

function _makeExtField(node, name, bg, edge) {
    if (_w(node, name)) return;                     // idempotent
    if (typeof node.addDOMWidget !== "function") return;
    let w;
    try {
        const el = document.createElement("textarea");
        el.readOnly = true;                         // read-only: a display, never edited
        el.classList.add("comfy-multiline-input");  // match ComfyUI's textarea styling
        el.style.width = "100%";
        el.style.boxSizing = "border-box";
        el.style.height = EXT_H + "px";             // FIXED height, not auto-fit
        el.style.overflowY = "auto";                // a long caption scrolls, does not grow
        el.value = "";
        w = node.addDOMWidget(name, "customtext", el, { serialize: false });
        w.serialize = false;                        // never part of the saved workflow
        w._plsExt = true;
        w.hidden = true;                            // hidden until its PIN is wired
        el.style.display = "none";
        w.computeSize = (width) => (w.hidden ? [0, -4] : [width, EXT_H]);
        _tint(w, bg, edge);                         // green = positive, brown = negative
    } catch (e) { /* a display field must never break node creation */ }
    return w;
}

function _extConnected(node, inputName) {
    const inp = (node.inputs || []).find((i) => i && i.name === inputName);
    return !!(inp && inp.link != null);
}

// Show an EXT field only when its input PIN is wired; fill it with the last run's resolved
// external text. Refit at the end so the node grows/shrinks with the field.
function _syncExternal(node) {
    for (const ef of EXT_FIELDS) {
        const w = _w(node, ef[0]);
        if (!w) continue;
        const on = _extConnected(node, ef[1]);
        w.hidden = !on;
        if (w.element) {
            w.element.style.display = on ? "" : "none";
            if (on) {
                const txt = (node._cteExt && node._cteExt[ef[2]]) || "";
                w.element.value = txt;
                w.element.placeholder = txt ? "" : ("external " + ef[2] + " \u2014 run to load");
            }
        }
    }
    _refitNextFrame(node);
}

/** Live word/char count from the widget values - no run, no tokenizer, no pretending. */
// Count words + chars of one text, applying the node's comment-strip setting -- the SAME
// rule for the positive block and the negative one, so both read consistently.
function _countText(rawTxt, node) {
    const marks = String(_w(node, "comment_markers")?.value ?? "//")
        .split(/\s+/).filter(Boolean)
        .map((m) => m.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    const stripped = marks.length && _w(node, "strip_comments")?.value !== false
        ? rawTxt.replace(new RegExp("(^|\\s)(?:" + marks.join("|") + ").*$", "gm"), " ")
        : rawTxt;
    const s = stripped.trim();
    return { words: s ? s.split(/\s+/).filter(Boolean).length : 0, chars: s.length };
}

// Live word/char counts from the widget values - no run, no tokenizer, no pretending. Both
// the positive block (visible segments) and the negative field, counted the same way.
function _liveCount(node) {
    const n = Math.max(1, Math.min(Number(_w(node, "segments")?.value) || 1, MAX_SEGMENTS));
    let pos = "";
    for (let i = 1; i <= n; i++) pos += " " + (_w(node, `pos_${i}`)?.value || "");
    const neg = _w(node, "use_negative")?.value ? (_w(node, "neg_1")?.value || "") : "";
    const p = _countText(pos, node);
    const g = _countText(neg, node);
    return { words: p.words, chars: p.chars, negWords: g.words, negChars: g.chars };
}

// The wordcount is DRAWN on the node in a footer bar, NOT a DOM widget. A DOM widget rides
// last and gets pushed off the bottom (Frank: "dead for versions"). Painting it means no
// widget layout can ever hide it. Hard facts only -- words, chars, tokens; no limit and no
// warning here (that is the Token Counter node's job, with a manual limit).
function _counterText(node) {
    const c = _liveCount(node);   // live: pos/neg words, pos chars (fields only)
    const t = node._cteTokens;    // last run: tokens from the backend tokenizer
    // v619: fold the last run's resolved external text into the word/char counts, so the
    // readout matches what was actually encoded (pos_external / neg_external). This closes
    // the words-vs-tokens gap -- the tokens already include the external text.
    const ex = node._cteExt || { pos: "", neg: "" };
    const ep = _countText(ex.pos || "", node);
    const en = _countText(ex.neg || "", node);
    const posWords = c.words + ep.words;
    const negWords = c.negWords + en.words;
    const chars = c.chars + ep.chars;
    const kb = chars > 999 ? (chars / 1000).toFixed(1) + "k" : String(chars);
    let txt = `pos ${posWords} words \u00b7 neg ${negWords} words \u00b7 ${kb} chars`;
    if (t) {
        // hard facts only -- pos and neg token counts from the last run, plus the method.
        // No limit and no over-budget warning here: the limit is model-dependent and lives
        // in the Token Counter node (manual model_limit), which owns the warning.
        txt += ` \u00b7 last run: pos ${t.pos} \u00b7 neg ${t.neg} tokens`
             + (t.method ? ` (${t.method})` : "");
    } else {
        txt += ` \u00b7 run to count tokens`;
    }
    return txt;
}

// v615 -- the counter is a FOOTER bar at the BOTTOM of the node. Frank ran the graph, saw the
// v614 title chip work, and asked for it back in the footer where the old pane lived, spelled
// out in full. Still PAINTED (no DOM widget can be pushed off the bottom -- that was the pane's
// fate), drawn in the band _refit reserves at the node's foot. v618: amber is a style element.
// v615 -- air ABOVE the use_negative toggle. v614 padded the toggle's own row (a canvas
// widget), which either did nothing or put the air BELOW it; Frank still found it cramped.
// A prompt field's computeSize IS provably honoured (the whole v613 auto-fit rides on it),
// so the air is padded onto the LAST visible positive field's reserved height instead -- it
// lands above the toggle no matter how the frontend sizes the toggle row.
const NEG_GAP = 14;

app.registerExtension({
    name: "polyhedron.clip_encode",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const _created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = _created ? _created.apply(this, arguments) : undefined;
            const self = this;
            self._cteTokens = null;

            // Wire every prompt field for auto-fit BEFORE the first _applyVisibility --
            // _hide() snapshots computeSize by hasOwnProperty, so ours must exist first.
            for (let i = 1; i <= MAX_SEGMENTS; i++) _wireField(_w(self, `pos_${i}`));
            _wireField(_w(self, "neg_1"));

            // v619: create the read-only external display fields (hidden until their PIN is
            // wired). They ride last (not in DISPLAY) and never serialize.
            for (const ef of EXT_FIELDS) _makeExtField(self, ef[0], ef[3], ef[4]);

            for (const name of ["segments", "use_negative"]) {
                const w = _w(self, name);
                if (!w) continue;
                const cb = w.callback;
                w.callback = function () {
                    const rv = cb ? cb.apply(this, arguments) : undefined;
                    _applyVisibility(self);
                    _refit(self);          // the visible field set changed -- refit
                    return rv;
                };
            }
            // Typing or pasting into ANY field (pos or neg) refits: the field grows to
            // show its text and stops. _refit also redraws, so the counter updates too.
            for (let i = 1; i <= MAX_SEGMENTS; i++) {
                const w = _w(self, `pos_${i}`);
                if (w && w.element) w.element.addEventListener("input", () => _refit(self));
            }
            const negw = _w(self, "neg_1");
            if (negw && negw.element) negw.element.addEventListener("input", () => _refit(self));

            // v604: the filters go ABOVE the prompt boxes -- on screen only. The
            // canon on disk does not move, so no saved graph can be hurt by this.
            // v606: the rows are permuted LATER, in the timeout below. While LiteGraph
            // pours widgets_values in, node.widgets must still be in CANON order --
            // that is what makes the values land correctly WITHOUT configure() having
            // to cooperate, and configure() has been measured not cooperating.
            setTimeout(() => {
                try {
                    _rescueScrambled(self);          // a graph poisoned by v603-v605
                    _reorderWidgetsToDisplay(self);  // filters on top -- rows only
                    _applyVisibility(self);
                    _applyTints(self);
                    _syncExternal(self);             // v619: show EXT fields for wired PINs; refits
                } catch (e) { /* never break a load */ }
            }, 0);
            _applyVisibility(self);
            _applyTints(self);   // v560: green = positive, brown = negative
            return r;
        };

        // v618: the wordcount is PAINTED (v613's win -- no widget layout can hide it), drawn as
        // a FLUSH, full-width status bar hugging the node's foot in the band _refit reserves --
        // integrated, not a floating card. A top separator divides it from the fields; the bottom
        // corners follow the node's radius. The amber is a STYLE element now, always on -- not a
        // warning (the over-limit warning lives in the Token Counter node, with a manual limit).
        const _drawFg = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            const r = _drawFg ? _drawFg.apply(this, arguments) : undefined;
            if (this.flags && this.flags.collapsed) return r;
            try {
                // v715: the SAME function that reserved the height decides what
                // is painted -- one source, so the strip and its content can
                // never disagree. Rows come pre-wrapped at the counter's own
                // separators, so nothing runs off the right edge.
                const lines = _barLines(this);
                const barH = _barHeight(this);
                const W = this.size[0], Hn = this.size[1];
                const topY = Hn - barH;                      // the reserved band
                const rad = 8;                               // follow the node's corner radius
                ctx.save();
                // a flush band across the whole foot -- square where it meets the fields,
                // rounded where it meets the node's bottom corners, so it reads as one piece.
                ctx.beginPath();
                ctx.moveTo(0, topY);
                ctx.lineTo(W, topY);
                ctx.lineTo(W, Hn - rad);
                ctx.arcTo(W, Hn, W - rad, Hn, rad);
                ctx.lineTo(rad, Hn);
                ctx.arcTo(0, Hn, 0, Hn - rad, rad);
                ctx.closePath();
                ctx.fillStyle = "rgba(70,36,0,0.92)";        // amber band -- style, not warning
                ctx.fill();
                // the separator that ties the bar to the fields above
                ctx.beginPath();
                ctx.moveTo(0, topY + 0.5);
                ctx.lineTo(W, topY + 0.5);
                ctx.lineWidth = 1;
                ctx.strokeStyle = "rgba(255,140,0,0.75)";
                ctx.stroke();
                ctx.font = BAR_FONT;
                // Belt and braces: on an absurdly narrow node a single segment
                // can still be wider than the band. Shrink it to fit rather
                // than let it run out (ph_reference.js line 263ff). Never below
                // 7px -- past that the honest answer is a wider node.
                const avail = W - BAR_PAD_X * 2;
                let widest = 0;
                for (const l of lines) {
                    const m = ctx.measureText(l).width;
                    if (m > widest) widest = m;
                }
                if (widest > avail && widest > 0) {
                    const px = Math.max(7, Math.floor(12 * avail / widest));
                    ctx.font = px + "px monospace";
                }
                ctx.textAlign = "left";
                ctx.textBaseline = "middle";
                ctx.fillStyle = "#ff8c00";
                let y = topY + BAR_PAD_Y + BAR_ROW_H * 0.5;
                for (const l of lines) {
                    ctx.fillText(l, BAR_PAD_X, y);
                    y += BAR_ROW_H;
                }
                ctx.restore();
            } catch (e) { /* a counter must never break a draw */ }
            return r;
        };

        // v600: a workflow saved BEFORE v600 carries the height of the old mural --
        // 300-400px of pane that no longer exists. Left alone, the node loads as a
        // tall empty box and Frank drags it shut by hand, once per node, in every
        // graph he owns. So the height is recomputed on load.
        //
        // HEIGHT ONLY (v531). The width is the user's: he widened these nodes to
        // read long prompts, and a node that snaps its own width back on every load
        // is a node that argues with its owner.
        // v602: this used to be `const _configure`, 25 lines above v557's OWN
        // `const _configure`. Two consts, one scope, one SyntaxError -- and the
        // whole module died on load. No tints, no visibility, six segments where
        // two were asked for. The node did not misbehave; it never ran.
        const _cfgHeight = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = _cfgHeight ? _cfgHeight.apply(this, arguments) : undefined;
            // v613: a pre-v613 graph may carry an oversized height. Refitting to the
            // content collapses any dead space. HEIGHT ONLY (v531): width stays the user's.
            try { _refitNextFrame(this); } catch (e) { /* never break a load */ }
            return r;
        };


        // v604: LiteGraph writes widgets_values from node.widgets -- which is in
        // DISPLAY order. Map it back to CANON before it hits the disk, so the file
        // is always in the order INPUT_TYPES declares, and always will be, no matter
        // how the rows are shuffled on screen tomorrow. THIS hook is what makes the
        // reorder safe; without it the reorder is v603 all over again.
        // v606: serialize(), not onSerialize().
        //
        // onSerialize is a CALLBACK the base method chooses to invoke. I have now
        // watched this frontend decline to invoke a base method (configure), so I am
        // not betting a user's graph on it declining to invoke a callback next.
        // serialize() IS the method -- there is no saving a graph without it.
        //
        // The rows swing back to CANON for the length of one call, so the array
        // LiteGraph builds from node.widgets is canon by construction. No mapping, no
        // permutation table, nothing to get backwards. Then they swing back.
        const _serialize = nodeType.prototype.serialize;
        nodeType.prototype.serialize = function () {
            const wasDisplayed = this._plsDisplayed;
            try { if (wasDisplayed) _canonOrder(this); } catch (e) { /* ignore */ }
            const o = _serialize ? _serialize.apply(this, arguments) : {};
            try { if (wasDisplayed) _reorderWidgetsToDisplay(this); } catch (e) { /* ignore */ }
            // v623: persist each prompt field's fitted height into the saved workflow, so a
            // page reload (F5) -- which wipes the runtime _plsH -- can restore it. Applied
            // back in onConfigure BEFORE any refit.
            try {
                const _h = _captureFieldHeights(this);
                if (o && Object.keys(_h).length) o.pls_field_heights = _h;
            } catch (e) { /* never break a save over a height cache */ }
            return o;
        };

        const _exec = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = _exec ? _exec.apply(this, arguments) : undefined;
            try {
                const d = message && message.pls_cte && message.pls_cte[0];
                if (d) {
                    this._cteTokens = { pos: d.pos_tokens, neg: d.neg_tokens,
                                        method: d.method,
                                        posLen: d.pos_len, negLen: d.neg_len };
                    // v619: the resolved external text, for the read-only EXT fields.
                    this._cteExt = { pos: d.pos_ext || "", neg: d.neg_ext || "" };
                    _syncExternal(this);               // fill + resize the EXT fields (redraws)
                }
            } catch (e) { /* a preview must never break a run */ }
            return r;
        };

        // v715: RESERVE THE BAND IN computeSize -- the ph_reference.js pattern
        // (its line 199ff). LiteGraph's own minimum height then accounts for
        // the footer on every path, including Frank dragging the bottom edge.
        // Before this, the reservation existed only inside _refit, so a manual
        // drag put the band underneath the neg_1 textarea.
        const _computeSize = nodeType.prototype.computeSize;
        nodeType.prototype.computeSize = function () {
            const size = _computeSize ? _computeSize.apply(this, arguments)
                                      : [140, 60];
            try { size[1] += _barHeight(this); } catch (e) { /* ignore */ }
            return size;
        };

        // v715: NO onResize hook here, deliberately. The first draft had one to
        // refit when a narrower node wraps the band onto a second row -- and
        // guard #604 stopped it: an onResize hook is a PROVEN dead end in this
        // node (it fought the native resize and re-measured, the v610/v611
        // loop). It is also unnecessary: _barLines reads node.size[0] on every
        // call, so the band height already follows the width, and LiteGraph
        // clamps a manual resize against computeSize -- which now carries that
        // height. The width case is handled by the reservation itself.

        // v619: a PIN being wired or unwired shows/hides the matching EXT field.
        const _connChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const r = _connChange ? _connChange.apply(this, arguments) : undefined;
            try { _syncExternal(this); } catch (e) { /* never break on a wiring change */ }
            // v715: _syncExternal refits on the NEXT frame. Grow immediately as
            // well (the ph_reference.js line 209ff pattern), so the frame in
            // between never shows the band under the field that just appeared.
            try {
                const want = this.computeSize();
                if (this.size && this.size[1] < want[1]) this.size[1] = want[1];
                this.setDirtyCanvas(true, true);
            } catch (e) { /* never break a connection */ }
            return r;
        };

        // v557: heal BEFORE LiteGraph applies widgets_values by position -
        // hooking prototype.configure is the live-proven point (ph_power_upscale).
        const _configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (info) {
            try {
                // v606: NO display mapping here. This hook has been measured NOT
                // FIRING in Frank's frontend -- the rows moved (onNodeCreated) and the
                // values did not (this). Everything load-bearing now lives in the
                // timeout, which is proven to run. What stays is v557's length heal,
                // which is harmless if it never fires and correct if it does.
                if (info && Array.isArray(info.widgets_values)) {
                    info.widgets_values = _healPreV557(info.widgets_values);
                }
            } catch (err) { /* never break configure */ }
            return _configure ? _configure.apply(this, arguments) : undefined;
        };

        // Loading a saved graph: re-apply visibility once LiteGraph is done.
        const _cfg = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            // v623: restore fitted heights saved by serialize() BEFORE any refit runs, so a
            // cold page reload (F5) -- which wiped the runtime _plsH -- lands the fields at the
            // remembered height with no measurement (also sidesteps the load-time layout-timing
            // gap). Runs first, ahead of _cfg's own refit.
            try { _restoreFieldHeights(this, arguments[0]); } catch (e) { /* never break a load */ }
            const r = _cfg ? _cfg.apply(this, arguments) : undefined;
            const self = this;
            setTimeout(() => { try { _applyVisibility(self); _applyTints(self);
                                     _refitNextFrame(self); }
                               catch (e) { /* ignore */ } }, 0);
            return r;
        };
    },
});
