/**
 * ph_cte_search.js -- v992: a search row for the CLIP Text Encode.
 *
 * Frank, 22.09.2026: "eine Suchspalte unter den Segments ... ein Wort, einen
 * Halbsatz oder einen Abschnitt ... wird im Prompt markiert, dazu zwei
 * Vorwaerts-/Rueckwaerts-Dreieck-Pfeile wie beim LoRA-Stack ... ersetzt die
 * system-eigene Suchfunktion ... gerne direkte live-Markierung".
 *
 * THE ROW: one DOM widget (serialize:false -- it never touches widgets_values,
 * guard #577): [ search field ][ \u25c0 ][ 3/12 ][ \u25b6 ][ \u2715 ] -- v993: the
 * counter shows a dim dash while there is no search, so the middle never
 * gapes; typing MARKS live but never moves the view (a jump while typing
 * breaks the typing) -- only Enter / Shift+Enter / the arrows jump.
 *   typing        marks every hit live, in every VISIBLE prompt field (the
 *                 positive segments in order, then the negative), and jumps to
 *                 the first one
 *   Enter / \u25b6    next hit (wraps), Shift+Enter / \u25c0 previous
 *   Esc / \u2715      clear
 *   Ctrl+F        inside a prompt field opens THIS search (prefilled with the
 *                 selected text) instead of the browser's own find bar
 * Matching is literal and case-insensitive; any run of whitespace in the query
 * matches any run of whitespace in the prompt, so a pasted half-sentence or a
 * whole paragraph is found even across a line break.
 *
 * THE MARKS: the textarea itself is never touched (tints, auto-fit, the Vue
 * relay all stay as they are). A layer sits over the field, pointer-events
 * off; an invisible mirror with the field's own font, padding and wrapping
 * measures where every hit falls (getClientRects), and the layer paints a
 * translucent box there -- amber for all hits, a brighter box for the current
 * one. It works for whichever textarea is on screen: the classic DOM element,
 * or under Nodes 2.0 the Vue field standing in for it (the CTE hands both in).
 *
 * THE JUMP: when the current hit is outside the canvas view, the canvas is
 * panned (ds.offset) so the hit lands in the middle -- the same move in both
 * renderers, because both place the node from ds.
 *
 * The pure part (findMatches, stepIndex, counterText) has no DOM and is what
 * the guard drives in node.
 */

import { app } from "../../scripts/app.js";

export const SEARCH_W = "pls_search";
export const ROW_H = 34;          // v993: roomier (Frank: 'ein wenig aufspreizen')
export const MAX_HITS = 999;
const DEBOUNCE_MS = 90;
const HIT_BG = "rgba(255, 196, 0, 0.30)";
const CUR_BG = "rgba(255, 140, 0, 0.55)";
const CUR_EDGE = "1px solid #ffb000";
const LAYER_CLASS = "pls-cte-hl";
export const IDLE = "\u2013";         // v993: the counter while there is no search

// ---------------------------------------------------------------- pure core

function _esc(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }

/** [[start, end], ...] of every hit of `query` in `text`: literal,
 *  case-insensitive, any whitespace run matches any whitespace run. */
export function findMatches(text, query) {
    const toks = String(query || "").trim().split(/\s+/).filter(Boolean);
    if (!toks.length) return [];
    const re = new RegExp(toks.map(_esc).join("\\s+"), "gi");
    const t = String(text || "");
    const out = [];
    let m;
    while ((m = re.exec(t)) !== null) {
        if (!m[0].length) { re.lastIndex++; continue; }
        out.push([m.index, m.index + m[0].length]);
        if (out.length >= MAX_HITS) break;
    }
    return out;
}

/** The next index after `cur` by `dir` (+1 / -1), wrapping; -1 when n == 0. */
export function stepIndex(cur, dir, n) {
    if (!n) return -1;
    if (cur < 0 || cur >= n) return dir < 0 ? n - 1 : 0;
    return ((cur + dir) % n + n) % n;
}

/** "3/12", "0/0" for no hit, "" for no query. */
export function counterText(cur, n, query) {
    if (!String(query || "").trim()) return "";
    return n ? `${cur + 1}/${n}${n >= MAX_HITS ? "+" : ""}` : "0/0";
}

/** All hits over the fields, in field order: [{w, start, end}]. */
export function collectHits(fields, query) {
    const hits = [];
    for (const w of fields) {
        const v = w && typeof w.value === "string" ? w.value : "";
        for (const [s, e] of findMatches(v, query)) {
            hits.push({ w, start: s, end: e });
            if (hits.length >= MAX_HITS) return hits;
        }
    }
    return hits;
}

// ---------------------------------------------------------------- the view

const MIRROR_KEYS = ["boxSizing", "paddingTop", "paddingRight", "paddingBottom", "paddingLeft",
    "borderTopWidth", "borderRightWidth", "borderBottomWidth", "borderLeftWidth",
    "fontFamily", "fontSize", "fontWeight", "fontStyle", "fontVariant", "lineHeight",
    "letterSpacing", "wordSpacing", "textIndent", "textTransform", "tabSize",
    "whiteSpace", "wordWrap", "overflowWrap", "wordBreak", "direction"];

function _layerFor(ta) {
    const parent = ta.parentElement;
    if (!parent) return null;
    let layer = ta._plsLayer;
    if (!layer || layer.parentElement !== parent) {
        if (layer && layer.parentElement) layer.parentElement.removeChild(layer);
        layer = document.createElement("div");
        layer.className = LAYER_CLASS;
        layer.setAttribute("aria-hidden", "true");
        layer.style.cssText = "position:absolute;pointer-events:none;overflow:hidden;z-index:2;";
        try {
            if (getComputedStyle(parent).position === "static") parent.style.position = "relative";
        } catch (e) { /* never break */ }
        parent.appendChild(layer);
        ta._plsLayer = layer;
    }
    return layer;
}

function _clearLayer(ta) {
    const layer = ta && ta._plsLayer;
    if (layer) layer.textContent = "";
}

/** Paint the boxes of `hits` (all in this textarea's text) over `ta`.
 *  Returns the box element of the current hit, or null. */
function _paint(ta, text, hits, curHit) {
    const layer = _layerFor(ta);
    if (!layer) return null;
    layer.textContent = "";
    if (!hits.length || ta.offsetParent === null || !ta.offsetWidth) return null;
    layer.style.left = ta.offsetLeft + "px";
    layer.style.top = ta.offsetTop + "px";
    layer.style.width = ta.offsetWidth + "px";
    layer.style.height = ta.offsetHeight + "px";

    const cs = getComputedStyle(ta);
    const mirror = document.createElement("div");
    for (const k of MIRROR_KEYS) mirror.style[k] = cs[k];
    const bl = parseFloat(cs.borderLeftWidth) || 0, br = parseFloat(cs.borderRightWidth) || 0;
    mirror.style.boxSizing = "border-box";
    mirror.style.borderStyle = "solid";
    mirror.style.borderColor = "transparent";
    mirror.style.position = "absolute";
    mirror.style.left = "0px";
    mirror.style.top = "0px";
    mirror.style.visibility = "hidden";
    mirror.style.overflow = "hidden";
    mirror.style.width = (ta.clientWidth + bl + br) + "px";   // the scrollbar is not text room
    mirror.style.height = "auto";
    if (cs.whiteSpace === "normal" || !cs.whiteSpace) mirror.style.whiteSpace = "pre-wrap";
    let pos = 0;
    const spans = [];
    for (const h of hits) {
        if (h.start > pos) mirror.appendChild(document.createTextNode(text.slice(pos, h.start)));
        const sp = document.createElement("span");
        sp.textContent = text.slice(h.start, h.end);
        mirror.appendChild(sp);
        spans.push([sp, h]);
        pos = h.end;
    }
    mirror.appendChild(document.createTextNode(text.slice(pos) + "\u200b"));
    layer.appendChild(mirror);

    const mr = mirror.getBoundingClientRect();
    const ratio = mirror.offsetWidth ? (mr.width / mirror.offsetWidth) || 1 : 1;
    let curBox = null;
    const frag = document.createDocumentFragment();
    for (const [sp, h] of spans) {
        const isCur = h === curHit;
        for (const r of sp.getClientRects()) {
            const b = document.createElement("div");
            const x = (r.left - mr.left) / ratio - ta.scrollLeft;
            const y = (r.top - mr.top) / ratio - ta.scrollTop;
            b.style.cssText = "position:absolute;border-radius:2px;"
                + `left:${x - 1}px;top:${y}px;width:${r.width / ratio + 2}px;height:${r.height / ratio}px;`
                + `background:${isCur ? CUR_BG : HIT_BG};` + (isCur ? `outline:${CUR_EDGE};` : "");
            if (isCur) { b.dataset.cur = "1"; if (!curBox) curBox = b; }
            frag.appendChild(b);
        }
    }
    layer.removeChild(mirror);
    layer.appendChild(frag);
    return curBox;
}

/** Pan the canvas so `el` sits in the middle, when it is outside the view. */
function _reveal(el) {
    const cv = app && app.canvas;
    const canvasEl = cv && cv.canvas;
    if (!el || !cv || !cv.ds || !canvasEl) return false;
    const r = el.getBoundingClientRect(), c = canvasEl.getBoundingClientRect();
    const M = 60;
    const outY = r.top < c.top + M || r.bottom > c.bottom - M;
    const outX = r.left < c.left + M || r.right > c.right - M;
    if (!outX && !outY) return false;
    // only the axis that is out moves -- a hit below the view scrolls down,
    // it does not also yank the view sideways
    const s = cv.ds.scale || 1;
    if (outX) cv.ds.offset[0] += ((c.left + c.width / 2) - (r.left + r.width / 2)) / s;
    if (outY) cv.ds.offset[1] += ((c.top + c.height / 2) - (r.top + r.height / 2)) / s;
    try { cv.setDirty(true, true); } catch (e) { /* ignore */ }
    try { app.graph && app.graph.setDirtyCanvas && app.graph.setDirtyCanvas(true, true); } catch (e) { /* ignore */ }
    return true;
}

// A textarea's own scroll (a field past the CTE's 2000 px cap): bring the
// current hit into the field's view first.
function _scrollIntoField(ta, box) {
    if (!ta || !box) return false;
    const top = parseFloat(box.style.top) || 0;
    if (top >= 0 && top <= ta.clientHeight - 20) return false;
    ta.scrollTop = Math.max(0, ta.scrollTop + top - ta.clientHeight / 2);
    return true;
}

// ---------------------------------------------------------------- the node

const REGISTRY = new Set();

function _state(node) {
    if (!node._plsSearch) node._plsSearch = { q: "", hits: [], cur: -1, timer: 0, ro: null, seen: new Set() };
    return node._plsSearch;
}

/** Recompute the hits and repaint. jump: "first" (current = first hit, pan to
 *  it) | "mark" (current = first hit, no pan) | "keep" | "none". */
export function refresh(node, jump = "keep") {
    const st = node && node._plsSearch;
    if (!st || !node._plsSearchApi) return;
    const api = node._plsSearchApi;
    const fields = api.fields() || [];
    const prev = st.hits[st.cur];
    st.hits = st.q.trim() ? collectHits(fields, st.q) : [];
    if (jump === "first" || jump === "mark") st.cur = st.hits.length ? 0 : -1;
    else if (prev) {
        const i = st.hits.findIndex((h) => h.w === prev.w && h.start === prev.start);
        st.cur = i >= 0 ? i : Math.min(st.cur, st.hits.length - 1);
    } else st.cur = st.hits.length ? Math.min(Math.max(st.cur, 0), st.hits.length - 1) : -1;
    _updateCounter(node);
    _repaint(node, jump === "first");                        // "mark" / "keep" / "none" never pan
}

function _repaint(node, doJump) {
    const st = node._plsSearch, api = node._plsSearchApi;
    if (!st || !api) return;
    if (!st.q.trim()) {                                   // nothing to mark: no layers, no observers
        for (const ta of st.seen) _clearLayer(ta);
        return;
    }
    const cur = st.hits[st.cur] || null;
    const active = new Set();
    let curBox = null, curTa = null;
    for (const w of api.fields() || []) {
        const ta = api.textareaFor(w);
        if (!ta) continue;
        active.add(ta);
        _observe(node, ta);
        const mine = st.hits.filter((h) => h.w === w);
        const text = typeof w.value === "string" ? w.value : (ta.value || "");
        const box = _paint(ta, text, mine, cur);
        if (box) { curBox = box; curTa = ta; }
    }
    for (const ta of st.seen) if (!active.has(ta)) _clearLayer(ta);   // hidden segments
    if (doJump && curBox) {
        if (_scrollIntoField(curTa, curBox)) curBox = _paint(curTa, curTa.value, st.hits.filter((h) => h.w === cur.w), cur) || curBox;
        _reveal(curBox);
    }
}

function _observe(node, ta) {
    const st = node._plsSearch;
    if (st.seen.has(ta)) return;
    st.seen.add(ta);
    ta.addEventListener("scroll", () => _repaint(node, false), { passive: true });
    if (typeof ResizeObserver === "function") {
        if (!st.ro) st.ro = new ResizeObserver(() => { if (st.q.trim()) schedule(node, "keep"); });
        st.ro.observe(ta);   // repaint only -- never a setSize, so no loop (#604)
    }
}

/** Debounced refresh -- typing in the search or in a prompt field. */
export function schedule(node, jump = "keep") {
    const st = node && node._plsSearch;
    if (!st) return;
    if (st.timer) clearTimeout(st.timer);
    st.timer = setTimeout(() => { st.timer = 0; try { refresh(node, jump); } catch (e) { /* never break */ } }, DEBOUNCE_MS);
}

export function step(node, dir) {
    const st = node && node._plsSearch;
    if (!st) return;
    if (st.timer) { clearTimeout(st.timer); st.timer = 0; refresh(node, "none"); }
    st.cur = stepIndex(st.cur, dir, st.hits.length);
    _updateCounter(node);
    _repaint(node, true);
}

export function clear(node) {
    const st = node && node._plsSearch;
    if (!st) return;
    st.q = "";
    if (node._plsSearchInput) node._plsSearchInput.value = "";
    refresh(node, "none");
}

function _updateCounter(node) {
    const st = node._plsSearch, c = node._plsSearchCounter;
    if (!c) return;
    const t = counterText(st.cur, st.hits.length, st.q);
    c.textContent = t || IDLE;                               // v993: a dim dash, never empty
    c.style.opacity = t ? "1" : ".5";
    c.style.color = st.q.trim() && !st.hits.length ? "#e06060" : "";
}

function _btn(label, title, onClick) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    b.title = title;
    b.style.cssText = "flex:0 0 auto;width:28px;height:28px;padding:0;border-radius:5px;cursor:pointer;"
        + "border:1px solid var(--border-color,#444);background:var(--comfy-input-bg,#222);"
        + "color:var(--input-text,#ddd);font-size:12px;line-height:26px;";
    b.addEventListener("mousedown", (e) => e.preventDefault());   // keep the focus in the field
    b.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); onClick(); });
    return b;
}

/**
 * Build the row and wire it. api = { fields(): visible prompt widgets in
 * order, textareaFor(w): the textarea on screen for w (or null) }.
 * Returns the widget, or null when the frontend has no DOM widgets.
 */
export function installSearch(node, api) {
    if (!node || typeof node.addDOMWidget !== "function") return null;
    const have = (node.widgets || []).find((w) => w && w.name === SEARCH_W);
    if (have) return have;
    _state(node);
    node._plsSearchApi = api;

    const row = document.createElement("div");
    row.style.cssText = "display:flex;align-items:center;gap:6px;width:100%;height:" + ROW_H
        + "px;box-sizing:border-box;padding:3px 0;";
    const input = document.createElement("input");
    input.type = "text";
    input.spellcheck = false;
    input.placeholder = "\u2315 search prompt\u2026";
    input.title = "Search the prompts -- Enter: next hit, Shift+Enter: previous, Esc: clear. "
        + "Ctrl+F inside a prompt field jumps here.";
    input.style.cssText = "flex:1 1 auto;min-width:60px;height:28px;box-sizing:border-box;padding:0 8px;"
        + "border-radius:5px;border:1px solid var(--border-color,#444);"
        + "background:var(--comfy-input-bg,#222);color:var(--input-text,#ddd);font-size:13px;";
    const counter = document.createElement("span");
    counter.style.cssText = "flex:0 0 auto;min-width:44px;text-align:center;font:12px ui-monospace,monospace;"
        + "color:var(--descrip-text,#aaa);opacity:.5;";
    counter.textContent = IDLE;
    const prev = _btn("\u25c0", "previous hit (Shift+Enter)", () => step(node, -1));
    const next = _btn("\u25b6", "next hit (Enter)", () => step(node, +1));
    const x = _btn("\u2715", "clear the search (Esc)", () => clear(node));
    row.append(input, prev, counter, next, x);

    // v993: typing marks live and makes the first hit current -- it never moves
    // the view (Frank: "springen geht nicht, weil man dann beim Tippen
    // unterbrochen wird"). Enter / the arrows jump.
    input.addEventListener("input", () => { _state(node).q = input.value; schedule(node, "mark"); });
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") { e.preventDefault(); e.stopPropagation(); step(node, e.shiftKey ? -1 : +1); }
        else if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); clear(node); }
        else e.stopPropagation();   // the canvas shortcuts stay out of the field
    });
    node._plsSearchInput = input;
    node._plsSearchCounter = counter;

    const w = node.addDOMWidget(SEARCH_W, "pls_search", row, {
        serialize: false, hideOnZoom: false,
        getMinHeight: () => ROW_H, getMaxHeight: () => ROW_H,
    });
    w.serialize = false;                      // never in widgets_values (#577)
    w.computeSize = (width) => [width, ROW_H];
    REGISTRY.add(node);
    const removed = node.onRemoved;
    node.onRemoved = function () {
        REGISTRY.delete(this);
        try { const st = this._plsSearch; if (st && st.ro) st.ro.disconnect(); } catch (e) { /* ignore */ }
        return removed ? removed.apply(this, arguments) : undefined;
    };
    return w;
}

/** Ctrl+F inside a CTE prompt field opens the node's own search. */
export function nodeForTextarea(ta) {
    for (const node of REGISTRY) {
        const api = node._plsSearchApi;
        if (!api) continue;
        for (const w of api.fields() || []) if (api.textareaFor(w) === ta) return node;
    }
    return null;
}

function _onKey(e) {
    if (!(e.ctrlKey || e.metaKey) || String(e.key).toLowerCase() !== "f" || e.altKey) return;
    const t = e.target;
    if (!t || t.tagName !== "TEXTAREA") return;
    const node = nodeForTextarea(t);
    if (!node || !node._plsSearchInput) return;
    e.preventDefault();
    e.stopPropagation();
    const sel = (typeof t.selectionStart === "number" && t.selectionEnd > t.selectionStart)
        ? t.value.slice(t.selectionStart, t.selectionEnd) : "";
    const inp = node._plsSearchInput;
    if (sel && sel.length <= 400) {
        inp.value = sel;
        _state(node).q = sel;
        schedule(node, "mark");
    }
    inp.focus();
    inp.select();
}

if (typeof document !== "undefined" && document.addEventListener && !globalThis.__plsCteSearchKeys) {
    globalThis.__plsCteSearchKeys = true;
    document.addEventListener("keydown", _onKey, true);
}

// ---------------------------------------------------------------- the hook
// Its own extension: the CTE only publishes node._plsCteApi and calls
// node._plsSearchSchedule() -- no import between the two files, so every
// guard that lifts ph_clip_encode.js on its own keeps working unchanged.
// The row is built synchronously inside onNodeCreated, BEFORE the CTE's
// setTimeout(0) that moves it under `segments`.
app.registerExtension({
    name: "polyhedron.cte_search",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "ULSCLIPTextEncode") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = created ? created.apply(this, arguments) : undefined;
            const node = this;
            try {
                installSearch(node, {
                    fields: () => (node._plsCteApi ? node._plsCteApi.fields() : []),
                    textareaFor: (w) => (node._plsCteApi ? node._plsCteApi.textareaFor(w) : null),
                });
                node._plsSearchSchedule = (jump) => schedule(node, jump || "keep");
            } catch (e) { /* the search must never break node creation */ }
            return r;
        };
    },
});
