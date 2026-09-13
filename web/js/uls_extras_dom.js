/**
 * uls_extras_dom.js -- the painted extras of five nodes, under Nodes 2.0 (v941, S5b).
 *
 * S5 measured (10.09.2026) that eighteen of the twenty "C2" nodes are NOT
 * empty under Nodes 2.0 -- their widgets render; only what their canvas code
 * paints is missing. Three of those painted parts are FUNCTIONS: Int's preset
 * chips and "+", Filter's Reset. The rest is information: CLIP Encode's word
 * band, the Load nodes' status line (Load CLIP's amber "no model connected"),
 * Everywhere's "feeds / fed" line. This file gives each of them a small view
 * through the shared switch point (uls_vue_views.js): the view exists only
 * while LiteGraph.vueNodesMode is true, stores nothing, and under the classic
 * renderer the painted parts stay exactly what they were.
 *
 * One truth: every view CALLS the node's own pieces (moved verbatim out of its
 * draw/onMouseDown in v941) -- no rule is rebuilt here. Views that mirror
 * state the node changes by itself carry a signature, so the switch point
 * redraws them when that state moves (a word count while typing).
 *
 * Not built, on purpose: WanSigmaSchedule's inline output dots (deprecated
 * node; its outputs work under Nodes 2.0) and the Load nodes' slot "x" (a
 * custom canvas widget -- measured separately before anything is built).
 */
import { registerVueView } from "./uls_vue_views.js";
import { getValue, setValue, addIntPreset, removeIntPreset, INT_EMPTY_HINT } from "./ph_int.js";
import { _pfReset } from "./ph_filter.js";
import { _barLines, _counterText } from "./ph_clip_encode.js";
import { loadStatusLine } from "./ph_basics.js";
// public build: ULSEverywhere is an internal-only node -- its view and the
// ph_everywhere.js import are not carried here (v374).

const CSS = `
.uls-x { font:11px 'Segoe UI',Arial; color:#9a9a9a; padding:2px 4px;
  --comfy-widget-min-height:16px; }
.uls-x-chips { display:flex; flex-wrap:wrap; gap:4px; align-items:center; }
.uls-x-chip { background:#1a1a2a; border:1px solid #2a2a3a; border-radius:4px;
  color:#a8a8b8; cursor:pointer; padding:1px 8px; font:10px 'Segoe UI',Arial; }
.uls-x-chip.on { background:#2f6a3a; border-color:#7fd08a; color:#eaffea; font-weight:bold; }
.uls-x-add { background:#16161f; border:1px solid #2a2a3a; border-radius:4px;
  color:#6a6a7a; cursor:pointer; padding:1px 8px; font:bold 11px 'Segoe UI',Arial; }
.uls-x-add:hover { border-color:#4a4a66; color:#8fb8ff; }
.uls-x-hint { color:#5a5a6a; font:italic 10px 'Segoe UI',Arial; }
.uls-x-btn { background:#1e1a28; border:1px solid #3a3446; border-radius:4px;
  color:#c8c0d8; cursor:pointer; padding:2px 10px; font:10px 'Segoe UI',Arial; }
.uls-x-band { background:rgba(70,36,0,0.92); border-top:1px solid rgba(255,140,0,0.75);
  border-radius:0 0 8px 8px; color:#ff8c00; font:12px monospace; padding:3px 8px; }
.uls-x-line { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
`;
let _css = false;
function ensureCss() {
    if (_css) return;
    _css = true;
    const st = document.createElement("style");
    st.textContent = CSS;
    document.head.appendChild(st);
}

function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
}

// ── Int: the preset chips (click = select, right-click = drop) and "+" ──────
function renderInt(node, root) {
    root.textContent = "";
    const st = node._phi;
    const rows = (st && st.rows) || [];
    const value = getValue(node);
    const box = el("div", "uls-x-chips");
    rows.forEach((row, i) => {
        const c = el("button", "uls-x-chip" + (row.value === value ? " on" : ""),
                     row.name || String(row.value));
        c.title = row.name ? row.name + " = " + row.value + "  (right-click: remove)"
                           : "right-click: remove";
        c.onclick = () => { setValue(node, row.value); renderInt(node, root); };
        c.oncontextmenu = (e) => {
            e.preventDefault();
            removeIntPreset(node, i);
            renderInt(node, root);
        };
        box.appendChild(c);
    });
    const add = el("button", "uls-x-add", "+");
    add.title = "Name the current value as a preset";
    add.onclick = () => { addIntPreset(node); renderInt(node, root); };
    box.appendChild(add);
    if (!rows.length) box.appendChild(el("span", "uls-x-hint", INT_EMPTY_HINT));
    root.appendChild(box);
}

// ── Filter: the Reset chip ─────────────────────────────────────────────────
function renderFilter(node, root) {
    root.textContent = "";
    const b = el("button", "uls-x-btn", "Reset");
    b.title = "Every grading control back to its default";
    b.onclick = () => _pfReset(node);
    root.appendChild(b);
}

// ── CLIP Text Encode: the word band ────────────────────────────────────────
function renderClipBand(node, root) {
    root.textContent = "";
    const band = el("div", "uls-x-band");
    for (const l of _barLines(node)) band.appendChild(el("div", "uls-x-line", l));
    root.appendChild(band);
}

// ── Load CLIP / Model / VAE: the status line ───────────────────────────────
function renderLoadStatus(node, root) {
    root.textContent = "";
    const { text, colour } = loadStatusLine(node);
    const line = el("div", "uls-x-line", text || "");
    if (colour) line.style.color = colour;
    root.appendChild(line);
}

const common = { className: "uls-x", prepare: ensureCss };

registerVueView("ULSInt", Object.assign({}, common, {
    widgetName: "uls_int_dom", label: "Int presets",
    ready: (node) => !!node._phi,
    signature: (node) => JSON.stringify([(node._phi && node._phi.rows) || [], getValue(node)]),
    render: renderInt,
}));
registerVueView("ULSFilter", Object.assign({}, common, {
    widgetName: "uls_filter_dom", label: "Filter reset",
    render: renderFilter,
}));
registerVueView("ULSCLIPTextEncode", Object.assign({}, common, {
    widgetName: "uls_cte_dom", label: "CLIP word band",
    signature: (node) => _counterText(node) + "|" + Math.round((node.size && node.size[0]) || 0),
    render: renderClipBand,
}));
const loadSpec = () => Object.assign({}, common, {
    widgetName: "uls_load_status_dom", label: "Load status",
    signature: (node) => { const s = loadStatusLine(node); return s.text + "|" + s.colour; },
    render: renderLoadStatus,
});
// written out, not looped: the renderer scan reads the class from the call
registerVueView("ULSLoadCLIP", loadSpec());
registerVueView("ULSLoadModel", loadSpec());
registerVueView("ULSLoadVAE", loadSpec());
