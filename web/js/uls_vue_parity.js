// uls_vue_parity.js -- v944/v945: the classic node stays the benchmark under Nodes 2.0.
//
// M1 SIZE BRIDGE. Nodes 2.0 never asks computeSize() while resizing: its floor is
// the element's inline min-width (else the frontend's MIN_NODE_WIDTH) and the
// measured content height (useNodeResize.ts, frontend 1.49.6). After a resize the
// layout sync writes node.size and calls onResize(node.size) -- so our clamps DO
// run, but what they change never reaches the element. The bridge restores the
// classic order (computeSize floor first, then the node's own clamp, exactly as
// LGraphCanvas does) and writes the clamped size back to the element, and it pins
// what the clamp revealed as min-width / min-height / max-height so the frontend's
// own resize respects it while the user drags. No table, no probe: the node's
// clamp is the one source. Classic mode passes straight through.
//
// M2 TINT BRIDGE. A suite text field is tinted on its classic element
// (ph_clip_encode: green positive, brown negative; ph_outpaint: green prompt).
// Nodes 2.0 renders its own textarea, so the tint never shows. The classic
// element's inline style is the source; it is copied to the Vue textarea of the
// same widget. Vue renders the visible widgets in node.widgets order; if the
// counts differ the field is left alone -- never a guess.
//
// M3 GROWTH RULE (v945). LiteGraph grows a widget only when it has NO computeSize
// (LGraphNode arrangeWidgets, frontend 1.49.6: computeSize = fixed height, else
// computeLayoutSize = growable up to its maxHeight). Nodes 2.0 never asks
// computeSize: its widget grid gives a row the growing track 'auto' for every
// widget with a computeLayoutSize -- every DOM widget inherits one -- and for the
// expanding types (textarea, ...), so content-fitted fields and fixed panels
// swallow the height the user drags in (P1 class F10). The rule reads the SAME
// property the classic layout reads: a row whose widget has a computeSize gets
// 'min-content', every other row keeps the frontend's own track. Rows are matched
// to widgets in the frontend's order (visible widgets); a DOM element mounted in
// the node must sit in its own row, and the counts must agree -- otherwise nothing
// is touched, never a guess. The template goes into ONE stylesheet rule per node
// (keyed by data-node-id, !important): Vue re-applies its inline
// grid-template-rows on every patch, a stylesheet rule stays outside that binding.
//
// Only min-width/min-height/max-height and the field styles are written: Vue
// re-applies every key of its own style binding (--min-node-width, transform,
// zIndex, opacity) on each patch, so those are never touched.

import { app } from "../../scripts/app.js";
import { vueMode } from "./uls_vue_views.js";
import { isPackNode, classicHidden } from "./uls_vue_hidden.js";

const POLL_MS = 400;
const EPS = 0.5;
const TINT_KEYS = ["backgroundColor", "borderLeft", "borderRadius"];
// v949 HEIGHT BRIDGE. A content-fitted field (CLIP Text Encode v613) writes its
// measured height on the CLASSIC textarea; the Vue textarea keeps the frontend's
// 64 px and scrolls while the node box already carries the full height. Copy the
// inline height too -- only where the classic element states one. The Vue field
// sets its own line height (15 px) and padding (20/12/8) against classic 'normal'
// and 2 px, so the same text needs MORE room there (measured 838 vs 532 px):
// the field gets the larger of the classic height and its own content height,
// measured with height 'auto' -- synchronous, event-free, so it cannot loop.
const HEIGHT_KEY = "height";
export function bridgedHeight(classicH, vueEl) {
    const c = parseInt(classicH, 10) || 0;
    const prev = vueEl.style.height;
    vueEl.style.height = "auto";
    const need = Math.ceil(vueEl.scrollHeight || 0);
    vueEl.style.height = prev;
    return Math.max(c, need);
}

function titleH() {
    return (typeof LiteGraph !== "undefined" && LiteGraph.NODE_TITLE_HEIGHT) || 30;
}

export function vueElement(node) {
    return document.querySelector(`[data-node-id="${node.id}"]`);
}

/** The classic resize floor: LiteGraph applies computeSize() before setSize(). */
export function computeFloor(node, size) {
    let min = null;
    try { min = node.computeSize ? node.computeSize() : null; } catch (e) { min = null; }
    if (min && isFinite(min[0]) && size[0] < min[0]) size[0] = min[0];
    if (min && isFinite(min[1]) && size[1] < min[1]) size[1] = min[1];
    return size;
}

/** What one clamp pass revealed: raised axis = floor, lowered axis = cap. */
export function learnBounds(node, want, got) {
    const b = node._ulsBounds || (node._ulsBounds = { floor: [0, 0], cap: [0, 0] });
    for (const i of [0, 1]) {
        if (got[i] > want[i] + EPS) b.floor[i] = got[i];
        else if (got[i] < want[i] - EPS) b.cap[i] = got[i];
    }
    return b;
}

/** Content height as the frontend measures it (useNodeResize: --node-height 0). */
export function contentMin(el) {
    const saved = el.style.getPropertyValue("--node-height");
    el.style.setProperty("--node-height", "0px");
    const h = el.getBoundingClientRect().height;
    el.style.setProperty("--node-height", saved || "");
    return h;
}

/** Push a clamped size and the learned bounds onto the Vue element. */
export function pushSize(node, size, el) {
    el = el || vueElement(node);
    if (!el) return false;
    const T = titleH(), b = node._ulsBounds;
    el.style.setProperty("--node-width", `${size[0]}px`);
    el.style.setProperty("--node-height", `${size[1] + T}px`);
    if (b) {
        if (b.floor[0] > 0) el.style.minWidth = `${b.floor[0]}px`;
        if (b.floor[1] > 0) el.style.minHeight = `max(var(--node-height), ${b.floor[1] + T}px)`;
        // a cap below the Vue content would clip it -- the taller Vue rows win
        if (b.cap[1] > 0 && b.cap[1] + T >= contentMin(el) - EPS) el.style.maxHeight = `${b.cap[1] + T}px`;
    }
    return true;
}

/** Wrap onResize once; re-wraps if the node later installs its own. */
export function wrapResize(node) {
    const cur = node.onResize;
    if (cur && cur._ulsParity) return false;
    const orig = cur;
    const wrapped = function (size) {
        if (!vueMode() || !size || size.length < 2) {
            return orig ? orig.apply(this, arguments) : undefined;
        }
        const want = [size[0], size[1]];
        computeFloor(this, size);
        const r = orig ? orig.apply(this, arguments) : undefined;
        if (Math.abs(size[0] - want[0]) > EPS || Math.abs(size[1] - want[1]) > EPS) {
            learnBounds(this, want, size);
            pushSize(this, size);
        }
        return r;
    };
    wrapped._ulsParity = true;
    node.onResize = wrapped;
    return true;
}

/** Visible text widgets whose classic element is NOT mounted in the Vue node. */
export function tintPairs(node, host) {
    const vts = [...host.querySelectorAll("textarea")].filter(t => !t.closest(".dom-widget"));
    const ws = (node.widgets || []).filter(w => w && !classicHidden(w) && !(w.options && w.options.hidden)
        && w.element && w.element.tagName === "TEXTAREA" && !host.contains(w.element));
    if (vts.length !== ws.length) return [];
    return ws.map((w, i) => [w, vts[i]]);
}

/** The Vue textarea standing in for a classic text widget under Nodes 2.0, or null. */
export function vueFieldFor(node, w) {
    if (!vueMode()) return null;
    const host = vueElement(node);
    if (!host) return null;
    for (const [cw, t] of tintPairs(node, host)) if (cw === w) return t;
    return null;
}

// ── v954 CONTROL ROW. LiteGraph paints "control after generate" as its own row
// under the seed (fixed / increment / decrement / randomize). Nodes 2.0 folds
// that widget into a dice button on the seed row and lists no row for it
// (Frank, 12.09.: "hier fehlt ein Fixed/Randomize Feld"). The row is put back
// as a small select, straight after the seed row, wired to the very same
// widget -- every pack node that carries one (Seed, Sampler, Power Upscale, ...).
const CTRL_CLASS = "uls-ctrl-row";
const CTRL_VALUES = ["fixed", "increment", "decrement", "randomize"];
function controlWidgets(node) {
    return (node.widgets || []).filter((w) => w && /control_after_generate$/.test(String(w.name || "")) && !w.hidden);
}
function seedRowFor(host, ctrlW, node) {
    // the row of the widget this control belongs to (linkedWidgets owner), by name
    const owner = (node.widgets || []).find((w) => w && w.linkedWidgets && w.linkedWidgets.includes(ctrlW));
    const name = owner ? String(owner.name) : "seed";
    for (const r of host.querySelectorAll('[data-testid="node-widget"]')) {
        const t = (r.textContent || "").trim();
        if (t === name || t.startsWith(name)) return r;
    }
    return null;
}
export function applyControlRows(node, host) {
    const ctrls = controlWidgets(node);
    if (!ctrls.length) return;
    for (const c of ctrls) {
        let row = host.querySelector('.' + CTRL_CLASS + '[data-ctrl="' + String(c.name) + '"]');
        if (!row) {
            const anchor = seedRowFor(host, c, node);
            if (!anchor || !anchor.parentElement) continue;
            row = document.createElement("div");
            row.className = CTRL_CLASS;
            row.setAttribute("data-ctrl", String(c.name));
            row.setAttribute("data-testid", "uls-ctrl-row");   // NOT node-widget: the frontend's row list must stay intact
            // spans both grid columns and lays its two cells on the grid's own
            // columns (subgrid), so the label column of the other rows is untouched
            row.style.cssText = "grid-column:1 / -1;display:grid;grid-template-columns:subgrid;align-items:center;min-height:24px;box-sizing:border-box;padding-right:12px;";   // pr-3 like the frontend rows
            // the grid's first column is the frontend's 12 px gutter -- keep it empty
            const gutter = document.createElement("div");
            row.appendChild(gutter);
            const lab = document.createElement("span");
            lab.textContent = String(c.label || c.name).replace(/_/g, " ");
            lab.style.cssText = "font-size:12px;opacity:.85;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
            const sel = document.createElement("select");
            sel.style.cssText = "min-width:0;width:100%;height:24px;font-size:12px;border-radius:6px;padding:0 6px;box-sizing:border-box;" +
                "background:var(--comfy-input-bg, #222);color:var(--input-text, #ddd);border:1px solid var(--border-color, #444);";
            for (const v of (c.options && c.options.values) || CTRL_VALUES) {
                const o = document.createElement("option"); o.value = v; o.textContent = v; sel.appendChild(o);
            }
            sel.onchange = () => {
                c.value = sel.value;
                try { if (c.callback) c.callback(c.value, app.canvas, node); } catch (e) { /* never break */ }
                node.setDirtyCanvas(true, true);
            };
            row.appendChild(lab); row.appendChild(sel);
            anchor.insertAdjacentElement("afterend", row);
        }
        const sel = row.querySelector("select");
        if (sel && sel.value !== String(c.value) && document.activeElement !== sel) sel.value = String(c.value);
    }
}

export function applyTints(node, host) {
    let n = 0;
    for (const [w, t] of tintPairs(node, host)) {
        const src = w.element.style;
        for (const k of TINT_KEYS) {
            const v = src[k] || "";
            if (t.style[k] !== v) t.style[k] = v;
        }
        if (src.backgroundColor) n++;
        const h = src[HEIGHT_KEY] || "";
        if (h) {
            // v952: copy, do not max. The classic auto-fit (CLIP Text Encode _refit)
            // now measures THIS Vue field when its own element is not laid out, so
            // the classic inline height already is the Vue content height -- and a
            // max() against it could only ever grow (Frank, 12.09.: 'the field does
            // not shrink when I take text out').
            if (t.style[HEIGHT_KEY] !== h) { t.style[HEIGHT_KEY] = h; t.style.minHeight = h; }
            if (t.style.overflowY !== "hidden") t.style.overflowY = "hidden";
            // typing in the Vue field never reaches the classic element's 'input'
            // listener (that is what drives the auto-fit) -- relay it on change
            if (t._ulsLastVal !== t.value) {
                t._ulsLastVal = t.value;
                try { w.element.dispatchEvent(new Event("input")); } catch (e) { /* never break */ }
            }
        }
    }
    return n;
}

const GROWTH_STYLE_ID = "uls-vue-growth";

/** The widgets Nodes 2.0 gives a row, in its order (useProcessedWidgets: renders
 *  every typed, non-canvasOnly widget; shows it unless hidden or advanced). */
export function vueRowWidgets(node) {
    return (node.widgets || []).filter(w => {
        const o = (w && w.options) || {};
        return !!(w && w.type) && !o.canvasOnly && !o.hidden && !o.advanced;
    });
}

/** The classic growth rule on the frontend's track list; null = leave it alone. */
export function growthTracks(tracks, widgets, rows, host) {
    if (!tracks.length || tracks.length !== widgets.length || widgets.length !== rows.length) return null;
    for (let i = 0; i < widgets.length; i++) {
        const el = widgets[i].element;
        if (el && host.contains(el) && !rows[i].contains(el)) return null;
    }
    return tracks.map((t, i) => (t === "auto" && widgets[i].computeSize ? "min-content" : t));
}

/** The template this node's grid should use under Nodes 2.0 ("" = the frontend's own). */
export function growthTemplate(node, host) {
    const grid = host.querySelector('[data-testid="node-widgets"]');
    if (!grid) return "";
    const tracks = String(grid.style.gridTemplateRows || "").trim().split(/\s+/).filter(Boolean);
    const rows = [...grid.children].filter(c => c.getAttribute && c.getAttribute("data-testid") === "node-widget");
    const t = growthTracks(tracks, vueRowWidgets(node), rows, host) || tracks;
    // v954: the frontend sizes every track itself (fixed px per row); a control
    // row (see applyControlRows) would fall into an implicit track and push the
    // rows below it (Frank, 12.09.: preview box crushed). Give it its own
    // 24 px track exactly where it sits.
    const out = [];
    let ri = 0;
    for (const c of grid.children) {
        if (c.classList && c.classList.contains(CTRL_CLASS)) out.push("24px");
        else if (c.getAttribute && c.getAttribute("data-testid") === "node-widget") out.push(t[ri++] || "auto");
    }
    if (ri !== t.length) return "";       // rows and tracks disagree -- leave the frontend alone
    return out.join(" ") !== tracks.join(" ") ? out.join(" ") : "";
}

/** Write all rules in one pass; an empty map removes them (classic mode). */
export function writeGrowthRules(rules, packIds) {
    let st = document.getElementById(GROWTH_STYLE_ID);
    const esc = (id) => String(id).replace(/["\\]/g, "");
    // v950: a disabled row is dimmed like LiteGraph dims it (the Vue input is
    // already disabled, it just does not look it) -- pack nodes only. The FIELD
    // must be disabled, not a stepper button at its range limit (denoise 1.00
    // disables '+', start_at_step 0 disables '-'; measured 12.09.).
    const dim = [...(packIds || [])].map((id) =>
        `[data-node-id="${esc(id)}"] [data-testid="node-widget"]:has(input:disabled,select:disabled,textarea:disabled,[role="switch"]:disabled,[role="combobox"]:disabled){opacity:.45}`);
    const css = [...rules].map(([id, t]) =>
        `[data-node-id="${esc(id)}"] [data-testid="node-widgets"]{grid-template-rows:${t} !important}`).concat(dim).join("\n");
    if (!st) {
        if (!css) return;
        st = document.createElement("style");
        st.id = GROWTH_STYLE_ID;
        document.head.appendChild(st);
    }
    if (st.textContent !== css) st.textContent = css;
}

function allNodes() {
    const g = app.graph;
    return g ? (g._nodes || g.nodes || []) : [];
}

function tick() {
    if (!vueMode()) { writeGrowthRules(new Map()); return; }
    const rules = new Map();
    const packIds = [];
    for (const node of allNodes()) {
        if (!isPackNode(node)) continue;
        const host = vueElement(node);
        if (!host) continue;
        packIds.push(node.id);
        try {
            const t = growthTemplate(node, host);
            if (t) rules.set(node.id, t);
        } catch (e) { /* a layout rule must never break the node */ }
        if (wrapResize(node) && node.size) {
            try { node.onResize(node.size); } catch (e) { /* a clamp must never break the view */ }
        }
        try { applyTints(node, host); } catch (e) { /* styling must never break the node */ }
        try { applyControlRows(node, host); } catch (e) { /* the control row must never break the node */ }
    }
    writeGrowthRules(rules, packIds);
}

let _watch = null;
app.registerExtension({
    name: "Polyhedron.vueParity",
    setup() {
        if (!_watch) _watch = setInterval(() => { try { tick(); } catch (e) { /* never throw */ } }, POLL_MS);
    },
});
