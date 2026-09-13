/**
 * uls_engine_dom.js -- the Polyhedron Engine's view under Nodes 2.0 (v938, S4).
 *
 * The Engine (ULSAccelerator) paints its whole body on the LiteGraph canvas,
 * so under the Vue renderer it was empty. This is its DOM view, registered with
 * the shared switch point (uls_vue_views.js): it exists only while
 * LiteGraph.vueNodesMode is true, and under the classic renderer the painted
 * Engine is untouched -- the classic node stays the benchmark.
 *
 * One truth, two views: the view reads and writes the node's OWN state
 * (node._uls, persisted by the node's own _ulsSync) and CALLS the painted
 * Engine's pieces from uls_node.js -- the S|C|D modes and their labels, the
 * weight step and weight input rules, the LoRA picker, the preview overlay,
 * the apply pill's cycle. Nothing is rebuilt here. It wears the Stack panel's
 * styles (ensureCss) plus a few of its own for the mode buttons.
 */
import { registerVueView } from "./uls_vue_views.js";
import { ensureCss } from "./uls_stack_dom.js";
import {
    openLoraSelect, loadLoraList, getLoraList, applyNorm, applyNext,
    WEIGHT_HDR_TIP_LINES, newEngineRow, APPLY_INFO, openGroupPreviewOverlay,
    ENGINE_MODES, ENGINE_MODE_LABELS, ENGINE_MODE_TIPS,
    stepEngineWeight, editEngineWeight,
} from "./uls_node.js";

const ENGINE_CLASS = "ULSAccelerator";
const WIDGET_NAME = "uls_engine_dom";

const ENGINE_CSS = `
.uls-eng-mode { flex:0 0 auto; min-width:30px; background:#1e1a28; border:1px solid #3a3446;
  border-radius:4px; color:#8f8aa8; cursor:pointer; padding:2px 8px; font:bold 11px 'Segoe UI',Arial; }
.uls-eng-mode.on { background:#2a2438; }
.uls-eng-line { font-size:10px; padding:0 0 2px 0; }
.uls-dom-w.uls-eng-w { flex:0 0 auto; min-width:62px; white-space:nowrap; }
`;
let _engCss = false;
function ensureEngineCss() {
    if (_engCss) return;
    _engCss = true;
    const st = document.createElement("style");
    st.textContent = ENGINE_CSS;
    document.head.appendChild(st);
}

function fileNameOf(row) {
    return (row.name && row.name !== "None")
        ? row.name.split(/[/\\]/).pop().replace(/\.safetensors$/i, "") : "";
}

/** Persist through the node's OWN sync, then repaint the view. */
function commit(node, root) {
    try {
        node._ulsSync?.();
    } catch (e) {
        console.warn("[ULS] Engine panel: sync failed", e);
    }
    node.setDirtyCanvas?.(true, true);
    renderEngine(node, root);
}

/** Rebuild the view from node._uls. Never caches rows -- _uls is the truth. */
function renderEngine(node, root) {
    root.textContent = "";
    const uls = node._uls;
    if (!uls) return;
    const mode = uls.mode || "SEQ";

    // ── header: S | C | D, the DARE variant (in DARE only), the Apply pill ──
    const head = document.createElement("div");
    head.className = "uls-dom-head";
    for (const m of ENGINE_MODES) {
        const b = document.createElement("button");
        b.className = "uls-eng-mode" + (m.key === mode ? " on" : "");
        b.textContent = m.letter;
        const tip = ENGINE_MODE_TIPS[m.key];      // the painted hover tooltip's text
        b.title = tip ? tip.label + " \u2014 " + tip.hint : m.key;
        if (m.key === mode) { b.style.color = m.color; b.style.borderColor = m.color; }
        b.onclick = () => { uls.mode = m.key; commit(node, root); };
        head.appendChild(b);
    }
    if (mode === "DARE") {
        const dv = document.createElement("button");
        dv.className = "uls-dom-pill uls-eng-dv";
        const variant = uls.dareVariant || "channel";
        dv.textContent = variant === "channel" ? "CHAN" : "ELEM";
        dv.title = "DARE variant: channel / element";
        dv.onclick = () => {
            uls.dareVariant = (uls.dareVariant === "channel") ? "element" : "channel";
            commit(node, root);
        };
        head.appendChild(dv);
    }
    const ap = document.createElement("button");
    ap.className = "uls-dom-pill uls-eng-apply";
    const info = APPLY_INFO[applyNorm(uls.apply)] || { label: String(applyNorm(uls.apply)) };
    ap.textContent = String(info.label).toUpperCase();
    if (info.color) ap.style.color = info.color;
    ap.title = "Apply mode: auto / bypass / patch";
    ap.onclick = () => { uls.apply = applyNext(uls.apply); commit(node, root); };
    head.appendChild(ap);
    root.appendChild(head);

    const line = document.createElement("div");
    line.className = "uls-eng-line";
    const cur = ENGINE_MODES.find(m => m.key === mode);
    line.textContent = "\u25b8 " + (ENGINE_MODE_LABELS[mode] || mode);
    if (cur) line.style.color = cur.color;
    root.appendChild(line);

    const cols = document.createElement("div");
    cols.className = "uls-dom-cols";
    const sp = document.createElement("span"); sp.className = "sp";
    cols.appendChild(sp);
    const wh = document.createElement("span");
    wh.className = "uls-dom-whdr";
    wh.textContent = "Weight / CLIP Strength \u24d8";
    wh.title = WEIGHT_HDR_TIP_LINES.join("\n");
    cols.appendChild(wh);
    root.appendChild(cols);

    const rows = uls.rows;
    if (!Array.isArray(rows)) return;
    rows.forEach((row, i) => {
        const el = document.createElement("div");
        el.className = "uls-dom-row";

        const ord = document.createElement("div");
        ord.className = "uls-dom-ord";
        const up = document.createElement("button");
        up.textContent = "\u25b2"; up.title = "Move up";
        up.onclick = () => {
            if (i === 0) return;
            const r = rows.splice(i, 1)[0]; rows.splice(i - 1, 0, r);
            commit(node, root);
        };
        const dn = document.createElement("button");
        dn.textContent = "\u25bc"; dn.title = "Move down";
        dn.onclick = () => {
            if (i >= rows.length - 1) return;
            const r = rows.splice(i, 1)[0]; rows.splice(i + 1, 0, r);
            commit(node, root);
        };
        ord.appendChild(up); ord.appendChild(dn);
        el.appendChild(ord);

        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.checked = !!row.enabled;
        cb.title = "Enable / bypass this LoRA";
        cb.onchange = () => { row.enabled = cb.checked; commit(node, root); };
        el.appendChild(cb);

        const fname = fileNameOf(row);
        const badge = document.createElement("div");
        badge.className = "uls-dom-badge";
        badge.textContent = fname ? fname.charAt(0).toUpperCase() : "\u00b7";
        badge.title = fname ? "Preview" : "";
        badge.onclick = (e) => {
            if (fname) openGroupPreviewOverlay(row, e, null);   // as the painted thumb
        };
        el.appendChild(badge);

        const name = document.createElement("button");
        name.className = "uls-dom-name";
        name.textContent = fname || "Select LoRA...";
        name.title = row.name || "";
        name.onclick = (e) => {
            openLoraSelect(row, getLoraList(), e, node, () => commit(node, root));
        };
        el.appendChild(name);

        const dec = document.createElement("button");
        dec.className = "uls-dom-step";
        dec.textContent = "\u25c0";
        dec.title = "-0.05 (Shift: CLIP strength)";
        dec.onclick = (e) => { stepEngineWeight(row, -1, !!e.shiftKey); commit(node, root); };
        el.appendChild(dec);

        const w = document.createElement("button");
        w.className = "uls-dom-w uls-eng-w";   // v939: "1.00 / 0.80" on one line
        const wt = (row.weight || 0).toFixed(2);
        w.textContent = (typeof row.wClip === "number")
            ? wt + " / " + row.wClip.toFixed(2) : wt;
        w.title = "Click: weight.  Shift+Click: CLIP strength";
        w.onclick = (e) => {
            editEngineWeight(row, e, !!e.shiftKey, () => commit(node, root));
        };
        el.appendChild(w);

        const inc = document.createElement("button");
        inc.className = "uls-dom-step";
        inc.textContent = "\u25b6";
        inc.title = "+0.05 (Shift: CLIP strength)";
        inc.onclick = (e) => { stepEngineWeight(row, +1, !!e.shiftKey); commit(node, root); };
        el.appendChild(inc);

        const x = document.createElement("button");
        x.className = "uls-dom-x";
        x.textContent = "\u00d7";
        x.title = "Remove this row";
        x.onclick = () => {
            rows.splice(i, 1);
            if (!rows.length) rows.push(newEngineRow());
            commit(node, root);
        };
        el.appendChild(x);

        root.appendChild(el);
    });

    const add = document.createElement("button");
    add.className = "uls-dom-add";
    add.textContent = "+";
    add.title = "Add a LoRA row";
    add.onclick = () => { rows.push(newEngineRow()); commit(node, root); };
    root.appendChild(add);
}

registerVueView(ENGINE_CLASS, {
    widgetName: WIDGET_NAME,
    className: "uls-dom",
    label: "Engine panel",
    prepare() {
        ensureCss();
        ensureEngineCss();
        void loadLoraList();
    },
    // _uls is filled by the Engine's own onNodeCreated/onConfigure.
    ready: (node) => !!node._uls?.rows && node._uls.isEngine === true,
    render: renderEngine,
    // Nodes 2.0 grew the node around the view; hand the height back to the
    // painted Engine's OWN rule.
    leave: (node) => node._engineResize?.(),
});
