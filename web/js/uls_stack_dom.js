/**
 * uls_stack_dom.js -- the LoRA Stack row list as DOM, for Nodes 2.0.
 *
 * WHY THIS FILE EXISTS
 * --------------------
 * ComfyUI's Vue renderer ("Nodes 2.0") never calls onDrawForeground. The
 * Stack builds its ENTIRE body there -- 545 lines of canvas painting plus
 * ~800 lines of hit-testing in onMouseDown/Move/Up -- and adds no DOM widget
 * at all. Under Nodes 2.0 the node therefore shows sockets and nothing else
 * (field check 09.09.2026): no rows, no weights, no add button. The suite's
 * central node is unusable there.
 *
 * Nodes whose UI is built from DOM widgets came through that same field check
 * intact -- Mask Editor's toolbar, Filter's A/B strip, Layers' hint. So DOM is
 * the way across, and this file is the first step of moving the Stack onto it.
 *
 * THE RULE THIS FILE FOLLOWS: ONE TRUTH, TWO VIEWS
 * ------------------------------------------------
 * State stays exactly where it has always been: `node._uls`, persisted by
 * `node._ulsSync()` into the hidden `uls_config` widget that Python reads.
 * This panel edits that object and calls that function -- it owns no state of
 * its own, invents no format, and touches no serialization. Saved workflows,
 * the backend and the widget baseline are untouched by design.
 *
 * For the same reason the LoRA picker and the weight dialog are IMPORTED from
 * uls_node.js rather than rebuilt here. They are floating overlays, so they
 * already work under both renderers. A second picker would be a second truth.
 *
 * WHY IT ONLY EXISTS UNDER THE VUE RENDERER (v934)
 * ------------------------------------------------
 * Under LiteGraph the painted UI is years matured and is what Frank works with
 * daily, and it must stay usable AT ANY TIME. v922-v928 built the panel always
 * and tried to keep it out of sight; the frontend's container around it stayed
 * clickable and made the painted node dead (field 10.09.2026). So the panel is
 * no longer hidden from the classic renderer -- under the classic renderer it
 * is not created at all. It is attached when LiteGraph.vueNodesMode reads true
 * and marked widget.hidden when it reads false again. Since v936 that logic
 * lives in uls_vue_views.js, shared by every node with a DOM view.
 *
 * SCOPE OF THIS FIRST CUT
 * -----------------------
 * The row list: enable toggle, LoRA name (opens the existing picker), weight
 * (opens the existing weight dialog), remove, and add. That is what a working
 * stack needs. Groups, DARE/Trim, ordering arrows, trigger words and the
 * header switches still live only in the painted view and come in later cuts.
 * The panel says so itself rather than pretending to be complete.
 */
import { registerVueView } from "./uls_vue_views.js";
import {
    openLoraSelect, showWeightInput, newRow, loadLoraList, getLoraList,
    applyNorm, applyNext, showGroupModePopup, openPreviewOverlay,
    // v937: parity with the painted view -- CALLED, never rebuilt
    insertRowTrigger, openStackOrderInput, checkConflicts, WEIGHT_HDR_TIP_LINES,
    // v949: group colours -- the painted row's palette, never a copy
    GROUP_COLORS,
    // v950: the Apply pill's label and tone -- the painted header's source
    APPLY_INFO,
} from "./uls_node.js";

const STACK_CLASS = "UltimateLoraStack";
const WIDGET_NAME = "uls_rows_dom";

// v936: WHEN THIS PANEL EXISTS IS DECIDED IN uls_vue_views.js.
//
// The v934 rules -- no widget under the classic renderer, attach on
// LiteGraph.vueNodesMode, widget.hidden when switching back, both serialize
// flags false, one global watch -- moved unchanged into the shared switch
// point, so the next node with a DOM view uses the same logic instead of a
// copy of it. This file keeps only what is the Stack's own: how its view is
// built (render), when a node is ready (_uls filled), and what happens on the
// way back (the painted view's own _ulsResize). See registerVueView below.

const CSS = `
/* v950: no gap between rows -- the painted row (ROW_H 28) carries its own air;
   6 px per row was exactly the Stack's +89 px over the painted node */
.uls-dom { display:flex; flex-direction:column; gap:0; padding:6px 8px;
  font:12px/1.35 system-ui,sans-serif; color:#ddd; box-sizing:border-box; }
.uls-dom-note { font-size:10px; color:#9a8fb0; padding-bottom:2px; }
.uls-dom-trig { flex:0 0 auto; background:#1e1a28; border:1px solid #3a3446;
  border-radius:4px; color:#8f8aa8; cursor:pointer; padding:1px 6px; font-size:11px; }
.uls-dom-trig:hover { border-color:#c060ff; color:#ddd; }
.uls-dom-obadge { display:inline-block; width:12px; height:12px; line-height:12px;
  margin-right:4px; border-radius:50%; border:1px dashed #f0c05066;
  font:bold 8px 'Segoe UI',Arial; text-align:center; vertical-align:middle;
  cursor:pointer; color:#1a1a2a; }
.uls-dom-obadge.set { background:#f0c050; border:1px solid #f0c050; }
.uls-dom-obadge.conflict { background:#ff4444; border:1px solid #ff4444; color:#fff; }
.uls-dom-rwarn { color:#ff7744; font-size:11px; cursor:help; }
.uls-dom-gwarn { color:#88aaee; font-size:10px; padding:1px 0; }
.uls-dom-gwarn.warn { color:#ff8844; }
.uls-dom-whdr { cursor:help; }
.uls-dom-row { display:flex; align-items:center; gap:6px; height:28px; box-sizing:border-box; }
.uls-dom-row.off { opacity:.45; }
.uls-dom-name { flex:1 1 auto; min-width:0; overflow:hidden;
  text-overflow:ellipsis; white-space:nowrap; text-align:left;
  background:#1c1a24; border:1px solid #3a3446; border-radius:4px;
  color:#ddd; padding:3px 6px; cursor:pointer; }
.uls-dom-name:hover { border-color:#c060ff; }
.uls-dom-w { flex:0 0 62px; background:#2a1f10; border:1px solid #6b4f1e;
  border-radius:4px; color:#ffb648; padding:3px 4px; cursor:pointer;
  text-align:center; font-variant-numeric:tabular-nums; }
.uls-dom-x { flex:0 0 auto; background:none; border:none; color:#a08; 
  cursor:pointer; font-size:13px; padding:0 4px; }
.uls-dom-x:hover { color:#f46; }
.uls-dom-add { align-self:center; background:none; border:1px dashed #4a4358;
  border-radius:4px; color:#9a8fb0; cursor:pointer; padding:2px 14px; }
.uls-dom-add:hover { border-color:#c060ff; color:#ddd; }
.uls-dom-head { display:flex; align-items:center; gap:6px; padding-bottom:4px; }
.uls-dom-pill { background:none; border:1px solid #6a4a8a; border-radius:10px;
  color:#c9a6e8; cursor:pointer; padding:2px 10px; font-size:10px;
  letter-spacing:.4px; }
.uls-dom-pill:hover { border-color:#c060ff; color:#fff; }
.uls-dom-pill.on { background:#3a2450; color:#e6ccff; }
.uls-dom-cols { display:flex; gap:6px; font-size:9px; color:#7d7490;
  padding:0 0 1px 26px; }
.uls-dom-cols .sp { flex:1 1 auto; }
.uls-dom-ord { display:flex; flex-direction:column; gap:1px; flex:0 0 auto; }
.uls-dom-ord button { background:none; border:none; color:#6f6885;
  cursor:pointer; font-size:8px; line-height:8px; padding:0 2px; }
.uls-dom-ord button:hover { color:#c060ff; }
.uls-dom-grp { flex:0 0 auto; background:#241f2e; border:1px solid #3a3446;
  border-radius:9px; color:#9a8fb0; cursor:pointer; font-size:9px;
  padding:2px 7px; }
.uls-dom-grp:hover { border-color:#c060ff; color:#ddd; }
.uls-dom-step { flex:0 0 auto; background:none; border:none; color:#a8762c;
  cursor:pointer; padding:0 2px; font-size:10px; }
.uls-dom-step:hover { color:#ffb648; }
.uls-dom-badge { flex:0 0 26px; height:20px; display:flex; align-items:center;
  justify-content:center; background:#2b2536; border:1px solid #453d58;
  border-radius:3px; color:#b9aecd; font-size:11px; cursor:pointer; }
.uls-dom-badge:hover { border-color:#c060ff; color:#fff; }
`;

let _cssDone = false;
/** v938: exported -- the Engine's view wears the same styles, not a copy. */
export function ensureCss() {
    if (_cssDone) return;
    _cssDone = true;
    const st = document.createElement("style");
    st.textContent = CSS;
    document.head.appendChild(st);
}

/** Rebuild the panel from node._uls. Never caches rows -- _uls is the truth. */
function render(node, root) {
    root.textContent = "";

    const uls = node._uls;

    // ── header: the two pills the painted view carries ────────────────────
    const head = document.createElement("div");
    head.className = "uls-dom-head";

    const flat = document.createElement("button");
    flat.className = "uls-dom-pill" + (uls?.flatMode ? "" : " on");
    flat.textContent = uls?.flatMode ? "FLAT" : "GROUP STACK";
    flat.title = "Group stack vs flat list";
    flat.onclick = () => { uls.flatMode = !uls.flatMode; commit(node, root); };
    head.appendChild(flat);

    const ap = document.createElement("button");
    ap.className = "uls-dom-pill";
    // v950: label and colour from APPLY_INFO, as the painted pill and the Engine
    // panel do -- "BAKED" in amber, not the raw value "PATCH" in grey
    const apInfo = APPLY_INFO[applyNorm(uls?.apply)] || { label: String(applyNorm(uls?.apply)) };
    ap.textContent = String(apInfo.label).toUpperCase();
    if (apInfo.color) {
        ap.style.color = apInfo.color;
        ap.style.background = apInfo.color + "18";
        ap.style.borderColor = apInfo.color + "88";
    }
    ap.title = apInfo.hint ? "Apply mode: " + apInfo.hint : "Apply mode: auto / bypass / patch";
    ap.onclick = () => { uls.apply = applyNext(uls.apply); commit(node, root); };
    head.appendChild(ap);
    root.appendChild(head);

    const cols = document.createElement("div");
    cols.className = "uls-dom-cols";
    const sp = document.createElement("span"); sp.className = "sp";
    cols.appendChild(sp);
    for (const t of ["Trigger", "Group", "Weight / CLIP Strength"]) {
        const c = document.createElement("span"); c.textContent = t;
        if (t.startsWith("Weight")) {
            // v937: the painted header's explainer, from the SAME lines
            c.textContent = t + " \u24d8";
            c.title = WEIGHT_HDR_TIP_LINES.join("\n");
            c.className = "uls-dom-whdr";
        }
        cols.appendChild(c);
    }
    root.appendChild(cols);

    // v950: no hint line -- the painted node has none (its text now sits on the
    // order arrows' tooltip); 16 px of the Stack's surplus over the painted node

    const rows = node._uls?.rows;
    if (!Array.isArray(rows)) return;
    // v937: the painted view's own conflict rules, not a second set
    const conflicts = checkConflicts(rows) || [];

    rows.forEach((row, i) => {
        const el = document.createElement("div");
        el.className = "uls-dom-row" + (row.enabled ? "" : " off");
        // v949: the 3 px group stripe of the painted row (grey when the row is off)
        const gc = GROUP_COLORS[row.group] || "#404050";
        el.style.borderLeft = "3px solid " + (row.enabled ? gc : "#282838");
        el.style.paddingLeft = "3px";

        const ord = document.createElement("div");
        ord.className = "uls-dom-ord";
        const up = document.createElement("button");
        up.textContent = "\u25b2"; up.title = "Move up (drag-to-reorder is classic-only)";
        up.onclick = () => {
            if (i === 0) return;
            const r = rows.splice(i, 1)[0]; rows.splice(i - 1, 0, r);
            commit(node, root);
        };
        const dn = document.createElement("button");
        dn.textContent = "\u25bc"; dn.title = "Move down (drag-to-reorder is classic-only)";
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

        // Badge: the painted view draws the LoRA preview here and falls back
        // to the first letter of the FILE name -- that is what the "M" and
        // "P" in the classic node are. Same fallback, same source.
        const badge = document.createElement("div");
        badge.className = "uls-dom-badge";
        const fileName = (row.name && row.name !== "None")
            // Benchmark drops the extension too.
            ? row.name.split(/[/\\]/).pop().replace(/\.safetensors$/i, "")
            : "";
        badge.textContent = fileName ? fileName.charAt(0).toUpperCase() : "\u00b7";
        badge.style.color = gc;           // v949: initial in the group colour, as painted
        badge.title = fileName ? "Preview / group overlay" : "";
        badge.onclick = (e) => {
            if (row.name && row.name !== "None") openPreviewOverlay(row.name, e);
        };
        el.appendChild(badge);

        const name = document.createElement("button");
        name.className = "uls-dom-name";
        // Benchmark shows the FILE name, not the folder path.
        name.textContent = fileName || "Select LoRA...";
        name.title = row.name || "";
        name.onclick = (e) => {
            // The existing floating picker -- works under both renderers.
            // v938: repaint AFTER the pick (onPicked). The v928 repaint 60 ms
            // after OPENING the picker ran before any choice was made -- the
            // panel kept the old name (measured 10.09.: state test_b, panel
            // still "Select LoRA...").
            openLoraSelect(row, getLoraList(), e, node, () => commit(node, root));
        };
        el.appendChild(name);

        const grp = document.createElement("button");
        // v937: the trigger button, left of the group pill as in the painted row
        const trig = document.createElement("button");
        trig.className = "uls-dom-trig";
        trig.textContent = "\u21b5";
        trig.title = "Insert this LoRA's trigger word at the cursor of the " +
                     "last prompt field";
        trig.onclick = (e) => insertRowTrigger(row, e);
        el.appendChild(trig);

        grp.className = "uls-dom-grp";
        // v949: pill in the group colour exactly as the painted GRP pill --
        // fill gc+"22", border gc+"55", text gc; label = first four letters, upper case
        grp.style.background = gc + "22";
        grp.style.borderColor = gc + "55";
        grp.style.color = gc;
        // v937: the stack-order badge in the pill -- gold with its number,
        // dashed when unset, red "!" while a conflict flashes; same rules as
        // the painted pill (none in flat mode, none for the no-group row)
        if (!uls.flatMode && row.group && row.group !== "\u2014") {
            const ov = (uls.groupOrder || {})[row.group];
            const hasOrder = ov !== undefined && ov !== null && ov !== "";
            const isConflict = uls._orderConflictGroup === row.group;
            const ob = document.createElement("span");
            ob.className = "uls-dom-obadge" +
                (isConflict ? " conflict" : hasOrder ? " set" : "");
            ob.textContent = isConflict ? "!" : hasOrder ? String(ov) : "";
            ob.title = "Stack order of this group -- click: 1-8, 0 = clear";
            ob.onclick = (e) => {
                e.stopPropagation();            // not the group dialog
                openStackOrderInput(node, row, e, () => render(node, root));
            };
            grp.appendChild(ob);
        }
        const gl = document.createElement("span");
        gl.textContent = row.group && row.group !== "\u2014"
            ? row.group.slice(0, 4).toUpperCase() : "GRP";
        grp.appendChild(gl);
        grp.title = "Group and its merge mode";
        grp.onclick = (e) => {
            // The existing floating group popup -- shared, not rebuilt.
            showGroupModePopup(
                row.group, uls.groupModes?.[row.group],
                uls.dare_variant, uls.groupTrim?.[row.group],
                uls.groupResolve?.[row.group],
                uls.groupTrimAmount?.[row.group], e,
                () => commit(node, root), () => commit(node, root),
                uls.apply);
        };
        el.appendChild(grp);

        const step = (dir) => {
            const b = document.createElement("button");
            b.className = "uls-dom-step";
            b.textContent = dir < 0 ? "\u25c0" : "\u25b6";
            b.title = (dir < 0 ? "Decrease" : "Increase") +
                      " weight (Shift = CLIP strength)";
            b.onclick = (e) => {
                const clip = e.shiftKey;
                const base = clip
                    ? (typeof row.wClip === "number" ? row.wClip : 1.0)
                    : (typeof row.wLow === "number" ? row.wLow : 1.0);
                const v = Math.round((base + dir * 0.05) * 100) / 100;
                if (clip) { row.wClip = v; }
                else { row.wHigh = v; row.wLow = v; }
                commit(node, root);
            };
            return b;
        };
        el.appendChild(step(-1));

        const w = document.createElement("button");
        w.className = "uls-dom-w";
        const cur = (typeof row.wLow === "number" ? row.wLow
                   : typeof row.wHigh === "number" ? row.wHigh : 1.0);
        w.textContent = Number(cur).toFixed(2);
        w.title = "Weight -- click to type a value";
        w.onclick = (e) => {
            showWeightInput(e, cur, (v) => {
                row.wHigh = v; row.wLow = v;
                commit(node, root);
            }, "Weight", "#ffb648");
        };
        el.appendChild(w);
        el.appendChild(step(+1));

        // v937: the painted row's conflict marker, with its message
        const rw = conflicts.find(c => c.row === i);
        if (rw) {
            const wm = document.createElement("span");
            wm.className = "uls-dom-rwarn";
            wm.textContent = "\u26a0";
            wm.title = rw.msg;
            el.appendChild(wm);
        }

        const x = document.createElement("button");
        x.className = "uls-dom-x";
        x.textContent = "\u00d7";
        x.title = "Remove this row";
        x.onclick = () => {
            node._uls.rows.splice(i, 1);
            if (!node._uls.rows.length) node._uls.rows.push(newRow());
            commit(node, root);
        };
        el.appendChild(x);

        root.appendChild(el);
    });

    // v937: the painted view's global warnings, under the list
    for (const w of conflicts.filter(c => c.row === -1)) {
        const gw = document.createElement("div");
        gw.className = "uls-dom-gwarn" + (w.level === "warn" ? " warn" : "");
        gw.textContent = w.msg;
        root.appendChild(gw);
    }

    const add = document.createElement("button");
    add.className = "uls-dom-add";
    add.textContent = "+";
    add.title = "Add a LoRA row";
    add.onclick = () => { node._uls.rows.push(newRow()); commit(node, root); };
    root.appendChild(add);
}

/** Persist through the node's OWN sync, then repaint. */
function commit(node, root) {
    try {
        node._ulsSync?.();
    } catch (e) {
        console.warn("[ULS] DOM panel: sync failed", e);
    }
    node.setDirtyCanvas?.(true, true);
    render(node, root);
}

registerVueView(STACK_CLASS, {
    widgetName: WIDGET_NAME,
    className: "uls-dom",
    label: "DOM panel",               // console wording unchanged since v934
    prepare() {
        ensureCss();
        void loadLoraList();
    },
    // _uls is filled by uls_node.js during its own nodeCreated/configure.
    ready: (node) => !!node._uls?.rows,
    render,
    // Nodes 2.0 grew the node around the panel; hand the height back to the
    // painted view's OWN rule (the call it makes on every row change).
    leave: (node) => node._ulsResize?.(),
});
