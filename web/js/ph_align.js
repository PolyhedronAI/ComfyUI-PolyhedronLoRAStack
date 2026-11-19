/**
 * ph_align.js -- v1333, reworked in v1334
 *
 * Polyhedron Align: a floating bar that aligns, sizes and distributes the
 * selected NODES AND GROUPS (Frank 15.11.: "markiert man zwei Gruppen und
 * drueckt Align ... das geht aktuell nicht"). A group moves with everything
 * inside it, so aligning two groups aligns two whole blocks of the graph.
 *
 * Own implementation under the Suite's MIT licence. The idea of a node align
 * bar comes from ComfyUI-NodeAligner (Tenney95, GPL-3.0); no code, icon or
 * style of it is used here -- the geometry below and the icons are written
 * from scratch.
 *
 * THE BAR (v1334, Frank 15.11.: "als einzelnes Float-Menue ... mit sauberer
 * Standard-Leiste ... laed, wenn man einen kleinen Custom-Node-Baustein
 * reinlaedt, eben als wuerde man eine Note-Node setzen ... nur, wenn man es
 * will"): one fixed floating bar of square buttons, shown for as long as the
 * graph holds a "Polyhedron Align" node that runs (not muted, not bypassed).
 * No node, no bar; v1333's bar that popped up on a selection is gone. The
 * Suite's hexagon at its front left is the drag handle; the bar remembers
 * where it was put (localStorage, per browser); the node's "reset" button
 * puts it back. A button that cannot act on the current selection is dimmed.
 *
 *   align      left | centre | right       top | middle | bottom
 *   size       equal width | equal height  (the largest of the selection)
 *   distribute horizontally | vertically   (equal gaps, needs three)
 *   group      a new Polyhedron group around the selection
 *              selected groups: Polyhedron group on / off
 *
 * WHAT MOVES WITH A GROUP: every node whose centre lies in it (the rule the
 * Group Switch counts by, ph_groups_core.memberNodes) and every group lying
 * wholly inside it. A selected node or group that already moves with a
 * selected group is not aligned a second time. The group frame moves with
 * skipChildren, the members are moved here -- one rule, not the frontend's
 * children list plus ours.
 */
import { app } from "../../scripts/app.js";
import { boundsOf, containsRect, setPhGroup, isPhGroup } from "./ph_groups_core.js";
import { addSuiteBadge, patchFrontendDef } from "./ph_group_switch.js";

export const ALIGN_TYPE = "PolyhedronAlign";
export const ALIGN_TITLE = "\u2B21 Polyhedron Align";

export const OPS = ["left", "hcenter", "right", "top", "vcenter", "bottom", "width", "height", "hdist", "vdist"];

// Pure: new rects for op. rects: [{x, y, w, h}]; returns new objects in the
// same order (unchanged copies where op does nothing).
export function planAlign(rects, op) {
    const out = rects.map((r) => ({ x: r.x, y: r.y, w: r.w, h: r.h }));
    if (out.length < 2) return out;
    const minX = Math.min(...out.map((r) => r.x));
    const maxX = Math.max(...out.map((r) => r.x + r.w));
    const minY = Math.min(...out.map((r) => r.y));
    const maxY = Math.max(...out.map((r) => r.y + r.h));
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    switch (op) {
        case "left": out.forEach((r) => { r.x = minX; }); break;
        case "right": out.forEach((r) => { r.x = maxX - r.w; }); break;
        case "hcenter": out.forEach((r) => { r.x = cx - r.w / 2; }); break;
        case "top": out.forEach((r) => { r.y = minY; }); break;
        case "bottom": out.forEach((r) => { r.y = maxY - r.h; }); break;
        case "vcenter": out.forEach((r) => { r.y = cy - r.h / 2; }); break;
        case "width": { const w = Math.max(...out.map((r) => r.w)); out.forEach((r) => { r.w = w; }); break; }
        case "height": { const h = Math.max(...out.map((r) => r.h)); out.forEach((r) => { r.h = h; }); break; }
        case "hdist":
        case "vdist": {
            if (out.length < 3) break;
            const H = op === "hdist";
            const pos = H ? "x" : "y", len = H ? "w" : "h";
            const order = out.map((r, i) => i).sort((a, b) => out[a][pos] - out[b][pos] || a - b);
            const first = out[order[0]];
            const last = out[order[order.length - 1]];
            const span = last[pos] + last[len] - first[pos];
            const sum = order.reduce((s, i) => s + out[i][len], 0);
            const gap = (span - sum) / (order.length - 1);
            let at = first[pos];
            for (const i of order) { out[i][pos] = at; at += out[i][len] + gap; }
            break;
        }
        default: break;
    }
    return out;
}

// Pure: which selected items get aligned. items: [{kind: "node"|"group",
// rect: [x, y, w, h], ref}]; a node whose centre lies in a selected group, or
// a group wholly inside a selected group, moves with that group and drops out.
export function topLevel(items) {
    const groups = items.filter((it) => it.kind === "group");
    const inside = (it) => groups.some((g) => {
        if (g === it) return false;
        if (it.kind === "group") return containsRect(g.rect, it.rect);
        const [x, y, w, h] = it.rect;
        const cx = x + w / 2, cy = y + h / 2;
        const [gx, gy, gw, gh] = g.rect;
        return cx >= gx && cx <= gx + gw && cy >= gy && cy <= gy + gh;
    });
    return items.filter((it) => !inside(it));
}

// Pure: what moves with a group (graph = {_nodes, _groups}): every node
// whose centre lies in it -- the Switch's centre rule, but here the Switch
// node and the frontend's virtual nodes move along too -- and every group
// wholly inside it.
export function groupMembers(graph, group, titleH) {
    const [gx, gy, gw, gh] = boundsOf(group);
    const th = Number(titleH) || 30;
    const nodes = (graph._nodes || []).filter((n) => {
        if (!n || !n.pos || !n.size) return false;
        const cx = n.pos[0] + n.size[0] / 2;
        const cy = n.pos[1] - th + (n.size[1] + th) / 2;
        return cx >= gx && cx <= gx + gw && cy >= gy && cy <= gy + gh;
    });
    const b = [gx, gy, gw, gh];
    return { nodes, groups: (graph._groups || []).filter((g) => g !== group && containsRect(b, boundsOf(g))) };
}

function titleH() {
    return Number((globalThis.LiteGraph || {}).NODE_TITLE_HEIGHT) || 30;
}

function isGroup(o) {
    const G = globalThis.LiteGraph && globalThis.LiteGraph.LGraphGroup;
    return !!o && ((G && o instanceof G) || (!!(o._bounding || o.bounding) && !o.inputs && typeof o.move === "function"));
}

// The selection as items (nodes with their title bar, groups with their frame).
export function itemsOf(selected, TH) {
    const th = Number(TH) || 30;
    const items = [];
    for (const o of selected || []) {
        if (isGroup(o)) items.push({ kind: "group", rect: boundsOf(o), ref: o });
        else if (o && o.pos && o.size && o.type !== undefined) {
            const collapsed = o.flags && o.flags.collapsed;
            const w = collapsed && o._collapsed_width ? o._collapsed_width : o.size[0];
            const h = collapsed ? 0 : o.size[1];
            items.push({ kind: "node", rect: [o.pos[0], o.pos[1] - th, w, h + th], ref: o });
        }
    }
    return items;
}

function selection() {
    const c = app.canvas;
    if (!c) return [];
    if (c.selectedItems && typeof c.selectedItems[Symbol.iterator] === "function") return Array.from(c.selectedItems);
    return Object.values(c.selected_nodes || {});
}

function changed() {
    try { app.graph.setDirtyCanvas(true, true); } catch (e) { /* ignore */ }
    try { if (app.graph.change) app.graph.change(); } catch (e) { /* ignore */ }
    try {
        const t = app.extensionManager && app.extensionManager.workflow
            && app.extensionManager.workflow.activeWorkflow
            && app.extensionManager.workflow.activeWorkflow.changeTracker;
        if (t && t.checkState) t.checkState();
    } catch (e) { /* ignore */ }
}

function moveGroupBlock(graph, group, dx, dy, moved) {
    if (!dx && !dy) return;
    const th = titleH();
    const m = groupMembers(graph, group, th);
    if (!moved.has(group)) { moved.add(group); group.move(dx, dy, true); }
    for (const g of m.groups) if (!moved.has(g)) { moved.add(g); g.move(dx, dy, true); }
    for (const n of m.nodes) if (!moved.has(n)) { moved.add(n); n.pos = [n.pos[0] + dx, n.pos[1] + dy]; }
}

// Apply op to the current selection; returns how many items changed.
export function runAlign(op) {
    const graph = (app.canvas && app.canvas.graph) || app.graph;
    if (!graph) return 0;
    const th = titleH();
    const items = topLevel(itemsOf(selection(), th));
    if (items.length < 2) return 0;
    const plan = planAlign(items.map((it) => ({ x: it.rect[0], y: it.rect[1], w: it.rect[2], h: it.rect[3] })), op);
    const moved = new Set();
    let n = 0;
    items.forEach((it, i) => {
        const p = plan[i];
        const dx = p.x - it.rect[0], dy = p.y - it.rect[1];
        const dw = p.w - it.rect[2], dh = p.h - it.rect[3];
        if (!dx && !dy && !dw && !dh) return;
        n++;
        if (it.kind === "group") {
            moveGroupBlock(graph, it.ref, dx, dy, moved);
            if (dw || dh) {
                const b = boundsOf(it.ref);
                if (typeof it.ref.setBounds === "function") it.ref.setBounds([b[0], b[1], b[2] + dw, b[3] + dh]);
                else { it.ref._size[0] += dw; it.ref._size[1] += dh; }
            }
        } else {
            const node = it.ref;
            if (!moved.has(node)) { moved.add(node); node.pos = [node.pos[0] + dx, node.pos[1] + dy]; }
            if (dw || dh) {
                const min = typeof node.computeSize === "function" ? node.computeSize() : [0, 0];
                const w = Math.max(node.size[0] + dw, min[0] || 0);
                const h = Math.max(node.size[1] + dh, min[1] || 0);
                if (typeof node.setSize === "function") node.setSize([w, h]); else node.size = [w, h];
            }
        }
    });
    if (n) changed();
    return n;
}

// A new Polyhedron group around the selection (the frontend's resizeTo,
// patched in ph_group_title.js to leave the bottom bar free).
export function groupAroundSelection() {
    const LG = globalThis.LiteGraph;
    const graph = (app.canvas && app.canvas.graph) || app.graph;
    const sel = selection();
    if (!LG || !LG.LGraphGroup || !graph || !sel.length) return null;
    const g = new LG.LGraphGroup("Group");
    setPhGroup(g, true);
    graph.add(g);
    g.resizeTo(sel, 10);
    changed();
    return g;
}

// Selected groups: all Polyhedron -> all normal, otherwise all Polyhedron.
export function toggleSelectedGroups(toggle) {
    const gs = selection().filter(isGroup);
    if (!gs.length) return 0;
    const on = !gs.every(isPhGroup);
    for (const g of gs) if (isPhGroup(g) !== on) toggle(g);
    return gs.length;
}

// ---------------------------------------------------------------- the bar

const ICON = {
    // own drawings: a guide line and the boxes it lines up
    left: '<path d="M3 2v16"/><rect x="5" y="4" width="11" height="4"/><rect x="5" y="11" width="7" height="4"/>',
    hcenter: '<path d="M10 2v16"/><rect x="4" y="4" width="12" height="4"/><rect x="6" y="11" width="8" height="4"/>',
    right: '<path d="M17 2v16"/><rect x="4" y="4" width="11" height="4"/><rect x="8" y="11" width="7" height="4"/>',
    top: '<path d="M2 3h16"/><rect x="4" y="5" width="4" height="11"/><rect x="11" y="5" width="4" height="7"/>',
    vcenter: '<path d="M2 10h16"/><rect x="4" y="4" width="4" height="12"/><rect x="11" y="6" width="4" height="8"/>',
    bottom: '<path d="M2 17h16"/><rect x="4" y="4" width="4" height="11"/><rect x="11" y="8" width="4" height="7"/>',
    width: '<rect x="3" y="4" width="14" height="4"/><rect x="3" y="12" width="14" height="4"/><path d="M1 2v16M19 2v16"/>',
    height: '<rect x="4" y="3" width="4" height="14"/><rect x="12" y="3" width="4" height="14"/><path d="M2 1h16M2 19h16"/>',
    hdist: '<rect x="2" y="6" width="3" height="8"/><rect x="8.5" y="6" width="3" height="8"/><rect x="15" y="6" width="3" height="8"/><path d="M5 10h3.5M11.5 10h3.5" stroke-dasharray="1.5 1"/>',
    vdist: '<rect x="6" y="2" width="8" height="3"/><rect x="6" y="8.5" width="8" height="3"/><rect x="6" y="15" width="8" height="3"/><path d="M10 5v3.5M10 11.5v3.5" stroke-dasharray="1.5 1"/>',
    group: '<rect x="2" y="2" width="16" height="16" rx="1" stroke-dasharray="2 1.5"/><rect x="5" y="6" width="4" height="4"/><rect x="11" y="9" width="4" height="5"/>',
    phtoggle: '<path d="M10 2.5l6.5 3.75v7.5L10 17.5l-6.5-3.75v-7.5z"/><path d="M7 10h6M10 7v6"/>',
};

const TIPS = {
    left: "Align left", hcenter: "Align centre", right: "Align right",
    top: "Align top", vcenter: "Align middle", bottom: "Align bottom",
    width: "Equal width (the widest)", height: "Equal height (the tallest)",
    hdist: "Distribute horizontally (equal gaps, three or more)",
    vdist: "Distribute vertically (equal gaps, three or more)",
    group: "New Polyhedron group around the selection",
    phtoggle: "Selected groups: Polyhedron group on / off",
};

export const BAR_GROUPS = [["left", "hcenter", "right"], ["top", "vcenter", "bottom"], ["width", "height"],
                           ["hdist", "vdist"], ["group", "phtoggle"]];

// v1335 (Frank 15.11.: "wo ist jetzt die Bar?"): v1334 pinned the bar to the
// WINDOW at 240/50 -- right under the tab row, where a taller top menu (the
// Manager row in a newer frontend) covers it. The bar now lives on the
// CANVAS: its place is kept relative to the canvas' box and always pulled
// back inside it; the default is the top middle of the canvas. A new key, so
// a v1334 window position is not read as a canvas offset.
const POS_KEY = "ph_align_bar_pos_canvas";
const EDGE = 10;     // the default gap to the canvas' top edge

// Pure: the bar's screen position. canvas = the canvas' box {left, top, width,
// height}; bar = {w, h}; stored = {dx, dy} relative to the canvas or null.
export function barPlace(canvas, bar, stored) {
    const c = canvas || { left: 0, top: 0, width: 800, height: 600 };
    const w = bar.w || 0, h = bar.h || 0;
    const dx = stored ? stored.dx : (c.width - w) / 2;
    const dy = stored ? stored.dy : EDGE;
    const cx = Math.max(0, Math.min(Math.max(0, c.width - w), dx));
    const cy = Math.max(0, Math.min(Math.max(0, c.height - h), dy));
    return { x: Math.round(c.left + cx), y: Math.round(c.top + cy) };
}

// Measured in both frontends (1.49.6, 1.53.10): the canvas element spans the
// whole window and the tab row, the menus and the Manager row lie OVER it. So
// the canvas' box alone is not the free area: freeTop walks down from the
// canvas' top until a row is the canvas itself (or our bar) at the bar's left,
// middle and right -- whatever menus a frontend stacks there, the bar starts
// below them.
// Pure: hit(x, y) = true where the canvas (or our bar) is the topmost element.
export function freeTop(box, xs, hit, step = 4) {
    const limit = Math.min(box.height / 2, 400);
    for (let dy = 0; dy <= limit; dy += step) {
        if (xs.every((x) => hit(x, box.top + dy))) return dy;
    }
    return 0;
}

function canvasBox(bar) {
    const el = app.canvas && app.canvas.canvas;
    if (!el || !el.getBoundingClientRect) return null;
    const r = el.getBoundingClientRect();
    if (!(r.width > 0 && r.height > 0)) return null;
    let top = 0;
    try {
        const w = bar ? (bar.getBoundingClientRect().width || 450) : 450;
        const mid = r.left + r.width / 2;
        const xs = [mid - w / 2 + 4, mid, mid + w / 2 - 4];     // where the bar sits by default
        const hit = (x, y) => {
            const e = document.elementFromPoint(x, y);
            return !!e && (e === el || (bar && bar.contains(e)));
        };
        top = freeTop({ top: r.top, height: r.height }, xs, hit);
    } catch (e) { top = 0; }
    return { left: r.left, top: r.top + top, width: r.width, height: r.height - top };
}
const HEX_SVG = '<svg width="18" height="20" viewBox="0 0 18 20"><path d="M9 1.5l7 4v9l-7 4-7-4v-9z" fill="none" stroke="#9fb4ff" stroke-width="1.6"/></svg>';

// Pure: the bar shows while the graph holds a Polyhedron Align node that runs.
// v1336 (Frank 15.11.: "in der Node einen On/Off-Toggle"): the node's own
// switch, properties.ph_bar (saved with the workflow, default on), turns the
// bar off without muting or deleting the node.
export function alignWanted(nodes) {
    return (nodes || []).some((n) => n && n.type === ALIGN_TYPE && !(n.mode === 2 || n.mode === 4) && barOn(n));
}

export function barOn(node) {
    return !(node && node.properties && node.properties.ph_bar === false);
}

// Pure: which buttons can act on the selection items (itemsOf).
export function enabledOps(items) {
    const top = topLevel(items || []);
    const on = new Set();
    if (top.length >= 2) for (const op of ["left", "hcenter", "right", "top", "vcenter", "bottom", "width", "height"]) on.add(op);
    if (top.length >= 3) { on.add("hdist"); on.add("vdist"); }
    if ((items || []).length >= 1) on.add("group");
    if ((items || []).some((it) => it.kind === "group")) on.add("phtoggle");
    return on;
}

function svg(body) {
    return '<svg width="18" height="18" viewBox="0 0 20 20" fill="none" stroke="#d6d9e2" stroke-width="1.3">' + body + "</svg>";
}

function readPos() {
    try { const p = JSON.parse(localStorage.getItem(POS_KEY) || "null"); if (p && isFinite(p.dx) && isFinite(p.dy)) return p; } catch (e) { /* none */ }
    return null;
}

function writePos(p) {
    try { if (p) localStorage.setItem(POS_KEY, JSON.stringify(p)); else localStorage.removeItem(POS_KEY); } catch (e) { /* not kept */ }
}

let BAR = null;

function place(bar) {
    const r = bar.getBoundingClientRect();
    const p = barPlace(canvasBox(bar), { w: r.width || 450, h: r.height || 38 }, readPos());
    bar.style.left = p.x + "px";
    bar.style.top = p.y + "px";
}

export function resetBarPosition() {
    writePos(null);
    if (BAR) place(BAR);
}

function buildBar(toggle) {
    const bar = document.createElement("div");
    bar.className = "ph-align-bar";
    bar.style.cssText = "position:fixed;z-index:1000;display:none;align-items:center;gap:3px;padding:4px 5px 4px 3px;"
        + "background:#202024;border:1px solid #3c3c44;border-radius:7px;box-shadow:0 3px 12px rgba(0,0,0,.45);"
        + "user-select:none;font:12px Arial,sans-serif";
    const handle = document.createElement("div");
    handle.className = "ph-align-handle";
    handle.title = "Polyhedron Align -- drag to move";
    handle.style.cssText = "cursor:grab;width:22px;height:28px;display:flex;align-items:center;justify-content:center;margin-right:2px";
    handle.innerHTML = HEX_SVG;
    bar.appendChild(handle);
    BAR_GROUPS.forEach((ops, gi) => {
        if (gi) {
            const sep = document.createElement("div");
            sep.className = "ph-align-sep";
            sep.style.cssText = "width:1px;height:20px;background:#4a4a55;margin:0 3px";
            bar.appendChild(sep);
        }
        for (const op of ops) {
            const b = document.createElement("button");
            b.type = "button";
            b.dataset.op = op;
            b.title = TIPS[op];
            b.innerHTML = svg(ICON[op]);
            b.style.cssText = "width:28px;height:28px;background:#33333a;border:1px solid #44444e;border-radius:4px;"
                + "padding:0;cursor:pointer;display:flex;align-items:center;justify-content:center";
            b.onmouseenter = () => { if (!b.disabled) b.style.background = "#2c3e5a"; };
            b.onmouseleave = () => { b.style.background = "#33333a"; };
            b.onpointerdown = (e) => e.stopPropagation();
            b.onclick = (e) => {
                e.stopPropagation();
                if (b.disabled) return;
                try {
                    if (op === "group") groupAroundSelection();
                    else if (op === "phtoggle") toggleSelectedGroups(toggle);
                    else runAlign(op);
                } catch (err) { console.warn("[PLS] ph_align:", err); }
            };
            bar.appendChild(b);
        }
    });

    // the hexagon drags the bar
    handle.addEventListener("pointerdown", (e) => {
        e.preventDefault();
        e.stopPropagation();
        const r = bar.getBoundingClientRect();
        const ox = e.clientX - r.left, oy = e.clientY - r.top;
        handle.style.cursor = "grabbing";
        const box = canvasBox(bar) || { left: 0, top: 0, width: window.innerWidth, height: window.innerHeight };
        const mv = (ev) => {
            const p = barPlace(box, { w: r.width, h: r.height }, { dx: ev.clientX - ox - box.left, dy: ev.clientY - oy - box.top });
            bar.style.left = p.x + "px"; bar.style.top = p.y + "px";
        };
        const up = () => {
            handle.style.cursor = "grab";
            window.removeEventListener("pointermove", mv);
            window.removeEventListener("pointerup", up);
            const rr = bar.getBoundingClientRect();
            writePos({ dx: rr.left - box.left, dy: rr.top - box.top });
        };
        window.addEventListener("pointermove", mv);
        window.addEventListener("pointerup", up);
    });
    return bar;
}

function setEnabled(bar, on) {
    for (const b of bar.querySelectorAll("button[data-op]")) {
        const en = on.has(b.dataset.op);
        if (b.disabled === !en) continue;
        b.disabled = !en;
        b.style.opacity = en ? "1" : "0.35";
        b.style.cursor = en ? "pointer" : "default";
        if (!en) b.style.background = "#33333a";
    }
}

function graphNodes() {
    const out = [];
    const seen = new Set();
    for (const g of [app.canvas && app.canvas.graph, app.graph]) {
        if (!g || seen.has(g)) continue;
        seen.add(g);
        for (const n of g._nodes || []) out.push(n);
    }
    return out;
}

// ---------------------------------------------------------------- the node

function markChanged(node) {
    try { node.setDirtyCanvas(true, true); } catch (e) { /* ignore */ }
    try { if (app.graph && app.graph.change) app.graph.change(); } catch (e) { /* ignore */ }
    try {
        const t = app.extensionManager && app.extensionManager.workflow
            && app.extensionManager.workflow.activeWorkflow
            && app.extensionManager.workflow.activeWorkflow.changeTracker;
        if (t && t.checkState) t.checkState();
    } catch (e) { /* ignore */ }
}

const NODE_BODY = "#1a1a2a";     // ph_palette.js: the Suite's node colours
const NODE_HEAD = "#333";

export function makeAlignClass(LG) {
    class AlignNode extends LG.LGraphNode {
        constructor(title) {
            super(title || ALIGN_TITLE);
            this.isVirtualNode = true;
            this.serialize_widgets = false;
            this.color = NODE_HEAD;
            this.bgcolor = NODE_BODY;
            this.size = [250, 52];
            this.properties = Object.assign({ ph_bar: true }, this.properties || {});
            this._phHit = null;
            this._phToggle = null;
            addSuiteBadge(this);
        }

        computeSize() { return [Math.max(this.size ? this.size[0] : 250, 210), 52]; }

        onDrawForeground(ctx) {
            if (this.flags && this.flags.collapsed) return;
            const muted = this.mode === 2 || this.mode === 4;
            const on = barOn(this);
            ctx.save();
            ctx.font = "12px Arial";
            ctx.textBaseline = "middle";
            ctx.textAlign = "left";
            // the toggle: a pill with a knob, green on / grey off
            const tx = 10, ty = 8, tw = 30, th = 16;
            ctx.fillStyle = on ? "#3f8f4a" : "#4a4a52";
            ctx.beginPath();
            if (ctx.roundRect) ctx.roundRect(tx, ty, tw, th, th / 2); else ctx.rect(tx, ty, tw, th);
            ctx.fill();
            ctx.fillStyle = "#e8e8e8";
            ctx.beginPath();
            ctx.arc(on ? tx + tw - th / 2 : tx + th / 2, ty + th / 2, th / 2 - 2, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = muted || !on ? "#8a8a8a" : "#d6d9e2";
            ctx.fillText(!on ? "Align bar: off" : (muted ? "Align bar: off (node muted / bypassed)" : "Align bar: on"),
                         tx + tw + 8, ty + th / 2);
            this._phToggle = { x: tx - 2, y: ty - 2, w: 200, h: th + 4 };
            const bw = 120, bx = 10, by = 28;
            ctx.fillStyle = "#2c3e5a";
            ctx.fillRect(bx, by, bw, 18);
            ctx.fillStyle = "#dddddd";
            ctx.fillText("reset bar position", bx + 8, by + 9);
            ctx.restore();
            this._phHit = { x: bx, y: by, w: bw, h: 18 };
        }

        onMouseDown(e, pos) {
            const h = this._phHit;
            if (h && pos[0] >= h.x && pos[0] <= h.x + h.w && pos[1] >= h.y && pos[1] <= h.y + h.h) {
                resetBarPosition();
                return true;
            }
            const t = this._phToggle;
            if (t && pos[0] >= t.x && pos[0] <= t.x + t.w && pos[1] >= t.y && pos[1] <= t.y + t.h) {
                this.properties.ph_bar = !barOn(this);
                markChanged(this);
                return true;
            }
            return false;
        }
    }
    AlignNode.title = ALIGN_TITLE;
    AlignNode.category = "Polyhedron/Utility";
    AlignNode.description = "Shows the Polyhedron Align bar while this node is in the graph and its switch is on "
        + "(the switch, mute or bypass hide the bar, deleting the node removes it). Aligns, sizes and distributes the selected nodes and groups -- "
        + "a group moves with everything inside it. Frontend only, never runs.";
    return AlignNode;
}

app.registerExtension({
    name: "polyhedron.align",
    beforeRegisterVueAppNodeDefs(defs) {
        try { patchFrontendDef(defs, ALIGN_TYPE, ALIGN_TITLE); } catch (e) { /* the badge stays the frontend's */ }
    },
    registerCustomNodes() {
        const LG = globalThis.LiteGraph;
        if (!LG || !LG.LGraphNode || !LG.registerNodeType) {
            console.warn("[PLS] ph_align: LiteGraph not reachable, node not registered");
            return;
        }
        if (LG.registered_node_types && LG.registered_node_types[ALIGN_TYPE]) return;
        LG.registerNodeType(ALIGN_TYPE, makeAlignClass(LG));
    },
    async setup() {
        try {
            const mod = await import("./ph_group_title.js");
            const toggle = typeof mod.togglePhGroup === "function" ? mod.togglePhGroup : (g) => setPhGroup(g, !isPhGroup(g));
            const bar = buildBar(toggle);
            BAR = bar;
            document.body.appendChild(bar);
            window.addEventListener("resize", () => { if (bar.style.display !== "none") place(bar); });
            let shown = false;
            setInterval(() => {
                let want = false;
                try { want = alignWanted(graphNodes()); } catch (e) { want = false; }
                if (want !== shown) {
                    shown = want;
                    if (want) { bar.style.display = "flex"; place(bar); } else bar.style.display = "none";
                }
                if (shown) {
                    try { setEnabled(bar, enabledOps(itemsOf(selection(), titleH()))); } catch (e) { /* keep */ }
                }
            }, 200);
        } catch (e) {
            console.warn("[PLS] ph_align: not installed", e);
        }
    },
});
