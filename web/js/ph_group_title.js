/**
 * ph_group_title.js -- v1330, reworked in v1331
 *
 * Groups look exactly like the frontend's own groups -- the title bar in the
 * group's colour -- with two additions:
 *
 *  1. A title wider than its group wraps at spaces into more lines (v1330).
 *     The bar grows with it, in the group's colour; titleHeight grows too, so
 *     dragging works on every line and "Fit Group to Nodes" leaves room.
 *  2. A bar at the BOTTOM of every group (v1331, Frank 15.11.): "none" and the
 *     nine colours of the frontend's own table (the Note node's row, applied
 *     to the group), a Fit button (the group around its nodes) and an on/off
 *     button that bypasses or brings back every node inside -- the same state
 *     the Polyhedron Group Switch shows, read from the nodes' own modes.
 *
 * v1331 drops v1330's grey "Suite look" and its "polyhedron" menu entry: Frank
 * wants the groups exactly like the normal ones.
 *
 * WHAT THE FRONTEND DOES (comfyui-frontend-package 1.49.6, litegraph):
 * LGraphGroup.draw() fills the title bar (NODE_TITLE_HEIGHT) and the body in
 * the group colour at alpha 0.25 and writes the title in ONE fillText that is
 * never measured. titleHeight is a getter; resizeTo() ("Fit Group to Nodes")
 * puts the nodes padding + titleHeight below the top and padding above the
 * bottom.
 *
 * Clicks on the bottom bar are taken in the CAPTURE phase on the canvas'
 * parent, before litegraph sees them -- only when the point lies on a bar
 * button and NO node lies there (a node over the bar always wins).
 *
 * v1333 (Frank 15.11.: "nur unsere Gruppen ... komplett vom Core abspalten"):
 * ALL of this applies only to a POLYHEDRON group -- one whose flags carry
 * ph_group (ph_groups_core.js). Such a group shows the Suite's hexagon in
 * front of its title (the same "\u2B21 " the Suite's node titles carry) and
 * always has its title drawn here. Every other group is the frontend's own:
 * its draw, its titleHeight and "Fit Group to Nodes" pass straight through.
 * A group becomes ours by its right-click menu ("\u2B21 Polyhedron group") or
 * by the "group around selection" button of the Polyhedron Align bar.
 *
 * Anything unexpected in a future frontend (no group class, no draw, an error
 * while measuring): the file does nothing and the frontend draws as before.
 */
import { app } from "../../scripts/app.js";
import { memberNodes, stateOf, nextOn, setNodes, MODE_BYPASS, isPhGroup, setPhGroup } from "./ph_groups_core.js";

export const BAR_H = 22;          // the bottom bar, inside the group
export const HEX = "\u2B21 ";     // v1333: the Suite's hexagon, as in its node titles
const SW = 13;                    // swatch size, as the Note row
const SW_GAP = 5;
const PAD = 8;
const MIN_SCALE = 0.45;           // below this zoom the bar is not drawn or clickable
const FIT_PAD = 10;               // the frontend's own default padding for resizeTo

// Pure: lines of `text` no wider than maxW; measure(s) gives the width of s.
export function wrapTitle(text, maxW, measure) {
    const t = String(text == null ? "" : text);
    if (!(maxW > 0) || measure(t) <= maxW) return [t];
    const lines = [];
    let cur = "";
    const push = (w) => {
        // a word wider than the group alone: break it by characters
        if (measure(w) <= maxW) return w;
        let part = "";
        for (const ch of w) {
            if (part && measure(part + ch) > maxW) { lines.push(part); part = ch; }
            else part += ch;
        }
        return part;
    };
    for (const word of t.split(" ")) {
        if (!word) continue;
        const next = cur ? cur + " " + word : word;
        if (!cur || measure(next) <= maxW) {
            cur = cur ? next : push(word);
        } else {
            lines.push(cur);
            cur = push(word);
        }
    }
    if (cur) lines.push(cur);
    return lines.length ? lines : [t];
}

// Pure: the title bar for n lines.
export function barHeight(n, titleH, lineH) {
    return titleH + Math.max(0, n - 1) * lineH;
}

// Pure: the buttons of the bottom bar, in graph coordinates.
// names: the colour names of the frontend's table, in its order.
export function barLayout(bounds, names) {
    const [x, y, w, h] = bounds;
    const top = y + h - BAR_H + (BAR_H - SW) / 2;
    const hits = [];
    let cx = x + PAD;
    hits.push({ kind: "colour", name: null, x: cx, y: top, w: SW, h: SW });
    cx += SW + SW_GAP;
    for (const name of names || []) {
        hits.push({ kind: "colour", name, x: cx, y: top, w: SW, h: SW });
        cx += SW + SW_GAP;
    }
    const right = x + w - PAD;
    hits.push({ kind: "toggle", x: right - 46, y: top - 2, w: 46, h: SW + 4 });
    hits.push({ kind: "fit", x: right - 46 - 6 - 40, y: top - 2, w: 40, h: SW + 4 });
    // too narrow for everything: the colours give way, the buttons stay
    const firstButton = right - 46 - 6 - 40 - 6;
    return hits.filter((hit) => hit.kind !== "colour" || hit.x + hit.w <= firstButton);
}

export function hitAt(hits, px, py) {
    for (const hit of hits || []) {
        if (px >= hit.x && px <= hit.x + hit.w && py >= hit.y && py <= hit.y + hit.h) return hit;
    }
    return null;
}

function lg() {
    const LG = globalThis.LiteGraph || {};
    return { LG, titleH: Number(LG.NODE_TITLE_HEIGHT) || 30 };
}

function fontOf(group) {
    const { LG } = lg();
    const size = Number(group.font_size) || Number(LG.GROUP_TEXT_SIZE) || 24;
    return { size, font: size + "px " + (LG.GROUP_FONT || "Arial") };
}

let _mctx = null;
function linesOf(group) {
    const { size, font } = fontOf(group);
    const w = group._size ? group._size[0] : (group.size ? group.size[0] : 0);
    const title = HEX + String(group.title || "") + (group.pinned ? "\u{1F4CC}" : "");
    const key = title + "|" + w + "|" + font;
    if (group._phTitleKey === key && group._phTitleLines) return group._phTitleLines;
    if (!_mctx) _mctx = document.createElement("canvas").getContext("2d");
    _mctx.font = font;
    const lines = wrapTitle(title, w - size, (s) => _mctx.measureText(s).width);
    group._phTitleKey = key;
    group._phTitleLines = lines;
    return lines;
}

function lineHeightOf(group) {
    const { titleH } = lg();
    return Math.max(titleH, Math.round(fontOf(group).size * 1.25));
}

function colourTable() {
    const C = (app.canvas && app.canvas.constructor) || globalThis.LGraphCanvas;
    return (C && C.node_colors) || null;
}

function colourName(group, colours) {
    if (!colours || !group.color) return null;
    for (const name of Object.keys(colours)) {
        if (colours[name] && colours[name].groupcolor === group.color) return name;
    }
    return undefined;   // a colour of its own, not from the table
}

function groupNodes(group) {
    const graph = group.graph || app.graph;
    return memberNodes(graph ? graph._nodes : [], [group._pos[0], group._pos[1], group._size[0], group._size[1]],
                       lg().titleH);
}

export function install(G) {
    if (!G || !G.prototype || typeof G.prototype.draw !== "function" || G.prototype._phTitleWrap) return false;
    const proto = G.prototype;
    proto._phTitleWrap = true;

    // titleHeight: the frontend's constant plus the extra lines
    const desc = Object.getOwnPropertyDescriptor(proto, "titleHeight");
    if (desc && typeof desc.get === "function") {
        const baseGet = desc.get;
        Object.defineProperty(proto, "titleHeight", {
            configurable: true,
            get() {
                const base = baseGet.call(this);
                if (!isPhGroup(this)) return base;      // v1333: a core group stays the frontend's
                try {
                    const n = linesOf(this).length;
                    return n > 1 ? barHeight(n, base, lineHeightOf(this)) : base;
                } catch (e) {
                    return base;
                }
            },
        });
    }

    // Fit Group to Nodes (the frontend's and ours) leaves room for the bottom bar
    if (typeof proto.resizeTo === "function") {
        const _resizeTo = proto.resizeTo;
        proto.resizeTo = function (objects, padding) {
            // the frontend returns early (and changes nothing) for no objects
            const any = objects != null && Array.from(objects).length > 0;
            const r = _resizeTo.apply(this, arguments);
            if (any && this._size && isPhGroup(this)) this._size[1] += BAR_H;
            return r;
        };
    }

    const _draw = proto.draw;
    proto.draw = function (canvas, ctx) {
        if (!isPhGroup(this)) { this._phBarHits = null; return _draw.apply(this, arguments); }
        let lines;
        try {
            lines = linesOf(this);
        } catch (e) {
            return _draw.apply(this, arguments);
        }
        // v1333: a Polyhedron group's title is always ours (the hexagon in front)
        let r;
        {
            const title = this.title, pinned = this.flags && this.flags.pinned;
            this.title = "";
            if (pinned) this.flags.pinned = false;     // the pin sign rides on our last line
            try {
                r = _draw.apply(this, arguments);
            } finally {
                this.title = title;
                if (pinned) this.flags.pinned = pinned;
            }
        }
        try {
            paintTitle(this, canvas, ctx, lines);
            paintBar(this, canvas, ctx);
        } catch (e) {
            /* a decoration must never break the canvas */
        }
        return r;
    };
    return true;
}

function alphaOf(canvas) {
    return canvas && canvas.editor_alpha != null ? canvas.editor_alpha : 1;
}

function paintTitle(group, canvas, ctx, lines) {
    const { titleH } = lg();
    const { size, font } = fontOf(group);
    const lineH = lineHeightOf(group);
    const [x, y] = group._pos;
    const w = group._size[0];
    const colour = group.color || "#335";
    const alpha = alphaOf(canvas);
    ctx.save();
    // the frontend's bar is NODE_TITLE_HEIGHT; the extra lines get the same fill
    if (lines.length > 1) {
        ctx.globalAlpha = 0.25 * alpha;
        ctx.fillStyle = colour;
        ctx.fillRect(x + 0.5, y + titleH + 0.5, w, barHeight(lines.length, titleH, lineH) - titleH);
    }
    ctx.globalAlpha = alpha;
    ctx.fillStyle = colour;
    ctx.font = font;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    for (let i = 0; i < lines.length; i++) {
        ctx.fillText(lines[i], x + size / 2, y + titleH / 2 + 1 + i * lineH);
    }
    ctx.restore();
}

function paintBar(group, canvas, ctx) {
    const scale = canvas && canvas.ds ? canvas.ds.scale : 1;
    if (scale < MIN_SCALE) { group._phBarHits = null; return; }
    const colours = colourTable();
    const names = colours ? Object.keys(colours) : [];
    const b = [group._pos[0], group._pos[1], group._size[0], group._size[1]];
    if (b[3] < BAR_H + lg().titleH) return;
    const hits = barLayout(b, names);
    group._phBarHits = hits;
    const colour = group.color || "#335";
    const alpha = alphaOf(canvas);
    const active = colourName(group, colours);
    const state = stateOf(groupNodes(group));
    ctx.save();
    // the bar is a second, lower header in the group's own colour
    ctx.globalAlpha = 0.25 * alpha;
    ctx.fillStyle = colour;
    ctx.fillRect(b[0] + 0.5, b[1] + b[3] - BAR_H + 0.5, b[2], BAR_H);
    ctx.globalAlpha = alpha;
    ctx.lineWidth = 1;
    for (const hit of hits) {
        if (hit.kind === "colour") {
            if (hit.name === null) {
                ctx.strokeStyle = "#777";
                ctx.strokeRect(hit.x + 0.5, hit.y + 0.5, SW - 1, SW - 1);
                ctx.beginPath();
                ctx.moveTo(hit.x + 2, hit.y + SW - 2);
                ctx.lineTo(hit.x + SW - 2, hit.y + 2);
                ctx.stroke();
            } else {
                const o = colours[hit.name];
                ctx.fillStyle = o.groupcolor;
                ctx.fillRect(hit.x, hit.y, SW, SW);
                ctx.strokeStyle = o.color || "#000";
                ctx.strokeRect(hit.x + 0.5, hit.y + 0.5, SW - 1, SW - 1);
            }
            if (hit.name === active) {
                ctx.strokeStyle = "#ffffff";
                ctx.lineWidth = 2;
                ctx.strokeRect(hit.x, hit.y, SW, SW);
                ctx.lineWidth = 1;
            }
        } else {
            ctx.fillStyle = "rgba(0,0,0,0.35)";
            ctx.fillRect(hit.x, hit.y, hit.w, hit.h);
            ctx.strokeStyle = colour;
            ctx.strokeRect(hit.x + 0.5, hit.y + 0.5, hit.w - 1, hit.h - 1);
            ctx.font = "11px Arial";
            ctx.textBaseline = "middle";
            ctx.textAlign = "center";
            const cy = hit.y + hit.h / 2 + 0.5;
            if (hit.kind === "fit") {
                ctx.fillStyle = "#dddddd";
                ctx.fillText("\u2922 Fit", hit.x + hit.w / 2, cy);
            } else {
                const dot = state === "on" ? "#5ad25a" : (state === "mixed" ? "#e8b44a" : "#777777");
                ctx.fillStyle = dot;
                ctx.beginPath();
                ctx.arc(hit.x + 9, cy, 4, 0, Math.PI * 2);
                ctx.fill();
                ctx.fillStyle = "#dddddd";
                ctx.textAlign = "left";
                ctx.fillText(state === "empty" ? "--" : state, hit.x + 16, cy);
            }
        }
    }
    ctx.restore();
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

// Fit the group around its nodes with the frontend's own method (patched
// above to leave the bottom bar free).
export function fitGroup(group) {
    if (typeof group.recomputeInsideNodes === "function") group.recomputeInsideNodes();
    const inside = groupNodes(group);
    if (!inside.length || typeof group.resizeTo !== "function") return false;
    group.resizeTo(inside, FIT_PAD);
    return true;
}

// v1333: make a group ours, or give it back to the frontend. Making it ours
// adds the bottom bar's room below its nodes (fit), giving it back takes the
// room away again, so neither way leaves a bar over a node or a gap.
export function togglePhGroup(group) {
    const on = !isPhGroup(group);
    setPhGroup(group, on);
    group._phTitleKey = null;
    if (on) fitGroup(group);
    else if (group._size && group._size[1] > BAR_H * 3) group._size[1] -= BAR_H;
    changed();
    return on;
}

export function installMenu(G) {
    const proto = G && G.prototype;
    if (!proto || typeof proto.getMenuOptions !== "function" || proto._phMenu) return;
    proto._phMenu = true;
    const _get = proto.getMenuOptions;
    proto.getMenuOptions = function () {
        const opts = _get.apply(this, arguments);
        try {
            if (Array.isArray(opts)) {
                const group = this;
                opts.push(null, {
                    content: isPhGroup(group) ? "\u2B21 Polyhedron group: off (back to a normal group)"
                                              : "\u2B21 Polyhedron group: on",
                    callback: () => togglePhGroup(group),
                });
            }
        } catch (e) { /* the menu stays the frontend's */ }
        return opts;
    };
}

function onBarClick(group, hit) {
    if (hit.kind === "colour") {
        const colours = colourTable();
        group.color = hit.name && colours && colours[hit.name] ? colours[hit.name].groupcolor : undefined;
    } else if (hit.kind === "fit") {
        fitGroup(group);
    } else if (hit.kind === "toggle") {
        const nodes = groupNodes(group);
        setNodes(nodes, nextOn(stateOf(nodes)), MODE_BYPASS);
    }
    changed();
}

function installClicks() {
    const canvas = app.canvas;
    const el = canvas && canvas.canvas;
    const host = el && el.parentElement;
    if (!host || host._phGroupBar) return;
    host._phGroupBar = true;
    host.addEventListener("pointerdown", (e) => {
        try {
            if (e.button !== 0 || e.target !== el) return;
            const ds = canvas.ds;
            if (!ds || ds.scale < MIN_SCALE) return;
            const rect = el.getBoundingClientRect();
            const p = ds.convertCanvasToOffset([e.clientX - rect.left, e.clientY - rect.top]);
            const graph = canvas.graph || app.graph;
            if (!graph) return;
            if (graph.getNodeOnPos && graph.getNodeOnPos(p[0], p[1])) return;   // a node wins
            const groups = graph._groups || [];
            for (let i = groups.length - 1; i >= 0; i--) {
                if (!isPhGroup(groups[i])) continue;
                const hit = hitAt(groups[i]._phBarHits, p[0], p[1]);
                if (hit) {
                    e.stopPropagation();
                    e.preventDefault();
                    onBarClick(groups[i], hit);
                    return;
                }
            }
        } catch (err) {
            /* never block the canvas */
        }
    }, true);
}

function findGroupClass() {
    const LG = globalThis.LiteGraph;
    if (LG && LG.LGraphGroup) return LG.LGraphGroup;
    const g = app.graph && app.graph._groups && app.graph._groups[0];
    return g ? g.constructor : null;
}

// v1333: the entry also on the canvas menu's top level (right click on a
// group); the group's own menu ("Edit Group") carries it via installMenu.
export function canvasMenuItems(canvas) {
    const graph = canvas && canvas.graph;
    const m = canvas && canvas.graph_mouse;
    if (!graph || !m || typeof graph.getGroupOnPos !== "function") return [];
    const group = graph.getGroupOnPos(m[0], m[1]);
    if (!group) return [];
    return [null, {
        content: isPhGroup(group) ? "\u2B21 Polyhedron group: off (back to a normal group)"
                                  : "\u2B21 Polyhedron group: on",
        callback: () => togglePhGroup(group),
    }];
}

app.registerExtension({
    name: "polyhedron.group_title",
    getCanvasMenuItems(canvas) {
        try { return canvasMenuItems(canvas); } catch (e) { return []; }
    },
    async setup() {
        try {
            installMenu(findGroupClass());
            if (!install(findGroupClass())) {
                // no group yet and no global class: install on the first group drawn
                const _onDraw = app.canvas && app.canvas.onDrawBackground;
                let done = false;
                if (app.canvas) app.canvas.onDrawBackground = function () {
                    if (!done) { done = install(findGroupClass()); if (done) installMenu(findGroupClass()); }
                    return _onDraw ? _onDraw.apply(this, arguments) : undefined;
                };
            }
            installClicks();
            if (app.canvas) app.canvas.setDirty(true, true);
        } catch (e) {
            console.warn("[PLS] ph_group_title: not installed", e);
        }
    },
});
