/**
 * ph_group_switch.js -- v1331
 *
 * Polyhedron Group Switch: a node that switches whole groups on and off from
 * anywhere in the graph (Frank 15.11.: "Gruppen ueber weite Nodes an und
 * ausschalten"). A frontend-only node (isVirtualNode): it never reaches the
 * backend and never runs.
 *
 * THE ASSIGNMENT RULE -- because matching groups by title or colour (the old
 * group bypasser nodes) breaks the moment a group is renamed or recoloured:
 *  - a group is known by its ID, which the workflow keeps through renaming,
 *    moving and recolouring;
 *  - by default the switch lists EVERY group, in reading order (left to
 *    right, then top to bottom), nested groups indented under their parent;
 *  - x takes a group off THIS switch (its id goes into ph_hidden), "+ all"
 *    (v1332: "N hidden - show all") brings every group back; the hidden
 *    groups stay listed in grey under the switch's rows, a click on one brings
 *    that group back alone (v1332, Frank 15.11.: a preset switch showed three
 *    rows and nothing said that ten were hidden);
 *    brings every group back; several switches can each carry their own set;
 *  - which nodes a group holds is measured at the click, never stored.
 *
 * Per switch: off = bypass or mute; rule = free, or one at a time (switching
 * a group on switches the other listed groups off; nested groups are not
 * rivals). The row shows on / off / mixed -- read from the nodes' own modes,
 * so the row and the group's own bar button can never disagree. A click on
 * "mixed" brings the whole group back on.
 *
 * v1333 (Frank 15.11.): the switch lists only POLYHEDRON groups (flag
 * ph_group, ph_groups_core.js) -- the frontend's own groups are not its
 * business. Its title carries the Suite's hexagon, and its badge names the
 * Suite: the frontend builds a frontend-only node's def with python_module
 * "custom_nodes.frontend_only" (badge "frontend_only"); the hook
 * beforeRegisterVueAppNodeDefs hands it the module of a Suite node instead.
 * The canvas badge itself is pushed by the frontend's NodeBadge extension in
 * nodeCreated -- which never runs for a class registered straight with
 * LiteGraph (measured in 1.49.6: node.badges stays empty). So the switch
 * carries its own badge, drawn like the frontend's LGraphBadge (12px
 * sans-serif, padding 6, height 20, radius 5, #FFF on #0F1F0F); it steps
 * aside the moment the frontend adds one of its own, and hides when the
 * source badge setting is "None".
 */
import { app } from "../../scripts/app.js";
import {
    memberNodes, stateOf, nextOn, setNodes, readingOrder, rivalsOf, boundsOf,
    MODE_BYPASS, MODE_MUTE, SWITCH_TYPE, isPhGroup,
} from "./ph_groups_core.js";

export const TITLE = "\u2B21 Polyhedron Group Switch";
const HEAD_H = 24;
const ROW_H = 22;
const PAD = 8;
const INDENT = 12;
const HID_SEP = 8;         // v1332: the gap and line above the hidden groups
const HID_H = 18;          // v1332: one hidden group
const BODY = "#1a1a2a";    // ph_palette.js: the Suite's node colours
const HEAD = "#333";

// Pure (v1333): only our groups; nesting depth counts among them alone.
export function phOnly(groups) {
    return (groups || []).filter(isPhGroup);
}

// Pure (v1333): the def the frontend built for the switch gets the Suite's
// module (badge) and the hexagon title; true when a def was patched.
export function patchSwitchDef(defs) {
    return patchFrontendDef(defs, SWITCH_TYPE, TITLE);
}

// v1334: the same for every frontend-only Suite node (Polyhedron Align too).
export function patchFrontendDef(defs, name, title) {
    const list = Array.isArray(defs) ? defs : Object.values(defs || {});
    const mine = list.find((d) => d && d.name === name);
    if (!mine) return false;
    const suite = list.find((d) => d && d !== mine && typeof d.python_module === "string"
        && /PolyhedronLoRAStack/i.test(d.python_module));
    mine.python_module = suite ? suite.python_module : "custom_nodes.ComfyUI-PolyhedronLoRAStack";
    mine.display_name = title;
    mine.category = "Polyhedron/Utility";
    return true;
}

// v1334: the Suite badge on a frontend-only node (see SuiteBadge).
export function addSuiteBadge(node) {
    if (!node || !Array.isArray(node.badges)) return false;
    const mine = () => badgeFor(node.badges, mine, sourceBadgeMode(), frontendModeOf(node));
    node.badges.push(mine);
    node.badgePosition = "top-right";
    return true;
}

export const BADGE_TEXT = "PolyhedronLoRAStack";

// The badge, drawn as the frontend's LGraphBadge draws its own.
export class SuiteBadge {
    constructor(text) {
        this.text = text; this.fgColor = "#FFF"; this.bgColor = "#0F1F0F";
        this.fontSize = 12; this.padding = 6; this.height = 20; this.cornerRadius = 5;
    }
    get visible() { return (this.text || "").length > 0; }
    getWidth(ctx) {
        if (!this.visible) return 0;
        const f = ctx.font;
        ctx.font = this.fontSize + "px sans-serif";
        const w = ctx.measureText(this.text).width;
        ctx.font = f;
        return w + this.padding * 2;
    }
    draw(ctx, x, y) {
        if (!this.visible) return;
        const { font, fillStyle, textBaseline, textAlign } = ctx;
        const w = this.getWidth(ctx);
        ctx.font = this.fontSize + "px sans-serif";
        ctx.fillStyle = this.bgColor;
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(x, y, w, this.height, this.cornerRadius); else ctx.rect(x, y, w, this.height);
        ctx.fill();
        ctx.fillStyle = this.fgColor;
        ctx.textBaseline = "middle";
        ctx.textAlign = "left";
        ctx.fillText(this.text, x + this.padding, y + this.height / 2 + 1);
        Object.assign(ctx, { font, fillStyle, textBaseline, textAlign });
    }
}

// Pure: our badge, or none when the frontend already gave the node one or the
// source badge is switched off.
export function badgeFor(badges, mine, mode, frontend = "array") {
    const others = (badges || []).filter((b) => b !== mine);
    return others.length || mode === "None" || frontend !== "array"
        ? new SuiteBadge("") : new SuiteBadge(BADGE_TEXT);
}

// v1335 (Frank 15.11., frontend newer than 1.49.6: "PolyhedronLoRAStack" twice).
// Pure: how the running frontend gives its nodes their source badge, read off
// the OTHER nodes of the graph -- node definitions with nodeData:
//   "array"    -- pushed into node.badges in nodeCreated (1.49.6): such a node
//                 carries entries, and our frontend-only nodes get none, so
//                 they draw their own;
//   "provider" -- drawn from the node def store for every node (1.53.10):
//                 those nodes carry no entries, and the frontend draws our
//                 badge itself, so ours stays empty;
//   null       -- no such node in the graph yet: undecided, ours stays empty.
export function frontendBadgeMode(nodes) {
    let seen = false;
    for (const n of nodes || []) {
        if (!n || n.isVirtualNode || !n.constructor || !n.constructor.nodeData) continue;
        seen = true;
        if (Array.isArray(n.badges) && n.badges.length > 0) return "array";
    }
    return seen ? "provider" : null;
}

let _frontendMode = null;     // decided once per page, from the first graph that tells

function frontendModeOf(node) {
    if (_frontendMode) return _frontendMode;
    const g = node && (node.graph || app.graph);
    _frontendMode = frontendBadgeMode(g ? g._nodes : []);
    return _frontendMode;
}

function sourceBadgeMode() {
    try {
        const s = app.extensionManager && app.extensionManager.setting;
        return s && s.get ? s.get("Comfy.NodeBadge.NodeSourceBadgeMode") : null;
    } catch (e) {
        return null;
    }
}

// Pure: the rows this switch shows.
export function rowsOf(groups, hidden) {
    const skip = new Set((hidden || []).map(String));
    return readingOrder(phOnly(groups)).filter((r) => !skip.has(String(r.group.id)));
}

// Pure: the groups this switch hides -- only groups that still exist (the id
// of a deleted group is not counted), in reading order.
export function hiddenRowsOf(groups, hidden) {
    const want = new Set((hidden || []).map(String));
    return readingOrder(phOnly(groups)).filter((r) => want.has(String(r.group.id)));
}

// Pure: the label of the "show all" chip, null when nothing is hidden.
export function hiddenLabel(k) {
    return k > 0 ? k + " hidden \u00b7 show all" : null;
}

// Pure: the height the node needs for n rows and k hidden groups.
export function heightFor(n, k = 0) {
    return HEAD_H + Math.max(1, n) * ROW_H + (k > 0 ? HID_SEP + k * HID_H : 0) + PAD;
}

function titleH() {
    return Number((globalThis.LiteGraph || {}).NODE_TITLE_HEIGHT) || 30;
}

function nodesOf(graph, group) {
    return memberNodes(graph ? graph._nodes : [], boundsOf(group), titleH());
}

function offModeOf(node) {
    return node.properties && node.properties.ph_off === "mute" ? MODE_MUTE : MODE_BYPASS;
}

function changed(node) {
    try { node.setDirtyCanvas(true, true); } catch (e) { /* ignore */ }
    try { if (app.graph.change) app.graph.change(); } catch (e) { /* ignore */ }
    try {
        const t = app.extensionManager && app.extensionManager.workflow
            && app.extensionManager.workflow.activeWorkflow
            && app.extensionManager.workflow.activeWorkflow.changeTracker;
        if (t && t.checkState) t.checkState();
    } catch (e) { /* ignore */ }
}

// Pure-ish: what a click on a row's state does (graph = { _nodes, _groups }).
export function switchGroup(graph, group, listedGroups, offMode, oneAtATime) {
    const nodes = nodesOf(graph, group);
    const on = nextOn(stateOf(nodes));
    if (on && oneAtATime) {
        for (const g of rivalsOf(group, listedGroups)) setNodes(nodesOf(graph, g), false, offMode);
    }
    setNodes(nodes, on, offMode);
    return on;
}

export function makeClass(LG) {
    class GroupSwitch extends LG.LGraphNode {
        constructor(title) {
            super(title || TITLE);
            this.isVirtualNode = true;
            this.serialize_widgets = false;
            this.properties = Object.assign({ ph_hidden: [], ph_off: "bypass", ph_rule: "free" }, this.properties || {});
            this.color = HEAD;
            this.bgcolor = BODY;
            this.size = [320, heightFor(3)];
            this._phHits = [];
            addSuiteBadge(this);     // top right, as the frontend places its own badge
        }

        computeSize() {
            const graph = this.graph || app.graph;
            const gs = graph ? graph._groups || [] : [];
            const n = graph ? rowsOf(gs, this.properties.ph_hidden).length : 1;
            const k = hiddenRowsOf(gs, this.properties.ph_hidden).length;
            return [Math.max(this.size ? this.size[0] : 320, 240), heightFor(n, k)];
        }

        onDrawForeground(ctx) {
            if (this.flags && this.flags.collapsed) return;
            const graph = this.graph || app.graph;
            if (!graph) return;
            const rows = rowsOf(graph._groups || [], this.properties.ph_hidden);
            const hiddenRows = hiddenRowsOf(graph._groups || [], this.properties.ph_hidden);
            const need = heightFor(rows.length, hiddenRows.length);
            if (this.size[1] !== need) this.size[1] = need;
            const w = this.size[0];
            const hits = [];
            ctx.save();
            ctx.font = "12px Arial";
            ctx.textBaseline = "middle";

            // header: off mode, rule, "+ all"
            const chips = [
                { kind: "off", label: this.properties.ph_off === "mute" ? "off = mute" : "off = bypass" },
                { kind: "rule", label: this.properties.ph_rule === "one" ? "one at a time" : "free" },
            ];
            const hl = hiddenLabel(hiddenRows.length);
            if (hl) chips.push({ kind: "all", label: hl });
            let cx = PAD;
            for (const c of chips) {
                const tw = ctx.measureText(c.label).width + 14;
                ctx.fillStyle = "#2c3e5a";
                ctx.fillRect(cx, 4, tw, HEAD_H - 8);
                ctx.fillStyle = "#dddddd";
                ctx.textAlign = "left";
                ctx.fillText(c.label, cx + 7, HEAD_H / 2);
                hits.push({ kind: c.kind, x: cx, y: 4, w: tw, h: HEAD_H - 8 });
                cx += tw + 6;
            }

            if (!rows.length) {
                ctx.fillStyle = "#888888";
                ctx.fillText(hiddenRows.length ? "every Polyhedron group is hidden" : "no Polyhedron groups -- group menu or Align bar", PAD, HEAD_H + ROW_H / 2);
            }
            rows.forEach((r, i) => {
                const y = HEAD_H + i * ROW_H;
                const g = r.group;
                const state = stateOf(nodesOf(graph, g));
                const x0 = PAD + r.depth * INDENT;
                ctx.fillStyle = g.color || "#335";
                ctx.fillRect(x0, y + 5, 10, ROW_H - 10);
                // state button
                const bx = w - PAD - 18 - 6 - 64;
                const dot = state === "on" ? "#5ad25a" : (state === "mixed" ? "#e8b44a" : "#777777");
                ctx.fillStyle = "rgba(0,0,0,0.35)";
                ctx.fillRect(bx, y + 3, 64, ROW_H - 6);
                ctx.fillStyle = dot;
                ctx.beginPath();
                ctx.arc(bx + 10, y + ROW_H / 2, 4, 0, Math.PI * 2);
                ctx.fill();
                ctx.fillStyle = "#dddddd";
                ctx.textAlign = "left";
                ctx.fillText(state === "empty" ? "--" : state, bx + 19, y + ROW_H / 2);
                hits.push({ kind: "state", group: g, x: bx, y: y + 3, w: 64, h: ROW_H - 6 });
                // remove from this switch
                const xx = w - PAD - 18;
                ctx.fillStyle = "#999999";
                ctx.textAlign = "center";
                ctx.fillText("\u00d7", xx + 9, y + ROW_H / 2);
                hits.push({ kind: "hide", group: g, x: xx, y: y + 3, w: 18, h: ROW_H - 6 });
                // title, cut to the room left
                let t = String(g.title || "(untitled)");
                const room = bx - 6 - (x0 + 16);
                ctx.textAlign = "left";
                ctx.fillStyle = state === "on" ? "#e8e8e8" : "#9a9a9a";
                if (ctx.measureText(t).width > room) {
                    while (t.length > 1 && ctx.measureText(t + "\u2026").width > room) t = t.slice(0, -1);
                    t += "\u2026";
                }
                ctx.fillText(t, x0 + 16, y + ROW_H / 2);
                // the whole row (title) also switches
                hits.push({ kind: "state", group: g, x: x0, y, w: bx - x0, h: ROW_H });
            });
            // v1332: the hidden groups, grey, under a thin line; a click brings one back
            if (hiddenRows.length) {
                const top = HEAD_H + Math.max(1, rows.length) * ROW_H;
                ctx.strokeStyle = "#3a3a4a";
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(PAD, top + HID_SEP / 2 + 0.5);
                ctx.lineTo(w - PAD, top + HID_SEP / 2 + 0.5);
                ctx.stroke();
                ctx.font = "11px Arial";
                hiddenRows.forEach((r, i) => {
                    const y = top + HID_SEP + i * HID_H;
                    let t = "+ " + String(r.group.title || "(untitled)");
                    const room = w - 2 * PAD - r.depth * INDENT;
                    if (ctx.measureText(t).width > room) {
                        while (t.length > 1 && ctx.measureText(t + "\u2026").width > room) t = t.slice(0, -1);
                        t += "\u2026";
                    }
                    ctx.fillStyle = "#6f6f7f";
                    ctx.textAlign = "left";
                    ctx.fillText(t, PAD + r.depth * INDENT, y + HID_H / 2);
                    hits.push({ kind: "unhide", group: r.group, x: PAD, y, w: w - 2 * PAD, h: HID_H });
                });
            }
            ctx.restore();
            this._phHits = hits;
        }

        onMouseDown(e, pos) {
            const graph = this.graph || app.graph;
            for (const h of this._phHits || []) {
                if (pos[0] < h.x || pos[0] > h.x + h.w || pos[1] < h.y || pos[1] > h.y + h.h) continue;
                const p = this.properties;
                if (h.kind === "off") p.ph_off = p.ph_off === "mute" ? "bypass" : "mute";
                else if (h.kind === "rule") p.ph_rule = p.ph_rule === "one" ? "free" : "one";
                else if (h.kind === "all") p.ph_hidden = [];
                else if (h.kind === "hide") p.ph_hidden = (p.ph_hidden || []).concat([String(h.group.id)]);
                else if (h.kind === "unhide") {
                    p.ph_hidden = (p.ph_hidden || []).filter((id) => String(id) !== String(h.group.id));
                }
                else if (h.kind === "state") {
                    const listed = rowsOf(graph._groups || [], p.ph_hidden).map((r) => r.group);
                    switchGroup(graph, h.group, listed, offModeOf(this), p.ph_rule === "one");
                }
                changed(this);
                return true;
            }
            return false;
        }
    }
    GroupSwitch.title = TITLE;
    GroupSwitch.category = "Polyhedron/Utility";
    GroupSwitch.description = "Switches whole groups on and off from anywhere: one row per group, "
        + "known by its id (renaming, moving and recolouring break nothing). x takes a group off this "
        + "switch; the hidden groups stay listed in grey below, a click brings one back, 'show all' brings every one. Off = bypass or mute; free or one at a time. "
        + "Frontend only, never runs.";
    return GroupSwitch;
}

app.registerExtension({
    name: "polyhedron.group_switch",
    beforeRegisterVueAppNodeDefs(defs) {
        try { patchSwitchDef(defs); } catch (e) { /* the badge stays the frontend's */ }
    },
    registerCustomNodes() {
        const LG = globalThis.LiteGraph;
        if (!LG || !LG.LGraphNode || !LG.registerNodeType) {
            console.warn("[PLS] ph_group_switch: LiteGraph not reachable, node not registered");
            return;
        }
        if (LG.registered_node_types && LG.registered_node_types[SWITCH_TYPE]) return;
        LG.registerNodeType(SWITCH_TYPE, makeClass(LG));
    },
});
