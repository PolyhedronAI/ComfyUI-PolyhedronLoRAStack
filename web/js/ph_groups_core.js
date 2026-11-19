/**
 * ph_groups_core.js -- v1331
 *
 * The one place that knows what a group IS for switching: which nodes belong
 * to it, what state it is in, how it is switched, in which order groups are
 * read. Used by ph_group_title.js (the bar at the bottom of every group) and
 * ph_group_switch.js (the Polyhedron Group Switch node). Both read the SAME
 * state from the nodes' own modes -- they cannot disagree, because neither
 * stores a state of its own.
 *
 * Pure functions only (no app, no DOM): the guard drives them in node.
 *
 * Membership: a node belongs to a group when the centre of its box (title
 * bar included, as the frontend's own containsCentre reads it) lies inside
 * the group. Computed when asked, never stored: a node dragged into a group
 * belongs to it at the next click.
 */

export const MODE_ON = 0;        // LiteGraph ALWAYS
export const MODE_MUTE = 2;      // LiteGraph NEVER  -- ComfyUI "mute"
export const MODE_BYPASS = 4;    // ComfyUI bypass

export const SWITCH_TYPE = "PolyhedronGroupSwitch";

// v1333 (Frank 15.11.: "nur unsere Gruppen ... komplett vom Core abspalten"):
// a Polyhedron group carries this flag in group.flags, which the frontend
// serializes with the group (id, title, bounding, color, flags). Only such a
// group gets the hexagon, the wrapped title, the bottom bar and a row in the
// Group Switch; every other group stays the frontend's own, untouched.
export const PH_FLAG = "ph_group";

export function isPhGroup(group) {
    return !!(group && group.flags && group.flags[PH_FLAG]);
}

export function setPhGroup(group, on) {
    if (!group) return;
    group.flags = group.flags || {};
    if (on) group.flags[PH_FLAG] = true;
    else delete group.flags[PH_FLAG];
}

export function boundsOf(group) {
    const b = group._bounding || group.bounding;
    return [Number(b[0]), Number(b[1]), Number(b[2]), Number(b[3])];
}

export function containsRect(outer, inner) {
    return inner[0] >= outer[0] && inner[1] >= outer[1]
        && inner[0] + inner[2] <= outer[0] + outer[2]
        && inner[1] + inner[3] <= outer[1] + outer[3]
        && !(inner[0] === outer[0] && inner[1] === outer[1]
             && inner[2] === outer[2] && inner[3] === outer[3]);
}

// The nodes a group switches: centre inside, never a Group Switch, never a
// virtual node (reroutes, primitives of the frontend -- they do not run).
export function memberNodes(nodes, bounds, titleH) {
    const [gx, gy, gw, gh] = bounds;
    const th = Number(titleH) || 30;
    const out = [];
    for (const n of nodes || []) {
        if (!n || n.type === SWITCH_TYPE || n.isVirtualNode) continue;
        const cx = n.pos[0] + n.size[0] / 2;
        const cy = n.pos[1] - th + (n.size[1] + th) / 2;
        if (cx >= gx && cx <= gx + gw && cy >= gy && cy <= gy + gh) out.push(n);
    }
    return out;
}

// "on" every node runs, "off" every node is muted or bypassed, "mixed" some
// were switched by hand, "empty" the group holds nothing to switch.
export function stateOf(nodes) {
    if (!nodes || !nodes.length) return "empty";
    let on = 0, off = 0;
    for (const n of nodes) {
        if (n.mode === MODE_ON) on++;
        else if (n.mode === MODE_MUTE || n.mode === MODE_BYPASS) off++;
    }
    if (on === nodes.length) return "on";
    if (off === nodes.length) return "off";
    return "mixed";
}

// A click: "on" goes off, "off" and "mixed" go on (mixed is brought back whole).
export function nextOn(state) {
    return state !== "on";
}

export function setNodes(nodes, on, offMode) {
    const m = on ? MODE_ON : (offMode === MODE_MUTE ? MODE_MUTE : MODE_BYPASS);
    for (const n of nodes || []) n.mode = m;
}

// Reading order: left to right, then top to bottom; depth = how many other
// groups contain this one (nested groups are listed indented).
export function readingOrder(groups) {
    const list = (groups || []).map((g) => ({ g, b: boundsOf(g) }));
    list.sort((a, c) => (a.b[0] - c.b[0]) || (a.b[1] - c.b[1]));
    return list.map(({ g, b }) => ({
        group: g,
        depth: list.filter((o) => o.g !== g && containsRect(o.b, b)).length,
    }));
}

// "one at a time": switching a group ON switches the other listed groups
// OFF -- except groups inside it or around it (nesting is not a rival).
export function rivalsOf(target, listed) {
    const tb = boundsOf(target);
    return (listed || []).filter((g) => {
        if (g === target) return false;
        const b = boundsOf(g);
        return !containsRect(b, tb) && !containsRect(tb, b);
    });
}
