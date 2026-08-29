/**
 * ph_note.js -- the view half of \u2b21 Polyhedron Note.
 *
 * v893: THE NOTE STOPS HAVING A COLOUR SYSTEM OF ITS OWN.
 *
 * Frank, on the v892 row: "diese Farben sind nicht deckungsgleich zu den
 * Farben, die dann tatsaechlich verwendet werden ... generell ein wenig
 * Vapor-Ware ... Man kann die Farben eben auch ganz oldschool mit Rechtsklick
 * auswaehlen ueber den ComfyUI-Standard". Both halves of that were right, and
 * the second one was worse than he knew:
 *
 *   THE REGRESSION v892 SHIPPED. To stop the derived colour being written to
 *   the file twice, v892 deleted `color`/`bgcolor` in serialize() -- and so
 *   also deleted any colour set through ComfyUI's own right-click menu.
 *   Driven: pick green, save, reload, and the note is amber again. Silent, and
 *   only visible after a reload. A guard that pins "no second copy" is worth
 *   nothing if the way it keeps that promise is to throw away the FIRST copy.
 *
 *   THE MISMATCH. The swatch painted the raw palette entry (#ff00ff) while the
 *   body took a darkened derivation (#451745). A control that does not show
 *   what it does is exactly the "Vapor-Ware" charge, and it was fair.
 *
 * THE FIX IS A SUBTRACTION. There is no note palette, no derivation, no colour
 * property and no serialize patch any more. The row drives LiteGraph's OWN
 * colour set -- the nine of `LGraphCanvas.node_colors` plus "none" -- through
 * the very methods the right-click menu uses:
 *
 *     setColorOption(o)   o == null ? delete color, delete bgcolor
 *                                   : color = o.color, bgcolor = o.bgcolor
 *     getColorOption()    finds the entry matching color AND bgcolor
 *
 * So the row and the context menu are ONE system that cannot disagree, the
 * swatch is painted with the very bgcolor the node will take -- honest by
 * construction, not by promise -- and the state lives where LiteGraph already
 * keeps it. Nothing of ours is stored at all.
 *
 * Measured before adopting them (v892 report): the nine sit at 17-33%
 * saturation and 27-30% lightness -- muted, which is what Frank asked for --
 * and keep 4.23-5.58 contrast for the #999 title text against the 4.43 the
 * tree's own #333 already shows.
 *
 * KEPT FROM v891, deliberately: the row is a CUSTOM WIDGET whose height is
 * declared INSIDE computeSize. v890 painted it in onDrawForeground at
 * `size[1] - PAD` and the `text` DOM textarea covered it -- the wound
 * ph_clip_encode.js documents at lines 35ff. Collapsing changes what
 * computeSize RETURNS; it never goes back to painting off the bottom edge.
 */

import { app } from "../../scripts/app.js";

const NODE_TYPE = "ULSNote";

// v892 stored the pick here. Read once on load, translated, then removed.
const PROP_LEGACY_COLOUR = "ph_note_colour";
const PROP_COLLAPSED = "ph_note_bar_collapsed";

/*
 * v892 -> LiteGraph, by smallest CIELab distance between the derived body
 * colour and the nine standard bodies. Every one of the five was unambiguous:
 * the runner-up was 3.3 to 19.0 dE further away.
 */
const LEGACY_MAP = {
    magenta: "purple",     // #451745 -> #535, dE 13.0 (next: blue 20.5)
    green: "green",        // #174517 -> #353, dE 11.8 (next: yellow 30.8)
    orange: "brown",       // #452a17 -> #593930, dE 10.0 (next: red 13.3)
    amber: "yellow",       // #453717 -> #653, dE 13.2 (next: brown 15.9)
    blue: "blue",          // #172e45 -> #335, dE 11.8 (next: pale_blue 18.6)
};

const PAD_X = 10;
const SW_SIZE = 13;
const SW_GAP = 5;
const CHEV_W = 14;
const ROW_H = 19;
const BAR_H = ROW_H + 6;
const COLLAPSED_H = 12;

const WIDGET_TYPE = "uls_note_bar";
const WIDGET_NAME = "$ph_note_bar";

/*
 * LiteGraph's own table, never a copy of it. Reached through the canvas'
 * constructor first because that is the instance actually drawing this graph;
 * the global is the fallback for frontends that do not re-export it.
 */
export function nodeColors() {
    return (app && app.canvas && app.canvas.constructor
            && app.canvas.constructor.node_colors)
        || (globalThis.LGraphCanvas && globalThis.LGraphCanvas.node_colors)
        || null;
}

/* Which entry is this node currently on -- READ from the node, never stored. */
export function currentName(node, colors) {
    if (!node || !colors) return null;
    for (const name of Object.keys(colors)) {
        const o = colors[name];
        if (o && o.color === node.color && o.bgcolor === node.bgcolor) return name;
    }
    return null;
}

/* Apply through LiteGraph's own method so the row and the menu stay one. */
export function applyColour(node, option) {
    if (typeof node.setColorOption === "function") {
        node.setColorOption(option || null);
        return;
    }
    if (option) {
        node.color = option.color;
        node.bgcolor = option.bgcolor;
    } else {
        delete node.color;
        delete node.bgcolor;
    }
}

export function isCollapsed(node) {
    return !!(node && node.properties && node.properties[PROP_COLLAPSED]);
}

/* One-time translation of a note coloured by v892. */
export function migrateLegacy(node) {
    if (!node || !node.properties) return null;
    const old = node.properties[PROP_LEGACY_COLOUR];
    if (!old) return null;
    delete node.properties[PROP_LEGACY_COLOUR];
    const colors = nodeColors();
    const name = LEGACY_MAP[old];
    if (!colors || !name || !colors[name]) return null;
    applyColour(node, colors[name]);
    return name;
}

export function makeBarWidget() {
    const w = {
        type: WIDGET_TYPE,
        name: WIDGET_NAME,
        value: null,
        serialize: false,               // never in widgets_values (#577)
        _hits: [],
        _chev: null,
        computeSize(width) {
            return [width, w._node && isCollapsed(w._node) ? COLLAPSED_H : BAR_H];
        },
        draw(ctx, node, width, y) {
            w._node = node;
            ctx.save();
            try {
                w._hits = [];
                const collapsed = isCollapsed(node);
                const top = y + (collapsed ? 1 : 3);
                const h = collapsed ? 8 : SW_SIZE;

                // The chevron is always there, so a collapsed row can always
                // be brought back -- hiding with no way home is a trap.
                ctx.fillStyle = "#888";
                ctx.beginPath();
                if (collapsed) {
                    ctx.moveTo(PAD_X, top + 1);
                    ctx.lineTo(PAD_X + 8, top + 1);
                    ctx.lineTo(PAD_X + 4, top + 6);
                } else {
                    ctx.moveTo(PAD_X, top + 2);
                    ctx.lineTo(PAD_X + 5, top + 6);
                    ctx.lineTo(PAD_X, top + 10);
                }
                ctx.closePath();
                ctx.fill();
                w._chev = { x: PAD_X - 3, y: top - 2, w: CHEV_W, h: h + 4 };
                if (collapsed) { ctx.restore(); return; }

                const colors = nodeColors();
                if (!colors) { ctx.restore(); return; }
                const active = currentName(node, colors);

                let x = PAD_X + CHEV_W;

                // "none" first, as in ComfyUI's own menu.
                ctx.strokeStyle = "#777";
                ctx.lineWidth = 1;
                ctx.strokeRect(x + 0.5, top + 0.5, SW_SIZE - 1, SW_SIZE - 1);
                ctx.beginPath();
                ctx.moveTo(x + 2, top + SW_SIZE - 2);
                ctx.lineTo(x + SW_SIZE - 2, top + 2);
                ctx.stroke();
                if (active === null) {
                    ctx.strokeStyle = "#ffffff";
                    ctx.lineWidth = 2;
                    ctx.strokeRect(x, top, SW_SIZE, SW_SIZE);
                }
                w._hits.push({ x, y: top, w: SW_SIZE, h: SW_SIZE, name: null });
                x += SW_SIZE + SW_GAP;

                for (const name of Object.keys(colors)) {
                    const o = colors[name];
                    // THE SWATCH IS THE OUTCOME: filled with the very bgcolor
                    // the node takes, rimmed with its title colour.
                    ctx.fillStyle = o.bgcolor;
                    ctx.fillRect(x, top, SW_SIZE, SW_SIZE);
                    ctx.strokeStyle = o.color;
                    ctx.lineWidth = 1;
                    ctx.strokeRect(x + 0.5, top + 0.5, SW_SIZE - 1, SW_SIZE - 1);
                    if (name === active) {
                        ctx.strokeStyle = "#ffffff";
                        ctx.lineWidth = 2;
                        ctx.strokeRect(x, top, SW_SIZE, SW_SIZE);
                    }
                    w._hits.push({ x, y: top, w: SW_SIZE, h: SW_SIZE, name });
                    x += SW_SIZE + SW_GAP;
                }
            } catch (e) {
                /* a decoration must never break the canvas */
            }
            ctx.restore();
        },
        mouse(event, pos, node) {
            const et = event && event.type;
            if (et !== "pointerdown" && et !== "mousedown") return false;
            const mx = pos[0];
            const my = pos[1];

            const c = w._chev;
            if (c && mx >= c.x && mx <= c.x + c.w && my >= c.y && my <= c.y + c.h) {
                const was = isCollapsed(node);
                node.properties = node.properties || {};
                node.properties[PROP_COLLAPSED] = !was;
                // Give the node back exactly the height the row stops using,
                // so the textarea neither grows nor gets clipped.
                const delta = (was ? BAR_H - COLLAPSED_H : COLLAPSED_H - BAR_H);
                if (node.size) node.size[1] += delta;
                node.setDirtyCanvas(true, true);
                return true;
            }
            if (isCollapsed(node)) return false;

            const colors = nodeColors();
            for (const h of w._hits || []) {
                if (mx >= h.x && mx <= h.x + h.w && my >= h.y && my <= h.y + h.h) {
                    applyColour(node, h.name && colors ? colors[h.name] : null);
                    node.setDirtyCanvas(true, true);
                    return true;
                }
            }
            return false;
        },
    };
    return w;
}

function ensureBar(node) {
    if (typeof node.addCustomWidget !== "function") return false;
    if ((node.widgets || []).some((x) => x && x.name === WIDGET_NAME)) return false;
    const w = node.addCustomWidget(makeBarWidget());
    if (w) w._node = node;
    return true;
}

app.registerExtension({
    name: "polyhedron.note",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated ? onCreated.apply(this, arguments) : undefined;
            this.properties = this.properties || {};
            ensureBar(this);
            if (!this.size || this.size[0] < 300) this.size = [300, 210];
            return r;
        };

        // A note saved before v891 has no bar widget; one saved by v892 has a
        // colour property and -- because v892's serialize deleted them -- no
        // color/bgcolor at all. Both are healed here, once.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            if (ensureBar(this) && this.size && this.size[1] < BAR_H + 140) {
                this.size[1] = BAR_H + 140;
            }
            migrateLegacy(this);
            return r;
        };

        // NO serialize() patch. v892 had one and it ate the right-click
        // colour; LiteGraph's own color/bgcolor are the single source now, and
        // ph_palette.js already strips them while they are still its default.
    },
});

export {
    PROP_LEGACY_COLOUR, PROP_COLLAPSED, LEGACY_MAP,
    BAR_H, COLLAPSED_H, WIDGET_TYPE, WIDGET_NAME,
};
