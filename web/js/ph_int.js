/* ph_int.js -- v879 (rebuilt; v878's own drawn number field is gone)
 *
 * ⬡ Polyhedron Int -- named presets under an ORDINARY LiteGraph int widget.
 *
 * WHAT v878 GOT WRONG, and why this is a rebuild rather than a repair:
 *
 *   1. It HID the canon `value` widget and painted its own number band. That
 *      removed LiteGraph's input: no click into the field, no keyboard, no
 *      typing "20". A redrawn control must do everything the original did, or
 *      it is a regression with better paint.
 *   2. The band was 46px plus a chip row -- more than twice the height of the
 *      foreign node it replaces, which held the same thing in two 20px rows.
 *      In a graph, bigger is not prettier.
 *   3. layout() ran INSIDE onDrawForeground and called setSize there. A size
 *      written mid-paint lands one frame late; while dragging, that frame
 *      never catches up, so the chips stood outside the box.
 *   4. onResize deliberately did NOT floor the height, reasoning that two
 *      places must not clamp one number. Right rule, wrong end: it guards
 *      against clamping UPWARD. Downward the content needs a floor, or the box
 *      is simply dragged smaller than what it holds.
 *
 *   Noted honestly: the scrub gesture v878 advertised as its own trick ALREADY
 *   EXISTS -- LiteGraph drags number widgets horizontally out of the box. v878
 *   hid the widget that had it and then re-implemented it worse. Gone; the
 *   stock behaviour is back.
 *
 * WHAT IS DRAWN NOW: only what core has no answer for -- a row of named preset
 * chips below the widgets, the lit one telling you which meaning is active.
 * The number is LiteGraph's, with its arrows, its drag and its keyboard.
 *
 * ONE SOURCE OF HEIGHT (the v809 lesson, lifted rather than re-derived):
 * node.computeSize()[1] is title + sockets + widgets. The chip block is added
 * to THAT, and the total is written on EVENTS -- never during a paint.
 */

import { app } from "../../scripts/app.js";

const NODE_TYPE = "ULSInt";

const PAD = 8;
const CHIP_H = 18;
const CHIP_GAP = 5;
const CHIP_PAD_X = 8;
const ADD_W = 20;
const BLOCK_TOP = 2;
const BLOCK_BOT = 6;

/* MEASURED floor: PAD + two useful chips + gap + PAD = 8 + 2*46 + 5 + 8 = 113.
 * Rounded to 140 so a chip pair never wraps at the smallest size. LiteGraph's
 * own minimum for the int widget row is wider still, so in practice the widget
 * binds -- this floor only stops the node being crushed below its contents. */
const MIN_W = 140;
const START_W = 210;

function widget(node, name) {
    return node.widgets ? node.widgets.find(w => w.name === name) : null;
}

function getValue(node) {
    const w = widget(node, "value");
    const v = w ? parseInt(w.value, 10) : 0;
    return Number.isFinite(v) ? v : 0;
}

/* Every write goes through here: clamp, store, fire the widget callback so
 * anything chained to it hears about it. The widget stays the only holder of
 * the number -- nothing here keeps a copy. */
function setValue(node, v) {
    const w = widget(node, "value");
    if (!w) return;
    v = Math.round(Number(v) || 0);
    v = Math.max(-2147483648, Math.min(2147483647, v));
    if (w.value === v) return;
    w.value = v;
    if (w.callback) { try { w.callback(v); } catch { /* not ours to own */ } }
    node.setDirtyCanvas?.(true, true);
}

function getPresets(node) {
    const w = widget(node, "preset_config");
    if (!w || !w.value) return [];
    try {
        const d = JSON.parse(w.value);
        const rows = Array.isArray(d) ? d : d.presets;
        if (!Array.isArray(rows)) return [];
        return rows
            .filter(r => r && Number.isFinite(Number(r.value)))
            .map(r => ({ name: String(r.name ?? ""),
                         value: Math.round(Number(r.value)) }));
    } catch { return []; }
}

function setPresets(node, rows) {
    let w = widget(node, "preset_config");
    if (!w) w = node.addWidget("text", "preset_config", "", () => {});
    w.value = JSON.stringify({ presets: rows });
    hideConfig(node);
    relayout(node);
}

/* Hidden by TYPE, not by a flag: a bare `hidden = true` is honoured by the
 * painter but not by the layout pass, and the node keeps a gap where the row
 * was (the v753/v755 lesson). */
function hideConfig(node) {
    const w = widget(node, "preset_config");
    if (!w) return;
    w.hidden = true;
    w.type = "hidden";
    w.computeSize = () => [0, -4];
}

/* Geometry is computed HERE and read by both the painter and the hit test.
 * Two places measuring the same rectangle is how a click lands one chip to the
 * left. NEVER called from a paint. */
function relayout(node) {
    const rows = getPresets(node);
    const base = node.computeSize()[1];      // title + sockets + widgets
    const W = Math.max(node.size[0] || 0, MIN_W);
    const avail = W - PAD * 2;

    const zones = [];
    let x = PAD, y = base + BLOCK_TOP, lines = 1;
    for (let i = 0; i < rows.length; i++) {
        const text = rows[i].name || String(rows[i].value);
        const w = Math.max(34, Math.min(avail, text.length * 6 + CHIP_PAD_X * 2));
        if (x > PAD && x + w > PAD + avail) {
            x = PAD; y += CHIP_H + CHIP_GAP; lines++;
        }
        zones.push({ i, x, y, w, h: CHIP_H, row: rows[i] });
        x += w + CHIP_GAP;
    }
    if (x > PAD && x + ADD_W > PAD + avail) {
        x = PAD; y += CHIP_H + CHIP_GAP; lines++;
    }
    const addZone = { x, y, w: ADD_W, h: CHIP_H };

    const blockH = BLOCK_TOP + lines * CHIP_H + (lines - 1) * CHIP_GAP + BLOCK_BOT;
    const st = node._phi || (node._phi = {});
    st.zones = zones;
    st.addZone = addZone;
    st.rows = rows;
    st.minH = base + blockH;          // the floor onResize needs

    // grow-only width, height follows the content (v877)
    const w = Math.max(MIN_W, node.size[0] || 0);
    node.size[0] = w;
    node.size[1] = st.minH;
    if (node.setSize) node.setSize([w, st.minH]);
    node.setDirtyCanvas?.(true, true);
    return st.minH;
}

function roundRect(ctx, x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
}

app.registerExtension({
    name: "Polyhedron.int",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const _created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            _created?.apply(this, arguments);
            this._phi = { hover: -1, hoverAdd: false,
                          zones: [], addZone: null, rows: [], minH: 0 };
            this.size[0] = Math.max(this.size[0] || 0, START_W);
            hideConfig(this);
            relayout(this);
        };

        const _cfg = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (o) {
            _cfg?.apply(this, arguments);
            this._phi = this._phi || { hover: -1, hoverAdd: false,
                                       zones: [], addZone: null, rows: [],
                                       minH: 0 };
            hideConfig(this);
            relayout(this);
        };

        /* The height floor lives HERE, and it is the height relayout computed.
         * v878 left it out on purpose and the box could be dragged smaller
         * than its own contents. Neither floor clamps upward, so the node
         * still grows freely. */
        nodeType.prototype.onResize = function (size) {
            if (size[0] < MIN_W) size[0] = MIN_W;
            const minH = this._phi?.minH || 0;
            if (minH > 0 && size[1] < minH) size[1] = minH;
        };

        // ── Draw: chips only. No layout, no setSize. ───────────────────
        nodeType.prototype.onDrawForeground = function (ctx) {
            if (this.flags?.collapsed) return;
            const st = this._phi; if (!st || !st.zones) return;

            const value = getValue(this);
            const active = st.rows.findIndex(r => r.value === value);

            for (const z of st.zones) {
                const isActive = z.i === active;
                const isHover = st.hover === z.i;
                ctx.fillStyle = isActive ? "#2f6a3a" : (isHover ? "#26263a" : "#1a1a2a");
                roundRect(ctx, z.x, z.y, z.w, z.h, 4); ctx.fill();
                ctx.strokeStyle = isActive ? "#7fd08a" : (isHover ? "#4a4a66" : "#2a2a3a");
                ctx.lineWidth = isActive ? 1.3 : 0.8;
                roundRect(ctx, z.x, z.y, z.w, z.h, 4); ctx.stroke();
                ctx.fillStyle = isActive ? "#eaffea" : "#a8a8b8";
                ctx.font = (isActive ? "bold " : "") + "10px 'Segoe UI',Arial";
                ctx.textAlign = "center"; ctx.textBaseline = "middle";
                ctx.fillText(z.row.name || String(z.row.value),
                             z.x + z.w / 2, z.y + z.h / 2 + 0.5);
            }

            const a = st.addZone;
            if (a) {
                ctx.fillStyle = st.hoverAdd ? "#26263a" : "#16161f";
                roundRect(ctx, a.x, a.y, a.w, a.h, 4); ctx.fill();
                ctx.strokeStyle = st.hoverAdd ? "#4a4a66" : "#2a2a3a";
                ctx.lineWidth = 0.8;
                roundRect(ctx, a.x, a.y, a.w, a.h, 4); ctx.stroke();
                ctx.fillStyle = st.hoverAdd ? "#8fb8ff" : "#6a6a7a";
                ctx.font = "bold 11px 'Segoe UI',Arial";
                ctx.textAlign = "center"; ctx.textBaseline = "middle";
                ctx.fillText("+", a.x + a.w / 2, a.y + a.h / 2 + 0.5);
                if (!st.rows.length) {
                    ctx.fillStyle = "#5a5a6a";
                    ctx.font = "italic 10px 'Segoe UI',Arial";
                    ctx.textAlign = "left";
                    ctx.fillText("name a value", a.x + a.w + 6, a.y + a.h / 2 + 0.5);
                }
            }
            ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
        };

        // ── Hit tests: read the shared geometry, never re-measure ───────
        const inRect = (lx, ly, r) =>
            r && lx >= r.x && lx <= r.x + r.w && ly >= r.y && ly <= r.y + r.h;

        nodeType.prototype.onMouseMove = function (e, [lx, ly]) {
            const st = this._phi; if (!st) return false;
            const prevH = st.hover, prevA = st.hoverAdd;
            st.hover = -1; st.hoverAdd = false;
            for (const z of st.zones) if (inRect(lx, ly, z)) { st.hover = z.i; break; }
            if (inRect(lx, ly, st.addZone)) st.hoverAdd = true;
            if (prevH !== st.hover || prevA !== st.hoverAdd) {
                this.setDirtyCanvas?.(true, false);
            }
            return false;
        };

        nodeType.prototype.onMouseDown = function (e, [lx, ly]) {
            const st = this._phi; if (!st) return false;

            for (const z of st.zones) {
                if (!inRect(lx, ly, z)) continue;
                if (e && e.button === 2) {
                    const rows = st.rows.slice();
                    rows.splice(z.i, 1);
                    setPresets(this, rows);
                    return true;
                }
                setValue(this, z.row.value);
                return true;
            }

            if (inRect(lx, ly, st.addZone)) {
                const name = prompt("Preset name", "");
                if (name === null) return true;
                const raw = prompt("Value", String(getValue(this)));
                if (raw === null) return true;
                const v = parseInt(raw, 10);
                if (!Number.isFinite(v)) return true;
                const rows = st.rows.slice();
                rows.push({ name: String(name), value: v });
                setPresets(this, rows);
                setValue(this, v);
                return true;
            }
            return false;   // everything above belongs to LiteGraph's widget
        };

        nodeType.prototype.onMouseLeave = function () {
            const st = this._phi; if (!st) return;
            st.hover = -1; st.hoverAdd = false;
            this.setDirtyCanvas?.(true, false);
        };
    },
});
