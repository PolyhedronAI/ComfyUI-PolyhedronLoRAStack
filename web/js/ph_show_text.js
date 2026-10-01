/**
 * ph_show_text.js -- the view half of ⬡ Polyhedron Show Text.
 *
 * One read-only DOM textarea, monospace, then the Polyhedron Note's colour
 * row -- imported from ph_note.js, never copied, so the two rows cannot drift.
 *
 * v991 -- sized like the CLIP Text Encode (Frank, 22.09.: "die spreizt sich in
 * die Breite statt in der Laenge ... schau, wie das bei der Clip Text Encode
 * geloest ist"):
 *   - the text WRAPS at the node's width; the width is the user's, the node
 *     never widens itself (v990's auto-widening is gone);
 *   - the field's reserved height is the height the text needs AT THE CURRENT
 *     WIDTH. It is computed on every layout call from node.size[0] -- the CTE's
 *     v715 rule: the reservation follows the width by itself, and LiteGraph
 *     clamps a manual resize against computeSize, so dragging the node
 *     narrower makes it taller and the text never hides;
 *   - NO onResize hook (guard #604: a proven dead end, the v610/v611 loop);
 *   - past SHOW_MAX_H the field stops growing and scrolls.
 * A run or a reload refits the height once laid out (refit(), never
 * setSize(computeSize()) -- guard v897), then adds what the renderer kept back
 * (Nodes 2.0 keeps ~24 px more for its frame; v990 probe).
 *
 * The shown text lives in node.properties.ph_show_text -- NOT in a widget
 * value: the field has serialize:false, so widgets_values stays empty and a
 * saved workflow can never shift (guard #577).
 */

import { app } from "../../scripts/app.js";
import { refit } from "./ph_widget_vis.js";
import { makeBarWidget, BAR_H } from "./ph_note.js";

const NODE_TYPE = "ULSShowText";
const FIELD = "ph_show_text_field";
const PROP_TEXT = "ph_show_text";
const BAR_NAME = "$ph_note_bar";

export const LINE_H = 14;            // px per line at 11 px monospace
export const SHOW_PAD = 10;          // textarea padding, top + bottom
export const SHOW_MIN_H = 60;
export const SHOW_MAX_H = 2000;      // then it scrolls (the CTE's FIELD_MAX_H)
export const CHAR_W = 6.7;           // px per character at 11 px monospace (fallback)
export const FRAME_W = 44;           // node width minus the field's text width
export const DEFAULT_W = 420;        // a new node's width
const FONT_FAMILY = "ui-monospace, 'Cascadia Mono', Consolas, 'DejaVu Sans Mono', monospace";
const EMPTY = "run the graph to show the text here";

// The width of one monospace character in the field's own font; a canvas
// measureText where there is one, CHAR_W otherwise (tests).
let _cw = 0;
export function charWidth() {
    if (_cw) return _cw;
    try {
        if (typeof document !== "undefined" && document.createElement) {
            const c = document.createElement("canvas");
            const ctx = c && c.getContext ? c.getContext("2d") : null;
            if (ctx && ctx.measureText) {
                ctx.font = "11px " + FONT_FAMILY;
                const w = ctx.measureText("0123456789").width / 10;
                if (w > 0) { _cw = w; return _cw; }
            }
        }
    } catch (e) { /* fall back */ }
    return CHAR_W;
}

// Visual lines of `text` wrapped at `width` px of node.
export function wrappedLines(text, width) {
    const t = String(text || "");
    if (!t) return 3;
    const cols = Math.max(8, Math.floor((Math.max(0, width - FRAME_W)) / charWidth()));
    let n = 0;
    for (const l of t.split("\n")) n += Math.max(1, Math.ceil(l.length / cols));
    return n;
}

// THE height: what the text needs at this node width, between the caps.
export function fieldHeight(text, width) {
    const n = wrappedLines(text, width || DEFAULT_W);
    return Math.max(SHOW_MIN_H, Math.min(SHOW_MAX_H, n * LINE_H + SHOW_PAD + 4));
}

function field(node) {
    return (node.widgets || []).find((w) => w && w.name === FIELD) || null;
}

function _setOverflow(node, el) {
    if (!el || !el.style) return;
    const w = node && node.size ? node.size[0] : DEFAULT_W;
    const need = wrappedLines(el.value, w) * LINE_H + SHOW_PAD;
    el.style.overflowX = "hidden";                        // it wraps -- never sideways
    el.style.overflowY = need > SHOW_MAX_H ? "auto" : "hidden";
}

function ensureWidgets(node) {
    if (typeof node.addDOMWidget !== "function") return null;
    let w = field(node);
    if (!w) {
        const el = document.createElement("textarea");
        el.readOnly = true;
        el.wrap = "soft";                         // v991: wrap at the node's width
        el.spellcheck = false;
        el.classList.add("comfy-multiline-input");
        Object.assign(el.style, {
            width: "100%", boxSizing: "border-box", resize: "none",
            fontFamily: FONT_FAMILY,
            fontSize: "11px", lineHeight: LINE_H + "px",
            whiteSpace: "pre-wrap", overflowWrap: "anywhere", wordBreak: "break-all",
            overflowX: "hidden", overflowY: "hidden", padding: "5px 6px",
        });
        el.placeholder = EMPTY;
        // Height through the DOM widget's own layout hook (getMinHeight), which
        // both renderers read, computed from the CURRENT width on every call.
        w = node.addDOMWidget(FIELD, "ph_show_text", el, {
            serialize: false,
            getMinHeight: () => fieldHeight(el.value, node.size ? node.size[0] : DEFAULT_W),
        });
        w.serialize = false;                     // never in widgets_values (#577)
    }
    // The colour row sits BELOW the text -- move it to the end if it exists.
    const list = node.widgets || [];
    const bi = list.findIndex((x) => x && x.name === BAR_NAME);
    if (bi >= 0 && bi !== list.length - 1) list.push(list.splice(bi, 1)[0]);
    if (bi < 0 && typeof node.addCustomWidget === "function") {
        const bar = node.addCustomWidget(makeBarWidget());
        if (bar) bar._node = node;
    }
    return w;
}

// After layout: add what the renderer withheld (scrollHeight > clientHeight),
// once, from a one-shot frame. Skipped while not laid out or past the cap.
function _fitHeight(node, el) {
    if (!el || !node || !node.size || typeof node.setSize !== "function") return;
    const ch = el.clientHeight || 0, sh = el.scrollHeight || 0;
    if (ch <= 0 || !el.value) return;
    if (wrappedLines(el.value, node.size[0]) * LINE_H + SHOW_PAD > SHOW_MAX_H) return;
    const miss = Math.ceil(sh - ch);
    if (miss > 0 && miss < SHOW_MAX_H) {
        node.setSize([node.size[0], node.size[1] + miss]);   // height only
        node.setDirtyCanvas?.(true, true);
    }
}

function _refitLaidOut(node, el) {
    refit(node);
    _setOverflow(node, el);
    const raf = (typeof requestAnimationFrame === "function")
        ? requestAnimationFrame : (fn) => setTimeout(fn, 0);
    raf(() => {
        refit(node);
        raf(() => { _fitHeight(node, el); _setOverflow(node, el); });
    });
}

export function showText(node, text) {
    const w = ensureWidgets(node);
    const t = text == null ? "" : String(text);
    node.properties = node.properties || {};
    node.properties[PROP_TEXT] = t;
    if (!w || !w.element) return;
    w.element.value = t;
    _refitLaidOut(node, w.element);          // height only -- the width is the user's
}

app.registerExtension({
    name: "polyhedron.showtext",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated ? onCreated.apply(this, arguments) : undefined;
            this.properties = this.properties || {};
            ensureWidgets(this);
            if (!this.size || this.size[0] < 320) this.size = [DEFAULT_W, this.size ? this.size[1] : 140];
            refit(this);
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            const txt = message && message.text;
            showText(this, Array.isArray(txt) ? txt.join("\n") : (txt ?? ""));
            return r;
        };

        // NO onResize hook (guard #604). The reservation reads node.size[0] on
        // every call, so a narrower node reserves more height by itself.

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            const t = this.properties && this.properties[PROP_TEXT];
            if (t) showText(this, t);
            else ensureWidgets(this);
            return r;
        };
    },
});

export { NODE_TYPE, FIELD, PROP_TEXT, BAR_H };
