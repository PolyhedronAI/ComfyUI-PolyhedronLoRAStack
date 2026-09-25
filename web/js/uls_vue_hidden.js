/**
 * uls_vue_hidden.js -- a widget hidden in the classic view is hidden in
 * Nodes 2.0 too (v940, S5).
 *
 * THE FINDING (measured 10.09.2026, ComfyUI 0.33.4 / frontend 1.49.6)
 * ------------------------------------------------------------------
 * The two renderers read two different switches:
 *   classic   LGraphNode.isWidgetVisible   -> widget.hidden
 *   Nodes 2.0 useProcessedWidgets.isWidgetVisible -> widget.options.hidden
 * The suite hides widgets the classic way -- a "pls-hidden-" type prefix, a
 * zero computeSize, widget.hidden, or type "hidden" -- at some fifty places in
 * thirteen files. v940 read the first, third and fourth of those; the ZERO
 * HEIGHT was named here but never asked for, which is why one row (the Empty
 * Latent's frame pill) kept coming through until v947 -- a promise is only kept
 * where it is actually read. None of that reaches options.hidden, so under Nodes 2.0 the
 * internal widgets came back as visible, EDITABLE fields: the Mask Editor's
 * mask_store / points_store / path_store / layers_store / bg_key (typing into
 * one destroys the node's state), blank rows in Cutout and Empty Latent.
 *
 * THE RULE, one place for all of them
 * -----------------------------------
 * For every widget of this pack's nodes, options.hidden is DERIVED from the
 * classic state: hidden, type "hidden", or a "pls-hidden-" type. Every
 * existing hiding site -- and every future one -- is covered without touching
 * it; there is no second list to keep in step. The classic renderer never
 * reads options.hidden (it reads widget.hidden), so the painted views are not
 * touched by this. The frontend's parameter side panel does read it and stops
 * listing the same internal widgets -- intended.
 *
 * An explicit write to options.hidden (the frontend's own code, or ours) is
 * kept and wins until it is cleared with undefined; a pre-existing
 * options.hidden === true is kept as such an explicit value.
 *
 * RUNTIME CHANGES. Nodes 2.0 reads options.hidden as a SNAPSHOT when it
 * extracts a node's data (useGraphNodeManager.extractVueNodeData) and
 * re-evaluates visibility only when something reactive changes -- a widget
 * value does, a widget.hidden flip does not. Measured: the Stack's panel,
 * released a moment after Nodes 2.0 came back on, stayed out of the page. So
 * the watch remembers each widget's classic state and, under Nodes 2.0, gives
 * a node whose state changed ONE nudge: node:property:changed for
 * showAdvanced with its unchanged value -- the channel a node itself uses to
 * report a property, and one that makes Nodes 2.0 re-evaluate exactly this
 * node's widgets without changing anything. Under the classic renderer
 * nothing is sent.
 *
 * Pack nodes are recognised by class name (every class of this pack starts
 * "ULS" or is "UltimateLoraStack" -- the NODE_IDS baseline). Foreign nodes are
 * never touched.
 */
import { app } from "../../scripts/app.js";
// v942: the renderer test has ONE home -- the switch point. A private copy
// here (v940) was the same one-liner, and a mirror that can drift.
import { vueMode } from "./uls_vue_views.js";

const HIDDEN_TYPE_PREFIX = "pls-hidden-";   // ph_widget_vis.HIDDEN_PREFIX and its copies
const MIRROR_POLL_MS = 400;

/** A row the classic layout gives NO height is invisible there -- v947.
 *
 *  Measured 12.09. over all 58 nodes: 65 widgets report a computeSize height of
 *  zero or less, and 64 of them already carry one of the flags above. The one
 *  that does not is the Empty Latent's frame pill, which is a real widget that
 *  simply has nothing to say for most model types -- it says so by reporting
 *  height 0. Classic then draws nothing; Nodes 2.0 gave it a row (P1 class F2).
 *
 *  Only widgets WITHOUT a mounted element are judged this way. A DOM widget that
 *  momentarily reports no height would otherwise lose its element, and an element
 *  that is not in the page cannot grow back -- the panel would be gone for good.
 *  Those hide themselves through widget.hidden anyway (all 64 above do).
 */
export function zeroHeight(w) {
    if (!w || w.element || typeof w.computeSize !== "function") return false;
    let h;
    try { h = (w.computeSize(0) || [])[1]; } catch (e) { return false; }
    return typeof h === "number" && isFinite(h) && h <= 0;
}

/** The classic "is this widget hidden" -- every form the suite uses. */
export function classicHidden(w) {
    if (!w) return false;
    if (w.hidden) return true;
    const t = w.type;
    if (typeof t === "string" && (t === "hidden" || t.startsWith(HIDDEN_TYPE_PREFIX))) return true;
    return zeroHeight(w);
}

export function isPackNode(node) {
    const c = String(node?.comfyClass || node?.type || "");
    return c.startsWith("ULS") || c === "UltimateLoraStack";
}

/** v965: does this frontend already derive widget.hidden FROM options.hidden?
 *
 * Measured 19.09.2026. Frontend 1.53.6 gives BaseWidget
 *   get hidden(){ return this._state.options.hidden }   (settingStore-*.js)
 * and its classic and Vue isWidgetVisible both read widget.hidden -- the two
 * truths this file was written for are ONE truth there. Our getter mirror on
 * options.hidden then reads w.hidden -> options.hidden -> our getter -> ...
 * and no workflow loads at all (RangeError, measured on every pack node).
 * Frontend 1.49.6 keeps `hidden` a plain field on the instance: no accessor
 * anywhere on the prototype chain, the mirror is needed and harmless.
 *
 * The decision is read from the prototype chain of the widget itself, never
 * from a version string: the accessor IS the fact that matters. */
export function hiddenIsAccessor(w) {
    let p = (w && typeof w === "object") ? Object.getPrototypeOf(w) : null;
    while (p && p !== Object.prototype) {
        const d = Object.getOwnPropertyDescriptor(p, "hidden");
        if (d) return typeof d.get === "function";
        p = Object.getPrototypeOf(p);
    }
    return false;
}

/** "getter" = options.hidden is derived live (1.49.6 rule);
 *  "value"  = the frontend derives widget.hidden from options.hidden itself,
 *             so we only WRITE options.hidden for the forms it cannot see
 *             (a "pls-hidden-" type, type "hidden", a zero-height row). */
export function mirrorMode(w) {
    return hiddenIsAccessor(w) ? "value" : "getter";
}

/** value mode: push the classic verdict into options.hidden, never define a
 *  getter. Only the hidden direction is ever written: a widget the frontend
 *  hides through options.hidden already reads back as hidden here. */
function syncValue(w) {
    const o = w.options;
    if (classicHidden(w) && o.hidden !== true) {
        o.hidden = true;
        return true;
    }
    return false;
}

/** Make options.hidden follow the classic state. Idempotent; true if installed. */
export function mirrorWidget(w) {
    if (!w || typeof w !== "object") return false;
    if (!w.options || typeof w.options !== "object") w.options = {};
    const o = w.options;
    if (hiddenIsAccessor(w)) {
        // v965: never a getter on an accessor frontend (see hiddenIsAccessor)
        const first = !Object.prototype.hasOwnProperty.call(w, "__ulsHiddenSync");
        if (first) Object.defineProperty(w, "__ulsHiddenSync", { value: true, enumerable: false });
        syncValue(w);
        return first;
    }
    if (Object.prototype.hasOwnProperty.call(o, "__ulsHiddenMirror")) return false;
    let override = (o.hidden === true) ? true : undefined;
    Object.defineProperty(o, "hidden", {
        configurable: true,
        enumerable: true,
        get() { return override !== undefined ? override : classicHidden(w); },
        set(v) { override = (v === undefined || v === null) ? undefined : !!v; },
    });
    Object.defineProperty(o, "__ulsHiddenMirror", { value: true, enumerable: false });
    return true;
}

export function mirrorNode(node) {
    if (!isPackNode(node)) return 0;
    let n = 0;
    for (const w of (node.widgets || [])) if (mirrorWidget(w)) n++;
    return n;
}

function allNodes() {
    const graphs = new Set();
    const g = app.graph;
    if (g) graphs.add(g);
    if (app.canvas?.graph) graphs.add(app.canvas.graph);
    try {
        const subs = g?.subgraphs;
        if (subs?.values) for (const sg of subs.values()) graphs.add(sg);
    } catch (e) { /* older frontends have no subgraphs */ }
    const out = [];
    for (const gr of graphs) for (const n of (gr?._nodes || gr?.nodes || [])) out.push(n);
    return out;
}

const _lastHidden = new WeakMap();   // widget -> classic state seen last tick

/** Ask Nodes 2.0 to re-evaluate this node's widgets (see RUNTIME CHANGES). */
export function nudgeVue(node) {
    try {
        node.graph?.trigger?.("node:property:changed", {
            nodeId: node.id, property: "showAdvanced",
            oldValue: node.showAdvanced, newValue: node.showAdvanced,
        });
        return true;
    } catch (e) {
        return false;
    }
}

/** Mirror + notice changes; true when a widget's classic state changed. */
function syncNode(node) {
    mirrorNode(node);
    let changed = false;
    for (const w of (node.widgets || [])) {
        if (!w || typeof w !== "object") continue;
        const h = classicHidden(w);
        if (h && hiddenIsAccessor(w) && w.options && w.options.hidden !== true) {
            w.options.hidden = true;   // v965 value mode: a late "pls-hidden-" / zero-height verdict
        }
        if (_lastHidden.get(w) !== h) {
            if (_lastHidden.has(w)) changed = true;
            _lastHidden.set(w, h);
        }
    }
    return changed;
}

// Widgets added after creation (progressive slots, stores created on first
// configure, DOM views attached later) are picked up here; installing is a
// no-op once done, so a tick costs a flag read per widget.
let _watch = null;
function startWatch() {
    if (_watch) return;
    _watch = setInterval(() => {
        try {
            const vue = vueMode();
            for (const n of allNodes()) {
                if (!isPackNode(n)) continue;
                if (syncNode(n) && vue) nudgeVue(n);
            }
        } catch (e) { /* never throw */ }
    }, MIRROR_POLL_MS);
}

app.registerExtension({
    name: "Polyhedron.vueHidden",

    setup() {
        startWatch();
    },

    nodeCreated(node) {
        if (!isPackNode(node)) return;
        startWatch();
        mirrorNode(node);
        setTimeout(() => mirrorNode(node), 0);   // after the node's own setup
    },

    loadedGraphNode(node) {
        mirrorNode(node);
    },
});
