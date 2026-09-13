/**
 * uls_vue_views.js -- the ONE switch point between the two renderers (v936).
 *
 * WHAT THIS IS
 * ------------
 * Nodes whose body is painted on the LiteGraph canvas show nothing under
 * ComfyUI's Vue renderer ("Nodes 2.0"), which never calls onDrawForeground.
 * Each such node gets a second VIEW built from DOM, shown only under that
 * renderer. This module owns the part every one of those views shares: when a
 * view exists, when it is shown, when it goes away. A node brings only what is
 * its own -- how to build its view and when it is ready -- via registerVueView.
 * One copy of the switching logic instead of one per node: copies drift.
 *
 * THE RULES, and why each holds by construction (v934, measured in a real
 * browser against ComfyUI 0.33.4 / frontend 1.49.6):
 *   * The renderer is read from LiteGraph.vueNodesMode, a runtime flag the
 *     frontend flips itself. Anything but `true` means classic. (A settings
 *     store may not be ready when nodes are built; "no canvas frame for a
 *     while" is not a fact either -- LiteGraph does not paint while idle.)
 *   * Under the classic renderer NO widget is created. The painted node stays
 *     exactly what it was; this module leaves no hook, timer or widget on it.
 *     The classic node is the benchmark and must be usable at any time.
 *   * A view is attached the first time the flag reads true. Switching back
 *     sets widget.hidden = true -- the one switch that both LiteGraph's layout
 *     and the frontend's DOM-widget overlay honour. Hiding only the inner
 *     element leaves the overlay's container clickable over the node: that
 *     was the v922-v928 click wall (and is Power Upscale's today, S0 10.09.).
 *   * widget.serialize AND options.serialize are false: the frontend does not
 *     copy one onto the other (1.49.6), and a view must store nothing -- not in
 *     the workflow, not in the prompt.
 *   * One watch for all registered views, working only on transitions.
 *   * No draw hooks, no computeSize edits, no timers that guess.
 *
 * THE CONTRACT (registerVueView(className, spec))
 *   spec.widgetName   name of the DOM widget (must not collide with inputs)
 *   spec.render(node, root)  build the view into `root` from the node's OWN
 *                     state; called when the view is shown and on refresh
 *   spec.ready(node)  optional; false = not configured yet, try next tick
 *   spec.leave(node)  optional; called when switching back to classic (the
 *                     Stack hands its height back to the painted view here)
 *   spec.prepare()    optional; called before the first view is built
 *   spec.className    optional CSS class for the root element
 *   spec.label        optional; the console names the view with it
 *   spec.signature(node)  optional (v941); a cheap value that changes when
 *                     the view should be redrawn -- checked every tick while
 *                     the view is on screen, render() only on a change. For
 *                     views that mirror state the node changes by itself
 *                     (a word count, a status line). Without it: unchanged.
 * The view owns no state. It edits the node's own state and persists it the
 * way the painted view does.
 *
 * This file names no node class on purpose -- a class appears only where it
 * registers.
 */
import { app } from "../../scripts/app.js";

const MODE_POLL_MS = 400;
const _views = new Map();          // comfyClass -> spec

/** The only renderer test. Unknown or missing => classic. */
export function vueMode() {
    try {
        return globalThis.LiteGraph?.vueNodesMode === true;
    } catch (e) {
        return false;
    }
}

function classOf(node) {
    return node?.comfyClass || node?.type;
}

function specOf(node) {
    return _views.get(classOf(node)) || null;
}

/** Register the DOM view of one node class. Safe to call before any node. */
export function registerVueView(className, spec) {
    if (!className || !spec || !spec.widgetName
            || typeof spec.render !== "function") {
        throw new Error("[ULS] registerVueView needs a class, a widgetName "
                        + "and a render(node, root)");
    }
    _views.set(className, spec);
    startWatch();
}

/** Build the view once, on the first moment it is actually needed. */
function attach(node, spec) {
    if (node._ulsVueView) return node._ulsVueView;
    if (!_prepared.has(spec)) {
        _prepared.add(spec);
        try { spec.prepare?.(); } catch (e) { /* a view never breaks a node */ }
    }
    const root = document.createElement("div");
    if (spec.className) root.className = spec.className;
    const widget = node.addDOMWidget(spec.widgetName, "div", root, {
        serialize: false,   // frontend: keeps the view out of the PROMPT
        hideOnZoom: false,
    });
    // The frontend does NOT copy options.serialize onto the widget, and
    // LiteGraph's workflow writer reads widget.serialize only.
    widget.serialize = false;
    node._ulsVueView = { widget, root, shown: false };
    return node._ulsVueView;
}
const _prepared = new Set();

/** Bring ONE node in line with the renderer; works only on a transition.
 *  Returns the view's label when it switched on/off, else null (v942: the
 *  watch collects them into one console line per tick). */
function syncNode(node, vue) {
    const spec = specOf(node);
    if (!spec) return;
    const label = spec.label || spec.widgetName;
    if (vue) {
        if (spec.ready && !spec.ready(node)) return;   // not configured yet
        const v = attach(node, spec);
        if (v.shown && !v.widget.hidden) {
            if (spec.signature) {                      // v941: live, on change only
                const sig = signatureOf(spec, node);
                if (sig !== v.sig) {
                    v.sig = sig;
                    spec.render(node, v.root);
                }
            }
            return;
        }
        v.widget.hidden = false;
        v.shown = true;
        v.sig = spec.signature ? signatureOf(spec, node) : undefined;
        spec.render(node, v.root);                     // state may have moved
        node.setDirtyCanvas?.(true, true);
        return label;
    } else {
        const v = node._ulsVueView;
        if (!v) return;                                // never attached
        if (!v.shown && v.widget.hidden) return;
        v.widget.hidden = true;
        v.shown = false;
        spec.leave?.(node);
        node.setDirtyCanvas?.(true, true);
        return label;
    }
    return null;
}

/** v942: ONE console line for all views that switched in one tick -- before,
 *  every view wrote its own, so a large graph filled the console on every
 *  renderer switch. Same words as before, with a count per label. */
function reportSwitch(vue, labels) {
    if (!labels.length) return;
    const n = {};
    for (const l of labels) n[l] = (n[l] || 0) + 1;
    const parts = Object.keys(n).map(l => (n[l] > 1 ? l + " x" + n[l] : l));
    console.log(`[ULS] ${parts.join(", ")} ${vue ? "on -- Nodes 2.0" : "off -- classic"} renderer`);
}

function signatureOf(spec, node) {
    try {
        return String(spec.signature(node));
    } catch (e) {
        return "\u0000error";                          // a broken signature never throws out
    }
}

/** Re-render a view that is on screen. False when there is none to redraw. */
export function refreshVueView(node) {
    const spec = specOf(node);
    const v = node?._ulsVueView;
    if (!spec || !v || !v.shown) return false;
    spec.render(node, v.root);
    return true;
}

/** Every node the user can reach: root graph, open graph, all subgraphs. */
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
    for (const gr of graphs) {
        for (const n of (gr?._nodes || gr?.nodes || [])) out.push(n);
    }
    return out;
}

function syncAll() {
    if (_views.size === 0) return;
    let vue;
    try { vue = vueMode(); } catch (e) { return; }
    const switched = [];
    for (const n of allNodes()) {
        if (!specOf(n)) continue;
        try {
            const l = syncNode(n, vue);
            if (l) switched.push(l);
        } catch (e) {
            // one broken view must not stop the others, nor the watch
            console.warn("[ULS] vue view failed:", classOf(n), e);
        }
    }
    reportSwitch(vue, switched);
}

let _watch = null;
function startWatch() {
    if (_watch) return;
    _watch = setInterval(syncAll, MODE_POLL_MS);
}

app.registerExtension({
    name: "Polyhedron.vueViews",

    setup() {
        startWatch();
    },

    nodeCreated(node) {
        if (!specOf(node)) return;
        // v924 mark: under Nodes 2.0 this node HAS a replacement view, so
        // uls_compat.js must not tell the user to switch renderers.
        node._ulsDomPanel = true;
        startWatch();
        // Deferred by a tick: a node fills its own state during its own
        // nodeCreated/configure. Under the classic renderer this does NOTHING.
        setTimeout(() => {
            try {
                const vue = vueMode();
                const l = syncNode(node, vue);
                if (l) reportSwitch(vue, [l]);
            } catch (e) { /* next tick */ }
        }, 0);
    },
});
