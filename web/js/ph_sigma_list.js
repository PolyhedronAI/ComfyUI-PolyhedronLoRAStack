// ph_sigma_list.js v984 -- the Sigma List shows what it does, and draws it.
//
// v973 made a preset override the widgets at RUN time. That was right, but it
// left the node lying on screen: `shift` read 1.00 while the run used 12.0,
// and the only witness was a console line. A preset you cannot see is half a
// preset -- the same trap in a new place.
//
// So: picking a preset WRITES its values into the widgets and greys them out.
// Back to `custom` hands them back, restoring what the user had typed. The
// backend override in v973 stays as the belt to this braces: a workflow saved
// before v974, or one edited by hand, still runs the preset's recipe.
//
// The values come from /pls/sigma_presets -- the ONE table in
// wan_sigma_schedule.py. Nothing about a grid is repeated in this file.

import { app } from "../../scripts/app.js";
// The ONE way to let a node re-measure its HEIGHT. Never setSize(computeSize()):
// that assigns the computed WIDTH back too and throws away whatever the user
// dragged out (guard v897, the law v888 gave one home).
import { refit } from "./ph_widget_vis.js";
// v984: the curve is drawn by the ONE sigma plot module, shared with the Sigma
// Curve and the Dual Sigma Curve (ph_sigma_curves.js). Height bounds, dot size
// and the refusal text live there once.
import { plotHeightFor, drawSigmaPlot } from "./ph_sigma_plot.js";

const NODE_ID = "ULSSigmaList";
const ROUTE = "/pls/sigma_presets";
const CUSTOM = "custom";
const USER_KEY = "_slUserValues";   // node.properties: what the user typed
const PREVIEW_ROUTE = "/pls/sigma_preview";
// v977: the plot is a WIDGET, not paint on the node's lower edge. v976 drew
// into `node.size[1] - PLOT_H`, which is wherever the widgets happen to be --
// the multiline field stretches into any height the node gains, so the curve
// landed on top of them. A custom widget is measured and placed by LiteGraph,
// so nothing can overlap it and it cannot overlap anything.
//
// It carries no value and declares serialize:false: it must never appear in
// widgets_values, or it would renumber every saved workflow (guard #577).
function plotHeight(node) {
    const p = node._slPreview;
    return plotHeightFor(p && p.ok && p.sigmas ? p.sigmas.length : 9);
}

let _presets = null;
let _pending = null;

async function loadPresets() {
    if (_presets) return _presets;
    if (_pending) return _pending;
    _pending = (async () => {
        for (const url of [`/api${ROUTE}`, ROUTE]) {
            try {
                const r = await fetch(url);
                if (!r.ok) continue;
                const j = await r.json();
                if (j && j.ok && j.presets) {
                    _presets = j.presets;
                    return _presets;
                }
            } catch (e) { /* try the next form */ }
        }
        // Served table unreachable: leave every widget editable rather than
        // freezing fields we cannot fill. The backend still applies the recipe.
        console.warn("[PLS] Sigma List: preset table unreachable; widgets stay editable.");
        _presets = {};
        return _presets;
    })();
    return _pending;
}

// ---- v976: the curve, drawn from what the RUN would compute ---------------
//
// The node draws the grid the backend resolves, never one this file works out
// for itself: the shift arithmetic lives in wan_sigma_schedule.py, and a
// second copy here would drift from it. So we ask and we draw. A refusal comes
// back as text and is shown as text -- which is how a typo in the list becomes
// visible in the node instead of at run time.

async function fetchPreview(node) {
    const w = widgetsOf(node);
    const body = JSON.stringify({
        sigmas_text: w.text ? String(w.text.value ?? "") : "",
        shift: w.shift ? Number(w.shift.value ?? 1.0) : 1.0,
        enforce_terminal_zero: w.zero ? !!w.zero.value : true,
        preset: w.preset ? String(w.preset.value ?? CUSTOM) : CUSTOM,
    });
    if (node._slLastBody === body) return;        // nothing changed
    node._slLastBody = body;
    for (const url of [`/api${PREVIEW_ROUTE}`, PREVIEW_ROUTE]) {
        try {
            const r = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body,
            });
            if (!r.ok) continue;
            node._slPreview = await r.json();
            // The height follows the point count, so a new answer may need a
            // taller node. refit() re-measures the HEIGHT and leaves the width
            // exactly as the user left it.
            refit(node);
            node.setDirtyCanvas(true, true);
            redrawVue(node);
            return;
        } catch (e) { /* try the next form */ }
    }
    node._slPreview = null;                        // unreachable: draw nothing
    node.setDirtyCanvas(true, true);
    redrawVue(node);
}

// v984: under Nodes 2.0 a canvas widget is painted into its OWN small canvas,
// once, and again only when the frontend's triggerDraw() is called (it hooks
// the widget's callback and a resize). A preview that lands later -- the
// normal case, it is a fetch -- would otherwise stay "no preview" forever.
// Measured in the v984 browser probe. The classic canvas redraws by itself.
function redrawVue(node) {
    const w = node._slPlot;
    if (w && typeof w.triggerDraw === "function") {
        // Its draw() takes the height from `computedHeight` FIRST (LiteGraph's
        // last layout pass) and only then asks computeSize -- so a plot that
        // grew with its knots stayed clipped at the old height (probe: 124 px
        // for a 41-point curve that wants 260). Hand it the new height.
        w.computedHeight = plotHeight(node);
        try { w.triggerDraw(); } catch (e) { /* a stale binding: ignore */ }
    }
}

function schedulePreview(node) {
    clearTimeout(node._slTimer);
    node._slTimer = setTimeout(() => fetchPreview(node), 120);
}

function drawCurve(ctx, node, widgetWidth, y, height) {
    const p = node._slPreview;
    let spec;
    if (!p) spec = { state: null };
    else if (!p.ok) spec = { state: { error: p.error || "invalid" } };
    else {
        const sig = p.sigmas || [];
        spec = {
            header: sig.length >= 2
                ? `${p.steps} steps  \u00b7  sigma ${sig[0].toFixed(3)} -> ` +
                  `${sig[sig.length - 2].toFixed(3)}  \u00b7  shift ${p.shift}`
                : "",
            series: [{ sigmas: sig, color: "#8fff8f" }],   // the suite's green
        };
    }
    drawSigmaPlot(ctx, widgetWidth, y, height, spec);
}

function ensurePlotWidget(node) {
    if (node._slPlot) return node._slPlot;
    const wdg = {
        type: "poly_sigma_curve",
        name: "curve",
        value: null,
        serialize: false,               // NEVER in widgets_values (guard #577)
        options: { serialize: false },
        draw(ctx, n, widgetWidth, y) {
            drawCurve(ctx, n, widgetWidth, y, plotHeight(n));
        },
        computeSize(width) {
            return [width, plotHeight(node)];
        },
    };
    node._slPlot = wdg;
    (node.widgets = node.widgets || []).push(wdg);
    return wdg;
}

function widgetsOf(node) {
    const by = {};
    for (const w of node.widgets || []) by[w.name] = w;
    return {
        text: by["sigmas_text"],
        shift: by["shift"],
        zero: by["enforce_terminal_zero"],
        preset: by["preset"],
    };
}

function domEl(w) {
    // A multiline widget IS a DOM textarea. Setting widget.value alone updates
    // the model but not always what the eye sees -- v974 shipped with an empty
    // grid field for exactly this reason. Write BOTH, always.
    if (!w) return null;
    return w.inputEl || (w.element && w.element.querySelector("textarea,input")) || null;
}

function setValue(w, v) {
    if (!w) return;
    w.value = v;
    const el = domEl(w);
    if (el && el.value !== undefined) el.value = v;
}

function setDisabled(w, off) {
    if (!w) return;
    w.disabled = !!off;                       // LiteGraph draws it greyed
    const el = domEl(w);
    if (el) {                                 // DOM widget (the multiline text)
        el.readOnly = !!off;
        el.style.opacity = off ? "0.55" : "";
        el.style.cursor = off ? "not-allowed" : "";
    }
}

function rememberUser(node, w) {
    // Only ever snapshot what the USER had; never a preset's own values, or a
    // second switch would bake the preset in as "the user's".
    node.properties = node.properties || {};
    if (node.properties[USER_KEY]) return;
    node.properties[USER_KEY] = {
        sigmas_text: w.text ? w.text.value : "",
        shift: w.shift ? w.shift.value : 1.0,
        enforce_terminal_zero: w.zero ? w.zero.value : true,
    };
}

function restoreUser(node, w) {
    const saved = (node.properties || {})[USER_KEY];
    if (!saved) return;
    if (saved.sigmas_text !== undefined) setValue(w.text, saved.sigmas_text);
    if (saved.shift !== undefined) setValue(w.shift, saved.shift);
    if (saved.enforce_terminal_zero !== undefined) setValue(w.zero, saved.enforce_terminal_zero);
    delete node.properties[USER_KEY];
}

function applyPreset(node, presets) {
    const w = widgetsOf(node);
    if (!w.preset) return;
    const name = String(w.preset.value || CUSTOM);
    const entry = presets ? presets[name] : null;

    if (!entry || !entry.grid) {            // custom, or a name this build lacks
        restoreUser(node, w);
        setDisabled(w.text, false);
        setDisabled(w.shift, false);
        setDisabled(w.zero, false);
    } else {
        rememberUser(node, w);
        setValue(w.text, entry.grid.join(", "));
        if (entry.shift !== null && entry.shift !== undefined) setValue(w.shift, entry.shift);
        if (entry.enforce_terminal_zero !== null && entry.enforce_terminal_zero !== undefined) {
            setValue(w.zero, entry.enforce_terminal_zero);
        }
        setDisabled(w.text, true);
        setDisabled(w.shift, true);
        setDisabled(w.zero, true);
    }
    node.setDirtyCanvas(true, true);
    schedulePreview(node);
}

app.registerExtension({
    name: "Polyhedron.SigmaList",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_ID) return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            const node = this;
            const w = widgetsOf(node);
            if (w.preset) {
                const prevCallback = w.preset.callback;
                w.preset.callback = function (value) {
                    const out = prevCallback?.apply(this, arguments);
                    loadPresets().then((p) => applyPreset(node, p));
                    return out;
                };
            }
            // Every widget change re-asks the backend; the strip redraws when
            // the answer lands. Debounced, and a repeated body is not re-sent.
            for (const name of ["sigmas_text", "shift", "enforce_terminal_zero"]) {
                const ww = (node.widgets || []).find((x) => x.name === name);
                if (!ww) continue;
                const prev = ww.callback;
                ww.callback = function () {
                    const out = prev?.apply(this, arguments);
                    schedulePreview(node);
                    return out;
                };
            }

            // The plot is a widget now: LiteGraph measures and places it, so
            // it can neither cover a field nor be covered by one.
            ensurePlotWidget(node);

            loadPresets().then((p) => applyPreset(node, p));
            schedulePreview(node);
            return r;
        };

        // A loaded workflow restores widget values AFTER creation, so the
        // greying has to run again once the values are in place.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            const node = this;
            loadPresets().then((p) => applyPreset(node, p));
            return r;
        };
    },
});
