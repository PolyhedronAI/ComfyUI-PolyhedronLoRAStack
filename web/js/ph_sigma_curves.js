// ph_sigma_curves.js v984 -- the Sigma Curve and the Dual Sigma Curve draw
// the curve they output.
//
// Same rule as the Sigma List (v976): the node ASKS the backend for its curve
// (/pls/sigma_curve_preview -> wan_sigma_schedule.curve_preview ->
// universal_curve / split_curves, the functions compute() runs) and draws the
// answer. Nothing here computes a sigma. A second copy of eight schedules in
// JavaScript would be a second truth, and two truths drift.
//
// The Dual Sigma Curve shows BOTH lists in full, faint, and each one bold on
// the knots its sampler really runs: HIGH 0..split, LOW split..end, with a
// marker at the handoff. That is the one picture that answers "what does each
// pass get" -- both outputs are full-length lists and the samplers slice.
//
// The deprecated ULSWanSigmaSchedule gets no plot: it is kept only so old
// workflows load.

import { app } from "../../scripts/app.js";
// Never setSize(computeSize()) -- that throws away the width the user dragged
// out (guard v897). refit() re-measures the HEIGHT only.
import { refit } from "./ph_widget_vis.js";
import { plotHeightFor, drawSigmaPlot } from "./ph_sigma_plot.js";

const ROUTE = "/pls/sigma_curve_preview";
const HIGH_COLOR = "#ffb060";
const LOW_COLOR = "#6fc3ff";
const ONE_COLOR = "#8fff8f";                 // the suite's green, as the list

// node class -> which curve and which widgets feed it (INPUT_TYPES names)
const NODES = {
    ULSUniversalSigmaCurve: {
        kind: "curve",
        widgets: ["sigma_schedule", "steps", "sigma_max", "sigma_min", "rho"],
    },
    ULSWanSplitNoiseSchedule: {
        kind: "dual",
        widgets: ["schedule_high", "schedule_low", "total_steps", "split_step",
                  "sigma_max", "sigma_min", "rho_high", "rho_low"],
    },
};

function valueOf(node, name) {
    const w = (node.widgets || []).find((x) => x.name === name);
    return w ? w.value : undefined;
}

function pointCount(node) {
    const p = node._scPreview;
    if (!p || !p.ok) return 9;
    const s = p.sigmas || p.sigmas_high || [];
    return s.length || 9;
}

async function fetchPreview(node, def) {
    const req = { node: def.kind };
    for (const name of def.widgets) req[name] = valueOf(node, name);
    const body = JSON.stringify(req);
    if (node._scLastBody === body) return;         // nothing changed
    node._scLastBody = body;
    for (const url of [`/api${ROUTE}`, ROUTE]) {
        try {
            const r = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body,
            });
            if (!r.ok) continue;
            node._scPreview = await r.json();
            node._scAsked = req;
            refit(node);                            // height follows the knots
            node.setDirtyCanvas(true, true);
            redrawVue(node);
            return;
        } catch (e) { /* try the next form */ }
    }
    node._scPreview = null;                         // unreachable: draw nothing
    node._scLastBody = null;                        // ask again on the next change
    node.setDirtyCanvas(true, true);
    redrawVue(node);
}

// v984: under Nodes 2.0 a canvas widget is painted into its OWN small canvas,
// once, and again only when the frontend's triggerDraw() is called (it hooks
// the widget's callback and a resize). A preview that lands later -- the
// normal case, it is a fetch -- would otherwise stay "no preview" forever.
// Measured in the v984 browser probe. The classic canvas redraws by itself.
function redrawVue(node) {
    const w = node._scPlot;
    if (w && typeof w.triggerDraw === "function") {
        // Its draw() takes the height from `computedHeight` FIRST (LiteGraph's
        // last layout pass) and only then asks computeSize -- so a plot that
        // grew with its knots stayed clipped at the old height (probe: 124 px
        // for a 41-point curve that wants 260). Hand it the new height.
        w.computedHeight = plotHeight(node);
        try { w.triggerDraw(); } catch (e) { /* a stale binding: ignore */ }
    }
}

function schedulePreview(node, def) {
    clearTimeout(node._scTimer);
    node._scTimer = setTimeout(() => fetchPreview(node, def), 120);
}

function fmt(v) {
    const a = Math.abs(v);
    return a >= 10 ? v.toFixed(2) : v.toFixed(3);
}

// The answer -> what ph_sigma_plot draws. Exported for the guard.
export function specFor(p, asked) {
    if (!p) return { state: null };
    if (!p.ok) return { state: { error: p.error || "invalid" } };
    const a = asked || {};
    const warn = [];
    if (p.swapped) warn.push("sigma_min >= sigma_max -- the node swaps them");
    if (p.node === "dual") {
        const hi = p.sigmas_high || [], lo = p.sigmas_low || [];
        const k = p.split_step, n = hi.length - 1;
        if (a.split_step !== undefined && Number(a.split_step) !== k) {
            warn.push(`split_step ${a.split_step} -> ${k} (must be 1..${n - 1})`);
        }
        return {
            header: `${n} steps  ·  HIGH ${a.schedule_high ?? ""} 0..${k}` +
                    `  ·  LOW ${a.schedule_low ?? ""} ${k}..${n}`,
            warn: warn.join("  ·  "),
            series: [
                { sigmas: lo, color: LOW_COLOR, from: k, to: lo.length - 1, label: "LOW" },
                { sigmas: hi, color: HIGH_COLOR, from: 0, to: k, label: "HIGH" },
            ],
            marker: { index: k, text: `split ${k}  ·  sigma ${fmt(hi[k] ?? 0)}` },
        };
    }
    const s = p.sigmas || [];
    return {
        header: s.length >= 2
            ? `${s.length - 1} steps  ·  ${a.sigma_schedule ?? ""}  ·  ` +
              `sigma ${fmt(s[0])} -> ${fmt(s[s.length - 2])}`
            : "",
        warn: warn.join("  ·  "),
        series: [{ sigmas: s, color: ONE_COLOR }],
    };
}

function plotHeight(node) {
    return plotHeightFor(pointCount(node));
}

function ensurePlotWidget(node) {
    if (node._scPlot) return node._scPlot;
    const wdg = {
        type: "poly_sigma_curve",
        name: "curve",
        value: null,
        serialize: false,               // NEVER in widgets_values (guard #577)
        options: { serialize: false },
        draw(ctx, n, widgetWidth, y) {
            drawSigmaPlot(ctx, widgetWidth, y, plotHeight(n), specFor(n._scPreview, n._scAsked));
        },
        computeSize(width) {
            return [width, plotHeight(node)];
        },
    };
    node._scPlot = wdg;
    (node.widgets = node.widgets || []).push(wdg);
    return wdg;
}

app.registerExtension({
    name: "Polyhedron.SigmaCurves",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const def = NODES[nodeData?.name];
        if (!def) return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            const node = this;
            for (const name of def.widgets) {
                const ww = (node.widgets || []).find((x) => x.name === name);
                if (!ww) continue;
                const prev = ww.callback;
                ww.callback = function () {
                    const out = prev?.apply(this, arguments);
                    schedulePreview(node, def);
                    return out;
                };
            }
            ensurePlotWidget(node);
            schedulePreview(node, def);
            return r;
        };

        // A loaded workflow restores widget values AFTER creation.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            schedulePreview(this, def);
            return r;
        };
    },
});
