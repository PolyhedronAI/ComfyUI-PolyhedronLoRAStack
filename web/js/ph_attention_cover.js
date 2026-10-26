/**
 * ph_attention_cover.js -- v1224: how much of the attention work the chosen
 * kernel really carried, shown ON the node, not only in the console.
 *
 * Frank, 25.10.2026: a joint refine ran 2.5x as long as it had to because every
 * self-attention call fell back from sageattention to the model's own backend.
 * The console said it in one quiet line. The Attention node now books every
 * call (nodes/ph_attention.py, CoverageLedger) and a stage that ran the model
 * (Sampler, Power Upscale refine) sends one event when it is done:
 *
 *   pls.attention_cover  {nodes: [ids], level: ok|note|warn, text, tip, stage}
 *
 * Where it lands:
 *   * Polyhedron Attention, classic canvas: one painted row under the widgets,
 *     zero height until the first report (a node with nothing to say does not
 *     grow), amber on WARN.
 *   * Polyhedron Attention, Nodes 2.0: a chip in the title bar (the v962 header
 *     slot of uls_vue_views) -- the painted row stays at zero height there.
 *   * Power Upscale: a line in its process pane header, next to the HUD.
 * Everything here is a readout: no widget is serialised, nothing is sent back.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { registerVueView, vueMode } from "./uls_vue_views.js";

/* Grow-only: the row may need 20 px, but a node Frank made taller must not shrink
 * (ph_widget_vis.refit fits the height exactly -- right for a row that hides,
 * wrong for a readout that appears in a node with room to spare). Width untouched. */
function growToFit(node) {
    if (!node || !node.setSize || !node.computeSize || !node.size) return;
    try {
        const need = node.computeSize()[1];
        if (need > node.size[1]) node.setSize([node.size[0], need]);
        node.setDirtyCanvas?.(true, true);
    } catch (e) { /* a readout never breaks the canvas */ }
}

const EVENT = "pls.attention_cover";
const FILE_VERSION = "v1227";
// v1227: the file says which version the browser really holds -- Frank 25.10. saw the coverage line in the pane
// but no toast: the v1224 file (no toast) had survived in the browser cache. One line in F12 settles that.
console.info("[PLS] ph_attention_cover.js " + FILE_VERSION + " loaded (attention coverage: node line, pane line, toast)");
const ATTN = "ULSAttention";
const PU = "ULSPowerUpscale";
const ROW_NAME = "$pls_attn_cover";
const ROW_H = 20;
const COLOURS = { ok: "#8fbf8f", note: "#c9c9a0", warn: "#f0b45a" };

/** Pure: the text a node shows for a report (exported for the guard). */
export function coverLine(rep) {
    if (!rep || !rep.text) return "";
    const head = rep.level === "warn" ? "\u26a0 " : "";
    const stage = rep.stage ? " (" + rep.stage + ")" : "";
    return head + rep.text + stage + (rep.tip ? " \u2014 " + rep.tip : "");
}

export function colourOf(level) {
    return COLOURS[level] || COLOURS.note;
}

function makeRow() {
    const w = {
        type: "pls_attn_cover",
        name: ROW_NAME,
        value: null,
        serialize: false,
        options: { serialize: false },   // keep it out of the PROMPT too (the frontend reads options.serialize)
        computeSize(width) {
            const n = w._node;
            return [width, (n && n._plsCover && !vueMode()) ? ROW_H : 0];
        },
        draw(ctx, node, width, y) {
            w._node = node;
            const rep = node._plsCover;
            if (!rep || vueMode()) return;
            ctx.save();
            try {
                ctx.font = "11px sans-serif";
                ctx.textBaseline = "middle";
                ctx.fillStyle = colourOf(rep.level);
                let txt = coverLine(rep);
                const max = width - 28;
                while (txt.length > 4 && ctx.measureText(txt).width > max) txt = txt.slice(0, -2);
                if (txt !== coverLine(rep)) txt = txt.slice(0, -1) + "\u2026";
                ctx.fillText(txt, 14, y + ROW_H / 2);
            } catch (e) {
                /* a readout must never break the canvas */
            }
            ctx.restore();
        },
    };
    return w;
}

function ensureRow(node) {
    if (typeof node.addCustomWidget !== "function") return;
    if ((node.widgets || []).some((x) => x && x.name === ROW_NAME)) return;
    const w = node.addCustomWidget(makeRow());
    if (w) w._node = node;
}

function paintChip(node) {
    const chip = node._plsCoverChip;
    if (!chip) return;
    const rep = node._plsCover;
    chip.textContent = rep ? (rep.level === "warn" ? "\u26a0 " : "") + rep.text : "";
    chip.title = rep ? coverLine(rep) : "";
    chip.style.cssText = "font:11px sans-serif;margin-left:8px;white-space:nowrap;"
        + "color:" + (rep ? colourOf(rep.level) : "inherit") + ";";
}

function paintPU(node) {
    const hud = node._procHud;
    const rep = node._plsCover;
    if (!hud || !hud.parentElement || !rep) return;
    let span = node._plsCoverSpan;
    if (!span || span.parentElement !== hud.parentElement) {
        span = document.createElement("span");
        span.style.cssText = "margin-left:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
            + "min-width:0;flex:0 1 auto;";
        hud.parentElement.insertBefore(span, hud);
        node._plsCoverSpan = span;
    }
    span.textContent = (rep.level === "warn" ? "\u26a0 " : "") + rep.text;
    span.title = coverLine(rep);
    span.style.color = colourOf(rep.level);
}

/** v1225: the toast for a report that cost real time (the server decides: detail.toast). Once per report. */
export function toastOf(detail) {
    if (!detail || !detail.toast) return null;
    const stage = detail.stage ? " -- " + detail.stage : "";
    return { severity: "warn", summary: "Attention fell back" + stage,
             detail: String(detail.text || "") + (detail.tip ? "\n" + detail.tip : "")
                     + "\n(console: the coverage report says why and what helps)",
             life: 15000 };
}

function showToast(detail) {
    const t = toastOf(detail);
    if (!t) return false;
    try {
        const tm = app.extensionManager && app.extensionManager.toast;
        if (tm && typeof tm.add === "function") {
            tm.add(t);
            console.info("[PLS] attention toast shown: " + t.summary);
            return true;
        }
        if (tm && typeof tm.addAlert === "function") {      // the house fallback (uls_token_toast.js)
            tm.addAlert(t.summary + " -- " + t.detail);
            console.info("[PLS] attention toast shown (alert): " + t.summary);
            return true;
        }
    } catch (e) { /* the console trace below */ }
    console.warn("[PLS] attention toast could NOT be shown (no toast API here): " + t.summary + " -- " + t.detail);
    return false;
}

/** Apply one report to the graph (exported for the guard). */
export function takeCover(detail) {
    if (!detail || !Array.isArray(detail.nodes)) return 0;
    const graph = app.graph;
    if (!graph || typeof graph.getNodeById !== "function") return 0;
    let n = 0;
    const seen = new Set();
    for (const id of detail.nodes) {
        if (seen.has(String(id))) continue;
        seen.add(String(id));
        const node = graph.getNodeById(Number(id));
        if (!node) continue;
        const cls = node.comfyClass || node.type;
        node._plsCover = { level: detail.level, text: detail.text, tip: detail.tip || "",
                           stage: detail.stage || "" };
        if (cls === ATTN) {
            ensureRow(node);
            paintChip(node);
            growToFit(node);
        } else if (cls === PU) {
            paintPU(node);
        }
        node.setDirtyCanvas?.(true, true);
        n += 1;
    }
    return n;
}

registerVueView(ATTN, {
    label: "Attention coverage",
    header(node, chip) {
        node._plsCoverChip = chip;
        paintChip(node);
    },
});

app.registerExtension({
    name: "polyhedron.attention.cover",
    setup() {
        api.addEventListener(EVENT, (ev) => {
            try { takeCover(ev && ev.detail); } catch (e) { /* diagnostic only */ }
            try { showToast(ev && ev.detail); } catch (e) { /* diagnostic only */ }
        });
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== ATTN) return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated ? onCreated.apply(this, arguments) : undefined;
            ensureRow(this);
            return r;
        };
    },
});
