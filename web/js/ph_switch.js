/*
 * ph_switch.js -- v537
 *
 * Frontend for ⬡ Polyhedron Switch (ULSAnySwitch) and
 * ⬡ Polyhedron Switch Inverse (ULSAnySwitchInv).
 *
 * Responsibilities:
 *   1. Dynamic slots with exactly ONE trailing spare: connect the spare and a
 *      new one appears; disconnect trailing slots and they collapse back.
 *      Holes in the middle are kept (their links must not shift).
 *   2. `select` auto-max: options.max always equals the number of CONNECTED
 *      slots (min 1); the value is clamped when slots are disconnected.
 *   3. Per-run status line from the backend ui channel ("pls_switch"),
 *      drawn at the bottom of the node (blocked / fallback / chosen slot).
 *
 * Slot-name prefixes MUST stay identical to nodes/ph_switch.py (_IN/_OUT);
 * tests/test_v537_switch.py guards the parity.
 */

import { app } from "../../scripts/app.js";

console.info("[PLS] ph_switch.js v874 loaded");

const IN = "input_";
const OUT = "out_";
const ANY = "*";
const FWD = "ULSAnySwitch";
const INV = "ULSAnySwitchInv";
const STATUS_H = 16;                    // room reserved for the status line

// v874: the widest label/value pair this node draws is "on_missing" against
// "use next active", plus the toggle circle and LiteGraph's two margins. Below
// this the two texts draw ON TOP OF EACH OTHER.
const MIN_W = 260;

function fitSize(node) {
    /* NEVER SHRINK THE WIDTH.
     *
     * tidy() used to end with node.setSize(node.computeSize()), which hard-
     * resets the node to LiteGraph's computed MINIMUM. That runs on every
     * connection change AND on every workflow load (onConfigure -> double-rAF
     * -> tidy), so two things went wrong: a width the user had dragged out was
     * thrown away on every restart, and the minimum itself is narrower than the
     * on_missing pair needs -- which is the overlap Frank kept seeing come back.
     *
     * Height still FOLLOWS computeSize: collapsing a spare slot must shrink the
     * box, and making room for it is what tidy() is for. Only the width is
     * grow-only, because only the width carries text that can collide. */
    const want = node.computeSize();
    const cur = (node.size && node.size[0]) || 0;
    node.setSize([Math.max(MIN_W, want[0], cur), want[1]]);
}

const LG_INPUT = (typeof LiteGraph !== "undefined" && LiteGraph.INPUT) || 1;
const LG_OUTPUT = (typeof LiteGraph !== "undefined" && LiteGraph.OUTPUT) || 2;

function selWidget(node) {
    return node.widgets ? node.widgets.find((w) => w.name === "select") : null;
}

function updateSelectMax(node, connectedCount) {
    const w = selWidget(node);
    if (!w) return;
    w.options = w.options || {};
    w.options.max = Math.max(1, connectedCount);
    if (w.value > w.options.max) w.value = w.options.max;
    if (w.value < 1) w.value = 1;
}

/* ------------------------------ forward: inputs ------------------------- */

function dynInputs(node) {
    return (node.inputs || []).filter((s) => s.name && s.name.startsWith(IN));
}

function tidyInputs(node) {
    if (!node.inputs) node.inputs = [];
    // collapse trailing spares down to one
    let dyn = dynInputs(node);
    while (dyn.length >= 2 &&
           dyn[dyn.length - 1].link == null &&
           dyn[dyn.length - 2].link == null) {
        node.removeInput(node.inputs.indexOf(dyn[dyn.length - 1]));
        dyn = dynInputs(node);
    }
    // ensure exactly one trailing spare
    dyn = dynInputs(node);
    if (dyn.length === 0 || dyn[dyn.length - 1].link != null) {
        node.addInput(IN + (dyn.length + 1), ANY);
    }
    // keep names dense (only trailing slots are ever removed, so this is a
    // no-op in normal flow and a repair after odd configure states)
    let i = 1;
    for (const s of node.inputs) if (s.name && s.name.startsWith(IN)) s.name = IN + (i++);

    updateSelectMax(node, dynInputs(node).filter((s) => s.link != null).length);
    fitSize(node);
    node.setDirtyCanvas(true, true);
}

/* ------------------------------ inverse: outputs ------------------------ */

function dynOutputs(node) {
    return (node.outputs || []).filter((s) => s.name && s.name.startsWith(OUT));
}

function outUsed(slot) {
    return slot.links != null && slot.links.length > 0;
}

function tidyOutputs(node) {
    if (!node.outputs) node.outputs = [];
    let dyn = dynOutputs(node);
    while (dyn.length >= 2 &&
           !outUsed(dyn[dyn.length - 1]) &&
           !outUsed(dyn[dyn.length - 2])) {
        node.removeOutput(node.outputs.indexOf(dyn[dyn.length - 1]));
        dyn = dynOutputs(node);
    }
    dyn = dynOutputs(node);
    if (dyn.length === 0 || outUsed(dyn[dyn.length - 1])) {
        node.addOutput(OUT + (dyn.length + 1), ANY);
    }
    let i = 1;
    for (const s of node.outputs) if (s.name && s.name.startsWith(OUT)) s.name = OUT + (i++);

    updateSelectMax(node, dynOutputs(node).filter(outUsed).length);
    fitSize(node);
    node.setDirtyCanvas(true, true);
}

/* ------------------------------ registration ---------------------------- */

app.registerExtension({
    name: "polyhedron.switch",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== FWD && nodeData.name !== INV) return;
        const isFwd = nodeData.name === FWD;
        const tidy = isFwd ? tidyInputs : tidyOutputs;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            // reserve room for the status line, once per node instance
            const baseCS = this.computeSize.bind(this);
            this.computeSize = (out) => {
                const s = baseCS(out);
                s[1] += STATUS_H;
                return s;
            };
            tidy(this);
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type) {
            onConnectionsChange?.apply(this, arguments);
            if ((isFwd && type === LG_INPUT) || (!isFwd && type === LG_OUTPUT)) {
                tidy(this);
            }
        };

        // After a workflow load, links restore asynchronously -- normalize on
        // the double-rAF (the proven v531 onConfigure pattern).
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            requestAnimationFrame(() => requestAnimationFrame(() => tidy(this)));
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const t = message ? message.pls_switch : null;
            this._plsStatus = Array.isArray(t) ? t[0] : t || this._plsStatus;
            this.setDirtyCanvas(true, true);
        };

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            onDrawForeground?.apply(this, arguments);
            if ((this.flags && this.flags.collapsed) || !this._plsStatus) return;
            ctx.save();
            ctx.font = "11px Arial";
            ctx.fillStyle = "#9a9a9a";
            ctx.textAlign = "left";
            ctx.fillText(this._plsStatus, 8, this.size[1] - 5);
            ctx.restore();
        };
    },
});
