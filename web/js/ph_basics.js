/*
 * ph_basics.js -- v718
 *
 * Frontend for ⬡ Polyhedron Load CLIP (ULSLoadCLIP), ⬡ Polyhedron Load VAE
 * (ULSLoadVAE) and ⬡ Polyhedron Load Model (ULSLoadModel).
 *
 * Two jobs:
 *
 *   1. Status line from the backend ui channel "pls_basics", drawn at the node
 *      bottom -- file, resolved type / channel family, dtype. One canvas line,
 *      no DOM, no timers (house pattern, see ph_seed.js / ph_switch.js).
 *
 *   2. PROGRESSIVE SLOTS (v716): a slot only appears once the one before it is
 *      filled, so a node with six model slots is not six empty rows wide.
 *
 * WHY THE SWITCH MECHANIC IS NOT COPYABLE HERE
 * --------------------------------------------
 * ph_switch.js grows and shrinks real PINS with addInput/removeInput. These two
 * nodes use WIDGETS, and widgets serialise POSITIONALLY into widgets_values.
 * Removing a widget at runtime shifts every value behind it on the next load --
 * saved workflows would break silently, which is the v585 append-only law.
 *
 * So we hide instead of remove, with the mechanism ph_clip_encode.js has run
 * since v613: swap the widget's TYPE to a marker prefix (LiteGraph skips a
 * widget in the LAYOUT pass by its type) and give it a zero computeSize. The
 * widget stays in node.widgets throughout, keeps its position and keeps
 * serialising -- it is only invisible.
 *
 * Two traps this pattern already paid for, both preserved here:
 *   * on show, NEVER write `computeSize = undefined` back -- delete the own
 *     property instead, or the widget stays zero-height forever (the v556 bug).
 *   * hide/show must be idempotent, or repeated calls stack the type prefix.
 *
 * The placeholder strings MUST stay identical to nodes/ph_basics.py
 * (_MODEL_PLACEHOLDER / _CLIP_PLACEHOLDER) -- "filled" means the same thing on
 * both sides or the two disagree about which slots exist.
 * tests/test_v716_progressive.py guards that parity.
 */

import { app } from "../../scripts/app.js";
import { refit } from "./ph_widget_vis.js";

console.info("[PLS] ph_basics.js v542 loaded");

const NODES = ["ULSLoadCLIP", "ULSLoadVAE", "ULSLoadModel"];
const STATUS_H = 16;

/* Parity with nodes/ph_basics.py -- em dashes included. */
const MODEL_PLACEHOLDER = "\u2014 select model \u2014";
const CLIP_PLACEHOLDER = "\u2014 none \u2014";

/* Slot layout per node: widget name pattern and how many there are. */
/*
 * v718 -- CANON vs DISPLAY for Load CLIP.
 *
 * Frank: "the type combo has to go either all the way up or all the way down --
 * up, because too much happens down there." Correct: `type` appears and
 * disappears with the encoder count, and doing that in the MIDDLE of the slot
 * block makes the rows below it jump.
 *
 * This is the most dangerous edit class in the tree and it has drawn blood
 * before (v584/v603: a widget moved into the display middle and every saved
 * graph loaded shifted by one -- Frank's prompt ended up inside `separator`).
 * The law that came out of it, from ph_clip_encode.js:
 *
 *     PYTHON = CANON. Never re-ordered. What is on disk, forever.
 *     JS     = DISPLAY. Permuted freely.
 *     load   : the rows are still in CANON while widgets_values is poured in,
 *              so nothing needs mapping -- permute AFTERWARDS.
 *     save   : serialize() swings the rows back to CANON for the length of one
 *              call. serialize(), NOT onSerialize -- this frontend has been
 *              observed declining to invoke a base callback, and there is no
 *              saving a graph without serialize() itself.
 *
 * The file on disk never learns that the display moved.
 *
 * ONE STEP FURTHER THAN ASKED, and easy to undo: `device` moves up as well.
 * Lifting only `type` would have left `device` sitting between clip_name and
 * clip_name_2, still splitting the slots into two groups. With both up, the
 * slots are one uninterrupted block above the remove button -- the same shape
 * ph_clip_encode.js uses (filters on top, fields below). Revert = drop "device"
 * from CLIP_DISPLAY's head and put it back after "clip_name".
 */
const CLIP_CANON = ["clip_name", "type", "device",
                    "clip_name_2", "clip_name_3", "clip_name_4"];
const CLIP_DISPLAY = ["type", "device",
                      "clip_name", "clip_name_2", "clip_name_3", "clip_name_4"];

/*
 * Permute the widget ROW ORDER. Values are untouched -- only the order they are
 * drawn in changes. Anything not in the canon (the remove button) rides at the
 * BACK, which v717 measured to be load-bearing: serialise writes by index and
 * configure reads with a running counter, so a non-canon widget anywhere but
 * the end shifts values.
 */
function _reorder(node, order) {
    if (!node.widgets) return false;
    const by = new Map(node.widgets.map((w) => [w.name, w]));
    const rows = order.map((n) => by.get(n)).filter(Boolean);
    if (rows.length !== order.length) return false;   // a widget missing: do not guess
    const extras = node.widgets.filter((w) => !order.includes(w.name));
    node.widgets = rows.concat(extras);
    return true;
}

function _toDisplay(node, spec) {
    if (!spec.display || node._plsDisplayed) return;
    if (_reorder(node, spec.display)) node._plsDisplayed = true;
}

function _toCanon(node, spec) {
    if (!spec.canon || !node._plsDisplayed) return;
    if (_reorder(node, spec.canon)) node._plsDisplayed = false;
}

/*
 * THE SAVE PATH, as a named function so a guard can LIFT AND DRIVE it rather
 * than reimplement it. A test that rebuilds this dance in its own harness
 * answers a question about the harness -- the first version of
 * test_v718_display_order.py did exactly that and let the v603 wound sail
 * straight through.
 *
 * The rows swing back to canon for the length of one call, so widgets_values
 * lands on disk in canon order no matter what the screen is showing.
 */
function _serializeInCanon(node, spec, base, args) {
    const wasDisplayed = node._plsDisplayed;
    _toCanon(node, spec);
    let out;
    try {
        out = base ? base.apply(node, args) : undefined;
    } finally {
        // in a finally, so a throw inside the base method cannot leave the node
        // parked in canon order on screen
        if (wasDisplayed) _toDisplay(node, spec);
    }
    return out;
}

const SLOTS = {
    ULSLoadModel: { names: ["model_1", "model_2", "model_3", "model_4",
                            "model_5", "model_6"],
                    placeholder: MODEL_PLACEHOLDER,
                    hasSelect: true,
                    /*
                     * How many slots the remove button refuses to go below.
                     *
                     * Load Model: 1 since v719. It was 0, on the reasoning that
                     * slot 1 HAS a placeholder in its own list, so the button
                     * should not be less capable than the dropdown. Frank found
                     * the cost on screen: with only slot 1 filled -- the normal
                     * state of a one-model node -- the button sits there
                     * offering to empty the node. It is a control that is
                     * always present and almost never wanted.
                     *
                     * Nothing is lost. Emptying slot 1 stays expressible, it
                     * just goes back through the dropdown that owns the
                     * placeholder. The button now means one thing only: get rid
                     * of the EXTRA slot -- and it appears exactly when there is
                     * an extra slot to get rid of. Same rule as Load CLIP now,
                     * for a different reason (there slot 1 cannot be emptied at
                     * all, here it can, just not from this button).
                     */
                    floor: 1 },
    ULSLoadCLIP: { names: ["clip_name", "clip_name_2", "clip_name_3",
                           "clip_name_4"],
                   placeholder: CLIP_PLACEHOLDER,
                   hasSelect: false,
                   /*
                    * Load CLIP: 1. Not a special rule invented here --
                    * `clip_name` simply has no placeholder entry in its list, so
                    * emptying it is not expressible in the first place. A Load
                    * CLIP node with no encoder has nothing to do.
                    */
                   floor: 1,
                   canon: CLIP_CANON,
                   display: CLIP_DISPLAY,
                   /*
                    * MEASURED in comfy/sd.py (2026-07-23): with three or four
                    * encoders load_text_encoder_state_dicts never consults
                    * clip_type -- three always builds sd3, four always hidream.
                    * Core's own Triple and Quadruple loaders have no `type`
                    * widget for exactly that reason.
                    *
                    * So from three filled slots the field is HIDDEN rather than
                    * shown-but-inert. A control that is visible and does nothing
                    * is worse than one that is absent: it invites the user to
                    * set it and then quietly ignores them. It keeps its value
                    * and its canon position throughout and comes back the
                    * moment the count drops to two, and the backend readout
                    * still states that the count decided.
                    */
                   typeGateFrom: 3 },
};

/*
 * v717 -- the remove button.
 *
 * WHY A BUTTON WIDGET AND NOT A PAINTED FOOTER BUTTON
 * ---------------------------------------------------
 * A painted button needs a hit area that agrees with the drawn rectangle across
 * zoom, node width and every resize -- two numbers that must match. That is the
 * exact shape of the wound the mask editor bled from between v707 and v714, and
 * the lesson from it is standing: never two layout layers that have to be
 * brought into agreement by arithmetic. LiteGraph draws and hit-tests a button
 * widget itself, so there is no second layer to keep in sync.
 *
 * WHY IT IS SAFE AGAINST THE CANON -- MEASURED, not assumed
 * ---------------------------------------------------------
 * Read out of comfyui-frontend-package 1.47.10 (2026-07-23), the two directions
 * do NOT mirror each other:
 *
 *   serialise: widgets_values[INDEX_IN_node.widgets] = value, skipping
 *              serialize===false (so a skipped widget leaves a HOLE)
 *   configure: a RUNNING counter over the widgets with serialize !== false,
 *              which does NOT advance for skipped ones
 *
 * A serialize:false widget in the MIDDLE therefore shifts everything behind it
 * on load. That is the measured reason behind the house rule "anything outside
 * the canon rides at the BACK" -- previously held on experience alone.
 *
 * Two belts, either of which is sufficient, both worn:
 *   1. the button is the LAST entry in node.widgets, so nothing sits behind it
 *      to be shifted;
 *   2. serialize:false, so it never occupies a widgets_values slot at all.
 * Together these also make it irrelevant whether configure() runs before or
 * after the button is appended.
 */
function lastFilledSlot(values, placeholder) {
    let last = 0;
    for (let i = 0; i < values.length; i++) {
        if (_filled(values[i], placeholder)) last = i + 1;
    }
    return last;                       // 1-based; 0 means nothing is filled
}

/*
 * Clear the LAST FILLED slot. Deliberately not "slot N" with per-row buttons:
 * removing from the middle would mean either leaving a hole or shifting values
 * between canon positions, and shifting is the thing that silently renumbers
 * saved workflows. Clearing from the back needs no shifting at all -- the
 * existing visibility rule collapses the row on its own.
 *
 * Side effect worth having: if a hole was made through the dropdown (1 and 3
 * filled, 2 empty), this tidies up from the back until the hole is gone.
 */
function removeLastSlot(node, spec) {
    const widgets = spec.names.map((n) => _w(node, n));
    const values = widgets.map((w) => (w ? w.value : null));
    const last = lastFilledSlot(values, spec.placeholder);
    if (last <= (spec.floor || 0)) return;      // nothing this button may clear
    const w = widgets[last - 1];
    if (!w) return;
    w.value = spec.placeholder;
    applySlots(node, spec);
}

const HIDDEN_PREFIX = "pls-hidden-";

function _w(node, name) {
    return (node.widgets || []).find((w) => w.name === name);
}

/* Hide a widget without dropping it from the canon. See header note. */
function _hide(w, hidden) {
    if (!w) return;
    const isHidden = String(w.type || "").startsWith(HIDDEN_PREFIX);
    if (hidden === isHidden) return;          // idempotent
    if (hidden) {
        w._pls_type = w.type;
        w._pls_hadCS = Object.prototype.hasOwnProperty.call(w, "computeSize");
        w._pls_cs = w.computeSize;
        w.type = HIDDEN_PREFIX + w.type;
        w.computeSize = () => [0, -4];
        w.hidden = true;
        if (w.element) { w.element.style.display = "none"; w.element.hidden = true; }
    } else {
        w.type = w._pls_type !== undefined
            ? w._pls_type : String(w.type).slice(HIDDEN_PREFIX.length);
        if (w._pls_hadCS) w.computeSize = w._pls_cs;
        else delete w.computeSize;            // v556 bug: never assign undefined
        w.hidden = false;
        if (w.element) { w.element.style.display = ""; w.element.hidden = false; }
    }
}

function _filled(value, placeholder) {
    if (value === null || value === undefined) return false;
    const text = String(value).trim();
    return text !== "" && text !== placeholder;
}

/*
 * The rule, one place only: slot 1 is always visible, and everything up to ONE
 * PAST the last filled slot is visible. That is literally the switch node's
 * "exactly one spare at the end", applied to visibility instead of existence.
 *
 * A hole in the middle stays visible -- the last filled index is what counts,
 * not the count of filled slots. Refusing to show a hole would hide a slot the
 * user can see is in use.
 */
function visibleCount(values, placeholder) {
    let last = 0;
    for (let i = 0; i < values.length; i++) {
        if (_filled(values[i], placeholder)) last = i + 1;
    }
    return Math.max(1, Math.min(values.length, last + 1));
}

/*
 * `select` on Load Model: the upper bound follows the LAST FILLED slot, so the
 * selector cannot point past the end of what exists.
 *
 * HONEST LIMIT, stated rather than papered over: with a hole (slots 1 and 3
 * filled, 2 empty) select can still be pointed at the empty 2. The backend
 * names that case precisely, so it is a clear message rather than a wrong load.
 * Silently jumping the value to the next filled slot would be a hidden edit of
 * the user's choice, which is worse.
 */
function updateSelectMax(node, values, placeholder) {
    const w = _w(node, "select");
    if (!w) return;
    let last = 0;
    for (let i = 0; i < values.length; i++) {
        if (_filled(values[i], placeholder)) last = i + 1;
    }
    w.options = w.options || {};
    w.options.max = Math.max(1, last);
    if (w.value > w.options.max) w.value = w.options.max;
    if (w.value < 1) w.value = 1;
}

function applySlots(node, spec) {
    if (!spec || !node.widgets) return;
    const widgets = spec.names.map((n) => _w(node, n));
    const values = widgets.map((w) => (w ? w.value : null));
    const show = visibleCount(values, spec.placeholder);

    for (let i = 0; i < widgets.length; i++) _hide(widgets[i], i >= show);

    if (spec.hasSelect) updateSelectMax(node, values, spec.placeholder);

    if (spec.typeGateFrom) {
        const filled = values.filter((v) => _filled(v, spec.placeholder)).length;
        _hide(_w(node, "type"), filled >= spec.typeGateFrom);
    }

    // The button names the slot it would clear, so the "from the bottom up"
    // rule is visible rather than something the user has to be told. When
    // there is nothing it may clear, it goes away instead of sitting there
    // disabled -- same reasoning as the `type` gate.
    const btn = _w(node, REMOVE_BTN);
    if (btn) {
        const canRemove = lastFilledSlot(values, spec.placeholder)
                          > (spec.floor || 0);
        btn.label = REMOVE_GLYPH;      // never a slot number -- see addRemoveButton
        _hide(btn, !canRemove);
    }

    // Height only. setSize(computeSize()) is a full reset that also throws the
    // WIDTH away -- the lesson from the v637 session; a relayout must never
    // resize the node the user shaped.
    node.setSize([node.size[0], node.computeSize()[1]]);
    node.setDirtyCanvas(true, true);
}

/* Re-run the layout whenever a slot value changes. */
function wireSlots(node, spec) {
    for (const name of spec.names) {
        const w = _w(node, name);
        if (!w || w._plsWired) continue;
        w._plsWired = true;
        const original = w.callback;
        w.callback = function () {
            const r = original ? original.apply(this, arguments) : undefined;
            applySlots(node, spec);
            return r;
        };
    }
}

/*
 * Append the remove button ONCE, and always last. See the REMOVE_BTN note above
 * for why "last" and "serialize:false" are both load-bearing.
 *
 * v718 -- the label is a bare ✕ and names no slot number. It used to read
 * "remove slot N", and Frank found the flaw on screen: with slot 1 filled and
 * slot 2 empty, it said "remove slot 1" while what visibly disappeared was ROW
 * 2. Neither number is wrong -- the button clears the last FILLED slot, and the
 * row that vanishes is the trailing empty one. "Slot" simply means two
 * different things in those two sentences, so no number can be right for both.
 * A symbol makes no claim it cannot keep.
 *
 * The colour comes from the suite's amber (#ff8c00, the same tone the CTE
 * footer uses). Painting it is safe here in a way a painted FOOTER button would
 * not be: LiteGraph passes y and height into widget.draw and sets last_y for
 * its own hit-testing beforehand (measured in comfyui-frontend-package 1.47.10:
 * `typeof s.draw === "function" ? s.draw(ctx, this, w, y, h, lq) : ...`). We
 * paint INSIDE the row we were handed and never compute a hit area, so there is
 * no second layout layer to keep in agreement -- the v707-v714 lesson holds.
 */
const REMOVE_BTN = "pls_remove_slot";
const AMBER = "#ff8c00";
const REMOVE_GLYPH = "\u2715";

/*
 * v828 -- the combo entries carry sizes now ("name \u00b7 1.2 GB"), and the
 * decoration is part of the VALUE (the cutout's v752 trade, same reasons).
 * A graph saved before this cut holds the bare filename, which the list no
 * longer offers -- the widget would show a value with no matching entry.
 * This normalises a stored value onto the entry the list offers RIGHT NOW,
 * by comparing both sides stripped. The strip mirrors the backend's
 * _SIZE_SUFFIX exactly: one trailing " \u00b7 <n> <unit>", nothing broader,
 * because filenames may legitimately contain dashes and dots.
 */
const SIZE_SUFFIX = /\s\u00b7\s[\d.,]+\s?(KB|MB|GB|TB)$/;

function stripSize(v) {
    return String(v == null ? "" : v).replace(SIZE_SUFFIX, "").trimEnd();
}

function normalizeSized(node) {
    for (const w of node.widgets || []) {
        const opts = w && w.options && w.options.values;
        if (!Array.isArray(opts) || !opts.length) continue;
        if (opts.includes(w.value)) continue;
        const bare = stripSize(w.value);
        const hit = opts.find((o) => stripSize(o) === bare);
        if (hit !== undefined) w.value = hit;
    }
}

/*
 * v829 -- the no-model warning as a BUBBLE, top right (Frank's ask: the
 * official toast form, "an der Node selbst geht es natuerlich auch" -- so
 * the amber status line STAYS and this is the second voice). The form is
 * the house pattern lifted from ph_cutout.js's download toast (fixed,
 * top/right, own class), reduced to an info bubble: no buttons, it
 * dismisses itself. ONE bubble per node instance and disconnect -- the
 * flag resets when a model is connected, so re-unplugging warns again,
 * but a redraw never spams.
 */
const NO_MODEL_TEXT =
    "no model connected -- the encoder set is checked from file headers "
    + "only; the match AGAINST THE MODEL cannot run. Connect 'model' to "
    + "verify.";

let _toastCSS = false;

function _ensureToastCSS() {
    if (_toastCSS) return;
    _toastCSS = true;
    const st = document.createElement("style");
    st.textContent =
        ".ph-basics-toast { position:fixed; top:16px; right:16px;"
        + " z-index:10000; max-width:340px; background:#1d1d1d;"
        + " border:1px solid " + AMBER + "; border-left:4px solid " + AMBER
        + "; color:#ddd; font:12px sans-serif; padding:8px 10px;"
        + " border-radius:4px; box-shadow:0 4px 14px rgba(0,0,0,.5); }"
        + ".ph-basics-toast h4 { margin:0 0 4px; font-size:12px;"
        + " color:" + AMBER + "; }"
        + ".ph-basics-toast p { margin:0; color:#bbb; line-height:1.35; }";
    document.head.appendChild(st);
}

function toastNoModel(node) {
    if (node._plsNoModelToasted) return;
    node._plsNoModelToasted = true;
    try {
        _ensureToastCSS();
        const box = document.createElement("div");
        box.className = "ph-basics-toast";
        const h = document.createElement("h4");
        h.textContent = "\u26a0 Load CLIP: check cannot run";
        const p = document.createElement("p");
        p.textContent = NO_MODEL_TEXT;
        box.appendChild(h);
        box.appendChild(p);
        document.body.appendChild(box);
        setTimeout(() => { try { box.remove(); } catch (e) { } }, 9000);
    } catch (e) { /* never break the ui */ }
}

function modelUnlinked(node) {
    const inp = (node.inputs || []).find((i) => i && i.name === "model");
    return !inp || inp.link == null;
}

function _drawRemove(ctx, node, width, y, height) {
    const h = Math.max(10, height - 4);
    const w = Math.min(h, width - 30);
    const x = (width - w) / 2;
    ctx.save();
    ctx.strokeStyle = "#4a4a4a";
    ctx.fillStyle = "#222";
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y + 2, w, h, 4);
    else ctx.rect(x, y + 2, w, h);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = AMBER;
    ctx.font = `bold ${Math.round(h * 0.62)}px Arial`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(REMOVE_GLYPH, width / 2, y + 2 + h / 2 + 1);
    ctx.restore();
}

function addRemoveButton(node, spec) {
    if (_w(node, REMOVE_BTN)) return;                  // idempotent
    const w = node.addWidget("button", REMOVE_BTN, null,
                             () => removeLastSlot(node, spec));
    if (!w) return;
    w.serialize = false;
    w.options = w.options || {};
    w.options.serialize = false;   // both spellings are read in the wild
    w.label = REMOVE_GLYPH;
    w.draw = _drawRemove;
}

app.registerExtension({
    name: "polyhedron.basics",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODES.includes(nodeData.name)) return;
        const spec = SLOTS[nodeData.name] || null;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            const baseCS = this.computeSize.bind(this);
            this.computeSize = (out) => {
                const s = baseCS(out);
                s[1] += STATUS_H;
                return s;
            };
            // v897: height only. `setSize(computeSize())` is a HARD RESET to
            // LiteGraph's computed MINIMUM -- it throws away a width the user
            // dragged out. ph_switch.js:39 records what that cost when tidy() did
            // it; v888 pinned the law and gave it one home. These were the last
            // four callers still doing the reset.
            refit(this);
            // v828: runs for ALL three loaders (the VAE has no slot spec but
            // its combo is sized too). Same timing as the slot wiring: the
            // setTimeout(0) lands AFTER LiteGraph poured widgets_values in,
            // so a stored bare value is on the widget by the time we look.
            setTimeout(() => {
                try { normalizeSized(this); } catch (e) { }
                // v829: the bubble, once, after the links are poured too.
                try {
                    if (nodeData.name === "ULSLoadCLIP"
                        && modelUnlinked(this)) toastNoModel(this);
                } catch (e) { }
            }, 0);
            if (spec) {
                // Everything load-bearing goes in the setTimeout(0) of
                // onNodeCreated -- nodeType.prototype.configure does NOT fire in
                // this frontend (proven v606), and the widgets are not all in
                // place during the constructor.
                setTimeout(() => {
                    try {
                        wireSlots(this, spec);
                        addRemoveButton(this, spec);
                        // AFTER the load, never during it: while LiteGraph is
                        // pouring widgets_values in, the rows must still be in
                        // canon order so the values land where they belong.
                        _toDisplay(this, spec);
                        applySlots(this, spec);
                    } catch (e) { /* never break the ui */ }
                }, 0);
            }
        };

        if (spec) {
            // After a workflow load the values restore asynchronously --
            // normalise on the double-rAF (the proven v531 onConfigure pattern).
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                onConfigure?.apply(this, arguments);
                requestAnimationFrame(() => requestAnimationFrame(() => {
                    try {
                        wireSlots(this, spec);
                        addRemoveButton(this, spec);
                        _toDisplay(this, spec);
                        applySlots(this, spec);
                    } catch (e) { /* never break the ui */ }
                }));
            };

            // It has to be serialize() and not onSerialize(): onSerialize is a
            // callback the base method chooses to invoke, and this frontend has
            // been watched declining to invoke a base method. There is no
            // saving a graph without serialize() itself.
            if (spec.canon) {
                const _base = nodeType.prototype.serialize;
                nodeType.prototype.serialize = function () {
                    return _serializeInCanon(this, spec, _base, arguments);
                };
            }
        }

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const p = message ? message.pls_basics : null;
            const text = Array.isArray(p) ? p[0] : p;
            if (text) {
                this._plsStatus = String(text);
                this.setDirtyCanvas(true, true);
            }
        };

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        // v829: connect resets the flag, unplug warns again -- state
        // CHANGES speak, redraws never do.
        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            onConnectionsChange?.apply(this, arguments);
            if (nodeData.name !== "ULSLoadCLIP") return;
            try {
                if (!modelUnlinked(this)) this._plsNoModelToasted = false;
                else toastNoModel(this);
            } catch (e) { }
        };
        nodeType.prototype.onDrawForeground = function (ctx) {
            onDrawForeground?.apply(this, arguments);
            if (this.flags && this.flags.collapsed) return;
            // v827 -- Frank's ask: when NO model is attached, the
            // model-side compatibility check simply cannot run, and the
            // node should SAY so, in the suite's official amber, right
            // where the status lives. The header check (safetensors
            // slate, no weights read) still runs either way -- the
            // warning names exactly what is missing, nothing more.
            let text = this._plsStatus ? String(this._plsStatus) : "";
            let colour = "#9a9a9a";
            if (nodeData.name === "ULSLoadCLIP" && modelUnlinked(this)) {
                // v829: same words as the bubble -- ONE source
                // (NO_MODEL_TEXT), two voices.
                text = "\u26a0 " + NO_MODEL_TEXT;
                colour = AMBER;
            }
            if (!text) return;
            ctx.save();
            ctx.font = "11px Arial";
            ctx.fillStyle = colour;
            ctx.textAlign = "left";
            const maxW = Math.max(40, this.size[0] - 16);
            while (text.length > 8 && ctx.measureText(text).width > maxW) {
                text = text.slice(0, -6) + "\u2026";
            }
            ctx.fillText(text, 8, this.size[1] - 5);
            ctx.restore();
        };
    },
});
