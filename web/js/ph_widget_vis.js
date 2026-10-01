/*
 * ph_widget_vis.js -- ONE way to take a widget off the node (v888)
 * ════════════════════════════════════════════════════════════════
 * Frank, 26.08.: "Kann man diese ausgegrauten Eintraege auch komplett
 * verstecken ...? Fuer WAN mag das okay sein, aber so sieht es unfertig aus."
 *
 * He is right, and the old reasoning had an expiry date on it. Greying was
 * chosen because it keeps the node's GEOMETRY constant -- every widget keeps
 * its slot and its height, so nothing can jump. That was the safe move before
 * the pack had a proven hide. It has one now (ph_clip_encode -> ph_cutout ->
 * ph_mask_editor, v753/v755), and a row that can do nothing is noise.
 *
 * WHY A TYPE MARKER AND NOT `hidden`
 * LiteGraph skips a widget in the LAYOUT pass by its TYPE. Setting only
 * `hidden` (plus computeSize) leaves the canvas widgets closing the gap while
 * a DOM element stays put -- exactly the overlap measured in v753. And on
 * show, computeSize must be DELETED when the widget never had one: writing
 * `undefined` back leaves the row zero-height forever.
 *
 * WHAT DOES NOT CHANGE, and this is the load-bearing part:
 * the widget STAYS IN node.widgets. ComfyUI serialises widgets_values BY
 * INDEX, so removing a row would renumber every saved workflow (the #577 law).
 * A hidden widget keeps its slot and its value; only the pixels go.
 *
 * THE LABEL still gets its base restored on show, so the INACTIVE_MARK of the
 * old greying can never end up baked into a visible row.
 */

export const HIDDEN_PREFIX = "pls-hidden-";

export function isHidden(w) {
    return !!w && String(w.type || "").startsWith(HIDDEN_PREFIX);
}

/* Take a widget out of the layout, or put it back. Idempotent. */
export function setHidden(w, hidden) {
    if (!w) return false;
    const was = isHidden(w);
    if (!!hidden === was) return false;          // nothing to do, nothing to refit
    if (hidden) {
        w._plsVisType = w.type;
        w._plsVisHadCS = Object.prototype.hasOwnProperty.call(w, "computeSize");
        w._plsVisCS = w.computeSize;
        w.type = HIDDEN_PREFIX + w.type;
        w.computeSize = () => [0, -4];
        w.hidden = true;
    } else {
        w.type = w._plsVisType !== undefined
            ? w._plsVisType : String(w.type).slice(HIDDEN_PREFIX.length);
        if (w._plsVisHadCS) w.computeSize = w._plsVisCS;
        else delete w.computeSize;
        w.hidden = false;
    }
    return true;                                  // the layout really changed
}

/*
 * Height-only refit after a visibility change.
 *
 * WIDTH IS NEVER TOUCHED (the v531 law): a node the user widened must stay
 * that width, and a pane that measured its own width must keep it. Only the
 * height follows the rows that are actually there.
 *
 * Guarded end to end -- a cosmetic refit may never break a graph.
 */
export function refit(node) {
    if (!node || !node.setSize || !node.computeSize) return;
    try {
        node.setSize([node.size[0], node.computeSize()[1]]);
        node.setDirtyCanvas?.(true, true);
    } catch (e) { /* never break the canvas over a layout nicety */ }
}

/*
 * v1022: a floor for text fields, the same on both renderers.
 *
 * Nodes 2.0 gives every textarea 64 px. Classic sizes a multiline DOM field
 * from the node's computed minimum, and with two text fields on one node
 * (MiniMax Keyframes, MiniMax Reference: prompt + tags since v1017) each got
 * 38 px -- one line, and P1 class F5 ("64 px in Nodes 2.0 vs 38 px classic").
 * The floor goes through the DOM widget's own layout hook (getMinHeight, the
 * ph_show_text.js way), so computeSize and arrangeWidgets both see it.
 * Measured on 1.49.6: the element is drawn DOM_FIELD_MARGIN px smaller than
 * its layout height, so the hook asks for px + margin.
 */
export const FIELD_MIN_PX = 64;
export const DOM_FIELD_MARGIN = 12;

/* While a field's input is WIRED (1.49.6: the widget stays in the layout,
 * only its element is hidden) it keeps one narrow row for its socket dot --
 * no empty field-sized gap (v1021 left ~50 px, measured), and the dot does
 * not land on the next widget's row (it did at 0 px, measured). */
export const WIRED_ROW_PX = 24;

export function fieldWired(node, name) {
    return (node && node.inputs || []).some((i) => i && i.link != null
        && (i.name === name || (i.widget && i.widget.name === name)));
}

export function fieldFloor(node, names, px = FIELD_MIN_PX) {
    for (const name of names || []) {
        const w = (node && node.widgets || []).find((x) => x && x.name === name);
        if (!w || !w.element || w.element.tagName !== "TEXTAREA") continue;
        w.options = w.options || {};
        if (w.options._plsFloor) continue;
        const prevMin = w.options.getMinHeight;
        const prevMax = w.options.getMaxHeight;
        w.options.getMinHeight = () => (fieldWired(node, name) ? WIRED_ROW_PX
            : Math.max(px + DOM_FIELD_MARGIN, typeof prevMin === "function" ? (prevMin() || 0) : 0));
        w.options.getMaxHeight = () => (fieldWired(node, name) ? WIRED_ROW_PX
            : (typeof prevMax === "function" ? prevMax() : undefined));
        w.options._plsFloor = true;
    }
}

/* Grow-only height fit: a node smaller than its computed minimum (a save from
 * before the floor) grows to it; a node the user made taller keeps its size.
 * Width is never touched (v531). */
export function growToFloor(node) {
    if (!node || !node.setSize || !node.computeSize) return;
    try {
        const h = node.computeSize()[1];
        if (node.size[1] < h) {
            node.setSize([node.size[0], h]);
            node.setDirtyCanvas?.(true, true);
        }
    } catch (e) { /* never break the canvas over a layout nicety */ }
}
