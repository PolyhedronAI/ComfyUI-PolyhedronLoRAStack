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
