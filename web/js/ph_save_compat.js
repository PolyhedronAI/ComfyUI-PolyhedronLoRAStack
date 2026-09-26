/*
 * ph_save_compat.js -- v1021: the canon save path, on EVERY frontend.
 *
 * Three nodes show their widgets in a DISPLAY order that differs from the
 * order widgets_values is written in (the CANON, append-only, #577):
 * Load CLIP (ph_basics.js), CLIP Text Encode (ph_clip_encode.js) and
 * Reference (ph_reference.js). widgets_values is read BY POSITION on load, so
 * the save has to be canon. Since v606 each of them overrides
 * nodeType.prototype.serialize and swings the rows to canon for the length of
 * that one call.
 *
 * MEASURED 26.09. on Core master's frontend 1.53.6: graph.serialize() no
 * longer calls node.serialize() at all (0 calls, 1 on 1.49.6). It writes the
 * node itself (serialiseWidgetValues: live row order, serialize === false
 * skipped) and then calls the node's onSerialize -- UNBOUND, as
 * `r?.(o)` inside runExtensionSerializeHook, so `this` is undefined there.
 * Result: display order on disk, and the next load pours the values into the
 * wrong rows (Load CLIP type=minimax -> "default", CTE prompt -> separator).
 * 1.53.6 also writes widgets_values_named, but restores by name only with
 * Comfy.Workflow.NamedValuesRestore, which defaults to OFF.
 *
 * THE FIX, for both worlds at once: a per-INSTANCE onSerialize closure (the
 * node is captured, `this` is not needed). When the node is still in display
 * order at that moment -- 1.53.6 -- it rebuilds widgets_values from the rows
 * in canon order with exactly 1.53.6's rule. On 1.49.6 the serialize() swing
 * has already put the rows in canon when onSerialize runs, so the closure
 * sees "not displayed" and touches nothing: the old path stays byte for byte
 * what it was. tests/test_v1021_save_compat_js.py guards both halves;
 * tools/browser_roundtrip_probe.py measures save -> reload on a live frontend.
 */

/* 1.53.6's serialiseWidgetValues, mirrored: skip serialize === false, deep
 * copy objects, undefined -> null. Pure. */
export function serialiseValues(widgets) {
    const out = [];
    for (const w of widgets || []) {
        if (!w || w.serialize === false) continue;
        const v = w.value;
        out.push(typeof v === "object" && v !== null ? JSON.parse(JSON.stringify(v)) : (v ?? null));
    }
    return out;
}

/*
 * Install on ONE node instance (from onNodeCreated).
 *   isDisplayed(node)   -> true while the rows are in display order
 *   withCanon(node, fn) -> runs fn with the rows in canon order, restores
 *                          display order afterwards (in a finally), returns
 *                          fn's result
 *   extra(node, o)      -> optional: more fields for the saved object
 */
export function saveInCanon(node, isDisplayed, withCanon, extra) {
    if (!node || node._plsSaveCompat) return;
    node._plsSaveCompat = true;
    const prev = node.onSerialize;
    node.onSerialize = function (o) {
        try { if (typeof prev === "function") prev.call(node, o); } catch (e) { /* not ours */ }
        if (!o) return;
        try {
            if (isDisplayed(node)) {
                const vals = withCanon(node, () => serialiseValues(node.widgets));
                if (Array.isArray(vals)) o.widgets_values = vals;
            }
        } catch (e) { /* never break a save */ }
        try { if (extra) extra(node, o); } catch (e) { /* never break a save */ }
    };
}
