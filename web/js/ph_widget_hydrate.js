/**
 * Polyhedron -- widget hydration for APPENDED widgets (v686).
 *
 * THE FIELD BUG THIS EXISTS FOR (v685, Frank's screen):
 *
 *   ULSSeed 86: Failed to convert an input value to a FLOAT value:
 *   noise_strength, None
 *
 * Append-only protects the NUMBERING of widgets_values -- it does not fill the
 * new positions. A graph saved before the append carries a shorter array; on
 * load LiteGraph creates the new widgets and then pours the old array over the
 * front of the list, leaving the tail at `undefined`. The node then sends null,
 * and ComfyUI's type validation rejects the prompt BEFORE our python runs --
 * which is why a python-side default (`noise_strength=1.0`) cannot save it.
 *
 * THE FIX, and why it needs no node definition: when LiteGraph builds a node it
 * creates every widget with its DEFAULT value. That happens before configure()
 * pours in the saved array. So a snapshot taken in onNodeCreated IS the default
 * set -- no lookup, no duplicated table that can drift from python.
 *
 * Use it on any node that gains widgets after release:
 *
 *     import { snapshotDefaults, hydrateMissing } from "./ph_widget_hydrate.js";
 *     ... in onNodeCreated:  snapshotDefaults(this);
 *     ... in a setTimeout(0) after onConfigure:  hydrateMissing(this);
 *
 * The setTimeout matters for the same reason it matters everywhere else in this
 * tree: while widgets_values is being poured in, nothing may touch the widget
 * list. Hydration runs after, never during.
 */

/** Remember every widget's current value as its default. Call in onNodeCreated,
 *  where the values ARE the defaults. */
export function snapshotDefaults(node) {
    if (!node || !Array.isArray(node.widgets)) return 0;
    const d = {};
    for (const w of node.widgets) {
        if (w && w.name !== undefined) d[w.name] = w.value;
    }
    node._plsWidgetDefaults = d;
    return Object.keys(d).length;
}

/** Is this widget's value missing in the sense that a short saved array
 *  left it behind?
 *
 *  undefined/null is the shape the FLOAT case (v685) showed. v796 adds
 *  the third face, found in the field: a COMBO left behind arrives as an
 *  EMPTY STRING, not as undefined -- so the old test walked straight
 *  past it and the empty string went to the server, where ComfyUI's type
 *  validation rejected the whole prompt with
 *      Value not in list: image_mode: '' not in [...]
 *  BEFORE any of our python could run. The test is deliberately narrow:
 *  a value OUT OF ITS OWN LIST is broken whatever it is, while an empty
 *  string on a text widget is a perfectly legitimate value and must be
 *  left alone. So this only widens for widgets that carry a value list.
 */
export function isMissing(w, def) {
    if (!w) return false;
    if (w.value === undefined || w.value === null) return true;
    const vals = w.options && w.options.values;
    if (Array.isArray(vals) && vals.length && !vals.includes(w.value)) {
        return true;
    }
    if (def !== undefined && def !== null && !_fitsType(w.value, def)) {
        return true;
    }
    return false;
}

/* v817 -- THE FOURTH FACE, and why the test moved from the LIST to the TYPE.
 *
 * FIELD BUG (Frank's screen, v816 running a graph saved before v815):
 *     ULSOutpaint 156: Failed to convert an input value to a INT value:
 *     canvas_w, , invalid literal for int() with base 10: ''
 *
 * Same empty string as v796, on a widget that has no value list -- so the
 * combo test walked straight past it and the empty string went to the
 * server, where ComfyUI's type validation rejected the whole prompt
 * before any of our python could run. Exactly the shape v796 was written
 * to stop, one widget type over.
 *
 * v796 pinned the OBSERVATION ("an empty string means missing") to the
 * COMBO case, because that is where it was seen. The real rule is the
 * one underneath: A VALUE THAT CANNOT BE ITS OWN TYPE IS MISSING. The
 * remembered default carries that type -- it was snapshotted from the
 * node definition itself, so there is no table here that can drift from
 * python.
 *
 * Deliberately narrow, in both directions:
 *   - only NUMBER and BOOLEAN defaults are judged. A string default says
 *     nothing, because an empty string on a text widget is a perfectly
 *     legitimate value (the v796 counter-case, still pinned).
 *   - a string that parses cleanly to a finite number is NOT missing.
 *     It is untidy, not broken, and healing it would overwrite a value
 *     the user may have typed. This only rescues what would otherwise
 *     fail hard.
 */
function _fitsType(value, def) {
    if (typeof def === "number") {
        if (typeof value === "number") return Number.isFinite(value);
        if (typeof value === "string") {
            const t = value.trim();
            return t !== "" && Number.isFinite(Number(t));
        }
        return false;
    }
    if (typeof def === "boolean") return typeof value === "boolean";
    return true;
}

/** Fill widgets left null/undefined by a short saved array. Returns how many
 *  were filled. A widget with no remembered default is left ALONE -- guessing a
 *  value is worse than the loud error, because it would be silently wrong. */
export function hydrateMissing(node) {
    if (!node || !Array.isArray(node.widgets)) return 0;
    const d = node._plsWidgetDefaults;
    if (!d) return 0;
    let filled = 0;
    for (const w of node.widgets) {
        if (!w || w.name === undefined) continue;
        if (!isMissing(w, d[w.name])) continue;
        if (d[w.name] === undefined || d[w.name] === null) continue;
        w.value = d[w.name];
        filled++;
    }
    if (filled && node.setDirtyCanvas) node.setDirtyCanvas(true, false);
    return filled;
}

/** Both halves wired onto a node type in one call. Returns nothing; it patches
 *  onNodeCreated and onConfigure, chaining whatever was there before. */
/* =========================================================================
 * PIN MIGRATION (v805) -- the same problem one floor up.
 *
 * THE FIELD BUG THIS EXISTS FOR (v804, Frank's screen): the Outpaint node
 * renamed `prompt_text` to `positive_text` and dropped `negative_text`.
 * The node definition changed; the SAVED GRAPH did not. LiteGraph adds
 * the new socket on load but never removes the old ones -- least of all
 * while a wire still hangs on them -- so the node showed THREE text pins,
 * two of which it no longer knows. And I had told Frank "links are stored
 * by index, the rename should survive", which answered the wrong
 * question: the LINK survives, the stale PIN stays.
 *
 * The shape is lifted from ph_power_upscale.js `_reorderInputsToDisplay`
 * (v548), which already rebuilds node.inputs after configure(). Two
 * things are taken from it verbatim, both paid for there:
 *   - the LINK TABLE has to be repaired. LLink.target_slot must match the
 *     new index or wires render, and disconnect, against the wrong socket.
 *   - fail LOUD, never a silent skip (the v540 lesson; a silent no-op
 *     cost a whole live round in v547).
 *
 * Renaming CARRIES THE WIRE ACROSS instead of dropping it, which is the
 * whole point: a user should not have to rewire because we renamed a pin.
 * ========================================================================= */

function _linkOf(node, id) {
    const links = node.graph && node.graph.links;
    if (!links || id == null) return null;
    return (typeof links.get === "function") ? links.get(id) : links[id];
}

/*
 * Apply a pin migration to ONE node.
 *
 *   spec.renamed  {oldName: newName}  -- move the wire, drop the old pin
 *   spec.removed  [name, ...]         -- drop the pin and its wire
 *
 * Returns a list of what it did (empty when there was nothing to do, so a
 * second load is a silent no-op -- idempotent by construction).
 */
export function migrateInputs(node, spec) {
    const done = [];
    if (!node || !Array.isArray(node.inputs)) return done;
    const renamed = (spec && spec.renamed) || {};
    const removed = new Set((spec && spec.removed) || []);
    const byName = new Map();
    for (const inp of node.inputs) if (inp && inp.name) byName.set(inp.name, inp);

    // 1. RENAME: carry the wire to the new socket, then drop the old pin.
    for (const oldName of Object.keys(renamed)) {
        const from = byName.get(oldName);
        if (!from) continue;
        const to = byName.get(renamed[oldName]);
        if (!to) {
            // The new socket does not exist -- the node definition is not
            // what this table expects. Say so; do NOT drop the old pin,
            // or the wire is lost with nowhere to go.
            console.warn("[PLS] pin migration skipped: `" + oldName +
                         "` cannot move to `" + renamed[oldName] +
                         "`, which this node does not have");
            continue;
        }
        if (from.link != null) {
            if (to.link != null) {
                // Both wired: keep what is on the NEW pin, drop the old
                // wire. Guessing which one the user meant would be worse.
                console.warn("[PLS] pin migration: `" + renamed[oldName] +
                             "` is already wired, dropping the old wire " +
                             "on `" + oldName + "`");
            } else {
                to.link = from.link;
                const l = _linkOf(node, from.link);
                if (l) l.target_slot = node.inputs.indexOf(to);
                done.push(oldName + " -> " + renamed[oldName] + " (wire carried)");
            }
            from.link = null;
        } else {
            done.push(oldName + " -> " + renamed[oldName]);
        }
        removed.add(oldName);
    }

    // 2. REMOVE: take the pin out and disconnect whatever hung on it. A
    //    link pointing at a pin that no longer exists is a zombie: it
    //    survives the next save and reappears as a wire to nowhere.
    const keep = [];
    for (const inp of node.inputs) {
        if (inp && removed.has(inp.name)) {
            if (inp.link != null) {
                const l = _linkOf(node, inp.link);
                const links = node.graph && node.graph.links;
                if (l && links) {
                    try {
                        if (typeof links.delete === "function") links.delete(inp.link);
                        else delete links[inp.link];
                    } catch (e) { /* never break a load */ }
                }
                if (!done.some((d) => d.indexOf(inp.name + " ->") === 0)) {
                    done.push(inp.name + " (removed, wire dropped)");
                }
            } else if (!done.some((d) => d.indexOf(inp.name + " ->") === 0)) {
                done.push(inp.name + " (removed)");
            }
            continue;
        }
        keep.push(inp);
    }
    if (keep.length !== node.inputs.length) node.inputs = keep;

    // 3. REPAIR every remaining slot index -- the v548 rule. Removing one
    //    pin shifts every socket behind it, and a wire whose target_slot
    //    still points at the old index lands on its neighbour.
    for (let i = 0; i < node.inputs.length; i++) {
        const inp = node.inputs[i];
        if (!inp || inp.link == null) continue;
        const l = _linkOf(node, inp.link);
        if (l) l.target_slot = i;
    }
    if (done.length) {
        console.info("[PLS] pin migration on " + (node.type || "node") +
                     ": " + done.join(", "));
        node.setDirtyCanvas?.(true, true);
    }
    return done;
}

/*
 * Hang a pin migration on a node type. Runs AFTER configure() has poured
 * the saved graph in -- same deferral as the widget hydration, and for
 * the same reason: nothing may touch the node while it is being filled.
 */
export function attachInputMigration(nodeType, spec) {
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
        const r = onConfigure?.apply(this, arguments);
        const self = this;
        setTimeout(() => {
            try { migrateInputs(self, spec); } catch (e) { /* never break a load */ }
        }, 0);
        return r;
    };
}

export function attachHydration(nodeType) {
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        const r = onCreated?.apply(this, arguments);
        try { snapshotDefaults(this); } catch (e) { /* never break a drop */ }
        return r;
    };
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
        const r = onConfigure?.apply(this, arguments);
        const self = this;
        setTimeout(() => {
            try { hydrateMissing(self); } catch (e) { /* never break a load */ }
        }, 0);
        return r;
    };
}
