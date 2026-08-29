/**
 * ph_palette.js -- v600
 *
 * ONE colour for all 40 nodes of the Suite, set in ONE place.
 *
 * THE COLOUR IS MEASURED, NOT CHOSEN. Frank said "the blue of the LoRA Stack
 * node". The LoRA Stack node's FRAME is not blue -- it is LiteGraph's default
 * grey (#333 title, #353535 body), same as every other node in the tree. The
 * blue he sees is the node's OWN widget canvas, painted by ph_lora_stack.js,
 * and it reads #1a1a2a off the screenshot. So his sentence -- "this dark blue,
 * paired with the standard grey" -- describes exactly this: a #1a1a2a body
 * inside a grey frame. That pairing is what spreads across the Suite here.
 *
 * THE HOOK IS THE CATEGORY. All 40 node classes sit under "Polyhedron/" --
 * verified by walking the AST of nodes/, not by assuming. Anything that ever
 * leaves that prefix silently loses its colour, which is a visible failure and
 * therefore an honest one.
 *
 * THE TRAP, AND WHY serialize() IS PATCHED.
 * LiteGraph writes `color`/`bgcolor` into the workflow JSON whenever they are
 * set. Set the default in onNodeCreated and walk away, and the default is FROZEN
 * into every workflow the moment it is saved -- so changing this constant later
 * would repaint new nodes and leave every existing graph on the old colour, for
 * good. Frank's own condition ("nachtraegliche Farb-Anpassungen nicht
 * ausgeschlossen") would have been dead on arrival, and it would have died
 * quietly, which is the worst way.
 *
 * So: the default is applied on creation, and stripped again on serialize IF it
 * is still the default. A colour the user actually changed differs from the
 * constant, survives the strip, is saved, and beats the default on reload --
 * because LiteGraph's configure() copies whatever the JSON carries over the top
 * of what onNodeCreated set. Hand-set colours win. Untouched nodes stay
 * repaintable, forever, from this one line.
 *
 * ---------------------------------------------------------------------------
 * v895 -- THE ONE COLOUR THIS COULD NOT KEEP: "No color".
 *
 * Reported since v600 and driven again on 28.08: pick right-click > Colors >
 * "No color" on any of the 40 nodes and it goes grey as asked -- then quietly
 * comes back Pack-blue on the next reload. Measured in the real bundle
 * (@comfyorg/litegraph 0.17.2), not assumed:
 *
 *   menu     -> inner_clicked -> item.setColorOption(v.value ? ... : null)
 *   node     -> setColorOption(null) => delete this.color; delete this.bgcolor
 *   serialize-> `if (this.color) o.color = ...` -- deleted keys write NOTHING
 *
 * So in the saved file "the user chose no colour" and "the user never touched
 * it" are THE SAME STATE: absence. onNodeCreated then repaints the default and
 * configure() has nothing to override it with. The strip above is not the
 * culprit -- the information was never there to strip.
 *
 * THE FIX RECORDS THE CHOICE, not the colour. `setColorOption` is wrapped, and
 * a null pick sets one flag in node.properties -- serialised with the workflow
 * (LGraphNode.serialize clones properties), never seen by the backend, the same
 * mechanism as the v740 image lock and the v750 cutout row. Any real colour
 * clears the flag again. onConfigure -- which LiteGraph calls at the END of
 * configure(), after the saved values have landed -- honours the flag by taking
 * the default back off. Nothing else changes: an untouched node still carries
 * no colour in the JSON and is still repaintable from the one line below.
 *
 * HONEST LIMIT, said out loud: the menu offers the nine plus "none", so once a
 * node is on "No color" the menu cannot put it back on the SUITE default -- it
 * can only reach the nine. Clearing `ph_no_colour` in the properties panel
 * restores it. Adding a "Suite default" menu entry is a separate job with its
 * own go.
 */
import { app } from "../../scripts/app.js";

// ---- The one line. Change these two, repaint the Suite. -----------------
const PH_BGCOLOR = "#1a1a2a";   // body  -- the LoRA Stack's dark blue, measured
const PH_COLOR   = "#333";      // title -- LiteGraph's standard grey, as paired
// ------------------------------------------------------------------------

// v895: the user's "No color" lives here, because absence cannot carry it.
const PROP_NO_COLOUR = "ph_no_colour";

const MINE = (nodeData) =>
    String(nodeData?.category || "").startsWith("Polyhedron");

app.registerExtension({
    name: "polyhedron.palette",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!MINE(nodeData)) return;

        // --- apply the default on creation -------------------------------
        // A node being LOADED gets this too -- onNodeCreated runs before
        // configure() -- which is why onConfigure below takes it back off for a
        // node whose saved properties say "no colour".
        //
        // NOT GUARDED HERE, and that is a measurement, not an oversight: my
        // first v895 draft also checked the flag in this hook, and the mutation
        // round found the check could not fail -- removing it left the guard
        // green. Read in 0.17.2: every path that can carry saved properties
        // goes through configure() (clone() does `node.configure(data)` too),
        // and at onNodeCreated time properties hold class defaults, never the
        // saved ones. A second gate that can never fire is not belt and braces;
        // it is a second place to keep in step, and it would have made the real
        // gate below look optional.
        const _created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = _created ? _created.apply(this, arguments) : undefined;
            this.color = PH_COLOR;
            this.bgcolor = PH_BGCOLOR;
            return r;
        };

        // --- v895: remember a "No color" pick ----------------------------
        // LiteGraph's own method, wrapped: null clears both keys, and that
        // absence is indistinguishable from "never touched" in the file. The
        // flag is the difference. Any real pick clears it, so the flag can only
        // ever describe the CURRENT state.
        const _setColorOption = nodeType.prototype.setColorOption;
        nodeType.prototype.setColorOption = function (colorOption) {
            const r = _setColorOption
                ? _setColorOption.apply(this, arguments) : undefined;
            this.properties = this.properties || {};
            if (colorOption == null) this.properties[PROP_NO_COLOUR] = true;
            else delete this.properties[PROP_NO_COLOUR];
            return r;
        };

        // --- and honour it on load ---------------------------------------
        // configure() calls onConfigure LAST, after the saved fields are in.
        const _configure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = _configure ? _configure.apply(this, arguments) : undefined;
            const flagged = (info && info.properties
                             && info.properties[PROP_NO_COLOUR])
                         || (this.properties && this.properties[PROP_NO_COLOUR]);
            if (flagged) {
                delete this.color;
                delete this.bgcolor;
            }
            return r;
        };

        // --- and REFUSE to freeze it into the file ------------------------
        // Only a colour that DIFFERS from the default is the user's own, and
        // only that one gets written. Everything else stays unpainted in the
        // JSON and inherits whatever this file says next time.
        const _serialize = nodeType.prototype.serialize;
        nodeType.prototype.serialize = function () {
            const o = _serialize ? _serialize.apply(this, arguments) : {};
            if (o && o.color === PH_COLOR && o.bgcolor === PH_BGCOLOR) {
                delete o.color;
                delete o.bgcolor;
            }
            return o;
        };
    },
});
