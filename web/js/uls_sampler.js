/**
 * Polyhedron Sampler — combi-node, split-setup, constant geometry
 * ─────────────────────────────────────────────────────────────────────────
 * ONE node handles both sampling paths. The `dual_moe` BOOLEAN (label "🛈 Mode")
 * is the selector at the top: Single ⟷ High + Low. Both paths live in the node at
 * once (a "split setup"): the fields of the inactive path are NOT hidden — they
 * are DISABLED (greyed + locked) and prefixed with "⊘", while the active path's
 * fields are live. Nothing is added or removed on toggle, so:
 *
 *   - The node NEVER changes size when you switch modes (constant geometry). All
 *     widgets stay laid out; only their enabled/disabled state changes.
 *   - The `model_low` input is ALWAYS present. It is declared optional in
 *     INPUT_TYPES, so an unconnected slot in Single is harmless (the backend only
 *     reads it in High + Low). Switching to Single never removes the slot, so a
 *     connected LOW-expert link is NEVER dropped.
 *
 * The two paths:
 *   - High + Low (noise-split MoE): Handoff + cfg_low are live; the manual
 *     start/end/leftover controls are disabled (the Handoff drives the split).
 *   - Single: the manual start/end/leftover controls are live; Handoff + cfg_low
 *     are disabled.
 *
 * NAMING (display only — internal names unchanged): pill states Single /
 * "High + Low"; `boundary` shows as "🛈 Handoff".
 *
 * SERIALISATION: the selector is the serialised `dual_moe` BOOLEAN (index 0).
 * There is no non-serialised widget, so nothing can leak into widgets_values
 * (that is what the removed v406 "Tune for" preset did). The v408 phantom-heal
 * migration is kept to repair graphs the preset already corrupted, alongside the
 * pre-v404 reorder migration.
 *
 * v409 — replaces the v407/v408 true show/hide (which RESIZED the node and removed
 * the model_low slot in Single, dropping the link). Same two paths, but now a
 * constant-geometry split-setup: disable in place instead of hide, model_low
 * pinned. Mode toggling no longer changes the node's dimensions or its inputs.
 *
 * v413 — adds the serialised `preview_mode` combo at the END of the widget list
 * (15 fields now). Appending keeps every existing index stable. Because a new
 * serialised slot shifts the widgets_values length, the legacy length heuristics
 * are DECOUPLED to frozen literals (LEN_PHANTOM/LEN_PRE_V404/LEN_OLD_CURRENT) so
 * a v413 current save (15) is not mistaken for a v406/v407 phantom save (also 15);
 * the two are told apart by the type of wv[1] (number boundary vs. preset string).
 * Old graphs (pre-v404, phantom, and 14-field v404..v412 current) all heal to the
 * 15-field order with preview_mode defaulted — closing the v409 phantom-heal item.
 * preview_mode is always live (not part of either path's disable set).
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { setHidden, refit } from "./ph_widget_vis.js";

const NODE_TYPE = "ULSSampler";
// live only in High + Low — the Handoff drives the split itself
// v545: widgets whose VALUE decides another widget's enabled state -- each one gets a
// callback that re-runs _applyModeState (dual_moe: the whole Single/Dual split;
// handoff_mode: whether scheduler_low is honoured).
const STATE_WIDGETS = ["dual_moe", "handoff_mode"];

const DUAL_ONLY = ["boundary", "cfg_low", "handoff_mode", "sampler_low", "scheduler_low",
                   "sigma_shift_low"];   // v495: handoff greys in Single too; v839: + the LOW shift
// live only in Single — manual HIGH/LOW step slicing
const SINGLE_ONLY = ["start_at_step", "end_at_step", "return_with_leftover_noise"];
// v416: schedule generators an external sigma schedule overrides (in either mode)
// v839 NOTE: sigma_shift_low is NOT here on purpose. This list is applied with
// _setDisabled(w, sigActive), which RE-ENABLES its members when no external
// curve is wired -- that pass would overwrite the DUAL_ONLY greying in Single.
// The LOW shift follows the scheduler_low pattern instead: greyed by DUAL_ONLY
// in Single, and by the disable-only dual block below in 'Continuous' or with
// an engaged external curve.
const SIGMA_INERT = ["scheduler", "steps", "denoise", "sigma_shift"];   // v495: + sigma_shift (cannot shift an external curve)

// Leading marker on the DISABLED path's field labels (renderer-independent cue;
// the w.disabled greying reinforces it where the renderer honours disabled).
const INACTIVE_MARK = "⊘ ";

// ── Current serialised widget order (17 fields since v492: handoff_mode appended
//    at the END for index stability — every existing slot keeps its position, and
//    preview_mode stays at index 14. v491 appended sigma_shift, v413 preview_mode,
//    the same way.) ────
const ORDER_V404 = [
    "dual_moe", "boundary", "cfg_low", "seed", "control_after_generate", "steps", "cfg",
    "sampler_name", "scheduler", "denoise", "add_noise", "start_at_step", "end_at_step",
    "return_with_leftover_noise", "preview_mode", "sigma_shift", "handoff_mode",
    // v544: per-expert sampler / scheduler. Appended at the END like every widget
    // since v413 -- every existing slot keeps its index, old saves stay readable.
    "sampler_low", "scheduler_low",
    // v839: per-expert sigma shift. Appended at the END the same way.
    "sigma_shift_low",
];

// Default for the v413 preview_mode widget (MUST match nodes INPUT_TYPES default).
const DEFAULT_PREVIEW_MODE = "Still · ComfyUI";

// Default for the v491 sigma_shift widget (MUST match nodes INPUT_TYPES default: 0.0 = OFF).
const DEFAULT_SIGMA_SHIFT = 0.0;

// Default for the handoff_mode widget (MUST match nodes INPUT_TYPES default).
const DEFAULT_HANDOFF_MODE = "Continuous";

// Defaults for the v544 per-expert widgets (MUST match nodes INPUT_TYPES).
// "same as high" = the LOW expert reuses the HIGH values -> byte-identical to v543.
const SAME_AS_HIGH = "same as high";
const DEFAULT_SAMPLER_LOW = SAME_AS_HIGH;
const DEFAULT_SCHEDULER_LOW = SAME_AS_HIGH;

// Default for the v839 sigma_shift_low widget (MUST match nodes INPUT_TYPES:
// -1 = the "same as high" sentinel -> the LOW expert follows sigma_shift).
const DEFAULT_SIGMA_SHIFT_LOW = -1.0;

// Index of preview_mode in the current 17-field order (14 — sigma_shift then handoff_mode
// trail it). The v420 rename keys off THIS slot, not the last field, since they trail it.
const PREVIEW_MODE_IDX = ORDER_V404.indexOf("preview_mode");

// Index of handoff_mode (16 — last field). The v493 rename keys off THIS slot.
const HANDOFF_MODE_IDX = ORDER_V404.indexOf("handoff_mode");

// v420: preview_mode combo values were renamed (Still· / Video·). Saved graphs hold the
// OLD value; translate it on load so an existing node keeps its mode instead of falling
// back to the default. Keys are pre-v420 values, values are the new ones.
const PREVIEW_MODE_RENAMES = {
    "Standard (ComfyUI)": "Still · ComfyUI",
    "Standard (latent2rgb)": "Still · latent2rgb",
    "latent2rgb (smooth)": "Video · latent2rgb (smooth)",
    "latent2rgb (crisp)": "Video · latent2rgb (crisp)",
    "taesd (taew2_1)": "Video · TAE (taew2_1)",
    "taesd (lighttaew2_1)": "Video · TAE (lighttaew2_1)",
};

// v493: handoff_mode combo values were renamed for clarity. Keys are the pre-v493 (v492)
// values, values are the new ones; translated on load so a v492 save keeps its mode.
const HANDOFF_MODE_RENAMES = {
    "Continue · leftover": "Continuous",
    "Rebase · x0 + renoise": "Wan MoE parity",
};

// ── v494: on-screen widget layout (DISPLAY order) ───────────────────────────
// The SERIALISED order stays ORDER_V404 (canonical) — this ONLY reorders what the
// user sees. handoff_mode sits directly under Handoff (boundary), sigma_shift under
// scheduler (like the Wan MoE KSampler), preview_mode last (above the preview pane).
// Same 17 names as ORDER_V404, permuted. Everything below is index-based (widgets_values
// is positional: value[i] == node.widgets[i].value), so it never relies on widget .name.
const DISPLAY_ORDER = [
    "dual_moe", "boundary", "handoff_mode", "seed", "control_after_generate",
    // v544: every LOW twin sits directly under its HIGH partner -- cfg_low under cfg,
    // sampler_low under sampler_name, scheduler_low under scheduler (v839:
    // sigma_shift_low under sigma_shift, the same rule).
    "steps", "cfg", "cfg_low", "sampler_name", "sampler_low", "scheduler", "scheduler_low",
    "sigma_shift", "sigma_shift_low", "denoise", "add_noise",
    "start_at_step", "end_at_step", "return_with_leftover_noise", "preview_mode",
];
// canonical index shown at each DISPLAY position, and the DISPLAY position of each canonical field
const CANON_IDX_AT_DISPLAY = DISPLAY_ORDER.map((n) => ORDER_V404.indexOf(n));
const DISPLAY_POS_OF_CANON = ORDER_V404.map((n) => DISPLAY_ORDER.indexOf(n));

// Reorder node.widgets from canonical (INPUT_TYPES) order to DISPLAY order, by POSITION.
// Fail-safe: only when node.widgets is exactly the 17 canonical widgets; else no-op (the
// node stays in canonical order, byte-identical to v493). Returns whether it reordered.
function _reorderWidgetsToDisplay(node) {
    const ws = node && node.widgets;
    if (!Array.isArray(ws) || ws.length !== ORDER_V404.length) return false;
    if (CANON_IDX_AT_DISPLAY.some((i) => i < 0)) return false;   // name set mismatch -> skip
    node.widgets = CANON_IDX_AT_DISPLAY.map((ci) => ws[ci]);
    return true;
}
// canonical widgets_values -> DISPLAY order (apply positionally to display widgets on load)
function _canonToDisplay(arr) {
    if (!Array.isArray(arr) || arr.length !== ORDER_V404.length) return arr;
    return CANON_IDX_AT_DISPLAY.map((ci) => arr[ci]);
}
// DISPLAY-order widgets_values -> canonical (the one true serialised order, on save)
function _displayToCanon(arr) {
    if (!Array.isArray(arr) || arr.length !== ORDER_V404.length) return arr;
    return DISPLAY_POS_OF_CANON.map((di) => arr[di]);
}

// ── Frozen serialised-array lengths (v413) ──────────────────────────────────
// DECOUPLED from ORDER_V404.length on purpose: appending preview_mode made
// ORDER_V404 15, which would have shifted the legacy length heuristics below and
// collided new saves with old phantom saves. These literals are the lengths of the
// on-disk formats we must still recognise (disambiguated by the type of wv[1]):
//   14  v404..v412 current   — NO preview_mode; wv[1] = boundary (number)
//   14  pre-v404              — Mode block at bottom; wv[1] = control_after_generate (string)
//   15  v406/v407 phantom     — 14 + the removed "Tune for" preset string at wv[1] (string)
//   15  v413..v490 current    — 14 + preview_mode at the end; wv[1] = boundary (number)
//   16  v491 current          — 15 + sigma_shift at the end; wv[1] = boundary (number)
//   17  v492 current          — 16 + handoff_mode at the end; wv[1] = boundary (number)
const LEN_OLD_CURRENT = 14;   // v404..v412 current, lacks preview_mode
const LEN_PRE_V404 = 14;      // pre-v404 reorder
const LEN_PHANTOM = 15;       // v406/v407 phantom-preset save (previously derived from ORDER_V404.length)
const LEN_V413_CURRENT = 15;  // v413..v490 current: has preview_mode, lacks sigma_shift; wv[1] = boundary (number)
const LEN_V491_CURRENT = 16;  // v491 current: has preview_mode+sigma_shift, lacks handoff_mode; wv[1] = boundary (number)
const LEN_V492_CURRENT = 17;  // v492..v543 current: lacks sampler_low/scheduler_low; wv[1] = boundary (number)
const LEN_V544_CURRENT = 19;  // v544..v838 current: lacks sigma_shift_low; wv[1] = boundary (number)

// ── Legacy "Tune for" preset values (the widget is gone — these strings are
// kept ONLY to recognise and heal the phantom slot in v406/v407 saves) ───────
const LEGACY_PRESET_VALUES = ["Custom", "Text→Video", "Image→Video"];

// ── Pre-v404 graph (Mode block at the bottom) ───────────────────────────────
const ORDER_PRE_V404 = [
    "seed", "control_after_generate", "steps", "cfg", "sampler_name", "scheduler",
    "denoise", "add_noise", "start_at_step", "end_at_step", "return_with_leftover_noise",
    "dual_moe", "boundary", "cfg_low",
];
const CONTROL_VALUES = ["fixed", "increment", "decrement", "randomize"];

function _looksPhantomPreset(wv) {
    // v406/v407 save: frozen length 15 (14 + the removed "Tune for" preset string
    // at wv[1]). Disambiguated from a v413 current save (also 15) by wv[1] being a
    // preset STRING here vs. the boundary NUMBER in a current save.
    return Array.isArray(wv)
        && wv.length === LEN_PHANTOM
        && typeof wv[1] === "string"
        && LEGACY_PRESET_VALUES.indexOf(wv[1]) !== -1;
}

function _healPhantomPreset(wv) {
    const out = wv.slice();
    out.splice(1, 1);                 // drop the phantom preset string at index 1 → 14 fields
    out.push(DEFAULT_PREVIEW_MODE);   // append the v413 preview_mode default → 15 fields
    out.push(DEFAULT_SIGMA_SHIFT);    // append the v491 sigma_shift default → 16 fields
    out.push(DEFAULT_HANDOFF_MODE);   // append the v492 handoff_mode default → 17 fields
    out.push(DEFAULT_SAMPLER_LOW);    // v544 → 18 (v839 straggler fix: this heal
    out.push(DEFAULT_SCHEDULER_LOW);  // v544 → 19  had stopped at 17 since v544)
    out.push(DEFAULT_SIGMA_SHIFT_LOW); // v839 → 20
    return out;
}

function _looksPreV404(wv) {
    // Pre-v404 save: frozen length 14, Mode block at the bottom so wv[1] is the
    // control_after_generate STRING. Disambiguated from a v404..v412 current save
    // (also 14) by wv[1] being a control string rather than the boundary number.
    return Array.isArray(wv)
        && wv.length === LEN_PRE_V404
        && typeof wv[1] === "string"
        && CONTROL_VALUES.indexOf(wv[1]) !== -1;
}

function _migratePreV404(wv) {
    const byName = {};
    for (let i = 0; i < ORDER_PRE_V404.length; i++) byName[ORDER_PRE_V404[i]] = wv[i];
    // Map the 14 known fields into the new order; preview_mode (absent pre-v404),
    // sigma_shift (absent pre-v491) and handoff_mode (absent pre-v492) are not in byName,
    // so fill each with its default → 17.
    return ORDER_V404.map((name) => {
        if (name === "preview_mode") return DEFAULT_PREVIEW_MODE;
        if (name === "sigma_shift") return DEFAULT_SIGMA_SHIFT;
        if (name === "handoff_mode") return DEFAULT_HANDOFF_MODE;
        // v839 straggler fix: the two v544 twins were missing here, so a
        // pre-v404 save migrated with two undefined tail values.
        if (name === "sampler_low") return DEFAULT_SAMPLER_LOW;
        if (name === "scheduler_low") return DEFAULT_SCHEDULER_LOW;
        if (name === "sigma_shift_low") return DEFAULT_SIGMA_SHIFT_LOW;
        return byName[name];
    });
}

function _looksOldCurrent(wv) {
    // v404..v412 current save: frozen length 14, NO preview_mode, wv[1] = boundary
    // (a number). Not phantom (length 15) and not pre-v404 (wv[1] is a string).
    return Array.isArray(wv)
        && wv.length === LEN_OLD_CURRENT
        && typeof wv[1] === "number";
}

function _healOldCurrent(wv) {
    const out = wv.slice();
    out.push(DEFAULT_PREVIEW_MODE);   // append the v413 preview_mode default → 15 fields
    out.push(DEFAULT_SIGMA_SHIFT);    // append the v491 sigma_shift default → 16 fields
    out.push(DEFAULT_HANDOFF_MODE);   // append the v492 handoff_mode default → 17 fields
    out.push(DEFAULT_SAMPLER_LOW);    // v544 → 18
    out.push(DEFAULT_SCHEDULER_LOW);  // v544 → 19
    out.push(DEFAULT_SIGMA_SHIFT_LOW); // v839 → 20
    return out;
}

function _looksV413Current(wv) {
    // v413..v490 current save: frozen length 15 (14 + preview_mode), NO sigma_shift.
    // wv[1] = boundary (a number). Disambiguated from the phantom (also 15) whose wv[1]
    // is a preset STRING. Healed by appending the sigma_shift default → 16 fields.
    return Array.isArray(wv)
        && wv.length === LEN_V413_CURRENT
        && typeof wv[1] === "number";
}

function _healV413Current(wv) {
    const out = wv.slice();
    out.push(DEFAULT_SIGMA_SHIFT);    // append the v491 sigma_shift default → 16 fields
    out.push(DEFAULT_HANDOFF_MODE);   // append the v492 handoff_mode default → 17 fields
    out.push(DEFAULT_SAMPLER_LOW);    // v544 → 18
    out.push(DEFAULT_SCHEDULER_LOW);  // v544 → 19
    out.push(DEFAULT_SIGMA_SHIFT_LOW); // v839 → 20
    return out;
}

function _looksV491Current(wv) {
    // v491 current save: frozen length 16 (15 + sigma_shift), NO handoff_mode.
    // wv[1] = boundary (a number). Length uniquely identifies it (only v491 current is 16);
    // the wv[1] type check matches the sibling heals. Healed by appending the handoff_mode
    // default → 17 fields.
    return Array.isArray(wv)
        && wv.length === LEN_V491_CURRENT
        && typeof wv[1] === "number";
}

function _healV491Current(wv) {
    const out = wv.slice();
    out.push(DEFAULT_HANDOFF_MODE);   // append the v492 handoff_mode default → 17 fields
    out.push(DEFAULT_SAMPLER_LOW);    // v544 → 18
    out.push(DEFAULT_SCHEDULER_LOW);  // v544 → 19
    out.push(DEFAULT_SIGMA_SHIFT_LOW); // v839 → 20
    return out;
}

function _looksV492Current(wv) {
    // v492..v543 current save: frozen length 17, NO sampler_low/scheduler_low.
    // wv[1] = boundary (a number), same disambiguation as the sibling heals.
    // Healed by appending the two v544 defaults ("same as high") → 19 fields, which
    // reproduces the v543 run exactly.
    return Array.isArray(wv)
        && wv.length === LEN_V492_CURRENT
        && typeof wv[1] === "number";
}

function _healV492Current(wv) {
    const out = wv.slice();
    out.push(DEFAULT_SAMPLER_LOW);    // v544 → 18
    out.push(DEFAULT_SCHEDULER_LOW);  // v544 → 19
    out.push(DEFAULT_SIGMA_SHIFT_LOW); // v839 → 20
    return out;
}

function _looksV544Current(wv) {
    // v544..v838 current save: frozen length 19, NO sigma_shift_low.
    // wv[1] = boundary (a number), same disambiguation as the sibling heals.
    // Healed by appending the -1 "same as high" default → 20 fields, which
    // reproduces the v838 run exactly (the LOW expert follows sigma_shift).
    return Array.isArray(wv)
        && wv.length === LEN_V544_CURRENT
        && typeof wv[1] === "number";
}

function _healV544Current(wv) {
    const out = wv.slice();
    out.push(DEFAULT_SIGMA_SHIFT_LOW);  // v839 → 20
    return out;
}

function _findWidget(node, name) {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
}

// v416: is the named INPUT slot connected? (sigma overrides key off this.)
function _inputLinked(node, name) {
    const inp = node.inputs ? node.inputs.find((i) => i && i.name === name) : null;
    return !!(inp && inp.link != null);
}

// ── Enable / disable a widget IN PLACE (no hide, no resize) ──────────────────
// The widget keeps its slot and its height, so the node's geometry is constant.
// We grey + lock it (w.disabled) and prefix its label with INACTIVE_MARK so the
// state is visible even if the renderer does not grey disabled widgets. The base
// label is captured once so the marker never stacks.
// v888: it HIDES now. Frank: "so sieht es unfertig aus." The name and every
// call site are unchanged on purpose -- what "off" LOOKS like is a rendering
// decision, and moving it here keeps the mode logic above untouched. The
// widget stays in node.widgets (values serialise BY INDEX -- #577), so a
// hidden row keeps its slot and its value; only the pixels go. The label is
// still restored on show, so INACTIVE_MARK can never bake into a visible row.
// Returns true when the LAYOUT changed, so the caller refits once per pass
// instead of once per widget.
let _visMoved = false;

function _setDisabled(w, disabled) {
    if (!w) return false;
    if (w._uls_baseLabel === undefined) w._uls_baseLabel = (w.label != null) ? w.label : w.name;
    w.disabled = disabled;
    w.label = w._uls_baseLabel;
    const moved = setHidden(w, disabled);
    if (moved) _visMoved = true;
    return moved;
}

/* Did this pass move any row? Reads AND clears -- one answer per pass. */
function _visChanged(_node) {
    const moved = _visMoved;
    _visMoved = false;
    return moved;
}

// ── Apply the mode state: live path enabled, off path disabled ──────────────
// No widget is shown/hidden and no input is added/removed, so the node never
// resizes. model_low stays as ComfyUI created it (declared optional in
// INPUT_TYPES) — always present — so a connected LOW link is never dropped.
function _applyModeState(node) {
    const pill = _findWidget(node, "dual_moe");
    const dual = pill ? !!pill.value : false;

    for (const name of DUAL_ONLY) _setDisabled(_findWidget(node, name), !dual);
    for (const name of SINGLE_ONLY) _setDisabled(_findWidget(node, name), dual);

    // v416: external sigma schedules override the built-in one. Single keys off a
    // connected `sigmas`; High + Low keys off BOTH `sigmas_high` and `sigmas_low`.
    // When engaged, the schedule generators go inert, and so do the split determiners
    // (boundary in High + Low; start/end/leftover in Single) -- the sigma node owns
    // the whole curve and the sampler ignores those widgets. These only ADD greying
    // on top of the mode state above (never re-enable a mode-disabled widget).
    const sigSingle = _inputLinked(node, "sigmas");
    const sigDual = _inputLinked(node, "sigmas_high") && _inputLinked(node, "sigmas_low");
    // v495: in High + Low, EITHER the segment pair (both high+low) OR the single
    // curve engages the external schedule (the sampler splits the single curve at the
    // boundary, so boundary stays ACTIVE in that wiring -- only sigDual greys it below).
    const sigActive = dual ? (sigDual || sigSingle) : sigSingle;
    for (const name of SIGMA_INERT) _setDisabled(_findWidget(node, name), sigActive);
    if (dual && sigDual) {
        _setDisabled(_findWidget(node, "boundary"), true);   // the two arrays drive the split
    }
    // v544: scheduler_low only bites where the two segments are INDEPENDENT, i.e. in
    // "Wan MoE parity". In "Continuous" both experts share ONE schedule by construction,
    // and an external curve owns the schedule outright -- grey it in both cases so the
    // widget never promises something the run does not honour. (sampler_low stays live:
    // a sampler consumes the schedule, it does not define it.)
    if (dual) {
        const hm = _findWidget(node, "handoff_mode");
        const parity = hm && /moe|rebase/i.test(String(hm.value || ""));
        if (!parity || sigActive) {
            _setDisabled(_findWidget(node, "scheduler_low"), true);
            // v839: same truth for the LOW shift -- in 'Continuous' ONE schedule is
            // built from the HIGH expert, so an own LOW shift cannot bite.
            _setDisabled(_findWidget(node, "sigma_shift_low"), true);
        }
    }
    if (!dual && sigSingle) {                                 // the array owns the slice
        for (const name of SINGLE_ONLY) _setDisabled(_findWidget(node, name), true);
    }

    // v888: ONE refit per pass, and only when a row really appeared or went.
    // Per-widget refitting would re-measure the node a dozen times inside a
    // single mode switch; _setDisabled returns whether the layout moved so
    // this stays a single, cheap decision.
    if (_visChanged(node)) refit(node);
    if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
}

// ── v415: live (mid-render) preview_mode switch ─────────────────────────────
// Track which node ComfyUI is currently executing, so a preview_mode change can be
// pushed to the backend ONLY while THIS sampler node is rendering. On idle/teardown
// the id is null. Wrapped so an unexpected api shape just disables the live push
// (the widget value still applies from the next render).
let _executingNodeId = null;
try {
    api.addEventListener("executing", (e) => {
        const d = e && e.detail;
        _executingNodeId = (d === null || d === undefined) ? null : String(d);
    });
} catch (err) { /* live switch disabled on this frontend */ }

// Tell the backend to switch the RUNNING render's preview. Fire-and-forget: never
// throws into the UI, and a failure just leaves the in-flight preview unchanged.
function _postLivePreviewMode(nodeId, mode) {
    try {
        const body = JSON.stringify({ node_id: String(nodeId), mode });
        const opts = { method: "POST", headers: { "Content-Type": "application/json" }, body };
        const doFetch = (api && typeof api.fetchApi === "function")
            ? api.fetchApi.bind(api)            // adds the /api prefix (registered as an alias)
            : (u, o) => fetch(u, o);            // bare path (also registered) on older frontends
        doFetch("/pls/sampler/preview_mode", opts).catch(() => {});
    } catch (err) { /* no-op */ }
}

// ── v830: the TAE preview decoder says when it is missing ───────────────────
// The two TAE combo entries carry decoration now ("\u25c8 Video · TAE
// (taew2_1) \u00b7 22 MB"); the strip mirrors the backend's _MODE_SIZE_SUFFIX
// exactly -- anchored on a trailing SIZE UNIT, because the mode labels
// themselves contain middots (the v828 rule, sharpened here).
const MODE_SIZE_SUFFIX = /\s\u00b7\s[\d.,]+\s?(KB|MB|GB|TB)$/;

function stripMode(v) {
    let out = String(v == null ? "" : v).replace(MODE_SIZE_SUFFIX, "").trimEnd();
    if (out.startsWith("\u25c8 ")) out = out.slice(2);
    return out;
}

function _samplerFetch(path, payload) {
    const body = JSON.stringify(payload);
    const opts = { method: "POST", headers: { "Content-Type": "application/json" }, body };
    const doFetch = (api && typeof api.fetchApi === "function")
        ? api.fetchApi.bind(api) : (u, o) => fetch(u, o);
    return doFetch(path, opts).then((r) => r.json());
}

// The bubble: the house form (ph_cutout's download toast / ph_basics'
// info bubble), amber, top right. With a pinned source it carries the
// Download button and goes through the one door; without one it informs
// and dismisses itself.
let _taeToastCSS = false;

function _ensureTaeToastCSS() {
    if (_taeToastCSS) return;
    _taeToastCSS = true;
    const st = document.createElement("style");
    st.textContent =
        ".pls-tae-toast { position:fixed; top:16px; right:16px; z-index:10000;"
        + " max-width:360px; background:#1d1d1d; border:1px solid #ff8c00;"
        + " border-left:4px solid #ff8c00; color:#ddd; font:12px sans-serif;"
        + " padding:8px 10px; border-radius:4px;"
        + " box-shadow:0 4px 14px rgba(0,0,0,.5); }"
        + ".pls-tae-toast h4 { margin:0 0 4px; font-size:12px; color:#ff8c00; }"
        + ".pls-tae-toast p { margin:0 0 8px; color:#bbb; line-height:1.35; }"
        + ".pls-tae-toast .row { display:flex; gap:8px; }"
        + ".pls-tae-toast button { flex:1 1 0; padding:4px 6px; cursor:pointer;"
        + " background:#2a2a2a; color:#ddd; border:1px solid #555; }"
        + ".pls-tae-toast button.go { background:#8fff8f; color:#123;"
        + " border-color:#8fff8f; }";
    document.head.appendChild(st);
}

function _taeToast(title, text, goLabel, onGo) {
    _ensureTaeToastCSS();
    const box = document.createElement("div");
    box.className = "pls-tae-toast";
    const h = document.createElement("h4");
    h.textContent = title;
    const p = document.createElement("p");
    p.textContent = text;
    box.appendChild(h);
    box.appendChild(p);
    const close = () => { try { box.remove(); } catch (e) { /* gone */ } };
    if (goLabel && onGo) {
        const row = document.createElement("div");
        row.className = "row";
        const go = document.createElement("button");
        go.className = "go";
        go.textContent = goLabel;
        const no = document.createElement("button");
        no.textContent = "Later";
        no.onclick = close;
        go.onclick = () => {
            go.disabled = true;
            go.textContent = "downloading\u2026";
            onGo(box, p, row);
        };
        row.appendChild(go);
        row.appendChild(no);
        box.appendChild(row);
    } else {
        setTimeout(close, 12000);
    }
    document.body.appendChild(box);
    return { box: box, p: p, close: close };
}

// Once per node instance and per picked decoder; re-picking the same
// missing decoder after a dismissal does not re-warn until reload (a
// combo is easy to wiggle -- state changes speak, wiggles do not).
function _checkTaePreview(node, modeValue) {
    const bare = stripMode(modeValue);
    if (!bare.includes("TAE")) return;
    const tae = bare.includes("lighttaew2_1") ? "lighttaew2_1" : "taew2_1";
    node._plsTaeWarned = node._plsTaeWarned || {};
    if (node._plsTaeWarned[tae]) return;
    _samplerFetch("/pls/sampler/tae_status", { mode: tae }).then((st) => {
        if (!st || !st.ok || st.found) return;
        node._plsTaeWarned[tae] = true;
        const info = "'" + st.file + "' is not in " + st.folder + " -- the "
            + "preview runs as latent2rgb (smooth) until it is. Source: "
            + st.source + ".";
        if (st.downloadable) {
            _taeToast("\u26a0 Preview decoder missing", info,
                      "Download (" + Math.round(st.size / 1048576) + " MB)",
                      (box, p) => {
                _samplerFetch("/pls/sampler/tae_install", { name: tae })
                    .then((r) => {
                        p.textContent = (r && r.ok)
                            ? "Installed \u2713  " + (r.path || "")
                              + "  -- the TAE preview applies from the "
                              + "next render."
                            : "Download failed: "
                              + ((r && r.error) || "unknown error");
                        setTimeout(() => { try { box.remove(); } catch (e) { } },
                                   (r && r.ok) ? 6000 : 15000);
                    })
                    .catch(() => {
                        p.textContent = "Download failed: network error.";
                    });
            });
        } else {
            _taeToast("\u26a0 Preview decoder missing", info, null, null);
        }
    }).catch(() => { /* status route unreachable -- console fallback stands */ });
}

app.registerExtension({
    name: "polyhedron.sampler.moe",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        // Loading a saved graph: heal a corrupted (phantom-preset) array or a
        // pre-v404 array BEFORE LiteGraph applies widgets_values, then apply the
        // mode state (enable/disable the two paths). No input slot is touched.
        const _configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (info) {
            if (info && Array.isArray(info.widgets_values)) {
                if (_looksPhantomPreset(info.widgets_values)) {
                    info.widgets_values = _healPhantomPreset(info.widgets_values);
                } else if (_looksPreV404(info.widgets_values)) {
                    info.widgets_values = _migratePreV404(info.widgets_values);
                } else if (_looksOldCurrent(info.widgets_values)) {
                    // v404..v412 graph (no preview_mode) -> append preview_mode + sigma_shift
                    // + handoff_mode so the saved array matches the 17-field v492 order exactly.
                    info.widgets_values = _healOldCurrent(info.widgets_values);
                } else if (_looksV413Current(info.widgets_values)) {
                    // v413..v490 graph (has preview_mode, no sigma_shift) -> append the
                    // sigma_shift + handoff_mode defaults so the array reaches the 17-field order.
                    info.widgets_values = _healV413Current(info.widgets_values);
                } else if (_looksV491Current(info.widgets_values)) {
                    // v491 graph (has preview_mode+sigma_shift, no handoff_mode) -> append the
                    // handoff_mode default so the array reaches the 17-field v492 order.
                    info.widgets_values = _healV491Current(info.widgets_values);
                } else if (_looksV492Current(info.widgets_values)) {
                    // v492..v543 save (17): append the two v544 "same as high" defaults.
                    info.widgets_values = _healV492Current(info.widgets_values);
                } else if (_looksV544Current(info.widgets_values)) {
                    // v544..v838 save (19): append the -1 "same as high" low-shift default.
                    info.widgets_values = _healV544Current(info.widgets_values);
                }
                // v420: translate a renamed preview_mode value so a saved graph keeps its
                // mode after the Still·/Video· rename. v491/v492: preview_mode is not the
                // LAST field (sigma_shift then handoff_mode trail it) — key off its fixed index.
                const wv = info.widgets_values;
                if (Array.isArray(wv) && wv.length > PREVIEW_MODE_IDX) {
                    const cur = wv[PREVIEW_MODE_IDX];
                    if (Object.prototype.hasOwnProperty.call(PREVIEW_MODE_RENAMES, cur)) {
                        wv[PREVIEW_MODE_IDX] = PREVIEW_MODE_RENAMES[cur];
                    }
                    // v830: the two TAE entries carry decoration now; snap a
                    // stored bare value onto the entry the list offers, by
                    // comparing both sides stripped (the v828/v829 form).
                    try {
                        const w = _findWidget(this, "preview_mode");
                        const opts = w && w.options && w.options.values;
                        const cur2 = wv[PREVIEW_MODE_IDX];
                        if (Array.isArray(opts) && !opts.includes(cur2)) {
                            const hit = opts.find(
                                (o) => stripMode(o) === stripMode(cur2));
                            if (hit !== undefined) wv[PREVIEW_MODE_IDX] = hit;
                        }
                    } catch (err) { /* never break configure */ }
                }
                // v493: translate a renamed handoff_mode value so a v492 save keeps its mode
                // after the "Continue · leftover"/"Rebase · x0 + renoise" -> "Continuous"/"Wan
                // MoE parity" rename (keys off the fixed handoff_mode slot).
                if (Array.isArray(wv) && wv.length > HANDOFF_MODE_IDX) {
                    const curH = wv[HANDOFF_MODE_IDX];
                    if (Object.prototype.hasOwnProperty.call(HANDOFF_MODE_RENAMES, curH)) {
                        wv[HANDOFF_MODE_IDX] = HANDOFF_MODE_RENAMES[curH];
                    }
                }
                // v494: heals/renames above all operate on the CANONICAL (ORDER_V404) array.
                // base configure then applies widgets_values POSITIONALLY; since the widgets
                // are shown in DISPLAY order, remap the canonical array to DISPLAY so each value
                // lands on the right widget. Only when this node's display reorder took.
                if (this._plsDisplayReordered) {
                    info.widgets_values = _canonToDisplay(info.widgets_values);
                }
            }
            const r = _configure ? _configure.apply(this, arguments) : undefined;
            _applyModeState(this);
            return r;
        };

        // On creation: label the Handoff + pill, wrap the pill callback, and apply
        // the mode state. model_low (optional in INPUT_TYPES) is already present
        // and is left untouched, so it persists in both modes.
        const _onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = _onCreated ? _onCreated.apply(this, arguments) : undefined;
            const self = this;

            const boundary = _findWidget(self, "boundary");
            if (boundary) boundary.label = "🛈 Handoff";

            const pill = _findWidget(self, "dual_moe");
            if (pill) pill.label = "🛈 Mode";

            // v545: EVERY widget whose VALUE feeds _applyModeState must re-run it on
            // change. Until v544 only dual_moe did, so switching handoff_mode left the
            // state stale -- scheduler_low stayed greyed in "Wan MoE parity" even though
            // the run would have honoured it. One list, so the next such dependency
            // cannot be forgotten.
            for (const name of STATE_WIDGETS) {
                const w = _findWidget(self, name);
                if (!w) continue;
                const _cb = w.callback;
                w.callback = function () {
                    const rr = _cb ? _cb.apply(this, arguments) : undefined;
                    _applyModeState(self);
                    return rr;
                };
            }

            // v415: live preview switch. Changing preview_mode always updates the
            // serialised value (applies from the next render); if THIS node is mid-
            // render, also push it so the in-flight preview switches on the next step.
            const previewMode = _findWidget(self, "preview_mode");
            if (previewMode) {
                const _pcb = previewMode.callback;
                previewMode.callback = function () {
                    const rr = _pcb ? _pcb.apply(this, arguments) : undefined;
                    try {
                        if (_executingNodeId !== null && String(self.id) === _executingNodeId) {
                            _postLivePreviewMode(self.id, previewMode.value);
                        }
                    } catch (err) { /* no-op: never break the widget callback */ }
                    try {                       // v830: warn if the decoder is absent
                        _checkTaePreview(self, previewMode.value);
                    } catch (err) { /* no-op */ }
                    return rr;
                };
                // v830: one check after the pour, so a saved graph sitting on
                // a TAE mode warns on load, not only on the next click.
                setTimeout(() => {
                    try { _checkTaePreview(self, previewMode.value); } catch (err) { }
                }, 0);
            }

            // v494: reorder the VISIBLE widgets into DISPLAY order (handoff_mode under
            // Handoff, sigma_shift under scheduler, preview_mode last). The SERIALISED order
            // is unchanged — onSerialize maps back to canonical, configure maps in. Fail-safe:
            // if the widget set is not exactly the 17 canonical widgets this is a no-op.
            self._plsDisplayReordered = _reorderWidgetsToDisplay(self);

            _applyModeState(self);
            return r;
        };
        const _onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = _onConfigure ? _onConfigure.apply(this, arguments) : undefined;
            _applyModeState(this);
            return r;
        };

        // v494: the widgets are shown in DISPLAY order, so base serialize emits widgets_values
        // in DISPLAY order. Remap it back to canonical ORDER_V404 here so the saved file — and
        // every migration/recogniser/rename path, all of which assume the canonical order —
        // stays in the one true serialised order. Fail-safe + guarded (never breaks a save).
        const _onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (o) {
            const r = _onSerialize ? _onSerialize.apply(this, arguments) : undefined;
            try {
                if (this._plsDisplayReordered && o && Array.isArray(o.widgets_values)) {
                    o.widgets_values = _displayToCanon(o.widgets_values);
                }
            } catch (err) { /* never break serialize */ }
            return r;
        };

        // v416: a sigma input connecting/disconnecting flips the schedule widgets
        // inert/active -> re-apply on any link change (guarded; cheap).
        const _onConn = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const r = _onConn ? _onConn.apply(this, arguments) : undefined;
            try { _applyModeState(this); } catch (e) { /* never break link handling */ }
            return r;
        };
    },
});
