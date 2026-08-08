/*
 * ph_power_upscale.js — mode-aware widget state for ⬡ Polyhedron Power Upscale.
 *
 * ONE node carries both upscaling architectures. The `dual_moe` BOOLEAN (labelled
 * "🛈 Mode", mirroring the sampler) selects Single vs. High + Low. The four
 * stage-L dials — upscale_by_low / denoise_low / steps_low / cfg_low — are ONLY
 * consumed by the High + Low stage chain (uls_tile_math.plan_stages reads none
 * of them on the Single path), so in Single they are DISABLED (greyed + locked)
 * and prefixed with "⊘". There is NO Single-only set: Single is a strict subset
 * of the stage chain, so greying runs in one direction only.
 *
 * STABLE GEOMETRY (the sampler's v405 rule, inherited): widgets stay laid out;
 * only their enabled/disabled state and label prefix change. No input slot is
 * touched — model_low / upscale_model_low are optional in INPUT_TYPES, always
 * present, so a connected LOW link is never dropped by toggling the mode.
 *
 * SERIALISATION: nothing here serialises. `disabled` and `label` are derived
 * from the dual_moe value on create/configure, so saved graphs need no heal or
 * migration (the node shipped in v514 with this exact widget order).
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// v531 doctrine: every JS file caches INDIVIDUALLY in Firefox -> self-proving banner.
// (v548: this file never had one - which is exactly why the v547 silent no-op below
// stayed invisible for a whole live round.)
console.info("[PLS] ph_power_upscale.js v597 loaded");

const NODE_TYPE = "ULSPowerUpscale";
// live only in High + Low — the stage-L dials (plan_stages ignores them in Single)
// v546: + the per-stage sampler/scheduler. They need NO further gate: this node is a
// STAGE CHAIN, not a boundary split — stage L is an INDEPENDENT run with its own
// schedule. (Contrast the Sampler, where scheduler_low only bites in "Wan MoE parity".)
const DUAL_ONLY = ["upscale_by_low", "denoise_low", "steps_low", "cfg_low",
                   "sampler_low", "scheduler_low"];

// ── v546: canonical (SERIALISED) widget order = INPUT_TYPES order. The two new fields
//    are appended at the END, so every existing index keeps its slot in old saves. ──
const ORDER_CANON = [
    "dual_moe", "upscale_by", "denoise", "steps", "cfg",
    "upscale_by_low", "denoise_low", "steps_low", "cfg_low",
    "seed", "control_after_generate", "sampler_name", "scheduler",
    "tile_size", "tile_overlap", "sigma_shift",
    "sampler_low", "scheduler_low",
    "result_preview",                     // v549 (appended - every old index keeps its slot)
    "process_preview",                    // v550 (appended)
    "mute_staging_logs",                  // v553 (appended)
    "resize_method",                      // v555 (appended)
    "per_batch",                          // v560 (appended)
    "vae_tiling",                         // v562 (appended)
    "pixel_stage",                        // v564 (appended)
    "final_upscale_by",                   // v582 (appended - every old index keeps its slot)
    "sigma_shift_low",                    // v851 (appended - every old index keeps its slot)
];
// What the USER sees: every LOW twin directly under its HIGH partner (the Sampler's v494
// pattern). Same names, permuted; everything below is index-based (widgets_values is
// positional: value[i] === node.widgets[i].value), never by widget .name.
const DISPLAY_ORDER = [
    "dual_moe",
    "upscale_by", "upscale_by_low",
    "final_upscale_by",   // v589: the third size dial joins its family (the law
                          // of proximity). Legal now - NOT because the risk was
                          // waved through, but because configure stopped
                          // GUESSING the save order: marked saves are proven
                          // canon (v588), unmarked saves are TYPE-fingerprinted
                          // (see _saveOrderOf) and the v587 display order loads
                          // through its own legacy map. Rollback anchor: v588.
    "denoise", "denoise_low",
    "steps", "steps_low",
    "cfg", "cfg_low",
    "seed", "control_after_generate",
    "sampler_name", "sampler_low",
    "scheduler", "scheduler_low",
    "tile_size", "tile_overlap",
    "sigma_shift", "sigma_shift_low",   // v852: the LAST twin joins its partner.
                          // v851 parked it at the end because DISPLAY_ORDER is
                          // append-only; this is the separate, measured cut that
                          // was announced there. Legal by the SAME ceremony v589
                          // used: the order it leaves behind is frozen as
                          // DISPLAY_LEGACY_V851, the fingerprint learned to name
                          // it, and the guard was rewritten in the same cut.
                          // Rollback anchor: v851.
    "result_preview", "process_preview", "mute_staging_logs", "resize_method",
    "per_batch", "vae_tiling", "pixel_stage",
];
// ── v585 LAW (measured 2026-07-13, the hard way): the live frontend
//    serialises widgets_values in the WIDGET (display) order, not in
//    ORDER_CANON. The v584 cut moved final_upscale_by into the display
//    middle and every pre-v584 save loaded shifted by one slot from
//    position 4 on - the seed landed in cfg_low. Therefore DISPLAY_ORDER
//    is APPEND-ONLY, exactly like ORDER_CANON: new widgets join at the
//    END of both lists. Re-sorting the display is only legal after the
//    save path is normalised through the canon mapping - a measured
//    project of its own, not a side effect. ──────────────────────────────
// ── v547: INPUT sockets follow the same law of proximity. ComfyUI builds the sockets as
//    {...required, ...optional}, so an OPTIONAL input (image/video/model_low/
//    upscale_model_low) can never rise above the required ones through INPUT_TYPES —
//    only a display permutation can. LGraphNode.configure() overwrites node.inputs from
//    the SAVE (element-wise), so old graphs keep their links; we permute AFTER that and
//    repair every incoming link's target_slot to the new index. ──
const INPUT_DISPLAY_ORDER = [
    "image", "video",                       // what you are upscaling -> first
    "model", "model_low",                   // the expert pair, together
    "positive", "negative", "vae",
    "upscale_model", "upscale_model_low",   // the ESRGAN pair, together
];

function _reorderInputsToDisplay(node) {
    const ins = node && node.inputs;
    if (!Array.isArray(ins) || !ins.length) return false;
    // v548: stable PARTITION instead of a rebuild-from-list. A wired widget (e.g. a
    // 'seed' fed by the Seed node) occupies an entry in node.inputs too; the v547
    // strict length-9 guard turned the whole reorder into a silent no-op on such
    // nodes (measured live, 2026-07-11) - and the rebuild would have DROPPED that
    // entry, orphaning its link. Now: the nine named sockets take the display
    // order, every other input keeps its relative order AFTER them (wired-widget
    // inputs render at their widget row anyway; only target_slot cares).
    const rank = new Map(INPUT_DISPLAY_ORDER.map((n, i) => [n, i]));
    const known = [];
    const rest = [];
    for (const inp of ins) {
        if (inp && rank.has(inp.name)) known.push(inp); else rest.push(inp);
    }
    if (known.length !== INPUT_DISPLAY_ORDER.length) {
        // fail-LOUD (the v540 lesson): a silent skip cost the v547 live round.
        console.warn("[PLS v548] PU socket reorder skipped: expected "
            + JSON.stringify(INPUT_DISPLAY_ORDER) + ", found "
            + JSON.stringify(ins.map((i) => (i ? i.name : null))));
        return false;
    }
    known.sort((a, b) => rank.get(a.name) - rank.get(b.name));
    const next = known.concat(rest);
    let changed = false;
    for (let i = 0; i < ins.length; i++) {
        if (ins[i] !== next[i]) { changed = true; break; }
    }
    if (!changed) return true;   // already in display order -> idempotent no-op
    node.inputs = next;
    // Repair the link table: LLink.target_slot must match the NEW slot index, or links
    // render (and disconnect) against the wrong socket. Full array on purpose:
    // sockets AND wired-widget inputs.
    const links = node.graph && node.graph.links;
    if (links) {
        for (let i = 0; i < node.inputs.length; i++) {
            const lid = node.inputs[i] ? node.inputs[i].link : null;
            if (lid == null) continue;
            const l = (typeof links.get === "function") ? links.get(lid) : links[lid];
            if (l) l.target_slot = i;
        }
    }
    if (typeof node.setDirtyCanvas === "function") node.setDirtyCanvas(true, true);
    return true;
}

const CANON_IDX_AT_DISPLAY = DISPLAY_ORDER.map((n) => ORDER_CANON.indexOf(n));
const DISPLAY_POS_OF_CANON = ORDER_CANON.map((n) => DISPLAY_ORDER.indexOf(n));
const LEN_PRE_V546 = 16;               // v514..v545 save: canonical, lacks the two new fields
const SAME_AS_HIGH = "same as high";   // MUST match the node's INPUT_TYPES default

function _reorderWidgetsToDisplay(node) {
    const ws = node && node.widgets;
    if (!Array.isArray(ws) || ws.length !== ORDER_CANON.length) return false;
    if (CANON_IDX_AT_DISPLAY.some((i) => i < 0)) return false;
    node.widgets = CANON_IDX_AT_DISPLAY.map((ci) => ws[ci]);
    return true;
}
// ═══ v563: THE SAFETY NET ═══════════════════════════════════════════════════════
// The length-exact heal cascade below is precise but brittle: it only fires on the
// EXACT previous length. A stale browser cache (old JS, new backend) therefore left
// a 23-value save in a 24-widget node - vae_tiling never got a value, ComfyUI
// rejected the prompt ("Value not in list: ''"), and the node could not repair
// itself. Two nets, in this order:
//   1. _padToCanon  - ANY short array is topped up with the canonical defaults.
//   2. _sanitize    - after configure, every widget is checked against its OWN
//                     option list; an invalid or empty value falls back to its
//                     default. Generic, so future widgets are covered for free.
// v586: COMPLETE mirror of the python INPUT_TYPES defaults. The v584/v585
// field runs measured the gap: as a PAD table (v563) slots 18-25 sufficed,
// but as the heal's default SOURCE (v584) the table must cover every slot -
// a conserved string landed on tile_size (slot 13), found no source, and
// stayed as NaN all the way to the validator. Combos self-heal through their
// own value list (list[0]); their entries here are pad/documentation.
const CANON_DEFAULTS = {
    0: false,            // dual_moe
    1: 1.10,             // upscale_by
    2: 0.19,             // denoise
    3: 3,                // steps
    4: 1.6,              // cfg
    5: 1.30,             // upscale_by_low
    6: 0.25,             // denoise_low
    7: 5,                // steps_low
    8: 1.9,              // cfg_low
    9: 0,                // seed
    10: "randomize",     // control_after_generate (frontend widget)
    13: 1024,            // tile_size
    14: 64,              // tile_overlap
    15: 0.0,             // sigma_shift
    18: true,            // result_preview
    19: "Off",           // process_preview
    20: true,            // mute_staging_logs
    21: "lanczos (cpu)", // resize_method
    22: 8,               // per_batch
    23: "Off",           // vae_tiling
    24: "model + fit",   // pixel_stage
    25: 1.0,             // final_upscale_by (v582: 1.0 = the old law, bit for bit)
    26: -1.0,            // sigma_shift_low (v851: -1 = "same as high" = the old law)
};
// v584: the numeric ranges, MIRRORED from INPUT_TYPES for the same reason
// CANON_DEFAULTS exists - a live ComfyUI widget is not guaranteed to expose
// options.min/max/default, and the heal must not depend on what the frontend
// happens to attach. One entry per numeric canon widget the net guards.
const CANON_RANGES = {
    1: [1.0, 8.0],                       // upscale_by
    2: [0.0, 1.0],                       // denoise
    3: [1, 10000],                       // steps
    4: [0.0, 100.0],                     // cfg
    5: [1.0, 8.0],                       // upscale_by_low
    6: [0.0, 1.0],                       // denoise_low
    7: [1, 10000],                       // steps_low
    8: [0.0, 100.0],                     // cfg_low
    9: [0, 0xffffffffffffffff],          // seed
    13: [64, 4096],                      // tile_size
    14: [0, 1024],                       // tile_overlap
    15: [0.0, 20.0],                     // sigma_shift
    22: [1, 256],                        // per_batch
    25: [0.25, 8.0],                     // final_upscale_by
    26: [-1.0, 20.0],                    // sigma_shift_low (v851: -1 is the sentinel)
};                                       // keep in step with the python widget
function _padToCanon(arr) {
    if (!Array.isArray(arr) || arr.length >= ORDER_CANON.length) return arr;
    const out = arr.slice();
    while (out.length < ORDER_CANON.length) {
        const d = CANON_DEFAULTS[out.length];
        out.push(d === undefined ? null : d);
    }
    console.info(`[PLS] PU: padded a short save (${arr.length} -> ${out.length} values)`);
    return out;
}

/** Every widget must hold a value its own options allow. */
function _sanitize(node) {
    let fixed = 0;
    for (const w of node.widgets || []) {
        const opts = w.options || {};
        // v584: source CHAINS. v583 read the default from options.default and
        // the range from options.min/max - fields the harness mocks carried
        // but a live ComfyUI widget does not guarantee (measured: the healed
        // slot stayed 0.00 in the field). Our own tables are always there:
        // options first, canon mirror second.
        const ci = ORDER_CANON.indexOf(w.name);
        const dflt = (opts.default !== undefined) ? opts.default
                   : (ci >= 0 ? CANON_DEFAULTS[ci] : undefined);
        const rng = (typeof opts.min === "number" || typeof opts.max === "number")
                  ? [opts.min, opts.max]
                  : (ci >= 0 && CANON_RANGES[ci]) ? CANON_RANGES[ci]
                  : [undefined, undefined];
        const list = Array.isArray(opts.values) ? opts.values : null;
        if (list && list.length) {
            if (w.value === undefined || w.value === null || w.value === "" ||
                !list.includes(w.value)) {
                w.value = (dflt !== undefined && list.includes(dflt))
                    ? dflt : list[0];
                fixed++;
            }
        } else if (w.value === undefined || w.value === null ||
                   (typeof w.value === "number" && Number.isNaN(w.value))) {
            if (dflt !== undefined) { w.value = dflt; fixed++; }
        } else if (typeof dflt === "number") {
            // v583: NUMBER-widget hygiene, measured in the field (2026-07-13):
            // a stale-cache first run conserved '' into the autosave's slot 25
            // (final_upscale_by). The widget RENDERED it as 0.00 (Number('')
            // is 0) while the prompt still carried the raw '' - rejected by
            // ComfyUI's validator before any backend code could run. And a
            // save that stores the cast artefact 0 as a NUMBER passes every
            // type check yet sits below the widget's own min - the error
            // merely changes shape. The rule, consistent with the combo
            // branch above (out-of-list -> default): a non-number is rescued
            // if parseable, else default; a number outside [min, max] ->
            // default, never a silent clamp (0 was never a dial position).
            let v = w.value;
            if (typeof v !== "number") {
                const n = Number(v);
                v = (String(v).trim() !== "" && Number.isFinite(n))
                    ? n : dflt;
            }
            if (Number.isNaN(v) ||
                (typeof rng[0] === "number" && v < rng[0]) ||
                (typeof rng[1] === "number" && v > rng[1])) {
                v = dflt;
            }
            if (v !== w.value) { w.value = v; fixed++; }
        }
    }
    if (fixed) {
        console.info(`[PLS] PU: repaired ${fixed} invalid widget value(s) `
                     + `- the node can run again`);
        node.setDirtyCanvas(true, true);
    }
    return fixed;
}

// ── v588: THE CANON MARKER ────────────────────────────────────────────────
// The v546 normalisation (configure maps canon->display, onSerialize maps
// back) has existed for 40+ cuts - but a save never PROVED which order it
// carries. If any frontend path serialises without our onSerialize hook,
// configure's assumption is a guess, and v584 measured what a wrong guess
// costs (the seed landed in cfg_low). A save that went through OUR
// onSerialize now carries this property; configure treats marked saves as
// proven-canonical and says so, unmarked saves load exactly as before (the
// status quo, zero risk) and say that too. Frank's next save->reload cycle
// is the field proof - and only AFTER that proof does a display re-sort
// (the v585 law) become a legal one-liner.
const CANON_MARKER = "pls_widgets_canon";
// v589: the display order of v514..v588, frozen. An unmarked save that a
// stray frontend path wrote in DISPLAY order (the v584 phantom) carries THIS
// order - and loads correctly through the map below instead of shifting.
const DISPLAY_LEGACY_V587 = [
    "dual_moe", "upscale_by", "upscale_by_low",
    "denoise", "denoise_low", "steps", "steps_low", "cfg", "cfg_low",
    "seed", "control_after_generate",
    "sampler_name", "sampler_low", "scheduler", "scheduler_low",
    "tile_size", "tile_overlap", "sigma_shift",
    "result_preview", "process_preview", "mute_staging_logs",
    "resize_method", "per_batch", "vae_tiling", "pixel_stage",
    "final_upscale_by",
];
// v851: the table above is FROZEN history and stays VERBATIM - but the canon
// kept growing, and _padToCanon runs BEFORE this map, so a padded legacy save is
// now LONGER than the historic order and still passes the length gate. A name the
// historic table does not know is a widget appended after v588; both lists are
// APPEND-ONLY, so such a name sits at the SAME index in the canon and in the save,
// and its value is already where it belongs. Read it from its canon slot instead
// of asking a table that predates it - exact, and true for every future append.
// v852: the display order of v589..v851, frozen the moment it was left. Same
// role DISPLAY_LEGACY_V587 plays for the pre-v589 era: an unmarked save that a
// stray frontend path wrote in DISPLAY order carries THIS order and loads
// through it instead of shifting. Never edit either table - they are history.
const DISPLAY_LEGACY_V851 = [
    "dual_moe", "upscale_by", "upscale_by_low", "final_upscale_by",
    "denoise", "denoise_low", "steps", "steps_low", "cfg", "cfg_low",
    "seed", "control_after_generate",
    "sampler_name", "sampler_low", "scheduler", "scheduler_low",
    "tile_size", "tile_overlap", "sigma_shift",
    "result_preview", "process_preview", "mute_staging_logs",
    "resize_method", "per_batch", "vae_tiling", "pixel_stage",
    "sigma_shift_low",
];
// v852: ONE mapper for every historic order. A name the table does not know is
// a widget appended after that table was frozen; both orders are append-only,
// so it sits at the SAME index in canon and in the save (the v851 rule, now
// shared instead of duplicated).
function _tableToCanon(arr, table) {
    if (!Array.isArray(arr) || arr.length !== ORDER_CANON.length) return arr;
    return ORDER_CANON.map((name, ci) => {
        const li = table.indexOf(name);
        return arr[li < 0 ? ci : li];
    });
}
function _legacyDisplayToCanon(arr) {
    return _tableToCanon(arr, DISPLAY_LEGACY_V587);
}
// v589: a save STATES its order or gets read by its TYPES. Two witnesses per
// verdict, so ONE corrupt slot (the '' saga) cannot flip it. No majority ->
// "unknown" -> status quo (treated as canon, the pre-v589 behaviour), loudly.
//
// v852: there are now THREE historic layouts to tell apart, and slots 13/14 +
// 16/17 alone can no longer do it - the v851 display order carries the SAME
// string/number shape there as the pre-v589 one. Measured over the four
// orders, these witnesses separate them (S=string, N=number, B=bool):
//
//   slot            10      13/14    16/17    18      25
//   canon            S       N N      S S      B       N
//   legacy v587      S       S S      N N      B       N
//   legacy v851      N       S S      N N      N       S
//   current v852     N       S S      N N      N       S
//
// So 13/14 + 16/17 answer "canon or a display order", and 10 + 25 answer
// "which display era". The last two rows are identical here on purpose: that
// pair differs only in slots 19..24, which is what _displayEra() then reads.
function _saveOrderOf(vals, marked) {
    if (marked) return "canon";
    if (!Array.isArray(vals) || vals.length < 24) return "canon";  // pre-Vue era: the hook ran
    const num = (x) => typeof x === "number";
    const str = (x) => typeof x === "string";
    const bool = (x) => typeof x === "boolean";
    const canonW  = (num(vals[13]) || num(vals[14])) && (str(vals[16]) || str(vals[17]));
    const legacyW = (str(vals[13]) || str(vals[14])) && (num(vals[16]) || num(vals[17]));
    if (canonW && !legacyW) return "canon";
    if (!(legacyW && !canonW)) return "unknown";
    // it is SOME display order - now which era? TWO witnesses per side again,
    // never a single slot: slot 10 (control_after_generate vs seed) pairs with
    // 18 (result_preview vs a number), and 25 (final_upscale_by vs a string
    // dial) pairs with 19.
    const oldEraW = (str(vals[10]) || bool(vals[18])) && (str(vals[19]) || num(vals[25]));
    const newEraW = (num(vals[10]) || num(vals[18])) && (str(vals[25]) || !str(vals[19]));
    if (oldEraW && !newEraW) return "legacy-display";
    if (!(newEraW && !oldEraW)) return "unknown";
    return _displayEra(vals);
}

// v852: v851 and the current display run apart from slot 19 on, because
// sigma_shift_low left the tail and pushed everything after sigma_shift down
// by one. Two witnesses again: 19 and 24.
//
//   slot            19                  21                 24            26
//   legacy v851      B result_preview     B mute_staging     S vae_tiling  N shift_low
//   current v852     N sigma_shift_low    S process_preview  N per_batch   S pixel_stage
//
// Paired as (19|21) and (24|26), so ONE corrupt slot cannot even degrade the
// verdict to "unknown" -- the '' saga taught that a single junk value must be
// survivable, not merely non-fatal.
function _displayEra(vals) {
    const num = (x) => typeof x === "number";
    const bool = (x) => typeof x === "boolean";
    const str = (x) => typeof x === "string";
    const v851W = (bool(vals[19]) || bool(vals[21])) && (str(vals[24]) || num(vals[26]));
    const currW = (num(vals[19]) || str(vals[21])) && (num(vals[24]) || str(vals[26]));
    if (v851W && !currW) return "legacy-display-851";
    if (currW && !v851W) return "display-current";
    return "unknown";
}

function _canonToDisplay(arr) {
    if (!Array.isArray(arr) || arr.length !== ORDER_CANON.length) return arr;
    return CANON_IDX_AT_DISPLAY.map((ci) => arr[ci]);
}
function _displayToCanon(arr) {
    if (!Array.isArray(arr) || arr.length !== ORDER_CANON.length) return arr;
    return DISPLAY_POS_OF_CANON.map((di) => arr[di]);
}
// A pre-v546 save is canonical and 16 long -> append the two "same as high" defaults,
// which reproduces the v545 run exactly.
function _healPreV546(wv) {
    if (!Array.isArray(wv) || wv.length !== LEN_PRE_V546) return wv;
    const out = wv.slice();
    out.push(SAME_AS_HIGH);   // sampler_low   -> 17
    out.push(SAME_AS_HIGH);   // scheduler_low -> 18
    return out;
}
const LEN_PRE_V549 = 18;   // a v546..v548 save lacks result_preview
function _healPreV549(wv) {
    if (!Array.isArray(wv) || wv.length !== LEN_PRE_V549) return wv;
    const out = wv.slice();
    out.push(true);   // result_preview -> 19 (PURE ui - the outputs are unchanged)
    return out;
}
const LEN_PRE_V550 = 19;   // a v549 save lacks process_preview
function _healPreV550(wv) {
    if (!Array.isArray(wv) || wv.length !== LEN_PRE_V550) return wv;
    const out = wv.slice();
    out.push("Off");   // process_preview -> 20 (Off = the probe is never built)
    return out;
}
const LEN_PRE_V553 = 20;   // a v550..v552 save lacks mute_staging_logs
function _healPreV553(wv) {
    if (!Array.isArray(wv) || wv.length !== LEN_PRE_V553) return wv;
    const out = wv.slice();
    out.push(true);   // mute_staging_logs -> 21 (pure logging - run untouched)
    return out;
}
const LEN_PRE_V555 = 21;   // a v553/v554 save lacks resize_method
function _healPreV555(wv) {
    if (!Array.isArray(wv) || wv.length !== LEN_PRE_V555) return wv;
    const out = wv.slice();
    out.push("lanczos (cpu)");   // resize_method -> 22 (the historic path: byte-identical)
    return out;
}
const LEN_PRE_V560 = 22;   // a v555..v559 save lacks per_batch
function _healPreV560(wv) {
    if (!Array.isArray(wv) || wv.length !== LEN_PRE_V560) return wv;
    const out = wv.slice();
    out.push(8);   // per_batch -> 23 (the OOM guard; 0 would be the old behaviour)
    return out;
}
const LEN_PRE_V562 = 23;   // a v560/v561 save lacks vae_tiling
function _healPreV562(wv) {
    if (!Array.isArray(wv) || wv.length !== LEN_PRE_V562) return wv;
    const out = wv.slice();
    out.push("Off");   // vae_tiling -> 24 (the historic path: byte-identical)
    return out;
}
const LEN_PRE_V564 = 24;   // a v562/v563 save lacks pixel_stage
function _healPreV564(wv) {
    if (!Array.isArray(wv) || wv.length !== LEN_PRE_V564) return wv;
    const out = wv.slice();
    out.push("model + fit");   // pixel_stage -> 24 (the historic behaviour)
    return out;
}

// Leading marker on the DISABLED path's field labels (renderer-independent cue;
// the w.disabled greying reinforces it where the renderer honours disabled).
const INACTIVE_MARK = "⊘ ";

function _findWidget(node, name) {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
}

// ── Enable / disable a widget IN PLACE (no hide, no resize) ──────────────────
// The widget keeps its slot and its height, so the node's geometry is constant.
// We grey + lock it (w.disabled) and prefix its label with INACTIVE_MARK so the
// state is visible even if the renderer does not grey disabled widgets. The base
// label is captured once so the marker never stacks.
function _setDisabled(w, disabled) {
    if (!w) return;
    if (w._uls_baseLabel === undefined) w._uls_baseLabel = (w.label != null) ? w.label : w.name;
    w.disabled = disabled;
    w.label = disabled ? INACTIVE_MARK + w._uls_baseLabel : w._uls_baseLabel;
}

// ── Apply the mode state: stage-L dials live only in High + Low ──────────────
// No widget is shown/hidden and no input is added/removed, so the node never
// resizes. model_low / upscale_model_low stay as ComfyUI created them.
function _applyModeState(node) {
    const pill = _findWidget(node, "dual_moe");
    const dual = pill ? !!pill.value : false;

    for (const name of DUAL_ONLY) _setDisabled(_findWidget(node, name), !dual);

    if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
}

// ═══ v549: result viewer ════════════════════════════════════════════════════════
// The finished frames, INSIDE the node - the ph_save v532-v536 mechanics as a
// lean second instance (deliberately NOT a refactor of ph_save.js, live-OK at
// v536): box-fit (v535), stored-height computeSize + height-only setSize
// (v518/v531), one shared control bar with play / scrub / i-N / loop (v533),
// loop default OFF = play once then stop (v534), hideOnZoom:false (v542).
const PV_MIN_H = 96, PV_MAX_H = 768, NODE_MIN_W = 300;
// v591 (Frank's ask): preview PARITY with the render sampler. The sampler rides
// ComfyUI's native imgs + setSizeForImage, which sizes the NODE to the image -
// on a 768x768 run that lands ~512 px. This viewer is a DOM widget and has
// always inherited whatever width the node happened to carry (~390 px on
// Frank's canvas), so the same frames showed up smaller here than there. The
// first payload of a run now WIDENS a too-narrow node to this. It never SHRINKS
// one (the v531 law) - a node pulled wider stays wider. The aspect ratio still
// comes from the frames themselves, so a 16:9 render lands 512x288, not a
// squashed square: the target is a WIDTH, not a box.
const PV_TARGET_W = 512;
let _pvFirstLogged = false;   // v552: prove the result chain once per session
let _pvWidenLogged = false;   // v591: prove the widen fired, once per session

// v592 (Frank, mid-run screenshot): v591 hung the widen on the RESULT payload -
// which arrives when the work is over. During the only minutes anyone actually
// watches the node, it stayed at whatever width it had. The widen is now its own
// function and BOTH views call it: the process pane on its first probe (seconds
// in) and the result viewer on its payload (belt and braces, for a node created
// after the run began). Grows to the target, never shrinks below what the user
// pulled - the v531 law is in the max(), not in a comment.
function _widenToTarget(node, target) {
    const want = Math.max(NODE_MIN_W, node.size[0], target || PV_TARGET_W);
    if (want <= node.size[0]) return false;
    node.setSize([want, node.computeSize()[1]]);
    if (!_pvWidenLogged) {   // measure > believe
        _pvWidenLogged = true;
        console.info("[PLS] PU: node widened to", want,
                     "px - sampler preview parity (v592)");
    }
    return true;
}

function _viewURL(item) {
    const p = new URLSearchParams({
        filename: item.filename, subfolder: item.subfolder || "",
        type: item.type || "temp",
    });
    return `/view?${p.toString()}&r=${Date.now()}`;   // cache-bust each run
}

function _pvStop(node) {
    if (node._pvTimer) { clearInterval(node._pvTimer); node._pvTimer = null; }
    if (node._pvBtn) node._pvBtn.textContent = "▶";
}
function _pvBlit(node, im) {
    // object-fit:contain, by hand, onto a canvas sized to its OWN box.
    const cv = node._pvImg;
    if (!cv || !im || !im.complete || !im.naturalWidth) return;
    const r = Math.min(2, window.devicePixelRatio || 1);   // cap the backing store
    const bw = Math.max(1, Math.round(cv.clientWidth * r));
    const bh = Math.max(1, Math.round(cv.clientHeight * r));
    if (cv.clientWidth < 1 || cv.clientHeight < 1) return;  // still laid out
    if (cv.width !== bw || cv.height !== bh) { cv.width = bw; cv.height = bh; }
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, bw, bh);
    const s = Math.min(bw / im.naturalWidth, bh / im.naturalHeight);
    const dw = im.naturalWidth * s, dh = im.naturalHeight * s;
    ctx.drawImage(im, (bw - dw) / 2, (bh - dh) / 2, dw, dh);
}

function _pvShow(node, i) {
    const fr = node._pvFrames || [];
    if (!fr.length) return;
    node._pvIdx = ((i % fr.length) + fr.length) % fr.length;
    _pvBlit(node, fr[node._pvIdx]);        // v579: blit, never re-src
    node._pvCount.textContent = (node._pvIdx + 1) + " / " + fr.length;
    node._pvScrub.value = String(node._pvIdx);
}
function _pvPlay(node) {
    const fr = node._pvFrames || [];
    if (fr.length < 2) return;
    _pvStop(node);
    node._pvBtn.textContent = "⏸";
    const dt = 1000 / Math.min(30, Math.max(1, node._pvFps || 12));
    node._pvTimer = setInterval(() => {
        const next = node._pvIdx + 1;
        if (next >= fr.length && !node._pvLoop) {   // v534: play once, then stop
            _pvShow(node, fr.length - 1); _pvStop(node); return;
        }
        _pvShow(node, next);
    }, dt);
}

function _buildViewer(node) {
    const box = document.createElement("div");
    box.style.cssText = "position:relative; width:100%; height:100%; display:none;" +
        " background:#000; border-radius:6px; overflow:hidden;";
    // v579: a CANVAS, not an <img>. The old viewer did `img.src = frame.src`
    // on every animation tick - and re-assigning src makes the browser TEAR
    // DOWN and RE-DECODE the image even though the frames are already decoded
    // Image objects. Between teardown and paint the <img> is EMPTY: that is the
    // flicker. Worse, an <img> hands the compositor the 896px SOURCE texture and
    // asks it to rescale on every zoom step - and this widget runs with
    // hideOnZoom:false (v542), so it is never allowed to bow out. A canvas backed
    // at its own DISPLAYED size is a fraction of that, and drawImage on an
    // already-decoded Image is a blit: no decode, no empty frame.
    const img = document.createElement("canvas");
    img.style.cssText = "width:100%; height:calc(100% - 24px); display:block;" +
        " background:#000;";
    const bar = document.createElement("div");
    bar.style.cssText = "position:absolute; left:0; right:0; bottom:0; height:24px;" +
        " display:flex; align-items:center; gap:6px; padding:0 8px;" +
        " background:rgba(0,0,0,.55); font:11px monospace; color:#ccc;";
    const btn = document.createElement("span");
    btn.textContent = "▶"; btn.style.cssText = "cursor:pointer; user-select:none;";
    const scrub = document.createElement("input");
    scrub.type = "range"; scrub.min = "0"; scrub.value = "0";
    scrub.style.cssText = "flex:1; height:12px;";
    const count = document.createElement("span");
    const loop = document.createElement("span");
    loop.textContent = "⟳"; loop.title = "Loop";
    loop.style.cssText = "cursor:pointer; user-select:none; opacity:.4;";
    bar.append(btn, scrub, count, loop);
    box.append(img, bar);

    btn.onclick = () => { node._pvTimer ? _pvStop(node) : _pvPlay(node); };
    loop.onclick = () => {
        node._pvLoop = !node._pvLoop;
        loop.style.opacity = node._pvLoop ? "1" : ".4";
    };
    scrub.oninput = () => { _pvStop(node); _pvShow(node, Number(scrub.value)); };

    // The canvas keeps its backing store while the graph ZOOMS (ComfyUI scales
    // DOM widgets with a CSS transform, which does not touch clientWidth), so
    // this fires only when the NODE itself is resized - exactly when a re-blit
    // is needed and never on the zoom path. Dies with the node; nothing to unhook.
    try {
        const _ro = new ResizeObserver(() => {
            const f = node._pvFrames || [];
            if (f.length) _pvBlit(node, f[node._pvIdx || 0]);
        });
        _ro.observe(img);
        node._pvRO = _ro;
    } catch (_e) { /* no ResizeObserver: the canvas just CSS-scales. Still fine. */ }

    node._pvBox = box; node._pvImg = img; node._pvBar = bar; node._pvBtn = btn;
    node._pvScrub = scrub; node._pvCount = count;
    node._pvLoop = false; node._pvFrames = []; node._pvIdx = 0;
    node._pvPrevH = 0;   // zero height (invisible) until the first run delivers

    const w = node.addDOMWidget("pls_pu_result", "div", box,
        { serialize: false, hideOnZoom: false });   // v542: survive zoom-out
    w.computeSize = (width) => [width, node._pvPrevH];
    return w;
}

function _pvApply(node, list) {
    if (!Array.isArray(list) || !list.length || !node._pvBox) return;
    if (!_pvFirstLogged) {   // v552 (measure > believe)
        _pvFirstLogged = true;
        console.info("[PLS] PU result viewer: first payload", list.length, "frame(s)");
    }
    _pvStop(node);
    _procHide(node);   // v591: the result is in - the process pane bows out
    node._pvFps = Number(list[0].fps) || 12;
    node._pvFrames = list.map((it) => {
        const im = new Image(); im.src = _viewURL(it); return im;
    });
    node._pvScrub.max = String(node._pvFrames.length - 1);
    const multi = node._pvFrames.length > 1;
    node._pvBar.style.display = multi ? "flex" : "none";
    node._pvImg.style.height = multi ? "calc(100% - 24px)" : "100%";
    node._pvBox.style.display = "block";
    node._pvFrames[0].onload = () => {
        // v531 lesson: fit the HEIGHT only - a run must never SHRINK the width.
        // v592: the widen is shared with the process pane and has usually
        // already fired by now (first probe). This call is the belt for a node
        // that only ever sees the result.
        _widenToTarget(node);
        const iw = node._pvFrames[0].naturalWidth || 1;
        const ih = node._pvFrames[0].naturalHeight || 1;
        const boxW = Math.max(NODE_MIN_W, node.size[0], PV_TARGET_W);
        const grew = boxW > node.size[0];
        const barH = multi ? 24 : 0;
        const fitH = Math.max(PV_MIN_H,
            Math.min(PV_MAX_H, Math.round(boxW * ih / iw) + barH));
        if (grew || Math.abs(fitH - node._pvPrevH) > 2) {   // deadband (v531)
            node._pvPrevH = fitH;
            node.setSize([boxW, node.computeSize()[1]]);
        }
        _pvShow(node, 0);
        if (multi) _pvPlay(node);   // plays once - loop is off by default (v534)
        node.setDirtyCanvas(true, true);
    };
    _pvShow(node, 0);
}

// ═══ v550: process view (the tile being refined RIGHT NOW) ═════════════
// A collapsible pane fed by the backend probe over "polyhedron.pu_tile": the
// current tile as latent2rgb, a minimap locating it on the stage canvas, and
// a HUD (stage · tile i/n · step s/t). Runtime-only - nothing serialises.
const PROC_HEAD_H = 22, PROC_BODY_H = 160;
// v596 (Frank, 11:51: "Noe. :D"). v595 capped the tile at 512 and left the rest
// of a wide node as a black slab - 188px at node 900, 388px at 1100 - with the
// minimap stranded at the far edge of it. The cap was right; the LAYOUT was
// still wrong, because the two boxes were pinned to the left and the leftover
// was dumped on the right.
//
// Frank's instruction: the two views sit together, side by side, each in its own
// area, with a little air between them. So:
//
//   * the pair is CENTRED in the body. A node pulled wider than the pair needs
//     puts equal air on both sides - no slab, no stranded instrument.
//   * each view gets its OWN framed area (border + inset background), so they
//     read as two panels rather than one image with a smudge next to it.
//   * the map is sized off the TILE (a third of it), not off the node - it is a
//     companion to the preview, not a consumer of whatever width is lying about.
//   * the tile is still capped at 512/edge (v595, Frank's law) and still fitted
//     with object-fit:contain (v589's quiet virtue: it EMBEDS the frame, it
//     never inflates it).
//
// PROC_NAT_W is what the pane actually wants: pad + 512 + gap + map + pad. The
// widen aims there now, so a default-width node shows the full 512 instead of
// a 382px compromise.
// v597 (Frank, 13:07/13:12). Two remarks, one cause:
//
//   "die Vorschau ist mir immer noch ein wenig zu gross"
//   "vae (sharp) ... ist allerdings jetzt recht kachelig"
//
// They are the same sentence. At 512 the pane showed the 512px probe jpeg 1:1 -
// so every artefact in it landed on screen at full size. And there ARE artefacts:
// the sharp path decodes ONE latent frame through a 3D VIDEO vae whose temporal
// convolutions expect a neighbour. T=1 is the degenerate case; it blocks. (The
// pixel stage proves it: same 512px jpeg, same Q80, real RGB instead of a decode
// - and Frank's screenshot of it is razor sharp.)
//
// Downscaling is a low-pass. The OLD pane looked clean precisely BECAUSE it was
// small - it averaged the blocks away. 384 restores that virtue without going
// back to a postage stamp: the 512px jpeg lands x1.33 down, which is smoothing
// for free, and the node's natural width drops from 710 to 541.
//
// The invariant that matters is NOT the number 384. It is: the pane never shows
// a jpeg larger than 1:1. Anything above _PROBE_MAX_EDGE is upscaling, and an
// upscaled preview is a blurrier preview. 384 is taste; <= 512 is law.
const PROC_TILE_MAX = 384;
const PROC_MAP_RATIO = 0.32;                    // the map is ~1/3 of the tile
const PROC_MAP_MIN = 120, PROC_MAP_MAX = 200;
const PROC_GAP = 10, PROC_PAD = 12;
const PROC_NAT_W = PROC_PAD + PROC_TILE_MAX + PROC_GAP
                 + Math.round(PROC_TILE_MAX * PROC_MAP_RATIO) + PROC_PAD;

// One place computes the pane's geometry, from the frame that is actually in it.
// Returns nothing; writes explicit pixels. Flexbox was asked to infer this in
// v594 and inferred "take everything", which is the one thing the tile must not
// do. Arithmetic does not negotiate.
function _procFit(node) {
    const im = node._procTile;
    if (!im) return;
    const iw = im.naturalWidth || 0, ih = im.naturalHeight || 0;
    const avail = Math.max(240, node.size[0] - 2 * PROC_PAD);

    // the tile first, capped; then the map as a FRACTION OF THE TILE
    let tileW = Math.min(PROC_TILE_MAX,
        Math.round((avail - PROC_GAP) / (1 + PROC_MAP_RATIO)));
    let tileH = (iw && ih) ? Math.round(tileW * ih / iw) : tileW;
    if (tileH > PROC_TILE_MAX) {          // portrait: the cap is per EDGE
        tileH = PROC_TILE_MAX;
        tileW = (iw && ih) ? Math.round(tileH * iw / ih) : tileW;
    }
    let mapW = Math.max(PROC_MAP_MIN,
        Math.min(PROC_MAP_MAX, Math.round(tileW * PROC_MAP_RATIO)));

    // a very wide node: the pair simply centres and the surplus becomes air on
    // BOTH sides. Nothing is stretched to fill a hole that should not be filled.
    if (node._procTileBox) {
        node._procTileBox.style.width = tileW + "px";
        node._procTileBox.style.height = tileH + "px";
    }
    im.style.width = tileW + "px";
    im.style.height = tileH + "px";
    if (node._procMapBox) {
        node._procMapBox.style.width = mapW + "px";
        node._procMapBox.style.height = tileH + "px";   // same shoulder height
    }

    const bodyH = Math.max(PROC_BODY_H, tileH + 2 * PROC_PAD);
    node._procBodyH = bodyH;
    const want = PROC_HEAD_H + (node._procOpen ? bodyH : 0);
    if (Math.abs(want - node._procH) > 2) {   // deadband (v531)
        node._procH = want;
        node.setSize([node.size[0], node.computeSize()[1]]);   // height only
        node.setDirtyCanvas(true, true);
    }
}
// v565: the pixel stage (P) feeds the same pane as the refine stages (H / L).
// It arrives as finished RGB, so it needs no latent2rgb - see _make_pixel_probe.
const STAGE_MARK = { high: "H", low: "L", pixel: "P" };
// v567: the pane shares the backend's clock. elapsed/eta ride the probe
// payload, so console, bar and HUD tick on the SAME numbers - Frank's
// congruence ask. Format mirrors the backend's _fmt_clock exactly.
function _fmtClock(sec) {
    const s = Math.max(0, Math.floor(sec));
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
    const mm = String(m).padStart(2, "0"), rr = String(r).padStart(2, "0");
    return h ? (h + ":" + mm + ":" + rr) : (m + ":" + rr);
}
let _procFirstLogged = false;   // v552: prove the receive side once per session

function _buildProcView(node) {
    const box = document.createElement("div");
    box.style.cssText = "width:100%; height:100%; display:none; overflow:hidden;" +
        " background:#111; border-radius:6px; font:11px monospace; color:#ccc;";
    const head = document.createElement("div");
    head.style.cssText = "height:" + PROC_HEAD_H + "px; display:flex;" +
        " align-items:center; gap:6px; padding:0 8px; background:#1c1c1c;" +
        " cursor:pointer; user-select:none;";
    const arrow = document.createElement("span");
    arrow.textContent = "▾";
    const title = document.createElement("span");
    title.textContent = "Process view";
    const hud = document.createElement("span");
    hud.style.cssText = "margin-left:auto; opacity:.8;";
    head.append(arrow, title, hud);
    const body = document.createElement("div");
    // v596: CENTRED. A node wider than the pair needs puts equal air on both
    // sides instead of a black slab on one.
    body.style.cssText = "height:calc(100% - " + PROC_HEAD_H + "px); display:flex;" +
        " gap:" + PROC_GAP + "px; padding:" + PROC_PAD + "px; box-sizing:border-box;" +
        " align-items:center; justify-content:center;";
    const PANEL = "flex:0 0 auto; box-sizing:border-box; background:#0b0b0b;" +
        " border:1px solid #3a3a3a; border-radius:6px; overflow:hidden;" +
        " display:flex; align-items:center; justify-content:center;";
    // v596: the tile lives in its OWN panel now - two areas, not one image with
    // a smudge beside it. No flex:1 anywhere: both boxes are computed.
    const tileBox = document.createElement("div");
    tileBox.style.cssText = PANEL;
    const tile = document.createElement("img");
    tile.style.cssText = "display:block; object-fit:contain;";   // v589's virtue:
    // contain EMBEDS the frame; it never inflates it.
    tileBox.append(tile);
    const mapBox = document.createElement("div");
    mapBox.style.cssText = PANEL + " padding:8px;";
    const map = document.createElement("div");
    map.style.cssText = "position:relative; max-width:100%; max-height:100%;" +
        " width:100%; border:1px solid #444; background:#000;";
    const rectEl = document.createElement("div");
    rectEl.style.cssText = "position:absolute; background:rgba(255,140,0,.55);" +
        " border:1px solid #ff8c00;";
    // v600: the orange rectangle says WHERE. It never said HOW MANY -- and without
    // that, the instrument is half mute. A single rect filling the frame and a rect
    // that is one of 32 are the same shape until you stop and count grid lines. The
    // count IS the thing you glance at this to check.
    const mapLbl = document.createElement("div");
    mapLbl.style.cssText = "position:absolute; top:3px; left:5px; z-index:2;" +
        " font:10px/1.2 ui-monospace,monospace; color:#ff8c00; font-weight:bold;" +
        " text-shadow:0 0 3px #000,0 0 3px #000,0 0 3px #000; pointer-events:none;";
    map.append(rectEl, mapLbl); mapBox.append(map); body.append(tileBox, mapBox);
    node._procMapLbl = mapLbl;
    box.append(head, body);

    head.onclick = () => {
        node._procOpen = !node._procOpen;
        arrow.textContent = node._procOpen ? "▾" : "▸";
        body.style.display = node._procOpen ? "flex" : "none";
        // v592: the measured height, not the old hard 160
        node._procH = PROC_HEAD_H
            + (node._procOpen ? (node._procBodyH || PROC_BODY_H) : 0);
        node.setSize([node.size[0], node.computeSize()[1]]);   // height only (v531)
        node.setDirtyCanvas(true, true);
    };

    // v592: the jpeg decides the pane's height, so the pane must wait for it.
    // Fires on every probe frame; _procFit deadbands, so a steady stream of
    // same-shape tiles costs one comparison each.
    tile.onload = () => { try { _procFit(node); } catch (_e) { /* never break */ } };

    node._procBox = box; node._procHud = hud; node._procTile = tile;
    node._procTileBox = tileBox; node._procMap = map; node._procRect = rectEl; node._procMapBox = mapBox;
    node._procOpen = true; node._procSeen = false;
    node._procH = 0;   // zero height until the first probe event arrives

    const w = node.addDOMWidget("pls_pu_process", "div", box,
        { serialize: false, hideOnZoom: false });   // v542: survive zoom-out
    w.computeSize = (width) => [width, node._procH];
    return w;
}

// v591 (Frank's ask): the process view is a window into work that is HAPPENING.
// The moment the result frames land, that work is over - the pane then costs
// node height to show a frozen last tile and a HUD reading "Chunk 9/9". Fold it
// away and RE-ARM it: the next run's first probe reveals it again, at full
// height, with _procOpen (the user's collapse choice) intact. An errored or
// interrupted run keeps the pane - no result payload ever arrives - which is
// exactly the run where you want to see where it stopped.
// v601: THE MIRROR OF _procHide, and it was missing.
//
// v591 taught the process pane to bow out when the result arrives. Nobody taught
// the RESULT viewer to bow out when a new run STARTS -- so a finished picture from
// the last run sat above a job that was three minutes from done, and a finished
// picture above a running job looks exactly like a finished job. Frank read it as
// output and it was history.
//
// Symmetry, not decoration: the pane that is TRUE right now is the pane on screen.
// Running -> process view. Done -> result. Never both, because "both" is the state
// where the user cannot tell which one to believe.
//
// An interrupted run keeps the process pane and does NOT restore the old picture:
// that run produced no result, and the previous run's image is not this run's
// answer. Showing it would be the same lie in a friendlier font.
function _pvHide(node) {
    if (!node._pvBox || node._pvBox.style.display === "none") return;
    _pvStop(node);                                        // the loop stops too
    node._pvBox.style.display = "none";
    node._pvPrevH = 0;
    node.setSize([node.size[0], node.computeSize()[1]]);   // height only (v531)
    node.setDirtyCanvas(true, true);
}

function _procHide(node) {
    if (!node._procBox || !node._procSeen) return;
    node._procBox.style.display = "none";
    node._procH = 0;
    node._procSeen = false;                                // re-arms for next run
    node.setSize([node.size[0], node.computeSize()[1]]);   // height only (v531)
    node.setDirtyCanvas(true, true);
}

function _procApply(node, d) {
    if (!node._procBox) return;
    if (!node._procSeen) {   // first data: reveal ONCE (a collapse is respected)
        node._procSeen = true;
        node._procBox.style.display = "block";
        _pvHide(node);   // v601: and the LAST run's result stops pretending to be this one

        // v592: widen HERE - seconds into the run, not when it is over. This is
        // the pane that is actually on screen while the work happens; v591 only
        // widened for the result viewer, which is still empty at this moment.
        // v596: the PROCESS pane needs room for BOTH panels, not just 512.
        _widenToTarget(node, PROC_NAT_W);
        node._procH = PROC_HEAD_H + (node._procOpen ? PROC_BODY_H : 0);
        node.setSize([node.size[0], node.computeSize()[1]]);   // height only (v531)
    }
    node._procTile.src = "data:image/jpeg;base64," + d.jpeg;   // onload -> _procFit
    // v565: three stages now, not two. The old ternary was `high ? "H" : "L"`,
    // which would have labelled every pixel-stage frame "L" - a confident lie.
    // A stage this pane does not know must SAY so ("?"), never guess.
    const st = STAGE_MARK[d.stage] || "?";
    // v567: same clock as the console line and the progress bar.
    const clk = (d.elapsed !== undefined && d.elapsed !== null)
        ? " · " + _fmtClock(d.elapsed) +
          ((d.eta !== undefined && d.eta !== null)
              ? " · ETA ~" + _fmtClock(d.eta) : "")
        : "";
    node._procHud.textContent = ((d.stage === "pixel")
        ? st + " · Chunk " + d.tile + "/" + d.tiles +
          " · Frame " + d.step + "/" + d.steps
        : st + " · Tile " + d.tile + "/" + d.tiles +
          " · Step " + d.step + "/" + d.steps) + clk;
    if (node._procMapLbl) {
        const nT = Math.max(1, d.tiles | 0);
        const noun = (d.stage === "pixel") ? (nT === 1 ? "Chunk" : "chunks")
                                           : (nT === 1 ? "Tile"  : "tiles");
        node._procMapLbl.textContent = nT + " " + noun;
    }
    const cw = Math.max(1, d.canvas[0]), ch = Math.max(1, d.canvas[1]);
    // v579 hid the minimap when a single tile filled the whole stage: the orange
    // rect covered 100% of it, and that "said nothing". v592 hid it on the pixel
    // stage too, where the chunks are frames in TIME and the rect is the whole
    // canvas by construction.
    //
    // v594 overrules BOTH, on Frank's call, and he is right. The minimap is a
    // CONTROL INSTRUMENT, not decoration. It is how you see at a glance whether
    // the tiling is set up the way you meant it. A single rectangle filling the
    // frame says exactly one thing - "no grid here, one tile is the canvas" -
    // and that is the thing being checked for. Hide it, and a mis-set tile_size
    // looks identical to a correct one: both show nothing. Dial tile_size down
    // to 512 on a 768 source and the 2x2 grid appears with the rect walking
    // through it. That is the instrument working.
    //
    // The rect maths below is unchanged and already correct in every case: one
    // tile -> 100%, a real grid -> the cell being refined.
    node._procMapBox.style.display = "flex";
    node._procMap.style.aspectRatio = cw + " / " + ch;
    node._procRect.style.left = (100 * d.rect[0] / cw) + "%";
    node._procRect.style.top = (100 * d.rect[1] / ch) + "%";
    node._procRect.style.width = (100 * d.rect[2] / cw) + "%";
    node._procRect.style.height = (100 * d.rect[3] / ch) + "%";
    _procFit(node);   // v592: the map just moved - the tile's width changed with it
    node.setDirtyCanvas(true, false);
}

app.registerExtension({
    name: "polyhedron.power_upscale.mode",
    async setup() {
        // v550: ONE global listener feeds every Power Upscale process pane.
        // v552: routing now matches uls_live_preview (the LIVE-PROVEN pattern):
        // node.type, not comfyClass - comfyClass may be unset on the graph
        // node object, which silently dropped every event. Also the robust
        // detail extraction and an armed marker, so F12 can prove this ran.
        console.info("[PLS] PU process listener armed (polyhedron.pu_tile)");
        api.addEventListener("polyhedron.pu_tile", (e) => {
            try {
                const d = (e && e.detail) || {};
                if (!d.node || !d.jpeg) return;
                const node = app.graph.getNodeById(Number(d.node));
                if (!node || node.type !== NODE_TYPE) return;
                if (!_procFirstLogged) {
                    _procFirstLogged = true;
                    console.info("[PLS] PU process view: first event (node", d.node + ")");
                }
                _procApply(node, d);
            } catch (err) { /* a preview must never break the ui */ }
        });
        const _tag = (txt) => {
            try {
                (app.graph._nodes || []).forEach((n) => {
                    if (n.type === NODE_TYPE && n._procSeen && n._procHud &&
                        !n._procHud.textContent.endsWith(txt)) {
                        n._procHud.textContent += txt;
                    }
                });
            } catch (e) { /* ignore */ }
        };
        api.addEventListener("execution_success", () => _tag(" · done"));
        api.addEventListener("execution_error", () => _tag(" · stopped"));
        api.addEventListener("execution_interrupted", () => _tag(" · stopped"));
    },

    // v548: belt and braces - re-apply once more after the WHOLE graph finished
    // loading. Idempotent (a graph already in display order is a no-op), and it
    // catches frontend variants where the onConfigure ordering differs.
    loadedGraphNode(node) {
        if (!node) return;
        if (node.comfyClass !== NODE_TYPE && node.type !== NODE_TYPE) return;
        try { _reorderInputsToDisplay(node); } catch (e) { /* never break loading */ }
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        // On creation: label the pill, wrap its callback, apply the mode state.
        const _onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = _onCreated ? _onCreated.apply(this, arguments) : undefined;
            const self = this;

            const pill = _findWidget(self, "dual_moe");
            if (pill) {
                pill.label = "🛈 Mode";
                const _cb = pill.callback;
                pill.callback = function () {
                    const rr = _cb ? _cb.apply(this, arguments) : undefined;
                    _applyModeState(self);
                    return rr;
                };
            }

            // v546: show the widgets in DISPLAY order (twins adjacent). The SERIALISED
            // order stays canonical — configure maps in, onSerialize maps back. Fail-safe:
            // if the widget set is not exactly the 18 canonical widgets this is a no-op.
            self._plsDisplayReordered = _reorderWidgetsToDisplay(self);
            _reorderInputsToDisplay(self);      // fresh node: no links yet, nothing to repair

            _applyModeState(self);
            _buildViewer(self);   // v549: result viewer pane (zero height until fed)
            _buildProcView(self);   // v550: process pane (zero height until fed)
            return r;
        };

        // v546: heal a pre-v546 save (16, canonical) to 18, THEN map canonical -> DISPLAY
        // (the widgets are shown permuted, and LiteGraph applies widgets_values by position).
        const _configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (info) {
            try {
                if (info && Array.isArray(info.widgets_values)) {
                    // v588: the save states its own order - v589: or gets read
                    // by its types. Guessing is over either way.
                    const marked = !!(info.properties && info.properties[CANON_MARKER]);
                    info.widgets_values = _healPreV546(info.widgets_values);
                    info.widgets_values = _healPreV549(info.widgets_values);
                    info.widgets_values = _healPreV550(info.widgets_values);
                    info.widgets_values = _healPreV553(info.widgets_values);
                    info.widgets_values = _healPreV555(info.widgets_values);
                    info.widgets_values = _healPreV560(info.widgets_values);
                    info.widgets_values = _healPreV562(info.widgets_values);
                    info.widgets_values = _healPreV564(info.widgets_values);
                    info.widgets_values = _padToCanon(info.widgets_values);  // v563
                    const _ord = _saveOrderOf(info.widgets_values, marked);
                    // v852: three historic layouts can reach us now. Each one
                    // is mapped home through ITS OWN frozen table; "canon" and
                    // "unknown" are left alone (status quo, said out loud).
                    if (_ord === "legacy-display") {
                        info.widgets_values = _legacyDisplayToCanon(info.widgets_values);
                    } else if (_ord === "legacy-display-851") {
                        info.widgets_values = _tableToCanon(info.widgets_values,
                                                            DISPLAY_LEGACY_V851);
                    } else if (_ord === "display-current") {
                        // a stray save in the order we display TODAY - the v584
                        // phantom class, which until now had no map at all
                        // because the current order never had a table.
                        info.widgets_values = _displayToCanon(info.widgets_values);
                    }
                    console.info(marked
                        ? "[PLS] PU: canon-marked save (order proven by its own "
                          + "serialize - immune to display re-sorts)"
                        : (_ord === "canon"
                            ? "[PLS] PU: legacy save (no canon marker) - TYPE "
                              + "fingerprint says canon order; the marker is "
                              + "written on the next save"
                            : (_ord === "unknown"
                                ? "[PLS] PU: legacy save (no canon marker) - "
                                  + "fingerprint INCONCLUSIVE; loading as canon "
                                  + "(the pre-v589 status quo). Check the dials "
                                  + "once; the marker is written on the next save"
                                : "[PLS] PU: legacy save (no canon marker) - TYPE "
                                  + "fingerprint says " + _ord + "; mapped "
                                  + "through its table, values keep their names")));
                    if (this._plsDisplayReordered) {
                        info.widgets_values = _canonToDisplay(info.widgets_values);
                    }
                }
            } catch (err) { /* never break configure */ }
            const r = _configure ? _configure.apply(this, arguments) : undefined;
            try { _sanitize(this); } catch (e) { /* the net must never tear */ }
            return r;
        };

        // v546: base serialize emits widgets_values in DISPLAY order -> map back to the ONE
        // true canonical order, so the saved file matches INPUT_TYPES (and the heal above).
        const _onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (o) {
            const r = _onSerialize ? _onSerialize.apply(this, arguments) : undefined;
            try {
                if (this._plsDisplayReordered && o && Array.isArray(o.widgets_values)) {
                    o.widgets_values = _displayToCanon(o.widgets_values);
                    // v588: the save now CARRIES the proof. Marked = these values
                    // are canon-ordered by construction; only this branch may mark.
                    o.properties = o.properties || {};
                    o.properties[CANON_MARKER] = 588;
                }
            } catch (err) { /* never break serialize */ }
            return r;
        };

        // v549: lower-edge drag scales the viewer pane between min and max
        // (the ph_save pattern) - the widget stack above keeps its exact height.
        const _onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            const r = _onResize ? _onResize.apply(this, arguments) : undefined;
            if (this._pvPrevH > 0) {
                const stackAbove = this.computeSize()[1] - this._pvPrevH;
                const want = Math.max(PV_MIN_H,
                    Math.min(PV_MAX_H, size[1] - stackAbove));
                if (Math.abs(want - this._pvPrevH) > 2) this._pvPrevH = want;
                size[1] = stackAbove + this._pvPrevH;
            }
            return r;
        };

        // v549: the backend ui payload (pls_pu_preview) feeds the viewer.
        const _onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = _onExecuted ? _onExecuted.apply(this, arguments) : undefined;
            try {
                _pvApply(this, message && message.pls_pu_preview);
            } catch (e) { /* a preview must never break a run */ }
            return r;
        };

        const _onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            _pvStop(this);
            return _onRemoved ? _onRemoved.apply(this, arguments) : undefined;
        };

        // Loading a saved graph: re-derive the state after LiteGraph applied
        // widgets_values (disabled/label are not serialised — always derived).
        const _onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = _onConfigure ? _onConfigure.apply(this, arguments) : undefined;
            // configure() restored node.inputs from the SAVE (old order, links intact) ->
            // permute to the display order and repair the link slots. Idempotent: a graph
            // saved in the new order permutes to itself.
            try { _reorderInputsToDisplay(this); } catch (e) { /* never break loading */ }
            _applyModeState(this);
            return r;
        };
    },
});
