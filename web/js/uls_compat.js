/**
 * Polyhedron Suite — Compatibility & Diagnostics Layer
 * ═════════════════════════════════════════════════════════
 *
 * Purpose
 * -------
 * The Stack and Engine nodes render their entire UI by hand on the LiteGraph
 * canvas (onDrawForeground + onMouseDown/Move/Up). ComfyUI's frontend now ships
 * a SECOND renderer — the Vue-based "Nodes 2.0" path — which does NOT call
 * onDrawForeground for a node. Under that renderer the hand-built Polyhedron UI
 * would simply not appear, with no error: the user just sees an empty node and
 * has no idea why.
 *
 * This module is the early-warning system for exactly that situation. It is:
 *
 *   • Purely ADDITIVE — it registers its own extension namespace, reads state,
 *     adds an About-page badge / a setting / a one-time toast. It does NOT
 *     change how any node renders or behaves. It cannot break the plugin.
 *
 *   • RENDERER-AGNOSTIC by design. It does not sniff version strings or guess
 *     setting keys (both churn release-to-release). Instead it observes a fact
 *     that is true regardless of how ComfyUI names its modes: did our node's
 *     canvas draw path actually run? uls_node.js sets `node._ulsDrawFired`
 *     inside onDrawForeground; if that never fires for a placed Polyhedron
 *     canvas node, the active renderer isn't drawing our UI → warn, with
 *     actionable guidance.
 *
 * Why this has lasting value
 * --------------------------
 * The recurring pain in this project's history is "ComfyUI changed under us
 * and a node silently broke". This layer turns that silent failure mode into a
 * clear, actionable message — and centralises the capability checks in ONE
 * place, so future ComfyUI changes surface here instead of scattered across
 * the 3000-line main UI file. It is also the detection foundation any eventual
 * Vue/Nodes-2.0 migration will build on (you cannot migrate what you cannot
 * reliably detect).
 *
 * Inspect at runtime via the browser console:  window.__POLYHEDRON_COMPAT__
 */

import { app } from "../../scripts/app.js";

// Node class names whose UI is hand-drawn on the LiteGraph canvas.
// Backend-only nodes (Bridge, Sigma, FrameInflate, Token Counter, Inspector,
// Model Switch) render via standard widgets and are unaffected by the renderer
// change — they are deliberately NOT in this set.
const POLY_CANVAS_NODES = new Set(["UltimateLoraStack", "ULSAccelerator"]);

const PLUGIN_VERSION = "v364";
// v303: 8s — 3s false-positived on large workflows / slow first draws.
// LiteGraph culls offscreen nodes (onDrawForeground never runs for them), so
// the notice can still appear for an offscreen-but-healthy node; the draw
// path in uls_node.js now removes it the moment the node provably draws
// (self-healing), which makes the whole mechanism false-positive-safe.
const GRACE_MS       = 8000;

// ── Best-effort environment snapshot (all reads guarded) ──────────────────
function readFrontendVersion() {
    try {
        if (typeof window !== "undefined" && window.__COMFYUI_FRONTEND_VERSION__) {
            return String(window.__COMFYUI_FRONTEND_VERSION__);
        }
    } catch (e) { /* ignore */ }
    return "unknown";
}

const COMPAT = {
    plugin:             PLUGIN_VERSION,
    frontendVersion:    readFrontendVersion(),
    hasExtensionManager: false,   // new (Vue-capable) frontend exposes this
    hasLiteGraph:        (typeof LiteGraph !== "undefined"),
    canvasDrawObserved:  false,   // set true once any Polyhedron canvas node draws
    rendererWarningShown: false,
    fallbackWidgetNodes: [],      // v301: node ids that received the in-node notice
    probeRearms:         0,       // v306: deferred judgements (all silent nodes offscreen)
    lastJudgement:       null,    // v306: "drew" | "toast" | "deferred" | "gave-up"
    checkedAt:           null,
    note: "Inspect this object for a quick Polyhedron environment report.",
};
try { window.__POLYHEDRON_COMPAT__ = COMPAT; } catch (e) { /* ignore */ }

// ── One-time, dismissible renderer warning ────────────────────────────────
let _warned = false;

function suppressed() {
    try {
        return !!app.extensionManager?.setting?.get?.(
            "Polyhedron.compat.suppressRendererWarning");
    } catch (e) { return false; }
}

function notify(severity, summary, detail, life) {
    // Prefer the documented toast API; fall back gracefully to console only.
    try {
        const tm = app.extensionManager?.toast;
        if (tm?.add) { tm.add({ severity, summary, detail, life }); return; }
        if (tm?.addAlert) { tm.addAlert(detail); return; }
    } catch (e) { /* ignore */ }
}

function warnRendererOnce() {
    if (_warned) return;
    _warned = true;
    COMPAT.rendererWarningShown = true;
    const detail =
        "A Polyhedron Stack/Engine node is placed but its custom UI hasn't " +
        "rendered. Your ComfyUI is likely using the new Vue node renderer " +
        "(\"Nodes 2.0\"), which doesn't draw the hand-built canvas UI. To use " +
        "the full Polyhedron interface, switch to LiteGraph rendering in " +
        "Settings (disable Nodes 2.0 / Vue nodes). Backend nodes — Bridge, " +
        "Sigma, Token Counter, Inspector, Model Switch — work fine either way.";
    console.warn("[Polyhedron] ⚠ " + detail +
        "  (Suppress via Settings → Polyhedron, or set " +
        "window.__POLYHEDRON_COMPAT__ aside for debugging.)");
    notify("warn", "Polyhedron Suite — renderer notice", detail, 15000);
}

// ── v301: visible in-node fallback when the canvas UI never drew ───────────
// A toast is transient; the broken node is what the user keeps staring at.
// Standard widgets ARE rendered by the Vue ("Nodes 2.0") renderer, so we put
// the guidance right inside the affected node. The widget is display-only and
// explicitly excluded from serialization (the Stack stores its rows JSON in
// widgets_values — a stray serialized widget would corrupt that layout).
const FALLBACK_WIDGET_NAME = "polyhedron_renderer_notice";

function injectFallbackWidget(node) {
    try {
        if (!node || !node.addWidget) return;
        if ((node.widgets || []).some(w => w?.name === FALLBACK_WIDGET_NAME)) return;
        const w = node.addWidget(
            "text",
            FALLBACK_WIDGET_NAME,
            "UI needs LiteGraph renderer — disable 'Modern Node Design' " +
            "(Nodes 2.0) in Settings. Your rows are safe.",
            () => {},
            { serialize: false }
        );
        if (w) {
            w.serialize = false;                       // belt + suspenders:
            if (w.options) w.options.serialize = false; // never enter widgets_values
            w.disabled = true;
        }
        COMPAT.fallbackWidgetNodes.push(node.id);
        node.setDirtyCanvas?.(true, true);
    } catch (e) {
        console.debug("[Polyhedron] fallback widget skipped:", e);
    }
}

// ── v306: viewport test — is a node inside the visible canvas area? ────────
// LiteGraph culls offscreen nodes (onDrawForeground never runs for them), so
// "placed but never drew" is only meaningful evidence for a node the canvas
// would actually have drawn. Fails OPEN (returns true) on any doubt: under a
// genuine Vue/Nodes-2.0 renderer these internals may be absent, and that is
// exactly the case where the warning must not be swallowed.
function nodeInViewport(n) {
    try {
        const c  = app.canvas;
        const ds = c?.ds;
        const el = c?.canvas;
        if (!c || !ds || !el || !Array.isArray(ds.offset)) return true;
        const scale = ds.scale || 1;
        const vx = -ds.offset[0], vy = -ds.offset[1];
        const vw = el.width / scale, vh = el.height / scale;
        const TITLE = 30;  // node title bar sits above pos[1]
        const nx = n.pos[0], ny = n.pos[1] - TITLE;
        const nw = n.size[0], nh = n.size[1] + TITLE;
        return nx + nw > vx && nx < vx + vw && ny + nh > vy && ny < vy + vh;
    } catch (e) {
        return true;   // fail open → rather warn once too often than never
    }
}

const REARM_MAX = 12;   // ≈ REARM_MAX × GRACE_MS of deferred judgement, then give up

// ── Probe: after a node is created, give it a grace period to draw ─────────
function armProbe() {
    setTimeout(() => {
        try {
            if (suppressed()) return;

            const nodes = app.graph?._nodes || [];
            let present = false;
            let drew = false;
            const silent = [];
            for (const n of nodes) {
                const t = n?.comfyClass || n?.type;
                if (!POLY_CANVAS_NODES.has(t)) continue;
                present = true;
                if (n._ulsDrawFired) { drew = true; }
                else { silent.push(n); }
            }

            // If the user removed the node again, there's nothing to warn about.
            if (!present) return;

            COMPAT.canvasDrawObserved = drew;
            COMPAT.checkedAt = new Date().toISOString();

            if (silent.length === 0) return;   // all drew → all good, silent

            // v306: judge with viewport evidence instead of time alone.
            // 1) ANY Polyhedron node drew → the canvas path is alive; silent
            //    siblings are merely offscreen (LiteGraph culls them). No
            //    toast, no injection — if one ever WERE broken, the v303
            //    self-healing widget path would still cover it on draw.
            if (drew) { COMPAT.lastJudgement = "drew"; return; }

            // 2) None drew. Only a node the canvas would have drawn — i.e.
            //    one inside the viewport — is evidence of a dead renderer.
            const visibleSilent = silent.filter(nodeInViewport);
            if (visibleSilent.length > 0) {
                COMPAT.lastJudgement = "toast";
                for (const n of silent) injectFallbackWidget(n);
                warnRendererOnce();    // visible but never drew → real notice
                return;
            }

            // 3) Everything offscreen → no evidence either way; re-check
            //    later instead of guessing (bounded, then give up silently).
            if (COMPAT.probeRearms < REARM_MAX) {
                COMPAT.probeRearms += 1;
                COMPAT.lastJudgement = "deferred";
                armProbe();
            } else {
                COMPAT.lastJudgement = "gave-up";
                console.debug("[Polyhedron] compat probe gave up: all Stack/" +
                    "Engine nodes stayed offscreen; no renderer judgement made.");
            }
        } catch (e) {
            // Diagnostics must never throw into ComfyUI.
            console.debug("[Polyhedron] compat probe skipped:", e);
        }
    }, GRACE_MS);
}

app.registerExtension({
    name: "Polyhedron.compat",

    // About-page badge (frontend ignores this field on older versions).
    aboutPageBadges: [
        {
            label: "Polyhedron Suite " + PLUGIN_VERSION,
            url: "https://civitai.red/user/Polyhedron_AI",
            icon: "pi pi-box",
        },
    ],

    // User-facing toggle (frontend ignores this field on older versions).
    settings: [
        {
            id: "Polyhedron.compat.suppressRendererWarning",
            name: "Polyhedron: suppress renderer-compatibility warning",
            type: "boolean",
            defaultValue: false,
            tooltip: "Hide the one-time notice shown when the active renderer " +
                     "(e.g. Vue / Nodes 2.0) doesn't draw the Polyhedron canvas UI.",
        },
    ],

    async setup() {
        COMPAT.hasExtensionManager = !!(app && app.extensionManager);
        COMPAT.checkedAt = new Date().toISOString();
        console.log(
            "[Polyhedron] compat layer ready — " +
            `plugin=${COMPAT.plugin}, frontend=${COMPAT.frontendVersion}, ` +
            `extensionManager=${COMPAT.hasExtensionManager}, ` +
            `litegraph=${COMPAT.hasLiteGraph}`
        );
    },

    // Documented lifecycle hook — fires for every node added to the graph.
    nodeCreated(node) {
        const t = node?.comfyClass || node?.type;
        if (!POLY_CANVAS_NODES.has(t)) return;
        armProbe();
    },
});
