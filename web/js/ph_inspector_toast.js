/*
 * ph_inspector_toast.js — v910
 *
 * An ORANGE (severity "warn") ComfyUI toast for ⬡ Polyhedron LoRA Inspector
 * when a feed is missing.
 *
 * WHY THIS FILE EXISTS. Until v910 both of the Inspector's inputs were
 * `required`. Mute or bypass the Stack that feeds it and ComfyUI answered with
 * `required_input_missing`: the node was painted in NODE_ERROR_COLOUR (#E00)
 * and the queue failed with the core's RED `failedToQueue` toast. That is the
 * core's ERROR channel, spent on a passive read-only node whose entire job is
 * to report what it can see — and red is the wrong colour for "your Stack is
 * muted". The inputs are optional now, so the run goes through; this file
 * supplies the notice that is actually appropriate, in the colour the pack
 * already uses for advice (uls_token_toast.js, ph_vectorize.js).
 *
 * toast() is a DECLARED MIRROR of the helper in uls_token_toast.js: same API,
 * same fallback chain, same console last resort. The guard drives both copies
 * through the same cases and compares them (the mirror rule).
 *
 * The backend (uls_stack_node.ULSInspector.inspect) hands us a structured
 * state via the UI channel ({"ui": {"pls_inspector": [...]}}) — this file
 * never parses the report text (the uls_token_toast rule).
 */

import { app } from "../../scripts/app.js";

console.info("[PLS] ph_inspector_toast.js v910 loaded");

const NODE = "ULSInspector";

function toast(severity, summary, detail, life) {
    try {
        const tm = app.extensionManager?.toast;
        if (tm?.add) { tm.add({ severity, summary, detail, life }); return; }
        if (tm?.addAlert) { tm.addAlert(detail); return; }
    } catch (e) { /* console-only fallback below */ }
    // Last resort: at least leave a console trace.
    console.warn(`[PLS Inspector] ${summary} — ${detail}`);
}

// The notice for one run, or null when there is nothing worth interrupting
// for. Pure, so the guard can drive it directly. NEVER returns "error":
// a missing feed is advice, not a failure — that is the whole point of v910.
export function runNotice(info) {
    if (!info) return null;
    if (info.state === "no_config") {
        return { sig: "no_config", severity: "warn", life: 8000,
                 summary: "LoRA Inspector has no Stack",
                 detail: "Nothing arrived on uls_config_out. The Stack is not "
                       + "connected, or its node is muted/bypassed. The "
                       + "Inspector ran and reported that instead of failing "
                       + "the queue." };
    }
    if (info.state === "no_prompt") {
        return { sig: `no_prompt:${info.loras}`, severity: "warn", life: 8000,
                 summary: "LoRA Inspector has no prompt",
                 detail: `Nothing arrived on the prompt input, so all `
                       + `${info.loras} active LoRA(s) read as "NOT IN PROMPT" `
                       + `for that reason alone. Connect the prompt, or read `
                       + `the report's note.` };
    }
    return null;
}

app.registerExtension({
    name: "Polyhedron.Inspector.Toast",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE) return;

        const origExec = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            origExec?.apply(this, arguments);
            try {
                const arr = message?.pls_inspector;
                const notice = runNotice(Array.isArray(arr) ? arr[0] : arr);
                if (!notice) { this._plsInspectorSig = null; return; }
                // One notice per state (the v492 lesson): onExecuted fires on
                // every run, so an unchanged situation must not stack up. A
                // CHANGED one speaks again.
                if (notice.sig === this._plsInspectorSig) return;
                this._plsInspectorSig = notice.sig;
                toast(notice.severity, notice.summary, notice.detail,
                      notice.life);
            } catch (e) {
                console.warn("[PLS Inspector] toast hook:", e);
            }
        };
    },
});
