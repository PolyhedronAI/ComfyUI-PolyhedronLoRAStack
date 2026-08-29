/*
 * uls_token_toast.js — v318
 *
 * Raises a NATIVE ComfyUI toast (app.extensionManager.toast) when the
 * ⬡ Polyhedron Token Counter runs and the prompt is over (or near) the
 * model's token limit.
 *
 * Why a toast and not just the report: the user is often working elsewhere on
 * the graph and can't see — or doesn't want to open — the counter's text
 * output. ComfyUI's toast is the same transient top-right notice the core uses
 * for warnings: theme-aware, non-blocking, visible regardless of where focus
 * is. The counter still prints the full report; this is the "look over here"
 * nudge on top.
 *
 * The backend (uls_stack_node.ULSTokenCounter.count) hands us structured
 * numbers via the UI channel ({"ui": {"pls_tokens": [...]}}), so we never
 * parse the report string. Mirrors the onExecuted pattern used by the 3D
 * Cockpit (ph_viewport3d.js).
 */

import { app } from "../../scripts/app.js";

// Documented toast API with a graceful fallback (same approach as
// uls_compat.js notify()). Returns silently if no toast manager exists.
function toast(severity, summary, detail, life) {
    try {
        const tm = app.extensionManager?.toast;
        if (tm?.add) { tm.add({ severity, summary, detail, life }); return; }
        if (tm?.addAlert) { tm.addAlert(detail); return; }
    } catch (e) { /* console-only fallback below */ }
    // Last resort: at least leave a console trace.
    console.warn(`[PLS Tokens] ${summary} — ${detail}`);
}

app.registerExtension({
    name: "Polyhedron.TokenCounter.Toast",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "ULSTokenCounter") return;

        const origExec = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            origExec?.apply(this, arguments);
            try {
                const arr = message?.pls_tokens;
                const info = Array.isArray(arr) ? arr[0] : arr;
                if (!info) return;

                // v492: de-duplicate. onExecuted fires every run; the over_limit toast
                // is life:0 (sticky), so without a guard identical notices stack up. One
                // signature per (state, worst, limit); re-notify only when it changes (a
                // changed token count, or over->ok->over). Per-node state, reset on reload
                // (one fresh warning per session is intended).
                const worst = Math.max(info.pos, info.neg);
                const sig = info.over_limit ? `over:${worst}:${info.limit}`
                          : info.near_limit ? `near:${worst}:${info.limit}`
                          : "ok";
                if (sig === this._plsLastTokenSig) return;
                this._plsLastTokenSig = sig;

                if (info.over_limit) {
                    const over = worst - info.limit;
                    // v908 -- THE TOAST USED TO LIE, LOUDLY.
                    //
                    // It promised "may be silently truncated or crash kijai's
                    // WanVideoSampler" for EVERY over-budget run, on every
                    // encoder. On MiniMax H3 both halves are false: qwen3vl
                    // carries max_length=99999999 and there is no kijai sampler
                    // anywhere in that graph. Frank trimmed a whole prompt
                    // session against that sentence. A warning nobody can act
                    // on correctly is worse than none, because it is the one
                    // thing that pops up on its own.
                    //
                    // The backend now sends `can_truncate` (does any live
                    // encoder actually have a cap) and `encoder`. With no clip
                    // wired can_truncate is true, so the old caveat still
                    // appears exactly where it is still warranted.
                    const named = info.encoder ? ` (${info.encoder})` : "";
                    const body = info.can_truncate
                        ? `Prompt is ${worst}/${info.limit} tokens (over by ${over}). ` +
                          `It may be silently truncated or crash kijai's WanVideoSampler. ` +
                          `Shorten the prompt or route through WanVideoTextEncode.`
                        : `Prompt is ${worst}/${info.limit} tokens (over by ${over}). ` +
                          `Nothing is truncated${named} — this is your own budget, ` +
                          `not a cap. See the report for what length really costs here.`;
                    toast(
                        "warn",
                        info.can_truncate ? "Token limit exceeded" : "Over your token budget",
                        body,
                        0,   // life 0 = sticky: an over-limit run must not auto-vanish
                    );
                } else if (info.near_limit) {
                    toast(
                        "warn",
                        "Token budget almost full",
                        `Prompt is ${worst}/${info.limit} tokens ` +
                        `(warn at ${info.warn_at}). Quality may start to degrade.`,
                        6000,
                    );
                }
            } catch (e) {
                console.warn("[PLS Tokens] toast hook:", e);
            }
        };
    },
});
