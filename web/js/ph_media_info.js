/*
 * ph_media_info.js — v391
 *
 * Persistent one-line readout for the ⬡ Polyhedron Media Info node.
 *
 * The backend (ph_media_info.ULSMediaInfo.info) hands us the already-formatted
 * line plus the raw numbers over the UI channel ({"ui": {"pls_mediainfo": [...]}}),
 * so we never parse anything — we just drop the text into a small DOM widget on
 * the node. Mirrors the onExecuted/UI-channel pattern used by the Token Counter
 * (uls_token_toast.js) and the 3D Cockpit (ph_viewport3d.js), but renders an
 * in-node label instead of a transient toast: the dimensions stay visible.
 */

import { app } from "../../scripts/app.js";

const READOUT_CSS =
    "font-family:monospace; font-size:11px; line-height:1.4; color:#cfe7ff;" +
    "background:#1c1c1c; border:1px solid #333; border-radius:6px;" +
    "padding:7px 9px; margin:2px 0; box-sizing:border-box;" +
    "white-space:nowrap; overflow:hidden; text-overflow:ellipsis;";

app.registerExtension({
    name: "Polyhedron.MediaInfo.Readout",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "ULSMediaInfo") return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onCreated?.apply(this, arguments);
            try {
                const el = document.createElement("div");
                el.style.cssText = READOUT_CSS;
                el.textContent = "— run to read —";
                this.addDOMWidget("ph_mediainfo_readout", "div", el, { serialize: false });
                this._phInfoEl = el;
                // give the one-line readout room so it isn't clipped
                const w = Math.max((this.size && this.size[0]) || 0, 280);
                const h = Math.max((this.size && this.size[1]) || 0, 200);
                this.setSize([w, h]);
            } catch (e) {
                console.warn("[PLS MediaInfo] create:", e);
            }
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            try {
                const arr = message?.pls_mediainfo;
                const info = Array.isArray(arr) ? arr[0] : arr;
                if (info && this._phInfoEl) this._phInfoEl.textContent = info.text || "";
            } catch (e) {
                console.warn("[PLS MediaInfo] readout:", e);
            }
        };
    },
});
