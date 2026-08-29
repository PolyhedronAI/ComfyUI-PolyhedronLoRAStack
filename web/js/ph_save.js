/*
 * ph_save.js -- v536
 *
 * ONE in-node preview for the ⬡ Polyhedron Save node. A single container widget
 * holds EITHER an <img> (still / GIF / animated WebP) OR a <video> -- never both
 * stacked. The backend hands the result over one channel: {"ui":{"ph_save":[{
 * filename, subfolder, type, kind:"image"|"video", format}]}}. The <video> uses
 * plain controls by default (single-frame clips never jump); an opt-in autoplay
 *
 * The media_kind selector (auto/image/video) greys out the branch that doesn't
 * apply (disabled in place, the Sampler's house pattern): "image" disables the
 * video widgets, "video" disables the image widgets, "auto" enables all.
 */

import { app } from "../../scripts/app.js";
import { setHidden, refit } from "./ph_widget_vis.js";

// v531 DIAG: per-file cache evidence. Every JS file caches INDIVIDUALLY in the
// browser, so the Cockpit banner cannot vouch for THIS file being fresh. If this
// line is missing from the console, Firefox is still serving an old ph_save.js
// (Ctrl+Shift+R). Drop with the other v531 diagnostics once Bug A is confirmed.
console.info("[PLS v536 DIAG] ph_save.js v542 loaded (preview aspect-fit active)");

const IMAGE_WIDGETS = ["image_format", "image_quality"];
const VIDEO_WIDGETS = ["video_preset", "quality", "frame_rate", "autoplay",
                       "pingpong", "loop_count", "trim_to_audio"];

// v529 (Bug A): the in-node preview follows the media's aspect ratio instead of a
// fixed floor. These bound the reserved preview height (px). Same house measures the
// Empty-Latent / Media-Loader nodes use.
const PREVIEW_MIN_H = 120;   // floor for the reserved preview height
const PREVIEW_MAX_H = 1400;  // ceiling for the reserved preview height (drag-to-upscale)
const PREVIEW_DEF_H = 260;   // default reserved height before the media loads
const NODE_MIN_W    = 300;   // node width floor (fits the Save widget rows)
const PREV_MARGIN   = 8;     // left+right breathing room inside the preview box
const PREV_PAD      = 8;     // vertical pad added to the fitted image height
const BAR_H         = 26;    // v533: shared play/pause + scrubber control bar height

function viewURL(item) {
    const p = new URLSearchParams({
        filename: item.filename || "",
        subfolder: item.subfolder || "",
        type: item.type || "output",
    });
    return `/view?${p.toString()}&r=${Date.now()}`;   // cache-bust each run
}

function getWidget(node, name) {
    return (node.widgets || []).find((w) => w.name === name);
}

function applyMediaKindGrey(node) {
    try {
        const kind = getWidget(node, "media_kind")?.value || "auto";
        let moved = false;
        for (const n of IMAGE_WIDGETS) {
            const w = getWidget(node, n);
            if (!w) continue;
            w.disabled = (kind === "video");
            // v888: a row that cannot bite is GONE now, not greyed. It keeps
            // its slot in node.widgets, so widgets_values is untouched (#577).
            if (setHidden(w, kind === "video")) moved = true;
        }
        for (const n of VIDEO_WIDGETS) {
            const w = getWidget(node, n);
            if (!w) continue;
            w.disabled = (kind === "image");
            if (setHidden(w, kind === "image")) moved = true;
        }
        if (moved) refit(node);
        app.graph?.setDirtyCanvas(true, true);
    } catch (e) {
        console.warn("[PLS Save] grey:", e);
    }
}

app.registerExtension({
    name: "Polyhedron.Save.Preview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "ULSSave") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            try {
                const box = document.createElement("div");
                box.style.cssText =
                    "width:100%; display:flex; justify-content:center; margin:2px 0;";
                // v542: hideOnZoom:false -- without it the preview DOM element is hidden
                // by the frontend below the low-quality zoom threshold (the 3D nodes
                // already opt out). Canvas-drawn core previews never had this problem.
                const pw = this.addDOMWidget("ph_save_preview", "div", box,
                    { serialize: false, hideOnZoom: false });
                this._phBox = box;

                // v529 (Bug A): reserve a preview height derived from the media aspect
                // ratio (set on load), NOT a fixed floor. computeSize returns the STORED
                // budget so the node never feeds its own size back into layout (feedback-
                // free; only onExecuted's onload or a lower-edge drag change it). This
                // mirrors the ph_empty_latent.js reserved-height mechanic, adapted to the
                // DOM widget, so the node hugs the image instead of squashing it small.
                this._phPrevH = PREVIEW_DEF_H;      // reserved preview height (px)
                this._phBudget = PREVIEW_DEF_H;     // v535: drag-controlled height budget
                this._phAR = null;                  // last media aspect ratio (w/h)
                const pWidget = pw || getWidget(this, "ph_save_preview");
                if (pWidget) pWidget.computeSize = (width) => [width, this._phPrevH];

                const _prevResize = this.onResize;
                this.onResize = function (size) {
                    // v536: firm node bounds so the widget rows can NEVER telescope /
                    // overlap. One computeSize() call gives LiteGraph's own minimum for
                    // BOTH axes (it already accounts for the input slots + every widget
                    // row, with our preview reserve as the last row).
                    const cs = this.computeSize();
                    if (size && size.length >= 1) {
                        const minW = Math.max(NODE_MIN_W, Math.ceil(cs[0] || 0));
                        if (size[0] < minW) size[0] = minW;
                    }
                    if (size && size.length >= 2) {
                        // widgetStack = everything above the preview (slots + rows + title);
                        // stable because cs[1] = widgetStack + this._phPrevH. The height can
                        // never drop below the full stack + a minimum preview -> the rows
                        // always keep their own space (media-loader v461 floor philosophy).
                        const bar = this._phHasBar ? BAR_H : 0;
                        const widgetStack = Math.max(0, Math.ceil(cs[1]) - this._phPrevH);
                        const floorTotal = widgetStack + PREVIEW_MIN_H + PREV_PAD + bar;
                        if (size[1] < floorTotal) size[1] = floorTotal;
                        // the lower-edge drag sets the preview height BUDGET; _phRelayout
                        // fits the media into (node width x budget) and reserves exactly
                        // the fitted box, so the node hugs the viewer with no dead space.
                        this._phBudget = Math.max(PREVIEW_MIN_H, Math.min(PREVIEW_MAX_H,
                            (size[1] - widgetStack) - PREV_PAD - bar));
                        if (this._phRelayout) this._phRelayout(false);
                        // snap the node to hug the fitted box once the drag settles --
                        // removes over-drag dead space and enforces the width-limited
                        // maximum (RAF + deadband, converges like v461). The snap target is
                        // computeSize()[1] which is >= floorTotal, so rows never overlap.
                        if (this._phSnapRAF) cancelAnimationFrame(this._phSnapRAF);
                        this._phSnapRAF = requestAnimationFrame(() => {
                            this._phSnapRAF = 0;
                            try {
                                const fitH = this.computeSize()[1];
                                if (this.size && Math.abs(this.size[1] - fitH) > 2) {
                                    this.setSize([this.size[0], fitH]);
                                    this.setDirtyCanvas?.(true, true);
                                }
                            } catch (e) { /* ignore */ }
                        });
                    }
                    const r = _prevResize ? _prevResize.apply(this, arguments) : undefined;
                    return r;
                };

                // hook the media_kind selector so greying follows the choice
                const mk = getWidget(this, "media_kind");
                if (mk) {
                    const prev = mk.callback;
                    mk.callback = function () {
                        const r = prev ? prev.apply(this, arguments) : undefined;
                        applyMediaKindGrey(this.node || this);
                        return r;
                    }.bind(this);
                }
                applyMediaKindGrey(this);
            } catch (e) {
                console.warn("[PLS Save] create:", e);
            }
        };

        // v531: after a workflow load the SERIALIZED node size is restored -- which
        // is exactly the state the Bug-A report shows ("klitzeklein" right after a
        // restart). Re-fit once the DOM settles (double rAF): derive the preview
        // budget from the RESTORED height (respects a size the user chose within
        // bounds), then repair the node height if it sits outside stack+budget.
        // Height only, width untouched, 8px deadband -- the gizmo's proven v509
        // mechanic against both the squeeze and the old one-way ratchet.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (o) {
            const r = onConfigure?.apply(this, arguments);
            try {
                requestAnimationFrame(() => requestAnimationFrame(() => {
                    try {
                        if (!this.computeSize || !this.size) return;
                        // v535: a saved node height carries the user's chosen preview size.
                        // Recover the budget from it so the dragged size persists across a
                        // reload; the box is fitted on the next execute (_phRelayout), or
                        // now if the media is already present.
                        const bar = this._phHasBar ? BAR_H : 0;
                        const stackAbove = this.computeSize()[1] - this._phPrevH;
                        const b = (this.size[1] - stackAbove) - PREV_PAD - bar;
                        if (isFinite(b)) {
                            this._phBudget = Math.max(PREVIEW_MIN_H, Math.min(PREVIEW_MAX_H, b));
                        }
                        if (this._phRelayout) this._phRelayout(true);
                    } catch (e) { /* sizing must never take the node down */ }
                }));
            } catch (e) { /* ignore */ }
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            try {
                if (!this._phBox) return;
                // v533/v534: ONE preview for both media kinds. A video and an image
                // sequence are the same thing -- N frames at fps -- so they share ONE
                // control bar: play/pause + scrubber + "i / N" counter + a Loop toggle
                // (default OFF -> plays once and stops). Muted (browser autoplay rule).
                // Standard reference: VideoHelperSuite renders <video muted loop controls
                // controlslist="nodownload noremoteplayback noplaybackrate">; we drive a
                // custom bar so the still sequence (separate PNGs, no native player) looks
                // and behaves IDENTICALLY to the video, and dragging the node's lower edge
                // scales the whole preview cleanly (see onResize).
                const raw = message?.ph_save;
                const items = (Array.isArray(raw) ? raw : (raw ? [raw] : []))
                    .filter((it) => it && it.filename);
                if (this._phFlipTimer) { clearInterval(this._phFlipTimer); this._phFlipTimer = null; }
                this._phBox.replaceChildren();
                if (!items.length) return;

                const last = items[items.length - 1];
                const stills = items.filter((it) => it.kind !== "video");
                const isVideo = last.kind === "video";
                const N = isVideo ? Math.max(1, parseInt(last.frames, 10) || 1) : stills.length;
                const fpsW = getWidget(this, "frame_rate");
                const rawFps = isVideo
                    ? (parseFloat(last.frame_rate) || parseFloat(fpsW && fpsW.value) || 24)
                    : (parseFloat(fpsW && fpsW.value) || 8);
                const fps = Math.max(1, Math.min(60, rawFps));
                const hasBar = N > 1;
                this._phHasBar = hasBar;                 // onResize reserves room for the bar
                this._phApplyLoop = null;                // each driver registers its loop hook
                if (this._phLoop === undefined) this._phLoop = false;  // persists across runs

                // ── layout: media row on top, shared bar below ──
                const wrap = document.createElement("div");
                wrap.style.cssText =
                    "display:flex; flex-direction:column; align-items:center; width:100%;";
                const mediaRow = document.createElement("div");
                mediaRow.style.cssText =
                    "position:relative; display:flex; justify-content:center; width:100%;";
                wrap.appendChild(mediaRow);

                let fitLogged = false;
                // v535: fit the media's aspect ratio into (node width x height budget) and
                // size the element EXPLICITLY to that box, so the reserved height equals
                // the pixels shown -> the node hugs the viewer (no dead space) and a
                // lower-edge drag scales the media up/down between a min and a
                // (width-limited) max. doFit snaps the node to hug; onResize passes false
                // (it snaps via its own RAF).
                this._phRelayout = (doFit) => {
                    if (!this._phAR || !this._phMedia) return;
                    const nodeW = (this.size && this.size[0]) || NODE_MIN_W;
                    const availW = Math.max(64, nodeW - 2 * PREV_MARGIN);
                    const budgetH = Math.max(PREVIEW_MIN_H,
                        Math.min(PREVIEW_MAX_H, this._phBudget || PREVIEW_DEF_H));
                    let boxH = budgetH, boxW = budgetH * this._phAR;
                    if (boxW > availW) { boxW = availW; boxH = availW / this._phAR; }
                    boxW = Math.round(boxW); boxH = Math.round(boxH);
                    this._phMedia.style.width = boxW + "px";
                    this._phMedia.style.height = boxH + "px";
                    this._phPrevH = boxH + PREV_PAD + (this._phHasBar ? BAR_H : 0);
                    if (doFit) {
                        const fitH = this.computeSize()[1];        // stack + _phPrevH
                        const curH = (this.size && this.size[1]) || 0;
                        if (Math.abs(curH - fitH) > 2) this.setSize([this.size[0], fitH]);
                    }
                    app.graph?.setDirtyCanvas(true, true);
                };
                const fitToMedia = (mw, mh) => {
                    if (!mw || !mh) return;
                    this._phAR = mw / mh;
                    if (!fitLogged) {
                        console.info("[PLS v536 DIAG] Save preview fit: " + mw + "x" + mh
                                     + " AR=" + this._phAR.toFixed(3)
                                     + " kind=" + (isVideo ? "video" : "stills")
                                     + " N=" + N + " budget=" + Math.round(this._phBudget));
                        fitLogged = true;
                    }
                    this._phRelayout(true);
                };

                // ── shared control bar (only when there is something to play) ──
                const BTN = "flex:0 0 auto; width:24px; height:18px; cursor:pointer;"
                    + "background:#0006; color:#cfe3d8; border:1px solid #ffffff22;"
                    + "border-radius:3px; font:12px monospace; line-height:1; padding:0;";
                let btn = null, slider = null, readout = null;
                if (hasBar) {
                    const bar = document.createElement("div");
                    bar.style.cssText =
                        "display:flex; align-items:center; gap:6px; width:100%;"
                        + "box-sizing:border-box; padding:3px 4px 0; height:" + BAR_H + "px;"
                        + "font:11px monospace; color:#cfe3d8;";
                    btn = document.createElement("button");
                    btn.style.cssText = BTN;
                    slider = document.createElement("input");
                    slider.type = "range"; slider.min = "0"; slider.step = "1"; slider.value = "0";
                    slider.style.cssText =
                        "flex:1 1 auto; min-width:40px; height:14px; cursor:pointer;"
                        + "accent-color:#7fb3d1;";
                    readout = document.createElement("span");
                    readout.style.cssText =
                        "flex:0 0 auto; min-width:54px; text-align:right; white-space:nowrap;";
                    // v534: Loop toggle in the old fullscreen slot. Default OFF -> the
                    // media plays through once and stops; ON -> it loops. Glyph is a
                    // loop/refresh spinner; the state persists across runs on the node.
                    const loopBtn = document.createElement("button");
                    loopBtn.textContent = "\u27f3";       // loop / refresh spinner
                    const styleLoop = () => {
                        loopBtn.style.cssText = BTN + (this._phLoop
                            ? "background:#2b6a8f; color:#eaf6ff; border-color:#7fb3d1;" : "");
                        loopBtn.title = this._phLoop
                            ? "Loop: ON (click = play once)" : "Loop: OFF (click = loop)";
                    };
                    styleLoop();
                    loopBtn.addEventListener("click", (e) => {
                        e.stopPropagation();
                        this._phLoop = !this._phLoop;
                        styleLoop();
                        if (this._phApplyLoop) this._phApplyLoop();
                    });
                    bar.appendChild(btn);
                    bar.appendChild(slider);
                    bar.appendChild(readout);
                    bar.appendChild(loopBtn);
                    wrap.appendChild(bar);
                }

                // ── media element + its driver ──
                let el;
                if (isVideo) {
                    el = document.createElement("video");
                    el.muted = true;                        // browser autoplay requirement
                    el.loop = !!this._phLoop;               // v534: loop only when toggled on
                    el.autoplay = true;                     // plays through once immediately
                    el.playsInline = true;
                    el.preload = "auto";
                    el.setAttribute("controlslist",
                                    "nodownload noremoteplayback noplaybackrate");
                    el.onloadedmetadata = () => fitToMedia(el.videoWidth, el.videoHeight);
                    if (hasBar) {
                        const glyph = () => { btn.textContent = el.paused ? "\u25b6" : "\u23f8"; };
                        const toggle = () => {
                            if (el.paused) {
                                if (el.ended || el.currentTime >= (el.duration || 0)) el.currentTime = 0;
                                const p = el.play(); if (p && p.catch) p.catch(() => {});
                            } else el.pause();
                        };
                        btn.addEventListener("click", (e) => { e.stopPropagation(); toggle(); });
                        el.addEventListener("click", toggle);
                        el.addEventListener("play", glyph);
                        el.addEventListener("pause", glyph);
                        el.addEventListener("ended", glyph);   // v534: show play when it stops
                        el.addEventListener("timeupdate", () => {
                            const d = el.duration || (N / fps);
                            if (d > 0) {
                                slider.value = String(Math.round((el.currentTime / d) * 1000));
                                const fr = Math.min(N, Math.floor(el.currentTime * fps) + 1);
                                readout.textContent = fr + " / " + N;
                            }
                        });
                        slider.max = "1000";
                        slider.addEventListener("input", () => {
                            const d = el.duration || (N / fps);
                            el.currentTime = (parseFloat(slider.value) / 1000) * d;
                        });
                        readout.textContent = "1 / " + N;
                        glyph();
                    }
                    el.src = viewURL(last);
                    this._phMedia = el;
                    this._phApplyLoop = () => {
                        el.loop = !!this._phLoop;
                        if (this._phLoop && el.paused) {
                            const p = el.play(); if (p && p.catch) p.catch(() => {});
                        }
                    };
                } else if (N > 1) {
                    el = document.createElement("img");
                    const urls = stills.map(viewURL);       // built ONCE per run
                    let i = 0, playing = true;
                    const glyph = () => { btn.textContent = playing ? "\u23f8" : "\u25b6"; };
                    const show = () => {
                        el.src = urls[i];
                        slider.value = String(i);
                        readout.textContent = (i + 1) + " / " + N;
                    };
                    slider.max = String(N - 1);
                    this._phFlipTimer = setInterval(() => {
                        if (!playing) return;
                        if (i >= N - 1) {
                            if (this._phLoop) { i = 0; show(); }   // loop on -> wrap
                            else { playing = false; glyph(); }     // v534: play once, stop at last
                        } else { i = i + 1; show(); }
                    }, Math.round(1000 / fps));
                    const toggle = () => {
                        if (!playing && i >= N - 1) i = 0;         // replay from the start
                        playing = !playing; glyph();
                        if (playing) show();
                    };
                    btn.addEventListener("click", (e) => { e.stopPropagation(); toggle(); });
                    el.style.cursor = "pointer";
                    el.addEventListener("click", toggle);
                    slider.addEventListener("input", () => {
                        playing = false; glyph();
                        i = parseInt(slider.value, 10) || 0;
                        show();
                    });
                    el.onload = () => fitToMedia(el.naturalWidth, el.naturalHeight);
                    glyph();
                    show();
                    this._phMedia = el;
                    this._phApplyLoop = () => {
                        if (this._phLoop && !playing) {
                            if (i >= N - 1) i = 0;
                            playing = true; glyph(); show();
                        }
                    };
                } else {
                    // single still (or 1-frame clip): just the frame, no bar to play
                    el = document.createElement("img");
                    el.onload = () => fitToMedia(el.naturalWidth, el.naturalHeight);
                    el.src = viewURL(last);
                    this._phMedia = el;
                }

                // v535: the element is sized EXPLICITLY by _phRelayout (box-fit); object-
                // fit keeps the aspect if the box rounds off. No max-width/height:auto --
                // that pinned the media to natural size and left dead space below.
                el.style.cssText +=
                    ";display:block; object-fit:contain; border-radius:6px; background:#000;";
                mediaRow.appendChild(el);
                this._phBox.appendChild(wrap);

                // v536: enforce the width bound on execute too (rows can't telescope)
                const _minW = Math.max(NODE_MIN_W, Math.ceil((this.computeSize()[0]) || 0));
                if (this.size && this.size[0] < _minW) {
                    this.setSize([_minW, this.size[1]]);
                }
                app.graph?.setDirtyCanvas(true, true);
            } catch (e) {
                console.warn("[PLS Save] preview:", e);
            }
        };

        // v532: stop the flipbook timer when the node goes away.
        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            try {
                if (this._phFlipTimer) { clearInterval(this._phFlipTimer); this._phFlipTimer = null; }
            } catch (e) { /* ignore */ }
            return onRemoved?.apply(this, arguments);
        };
    },
});
