/**
 * Ultimate LoRA Stack — Frontend v2
 * ══════════════════════════════════
 * LiteGraph integration without DOM-node assumptions:
 *  - Everything drawn on canvas (onDrawForeground)
 *  - Preview popup = real DOM overlay, positioned via
 *    canvas.getBoundingClientRect() + ds.scale/offset
 *  - Mouse events: onMouseMove / onMouseDown / onMouseUp / onMouseLeave
 *  - State serialized via hidden "uls_config" widget → Python reads it
 *  - Drag-to-reorder with visual indicator line
 *  - Conflict analysis: order + weight sums
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

console.log("[Polyhedron LoRA Stack] uls_node.js loaded ✓");

const NODE_TYPE        = "UltimateLoraStack";
const NODE_TYPE_ENGINE = "ULSAccelerator";   // Polyhedron Engine

// ── v263: HTML-escape for file/network-derived strings ────────────────────
// LoRA names, folder paths, trigger words and safetensors metadata all flow
// from disk or Civitai — i.e. they are NOT trusted. Anywhere such a value is
// interpolated into innerHTML it must be escaped first, or a crafted LoRA
// (e.g. a trigger field containing `<img src=x onerror=…>`) could inject
// script into the ComfyUI page. Static template markup needs no escaping;
// only interpolated foreign data does. (textContent would also be safe, but
// several sites mix trusted markup with a single untrusted value, so a scalar
// escaper is the smaller, surgical change.)
function escapeHtml(s) {
    if (s == null) return "";
    return String(s)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

// Layout
const PAD        = 8;
const ROW_H      = 28;
const HEADER_H   = 130; // LiteGraph title(30) + 5 output pins × ~20px = ~130px
const FOOTER_H   = 0;   // Slider removed — footer no longer needed
const OUT_W      = 0;   // Output pins extend outside node rect — no space reserved
// Content area: 0 to W
// Row layout left→right: ▲▼(14) + CB(12) + Thumb(30) + Name(flex) + ↵(22) + GRP(36) + Weight(52)
// Last "row" = "+" add button

const GROUPS = ["—", "acc", "style", "scene", "motion", "subject", "detail", "custom"];
const GROUP_COLORS = {
    "acc":     "#e85d5d",
    "style":     "#8b6fe8",
    "scene":    "#4a9eff",
    "motion":    "#43c9c9",
    "subject": "#ff6b9d",
    "detail":    "#51cf66",
    "custom":    "#ff8c42",
    "—":         "#404050",
};

// ─── Caches ────────────────────────────────────────────────────────────────

const metaCache    = new Map();  // name → metadata object | null
const previewCache = new Map();  // name → { img, vid, loaded }

async function fetchMeta(name) {
    if (metaCache.has(name)) return metaCache.get(name);
    try {
        const r = await api.fetchApi(`/uls/metadata?lora=${encodeURIComponent(name)}`);
        const d = r.ok ? await r.json() : null;
        metaCache.set(name, d);
        return d;
    } catch { metaCache.set(name, null); return null; }
}

function ensurePreview(name) {
    if (!name || name === "None" || previewCache.has(name)) return;
    const entry = { img: null, vid: null, loaded: false };
    previewCache.set(name, entry);
    fetchMeta(name).then(meta => {
        if (!meta) { entry.loaded = true; return; }
        let pending = 0;
        if (meta.has_preview_image) {
            pending++;
            const img = new Image();
            img.onload  = () => { entry.img = img; if (--pending === 0) entry.loaded = true; app.graph?.setDirtyCanvas(true, false); };
            img.onerror = () => { if (--pending === 0) entry.loaded = true; };
            img.src = api.apiURL(`/uls/preview/image?lora=${encodeURIComponent(name)}`);
        }
        if (meta.has_preview_video) {
            pending++;
            const vid = document.createElement("video");
            vid.muted = true; vid.loop = true; vid.autoplay = false; vid.preload = "auto";
            vid.onloadeddata = () => {
                entry.vid = vid;
                // Ersten Frame als Standbild ins Cache rendern
                try {
                    const oc = document.createElement("canvas");
                    oc.width = 120; oc.height = 120;
                    const octx = oc.getContext("2d");
                    octx.drawImage(vid, 0, 0, 120, 120);
                    const snap = new Image();
                    snap.onload = () => {
                        if (!entry.img) entry.img = snap;  // nur wenn kein Standbild vorhanden
                        if (--pending === 0) entry.loaded = true;
                        app.graph?.setDirtyCanvas(true, false);
                    };
                    snap.onerror = () => { if (--pending === 0) entry.loaded = true; };
                    snap.src = oc.toDataURL("image/jpeg", 0.8);
                } catch {
                    if (--pending === 0) entry.loaded = true;
                }
            };
            vid.onerror = () => { if (--pending === 0) entry.loaded = true; };
            vid.src = api.apiURL(`/uls/preview/video?lora=${encodeURIComponent(name)}`);
        }
        if (pending === 0) entry.loaded = true;
    });
}

// ─── Globale LoRA-Liste ───────────────────────────────────────────────────

let _loraList = [];        // alle LoRAs inkl. Unterordner
let _loraListLoading = false;
let _loraListLoaded  = false;

async function loadLoraList() {
    if (_loraListLoading) return;
    _loraListLoading = true;
    try {
        const r = await api.fetchApi("/uls/list");
        const list = await r.json();
        _loraList = list.map(l => l.name);
        _loraListLoaded = true;
        console.log(`[ULS] LoRA list loaded: ${_loraList.length} LoRAs`);
        // Preload previews
        for (const item of list) ensurePreview(item.name);
    } catch(e) {
        console.warn("[ULS] Could not load LoRA list:", e);
    }
    _loraListLoading = false;
}

// Sofort laden wenn Extension initialisiert wird
loadLoraList();

// ─── Fokus-Tracker: merkt sich das zuletzt aktive Textfeld ────────────────
//
// ComfyUI CLIP Text Encode + ULS Dual Prompt Textareas sind echte DOM-
// Elemente. Wir tracken das zuletzt fokussierte Textfeld global, damit
// ein Klick auf eine LoRA-Zeile im Canvas den Trigger dort einfügen kann.
//
// Warum nicht document.activeElement? Weil der Canvas-Klick das Fokus
// sofort weg nimmt — wir brauchen den Wert von VOR dem Klick.

let _lastFocusedTextarea = null;
let _lastCursorPos       = 0;  // selectionStart zum Zeitpunkt des Blur

document.addEventListener("focusin", (e) => {
    const el = e.target;
    if (el.tagName === "TEXTAREA" || (el.tagName === "INPUT" && el.type === "text")) {
        _lastFocusedTextarea = el;
    }
}, true);

document.addEventListener("selectionchange", () => {
    const el = document.activeElement;
    if (el && (el.tagName === "TEXTAREA" || el.tagName === "INPUT")) {
        _lastCursorPos = el.selectionStart ?? 0;
    }
}, true);

document.addEventListener("focusout", (e) => {
    const el = e.target;
    if (el === _lastFocusedTextarea) {
        // Cursor-Position beim Verlassen des Feldes merken
        _lastCursorPos = el.selectionStart ?? el.value?.length ?? 0;
    }
}, true);

/**
 * Leitet alle Trigger-Words einer LoRA ab — gibt ein Array zurück.
 * Reihenfolge: längste zuerst (vollständigste zuerst).
 *
 * Strategie:
 *  1. Metadaten → alle komma-getrennten Trigger-Words
 *  2. Dateiname → abgeleiteter Fallback
 */
function deriveTriggers(loraName, meta) {
    // 1. Echte Trigger-Words aus Metadata
    if (meta) {
        let tw = meta.trigger_words || "";
        if (typeof tw !== "string") {
            // ss_tag_frequency ist ein Dict {tag: count}
            tw = Object.keys(tw).join(", ");
        }
        // JSON-artige Strings parsen
        if (tw.startsWith("{") || tw.startsWith("[")) {
            try {
                const parsed = JSON.parse(tw);
                tw = typeof parsed === "object" ? Object.keys(parsed).join(", ") : String(parsed);
            } catch {}
        }
        const parts = tw.split(",")
            .map(s => s.trim())
            .filter(s => s.length > 0 && s.length < 60);
        if (parts.length > 0) {
            // Längste zuerst
            return parts.sort((a, b) => b.length - a.length);
        }
    }

    // 2. Aus Dateiname ableiten
    const base = loraName
        .split(/[/\\]/).pop()
        .replace(/\.safetensors$/i, "")
        .replace(/^[^_]+-/, "");

    const cleaned = base
        .replace(/_?(high|low|hd|ld)_noise$/i, "")
        .replace(/_?(high|low)$/i, "")
        .replace(/wan\d+[\._]\d+_?/i, "")
        .replace(/polyhedron_?/i, "")
        .replace(/v\d+$/i, "")
        .replace(/_+/g, "_")
        .replace(/^_|_$/g, "");

    const parts = cleaned.split("_").filter(Boolean);
    if (parts.length > 0) {
        const candidate = parts[parts.length - 1];
        if (candidate.length >= 2) return [candidate];
    }

    return [base.split("_")[0] || base];
}

// Compat-Wrapper für einzelnen Trigger (Fallback)
function deriveTrigger(loraName, meta) {
    return deriveTriggers(loraName, meta)[0] || "";
}

/**
 * Fügt Text an der gespeicherten Cursor-Position im letzten Textfeld ein.
 * Danach wird ein natürliches Input-Event gefeuert damit ComfyUI
 * den geänderten Wert mitbekommt.
 */
function insertTriggerAtCursor(triggerText) {
    const el  = _lastFocusedTextarea;
    if (!el) {
        // Kein Textfeld fokussiert → Feedback ans Canvas
        return false;
    }

    const pos  = _lastCursorPos;
    const val  = el.value || "";

    // Leerzeichen-Padding: füge ein Leerzeichen vor/nach ein wenn nötig
    const before = val.slice(0, pos);
    const after  = val.slice(pos);
    const needSpaceBefore = before.length > 0 && !/\s$/.test(before);
    const needSpaceAfter  = after.length  > 0 && !/^\s/.test(after);

    const insert = (needSpaceBefore ? " " : "") + triggerText + (needSpaceAfter ? " " : "");
    const newVal = before + insert + after;
    const newPos = pos + insert.length;

    // Wert setzen (React-kompatibel für ComfyUI-Widgets)
    const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype, "value"
    )?.set;
    if (nativeInputValueSetter) {
        nativeInputValueSetter.call(el, newVal);
    } else {
        el.value = newVal;
    }

    // Cursor ans Ende des eingefügten Texts setzen
    el.selectionStart = newPos;
    el.selectionEnd   = newPos;
    _lastCursorPos    = newPos;

    // ComfyUI Events feuern
    el.dispatchEvent(new Event("input",  { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));

    // Textfeld kurz highlighten als visuelles Feedback
    const origBg = el.style.background;
    el.style.background = "rgba(139, 111, 232, 0.25)";
    setTimeout(() => { el.style.background = origBg; }, 400);

    return true;
}

/**
 * Polyhedron-styled confirm dialog (replaces the bare white window.confirm).
 * Matches the dark popup vocabulary used elsewhere in this file. Calls
 * onConfirm() if the user accepts; otherwise onCancel(). Positioned near the
 * click when screenPos is given, else centered.
 */
function showConfirmDialog({ title, message, confirmLabel = "OK", cancelLabel = "Cancel",
                            screenPos, onConfirm, onCancel }) {
    document.getElementById("uls-confirm")?.remove();
    const scale = (app.canvas?.ds?.scale) || 1;

    const overlay = document.createElement("div");
    overlay.id = "uls-confirm";
    overlay.style.cssText =
        "position:fixed;inset:0;z-index:1000000;background:rgba(0,0,0,0.45);";

    const box = document.createElement("div");
    box.style.cssText = [
        "position:absolute",
        "min-width:300px", "max-width:380px",
        "background:#14141e",
        "border:1px solid #7a3a3a",          // warm/alert border, like the conflict flash
        "border-radius:10px",
        "box-shadow:0 8px 32px rgba(0,0,0,0.7)",
        "overflow:hidden",
        "font:13px 'Segoe UI',Arial,sans-serif",
        "color:#e0e0ea",
    ].join(";");

    const head = document.createElement("div");
    head.style.cssText =
        "padding:10px 14px;background:#1c1018;border-bottom:1px solid #3a2a2a;" +
        "color:#ffb0b0;font-weight:bold;display:flex;align-items:center;gap:8px;";
    head.textContent = "⚠  " + title;
    box.appendChild(head);

    const body = document.createElement("div");
    body.style.cssText = "padding:14px;line-height:1.5;white-space:pre-wrap;color:#c8c8d4;";
    body.textContent = message;
    box.appendChild(body);

    const foot = document.createElement("div");
    foot.style.cssText =
        "padding:10px 14px;border-top:1px solid #2a2a3a;background:#101019;" +
        "display:flex;justify-content:flex-end;gap:8px;";
    const mkBtn = (label, primary) => {
        const b = document.createElement("button");
        b.textContent = label;
        b.style.cssText = [
            "padding:6px 16px", "border-radius:5px", "cursor:pointer",
            "font:bold 12px 'Segoe UI',Arial", "outline:none",
            primary ? "background:#3a5a8a" : "background:#2a2a3a",
            primary ? "border:1px solid #4a7ac0" : "border:1px solid #3a3a5a",
            primary ? "color:#dce8ff" : "color:#b0b0c0",
        ].join(";");
        return b;
    };
    const cancelBtn = mkBtn(cancelLabel, false);
    const okBtn = mkBtn(confirmLabel, true);
    foot.appendChild(cancelBtn); foot.appendChild(okBtn);
    box.appendChild(foot);

    overlay.appendChild(box);
    document.body.appendChild(overlay);

    // position
    requestAnimationFrame(() => {
        const r = box.getBoundingClientRect();
        let x, y;
        if (screenPos) {
            x = screenPos.x; y = screenPos.y;
        } else {
            x = (window.innerWidth - r.width) / 2;
            y = (window.innerHeight - r.height) / 2;
        }
        x = Math.min(Math.max(8, x), window.innerWidth - r.width - 8);
        y = Math.min(Math.max(8, y), window.innerHeight - r.height - 8);
        box.style.left = `${x}px`; box.style.top = `${y}px`;
    });

    const done = (fn) => { overlay.remove(); document.removeEventListener("keydown", onKey, true); fn?.(); };
    const onKey = (ev) => {
        if (ev.key === "Enter")  { ev.preventDefault(); ev.stopPropagation(); done(onConfirm); }
        if (ev.key === "Escape") { ev.preventDefault(); ev.stopPropagation(); done(onCancel); }
    };
    okBtn.addEventListener("click", () => done(onConfirm));
    cancelBtn.addEventListener("click", () => done(onCancel));
    overlay.addEventListener("click", (e) => { if (e.target === overlay) done(onCancel); });
    document.addEventListener("keydown", onKey, true);
    requestAnimationFrame(() => okBtn.focus());
}

/**
 * Toast-Notification wenn kein Textfeld fokussiert war.
 * If nodePos is provided, toast appears above the node instead of bottom-center.
 */
function showInsertToast(text, ok, nodePos) {
    const existing = document.getElementById("uls-toast");
    existing?.remove();

    const toast = document.createElement("div");
    toast.id = "uls-toast";

    let posStyle;
    if (nodePos) {
        posStyle = `position:fixed; left:${nodePos.x}px; top:${nodePos.y - 44}px; transform:none;`;
    } else {
        posStyle = `position:fixed; bottom:24px; left:50%; transform:translateX(-50%);`;
    }

    toast.style.cssText = `
        ${posStyle}
        z-index:999999;
        background:${ok ? "#1a2a1a" : "#2a1a1a"};
        border:1px solid ${ok ? "#3a7a3a" : "#7a3a3a"};
        color:${ok ? "#8dff8d" : "#ff8d8d"};
        font:12px 'Segoe UI',Arial,sans-serif;
        padding:6px 14px; border-radius:6px;
        box-shadow:0 4px 20px rgba(0,0,0,0.6);
        pointer-events:none;
        transition: opacity 0.3s ease;
        white-space:nowrap;
    `;
    toast.textContent = ok
        ? `✓ "${text}" inserted`
        : `⚠ Click into a Prompt node first`;
    document.body.appendChild(toast);
    setTimeout(() => { toast.style.opacity = "0"; }, 1800);
    setTimeout(() => { toast.remove(); }, 2200);
}

/**
 * Zeigt ein kleines Popup mit allen verfügbaren Triggern zur Auswahl.
 * Klick auf einen Trigger fügt ihn im Format (trigger:weight) ein.
 */
function openTriggerSelectPopup(triggers, weight, e) {
    document.getElementById("uls-trigger-select")?.remove();

    const wrap = document.createElement("div");
    wrap.id = "uls-trigger-select";
    wrap.style.cssText = `
        position:fixed; left:${e.clientX}px; top:${e.clientY + 6}px;
        z-index:999999; min-width:200px; max-width:340px;
        background:#14141e; border:1px solid #3a3a5a;
        border-radius:8px; overflow:hidden;
        box-shadow:0 8px 32px rgba(0,0,0,.8);
        font:12px 'Segoe UI',Arial,sans-serif;
    `;

    const header = document.createElement("div");
    header.style.cssText = "padding:6px 10px;color:#666;font-size:10px;border-bottom:1px solid #2a2a3a;";
    header.textContent = "Select trigger — inserts as (trigger:weight)";
    wrap.appendChild(header);

    for (const trigger of triggers) {
        const item = document.createElement("div");
        item.style.cssText = "padding:7px 12px;cursor:pointer;color:#c0d0ff;border-bottom:1px solid #1e1e2a;display:flex;justify-content:space-between;align-items:center;";
        const label = document.createElement("span");
        label.textContent = trigger;
        const preview = document.createElement("span");
        preview.style.cssText = "color:#f0a030;font-size:10px;margin-left:8px;opacity:0.7;";
        preview.textContent = `(${trigger}:${weight})`;
        item.appendChild(label);
        item.appendChild(preview);
        item.addEventListener("mouseenter", () => item.style.background = "#28284a");
        item.addEventListener("mouseleave", () => item.style.background = "");
        item.addEventListener("mousedown", (ev) => {
            ev.preventDefault();
            const text = `(${trigger}:${weight})`;
            showInsertToast(text, insertTriggerAtCursor(text));
            wrap.remove();
            document.removeEventListener("pointerdown", closeH, true);
        });
        wrap.appendChild(item);
    }

    document.body.appendChild(wrap);

    // Viewport-Korrektur
    requestAnimationFrame(() => {
        const r = wrap.getBoundingClientRect();
        if (r.right  > window.innerWidth  - 8) wrap.style.left = `${window.innerWidth  - r.width  - 8}px`;
        if (r.bottom > window.innerHeight - 8) wrap.style.top  = `${e.clientY - r.height - 6}px`;
    });

    const closeH = (ev) => {
        if (!wrap.contains(ev.target)) {
            wrap.remove();
            document.removeEventListener("pointerdown", closeH, true);
        }
    };
    setTimeout(() => document.addEventListener("pointerdown", closeH, true), 100);
}

/**
 * Group apply-mode picker — opens on right-click of a GRP-pill.
 *
 * v252: this is a STAY-OPEN panel. The four base modes (SEQ / CONCAT /
 * Smooth-Mix Channel / Smooth-Mix Element) are mutually-exclusive radio rows;
 * picking one moves the marker live and persists immediately, but does NOT
 * close the popup. That lets the user then layer a Cleanup switch (Trim /
 * Resolve) on top and close deliberately — via the "Done" button, a
 * click outside, or Escape. The Cleanup section is live-gated: it greys out
 * for SEQ (nothing is side-by-side to clean) and lights up the moment a
 * CONCAT/DARE mode is chosen, without rebuilding the popup. A Trim value set
 * under CONCAT/DARE is KEPT (only greyed) when switching to SEQ and returns
 * if you switch back — SEQ ignores Trim on the backend anyway.
 *
 * The whole thing is a plain DOM overlay (no canvas painting), consistent with
 * the v250 cleanup-popup approach and the Nodes-2.0 migration design note.
 *
 * @param {string}   group            — group name ("character", "detail", ...)
 * @param {string}   currentMode      — "SEQ" | "CONCAT" | "DARE"
 * @param {string}   currentDareVariant — "channel" | "element"
 * @param {boolean}  currentTrim      — Trim switch state
 * @param {boolean}  currentResolve   — Resolve switch state (live since v256)
 * @param {object}   clickEvent       — original event for positioning
 * @param {function} onChange         — called with the new composite mode key after persistence kicks off
 * @param {function} onToggle         — called (which, value) when a Cleanup switch flips
 */
function showGroupModePopup(group, currentMode, currentDareVariant, currentTrim, currentResolve, currentTrimAmount, clickEvent, onChange, onToggle) {
    document.getElementById("uls-mode-popup")?.remove();

    const curDV   = currentDareVariant || "channel";
    let   trimOn  = !!currentTrim;
    let   resOn   = !!currentResolve;

    // v252: single source of truth for the active mode. Repainting reads it,
    // so the marker can move (and the Cleanup gating re-evaluate) without
    // tearing down and rebuilding the popup.
    //   "SEQ" | "CONCAT" | "DARE:channel" | "DARE:element"
    const _cur = (currentMode || "SEQ").toUpperCase();
    let   selectedKey = _cur === "DARE" ? ("DARE:" + curDV) : _cur;
    const isSeq = () => selectedKey === "SEQ";

    const wrap = document.createElement("div");
    wrap.id = "uls-mode-popup";
    wrap.style.cssText = `
        position:fixed; left:${clickEvent.clientX}px; top:${clickEvent.clientY + 6}px;
        z-index:999999; min-width:200px;
        background:#14141e; border:1px solid #4a3a6a;
        border-radius:8px; overflow:hidden;
        box-shadow:0 8px 32px rgba(0,0,0,.8);
        font:12px 'Segoe UI',Arial,sans-serif;
    `;

    const header = document.createElement("div");
    header.style.cssText = "padding:6px 10px;color:#aaa;font-size:10px;border-bottom:1px solid #2a2a3a;";
    header.textContent = `Group "${group}" — Apply mode`;
    wrap.appendChild(header);

    const MODES = [
        { key: "SEQ",          letter: "S",  color: "#7a7a8a",
          label: "Sequential (SEQ)",
          hint:  "stacks LoRAs one by one — classic, full effect of each"              },
        { key: "CONCAT",       letter: "C",  color: "#f0c050",
          label: "Combined (CONCAT)",
          hint:  "merges the group's LoRAs into one — gentler than stacking them separately"             },
        { key: "DARE:channel", letter: "D·C", color: "#40c0ff",
          label: "Smooth Mix — Channel Drop",
          hint:  "trims each LoRA's overlapping detail before merging, so they clash less" },
        { key: "DARE:element", letter: "D·E", color: "#7af0c0",
          label: "Smooth Mix — Element Drop",
          hint:  "thins each LoRA in tiny pieces — subtler and finer than Channel Drop" },
    ];

    // Build the four mode rows. Each keeps a repaint closure so the active
    // marker can move on selection without a rebuild.
    const modePainters = [];
    for (const m of MODES) {
        const item = document.createElement("div");

        const dot = document.createElement("div");
        dot.style.cssText = `
            width:24px; height:18px; border-radius:4px;
            background:${m.color};
            display:flex; align-items:center; justify-content:center;
            color:#1a1a2a; font-weight:bold; font-size:9px;
            flex-shrink:0; letter-spacing:-0.5px;
        `;
        dot.textContent = m.letter;
        item.appendChild(dot);

        const txt = document.createElement("div");
        txt.style.cssText = "flex:1; min-width:0;";
        const lbl = document.createElement("div");
        const hnt = document.createElement("div");
        hnt.style.cssText = "font-size:10px; color:#888; margin-top:2px;";
        hnt.textContent = m.hint;
        txt.appendChild(lbl); txt.appendChild(hnt);
        item.appendChild(txt);

        const paint = () => {
            const isActive = selectedKey === m.key;
            item.style.cssText = `
                padding:8px 10px; cursor:pointer; color:${isActive ? "#fff" : "#c0c0d0"};
                border-bottom:1px solid #1e1e2a;
                display:flex; align-items:center; gap:10px;
                background:${isActive ? m.color + "22" : "transparent"};
                border-left:3px solid ${isActive ? m.color : "transparent"};
            `;
            lbl.style.cssText = `font-weight:bold; color:${isActive ? m.color : "#d0d0e0"};`;
            lbl.textContent = m.label + (isActive ? "  ●" : "");
        };
        modePainters.push(paint);

        item.addEventListener("mouseenter", () => {
            if (selectedKey !== m.key) item.style.background = "#28284a";
        });
        item.addEventListener("mouseleave", paint);
        item.addEventListener("mousedown", (ev) => {
            ev.preventDefault();
            ev.stopPropagation();
            // v252: select + persist, then STAY OPEN and repaint everything so
            // the marker moves and Cleanup re-gates. No wrap.remove() here.
            if (selectedKey === m.key) return;   // re-click on the active row → no-op
            selectedKey = m.key;
            onChange?.(m.key);
            paintModes();
            paintCleanup();
        });

        wrap.appendChild(item);
        paint();
    }

    function paintModes() { for (const p of modePainters) p(); }

    // ── Cleanup switches (v250; live-gated since v252) ───────────────────
    // Sit BESIDE the four base modes: four familiar rows stay, two switches
    // layer on top — the combinatorics live behind the lean UI. Greyed for SEQ
    // (SEQ never puts LoRAs side-by-side, so there is nothing to compare/clean).
    // Toggling does NOT close the popup.
    const sub = document.createElement("div");
    sub.style.cssText = "padding:6px 10px 4px;color:#888;font-size:10px;"
        + "border-top:1px solid #2a2a3a;border-bottom:1px solid #1e1e2a;"
        + "letter-spacing:.3px;";
    wrap.appendChild(sub);

    const togglePainters = [];

    function makeToggle(opts) {
        const { color, label, hint, get, set, disabled } = opts;
        const row = document.createElement("div");
        const box = document.createElement("div");
        const txt = document.createElement("div");
        const lbl = document.createElement("div");
        const hnt = document.createElement("div");
        const paint = () => {
            const on  = get();
            const dim = disabled || isSeq();
            row.style.cssText = `
                padding:8px 10px; cursor:${dim ? "default" : "pointer"};
                border-bottom:1px solid #1e1e2a;
                display:flex; align-items:center; gap:10px;
                opacity:${dim ? 0.4 : 1};
                background:${on && !dim ? color + "22" : "transparent"};
                border-left:3px solid ${on && !dim ? color : "transparent"};
            `;
            box.textContent     = on ? "✔" : "";
            box.style.background = on && !dim ? color : "transparent";
            box.style.color      = on && !dim ? "#1a1a2a" : "#666";
            lbl.style.color      = on && !dim ? color : "#d0d0e0";
        };
        box.style.cssText = `width:24px;height:18px;border-radius:4px;border:1px solid ${color};
            display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:11px;flex-shrink:0;`;
        row.appendChild(box);
        txt.style.cssText = "flex:1; min-width:0;";
        lbl.style.cssText = "font-weight:bold;";
        lbl.textContent   = label;
        hnt.style.cssText = "font-size:10px; color:#888; margin-top:2px;";
        hnt.textContent   = hint;
        txt.appendChild(lbl); txt.appendChild(hnt);
        row.appendChild(txt);

        // v252: listeners attached unconditionally; the handler bails while the
        // switch is dimmed (disabled, or SEQ active). This lets the row go live
        // the instant a CONCAT/DARE mode is picked, without a rebuild. The
        // backing value (e.g. trimOn) is only changed on a real toggle, so it
        // survives an SEQ detour and reappears on switching back.
        row.addEventListener("mouseenter", () => {
            if (!disabled && !isSeq() && !get()) row.style.background = "#28284a";
        });
        row.addEventListener("mouseleave", paint);
        row.addEventListener("mousedown", (ev) => {
            ev.preventDefault();
            ev.stopPropagation();
            if (disabled || isSeq()) return;
            set(!get());
            paintCleanup();   // v261: repaint section so the Trim stepper shows/hides
        });

        togglePainters.push(paint);
        return row;
    }

    wrap.appendChild(makeToggle({
        color: "#ff9f40", label: "Trim — keep strongest",
        hint: "keeps each LoRA's strongest parts and drops the weakest — reduces interference when you stack many",
        get: () => trimOn,
        set: (v) => { trimOn = v; onToggle?.("trim", v); },
        disabled: false,
    }));
    const resolveRow = makeToggle({
        color: "#c080ff", label: "Resolve — resolve conflicts",
        hint: "when LoRAs pull opposite ways, keeps the winning side instead of letting them cancel out — works with Trim",
        get: () => resOn,
        set: (v) => { resOn = v; onToggle?.("resolve", v); },
        disabled: false,
    });
    wrap.appendChild(resolveRow);

    // ── v261: Trim strength stepper ──────────────────────────────────────
    // Sits under the Trim toggle; visible only when Trim is on (and not SEQ).
    // "Auto" = the group-size formula (bit-identical to v260); the numeric
    // stops are fixed kept-fractions. The value leaves via
    // onToggle("trim_amount", <fraction|null>) — null = Auto. Pure DOM, no
    // canvas paint (consistent with the Nodes-2.0 risk note).
    const TRIM_STEPS = [null, 0.9, 0.8, 0.7, 0.6, 0.5];   // index 0 = Auto
    // v262: lay-friendly strength words per stop; the kept-% rides along as a
    // secondary readout. Display only — the stored value stays the fraction.
    const TRIM_WORDS = { 1: "Gentle", 2: "Light", 3: "Medium", 4: "Strong", 5: "Max" };
    let trimAmt = (typeof currentTrimAmount === "number") ? currentTrimAmount : null;

    const stepRow = document.createElement("div");
    stepRow.style.cssText = "padding:6px 10px 8px 47px; display:flex; align-items:center;"
        + "gap:10px; border-bottom:1px solid #1e1e2a; background:#ff9f400d;";
    const stepLbl = document.createElement("div");
    stepLbl.style.cssText = "flex:1; min-width:0; font-size:10px; color:#c0a070;";
    stepLbl.textContent = "Trim strength";
    const ctrl = document.createElement("div");
    ctrl.style.cssText = "display:flex; align-items:center;"
        + "border:1px solid #ff9f4066; border-radius:5px; overflow:hidden; flex-shrink:0;";
    const mkBtn = (txt) => {
        const b = document.createElement("div");
        b.textContent = txt;
        b.style.cssText = "width:22px; height:20px; display:flex; align-items:center;"
            + "justify-content:center; cursor:pointer; color:#ff9f40; font-weight:bold;"
            + "font-size:13px; user-select:none;";
        b.addEventListener("mouseenter", () => { b.style.background = "#ff9f4022"; });
        b.addEventListener("mouseleave", () => { b.style.background = "transparent"; });
        return b;
    };
    const dec = mkBtn("‹");
    const inc = mkBtn("›");
    const valBox = document.createElement("div");
    valBox.style.cssText = "min-width:92px; text-align:center; font-size:11px;"
        + "font-weight:bold; color:#ffcf90; padding:0 4px;"
        + "border-left:1px solid #ff9f4033; border-right:1px solid #ff9f4033;";
    ctrl.appendChild(dec); ctrl.appendChild(valBox); ctrl.appendChild(inc);
    stepRow.appendChild(stepLbl); stepRow.appendChild(ctrl);
    // v262: place the stepper directly under the Trim toggle (before Resolve),
    // so it reads as part of Trim rather than floating at the bottom.
    wrap.insertBefore(stepRow, resolveRow);

    function trimIdx() {
        if (trimAmt == null) return 0;
        let best = 1, bd = Infinity;
        for (let i = 1; i < TRIM_STEPS.length; i++) {
            const d = Math.abs(TRIM_STEPS[i] - trimAmt);
            if (d < bd) { bd = d; best = i; }
        }
        return best;
    }
    function setTrimIdx(i) {
        i = Math.max(0, Math.min(TRIM_STEPS.length - 1, i));
        trimAmt = TRIM_STEPS[i];
        onToggle?.("trim_amount", trimAmt);   // null = Auto, else kept-fraction
        paintTrimAmt();
    }
    function paintTrimAmt() {
        const show = trimOn && !isSeq();
        stepRow.style.display = show ? "flex" : "none";
        if (trimAmt == null) {
            valBox.textContent = "Auto";
        } else {
            valBox.textContent = (TRIM_WORDS[trimIdx()] || "") + " · " + Math.round(trimAmt * 100) + "%";
        }
    }
    dec.addEventListener("mousedown", (ev) => {
        ev.preventDefault(); ev.stopPropagation();
        if (!trimOn || isSeq()) return;
        setTrimIdx(trimIdx() - 1);   // ‹ steps toward Auto (gentler)
    });
    inc.addEventListener("mousedown", (ev) => {
        ev.preventDefault(); ev.stopPropagation();
        if (!trimOn || isSeq()) return;
        setTrimIdx(trimIdx() + 1);   // › steps toward 50% (stronger)
    });
    togglePainters.push(paintTrimAmt);

    function paintCleanup() {
        sub.textContent = "Cleanup" + (isSeq() ? "  — needs Combined or Smooth Mix" : "");
        for (const p of togglePainters) p();
    }
    paintCleanup();

    // ── Footer: deliberate close ─────────────────────────────────────────
    // The popup no longer closes on a mode click, so it needs an explicit
    // dismiss. "Done" + click-outside + Escape all route through close().
    const footer = document.createElement("div");
    footer.style.cssText = "display:flex; justify-content:flex-end; align-items:center; gap:8px;"
        + "padding:7px 10px; border-top:1px solid #2a2a3a; background:#101019;";
    const footHint = document.createElement("div");
    footHint.style.cssText = "flex:1; min-width:0; color:#666; font-size:10px;";
    footHint.textContent = "Pick a mode, add cleanup, then close.";
    footer.appendChild(footHint);
    const doneBtn = document.createElement("div");
    doneBtn.textContent = "Done";
    doneBtn.style.cssText = "padding:5px 14px; cursor:pointer; border-radius:5px;"
        + "background:#2a2a44; color:#d0d0e0; font-weight:bold; font-size:11px;"
        + "border:1px solid #4a3a6a; flex-shrink:0;";
    doneBtn.addEventListener("mouseenter", () => { doneBtn.style.background = "#3a3a5a"; });
    doneBtn.addEventListener("mouseleave", () => { doneBtn.style.background = "#2a2a44"; });
    doneBtn.addEventListener("mousedown", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        close();
    });
    footer.appendChild(doneBtn);
    wrap.appendChild(footer);

    document.body.appendChild(wrap);

    // Viewport correction (unchanged from v250).
    requestAnimationFrame(() => {
        const r = wrap.getBoundingClientRect();
        if (r.right  > window.innerWidth  - 8) wrap.style.left = `${window.innerWidth  - r.width  - 8}px`;
        if (r.bottom > window.innerHeight - 8) wrap.style.top  = `${clickEvent.clientY - r.height - 6}px`;
    });

    // ── Close handling: click-away + Escape, both torn down on close ─────
    function close() {
        wrap.remove();
        document.removeEventListener("pointerdown", closeH, true);
        document.removeEventListener("keydown", escH, true);
    }
    const closeH = (ev) => { if (!wrap.contains(ev.target)) close(); };
    const escH   = (ev) => { if (ev.key === "Escape") { ev.stopPropagation(); close(); } };
    // Arm after a tick so the opening right-click's own pointerdown doesn't
    // immediately close the popup.
    setTimeout(() => {
        document.addEventListener("pointerdown", closeH, true);
        document.addEventListener("keydown", escH, true);
    }, 100);
}

// ─── DOM Popup ─────────────────────────────────────────────────────────────

let popup = null;
let popupName = null;

function showPopup(name, screenX, screenY) {
    if (popupName === name && popup) return;
    closePopup();
    popupName = name;

    const el = document.createElement("div");
    el.style.cssText = [
        "position:fixed", `left:${screenX + 16}px`, `top:${screenY - 10}px`,
        "z-index:99999", "max-width:260px", "background:#111118",
        "border:1px solid #3a3a5a", "border-radius:10px", "padding:10px",
        "box-shadow:0 10px 40px rgba(0,0,0,.85)", "pointer-events:none",
        "font:12px 'Segoe UI',Arial,sans-serif", "color:#ddd",
    ].join(";");

    // Title
    const title = document.createElement("b");
    title.style.cssText = "display:block;color:#a0c4ff;margin-bottom:6px;font-size:11px;word-break:break-all;";
    title.textContent = name.split(/[/\\]/).pop().replace(/\.safetensors$/i, "");
    el.appendChild(title);

    const pv  = previewCache.get(name);
    const meta = metaCache.get(name);

    if (pv?.vid) {
        const v = document.createElement("video");
        v.src = api.apiURL(`/uls/preview/video?lora=${encodeURIComponent(name)}`); v.muted = true; v.loop = true; v.autoplay = true; v.playsInline = true;
        v.style.cssText = "width:100%;border-radius:6px;display:block;margin-bottom:8px;";
        el.appendChild(v);
        v.play().catch(() => {});
    } else if (pv?.img) {
        const img = document.createElement("img");
        img.src = pv.img.src;
        img.style.cssText = "width:100%;border-radius:6px;display:block;margin-bottom:8px;";
        el.appendChild(img);
    } else {
        const ph = document.createElement("div");
        ph.style.cssText = "color:#444;font-size:10px;margin-bottom:6px;";
        ph.textContent = meta?.has_preview_image || meta?.has_preview_video
            ? "⏳ Loading preview…" : "📷 No preview";
        el.appendChild(ph);
    }

    if (meta) {
        const infoEl = document.createElement("div");
        infoEl.style.cssText = "font-size:10px;line-height:1.8;";
        const rows = [
            ["Base",  meta.base_model],
            ["Rank",  meta.rank],
            ["Algo",  (meta.algo || "").split(".").pop()],
        ].filter(([, v]) => v && v !== "unknown" && v !== "?");
        infoEl.innerHTML = rows.map(([k, v]) =>
            `<span style="color:#666">${k}:</span> <span style="color:#ccc">${escapeHtml(v)}</span> &nbsp;`
        ).join("");
        el.appendChild(infoEl);

        const tw = typeof meta.trigger_words === "string"
            ? meta.trigger_words
            : Object.keys(meta.trigger_words || {}).slice(0, 12).join(", ");
        if (tw?.length > 1) {
            const twEl = document.createElement("div");
            twEl.style.cssText = "margin-top:5px;font-size:10px;";
            twEl.innerHTML = `<span style="color:#555">Trigger: </span><span style="color:#90c4f9">${escapeHtml(tw.slice(0,180))}</span>`;
            el.appendChild(twEl);
        }
    }

    document.body.appendChild(el);
    popup = el;

    // Viewport-Begrenzung
    requestAnimationFrame(() => {
        const r = el.getBoundingClientRect();
        if (r.right  > window.innerWidth  - 8) el.style.left = `${screenX - r.width - 16}px`;
        if (r.bottom > window.innerHeight - 8) el.style.top  = `${screenY - r.height + 10}px`;
    });
}

function closePopup() {
    popup?.remove(); popup = null; popupName = null;
}

// ─── Row Fabrik ────────────────────────────────────────────────────────────

function newRow() {
    return { enabled: true, name: "None", wHigh: 1.0, wLow: 1.0, group: "—" };
}

// ─── Conflict-Analyse ──────────────────────────────────────────────────────

function checkConflicts(rows) {
    if (!rows || !Array.isArray(rows)) return [];
    const warnings = [];
    const active = rows.filter(r => r.enabled && r.name !== "None");

    // Aufeinanderfolgende Style-LoRAs mit hohen Weights
    for (let i = 0; i < active.length - 1; i++) {
        if (active[i].group === "style" && active[i+1].group === "style") {
            const w = Math.max(active[i].wHigh, active[i+1].wHigh);
            if (w > 0.75) warnings.push({ row: rows.indexOf(active[i+1]), level: "warn",
                msg: `⚠ Two style LoRAs stacked (×${w.toFixed(2)}) — artifact risk` });
        }
    }

    // Weight sums — DARE-Merging bei Gruppen reduziert Interferenz,
    // daher ist der Threshold großzügiger als beim alten sequentiellen Stacking.
    const sumH = active.reduce((s, r) => s + Math.abs(r.wHigh), 0);
    const sumL = active.reduce((s, r) => s + Math.abs(r.wLow),  0);
    if (sumH > 10) warnings.push({ row: -1, level: "warn",
        msg: `⚠ Total weight sum ${sumH.toFixed(1)} is very high — consider reducing individual LoRA weights` });
    if (sumL > 10) warnings.push({ row: -1, level: "warn",
        msg: `⚠ Total weight sum ${sumL.toFixed(1)} is very high — consider reducing individual LoRA weights` });

    return warnings;
}

// ─── Multiplier Info Tooltip ──────────────────────────────────────────────────

function showMultiplierTooltip(canvasX, canvasY, node) {
    document.getElementById("uls-mult-tooltip")?.remove();
    const canvas = app.canvas?.canvas;
    if (!canvas) return;
    const rect  = canvas.getBoundingClientRect();
    const scale = app.canvas?.ds?.scale ?? 1;
    const off   = app.canvas?.ds?.offset ?? {0:0,1:0};
    const sx = rect.left + (node.pos[0] + canvasX) * scale + off[0] * scale;
    const sy = rect.top  + (node.pos[1] + canvasY) * scale + off[1] * scale;

    const el = document.createElement("div");
    el.id = "uls-mult-tooltip";
    el.style.cssText = [
        `position:fixed`, `left:${sx - 160}px`, `top:${sy - 120}px`,
        "z-index:999999", "width:200px",
        "background:#14141e", "border:1px solid #3a3a5a",
        "border-radius:8px", "padding:10px 12px",
        "box-shadow:0 4px 20px rgba(0,0,0,0.8)",
        "font:11px 'Segoe UI',Arial,sans-serif", "color:#aaa",
        "pointer-events:none",
    ].join(";");
    el.innerHTML = `
        <b style="color:#a080ff">Global Multiplier</b><br><br>
        Scales <b>all</b> LoRA weights simultaneously.<br><br>
        <span style="color:#4a9eff">×0.75</span> = 75% of all weights<br>
        <span style="color:#7060cc">×1.00</span> = default<br>
        <span style="color:#ff7744">×1.50</span> = 150% boosted
    `;
    document.body.appendChild(el);
    // Auto-remove nach 3s
    setTimeout(() => el?.remove(), 3000);
}

function showMultiplierInfo(e) {
    const existing = document.getElementById("uls-mult-info");
    if (existing) { existing.remove(); return; }

    const el = document.createElement("div");
    el.id = "uls-mult-info";
    el.style.cssText = [
        "position:fixed",
        `left:${e.clientX - 160}px`,
        `top:${e.clientY - 180}px`,
        "z-index:999999",
        "width:300px",
        "background:#14141e",
        "border:1px solid #3a3a5a",
        "border-radius:10px",
        "padding:14px 16px",
        "box-shadow:0 8px 32px rgba(0,0,0,0.8)",
        "font:12px 'Segoe UI',Arial,sans-serif",
        "color:#ccc",
    ].join(";");

    el.innerHTML = `
        <div style="font-weight:bold;color:#a080ff;margin-bottom:8px;font-size:13px;">
            ⚡ Global Multiplier
        </div>
        <div style="line-height:1.7;color:#aaa;">
            Scales <b style="color:#ddd">all</b> LoRA weights proportionally at once.<br><br>
            <b style="color:#4a9eff">×1.00</b> = no change (default)<br>
            <b style="color:#51cf66">×0.75</b> = all weights reduced to 75%<br>
            <b style="color:#ff7744">×1.50</b> = all weights boosted by 50%<br><br>
            <span style="color:#666;font-size:11px;">
            Useful when all LoRAs together are too strong or
            too weak — without adjusting each one individually.
            </span>
        </div>
        <div style="text-align:right;margin-top:10px;">
            <span style="color:#444;font-size:10px;cursor:pointer;" id="uls-mult-close">✕ close</span>
        </div>
    `;

    document.body.appendChild(el);
    document.getElementById("uls-mult-close")?.addEventListener("click", () => el.remove());

    // Viewport-Korrektur
    requestAnimationFrame(() => {
        const r = el.getBoundingClientRect();
        if (r.right  > window.innerWidth  - 8) el.style.left = `${window.innerWidth  - r.width  - 8}px`;
        if (r.bottom > window.innerHeight - 8) el.style.top  = `${window.innerHeight - r.height - 8}px`;
        if (r.top    < 8)                       el.style.top  = "8px";
    });

    // Auto-close nach 8 Sekunden
    setTimeout(() => el?.remove(), 8000);
}

// ─── Haupt Extension ───────────────────────────────────────────────────────

app.registerExtension({
    name: "Polyhedron.stack",

    async setup() {
        console.log("[ULS] Extension setup() ✓");
        // Preload groups DB
        try {
            const r = await api.fetchApi("/uls/groups");
            window._ulsGroupsDB = await r.json();
            console.log(`[ULS] Groups DB: ${Object.keys(window._ulsGroupsDB).length} entries`);
        } catch(e) {
            window._ulsGroupsDB = {};
        }

        // ── Global pointerup: stop slider drag even if mouse leaves node ──
        document.addEventListener("pointerup", () => {
            if (app.graph?._nodes) {
                for (const node of app.graph._nodes) {
                    if (node._uls?._sliderDragging) {
                        node._uls._sliderDragging = false;
                        app.graph?.setDirtyCanvas(true, false);
                    }
                }
            }
        }, false);

        // ── Close all ULS overlays on canvas click ───────────────────
        setTimeout(() => {
            const canvasEl = app.canvas?.canvas;
            if (!canvasEl) return;

            function closeAllUlsOverlays() {
                document.getElementById("uls-lora-select")?.remove();
                document.getElementById("uls-group-overlay")?.remove();
                document.getElementById("uls-trigger-select")?.remove();
                document.getElementById("uls-weight-input")?.remove();
                document.getElementById("uls-mult-tooltip")?.remove();
                document.getElementById("uls-mult-info")?.remove();
                document.getElementById("uls-mode-popup")?.remove();
                closePopup();
            }

            // document-level capture: fires before LiteGraph sees the event.
            // We use this for two things:
            //   (a) close ULS overlays when clicking the bare canvas
            //   (b) intercept right-click on a GRP-pill before LiteGraph builds
            //       its tree menu, and open our mode-picker popup instead.
            document.addEventListener("pointerdown", (ev) => {
                // (a) Bare canvas click → close overlays
                if (ev.target === canvasEl && ev.button !== 2) {
                    closeAllUlsOverlays();
                }

                // (b) Right-click on canvas → check if cursor is over a GRP-pill
                if (ev.button === 2 && ev.target === canvasEl) {
                    const node = app.canvas?.node_over;
                    if (!node || !node._uls) return;
                    const uls = node._uls;
                    if (uls.hoverZone !== "grp" || uls.hoverRow < 0) return;
                    const row = uls.rows[uls.hoverRow];
                    if (!row || !row.group || row.group === "—") return;

                    // Suppress LiteGraph's tree menu entirely.
                    ev.preventDefault();
                    ev.stopPropagation();
                    ev.stopImmediatePropagation();

                    const cur   = (uls.groupModes || {})[row.group] || "SEQ";
                    const curDV = (uls.groupDare  || {})[row.group] || "channel";
                    const curTrim = !!((uls.groupTrim    || {})[row.group]);
                    const curRes  = !!((uls.groupResolve || {})[row.group]);
                    const curTrimAmt = (uls.groupTrimAmount || {})[row.group];   // number | undefined (=Auto)
                    showGroupModePopup(row.group, cur, curDV, curTrim, curRes, curTrimAmt, ev, (compositeKey) => {
                        if (!node._uls.groupModes) node._uls.groupModes = {};
                        if (!node._uls.groupDare)  node._uls.groupDare  = {};
                        if (compositeKey.startsWith("DARE:")) {
                            const dv = compositeKey.slice(5); // "channel" or "element"
                            node._uls.groupModes[row.group] = "DARE";
                            node._uls.groupDare[row.group]  = dv;
                        } else {
                            if (compositeKey === "SEQ") delete node._uls.groupModes[row.group];
                            else node._uls.groupModes[row.group] = compositeKey;
                            // Clear dare variant when leaving DARE mode
                            delete node._uls.groupDare[row.group];
                        }
                        api.fetchApi("/uls/group_modes", {
                            method: "POST",
                            headers: {"Content-Type": "application/json"},
                            body: JSON.stringify({ group: row.group, mode: node._uls.groupModes[row.group] || "SEQ" })
                        }).catch(() => {});
                        node._ulsSync?.();
                        app.graph?.setDirtyCanvas(true, false);
                    }, (which, value) => {
                        // Cleanup-switch toggled. Persisted in uls_config (like group_dare),
                        // so it travels inside the saved workflow.
                        if (which === "trim") {
                            if (!node._uls.groupTrim) node._uls.groupTrim = {};
                            if (value) node._uls.groupTrim[row.group] = true;
                            else       delete node._uls.groupTrim[row.group];
                        } else if (which === "resolve") {
                            if (!node._uls.groupResolve) node._uls.groupResolve = {};
                            if (value) node._uls.groupResolve[row.group] = true;
                            else       delete node._uls.groupResolve[row.group];
                        } else if (which === "trim_amount") {
                            // v261: per-group Trim strength. null = Auto → drop the key
                            // so the backend falls back to the group-size formula.
                            if (!node._uls.groupTrimAmount) node._uls.groupTrimAmount = {};
                            if (typeof value === "number") node._uls.groupTrimAmount[row.group] = value;
                            else                           delete node._uls.groupTrimAmount[row.group];
                        }
                        node._ulsSync?.();
                        app.graph?.setDirtyCanvas(true, false);
                    });
                }
            }, true);

            // Belt-and-suspenders: also block the contextmenu event itself
            // when it would fire over a GRP-pill, in case pointerdown didn't
            // catch it (e.g. trackpad two-finger right-click).
            canvasEl.addEventListener("contextmenu", (ev) => {
                const node = app.canvas?.node_over;
                if (!node || !node._uls) return;
                const uls = node._uls;
                if (uls.hoverZone !== "grp" || uls.hoverRow < 0) return;
                const row = uls.rows[uls.hoverRow];
                if (!row || !row.group || row.group === "—") return;
                ev.preventDefault();
                ev.stopPropagation();
                ev.stopImmediatePropagation();
            }, true);
        }, 500);
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        // LoRA-Liste: globale Variable - wird beim ersten Öffnen geladen
        // (loraList ist global oben im File definiert)

        // ── onCreate ────────────────────────────────────────────────────
        const _orig_onCreate = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            _orig_onCreate?.apply(this, arguments);
            this._uls = { rows: [newRow()], mult: 1.0,
                          hoverRow: -1, hoverZone: "",
                          dragSrc: -1, dragDest: -1,
                          groupModes: {},
                          groupDare: {},
                          groupTrim: {},
                          groupResolve: {},
                          groupTrimAmount: {},
                          flatMode: false,
                          groupOrder: {} };
            this.size[0] = Math.max(this.size[0], 460);
            this._ulsResize();
            // Hide config widget after a short delay (widgets are built after onCreate)
            setTimeout(() => this._ulsHideConfigWidget?.(), 100);
            // Load persisted group modes from backend
            const _self = this;
            api.fetchApi("/uls/group_modes").then(r => r.json()).then(data => {
                if (data && typeof data === "object") {
                    _self._uls.groupModes = data;
                    _self._ulsSync?.();
                    app.graph?.setDirtyCanvas(true, false);
                }
            }).catch(() => {});
        };

        nodeType.prototype._ulsResize = function () {
            if (!this._uls?.rows) return;
            const warns = checkConflicts(this._uls.rows).filter(w => w.row === -1).length;
            const warnH = warns > 0 ? warns * 20 + 4 : 0;
            const h = HEADER_H + (this._uls.rows.length + 1) * ROW_H + warnH + FOOTER_H + 8;
            // Preserve user-resized width — only enforce minimum
            const currentW = this.size[0] || 460;
            const w = Math.max(currentW, 460);
            this.size[0] = w;
            this.size[1] = h;
            if (this.setSize) this.setSize([w, h]);
            this._uls.hoverRow  = -1;
            this._uls.hoverZone = "";
            app.graph?.setDirtyCanvas(true, false);
        };

        // onResize: verhindert dass ComfyUI die Node unter unsere Mindestgröße schrumpft
        nodeType.prototype.onResize = function (size) {
            if (!this._uls?.rows) return;
            const warns = checkConflicts(this._uls.rows).filter(w => w.row === -1).length;
            const warnH = warns > 0 ? warns * 20 + 4 : 0;
            const minH = HEADER_H + (this._uls.rows.length + 1) * ROW_H + warnH + FOOTER_H + 8;
            if (size[0] < 460) size[0] = 460;
            if (size[1] < minH) size[1] = minH;
        };

        // ── Serialize ───────────────────────────────────────────────────
        const _orig_ser = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (o) {
            _orig_ser?.apply(this, arguments);
            if (this._uls?.rows) o._uls = JSON.stringify({
                rows: this._uls.rows.map(r => ({
                    enabled: r.enabled, name: r.name,
                    weight: r.wLow ?? r.wHigh ?? 1.0,
                    wHigh: r.wHigh, wLow: r.wLow,
                    wClip: r.wClip,   // v302: optional per-row CLIP strength
                    group: r.group,
                })),
                mult: this._uls.mult,
                groupDare: this._uls.groupDare || {},
                // v259: persist the Trim/Resolve cleanup toggles into the saved
                // workflow. onConfigure already reads d.group_trim / d.group_resolve
                // (snake_case) — they were just never written here, so they reset
                // to {} on every reload/restart. Writing them closes that gap.
                group_trim: this._uls.groupTrim || {},
                group_resolve: this._uls.groupResolve || {},
                // v261: persist the per-group Trim strength too, same chain.
                group_trim_amount: this._uls.groupTrimAmount || {},
                flatMode: this._uls.flatMode || false,
                groupOrder: this._uls.groupOrder || {},
            });
        };

        const _orig_cfg = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (o) {
            _orig_cfg?.apply(this, arguments);
            // _uls sicher initialisieren falls onNodeCreated noch nicht lief
            if (!this._uls || typeof this._uls !== "object" || !this._uls.rows) {
                this._uls = { rows: [newRow()], mult: 1.0,
                              hoverRow: -1, hoverZone: "", dragSrc: -1, dragDest: -1,
                              groupModes: {}, groupDare: {},
                              groupTrim: {}, groupResolve: {},
                              groupTrimAmount: {},
                              flatMode: false, groupOrder: {} };
            }
            if (!this._uls.groupModes) this._uls.groupModes = {};
            if (!this._uls.groupDare)  this._uls.groupDare  = {};
            if (!this._uls.groupTrim)    this._uls.groupTrim    = {};
            if (!this._uls.groupResolve) this._uls.groupResolve = {};
            if (!this._uls.groupTrimAmount) this._uls.groupTrimAmount = {};
            if (this._uls.flatMode === undefined) this._uls.flatMode = false;
            if (!this._uls.groupOrder) this._uls.groupOrder = {};
            // uls_config Widget immer verstecken
            this._ulsHideConfigWidget();
            if (o._uls && this._uls) {
                try {
                    const d = JSON.parse(o._uls);
                    // Client-side group migration — same map as server uls_routes.py
                    const _GRP_MIGRATE = {
                        "character": "subject", "lighting": "scene", "artist": "style",
                        "CHAR": "subject", "LIGH": "scene", "SUBJ": "subject",
                        "SCEN": "scene", "DETA": "detail", "STYL": "style",
                        "MOTI": "motion", "CUST": "custom", "ACC": "acc",
                    };
                    this._uls.rows = d.rows.map(r => {
                        const base = newRow();
                        const merged = { ...base, ...r };
                        const w = (typeof r.weight === "number" ? r.weight
                                : typeof r.wLow   === "number" ? r.wLow
                                : typeof r.wHigh  === "number" ? r.wHigh
                                : 1.0);
                        merged.wLow  = w;
                        merged.wHigh = w;
                        // Migrate old group names
                        if (merged.group && _GRP_MIGRATE[merged.group]) {
                            merged.group = _GRP_MIGRATE[merged.group];
                        }
                        return merged;
                    });
                    this._uls.mult = d.mult ?? 1.0;
                    this._uls.flatMode   = d.flatMode   === true;
                    this._uls.groupOrder = (d.groupOrder && typeof d.groupOrder === "object")
                        ? d.groupOrder : {};
                    // v098: per-group dare. Migrate legacy global dareVariant if present.
                    if (d.groupDare && typeof d.groupDare === "object") {
                        this._uls.groupDare = d.groupDare;
                    } else if (d.dareVariant === "channel" || d.dareVariant === "element") {
                        // Old workflow: had a single global dareVariant.
                        // We can't know which groups had DARE active, so we
                        // leave groupDare empty — user will set per group as needed.
                        this._uls.groupDare = {};
                    }
                    // v250: per-group cleanup switches. Absent → empty (off).
                    this._uls.groupTrim    = (d.group_trim    && typeof d.group_trim    === "object") ? d.group_trim    : {};
                    this._uls.groupResolve = (d.group_resolve && typeof d.group_resolve === "object") ? d.group_resolve : {};
                    // v261: restore per-group Trim strength.
                    this._uls.groupTrimAmount = (d.group_trim_amount && typeof d.group_trim_amount === "object") ? d.group_trim_amount : {};
                    this._uls.rows.forEach(r => ensurePreview(r.name));
                    this._ulsResize();
                } catch {}
            }
            // Always re-fetch group modes from backend (single source of truth)
            const _self = this;
            api.fetchApi("/uls/group_modes").then(r => r.json()).then(data => {
                if (data && typeof data === "object") {
                    _self._uls.groupModes = data;
                    _self._ulsSync?.();
                    app.graph?.setDirtyCanvas(true, false);
                }
            }).catch(() => {});
        };

        nodeType.prototype._ulsHideConfigWidget = function () {
            if (!this.widgets) return;
            for (const w of this.widgets) {
                if (w.name === "uls_config") {
                    w.hidden = true;
                    w.type = "hidden";
                    w.computeSize = () => [0, -4];
                    // ComfyUI neu-Layout erzwingen — Breite explizit bewahren
                    if (this.setSize) {
                        const s = this.computeSize();
                        this.setSize([Math.max(this.size[0] || 460, 460), s[1]]);
                    }
                    break;
                }
            }
        };

        // Sync zum hidden widget für Python-Backend
        nodeType.prototype._ulsSync = function () {
            if (!this._uls || !this.widgets) return;
            let w = this.widgets.find(x => x.name === "uls_config");
            if (!w) {
                w = this.addWidget("text", "uls_config", "", () => {});
            }
            // Widget immer verstecken — Wert wird intern verwaltet
            w.hidden = true;
            w.type = "hidden";
            w.computeSize = () => [0, -4];
            w.tooltip = "";
            w.options = w.options || {};
            w.options.hideOnZoom = true;  // Nimmt keinen Platz ein
            w.value = JSON.stringify({
                rows: this._uls.rows.map(r => {
                    const wt = (typeof r.wLow === "number" ? r.wLow
                              : typeof r.wHigh === "number" ? r.wHigh
                              : 1.0);
                    return {
                        enabled: r.enabled, name: r.name,
                        weight: wt,
                        wHigh: r.wHigh, wLow: r.wLow,
                        wClip: r.wClip,   // v302: optional per-row CLIP strength
                        group: r.group,
                    };
                }),
                mult: this._uls.mult,
                group_modes: this._uls.groupModes || {},
                group_dare: this._uls.groupDare || {},
                group_trim: this._uls.groupTrim || {},
                group_resolve: this._uls.groupResolve || {},
                group_trim_amount: this._uls.groupTrimAmount || {},
                flatMode: this._uls.flatMode || false,
                groupOrder: this._uls.groupOrder || {},
                dare_variant: "channel",
            });
            // Widget nach jedem Sync verstecken
            this._ulsHideConfigWidget?.();
        };

        // ── Draw ────────────────────────────────────────────────────────
        nodeType.prototype.onDrawForeground = function (ctx) {
            // Compat probe (uls_compat.js): records that the LiteGraph canvas
            // draw path actually ran for this node. Used to detect renderers
            // (e.g. Nodes 2.0 / Vue) that never call onDrawForeground.
            this._ulsDrawFired = true;
            // v303: self-healing — if the compat layer injected the renderer
            // notice (uls_compat.js, name below) but the canvas path IS alive,
            // that was a false positive (offscreen culling / slow first draw:
            // onDrawForeground only runs for nodes inside the viewport).
            // Remove the notice the moment we provably draw.
            if (this.widgets?.some(w => w?.name === "polyhedron_renderer_notice")) {
                this.widgets = this.widgets.filter(
                    w => w?.name !== "polyhedron_renderer_notice");
                this.setDirtyCanvas?.(true, true);
            }
            const uls = this._uls;
            if (!uls || typeof uls !== "object" || !uls.rows) return;
            const W = this.size[0];
            const rows = uls.rows;
            const conflicts = checkConflicts(rows);
            ctx.save();

            // ── Polyhedron Wireframe Icon im Titel-Bereich ──────────────
            // Kleines Wireframe-Hexagon rechts oben im Node-Titel
            {
                const ix = W - 22, iy = -18, ir = 8;
                ctx.strokeStyle = "#c060ff";
                ctx.lineWidth = 1.2;
                ctx.globalAlpha = 0.8;
                // Äußeres Hexagon
                ctx.beginPath();
                for (let k = 0; k < 6; k++) {
                    const a = (k * Math.PI / 3) - Math.PI / 6;
                    const x = ix + ir * Math.cos(a);
                    const y = iy + ir * Math.sin(a);
                    k === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
                }
                ctx.closePath(); ctx.stroke();
                // Innere Speichen (Kanten zum Mittelpunkt)
                for (let k = 0; k < 6; k += 2) {
                    const a = (k * Math.PI / 3) - Math.PI / 6;
                    ctx.beginPath();
                    ctx.moveTo(ix, iy);
                    ctx.lineTo(ix + ir * Math.cos(a), iy + ir * Math.sin(a));
                    ctx.stroke();
                }
                ctx.globalAlpha = 1;
            }

            // Spalten-Header
            ctx.font = "bold 10px 'Segoe UI',Arial";
            ctx.textBaseline = "middle";
            ctx.fillStyle = "#505060";
            // Column headers above rows
            // v307: these MUST mirror the row layout constants (row loop:
            // GRP_W=50, insertW=28, WEIGHT_W=72, DEL_W=18, btnGap=4). The
            // header had a stale _GRP_W=36, which pushed the "Group" label
            // 7px and the "Trigger" label 14px right of their columns.
            const _DEL_W   = 18, _btnGap = 4, _WEIGHT_W = 72;
            const _weightX = W - PAD - _DEL_W - _btnGap - _WEIGHT_W;
            const _GRP_W   = 50, _INSERT_W = 28;
            const _grpX    = _weightX - _btnGap - _GRP_W;
            const _insertX = _grpX - _btnGap - _INSERT_W;

            // ── Flat-Mode Toggle Pill ────────────────────────────────────
            {
                const flat    = uls.flatMode || false;
                const isHov   = uls.hoverZone === "flatMode";
                const pillW   = 90, pillH = 16;
                const pillX   = PAD;
                const pillY   = HEADER_H - 34;
                const accent  = flat ? "#f0c050" : "#a060ff";
                ctx.fillStyle = isHov ? accent + "33" : accent + "18";
                roundRect(ctx, pillX, pillY, pillW, pillH, 8); ctx.fill();
                ctx.strokeStyle = isHov ? accent : accent + "88";
                ctx.lineWidth = 1;
                roundRect(ctx, pillX, pillY, pillW, pillH, 8); ctx.stroke();
                ctx.font = "bold 9px 'Segoe UI',Arial";
                ctx.fillStyle = flat ? "#f0c050" : "#a060ff";
                ctx.textAlign = "center"; ctx.textBaseline = "middle";
                ctx.fillText(flat ? "LIST STACK" : "GROUP STACK", pillX + pillW/2, pillY + pillH/2);
                ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
                uls._flatModePillRect = { x: pillX, y: pillY, w: pillW, h: pillH };

                // No tooltip — the input label "Stack Order" already explains the concept
            }

            ctx.font = "9px 'Segoe UI',Arial";
            ctx.fillStyle = "#b07820";
            ctx.textAlign = "center";
            {
                // v305: two-tone header (amber Weight + blue /CLIP + info
                // icon), centered over the column via measureText — the v304
                // eyeballed offsets sat visibly right of the cell center.
                // The icon is decorative: the WHOLE header cell is the hover
                // area for the explainer tooltip (uls._weightHdrRect below).
                const _hy = HEADER_H - 8;
                // v308: "Weight / CLIP Strength" is wider than the 72px
                // weight cell, so the composite is RIGHT-anchored: the icon's
                // right edge sits at the node content edge (above the ✕
                // column, which has no header label), and the text extends
                // left into the cell. That keeps maximum clearance from the
                // "Group" label. Measured, never eyeballed (v304 lesson).
                ctx.textAlign = "left";
                ctx.font = "9px 'Segoe UI',Arial";
                const _wW = ctx.measureText("Weight").width;
                ctx.font = "8px 'Segoe UI',Arial";
                const _wC = ctx.measureText(" / CLIP Strength").width;
                const _ICO_R = 3.2, _ICO_GAP = 4;
                const _total = _wW + _wC + _ICO_GAP + 2 * _ICO_R;
                const _right = W - PAD;            // = contentR of the rows
                let _hx = _right - _total;
                ctx.font = "9px 'Segoe UI',Arial";
                ctx.fillStyle = "#b07820";
                ctx.fillText("Weight", _hx, _hy);
                _hx += _wW;
                ctx.font = "8px 'Segoe UI',Arial";
                ctx.fillStyle = "#6aa0d0";
                ctx.fillText(" / CLIP Strength", _hx, _hy);
                // 🛈 — small stroked circle with an "i", same CLIP blue
                const _ix = _hx + _wC + _ICO_GAP + _ICO_R;
                ctx.beginPath();
                ctx.arc(_ix, _hy - 3, _ICO_R, 0, Math.PI * 2);
                ctx.strokeStyle = "#6aa0d0"; ctx.lineWidth = 0.9;
                ctx.stroke();
                ctx.font = "bold 5.5px 'Segoe UI',Arial";
                ctx.textAlign = "center";
                ctx.fillText("i", _ix, _hy - 1.2);
                ctx.font = "9px 'Segoe UI',Arial";
                // Hover area = the full composite (text + icon), not just
                // the 72px cell — the tooltip promise is "whole surface".
                uls._weightHdrRect = { x: _right - _total - 2, y: HEADER_H - 18,
                                       w: _total + 4, h: 14 };
            }
            ctx.fillStyle = "#7a6aaa";
            ctx.fillText("Group", _grpX + _GRP_W / 2, HEADER_H - 8);
            ctx.fillStyle = "#4a8a6a";
            ctx.fillText("Trigger", _insertX + _INSERT_W / 2, HEADER_H - 8);
            ctx.textAlign = "left";
            ctx.fillStyle = "#404055";
            ctx.strokeStyle = "#252535"; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(0, HEADER_H - 2); ctx.lineTo(W, HEADER_H - 2); ctx.stroke();

            // Rows
            rows.forEach((row, i) => {
                const y = HEADER_H + i * ROW_H;
                const isHov  = uls.hoverRow === i;
                const isDSrc = uls.dragSrc  === i;
                const isDDst = uls.dragDest === i && uls.dragSrc >= 0 && uls.dragSrc !== i;

                // Zebra / Hover Hintergrund — volle Breite
                ctx.fillStyle = isDSrc ? "#141420" : isHov ? "#21213a" : i % 2 ? "#1c1c2c" : "#191928";
                ctx.fillRect(0, y, W, ROW_H);

                // DragDest Linie
                if (isDDst) {
                    ctx.strokeStyle = "#8b6fe8"; ctx.lineWidth = 2;
                    ctx.beginPath(); ctx.moveTo(PAD, y); ctx.lineTo(W - PAD, y); ctx.stroke();
                }

                // Gruppen-Streifen
                ctx.fillStyle = row.enabled ? (GROUP_COLORS[row.group] || "#404050") : "#282838";
                ctx.fillRect(0, y, 3, ROW_H);

                const alpha = row.enabled ? 1 : 0.4;
                ctx.globalAlpha = alpha;

                // ▲▼ Buttons
                const upHov  = isHov && uls.hoverZone === "up";
                const dnHov  = isHov && uls.hoverZone === "down";
                // ▲ Button
                ctx.fillStyle = upHov ? "#2a2a4a" : "#1a1a2e";
                roundRect(ctx, PAD, y + 2, 10, 11, 2); ctx.fill();
                ctx.fillStyle = (i === 0) ? "#333" : (upHov ? "#a0a0ff" : "#5566aa");
                ctx.font = "8px Arial"; ctx.textAlign = "center";
                ctx.fillText("▲", PAD + 5, y + 10);
                // ▼ Button
                ctx.fillStyle = dnHov ? "#2a2a4a" : "#1a1a2e";
                roundRect(ctx, PAD, y + 15, 10, 11, 2); ctx.fill();
                ctx.fillStyle = (i === uls.rows.length-1) ? "#333" : (dnHov ? "#a0a0ff" : "#5566aa");
                ctx.fillText("▼", PAD + 5, y + 23);
                ctx.textAlign = "left";

                // Checkbox
                const cbx = PAD + 30, cby = y + 8;
                ctx.strokeStyle = row.enabled ? "#5555bb" : "#383848"; ctx.lineWidth = 1.5;
                ctx.strokeRect(cbx, cby, 12, 12);
                if (row.enabled) {
                    ctx.strokeStyle = "#8080ff"; ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.moveTo(cbx+2, cby+6); ctx.lineTo(cbx+5, cby+9); ctx.lineTo(cbx+10, cby+3);
                    ctx.stroke();
                }

                // ── Button-Layout ──────────────────────────────────────
                // [THUMB][Name-Button][↵][GRP-Pill][◀ Weight ▶][✕]
                const THUMB_W  = 30;
                const thumbX   = PAD + 48;
                const insertW  = 28;
                const GRP_W    = 50;
                const WEIGHT_W = 72;  // breit genug für ◀ value ▶
                const ARROW_W  = 14;  // Breite jedes Pfeil-Buttons
                const DEL_W    = 18;
                const btnGap   = 4;
                const nameX    = thumbX + THUMB_W + 4;
                const contentR = W - PAD;
                // Von rechts: ✕ + gap + [◀ Weight ▶] + gap + GRP + gap + insert + gap
                const delX     = contentR - DEL_W;
                const weightX  = delX - btnGap - WEIGHT_W;
                const grpPillX = weightX - btnGap - GRP_W;
                const insertX  = grpPillX - btnGap - insertW;
                const nameMaxW = insertX - btnGap - nameX;
                // Pfeil-Positionen innerhalb der Weight-Box
                const wArrowLX = weightX;                       // ◀ links
                const wArrowRX = weightX + WEIGHT_W - ARROW_W; // ▶ rechts
                const wValX    = weightX + ARROW_W;             // Wert-Bereich

                const nameHover   = isHov && uls.hoverZone === "name";
                const insertHover = isHov && uls.hoverZone === "insert";
                const delHover    = isHov && uls.hoverZone === "del";
                const wDecHover   = isHov && uls.hoverZone === "wDec";
                const wIncHover   = isHov && uls.hoverZone === "wInc";

                // Thumbnail zeichnen
                const pv2 = previewCache.get(row.name);
                ctx.fillStyle = "#111118";
                roundRect(ctx, thumbX, y + 3, THUMB_W, ROW_H - 6, 3); ctx.fill();
                ctx.strokeStyle = "#2a2a3a"; ctx.lineWidth = 0.5;
                roundRect(ctx, thumbX, y + 3, THUMB_W, ROW_H - 6, 3); ctx.stroke();

                if (pv2?.img) {
                    // Bild als Thumbnail zeichnen
                    try {
                        ctx.save();
                        roundRect(ctx, thumbX, y + 3, THUMB_W, ROW_H - 6, 3);
                        ctx.clip();
                        const ir = pv2.img.naturalWidth / pv2.img.naturalHeight;
                        const th = ROW_H - 6;
                        const tw2 = th * ir;
                        const tx = thumbX + (THUMB_W - tw2) / 2;
                        ctx.drawImage(pv2.img, tx, y + 3, tw2, th);
                        ctx.restore();
                    } catch(e) {}
                } else if (pv2?.vid) {
                    // Video-Frame als Thumbnail
                    try {
                        ctx.save();
                        roundRect(ctx, thumbX, y + 3, THUMB_W, ROW_H - 6, 3);
                        ctx.clip();
                        ctx.drawImage(pv2.vid, thumbX, y + 3, THUMB_W, ROW_H - 6);
                        ctx.restore();
                        // Play-Icon overlay
                        ctx.fillStyle = "rgba(255,255,255,0.7)";
                        ctx.font = "8px Arial";
                        ctx.textAlign = "center";
                        ctx.fillText("▶", thumbX + THUMB_W/2, y + ROW_H/2 + 3);
                        ctx.textAlign = "left";
                    } catch(e) {}
                } else if (row.name !== "None") {
                    // Kein Preview: Initials anzeigen
                    const initial = row.name.split(/[/\\]/).pop().charAt(0).toUpperCase();
                    ctx.fillStyle = GROUP_COLORS[row.group] || "#404050";
                    ctx.font = "bold 12px Arial";
                    ctx.textAlign = "center";
                    ctx.fillText(initial, thumbX + THUMB_W/2, y + ROW_H/2 + 4);
                    ctx.textAlign = "left";
                }

                // Name-Button
                ctx.fillStyle = nameHover ? "#2a2244" : "#1a1a2a";
                roundRect(ctx, nameX - 2, y + 4, nameMaxW + 4, ROW_H - 8, 4); ctx.fill();
                ctx.strokeStyle = nameHover ? "#6655aa" : "#2e2e44"; ctx.lineWidth = 0.5;
                roundRect(ctx, nameX - 2, y + 4, nameMaxW + 4, ROW_H - 8, 4); ctx.stroke();
                ctx.save();
                ctx.beginPath(); ctx.rect(nameX, y + 4, nameMaxW, ROW_H - 8); ctx.clip();
                ctx.fillStyle = row.name === "None" ? "#555566" : nameHover ? "#e8e0ff" : "#d0d0e0";
                ctx.font = "12px 'Segoe UI',Arial";
                const dispName = row.name === "None"
                    ? "  Select LoRA…"
                    : "  " + row.name.split(/[/\\]/).pop().replace(/\.safetensors$/i, "");
                ctx.fillText(dispName, nameX, y + 17);
                ctx.restore();

                // ↵ Insert-Button
                ctx.fillStyle = insertHover ? "#2a1f50" : "#1a1530";
                roundRect(ctx, insertX, y + 4, insertW, ROW_H - 8, 4); ctx.fill();
                ctx.strokeStyle = insertHover ? "#8866dd" : "#2e2844"; ctx.lineWidth = 0.5;
                roundRect(ctx, insertX, y + 4, insertW, ROW_H - 8, 4); ctx.stroke();
                ctx.fillStyle = row.name === "None" ? "#333" : insertHover ? "#c0a8ff" : "#7060aa";
                ctx.font = "13px Arial"; ctx.textAlign = "center";
                ctx.fillText("\u21b5", insertX + insertW / 2, y + 17);
                ctx.textAlign = "left";

                // GRP-Pill (nach ↵ Button)
                const gc2 = GROUP_COLORS[row.group] || "#404050";
                const grpHov2 = isHov && uls.hoverZone === "grp";
                ctx.fillStyle = grpHov2 ? gc2 + "55" : gc2 + "22";
                roundRect(ctx, grpPillX, y + 5, GRP_W, ROW_H - 10, 4); ctx.fill();
                ctx.strokeStyle = grpHov2 ? gc2 : gc2 + "55"; ctx.lineWidth = 0.5;
                roundRect(ctx, grpPillX, y + 5, GRP_W, ROW_H - 10, 4); ctx.stroke();
                ctx.fillStyle = gc2; ctx.font = "bold 8px 'Segoe UI',Arial";
                ctx.textAlign = "center";
                const grpLabel = row.group === "—" ? "GRP" : row.group.slice(0,4).toUpperCase();
                ctx.fillText(grpLabel, grpPillX + GRP_W/2, y + ROW_H/2 + 3);
                ctx.textAlign = "left";

                // Order badge — top-left corner of GRP pill, inside the pill.
                // Gold filled = has a custom order number. Dashed = no number (click to set).
                // Red flash = conflict (duplicate number rejected).
                if (!uls.flatMode && row.group !== "—") {
                    const orderVal = (uls.groupOrder || {})[row.group];
                    const hasOrder = orderVal !== undefined && orderVal !== null && orderVal !== "";
                    const isConflict = uls._orderConflictGroup === row.group;
                    const obcx = grpPillX + 7, obcy = y + 8;
                    if (isConflict) {
                        ctx.fillStyle = "#ff4444";
                        ctx.beginPath(); ctx.arc(obcx, obcy, 7, 0, Math.PI*2); ctx.fill();
                        ctx.fillStyle = "#ffffff";
                        ctx.font = "bold 8px 'Segoe UI',Arial";
                        ctx.textAlign = "center"; ctx.textBaseline = "middle";
                        ctx.fillText("!", obcx, obcy + 0.5);
                        ctx.textBaseline = "alphabetic"; ctx.textAlign = "left";
                    } else if (hasOrder) {
                        ctx.fillStyle = "#f0c050";
                        ctx.beginPath(); ctx.arc(obcx, obcy, 7, 0, Math.PI*2); ctx.fill();
                        ctx.fillStyle = "#1a1a2a";
                        ctx.font = "bold 8px 'Segoe UI',Arial";
                        ctx.textAlign = "center"; ctx.textBaseline = "middle";
                        ctx.fillText(String(orderVal), obcx, obcy + 0.5);
                        ctx.textBaseline = "alphabetic"; ctx.textAlign = "left";
                    } else {
                        // Dashed hint circle — shows badge is clickable
                        ctx.strokeStyle = "#f0c05066";
                        ctx.lineWidth = 0.8;
                        ctx.setLineDash([2, 2]);
                        ctx.beginPath(); ctx.arc(obcx, obcy, 6, 0, Math.PI*2); ctx.stroke();
                        ctx.setLineDash([]);
                    }

                    // No tooltip on GRP-pill hover — too noisy
                }

                // Mode indicator: always visible. Shows current apply mode for this group.
                // S = SEQ (neutral grey), C = CONCAT (gold), D = DARE (cyan).
                // Hidden only for the "no group" placeholder ("—").
                if (row.group !== "—") {
                    const _modeForGrp = ((uls.groupModes || {})[row.group] || "SEQ").toUpperCase();
                    const isDare = _modeForGrp === "DARE";
                    const mLetter = isDare ? "D" : (_modeForGrp === "CONCAT" ? "C" : "S");
                    const mColor  = isDare ? "#40c0ff"
                                   : _modeForGrp === "CONCAT" ? "#f0c050"
                                   : "#7a7a8a";
                    const mx = grpPillX + GRP_W - 5, my = y + 7;
                    ctx.fillStyle = mColor;
                    ctx.beginPath(); ctx.arc(mx, my, 4, 0, Math.PI * 2); ctx.fill();
                    ctx.fillStyle = "#1a1a2a"; ctx.font = "bold 7px 'Segoe UI',Arial";
                    ctx.textAlign = "center"; ctx.textBaseline = "middle";
                    ctx.fillText(mLetter, mx, my + 0.5);
                    ctx.textBaseline = "alphabetic"; ctx.textAlign = "left";

                    // DARE variant mini-badge: "·C" or "·E" below the mode dot, only when DARE active
                    if (isDare) {
                        const dv = ((uls.groupDare || {})[row.group] || "channel");
                        const dvLetter = dv === "element" ? "E" : "C";
                        const dvColor  = dv === "element" ? "#f0c87a" : "#7af0c0";
                        ctx.fillStyle = dvColor;
                        ctx.font = "bold 7px 'Segoe UI',Arial";
                        ctx.textAlign = "center"; ctx.textBaseline = "middle";
                        ctx.fillText("·" + dvLetter, mx, my + 9);
                        ctx.textBaseline = "alphabetic"; ctx.textAlign = "left";
                    }
                }

                // Weight Box mit ◀ ▶ Buttons
                ctx.fillStyle = "#221a10";
                roundRect(ctx, weightX, y + 5, WEIGHT_W, 18, 4); ctx.fill();
                ctx.strokeStyle = "#f0a03044"; ctx.lineWidth = 0.5;
                roundRect(ctx, weightX, y + 5, WEIGHT_W, 18, 4); ctx.stroke();
                // ◀ Button
                ctx.fillStyle = wDecHover ? "#f0a03044" : "transparent";
                roundRect(ctx, wArrowLX, y + 5, ARROW_W, 18, 4); ctx.fill();
                ctx.fillStyle = wDecHover ? "#f0a030" : "#7a5018";
                ctx.font = "bold 9px Arial";
                ctx.textAlign = "center"; ctx.textBaseline = "middle";
                ctx.fillText("◀", wArrowLX + ARROW_W / 2, y + ROW_H / 2);
                // ▶ Button
                ctx.fillStyle = wIncHover ? "#f0a03044" : "transparent";
                roundRect(ctx, wArrowRX, y + 5, ARROW_W, 18, 4); ctx.fill();
                ctx.fillStyle = wIncHover ? "#f0a030" : "#7a5018";
                ctx.fillText("▶", wArrowRX + ARROW_W / 2, y + ROW_H / 2);
                // Wert
                ctx.fillStyle = "#f0a030"; ctx.font = "bold 11px monospace";
                if (typeof row.wClip === "number" && row.wClip !== row.wLow) {
                    // v302: decoupled CLIP strength — two-line cell
                    ctx.font = "bold 9px monospace";
                    ctx.fillText(row.wLow.toFixed(2),
                                 wValX + (WEIGHT_W - 2*ARROW_W) / 2, y + ROW_H / 2 - 4);
                    ctx.font = "7px monospace";
                    ctx.fillStyle = "#6aa0d0";
                    ctx.fillText("c " + row.wClip.toFixed(2),
                                 wValX + (WEIGHT_W - 2*ARROW_W) / 2, y + ROW_H / 2 + 5);
                } else {
                    ctx.fillText(row.wLow.toFixed(2), wValX + (WEIGHT_W - 2*ARROW_W) / 2, y + ROW_H / 2);
                }
                ctx.textBaseline = "alphabetic"; ctx.textAlign = "left";

                // ✕ Delete Button
                ctx.fillStyle = delHover ? "#3a1010" : "transparent";
                roundRect(ctx, delX, y + 5, DEL_W, 18, 3); ctx.fill();
                ctx.fillStyle = delHover ? "#ff6060" : "#444455";
                ctx.font = "bold 11px Arial";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText("✕", delX + DEL_W / 2, y + ROW_H / 2);
                ctx.textBaseline = "alphabetic";
                ctx.textAlign = "left";

                // Conflict-Marker
                const rowWarn = conflicts.find(c => c.row === i);
                if (rowWarn) {
                    ctx.fillStyle = "#ff7744";
                    ctx.font = "11px Arial";
                    ctx.fillText("⚠", delX - 14, y + 17);
                }

                ctx.globalAlpha = 1;
            });

            // Globale Warnungen
            const globalWarns = conflicts.filter(c => c.row === -1);
            let warnY = HEADER_H + rows.length * ROW_H + 5;
            for (const w of globalWarns) {
                ctx.fillStyle = w.level === "warn" ? "#ff8844" : "#88aaee";
                ctx.font = "10px 'Segoe UI',Arial";
                ctx.fillText(w.msg, PAD, warnY + 12);
                warnY += 20;
            }

            // ── "+" Add-Button Row (direkt unter letzter LoRA) ──────────
            const addY = HEADER_H + rows.length * ROW_H + (globalWarns.length > 0 ? globalWarns.length * 20 + 4 : 0);
            const addHov = uls.hoverZone === "addRow";
            ctx.fillStyle = addHov ? "#1a2a1a" : "#161620";
            ctx.fillRect(0, addY, W, ROW_H);
            ctx.strokeStyle = "#252535"; ctx.lineWidth = 0.5;
            ctx.beginPath(); ctx.moveTo(0, addY); ctx.lineTo(W, addY); ctx.stroke();
            // "＋" zentriert
            ctx.fillStyle = addHov ? "#6acc6a" : "#3a5a3a";
            ctx.font = "bold 14px Arial";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("＋", W / 2, addY + ROW_H / 2);
            ctx.textBaseline = "alphabetic";
            ctx.textAlign = "left";

            // ── Weight/CLIP header tooltip — drawn LAST (v304) ──────────
            if (uls.hoverZone === "weightHdr" && uls._weightHdrRect) {
                const hr = uls._weightHdrRect;
                const lines = [
                    "Weight / CLIP Strength",
                    "Click: model weight.  Shift+Click: set a per-LoRA",
                    "CLIP strength (decoupled).  Shift+\u25C0 \u25B6 steps it.",
                    "Enter the model weight again to re-link.",
                ];
                const LH = 13, PAD_T = 6, TW = 250;
                const TH = PAD_T * 2 + LH * lines.length;
                // Header sits near the right edge → anchor the tooltip LEFT
                const TX = Math.max(PAD, hr.x - TW - 8);
                const TY = Math.max(2, hr.y - 2);
                ctx.fillStyle = "#0e0e18";
                roundRect(ctx, TX, TY, TW, TH, 5); ctx.fill();
                ctx.strokeStyle = "#6aa0d0aa"; ctx.lineWidth = 1;
                roundRect(ctx, TX, TY, TW, TH, 5); ctx.stroke();
                lines.forEach((line, i) => {
                    ctx.font = i === 0
                        ? "bold 10px 'Segoe UI',Arial"
                        : "9.5px 'Segoe UI',Arial";
                    ctx.fillStyle = i === 0 ? "#6aa0d0" : "#c8c8d8";
                    ctx.textAlign = "left";
                    ctx.fillText(line, TX + 8, TY + PAD_T + 10 + i * LH);
                });
            }

            // ── Pill Tooltip — drawn LAST so rows can't paint over it ────
            if (uls.hoverZone === "flatMode" && uls._flatModePillRect) {
                const pr = uls._flatModePillRect;
                const flat = uls.flatMode || false;
                const accent = flat ? "#f0c050" : "#a060ff";
                const lines = flat ? [
                    "List Stack",
                    "Applies LoRAs top to bottom, ignoring group categories.",
                ] : [
                    "Group Stack",
                    "Applies LoRAs by category:",
                    "ACC→STYL→SCEN→MOTI→SUBJ→DETA→CUST  ·  Use \u2460 badges to reorder.",
                ];
                const LH = 13, PAD_T = 6;
                const TX = pr.x + pr.w + 8;
                const TW = Math.min(W - TX - PAD, 360);
                const TH = PAD_T * 2 + LH * lines.length;
                // Anchor to header — tooltip sits aligned with pill, never extends into rows
                const TY = Math.max(2, pr.y - (TH - pr.h) / 2);
                ctx.fillStyle = "#0e0e18";
                roundRect(ctx, TX, TY, TW, TH, 5); ctx.fill();
                ctx.strokeStyle = accent + "aa"; ctx.lineWidth = 1;
                roundRect(ctx, TX, TY, TW, TH, 5); ctx.stroke();
                lines.forEach((line, i) => {
                    ctx.font = i === 0
                        ? "bold 10px 'Segoe UI',Arial"
                        : "9.5px 'Segoe UI',Arial";
                    ctx.fillStyle = i === 0 ? accent : "#aaaabc";
                    ctx.textAlign = "left"; ctx.textBaseline = "top";
                    ctx.fillText(line, TX + PAD_T, TY + PAD_T + i * LH);
                });
                ctx.textBaseline = "alphabetic";
            }

            ctx.restore();
        };

        // ── Mouse Wheel: Weight per Scroll ändern ────────────────────

        nodeType.prototype.onMouseWheel = function (e, [lx, ly]) {
            const uls = this._uls; if (!uls) return false;
            const W = this.size[0];
            const ri = rowAt(ly);
            const delta = e.deltaY < 0 ? 0.05 : -0.05;
            let changed = false;

            if (ri >= 0 && ri < uls.rows.length) {
                const row      = uls.rows[ri];
                const WEIGHT_W = 72, ARROW_W = 14, btnGap = 4, DEL_W = 18;
                const contentR = W - PAD;
                const delX2    = contentR - DEL_W;
                const weightX  = delX2 - btnGap - WEIGHT_W;
                if (lx >= weightX && lx <= weightX + WEIGHT_W) {
                    row.wLow  = Math.round(Math.max(-10, Math.min(10, row.wLow + delta)) * 100) / 100;
                    row.wHigh = row.wLow;
                    changed = true;
                }
            }
            if (changed) {
                this._ulsSync();
                app.graph?.setDirtyCanvas(true, false);
                e.preventDefault?.();
                return true;
            }
            return false;
        };

        // ── Mouse Events ─────────────────────────────────────────────────

        nodeType.prototype.onMouseMove = function (e, [lx, ly]) {
            const uls = this._uls; if (!uls) return;
            const W      = this.size[0];
            const rowIdx = rowAt(ly);
            let dirty = false;

            // v304: Weight/CLIP header hover → explains the Shift interaction
            const whR = uls._weightHdrRect;
            if (whR && lx >= whR.x && lx <= whR.x + whR.w
                    && ly >= whR.y && ly <= whR.y + whR.h) {
                if (uls.hoverZone !== "weightHdr") {
                    uls.hoverZone = "weightHdr"; dirty = true;
                }
                if (rowIdx !== uls.hoverRow) { uls.hoverRow = rowIdx; dirty = true; }
                if (dirty) app.graph?.setDirtyCanvas(true, false);
                return;
            }

            // Flat-Mode Pill Hover
            const fmR = uls._flatModePillRect;
            if (fmR && lx >= fmR.x && lx <= fmR.x + fmR.w
                    && ly >= fmR.y && ly <= fmR.y + fmR.h) {
                if (uls.hoverZone !== "flatMode") {
                    uls.hoverZone = "flatMode"; dirty = true;
                }
                if (rowIdx !== uls.hoverRow) { uls.hoverRow = rowIdx; dirty = true; }
                if (dirty) app.graph?.setDirtyCanvas(true, false);
                return;
            }

            if (rowIdx !== uls.hoverRow) { uls.hoverRow = rowIdx; dirty = true; }

            // "+" Add-Row Hover
            const addRowY = HEADER_H + uls.rows.length * ROW_H;
            const onAddRow = ly >= addRowY && ly < addRowY + ROW_H;
            const newZoneAdd = onAddRow ? "addRow" : "";

            const inFooter = ly > this.size[1] - FOOTER_H;
            // ℹ hover im Footer
            if (inFooter) {
                const footY2   = this.size[1] - FOOTER_H + 6;
                const sliderMid2 = footY2 + 14;
                const infoX3   = W - 16, infoY3 = sliderMid2;
                const onInfo   = Math.sqrt((lx-infoX3)**2 + (ly-infoY3)**2) < 12;
                if (onInfo !== (uls.hoverZone === "multInfo")) {
                    uls.hoverZone = onInfo ? "multInfo" : "";
                    dirty = true;
                    if (onInfo) {
                        showMultiplierTooltip(infoX3, infoY3, this);
                    } else {
                        document.getElementById("uls-mult-tooltip")?.remove();
                    }
                }
            } else if (onAddRow) {
                if (uls.hoverZone !== "addRow") { uls.hoverZone = "addRow"; dirty = true; }
            }

            // Zone-Hover berechnen
            if (rowIdx >= 0 && rowIdx < uls.rows.length) {
                const row      = uls.rows[rowIdx];
                const W        = this.size[0];
                const y        = HEADER_H + rowIdx * ROW_H;
                const THUMB_W  = 30;
                const thumbX   = PAD + 48;
                const insertW  = 28, btnGap = 4;
                const GRP_W    = 50;
                const WEIGHT_W = 72, ARROW_W = 14;
                const DEL_W    = 18;
                const nameX    = thumbX + THUMB_W + 4;
                const contentR = W - PAD;
                const delX     = contentR - DEL_W;
                const weightX  = delX - btnGap - WEIGHT_W;
                const grpPillX = weightX - btnGap - GRP_W;
                const insertX  = grpPillX - btnGap - insertW;
                const nameMaxW = insertX - btnGap - nameX;
                const wArrowLX = weightX;
                const wArrowRX = weightX + WEIGHT_W - ARROW_W;

                let zone = "";
                if (lx >= PAD && lx <= PAD + 14 && ly >= y + 1 && ly <= y + ROW_H / 2)               zone = "up";
                else if (lx >= PAD && lx <= PAD + 14 && ly >= y + ROW_H / 2 && ly <= y + ROW_H - 1)  zone = "down";
                else if (lx >= thumbX && lx <= thumbX + THUMB_W)                   zone = "thumb";
                else if (lx >= nameX - 2  && lx <= nameX + nameMaxW + 2)           zone = "name";
                else if (lx >= insertX && lx <= insertX + insertW)                 zone = "insert";
                else if (lx >= grpPillX && lx <= grpPillX + GRP_W)                zone = "grp";
                else if (lx >= wArrowLX && lx <= wArrowLX + ARROW_W)              zone = "wDec";
                else if (lx >= wArrowRX && lx <= wArrowRX + ARROW_W)              zone = "wInc";
                else if (lx >= weightX && lx <= weightX + WEIGHT_W)               zone = "weight";
                else if (lx >= delX && lx <= delX + DEL_W)                        zone = "del";

                if (zone !== uls.hoverZone) { uls.hoverZone = zone; dirty = true; }

                // Preview-Popup beim Thumbnail-Hover
                if (zone === "thumb" && row.name !== "None") {
                    ensurePreview(row.name);
                    const canvas = app.canvas.canvas;
                    const rect   = canvas.getBoundingClientRect();
                    const scale  = app.canvas?.ds?.scale ?? 1;
                    const off    = app.canvas?.ds?.offset ?? {0:0,1:0};
                    const sx = rect.left + (this.pos[0] + lx) * scale + off[0] * scale;
                    const sy = rect.top  + (this.pos[1] + ly) * scale + off[1] * scale;
                    showPopup(row.name, sx, sy);
                } else if (zone !== "thumb" && popupName) {
                    closePopup();
                }
            } else {
                if (uls.hoverZone !== "") { uls.hoverZone = ""; dirty = true; }
                closePopup();
            }

            // (Slider-Drag läuft über nativen pointermove-Handler — nicht hier)

            // Drag-Dest aktualisieren
            if (uls.dragSrc >= 0 && rowIdx >= 0 && rowIdx !== uls.dragDest) {
                uls.dragDest = rowIdx; dirty = true;
            }

            if (dirty) app.graph?.setDirtyCanvas(true, false);
        };

        nodeType.prototype.onMouseLeave = function () {
            const uls = this._uls;
            if (!uls || typeof uls !== "object") return;
            uls.hoverRow = -1; uls.hoverZone = "";
            if (uls.dragSrc >= 0) { uls.dragSrc = -1; uls.dragDest = -1; }
            // Stop slider drag when mouse leaves node
            if (uls._sliderDragging) {
                uls._sliderDragging = false;
                this._ulsSync();
            }
            closePopup();
            app.graph?.setDirtyCanvas(true, false);
        };

        nodeType.prototype.onMouseDown = function (e, [lx, ly]) {
            const uls = this._uls; if (!uls) return false;
            const W    = this.size[0];
            const footY  = this.size[1] - FOOTER_H + 5;

            // ── Flat-Mode Pill Click ────────────────────────────────────
            const fmR = uls._flatModePillRect;
            if (fmR && lx >= fmR.x && lx <= fmR.x + fmR.w
                    && ly >= fmR.y && ly <= fmR.y + fmR.h) {
                uls.flatMode = !uls.flatMode;
                this._ulsSync();
                app.graph?.setDirtyCanvas(true, false);
                return true;
            }

            // ── "+" Add-Row Klick ───────────────────────────────────────
            const addRowY = HEADER_H + uls.rows.length * ROW_H;
            if (ly >= addRowY && ly < addRowY + ROW_H) {
                uls.rows.push(newRow());
                this._ulsResize(); this._ulsSync();
                return true;
            }

            const ri = rowAt(ly);
            if (ri < 0 || ri >= uls.rows.length) return false;
            const row = uls.rows[ri];

            // Drag-Handle (ganz links)
            if (lx < PAD + 4) {
                uls.dragSrc = ri; uls.dragDest = ri;
                return true;
            }

            // Checkbox Toggle
            if (lx >= PAD + 30 && lx <= PAD + 42) {
                row.enabled = !row.enabled;
                app.graph?.setDirtyCanvas(true, false);
                this._ulsSync(); return true;
            }

            // Zonen-Koordinaten (müssen mit Draw übereinstimmen)
            const THUMB_W  = 30;
            const thumbX   = PAD + 48;
            const insertW  = 28, btnGap = 4;
            const GRP_W    = 50;
            const WEIGHT_W = 72, ARROW_W = 14;
            const DEL_W    = 18;
            const nameX    = thumbX + THUMB_W + 4;
            const contentR = W - PAD;
            const delX     = contentR - DEL_W;
            const weightX  = delX - btnGap - WEIGHT_W;
            const grpPillX = weightX - btnGap - GRP_W;
            const insertX  = grpPillX - btnGap - insertW;
            const nameMaxW = insertX - btnGap - nameX;
            const wArrowLX = weightX;
            const wArrowRX = weightX + WEIGHT_W - ARROW_W;

            // ── ▲▼ Reihenfolge ─────────────────────────────────────────
            if (ri >= 0 && ri < uls.rows.length && lx >= PAD && lx <= PAD + 14) {
                const y = HEADER_H + ri * ROW_H;
                // ▲ Nach oben (obere Hälfte der Row)
                if (ly >= y + 1 && ly <= y + ROW_H / 2) {
                    if (ri > 0) {
                        const [r] = uls.rows.splice(ri, 1);
                        uls.rows.splice(ri - 1, 0, r);
                        this._ulsSync(); this._ulsResize();
                        this._ulsHideConfigWidget();
                    }
                    return true;
                }
                // ▼ Nach unten (untere Hälfte der Row)
                if (ly >= y + ROW_H / 2 && ly <= y + ROW_H - 1) {
                    if (ri < uls.rows.length - 1) {
                        const [r] = uls.rows.splice(ri, 1);
                        uls.rows.splice(ri + 1, 0, r);
                        this._ulsSync(); this._ulsResize();
                        this._ulsHideConfigWidget();
                    }
                    return true;
                }
            }

            // ── Thumbnail Klick: Gruppen+Preview Overlay ─────────────
            if (lx >= thumbX && lx <= thumbX + THUMB_W) {
                openGroupPreviewOverlay(row, e, this);
                return true;
            }

            // ── ↵ Insert-Button: Trigger in CLIP einfügen ──────────────
            if (lx >= insertX && lx <= insertX + insertW) {
                const toastPos = { x: e.clientX - 10, y: e.clientY - 36 };

                if (row.name === "None") {
                    showInsertToast("No LoRA selected", false, toastPos);
                    return true;
                }
                const doInsert = (meta) => {
                    const triggers = deriveTriggers(row.name, meta);
                    if (triggers.length <= 1) {
                        const text = `(${triggers[0]}:1.00)`;
                        showInsertToast(text, insertTriggerAtCursor(text), toastPos);
                    } else {
                        openTriggerSelectPopup(triggers, "1.00", e);
                    }
                };
                const cached = metaCache.get(row.name);
                if (cached !== undefined) doInsert(cached);
                else fetchMeta(row.name).then(doInsert);
                return true;
            }

            // ── Name-Button: LoRA auswählen via DOM-Select ─────────────
            if (lx >= nameX - 2 && lx <= nameX + nameMaxW + 2) {
                openLoraSelect(row, _loraList, e, this);
                return true;
            }

            // ── ◀ Weight verringern (v302: Shift = CLIP strength) ──────
            if (lx >= wArrowLX && lx <= wArrowLX + ARROW_W) {
                if (e.shiftKey) {
                    const base = (typeof row.wClip === "number") ? row.wClip : row.wLow;
                    row.wClip = Math.round(Math.max(-10, base - 0.05) * 100) / 100;
                } else {
                    row.wLow  = Math.round(Math.max(-10, row.wLow - 0.05) * 100) / 100;
                    row.wHigh = row.wLow;
                }
                app.graph?.setDirtyCanvas(true, false); this._ulsSync();
                return true;
            }
            // ── ▶ Weight erhöhen (v302: Shift = CLIP strength) ──────────
            if (lx >= wArrowRX && lx <= wArrowRX + ARROW_W) {
                if (e.shiftKey) {
                    const base = (typeof row.wClip === "number") ? row.wClip : row.wLow;
                    row.wClip = Math.round(Math.min(10, base + 0.05) * 100) / 100;
                } else {
                    row.wLow  = Math.round(Math.min(10, row.wLow + 0.05) * 100) / 100;
                    row.wHigh = row.wLow;
                }
                app.graph?.setDirtyCanvas(true, false); this._ulsSync();
                return true;
            }

            // ── Weight Box (Klick = Eingabe; v302: Shift-Klick = CLIP) ──
            if (lx >= weightX + ARROW_W && lx <= weightX + WEIGHT_W - ARROW_W) {
                if (e.shiftKey) {
                    const cur = (typeof row.wClip === "number") ? row.wClip : row.wLow;
                    showWeightInput(e, cur, (v) => {
                        // entering the model weight re-links CLIP to it
                        if (v === row.wLow) delete row.wClip; else row.wClip = v;
                        app.graph?.setDirtyCanvas(true, false); this._ulsSync();
                    }, "CLIP Strength", "#6aa0d0");
                } else {
                    showWeightInput(e, row.wLow, (v) => {
                        row.wLow = v; row.wHigh = v;
                        app.graph?.setDirtyCanvas(true, false); this._ulsSync();
                    }, "Weight");
                }
                return true;
            }

            // ── ✕ Delete Row ───────────────────────────────────────────
            if (lx >= delX && lx <= delX + DEL_W) {
                uls.rows.splice(ri, 1);
                this._ulsResize(); this._ulsSync();
                return true;
            }

            // ── GRP-Pill Klick ──────────────────────────────────────────
            if (lx >= grpPillX && lx <= grpPillX + GRP_W) {
                const _modeForGrp = ((uls.groupModes || {})[row.group] || "SEQ").toUpperCase();
                const rowY = HEADER_H + ri * ROW_H;

                // Order badge hit area — top-left inside GRP pill (cx+7, cy+8, r=8)
                if (!uls.flatMode && row.group !== "—") {
                    const obcx = grpPillX + 7, obcy = rowY + 8;
                    if (Math.sqrt((lx - obcx)**2 + (ly - obcy)**2) <= 8) {
                        // Dedicated integer input — no decimals, range 1-8
                        document.getElementById("uls-weight-input")?.remove();
                        const canvasScale = app.canvas?.ds?.scale ?? 1;
                        const orderClickPos = { x: e.clientX, y: e.clientY }; // for the conflict dialog
                        const el = document.createElement("div");
                        el.id = "uls-weight-input";
                        el.style.cssText = [
                            `position:fixed`,
                            `left:${e.clientX - 50}px`,
                            `top:${e.clientY - 44}px`,
                            "z-index:999999",
                            "background:#14141e",
                            "border:1px solid #3a3a5a",
                            "border-radius:8px",
                            "padding:10px 12px",
                            "box-shadow:0 4px 20px rgba(0,0,0,0.8)",
                            "font:12px 'Segoe UI',Arial,sans-serif",
                        ].join(";");
                        const lbl = document.createElement("div");
                        lbl.style.cssText = "color:#888;font-size:10px;margin-bottom:6px;";
                        lbl.textContent = "Stack Order  (1–8,  0 = clear)";
                        el.appendChild(lbl);
                        const inp = document.createElement("input");
                        inp.type = "text";
                        inp.inputMode = "numeric";
                        const curVal = ((uls.groupOrder || {})[row.group] ?? "");
                        inp.value = curVal !== "" ? String(curVal) : "";
                        inp.style.cssText = [
                            "width:60px",
                            "background:#1a1030",
                            "border:1px solid #f0c050",
                            "border-radius:4px",
                            "color:#f0c050",
                            "padding:4px 8px",
                            "font:bold 14px monospace",
                            "outline:none",
                            "text-align:center",
                        ].join(";");
                        el.appendChild(inp);
                        document.body.appendChild(el);
                        requestAnimationFrame(() => {
                            el.style.transform = `scale(${canvasScale})`;
                            el.style.transformOrigin = "top left";
                            const r = el.getBoundingClientRect();
                            if (r.right  > window.innerWidth  - 8) el.style.left = `${window.innerWidth  - r.width  - 8}px`;
                            if (r.bottom > window.innerHeight - 8) el.style.top  = `${window.innerHeight - r.height - 8}px`;
                            inp.focus(); inp.select();
                        });
                        const applyOrder = () => {
                            el.remove();
                            if (!uls.groupOrder) uls.groupOrder = {};
                            const raw = parseInt(inp.value, 10);
                            if (isNaN(raw) || raw <= 0) {
                                delete uls.groupOrder[row.group];
                                this._ulsSync(); app.graph?.setDirtyCanvas(true, false);
                                return;
                            }
                            const n = Math.max(1, Math.min(8, raw));

                            // Which group (if any) currently holds this number?
                            const conflict = Object.entries(uls.groupOrder)
                                .find(([g, num]) => g !== row.group && num === n);

                            if (conflict) {
                                const otherGroup = conflict[0];
                                // Is the conflicting group still present among the
                                // live rows? A group goes "orphaned" when no active
                                // LoRA row carries that category anymore, yet its
                                // groupOrder entry lingers (invisible, no badge to
                                // see or clear). Such a ghost has no say — reclaim
                                // its number silently. Only a CURRENTLY VISIBLE
                                // group triggers a confirm, because that's a real
                                // user-facing collision they can act on.
                                const liveGroups = new Set(
                                    (uls.rows || [])
                                        .map(r => r && r.group)
                                        .filter(g => g && g !== "—"));
                                const otherIsOrphan = !liveGroups.has(otherGroup);

                                if (otherIsOrphan) {
                                    // Reclaim from the ghost — drop its stale entry.
                                    delete uls.groupOrder[otherGroup];
                                    uls.groupOrder[row.group] = n;
                                    this._ulsSync(); app.graph?.setDirtyCanvas(true, false);
                                    return;
                                }

                                // Visible collision → ask before stealing the slot.
                                const otherLabel = otherGroup.toUpperCase();
                                const thisLabel = row.group.toUpperCase();
                                // anchor the dialog near the click
                                const sp = orderClickPos;
                                showConfirmDialog({
                                    title: "Stack order already in use",
                                    message:
                                        `Stack order ${n} is already assigned to group ` +
                                        `"${otherLabel}".\n\nReassign ${n} to "${thisLabel}"? ` +
                                        `"${otherLabel}" will be left without an order number.`,
                                    confirmLabel: `Reassign to ${thisLabel}`,
                                    cancelLabel: "Keep current",
                                    screenPos: sp,
                                    onConfirm: () => {
                                        delete uls.groupOrder[otherGroup];
                                        uls.groupOrder[row.group] = n;
                                        this._ulsSync(); app.graph?.setDirtyCanvas(true, false);
                                    },
                                    onCancel: () => {
                                        // Flash the existing holder so the user sees who owns it.
                                        uls._orderConflictGroup = otherGroup;
                                        app.graph?.setDirtyCanvas(true, false);
                                        setTimeout(() => {
                                            uls._orderConflictGroup = null;
                                            app.graph?.setDirtyCanvas(true, false);
                                        }, 1200);
                                    },
                                });
                                return;
                            }
                            uls.groupOrder[row.group] = n;
                            this._ulsSync(); app.graph?.setDirtyCanvas(true, false);
                        };
                        inp.addEventListener("keydown", (ev) => {
                            if (ev.key === "Enter")  { ev.preventDefault(); applyOrder(); }
                            if (ev.key === "Escape") { el.remove(); }
                            ev.stopPropagation();
                        });
                        setTimeout(() => {
                            const closeH = (ev) => {
                                if (!el.contains(ev.target)) {
                                    applyOrder();
                                    document.removeEventListener("pointerdown", closeH, true);
                                }
                            };
                            document.addEventListener("pointerdown", closeH, true);
                        }, 200);
                        return true;
                    }
                }

                // DARE variant badge toggle (top-right)
                const badgeCX = grpPillX + GRP_W - 5;
                const badgeCY = rowY + 16;
                if (_modeForGrp === "DARE"
                        && lx >= badgeCX - 7 && lx <= badgeCX + 7
                        && ly >= badgeCY - 7 && ly <= badgeCY + 7) {
                    if (!uls.groupDare) uls.groupDare = {};
                    const curDV = uls.groupDare[row.group] || "channel";
                    uls.groupDare[row.group] = (curDV === "channel") ? "element" : "channel";
                    this._ulsSync();
                    app.graph?.setDirtyCanvas(true, false);
                    return true;
                }
                // Rest of pill: cycle group
                const cur = GROUPS.indexOf(row.group);
                row.group = GROUPS[(cur + 1) % GROUPS.length];
                api.fetchApi("/uls/groups", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({ lora_name: row.name, group: row.group })
                }).catch(() => {});
                app.graph?.setDirtyCanvas(true, false);
                this._ulsSync(); return true;
            }

            return false;
        };

        nodeType.prototype.onMouseUp = function (e, [lx, ly]) {
            const uls = this._uls; if (!uls) return false;
            // Slider-Drag beenden
            if (uls._sliderDragging) {
                uls._sliderDragging = false;
                this._ulsSync();
                app.graph?.setDirtyCanvas(true, false);
                return true;
            }
            if (uls.dragSrc < 0) return false;
            const src = uls.dragSrc, dst = uls.dragDest;
            uls.dragSrc = -1; uls.dragDest = -1;
            if (src !== dst && dst >= 0 && dst < uls.rows.length) {
                const [r] = uls.rows.splice(src, 1);
                // Einfügen: wenn dst > src schon korrigiert durch splice
                const insertAt = dst > src ? dst - 1 : dst;
                uls.rows.splice(insertAt, 0, r);
                this._ulsSync();
            }
            app.graph?.setDirtyCanvas(true, false);
            return true;
        };

        // Rechtsklick-Kontextmenü (auf Row, NICHT auf GRP-Pill — das hat
        // sein eigenes Popup. Dieses Tree-Menü deckt nur Row-Verwaltung ab.)
        const _origMenu = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
            _origMenu?.apply(this, arguments);
            const uls = this._uls; if (!uls) return;
            // Group/List mode toggle in the context menu — always available
            // (independent of the hovered row). The canvas pill stays the
            // primary control; this is just a convenient right-click alternative.
            options.push(null, {
                content: uls.flatMode ? "⬡ Switch to Group Stack" : "⬡ Switch to List Stack",
                callback: () => {
                    uls.flatMode = !uls.flatMode;
                    this._ulsSync?.();
                    app.graph?.setDirtyCanvas(true, false);
                },
            });
            const i   = uls.hoverRow;
            if (i < 0 || i >= uls.rows.length) return;

            options.push(null,
                { content: "⬆ Move up", callback: () => {
                    if (i > 0) { [uls.rows[i-1], uls.rows[i]] = [uls.rows[i], uls.rows[i-1]]; this._ulsSync(); app.graph?.setDirtyCanvas(true,false); }
                }},
                { content: "⬇ Move down", callback: () => {
                    if (i < uls.rows.length-1) { [uls.rows[i], uls.rows[i+1]] = [uls.rows[i+1], uls.rows[i]]; this._ulsSync(); app.graph?.setDirtyCanvas(true,false); }
                }},
                { content: "🗑 Delete row", callback: () => {
                    uls.rows.splice(i, 1); this._ulsResize(); this._ulsSync();
                }},
                { content: "📋 Copy stack info", callback: () => {
                    const txt = uls.rows.filter(r => r.enabled && r.name !== "None")
                        .map((r,j) => `[${r.group}] ${r.name}  ×${r.wLow}`)
                        .join("\n");
                    navigator.clipboard?.writeText(txt);
                }}
            );
        };
    }
});

// ─── Hilfsfunktionen ───────────────────────────────────────────────────────

/**
 * Öffnet ein DOM-Dropdown zur LoRA-Auswahl.
 * Positioniert direkt unter dem angeklickten Name-Button.
 * Filtert per Texteingabe — bei 20+ LoRAs unverzichtbar.
 */
/**
 * Öffnet ein großes Preview-Overlay mit Bild oder Video.
 * Klick irgendwo schließt es.
 */
function openGroupPreviewOverlay(row, e, node) {
    document.getElementById("uls-group-overlay")?.remove();

    const meta = metaCache.get(row.name);
    const pv   = previewCache.get(row.name);
    ensurePreview(row.name);

    const el = document.createElement("div");
    el.id = "uls-group-overlay";
    el.style.cssText = [
        "position:fixed",
        `left:${e.clientX + 10}px`,
        `top:${e.clientY - 20}px`,
        "z-index:999999",
        "width:240px",
        "background:#14141e",
        "border:1px solid #3a3a5a",
        "border-radius:10px",
        "padding:10px",
        "box-shadow:0 8px 32px rgba(0,0,0,0.8)",
        "font:12px 'Segoe UI',Arial,sans-serif",
        "color:#ccc",
    ].join(";");

    // LoRA Name
    const title = document.createElement("div");
    title.style.cssText = "font-weight:bold;color:#a0c4ff;margin-bottom:6px;font-size:11px;word-break:break-all;";
    title.textContent = row.name.split(/[/\\]/).pop().replace(/\.safetensors$/i,"");
    el.appendChild(title);

    // Preview
    if (pv?.img) {
        const img = document.createElement("img");
        img.src = pv.img.src;
        img.style.cssText = "width:100%;border-radius:6px;display:block;margin-bottom:8px;";
        el.appendChild(img);
    } else if (pv?.vid) {
        const v = document.createElement("video");
        v.src = api.apiURL(`/uls/preview/video?lora=${encodeURIComponent(row.name)}`);
        v.muted=true; v.loop=true; v.autoplay=true; v.controls=true;
        v.style.cssText = "width:100%;border-radius:6px;display:block;margin-bottom:8px;max-height:200px;";
        el.appendChild(v); v.play().catch(()=>{});
    } else {
        const ph = document.createElement("div");
        ph.style.cssText = "color:#444;font-size:10px;margin-bottom:6px;text-align:center;padding:10px;";
        ph.textContent = "📷 No preview";
        el.appendChild(ph);
    }

    // Trigger Words — editierbar
    const currentTriggers = (() => {
        if (meta?.trigger_words) {
            return typeof meta.trigger_words === "string"
                ? meta.trigger_words
                : Object.keys(meta.trigger_words||{}).slice(0,20).join(", ");
        }
        return "";
    })();

    const twLabel = document.createElement("div");
    twLabel.style.cssText = "color:#666;font-size:9px;margin-bottom:3px;margin-top:6px;";
    twLabel.textContent = "TRIGGER WORDS (editable):";
    el.appendChild(twLabel);

    const twInput = document.createElement("textarea");
    twInput.value = currentTriggers;
    twInput.placeholder = "Trigger words, comma separated…";
    twInput.style.cssText = [
        "width:100%","box-sizing:border-box",
        "background:#1a1a2e","border:1px solid #3a3a5a",
        "border-radius:4px","color:#90c4f9",
        "font:11px 'Segoe UI',Arial","padding:5px 7px",
        "resize:vertical","min-height:40px","max-height:100px",
        "outline:none","margin-bottom:4px",
    ].join(";");
    twInput.addEventListener("mousedown", ev => ev.stopPropagation());
    twInput.addEventListener("keydown",   ev => ev.stopPropagation());
    el.appendChild(twInput);

    const saveBtn = document.createElement("button");
    saveBtn.textContent = "💾 Save triggers";
    saveBtn.style.cssText = [
        "width:100%","padding:4px 8px","margin-bottom:8px",
        "background:#1a2a1a","border:1px solid #3a6a3a",
        "border-radius:4px","color:#6acc6a",
        "font:10px 'Segoe UI',Arial","cursor:pointer",
    ].join(";");
    saveBtn.addEventListener("mousedown", async (ev) => {
        ev.preventDefault(); ev.stopPropagation();
        const newTriggers = twInput.value.trim();
        try {
            const r = await api.fetchApi("/uls/triggers", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({ lora_name: row.name, trigger_words: newTriggers }),
            });
            if (r.ok) {
                metaCache.delete(row.name);
                saveBtn.textContent = "✓ Saved";
                saveBtn.style.color = "#8dff8d";
                setTimeout(() => { saveBtn.textContent = "💾 Save triggers"; saveBtn.style.color = "#6acc6a"; }, 1500);
            } else {
                saveBtn.textContent = "✗ Error";
                saveBtn.style.color = "#ff8d8d";
            }
        } catch {
            saveBtn.textContent = "✗ Connection error";
            saveBtn.style.color = "#ff8d8d";
        }
    });
    el.appendChild(saveBtn);

    // Civitai-Link (wenn civitai_id vorhanden)
    if (meta?.civitai_id) {
        const civLink = document.createElement("a");
        civLink.href = `https://civitai.com/models/${meta.civitai_id}`;
        civLink.target = "_blank";
        civLink.rel = "noopener";
        civLink.style.cssText = [
            "display:block","text-align:center","padding:4px 8px","margin-bottom:4px",
            "background:#1a1a2a","border:1px solid #4a3a5a",
            "border-radius:4px","color:#c084fc",
            "font:10px 'Segoe UI',Arial","text-decoration:none",
            "cursor:pointer",
        ].join(";");
        civLink.textContent = "🌐 View on Civitai";
        civLink.addEventListener("mouseenter", () => civLink.style.background = "#2a1a3a");
        civLink.addEventListener("mouseleave", () => civLink.style.background = "#1a1a2a");
        civLink.addEventListener("mousedown", ev => ev.stopPropagation());
        el.appendChild(civLink);
    } else {
        // Kein civitai_id — Suche-Link anbieten
        const shortName = row.name.split(/[/\\]/).pop().replace(/\.safetensors$/i, "");
        // Split on underscores, hyphens, and CamelCase boundaries → clean search terms
        const searchTerms = shortName
            .replace(/([a-z])([A-Z])/g, "$1 $2")   // CamelCase → "Camel Case"
            .replace(/([A-Z]+)([A-Z][a-z])/g, "$1 $2")
            .replace(/[_\-\.]+/g, " ")               // underscores/hyphens → spaces
            .replace(/\s+/g, " ").trim()
            + " LoRA";
        const civSearch = document.createElement("a");
        civSearch.href = `https://civitai.com/search/models?query=${encodeURIComponent(searchTerms)}`;
        civSearch.target = "_blank";
        civSearch.rel = "noopener";
        civSearch.style.cssText = [
            "display:block","text-align:center","padding:4px 8px","margin-bottom:4px",
            "background:#1a1a2a","border:1px solid #4a5a7a",
            "border-radius:4px","color:#7090d0",
            "font:10px 'Segoe UI',Arial","text-decoration:none","cursor:pointer",
        ].join(";");
        civSearch.textContent = "🔍 Search on Civitai";
        civSearch.title = searchTerms;  // tooltip shows the actual search query
        civSearch.addEventListener("mouseenter", () => { civSearch.style.background = "#1e2a3a"; civSearch.style.color = "#90b0f0"; });
        civSearch.addEventListener("mouseleave", () => { civSearch.style.background = "#1a1a2a"; civSearch.style.color = "#7090d0"; });
        civSearch.addEventListener("mousedown", ev => ev.stopPropagation());
        el.appendChild(civSearch);
    }

    // Fetch from Civitai button (always shown)
    const fetchBtn = document.createElement("button");
    fetchBtn.textContent = "📥 Fetch from Civitai";
    fetchBtn.style.cssText = [
        "display:block","width:100%","text-align:center","padding:4px 8px","margin-bottom:8px",
        "background:#1a1e2a","border:1px solid #2a3a5a",
        "border-radius:4px","color:#60a0e0",
        "font:10px 'Segoe UI',Arial","cursor:pointer",
    ].join(";");
    fetchBtn.addEventListener("mouseenter", () => fetchBtn.style.background = "#1e2a3a");
    fetchBtn.addEventListener("mouseleave", () => fetchBtn.style.background = "#1a1e2a");
    fetchBtn.addEventListener("mousedown", ev => ev.stopPropagation());
    fetchBtn.addEventListener("click", async (ev) => {
        ev.stopPropagation();
        fetchBtn.textContent = "⏳ Fetching…";
        fetchBtn.style.color = "#888";
        try {
            const r = await api.fetchApi("/uls/civitai_fetch", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ lora_name: row.name }),
            });
            const d = await r.json();
            if (d.ok) {
                fetchBtn.textContent = `✓ ${d.model_name || "Found"}`;
                fetchBtn.style.color = "#6acc6a";
                // Update trigger textarea if triggers were found
                if (d.trigger_words && twInput) {
                    twInput.value = d.trigger_words;
                }
                // Invalidate caches so preview reloads
                metaCache.delete(row.name);
                previewCache.delete(row.name);
                ensurePreview(row.name);
                app.graph?.setDirtyCanvas(true, false);
            } else {
                fetchBtn.textContent = `✗ ${d.error || "Not found"}`;
                fetchBtn.style.color = "#ff8d8d";
            }
        } catch(e) {
            fetchBtn.textContent = "✗ Connection error";
            fetchBtn.style.color = "#ff8d8d";
        }
        setTimeout(() => {
            fetchBtn.textContent = "📥 Fetch from Civitai";
            fetchBtn.style.color = "#60a0e0";
        }, 3000);
    });
    el.appendChild(fetchBtn);

    // Gruppen-Auswahl
    const grpLabel = document.createElement("div");
    grpLabel.style.cssText = "color:#666;font-size:9px;margin-bottom:4px;";
    grpLabel.textContent = "GROUP:";
    el.appendChild(grpLabel);

    const grpBtns = document.createElement("div");
    grpBtns.style.cssText = "display:flex;flex-wrap:wrap;gap:4px;";

    const GROUP_COLORS_DOM = {
        "—":"#505060","acc":"#e85d5d","style":"#8b6fe8","scene":"#4a9eff",
        "motion":"#43c9c9","subject":"#ff6b9d","detail":"#51cf66",
        "custom":"#ff8c42"
    };

    for (const [grp, color] of Object.entries(GROUP_COLORS_DOM)) {
        const btn = document.createElement("button");
        btn.style.cssText = [
            `background:${row.group===grp ? color+"44" : color+"11"}`,
            `border:1px solid ${row.group===grp ? color : color+"44"}`,
            `color:${color}`,
            "border-radius:4px","padding:2px 6px",
            "font:9px 'Segoe UI',Arial","cursor:pointer",
        ].join(";");
        btn.textContent = grp === "—" ? "—" : grp;
        btn.addEventListener("mousedown", (ev) => {
            ev.preventDefault(); ev.stopPropagation();
            row.group = grp;
            // Persistent speichern
            api.fetchApi("/uls/groups", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({ lora_name: row.name, group: grp })
            }).catch(e => console.warn("[ULS] Gruppe speichern:", e));
            // v267 (N-2): push the changed group into the hidden uls_config
            // widget IMMEDIATELY. Without this, a queue right after changing
            // the group via this overlay still ran with the OLD group (the
            // widget only updated on the next unrelated interaction). The
            // DARE-variant buttons below already did this; the GRP-pill
            // cycle path did too — this was the one path that missed it.
            // node is null when opened from the Engine (read-only feel) —
            // optional chaining keeps that path a no-op.
            node?._ulsSync?.();
            app.graph?.setDirtyCanvas(true, false);
            el.remove();
        });
        grpBtns.appendChild(btn);
    }
    el.appendChild(grpBtns);

    // ── DARE-Variant Auswahl (nur wenn Gruppe einen DARE-Modus hat) ──────
    const _currentMode = ((node?._uls?.groupModes || {})[row.group] || "SEQ").toUpperCase();
    if (_currentMode === "DARE" && row.group !== "—") {
        const dvLabel = document.createElement("div");
        dvLabel.style.cssText = "color:#666;font-size:9px;margin-top:10px;margin-bottom:4px;";
        dvLabel.textContent = "DARE VARIANT:";
        el.appendChild(dvLabel);

        const dvBtns = document.createElement("div");
        dvBtns.style.cssText = "display:flex;gap:6px;";
        const _curDV = ((node?._uls?.groupDare || {})[row.group] || "channel");

        for (const [dvKey, dvLabel2, dvColor] of [
            ["channel", "CHAN", "#7af0c0"],
            ["element", "ELEM", "#f0c87a"],
        ]) {
            const isActive = _curDV === dvKey;
            const b = document.createElement("button");
            b.style.cssText = [
                `background:${isActive ? dvColor + "33" : "#111"}`,
                `border:1px solid ${isActive ? dvColor : dvColor + "44"}`,
                `color:${isActive ? dvColor : dvColor + "88"}`,
                "border-radius:4px","padding:3px 10px",
                "font:bold 10px 'Segoe UI',Arial","cursor:pointer","flex:1",
            ].join(";");
            b.textContent = dvLabel2;
            b.addEventListener("mousedown", (ev) => {
                ev.preventDefault(); ev.stopPropagation();
                if (!node._uls.groupDare) node._uls.groupDare = {};
                node._uls.groupDare[row.group] = dvKey;
                node._ulsSync?.();
                app.graph?.setDirtyCanvas(true, false);
                el.remove();
            });
            dvBtns.appendChild(b);
        }
        el.appendChild(dvBtns);
    }

    document.body.appendChild(el);

    // Viewport-Korrektur
    requestAnimationFrame(() => {
        const r = el.getBoundingClientRect();
        if (r.right  > window.innerWidth  - 8) el.style.left = `${e.clientX - r.width - 10}px`;
        if (r.bottom > window.innerHeight - 8) el.style.top  = `${e.clientY - r.height}px`;
    });

    // Schließen bei Klick außerhalb
    const closeH = (ev) => {
        if (!el.contains(ev.target)) {
            el.remove();
            document.removeEventListener("mousedown", closeH, true);
        }
    };
    setTimeout(() => document.addEventListener("mousedown", closeH, true), 50);
}

function openPreviewOverlay(loraName, e) {
    document.getElementById("uls-preview-overlay")?.remove();

    const meta = metaCache.get(loraName);
    const pv   = previewCache.get(loraName);

    // Backdrop
    const backdrop = document.createElement("div");
    backdrop.id = "uls-preview-overlay";
    backdrop.style.cssText = [
        "position:fixed", "inset:0", "z-index:999999",
        "background:rgba(0,0,0,0.88)",
        "display:flex", "flex-direction:column",
        "align-items:center", "justify-content:center",
        "cursor:pointer",
        "padding:20px",
        "box-sizing:border-box",
        "overflow:auto",
    ].join(";");

    // Titel
    const title = document.createElement("div");
    const shortName = loraName.split(/[/\\]/).pop().replace(/\.safetensors$/i, "");
    title.style.cssText = "color:#ddd;font:14px 'Segoe UI',Arial;margin-bottom:12px;max-width:90vw;text-align:center;";
    title.textContent = shortName;
    backdrop.appendChild(title);

    // Inhalt
    if (pv?.vid) {
        const vid = document.createElement("video");
        vid.src = api.apiURL(`/uls/preview/video?lora=${encodeURIComponent(loraName)}`);
        vid.controls = true;
        vid.autoplay = true;
        vid.loop = true;
        vid.muted = false;
        vid.style.cssText = "max-width:90vw;max-height:80vh;border-radius:10px;box-shadow:0 0 60px rgba(0,0,0,0.8);";
        vid.addEventListener("click", ev => ev.stopPropagation());
        backdrop.appendChild(vid);
    } else if (pv?.img) {
        const img = document.createElement("img");
        img.src = pv.img.src;
        img.style.cssText = "max-width:90vw;max-height:80vh;border-radius:10px;box-shadow:0 0 60px rgba(0,0,0,0.8);object-fit:contain;";
        img.addEventListener("click", ev => ev.stopPropagation());
        backdrop.appendChild(img);
    } else {
        ensurePreview(loraName);
        const msg = document.createElement("div");
        msg.style.cssText = [
            "color:#666", "font:13px 'Segoe UI',Arial",
            "padding:40px", "text-align:center",
            "border:1px dashed #333", "border-radius:10px",
            "min-width:200px",
        ].join(";");
        msg.innerHTML = "📷 No preview<br><span style='font-size:11px;color:#444;margin-top:6px;display:block'>" +
            "Place a .jpg or .png<br>next to the .safetensors file</span>";
        backdrop.appendChild(msg);
    }

    // Trigger Words sauber formatieren
    if (meta?.trigger_words) {
        let tw = meta.trigger_words;
        if (typeof tw === "object" && tw !== null) {
            // ss_tag_frequency: {"tag": count} → nur Tags
            tw = Object.keys(tw).slice(0, 15).join(", ");
        }
        tw = String(tw || "").trim();
        // JSON-artige Strings erkennen und extrahieren
        if (tw.startsWith("{") || tw.startsWith("[")) {
            try {
                const parsed = JSON.parse(tw);
                if (typeof parsed === "object") {
                    tw = Object.keys(parsed).slice(0, 15).join(", ");
                }
            } catch {}
        }
        if (tw && tw.length > 1) {
            const twEl = document.createElement("div");
            twEl.style.cssText = [
                "color:#90c4f9", "font:12px 'Segoe UI',Arial",
                "margin-top:10px", "max-width:80vw",
                "text-align:center", "line-height:1.6",
                "background:rgba(0,0,100,0.3)",
                "padding:8px 14px", "border-radius:6px",
            ].join(";");
            twEl.textContent = "Trigger: " + tw.slice(0, 300);
            backdrop.appendChild(twEl);
        }
    }

    const hint = document.createElement("div");
    hint.style.cssText = "color:#555;font:11px Arial;margin-top:16px;";
    hint.textContent = "Click to close  •  Escape";
    backdrop.appendChild(hint);

    backdrop.addEventListener("click", () => backdrop.remove());
    document.addEventListener("keydown", function esc(ev) {
        if (ev.key === "Escape") { backdrop.remove(); document.removeEventListener("keydown", esc); }
    });

    document.body.appendChild(backdrop);
}

function showWeightInput(e, currentVal, onConfirm, label, accent) {
    // v304: accent themes the popup — orange (default) for model weight,
    // CLIP blue for the decoupled CLIP strength, matching the in-cell color.
    accent = accent || "#f0a030";
    document.getElementById("uls-weight-input")?.remove();

    // Match the canvas zoom: when the user zooms in, nodes appear larger,
    // so the input must scale up too. When zoomed out, nodes are small —
    // the input scales down correspondingly. This keeps the input visually
    // consistent with the row it belongs to at any zoom level.
    const canvasScale = app.canvas?.ds?.scale ?? 1;

    const el = document.createElement("div");
    el.id = "uls-weight-input";
    el.style.cssText = [
        `position:fixed`,
        `left:${e.clientX - 50}px`,
        `top:${e.clientY - 44}px`,
        "z-index:999999",
        "background:#14141e",
        "border:1px solid #3a3a5a",
        "border-radius:8px",
        "padding:10px 12px",
        "box-shadow:0 4px 20px rgba(0,0,0,0.8)",
        "font:12px 'Segoe UI',Arial,sans-serif",
    ].join(";");

    const lbl = document.createElement("div");
    lbl.style.cssText = `color:${accent};font-size:10px;margin-bottom:6px;`;
    lbl.textContent = label || "Weight";
    el.appendChild(lbl);

    const inp = document.createElement("input");
    inp.type = "text";
    inp.value = currentVal.toFixed(2);
    inp.style.cssText = [
        "width:80px",
        accent === "#f0a030" ? "background:#221a10" : "background:#101a22",
        `border:1px solid ${accent}`,
        "border-radius:4px",
        `color:${accent}`,
        "padding:4px 8px",
        "font:bold 13px monospace",
        "outline:none",
        "text-align:center",
    ].join(";");
    el.appendChild(inp);
    document.body.appendChild(el);

    requestAnimationFrame(() => {
        // Apply scale matching the canvas zoom, anchored at the click position
        // so the element grows/shrinks around the cursor.
        el.style.transform       = `scale(${canvasScale})`;
        el.style.transformOrigin = "top left";

        // Viewport correction AFTER scale — uses the actual rendered size.
        const r = el.getBoundingClientRect();
        if (r.right  > window.innerWidth  - 8) el.style.left = `${window.innerWidth  - r.width  - 8}px`;
        if (r.bottom > window.innerHeight - 8) el.style.top  = `${window.innerHeight - r.height - 8}px`;

        inp.focus();
        inp.setSelectionRange(0, inp.value.length);
    });

    function confirm() {
        const n = parseFloat(inp.value);
        if (!isNaN(n)) onConfirm(Math.max(-10, Math.min(10, n)));
        el.remove();
    }

    inp.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter")  { ev.preventDefault(); confirm(); }
        if (ev.key === "Escape") { el.remove(); }
        ev.stopPropagation();
    });

    // Schließen bei Klick außerhalb — mit Delay damit Input erst fokussiert
    setTimeout(() => {
        const closeHandler = (ev) => {
            if (!el.contains(ev.target)) {
                confirm();
                document.removeEventListener("pointerdown", closeHandler, true);
            }
        };
        document.addEventListener("pointerdown", closeHandler, true);
    }, 200);
}

function openLoraSelect(row, loraList, e, node) {
    // Bestehende Selects schließen
    document.getElementById("uls-lora-select")?.remove();

    // Falls Liste noch nicht geladen: jetzt laden und nach kurzer Pause neu öffnen
    if (_loraList.length === 0 && !_loraListLoading) {
        loadLoraList().then(() => openLoraSelect(row, _loraList, e, node));
        return;
    }
    // Immer aktuelle globale Liste verwenden
    const allLoras = _loraList.length > 0 ? _loraList : loraList;

    const canvas  = app.canvas.canvas;
    const rect    = canvas.getBoundingClientRect();
    const scale   = app.canvas?.ds?.scale ?? 1;
    const off     = app.canvas?.ds?.offset ?? {0:0,1:0};

    // Canvas-Position → Screen-Position
    const sx = rect.left + (node.pos[0] + 56) * scale + off[0] * scale;
    const sy = rect.top  + (node.pos[1] + (app.canvas.ds ? 0 : 0)) * scale + off[1] * scale + e.clientY - rect.top;

    const wrap = document.createElement("div");
    wrap.id = "uls-lora-select";
    wrap.style.cssText = `
        position:fixed; left:${e.clientX}px; top:${e.clientY + 4}px;
        z-index:99999; width:420px;
        background:#14141e; border:1px solid #3a3a5a;
        border-radius:8px; overflow:hidden;
        box-shadow:0 8px 32px rgba(0,0,0,.8);
        font:13px 'Segoe UI',Arial,sans-serif;
    `;

    // Suchfeld
    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = "Search LoRA…";
    input.style.cssText = `
        width:100%; box-sizing:border-box;
        padding:8px 10px; border:none; border-bottom:1px solid #2a2a3a;
        background:#1e1e2e; color:#ddd; font:13px 'Segoe UI',Arial,sans-serif;
        outline:none;
    `;
    wrap.appendChild(input);

    // Liste
    const list = document.createElement("div");
    list.style.cssText = "max-height:260px; overflow-y:auto;";
    wrap.appendChild(list);

    const allItems = ["None", ...allLoras.filter(n => n !== "None")];

    function renderList(filter) {
        list.innerHTML = "";
        const q = filter.toLowerCase().trim();
        // Mit Filter: alle Treffer zeigen; ohne Filter: nach Ordner sortiert, max 300
        let filtered;
        if (q) {
            // Mit Filter: alle Treffer, Ordner-Treffer zuerst
            filtered = allItems
                .filter(n => n.toLowerCase().includes(q))
                .sort((a, b) => {
                    const aHas = a.includes("/") || a.includes("\\");
                    const bHas = b.includes("/") || b.includes("\\");
                    if (aHas && !bHas) return -1;
                    if (!aHas && bHas) return  1;
                    return a.toLowerCase().localeCompare(b.toLowerCase());
                });
        } else {
            // Ohne Filter: erst None, dann Root-LoRAs, dann Unterordner alphabetisch
            const roots   = allItems.filter(n => n !== "None" && !n.includes("/") && !n.includes("\\")).sort((a,b) => a.toLowerCase().localeCompare(b.toLowerCase()));
            const subdirs = allItems.filter(n => n.includes("/") || n.includes("\\")).sort((a,b) => a.toLowerCase().localeCompare(b.toLowerCase()));
            filtered = ["None", ...roots, ...subdirs].slice(0, 400);
        }

        for (const name of filtered) {
            const item = document.createElement("div");
            const short = name === "None" ? "— None —"
                : name.split(/[/\\]/).pop().replace(/\.safetensors$/i, "");
            const folder = name === "None" ? ""
                : /[/\\]/.test(name)
                    ? name.replace(/[/\\][^/\\]+$/, "") : "";

            item.style.cssText = `
                padding:6px 10px; cursor:pointer; color:#ccc;
                border-bottom:1px solid #1e1e2a;
                ${name === row.name ? "background:#1e1e40;" : ""}
            `;
            // Mini-Thumbnail wenn vorhanden
            const pvItem = previewCache.get(name);
            const thumbHtml = pvItem?.img
                ? `<img src="${pvItem.img.src}" style="width:32px;height:32px;object-fit:cover;border-radius:3px;flex-shrink:0;">`
                : `<div style="width:32px;height:32px;background:#1a1a2a;border-radius:3px;flex-shrink:0;display:flex;align-items:center;justify-content:center;color:#333;font-size:10px;">?</div>`;
            item.style.cssText = item.style.cssText + ";display:flex;align-items:center;gap:8px;";
            const textPart = folder
                ? `<div><div style="color:#666;font-size:9px;">${escapeHtml(folder)}/</div><div>${escapeHtml(short)}</div></div>`
                : `<div>${escapeHtml(short)}</div>`;
            item.innerHTML = thumbHtml + textPart;

            item.addEventListener("mouseenter", () => item.style.background = "#28284a");
            item.addEventListener("mouseleave", () => item.style.background = name === row.name ? "#1e1e40" : "");
            item.addEventListener("mousedown", (ev) => {
                ev.preventDefault();
                row.name = name;
                ensurePreview(name);
                // Gespeicherte Gruppe laden
                api.fetchApi(`/uls/groups`)
                    .then(r => r.json())
                    .then(groups => {
                        if (groups[name] && groups[name] !== "—") {
                            row.group = groups[name];
                        }
                        app.graph?.setDirtyCanvas(true, false);
                        node._ulsSync();
                    })
                    .catch(() => {
                        app.graph?.setDirtyCanvas(true, false);
                        node._ulsSync();
                    });
                wrap.remove();
            });
            list.appendChild(item);
        }
        if (filtered.length === 0) {
            list.innerHTML = '<div style="padding:10px;color:#555;text-align:center;">Nothing found</div>';
        }
    }

    renderList("");
    input.addEventListener("input", () => renderList(input.value));

    document.body.appendChild(wrap);

    // Viewport-Korrektur
    requestAnimationFrame(() => {
        const r = wrap.getBoundingClientRect();
        if (r.right > window.innerWidth - 8)
            wrap.style.left = `${window.innerWidth - r.width - 8}px`;
        if (r.bottom > window.innerHeight - 8)
            wrap.style.top = `${e.clientY - r.height - 4}px`;
        input.focus();
    });

    // Schließen bei Klick außerhalb oder auf Canvas
    function doClose() {
        if (!document.getElementById("uls-lora-select")) return;
        wrap.remove();
        document.removeEventListener("pointerdown", closeHandler, true);
        document.removeEventListener("keydown",     keyHandler,   true);
    }

    const closeHandler = (ev) => {
        if (!wrap.contains(ev.target)) doClose();
    };

    const keyHandler = (ev) => {
        if (ev.key === "Escape") doClose();
    };

    setTimeout(() => {
        document.addEventListener("pointerdown", closeHandler, true);
        document.addEventListener("keydown",     keyHandler,   true);
    }, 100);
}

function rowAt(ly) {
    const rel = ly - HEADER_H;
    if (rel < 0) return -1;
    return Math.floor(rel / ROW_H);
}

function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
}

/**
 * Draw a two-line canvas tooltip (label + hint) with automatic text wrapping.
 * @param {CanvasRenderingContext2D} ctx
 * @param {number} TX - left edge
 * @param {number} TY - top edge
 * @param {number} TW - max width
 * @param {string} label - bold first line
 * @param {string} hint  - smaller second line (may wrap)
 * @param {string} color - accent color for label + border
 */
function drawCanvasTooltip(ctx, TX, TY, TW, label, hint, color) {
    const PAD_X = 8, LINE_H = 14;
    const maxTextW = TW - PAD_X * 2;

    // Measure hint wrapping
    ctx.font = "10px 'Segoe UI',Arial";
    const words = hint.split(" ");
    const lines = [];
    let cur = "";
    for (const w of words) {
        const test = cur ? cur + " " + w : w;
        if (ctx.measureText(test).width > maxTextW && cur) {
            lines.push(cur); cur = w;
        } else { cur = test; }
    }
    if (cur) lines.push(cur);

    const TH = 10 + LINE_H + lines.length * LINE_H + 6;

    ctx.fillStyle = "#12121e";
    roundRect(ctx, TX, TY, TW, TH, 5); ctx.fill();
    ctx.strokeStyle = color + "88"; ctx.lineWidth = 1;
    roundRect(ctx, TX, TY, TW, TH, 5); ctx.stroke();

    // Label
    ctx.fillStyle = color;
    ctx.font = "bold 11px 'Segoe UI',Arial"; ctx.textAlign = "left";
    ctx.fillText(label, TX + PAD_X, TY + 14);

    // Hint lines
    ctx.fillStyle = "#888"; ctx.font = "10px 'Segoe UI',Arial";
    lines.forEach((line, i) => {
        ctx.fillText(line, TX + PAD_X, TY + 14 + LINE_H * (i + 1));
    });
    ctx.textAlign = "left";
}

// ════════════════════════════════════════════════════════════════════════════
// ⬡ Polyhedron Engine — slim sibling of the main stack node
// ════════════════════════════════════════════════════════════════════════════
//
// Same Canvas-UI feel but stripped down for engine-class LoRAs:
//   - flat list (no groups → no GRP-pill, no trigger insert)
//   - one global apply-mode for the whole node (S / C / D button row in header)
//   - shares the global LoRA list, preview cache, and helpers
//     (openLoraSelect, showWeightInput, openGroupPreviewOverlay) with the
//     main stack — the helpers operate on a `row` and a `node._ulsSync()`,
//     both of which the engine node provides.
//
// Backend reads JSON: { rows: [{enabled, name, weight}], mode: "SEQ"|"CONCAT"|"DARE" }

const ENGINE_HEADER_H = 90;   // Title bar (~20) + pin zone (~30) + mode-buttons (22) + label (14) + padding
const ENGINE_NODE_TYPE = NODE_TYPE_ENGINE;

/* v877 -- the engine node could not be pushed together, and it lost an
 * aspired width on every workflow load. Both are the v874 wound (ph_switch.js),
 * a generation later, in a node whose twin was healed long ago: the STACK
 * block keeps the user's width ("Preserve user-resized width -- only enforce
 * minimum"), the ENGINE block did not.
 *
 * TWO defects, one family:
 *   1. _engineHideConfigWidget ended with setSize(computeSize()) -- a hard
 *      reset to LiteGraph's MINIMUM, run from onConfigure (every workflow
 *      load) and from _ulsSync (every LoRA pick). An aspired width was gone
 *      after every restart.
 *   2. onResize floored the width at 700 -- a number nothing in the renderer
 *      asks for. The engine ROW is narrower than the stack row (no group
 *      column, no low-weight column), so the floor was inherited, not
 *      measured. It made the node impossible to push together.
 *
 * ENGINE_MIN_W is COMPUTED from the drawn constants, not chosen:
 *   header : PAD + 3*MODE_BTN_W + 2*MODE_GAP + 4 + pillW + PAD      = 196
 *   row    : nameX(90) + NAME_MIN(120) + btnGap + WEIGHT_W
 *            + btnGap + DEL_W + PAD                                 = 316
 * The row is the binding constraint. 320 leaves the name field 124px.
 */
const ENGINE_MIN_W = 320;

/* The width a FRESH engine node opens at. This is comfort, not a floor:
 * onNodeCreated runs once, and grow-only keeps whatever the user then
 * pulls it to. Separating the two is the point of this cut -- the old
 * code used the same 700 for both, so the opening width WAS the floor. */
const ENGINE_START_W = 700;

/* Width is GROW-ONLY: never below ENGINE_MIN_W, never below what the node
 * already holds. Height FOLLOWS the content -- removing a row MUST shrink the
 * box, otherwise it freezes at its largest ever extent (the v874 lesson). */
function fitEngineSize(node, contentH) {
    const cur = (node.size && node.size[0]) || 0;
    const w = Math.max(ENGINE_MIN_W, cur);
    const h = (typeof contentH === "number" && contentH > 0)
        ? contentH : ((node.size && node.size[1]) || 0);
    node.size[0] = w;
    node.size[1] = h;
    if (node.setSize) node.setSize([w, h]);
    return [w, h];
}

function newEngineRow() {
    return { enabled: true, name: "None", weight: 1.0,
             // Compat shims so shared helpers (openLoraSelect / openGroupPreviewOverlay)
             // can read/write the same fields as the main stack without branching:
             group: "—", wHigh: 1.0, wLow: 1.0 };
}

app.registerExtension({
    name: "Polyhedron.engine",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== ENGINE_NODE_TYPE) return;

        // ── onCreate ───────────────────────────────────────────────────
        const _orig_onCreate = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            _orig_onCreate?.apply(this, arguments);
            this._uls = { rows: [newEngineRow()], mode: "SEQ",
                          hoverRow: -1, hoverZone: "",
                          dragSrc: -1, dragDest: -1,
                          isEngine: true,
                          dareVariant: "channel" };
            this.size[0] = Math.max(this.size[0], ENGINE_START_W);
            this._engineResize();
            setTimeout(() => this._engineHideConfigWidget?.(), 100);
        };

        nodeType.prototype._engineResize = function () {
            if (!this._uls?.rows) return;
            const h = ENGINE_HEADER_H + (this._uls.rows.length + 1) * ROW_H + 8;
            fitEngineSize(this, h);
            this._uls.hoverRow = -1; this._uls.hoverZone = "";
            app.graph?.setDirtyCanvas(true, false);
        };

        nodeType.prototype.onResize = function (size) {
            if (!this._uls?.rows) return;
            const minH = ENGINE_HEADER_H + (this._uls.rows.length + 1) * ROW_H + 8;
            if (size[0] < ENGINE_MIN_W) size[0] = ENGINE_MIN_W;
            if (size[1] < minH) size[1] = minH;
        };

        // ── Serialize / Configure ──────────────────────────────────────
        const _orig_ser = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (o) {
            _orig_ser?.apply(this, arguments);
            if (this._uls?.rows) o._engine = JSON.stringify({
                rows: this._uls.rows.map(r => ({
                    enabled: r.enabled, name: r.name, weight: r.weight,
                    wClip: r.wClip,   // v309: optional per-row CLIP strength (mirrors Stack v302)
                })),
                mode: this._uls.mode || "SEQ",
                dareVariant: this._uls.dareVariant || "channel",
            });
        };

        const _orig_cfg = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (o) {
            _orig_cfg?.apply(this, arguments);
            if (!this._uls || !this._uls.rows) {
                this._uls = { rows: [newEngineRow()], mode: "SEQ",
                              hoverRow: -1, hoverZone: "",
                              dragSrc: -1, dragDest: -1,
                              isEngine: true,
                              dareVariant: "channel" };
            }
            if (!this._uls.dareVariant) this._uls.dareVariant = "channel";
            this._engineHideConfigWidget();
            if (o._engine) {
                try {
                    const d = JSON.parse(o._engine);
                    this._uls.rows = (d.rows || []).map(r => ({ ...newEngineRow(), ...r }));
                    this._uls.mode = (d.mode || "SEQ").toUpperCase();
                    if (d.dareVariant === "channel" || d.dareVariant === "element") {
                        this._uls.dareVariant = d.dareVariant;
                    }
                    this._uls.rows.forEach(r => ensurePreview(r.name));
                    this._engineResize();
                } catch {}
            }
        };

        nodeType.prototype._engineHideConfigWidget = function () {
            if (!this.widgets) return;
            for (const w of this.widgets) {
                if (w.name === "engine_config") {
                    w.hidden = true;
                    w.type = "hidden";
                    w.computeSize = () => [0, -4];
                    // v877: was setSize(computeSize()) -- a hard reset to
                    // LiteGraph's minimum on every workflow load. Grow-only.
                    fitEngineSize(this);
                    break;
                }
            }
        };

        // Shared name with the stack node — openLoraSelect calls node._ulsSync().
        nodeType.prototype._ulsSync = function () {
            if (!this._uls || !this.widgets) return;
            let w = this.widgets.find(x => x.name === "engine_config");
            if (!w) w = this.addWidget("text", "engine_config", "", () => {});
            w.hidden = true; w.type = "hidden";
            w.computeSize = () => [0, -4];
            w.value = JSON.stringify({
                rows: this._uls.rows.map(r => ({
                    enabled: r.enabled, name: r.name, weight: r.weight,
                    wClip: r.wClip,   // v309: optional per-row CLIP strength (mirrors Stack v302)
                })),
                mode: this._uls.mode || "SEQ",
                dare_variant: this._uls.dareVariant || "channel",
            });
            this._engineHideConfigWidget?.();
        };

        // ── Draw ───────────────────────────────────────────────────────
        nodeType.prototype.onDrawForeground = function (ctx) {
            // Compat probe (uls_compat.js): see Stack node for rationale.
            this._ulsDrawFired = true;
            // v303: self-healing — if the compat layer injected the renderer
            // notice (uls_compat.js, name below) but the canvas path IS alive,
            // that was a false positive (offscreen culling / slow first draw:
            // onDrawForeground only runs for nodes inside the viewport).
            // Remove the notice the moment we provably draw.
            if (this.widgets?.some(w => w?.name === "polyhedron_renderer_notice")) {
                this.widgets = this.widgets.filter(
                    w => w?.name !== "polyhedron_renderer_notice");
                this.setDirtyCanvas?.(true, true);
            }
            const uls = this._uls;
            if (!uls || !uls.rows) return;
            const W = this.size[0];
            const rows = uls.rows;
            ctx.save();

            // Wireframe hexagon icon (top-right of title)
            {
                const ix = W - 22, iy = -18, ir = 8;
                ctx.strokeStyle = "#c060ff";
                ctx.lineWidth = 1.2;
                ctx.globalAlpha = 0.8;
                ctx.beginPath();
                for (let k = 0; k < 6; k++) {
                    const a = (k * Math.PI / 3) - Math.PI / 6;
                    const x = ix + ir * Math.cos(a);
                    const y = iy + ir * Math.sin(a);
                    k === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
                }
                ctx.closePath(); ctx.stroke();
                ctx.globalAlpha = 1;
            }

            // Mode-Switch row in the header (S | C | D buttons)
            // Sits below the output-pin zone (~52px from top of foreground area)
            const modeY = 52;
            const MODE_BTN_W = 38, MODE_BTN_H = 22, MODE_GAP = 4;
            const modesArr = [
                { key: "SEQ",    letter: "S", color: "#7a7a8a" },
                { key: "CONCAT", letter: "C", color: "#f0c050" },
                { key: "DARE",   letter: "D", color: "#40c0ff" },
            ];
            let mbx = PAD;
            uls._modeBtnZones = [];
            for (const m of modesArr) {
                const isActive = (uls.mode || "SEQ") === m.key;
                const isHov = uls.hoverZone === ("mode:" + m.key);
                ctx.fillStyle = isActive ? m.color + "44" : (isHov ? m.color + "22" : "#1a1a2a");
                roundRect(ctx, mbx, modeY, MODE_BTN_W, MODE_BTN_H, 4); ctx.fill();
                ctx.strokeStyle = isActive ? m.color : (isHov ? m.color + "88" : "#2a2a3a");
                ctx.lineWidth = isActive ? 1.5 : 0.8;
                roundRect(ctx, mbx, modeY, MODE_BTN_W, MODE_BTN_H, 4); ctx.stroke();
                ctx.fillStyle = isActive ? "#fff" : (isHov ? m.color : "#999");
                ctx.font = "bold 11px 'Segoe UI',Arial";
                ctx.textAlign = "center"; ctx.textBaseline = "middle";
                ctx.fillText(m.letter, mbx + MODE_BTN_W/2, modeY + MODE_BTN_H/2 + 1);
                ctx.textBaseline = "alphabetic"; ctx.textAlign = "left";
                uls._modeBtnZones.push({ key: m.key, x: mbx, y: modeY, w: MODE_BTN_W, h: MODE_BTN_H });
                mbx += MODE_BTN_W + MODE_GAP;
            }

            // DARE-Variant Toggle Pill — anchored right of D-button, only visible when DARE active.
            // Fix v098: single centered text, wider pill, no split-text layout bug.
            uls._dareVariantRect = null;
            if ((uls.mode || "SEQ") === "DARE") {
                const variant  = uls.dareVariant || "channel";
                const isHov    = uls.hoverZone === "dareVariant";
                const pillW    = 54, pillH = MODE_BTN_H;
                // Gap of 8px after the last mode button (mbx already advanced past D)
                const pillX    = mbx + 4;
                const pillY    = modeY;
                ctx.fillStyle  = isHov
                    ? (variant === "channel" ? "#1a3a2a" : "#3a2a0a")
                    : (variant === "channel" ? "#112218" : "#221808");
                roundRect(ctx, pillX, pillY, pillW, pillH, 4); ctx.fill();
                ctx.strokeStyle = isHov
                    ? (variant === "channel" ? "#7af0c0" : "#f0c87a")
                    : (variant === "channel" ? "#3a7a5a" : "#7a5a1a");
                ctx.lineWidth = 1;
                roundRect(ctx, pillX, pillY, pillW, pillH, 4); ctx.stroke();
                const dvText = variant === "channel" ? "CHAN" : "ELEM";
                ctx.fillStyle = variant === "channel" ? "#7af0c0" : "#f0c87a";
                ctx.font = "bold 10px 'Segoe UI',Arial";
                ctx.textAlign = "center"; ctx.textBaseline = "middle";
                ctx.fillText(dvText, pillX + pillW / 2, pillY + pillH / 2 + 0.5);
                ctx.textBaseline = "alphabetic"; ctx.textAlign = "left";
                uls._dareVariantRect = { x: pillX, y: pillY, w: pillW, h: pillH };
            }

            // Active mode label (small line below buttons)
            const modeLabels = { "SEQ": "Sequential (SEQ)", "CONCAT": "Combined (CONCAT)", "DARE": "Smooth Mix (DARE)" };
            const activeModeKey = uls.mode || "SEQ";
            const modeColors   = { "SEQ": "#7a7a8a", "CONCAT": "#f0c050", "DARE": "#40c0ff" };
            ctx.fillStyle = modeColors[activeModeKey] || "#888";
            ctx.font = "10px 'Segoe UI',Arial"; ctx.textAlign = "left";
            ctx.fillText("▸ " + (modeLabels[activeModeKey] || activeModeKey), PAD, modeY + MODE_BTN_H + 14);

            {
                // v310: "Weight / CLIP Strength" header — exact mirror of the
                // Stack v305/v308 composite. Shares the mode-label baseline
                // (one header line: label left, composite right) and is
                // RIGHT-anchored at the node content edge (above the ✕
                // column). Measured via measureText, never eyeballed
                // (v304/v307 lesson). The WHOLE composite is the hover area
                // for the explainer tooltip (uls._weightHdrRect below).
                const _hy = modeY + MODE_BTN_H + 14;
                ctx.textAlign = "left";
                ctx.font = "9px 'Segoe UI',Arial";
                const _wW = ctx.measureText("Weight").width;
                ctx.font = "8px 'Segoe UI',Arial";
                const _wC = ctx.measureText(" / CLIP Strength").width;
                const _ICO_R = 3.2, _ICO_GAP = 4;
                const _total = _wW + _wC + _ICO_GAP + 2 * _ICO_R;
                const _right = W - PAD;            // = contentR of the rows
                let _hx = _right - _total;
                ctx.font = "9px 'Segoe UI',Arial";
                ctx.fillStyle = "#b07820";
                ctx.fillText("Weight", _hx, _hy);
                _hx += _wW;
                ctx.font = "8px 'Segoe UI',Arial";
                ctx.fillStyle = "#6aa0d0";
                ctx.fillText(" / CLIP Strength", _hx, _hy);
                // 🛈 — small stroked circle with an "i", same CLIP blue
                const _ix = _hx + _wC + _ICO_GAP + _ICO_R;
                ctx.beginPath();
                ctx.arc(_ix, _hy - 3, _ICO_R, 0, Math.PI * 2);
                ctx.strokeStyle = "#6aa0d0"; ctx.lineWidth = 0.9;
                ctx.stroke();
                ctx.font = "bold 5.5px 'Segoe UI',Arial";
                ctx.textAlign = "center";
                ctx.fillText("i", _ix, _hy - 1.2);
                ctx.font = "10px 'Segoe UI',Arial";
                ctx.textAlign = "left";
                // Hover area = full composite; h=12 keeps it clear of row 0
                // (rows start at ENGINE_HEADER_H, 2px below the baseline).
                uls._weightHdrRect = { x: _right - _total - 2, y: _hy - 10,
                                       w: _total + 4, h: 12 };
            }

            // Row layout (same as main stack but without GRP-pill and trigger-insert)
            const THUMB_W  = 30;
            const thumbX   = PAD + 48;
            const btnGap   = 4;
            const WEIGHT_W = 72, ARROW_W = 14;
            const DEL_W    = 18;
            const nameX    = thumbX + THUMB_W + 4;
            const contentR = W - PAD;
            const delX     = contentR - DEL_W;
            const weightX  = delX - btnGap - WEIGHT_W;
            const nameMaxW = weightX - btnGap - nameX;
            const wArrowLX = weightX;
            const wArrowRX = weightX + WEIGHT_W - ARROW_W;
            const wValX    = weightX + ARROW_W;

            for (let ri = 0; ri < rows.length; ri++) {
                const row = rows[ri];
                const y = ENGINE_HEADER_H + ri * ROW_H;
                const isHov = uls.hoverRow === ri;

                // Hover tint
                if (isHov) {
                    ctx.fillStyle = "#1a1a2a";
                    roundRect(ctx, PAD, y + 1, W - 2*PAD, ROW_H - 2, 4);
                    ctx.fill();
                }

                // Drag-handle ▲▼ (left)
                const upHov   = isHov && uls.hoverZone === "up";
                const dnHov   = isHov && uls.hoverZone === "down";
                ctx.fillStyle = upHov ? "#aac" : "#556";
                ctx.font = "bold 9px 'Segoe UI',Arial"; ctx.textAlign = "center";
                ctx.fillText("▲", PAD + 7, y + ROW_H/2 - 3);
                ctx.fillStyle = dnHov ? "#aac" : "#556";
                ctx.fillText("▼", PAD + 7, y + ROW_H/2 + 8);

                // Checkbox (same style as stack)
                const cbx = PAD + 30, cby = y + 8;
                ctx.strokeStyle = row.enabled ? "#5555bb" : "#383848"; ctx.lineWidth = 1.5;
                ctx.strokeRect(cbx, cby, 12, 12);
                if (row.enabled) {
                    ctx.strokeStyle = "#8080ff"; ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.moveTo(cbx+2, cby+6); ctx.lineTo(cbx+5, cby+9); ctx.lineTo(cbx+10, cby+3);
                    ctx.stroke();
                }

                // Thumbnail (same as stack)
                const pv = previewCache.get(row.name);
                const thumbHov = isHov && uls.hoverZone === "thumb";
                ctx.fillStyle = "#111118";
                roundRect(ctx, thumbX, y + 3, THUMB_W, ROW_H - 6, 3); ctx.fill();
                ctx.strokeStyle = thumbHov ? "#aa88ff" : "#2a2a3a"; ctx.lineWidth = 0.5;
                roundRect(ctx, thumbX, y + 3, THUMB_W, ROW_H - 6, 3); ctx.stroke();
                if (pv?.img) {
                    try {
                        ctx.save();
                        roundRect(ctx, thumbX, y + 3, THUMB_W, ROW_H - 6, 3); ctx.clip();
                        const ir = pv.img.naturalWidth / pv.img.naturalHeight;
                        const th = ROW_H - 6;
                        const tw2 = th * ir;
                        const tx = thumbX + (THUMB_W - tw2) / 2;
                        ctx.drawImage(pv.img, tx, y + 3, tw2, th);
                        ctx.restore();
                    } catch(e) {}
                } else if (pv?.vid) {
                    try {
                        ctx.save();
                        roundRect(ctx, thumbX, y + 3, THUMB_W, ROW_H - 6, 3); ctx.clip();
                        ctx.drawImage(pv.vid, thumbX, y + 3, THUMB_W, ROW_H - 6);
                        ctx.restore();
                        ctx.fillStyle = "rgba(255,255,255,0.7)";
                        ctx.font = "8px Arial"; ctx.textAlign = "center";
                        ctx.fillText("▶", thumbX + THUMB_W/2, y + ROW_H/2 + 3);
                        ctx.textAlign = "left";
                    } catch(e) {}
                } else if (row.name !== "None") {
                    const initial = row.name.split(/[/\\]/).pop().charAt(0).toUpperCase();
                    ctx.fillStyle = "#404050";
                    ctx.font = "bold 12px Arial"; ctx.textAlign = "center";
                    ctx.fillText(initial, thumbX + THUMB_W/2, y + ROW_H/2 + 4);
                    ctx.textAlign = "left";
                }

                // Name button (wide, no GRP-pill)
                const nameHover = isHov && uls.hoverZone === "name";
                ctx.fillStyle = nameHover ? "#2a2244" : "#1a1a2a";
                roundRect(ctx, nameX - 2, y + 4, nameMaxW + 4, ROW_H - 8, 4); ctx.fill();
                ctx.strokeStyle = nameHover ? "#6655aa" : "#2e2e44"; ctx.lineWidth = 0.5;
                roundRect(ctx, nameX - 2, y + 4, nameMaxW + 4, ROW_H - 8, 4); ctx.stroke();
                ctx.save();
                ctx.beginPath(); ctx.rect(nameX, y + 4, nameMaxW, ROW_H - 8); ctx.clip();
                ctx.fillStyle = row.name === "None" ? "#555566" : (nameHover ? "#e8e0ff" : "#d0d0e0");
                ctx.font = "12px 'Segoe UI',Arial"; ctx.textAlign = "left";
                const dispName = row.name === "None"
                    ? "  Select LoRA…"
                    : "  " + row.name.split(/[/\\]/).pop().replace(/\.safetensors$/i, "");
                ctx.fillText(dispName, nameX, y + 17);
                ctx.restore();

                // Weight box
                const wDecHover = isHov && uls.hoverZone === "wDec";
                const wIncHover = isHov && uls.hoverZone === "wInc";
                ctx.fillStyle = "#221a10";
                roundRect(ctx, weightX, y + 5, WEIGHT_W, 18, 4); ctx.fill();
                ctx.strokeStyle = "#f0a03044"; ctx.lineWidth = 0.5;
                roundRect(ctx, weightX, y + 5, WEIGHT_W, 18, 4); ctx.stroke();
                ctx.fillStyle = wDecHover ? "#f0a03044" : "transparent";
                roundRect(ctx, wArrowLX, y + 5, ARROW_W, 18, 4); ctx.fill();
                ctx.fillStyle = wDecHover ? "#f0a030" : "#7a5018";
                ctx.font = "bold 9px Arial";
                ctx.textAlign = "center"; ctx.textBaseline = "middle";
                ctx.fillText("◀", wArrowLX + ARROW_W/2, y + 14);
                ctx.fillStyle = wIncHover ? "#f0a03044" : "transparent";
                roundRect(ctx, wArrowRX, y + 5, ARROW_W, 18, 4); ctx.fill();
                ctx.fillStyle = wIncHover ? "#f0a030" : "#7a5018";
                ctx.fillText("▶", wArrowRX + ARROW_W/2, y + 14);
                ctx.fillStyle = "#f0a030";
                if (typeof row.wClip === "number" && row.wClip !== row.weight) {
                    // v302: decoupled CLIP strength — two-line cell
                    ctx.font = "bold 9px monospace";
                    ctx.fillText((row.weight || 0).toFixed(2),
                                 wValX + (WEIGHT_W - 2*ARROW_W) / 2, y + 10);
                    ctx.font = "7px monospace";
                    ctx.fillStyle = "#6aa0d0";
                    ctx.fillText("c " + row.wClip.toFixed(2),
                                 wValX + (WEIGHT_W - 2*ARROW_W) / 2, y + 19);
                } else {
                    ctx.font = "bold 11px monospace";
                    ctx.fillText((row.weight || 0).toFixed(2),
                                 wValX + (WEIGHT_W - 2*ARROW_W) / 2, y + 14);
                }
                ctx.textBaseline = "alphabetic";

                // Delete ✕
                const delHov = isHov && uls.hoverZone === "del";
                ctx.fillStyle = delHov ? "#aa3344" : "#553344";
                ctx.font = "bold 12px Arial"; ctx.textAlign = "center";
                ctx.fillText("✕", delX + DEL_W/2, y + 18);
                ctx.textAlign = "left";
            }

            // "＋" Add-Button Row (same style as stack)
            const addRowY = ENGINE_HEADER_H + rows.length * ROW_H;
            const addHov = uls.hoverZone === "addRow";
            ctx.fillStyle = addHov ? "#1a2a1a" : "#161620";
            ctx.fillRect(0, addRowY, W, ROW_H);
            ctx.strokeStyle = "#252535"; ctx.lineWidth = 0.5;
            ctx.beginPath(); ctx.moveTo(0, addRowY); ctx.lineTo(W, addRowY); ctx.stroke();
            ctx.fillStyle = addHov ? "#6acc6a" : "#3a5a3a";
            ctx.font = "bold 14px Arial"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
            ctx.fillText("＋", W/2, addRowY + ROW_H/2);
            ctx.textBaseline = "alphabetic"; ctx.textAlign = "left";

            // ── Weight/CLIP header tooltip — drawn LAST (v310, Stack v304 mirror) ──
            if (uls.hoverZone === "weightHdr" && uls._weightHdrRect) {
                const hr = uls._weightHdrRect;
                const lines = [
                    "Weight / CLIP Strength",
                    "Click: model weight.  Shift+Click: set a per-LoRA",
                    "CLIP strength (decoupled).  Shift+\u25C0 \u25B6 steps it.",
                    "Enter the model weight again to re-link.",
                ];
                const LH = 13, PAD_T = 6, TW = 250;
                const TH = PAD_T * 2 + LH * lines.length;
                // Header sits near the right edge → anchor the tooltip LEFT
                const TX = Math.max(PAD, hr.x - TW - 8);
                const TY = Math.max(2, hr.y - 2);
                ctx.fillStyle = "#0e0e18";
                roundRect(ctx, TX, TY, TW, TH, 5); ctx.fill();
                ctx.strokeStyle = "#6aa0d0aa"; ctx.lineWidth = 1;
                roundRect(ctx, TX, TY, TW, TH, 5); ctx.stroke();
                lines.forEach((line, i) => {
                    ctx.font = i === 0
                        ? "bold 10px 'Segoe UI',Arial"
                        : "9.5px 'Segoe UI',Arial";
                    ctx.fillStyle = i === 0 ? "#6aa0d0" : "#c8c8d8";
                    ctx.textAlign = "left";
                    ctx.fillText(line, TX + 8, TY + PAD_T + 10 + i * LH);
                });
            }

            // Mode-button hover tooltip
            const hovMode = (uls.hoverZone || "").startsWith("mode:")
                ? (uls.hoverZone.split(":")[1]) : null;
            if (hovMode && uls._modeBtnZones) {
                const TOOLTIP_HINTS = {
                    "SEQ":    { label: "Sequential (SEQ)",   hint: "stacks LoRAs one by one — classic, full effect of each" },
                    "CONCAT": { label: "Combined (CONCAT)",  hint: "bundles all LoRAs into one — less stacking" },
                    "DARE":   { label: "Smooth Mix (DARE)",  hint: "bundles and spreads — best when many LoRAs target the same area" },
                };
                const TOOLTIP_COLORS = { "SEQ": "#7a7a8a", "CONCAT": "#f0c050", "DARE": "#40c0ff" };
                const zone = uls._modeBtnZones.find(z => z.key === hovMode);
                const info = TOOLTIP_HINTS[hovMode];
                if (zone && info) {
                    const TW = 300;
                    const TX = Math.min(zone.x, W - TW - PAD);
                    const TY = zone.y + zone.h + 4;
                    drawCanvasTooltip(ctx, TX, TY, TW, info.label, info.hint, TOOLTIP_COLORS[hovMode]);
                }
            }

            // DARE-Variant pill hover tooltip
            if (uls.hoverZone === "dareVariant" && uls._dareVariantRect) {
                const dvR = uls._dareVariantRect;
                const variant = uls.dareVariant || "channel";
                const DARE_TOOLTIPS = {
                    "channel": { label: "Channel Drop (CHAN)", color: "#7af0c0",
                        hint: "drops whole rank-channels — LoRA-aware, reduces overlap between concurrent LoRAs" },
                    "element":  { label: "Element Drop (ELEM)", color: "#f0c87a",
                        hint: "drops individual tensor elements — classic DARE paper, finer-grained sparsity" },
                };
                const info = DARE_TOOLTIPS[variant];
                if (info) {
                    const TW = 320;
                    const TX = Math.min(dvR.x, W - TW - PAD);
                    const TY = dvR.y + dvR.h + 4;
                    drawCanvasTooltip(ctx, TX, TY, TW, info.label, info.hint, info.color);
                }
            }

            ctx.restore();
        };

        // ── Hover tracking ────────────────────────────────────────────
        nodeType.prototype.onMouseMove = function (e, [lx, ly]) {
            const uls = this._uls; if (!uls?.rows) return false;
            const W = this.size[0];
            let dirty = false;

            // v310: Weight/CLIP header hover → explains the Shift interaction
            // (checked first, mirrors the Stack onMouseMove order)
            const whR = uls._weightHdrRect;
            if (whR && lx >= whR.x && lx <= whR.x + whR.w
                    && ly >= whR.y && ly <= whR.y + whR.h) {
                if (uls.hoverZone !== "weightHdr") {
                    uls.hoverZone = "weightHdr";
                    uls.hoverRow  = -1;
                    app.graph?.setDirtyCanvas(true, false);
                }
                return false;
            }

            // DARE-Variant Pill (header, right side)
            const dvR = uls._dareVariantRect;
            if (dvR && lx >= dvR.x && lx <= dvR.x + dvR.w
                    && ly >= dvR.y && ly <= dvR.y + dvR.h) {
                if (uls.hoverZone !== "dareVariant") {
                    uls.hoverZone = "dareVariant";
                    uls.hoverRow  = -1;
                    app.graph?.setDirtyCanvas(true, false);
                }
                return false;
            }

            // Mode buttons (header)
            const oldZone = uls.hoverZone;
            const oldRow  = uls.hoverRow;
            uls.hoverRow  = -1;
            uls.hoverZone = "";
            if (uls._modeBtnZones) {
                for (const z of uls._modeBtnZones) {
                    if (lx >= z.x && lx <= z.x + z.w && ly >= z.y && ly <= z.y + z.h) {
                        uls.hoverZone = "mode:" + z.key;
                        if (oldZone !== uls.hoverZone) dirty = true;
                        break;
                    }
                }
            }
            if (uls.hoverZone) {
                if (oldRow !== -1) dirty = true;
                if (dirty) app.graph?.setDirtyCanvas(true, false);
                return false;
            }

            // Add-row hover
            const addRowY = ENGINE_HEADER_H + uls.rows.length * ROW_H;
            if (ly >= addRowY && ly < addRowY + ROW_H) {
                uls.hoverZone = "addRow";
                if (uls.hoverZone !== oldZone) dirty = true;
                if (dirty) app.graph?.setDirtyCanvas(true, false);
                return false;
            }

            // Row zones
            const ri = Math.floor((ly - ENGINE_HEADER_H) / ROW_H);
            if (ri < 0 || ri >= uls.rows.length) {
                if (oldRow !== -1 || oldZone) { app.graph?.setDirtyCanvas(true, false); }
                return false;
            }
            uls.hoverRow = ri;

            const THUMB_W  = 30;
            const thumbX   = PAD + 48;
            const btnGap   = 4;
            const WEIGHT_W = 72, ARROW_W = 14;
            const DEL_W    = 18;
            const nameX    = thumbX + THUMB_W + 4;
            const delX     = (W - PAD) - DEL_W;
            const weightX  = delX - btnGap - WEIGHT_W;
            const nameMaxW = weightX - btnGap - nameX;
            const wArrowLX = weightX;
            const wArrowRX = weightX + WEIGHT_W - ARROW_W;

            let zone = "";
            if      (lx >= PAD && lx <= PAD + 14)                              zone = (ly < ENGINE_HEADER_H + ri*ROW_H + ROW_H/2) ? "up" : "down";
            else if (lx >= PAD + 30 && lx <= PAD + 42)                          zone = "checkbox";
            else if (lx >= thumbX && lx <= thumbX + THUMB_W)                    zone = "thumb";
            else if (lx >= nameX - 2 && lx <= nameX + nameMaxW + 2)             zone = "name";
            else if (lx >= wArrowLX && lx <= wArrowLX + ARROW_W)                zone = "wDec";
            else if (lx >= wArrowRX && lx <= wArrowRX + ARROW_W)                zone = "wInc";
            else if (lx >= weightX && lx <= weightX + WEIGHT_W)                 zone = "weight";
            else if (lx >= delX && lx <= delX + DEL_W)                          zone = "del";

            if (zone !== oldZone || ri !== oldRow) {
                uls.hoverZone = zone;
                app.graph?.setDirtyCanvas(true, false);
            }

            // Preview-Popup beim Thumbnail-Hover (same as stack)
            const row = uls.rows[ri];
            if (zone === "thumb" && row?.name !== "None") {
                ensurePreview(row.name);
                const canvas = app.canvas.canvas;
                const rect   = canvas.getBoundingClientRect();
                const scale  = app.canvas?.ds?.scale ?? 1;
                const off    = app.canvas?.ds?.offset ?? {0:0,1:0};
                const sx = rect.left + (this.pos[0] + lx) * scale + off[0] * scale;
                const sy = rect.top  + (this.pos[1] + ly) * scale + off[1] * scale;
                showPopup(row.name, sx, sy);
            } else if (zone !== "thumb" && popupName) {
                closePopup();
            }

            return false;
        };

        nodeType.prototype.onMouseLeave = function () {
            const uls = this._uls; if (!uls) return;
            uls.hoverRow = -1; uls.hoverZone = "";
            closePopup();
            app.graph?.setDirtyCanvas(true, false);
        };

        // ── Click ─────────────────────────────────────────────────────
        nodeType.prototype.onMouseDown = function (e, [lx, ly]) {
            const uls = this._uls; if (!uls) return false;
            const W = this.size[0];

            // DARE-Variant pill (header)
            const dvR = uls._dareVariantRect;
            if (dvR && lx >= dvR.x && lx <= dvR.x + dvR.w
                    && ly >= dvR.y && ly <= dvR.y + dvR.h) {
                uls.dareVariant = (uls.dareVariant === "channel") ? "element" : "channel";
                this._ulsSync();
                app.graph?.setDirtyCanvas(true, false);
                return true;
            }

            // Mode-button click (header)
            if (uls._modeBtnZones) {
                for (const z of uls._modeBtnZones) {
                    if (lx >= z.x && lx <= z.x + z.w && ly >= z.y && ly <= z.y + z.h) {
                        uls.mode = z.key;
                        this._ulsSync();
                        app.graph?.setDirtyCanvas(true, false);
                        return true;
                    }
                }
            }

            // Add-row
            const addRowY = ENGINE_HEADER_H + uls.rows.length * ROW_H;
            if (ly >= addRowY && ly < addRowY + ROW_H) {
                uls.rows.push(newEngineRow());
                this._engineResize(); this._ulsSync();
                return true;
            }

            const ri = Math.floor((ly - ENGINE_HEADER_H) / ROW_H);
            if (ri < 0 || ri >= uls.rows.length) return false;
            const row = uls.rows[ri];

            // Drag handle (left edge)
            if (lx < PAD + 4) {
                uls.dragSrc = ri; uls.dragDest = ri;
                return true;
            }

            // ▲▼ reorder
            if (lx >= PAD && lx <= PAD + 14) {
                const y = ENGINE_HEADER_H + ri * ROW_H;
                if (ly >= y + 1 && ly <= y + ROW_H/2) {
                    if (ri > 0) {
                        const [r] = uls.rows.splice(ri, 1);
                        uls.rows.splice(ri - 1, 0, r);
                        this._ulsSync(); this._engineResize();
                    }
                    return true;
                }
                if (ly >= y + ROW_H/2 && ly <= y + ROW_H - 1) {
                    if (ri < uls.rows.length - 1) {
                        const [r] = uls.rows.splice(ri, 1);
                        uls.rows.splice(ri + 1, 0, r);
                        this._ulsSync(); this._engineResize();
                    }
                    return true;
                }
            }

            // Checkbox
            if (lx >= PAD + 30 && lx <= PAD + 42) {
                row.enabled = !row.enabled;
                app.graph?.setDirtyCanvas(true, false);
                this._ulsSync(); return true;
            }

            // Zone coordinates
            const THUMB_W  = 30;
            const thumbX   = PAD + 48;
            const btnGap   = 4;
            const WEIGHT_W = 72, ARROW_W = 14;
            const DEL_W    = 18;
            const nameX    = thumbX + THUMB_W + 4;
            const delX     = (W - PAD) - DEL_W;
            const weightX  = delX - btnGap - WEIGHT_W;
            const nameMaxW = weightX - btnGap - nameX;
            const wArrowLX = weightX;
            const wArrowRX = weightX + WEIGHT_W - ARROW_W;

            // Thumbnail click → reuses the stack's preview overlay (read-only feel)
            if (lx >= thumbX && lx <= thumbX + THUMB_W) {
                openGroupPreviewOverlay(row, e, null);
                return true;
            }

            // Name click → LoRA select dropdown (shared helper)
            if (lx >= nameX - 2 && lx <= nameX + nameMaxW + 2) {
                openLoraSelect(row, _loraList, e, this);
                return true;
            }

            // Weight ◀ (v302: Shift = CLIP strength)
            if (lx >= wArrowLX && lx <= wArrowLX + ARROW_W) {
                if (e.shiftKey) {
                    const base = (typeof row.wClip === "number") ? row.wClip : (row.weight || 0);
                    row.wClip = Math.round(Math.max(-10, base - 0.05) * 100) / 100;
                } else {
                    row.weight = Math.round(Math.max(-10, (row.weight || 0) - 0.05) * 100) / 100;
                }
                app.graph?.setDirtyCanvas(true, false); this._ulsSync();
                return true;
            }
            // Weight ▶ (v302: Shift = CLIP strength)
            if (lx >= wArrowRX && lx <= wArrowRX + ARROW_W) {
                if (e.shiftKey) {
                    const base = (typeof row.wClip === "number") ? row.wClip : (row.weight || 0);
                    row.wClip = Math.round(Math.min(10, base + 0.05) * 100) / 100;
                } else {
                    row.weight = Math.round(Math.min(10, (row.weight || 0) + 0.05) * 100) / 100;
                }
                app.graph?.setDirtyCanvas(true, false); this._ulsSync();
                return true;
            }
            // Weight click → numeric input (v302: Shift-Klick = CLIP)
            if (lx >= weightX + ARROW_W && lx <= weightX + WEIGHT_W - ARROW_W) {
                if (e.shiftKey) {
                    const cur = (typeof row.wClip === "number") ? row.wClip : (row.weight || 0);
                    showWeightInput(e, cur, (v) => {
                        if (v === (row.weight || 0)) delete row.wClip; else row.wClip = v;
                        app.graph?.setDirtyCanvas(true, false); this._ulsSync();
                    }, "CLIP Strength", "#6aa0d0");
                } else {
                    showWeightInput(e, row.weight || 0, (v) => {
                        row.weight = v;
                        app.graph?.setDirtyCanvas(true, false); this._ulsSync();
                    }, "Weight");
                }
                return true;
            }

            // Delete ✕
            if (lx >= delX && lx <= delX + DEL_W) {
                uls.rows.splice(ri, 1);
                if (uls.rows.length === 0) uls.rows.push(newEngineRow());
                this._engineResize(); this._ulsSync();
                return true;
            }

            return false;
        };

        nodeType.prototype.onMouseUp = function (e, [lx, ly]) {
            const uls = this._uls; if (!uls) return false;
            if (uls.dragSrc < 0) return false;
            const src = uls.dragSrc, dst = uls.dragDest;
            uls.dragSrc = -1; uls.dragDest = -1;
            if (src !== dst && dst >= 0 && dst < uls.rows.length) {
                const [r] = uls.rows.splice(src, 1);
                const insertAt = dst > src ? dst - 1 : dst;
                uls.rows.splice(insertAt, 0, r);
                this._ulsSync();
            }
            app.graph?.setDirtyCanvas(true, false);
            return true;
        };

        // Right-click context menu — only the row-management entries
        const _origMenu = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
            _origMenu?.apply(this, arguments);
            const uls = this._uls; if (!uls) return;
            const i = uls.hoverRow;
            if (i < 0 || i >= uls.rows.length) return;
            options.push(null,
                { content: "⬆ Move up", callback: () => {
                    if (i > 0) {
                        [uls.rows[i-1], uls.rows[i]] = [uls.rows[i], uls.rows[i-1]];
                        this._ulsSync(); app.graph?.setDirtyCanvas(true,false);
                    }
                }},
                { content: "⬇ Move down", callback: () => {
                    if (i < uls.rows.length-1) {
                        [uls.rows[i], uls.rows[i+1]] = [uls.rows[i+1], uls.rows[i]];
                        this._ulsSync(); app.graph?.setDirtyCanvas(true,false);
                    }
                }},
                { content: "🗑 Delete row", callback: () => {
                    uls.rows.splice(i, 1);
                    if (uls.rows.length === 0) uls.rows.push(newEngineRow());
                    this._engineResize(); this._ulsSync();
                }},
            );
        };
    }
});
