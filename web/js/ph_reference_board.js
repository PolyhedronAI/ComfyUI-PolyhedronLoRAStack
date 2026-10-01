/*
 * ph_reference_board.js -- v1019 (Cine C2): the Reference Board, drawn.
 *
 * One field inside the node (no floating window -- Frank's rule), fixed height
 * with its own scroll. A toolbar (add files, counts against MiniMax's 9 / 3 / 3,
 * drop zone) and one CARD per reference:
 *
 *   [thumb]  @tag   role   retention   (MP | sound)      up down x
 *            description ....................................
 *
 * Images show as stills, videos play while the pointer is on them, audio has a
 * play button. Every edit writes the whole board back into the hidden `board`
 * widget as JSON -- the backend (nodes/ph_reference_board.py) reads nothing
 * else. Files are uploaded into input/pls_board through Core's /upload/image
 * (which takes any file, the way Core's own video/audio loaders use it).
 *
 * ROLES and RETENTION MUST match nodes/h3_prompt.py (ROLES / RETENTION);
 * tests/test_v1019_reference_board_js.py guards the parity and drives the pure
 * helpers below.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { setHidden, refit } from "./ph_widget_vis.js";

console.info("[PLS] ph_reference_board.js v1019 loaded");

const NODE = "ULSReferenceBoard";
const FIELD = "pls_board_view";
const HEIGHT = 460;
const SUB = "pls_board";

export const LIMIT = { image: 9, video: 3, audio: 3 };
export const ROLES = {
    image: ["subject", "scene", "style", "first frame", "last frame", "keyframe", "storyboard"],
    video: ["continuation", "motion", "structure", "edit source"],
    audio: ["voice", "music", "ambience", "sound effect"],
};
export const RETENTION = {
    visual: ["fully_preserved", "partially_preserved", "attribute_transfer", "weak_reference"],
    audio: ["fully_copy", "partially_copy", "reference", "weak_reference"],
};
const EXT = {
    image: [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".avif"],
    video: [".mp4", ".webm", ".mov", ".mkv", ".m4v"],
    audio: [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"],
};
const SLOT = /^(image|video|video_audio|audio)_\d+$|^(first|last)_frame$/;

/* ---------------------------------------------------------------- pure */

export function kindOf(name) {
    const n = String(name || "").toLowerCase();
    for (const k of Object.keys(EXT)) if (EXT[k].some((e) => n.endsWith(e))) return k;
    return null;
}

/* A tag from a file name: letters/digits/_ , not a slot name, unique. */
export function sanitizeTag(name, taken) {
    let t = String(name || "").replace(/\.[^.]*$/, "").toLowerCase()
        .replace(/[^a-z0-9_]+/g, "_").replace(/^_+|_+$/g, "");
    if (!t || /^[0-9]/.test(t)) t = "ref_" + t;
    if (SLOT.test(t)) t = "ref_" + t;
    t = t.slice(0, 24);
    let out = t, i = 2;
    const used = new Set(taken || []);
    while (used.has(out) || used.has(out.replace(/_sound$/, ""))) out = t + "_" + (i++);
    return out;
}

export function defaultEntry(kind, file, taken) {
    return {
        kind, file, tag: sanitizeTag(file && file.filename, taken),
        role: ROLES[kind][0],
        retention: kind === "audio" ? "reference" : "fully_preserved",
        desc: "", mp: 0, sound: "none",
    };
}

export function counts(entries) {
    const c = { image: 0, video: 0, audio: 0 };
    for (const e of entries || []) if (c[e.kind] !== undefined) c[e.kind]++;
    return c;
}

export function canAdd(entries, kind) {
    return counts(entries)[kind] < LIMIT[kind];
}

export function moveEntry(entries, i, d) {
    const j = i + d;
    if (i < 0 || j < 0 || i >= entries.length || j >= entries.length) return entries;
    const out = entries.slice();
    [out[i], out[j]] = [out[j], out[i]];
    return out;
}

export function parseBoard(text) {
    try { const v = JSON.parse(text || "[]"); return Array.isArray(v) ? v : []; }
    catch (e) { return []; }
}

export function takenTags(entries) {
    const t = [];
    for (const e of entries || []) { t.push(e.tag); if (e.kind === "video" && e.sound === "own") t.push(e.tag + "_sound"); }
    return t;
}

function viewURL(ref) {
    if (!ref || !ref.filename) return "";
    const p = "/view?filename=" + encodeURIComponent(ref.filename)
        + "&subfolder=" + encodeURIComponent(ref.subfolder || "") + "&type=" + encodeURIComponent(ref.type || "input");
    try { return api.apiURL(p); } catch (e) { return p; }
}

/* ---------------------------------------------------------------- DOM */

function _w(node, name) { return (node.widgets || []).find((w) => w && w.name === name); }
function _el(tag, css, text) {
    const e = document.createElement(tag);
    if (css) e.style.cssText = css;
    if (text !== undefined) e.textContent = text;
    return e;
}
const INPUT = "background:#1e1e1e;color:#ddd;border:1px solid #444;border-radius:3px;font-size:11px;padding:1px 3px;";

function _entries(node) { const w = _w(node, "board"); return parseBoard(w ? w.value : "[]"); }
function _save(node, entries) {
    const w = _w(node, "board");
    if (w) { w.value = JSON.stringify(entries); w.callback?.(w.value); }
    node.setDirtyCanvas?.(true, true);
}

async function _upload(file) {
    const body = new FormData();
    body.append("image", file);
    body.append("subfolder", SUB);
    body.append("type", "input");
    const r = await api.fetchApi("/upload/image", { method: "POST", body });
    if (r.status !== 200) throw new Error("upload " + r.status);
    const d = await r.json();
    return { filename: d.name, subfolder: d.subfolder || SUB, type: d.type || "input" };
}

async function _addFiles(node, files) {
    let entries = _entries(node);
    const st = node._plsBoard;
    for (const f of files) {
        const kind = kindOf(f.name);
        if (!kind) { st && (st.msg.textContent = f.name + ": not an image, video or audio file"); continue; }
        if (!canAdd(entries, kind)) { st && (st.msg.textContent = "MiniMax H3 takes at most " + LIMIT[kind] + " " + kind + "s"); continue; }
        try {
            const ref = await _upload(f);
            entries = entries.concat([defaultEntry(kind, ref, takenTags(entries))]);
            _save(node, entries);
            _render(node);
        } catch (e) { st && (st.msg.textContent = f.name + ": " + e.message); }
    }
}

function _select(options, value, onChange) {
    const s = _el("select", INPUT);
    for (const o of options) { const op = document.createElement("option"); op.value = o; op.textContent = o; s.appendChild(op); }
    s.value = value;
    s.addEventListener("change", () => onChange(s.value));
    return s;
}

function _card(node, entries, i) {
    const e = entries[i];
    const card = _el("div", "display:flex;gap:6px;padding:4px;border:1px solid #333;border-radius:4px;background:#191919;");
    const thumb = _el("div", "flex:0 0 96px;width:96px;height:54px;background:#111;position:relative;overflow:hidden;border-radius:2px;");
    const src = viewURL(e.file);
    if (e.kind === "image") {
        const im = _el("img", "width:96px;height:54px;object-fit:cover;display:block;"); im.src = src; thumb.appendChild(im);
    } else if (e.kind === "video") {
        const v = document.createElement("video");
        v.src = src; v.muted = true; v.loop = true; v.playsInline = true; v.preload = "metadata";
        v.style.cssText = "width:96px;height:54px;object-fit:cover;display:block;";
        thumb.addEventListener("mouseenter", () => v.play().catch(() => {}));
        thumb.addEventListener("mouseleave", () => { v.pause(); v.currentTime = 0; });
        thumb.appendChild(v);
    } else {
        const a = document.createElement("audio"); a.src = src; a.preload = "none";
        const b = _el("button", "width:96px;height:54px;background:#222;color:#9ccc65;border:0;font-size:18px;cursor:pointer;", "\u25B6");
        b.addEventListener("click", () => { if (a.paused) { a.play(); b.textContent = "\u275A\u275A"; } else { a.pause(); b.textContent = "\u25B6"; } });
        a.addEventListener("ended", () => { b.textContent = "\u25B6"; });
        thumb.append(b, a);
    }
    thumb.appendChild(_el("span", "position:absolute;left:2px;bottom:1px;font-size:9px;color:#fff;background:#0008;padding:0 2px;", e.kind));
    const col = _el("div", "flex:1 1 auto;display:flex;flex-direction:column;gap:3px;min-width:0;");
    const row = _el("div", "display:flex;gap:4px;align-items:center;flex-wrap:wrap;");
    const at = _el("span", "color:#f0c87a;font-size:12px;", "@");
    const tag = _el("input", INPUT + "width:90px;"); tag.value = e.tag;
    tag.addEventListener("change", () => { const all = _entries(node); all[i].tag = tag.value.trim(); _save(node, all); });
    row.append(at, tag,
        _select(ROLES[e.kind], e.role, (v) => { const all = _entries(node); all[i].role = v; _save(node, all); }),
        _select(RETENTION[e.kind === "audio" ? "audio" : "visual"], e.retention, (v) => { const all = _entries(node); all[i].retention = v; _save(node, all); }));
    if (e.kind === "image") {
        const mp = _el("input", INPUT + "width:44px;"); mp.type = "number"; mp.min = "0"; mp.max = "16"; mp.step = "0.1";
        mp.value = e.mp || 0; mp.title = "megapixels (0 = the Reference node's rule)";
        mp.addEventListener("change", () => { const all = _entries(node); all[i].mp = Math.max(0, parseFloat(mp.value) || 0); _save(node, all); });
        row.append(_el("span", "font-size:10px;color:#888;", "MP"), mp);
    }
    if (e.kind === "video") {
        row.append(_select(["none", "own"], e.sound || "none", (v) => { const all = _entries(node); all[i].sound = v; _save(node, all); _render(node); }));
        row.lastChild.title = "own = the video's own soundtrack goes along as @" + e.tag + "_sound";
    }
    const ctl = _el("span", "margin-left:auto;display:flex;gap:2px;");
    for (const [label, fn] of [["\u25B2", () => _save(node, moveEntry(_entries(node), i, -1))],
                               ["\u25BC", () => _save(node, moveEntry(_entries(node), i, +1))],
                               ["\u2715", () => { const all = _entries(node); all.splice(i, 1); _save(node, all); }]]) {
        const b = _el("button", "background:#2a2a2a;color:#ccc;border:1px solid #444;border-radius:3px;font-size:10px;cursor:pointer;padding:0 4px;", label);
        b.addEventListener("click", () => { fn(); _render(node); });
        ctl.appendChild(b);
    }
    row.appendChild(ctl);
    const desc = _el("textarea", INPUT + "resize:none;height:30px;width:100%;box-sizing:border-box;");
    desc.placeholder = "what this reference is -- it becomes the definition line";
    desc.value = e.desc || "";
    desc.addEventListener("change", () => { const all = _entries(node); all[i].desc = desc.value; _save(node, all); });
    col.append(row, desc);
    card.append(thumb, col);
    return card;
}

function _render(node) {
    const st = node._plsBoard;
    if (!st) return;
    const entries = _entries(node);
    const c = counts(entries);
    st.count.textContent = "images " + c.image + "/9 \u00B7 videos " + c.video + "/3 \u00B7 audios " + c.audio + "/3";
    st.list.replaceChildren();
    if (!entries.length) {
        st.list.appendChild(_el("div", "color:#888;font-size:11px;padding:14px;text-align:center;border:1px dashed #444;border-radius:4px;",
            "drop images, videos or audio here -- or use + add"));
    }
    entries.forEach((_, i) => st.list.appendChild(_card(node, entries, i)));
}

function _field(node) {
    if (_w(node, FIELD) || typeof node.addDOMWidget !== "function") return;
    const root = _el("div", "width:100%;height:" + HEIGHT + "px;box-sizing:border-box;display:flex;flex-direction:column;gap:4px;");
    const bar = _el("div", "display:flex;gap:6px;align-items:center;");
    const add = _el("button", "background:#2e7d32;color:#fff;border:0;border-radius:3px;font-size:11px;padding:2px 8px;cursor:pointer;", "+ add");
    const pick = document.createElement("input");
    pick.type = "file"; pick.multiple = true; pick.accept = "image/*,video/*,audio/*"; pick.style.display = "none";
    add.addEventListener("click", () => pick.click());
    pick.addEventListener("change", () => { _addFiles(node, Array.from(pick.files || [])); pick.value = ""; });
    const count = _el("span", "font-size:11px;color:#aaa;");
    const msg = _el("span", "font-size:11px;color:#e57373;margin-left:auto;");
    bar.append(add, pick, count, msg);
    const list = _el("div", "flex:1 1 auto;overflow-y:auto;display:flex;flex-direction:column;gap:4px;");
    root.addEventListener("dragover", (ev) => { ev.preventDefault(); root.style.outline = "1px dashed #9ccc65"; });
    root.addEventListener("dragleave", () => { root.style.outline = ""; });
    root.addEventListener("drop", (ev) => {
        ev.preventDefault(); ev.stopPropagation(); root.style.outline = "";
        _addFiles(node, Array.from((ev.dataTransfer && ev.dataTransfer.files) || []));
    });
    root.append(bar, list);
    try {
        const w = node.addDOMWidget(FIELD, "pls-board", root, { serialize: false });
        w.serialize = false;
        w.computeSize = (width) => [width, HEIGHT];
        node._plsBoard = { list, count, msg };
    } catch (e) { /* a view must never break node creation */ }
}

app.registerExtension({
    name: "polyhedron.reference_board",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = created?.apply(this, arguments);
            setHidden(_w(this, "board"), true);
            _field(this);
            _render(this);
            if (this.size && this.size[0] < 460) this.setSize([460, this.size[1]]);
            refit(this);
            return r;
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = configured?.apply(this, arguments);
            setHidden(_w(this, "board"), true);
            _render(this);
            return r;
        };
    },
});
