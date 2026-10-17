/**
 * Polyhedron model door -- web/js/ph_model_door.js (v1059)
 *
 * ONE place for three things every model download in the pack needs:
 *
 * 1. The list-form healer. Our model combos carry decoration ("file \u00b7 516 MB",
 *    "\u25c8 rife426.pth \u00b7 23 MB", "... \u00b7 not installed"). A graph saved
 *    by hand, by a template builder or by an older build holds the BARE file
 *    name. ComfyUI's frontend marks every combo whose value ends in a model
 *    extension and is not among the offered values as a MISSING MODEL (red
 *    node + its own notice) -- even when the file is on disk. The healer snaps
 *    such a value onto the entry the list offers, SYNCHRONOUSLY in
 *    onConfigure, i.e. before the frontend scans the loaded graph.
 *
 * 2. The notice: the house's amber box top right (the cutout / TAE form),
 *    with Download + Later and a progress bar. Once per file per session.
 *
 * 3. The install call + progress hub: POST /pls/weights/install {name}; the
 *    server fetches from the pinned https address, checks sha256 and size and
 *    writes into the folder ComfyUI reports. A second request for the same
 *    file joins the running one. Progress arrives as "pls.weights.progress"
 *    and is fanned out to every listener (node line AND notice).
 *
 * Nothing is ever downloaded without a click. House style: English UI text,
 * ASCII source.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const PACK_MODULE = "PolyhedronLoRAStack";
const AMBER = "#ff8c00";
const TAIL_MISSING = /\s\u00b7\snot installed$/;
const TAIL_SIZE = /\s\u00b7\s[\d.,]+\s?(KB|MB|GB|TB)$/;

/* ------------------------------------------------------------ 1. healer */

export function stripModelLabel(v) {
    let out = String(v == null ? "" : v).trim();
    out = out.replace(TAIL_MISSING, "").replace(TAIL_SIZE, "").trimEnd();
    if (out.startsWith("\u25c8 ")) out = out.slice(2);
    return out;
}

export function healCombos(node) {
    let n = 0;
    for (const w of (node && node.widgets) || []) {
        const opts = w && w.options && w.options.values;
        if (!Array.isArray(opts) || !opts.length || typeof w.value !== "string") continue;
        if (opts.includes(w.value)) continue;
        const bare = stripModelLabel(w.value);
        if (!bare) continue;
        const hit = opts.find((o) => typeof o === "string" && stripModelLabel(o) === bare);
        if (hit !== undefined) { w.value = hit; n++; }
    }
    return n;
}

/* ------------------------------------------------------------ 3. install + progress */

const _sinks = new Map();          // file name -> Set of (done, total, phase) callbacks
const _running = new Map();        // file name -> Promise<path>
let _hooked = false;

function hookProgress() {
    if (_hooked) return;
    _hooked = true;
    try {
        api.addEventListener("pls.weights.progress", (ev) => {
            const d = ev && ev.detail;
            const set = d && _sinks.get(d.name);
            if (!set) return;
            for (const fn of set) { try { fn(d.done, d.total, d.phase); } catch (e) { /* one sink */ } }
        });
    } catch (e) { /* older frontend: no bar, the result still arrives */ }
}

export function onProgress(name, fn) {
    hookProgress();
    if (!_sinks.has(name)) _sinks.set(name, new Set());
    _sinks.get(name).add(fn);
    return () => { const s = _sinks.get(name); if (s) s.delete(fn); };
}

export function isDownloading(name) {
    return _running.has(name);
}

export function installModel(name) {
    if (_running.has(name)) return _running.get(name);
    hookProgress();
    const p = (async () => {
        try {
            const r = await api.fetchApi("/pls/weights/install", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name }),
            });
            const d = r && await r.json();
            if (!d || !d.ok) throw new Error((d && d.error) || "no answer from the server");
            _doneHooks.forEach((fn) => { try { fn(name, d.path); } catch (e) { /* ignore */ } });
            return d.path;
        } finally {
            _running.delete(name);
        }
    })();
    _running.set(name, p);
    return p;
}

const _doneHooks = new Set();
export function onInstalled(fn) { _doneHooks.add(fn); return () => _doneHooks.delete(fn); }

export async function modelStatus(set) {
    const r = await api.fetchApi("/pls/weights/status?set=" + encodeURIComponent(set));
    const d = r && await r.json();
    return (d && d.ok && Array.isArray(d.models)) ? d.models : [];
}

export function fmtBytes(n) {
    n = Number(n) || 0;
    if (n >= 1073741824) return (n / 1073741824).toFixed(1) + " GB";
    if (n >= 1048576) return Math.round(n / 1048576) + " MB";
    return Math.max(1, Math.round(n / 1024)) + " KB";
}

export function refreshModelLists() {
    try { app.refreshComboInNodes && app.refreshComboInNodes(); } catch (e) { /* lists refresh on R */ }
}

/* ------------------------------------------------------------ 2. notice */

let _css = false;
let _stack = null;
const _shown = new Set();          // file names noticed this session

function ensureCss() {
    if (_css) return;
    _css = true;
    const st = document.createElement("style");
    st.textContent =
        ".pls-door-stack{position:fixed;top:16px;right:16px;z-index:10000;display:flex;"
        + "flex-direction:column;gap:8px;max-width:360px;}"
        + ".pls-door{background:#1d1d1d;border:1px solid " + AMBER + ";border-left:4px solid " + AMBER
        + ";color:#ddd;font:12px sans-serif;padding:8px 10px;border-radius:4px;"
        + "box-shadow:0 4px 14px rgba(0,0,0,.5);}"
        + ".pls-door h4{margin:0 0 4px;font-size:12px;color:" + AMBER + ";}"
        + ".pls-door p{margin:0 0 8px;color:#bbb;line-height:1.35;word-break:break-word;}"
        + ".pls-door .row{display:flex;gap:8px;}"
        + ".pls-door button{flex:1 1 0;padding:4px 6px;cursor:pointer;background:#2a2a2a;"
        + "color:#ddd;border:1px solid #555;border-radius:3px;}"
        + ".pls-door button.go{background:#8fff8f;color:#123;border-color:#8fff8f;}"
        + ".pls-door .bar{height:6px;border-radius:3px;background:#444;overflow:hidden;margin:0 0 6px;display:none;}"
        + ".pls-door .bar i{display:block;height:100%;width:0;background:#8fff8f;}";
    document.head.appendChild(st);
}

function stack() {
    if (_stack && document.body.contains(_stack)) return _stack;
    _stack = document.createElement("div");
    _stack.className = "pls-door-stack";
    _stack.setAttribute("data-ph-overlay", "1");
    document.body.appendChild(_stack);
    return _stack;
}

/**
 * Show the amber notice for one registry file.
 *   m      : a status entry {name, title, size, bytes, dir, installed}
 *   who    : the node title that needs it (text only)
 *   after  : optional callback(path) once installed
 * Returns false when it was already shown this session or the file is there.
 */
export function noticeMissing(m, who, after) {
    if (!m || m.installed || _shown.has(m.name)) return false;
    _shown.add(m.name);
    ensureCss();
    const box = document.createElement("div");
    box.className = "pls-door";
    const h = document.createElement("h4");
    // v1092: on disk but OUTDATED (an earlier pack version's label bank) is said as such; when the
    // door can rebuild it locally the click downloads nothing
    h.textContent = "\u26a0 " + m.title + (m.outdated ? ": outdated (" + m.outdated + ")" : " is not installed");
    const p = document.createElement("p");
    p.textContent = m.rebuild
        ? (who ? who + " needs " : "Needed: ") + "a rebuilt label bank. The click rebuilds it from the files already in "
          + m.dir + " and removes the old one -- nothing downloads."
        : (who ? who + " needs " : "Needed: ") + m.name + " (" + m.size + "). "
          + "It goes to " + m.dir + " -- from the pinned address, sha256-checked. "
          + (m.outdated ? "The outdated file is removed by the door. " : "")
          + "Nothing downloads unless you click.";
    const bar = document.createElement("div");
    bar.className = "bar";
    const fill = document.createElement("i");
    bar.appendChild(fill);
    const row = document.createElement("div");
    row.className = "row";
    const go = document.createElement("button");
    go.className = "go";
    go.textContent = m.rebuild ? "Rebuild (no download)" : "Download (" + m.size + ")";
    const no = document.createElement("button");
    no.textContent = "Later";
    row.appendChild(go);
    row.appendChild(no);
    box.appendChild(h);
    box.appendChild(p);
    box.appendChild(bar);
    box.appendChild(row);
    stack().appendChild(box);

    let off = null;
    const close = () => { if (off) off(); try { box.remove(); } catch (e) { /* gone */ } };
    const showBar = (done, total, phase) => {
        bar.style.display = "block";
        go.disabled = true;
        go.textContent = phase === "verify" ? "checking sha256 ..." : phase === "convert" ? "converting ..." : "downloading ...";
        const pct = total ? Math.min(100, done / total * 100) : 0;
        fill.style.width = pct.toFixed(1) + "%";
        if (total) p.textContent = fmtBytes(done) + " of " + fmtBytes(total) + " -> " + m.dir;
    };
    off = onProgress(m.name, showBar);
    no.onclick = close;
    go.onclick = async () => {
        showBar(0, m.bytes, "start");
        try {
            const path = await installModel(m.name);
            p.textContent = "\u2713 Installed: " + path;
            h.textContent = "\u2713 " + m.title + " installed";
            go.style.display = "none";
            no.textContent = "Close";
            bar.style.display = "none";
            refreshModelLists();
            if (after) { try { after(path); } catch (e) { /* caller */ } }
            setTimeout(close, 6000);
        } catch (e) {
            bar.style.display = "none";
            h.textContent = "\u26a0 " + m.title + ": download failed";
            p.textContent = String((e && e.message) || e);
            go.disabled = false;
            go.textContent = "Retry";
        }
    };
    // the same file being fetched from the node line: mirror it
    if (isDownloading(m.name)) showBar(0, m.bytes, "start");
    return true;
}

/* ------------------------------------------------------------ registration */

app.registerExtension({
    name: "polyhedron.model_door",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData || !String(nodeData.python_module || "").includes(PACK_MODULE)) return;
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = configured ? configured.apply(this, arguments) : undefined;
            try { healCombos(this); } catch (e) { /* never break a load */ }
            return r;
        };
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = created ? created.apply(this, arguments) : undefined;
            setTimeout(() => { try { healCombos(this); } catch (e) { /* ignore */ } }, 0);
            return r;
        };
    },
});
