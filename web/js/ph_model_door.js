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
            forgetStatus();                  // v1220: the folders changed -- the next status asks the server
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

/* v1220 (the UI freeze, Frank 23.10.): a workflow with many model lines asked the status route once per line and
 * per check -- the same set again and again while it loaded. One answer per set now serves every line that asks
 * within STATUS_KEEP ms (asked together = ONE request); an install forgets it. */
export const STATUS_KEEP = 1500;
const _status = new Map();          // set -> {t, p}
export function forgetStatus() { _status.clear(); }
export function modelStatus(set) {
    const key = String(set || ""), now = Date.now(), hit = _status.get(key);
    if (hit && now - hit.t < STATUS_KEEP) return hit.p;
    const p = (async () => {
        const r = await api.fetchApi("/pls/weights/status?set=" + encodeURIComponent(set));
        const d = r && await r.json();
        return (d && d.ok && Array.isArray(d.models)) ? d.models : [];
    })();
    _status.set(key, { t: now, p });
    p.catch(() => { if (_status.get(key) && _status.get(key).p === p) _status.delete(key); });   // a failure is not kept
    return p;
}

export function fmtBytes(n) {
    n = Number(n) || 0;
    if (n >= 1073741824) return (n / 1073741824).toFixed(1) + " GB";
    if (n >= 1048576) return Math.round(n / 1048576) + " MB";
    return Math.max(1, Math.round(n / 1024)) + " KB";
}

/* v1220 (the UI freeze): ONE list refresh for a burst -- a download's line, its notice and a set's loop each asked
 * for one (refreshComboInNodes re-reads object_info for EVERY node); now they fold into one, REFRESH_WAIT ms after
 * the last ask */
export const REFRESH_WAIT = 400;
let _refreshT = null;
export function refreshModelLists() {
    if (_refreshT) clearTimeout(_refreshT);
    _refreshT = setTimeout(() => {
        _refreshT = null;
        try { app.refreshComboInNodes && app.refreshComboInNodes(); } catch (e) { /* lists refresh on R */ }
    }, REFRESH_WAIT);
}

/* ------------------------------------------------------------ 2. notice */

let _css = false;
let _stack = null;
const _shown = new Set();          // file names noticed this session

/* v1204 (Frank 21.10.: "Das sollte dann erscheinen, sobald man es wirklich braucht ergo: nutzen moechte ... das muss
 * schaerfer sein ... immer auch in Hinblick auf den Malware-Download und unsere Redflags bei ComfyUI"): ONE rule for
 * every download offer of the suite -- a node that is muted (mode 2) or bypassed (mode 4, also through its group)
 * needs nothing, so it offers nothing: no line, no notice. When it comes back, the offer is re-checked. */
export function nodeActive(node) {
    const m = node ? node.mode : 0;
    return m !== 2 && m !== 4;
}
const _watch = new Set();
let _watchTimer = null;
/* fn(node) whenever node.mode changes (bypass, mute, back) -- one quiet timer for every watched node, any renderer */
export function watchMode(node, fn) {
    if (!node || typeof fn !== "function") return;
    _watch.add({ node, fn, mode: node.mode });
    if (_watchTimer) return;
    _watchTimer = setInterval(() => {
        for (const e of Array.from(_watch)) {
            if (!e.node.graph) { _watch.delete(e); continue; }
            if (e.node.mode !== e.mode) {
                e.mode = e.node.mode;
                try { e.fn(e.node); } catch (x) { /* an offer never breaks the canvas */ }
            }
        }
    }, 1200);
}

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
        + ".pls-door button{flex:1 1 0;padding:4px 6px;cursor:pointer;background:linear-gradient(#34476a,#2a3a57);"
        + "color:#e8eefb;border:1px solid #4d6898;border-radius:3px;}"
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
export function noticeMissing(m, who, after, node) {
    if (!m || m.installed || _shown.has(m.name)) return false;
    if (node && !nodeActive(node)) return false;   // v1204: a bypassed / muted node asks for nothing (not marked shown)
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

    /* v1217 (Frank 23.10.: the model was complete, the line green, the notice still said "1 KB of 1 KB ...
     * downloading ..." and never closed): a download started on the NODE LINE only fed this box its progress -- the
     * finish ran in this box's own click handler alone, and the server's last two events ("done" 1/1, "error" 0/0)
     * were drawn as more progress. Now the box ends on whatever ends the download: its own click, the line's
     * install (onInstalled), the server's "done" / "error" event -- once. */
    let off = null, offDone = null, ended = false;
    const close = () => { if (off) off(); if (offDone) offDone(); try { box.remove(); } catch (e) { /* gone */ } };
    const finish = (path) => {
        if (ended) return;
        ended = true;
        p.textContent = "\u2713 Installed: " + (path || m.dir);
        h.textContent = "\u2713 " + m.title + " installed";
        go.style.display = "none";
        no.textContent = "Close";
        bar.style.display = "none";
        refreshModelLists();
        if (after) { try { after(path); } catch (e) { /* caller */ } }
        setTimeout(close, 6000);
    };
    const fail = (msg) => {
        if (ended) return;
        bar.style.display = "none";
        h.textContent = "\u26a0 " + m.title + ": download failed";
        p.textContent = msg;
        go.disabled = false;
        go.textContent = "Retry";
    };
    const showBar = (done, total, phase) => {
        if (ended) return;
        if (phase === "done") { finish(""); return; }
        if (phase === "error") { fail("The download stopped -- the node's line names the reason; Retry tries again."); return; }
        bar.style.display = "block";
        go.disabled = true;
        go.textContent = phase === "verify" ? "checking sha256 ..." : phase === "convert" ? "converting ..." : "downloading ...";
        const pct = total ? Math.min(100, done / total * 100) : 0;
        fill.style.width = pct.toFixed(1) + "%";
        if (total) p.textContent = fmtBytes(done) + " of " + fmtBytes(total) + " -> " + m.dir;
    };
    off = onProgress(m.name, showBar);
    offDone = onInstalled((name, path) => { if (name === m.name) finish(path); });
    no.onclick = close;
    go.onclick = async () => {
        showBar(0, m.bytes, "start");
        try {
            finish(await installModel(m.name));
        } catch (e) {
            fail(String((e && e.message) || e));
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
