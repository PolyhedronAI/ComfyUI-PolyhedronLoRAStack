/*
 * ph_media_loader.js — ⬡ Polyhedron Media Loader (ULSMediaLoader)
 *
 * A Load-Image/Video node with:
 *   • a "Choose folder" picker — an in-canvas (themed) folder browser to pin ANY
 *     local folder via level navigation, recent-folder shortcuts, or an address-bar
 *     with live subfolder autocomplete; the pin is stored in node.properties and
 *     survives a reload,
 *   • a THUMBNAIL grid of that folder's images/videos/audio, paginated 20 per page
 *     (◀ Back / Next ▶); video tiles show the first frame + a ▶ badge, GIFs animate,
 *     audio tiles show a green faux-waveform card + a ♪ badge (v457),
 *   • "Browse" — a flat full list (thumb + name per row, filterable) for picking
 *     from long folders; videos and GIFs animate on row hover, audio plays on hover,
 *   • "Upload" — files always go to input/ (the view then switches to input),
 *   • "↻" jumps back to the input folder,
 *   • "▦ Batch…" defines an image batch (whole folder -> one IMAGE/VIDEO batch);
 *     "▦ Batch" toggles the node between that batch and single-file at any time —
 *     the grid stays a free browser either way, and the checked frames are kept,
 *   • a footer: pager on the left, full folder path on the right,
 *   • hover preview: image enlarges, VIDEO plays (muted loop) via /uls/media/file,
 *     AUDIO plays via /uls/media/file (one clip at a time); clicking an audio file
 *     puts it in the Selection with native <audio> controls. Audio is Stufe A:
 *     browse/preview only — no graph output yet (an AUDIO socket lands in Stufe B).
 *
 * The UI lives in a DOM widget (renderer-agnostic, Nodes-2.0 safe). The chosen
 * item is written into the node's hidden `media_ref` STRING widget as JSON
 * {folder, file, kind}; the backend (nodes/ph_media_loader.py) decodes whatever
 * that points at. node.properties.ph_media_state mirrors it for reload restore.
 *
 * Backend routes (nodes/uls_routes.py): /uls/media/{folders,list,thumb,file,upload}.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// v536 DIAG: per-file cache evidence (each JS file caches individually in the
// browser -- the Cockpit banner cannot vouch for THIS file). If this line is
// missing from the console, Firefox is serving an old ph_media_loader.js
// (Ctrl+Shift+R). Drop with the other v531 diagnostics once Bug B is measured.
console.info("[PLS v536 DIAG] ph_media_loader.js v544 loaded");

const NODE = "ULSMediaLoader";

// Solid play / pause glyphs for the batch-preview overlay (currentColor = button color).
const PLAY_SVG = '<svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><path d="M8 5v14l11-7z" fill="currentColor"/></svg>';
const PAUSE_SVG = '<svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><path d="M6 5h4v14H6zM14 5h4v14h-4z" fill="currentColor"/></svg>';

// Resize floor: below this the button bar / grid / selection have no room and
// would spill out of the node. Enforced in onResize; the CSS also degrades
// gracefully (wrapping bar, shrinkable selection) as a second line of defense.
const MIN_NODE_W = 360;
const MIN_NODE_H = 320;
// v683: one page of tiles. Shared by renderGrid and _syncDims so the
// dimensions fetched are exactly the tiles that get drawn.
const GRID_PAGE = 20;

// v684 (L2) — how far outside the visible strip a tile still counts as
// "about to be seen". Enough that scrolling never shows an empty cell,
// small enough that a folder page is not fetched wholesale.
const THUMB_MARGIN = "160px";

/** v684: reveal the thumbnails whose tiles have come into view. PURE apart
 *  from the two effects it exists for -- assigning src and unobserving --
 *  so a guard can drive it with fabricated entries.
 *
 *  A revealed image is unobserved IMMEDIATELY: the fetch must happen once,
 *  never again when the tile scrolls back into view. Anything without a
 *  pending source is unobserved too (it has already been revealed), so the
 *  observer's set only ever shrinks. */
function _thumbReveal(entries, unobserve) {
    let revealed = 0;
    for (const e of (entries || [])) {
        if (!e || !e.isIntersecting) continue;
        const img = e.target;
        if (!img) continue;
        const src = img.dataset ? img.dataset.phSrc : null;
        if (src) {
            img.src = src;
            delete img.dataset.phSrc;
            revealed++;
        }
        try { unobserve(img); } catch (err) { /* observer already gone */ }
    }
    return revealed;
}

// Tile grid sizing — computed deterministically in _layoutGrid (explicit pixel
// columns AND row height, so the old aspect-ratio/auto-row overlap can't happen).
// Tiles are square, grow toward TILE_MAX as the node widens (fewer, larger
// tiles), never shrink below TILE_MIN, and always keep TILE_GAP between them;
// leftover width centers the grid so the upscale radiates from the middle.
const TILE_MIN = 84;
const TILE_MAX = 140;
const TILE_GAP = 8;

// Selection preview: a small clip may upscale up to this factor (kept modest so
// the upscale stays reasonably crisp); large clips stay capped at the preview box,
// and the box itself grows with the node but is capped via .ph-media-preview max-width.
const PREVIEW_MAX_UPSCALE = 2;

// Recently pinned folders, browser-global (shared across all Media Loader nodes),
// surviving reloads. Pure convenience — never the source of truth for a pin.
const RECENT_KEY = "ph_media_recent_folders";
function getRecentFolders() {
    try { const a = JSON.parse(localStorage.getItem(RECENT_KEY) || "[]"); return Array.isArray(a) ? a : []; }
    catch (e) { return []; }
}
function pushRecentFolder(path) {
    if (!path) return;
    try {
        let arr = getRecentFolders().filter((p) => p !== path);
        arr.unshift(path);
        localStorage.setItem(RECENT_KEY, JSON.stringify(arr.slice(0, 12)));
    } catch (e) { /* ignore */ }
}

// ── v624: Solo-Selection view state ─────────────────────────────────────────
// Pure mode-switch bookkeeping, kept as a top-level function so the guard can
// drive it directly (test_v624_solo_selection). The node's size is remembered
// PER MODE: leaving a mode records the current size under that mode's slot;
// entering a mode returns its remembered size (null on the first visit — the
// caller then keeps the current size).
function _viewSwap(view, goSolo, curSize) {
    const v = Object.assign({ solo: false, tilesSize: null, soloSize: null }, view || {});
    v[v.solo ? "soloSize" : "tilesSize"] = [curSize[0], curSize[1]];
    v.solo = !!goSolo;
    const t = v.solo ? v.soloSize : v.tilesSize;
    return { view: v, size: (Array.isArray(t) && t.length === 2) ? [t[0], t[1]] : null };
}

// ── v625: fixed-length trim window ──────────────────────────────────────────
// Pure window bookkeeping, top-level so the guard can drive it directly
// (test_v625_fix_trim). Given a desired window START (seconds), a FIXED length
// (seconds), the clip duration and a snap granularity (1/native-fps for video —
// the very fps the backend's _slice_frames cuts with — or 0.1 for audio tenths),
// return the clamped, snapped [start, end]. THE LENGTH IS THE LAW: it never
// changes; only the position moves, kept fully inside the clip.
function _fixWindow(start, len, dur, snap) {
    const g = (snap && snap > 0) ? snap : 0.1;
    len = Math.max(g, Math.min(+len || 0, dur));
    let s = Math.max(0, Math.min(+start || 0, dur - len));
    s = Math.round(s / g) * g;                       // land on a frame/tenth boundary
    s = Math.max(0, Math.min(s, dur - len));         // snapping must not push past the end
    return [s, s + len];
}

// ── one-time CSS ────────────────────────────────────────────────────────────
function injectCSS() {
    if (document.getElementById("ph-media-css")) return;
    const s = document.createElement("style");
    s.id = "ph-media-css";
    s.textContent = `
.ph-media { display:flex; flex-direction:column; gap:6px; width:100%; height:100%;
    box-sizing:border-box; font-size:11px; color:#cfcfcf; }
.ph-media-bar { display:flex; gap:6px; align-items:center; flex-wrap:wrap; }
.ph-media-btn { background:#2a2a2a; border:1px solid #444; color:#ddd; border-radius:5px;
    padding:3px 8px; cursor:pointer; white-space:nowrap; }
.ph-media-btn:hover { background:#383838; }
.ph-media-btn.ph-batch-toggle.on, .ph-media-btn.ph-audio-toggle.on { background:#2f5d2f; border-color:#4f8f4f; color:#dfffdf; }
.ph-media-btn.ph-batch-toggle.on:hover, .ph-media-btn.ph-audio-toggle.on:hover { background:#356a35; }
/* Reserve the width of the wider ("… OFF") label on both toggles so the button
   does not change size when the label flips ON<->OFF (no layout jitter). */
.ph-media-btn.ph-batch-toggle, .ph-media-btn.ph-audio-toggle { min-width:104px; text-align:center; box-sizing:border-box; }
.ph-media-path { flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
    color:#7fd17f; opacity:.85; direction:rtl; font-size:10px; }
.ph-media-foot { display:flex; align-items:center; gap:10px; min-height:18px; }
.ph-media-main { display:flex; gap:6px; flex:1; min-height:0; position:relative; }
.ph-media-busy { position:absolute; inset:0; display:none; align-items:center; justify-content:center;
    gap:10px; background:#1c1c1ccc; z-index:20; border-radius:6px; pointer-events:none; }
.ph-media-busy.on { display:flex; }
.ph-media-busy .ph-busy-ring { width:26px; height:26px; border-radius:50%;
    border:3px solid #444; border-top-color:#ff8c00; animation:ph-busy-spin 0.9s linear infinite; }
.ph-media-busy .ph-busy-label { color:#bbb; font-size:12px; }
@keyframes ph-busy-spin { to { transform:rotate(360deg); } }
/* v692: a tile waits for its thumbnail -- since v684 the URL is only assigned
   when the tile comes near view, and the server still has to decode the file
   to make one. Until then the tile was simply EMPTY, which reads as "nothing is
   happening". Same amber ring as the folder-level spinner, pure CSS on a class:
   no timer, no observer, nothing to leak. Removed on load or error. */
.ph-media-tile.ph-thumb-wait::after { content:""; position:absolute; left:50%; top:50%;
    width:18px; height:18px; margin:-9px 0 0 -9px; border-radius:50%;
    border:2px solid #3a3a3a; border-top-color:#ff8c00;
    animation:ph-busy-spin 0.9s linear infinite; pointer-events:none; z-index:2; }
/* v664: the drop CHOICE. A drag over the node raises two zones — aiming at one
   IS the decision, so nothing has to be known or held down. The zones exist only
   while a drag is in flight; the node's face is unchanged otherwise. They wrap
   under each other on a narrow node (min-width does it, no JS threshold). */
.ph-media-drophint { position:absolute; inset:0; display:none; flex-wrap:wrap; gap:8px;
    align-items:stretch; justify-content:center; background:#101014e6; z-index:30;
    border-radius:6px; pointer-events:none; padding:8px; box-sizing:border-box; }
.ph-media-drophint.on { display:flex; }
.ph-media-drophint .ph-dz { flex:1 1 190px; min-width:170px; display:flex; flex-direction:column;
    gap:6px; align-items:center; justify-content:center; text-align:center; padding:10px;
    border:2px dashed #444; border-radius:8px; background:#16161acc; pointer-events:auto;
    transition:border-color .12s, background .12s; }
.ph-media-drophint .ph-dz-icon { font-size:22px; line-height:1; }
.ph-media-drophint .ph-dz-title { font-size:12.5px; font-weight:600; color:#cfcfcf; }
.ph-media-drophint .ph-dz-sub { font-size:11px; color:#8a8a92; line-height:1.35; }
.ph-media-drophint .ph-dz.hot { border-color:#5cd07a; background:#1d2a1dcc; }
.ph-media-drophint .ph-dz.hot .ph-dz-title { color:#dfffdf; }
.ph-media-drophint .ph-dz.dim { opacity:.62; }
.ph-media-drophint .ph-dz.dim.hot { opacity:1; border-color:#e0b060; background:#2a231dcc; }
.ph-media-grid { flex:3 1 220px; min-width:0; overflow:hidden auto; display:grid; gap:8px;
    grid-template-columns:repeat(auto-fill, minmax(84px,1fr)); align-content:start; justify-content:center;
    background:#1c1c1c; border:1px solid #333; border-radius:6px; padding:6px; min-height:80px; }
.ph-media-preview { flex:1 1 140px; min-width:130px; max-width:340px; display:flex; flex-direction:column; gap:4px;
    background:#1c1c1c; border:1px solid #333; border-radius:6px; padding:6px; overflow:hidden; }
.ph-media-prev-lbl { font-size:10px; color:#888; }
.ph-media-prev-media { position:relative; flex:1; min-height:0; display:flex; align-items:center; justify-content:center;
    background:#111; border-radius:4px; overflow:hidden; }
.ph-media-prev-media img, .ph-media-prev-media video { max-width:100%; max-height:100%; object-fit:contain; display:block; image-rendering:auto; }
.ph-media-prev-cap { font-size:9px; color:#bbb; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
/* v624: Solo-Selection — the ⛶ toggle in the Selection header hides the tile grid
   (and its pager) so the Selection fills the whole node like a plain Load node.
   Node sizes are remembered PER MODE (node.properties.ph_media_view) and restored
   on every toggle; the tile grid's own layout bails at clientWidth 0, so hiding
   it is a no-op for _layoutGrid. */
.ph-media.ph-solo .ph-media-grid, .ph-media.ph-solo .ph-media-pager { display:none; }
.ph-media.ph-solo .ph-media-preview { max-width:none; flex:1 1 auto; }
.ph-media-prev-lbl { display:flex; align-items:center; gap:6px; }
.ph-solo-toggle { margin-left:auto; flex:none; cursor:pointer; background:#262626; color:#bbb;
    border:1px solid #444; border-radius:4px; font:10px/1 system-ui,sans-serif; padding:2px 6px; }
.ph-solo-toggle:hover { color:#eee; border-color:#666; }
.ph-media.ph-solo .ph-solo-toggle { color:#f0a030; border-color:#f0a030aa; }
.ph-media-tile { position:relative; aspect-ratio:1/1; border:2px solid transparent; border-radius:5px;
    overflow:hidden; cursor:pointer; background:#262626; }
.ph-media-tile.sel { border-color:#f0a030; }
.ph-media-tile.picked { border-color:#7fd17f; }                  /* batch-selected frame (green) */
.ph-media-tile.ph-match { outline:2px dashed #7fb3d1aa; outline-offset:-2px; }  /* live filter-match preview (v528) */
.ph-selc { position:absolute; top:3px; left:3px; width:18px; height:18px; z-index:4;
  border:2px dashed #b9b9c2aa; border-radius:50%; background:#000a; cursor:pointer;
  display:flex; align-items:center; justify-content:center;
  font:bold 12px/1 system-ui,sans-serif; color:#dfffdf; box-sizing:border-box; }
.ph-selc:hover { border-color:#e6e6e6; background:#000d; }
.ph-selc.on { border:2px solid #4f8f4f; background:#2f5d2f; }
.ph-selc.on:hover { background:#356a35; }
.ph-media-tile.audsel { border-color:#7fd17f; }                  /* v458: audio selection (green) */
.ph-media-tile.audsel::after { content:"✓"; position:absolute; top:3px; left:26px; z-index:3;
    min-width:16px; height:16px; padding:0 3px; box-sizing:border-box; border-radius:8px;
    background:#2e7d32; color:#fff; font:11px/16px system-ui,sans-serif; text-align:center;
    box-shadow:0 0 0 1px rgba(0,0,0,.45); }
.ph-media-tile img { width:100%; height:100%; object-fit:cover; display:block; }
video.ph-tile-video { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; z-index:1; }
.ph-media-tile .ph-ph { display:flex; align-items:center; justify-content:center; width:100%; height:100%;
    color:#777; font-size:18px; }
.ph-media-tile .ph-name { position:absolute; left:0; right:0; bottom:0; padding:2px 4px; z-index:2;
    background:linear-gradient(transparent,#000c); color:#eee; font-size:9px;
    overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.ph-media-tile .ph-vid { position:absolute; top:26px; right:3px; background:#000a; color:#ff8c00; z-index:2;
    border-radius:3px; font-size:9px; padding:0 3px; }              /* v624: below the dims badge; v625: CTE amber — video reads as video at a glance */
/* v457: audio tile — a soft-grey card with a deterministic faux-waveform (green)
   + a green ♪ badge. No decode: the waveform is drawn from a filename hash so each
   file gets a stable, distinct shape. */
.ph-media-tile .ph-aud-card { display:flex; align-items:center; justify-content:center;
    width:100%; height:100%; box-sizing:border-box; padding:13% 9%;
    background:linear-gradient(160deg,#50504f,#3a3a39); }
.ph-aud-wave { width:100%; height:60%; display:block; }
.ph-aud-wave rect { fill:#7fd17f; }
.ph-media-tile .ph-aud { position:absolute; top:3px; right:3px; z-index:2;
    background:#0b0b0bcc; color:#7fd17f; border-radius:3px; font-size:11px;
    line-height:1; padding:2px 5px; }
/* v457: audio in the Selection column — the same waveform card above native controls. */
.ph-aud-prevwrap { display:flex; flex-direction:column; align-items:center; gap:8px;
    width:100%; padding:6px 4px; box-sizing:border-box; }
.ph-aud-prevwrap .ph-aud-card { width:70%; max-width:200px; aspect-ratio:16/7; height:auto;
    border-radius:6px; }
.ph-aud-prev { width:92%; max-width:300px; }
/* v458: collapsible audio companion pane in the Selection column (below the visual) */
.ph-media-prev-audio { flex:none; display:flex; flex-direction:column; gap:3px;
    margin-top:4px; padding-top:4px; border-top:1px solid #333; }
/* v476: video trim pane — a sibling under the video preview; reuses the .ph-trim strip */
.ph-media-prev-vtrim { flex:none; display:flex; flex-direction:column; gap:3px;
    margin-top:4px; padding-top:4px; border-top:1px solid #333; }
.ph-media-prev-vtrim .ph-vtrim-head { display:flex; align-items:center; gap:6px; }
.ph-media-prev-vtrim .ph-vtrim-label { font-size:10px; color:#cfcfcf; overflow:hidden;
    text-overflow:ellipsis; white-space:nowrap; }
.ph-aud-head { display:flex; align-items:center; gap:4px; }
.ph-aud-name { font-size:10px; color:#cfcfcf; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.ph-aud-btn { background:#2e2e2e; border:1px solid #444; color:#ddd; border-radius:3px; cursor:pointer;
    font-size:11px; line-height:1; padding:2px 5px; }
.ph-aud-btn:hover { background:#383838; }
.ph-aud-src { font-size:8px; color:#8a8a92; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.ph-media-prev-audio .ph-aud-prevwrap { padding:2px 4px; gap:4px; }
.ph-media-prev-audio .ph-aud-prevwrap .ph-aud-card { width:55%; max-width:150px; }
.ph-media-prev-audio .ph-aud-prev { width:96%; }
/* v464: audio trim strip — two handles + numeric in/out (bidirectional) */
.ph-trim { display:flex; flex-direction:column; gap:5px; width:100%; padding:3px 4px 4px; box-sizing:border-box; }
.ph-trim-track { position:relative; height:24px; border-radius:5px; background:#262626; border:1px solid #444;
    cursor:pointer; user-select:none; overflow:hidden; }
.ph-trim-dim { position:absolute; top:0; bottom:0; background:#000a; pointer-events:none; }
.ph-trim-keep { position:absolute; top:0; bottom:0; background:#5cd07a22; pointer-events:none; }
.ph-trim-handle { position:absolute; top:-2px; bottom:-2px; width:11px; margin-left:-6px; cursor:ew-resize;
    z-index:2; display:flex; align-items:center; justify-content:center; touch-action:none; }
.ph-trim-handle::after { content:""; width:3px; height:62%; background:#5cd07a; border-radius:2px;
    box-shadow:0 0 2px #000; }
/* v626: the fields row WRAPS. With the v625 Fix field the row (Start · End · Fix · len)
   outgrows a narrow Selection column, and the column's overflow:hidden CLIPPED it —
   fields were invisible until the node was dragged very wide (Frank's screens). Wrapping
   keeps every field visible at any width; the height floor counts the taller pane. */
.ph-trim-fields { display:flex; flex-wrap:wrap; align-items:center; gap:4px 6px; font-size:11px; color:#bbb; }
.ph-trim-fields label { display:flex; align-items:center; gap:3px; }
.ph-trim-num { width:66px; box-sizing:border-box; background:#1a1a1a; border:1px solid #555; color:#ddd;
    border-radius:4px; padding:2px 4px; font-size:11px; }
.ph-trim-num:disabled { opacity:.5; }
.ph-trim-len { margin-left:auto; color:#5cd07a; white-space:nowrap; }
/* v625: fixed-length trim. The length readout is a BUTTON (click = lock the current
   frame count / tenth-seconds; click again = unlock). Locked, the keep zone and both
   handles drag the WHOLE window (length is the law); the readout goes CTE amber. */
.ph-trim-len { cursor:pointer; }
.ph-trim-len:hover { text-decoration:underline; }
.ph-trim-len.on { color:#ff8c00; }
.ph-trim-track.fixed .ph-trim-keep { pointer-events:auto; cursor:grab; }
.ph-trim-track.fixed .ph-trim-handle { cursor:grab; }
.ph-trim-fix { width:46px; }
/* v466: play-selection loop — circular toggle left of the native player; the glyph
   spins while looping (the "Warte-Kreisel"). Bounds are read live so it follows the
   handles. */
.ph-aud-playrow { display:flex; align-items:center; gap:6px; width:100%; }
.ph-media-prev-audio .ph-aud-playrow .ph-aud-prev { width:auto; flex:1; min-width:0; }
/* v468: fixed compact size again — the v467 align-self:stretch + aspect-ratio blew the
   button up to fill the whole row (it ate the native player's width). Back to a 30px
   circle, centered in the row next to the play button. The glyph stays a CENTERED inline
   SVG (ring centered in the viewBox) so the spin runs around the true center — no wobble. */
.ph-trim-loop { flex:none; width:30px; height:30px; min-width:30px; padding:0; border-radius:50%;
    background:#2e2e2e; border:1px solid #444; color:#ddd; cursor:pointer; line-height:0;
    display:flex; align-items:center; justify-content:center; box-sizing:border-box; }
.ph-trim-loop:hover { background:#383838; }
.ph-trim-loop span { display:flex; align-items:center; justify-content:center; width:100%; height:100%; }
.ph-trim-loop svg { width:58%; height:58%; display:block; }
.ph-trim-loop.playing { color:#5cd07a; border-color:#5cd07a; background:#5cd07a1a; }
.ph-trim-loop.playing span { animation:ph-spin 1.4s linear infinite; }
@keyframes ph-spin { to { transform:rotate(360deg); } }
/* v624: pixel-dimension badge (W x H) — TOP-RIGHT on every view (tile, hover pop,
   Selection). The ○ check circle keeps the top-left corner; on video tiles the
   ▸duration badge moves below the dims (top:26px) so the two never overlap. */
.ph-media-tile .ph-dim, .ph-media-pop .ph-dim { position:absolute; top:3px; right:3px; z-index:2;
    background:#000a; color:#eee; border-radius:3px; font-size:9px; line-height:1.4; padding:0 3px;
    letter-spacing:.2px; pointer-events:none; }
.ph-media-prev-media .ph-dim-prev { position:absolute; top:5px; right:5px; z-index:3; background:#000b;
    color:#fff; border-radius:4px; font-size:11px; font-weight:600; line-height:1.3; padding:2px 7px;
    letter-spacing:.3px; box-shadow:0 1px 3px #0007; pointer-events:none; }
.ph-br-dim { color:#888; font-size:10px; margin-left:6px; flex:none; white-space:nowrap; }
.ph-media-empty { color:#777; padding:14px; text-align:center; grid-column:1 / -1; }
/* image-batch preview (Selection column when Batch is ON) — counter + play stage */
.ph-batch-prev-head { font-size:11px; color:#7fd17f; padding:2px 4px 0; text-align:center; line-height:1.3; }
.ph-batch-prev-warn { color:#e0a85a; }
/* batch animation preview: solid play overlay on the stage (the frames live in the grid) */
.ph-batch-stagewrap { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; cursor:pointer; }
.ph-batch-stage { display:block; max-width:100%; max-height:100%;
    border-radius:4px; background:#0a0a0a; object-fit:contain; border:1px solid #2a2a2a; }
.ph-batch-playov { position:absolute; left:50%; top:50%; transform:translate(-50%,-50%);
    width:46px; height:46px; border-radius:50%; pointer-events:none; box-sizing:border-box;
    background:rgba(15,15,15,0.60); border:1px solid rgba(255,255,255,0.45); color:#fff;
    display:flex; align-items:center; justify-content:center; opacity:.9;
    transition:opacity .15s, background .15s; }
.ph-batch-stagewrap:hover .ph-batch-playov { background:rgba(15,15,15,0.82); opacity:1; }
.ph-batch-playov.playing { opacity:.30; }                 /* fade during play, like native video controls */
.ph-batch-stagewrap:hover .ph-batch-playov.playing { opacity:.9; }
.ph-media-pop { position:fixed; z-index:10000; pointer-events:none; border:1px solid #555;
    border-radius:6px; box-shadow:0 6px 24px #000a; background:#111; padding:2px; }
.ph-media-pop img { max-width:320px; max-height:320px; display:block; border-radius:4px; }
/* folder picker modal — z-index above the Batch panel overlay (99999) so the
   picker (and the Browse modal, which shares .ph-fp-back) opens ON TOP of, and
   remains usable above, the Batch panel when launched from its Choose… button. */
.ph-fp-back { position:fixed; inset:0; z-index:100001; background:#0008; display:flex;
    align-items:center; justify-content:center; }
.ph-fp { width:520px; max-width:92vw; max-height:80vh; background:#222; border:1px solid #555;
    border-radius:8px; display:flex; flex-direction:column; color:#ddd; font-size:12px; }
.ph-fp-head { display:flex; gap:6px; align-items:center; padding:8px 10px; border-bottom:1px solid #444; }
.ph-fp-pastewrap { position:relative; flex:1; min-width:0; }
.ph-fp-paste { width:100%; box-sizing:border-box; background:#1a1a1a; border:1px solid #555; color:#ddd;
    border-radius:5px; padding:4px 8px; font-size:12px; }
.ph-fp-suggest { position:absolute; top:100%; left:0; right:0; z-index:6; display:none;
    background:#1a1a1a; border:1px solid #555; border-top:none; border-radius:0 0 5px 5px;
    max-height:210px; overflow-y:auto; }
.ph-fp-suggest-row { padding:4px 8px; cursor:pointer; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.ph-fp-suggest-row:hover { background:#333; }
.ph-fp-recent { padding:5px 10px; border-bottom:1px solid #383838; display:flex; flex-direction:column;
    gap:2px; max-height:118px; overflow-y:auto; flex:0 0 auto; }  /* v658: never squeezed by a tall list (the 54-row peek) */
.ph-fp-recent-lbl { color:#888; font-size:10px; }
.ph-fp-recent-row { color:#9ad; cursor:pointer; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
    direction:rtl; text-align:left; padding:1px 0; }
.ph-fp-recent-row:hover { color:#bcf; }
.ph-fp-hint { padding:12px 14px; color:#8a8a92; font-size:11px; line-height:1.7; flex:0 0 auto; }
.ph-fp-listview { flex:1; min-height:0; display:flex; flex-direction:column; }
.ph-fp-cur { min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
    direction:rtl; text-align:left; color:#9ad; padding:5px 10px; border-bottom:1px solid #383838;
    flex:0 0 auto; }  /* v659: same shrink law as Recent -- the tall list must not squeeze the path row */
.ph-fp-list { flex:1; overflow-y:auto; padding:6px; }
.ph-fp-row { display:flex; gap:6px; align-items:center; padding:5px 8px; border-radius:5px; cursor:pointer; }
.ph-fp-row:hover { background:#333; }
.ph-fp-foot { display:flex; gap:8px; justify-content:flex-end; align-items:center; padding:8px 10px; border-top:1px solid #444; }
.ph-fp-pin { background:#3a6; border:none; color:#fff; }
.ph-fp-x { background:#444; border:none; color:#ddd; }
.ph-fp-foot button { border-radius:5px; padding:5px 12px; cursor:pointer; }
.ph-media-btn:disabled { opacity:.4; cursor:default; }
.ph-media-pager { display:flex; gap:8px; align-items:center; min-height:18px; flex:none; }
.ph-media-pageinfo { color:#999; font-size:10px; }
.ph-media-pop video { max-width:320px; max-height:320px; display:block; border-radius:4px; background:#000; }
/* browse-folder modal (flat full list, thumb + name per row) */
.ph-br { width:560px; max-width:92vw; max-height:80vh; background:#222; border:1px solid #555;
    border-radius:8px; display:flex; flex-direction:column; color:#ddd; font-size:12px; }
.ph-br-head { display:flex; gap:6px; align-items:center; padding:8px 10px; border-bottom:1px solid #444; }
.ph-br-filter { flex:1; min-width:0; background:#1a1a1a; border:1px solid #555; color:#ddd;
    border-radius:5px; padding:4px 8px; font-size:12px; }
.ph-br-list { flex:1; overflow-y:auto; padding:4px; }
.ph-br-row { display:flex; gap:8px; align-items:center; padding:3px 6px; border-radius:5px;
    cursor:pointer; border:1px solid transparent; }
.ph-fp-media-hd { color:#9a9a9a; font-size:11px; padding:6px 4px 2px; }
.ph-fp-media-row { color:#7f7f7f; font-size:11px; padding:1px 4px 1px 12px; overflow:hidden;
    text-overflow:ellipsis; white-space:nowrap; cursor:default; }
.ph-br-row:hover { background:#2e2e2e; }
.ph-br-row.sel { border-color:#f0a030; }
.ph-br-thumb { width:40px; height:40px; object-fit:cover; border-radius:3px; background:#262626; flex:none; display:block; }
.ph-br-thumbwrap { position:relative; width:40px; height:40px; flex:none; overflow:hidden; border-radius:3px; }
.ph-br-name { min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; flex:1; }
.ph-br-vid { color:#f0a030; margin-left:6px; font-size:10px; }
.ph-br-aud { display:flex; align-items:center; justify-content:center; width:100%; height:100%;
    background:linear-gradient(160deg,#50504f,#3a3a39); color:#7fd17f; font-size:16px; }
`;
    document.head.appendChild(s);
}

// v655: listing signature for the silent focus re-read — name+size+mtime per
// entry, so additions, removals AND in-place replacements all register. Pure
// (guard-driven via node).
function _filesSig(files) {
    return (files || []).map((f) => f.name + "|" + (f.size ?? "") + "|" + (f.mtime ?? "")).join("\n");
}

// ── server-side folder picker modal ────────────────────────────────────────
async function openFolderPicker(startPath, onPin) {
    injectCSS();
    const back = document.createElement("div"); back.className = "ph-fp-back";
    const box = document.createElement("div"); box.className = "ph-fp";
    box.innerHTML = `
      <div class="ph-fp-head">
        <button class="ph-media-btn ph-fp-up">⬆ up</button>
        <button class="ph-media-btn ph-fp-down" title="Go one level deeper — into the first subfolder">⬇ in</button>
        <div class="ph-fp-pastewrap">
          <input class="ph-fp-paste" type="text" spellcheck="false" placeholder="type or paste a folder path…">
          <div class="ph-fp-suggest"></div>
        </div>
        <button class="ph-media-btn ph-fp-native" title="The Windows dialog — browse anywhere">📁</button>
      </div>
      <div class="ph-fp-recent"></div>
      <div class="ph-fp-hint">📁 — the Windows dialog: pick ANY file in your media folder, its folder gets pinned.<br>⬆ up / ⬇ in — step out to the parent folder, or into the first subfolder.</div>
      <div class="ph-fp-listview" style="display:none">
        <div class="ph-fp-cur"></div>
        <div class="ph-fp-list"></div>
      </div>
      <div class="ph-fp-foot">
        <button class="ph-fp-x">Cancel</button>
        <button class="ph-fp-pin">Choose</button>
      </div>`;
    back.appendChild(box); document.body.appendChild(back);

    const curEl = box.querySelector(".ph-fp-cur");
    const listEl = box.querySelector(".ph-fp-list");
    const pasteEl = box.querySelector(".ph-fp-paste");
    const suggestEl = box.querySelector(".ph-fp-suggest");
    const hintEl = box.querySelector(".ph-fp-hint");
    const listviewEl = box.querySelector(".ph-fp-listview");
    const downBtn = box.querySelector(".ph-fp-down");
    let cur = startPath || "";
    let parent = "";
    let firstChild = "";   // first subfolder of the current folder, for ⬇ in

    function hideSuggest() { suggestEl.style.display = "none"; suggestEl.innerHTML = ""; }

    // The folder list opens collapsed (just Recent + the browse buttons). Any
    // navigation — ⬆ up, ⬇ in, a Recent row, the address bar — reveals it.
    function showList() { hintEl.style.display = "none"; listviewEl.style.display = "flex"; }

    // Parent of a path, for ⬆ up (one level higher). A drive root ("D:" /
    // "D:\") has no parent in this list, so fall back to the roots ("").
    function parentOf(p) {
        if (!p) return "";
        const t = p.replace(/[\\/]+$/, "");
        const i = Math.max(t.lastIndexOf("\\"), t.lastIndexOf("/"));
        if (i <= 0) return "";
        let par = t.slice(0, i);
        if (/^[A-Za-z]:$/.test(par)) par += "\\";   // "D:" -> "D:\"
        return par;
    }

    async function nav(path) {
        hideSuggest();
        showList();
        downBtn.disabled = true;          // re-evaluated once this folder's children load
        listEl.textContent = "Loading…";
        let d = null;
        try {
            const r = await api.fetchApi("/uls/media/folders?path=" + encodeURIComponent(path || ""));
            if (r && r.ok) d = await r.json();
        } catch (e) { /* ignore */ }
        if (!d || !d.ok) { listEl.textContent = "Cannot read this folder."; return; }
        cur = d.path || ""; parent = d.parent || "";
        curEl.textContent = cur || "(roots)";
        curEl.title = cur;
        // v427: mirror the resolved location into the address field so it acts as a
        // live "you are here" bar. Every way of moving — a recent row, a folder row,
        // ↑ up, Go, and the initial open — funnels through nav(), so this single
        // assignment covers them all: after clicking a recent the path sits up top,
        // ready to Choose (as after a fresh pick) or to edit / ↑ up one level.
        pasteEl.value = cur;
        listEl.innerHTML = "";
        const kids = d.folders || [];
        firstChild = kids.length ? kids[0].path : "";   // ⬇ in descends into the first subfolder
        downBtn.disabled = !firstChild;                 // greyed at a leaf — nothing below to enter
        for (const f of kids) {
            const row = document.createElement("div"); row.className = "ph-fp-row";
            row.textContent = "📁 " + f.name;
            row.onclick = () => nav(f.path);
            listEl.appendChild(row);
        }
        if (!kids.length) {
            const e = document.createElement("div"); e.className = "ph-media-empty";
            e.textContent = (d.media_total
                ? "No subfolders — Choose pins this folder."
                : "No subfolders here — pin this one if it holds your media.");
            listEl.appendChild(e);
        }
        // v651: peek at the folder's OWN media so a pin is an informed pick —
        // sorted names with kind icons, capped server-side (total in the header).
        const media = d.media || [];
        if (media.length) {
            const hd = document.createElement("div"); hd.className = "ph-fp-media-hd";
            hd.textContent = d.media_total + " media file" + (d.media_total === 1 ? "" : "s") +
                (d.media_total > media.length ? " (first " + media.length + " shown)" : "") + ":";
            listEl.appendChild(hd);
            const icons = { image: "🖼", video: "🎬", audio: "♪" };
            for (const m of media) {
                const row = document.createElement("div"); row.className = "ph-fp-media-row";
                row.textContent = (icons[m.kind] || "•") + " " + m.name;
                row.title = m.name;
                listEl.appendChild(row);
            }
        }
    }

    // address-bar autocomplete: split typed text into <parent>/<partial>, fetch
    // the parent's subfolders, and offer the ones that start with <partial>.
    let suggestTimer = null;
    function parseParentPartial(val) {
        const i = Math.max(val.lastIndexOf("\\"), val.lastIndexOf("/"));
        if (i < 0) return { parent: "", partial: val };
        let par = val.slice(0, i);
        const partial = val.slice(i + 1);
        if (/^[A-Za-z]:$/.test(par)) par += "\\";   // "D:" -> "D:\"
        return { parent: par, partial };
    }
    async function updateSuggest() {
        const val = pasteEl.value;
        if (!val) { hideSuggest(); return; }
        const { parent: par, partial } = parseParentPartial(val);
        let d = null;
        try {
            const r = await api.fetchApi("/uls/media/folders?path=" + encodeURIComponent(par));
            if (r && r.ok) d = await r.json();
        } catch (e) { /* ignore */ }
        if (!d || !d.ok) { hideSuggest(); return; }
        const pl = partial.toLowerCase();
        const matches = (d.folders || []).filter((f) => !pl || f.name.toLowerCase().startsWith(pl)).slice(0, 12);
        if (!matches.length) { hideSuggest(); return; }
        suggestEl.innerHTML = "";
        for (const f of matches) {
            const row = document.createElement("div"); row.className = "ph-fp-suggest-row";
            row.textContent = "📁 " + f.name; row.title = f.path;
            row.onmousedown = (e) => { e.preventDefault(); pasteEl.value = f.path; hideSuggest(); nav(f.path); };
            suggestEl.appendChild(row);
        }
        suggestEl.style.display = "block";
    }

    const goPaste = () => { const v = pasteEl.value.trim(); if (v) nav(v); };
    box.querySelector(".ph-fp-up").onclick = () => nav(parent || parentOf(cur));
    // ⬇ in -> descend one level, into the current folder's FIRST subfolder. Disabled
    // (greyed) at a leaf, where there is nothing below to enter. 📁 (the OS dialog) is
    // for browsing anywhere; ⬆ up steps back out to the parent.
    downBtn.disabled = true;   // enabled once a navigation reveals subfolders below
    downBtn.onclick = () => { if (firstChild) nav(firstChild); };
    pasteEl.addEventListener("input", () => { clearTimeout(suggestTimer); suggestTimer = setTimeout(updateSuggest, 200); });
    pasteEl.addEventListener("blur", () => setTimeout(hideSuggest, 150));
    pasteEl.addEventListener("keydown", (e) => {
        if (e.key === "Enter") { e.preventDefault(); hideSuggest(); goPaste(); }
        else if (e.key === "Escape") hideSuggest();
    });
    box.querySelector(".ph-fp-x").onclick = () => back.remove();
    back.onclick = (e) => { if (e.target === back) back.remove(); };
    // v651: Choose honors the ADDRESS BAR — a fully typed/pasted path pins
    // directly (validated against the folders route), no Enter-then-Choose
    // detour; with an untouched bar it pins the navigated folder as before.
    box.querySelector(".ph-fp-pin").onclick = async () => {
        const typed = pasteEl.value.trim();
        let target = cur;
        if (typed && typed !== cur) {
            let d = null;
            try {
                const r = await api.fetchApi("/uls/media/folders?path=" + encodeURIComponent(typed));
                if (r && r.ok) d = await r.json();
            } catch (e) { /* ignore */ }
            if (d && d.ok && d.path) { target = d.path; }
            else { showList(); curEl.textContent = "Cannot read this folder: " + typed; curEl.title = typed; return; }
        }
        if (target) { onPin(target); back.remove(); }
    };
    // yellow 📁 -> the native Windows folder dialog (PowerShell FolderBrowserDialog,
    // server-side via /uls/media/native_pick), the way ⬆ Upload pops a native dialog.
    // Shared by both pickers (main Choose folder + the Batch panel's Choose…).
    // v437 sent the pick through nav() + Choose; v651 reverts that for THIS path
    // only: the Windows dialog's own OK already confirms, a second Choose was
    // double-confirmation friction (field). List/recent/paste still end at Choose.
    const nativeBtn = box.querySelector(".ph-fp-native");
    nativeBtn.onclick = async () => {
        const old = nativeBtn.textContent;
        nativeBtn.textContent = "⏳"; nativeBtn.disabled = true;
        let d = null;
        try { const r = await api.fetchApi("/uls/media/native_pick"); d = r && await r.json(); }
        catch (e) { /* ignore */ }
        nativeBtn.textContent = old; nativeBtn.disabled = false;
        if (d && d.ok && d.path) { onPin(d.path, d.file || ""); back.remove(); return; }
        else if (d && d.reason) {
            curEl.textContent = "Native dialog unavailable here — use Recent, the list, or paste a path.";
            curEl.title = "";
        }
        // cancelled -> leave the picker open
    };

    // recent folders — one click jumps there
    const recentEl = box.querySelector(".ph-fp-recent");
    const recents = getRecentFolders();
    if (recents.length) {
        const lbl = document.createElement("div"); lbl.className = "ph-fp-recent-lbl"; lbl.textContent = "Recent:";
        recentEl.appendChild(lbl);
        // Cap at the 5 newest folders: five short rows always fit the recent
        // list's height, so it can never overflow into the current-path row below
        // (8 long paths used to exceed the 118px cap and crowd that row).
        for (const p of recents.slice(0, 5)) {
            const row = document.createElement("div"); row.className = "ph-fp-recent-row";
            row.textContent = p; row.title = p;
            row.onclick = () => nav(p);
            recentEl.appendChild(row);
        }
    } else {
        recentEl.style.display = "none";
    }

    // open collapsed: Recent + the browse buttons + the hint. The folder list
    // loads only on demand (⬆ up / ⬇ in / a Recent row / the address bar),
    // never auto-onto a possibly-leaf source folder.
}

// ── per-node UI controller ─────────────────────────────────────────────────
class MediaLoaderUI {
    constructor(node) {
        injectCSS();
        this.node = node;
        this._pop = null;

        // Hide the JS-managed widgets (media_ref + batch_config). We collapse the
        // height AND no-op the draw, so they never render regardless of how this
        // frontend version reads the type flag — the optional batch_config widget
        // kept drawing its JSON string with type="hidden" alone.
        const hideWidget = (w) => {
            if (!w) return w;
            w.type = "hidden";
            w.computeSize = () => [0, -4];
            w.draw = () => {};
            return w;
        };
        this._refWidget = hideWidget(node.widgets?.find((w) => w.name === "media_ref"));
        this._cfgWidget = hideWidget(node.widgets?.find((w) => w.name === "batch_config"));
        this._procWidget = hideWidget(node.widgets?.find((w) => w.name === "proc_config"));

        const root = document.createElement("div");
        root.className = "ph-media";
        root.innerHTML = `
          <div class="ph-media-bar">
            <button class="ph-media-btn ph-upload" title="Upload a file into the ComfyUI input folder. Drag &amp; drop works too: dragging a file over the node raises two zones — drop on the left to copy it into input, on the right to load it from where it already lies (that one needs the file's path).">⬆ Upload File</button>
            <button class="ph-media-btn ph-batch" title="Set up the batch: pick files (✓ circles / filter), then run them as ▦ Frames (one batch after the consistency check) or ▶ Processing (one file per run).">▦▶ Batch…</button>
            <button class="ph-media-btn ph-folder">📁 Choose Folder</button>
            <button class="ph-media-btn ph-browse">🔎 Browse Folder</button>
            <button class="ph-media-btn ph-batch-toggle" title="Turn the batch on/off. The mode set in ▦▶ Batch… decides how the checked files run: ▦ Frames = one batch, ▶ Processing = one file per run. Your checked files stay remembered either way.">Batch: OFF</button>
            <button class="ph-media-btn ph-audio-toggle" title="Pair an audio file with your image/video — shown in the Selection and carried on the AUDIO / video_audio outputs. Picking an audio tile arms this; it stays remembered when OFF.">♪ Audio: OFF</button>
            <button class="ph-media-btn ph-proc-runall" title="Process every file in the folder now — queues one run per file and stops on its own. Shown while Batch Processing is on." style="display:none;">▶ Run all</button>
            <button class="ph-media-btn ph-reread" title="Re-read this folder — picks up files added since the last look. Page and checkmarks stay.">⟳</button>
            <button class="ph-media-btn ph-refresh" title="Back to input folder">↻</button>
          </div>
          <div class="ph-batch-status" style="font:11px/1.3 system-ui,sans-serif;padding:2px 4px;color:#8a8a92;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"></div>
          <div class="ph-media-main">
            <div class="ph-media-busy"><div class="ph-busy-ring"></div><div class="ph-busy-label">Working…</div></div>
            <div class="ph-media-drophint">
              <div class="ph-dz ph-dz-copy" data-zone="copy">
                <div class="ph-dz-icon">&#11014;</div>
                <div class="ph-dz-title">Copy into input</div>
                <div class="ph-dz-sub">A copy lands in the ComfyUI input folder and loads. Your own folders stay untouched.</div>
              </div>
              <div class="ph-dz ph-dz-pin" data-zone="pin">
                <div class="ph-dz-icon">&#128204;</div>
                <div class="ph-dz-title">Load from where it is</div>
                <div class="ph-dz-sub ph-dz-pinsub">Selects the file and pins the folder it already lives in. Nothing is copied.</div>
              </div>
            </div>
            <div class="ph-media-grid"></div>
            <div class="ph-media-preview">
              <div class="ph-media-prev-lbl"><span>Selection</span><button class="ph-solo-toggle" title="Hide the tile grid — the Selection fills the whole node like a plain Load node (resize freely; sizes are remembered per mode). Click again to bring the tiles back.">⛶ Solo</button></div>
              <div class="ph-media-prev-media"></div>
              <div class="ph-media-prev-vtrim"></div>
              <div class="ph-media-prev-cap"></div>
              <div class="ph-media-prev-audio"></div>
            </div>
          </div>
          <div class="ph-media-foot">
            <div class="ph-media-pager"></div>
            <div class="ph-media-path"></div>
          </div>`;
        this.root = root;
        this.pathEl = root.querySelector(".ph-media-path");
        this.gridEl = root.querySelector(".ph-media-grid");
        this.pagerEl = root.querySelector(".ph-media-pager");
        this.previewMediaEl = root.querySelector(".ph-media-prev-media");
        this.previewCapEl = root.querySelector(".ph-media-prev-cap");
        this.audioPaneEl = root.querySelector(".ph-media-prev-audio");   // v458: collapsible audio companion pane
        this.videoTrimEl = root.querySelector(".ph-media-prev-vtrim");    // v476: video trim strip under the preview
        this.barEl = root.querySelector(".ph-media-bar");                 // v461: measured live for the height floor (it wraps with width)
        this.soloBtn = root.querySelector(".ph-solo-toggle");              // v624: Solo-Selection toggle
        this.soloBtn.onclick = () => this._toggleSolo();
        this._applyViewClass();                                            // v624: cloned nodes may already carry ph_media_view

        root.querySelector(".ph-folder").onclick = () =>
            openFolderPicker(this.folder, (p, file) => {
                this.setFolder(p); pushRecentFolder(p);
                if (file) this._dropSelect(file);   // v660: the dialog's pick loads immediately (drop parity)
            });
        root.querySelector(".ph-browse").onclick = () => this.openBrowseModal();
        root.querySelector(".ph-refresh").onclick = async () => {
            const input = await this._resolveInputPath();
            if (input) this.setFolder(input); else this.refreshGrid();
        };
        this.batchStatusEl = root.querySelector(".ph-batch-status");
        // v528: Ctrl+A / Ctrl+X / Ctrl+I edit the checked set — hover-scoped: they
        // only fire while the pointer is over THIS grid (capture phase so the
        // canvas shortcuts never see them there; everywhere else stays untouched).
        this._gridHover = false;
        this.gridEl.addEventListener("mouseenter", () => { this._gridHover = true; });
        this.gridEl.addEventListener("mouseleave", () => { this._gridHover = false; });
        this._selKeyHandler = (e) => {
            if (!this._gridHover || !e.ctrlKey || e.altKey || e.metaKey) return;
            const k = String(e.key || "").toLowerCase();
            if (k === "a") { e.preventDefault(); e.stopPropagation(); this._selAll(this.folder); }
            else if (k === "x") { e.preventDefault(); e.stopPropagation(); this._selNone(); }
            else if (k === "i") { e.preventDefault(); e.stopPropagation(); this._selInvert(this.folder); }
        };
        document.addEventListener("keydown", this._selKeyHandler, true);
        root.querySelector(".ph-reread").onclick = () => this.refreshGrid(true);
        // v655: window regains focus (render finished elsewhere, files copied in)
        // -> silent re-read. Bound handler, removed in _destroy (the v624 leak law).
        this._onWinFocus = () => { this._focusReread(); };
        window.addEventListener("focus", this._onWinFocus);
        root.querySelector(".ph-batch").onclick = () => this.openBatchPanel();
        this.batchToggleEl = root.querySelector(".ph-batch-toggle");
        this.batchToggleEl.onclick = () => this._toggleBatch();
        this.audioToggleEl = root.querySelector(".ph-audio-toggle");   // v458
        this.audioToggleEl.onclick = () => this._toggleAudio();
        this.procRunAllEl = root.querySelector(".ph-proc-runall");
        this.procRunAllEl.onclick = () => this._procRunAll();

        const fileInput = document.createElement("input");
        fileInput.type = "file"; fileInput.accept = "image/*,video/*"; fileInput.multiple = true;
        fileInput.style.display = "none";
        fileInput.onchange = () => { if (fileInput.files?.length) this.upload(fileInput.files); fileInput.value = ""; };
        root.appendChild(fileInput);
        root.querySelector(".ph-upload").onclick = () => fileInput.click();

        // v542: hideOnZoom:false -- DOM widgets are hidden by the frontend below the
        // low-quality zoom threshold; the 3D nodes (viewport/gizmo/studio light)
        // already opt out. Without it the whole browser UI vanishes when zooming out.
        node.addDOMWidget("ph_media_ui", "div", root, { serialize: false, hideOnZoom: false });
        // v644 drag & drop, v664 zones. A drag over the node raises two drop zones;
        // aiming at one IS the choice. Shift survives only as a silent alias for the
        // "load from where it is" zone, and any drop that misses both zones falls
        // back to that same rule — one mental model, no competing explanations.
        this.dropHintEl = root.querySelector(".ph-media-drophint");
        root.addEventListener("dragenter", (ev) => { ev.preventDefault(); this._dropHint(true, ev); });
        root.addEventListener("dragover", (ev) => {
            ev.preventDefault();
            if (ev.dataTransfer) ev.dataTransfer.dropEffect = ev.shiftKey ? "link" : "copy";
            this._dropHint(true, ev);
        });
        root.addEventListener("dragleave", (ev) => {
            // only when the pointer really left the node, not on a child boundary
            if (!ev.relatedTarget || !root.contains(ev.relatedTarget)) this._dropHint(false, ev);
        });
        root.addEventListener("drop", (ev) => {
            ev.preventDefault();
            this._dropHint(false, ev);
            this._onDrop(ev, this._intentFor("", ev.shiftKey));
        });
        // The zones themselves: hovering one lights it, dropping on one decides.
        (this.dropHintEl ? this.dropHintEl.querySelectorAll(".ph-dz") : []).forEach((z) => {
            const zone = z.dataset.zone;
            z.addEventListener("dragenter", (ev) => { ev.preventDefault(); this._zoneHot(zone); });
            z.addEventListener("dragover", (ev) => {
                ev.preventDefault(); ev.stopPropagation();
                if (ev.dataTransfer) ev.dataTransfer.dropEffect = zone === "pin" ? "link" : "copy";
                this._dropHint(true, ev);
                this._zoneHot(zone);
            });
            z.addEventListener("dragleave", () => this._zoneHot(""));
            z.addEventListener("drop", (ev) => {
                ev.preventDefault(); ev.stopPropagation();
                this._dropHint(false, ev);
                this._onDrop(ev, this._intentFor(zone, ev.shiftKey));
            });
        });
        // v463: ensure a FRESH node is wide enough for the multi-column browser
        // layout AND tall enough to clear the content floor — both grow-only.
        // Width must be ensured REGARDLESS of the computed height: adding widgets
        // can push ComfyUI's computed height past a threshold, and the old
        // `height < 360` gate then wrongly skipped the nudge, leaving a fresh node
        // narrow with the grid cramped to the left.
        // A loaded node's saved size is restored in configure() afterwards, so this
        // only sizes fresh nodes. The dynamic floor below perfects the height.
        const _w = Math.max(node.size?.[0] || 0, 600);
        const _h = Math.max(node.size?.[1] || 0, 520);
        node.setSize([_w, _h]);
        this._renderBatchStatus();
        this._syncAudioToggle();   // v458: reflect the persisted ♪ Audio state on the button

        // Recompute the tile grid AND refit the preview video whenever either box
        // changes size (node resize, selection-pane reflow, first layout).
        if (typeof ResizeObserver !== "undefined") {
            this._ro = new ResizeObserver(() => this._scheduleLayout());
            this._ro.observe(this.gridEl);
            this._ro.observe(this.previewMediaEl);
            if (this.barEl) this._ro.observe(this.barEl);   // v461: bar wraps with width -> re-floor when its row count changes
        }

        this.renderPath();
        this._renderPreview();
        if (this.folder) this.refreshGrid();
        else this._maybeDefaultToInput();
    }

    async _resolveInputPath() {
        if (this._inputPath) return this._inputPath;
        try {
            const r = await api.fetchApi("/uls/media/folders");
            const d = r && await r.json();
            const inp = ((d && d.folders) || []).find((f) => f.name === "input/");
            if (inp && inp.path) { this._inputPath = inp.path; return inp.path; }
        } catch (e) { /* ignore */ }
        return "";
    }

    _maybeDefaultToInput() {
        // Fresh node (no pin) -> default to the ComfyUI input dir so its media
        // shows as thumbnails right away. Deferred a tick so a workflow reload's
        // restore() (onConfigure) wins if it pins a folder.
        setTimeout(async () => {
            if (this.folder) { this.refreshGrid(); return; }
            const p = await this._resolveInputPath();
            if (p && !this.folder) this.setFolder(p);
        }, 0);
    }

    // ── Image-batch config (batch_config widget) + panel ────────────────────
    _readCfg() {
        try {
            const c = JSON.parse(this._cfgWidget?.value || "{}");
            return (c && typeof c === "object") ? c : {};
        } catch (e) { return {}; }
    }

    _writeCfg(cfg) {
        if (this._cfgWidget) this._cfgWidget.value = JSON.stringify(cfg);
        // Sequence and batch-processing are mutually exclusive — arming one disarms
        // the other. v528: a cfg that carries a mode is the unified world and owns
        // the state outright, so the legacy proc widget is kept disarmed too.
        if ((cfg.enabled || cfg.mode) && this._procWidget) {
            const p = this._readProcCfg();
            if (p.enabled) this._procWidget.value = JSON.stringify({ ...p, enabled: false });
        }
        this._renderBatchStatus();
        // Batch on -> the grid mirrors the batch: jump it to the source folder so the
        // selected frames show (marked). A different source navigates (fetch + mark);
        // same folder just re-renders to update the marks for the new selectors.
        if (cfg.enabled && cfg.source && !this._samePath(this.folder, cfg.source)) {
            this.setFolder(cfg.source);   // setFolder: renderPath + _renderPreview + refreshGrid
        } else {
            this._renderPreview();   // reflect batch on/off in the Selection column live
            if (this.folder && this._files) this.renderGrid();   // re-mark the grid for the new cfg
        }
        try { this.node.setDirtyCanvas?.(true, true); } catch (e) { /* ignore */ }
        try { app.graph?.setDirtyCanvas?.(true, true); } catch (e) { /* ignore */ }
    }

    // ── Batch-processing config (proc_config widget) ────────────────────────
    _readProcCfg() {
        try {
            const c = JSON.parse(this._procWidget?.value || "{}");
            return (c && typeof c === "object") ? c : {};
        } catch (e) { return {}; }
    }

    _writeProcCfg(cfg) {
        if (this._procWidget) this._procWidget.value = JSON.stringify(cfg);
        // Batch-processing and sequence are mutually exclusive — arming one disarms the other.
        if (cfg.enabled && this._cfgWidget) {
            const c = this._readCfg();
            if (c.enabled) this._cfgWidget.value = JSON.stringify({ ...c, enabled: false });
        }
        this._renderBatchStatus();
        // Proc on -> mirror the source folder in the grid so the queued files show.
        if (cfg.enabled && cfg.source && !this._samePath(this.folder, cfg.source)) {
            this.setFolder(cfg.source);
        } else {
            this._renderPreview();
            if (this.folder && this._files) this.renderGrid();
        }
        try { this.node.setDirtyCanvas?.(true, true); } catch (e) { /* ignore */ }
        try { app.graph?.setDirtyCanvas?.(true, true); } catch (e) { /* ignore */ }
    }

    // ── v528 checked-files selection — THE set both batch modes consume ─────
    // Stored inside batch_config as {mode:"explicit", names:[…]} (null = legacy
    // rule pipeline), so it serializes with the workflow and survives restarts
    // like every other setting. The ○ circle on each tile, Ctrl+A/X/I over the
    // grid and the dialog's Checks row all edit this one set.
    _selNames() {
        const c = this._readCfg();
        const sel = c.selection;
        if (Array.isArray(sel?.names)) return sel.names.filter((n) => typeof n === "string" && n);
        if (Array.isArray(sel)) return sel.filter((n) => typeof n === "string" && n);
        return [];
    }

    _writeSel(names, adoptFolder) {
        const c = this._readCfg();
        // Checking a file in a folder that isn't the batch source (or with no
        // source yet) adopts THAT folder as the source — same philosophy as the
        // Batch toggle adopting the current folder on first arm.
        if (adoptFolder && (!c.source || !this._samePath(c.source, adoptFolder))) {
            c.source = adoptFolder;
        }
        const clean = (names || []).filter((n) => typeof n === "string" && n);
        c.selection = clean.length ? { mode: "explicit", names: clean } : null;
        this._writeCfg(c);   // re-renders status, toggle, grid marks, preview
    }

    _toggleSelName(name) {
        // With no explicit set yet but rule-derived marks showing (an armed Frames
        // batch), the first circle click materializes those marks into the set —
        // so unchecking one of 50 marked files leaves 49, never a set of one.
        let cur = this._selNames();
        if (!cur.length && this._gridMarks) cur = Object.keys(this._gridMarks);
        const has = cur.includes(name);
        this._writeSel(has ? cur.filter((n) => n !== name) : [...cur, name], this.folder);
    }

    async _selAll(folder) {
        const target = folder || this.folder;
        if (!target) return;
        const files = (this._samePath(this._filesFolder, target) && this._files)
            ? this._files : await this._listFolder(target);
        if (!files) return;
        this._writeSel(files.filter((f) => f.kind !== "audio").map((f) => f.name), target);
    }

    _selNone() { this._writeSel([], null); }

    async _selInvert(folder) {
        const target = folder || this.folder;
        if (!target) return;
        const files = (this._samePath(this._filesFolder, target) && this._files)
            ? this._files : await this._listFolder(target);
        if (!files) return;
        const cur = new Set(this._selNames());
        this._writeSel(files.filter((f) => f.kind !== "audio" && !cur.has(f.name))
                            .map((f) => f.name), target);
    }

    // Soft-highlight the dialog filter's live matches on the grid tiles (only
    // when the grid is showing the folder the matches belong to). null clears.
    _highlightGridMatches(matchSet, folder) {
        if (!this.gridEl) return;
        const active = !!(matchSet && folder && this._samePath(this.folder, folder));
        this.gridEl.querySelectorAll(".ph-media-tile").forEach((t) => {
            t.classList.toggle("ph-match", active && matchSet.has(t.dataset.file));
        });
    }

    _renderBatchStatus() {
        // v528 unified: batch_config carries mode ("frames" | "proc") + the checked
        // selection. A mode-less cfg with a legacy enabled proc_config still shows
        // as Proc (mirrors the Python precedence), so old workflows read correctly.
        const c = this._readCfg();
        const p = this._readProcCfg();
        const mode = (c.mode === "proc" || c.mode === "frames") ? c.mode : "";
        const legacyProc = !mode && !!(p.enabled && p.source);
        const procOn = (mode === "proc" && !!(c.enabled && c.source)) || legacyProc;
        const framesOn = !procOn && ((mode === "frames" || !mode) && !!(c.enabled && c.source));
        const nSel = Array.isArray(c.selection?.names) ? c.selection.names.length
                   : (Array.isArray(c.selection) ? c.selection.length : 0);
        const selTxt = nSel ? ` · ${nSel} checked` : "";
        // ▶ Run all lives on the node toolbar and is only useful while proc is armed.
        if (this.procRunAllEl) this.procRunAllEl.style.display = procOn ? "" : "none";
        const el = this.batchStatusEl;
        if (el) {
            if (procOn) {
                const src = legacyProc ? p : c;
                const base = String(src.source).replace(/[\\/]+$/, "").split(/[\\/]/).pop() || src.source;
                const flt = (src.name_filter && src.name_filter !== "*") ? ` · "${src.name_filter}"` : "";
                el.textContent = `▶ Batch ON (Separate files): ${base} · ${src.sort_mode || "name (natural)"}${flt}${selTxt} — each file its own job (Auto-Queue or ▶ Run all)`;
                el.style.color = "#7fb3d1";
            } else if (framesOn) {
                const nth = (c.every_nth && c.every_nth > 1) ? ` ×${c.every_nth}` : "";
                el.textContent = `▦ Batch ON (Video frames): ${c.sort_mode || "name (natural)"} · "${c.name_filter || "*"}"${nth}${selTxt} · ${c.resize_method || "none (strict)"}`;
                el.style.color = "#7fd17f";
            } else {
                // Single-file: lead with a 🛈 info affordance + the selection keys, so the
                // check mechanism is documented right where you use it. Static text only.
                el.innerHTML = '<span class="ph-batch-info" style="cursor:help" title="Single-file: the node emits the one clicked file. ▦▶ Batch… picks files and a mode: ▦ Frames = whole selection as one batch, ▶ Processing = one file per run. Check files via the ○ circle on each tile; Ctrl+A checks all, Ctrl+X none, Ctrl+I inverts (pointer over the grid). Your checked files are remembered.">🛈</span> Single-file — click a tile to load it · ○ circle checks it for the batch · Ctrl+A all · Ctrl+X none · Ctrl+I invert' + (nSel ? ` · <b style="color:#7fd17f">${nSel} checked</b>` : "");
                el.style.color = "#8a8a92";
            }
        }
        // ONE node-face toggle: on/off + which mode the checked files will run as.
        const bt = this.batchToggleEl;
        if (bt) {
            const on = procOn || framesOn;
            bt.classList.toggle("on", on);
            bt.textContent = procOn ? "▶ Batch: ON" : (framesOn ? "▦ Batch: ON" : "Batch: OFF");
            bt.title = on
                ? (procOn ? "Batch is ON in ▶ Processing mode — each run emits the next checked file. Click to turn it off (checks stay remembered)."
                          : "Batch is ON in ▦ Frames mode — the node emits the checked files as one batch. Click to turn it off (checks stay remembered).")
                : "Batch is OFF. Click to turn it on with the current settings from ▦▶ Batch… (checked files, mode, sort).";
        }
    }

    // ▶ Run all (node toolbar, shown while proc is armed): sweep the whole
    // selection/folder from the node. v528: reads the UNIFIED cfg when it carries
    // mode 'proc', else the legacy proc_config. Re-homes the cursor to "Start at",
    // resolves the exact run count — the checked set counts locally (the set is
    // the truth), the rule pipeline asks the backend (/uls/media/proc_count, the
    // same resolver load() uses) — then queues exactly that many runs. proc mode
    // is always-dirty (IS_CHANGED -> time.time()), so each queued run advances one
    // file and the sweep stops on its own. Queue call mirrors ph_viewport3d.js.
    async _procRunAll() {
        const c = this._readCfg();
        const unified = c.mode === "proc";
        const p = unified ? c : this._readProcCfg();
        if (!p.enabled || !p.source) return;   // button is only shown when armed; guard anyway
        if (!app || typeof app.queuePrompt !== "function") {
            alert("Run all needs ComfyUI's queue API, which isn't available here. Press Run instead.");
            return;
        }
        const selNames = unified
            ? (Array.isArray(c.selection?.names) ? c.selection.names
               : (Array.isArray(c.selection) ? c.selection : null))
            : null;
        let total = 0;
        if (selNames && selNames.length) {
            // intersect the checked set with what actually exists (vanished files drop)
            const files = (this._samePath(this._filesFolder, p.source) && this._files)
                ? this._files : await this._listFolder(p.source);
            if (!files) { alert("Run all: the source folder is unreadable."); return; }
            const existing = new Set(files.map((f) => f.name));
            total = selNames.filter((n) => existing.has(n)).length;
        } else {
            try {
                const r = await api.fetchApi(
                    "/uls/media/proc_count?folder=" + encodeURIComponent(p.source)
                    + "&sort=" + encodeURIComponent(p.sort_mode || "name (natural)")
                    + "&filter=" + encodeURIComponent(p.name_filter || "*"));
                const d = r && await r.json();
                if (!d || !d.ok) {
                    alert("Run all: could not count files: " + ((d && d.error) || "unknown error"));
                    return;
                }
                total = parseInt(d.total, 10) || 0;
            } catch (e) { alert("Run all: could not reach the server: " + e); return; }
        }
        if (total < 1) { alert("Run all: no matching files in the source folder."); return; }
        const armed = { ...p, reset_seq: (parseInt(p.reset_seq, 10) || 0) + 1 };   // re-home to Start-at
        if (unified) this._writeCfg(armed); else this._writeProcCfg(armed);        // arm BEFORE queuing
        try {
            app.queuePrompt(0, total);
            if (this.batchStatusEl) {
                this.batchStatusEl.textContent =
                    `\u25B6 Batch (Processing): queued ${total} run${total === 1 ? "" : "s"} \u2014 sweeping\u2026`;
                this.batchStatusEl.style.color = "#7fb3d1";
            }
            console.log(`[PLS] MediaLoader: Run all queued ${total} runs for ${p.source}`);
        } catch (e) { alert("Run all: queuePrompt failed: " + e); }
    }

    // Flip the node's ONE batch toggle. State lives in batch_config (the serialized
    // hidden widget) so it survives reload like every other setting. v528 unified:
    // the mode set in ▦▶ Batch… decides how the checked files run; turning on with
    // no batch ever defined adopts the CURRENT folder as source with sane Frames
    // defaults, so "Batch on" is immediately meaningful. Arming in Processing mode
    // re-homes the cursor (reset_seq bump), mirroring the old ▶ Proc toggle.
    _toggleBatch() {
        const c = this._readCfg();
        if (!c.enabled) {
            if (!c.source) {
                if (!this.folder) {
                    alert("Pick a folder first (Choose folder / Upload), then turn Batch on.");
                    return;
                }
                c.source = this.folder;
            }
            c.mode = (c.mode === "proc" || c.mode === "frames") ? c.mode : "frames";
            c.sort_mode = c.sort_mode || "name (natural)";
            c.name_filter = c.name_filter || "*";
            c.every_nth = c.every_nth || 1;
            c.resize_method = c.resize_method || "none (strict)";
            if (c.mode === "proc") {
                c.start_at = parseInt(c.start_at, 10) || 0;
                c.reset_seq = (parseInt(c.reset_seq, 10) || 0) + 1;   // re-home on arm
            }
            c.enabled = true;
        } else {
            c.enabled = false;
        }
        this._writeCfg(c);
    }

    openProcPanel() {
        // v528: the two batch dialogs are ONE — this alias keeps old call sites
        // working and simply opens the unified panel preset to Processing mode.
        return this.openBatchPanel("proc");
    }

    openBatchPanel(presetMode) {
        const cur = this._readCfg();
        // v528 migration: a mode-less cfg with a legacy enabled proc_config opens
        // preset to Processing with its values — first Apply materializes them into
        // the unified cfg (proc_config stays behind as an inert legacy mirror).
        const legacy = this._readProcCfg();
        const legacyProc = !cur.mode && !!legacy.enabled;
        let mode = presetMode === "proc" || presetMode === "frames" ? presetMode
                 : (cur.mode === "proc" || cur.mode === "frames") ? cur.mode
                 : (legacyProc ? "proc" : "frames");
        let source = cur.source || (legacyProc ? legacy.source : "") || this.folder || "";
        const overlay = document.createElement("div");
        overlay.className = "ph-batch-overlay";
        overlay.innerHTML = `
          <style>
          .ph-batch-overlay{position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:99999;display:flex;align-items:flex-start;justify-content:center;padding:24px 0;box-sizing:border-box;}
          .ph-batch-card{background:#1f1f22;color:#e6e6e6;border:1px solid #3a3a40;border-radius:10px;padding:16px 18px;width:min(580px,92vw);max-height:calc(100vh - 48px);overflow:auto;font:13px/1.4 system-ui,sans-serif;box-shadow:0 12px 40px rgba(0,0,0,.5);}
          .ph-batch-title{font-weight:600;margin-bottom:12px;font-size:14px;}
          .ph-batch-row{display:flex;align-items:center;gap:10px;margin:8px 0;}
          .ph-batch-row>label{flex:0 0 96px;color:#a9a9b2;}
          .ph-batch-row select,.ph-batch-row input[type=text],.ph-batch-row input[type=number]{flex:1;background:#141416;color:#e6e6e6;border:1px solid #3a3a40;border-radius:6px;padding:6px 8px;}
          .ph-batch-src{flex:1;display:flex;align-items:center;gap:8px;min-width:0;}
          .ph-batch-srcpath{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#cfcfd6;background:#141416;border:1px solid #3a3a40;border-radius:6px;padding:6px 8px;}
          .ph-batch-enable{display:flex;align-items:center;gap:8px;margin:12px 0 4px;color:#e6e6e6;}
          .ph-batch-note{color:#8a8a92;font-size:12px;margin:6px 0 12px;}
          .ph-batch-actions{display:flex;justify-content:flex-end;gap:8px;margin-top:6px;}
          .ph-batch-helpbtn{margin-right:auto;}
          .ph-batch-help{display:none;margin:10px 0;padding:10px 12px;border:1px solid #2f2f35;border-radius:8px;background:#191919;color:#bfc3cb;font-size:12px;line-height:1.5;}
          .ph-batch-help b{color:#e6e6e6;}
          .ph-batch-help ul{margin:6px 0;padding-left:18px;}
          .ph-batch-help li{margin:3px 0;}
          .ph-batch-seq{margin:10px 0 12px;padding:10px;border:1px solid #2f2f35;border-radius:8px;background:#191919;}
          .ph-batch-seq-hd{color:#a9a9b2;font-size:11.5px;margin-bottom:8px;line-height:1.35;}
          .ph-batch-seq-hd code{background:#26262b;color:#cdd6cd;padding:0 4px;border-radius:4px;}
          .ph-batch-seq-pick,.ph-batch-seq-new{display:flex;gap:8px;align-items:center;margin-bottom:6px;}
          .ph-batch-seq-pick .ph-seq-select,.ph-batch-seq-new .ph-seq-name{flex:1 1 auto;min-width:0;background:#202024;color:#d6d6dc;border:1px solid #34343a;border-radius:6px;padding:4px 6px;font-size:12px;}
          .ph-batch-seq-status{margin-top:4px;font-size:11px;color:#8a8a92;min-height:14px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
          .ph-seg{flex:1;display:flex;}
          .ph-seg-opt{flex:1;display:flex;align-items:center;justify-content:center;cursor:pointer;padding:5px 8px;border:1px solid #34343a;background:#202024;color:#a9a9b2;font-size:12px;user-select:none;}
          .ph-seg-opt:first-child{border-radius:6px 0 0 6px;}
          .ph-seg-opt:last-child{border-radius:0 6px 6px 0;border-left:none;}
          .ph-seg-opt input{display:none;}
          .ph-seg-opt.on{background:#2f5d2f;border-color:#4f8f4f;color:#dfffdf;}
          .ph-q{display:inline-flex;align-items:center;justify-content:center;width:14px;height:14px;border-radius:50%;border:1px solid #4a4a52;color:#8a8a92;font-size:9px;font-style:normal;cursor:help;margin-left:5px;}
          .ph-q:hover{color:#d6d6dc;border-color:#8a8a92;}
          .ph-check-lbl{flex:1;display:flex;gap:8px;align-items:center;cursor:pointer;color:#cfcfd6;font-size:12px;}
          .ph-sel-btns{flex:1;display:flex;gap:8px;align-items:center;flex-wrap:wrap;}
          .ph-batch-count{flex:none;min-width:112px;text-align:right;color:#8a8a92;font-size:11.5px;}
          .ph-batch-startname{flex:none;color:#8a8a92;font-size:11.5px;}
          .ph-sel-count{color:#7fd17f;font-size:11.5px;}
          .ph-adv{flex:1;}
          .ph-adv-toggle,.ph-seq-head{background:none;border:none;color:#8a8a92;font-size:11.5px;cursor:pointer;padding:0;text-align:left;}
          .ph-adv-toggle:hover,.ph-seq-head:hover{color:#d6d6dc;}
          .ph-adv-body{display:none;flex-wrap:wrap;gap:6px;margin-top:6px;}
          .ph-adv-body.on{display:flex;}
          .ph-ex{background:#202024;border:1px solid #34343a;border-radius:6px;color:#a9a9b2;font-size:11px;padding:3px 7px;cursor:pointer;font-family:inherit;}
          .ph-ex:hover{background:#26262b;color:#e6e6e6;border-color:#4a4a52;}
          .ph-batch-live{margin:10px 0 4px;padding:8px 10px;border:1px solid #2f4f2f;border-radius:8px;background:#16210f;color:#9fd39f;font-size:12px;line-height:1.4;}
          .ph-batch-live.warn{border-color:#5a4520;background:#221a0f;color:#e0a85a;}
          .ph-seq-body{display:none;margin-top:8px;}
          .ph-seq-body.on{display:block;}
          .ph-mode-info{margin:0 0 12px;color:#a9a9b2;font-size:12px;line-height:1.5;}
          .ph-rub{border:1px solid #2f4f2f;border-radius:8px;padding:8px 10px 4px;margin:10px 0;}
          .ph-rub.ph-mrow-proc{border-color:#2f4160;}
          .ph-rub-hd{font-size:11.5px;font-weight:600;color:#7fd17f;margin-bottom:6px;}
          .ph-rub.ph-mrow-proc .ph-rub-hd{color:#7fa8e0;}
          .ph-rub .ph-batch-row{margin-bottom:6px;}
          .ph-rub .ph-batch-seq{margin:6px 0 4px;}
          .ph-batch-live.proc{border-color:#2f4160;background:#0f1621;color:#9fbfe0;}
          </style>
          <div class="ph-batch-card">
            <div class="ph-batch-title">Batch</div>
            <div class="ph-batch-row">
              <label>Mode</label>
              <div class="ph-seg">
                <label class="ph-seg-opt ph-seg-frames" title="The checked images are the frames of ONE video.">
                  <input type="radio" name="ph-bmode" class="ph-mode-frames" value="frames"><span>\u25a6 Video frames</span></label>
                <label class="ph-seg-opt ph-seg-proc" title="Each checked file is its own job \u2014 one per run.">
                  <input type="radio" name="ph-bmode" class="ph-mode-proc" value="proc"><span>\u25b6 Separate files</span></label>
              </div>
            </div>
            <div class="ph-mode-info"></div>
            <div class="ph-batch-help">
              <b>Pick WHICH files run, then HOW they run.</b>
              <ul>
                <li><b>Which</b> \u2014 check files with the <b>\u25cb circle</b> on a tile. Over the grid: <b>Ctrl+A</b> all, <b>Ctrl+X</b> none, <b>Ctrl+I</b> invert. Checks are remembered in the workflow and beat the name filter.</li>
                <li><b>How</b> \u2014 <b>\u25a6 Video frames</b>: the checked images are the frames of ONE video and load as a single batch (after a size check). <b>\u25b6 Separate files</b>: each checked file (image or video) is its own job, fed in one per run; Auto-Queue or \u25b6 Run all sweeps them.</li>
                <li><b>Name contains</b> \u2014 plain text matches any name holding it. The <b>Examples</b> row shows the extra syntax (wildcards, either/or, exclude, regex); clicking an example fills the field.</li>
                <li><b>Saved sequences</b> \u2014 a renumbered copy of the batch under <code>output\\PLS_sequences</code>. Your originals are never touched.</li>
              </ul>
              <b>Apply</b> arms the batch; the <b>Batch</b> button on the node turns it on and off later (the checks stay).
            </div>
            <div class="ph-batch-row">
              <label>Folder</label>
              <div class="ph-batch-src">
                <span class="ph-batch-srcpath"></span>
                <button class="ph-media-btn ph-batch-open" title="Open this folder in your file manager">\ud83d\udcc2 Open</button>
                <button class="ph-media-btn ph-batch-choose">\ud83d\udcc1 Choose</button>
              </div>
            </div>
            <div class="ph-batch-row">
              <label>Sort by <i class="ph-q" title="What decides the order the files go in.">?</i></label>
              <select class="ph-batch-sort">
                <option value="name (natural)" title="Stacks by the numbers in the file name: img2 comes before img10. Padded names (0001_, 0002_) work either way.">Number</option>
                <option value="name (literal)" title="Pure character order \u2014 digits are just text, so img10 lands before img2.">A\u2013Z</option>
                <option value="mtime (oldest first)" title="Oldest first, by the file's last change.">Date modified</option>
                <option value="created" title="Oldest first, by when the file was made.">Date created</option>
              </select>
            </div>
            <div class="ph-batch-row">
              <label>Name contains <i class="ph-q" title="Type part of a file name to narrow the list down. Leave it at * for all files. See Examples for more ways to filter.">?</i></label>
              <input class="ph-batch-filter" type="text" spellcheck="false" placeholder="* \u2014 all files">
              <span class="ph-batch-count"></span>
            </div>
            <div class="ph-batch-row">
              <label></label>
              <div class="ph-adv">
                <button type="button" class="ph-adv-toggle">Examples \u25be</button>
                <div class="ph-adv-body">
                  <button type="button" class="ph-ex" data-ex="PH">PH \u2014 name holds PH</button>
                  <button type="button" class="ph-ex" data-ex="PH*">PH* \u2014 starts with PH</button>
                  <button type="button" class="ph-ex" data-ex="*.png">*.png \u2014 only PNGs</button>
                  <button type="button" class="ph-ex" data-ex="PH*, IL_*">PH*, IL_* \u2014 either one</button>
                  <button type="button" class="ph-ex" data-ex="!*REMIX*">!*REMIX* \u2014 all except REMIX</button>
                  <button type="button" class="ph-ex" data-ex="re:^\\d{4}">re:^\\d{4} \u2014 regex</button>
                </div>
              </div>
            </div>
            <div class="ph-rub ph-mrow-frames">
              <div class="ph-rub-hd">\u25a6 Video frames</div>
            <div class="ph-batch-row">
              <label>Use <i class="ph-q" title="Skip files to make a shorter batch: every 2nd file halves it, every 3rd thirds it.">?</i></label>
              <select class="ph-batch-nth">
                <option value="1">every file</option>
                <option value="2">every 2nd file</option>
                <option value="3">every 3rd file</option>
                <option value="4">every 4th file</option>
                <option value="5">every 5th file</option>
                <option value="custom">every Nth file\u2026</option>
              </select>
              <input class="ph-batch-nth-custom" type="number" min="1" step="1" value="1" style="display:none;max-width:76px;">
            </div>
            <div class="ph-batch-row">
              <label>If sizes differ <i class="ph-q" title="What to do when the frames are not all the same size.">?</i></label>
              <select class="ph-batch-resize">
                <option value="none (strict)" title="Refuses to run if the frames differ in size \u2014 and names the odd one out.">Stop</option>
                <option value="resize to first" title="Every frame is stretched to the first one's size (can squash).">Stretch</option>
                <option value="pad to first" title="Bars are added so every frame matches the first \u2014 nothing gets squashed.">Letterbox</option>
                <option value="center crop to first" title="The middle of every frame is cropped to the first one's size.">Crop center</option>
              </select>
            </div>
            <div class="ph-batch-seq">
              <button type="button" class="ph-seq-head">Saved sequences \u25be <i class="ph-q" title="A renumbered copy of the batch, stored under output\\PLS_sequences. Your original files are never changed.">?</i></button>
              <div class="ph-seq-body">
                <div class="ph-batch-seq-pick">
                  <select class="ph-seq-select" title="Saved sequences \u2014 choosing one loads it as the active batch"></select>
                  <button class="ph-media-btn ph-seq-use" title="Load the selected sequence as the batch and close">Use</button>
                  <button class="ph-media-btn ph-seq-delete" title="Delete the selected sequence folder">\ud83d\uddd1 Delete</button>
                </div>
                <div class="ph-batch-seq-new">
                  <input class="ph-seq-name" type="text" placeholder="new sequence name" title="Name for a new sequence folder" />
                  <button class="ph-media-btn ph-seq-build" title="Build the current files into a named sequence and load it">\u25a6 Build &amp; Use</button>
                </div>
                <div class="ph-batch-seq-status"></div>
              </div>
            </div>
            </div>
            <div class="ph-rub ph-mrow-proc">
              <div class="ph-rub-hd">\u25b6 Separate files</div>
            <div class="ph-batch-row">
              <label>Begin at file <i class="ph-q" title="Which file of the running set goes first. Apply sends the cursor back here.">?</i></label>
              <input class="ph-batch-start" type="number" min="1" step="1" value="1" style="max-width:76px;">
              <span class="ph-batch-startname"></span>
            </div>
            <div class="ph-batch-row">
              <label></label>
              <label class="ph-check-lbl" title="On: after the last file it starts over at the first. Off: it stays on the last file.">
                <input type="checkbox" class="ph-batch-wrap" style="flex:none;"> <span class="ph-wrap-lbl">Start over after the last file</span></label>
            </div>
            </div>
            <div class="ph-batch-row">
              <label>Selection <i class="ph-q" title="The checked files are what runs. Click the \u25cb circle on a tile to check it, or use these buttons (same as Ctrl+A / Ctrl+X / Ctrl+I over the grid).">?</i></label>
              <div class="ph-sel-btns">
                <button class="ph-media-btn ph-sel-all">Select all</button>
                <button class="ph-media-btn ph-sel-none">Clear</button>
                <button class="ph-media-btn ph-sel-invert">Invert</button>
                <button class="ph-media-btn ph-sel-fromfilter" title="Check exactly the files the name filter matches">Select the matches</button>
                <span class="ph-sel-count"></span>
              </div>
            </div>
            <div class="ph-batch-live"></div>
            <div class="ph-batch-actions">
              <button class="ph-media-btn ph-batch-helpbtn" title="What these settings do">🛈 Help</button>
              <button class="ph-media-btn ph-batch-cancel">Cancel</button>
              <button class="ph-media-btn ph-batch-apply">Apply</button>
            </div>
          </div>`;
        document.body.appendChild(overlay);
        const $ = (s) => overlay.querySelector(s);
        const srcEl = $(".ph-batch-srcpath");
        const renderSrc = () => { srcEl.textContent = source || "(none chosen)"; };
        renderSrc();
        const seed = legacyProc ? legacy : cur;               // legacy proc prefills once
        $(".ph-batch-sort").value = seed.sort_mode || "name (natural)";
        $(".ph-batch-filter").value = seed.name_filter || "*";
        // v542: "Every Nth" is a plain-language dropdown; values above the listed
        // ones fall back to a custom number field. Wire value stays an INT.
        const nthSel = $(".ph-batch-nth"), nthCustom = $(".ph-batch-nth-custom");
        const setNth = (n) => {
            n = Math.max(1, parseInt(n, 10) || 1);
            const known = [...nthSel.options].some((o) => o.value === String(n));
            nthSel.value = known ? String(n) : "custom";
            nthCustom.value = String(n);
            nthCustom.style.display = known ? "none" : "";
        };
        const getNth = () => (nthSel.value === "custom"
            ? Math.max(1, parseInt(nthCustom.value, 10) || 1)
            : Math.max(1, parseInt(nthSel.value, 10) || 1));
        setNth(cur.every_nth || 1);
        $(".ph-batch-resize").value = cur.resize_method || "none (strict)";
        $(".ph-batch-start").value = String((parseInt(seed.start_at, 10) || 0) + 1);
        $(".ph-batch-wrap").checked = !!seed.wrap;
        ($(mode === "proc" ? ".ph-mode-proc" : ".ph-mode-frames")).checked = true;

        // v542: the mode SWITCHES the form instead of greying half of it out.
        // Dead rows are gone, not dimmed — that alone removes half of the
        // perceived complexity. Values stay readable (never disabled), so Apply
        // still collects both branches exactly as before.
        const applyModeSwitch = () => {
            mode = $(".ph-mode-proc").checked ? "proc" : "frames";
            overlay.querySelectorAll(".ph-mrow-frames").forEach((row) => {
                row.style.display = mode === "proc" ? "none" : "";
            });
            overlay.querySelectorAll(".ph-mrow-proc").forEach((row) => {
                row.style.display = mode === "frames" ? "none" : "";
            });
            $(".ph-seg-frames").classList.toggle("on", mode === "frames");
            $(".ph-seg-proc").classList.toggle("on", mode === "proc");
            // v543: ONE explanatory line per mode -- the two rubrics are a purpose
            // split (build ONE video / push MANY files), and that has to be said
            // once. Everything else stays tooltip-only.
            $(".ph-mode-info").textContent = mode === "proc"
                ? "Each checked file (image or video) is its own job. One file is fed into the graph per run \u2014 Auto-Queue or \u25b6 Run all works through the whole set."
                : "The checked images are the frames of ONE video. They load together as a single batch \u2014 the node's preview already plays them as a clip.";
            $(".ph-batch-live").classList.toggle("proc", mode === "proc");
            updateLive();
        };
        $(".ph-mode-frames").onchange = applyModeSwitch;
        $(".ph-mode-proc").onchange = applyModeSwitch;

        // Live matcher feedback: the counter shows what the filter grabs in the
        // source folder, and the grid soft-highlights the matches when it is
        // showing that folder. The checked-set counter updates alongside.
        let lastList = null;

        // v542: one live sentence replaces the prose. It states the OUTCOME --
        // and makes the "checks beat the filter" rule visible instead of
        // explaining it: the running set IS the checked set when one exists.
        const ORD = (n) => (n === 2 ? "2nd" : n === 3 ? "3rd" : n + "th");
        const updateLive = () => {
            const live = $(".ph-batch-live");
            if (!live) return;
            const nameEl = $(".ph-batch-startname");
            if (nameEl) nameEl.textContent = "";
            if (!source) {
                live.classList.add("warn");
                live.textContent = "Choose a folder to see what will run.";
                return;
            }
            const names = lastList ? lastList.names : [];
            const total = names.length;
            const checked = this._selNames();
            const matches = this._matchNames(names, $(".ph-batch-filter").value || "*");
            const running = checked.length ? checked : matches;   // checks beat the filter
            const via = checked.length ? "checked" : "matching";
            const n = running.length;
            live.classList.toggle("warn", n === 0);
            if (!n) { live.textContent = "Nothing selected -- 0 of " + total + " files would run."; return; }
            if ($(".ph-mode-proc").checked) {
                const start = Math.min(Math.max(1, parseInt($(".ph-batch-start").value, 10) || 1), n);
                const tail = $(".ph-batch-wrap").checked ? "then starts over" : "then stops";
                // only name the first file when the order is name-based (time
                // orders are resolved on the backend -- do not fake it here)
                let firstTxt = "";
                if (String($(".ph-batch-sort").value).startsWith("name")) {
                    const ordered = names.filter((x) => running.includes(x));
                    const first = ordered[start - 1];
                    if (first) {
                        firstTxt = " \u00b7 starts at " + first;
                        if (nameEl) nameEl.textContent = "\u25b8 " + first;
                    }
                }
                live.textContent = "\u25b6 " + n + " of " + total + " files (" + via + ") \u2192 "
                    + (n - start + 1) + " separate jobs, one per run" + firstTxt + " \u00b7 " + tail + ".";
            } else {
                const nth = getNth();
                const frames = nth > 1 ? Math.ceil(n / nth) : n;
                const nthTxt = nth > 1 ? " \u00b7 every " + ORD(nth) + " file" : "";
                const rz = $(".ph-batch-resize").value === "none (strict)"
                    ? "sizes must match" : "sizes get fixed";
                live.textContent = "\u25a6 " + n + " of " + total + " files (" + via + ") \u2192 ONE video batch of "
                    + frames + " frame" + (frames === 1 ? "" : "s") + nthTxt + " \u00b7 " + rz + ".";
            }
        };

        const refreshCounts = async () => {
            const cntEl = $(".ph-batch-count"), selEl = $(".ph-sel-count");
            const selN = this._selNames().length;
            if (selEl) selEl.textContent = selN ? `${selN} checked` : "";
            if (!source) { if (cntEl) cntEl.textContent = ""; return; }
            if (!lastList || !this._samePath(lastList.folder, source)) {
                const files = (this._samePath(this._filesFolder, source) && this._files)
                    ? this._files : await this._listFolder(source);
                lastList = { folder: source, names: (files || []).map((f) => f.name) };
            }
            const expr = $(".ph-batch-filter").value || "*";
            const hits = this._matchNames(lastList.names, expr);
            if (cntEl) cntEl.textContent = `${hits.length} of ${lastList.names.length} match`;
            this._highlightGridMatches(new Set(hits), source);
            updateLive();
        };
        $(".ph-batch-filter").addEventListener("input", refreshCounts);
        for (const sel of [".ph-batch-sort", ".ph-batch-nth", ".ph-batch-nth-custom",
                           ".ph-batch-resize", ".ph-batch-start", ".ph-batch-wrap"]) {
            const el = $(sel);
            if (el) el.addEventListener("change", () => { if (sel === ".ph-batch-nth") setNth(getNth()); updateLive(); });
            if (el && el.type === "number") el.addEventListener("input", updateLive);
        }
        // Clickable filter examples: the syntax is learned by trying it, not by
        // reading a sentence about it.
        const advBody = $(".ph-adv-body");
        $(".ph-adv-toggle").onclick = () => {
            const on = advBody.classList.toggle("on");
            $(".ph-adv-toggle").textContent = on ? "Examples \u25b4" : "Examples \u25be";
        };
        overlay.querySelectorAll(".ph-ex").forEach((b) => {
            b.onclick = () => { $(".ph-batch-filter").value = b.dataset.ex; refreshCounts(); };
        });
        const seqBody = $(".ph-seq-body");
        $(".ph-seq-head").onclick = () => {
            const on = seqBody.classList.toggle("on");
            $(".ph-seq-head").innerHTML = (on ? "Saved sequences \u25b4" : "Saved sequences \u25be")
                + ' <i class="ph-q" title="A renumbered copy of the batch, stored under output\\PLS_sequences. Your original files are never changed.">?</i>';
        };

        // Checks editing from the dialog (mirrors the grid keys).
        $(".ph-sel-all").onclick = async () => { await this._selAll(source); refreshCounts(); };
        $(".ph-sel-none").onclick = () => { this._selNone(); refreshCounts(); };
        $(".ph-sel-invert").onclick = async () => { await this._selInvert(source); refreshCounts(); };
        $(".ph-sel-fromfilter").onclick = async () => {
            if (!source) { alert("Choose a source folder first."); return; }
            if (!lastList || !this._samePath(lastList.folder, source)) await refreshCounts();
            const hits = this._matchNames(lastList ? lastList.names : [], $(".ph-batch-filter").value || "*");
            this._writeSel(hits, source);
            refreshCounts();
        };
        applyModeSwitch();
        refreshCounts();

        const close = () => { this._highlightGridMatches(null); overlay.remove(); };
        // v663 (B-04): switching the source used to leave the counters — and the
        // node's status line — on the OLD folder: "100 of 100 match / 7 checked"
        // while the new folder held 13 files and no checked tile at all. The
        // cached listing is dropped and the checked set is intersected with what
        // the new folder actually contains, so every number on screen belongs to
        // the folder named above it.
        $(".ph-batch-choose").onclick = () =>
            openFolderPicker(source, async (p) => {
                source = p;
                renderSrc();
                pushRecentFolder(p);
                lastList = null;
                const keep = this._selNames();
                if (keep.length) {
                    const files = await this._listFolder(p);
                    const here = new Set((files || []).map((f) => f.name));
                    const kept = keep.filter((n) => here.has(n));
                    if (kept.length !== keep.length) this._writeSel(kept, p);
                }
                await refreshCounts();
            });
        // Open: reveal the current source folder in the OS file manager (local tool).
        $(".ph-batch-open").onclick = async () => {
            if (!source) { alert("Choose a source folder first."); return; }
            try {
                const r = await api.fetchApi("/uls/media/open_folder?path=" + encodeURIComponent(source),
                    { method: "POST" });
                const d = r && await r.json();
                if (!d || !d.ok) alert("Could not open the folder: " + ((d && d.error) || "unknown error"));
            } catch (e) { alert("Could not open the folder: " + e); }
        };
        $(".ph-batch-cancel").onclick = close;
        const helpEl = $(".ph-batch-help");
        $(".ph-batch-helpbtn").onclick = () => {
            const open = helpEl.style.display === "block";
            helpEl.style.display = open ? "none" : "block";
            if (!open) helpEl.scrollIntoView({ block: "nearest" });
        };
        overlay.onclick = (e) => { if (e.target === overlay) close(); };
        $(".ph-batch-apply").onclick = () => {
            // Apply = arm the unified batch. reset_seq bumps so a Processing cursor
            // re-homes to "Start at" on the next run (the old proc-Apply semantics).
            const prev = this._readCfg();
            const cfg = {
                enabled: true,   // applying a batch arms it; the Batch toggle turns it off later
                mode: $(".ph-mode-proc").checked ? "proc" : "frames",
                source: source || "",
                sort_mode: $(".ph-batch-sort").value,
                name_filter: $(".ph-batch-filter").value || "*",
                every_nth: getNth(),
                resize_method: $(".ph-batch-resize").value,
                start_at: Math.max(1, parseInt($(".ph-batch-start").value, 10) || 1) - 1,
                wrap: $(".ph-batch-wrap").checked,
                reset_seq: (parseInt(prev.reset_seq, 10) || 0) + 1,
                selection: prev.selection || null,   // checks are edited live, Apply keeps them
            };
            if (!cfg.source) { alert("Choose a source folder first."); return; }
            this._writeCfg(cfg);
            close();
        };

        // ── Sequences: a persistent library of named, renumbered sequence folders
        // under output/PLS_sequences. Pick one to load it as the active batch, or
        // Build & Use to write the current sorted/filtered selection into a new
        // named folder and load it. The raw source folder is only ever read.
        const seqStatus = $(".ph-batch-seq-status");
        const setSeq = (msg, kind) => {
            seqStatus.textContent = msg || "";
            seqStatus.style.color = kind === "err" ? "#e08a8a"
                                  : kind === "ok"  ? "#7fd17f" : "#8a8a92";
        };
        const seqSelect = $(".ph-seq-select");
        let seqPaths = {};   // name -> absolute path, from the last list fetch

        const loadProject = (path) => {
            if (!path) return;
            const cur = this._readCfg();
            // a built sequence is already ordered & uniform: load the whole folder
            this._writeCfg({
                enabled: true, mode: "frames", source: path,
                sort_mode: "name (natural)", name_filter: "*", every_nth: 1,
                resize_method: cur.resize_method || "none (strict)",
                selection: null,   // a built sequence is the whole folder by design
            });
        };

        const refreshProjects = async (selectName) => {
            try {
                const r = await api.fetchApi("/uls/media/seq/list");
                const d = r && await r.json();
                seqPaths = {};
                seqSelect.innerHTML = "";
                const list = (d && d.ok && d.projects) || [];
                if (!list.length) {
                    const o = document.createElement("option");
                    o.value = ""; o.textContent = "— no sequences yet —";
                    o.disabled = true; o.selected = true;
                    seqSelect.appendChild(o);
                    seqSelect.disabled = true;
                    return;
                }
                seqSelect.disabled = false;
                for (const p of list) {
                    seqPaths[p.name] = p.path;
                    const o = document.createElement("option");
                    o.value = p.name;
                    o.textContent = `${p.name}  (${p.count})`;
                    seqSelect.appendChild(o);
                }
                // programmatic .value does NOT fire onchange -> no double-load
                if (selectName && seqPaths[selectName]) seqSelect.value = selectName;
            } catch (e) { setSeq("✗ " + e, "err"); }
        };

        // choosing a saved sequence loads it directly as the batch (no second step)
        seqSelect.onchange = () => {
            const p = seqPaths[seqSelect.value];
            if (p) { loadProject(p); setSeq(`✓ loaded "${seqSelect.value}" as the batch.`, "ok"); }
        };

        // Use: load the picked sequence as the active batch, then close the panel
        // (the node shows the green-check selection + clip via loadProject's _writeCfg)
        $(".ph-seq-use").onclick = () => {
            const name = seqSelect.value;
            const p = seqPaths[name];
            if (!p) { setSeq("Pick a sequence to use.", "err"); return; }
            loadProject(p);
            close();
        };

        $(".ph-seq-build").onclick = async () => {
            if (!source) { setSeq("Choose a source folder first.", "err"); return; }
            const nameEl = $(".ph-seq-name");
            const name = (nameEl.value || "").trim();
            if (!name) { setSeq("Enter a name for the sequence.", "err"); return; }
            const q = "source=" + encodeURIComponent(source) +
                      "&name=" + encodeURIComponent(name) +
                      "&sort=" + encodeURIComponent($(".ph-batch-sort").value) +
                      "&filter=" + encodeURIComponent($(".ph-batch-filter").value || "*") +
                      "&nth=" + encodeURIComponent(String(getNth()));
            const build = async (overwrite) => {
                setSeq("Building…");
                const r = await api.fetchApi("/uls/media/seq/build?" + q + (overwrite ? "&overwrite=1" : ""),
                    { method: "POST" });
                return r && await r.json();
            };
            try {
                let d = await build(false);
                if (d && !d.ok && d.exists) {
                    if (!confirm(`Sequence "${d.name}" already exists — replace its contents?`)) {
                        setSeq("Kept the existing sequence.", ""); return;
                    }
                    d = await build(true);
                }
                if (d && d.ok) {
                    nameEl.value = "";
                    await refreshProjects(d.name);
                    loadProject(d.path);
                    close();   // built + loaded → close the panel, same as Use
                } else {
                    setSeq("✗ " + ((d && d.error) || "build failed"), "err");
                }
            } catch (e) { setSeq("✗ " + e, "err"); }
        };

        $(".ph-seq-delete").onclick = async () => {
            const name = seqSelect.value;
            if (!name || !seqPaths[name]) { setSeq("Pick a sequence to delete.", "err"); return; }
            if (!confirm(`Delete sequence "${name}"?\n\nThe folder and its frames will be removed. The original source images are not affected.`)) return;
            setSeq("Deleting…");
            try {
                const r = await api.fetchApi("/uls/media/seq/delete?name=" + encodeURIComponent(name),
                    { method: "POST" });
                const d = r && await r.json();
                if (d && d.ok) { await refreshProjects(); setSeq(d.removed ? `✓ deleted "${name}".` : `"${name}" was already gone.`, "ok"); }
                else setSeq("✗ " + ((d && d.error) || "delete failed"), "err");
            } catch (e) { setSeq("✗ " + e, "err"); }
        };

        refreshProjects();   // populate the picker when the panel opens
    }

    get state() { return (this.node.properties && this.node.properties.ph_media_state) || { folder: "", file: "", kind: "" }; }
    set state(s) {
        this.node.properties = this.node.properties || {};
        this.node.properties.ph_media_state = s;
        this._syncRef();
    }
    // v458: the audio companion is a SECOND selection slot, stored OUTSIDE media_ref
    // (mirrors `paused`): it persists across reload but must never leak into
    // media_ref / IS_CHANGED in Stufe A — audio has no graph output yet (Stufe B).
    get audioSel() { return (this.node.properties && this.node.properties.ph_media_audio) || null; }
    set audioSel(a) {
        this.node.properties = this.node.properties || {};
        this.node.properties.ph_media_audio = a || null;
        this._syncRef();
    }
    get audioOn() { return !!(this.node.properties && this.node.properties.ph_media_audio_on); }
    set audioOn(v) {
        this.node.properties = this.node.properties || {};
        this.node.properties.ph_media_audio_on = !!v;
        this._syncRef();
    }
    // media_ref (what the backend loads) = the VISUAL pick if present, else the audio
    // pick (audio-only -> the v457 placeholder branch in load()). When a visual is
    // chosen, the paired audio stays OUT of media_ref (preview-only until Stufe B).
    _syncRef() {
        if (!this._refWidget) return;
        const v = this.state, a = this.audioSel;
        const audOn = !!(this.audioOn && a && a.folder && a.file);
        const audObj = (extra) => Object.assign(extra, {
            folder: a.folder, file: a.file, mtime: a.mtime || 0,
            trimStart: +a.trimStart || 0, trimEnd: +a.trimEnd || 0,   // v464: trim window (seconds)
        });
        let ref = "";
        if (v && v.folder && v.file) {
            // v484 (D1 Stufe 2): the VIDEO trim window now rides into media_ref so the
            // backend slices the decoded frames (and IS_CHANGED re-fires, since media_ref
            // changed). 0/0 for a still or an untrimmed video -> backend no-op.
            const o = { folder: v.folder, file: v.file, kind: v.kind, mtime: v.mtime || 0,
                        vtrimStart: +v.trimStart || 0, vtrimEnd: +v.trimEnd || 0 };
            // Stufe B (v459): the paired audio rides in media_ref so the backend can
            // decode it for the AUDIO + muxed-VIDEO outputs (preview is unchanged).
            if (audOn) o.audio = audObj({});
            ref = JSON.stringify(o);
        } else if (audOn) {
            ref = JSON.stringify(audObj({ kind: "audio" }));
        }
        this._refWidget.value = ref;
    }
    // v458: the BROWSED folder is decoupled from the selection's folder, so a visual
    // from folder A survives navigating to folder B to pick an audio. `folder` is the
    // browsed folder; it falls back to the visual selection's folder for nodes saved
    // before v458 (which had no ph_media_browse).
    get folder() {
        const p = this.node.properties || {};
        if (typeof p.ph_media_browse === "string") return p.ph_media_browse;
        return this.state.folder || "";
    }

    // Selection-video paused state. Stored OUTSIDE ph_media_state on purpose: it
    // persists across reload (node.properties are serialized) but must NOT leak
    // into media_ref / IS_CHANGED — pausing the preview may never re-run the graph.
    get paused() { return !!(this.node.properties && this.node.properties.ph_media_paused); }
    set paused(v) {
        this.node.properties = this.node.properties || {};
        this.node.properties.ph_media_paused = !!v;
    }

    /* v644 drag & drop, v662 intent split. HONEST LIMIT: browsers never reveal
     * the origin PATH of an OS file drop (file.path is Electron-only), so in a
     * plain browser tab real files can only route through the upload machinery
     * into the PINNED folder; a dropped path TEXT — and a file.path when the
     * desktop client supplies one — can pin its origin folder instead.
     *
     * v662 makes the INTENT explicit instead of letting the source application
     * decide it (Frank, 2026-07-19: "manchmal wird der ganze Ordner mit
     * reingeladen, manchmal nur eine Datei"):
     *   plain drop  -> PIN the origin folder when a path is available,
     *                  otherwise copy (nothing else is possible).
     *   SHIFT drop  -> always COPY into the pinned folder.
     * Either way the node says afterwards which of the two ran. */
    _dropPathOf(dt) {
        // v665: comb EVERY type the drag offers. Firefox (unlike Chromium) often
        // ships a file:/// URI for an Explorer drag in text/x-moz-url or
        // text/uri-list — that IS the origin path, just dressed as a URI. This is
        // why "it worked before" for some drags and not others: it depends on what
        // the source puts on the wire, so we read all of it.
        const cand = [];
        try {
            for (const t of ["text/uri-list", "text/x-moz-url", "text/plain", "DownloadURL"]) {
                const v = (dt.getData && dt.getData(t)) || "";
                if (v) cand.push(...String(v).split(/[\r\n]+/));
            }
        } catch (e) { /* getData may throw during dragover — fine, drop-time calls succeed */ }
        // Electron / ComfyUI desktop hands the real path along with the file.
        const f = dt.files && dt.files.length ? dt.files[0] : null;
        if (f && (f.path || f.webkitRelativePath)) cand.push(f.path || f.webkitRelativePath);
        for (let c of cand) {
            c = String(c || "").trim().replace(/^"|"$/g, "");
            const p = this._pathFromUri(c);
            if (p) return p;
        }
        return "";
    }

    // file:///C:/a%20b/x.png -> C:\a b\x.png ; file:///home/y.png -> /home/y.png ;
    // a bare absolute path passes through; anything else -> "".
    _pathFromUri(c) {
        if (!c) return "";
        if (/^file:\/\//i.test(c)) {
            try {
                let p = decodeURIComponent(c.replace(/^file:\/+/i, "/"));
                const win = p.match(/^\/([A-Za-z]:)(\/.*)?$/);
                if (win) p = win[1] + (win[2] || "\\").replace(/\//g, "\\");
                return p;
            } catch (e) { return ""; }
        }
        return /^([A-Za-z]:[\\/]|\/)/.test(c) ? c : "";
    }

    // A short, self-clearing note in the node's status line — the drop must never
    // be silent about which of the two things it did.
    _dropNote(text, color) {
        const el = this.batchStatusEl;
        if (!el) return;
        if (this._dropNoteT) clearTimeout(this._dropNoteT);
        el.textContent = text;
        el.style.color = color || "#7fd17f";
        this._dropNoteT = setTimeout(() => {
            this._dropNoteT = null;
            this._renderBatchStatus();
        }, 6000);
    }

    // v664: the two drop zones. Raised only while a drag is in flight — the node's
    // face gains no permanent buttons. The "load from where it is" zone needs the
    // file's PATH, which a plain browser tab does not hand out for an OS file drag
    // (file.path is desktop-only); when the drag carries no path text that zone is
    // dimmed with the reason, but stays droppable: it then copies and says why.
    _dropHint(on, ev) {
        const el = this.dropHintEl;
        if (!el) return;
        el.classList.toggle("on", !!on);
        if (!on) { this._zoneHot(""); return; }
        const types = (ev && ev.dataTransfer && ev.dataTransfer.types)
            ? Array.from(ev.dataTransfer.types) : [];
        // v665: a path may arrive as text OR as a file:/// URI (Firefox often ships
        // one for an Explorer drag) — and even without any, a dropped FILE can be
        // relocated by name+size+mtime in the folders this node has visited. So the
        // pin zone only dims when neither route exists: no path-ish type AND no file.
        const maybePath = types.includes("text/plain") || types.includes("text/uri-list")
            || types.includes("text/x-moz-url") || types.includes("DownloadURL");
        const hasFiles = types.includes("Files") || types.includes("application/x-moz-file");
        const pin = el.querySelector(".ph-dz-pin");
        const sub = el.querySelector(".ph-dz-pinsub");
        if (pin) pin.classList.toggle("dim", !maybePath && !hasFiles);
        if (sub) {
            sub.textContent = maybePath
                ? "Selects the file and pins the folder it already lives in. Nothing is copied."
                : (hasFiles
                    ? "Finds the file again in recently used folders and pins it there. If it cannot be found, a copy goes to input instead."
                    : "Needs the file's path — the desktop client or a dropped path text. Without it this copies into input instead.");
        }
    }

    // Light the zone under the pointer, so the aim is confirmed before release.
    _zoneHot(zone) {
        const el = this.dropHintEl;
        if (!el) return;
        el.querySelectorAll(".ph-dz").forEach((z) => {
            z.classList.toggle("hot", !!zone && z.dataset.zone === zone);
        });
    }

    // The zone the file was dropped on WINS; Shift only decides a drop that missed
    // both zones (overlay not raised, or a drop on the node's frame).
    _intentFor(zone, shift) {
        if (zone === "copy" || zone === "pin") return zone;
        return shift ? "pin" : "copy";
    }

    async _onDrop(ev, intent) {
        const dt = ev.dataTransfer;
        if (!dt) return;
        // v663/v664: "pin" leaves the file where it is and pins its folder; "copy"
        // puts a copy in the official ComfyUI input folder. Nothing is ever written
        // into a browsed media folder ("sonst kriegt man Kuddelmuddel auf der
        // Festplatte"). Pinning needs a path; without one the drop copies and says so.
        const wantPin = intent === "pin";
        const path = wantPin ? this._dropPathOf(dt) : "";
        if (wantPin && path) {
            try {
                const r = await (await api.fetchApi("/uls/media/resolve?path=" + encodeURIComponent(path))).json();
                if (r && r.ok && r.folder) {
                    this.setFolder(r.folder);
                    pushRecentFolder(r.folder);
                    this._dropSelect(r.file);
                    this._dropNote("📌 Pinned the file's own folder and selected it — nothing was copied.",
                                   "#7fd17f");
                    return;
                }
            } catch (e) { /* drop is a convenience, never load-bearing */ }
        }
        if (wantPin && !path && dt.files && dt.files.length) {
            // v665: no path on the wire — RELOCATE the file instead of giving up.
            // The browser still hands out name, byte size and mtime, and that triple
            // is checked against the folders this node has actually visited (recents
            // + the current pin + input). Only an UNAMBIGUOUS hit pins — a miss or a
            // double is reported honestly and falls through to the copy. This is
            // verification against known ground, not a guess across the disk.
            const f = dt.files[0];
            try {
                const folders = [this.folder, ...getRecentFolders()].filter(Boolean);
                const body = JSON.stringify({ name: f.name, size: f.size,
                                              mtime: Math.round((f.lastModified || 0) / 1000),
                                              folders });
                const r = await (await api.fetchApi("/uls/media/locate",
                    { method: "POST", headers: { "Content-Type": "application/json" }, body })).json();
                if (r && r.ok && r.folder) {
                    this.setFolder(r.folder);
                    pushRecentFolder(r.folder);
                    this._dropSelect(r.file);
                    this._dropNote("📌 Found the file in " + r.folder + " — pinned its folder, nothing was copied.",
                                   "#7fd17f");
                    return;
                }
                this._dropNote(r && r.reason === "ambiguous"
                    ? "📌 Found the same file in several known folders — copying into input instead. Drop its path as text to pin one of them."
                    : "📌 Your browser hides the file's path and it was not found in recently used folders — copying into input. Tip: drag the path as text (Shift+right-click → Copy as path) to pin its folder.",
                    "#e0b060");
            } catch (e) {
                this._dropNote("📌 No path available for this drag — copying into the ComfyUI input folder instead.",
                               "#e0b060");
            }
        } else if (wantPin && !path) {
            this._dropNote("📌 No path available for this drag — copying into the ComfyUI input folder instead.",
                           "#e0b060");
        }
        if (dt.files && dt.files.length) {
            // v663 (Frank): a copy NEVER writes into whatever folder happens to be
            // pinned — that scattered stray files across the user's own media
            // directories ("Kuddelmuddel auf der Festplatte"). Dropped files land in
            // the official ComfyUI input folder, the same target the ⬆ Upload File
            // button uses, and the view follows them there.
            const input = await this._resolveInputPath();
            if (!input) {
                this._dropNote("⚠ Could not locate the ComfyUI input folder — nothing was copied.",
                               "#e0b060");
                return;
            }
            const fd = new FormData();
            for (const f of dt.files) fd.append("file", f, f.name);
            this._busyOn("Uploading…");
            try {
                const r = await (await api.fetchApi("/uls/media/upload?folder=" + encodeURIComponent(input),
                    { method: "POST", body: fd })).json();
                if (r && r.ok && r.names && r.names.length) {
                    this.setFolder(input);
                    pushRecentFolder(input);
                    await this.refreshGrid();
                    this._dropSelect(r.names[0]);
                    const n = r.names.length;
                    // Name what happened to ALL of them — a multi-file drop copies every
                    // file but can only select one, which read as "only one file arrived".
                    this._dropNote(`⬆ Copied ${n} file${n > 1 ? "s" : ""} into the ComfyUI input folder · `
                                   + `loaded ${r.names[0]}${n > 1 ? " (the rest are in the grid)" : ""}`,
                                   "#7fb3d1");
                } else {
                    this._dropNote("⚠ Nothing was copied — the drop held no image, video or audio file.",
                                   "#e0b060");
                }
            } catch (e) {
                this._dropNote("⚠ The copy failed — the grid is unchanged.", "#e0b060");
            } finally {
                this._busyOff();
            }
            return;
        }
        this._dropNote("⚠ Nothing to drop here — no image, video or audio file was found.",
                       "#e0b060");
    }

    _dropSelect(name, tries = 0) {
        // v692 -- this used to look for the TILE in the DOM and click it. Two
        // reasons that silently failed, both of them normal states:
        //   * renderGrid builds ONE PAGE of GRID_PAGE tiles, and a drop resets
        //     to page 1. In a folder of 204 files the new file is only findable
        //     when it happens to sort into the first twenty.
        //   * in Solo the grid is not built AT ALL (v683 L1), so there is never
        //     a tile to click.
        // Both ended in twenty silent retries and a node that just did not
        // change. The list is the truth, not the DOM: find the file there, page
        // the grid to where it lives so the user SEES the selection, and go
        // through select() -- which is exactly what the tile's own click calls,
        // so no logic is duplicated.
        const files = this._files || [];
        const idx = files.findIndex((f) => f && f.name === name);
        if (idx >= 0) {
            const page = Math.floor(idx / GRID_PAGE);
            if (this._page !== page) {
                this._page = page;
                // In Solo there is no grid to redraw; the page is remembered and
                // the swap back renders it (renderGrid sets _gridStale itself).
                if (!this.view.solo) this.renderGrid();
            }
            this.select(files[idx]);
            return;
        }
        // The listing may still be in flight -- keep the old retry window.
        if (tries < 20) { setTimeout(() => this._dropSelect(name, tries + 1), 150); return; }
        // v692: never fail silently again. If it is not in the list after three
        // seconds, say so instead of leaving a node that looks broken.
        this._dropNote("\u26a0 " + name + " was not found in the folder listing \u2014 "
                       + "it is not selected. Use \u27f3 to re-read the folder.", "#e0b060");
    }

    setFolder(path) {
        // v458: navigation sets the BROWSED folder only — the visual/audio selection
        // is preserved, so you can browse to another folder to pick an audio (or a new
        // image/video) without losing what you already chose.
        this.node.properties = this.node.properties || {};
        this.node.properties.ph_media_browse = path || "";
        this.renderPath();
        this._renderPreview();
        this.refreshGrid();
    }

    renderPath() {
        this.pathEl.textContent = this.folder || "no folder pinned";
        this.pathEl.title = this.folder || "";
    }

    select(f) {
        this._stopAudioHover();   // v457: committing to the Selection stops any hover-preview clip (no double audio)
        if (f.kind === "audio") {
            // v458: audio fills its OWN slot — the visual selection is kept. Clicking the
            // already-selected audio again clears it (toggle-off).
            const a = this.audioSel;
            if (a && a.file === f.name && this._samePath(a.folder || "", this.folder)) { this._clearAudio(); return; }
            this.audioSel = { folder: this.folder, file: f.name, mtime: f.mtime || 0 };
            this.audioOn = true;          // picking an audio arms + reveals the audio container
            this._pendingSplit = null;    // v473: a deliberate audio pick overrides any pending video-audio split
            this._syncAudioToggle();
        } else {
            // visual (image/video) fills the visual slot — the audio selection is kept.
            // v473: selecting a NEW video arms an embedded-audio split (resolved once the
            // video's metadata loads). Re-selecting the same video, or picking an image,
            // leaves the current audio untouched.
            const prev = this.state;
            const sameVideo = !!(prev && prev.kind === "video" && prev.file === f.name
                && this._samePath(prev.folder || "", this.folder));
            // v476: a NEW video resets the video trim (fresh state, no trim window); re-
            // selecting the SAME video carries its trim forward (like the audio selection,
            // which a re-click leaves untouched).
            this.state = sameVideo
                ? { folder: this.folder, file: f.name, kind: f.kind, mtime: f.mtime || 0,
                    trimStart: prev.trimStart, trimEnd: prev.trimEnd }
                : { folder: this.folder, file: f.name, kind: f.kind, mtime: f.mtime || 0 };
            this.paused = false;          // a fresh pick auto-plays (loop) until the user pauses
            this._pendingSplit = (f.kind === "video" && !sameVideo)
                ? { folder: this.folder, file: f.name, mtime: f.mtime || 0 } : null;
        }
        this._markGridSelection();
        this._renderPreview();
    }

    _fileURL(f) { return this._fileURLFor(this.folder, f); }
    _fileURLFor(folder, f) {
        return api.apiURL("/uls/media/file?folder=" + encodeURIComponent(folder || "") +
            "&file=" + encodeURIComponent(f.name) + "&t=" + Math.floor(f.mtime || 0));
    }

    // GIFs are kind "image" (a still to the loader), but in previews/thumbnails we
    // want them to ANIMATE. The thumb route flattens them to a static JPEG, so for
    // a GIF we point <img> at the raw file (served as image/gif) -> it animates.
    _isGif(f) { return /\.gif$/i.test((f && f.name) || ""); }

    _renderPreview() {
        if (!this.previewMediaEl) return;
        this._batchAnimStop();   // kill any running flipbook before we rebuild the preview
        this._renderAudioPane();   // v458: the audio pane is independent of the visual branch below
        this._renderVideoTrimPane(null);   // v476: collapse by default; the video branch below populates it
        // Batch is decoupled from browsing. The Selection column shows the whole
        // batch when nothing is selected yet OR the current selection is a batch
        // member (then the flipbook jumps to that frame). When batch is on but a
        // NON-member / other-folder file is selected, we fall through and preview
        // that single file — a peek; the node still EMITS the batch while the ▦ Batch
        // toggle is on (backend keys off batch_config.enabled, not this selection).
        const _cfg = this._readCfg();
        if (_cfg.enabled && _cfg.source) {
            const marks = this._gridBatchMarks();   // {name:idx} when on-source & enabled, else null
            const sel = this.state.file || "";
            const isMember = !!(marks && sel && (sel in marks));
            if (!sel || isMember) { this._renderBatchPreview(_cfg, isMember ? sel : null); return; }
            // else: peeking a non-member -> fall through to the single-file preview
        }
        const s = this.state;
        this.previewMediaEl.innerHTML = "";
        this.previewCapEl.textContent = s.file || "";
        this.previewCapEl.title = s.file || "";
        if (!s.folder || !s.file) {
            // v458: with no visual but an audio paired, the visual area collapses and
            // the audio pane (rendered above) carries the Selection — the audio-only
            // view (≈ v457). Otherwise the usual empty placeholder.
            const audOnly = !!(this.audioOn && this.audioSel && this.audioSel.file);
            this.previewMediaEl.innerHTML = audOnly ? "" : `<div class="ph-media-empty">No selection</div>`;
            this.previewMediaEl.style.flex = audOnly ? "none" : "";
            return;
        }
        this.previewMediaEl.style.flex = "";   // a visual is present -> the visual area fills again
        const f = { name: s.file, mtime: s.mtime || 0 };
        const fobj = (this._files || []).find((x) => x.name === s.file);   // backend w/h for an instant badge
        if (s.kind === "video") {
            const v = document.createElement("video");
            // v626: play ONCE and stop (Frank). A fresh selection or a reload runs the
            // clip exactly one time to its end and holds there; pressing play again
            // loops (the once-ended hook below restores loop for every later play).
            v.muted = true; v.loop = false; v.playsInline = true; v.controls = true;
            v.addEventListener("ended", () => { v.loop = true; }, { once: true });
            v.poster = this._thumbURLFor(s.folder, f);
            // The native play/pause button drives ph_media_paused, which survives
            // reload. isConnected guard: the 'pause' that fires when an OLD preview
            // video is detached on re-render must not corrupt the new state.
            // v626: the natural END also fires 'pause' -- that is NOT a user pause, so
            // it must not persist as one (or the next reload would start frozen instead
            // of running its one pass).
            v.addEventListener("pause", () => { if (v.isConnected && !v.ended) this.paused = true; });
            v.addEventListener("play", () => { if (v.isConnected) this.paused = false; });
            v.addEventListener("loadedmetadata", () => { this._fitPreviewVideo(); this._resolveEmbeddedAudio(v, s); });
            v.src = this._fileURLFor(s.folder, f);
            this.previewMediaEl.appendChild(v);
            if (fobj && fobj.w && fobj.h) this._setPrevDim(fobj.w, fobj.h);   // instant; _fitPreviewVideo confirms from videoWidth
            this._fitPreviewVideo();           // metadata may already be cached
            if (this.paused) {
                v.autoplay = false;            // restored paused -> first frame, wait for play
            } else {
                v.autoplay = true;             // fresh/again -> ONE pass to the end, then stop (v626)
                const p = v.play && v.play(); if (p && p.catch) p.catch(() => {});
            }
            this._renderVideoTrimPane(v);   // v476: build the trim strip bound to this preview video
        } else {
            const img = document.createElement("img");
            // The full image is the source of truth for the badge; if it fails and
            // we fall back to the THUMB, its (smaller) size must NOT become the badge.
            // v661 (B-03): when the THUMB fails too, the <img> used to stay silently
            // blank — a black Selection with no hint, and since navigation keeps the
            // selection (v458) it survived every folder change. Say so instead.
            let usingThumb = false;
            img.onerror = () => {
                if (!usingThumb) { usingThumb = true; img.src = this._thumbURLFor(s.folder, f); return; }
                const e = document.createElement("div");
                e.className = "ph-media-empty";
                e.textContent = "Cannot preview " + (f && f.name ? f.name : "this file");
                e.title = "The file could not be decoded as an image. Pick another tile to clear this.";
                if (img.isConnected) img.replaceWith(e);
            };
            img.onload = () => { if (!usingThumb && img.naturalWidth) this._setPrevDim(img.naturalWidth, img.naturalHeight); };
            img.src = this._fileURLFor(s.folder, f);
            this.previewMediaEl.appendChild(img);
            if (fobj && fobj.w && fobj.h) this._setPrevDim(fobj.w, fobj.h);   // instant from the listing (true source dims)
        }
    }

    _thumbURL(f) { return this._thumbURLFor(this.folder, f); }

    // v473: when a NEW video with an embedded audio track is selected, split that track
    // off automatically and load it into the audio pane (with the full trim/loop UI). It
    // points audioSel at the video FILE itself — _syncRef carries that into media_ref's
    // ['audio'], and the backend's _decode_paired_audio -> _load_audio reads the video's
    // own audio stream (no backend change needed). A separately picked audio file
    // overrides it; a silent video clears it. Detection is browser-side: Firefox's
    // mozHasAudio (plus audioTracks where a browser exposes it).
    _resolveEmbeddedAudio(videoEl, sel) {
        const p = this._pendingSplit;
        if (!p || !sel) return;
        if (p.file !== sel.file || !this._samePath(p.folder || "", sel.folder || "")) return;   // a stale/old video load
        this._pendingSplit = null;
        const hasAudio = videoEl.mozHasAudio === true
            || (videoEl.audioTracks && videoEl.audioTracks.length > 0);
        if (hasAudio) {
            this.audioSel = { folder: p.folder, file: p.file, mtime: p.mtime,
                              fromVideo: true, trimStart: 0, trimEnd: 0 };
            this.audioOn = true;
        } else {
            this.audioSel = null;          // silent video -> nothing to split
            this.audioOn = false;
        }
        this._syncAudioToggle();
        this._markGridSelection();
        this._renderAudioPane();
        if (hasAudio) {
            // v474: the embedded-audio <audio> shares the file the <video> is still
            // streaming, so on first paint its metadata can fail to settle — the pane's
            // player/trim then stay inert until a manual Audio-toggle (the reported hang).
            // Once the video has buffered, re-render the pane IF the audio duration never
            // resolved, so a fresh <audio> reloads from the now-ready file.
            const resettle = () => {
                const a = this.audioSel;
                if (a && a.fromVideo && a.file === p.file && !this._audioDur) this._renderAudioPane();
            };
            if (videoEl.readyState >= 3) resettle();                                  // HAVE_FUTURE_DATA — already buffered
            else videoEl.addEventListener("canplay", resettle, { once: true });
        }
    }

    _thumbURLFor(folder, f) {
        return api.apiURL("/uls/media/thumb?folder=" + encodeURIComponent(folder || "") +
            "&file=" + encodeURIComponent(f.name) + "&t=" + Math.floor(f.mtime || 0));
    }

    // ── v458: audio companion pane (collapsible, below the visual preview) ───────
    // Shown iff the audio toggle is ON and an audio is chosen. Renders the same
    // faux-waveform card as the grid above a native <audio> player, plus a header
    // with the file name, a 📁 jump-to-folder, and a ✕ clear. When the audio lives in
    // a different folder than the visual, its folder is shown so the pairing is clear.
    _renderAudioPane() {
        const pane = this.audioPaneEl;
        if (!pane) return;
        const a = this.audioSel;
        const on = !!(this.audioOn && a && a.folder && a.file);
        if (!on) { this._trim = null; this._loopOn = false; pane.innerHTML = ""; pane.style.display = "none"; return; }
        pane.style.display = "";
        pane.innerHTML = "";
        const f = { name: a.file, mtime: a.mtime || 0 };

        const head = document.createElement("div"); head.className = "ph-aud-head";
        const nm = document.createElement("span"); nm.className = "ph-aud-name";
        nm.textContent = "♪ " + a.file + (a.fromVideo ? "  ·  from video" : "");
        nm.title = a.fromVideo ? (a.file + " — embedded audio track of the selected video") : a.file;
        const spacer = document.createElement("span"); spacer.style.flex = "1";
        head.append(nm, spacer);
        // v477: the jump-to-folder button only makes sense when the audio lives in a
        // DIFFERENT folder than the one being browsed. For a "from video" track (the audio
        // IS the video file, same folder) or any same-folder audio the jump is a no-op, so
        // the button is omitted entirely.
        if (a.folder && !this._samePath(a.folder, this.folder)) {
            const jump = document.createElement("button"); jump.className = "ph-aud-btn"; jump.textContent = "📁";
            jump.title = "Jump the grid to this audio's folder (it shows marked there)";
            jump.onclick = () => this.setFolder(a.folder);
            head.append(jump);
        }
        const clr = document.createElement("button"); clr.className = "ph-aud-btn"; clr.textContent = "✕";
        clr.title = "Remove the audio selection";
        clr.onclick = () => this._clearAudio();
        head.append(clr);
        pane.appendChild(head);

        const vis = this.state;
        if (vis && vis.folder && a.folder && !this._samePath(vis.folder, a.folder)) {
            const src = document.createElement("div"); src.className = "ph-aud-src";
            src.textContent = a.folder; src.title = a.folder;
            pane.appendChild(src);
        }

        const wrap = document.createElement("div"); wrap.className = "ph-aud-prevwrap";
        wrap.appendChild(this._makeAudioCard(f));
        const au = document.createElement("audio");
        au.className = "ph-aud-prev"; au.controls = true;
        // v475: a "from video" track points at the SAME file the preview <video> is
        // muted-looping. Firefox treats two media elements on an IDENTICAL url as one
        // contended resource: the metadata settles (the duration shows) but playable
        // data never buffers, so the native play button hangs until a manual seek (a
        // Range fetch) or an Audio OFF->ON rebuild. handle_media_file reads only
        // folder+file (extra params are ignored) and honors Range, so giving the audio a
        // DISTINCT url + eager preload lets it buffer on its OWN connection and play on
        // the first click. Preview only -- media_ref carries folder/file/trim (not this
        // url), so the AUDIO / muxed-video outputs are unaffected.
        let aurl = this._fileURLFor(a.folder, f);   // the audio's OWN folder, not the browsed one
        if (a.fromVideo) { aurl += "&__a=1"; au.preload = "auto"; }
        au.src = aurl;
        wrap.appendChild(au);
        pane.appendChild(wrap);
        this._buildTrimUI(pane, a, au);   // v464: audio trim strip (handles + numeric in/out)
    }

    // v464: the audio trim strip — two draggable handles over a track + two numeric
    // fields (absolute start/end seconds), kept in sync both ways. The values are
    // stored on audioSel (node.properties) as trimStart (head) / trimEnd (tail) and
    // ride into media_ref via _syncRef(); the backend slices the audio to the window.
    _buildTrimUI(pane, a, au) {
        const sec = document.createElement("div"); sec.className = "ph-trim";
        const track = document.createElement("div"); track.className = "ph-trim-track";
        const dimL = document.createElement("div"); dimL.className = "ph-trim-dim";
        const dimR = document.createElement("div"); dimR.className = "ph-trim-dim";
        const keep = document.createElement("div"); keep.className = "ph-trim-keep";
        const hS = document.createElement("div"); hS.className = "ph-trim-handle"; hS.title = "Trim start";
        const hE = document.createElement("div"); hE.className = "ph-trim-handle"; hE.title = "Trim end";
        track.append(dimL, dimR, keep, hS, hE);
        const fields = document.createElement("div"); fields.className = "ph-trim-fields";
        const inS = document.createElement("input"); inS.className = "ph-trim-num"; inS.type = "text"; inS.inputMode = "text";
        const inE = document.createElement("input"); inE.className = "ph-trim-num"; inE.type = "text"; inE.inputMode = "text";
        const lblS = document.createElement("label"); lblS.append("Start", inS);
        const lblE = document.createElement("label"); lblE.append("End", inE);
        const len = document.createElement("span"); len.className = "ph-trim-len"; len.textContent = "—";
        fields.append(lblS, lblE, len);
        sec.append(track, fields);
        pane.appendChild(sec);
        // v625: fixed length input (seconds) — type e.g. 3 + Enter and the selection
        // becomes a fixed 3 s window you slide along the timeline. Empty = unlock.
        const fix = document.createElement("input"); fix.className = "ph-trim-num ph-trim-fix";
        fix.type = "text"; fix.inputMode = "decimal"; fix.placeholder = "s";
        fix.title = "Fixed length in seconds. Type e.g. 3 + Enter \u2014 the selection becomes a "
                  + "fixed 3 s window you slide along the timeline (handles and the green zone "
                  + "move it as a whole). Empty = unlock.";
        const lblF = document.createElement("label"); lblF.append("Fix", fix);
        fields.insertBefore(lblF, len);   // mirror the video row: Start · End · Fix · len
        this._trim = { track, dimL, dimR, keep, hS, hE, inS, inE, len, fix };

        const dur = () => this._audioDur || 0;
        const clamp = (x) => Math.max(0, Math.min(x, dur()));
        // v465: read the LIVE selection, not the captured `a`. _applyTrim REPLACES
        // the audioSel object, so a captured `a` goes stale mid-drag and makes the
        // dragged handle snap the OTHER one to its extreme (intro pushes outro to the
        // wall and vice versa).
        const curStart = () => { const c = this.audioSel; return clamp(+(c && c.trimStart) || 0); };
        const curEnd = () => { const c = this.audioSel; return clamp(dur() - (+(c && c.trimEnd) || 0)); };
        // v467: parse a typed value as m:ss(.mmm) OR plain seconds, matching the display
        // format. ":" → minutes:seconds (chains for h:m:s); otherwise raw seconds.
        const parseT = (str) => {
            str = String(str == null ? "" : str).trim();
            if (str === "") return NaN;
            if (str.indexOf(":") >= 0) {
                const parts = str.split(":");
                const sec = parseFloat(parts.pop());
                let total = Number.isFinite(sec) ? sec : 0, mult = 60;
                while (parts.length) { const p = parseFloat(parts.pop()); if (Number.isFinite(p)) total += p * mult; mult *= 60; }
                return total;
            }
            return parseFloat(str);
        };
        // v466: play-selection loop — a circular toggle to the left of the native
        // player that plays only the [start,end] window on a loop. The bounds are read
        // LIVE (curStart/curEnd), so while it loops it follows the handles as you drag.
        this._loopOn = false;
        const loop = document.createElement("button");
        loop.className = "ph-trim-loop";
        loop.title = "Play the selection on a loop (follows the handles live)";
        // v467: centered SVG glyph (ring centered in the 24x24 viewBox) so the spin is
        // truly centered — the bare U+27F3 char sat off-center in its box and wobbled.
        loop.innerHTML = "<span><svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><path d=\"M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8\"/><path d=\"M21 3v5h-5\"/></svg></span>";
        const playRow = document.createElement("div"); playRow.className = "ph-aud-playrow";
        au.parentNode.insertBefore(playRow, au);
        playRow.append(loop, au);
        const setLoop = (on) => {
            this._loopOn = on;
            loop.classList.toggle("playing", on);
            if (on) { try { au.currentTime = curStart(); } catch (e) {} au.play().catch(() => {}); }
            else { au.pause(); }
        };
        loop.onclick = () => setLoop(!this._loopOn);
        au.addEventListener("timeupdate", () => {
            if (!this._loopOn) return;
            const s = curStart(), e = curEnd();
            if (e - s < 0.05) return;                       // no usable window yet
            if (au.currentTime >= e - 0.015 || au.currentTime < s - 0.05) {
                try { au.currentTime = s; } catch (err) {}  // wrap to the live start
                if (au.paused) au.play().catch(() => {});
            }
        });
        au.addEventListener("ended", () => {                // selection end == file end
            if (!this._loopOn) return;
            try { au.currentTime = curStart(); } catch (e) {}
            au.play().catch(() => {});
        });
        au.addEventListener("pause", () => {                // native pause drops the loop
            if (this._loopOn && !au.ended) { this._loopOn = false; loop.classList.remove("playing"); }
        });
        const px2sec = (clientX) => {
            const r = track.getBoundingClientRect();
            return clamp(((clientX - r.left) / Math.max(1, r.width)) * dur());
        };
        // v470: a flat 1-second minimum selection — simple and sensible (selections here
        // run ~2-8 s). The v469 handle-width/pixel-scaling produced a silly ~10 s floor on
        // long tracks; that's gone. Clamp to the clip length for sub-second files.
        const MIN_GAP = 1.0;
        const minGap = () => Math.min(MIN_GAP, dur());
        const startDrag = (which, ev) => {
            if (dur() <= 0) return;
            ev.preventDefault();
            // v625: with a fixed window, keep the grab point's offset into the window.
            const sx0 = ev.touches ? ev.touches[0].clientX : ev.clientX;
            const grabOff = px2sec(sx0) - curStart();
            // v471: when the two handles sit within ~a handle-width on screen they can't be
            // grabbed apart (a buried START under END at the 1 s minimum). In that case the
            // edge NEAREST the cursor follows it — so you can pull START out to the left (or
            // END to the right) and still widen/narrow either way. When the handles are
            // clearly apart, the grabbed handle moves as usual.
            const r0 = track.getBoundingClientRect();
            const gapPx = (r0.width > 0 && dur() > 0) ? ((curEnd() - curStart()) / dur()) * r0.width : 999;
            const nearest = gapPx < 14;
            const move = (e) => {
                const x = e.touches ? e.touches[0].clientX : e.clientX;
                const g = minGap();
                const px = px2sec(x);
                // v625: fixed-length window — the grabbed point follows the cursor, the
                // length is the law. MIN_GAP does not apply to an explicit fix.
                const fx = this._aFixLen();
                if (fx) {
                    const s0 = (which === "s") ? px
                             : (which === "e") ? px - fx[0]
                             : px - grabOff;                     // whole-window drag
                    const w = _fixWindow(s0, fx[0], dur(), fx[1]);
                    this._applyTrim(w[0], w[1]);
                    return;
                }
                let s = curStart(), en = curEnd();
                if (nearest) {
                    if (Math.abs(px - s) <= Math.abs(px - en)) s = Math.min(Math.max(0, px), en - g);
                    else en = Math.max(Math.min(dur(), px), s + g);
                } else if (which === "s") {
                    s = Math.min(px, en - g);
                } else {
                    en = Math.max(px, s + g);
                }
                this._applyTrim(s, en);
            };
            const up = () => {
                window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up);
                window.removeEventListener("touchmove", move); window.removeEventListener("touchend", up);
            };
            window.addEventListener("mousemove", move); window.addEventListener("mouseup", up);
            window.addEventListener("touchmove", move, { passive: false }); window.addEventListener("touchend", up);
        };
        hS.addEventListener("mousedown", (e) => startDrag("s", e));
        hE.addEventListener("mousedown", (e) => startDrag("e", e));
        hS.addEventListener("touchstart", (e) => startDrag("s", e), { passive: false });
        hE.addEventListener("touchstart", (e) => startDrag("e", e), { passive: false });
        // v625: the green keep zone drags the whole fixed window (fixed-only; unfixed it
        // stays pointer-events:none per CSS).
        keep.addEventListener("mousedown", (e) => { if (this._aFixLen()) startDrag("w", e); });
        keep.addEventListener("touchstart", (e) => { if (this._aFixLen()) startDrag("w", e); }, { passive: false });
        fix.addEventListener("change", () => {
            const v = parseT(fix.value);
            this._setAudioFix(Number.isFinite(v) && v > 0 ? v : 0);
        });
        fix.addEventListener("keydown", (e) => { if (e.key === "Enter") fix.blur(); });
        len.addEventListener("click", () => {
            if (this._aFixLen()) { this._setAudioFix(0); return; }
            this._setAudioFix(Math.max(0.1, Math.round((curEnd() - curStart()) * 10) / 10));
        });

        const commit = (which) => {
            const d = dur(); if (d <= 0) return;
            let s = parseT(inS.value), en = parseT(inE.value);
            // v625: with a fixed window the edited field MOVES it.
            const fx = this._aFixLen();
            if (fx) {
                const s0 = (which === "e" && Number.isFinite(en)) ? en - fx[0]
                         : (Number.isFinite(s) ? s : curStart());
                const w = _fixWindow(s0, fx[0], d, fx[1]);
                this._applyTrim(w[0], w[1]);
                return;
            }
            if (!Number.isFinite(s)) s = curStart();
            if (!Number.isFinite(en)) en = curEnd();
            const g = minGap();
            s = Math.max(0, Math.min(s, d)); en = Math.max(0, Math.min(en, d));
            if (en - s < g) { en = Math.min(d, s + g); if (en - s < g) s = Math.max(0, en - g); }
            this._applyTrim(s, en);
        };
        inS.addEventListener("change", () => commit("s")); inE.addEventListener("change", () => commit("e"));
        inS.addEventListener("keydown", (e) => { if (e.key === "Enter") inS.blur(); });
        inE.addEventListener("keydown", (e) => { if (e.key === "Enter") inE.blur(); });

        this._audioDur = (isFinite(au.duration) && au.duration > 0) ? au.duration : 0;
        if (!this._audioDur) {
            au.addEventListener("loadedmetadata", () => {
                this._audioDur = (isFinite(au.duration) && au.duration > 0) ? au.duration : 0;
                this._updateTrimUI();
            }, { once: true });
        }
        this._updateTrimUI();
    }

    _applyTrim(startAbs, endAbs) {
        const a = this.audioSel; if (!a) return;
        const d = this._audioDur || 0;
        // v469: snap to tenths (one decimal — all we need) and store the END as the exact
        // tail of the tenths-rounded end, so the field, the loop window (curEnd = d -
        // trimEnd) and the backend slice all land on the same tenth.
        const r1 = (x) => Math.round(x * 10) / 10;
        let s = r1(Math.max(0, Math.min(startAbs, d)));
        let e = r1(Math.max(0, Math.min(endAbs, d)));
        if (e <= s) e = Math.min(d, r1(s + 0.1));
        this.audioSel = { ...a, trimStart: s, trimEnd: Math.max(0, d - e) };
        this._updateTrimUI();
    }

    // ── v625: fixed-length audio trim (seconds; tenths are audio's native grid) ──
    // [windowSeconds, snapSeconds] when a fix is armed; else null.
    _aFixLen() {
        const v = +(this.audioSel && this.audioSel.trimFixSec) || 0;
        return (v > 0) ? [Math.round(v * 10) / 10, 0.1] : null;
    }

    // Arm (sec>0) or release (sec=0) the fixed window on audioSel — _applyTrim
    // spreads the old object, so the fix survives every trim commit and reload.
    _setAudioFix(sec) {
        const a = this.audioSel; if (!a) return;
        this.audioSel = { ...a, trimFixSec: Math.max(0, Math.round((+sec || 0) * 10) / 10) };
        const fx = this._aFixLen();
        if (fx) {
            const d = this._audioDur || 0;
            const cs = Math.max(0, Math.min(+(this.audioSel.trimStart) || 0, d));
            const w = _fixWindow(cs, fx[0], d, fx[1]);
            this._applyTrim(w[0], w[1]);
        } else {
            this._updateTrimUI();
        }
    }

    _updateTrimUI() {
        const t = this._trim; if (!t) return;
        const a = this.audioSel; const d = this._audioDur || 0;
        // v469: timestamps as m:ss.t — one decimal (a tenth of a second) is all we need,
        // and it equals the stored tenths value, so field, loop and slice agree. Snap to a
        // tenth BEFORE splitting m:ss so 9:59.96 carries to 10:00.0 (not "9:60.0").
        const fmtT = (x) => { x = Math.max(0, Math.round(x * 10) / 10); const m = Math.floor(x / 60); return m + ":" + (x - m * 60).toFixed(1).padStart(4, "0"); };
        const fmtLen = (x) => Math.max(0, x).toFixed(1);
        if (d <= 0) {
            t.inS.value = ""; t.inE.value = ""; t.inS.disabled = t.inE.disabled = true;
            t.len.textContent = "loading…";
            t.dimL.style.width = "0%"; t.dimR.style.width = "0%";
            t.keep.style.left = "0%"; t.keep.style.right = "0%";
            t.hS.style.left = "0%"; t.hE.style.left = "100%";
            return;
        }
        t.inS.disabled = t.inE.disabled = false;
        const s = Math.max(0, Math.min(+(a && a.trimStart) || 0, d));
        const e = Math.max(0, Math.min(d - (+(a && a.trimEnd) || 0), d));
        const sp = (s / d) * 100, ep = (e / d) * 100;
        t.dimL.style.left = "0%"; t.dimL.style.width = sp + "%";
        t.dimR.style.left = ep + "%"; t.dimR.style.width = (100 - ep) + "%";
        t.keep.style.left = sp + "%"; t.keep.style.right = (100 - ep) + "%";
        t.hS.style.left = sp + "%"; t.hE.style.left = ep + "%";
        if (document.activeElement !== t.inS) t.inS.value = fmtT(s);
        if (document.activeElement !== t.inE) t.inE.value = fmtT(e);
        // v625: fixed-length state — lock glyph + amber, whole window draggable (CSS .fixed).
        const fx = this._aFixLen();
        t.track.classList.toggle("fixed", !!fx);
        t.len.classList.toggle("on", !!fx);
        if (t.fix && document.activeElement !== t.fix) {
            t.fix.value = (this.audioSel && +this.audioSel.trimFixSec > 0)
                ? String(this.audioSel.trimFixSec) : "";
        }
        if (fx) {
            t.len.textContent = "\ud83d\udd12 " + fmtLen(fx[0]) + " s";
            t.len.title = "Fixed window \u2014 drag the green zone or a handle to slide it. Click to unlock.";
        } else {
            t.len.textContent = "\u2192 " + fmtLen(Math.max(0, e - s)) + " s";
            t.len.title = "Click to LOCK this length \u2014 the selection then moves as a fixed window.";
        }
    }

    // ── v476 · Video trim (Stufe 1) ─────────────────────────────────────────────
    // A trim strip under the video preview, mirroring the audio trim 1:1 (track + 2
    // handles + m:ss.t fields + length readout + a loop-preview button that loops only
    // [start,end] on the PREVIEW video). The window lives on the visual selection
    // (ph_media_state) as trimStart (head) / trimEnd (tail, seconds): it survives reload
    // and resets when a NEW video is picked (a same-video re-click carries it). v484
    // (Stufe 2): _syncRef now carries trimStart/trimEnd into media_ref as vtrimStart/
    // vtrimEnd, the backend slices the decoded frames to that window, and IS_CHANGED
    // re-fires (media_ref changed). Parallel state slots (_vtrim / _videoDur / _vloopOn)
    // keep this from colliding with the audio trim.
    _renderVideoTrimPane(vid) {
        const pane = this.videoTrimEl;
        if (!pane) return;
        const s = this.state;
        const on = !!(vid && s && s.kind === "video" && s.folder && s.file);
        if (!on) { this._vtrim = null; this._vloopOn = false; pane.innerHTML = ""; pane.style.display = "none"; return; }
        pane.style.display = "";
        pane.innerHTML = "";
        this._buildVideoTrimUI(pane, vid);
    }

    _buildVideoTrimUI(pane, vid) {
        const head = document.createElement("div"); head.className = "ph-vtrim-head";
        const loop = document.createElement("button");
        loop.className = "ph-trim-loop";
        loop.title = "Play the video selection on a loop (follows the handles live)";
        loop.innerHTML = "<span><svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><path d=\"M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8\"/><path d=\"M21 3v5h-5\"/></svg></span>";
        const lbl = document.createElement("span"); lbl.className = "ph-vtrim-label"; lbl.textContent = "\u2702 Video trim";
        head.append(loop, lbl);
        pane.appendChild(head);

        const sec = document.createElement("div"); sec.className = "ph-trim";
        const track = document.createElement("div"); track.className = "ph-trim-track";
        const dimL = document.createElement("div"); dimL.className = "ph-trim-dim";
        const dimR = document.createElement("div"); dimR.className = "ph-trim-dim";
        const keep = document.createElement("div"); keep.className = "ph-trim-keep";
        const hS = document.createElement("div"); hS.className = "ph-trim-handle"; hS.title = "Trim start";
        const hE = document.createElement("div"); hE.className = "ph-trim-handle"; hE.title = "Trim end";
        track.append(dimL, dimR, keep, hS, hE);
        const fields = document.createElement("div"); fields.className = "ph-trim-fields";
        const inS = document.createElement("input"); inS.className = "ph-trim-num"; inS.type = "text"; inS.inputMode = "text";
        const inE = document.createElement("input"); inE.className = "ph-trim-num"; inE.type = "text"; inE.inputMode = "text";
        const lblS = document.createElement("label"); lblS.append("Start", inS);
        const lblE = document.createElement("label"); lblE.append("End", inE);
        const len = document.createElement("span"); len.className = "ph-trim-len"; len.textContent = "\u2014";
        // v625: fixed frame-count input — type e.g. 121 + Enter and the selection becomes
        // a fixed 121-frame window you slide along the timeline. Empty = unlock.
        const fix = document.createElement("input"); fix.className = "ph-trim-num ph-trim-fix";
        fix.type = "text"; fix.inputMode = "numeric"; fix.placeholder = "fr";
        fix.title = "Fixed frame count (native fps). Type e.g. 121 + Enter \u2014 the selection "
                  + "becomes a fixed 121-frame window you slide along the timeline (handles and "
                  + "the green zone move it as a whole). Empty = unlock.";
        const lblF = document.createElement("label"); lblF.append("Fix", fix);
        fields.append(lblS, lblE, lblF, len);
        sec.append(track, fields);
        pane.appendChild(sec);
        this._vtrim = { track, dimL, dimR, keep, hS, hE, inS, inE, len, fix };

        const dur = () => this._videoDur || 0;
        const clamp = (x) => Math.max(0, Math.min(x, dur()));
        // read the LIVE selection (mirrors the audio trim's v465 stale-closure fix):
        // _applyVideoTrim REPLACES ph_media_state, so a captured object would go stale.
        const curStart = () => { const c = this.state; return clamp(+(c && c.trimStart) || 0); };
        const curEnd = () => { const c = this.state; return clamp(dur() - (+(c && c.trimEnd) || 0)); };
        const parseT = (str) => {
            str = String(str == null ? "" : str).trim();
            if (str === "") return NaN;
            if (str.indexOf(":") >= 0) {
                const parts = str.split(":");
                const sec = parseFloat(parts.pop());
                let total = Number.isFinite(sec) ? sec : 0, mult = 60;
                while (parts.length) { const p = parseFloat(parts.pop()); if (Number.isFinite(p)) total += p * mult; mult *= 60; }
                return total;
            }
            return parseFloat(str);
        };

        // loop-preview: constrain the preview video's loop to [start,end]. ON disables the
        // native full-loop and wraps via timeupdate; OFF restores the full loop. Bounds are
        // read LIVE so the loop follows the handles as you drag.
        this._vloopOn = false;
        const setLoop = (onv) => {
            this._vloopOn = onv;
            loop.classList.toggle("playing", onv);
            if (onv) { vid.loop = false; try { vid.currentTime = curStart(); } catch (e) {} vid.play().catch(() => {}); }
            else { vid.loop = true; }
        };
        loop.onclick = () => setLoop(!this._vloopOn);
        vid.addEventListener("timeupdate", () => {
            if (!this._vloopOn) return;
            const s = curStart(), e = curEnd();
            if (e - s < 0.05) return;                       // no usable window yet
            if (vid.currentTime >= e - 0.015 || vid.currentTime < s - 0.05) {
                try { vid.currentTime = s; } catch (err) {}  // wrap to the live start
                if (vid.paused) vid.play().catch(() => {});
            }
        });
        vid.addEventListener("ended", () => {               // selection end == clip end
            if (!this._vloopOn) return;
            try { vid.currentTime = curStart(); } catch (e) {}
            vid.play().catch(() => {});
        });
        vid.addEventListener("pause", () => {               // a native pause drops the loop
            if (this._vloopOn && !vid.ended) { this._vloopOn = false; loop.classList.remove("playing"); vid.loop = true; }
        });

        const px2sec = (clientX) => {
            const r = track.getBoundingClientRect();
            return clamp(((clientX - r.left) / Math.max(1, r.width)) * dur());
        };
        const MIN_GAP = 1.0;                                // flat 1 s minimum (mirrors v470)
        const minGap = () => Math.min(MIN_GAP, dur());
        const startDrag = (which, ev) => {
            if (dur() <= 0) return;
            ev.preventDefault();
            // v625: with a fixed window, the grab point's offset into the window is kept,
            // so dragging the green zone doesn't jump the window under the cursor.
            const sx0 = ev.touches ? ev.touches[0].clientX : ev.clientX;
            const grabOff = px2sec(sx0) - curStart();
            // v471 nearest-edge: when the handles sit within a handle-width on screen, the
            // edge nearest the cursor follows it so a buried START/END can still be pulled out.
            const r0 = track.getBoundingClientRect();
            const gapPx = (r0.width > 0 && dur() > 0) ? ((curEnd() - curStart()) / dur()) * r0.width : 999;
            const nearest = gapPx < 14;
            const move = (e) => {
                const x = e.touches ? e.touches[0].clientX : e.clientX;
                const g = minGap();
                const px = px2sec(x);
                // v625: fixed-length window — the grabbed point follows the cursor, the
                // LENGTH IS THE LAW (_fixWindow clamps + frame-snaps). MIN_GAP does not
                // apply: whoever typed an explicit frame count meant it.
                const fx = this._vFixLen();
                if (fx) {
                    const s0 = (which === "s") ? px
                             : (which === "e") ? px - fx[0]
                             : px - grabOff;                     // whole-window drag
                    const w = _fixWindow(s0, fx[0], dur(), fx[1]);
                    this._applyVideoTrim(w[0], w[1], fx[1]);
                    return;
                }
                let st = curStart(), en = curEnd();
                if (nearest) {
                    if (Math.abs(px - st) <= Math.abs(px - en)) st = Math.min(Math.max(0, px), en - g);
                    else en = Math.max(Math.min(dur(), px), st + g);
                } else if (which === "s") {
                    st = Math.min(px, en - g);
                } else {
                    en = Math.max(px, st + g);
                }
                this._applyVideoTrim(st, en);
            };
            const up = () => {
                window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up);
                window.removeEventListener("touchmove", move); window.removeEventListener("touchend", up);
            };
            window.addEventListener("mousemove", move); window.addEventListener("mouseup", up);
            window.addEventListener("touchmove", move, { passive: false }); window.addEventListener("touchend", up);
        };
        hS.addEventListener("mousedown", (e) => startDrag("s", e));
        hE.addEventListener("mousedown", (e) => startDrag("e", e));
        hS.addEventListener("touchstart", (e) => startDrag("s", e), { passive: false });
        hE.addEventListener("touchstart", (e) => startDrag("e", e), { passive: false });
        // v625: the green keep zone drags the whole fixed window (only while fixed —
        // unfixed it stays pointer-events:none per CSS, so this never fires there).
        keep.addEventListener("mousedown", (e) => { if (this._vFixLen()) startDrag("w", e); });
        keep.addEventListener("touchstart", (e) => { if (this._vFixLen()) startDrag("w", e); }, { passive: false });
        // v625: type a frame count (121 + Enter) to lock; empty to unlock. Clicking the
        // length readout locks the CURRENT frame count / unlocks a set one.
        fix.addEventListener("change", () => {
            const n = parseInt(String(fix.value).trim(), 10);
            this._setVideoFix(Number.isFinite(n) && n > 0 ? n : 0);
        });
        fix.addEventListener("keydown", (e) => { if (e.key === "Enter") fix.blur(); });
        len.addEventListener("click", () => {
            if (this._vFixLen()) { this._setVideoFix(0); return; }
            const fps = this._vFixFps();
            if (fps > 0) this._setVideoFix(Math.max(1, Math.round((curEnd() - curStart()) * fps)));
        });

        const commit = (which) => {
            const d = dur(); if (d <= 0) return;
            let st = parseT(inS.value), en = parseT(inE.value);
            // v625: with a fixed window the edited field MOVES it (start pins the start,
            // end pins the end; the length is the law).
            const fx = this._vFixLen();
            if (fx) {
                const s0 = (which === "e" && Number.isFinite(en)) ? en - fx[0]
                         : (Number.isFinite(st) ? st : curStart());
                const w = _fixWindow(s0, fx[0], d, fx[1]);
                this._applyVideoTrim(w[0], w[1], fx[1]);
                return;
            }
            if (!Number.isFinite(st)) st = curStart();
            if (!Number.isFinite(en)) en = curEnd();
            const g = minGap();
            st = Math.max(0, Math.min(st, d)); en = Math.max(0, Math.min(en, d));
            if (en - st < g) { en = Math.min(d, st + g); if (en - st < g) st = Math.max(0, en - g); }
            this._applyVideoTrim(st, en);
        };
        inS.addEventListener("change", () => commit("s")); inE.addEventListener("change", () => commit("e"));
        inS.addEventListener("keydown", (e) => { if (e.key === "Enter") inS.blur(); });
        inE.addEventListener("keydown", (e) => { if (e.key === "Enter") inE.blur(); });

        this._videoDur = (isFinite(vid.duration) && vid.duration > 0) ? vid.duration : 0;
        if (!this._videoDur) {
            vid.addEventListener("loadedmetadata", () => {
                this._videoDur = (isFinite(vid.duration) && vid.duration > 0) ? vid.duration : 0;
                this._updateVideoTrimUI();
            }, { once: true });
        }
        this._updateVideoTrimUI();
    }

    // ── v625: fixed-length video trim ─────────────────────────────────────────
    // The frame count counts in NATIVE fps — the fps the backend's _slice_frames
    // cuts with (force_fps is only the output label). It comes from the listing
    // (v625 additive field); no fps in the listing -> the frame features degrade
    // honestly (no count shown, no lock available).
    _vFixFps() {
        const st = this.state;
        const f = (this._files || []).find((x) => x && x.name === (st && st.file));
        const v = f && +f.fps;
        return (Number.isFinite(v) && v > 0) ? v : 0;
    }

    // [windowSeconds, snapSeconds] when a fix is armed AND the fps is known; else null.
    _vFixLen() {
        const n = Math.round(+(this.state && this.state.trimFixFrames) || 0);
        const fps = this._vFixFps();
        return (n > 0 && fps > 0) ? [n / fps, 1 / fps] : null;
    }

    // Arm (n>0) or release (n=0) the fixed window. Stored on ph_media_state as
    // trimFixFrames — every state replace in this file spreads the old object,
    // so the fix survives trim commits and reloads. Arming immediately reshapes
    // the selection to exactly n frames from the current start (clamped).
    _setVideoFix(n) {
        const s0 = this.state; if (!s0 || s0.kind !== "video") return;
        this.state = { ...s0, trimFixFrames: Math.max(0, Math.round(+n || 0)) };
        const fx = this._vFixLen();
        if (fx) {
            const d = this._videoDur || 0;
            const cs = Math.max(0, Math.min(+(this.state.trimStart) || 0, d));
            const w = _fixWindow(cs, fx[0], d, fx[1]);
            this._applyVideoTrim(w[0], w[1], fx[1]);
        } else {
            this._updateVideoTrimUI();
        }
    }

    _applyVideoTrim(startAbs, endAbs, snap) {
        const s0 = this.state; if (!s0 || s0.kind !== "video") return;
        const d = this._videoDur || 0;
        // snap to tenths and store the END as the exact tail (mirrors the audio v469), so the
        // field, the loop window (curEnd = d - trimEnd) and the future backend slice agree.
        // v625: a fixed-length window passes snap = 1/native-fps instead — the backend's
        // _slice_frames rounds seconds*native_fps to frame indices, so frame-boundary
        // values make the frame count deterministic (121 means 121, not "about 121").
        const q = (snap && snap > 0) ? snap : 0.1;
        const r1 = (x) => Math.round(x / q) * q;
        let s = r1(Math.max(0, Math.min(startAbs, d)));
        let e = r1(Math.max(0, Math.min(endAbs, d)));
        if (e <= s) e = Math.min(d, r1(s + q));
        // store on ph_media_state; set state -> _syncRef carries trimStart/trimEnd into
        // media_ref as vtrimStart/vtrimEnd (v484 Stufe 2: backend slices the frames).
        this.state = { ...s0, trimStart: s, trimEnd: Math.max(0, d - e) };
        this._updateVideoTrimUI();
    }

    _updateVideoTrimUI() {
        const t = this._vtrim; if (!t) return;
        const a = this.state; const d = this._videoDur || 0;
        const fmtT = (x) => { x = Math.max(0, Math.round(x * 10) / 10); const m = Math.floor(x / 60); return m + ":" + (x - m * 60).toFixed(1).padStart(4, "0"); };
        const fmtLen = (x) => Math.max(0, x).toFixed(1);
        if (d <= 0) {
            t.inS.value = ""; t.inE.value = ""; t.inS.disabled = t.inE.disabled = true;
            t.len.textContent = "loading\u2026";
            t.dimL.style.width = "0%"; t.dimR.style.width = "0%";
            t.keep.style.left = "0%"; t.keep.style.right = "0%";
            t.hS.style.left = "0%"; t.hE.style.left = "100%";
            return;
        }
        t.inS.disabled = t.inE.disabled = false;
        const s = Math.max(0, Math.min(+(a && a.trimStart) || 0, d));
        const e = Math.max(0, Math.min(d - (+(a && a.trimEnd) || 0), d));
        const sp = (s / d) * 100, ep = (e / d) * 100;
        t.dimL.style.left = "0%"; t.dimL.style.width = sp + "%";
        t.dimR.style.left = ep + "%"; t.dimR.style.width = (100 - ep) + "%";
        t.keep.style.left = sp + "%"; t.keep.style.right = (100 - ep) + "%";
        t.hS.style.left = sp + "%"; t.hE.style.left = ep + "%";
        if (document.activeElement !== t.inS) t.inS.value = fmtT(s);
        if (document.activeElement !== t.inE) t.inE.value = fmtT(e);
        // v625: live frame count (native fps) beside the length; locked state reads
        // amber with the lock glyph, and the whole window becomes draggable (CSS .fixed).
        const fps = this._vFixFps();
        const fx = this._vFixLen();
        t.track.classList.toggle("fixed", !!fx);
        t.len.classList.toggle("on", !!fx);
        if (t.fix && document.activeElement !== t.fix) {
            t.fix.value = (this.state && +this.state.trimFixFrames > 0) ? String(this.state.trimFixFrames) : "";
            t.fix.disabled = !(fps > 0);
        }
        if (fx) {
            t.len.textContent = "\ud83d\udd12 " + Math.round(fx[0] * fps) + " fr \u00b7 " + fmtLen(fx[0]) + " s";
            t.len.title = "Fixed window \u2014 drag the green zone or a handle to slide it. Click to unlock.";
        } else {
            t.len.textContent = "\u2192 " + fmtLen(Math.max(0, e - s)) + " s"
                + (fps > 0 ? " \u00b7 " + Math.max(0, Math.round((e - s) * fps)) + " fr" : "");
            t.len.title = (fps > 0)
                ? "Click to LOCK this frame count \u2014 the selection then moves as a fixed window."
                : "";
        }
    }

    _clearAudio() {
        this._stopAudioHover();
        this.audioSel = null;
        this.audioOn = false;                                   // removing audio collapses the pane
        this._syncAudioToggle();
        if (this.folder && this._files) this.renderGrid();      // drop the green ✓ mark in the grid
        this._renderAudioPane();
    }

    // ♪ Audio: ON/OFF — like ▦ Batch / ▶ Proc, but NOT an exclusive output mode: it
    // arms/reveals the audio companion. Turning it ON jumps the grid to the chosen
    // audio's folder (marked there), mirroring Batch Frames's source-folder jump.
    // The chosen audio is remembered while OFF.
    _toggleAudio() {
        const on = !this.audioOn;
        this.audioOn = on;
        const a = this.audioSel;
        if (on && a && a.folder && !this._samePath(this.folder, a.folder)) {
            this.setFolder(a.folder);   // setFolder also re-renders the preview + grid
        } else {
            // v477: toggling audio visibility only changes the AUDIO pane. When a visual is
            // showing, re-render just that pane — don't rebuild the <video> (that churn
            // re-streams the clip and is the likely cause of the toggle needing a reload to
            // take effect). With no visual (audio-only), _renderPreview handles the collapse.
            const vis = this.state;
            if (vis && vis.folder && vis.file) this._renderAudioPane();
            else this._renderPreview();
            if (this.folder && this._files) this.renderGrid();
        }
        this._syncAudioToggle();
    }

    _syncAudioToggle() {
        const at = this.audioToggleEl;
        if (!at) return;
        const on = this.audioOn;
        at.classList.toggle("on", on);
        at.textContent = on ? "♪ Audio: ON" : "♪ Audio: OFF";
        at.title = on
            ? "Audio companion is ON — shown in the Selection and carried on the AUDIO / video_audio outputs. Click 📁 in the pane to jump to its folder; click here to set it aside (remembered)."
            : "Audio companion is OFF. Click an audio tile to pair an audio with your image/video; it is remembered even when OFF.";
    }

    // v458: mark BOTH slots in the grid — the visual selection (orange .sel) and the
    // audio selection (green .audsel) — but only when the slot's file is in the folder
    // currently shown, so a same-named file in another folder is not falsely marked.
    _markGridSelection() {
        if (!this.gridEl) return;
        const vis = this.state, aud = this.audioSel, bf = this.folder;
        const visFile = (vis && vis.file && this._samePath(vis.folder || "", bf)) ? vis.file : null;
        const audFile = (aud && !aud.fromVideo && aud.file && this._samePath(aud.folder || "", bf)) ? aud.file : null;
        this.gridEl.querySelectorAll(".ph-media-tile").forEach((t) => {
            t.classList.toggle("sel", t.dataset.file === visFile);
            t.classList.toggle("audsel", t.dataset.file === audFile);
        });
    }

    // v458: a node saved under v457 may have stored an audio-only pick in the VISUAL
    // slot (ph_media_state.kind === "audio"). Move it into the audio slot once so the
    // visual slot only ever holds images/videos from here on.
    _migrateAudioState() {
        const p = this.node.properties || {};
        const s = p.ph_media_state;
        if (s && s.kind === "audio" && s.file && !(p.ph_media_audio && p.ph_media_audio.file)) {
            this.audioSel = { folder: s.folder, file: s.file, mtime: s.mtime || 0 };
            this.audioOn = true;
            if (typeof p.ph_media_browse !== "string" && s.folder) p.ph_media_browse = s.folder;
            this.state = { folder: "", file: "", kind: "" };
        }
    }

    // ── image-batch preview (Selection column when Batch is ON) ──────────────
    // The selection pipeline below MIRRORS nodes/ph_media_util.select_frames so
    // the preview's frame count / order / first-frame size are EXACTLY what
    // load() will load. Keep the two in lock-step if either changes.
    _numWidget(name) {
        const w = this.node?.widgets?.find((x) => x.name === name);
        const v = w ? parseInt(w.value, 10) : 0;
        return Number.isFinite(v) ? v : 0;
    }

    _samePath(a, b) {
        const norm = (p) => String(p || "").replace(/[\\/]+/g, "\\").replace(/\\+$/, "").toLowerCase();
        return !!a && norm(a) === norm(b);
    }

    async _listFolder(folder) {
        if (!folder) return null;
        if (this._listCache && this._listCache.folder === folder) return this._listCache.files;
        let files = null;
        try {
            const r = await api.fetchApi("/uls/media/list?folder=" + encodeURIComponent(folder));
            const d = r && await r.json();
            files = (d && d.ok) ? (d.files || []) : null;
        } catch (e) { files = null; }
        this._listCache = { folder, files };
        return files;
    }

    // fnmatch.translate-equivalent for the globs folder filters use (* ? [..]).
    _fnmatchToRe(pat) {
        let re = "";
        for (let i = 0; i < pat.length; i++) {
            const c = pat[i];
            if (c === "*") re += ".*";
            else if (c === "?") re += ".";
            else if (c === "[") {
                let j = i + 1, neg = "";
                if (pat[j] === "!") { neg = "^"; j++; }
                let cls = "";
                while (j < pat.length && pat[j] !== "]") { cls += pat[j].replace(/[\\\]^]/g, "\\$&"); j++; }
                if (j >= pat.length) { re += "\\["; }
                else { re += "[" + neg + cls + "]"; i = j; }
            } else re += c.replace(/[.\\+^$(){}|/]/g, "\\$&");
        }
        return new RegExp("^" + re + "$", "i");
    }

    // PH_MATCH_BEGIN — parity mirror of ph_media_util.match_names. The v528 guard
    // extracts this fenced region, runs it in node on the SAME fixtures as the
    // Python side, and requires identical results (Messen schlaegt Glauben).
    _matchNames(names, expr) {
        const e = String(expr == null ? "*" : expr).trim();
        if (e === "" || e === "*") return names.slice();
        const low = names.map((n) => [n, String(n).toLowerCase()]);
        if (e.toLowerCase().startsWith("re:")) {
            let rx;
            try { rx = new RegExp(e.slice(3), "i"); }
            catch (err) { return []; }               // broken regex selects nothing (visibly)
            return low.filter(([n, _ln]) => rx.test(String(n))).map(([n]) => n);
        }
        const tokMatch = (tok, ln) => {
            if (/[*?\[]/.test(tok)) return this._fnmatchToRe(tok).test(ln);
            return ln.includes(tok);
        };
        const pos = [], neg = [];
        for (const raw of e.split(",")) {
            const t = raw.trim().toLowerCase();
            if (!t) continue;
            if (t.startsWith("!")) { const c = t.replace(/^!+/, "").trim(); if (c) neg.push(c); }
            else pos.push(t);
        }
        const kept = [];
        for (const [n, ln] of low) {
            let ok = pos.length ? pos.some((t) => tokMatch(t, ln)) : true;
            if (ok && neg.length && neg.some((t) => tokMatch(t, ln))) ok = false;
            if (ok) kept.push(n);
        }
        return kept;
    }
    // PH_MATCH_END

    _filterNames(names, pattern) {
        const pat = ((pattern || "*").trim()) || "*";
        if (pat === "*") return names.slice();
        const re = this._fnmatchToRe(pat.toLowerCase());
        return names.filter((n) => re.test(String(n).toLowerCase()));
    }

    _naturalCmp(a, b) {
        const ax = String(a).toLowerCase().split(/(\d+)/);
        const bx = String(b).toLowerCase().split(/(\d+)/);
        const n = Math.max(ax.length, bx.length);
        for (let i = 0; i < n; i++) {
            const at = ax[i] ?? "", bt = bx[i] ?? "";
            if (i % 2 === 1) { const d = (parseInt(at, 10) || 0) - (parseInt(bt, 10) || 0); if (d) return d; }
            else { if (at < bt) return -1; if (at > bt) return 1; }
        }
        return 0;
    }

    _orderNames(names, mode, byName) {
        const arr = names.slice();
        if (mode === "name (literal)") return arr.sort();   // lexicographic — matches Python sorted()
        if (mode === "mtime (oldest first)" || mode === "created") {
            return arr.sort((a, b) => ((byName[a]?.mtime || 0) - (byName[b]?.mtime || 0)) || this._naturalCmp(a, b));
        }
        return arr.sort((a, b) => this._naturalCmp(a, b));  // name (natural) — default
    }

    // v661: the slice stage on its own (skip -> every_nth -> cap), so a CHECKED
    // set can be sliced without running it through the name filter first — the
    // backend's select_slice does exactly this for an explicit selection.
    _sliceFrames(ordered, cfg, skip, cap) {
        const HARD = 2000;
        const s = Math.max(0, parseInt(skip, 10) || 0);
        const nth = Math.max(1, parseInt(cfg.every_nth, 10) || 1);
        const sliced = ordered.slice(s).filter((_, i) => i % nth === 0);
        let lim = (parseInt(cap, 10) || 0) > 0 ? parseInt(cap, 10) : HARD;
        lim = Math.min(lim, HARD);
        return sliced.slice(0, lim);
    }

    // filter -> order -> skip -> every_nth -> cap (== select_frames / select_slice)
    _selectFrames(names, cfg, skip, cap, byName) {
        const ordered = this._orderNames(this._filterNames(names, cfg.name_filter || "*"),
                                         cfg.sort_mode || "name (natural)", byName);
        return this._sliceFrames(ordered, cfg, skip, cap);
    }

    async _renderBatchPreview(cfg, seekName) {
        const host = this.previewMediaEl, cap = this.previewCapEl;
        if (!host) return;
        host.innerHTML = `<div class="ph-media-empty">Reading batch…</div>`;
        if (cap) { cap.textContent = ""; cap.title = cfg.source || ""; }
        // reuse the current grid listing only when it ACTUALLY belongs to the
        // source folder. During a batch-enable navigation this.folder is already
        // the source but this._files may still be the previous folder's listing
        // (refreshGrid is async) -> fetch fresh instead of filtering stale files.
        const files = (this._samePath(this._filesFolder, cfg.source) && this._files)
            ? this._files : await this._listFolder(cfg.source);
        // a newer render may have superseded this async one, or batch turned off
        const now = this._readCfg();
        if (!now.enabled || !this._samePath(now.source, cfg.source)) return;
        if (!files) { host.innerHTML = `<div class="ph-media-empty">Batch source unreadable</div>`; return; }
        const byName = {}; for (const f of files) byName[f.name] = f;
        // v661 (B-01): mirror the backend's v528 precedence. An explicit CHECKED set
        // wins over the name filter (_load_image_batch / _proc_resolve_files both take
        // that branch), so this preview has to judge the SAME set. It used to run the
        // rule pipeline over the WHOLE folder and then warn about sizes the run never
        // touches — "13 frames · 10 differ" while 10 uniform files were checked and the
        // run went through fine (measured 2026-07-19). skip / every-nth / cap still
        // slice ON the checked set, exactly like select_slice does server-side.
        const allNames = files.map((f) => f.name);
        const checkedSet = (typeof this._selNames === "function") ? (this._selNames() || []) : [];
        const chosen = checkedSet.length
            ? this._sliceFrames(this._orderNames(allNames.filter((nm) => checkedSet.includes(nm)),
                                                 cfg.sort_mode || "name (natural)", byName),
                                cfg, this._numWidget("frame_skip"), this._numWidget("frame_load_cap"))
            : this._selectFrames(allNames, cfg,
                                 this._numWidget("frame_skip"), this._numWidget("frame_load_cap"), byName);
        const n = chosen.length;
        const dims = chosen.map((nm) => [byName[nm]?.w || 0, byName[nm]?.h || 0]);
        const tgt = dims.length ? dims[0] : [0, 0];
        const differ = dims.filter((d) => d[0] !== tgt[0] || d[1] !== tgt[1]).length;

        host.innerHTML = "";
        if (!n) {
            const head = document.createElement("div");
            head.className = "ph-batch-prev-head ph-batch-prev-warn";
            head.textContent = `⚠ filter "${cfg.name_filter || "*"}" matches nothing here`;
            host.appendChild(head);
        } else if (differ) {
            // sizes differ -> none(strict) will refuse; a centered warning, no stage.
            const head = document.createElement("div");
            head.className = "ph-batch-prev-head ph-batch-prev-warn";
            head.innerHTML = `▦ ${n} frame${n > 1 ? "s" : ""} · sizes differ<br>⚠ ${differ} differ — none(strict) will refuse`;
            host.appendChild(head);
        } else {
            // Uniform set -> show the first frame CENTERED, like the normal Selection:
            // it fills the preview's vertical space, the size sits top-left as a badge
            // (reusing .ph-dim-prev), and the frame count goes into the caption below.
            // A >= 2-frame set IS a film sequence, so it also gets the play/pause
            // overlay + flipbook; the frames themselves stay marked in the main grid.
            const urls = chosen.map((nm) => this._thumbURLFor(cfg.source, byName[nm]));
            const playable = (n >= 2 && differ === 0);
            const wrap = document.createElement("div"); wrap.className = "ph-batch-stagewrap";
            const stage = document.createElement("img");
            stage.className = "ph-batch-stage"; stage.loading = "lazy"; stage.alt = "";
            stage.src = urls[0];
            wrap.appendChild(stage);
            let ov = null;
            if (playable) {
                ov = document.createElement("div"); ov.className = "ph-batch-playov"; ov.innerHTML = PLAY_SVG;
                wrap.appendChild(ov);
                wrap.addEventListener("click", (e) => { e.stopPropagation(); this._batchAnimToggle(); });
            }
            host.appendChild(wrap);
            this._setPrevDim(tgt[0], tgt[1]);   // top-left size badge, like the normal view
            if (playable) {
                this._batchAnim = { timer: null, idx: 0, urls, stage, ov };
                // a member click jumps the freshly built (paused) flipbook to its frame
                if (seekName) { const si = chosen.indexOf(seekName); if (si >= 0) this._batchAnimShow(si); }
            }
        }
        if (cap) {
            const first = chosen[0] || "", last = chosen[n - 1] || "";
            const range = n > 1 ? `${first} … ${last}` : first;
            cap.textContent = n ? `▦ ${n} frame${n > 1 ? "s" : ""} · ${range}` : (cfg.source || "");
            cap.title = cfg.source || "";
        }
    }

    // ── batch animation preview (flipbook) ──────────────────────────────────
    // A uniform-size batch is a film sequence; play it back as a looping
    // flipbook of the frame thumbnails at force_fps (fallback 16). Pure client
    // side — same ordered frame list the load() path uses, no decode here.
    _forceFps() {
        const w = this.node?.widgets?.find((x) => x.name === "force_fps");
        const v = w ? parseFloat(w.value) : 0;
        return Number.isFinite(v) ? v : 0;
    }

    _batchAnimStop() {
        const a = this._batchAnim;
        if (a && a.timer) clearInterval(a.timer);
        this._batchAnim = null;
    }

    // swap the stage to frame i. v434: a member click rebuilds the (paused) preview
    // via _renderBatchPreview(cfg, seekName), which calls this to land on the frame;
    // the old grid-driven per-tile seek is gone (browsing no longer pokes the stage).
    _batchAnimShow(i) {
        const a = this._batchAnim; if (!a || !a.urls.length) return;
        a.idx = ((i % a.urls.length) + a.urls.length) % a.urls.length;
        if (a.stage) a.stage.src = a.urls[a.idx];
    }

    _batchAnimIcon(playing) {
        const a = this._batchAnim; if (!a || !a.ov) return;
        a.ov.innerHTML = playing ? PAUSE_SVG : PLAY_SVG;
        a.ov.classList.toggle("playing", playing);
    }

    _batchAnimToggle() {
        const a = this._batchAnim;
        if (!a || !a.urls || a.urls.length < 2) return;
        if (a.timer) {                            // playing -> pause, hold current frame
            clearInterval(a.timer); a.timer = null;
            this._batchAnimIcon(false);
            return;
        }
        const fps = this._forceFps();             // live value at play time
        const period = Math.max(20, Math.round(1000 / (fps > 0 ? fps : 16)));   // 16 = WAN default
        a.urls.forEach((u) => { const im = new Image(); im.src = u; });         // warm the cache
        a.timer = setInterval(() => {
            // stage detached (re-render / node removed) -> self-stop, no orphan timer
            if (!a.stage || !a.stage.isConnected) { this._batchAnimStop(); return; }
            this._batchAnimShow(a.idx + 1);
        }, period);
        this._batchAnimIcon(true);
    }

    // ── pixel-dimension badges (W x H) ──────────────────────────────────────
    // Convention: width x height (matches the node's width/height outputs and
    // every tool, e.g. 1920x1080). Values come from /uls/media/list (image
    // header read / cv2 probe); the browser's authoritative naturalWidth /
    // videoWidth refines them once the real media loads (see callers below).
    _dimText(w, h) { return (w && h) ? `${w}\u00d7${h}` : ""; }          // 512×512 (compact)
    _dimTextPx(w, h) { return (w && h) ? `${w} \u00d7 ${h} px` : ""; }   // 512 × 512 px (prominent)

    _setTileDim(tile, w, h) {
        if (!tile || !w || !h) return;
        let el = tile.querySelector(".ph-dim");
        if (!el) { el = document.createElement("div"); el.className = "ph-dim"; tile.appendChild(el); }
        el.textContent = this._dimText(w, h);
    }

    _setPrevDim(w, h) {
        if (!this.previewMediaEl || !w || !h) return;
        let el = this.previewMediaEl.querySelector(".ph-dim-prev");
        if (!el) { el = document.createElement("div"); el.className = "ph-dim-prev"; this.previewMediaEl.appendChild(el); }
        el.textContent = this._dimTextPx(w, h);
    }

    // v684 (L2): one IntersectionObserver per loader UI, created lazily and
    // rooted in the grid's own scroller. Returns null where the API is absent
    // -- every call site then assigns src directly, i.e. the old behaviour.
    _thumbObserver() {
        if (this._thumbObs !== undefined) return this._thumbObs;
        let obs = null;
        try {
            if (typeof IntersectionObserver === "function" && this.gridEl) {
                obs = new IntersectionObserver(
                    (entries) => _thumbReveal(entries, (el) => obs.unobserve(el)),
                    { root: this.gridEl, rootMargin: THUMB_MARGIN, threshold: 0 });
            }
        } catch (e) { obs = null; }
        this._thumbObs = obs;
        return obs;
    }

    _makeTile(f) {
        const marks = this._gridMarks;                       // {name: indexInBatch} in batch mode, else null
        const inBatch = !!(marks && (f.name in marks));
        const bf = this.folder;
        const isVis = !!(this.state && this.state.file === f.name && this._samePath(this.state.folder || "", bf));
        const isAud = !!(this.audioSel && this.audioSel.file === f.name && this._samePath(this.audioSel.folder || "", bf));
        const tile = document.createElement("div");
        tile.className = "ph-media-tile"
            + (isVis ? " sel" : "")        // visual selection (orange) — folder-checked (v458)
            + (isAud ? " audsel" : "")     // v458: audio selection (green ✓)
            + (inBatch ? " picked" : "");
        tile.dataset.file = f.name;
        tile.dataset.kind = f.kind || "";
        // v644: tiles are drag SOURCES carrying the FULL path -- drop one onto
        // the Batch Pipeline Source (video_path) or anywhere that takes a path.
        tile.draggable = true;
        tile.addEventListener("dragstart", (ev) => {
            try {
                const sep = bf.includes("\\") ? "\\" : "/";
                ev.dataTransfer.setData("text/plain", bf.replace(/[\\/]+$/, "") + sep + f.name);
                ev.dataTransfer.effectAllowed = "copy";
            } catch (e) { /* drag stays inert */ }
        });
        if (f.kind === "audio") {
            // v457: audio tiles have no thumbnail to fetch — draw the deterministic
            // faux-waveform card + a green ♪ badge (Selection reuses the same card).
            tile.appendChild(this._makeAudioCard(f));
            const b = document.createElement("div"); b.className = "ph-aud"; b.textContent = "♪"; tile.appendChild(b);
        } else {
            const img = document.createElement("img");
            img.loading = "lazy";
            // v684 (L2): the URL is PARKED, not assigned. Native loading="lazy"
            // does not help inside this scroller (the node's DOM widget is
            // absolutely positioned on the canvas, so the browser treats the
            // whole page as visible and fetches every tile at once). The
            // observer below assigns src when the tile is actually near view;
            // without an observer the URL is assigned straight away, so an old
            // browser simply behaves as before.
            const url = this._isGif(f) ? this._fileURL(f) : this._thumbURL(f);
            const obs = this._thumbObserver();
            if (obs) { img.dataset.phSrc = url; obs.observe(img); }
            else { img.src = url; }
            tile.classList.add("ph-thumb-wait");            // v692: show that work is pending
            img.onload = () => tile.classList.remove("ph-thumb-wait");
            img.onerror = () => {
                tile.classList.remove("ph-thumb-wait");     // v692: stop spinning on a dead thumb
                img.style.display = "none";
                const ph = document.createElement("div"); ph.className = "ph-ph";
                ph.textContent = f.kind === "video" ? "🎞" : "🖼";
                tile.insertBefore(ph, tile.firstChild);
            };
            tile.appendChild(img);
            if (f.kind === "video") { const b = document.createElement("div"); b.className = "ph-vid"; b.textContent = "▶"; tile.appendChild(b); }
        }
        const nm = document.createElement("div"); nm.className = "ph-name"; nm.textContent = f.name; tile.appendChild(nm);
        if (f.w && f.h) this._setTileDim(tile, f.w, f.h);   // static dims from the listing (refined on play, below)
        if (f.kind !== "audio") {
            // v528: the ○ check circle — THE per-tile selection control. Empty
            // dashed = unchecked, green ✓ = in the batch set. stopPropagation so
            // checking never disturbs the free-browse click on the tile body.
            const sc = document.createElement("div");
            sc.className = "ph-selc" + (inBatch ? " on" : "");
            sc.textContent = inBatch ? "✓" : "";
            sc.title = inBatch
                ? "Checked for the batch — click to uncheck (Ctrl+A all · Ctrl+X none · Ctrl+I invert)"
                : "Click to check this file for the batch (Ctrl+A all · Ctrl+X none · Ctrl+I invert)";
            sc.onclick = (e) => { e.stopPropagation(); this._toggleSelName(f.name); };
            tile.appendChild(sc);
        }
        // The grid is always a free browser: click ANY tile to select it. select()
        // -> _renderPreview then decides what the Selection column shows — a batch
        // member shows the whole batch and jumps the flipbook to that frame; anything
        // else (a non-member here, or a file in another folder) shows that one file.
        // Whether the NODE emits the batch or a single file is the ▦ Batch toggle, not
        // this click — so browsing never disturbs an armed batch.
        tile.onclick = () => this.select(f);
        tile.onmouseenter = () => {
            if (f.kind === "video") this._playInTile(tile, f, (w, h) => this._setTileDim(tile, w, h));
            else if (f.kind === "audio") this._playAudioHover(f);
            else this.showPop(f, tile);
        };
        tile.onmouseleave = () => {
            if (f.kind === "video") this._stopInTile(tile);
            else if (f.kind === "audio") this._stopAudioHover();
            else this.hidePop();
        };
        return tile;
    }

    _playInTile(tile, f, onDims) {
        if (tile.querySelector("video")) return;
        const v = document.createElement("video");
        v.className = "ph-tile-video";
        v.src = this._fileURL(f); v.poster = this._thumbURL(f);
        v.muted = true; v.loop = true; v.autoplay = true; v.playsInline = true;
        if (onDims) v.addEventListener("loadedmetadata", () => {
            if (v.videoWidth && v.videoHeight) onDims(v.videoWidth, v.videoHeight);
        });
        const img = tile.querySelector("img"); if (img) img.style.visibility = "hidden";
        tile.appendChild(v);
        const p = v.play && v.play(); if (p && p.catch) p.catch(() => {});
    }

    _stopInTile(tile) {
        const v = tile.querySelector("video"); if (v) v.remove();
        const img = tile.querySelector("img"); if (img) img.style.visibility = "";
    }

    // v457: a stable, decode-free faux-waveform drawn from the filename hash, so
    // every audio file gets its own recognisable shape (purely decorative — the
    // Selection column reuses this exact card above the native <audio> controls).
    _makeAudioCard(f) {
        const card = document.createElement("div");
        card.className = "ph-aud-card";
        const N = 21, hsh = this._hashStr(f.name || "");
        let bars = "";
        for (let i = 0; i < N; i++) {
            const r = (Math.imul(hsh ^ (i + 1), 2654435761) >>> 8) & 0xff;   // stable per-bar pseudo-random
            const pct = 14 + (r / 255) * 72;                                 // 14%..86% of the band
            const slot = 100 / N, bw = slot * 0.52, x = i * slot + (slot - bw) / 2;
            bars += `<rect x="${x.toFixed(2)}" y="${((100 - pct) / 2).toFixed(2)}" `
                  + `width="${bw.toFixed(2)}" height="${pct.toFixed(2)}" rx="${(bw * 0.35).toFixed(2)}"></rect>`;
        }
        card.innerHTML = `<svg class="ph-aud-wave" viewBox="0 0 100 100" preserveAspectRatio="none">${bars}</svg>`;
        return card;
    }

    _hashStr(s) {
        let h = 2166136261 >>> 0;                        // FNV-1a, deterministic, no deps
        for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
        return h >>> 0;
    }

    // v457: hover-preview audio. ONE clip at a time — a new hover, a mouseleave, or
    // a click (via select()) all stop the previous. Audible by design, so the browser
    // autoplay policy may reject the very first play before any user gesture on the
    // page; .catch swallows that silently (the tile still shows; the Selection column
    // exposes full play/seek controls regardless).
    _playAudioHover(f) {
        this._stopAudioHover();
        const a = new Audio(this._fileURL(f));
        a.loop = false;
        this._hoverAudio = a;
        const p = a.play && a.play(); if (p && p.catch) p.catch(() => {});
    }

    _stopAudioHover() {
        if (this._hoverAudio) {
            try { this._hoverAudio.pause(); } catch (e) { /* ignore */ }
            this._hoverAudio.src = "";
            this._hoverAudio = null;
        }
    }

    // v656: cross-view busy signal. The grid's own "Loading…" text is invisible
    // in Solo view (the grid is display:none), so slow folder pins and uploads
    // looked like nothing was happening. Counter-based: overlapping phases
    // (upload -> re-read) keep the ring up until the LAST one finishes. The
    // silent focus re-read stays silent by design and never touches this.
    _busyOn(label) {
        this._busyCount = (this._busyCount || 0) + 1;
        const el = this.root && this.root.querySelector(".ph-media-busy");
        if (!el) return;
        if (label) el.querySelector(".ph-busy-label").textContent = label;
        el.classList.add("on");
    }

    _busyOff() {
        this._busyCount = Math.max(0, (this._busyCount || 0) - 1);
        if (this._busyCount) return;
        const el = this.root && this.root.querySelector(".ph-media-busy");
        if (el) el.classList.remove("on");
    }

    async refreshGrid(keepPage = false) {
        if (!this.folder) {
            this.gridEl.innerHTML = `<div class="ph-media-empty">Pick a folder to see its images and videos.</div>`;
            this.pagerEl.innerHTML = ""; return;
        }
        const target = this.folder;   // capture: a slow listing of a folder we've since LEFT must not clobber the grid
        this.gridEl.textContent = "Loading…"; this.pagerEl.innerHTML = "";
        this._busyOn("Reading folder…");
        // v657: listing sequence — every listing fetch (here and the focus
        // probe) takes a ticket; only the LATEST-STARTED fetch may apply its
        // response. Kills the drop race: the focus probe fired by the drop
        // itself used to return a PRE-upload listing after the refresh and
        // wipe the new tile (first drop "didn't take", second was debounced).
        const seq = (this._listSeq = (this._listSeq || 0) + 1);
        try {
            let d = null;
            try {
                const r = await api.fetchApi("/uls/media/list?dims=0&folder=" + encodeURIComponent(target));
                if (r && r.ok) d = await r.json();
            } catch (e) { /* ignore */ }
            if (seq !== this._listSeq) return;    // a newer listing fetch started -> this one is stale
            if (target !== this.folder) return;   // folder switched mid-flight -> this response is stale, drop it
            if (!d || !d.ok) { this.gridEl.innerHTML = `<div class="ph-media-empty">Cannot read this folder.</div>`; return; }
            this._files = d.files || [];
            this._filesFolder = target;   // label the files by the folder actually fetched, not the current one
            // v655: keepPage holds the current page across a re-read (renderGrid
            // clamps if the folder shrank); every other path starts at page 1.
            this._page = keepPage ? this._page : 0;
            this.renderGrid();
        } finally {
            this._busyOff();
        }
    }

    // v655: silent focus re-read — fetch the pinned folder's listing and only
    // re-render when it actually changed (no "Loading…" flicker on a no-op).
    // Debounced: at most one probe per 5 s of window-focus events.
    async _focusReread() {
        if (!this.folder) return;
        if (this._busyCount) return;   // an upload/refresh is running -- its own re-read is authoritative
        const now = Date.now();
        if (now - (this._lastFocusReread || 0) < 5000) return;
        this._lastFocusReread = now;
        const target = this.folder;
        const seq = (this._listSeq = (this._listSeq || 0) + 1);   // v657: same ticket rule as refreshGrid
        let d = null;
        try {
            const r = await api.fetchApi("/uls/media/list?dims=0&folder=" + encodeURIComponent(target));
            if (r && r.ok) d = await r.json();
        } catch (e) { return; }
        if (seq !== this._listSeq) return;   // a newer listing fetch started -> stale, drop
        if (!d || !d.ok || target !== this.folder) return;
        if (_filesSig(d.files) === _filesSig(this._files)) return;   // nothing new
        this._files = d.files || [];
        this._filesFolder = target;
        this.renderGrid();   // page + name-based selection survive on their own
    }

    // v683 — DEFERRED DIMENSIONS. The listing itself no longer opens a single
    // file (see uls_routes._scan_media_fast): a folder of a thousand items used
    // to cost a thousand file opens before anything appeared, in Solo view too,
    // where nothing but the selected file is visible. So the sizes are fetched
    // for what is ACTUALLY on screen -- the selected file always, the current
    // page only when the grid is showing -- and filled in afterwards.
    //
    // TERMINATION: every name asked for is marked _dimsTried, and only untried
    // names are ever requested. A failed probe therefore costs one attempt, not
    // an endless re-render loop (renderGrid calls back into here).
    async _syncDims() {
        if (!this.folder || !Array.isArray(this._files) || !this._files.length) return;
        const target = this.folder;
        const byName = new Map(this._files.map((f) => [f.name, f]));
        const want = new Set();
        const sel = this.state && this.state.file;
        if (sel) want.add(sel);                       // the selection needs w/h AND fps (fixed trim)
        if (!this.view.solo) {                        // the grid is drawn -> the page on screen
            const start = this._page * GRID_PAGE;
            for (const f of this._files.slice(start, start + GRID_PAGE)) want.add(f.name);
        }
        const names = [...want].filter((n) => {
            const f = byName.get(n);
            return f && f.w == null && !f._dimsTried;
        });
        if (!names.length) return;
        for (const n of names) { const f = byName.get(n); if (f) f._dimsTried = true; }
        const seq = this._listSeq;
        let d = null;
        try {
            const r = await api.fetchApi("/uls/media/dims?folder=" + encodeURIComponent(target) +
                                         "&files=" + encodeURIComponent(names.join("|")));
            if (r && r.ok) d = await r.json();
        } catch (e) { return; }
        if (seq !== this._listSeq || target !== this.folder) return;   // stale, drop (v657 ticket rule)
        if (!d || !d.ok || !d.dims) return;
        let changed = false;
        for (const [name, dim] of Object.entries(d.dims)) {
            const f = byName.get(name);
            if (!f || !dim) continue;
            if (dim.w != null) { f.w = dim.w; f.h = dim.h; changed = true; }
            if (dim.fps != null) { f.fps = dim.fps; changed = true; }
        }
        if (!changed) return;
        this.renderGrid();                    // badges on the tiles
        try { this._renderPreview(); } catch (e) { /* preview may not be up yet */ }
    }

    // which frames the active batch will load, as {name: indexInBatch}; null when
    // batch is off or the grid is not showing the batch source. Mirrors load()'s
    // select_frames so the marked tiles are EXACTLY the frames that get loaded.
    _gridBatchMarks() {
        // v528: the explicit checked set is THE mark source whenever the grid shows
        // the batch source — in every mode, ON or OFF, so composing a selection is
        // always visible. Without a set, an armed Frames batch falls back to its
        // rule-derived members (legacy behaviour, e.g. loaded sequences).
        const cfg = this._readCfg();
        const sel = this._selNames();
        if (sel.length && cfg.source && this._samePath(this.folder, cfg.source)) {
            const map = {}; sel.forEach((nm, i) => { map[nm] = i; });
            return map;
        }
        const framesOn = cfg.enabled && (cfg.mode ? cfg.mode === "frames" : true);
        if (!framesOn || !cfg.source || !this._samePath(this.folder, cfg.source)) return null;
        const files = this._files || [];
        const byName = {}; for (const f of files) byName[f.name] = f;
        const chosen = this._selectFrames(files.map((f) => f.name), cfg,
                                          this._numWidget("frame_skip"), this._numWidget("frame_load_cap"), byName);
        const map = {}; chosen.forEach((nm, i) => { map[nm] = i; });
        return map;
    }

    renderGrid() {
        const PAGE = GRID_PAGE;
        const files = this._files || [];
        const total = files.length;
        // v683 (L1) — in Solo the grid is display:none, but it was still BUILT:
        // every tile constructed, every thumbnail requested from the server.
        // Now it is not built at all; the flag makes the swap back rebuild it.
        // _files stays the truth, so filters, checkmarks and the batch config
        // keep working off it while the DOM is absent.
        if (this.view.solo) { this._gridStale = true; this._syncDims(); return; }
        this._gridStale = false;
        this._gridMarks = this._gridBatchMarks();   // {name: idx} when batch is on & grid is on-source, else null
        if (!total) {
            this.gridEl.innerHTML = `<div class="ph-media-empty">No images or videos in this folder.</div>`;
            this.pagerEl.innerHTML = ""; this._layoutGrid(); return;
        }
        const pages = Math.ceil(total / PAGE);
        if (this._page >= pages) this._page = pages - 1;
        if (this._page < 0) this._page = 0;
        const start = this._page * PAGE;
        // v684: the tiles about to be discarded are still being watched --
        // drop the whole watch set before the DOM under it disappears, or
        // the observer keeps every removed <img> reachable (v624 leak law).
        try { this._thumbObs?.disconnect(); } catch (e) { /* ignore */ }
        this.gridEl.innerHTML = "";
        for (const f of files.slice(start, start + PAGE)) this.gridEl.appendChild(this._makeTile(f));
        this.gridEl.scrollTop = 0;
        this._layoutGrid();

        this.pagerEl.innerHTML = "";
        this.pagerEl.style.display = "flex";
        if (pages > 1) {
            const prev = document.createElement("button"); prev.className = "ph-media-btn";
            prev.textContent = "◀ Back"; prev.disabled = this._page === 0;
            prev.onclick = () => { if (this._page > 0) { this._page--; this.renderGrid(); } };
            const info = document.createElement("div"); info.className = "ph-media-pageinfo";
            info.textContent = `Page ${this._page + 1} / ${pages} · ${total} files`;
            const next = document.createElement("button"); next.className = "ph-media-btn";
            next.textContent = "Next ▶"; next.disabled = this._page >= pages - 1;
            next.onclick = () => { if (this._page < pages - 1) { this._page++; this.renderGrid(); } };
            this.pagerEl.append(prev, info, next);
        } else {
            const info = document.createElement("div"); info.className = "ph-media-pageinfo";
            info.textContent = `${total} file${total === 1 ? "" : "s"}`;
            this.pagerEl.appendChild(info);
        }
        // v683: renderGrid is the funnel EVERY path goes through (refresh,
        // focus probe, page turn, filter, solo swap-back), so the deferred
        // dimension fetch hangs here once instead of at six call sites.
        this._syncDims();
    }

    _layoutGrid() {
        const grid = this.gridEl;
        if (!grid) return;
        // Message/empty state (no tiles) -> a single full-width column, so a
        // notice isn't squeezed into one tile cell.
        if (!grid.querySelector(".ph-media-tile")) {
            grid.style.gridTemplateColumns = "1fr";
            grid.style.gridAutoRows = "auto";
            grid.style.gap = TILE_GAP + "px";
            return;
        }
        const cs = getComputedStyle(grid);
        const padX = (parseFloat(cs.paddingLeft) || 0) + (parseFloat(cs.paddingRight) || 0);
        const avail = grid.clientWidth - padX;          // content width (excludes scrollbar)
        if (avail <= 0) return;                          // not laid out yet; ResizeObserver retries
        // Fewest columns that keep tiles <= TILE_MAX (largest tiles); drop a column
        // if that would push tiles below TILE_MIN. Result: square SIZExSIZE cells.
        let cols = Math.max(1, Math.ceil((avail + TILE_GAP) / (TILE_MAX + TILE_GAP)));
        let size = Math.floor((avail - (cols - 1) * TILE_GAP) / cols);
        while (cols > 1 && size < TILE_MIN) {
            cols -= 1;
            size = Math.floor((avail - (cols - 1) * TILE_GAP) / cols);
        }
        size = Math.max(1, Math.min(TILE_MAX, size));
        grid.style.gridTemplateColumns = `repeat(${cols}, ${size}px)`;
        grid.style.gridAutoRows = `${size}px`;
        grid.style.gap = TILE_GAP + "px";
    }

    _fitPreviewVideo() {
        const box = this.previewMediaEl;
        if (!box) return;
        const v = box.querySelector("video");
        if (!v || !v.videoWidth || !v.videoHeight) return;   // no video / metadata not ready
        const boxW = box.clientWidth, boxH = box.clientHeight;
        if (boxW <= 0 || boxH <= 0) return;
        // Scale to fit the box (contain), but never upscale a small clip beyond
        // PREVIEW_MAX_UPSCALE: large clips stay capped at the box, small clips grow
        // up to 2x for a more useful preview without exaggerating the blur. The
        // result is always <= the box, so the CSS max-width/height never clips it.
        const fitScale = Math.min(boxW / v.videoWidth, boxH / v.videoHeight);
        const scale = Math.min(fitScale, PREVIEW_MAX_UPSCALE);
        v.style.width = Math.round(v.videoWidth * scale) + "px";
        v.style.height = Math.round(v.videoHeight * scale) + "px";
        this._setPrevDim(v.videoWidth, v.videoHeight);   // authoritative video dimensions
    }

    // v461: dynamic, content-aware minimum height. The static MIN_NODE_H can't know
    // how tall the wrapped button-bar is (it wraps with node WIDTH) nor how far the
    // 10 output slots push the widget area down — so at narrow/short sizes the DOM
    // content (bar + grid + selection + foot) got squeezed into a few px and the rows
    // piled on top of each other. We measure the real minimum live instead and clamp
    // the node to it. FLOOR-ONLY: a larger size the user chose is never shrunk.
    //
    // topReserved = everything above the DOM widget (header + slots + number widgets).
    // The DOM element always fills (nodeHeight - topReserved), so at any SETTLED size
    // topReserved == node.size[1] - domHeight. We cache it from _refreshFloor (run in
    // the post-layout RAF, when sizes are consistent) and reuse it in onResize, where
    // node.size already holds the proposed height but the element hasn't resized yet.
    _canvasScale() { return app.canvas?.ds?.scale || 1; }

    _computeDomMin() {
        const sc = this._canvasScale();
        // Bar height depends on how many rows it wrapped into -> measure it live.
        const barH = this.barEl ? this.barEl.getBoundingClientRect().height / sc : 28;
        // The status line is only present in some modes; count it when visible.
        const statusH = (this.batchStatusEl && this.batchStatusEl.getBoundingClientRect().height > 0)
            ? this.batchStatusEl.getBoundingClientRect().height / sc : 0;
        // v626: the Selection column's panes (video trim, audio companion, caption) were
        // NOT part of the floor, so a short node clipped them at the column's
        // overflow:hidden — invisible until the node was dragged very tall (Frank's
        // screens). Measure them live (same visible-gating as the status line) so the
        // floor always leaves room for the FULL view. Floor-only as ever: a larger
        // user-chosen size is never shrunk, and the grow converges via the deadband.
        const _vis = (el) => {
            const h = el ? el.getBoundingClientRect().height : 0;
            return h > 0 ? h / sc : 0;
        };
        const paneH = _vis(this.videoTrimEl) + _vis(this.audioPaneEl) + _vis(this.previewCapEl);
        const MAIN_MIN = 92;   // .ph-media-grid min-height:80 + 2*6 padding
        const FOOT_MIN = 18;   // .ph-media-foot min-height:18
        const GAPS = 6 * 3;    // .ph-media column-gap:6 between its four children
        return Math.ceil(barH + statusH + paneH + MAIN_MIN + FOOT_MIN + GAPS);
    }

    _minNodeHeight() {
        const dom = this._computeDomMin();
        const top = (this._topReserved != null) ? this._topReserved : (MIN_NODE_H - dom);
        return Math.max(MIN_NODE_H, Math.ceil(top + dom));
    }

    _refreshFloor() {
        // Sizes are settled here (post-layout). Re-derive topReserved from the live
        // gap between node height and the DOM element height, then grow the node if it
        // now sits below the content floor (e.g. the bar gained a row on narrowing).
        // Converges: the grow re-enters this via the ResizeObserver, but once at/above
        // the floor the deadband stops it — so it never loops or fights a drag.
        const node = this.node;
        if (!node || !node.size || !this.root) return;
        const domNow = this.root.getBoundingClientRect().height / this._canvasScale();
        if (domNow > 0) this._topReserved = Math.max(0, node.size[1] - domNow);
        const floor = this._minNodeHeight();
        if (node.size[1] < floor - 1) node.setSize([node.size[0], floor]);
    }

    _scheduleLayout() {
        // Coalesce the bursts a ResizeObserver fires during a drag-resize.
        if (this._layoutRAF) return;
        this._layoutRAF = requestAnimationFrame(() => {
            this._layoutRAF = 0;
            this._layoutGrid();
            this._fitPreviewVideo();
            this._refreshFloor();   // v461: keep the node tall enough for the wrapped bar
        });
    }

    // v463: heal widget values after loading an OLDER graph. As the widget set
    // evolves (v462 added force_seconds + keep_input_fps; v464 removed
    // force_seconds), the positional widgets_values of a graph saved under a
    // different set can shift — a stray '' from the serialize:false DOM widget, or
    // an old neighbour's value, can land in a typed slot and ComfyUI rejects the
    // prompt ("could not convert string to float: ''"). Coerce the typed widgets
    // back to valid values; anything unparseable falls to a safe default. Runs only
    // for loaded nodes (onConfigure), so fresh nodes (correct defaults) are untouched.
    _healMigratedWidgets() {
        const ws = this.node?.widgets;
        if (!ws) return;
        const num = (name, def) => {
            const w = ws.find((x) => x.name === name);
            if (!w) return;
            const v = (typeof w.value === "number") ? w.value : parseFloat(w.value);
            w.value = Number.isFinite(v) ? v : def;
        };
        const bool = (name, def) => {
            const w = ws.find((x) => x.name === name);
            if (!w) return;
            if (typeof w.value === "boolean") return;
            if (w.value === "true") w.value = true;
            else if (w.value === "false") w.value = false;
            else w.value = (w.value == null || w.value === "") ? def : !!w.value;
        };
        num("force_fps", 0.0);        // pre-existing; 0 = native, preserved when valid
        bool("keep_input_fps", false);
        try { this.node?.setDirtyCanvas?.(true, true); } catch (e) { /* ignore */ }
    }

    // ── v624: Solo-Selection (hide the tile grid; Selection fills the node) ──
    // View state lives in its OWN properties key (ph_media_view), deliberately
    // separate from ph_media_state: trim commits REPLACE ph_media_state wholesale
    // (see _applyVideoTrim), and the view mode must not ride on that semantics.
    get view() { return (this.node.properties && this.node.properties.ph_media_view) || { solo: false, tilesSize: null, soloSize: null }; }
    set view(v) { this.node.properties = this.node.properties || {}; this.node.properties.ph_media_view = v; }

    _applyViewClass() {
        const solo = !!this.view.solo;
        this.root.classList.toggle("ph-solo", solo);
        if (this.soloBtn) this.soloBtn.textContent = solo ? "\u25a6 Tiles" : "\u26f6 Solo";
    }

    _toggleSolo() {
        const r = _viewSwap(this.view, !this.view.solo, this.node.size);
        this.view = r.view;
        this._applyViewClass();
        // v683: coming BACK to tiles, the grid may never have been built (L1).
        // _files is the truth, so a plain re-render restores everything --
        // page, filter highlight, checkmarks, batch marks -- and _syncDims
        // then fetches the sizes for the page that just appeared.
        if (!this.view.solo && this._gridStale && this.folder && this._files) {
            this.renderGrid();
        }
        if (r.size) {   // restore the size this mode was last used at (floor-clamped)
            let floor = MIN_NODE_H;
            try { floor = Math.max(floor, this._minNodeHeight()); } catch (e) { /* measure may not be ready */ }
            this.node.setSize([Math.max(MIN_NODE_W, r.size[0]), Math.max(floor, r.size[1])]);
        }
        this._noteSize();          // the size we landed on IS the new mode's remembered size
        this._scheduleLayout();
        try { this.node.setDirtyCanvas(true, true); } catch (e) { /* ignore */ }
    }

    // Record the CURRENT node size under the CURRENT mode (fed from onResize, so a
    // user drag in either mode keeps that mode's memory up to date).
    _noteSize() {
        const s = this.node?.size;
        if (!s) return;
        const v = Object.assign({ solo: false, tilesSize: null, soloSize: null }, this.view);
        v[v.solo ? "soloSize" : "tilesSize"] = [s[0], s[1]];
        this.view = v;
    }

    _destroy() {
        try { this._ro?.disconnect(); } catch (e) { /* ignore */ }
        try { this._thumbObs?.disconnect(); } catch (e) { /* ignore */ }   // v684
        if (this._layoutRAF) { cancelAnimationFrame(this._layoutRAF); this._layoutRAF = 0; }
        // v624: the ○-selection keydown handler is registered on document (capture
        // phase). Without this remove, every deleted loader node left a live handler
        // behind that ran on EVERY keydown and kept the whole UI tree reachable
        // (leak found in the v624 review).
        try { document.removeEventListener("keydown", this._selKeyHandler, true); } catch (e) { /* ignore */ }
        try { window.removeEventListener("focus", this._onWinFocus); } catch (e) { /* ignore */ }
        // v624: defensive — stop any batch flipbook / hover audio outliving the node
        // (the batch timer also self-stops via its isConnected check; this is belt).
        try { this._batchAnimStop(); } catch (e) { /* ignore */ }
        try { this._stopAudioHover(); } catch (e) { /* ignore */ }
    }

    async upload(fileList) {
        const input = await this._resolveInputPath();
        if (!input) { openFolderPicker("", (p, file) => { this.setFolder(p); pushRecentFolder(p); if (file) this._dropSelect(file); }); return; }
        const fd = new FormData();
        for (const f of fileList) fd.append("files", f, f.name);
        try {
            const r = await api.fetchApi("/uls/media/upload?folder=" + encodeURIComponent(input),
                { method: "POST", body: fd });
            const d = r && await r.json();
            if (d && d.ok) {
                // uploads always land in input/ — switch the view there to show them
                this.setFolder(input);   // v458: browse there without clearing the selection
                await this.refreshGrid();
                if (d.names && d.names.length) {
                    const first = d.names[0];
                    const kind = /\.(mp4|webm|mov|mkv|avi|m4v)$/i.test(first) ? "video" : "image";
                    const fobj = (this._files || []).find((x) => x.name === first) || { name: first, kind, mtime: 0 };
                    this.select(fobj);
                }
            }
        } catch (e) { /* ignore */ }
    }

    async openBrowseModal() {
        if (!this.folder) { openFolderPicker(this.folder, (p) => this.setFolder(p)); return; }
        // ensure the file list is loaded (the grid usually loaded it already)
        if (!this._files || !this._files.length) {
            const target = this.folder;   // same stale-guard as refreshGrid
            try {
                const r = await api.fetchApi("/uls/media/list?folder=" + encodeURIComponent(target));
                const d = r && await r.json();
                if (target === this.folder && d && d.ok) { this._files = d.files || []; this._filesFolder = target; }
            } catch (e) { /* ignore */ }
        }
        injectCSS();
        const back = document.createElement("div"); back.className = "ph-fp-back";
        const box = document.createElement("div"); box.className = "ph-br";
        box.innerHTML = `
          <div class="ph-br-head">
            <input class="ph-br-filter" type="text" spellcheck="false" placeholder="Filter list…">
            <button class="ph-media-btn ph-br-x">Close</button>
          </div>
          <div class="ph-br-list"></div>`;
        back.appendChild(box); document.body.appendChild(back);
        const listEl = box.querySelector(".ph-br-list");
        const filterEl = box.querySelector(".ph-br-filter");
        const files = this._files || [];
        const close = () => { this.hidePop(); back.remove(); };

        const render = (q) => {
            const ql = (q || "").toLowerCase();
            const shown = files.filter((f) => !ql || f.name.toLowerCase().includes(ql));
            listEl.innerHTML = "";
            if (!shown.length) { listEl.innerHTML = `<div class="ph-media-empty">No matches.</div>`; return; }
            for (const f of shown) {
                const row = document.createElement("div");
                row.className = "ph-br-row" + (f.name === this.state.file ? " sel" : "");
                const thumb = document.createElement("div"); thumb.className = "ph-br-thumbwrap";
                const im = document.createElement("img"); im.className = "ph-br-thumb";
                if (f.kind === "audio") {
                    // v457: audio has no thumbnail — show a small ♪ instead of a broken img.
                    im.style.display = "none";
                    const ph = document.createElement("div"); ph.className = "ph-br-aud"; ph.textContent = "♪";
                    thumb.appendChild(ph);
                } else {
                    im.loading = "lazy"; im.src = this._thumbURL(f);
                    im.onerror = () => { im.style.visibility = "hidden"; };
                }
                thumb.appendChild(im);
                const nm = document.createElement("div"); nm.className = "ph-br-name"; nm.textContent = f.name;
                if (f.kind === "video") { const b = document.createElement("span"); b.className = "ph-br-vid"; b.textContent = "▶ video"; nm.appendChild(b); }
                else if (f.kind === "audio") { const b = document.createElement("span"); b.className = "ph-br-vid"; b.textContent = "♪ audio"; nm.appendChild(b); }
                const dim = document.createElement("span"); dim.className = "ph-br-dim";
                if (f.w && f.h) dim.textContent = this._dimText(f.w, f.h);
                row.append(thumb, nm, dim);
                row.onclick = () => { this.select(f); close(); };
                // video plays inside the row thumbnail on hover (like the LoRA Stack),
                // not as a big floating popup; images just show their static thumb.
                if (f.kind === "video") {
                    row.onmouseenter = () => this._playInTile(thumb, f, (w, h) => { dim.textContent = this._dimText(w, h); });
                    row.onmouseleave = () => this._stopInTile(thumb);
                } else if (f.kind === "audio") {
                    // v457: hover plays the audio (one clip at a time), like the grid tiles.
                    row.onmouseenter = () => this._playAudioHover(f);
                    row.onmouseleave = () => this._stopAudioHover();
                } else if (this._isGif(f)) {
                    // GIF (kind "image"): the row thumb is a static JPEG; on hover point
                    // it at the raw file (served as image/gif) so it animates, swap back
                    // on leave to stop it and free the decode. Hover-only on purpose —
                    // a browse list can be long, so we don't animate them all at once
                    // the way the grid tiles do.
                    row.onmouseenter = () => { im.style.visibility = ""; im.src = this._fileURL(f); };
                    row.onmouseleave = () => { im.src = this._thumbURL(f); };
                }
                listEl.appendChild(row);
            }
        };

        filterEl.addEventListener("input", () => render(filterEl.value));
        box.querySelector(".ph-br-x").onclick = close;
        back.onclick = (e) => { if (e.target === back) close(); };
        render("");
        filterEl.focus();
    }

    showPop(f, anchor) {
        this.hidePop();
        const pop = document.createElement("div"); pop.className = "ph-media-pop";
        let media;
        if (f && f.kind === "video") {
            media = document.createElement("video");
            media.src = api.apiURL("/uls/media/file?folder=" + encodeURIComponent(this.folder) +
                "&file=" + encodeURIComponent(f.name) + "&t=" + Math.floor(f.mtime || 0));
            media.poster = this._thumbURL(f);   // real first frame while loading / if unplayable
            media.muted = true; media.loop = true; media.autoplay = true; media.playsInline = true;
            const p = media.play && media.play(); if (p && p.catch) p.catch(() => {});
        } else {
            media = document.createElement("img");
            media.src = this._isGif(f) ? this._fileURL(f) : this._thumbURL(f);
        }
        pop.appendChild(media);
        // pixel-dimension badge (top-left) — an image popup shows a THUMBNAIL, so
        // its naturalWidth would be wrong; use the listing's true source dims. A
        // video reports its authoritative size once metadata loads.
        const dEl = document.createElement("div"); dEl.className = "ph-dim";
        if (f && f.w && f.h) dEl.textContent = this._dimText(f.w, f.h);
        if (f && f.kind === "video") media.addEventListener("loadedmetadata", () => {
            if (media.videoWidth) dEl.textContent = this._dimText(media.videoWidth, media.videoHeight);
        });
        if (dEl.textContent || (f && f.kind === "video")) pop.appendChild(dEl);
        document.body.appendChild(pop);
        const r = anchor.getBoundingClientRect();
        pop.style.left = Math.min(r.right + 8, window.innerWidth - 340) + "px";
        pop.style.top = Math.min(r.top, window.innerHeight - 340) + "px";
        this._pop = pop;
    }
    hidePop() { if (this._pop) { this._pop.remove(); this._pop = null; } }

    restore() {
        // called on workflow load: migrate any v457 audio-only pick, re-arm media_ref,
        // sync the toggles, and refresh the grid at the BROWSED folder (v458: decoupled
        // from the selection's folder).
        // v536 DIAG: what does the RESTORED hidden widget hold right after workflow
        // load, BEFORE any click? Pairs with the backend [PLS v536 DIAG] line on the
        // same cold run: if this shows enabled/mode but the backend line does not,
        // the first graphToPrompt lost it (transport, in the deserialized case); if
        // both match and are enabled, the backend path is at fault; if THIS already
        // shows disabled/empty, the saved workflow itself carried that state. Drop
        // once the cold-restart case is measured.
        try {
            console.info("[PLS v536 DIAG] MediaLoader restore(): batch_config=",
                         this._cfgWidget ? this._cfgWidget.value : "(no widget)");
        } catch (e) { /* ignore */ }
        this._migrateAudioState();
        this._syncRef();
        this._applyViewClass();   // v624: re-apply the persisted Solo-Selection mode (node size comes from the workflow itself)
        this.renderPath();
        this._renderPreview();
        this._renderBatchStatus();
        this._syncAudioToggle();
        if (this.folder) this.refreshGrid();
    }
}

app.registerExtension({
    name: "Polyhedron.MediaLoader",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE) return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onCreated?.apply(this, arguments);
            try { this._mlUI = new MediaLoaderUI(this); }
            catch (e) { console.error("[PLS MediaLoader] init:", e); }
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (o) {
            onConfigure?.apply(this, arguments);
            try { this._mlUI?.restore(); } catch (e) { /* ignore */ }
            try { this._mlUI?._healMigratedWidgets(); } catch (e) { /* ignore */ }
        };

        // Resize floor: clamp in place so the button bar / grid / selection can
        // never be squeezed out of the node. LiteGraph passes node.size by
        // reference during a drag-resize, so mutating it here constrains the node.
        // v461: the height floor is now content-aware (measured wrapped-bar height +
        // the slot stack), with the static MIN_NODE_H kept as a hard fallback.
        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            if (size) {
                if (size[0] < MIN_NODE_W) size[0] = MIN_NODE_W;
                let floor = MIN_NODE_H;
                try { floor = Math.max(floor, this._mlUI?._minNodeHeight?.() || 0); } catch (e) { /* measure may not be ready */ }
                if (size[1] < floor) size[1] = floor;
                if (size[1] < MIN_NODE_H) size[1] = MIN_NODE_H;
            }
            onResize?.apply(this, arguments);
            try { this._mlUI?._noteSize?.(); } catch (e) { /* ignore */ }   // v624: per-mode size memory
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            try { this._mlUI?._destroy(); } catch (e) { /* ignore */ }
            onRemoved?.apply(this, arguments);
        };
    },
});
