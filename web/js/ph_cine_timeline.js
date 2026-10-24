/*
 * ph_cine_timeline.js -- v1028 (Cine C6 + S6/E1): the timeline AND the
 * cutting room, inside the node.
 *
 * One field (no floating window -- Frank's rule), fixed height:
 *
 *   length [ 6.00 ] s -> 149 frames = 6.21 s      style [ Live-action, cinematic, ]
 *   |0s    |1s    |2s    |3s    |4s    |5s    |6s     ruler (17-frame groups shaded)
 *   [ Scene 1              ](dissolve)[ Scene 2       ]  scenes + transition chips
 *   [ Shot 1          ][ Shot 2        ][ Shot 3     ]  cuts drag
 *   <>@fox                  <>image_1            <>@end  anchors drag
 *   + shot  - shot   + anchor [src v]  - anchor
 *   the selected shot (cut phrase, text, "starts a new scene") / anchor / transition
 *   sound: overall_soundscape / non_diegetic_music
 *   render [whole timeline | scene k v]  project [cine]  [render] [render all scenes] [assemble]
 *   stack: one tile per scene from the project's stack (plays on hover) + the whole render + the film
 *   preview: the prompt the RUN will produce + the H3 check
 *
 * Every edit writes the whole timeline into the hidden `timeline` widget as
 * JSON -- the backend (nodes/cine_timeline.py) reads nothing else. The
 * preview comes from POST /pls/cine/timeline, which runs the backend's own
 * compiler: the prompt shown here is the prompt the run produces, not a
 * JavaScript copy of it. The stack and the join are the video track (POST
 * /pls/cine/edit, POST /pls/cine/export; v1182: the v1028 routes are gone). The few numbers the drawing needs before the
 * server answers (the 17k+5 grid, the time label, the official cut phrases)
 * are mirrored below; tests/test_v1023_cine_timeline_js.py holds them equal
 * to the Python side.
 *
 * v1028 -- the render choice is the USER'S: "whole timeline (one run)" keeps
 * the v1027 behaviour; "scene k" renders that scene alone into
 * <day>/PH_<hhmmss>_<project>_scene_k_*.mp4 (v1152; the node's clip_prefix output feeds the
 * Save node). "render all scenes" queues one run per scene. Undo: every
 * finished edit is announced to the frontend's change tracker
 * (litegraph:canvas before-/after-change), so Ctrl+Z on the canvas steps
 * back through timeline edits like through any node edit.
 *
 * v1046 -- the MONITOR (picture + transport) on top, the timeline's TOOLBAR
 *   (cut, heal, delete, A/V linked, snap | reset, cuts -> prompt, keep tail,
 *   span | E4) right above the tracks; no take chip / list (V1 = the newest
 *   run); E4: "new render -> new track" (V1's take pinned while on),
 *   "restore on open", "takes -> tracks".
 *
 * v1045 -- laid out as the editors do (Frank 30.09., a Premiere screenshot):
 *   a head column on the left of every row (timecode, Scenes, Shots, the
 *   tracks, Anchors); picture tracks V4..V2 above the render's V1, sound
 *   tracks A1 (the render's), A2.. below; picture heads lock / Vn / eye /
 *   up / down / reset / x / size, sound heads lock / An / M / S / size; the
 *   active track's chip blue; the render stays V1/A1; "+ new track" is a row
 *   of its own that also takes a dropped video; solo.
 *
 * v1044 -- the LAYERS on screen (E2) and clips into them (E3): "+ layer", a
 *   row per layer around the render lane (head: eye, speaker, lock, up, down,
 *   reset, x, size; picture; sound), the ACTIVE layer for cut / heal /
 *   delete, move / slip / trim / speed on every layer, the player over the
 *   stack, span; drop (upper half copy into input/cine_media/<project>/,
 *   lower half read in place) and the native dialog. The height grip is a
 *   band you can see and reach at every height (Frank 30.09.), plus S/M/L/XL.
 *
 * v1043 -- the LAYERS, stage E1 (Frank, 30.09.): the model and the export.
 *   Emptiness is a piece (GAP), so a layer stays a gapless chain and every
 *   piece op works on it; deletePiece / movePiece / placeClip / normLayer and
 *   flattenRuns (the topmost layer switched on with material wins, picture
 *   and sound each on their own) mirror nodes/cine_edit.py -- the guard
 *   test_v1043_layers drives both. The edit carries the stack (TOP FIRST,
 *   "base" = the render layer, its pieces stay in edit.pieces) and a span;
 *   the rows on screen come with E2.
 *
 * v1037 -- the PIECES (Frank, 29.09.: "a razor to make three or four clips out
 *   of one, a plaster to heal it, then pull the pieces longer or shorter"):
 *   the film is a list of pieces (clip, source in/out, speed). Split at the
 *   playhead, heal a split, drag a seam (trim), drag the amber knob (the
 *   piece's length = its speed; slower frames are blended by ffmpeg, the live
 *   player plays at that rate), drag a clip (slip). A click on a clip shows
 *   its first frame in the player. The playhead is amber.
 *
 * v1036 -- the VIDEO track under the prompt track (Frank, 29.09.: "prompt
 *   transitions go into the run, video transitions onto finished clips --
 *   put both so they line up"):
 *   - a clip lane directly under the shots: a whole run = ONE clip over the
 *     whole length, scenes rendered one by one = one clip per scene on its
 *     place, frame for frame under its shots. Film strips, the overhang the
 *     17k+5 grid adds (a material bar under each clip), takes per scene.
 *   - the ffmpeg transitions sit on the SEAMS of that lane (the handle's
 *     choice is H3 only now); a blend sits centred on its cut and uses the
 *     overhang; where the material ends the last picture is held (named).
 *   - a playhead over both lanes, Play/Pause, click the ruler to jump; the
 *     blends are mixed live on a canvas; "exact preview" and "export" run
 *     ffmpeg on the SAME plan (nodes/cine_edit.py, route /pls/cine/edit).
 *   - fine cut: drag a seam (the film's cut), the end, or a clip's body
 *     (slip); the edit lives in node.properties.pls_cine_edit -- NOT in the
 *     timeline widget, so a fine cut never re-renders the graph. Ghost lines
 *     on the shots show where the film cuts differ from the prompt;
 *     "cuts -> prompt" takes them over for the next run.
 *
 * v1035 -- H3 transitions as text (Frank's field test 29.09.: the formula
 *   alone gave a hard cut, formula + description gave the blend):
 *   - "at this cut" has two groups: inside the clip (H3 paints it, one run)
 *     -- H3 cut, H3 blur to white, H3 fade through black, H3 cross-dissolve,
 *     H3 custom -- and between two clips (assemble, ffmpeg) as before.
 *   - an H3 transition is a preset: it sets the cut formula (the formula
 *     select is locked and shows it) and two sentences the COMPILER writes
 *     (end of the shot before, start of the shot after); the shot texts
 *     stay content only. "H3 custom" edits the two sentences and the
 *     formula, prefilled from the preset it came from.
 *   - the handle turns violet and carries a band of the asked length.
 *
 * v1034 -- three parts from one Run, and the stack without a cable:
 *   - render "all scenes, one run each": the ordinary Run (button, Ctrl+Enter,
 *     the node's own "render") is split into one queued run per scene, in
 *     order (active = 1..n); the timeline stays on "all".
 *   - after a run the editor reports the clip the Save node wrote
 *     (POST /pls/cine/register) -- wired to clip_prefix or not, the clip
 *     joins the project's stack and assemble finds it.
 *   - the stack line says what the next Run will do (whole / scene k / all).
 *   - scene labels keep room beside the transition chips.
 *
 * v1031 -- the width check no longer rides on the node's draw hook (it
 *   made the node count as hand-drawn for the Nodes 2.0 notice); it runs
 *   when the field comes on screen and when a pointer is let go.
 *
 * v1030 -- times by hand, and a preview that grows (Frank, 28.09.):
 *   - the selected shot shows two number fields, "at" (the cut) and
 *     "length"; both land on the frame, the neighbours keep one frame of
 *     room. Changing a length moves the NEXT cut (the last shot's length is
 *     the clip length, which the 17k+5 grid rounds up).
 *   - double-click a shot bar or a cut handle -> the "at" field takes the
 *     focus; on a handle the transition choice for THAT cut is lit as well
 *     (the chip on the scene lane stays the second way). "H3 cut (inside
 *     the clip)" is the default; any other choice makes the cut a scene
 *     start by itself, choosing "H3 cut" again takes the flag away -- a
 *     transition only exists between two clips.
 *   - double-click a scene bar -> its start and end as number fields.
 *   - "drag snap" (frame / 0.25 s / 0.5 s) lives in node.properties: a view
 *     setting, never in the timeline JSON, so the prompt stays byte-equal.
 *   - transitions follow their scene when scenes are added or removed
 *     (they are stored per boundary; v1028 left them on the old index).
 *   - "prompt the run will produce" fits its text like the CLIP Text Encode
 *     fields: measured from scrollHeight once laid out, kept otherwise, the
 *     node reserves it through computeSize -- no onResize hook, fixed px.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { setHidden, refit } from "./ph_widget_vis.js";

console.info("[PLS] ph_cine_timeline.js v1051 loaded");

const NODE = "ULSCineTimeline";
/* v1046: nodes already opened in this page session (restore on open) */
const _SEEN = new Set();
/* v1088: the key of a timeline node in a page session -- its graph's id (new frontends give every
 * workflow graph one) plus the node id; a graph without an id falls back to the node id alone */
export function seenKeyOf(node) {
    const g = node && node.graph;
    const gid = (g && (g.id || (g.extra && g.extra.ph_wf_id))) || "";
    return String(gid) + ":" + String(node && node.id);
}
/* v1088 (audit B7): the folders a dropped file is looked for in -- the Media Loader's recent
 * folders (the same list, shared through the browser) plus this node's own -- and where a
 * found file's folder is remembered */
export const RECENT_KEY = "ph_media_recent_folders";
export function recentFolders(own) {
    let shared = [];
    try { const a = JSON.parse(localStorage.getItem(RECENT_KEY) || "[]"); shared = Array.isArray(a) ? a : []; } catch (e) { shared = []; }
    const out = [];
    for (const f of [...(own || []), ...shared]) if (f && !out.includes(f)) out.push(f);
    return out.slice(0, 20);
}
export function rememberFolder(st, folder) {
    if (!folder) return;
    st.recent = [folder, ...(st.recent || []).filter((x) => x !== folder)].slice(0, 12);
    try {
        const a = JSON.parse(localStorage.getItem(RECENT_KEY) || "[]");
        const arr = [folder, ...(Array.isArray(a) ? a : []).filter((x) => x !== folder)].slice(0, 12);
        localStorage.setItem(RECENT_KEY, JSON.stringify(arr));
    } catch (e) { /* the node's own list still holds it */ }
}
const FIELD = "pls_timeline_view";
/* v1150 (Frank 17.10.: "the timeline is still too nested and complex ... the control areas must get bigger, above
 * all the buttons; check the whole arrangement" -- and: "nothing falls under the table just to look tidier"):
 * the field is a stack of SECTIONS, each with its height set in px from these numbers, and the node's height is
 * their sum (sections(node) -- the one list _layout applies and fieldH adds up; no measured BASE_H any more):
 *   top bar      mode (Text | Keyframes | Reference), length, project, render choice, render all, Render
 *   head row     the old triggers / style fields + "-> CTE" -- only while they hold text
 *   refs row     the wired pictures as thumbnails (hover: big, inside the node) -- Keyframes / Reference
 *   work         left: the selected shot (big text) + the sound; right: the monitor + its transport
 *   state, toolbar (every edit tool, grouped), FILM row, the tracks (all of them), the shot/anchor bar,
 *   status       the check as chips, what the next run does, clip_prefix, the prompt preview's switch
 *   preview      the prompt the run will produce -- open on demand (fitted to its text, as before) */
export const PREVIEW_MIN = 96, PREVIEW_DEF = 180, PREVIEW_MAX = 2000;
export const ROOT_GAP = 4, TOPBAR_H = 38, REFS_H = 40, WORK_H = 312, SND_H = 44, SND_W = 300, STATE_H = 14, BAR_H = 30, STATUS_H = 28;   // v1153: SND_W = the sound column
export const RULER_H = 16, SCENES_H = 22, SHOTS_H = 34, ANCH_H = 24, MON_W = 460, MON_H = 240;
/* v1163 (Frank 18.10.: "das Wiedergabe-Fenster ... an die Groesse des Clips anpassen", "beim Aufziehen und
 * Zusammenziehen sauber anpassen"): the monitor is a BOX the clip is fitted into (contain), never a 16:9 frame
 * the clip is stretched into; the work row's columns shrink down to their floors; the node has a floor width
 * through computeSize (the ph_reference / CTE pattern -- LiteGraph clamps a drag against it; no onResize). */
export const MON_MIN_W = 200, SND_MIN_W = 160, LEFT_MIN_W = 200, MON_AREA_H = 276;
export const TL_MIN_W = 760;   // the node's floor: the three work columns at their floors + gaps + the track heads
export const TAB_MODE_KEY = "pls_cine_mode", PREV_OPEN_KEY = "pls_cine_prev_open";
export const MODES = [["text", "Text (T2V)"], ["keys", "Keyframes (I2V / FL2V)"], ["ref", "Reference (Ref2V)"]];
const DBL_MS = 400;

export const FPS = 24;
export const CUTS = ["the camera cuts to", "the shot cuts to", "the shot transitions to",
                     "the shot changes to", "the shot switches to", "(none)"];
export const TRANSITION_NAMES = ["cut", "dissolve", "fade to black", "fade to white",
                                 "wipe left", "wipe right", "slide left"];
export const DEFAULT = { v: 1, seconds: 6.0, style: "Live-action, cinematic,",
    shots: [{ at: 0.0, cut: CUTS[0], text: "" }], anchors: [], soundscape: "", music: "N/A", from: "",
    render: "whole", active: 1, project: "cine", transitions: [], carry_mode: "pin", head_frames: 22, triggers: "", target: "fl2va" };
/* v1049: scene continuity -- a scene may BEGIN on the last frame of the scene
 * before (its start shot's `carry`); the frame leaves as first_frame, pinned
 * (Reference, ref2va) or as the first-frame anchor (Keyframes, fl2va) */
export const HEAD_FRAMES = [[5, "head 5 f (0.21 s)"], [22, "head 22 f (0.92 s)"], [39, "head 39 f (1.63 s)"]];   // v1185
export const CARRY_MODES = [["pin", "pin only (MiniMax Reference, ref2va)"], ["keyframe", "first-frame anchor (MiniMax Keyframes, fl2va)"],
    ["motion", "motion context: the last 22 frames + sound (MiniMax Continue K2, Core master)"]];   // v1176
export function carryOf(m, k) {
    if (!(k >= 2)) return false;
    const sc = scenesOf(m);
    return !!(sc[k - 1] && m.shots[sc[k - 1][0]].carry);
}
export function setCarry(m, k, on) {
    const sc = scenesOf(m);
    if (!(k >= 2) || !sc[k - 1]) return;
    const s = m.shots[sc[k - 1][0]];
    if (on) s.carry = true; else delete s.carry;
}
// v1051: 9 image pins (they grow with what is wired, ph_cine_timeline_pins.js)
export const N_IMAGE_SLOTS = 9;
export const TARGETS = [["fl2va", "fl2va (Keyframes)"], ["ref2va", "ref2va (Reference)"]];
const SLOTS = Array.from({ length: N_IMAGE_SLOTS }, (_, i) => "image_" + (i + 1));
/* v1051: frames of RENDERED clips as anchor sources -- the backend resolves
 * them at run time from the project's stack (scene K's lead take; K 0 = the
 * whole timeline). Offered in the source list, never sent as "wired". */
export function clipChoices(m) {
    const n = (m && Array.isArray(m.shots) && m.shots.length) ? scenesOf(m).length : 1;
    const out = [];
    if (n > 1) for (let k = 1; k <= n; k++) out.push("@clip:" + k + ":last", "@clip:" + k + ":first");
    out.push("@clip:0:last", "@clip:0:first");
    return out;
}
/* v1035: mirrored from nodes/cine_timeline.py (H3T_*) -- the guard holds them equal */
export const H3T_PRESETS = {
    "H3 blur to white": {
        formula: "the shot cross-dissolves to",
        out: "the whole image slowly drifts out of focus into a soft glowing blur while the light swells until the frame turns almost white",
        in: "The image resolves out of the white glow and comes back into sharp focus" },
    "H3 fade through black": {
        formula: "the shot fades to black, then fades in on",
        out: "the light drains from the scene and the whole image sinks into darkness until the frame is black",
        in: "Brightness returns and the image rises out of the dark" },
    "H3 cross-dissolve": {
        formula: "the shot cross-dissolves to",
        out: "the image begins to dissolve softly while the next image shows through it",
        in: "The previous image melts away over the new one" },
};
export const H3T_CUSTOM = "H3 custom";
export const H3T_NAMES = Object.keys(H3T_PRESETS).concat([H3T_CUSTOM]);
export const H3T_FORMULAS = ["the shot cross-dissolves to", "the shot fades to black, then fades in on",
                             "the camera cuts to", "(none)"];
export const H3T_MIN = 0.25, H3T_MAX = 2.0, H3T_DEFAULT = 0.5;

/* ---------------------------------------------------------------- pure */

export function alignFrames(n) {
    n = Math.max(5, Math.trunc(n));
    while (n % 17 !== 5) n += 1;
    return n;
}
export function framesFor(seconds) { return alignFrames(Math.round(Number(seconds) * FPS)); }
export function duration(model) { return framesFor(model.seconds) / FPS; }
export function snap(sec) { return Math.round(Number(sec) * FPS) / FPS; }
export function timeLabel(sec) {
    const ms = Math.round(Number(sec) * 1000);
    const p = (v, n) => String(v).padStart(n, "0");
    return p(Math.floor(ms / 60000), 2) + ":" + p(Math.floor(ms / 1000) % 60, 2) + "." + p(ms % 1000, 3);
}
export function parseTimeline(text) {
    let d = {};
    try { d = text && String(text).trim() ? JSON.parse(text) : {}; } catch (e) { d = {}; }
    const m = JSON.parse(JSON.stringify(DEFAULT));
    if (d && typeof d === "object") for (const k of Object.keys(DEFAULT)) if (k in d) m[k] = d[k];
    if (!Array.isArray(m.shots) || !m.shots.length) m.shots = [{ at: 0, cut: CUTS[0], text: "" }];
    if (!Array.isArray(m.anchors)) m.anchors = [];
    if (!Array.isArray(m.transitions)) m.transitions = [];
    if (m.render !== "scene" && m.render !== "all") m.render = "whole";
    m.active = Math.max(1, Math.trunc(Number(m.active) || 1));
    m.shots.sort((a, b) => a.at - b.at);
    return m;
}
/* split shot i at its middle (snapped); returns the new shot's index */
export function splitShot(m, i) {
    const dur = duration(m);
    const a = m.shots[i].at, b = i + 1 < m.shots.length ? m.shots[i + 1].at : dur;
    const at = snap((a + b) / 2);
    if (at <= a || at >= b) return i;
    m.shots.splice(i + 1, 0, { at, cut: CUTS[0], text: "" });
    return i + 1;
}
export function removeShot(m, i) {
    if (m.shots.length < 2) return 0;
    keepTransitions(m, () => {
        m.shots.splice(i, 1);
        m.shots[0].at = 0;
        delete m.shots[0].scene;        // shot 1 always begins scene 1
    });
    return Math.max(0, i - 1);
}
/* move the cut that starts shot i (i >= 1) -- one frame of room each side */
export function moveCut(m, i, sec) {
    if (i < 1 || i >= m.shots.length) return;
    const lo = m.shots[i - 1].at + 1 / FPS;
    const hi = (i + 1 < m.shots.length ? m.shots[i + 1].at : duration(m)) - 1 / FPS;
    const was = Math.round(m.shots[i].at * FPS);
    m.shots[i].at = Math.min(hi, Math.max(lo, snap(sec)));
    // v1219: an anchor ON a scene boundary goes with it -- it was the scene's first frame (and with 'both' the last
    // of the scene before); left behind it would turn into a guide of the scene before without a word
    if (m.shots[i].scene) for (const a of m.anchors || []) if (Math.round(Number(a.at) * FPS) === was) a.at = m.shots[i].at;
}
/* v1219: the frames on which a scene boundary lies (scene 2..n start) -- the backend's boundary_frames() */
export function boundaryFrames(m) {
    const sc = scenesOf(m), out = new Set();
    for (let k = 1; k < sc.length; k++) out.add(Math.round(Number(m.shots[sc[k][0]].at) * FPS));
    return out;
}
/* v1219: an anchor's role in the run that renders it -- the whole clip (anchorRole) or, rendered scene by scene, its
 * scene: {role: first | last | both | guide, scene: k}. Mirrors cine_timeline.scene_model + split_anchors (the guard
 * drives both): an anchor on a boundary is the first frame of the scene after it; marked `both` also the last frame
 * of the scene before; on a scene's last asked frame (or in its grid overhang) it is that scene's last frame. */
export function anchorRoleIn(m, a) {
    const sc = scenesOf(m);
    if (!m || m.render === "whole" || sc.length < 2) return { role: anchorRole(m, a), scene: 0 };
    const f = Math.round(Number(a.at) * FPS);
    for (let k = 2; k <= sc.length; k++) {
        if (Math.round(sceneSpan(m, k)[0] * FPS) === f) return { role: a.both ? "both" : "first", scene: k };
    }
    for (let k = 1; k <= sc.length; k++) {
        const [t0, t1] = sceneSpan(m, k), last = k === sc.length;
        if (Number(a.at) < t0 - 1e-9 || (Number(a.at) >= t1 - 1e-9 && !last)) continue;
        const grid = framesFor(t1 - t0), asked = Math.max(1, Math.min(grid, Math.floor((t1 - t0) * FPS + 0.5)));
        const fl = Math.max(0, Math.min(grid - 1, Math.round((Number(a.at) - t0) * FPS)));
        const wf = framesFor(m.seconds);
        if ((last && f >= wf - 1) || fl >= asked - 1) return { role: "last", scene: k };
        return { role: fl <= 0 ? "first" : "guide", scene: k };
    }
    return { role: anchorRole(m, a), scene: 0 };
}
/* v1219: the head a piece must not show -- the repeated head of a motion context (plan.guard, from the server);
 * Alt held = cut past it on purpose (the piece is marked free). -> {pieces, held: how many were held} */
export function guardHeads(pieces, guard, alt) {
    let held = 0;
    const out = (pieces || []).map((p) => {
        const q = Object.assign({}, p);
        if (isGap(q)) return q;
        const g = guard && guard[String(q.k)], head = g && g.repeat ? Number(g.head) || 0 : 0;
        if (head > 0 && q.a < head && !q.free) {
            if (alt) q.free = true;
            else { q.a = head; if (q.b - q.a < 1) q.b = q.a + 1; held++; }
        }
        return q;
    });
    return { pieces: out, held };
}
export function lastFrameTime(m) { return (framesFor(m.seconds) - 1) / FPS; }
export function moveAnchor(m, i, sec) {
    if (!m.anchors[i]) return;
    m.anchors[i].at = Math.min(lastFrameTime(m), Math.max(0, snap(sec)));
    // v1219: 'both scenes' only means something ON a boundary
    if (m.anchors[i].both && !boundaryFrames(m).has(Math.round(m.anchors[i].at * FPS))) delete m.anchors[i].both;
}
/* a free spot for a new anchor: first frame, else last, else the middle */
export function freeSpot(m) {
    const f = framesFor(m.seconds);
    const taken = new Set(m.anchors.map((a) => Math.round(a.at * FPS)));
    if (!taken.has(0)) return 0;
    if (!taken.has(f - 1)) return (f - 1) / FPS;
    return snap(lastFrameTime(m) / 2);
}
/* where an anchor goes: a keyframe (first/last) or a guide (between) */
export function anchorRole(m, a) {
    const f = Math.round(a.at * FPS), n = framesFor(m.seconds);
    return f <= 0 ? "first" : (f >= n - 1 ? "last" : "guide");
}
/* scenes as [firstShot, endShotExclusive] over the (sorted) shots -- the
 * backend's cine_timeline.scenes(); shot 1 begins scene 1 */
export function scenesOf(m) {
    const starts = [0];
    for (let i = 1; i < m.shots.length; i++) if (m.shots[i].scene) starts.push(i);
    return starts.map((a, k) => [a, k + 1 < starts.length ? starts[k + 1] : m.shots.length]);
}
export function sceneSpan(m, k) {
    const sc = scenesOf(m), [a, b] = sc[k - 1];
    return [m.shots[a].at, b < m.shots.length ? m.shots[b].at : duration(m)];
}
export function transitionAt(m, i) {
    const t = m.transitions[i];
    return (t && typeof t === "object" && TRANSITION_NAMES.includes(t.kind)) ? t : { kind: "cut", seconds: 0.5 };
}
export function transitionLabel(t) { return t.kind === "cut" ? "cut" : t.kind + " " + Number(t.seconds).toFixed(2) + "s"; }

/* v1030 ------------------------------------------------------------------ */

/* transitions are stored per scene BOUNDARY (the backend's and assemble's
 * form). Any edit that adds or removes a scene start re-numbers the
 * boundaries, so the edit runs inside this: each transition is held by the
 * shot that starts its scene (by identity -- splice, filter and sort keep
 * it), and afterwards laid out again, exactly one per boundary. */
export function keepTransitions(m, edit) {
    const held = new Map();
    const before = scenesOf(m);
    for (let k = 1; k < before.length; k++) {
        const t = m.transitions[k - 1];
        if (t && typeof t === "object" && TRANSITION_NAMES.includes(t.kind)) held.set(m.shots[before[k][0]], t);
    }
    edit();
    const after = scenesOf(m);
    m.transitions = after.slice(1).map(([a]) => held.get(m.shots[a]) || { kind: "cut", seconds: 0.5 });
}
/* the clip length; shots past the new end go, anchors stay inside */
export function setLength(m, sec) {
    const s = Number(sec);
    keepTransitions(m, () => {
        m.seconds = Math.max(0.5, Number.isFinite(s) && s > 0 ? s : m.seconds);
        const d = duration(m);
        m.shots = m.shots.filter((x, i) => i === 0 || x.at < d - 1 / FPS);
    });
    m.anchors.forEach((a) => { a.at = Math.min(a.at, lastFrameTime(m)); });
}
/* the length of shot i moves the NEXT cut; the last shot's length is the
 * clip length (the grid rounds it up to 17k+5 frames) */
export function setShotLength(m, i, sec) {
    if (!m.shots[i]) return;
    const d = Math.max(1 / FPS, Number(sec) || 0);
    if (i + 1 < m.shots.length) moveCut(m, i + 1, m.shots[i].at + d);
    else setLength(m, Math.round((m.shots[i].at + d) * 1000) / 1000);
}
export function shotLength(m, i) {
    return (i + 1 < m.shots.length ? m.shots[i + 1].at : duration(m)) - m.shots[i].at;
}
/* drag snap: 0 = the frame (1/24 s); otherwise the nearest step, then the frame */
export const SNAP_STEPS = [0, 0.25, 0.5];
export function snapTo(sec, step) {
    const g = Number(step) || 0;
    return snap(g > 0 ? Math.round(Number(sec) / g) * g : Number(sec));
}
/* the choice at a cut handle: H3's own cut inside the clip, or a transition
 * between two clips -- which makes the cut a scene start */
export const H3_CUT = "H3 cut (inside the clip)";
export function setSceneStart(m, i, on) {
    if (i < 1 || !m.shots[i]) return;
    keepTransitions(m, () => { if (on) m.shots[i].scene = true; else delete m.shots[i].scene; });
}
export function h3tOf(shot) {
    const t = shot && shot.h3t;
    return (!shot || shot.scene || !t || typeof t !== "object" || !H3T_NAMES.includes(t.kind)) ? null : t;
}
export function isH3(kind) { return kind === H3_CUT || H3T_NAMES.includes(kind); }
export function cutChoice(m, i) {
    if (i >= 1 && m.shots[i] && !m.shots[i].scene) {
        const t = h3tOf(m.shots[i]);
        return t ? { kind: t.kind, seconds: Number(t.seconds) || H3T_DEFAULT } : { kind: H3_CUT, seconds: H3T_DEFAULT };
    }
    if (i < 1 || !m.shots[i]) return { kind: H3_CUT, seconds: 0.5 };
    const k = scenesOf(m).findIndex(([a]) => a === i);
    return transitionAt(m, k - 1);
}
export function setCutChoice(m, i, kind, seconds) {
    if (i < 1 || !m.shots[i]) return;
    const s = Number(seconds);
    if (H3T_NAMES.includes(kind)) {
        // v1035: painted by H3 inside the clip -- never a scene cut
        setSceneStart(m, i, false);
        const was = m.shots[i].h3t && typeof m.shots[i].h3t === "object" ? m.shots[i].h3t : null;
        const t = { kind, seconds: Math.max(H3T_MIN, Math.min(H3T_MAX, Number.isFinite(s) && s > 0 ? s : H3T_DEFAULT)) };
        if (kind === H3T_CUSTOM) {
            // prefilled from the preset it came from (or its own sentences, kept)
            const src = was && was.kind === H3T_CUSTOM ? was : H3T_PRESETS[was && was.kind] || H3T_PRESETS["H3 blur to white"];
            t.formula = H3T_FORMULAS.includes(src.formula) ? src.formula : H3T_FORMULAS[0];
            t.out = String(src.out || ""); t.in = String(src.in || "");
        }
        m.shots[i].h3t = t;
        return;
    }
    delete m.shots[i].h3t;
    if (kind === H3_CUT || !TRANSITION_NAMES.includes(kind)) { setSceneStart(m, i, false); return; }
    setSceneStart(m, i, true);
    const k = scenesOf(m).findIndex(([a]) => a === i);
    m.transitions[k - 1] = { kind, seconds: Math.max(0.05, Math.min(5, Number.isFinite(s) && s > 0 ? s : 0.5)) };
}
/* v1035: the sentences of an H3 custom transition */
export function setH3Custom(m, i, part, value) {
    const t = m.shots[i] && m.shots[i].h3t;
    if (!t || t.kind !== H3T_CUSTOM || !["out", "in", "formula"].includes(part)) return;
    if (part === "formula" && !H3T_FORMULAS.includes(value)) return;
    t[part] = String(value);
}
/* scene k (1-based): its start and end are the cuts at its edges; the last
 * scene's end is the clip length */
export function setSceneEdge(m, k, edge, sec) {
    const sc = scenesOf(m);
    if (!sc[k - 1]) return;
    const [a, b] = sc[k - 1];
    if (edge === "start") { if (a > 0) moveCut(m, a, sec); return; }
    if (b < m.shots.length) moveCut(m, b, sec); else setLength(m, sec);
}
/* the preview's height for a measured text height: at least PREVIEW_MIN,
 * at most PREVIEW_MAX (a monstrous text scrolls past that) */
/* v1034: the order one Run is split in -- queued at the back (number 0,
 * the Run button) scene 1 first; queued at the FRONT (-1) the last scene
 * goes in first, so they still run 1..n */
export function splitOrder(n, number) {
    const ks = Array.from({ length: Math.max(0, n) }, (_, i) => i + 1);
    return Number(number) === -1 ? ks.reverse() : ks;
}
/* v1163 (Frank 18.10.: "Szene 2 bis 5 zusammen machen ... in einem Rutsch rendern"): every scene has a tick on
 * the Scenes row; "render all scenes" and the "all scenes" run queue the TICKED ones only, in order. The node
 * stores the UNticked scenes (node.properties, saved with the workflow) -- a scene added later starts ticked. */
export const SKIP_KEY = "pls_cine_skip";
export function skipOf(node) {
    const v = node && node.properties && node.properties[SKIP_KEY];
    return new Set(Array.isArray(v) ? v.map((x) => Math.trunc(Number(x))).filter((x) => x >= 1) : []);
}
export function pickedOrder(n, number, skip) {
    return splitOrder(n, number).filter((k) => !(skip && skip.has(k)));
}
function _setSkip(node, k, ticked) {
    const s = skipOf(node);
    if (ticked) s.delete(k); else s.add(k);
    node.properties = node.properties || {};
    node.properties[SKIP_KEY] = [...s].sort((a, b) => a - b);
    node.setDirtyCanvas?.(true, true);
}
const VIDEO_RE = /\.(mp4|webm|mov|mkv|m4v)$/i;
/* the video clips a node's run output names ({filename, subfolder, type}) --
 * the pack's Save ("ph_save") or any other node that lists files */
export function videoRefs(output) {
    const out = [];
    if (!output || typeof output !== "object") return out;
    for (const v of Object.values(output)) {
        if (!Array.isArray(v)) continue;
        for (const e of v) {
            if (e && typeof e === "object" && typeof e.filename === "string" && VIDEO_RE.test(e.filename)
                && (e.type || "output") === "output") out.push({ filename: e.filename, subfolder: e.subfolder || "", type: "output" });
        }
    }
    return out;
}
export function projectOf(clipPrefix) {
    // v1154 (audit): the day prefix (v1152) does not carry the project as its folder -- the run's ui names it
    const head = String(clipPrefix || "").split("/")[0];
    return head && head.indexOf("%") < 0 ? head : "cine";
}
/* what the next Run does, said where the clips appear */
export function stackHint(m, n) {
    // v1154 (audit): the clips lie where the Saves write them (the day folder, v1152) -- the hint names the project's stack
    const p = "the stack of " + (m.project || "cine");
    if (n <= 1) return "video track: the whole run from " + p + " -- on the clip lane above, play below";
    if (m.render === "all") return "video track: Run = " + n + " runs, scene 1.." + n + " one by one -> each clip lands on its place in the clip lane";
    if (m.render === "scene") return "video track: the clips of " + p + " -- Run renders scene " + Math.min(n, m.active) + " of " + n + " only";
    return "video track: whole = ONE clip, the " + n + " scenes as hard cuts inside it -- the seam transitions need the scenes one by one ('all scenes')";
}
/* v1036 ------------------------------------------------------------ video track */
export const EDIT_KEY = "pls_cine_edit";
export const LANE_CLIP_H = 44, LANE_AUDIO_H = 32;   // v1153: the sound lane 32 (the waveform readable)
/* v1038: the lanes' heights are a VIEW setting (node.properties, like the
 * drag snap): pulled at the divider under each lane (v1039). The film strip
 * keeps its aspect -- a taller lane shows bigger pictures, not stretched ones */
export const LANE_KEY = "pls_cine_lanes";
export const CLIP_H_MIN = LANE_CLIP_H, CLIP_H_MAX = 240, AUDIO_H_MIN = LANE_AUDIO_H, AUDIO_H_MAX = 120;
export function laneHeights(node) {
    const v = node && node.properties && node.properties[LANE_KEY];
    const n = (x, lo, hi, d) => { const y = Math.round(Number(x)); return Number.isFinite(y) ? Math.max(lo, Math.min(hi, y)) : d; };
    return { clip: n(v && v.clip, CLIP_H_MIN, CLIP_H_MAX, LANE_CLIP_H), audio: n(v && v.audio, AUDIO_H_MIN, AUDIO_H_MAX, LANE_AUDIO_H) };
}
/* the lanes' vertical layout: ruler 16, scenes 22, shots 30, gaps 4 -> the
 * picture lane starts 80 px down (offsetTop when laid out) */
export const LANE_GAP = 4, LANES_TOP = 84;     // v1150: shots 34 px (was 30)
/* v1044: the height grip under a lane, and the sizes to pick without dragging */
export const DIV_H = 10;
/* v1045: the head column (Premiere's track headers) and the new-track row */
export const GUT_W = 216, NEWROW_H = 22;
/* v1046: the timeline's toolbar (two rows at most) and the E4 switches */
export const TOOLS_H = 40, TOOLS_MAX = 140, AUTO_KEY = "auto_track", RESTORE_KEY = "pls_cine_restore";
export const AUTO_CHOICES = [["track", "new track"], ["render", "one track per render"], ["replace", "replace"]];   // v1163; v1198: + one track per render
/* v1047 (Frank 01.10.: "the total length free, e.g. 5 minutes -- prominent,
 * the view and the zoom follow"): the FILM row over the tracks -- the film's
 * length (= the edit span) and the zoom; the heads stay put while the lanes
 * scroll under them */
export const FILM_H = 32, ZOOM_KEY = "pls_cine_zoom", SPAN_MAX = 3600;
export const FILM_PRESETS = [15, 30, 60, 120, 180, 300, 600];
/* "5:00", "1:30.5", "300", "2m", "90s" -> seconds (NaN when it is not one) */
export function parseLen(s) {
    const t = String(s == null ? "" : s).trim().toLowerCase().replace(",", ".");
    if (!t) return NaN;
    let m = /^(\d+(?:\.\d+)?)\s*(m|min)$/.exec(t);
    if (m) return 60 * Number(m[1]);
    m = /^(\d+(?:\.\d+)?)\s*s?$/.exec(t);
    if (m) return Number(m[1]);
    m = /^(\d+):(\d{1,2}(?:\.\d+)?)$/.exec(t);
    if (m && Number(m[2]) < 60) return 60 * Number(m[1]) + Number(m[2]);
    m = /^(\d+):(\d{1,2}):(\d{1,2}(?:\.\d+)?)$/.exec(t);
    if (m && Number(m[2]) < 60 && Number(m[3]) < 60) return 3600 * Number(m[1]) + 60 * Number(m[2]) + Number(m[3]);
    return NaN;
}
/* seconds -> "m:ss" (with hundredths when it is not a whole second) */
export function fmtLen(sec) {
    const s = Math.max(0, Number(sec) || 0), whole = Math.abs(s - Math.round(s)) < 0.005;
    const m = Math.floor((whole ? Math.round(s) : s) / 60), r = (whole ? Math.round(s) : s) - 60 * m;
    return m + ":" + (whole ? String(Math.round(r)).padStart(2, "0") : r.toFixed(2).padStart(5, "0"));
}
/* the ruler's labelled step for a scale (px per second): >= 56 px apart */
export const TICK_STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
export function tickStep(pxPerSec) {
    const p = Number(pxPerSec) || 0;
    for (const s of TICK_STEPS) if (s * p >= 56) return s;
    return TICK_STEPS[TICK_STEPS.length - 1];
}
/* the zoom: 1 = the whole axis in view; at most a 2-second window */
export function zoomOf(node) { const z = Number(node && node.properties && node.properties[ZOOM_KEY]); return Number.isFinite(z) && z >= 1 ? z : 1; }
export function zoomClamp(z, ax) { return Math.max(1, Math.min(Math.max(1, (Number(ax) || 0) / 2), Number(z) || 1)); }
/* v1047 (Frank 01.10.: "new render -> new track did not work"): what a run
 * of this model renders -- the server's resolve() + clip_prefix() in JS, for
 * the runs whose timeline node came from the CACHE (no "executed" event then) */
export function projectName(m) {
    const raw = String((m && m.project) || "").trim();
    const safe = Array.from(raw).map((ch) => (/[A-Za-z0-9\-_]/.test(ch) || (ch.charCodeAt(0) > 127 && /\p{L}|\p{N}/u.test(ch)) ? ch : "_")).join("").replace(/^_+|_+$/g, "");
    return safe || "cine";
}
/* v1088: the run bookkeeping's pure parts (B3 / B9) */
export function freshRun(r, node, provisional) {
    return Object.assign({ sent: new Set(), rels: [], pending: [], held: [], node: String(node), provisional: !!provisional }, r);
}
export function sameRun(a, b) { return !!(a && b) && a.project === b.project && a.scene === b.scene && a.clip_prefix === b.clip_prefix; }
/* the node ids reachable from node `start` along its output links */
export function downstream(graph, start) {
    const seen = new Set();
    if (!graph) return seen;
    const nodes = graph._nodes || [];
    const byId = new Map(nodes.map((n) => [String(n.id), n]));
    const L = graph.links || {};
    const link = (id) => (L instanceof Map ? L.get(id) : L[id]);
    const todo = [String(start)];
    while (todo.length) {
        const n = byId.get(todo.pop());
        if (!n) continue;
        for (const o of n.outputs || []) for (const lid of o.links || []) {
            const l = link(lid);
            const t = l && String(l.target_id !== undefined ? l.target_id : l[3]);
            if (t && !seen.has(t)) { seen.add(t); todo.push(t); }
        }
    }
    return seen;
}
/* which timeline run a Save belongs to: the only one of the prompt, else the one it hangs below */
export function attributeSave(rec, saveId, graph) {
    const all = [...rec.tls.values()];
    if (all.length === 1) return all[0];
    const sid = String(saveId).split(":")[0];
    const hits = all.filter((r) => { const d = downstream(graph, r.node); return d.has(String(saveId)) || d.has(sid); });
    return hits.length === 1 ? hits[0] : null;
}
/* v1152: the day folder -- the server's cine_timeline.DAY_PREFIX (the Save expands the tokens) */
export const DAY_PREFIX = "%date:yyyy-MM-dd%/PH_%date:hhmmss%_";
/* a prefix as the Save writes it: every %token% stands for any text inside one folder level */
export function prefixRx(clipPrefix) {
    const esc = String(clipPrefix || "").split(/(%[^%/]*%)/).map((x) => /^%[^%/]*%$/.test(x) ? "[^/]*?" : x.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("");
    return new RegExp("^" + esc + "_");
}
export function runOf(m) {
    const n = scenesOf(m).length;
    const k = m && (m.render === "scene" || m.render === "all") ? Math.max(1, Math.min(n, parseInt(m.active, 10) || 1)) : 0;
    const p = projectName(m);
    return { project: p, scene: k, clip_prefix: DAY_PREFIX + p + "_" + (k ? "scene_" + k : "whole") };
}
/* the ONE file of a run that becomes a track: a Save wired to clip_prefix
 * wins, else the run's last video (the end of the chain: upscale, interpolate) */
export function pickTrackFile(rels, clipPrefix) {
    const L = (rels || []).filter(Boolean);
    if (!L.length) return null;
    const rx = clipPrefix ? prefixRx(clipPrefix) : null;
    for (const r of L) if (rx && rx.test(r)) return r;
    return L[L.length - 1];
}
const GUT = "flex:0 0 " + GUT_W + "px;width:" + GUT_W + "px;box-sizing:border-box;display:flex;align-items:center;gap:3px;padding:0 4px 0 9px;overflow:hidden;background:#1c1c1c;border-right:1px solid #2c2c2c;"
    + "position:sticky;left:0;z-index:8;";
function _row(g, laneEl) {
    const r = _el("div", "display:flex;align-items:stretch;flex:0 0 auto;");
    r.append(g, laneEl);
    return r;
}
export const SIZES = [["S", 44, 26], ["M", 90, 40], ["L", 150, 60], ["XL", 240, 90]];
/* v1039: the axis on screen -- the film plus room after its end */
export function viewAxis(ax) { const a = Number(ax) || 0; return a + Math.max(0.25, a * 0.04); }
/* what the lanes add beyond their minimum heights (v1150: lanesH counts the lanes themselves) */
export function laneExtra(node) { const h = laneHeights(node); return (h.clip - LANE_CLIP_H) + (h.audio - LANE_AUDIO_H); }
/* v1044: a layer's rows -- a head (the buttons), its picture, its sound. The
 * heights are a VIEW setting per layer (node.properties, like the lanes'):
 * a taller layer pushes the ones below down; the buttons keep their size */
export const HEAD_H = 18, LAYER_CLIP_H = 44, LAYER_AUDIO_H = 28, LH_KEY = "pls_cine_lh";
export function layerHeights(node, id) {
    const v = node && node.properties && node.properties[LH_KEY] && node.properties[LH_KEY][id];
    const n = (x, lo, hi, d) => { const y = Math.round(Number(x)); return Number.isFinite(y) ? Math.max(lo, Math.min(hi, y)) : d; };
    return { clip: n(v && v.clip, 30, CLIP_H_MAX, LAYER_CLIP_H), audio: n(v && v.audio, 14, AUDIO_H_MAX, LAYER_AUDIO_H) };
}
export function layerRowH(node, id) { const h = layerHeights(node, id); return h.clip + LANE_GAP + h.audio; }
/* what the stack adds to the field: the render layer's head, every other
 * layer's rows (each with the lanes' gap) -- nothing without a stack */
export function stackExtra(node) {
    // v1045: the heads sit left of the lanes (no head rows any more); the
    // new-track row is always there; every other track = its picture row +
    // its sound row, each with the lanes' gap
    const e = node && node.properties && node.properties[EDIT_KEY];
    let h = NEWROW_H + LANE_GAP;
    for (const x of stackLayers(e)) if (x.id !== BASE_ID) { const q = layerHeights(node, x.id); h += q.clip + LANE_GAP + q.audio + LANE_GAP; }
    return h;
}
/* v1038: picture and sound are LINKED by default (the NLE "linked
 * selection"); unlinked -- or Alt held for one drag -- the sound's cut has its
 * own grip (J-/L-cut). A view setting too. */
export const LINK_KEY = "pls_cine_linked";
export function isLinked(node) { const v = node && node.properties ? node.properties[LINK_KEY] : undefined; return v !== false; }
export function linkedFor(node, altKey) { return isLinked(node) !== !!altKey; }
/* v1153 (Frank 17.10.: "V and A held together, a clamp like Photoshop's; unlinked, each can be moved alone"):
 * every pair Vn/An has its own clamp. The render's pair (V1/A1) is the A/V switch above (its J-/L-cuts per
 * seam); another track's pair: linked = its pieces carry picture and sound; unlinked = the sound slides
 * against the picture as a whole (aoff, film frames) -- drag the sound lane, or the picture alone */
export function layerLinked(node, lay) { return !lay || lay.id === BASE_ID ? isLinked(node) : lay.link !== false; }
/* v1153: the tracks' order -- pairs (default: Vn over An, each pair together) or by type (every V above V1, every A below A1) */
export const BYTYPE_KEY = "pls_cine_bytype";
export function byType(node) { return !!(node && node.properties && node.properties[BYTYPE_KEY]); }
/* the speed as editors show it: percent of normal */
export function speedLabel(sp) { return Math.round(100 * (Number(sp) || 1)) + "%"; }
/* the node's edit (node.properties) -- a copy, never the live object */
export function editOf(node) {
    const e = node && node.properties && node.properties[EDIT_KEY];
    try { return e && typeof e === "object" ? JSON.parse(JSON.stringify(e)) : {}; } catch (x) { return {}; }
}
/* the time axis of all lanes: the prompt's grid length, or the film when longer */
export function axisOf(m, plan) {
    const d = duration(m);
    const L = plan && plan.plan ? plan.plan.length / FPS : 0;
    // v1044: the stack -- the film (span) and every layer's own end
    let F = 0;
    if (plan && plan.film) {
        F = Number(plan.film.length) || 0;
        for (const r of plan.film.layers || []) if (r && r.plan) F = Math.max(F, Number(r.plan.length) || 0);
    }
    return Math.max(d, L, F / FPS);
}
/* v1044: how long the film plays -- the stack's span, else the render's film */
export function filmLen(D) {
    if (D && D.film) return Number(D.film.length) || 0;
    return D && D.plan ? Number(D.plan.length) || 0 : 0;
}
/* what the film shows at time t (seconds): [{i, w}] -- one segment with weight
 * 1, or the two of a blend with its progress p (0 -> 1) and kind */
export function mixAt(plan, t) {
    const P = plan && plan.plan;
    if (!P) return { parts: [] };
    const f = t * FPS;
    const act = [];
    P.segments.forEach((s, i) => { if (s.film_in <= f && f < s.film_out) act.push(i); });
    if (act.length === 2) {
        const sm = P.seams[act[0]];
        const a0 = sm.at - sm.h1;
        const p = sm.d > 0 ? Math.max(0, Math.min(1, (f - a0) / sm.d)) : 1;
        return { parts: [{ i: act[0], w: 1 - p }, { i: act[1], w: p }], kind: sm.xfade || "fade", p, seam: act[0] };
    }
    return { parts: act.map((i) => ({ i, w: 1 })) };
}
/* the source time (seconds) segment s shows at film time t -- its speed,
 * a held first picture before its material, the last one after it */
export function srcTime(s, t) {
    const local = t * FPS - s.film_in - (s.hold_head || 0);
    const sp = Number(s.speed) || 1;
    const f = Math.max(s.src_in, Math.min(s.src_in + Math.max(0, local) * sp, s.src_out - 1));
    return (Math.floor(f) + 0.5) / FPS;
}

/* v1037 -- THE PIECES (nodes/cine_edit.py): the film is a list of pieces
 * {k, a, b, speed, tr}; a piece plays source frames [a, b) of clip k at its
 * speed, L = round((b - a) / speed) film frames. Every op below takes the
 * plan's effective list (piecesOf) and hands back a new one -- the editor
 * saves it into the edit; the server plans the film from it. */
export const SEG_MIN = 2, SPEED_MIN = 0.25, SPEED_MAX = 4;
export function piecesOf(D) { return JSON.parse(JSON.stringify((D && D.plan && D.plan.pieces) || [])); }
export function pieceLen(p) { return Math.max(1, Math.round((p.b - p.a) / p.speed)); }
export function starts(pieces) { const o = []; let acc = 0; for (const p of pieces) { o.push(acc); acc += pieceLen(p); } return o; }
export function framesOf(D, k) { const f = D && D.plan && D.plan.frames ? D.plan.frames[String(k)] : null; return f == null ? Infinity : f; }
/* the piece that holds film frame f (-1 = past the end) */
export function pieceAt(pieces, f) {
    const st = starts(pieces);
    for (let j = 0; j < pieces.length; j++) if (f >= st[j] && f < st[j] + pieceLen(pieces[j])) return j;
    return -1;
}
/* the razor: split the piece under film frame f -- two pieces of the same
 * clip; null when f is too close to a piece's edge */
export function splitAt(pieces, f) {
    const j = pieceAt(pieces, Math.round(f));
    if (j < 0) return null;
    const p = pieces[j], local = Math.round(f) - starts(pieces)[j];
    const s = p.a + Math.round(local * p.speed);
    if (local < SEG_MIN || pieceLen(p) - local < SEG_MIN || s - p.a < 1 || p.b - s < 1) return null;
    const out = pieces.map((x) => Object.assign({}, x));
    const left = Object.assign({}, p, { b: s }), right = { k: p.k, a: s, b: p.b, speed: p.speed };
    out.splice(j, 1, left, right);
    return out;
}
/* the plaster: the seam near film frame f that can be healed (the two
 * pieces touch: same clip, b == a) -> its index j (pieces j-1 | j), or -1 */
export function healableAt(pieces, f, tol = 6) {
    const st = starts(pieces);
    let best = -1, dist = Infinity;
    for (let j = 1; j < pieces.length; j++) {
        const q = pieces[j - 1], r = pieces[j], d = Math.abs(st[j] - f);
        if (q.k === r.k && q.b === r.a && d <= tol && d < dist) { best = j; dist = d; }
    }
    return best;
}
/* heal seam j: one piece again; the film keeps its length (the speed is
 * what both pieces took together) */
export function heal(pieces, j) {
    const q = pieces[j - 1], r = pieces[j];
    if (!q || !r || q.k !== r.k || q.b !== r.a) return null;
    const L = pieceLen(q) + pieceLen(r);
    const m = Object.assign({}, q, { b: r.b, speed: q.speed === r.speed ? q.speed : Math.max(SPEED_MIN, Math.min(SPEED_MAX, (r.b - q.a) / L)) });
    const out = pieces.map((x) => Object.assign({}, x));
    out.splice(j - 1, 2, m);
    return out;
}
/* a seam dragged to film frame cut: the piece before gives or takes what the
 * piece after takes or gives (at their speeds), inside their material */
export function trimSeam(pieces, j, cut, framesK) {
    const st = starts(pieces), q = pieces[j - 1], r = pieces[j];
    let d = Math.round(cut) - st[j];
    // v1154 (audit): a GAP may close completely (it kept 2 frames, SEG_MIN is for pieces of a clip)
    d = Math.max(d, -(pieceLen(q) - (isGap(q) ? 0 : SEG_MIN)), Math.ceil(-r.a / r.speed));
    d = Math.min(d, pieceLen(r) - (isGap(r) ? 0 : SEG_MIN), Math.floor((framesK(q.k) - q.b) / q.speed));
    const out = pieces.map((x) => Object.assign({}, x));
    out[j - 1].b = q.b + Math.round(d * q.speed);
    out[j].a = r.a + Math.round(d * r.speed);
    return out;
}
/* stretch piece j to len film frames: its speed (the pieces after it move) */
export function stretch(pieces, j, len) {
    const p = pieces[j], out = pieces.map((x) => Object.assign({}, x));
    const sp = (p.b - p.a) / Math.max(SEG_MIN, Math.round(len));
    out[j].speed = Math.round(Math.max(SPEED_MIN, Math.min(SPEED_MAX, sp)) * 10000) / 10000;
    return out;
}
/* slip piece j by d source frames (a and b together, inside the material) */
export function slipPiece(pieces, j, d, framesK) {
    const p = pieces[j], out = pieces.map((x) => Object.assign({}, x));
    const dd = Math.max(-p.a, Math.min(Math.round(d), framesK(p.k) - p.b));
    out[j].a = p.a + dd; out[j].b = p.b + dd;
    return out;
}
/* the film's end dragged to film frame end: the last piece takes or gives */
export function trimEnd(pieces, end, framesK) {
    const j = pieces.length - 1, p = pieces[j], st = starts(pieces)[j];
    const len = Math.max(SEG_MIN, Math.round(end) - st);
    const out = pieces.map((x) => Object.assign({}, x));
    out[j].b = Math.min(framesK(p.k), p.a + Math.max(1, Math.round(len * p.speed)));
    return out;
}
/* v1038 -- THE SOUND'S CUT (nodes/cine_edit.py ao_range, mirrored: the
 * guard drives both). ao of piece j = where the sound's cut of seam j (pieces
 * j-1 | j) sits, film frames from the picture's cut: > 0 the sound before runs
 * on (L-cut), < 0 the sound after starts early (J-cut). C = the pieces' ends
 * (running sums), d = the seams' blend lengths, prevA = the sound's cut before */
export function aoRange(pieces, C, d, framesK, j, prevA) {
    const at = C[j - 1], dh = d[j - 1] || 0, dp = j >= 2 ? (d[j - 2] || 0) : 0, dn = j < d.length ? (d[j] || 0) : 0;
    const nxt = j < C.length ? C[j] : at;
    let lo = prevA + Math.max(SEG_MIN, dp, dh) - at;
    let hi = nxt - Math.max(SEG_MIN, dh, dn) - at;
    const pa = pieces[j - 1], pb = pieces[j], fa = framesK(pa.k);
    if (Number.isFinite(fa)) hi = Math.min(hi, Math.floor((fa - pa.b) / pa.speed));
    lo = Math.max(lo, -Math.floor(pb.a / pb.speed));
    return [Math.min(lo, 0), Math.max(hi, 0)];
}
/* the running ends of the pieces */
export function endsOf(pieces) { const o = []; let acc = 0; for (const p of pieces) { acc += pieceLen(p); o.push(acc); } return o; }
/* the sound's cuts of all seams, clamped as the server does -> [A_1..A_m-1] */
export function soundCuts(pieces, d, framesK) {
    const C = endsOf(pieces), A = [0];
    for (let j = 1; j < pieces.length; j++) {
        const [lo, hi] = aoRange(pieces, C, d, framesK, j, A[A.length - 1]);
        A.push(C[j - 1] + Math.max(lo, Math.min(hi, Math.round(Number(pieces[j].ao) || 0))));
    }
    return A.slice(1);
}
/* the sound's cut of seam j (1-based) dragged to film frame f -> new pieces */
export function setSoundCut(pieces, j, f, d, framesK) {
    const out = pieces.map((x) => Object.assign({}, x));
    const C = endsOf(out), A = [0].concat(soundCuts(out, d, framesK));
    const [lo, hi] = aoRange(out, C, d, framesK, j, A[j - 1]);
    const ao = Math.max(lo, Math.min(hi, Math.round(f) - C[j - 1]));
    if (ao) out[j].ao = ao; else delete out[j].ao;
    return out;
}
/* the picture's cut of seam j moved to film frame cut: linked = the sound's
 * cut goes along (ao kept); unlinked = the sound's cut stays where it was */
export function trimCut(pieces, j, cut, framesK, linked, d) {
    const out = trimSeam(pieces, j, cut, framesK);
    if (linked || !d) return out;
    const before = soundCuts(pieces, d, framesK)[j - 1];
    return setSoundCut(out, j, before, d, framesK);
}
/* what the SOUND plays at time t: [{i, w}] -- like mixAt, on the sound's
 * segments and cuts (a J-/L-cut blends on the sound's cut) */
export function audioMixAt(plan, t) {
    const P = plan && plan.plan;
    if (!P || !P.asegments) return mixAt(plan, t);
    const f = t * FPS, act = [];
    P.asegments.forEach((s, i) => { if (s.film_in <= f && f < s.film_out) act.push(i); });
    if (act.length === 2) {
        const sm = P.seams[act[0]], at = typeof sm.a_at === "number" ? sm.a_at : sm.at;
        const p = sm.d > 0 ? Math.max(0, Math.min(1, (f - (at - sm.h1)) / sm.d)) : 1;
        return { parts: [{ i: act[0], w: 1 - p }, { i: act[1], w: p }] };
    }
    return { parts: act.map((i) => ({ i, w: 1 })) };
}
/* the source time (seconds) of a sound segment at film time t (no hold: the
 * sound and its picture share the source line -- sync is kept) */
export function asrcTime(s, t) {
    const sp = Number(s.speed) || 1;
    const f = Math.max(s.src_in, Math.min(s.src_in + Math.max(0, t * FPS - s.film_in - (s.pad_head || 0)) * sp, s.src_out - 1));
    return (Math.floor(f) + 0.5) / FPS;
}
/* v1038: the film strip in its aspect -- pictures of width h * ar side by
 * side over a box of boxW px; each shows the strip tile nearest to the
 * source frame under its middle. srcAt(x) -> source frame at box x (px) */
export function tilePlan(boxW, h, ar, tiles, step, srcAt) {
    const tw = Math.max(8, h * (ar > 0 ? ar : 16 / 9));
    const out = [];
    for (let x = 0; x < boxW - 0.5; x += tw) {
        const w = Math.min(tw, boxW - x);
        const idx = Math.max(0, Math.min(tiles - 1, Math.floor(srcAt(x + tw / 2) / Math.max(1, step))));
        out.push({ x, w, tw, idx });
    }
    return out;
}
/* v1038: the playhead snaps to the cuts within tol seconds (0 = off) */
export function snapHead(t, targets, tol) {
    let best = t, dist = tol;
    for (const x of targets) { const dd = Math.abs(x - t); if (dd <= dist) { best = x; dist = dd; } }
    return best;
}

/* v1043 -- THE LAYERS (nodes/cine_edit.py, mirrored; the guard drives both).
 * Emptiness is a piece: k = GAP plays nothing, so every layer stays a gapless
 * chain and the piece ops above work on it unchanged. normLayer after every
 * op: gaps at a = GAP_A, speed 1, merged, none at the end. */
export const GAP = -1, GAP_A = 100000;
export function isGap(p) { return !!p && Number(p.k) === GAP; }
export function gapPiece(n) { return { k: GAP, a: GAP_A, b: GAP_A + Math.max(1, Math.trunc(n)), speed: 1 }; }
export function normLayer(pieces) {
    const out = [];
    for (const p of pieces) {
        if (isGap(p)) {
            if (Number(p.b) - Number(p.a) <= 0) continue;   // v1154: a gap closed to nothing goes (pieceLen floors at 1)
            const n = pieceLen(p);
            if (out.length && isGap(out[out.length - 1])) out[out.length - 1] = gapPiece(pieceLen(out[out.length - 1]) + n);
            else out.push(gapPiece(n));
        } else out.push(Object.assign({}, p));
    }
    while (out.length && isGap(out[out.length - 1])) out.pop();
    return out;
}
/* a new layer: the whole clip from film frame `at` */
export function placeClip(frames, at) {
    return normLayer((Math.trunc(at) > 0 ? [gapPiece(at)] : []).concat([{ k: 0, a: 0, b: Math.max(1, Math.trunc(frames)), speed: 1 }]));
}
/* v1198: a boundary at film frame f (a gap or a piece split there; past the end the layer grows by a gap) */
export function cutAtFrame(pieces, f) {
    const out = (pieces || []).map((x) => Object.assign({}, x)), F = Math.max(0, Math.trunc(f));
    const st = starts(out), end = out.length ? st[out.length - 1] + pieceLen(out[out.length - 1]) : 0;
    if (F >= end) { if (F > end) out.push(gapPiece(F - end)); return out; }
    const j = pieceAt(out, F);
    if (j < 0 || F - st[j] <= 0) return out;
    const p = out[j], local = F - st[j];
    if (isGap(p)) { out.splice(j, 1, gapPiece(local), gapPiece(pieceLen(p) - local)); return out; }
    const s = Math.max(p.a + 1, Math.min(p.b - 1, p.a + Math.round(local * p.speed)));
    if (s <= p.a || s >= p.b) return out;
    out.splice(j, 1, Object.assign({}, p, { b: s }), { k: p.k, a: s, b: p.b, speed: p.speed });
    return out;
}
/* v1198: a piece laid on a layer at film frame `at` -- what lay there is overwritten (the editors' overwrite edit),
 * the rest keeps its place; at past the end = a gap up to it */
export function insertClip(pieces, piece, at) {
    const pc = { k: Math.trunc(piece.k), a: Math.trunc(piece.a), b: Math.trunc(piece.b), speed: Number(piece.speed) || 1 };
    const n = pieceLen(pc), a0 = Math.max(0, Math.trunc(at));
    const q = cutAtFrame(cutAtFrame(normLayer(pieces || []), a0), a0 + n), st = starts(q), out = [];
    let placed = false;
    q.forEach((p, j) => {
        if (st[j] >= a0 && st[j] < a0 + n) { if (!placed) { out.push(pc); placed = true; } return; }
        if (!placed && st[j] >= a0 + n) { out.push(pc); placed = true; }
        out.push(Object.assign({}, p));
    });
    if (!placed) out.push(pc);
    return normLayer(out);
}
/* v1198: a track's files -- k 0 = src, k i = more[i-1] (a render's scenes on one track; clips dragged in) */
export function layerSrcs(lay) {
    if (!lay || !lay.src || (lay.src.kind !== "render" && lay.src.kind !== "file")) return [];
    return [lay.src].concat(Array.isArray(lay.more) ? lay.more : []);
}
export function layerPaths(lay) { return layerSrcs(lay).map((x) => x.path); }
/* v1198: a file onto a track -> { lay, k } (an empty track takes it as its first file; a file it has is reused) */
export function addSource(lay, src) {
    const S = layerSrcs(lay), i = S.findIndex((x) => x.kind === src.kind && x.path === src.path);
    if (i >= 0) return { lay: Object.assign({}, lay), k: i };
    const sx = { kind: src.kind, path: src.path };
    if (src.at != null) sx.at = Math.max(0, Math.trunc(src.at));
    if (!S.length) return { lay: Object.assign({}, lay, { src: sx, pieces: [], more: undefined }), k: 0 };
    return { lay: Object.assign({}, lay, { more: (Array.isArray(lay.more) ? lay.more : []).concat([sx]) }), k: S.length };
}
/* v1198: a piece of file `src` laid on track `toId` at film frame `at` -> the new edit (the track's pieces with it) */
export function dropOnto(e, toId, src, piece, at) {
    const lay = layerOf(e, toId);
    if (!lay || toId === BASE_ID) return e;
    const r = addSource(lay, src);
    const pieces = insertClip(lay.src && lay.src.kind === "empty" ? [] : (lay.pieces || []), Object.assign({}, piece, { k: r.k }), at);
    const patch = { src: r.lay.src, pieces };
    if (r.lay.more) patch.more = r.lay.more;
    return setLayer(e, toId, patch);
}
/* v1198: a track of several files reset -- every file whole again at the place it was laid (in the order they came) */
export function resetMulti(lay, framesK) {
    let q = [];
    layerSrcs(lay).forEach((sx, k) => { const f = framesK(k); if (Number.isFinite(f) && f > 0) q = insertClip(q, { k, a: 0, b: f, speed: 1 }, sx.at || 0); });
    return q;
}
/* piece j becomes empty for its length -- the layers below show through */
export function deletePiece(pieces, j) {
    const out = pieces.map((x) => Object.assign({}, x));
    if (j >= 0 && j < out.length && !isGap(out[j])) out[j] = gapPiece(pieceLen(out[j]));
    return normLayer(out);
}
/* piece j moved d film frames: the gap on one side gives what the other
 * takes, up to the next piece; at the head a gap is made, at the end the
 * layer grows */
/* v1154: where the piece that was j of `before` starts in `after` -- movePiece keeps the order of the clip pieces
 * (only gaps grow, shrink or go), so it is the same n-th non-gap piece */
export function movedStart(before, j, after) {
    let nth = 0;
    for (let i = 0; i < j; i++) if (!isGap(before[i])) nth++;
    const s0 = starts(after);
    for (let i = 0, c = 0; i < after.length; i++) {
        if (isGap(after[i])) continue;
        if (c === nth) return s0[i];
        c++;
    }
    return starts(before)[j];
}
export function movePiece(pieces, j, d) {
    const out = pieces.map((x) => Object.assign({}, x));
    if (!(j >= 0 && j < out.length) || isGap(out[j])) return normLayer(out);
    d = Math.trunc(d);
    const before = j > 0 && isGap(out[j - 1]) ? pieceLen(out[j - 1]) : 0;
    const afterGap = j + 1 < out.length && isGap(out[j + 1]);
    const last = j === out.length - 1;
    if (d < 0) d = Math.max(d, -before);
    else if (d > 0 && !last) d = Math.min(d, afterGap ? pieceLen(out[j + 1]) : 0);
    if (d === 0) return normLayer(out);
    let n = out.slice(0, j);
    if (j > 0 && isGap(out[j - 1])) { const g = before + d; n = out.slice(0, j - 1).concat(g > 0 ? [gapPiece(g)] : []); }
    else if (d > 0) n = out.slice(0, j).concat([gapPiece(d)]);
    n.push(out[j]);
    let rest = out.slice(j + 1);
    if (rest.length && isGap(rest[0])) { const g = pieceLen(rest[0]) - d; rest = (g > 0 ? [gapPiece(g)] : []).concat(rest.slice(1)); }
    else if (rest.length && d < 0) rest = [gapPiece(-d)].concat(rest);
    return normLayer(n.concat(rest));
}
function _coverOf(segs, n) {
    const out = [];
    for (const s of segs || []) {
        if (s.gap || s.missing) continue;
        const a = Math.max(0, Math.trunc(s.film_in)), b = Math.min(n, Math.trunc(s.film_out));
        if (b > a) out.push([a, b]);
    }
    return out;
}
function _runsOf(owner) {
    const out = [];
    owner.forEach((o, f) => {
        const last = out[out.length - 1];
        if (last && last.layer === o) last.film_out = f + 1;
        else out.push({ film_in: f, film_out: f + 1, layer: o });
    });
    return out;
}
/* the stack (TOP FIRST: {video, audio, has_audio, track}) -> who shows at
 * each film frame: the topmost layer switched on with material there,
 * picture and sound each on their own; null = black / silence */
/* v1045: the playhead as the editors show it -- HH:MM:SS:FF at 24 fps */
export function timecode(sec) {
    const f = Math.max(0, Math.floor(Number(sec) * FPS + 1e-6));
    const p = (v) => String(v).padStart(2, "0");
    const s = Math.floor(f / FPS);
    return p(Math.floor(s / 3600)) + ":" + p(Math.floor(s / 60) % 60) + ":" + p(s % 60) + ":" + p(f % FPS);
}
export function flattenRuns(layers, length) {
    const n = Math.max(0, Math.trunc(length));
    const vo = new Array(n).fill(null), ao = new Array(n).fill(null), vcover = [], acover = [];
    const solo = layers.some((L) => L.solo);          // v1045: solo = only the solo tracks sound
    for (const L of layers) {
        const tr = L.track || {};
        vcover.push(L.video !== false ? _coverOf(tr.segments, n) : []);
        const ac = [];
        if (L.audio !== false && L.has_audio !== false && (!solo || L.solo)) {
            for (const s of tr.asegments || []) {
                if (s.gap || s.missing) continue;
                const a = Math.max(0, Math.trunc(s.film_in) + Math.trunc(s.pad_head || 0));
                const b = Math.min(n, Math.trunc(s.film_out) - Math.trunc(s.pad_tail || 0));
                if (b > a) ac.push([a, b]);
            }
        }
        acover.push(ac);
    }
    for (let i = layers.length - 1; i >= 0; i--) {
        for (const [a, b] of vcover[i]) for (let f = a; f < b; f++) vo[f] = i;
        for (const [a, b] of acover[i]) for (let f = a; f < b; f++) ao[f] = i;
    }
    return { vruns: _runsOf(vo), aruns: _runsOf(ao), vcover, acover };
}

/* v1044 -- the STACK in the edit (TOP FIRST; "base" = the render layer, its
 * pieces stay in edit.pieces). Every helper takes the edit and hands back a
 * new one -- the editor saves it (node.properties, never the timeline). */
export const BASE_ID = "base";
export function stackLayers(e) {
    // v1045: the render is V1 / A1 -- always the bottom of the stack
    const L = Array.isArray(e && e.layers) ? e.layers.filter((x) => x && x.id !== BASE_ID).map((x) => Object.assign({}, x)) : [];
    const b = Array.isArray(e && e.layers) ? e.layers.find((x) => x && x.id === BASE_ID) : null;
    L.push(Object.assign({ id: BASE_ID, src: { kind: "base" }, video: true, audio: true, lock: false }, b || {}));
    return L;
}
export function hasStack(e) { return Array.isArray(e && e.layers) && e.layers.some((x) => x && x.id !== BASE_ID); }
function _withLayers(e, L) { const o = JSON.parse(JSON.stringify(e || {})); o.layers = L; return o; }
export function newLayerId(e, now) {
    const ids = new Set(stackLayers(e).map((x) => x.id));
    let n = Math.trunc(Number(now) || Date.now()) % 1679616, id;
    do { id = "L" + (n++).toString(36); } while (ids.has(id));
    return id;
}
/* a new EMPTY layer on top (a clip is dropped in later) */
export function addLayer(e, id, name) {
    const L = stackLayers(e);
    L.unshift({ id, name: name || "", src: { kind: "empty" }, video: true, audio: true, lock: false, pieces: [] });
    return _withLayers(e, L);
}
/* one step up (dir -1) or down (+1) in the stack */
export function moveLayer(e, id, dir) {
    const L = stackLayers(e), i = L.findIndex((x) => x.id === id), j = i + (dir < 0 ? -1 : 1);
    if (i < 0 || j < 0 || j >= L.length || L[i].id === BASE_ID || L[j].id === BASE_ID) return _withLayers(e, L);
    const t = L[i]; L[i] = L[j]; L[j] = t;
    return _withLayers(e, L);
}
export function removeLayer(e, id) {
    if (id === BASE_ID) return _withLayers(e, stackLayers(e));
    return _withLayers(e, stackLayers(e).filter((x) => x.id !== id));
}
export function setLayer(e, id, patch) {
    return _withLayers(e, stackLayers(e).map((x) => (x.id === id ? Object.assign({}, x, patch) : x)));
}
export function layerOf(e, id) { return stackLayers(e).find((x) => x.id === id) || null; }
/* reset: the file again, whole, no cuts and no gaps -- where its material began */
export function resetPieces(pieces, frames) {
    const st = starts(pieces), j = pieces.findIndex((p) => !isGap(p));
    return placeClip(frames, j < 0 ? 0 : st[j]);
}
/* the owner of film frame f in flattened runs (null = nothing) */
export function ownerAt(runs, f) {
    for (const r of runs || []) if (f >= r.film_in && f < r.film_out) return r.layer;
    return null;
}

/* the film's cuts between the scenes into the prompt: scene k+1 starts
 * where the film changes from clip k to k+1 (the pieces in scene order) */
export function cutsIntoPrompt(m, plan) {
    const P = plan && plan.plan;
    const C = P ? P.place_cuts : [];
    const sc = scenesOf(m);
    if (C.length !== sc.length - 1) return m;
    C.forEach((f, j) => moveCut(m, sc[j + 1][0], f / FPS));
    return m;
}
/* the canvas mix of two pictures a -> b at progress p (0 -> 1), the xfade
 * kinds the editor offers -- ffmpeg's own formulas (vf_xfade.c; measured
 * against ffmpeg frame by frame in the browser probe). ops:
 *   ["a"|"b", x, alpha]      draw a frame (normal)
 *   ["bclip", x0, x1]        draw b inside [x0, x1)
 *   ["add", "a"|"b"|colour, w]  add w x the frame / colour (on black)
 * fade to black/white: ffmpeg's two-phase curve -- the old picture is gone
 * within the first fifth, the new one rises over the rest (smoothstep). */
export function smoothstep(e0, e1, x) { const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0))); return t * t * (3 - 2 * t); }
export function blendOps(kind, p, W) {
    switch (kind) {
    case "fadeblack": case "fadewhite": {
        // ffmpeg: progress P runs 1 -> 0 and mix(x, y, m) = x*m + y*(1-m):
        // out = P*(a*s(.8,1,P) + bg*(1-s)) + (1-P)*(bg*s(.2,1,P) + b*(1-s))
        const P = 1 - p, wa = P * smoothstep(0.8, 1, P), wb = (1 - P) * (1 - smoothstep(0.2, 1, P));
        const ops = [["add", "a", wa], ["add", "b", wb]];
        if (kind === "fadewhite") ops.push(["add", "#fff", 1 - wa - wb]);
        return ops;
    }
    case "wipeleft": return [["a", 0, 1], ["bclip", W * (1 - p), W]];
    case "wiperight": return [["a", 0, 1], ["bclip", 0, W * p]];
    case "slideleft": return [["a", -W * p, 1], ["b", W * (1 - p), 1]];
    default: return [["a", 0, 1], ["b", 0, p]];
    }
}
export function previewHeight(contentPx) {
    const h = Math.ceil(Number(contentPx) || 0);
    return Math.max(PREVIEW_MIN, Math.min(PREVIEW_MAX, h));
}
/* the text a shot bar shows: the number, then at most `max` characters */
export function barText(i, text, max) {
    const t = String(text || "").replace(/\s+/g, " ").trim();
    const head = "Shot " + (i + 1);
    if (!t) return head;
    return head + " \u00b7 " + (t.length > max ? t.slice(0, Math.max(1, max - 1)).trimEnd() + "\u2026" : t);
}

/* ---------------------------------------------------------------- dom */

function _w(node, name) { return (node.widgets || []).find((w) => w && w.name === name); }
function _el(tag, css, text) {
    const e = document.createElement(tag);
    if (css) e.style.cssText = css;
    if (text !== undefined) e.textContent = text;
    return e;
}
// v1150 (Frank 17.10.: "the control areas must get bigger, above all the buttons"): 12 px text, real padding
const INPUT = "background:#1e1e1e;color:#ddd;border:1px solid #4a4a4a;border-radius:4px;font-size:12px;padding:2px 6px;box-sizing:border-box;";
/* v1172 (Frank 19.10.: "alle echten Buttons ein wenig einfaerben ... das Grau der Eingabeboxen und das Grau der Buttons
 * verwirrt"): a button is a muted slate blue with a lighter top edge -- clearly not a field (fields stay grey) */
const BTN_BG = "linear-gradient(#34476a,#2a3a57)";
const BTN = "background:linear-gradient(#34476a,#2a3a57);color:#e8eefb;border:1px solid #4d6898;border-radius:4px;font-size:12px;cursor:pointer;padding:3px 10px;white-space:nowrap;box-shadow:inset 0 1px 0 rgba(255,255,255,0.08);";
const BTN_GO = "background:#2f8a4a;color:#fff;border:1px solid #3fae60;border-radius:5px;font-size:13px;font-weight:bold;cursor:pointer;padding:5px 20px;white-space:nowrap;";
const SHOT_COL = ["#3b5b7a", "#5b4a7a", "#3d6b56", "#7a5b3b", "#6b3d4d", "#4d6b3d"];
const SCENE_COL = ["#2b3d4f", "#3c3550", "#2e4a3c", "#4f3d2b"];
const H3T_COL = "#c5a3ff";          // v1035: an H3 transition (violet)

function _model(node) { const w = _w(node, "timeline"); return parseTimeline(w ? w.value : ""); }
/* v1028: tell the frontend's change tracker an edit happened, so Ctrl+Z
 * (on the canvas) steps back over it. Both frontends listen for this DOM
 * event (before-change / after-change -> captureCanvasState). */
export function commitEdit() {
    try {
        document.dispatchEvent(new CustomEvent("litegraph:canvas", { detail: { subType: "before-change" } }));
        document.dispatchEvent(new CustomEvent("litegraph:canvas", { detail: { subType: "after-change" } }));
    } catch (e) { /* no tracker -- no undo, nothing else */ }
}
/* v1030: an edit and the height its preview then needs are ONE undo step.
 * The edit opens the tracker's bracket (before-change: nothing is recorded
 * while it is open); the preview's fit closes it (after-change: one capture,
 * edit + height). Measured in the browser: without it, Ctrl+Z first took back
 * only the height, the edit needed a second Ctrl+Z. A safety timer closes a
 * bracket whose preview never comes back. */
function _begin(node) {
    const st = node._plsTl;
    if (!st) { commitEdit(); return; }
    if (!st.bracket) {
        st.bracket = true;
        try { document.dispatchEvent(new CustomEvent("litegraph:canvas", { detail: { subType: "before-change" } })); } catch (e) { /* no tracker */ }
    }
    clearTimeout(st.bracketTimer);
    st.bracketTimer = setTimeout(() => _end(node), 4000);
}
function _end(node) {
    const st = node._plsTl;
    if (!st || !st.bracket) return;
    st.bracket = false;
    clearTimeout(st.bracketTimer);
    try { document.dispatchEvent(new CustomEvent("litegraph:canvas", { detail: { subType: "after-change" } })); } catch (e) { /* no tracker */ }
}
/* a finished text edit: if its preview is still on the way, the fit closes
 * the bracket; otherwise the edit is recorded now */
function _commit(node) {
    const st = node._plsTl;
    if (st && st.previewPending) _begin(node); else commitEdit();
}
/* a fit that is NOT an edit (load, run, width): folded into the current
 * state without an undo step -- the frontend's own squashState does exactly
 * that; where a frontend lacks it, the height is simply one more step */
function _foldLayout() {
    try {
        const ct = app.extensionManager && app.extensionManager.workflow && app.extensionManager.workflow.activeWorkflow
            && app.extensionManager.workflow.activeWorkflow.changeTracker;
        if (ct && typeof ct.squashState === "function" && !(ct.changeCount > 0)) ct.squashState();
    } catch (e) { /* layout only */ }
}
function _save(node, m, commit = true) {
    const w = _w(node, "timeline");
    if (w) { w.value = JSON.stringify(m); w.callback?.(w.value); }
    node.setDirtyCanvas?.(true, true);
    if (commit) _begin(node);
    _preview(node);
}

function _link(node, id) {
    const g = node.graph || app.graph;
    if (!g || id == null) return null;
    return (g.links && (g.links[id] || (g.links.get && g.links.get(id)))) || null;
}
/* anchor sources: the board's image tags (linked Reference Board) + wired
 * image slots. null = not knowable here (then the server skips the check). */
export function sourcesOf(node) {
    const out = [];
    const refs = (node.inputs || []).find((i) => i && i.name === "refs");
    if (refs && refs.link != null) {
        const l = _link(node, refs.link);
        const g = node.graph || app.graph;
        const src = l && g && g.getNodeById ? g.getNodeById(l.origin_id) : null;
        const bw = src && (src.widgets || []).find((w) => w && w.name === "board");
        if (!bw) return null;
        try {
            for (const e of JSON.parse(bw.value || "[]")) if (e && e.kind === "image" && e.tag) out.push("@" + e.tag);
        } catch (e) { return null; }
    }
    for (const s of SLOTS) {
        const i = (node.inputs || []).find((x) => x && x.name === s);
        if (i && i.link != null) out.push(s);
    }
    return out;
}

function _preview(node) {
    const st = node._plsTl;
    if (!st) return;
    clearTimeout(st.timer);
    st.previewPending = true;
    st.timer = setTimeout(async () => {
        const w = _w(node, "timeline");
        /* v1183: an older answer that arrives after a newer request is dropped (as the video track's planSeq) */
        const seq = (st.previewSeq = (st.previewSeq || 0) + 1);
        try {
            const filled = fillOf(node);
            const r = await api.fetchApi("/pls/cine/timeline", { method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ timeline: w ? w.value : "", sources: sourcesOf(node), fill: filled }) });
            const d = await r.json();
            if (seq !== st.previewSeq || !node._plsTl) return;
            st.filled = filled;
            st.fillSig = JSON.stringify(st.filled);
            _chips(node, d);
            if (d.errors && d.errors.length) {
                st.prev.style.color = "#e57373";
                st.prev.value = "does not compile yet:\n  " + d.errors.join("\n  ");
            } else {
                st.prev.style.color = "#cfd8dc";
                st.prev.value = d.prompt + "\n\n-- " + (d.lines || []).join("\n-- ")
                    + ((d.check && d.check.length) ? "\n\ncheck:\n  " + d.check.join("\n  ") : "");
                if (d.clip_prefix) { st.target.textContent = "clip_prefix (optional) = " + d.clip_prefix + "  |  unwired: the run's saved clip joins the stack where your Save writes it"; st.target.title = st.target.textContent; }
            }
        } catch (e) { if (seq !== st.previewSeq) return; st.prev.value = "preview unavailable: " + e.message; }
        st.previewPending = false;
        _fitPreview(node);
        _end(node);
    }, 250);
}

function _secFromX(track, clientX, dur) {
    const r = track.getBoundingClientRect();
    return Math.max(0, Math.min(1, (clientX - r.left) / Math.max(1, r.width))) * dur;
}

/* v1030: the drag snap is a VIEW setting of this node (node.properties) --
 * it never enters the timeline JSON, so the compiled prompt cannot move */
export function snapStep(node) {
    const v = Number(node && node.properties && node.properties.pls_tl_snap);
    return SNAP_STEPS.includes(v) ? v : 0;
}

/* v1030: a double-click that survives a re-render. Every click re-renders
 * the lanes (the element under the pointer is replaced), so the browser's
 * dblclick has no stable target -- two pointer-downs on the same KEY within
 * DBL_MS are the double-click instead. */
export function isDouble(st, key, now) {
    const t = typeof now === "number" ? now : Date.now();
    const d = !!(st.lastDown && st.lastDown.key === key && t - st.lastDown.t < DBL_MS);
    st.lastDown = d ? null : { key, t };
    return d;
}

function _drag(node, handle, track, key, onMove, onDouble, onClick) {
    handle.addEventListener("pointerdown", (ev) => {
        ev.preventDefault(); ev.stopPropagation();
        if (onDouble && isDouble(node._plsTl, key)) { onDouble(); return; }
        handle.setPointerCapture?.(ev.pointerId);
        const dur = node._plsTl.ax || duration(_model(node));
        const x0 = ev.clientX, alt0 = !!ev.altKey;
        let moved = false;
        const at = (e) => snapTo(_secFromX(track, e.clientX, dur), snapStep(node));
        // v1038: Alt at the press or the release flips linked for this drag
        const mv = (e) => { e.stopPropagation(); if (Math.abs(e.clientX - x0) > 1) moved = true; if (moved) onMove(at(e), false, alt0 || !!e.altKey); };
        const up = (e) => {
            handle.removeEventListener("pointermove", mv); handle.removeEventListener("pointerup", up);
            // a click without a move is a selection, not an edit: no save,
            // no undo step, no re-render under the second click
            if (moved) onMove(at(e), true, alt0 || !!e.altKey);
            else if (onClick) onClick();
        };
        handle.addEventListener("pointermove", mv); handle.addEventListener("pointerup", up);
    });
}

/* v1030: "prompt the run will produce" fits its text -- the CLIP Text
 * Encode way (ph_clip_encode.js _refit): measured from scrollHeight only
 * when laid out, the stored height is kept otherwise; the node reserves it
 * through computeSize; one setSize; re-entry guarded; no onResize hook. */
/* v1150: the mode decides what is on screen. Reference = the model's target ref2va (the run changes with it);
 * Text vs Keyframes is a VIEW (node.properties, never in the timeline JSON): both feed fl2va, Keyframes shows the
 * anchors and the pictures. While anchors exist the view is Keyframes -- they go into the run, so they stay seen */
export function modeOf(node, m) {
    if (m && m.target === "ref2va") return "ref";
    if (m && Array.isArray(m.anchors) && m.anchors.length) return "keys";
    const v = node && node.properties && node.properties[TAB_MODE_KEY];
    return v === "keys" ? "keys" : "text";
}
export function headShown(m) { return !!(m && (String(m.triggers || "").trim() || String(m.style || "").trim())); }
export function prevOpen(node) { return !!(node && node.properties && node.properties[PREV_OPEN_KEY]); }
export function lanesH(node, mode) {
    const LH = laneHeights(node);
    const rows = [RULER_H, SCENES_H, SHOTS_H, LH.clip, LH.audio].concat(mode === "text" ? [] : [ANCH_H]);
    return rows.reduce((a, b) => a + b, 0) + LANE_GAP * (rows.length - 1) + stackExtra(node);
}
/* the sections, top to bottom: [key, height, shown] -- _layout sets exactly these, fieldH adds exactly these */
export function sections(node, m) {
    const st = node && node._plsTl;
    const mm = m || (st && st.m) || {};
    const mode = modeOf(node, mm);
    const prevH = st && typeof st.prevH === "number" ? st.prevH : PREVIEW_DEF;
    return [
        ["topbar", TOPBAR_H, true],
        ["headrow", HEAD_LINE_H + headExtra(node) + 6, headShown(mm)],
        ["refs", REFS_H, mode !== "text"],
        ["work", WORK_H, true],
        ["state", STATE_H, true],
        ["tools", st && typeof st.toolsH === "number" ? st.toolsH : TOOLS_H, true],
        ["film", FILM_H, true],
        ["scroller", lanesH(node, mode), true],
        ["bar", BAR_H, true],
        ["status", STATUS_H, true],
        ["prev", prevH, prevOpen(node)],
    ];
}
export function fieldH(node, m) {
    const on = sections(node, m).filter((x) => x[2]);
    return on.reduce((a, x) => a + x[1], 0) + ROOT_GAP * Math.max(0, on.length - 1);
}
/* v1064 (Frank 08.10.: "let the triggers and style fields grow down as soon as
 * the text is wider than the line -- we have a lot of room below, no text
 * half hidden"): the two head fields wrap and grow; what they grow beyond one
 * line is added to the field's height (0 = the old one-line head) */
export const HEAD_LINE_H = 19, HEAD_MAX_H = 400;
export function headExtra(node) {   // v1150: the head row's growth (the row shows only while it holds text)
    const st = node && node._plsTl;
    const x = st && typeof st.headX === "number" ? st.headX : 0;
    return x > 0 ? x : 0;
}
/* the height of one head field for its text: one line at least, HEAD_MAX_H at most */
export function headFieldH(scrollH) {
    const v = Math.ceil(Number(scrollH) || 0);
    return Math.max(HEAD_LINE_H, Math.min(HEAD_MAX_H, v));
}
/* v1150: the toolbar keeps every tool and wraps on a narrow node -- its height is fitted to its rows (measured
 * when laid out, like the preview; the stored one is kept otherwise) */
export function toolsHeight(scrollH) { const v = Math.ceil(Number(scrollH) || 0); return Math.max(TOOLS_H, Math.min(TOOLS_MAX, v)); }
function _fitTools(node) {
    const st = node._plsTl, el = st && st.tools;
    if (!el || el.offsetParent === null || !(el.clientWidth > 0)) return;
    const h0 = el.style.height;
    el.style.height = "auto";
    const h = toolsHeight(el.scrollHeight + 2);
    el.style.height = h0;
    st.toolsH = h;
}
/* v1150: the sections on screen and their heights -- exactly the list fieldH adds up */
function _layout(node) {
    const st = node._plsTl;
    if (!st || !st.sec) return;
    for (const [k, h, on] of sections(node, st.m)) {
        const el = st.sec[k];
        if (!el) continue;
        el.style.display = on ? (el._plsDisp || "flex") : "none";
        el.style.height = h + "px";
    }
    const fh = fieldH(node, st.m) + "px";
    if (st.root.style.height !== fh) { st.root.style.height = fh; try { refit(node); } catch (e) { /* layout only */ } }
}
function _fitHead(node) {
    const st = node._plsTl;
    if (!st || !st.trig || !st.style) return false;
    let hi = HEAD_LINE_H;
    for (const el of [st.trig, st.style]) {
        if (el.offsetParent === null || !(el.clientWidth > 0)) return false;   // not laid out: keep the stored one
        el.style.height = "0px";
        const h = headFieldH(el.scrollHeight + 2);
        el.style.height = h + "px";
        el.style.overflowY = h >= HEAD_MAX_H ? "auto" : "hidden";
        hi = Math.max(hi, h);
    }
    const x = hi - HEAD_LINE_H;
    if (x === (st.headX || 0)) return false;
    st.headX = x;
    _layout(node);
    return true;
}
function _fitPreview(node) {
    const st = node._plsTl;
    if (!st || st.fitting) return;
    st.fitting = true;
    try {
        const el = st.prev;
        if (el && el.offsetParent !== null && el.clientWidth > 0) {
            const h0 = el.style.height, m0 = el.style.minHeight;
            el.style.height = "auto"; el.style.minHeight = "0";
            const raw = el.scrollHeight;
            el.style.height = h0; el.style.minHeight = m0;
            st.prevH = previewHeight(raw + 2);
            st.fitW = el.clientWidth;
            st.fitNodeW = node.size ? node.size[0] : null;
        }
        // not laid out (collapsed, hidden): keep the stored height, and take
        // the width as seen -- otherwise every draw would ask again
        if (node.size) st.fitNodeW = node.size[0];
        const h = typeof st.prevH === "number" ? st.prevH : PREVIEW_DEF;
        el.style.height = h + "px";
        el.style.overflowY = h >= PREVIEW_MAX ? "auto" : "hidden";
        _fitHead(node);
        _fitTools(node);
        _layout(node);
        // v1038: the film strip is laid out in px -- a new width, new pictures
        // v1044: the layer rows' film strips are laid out in px too (v1157: _stripsStale / _relayStrips)
        if (st.plan && st.clips && _stripsStale(st)) _relayStrips(node);
        if (!st.bracket && !st.previewPending) _foldLayout();
    } catch (e) { /* a layout nicety must never break the node */ }
    finally { st.fitting = false; }
}
/* v1157 (Frank 17.10.: "the clips that came in shown right away -- the real frames with the real
 * sound, also after a reload"): the film strips are laid out in px of the lane. A plan that came
 * while the lanes were not laid out (off screen, a reload, the first frames after a load) drew
 * them at width 0 -- V1 fell back to the stretched strip, V2.. showed no picture at all -- and
 * nothing drew them again: the re-layout hung on the PREVIEW's width check, and the preview's
 * width had not changed (measured: lanes 1356 px wide, laneW still 0). Now the lanes have their
 * own check -- after every plan, when the lanes come on screen (IntersectionObserver), after
 * every pointer release; one-shot frames, no ResizeObserver (v610 / test_v1030). */
function _stripsStale(st) {
    const w = (st && st.clips && st.clips.clientWidth) || 0;
    return w > 0 && (Math.abs(w - (st.laneW || 0)) > 2 || Math.abs(w - (st.layW || 0)) > 2);
}
function _relayStrips(node) {
    const st = node._plsTl, ax = st.ax || 1, pct = (x) => (100 * x / ax) + "%";
    _stackRows(node, pct);
    _clipLane(node, _model(node), pct);
}
function _stripCheck(node, fresh) {
    const st = node._plsTl;
    if (!st || !st.plan || !st.clips) return;
    if (fresh) st.stripTries = 0;
    if (st.stripQueued) return;
    const raf = typeof requestAnimationFrame === "function" ? requestAnimationFrame : (fn) => setTimeout(fn, 0);
    st.stripQueued = true;
    raf(() => {
        st.stripQueued = false;
        if (node._plsTl !== st || st.zoomTimer) return;          // gone, or a zoom whose own render follows
        if ((st.clips.clientWidth || 0) > 0) {
            if (_stripsStale(st)) { try { _relayStrips(node); } catch (e) { /* view only */ } }
            return;
        }
        // not laid out yet: classic lays the DOM out a frame or two after it comes on screen --
        // look again for a few frames, then the next trigger (the lanes coming on screen)
        if ((st.stripTries || 0) < 10) { st.stripTries = (st.stripTries || 0) + 1; _stripCheck(node); }
    });
}
/* the text wraps with the width: when the node got wider or narrower since
 * the last fit, fit once more on the next frame (one-shot, not an observer) */
function _widthCheck(node, fresh) {
    const st = node._plsTl;
    if (!st || !st.prev) return;
    _monitorFit(node);                                   // v1163: the monitor follows its column
    if (fresh) st.firstTries = 0;
    if (st.fitQueued) return;
    const w = st.prev.clientWidth;
    const raf0 = typeof requestAnimationFrame === "function" ? requestAnimationFrame : (fn) => setTimeout(fn, 0);
    // never measured yet (the node was off screen when its text came): it
    // fits the first frame it is laid out. Classic lays the DOM out a frame
    // or two AFTER it comes on screen (measured), so a trigger may look again
    // for a few frames -- at most ten, then the next trigger.
    if (typeof st.fitW !== "number") {
        if (w > 0) { st.fitQueued = true; raf0(() => { st.fitQueued = false; _fitPreview(node); }); }
        else if ((st.firstTries || 0) < 10) { st.firstTries = (st.firstTries || 0) + 1; st.fitQueued = true; raf0(() => { st.fitQueued = false; _widthCheck(node); }); }
        return;
    }
    // the node's own width counts too: in classic the DOM element takes the
    // new width only after the draw that saw node.size change (measured)
    const nodeMoved = node.size && typeof st.fitNodeW === "number" && Math.abs(node.size[0] - st.fitNodeW) > 2;
    if ((w > 0 && Math.abs(w - st.fitW) > 2) || nodeMoved) {
        st.fitQueued = true;
        const raf = typeof requestAnimationFrame === "function" ? requestAnimationFrame : (fn) => setTimeout(fn, 0);
        // two frames: the element is laid out at its new width by then; the
        // fit stores the new widths, so this does not come back
        raf(() => raf(() => { st.fitQueued = false; _fitPreview(node); }));
    }
}

/* the source list reads the board NOW -- it is refilled whenever it is
 * opened, so a board edited after wiring shows up without a timeline edit */
function _fillSources(node) {
    const st = node._plsTl;
    const wired = sourcesOf(node) || [];
    const srcs = (wired.length ? wired : ["image_1"]).concat(clipChoices(_model(node)));   // v1051: + clip frames
    const keep = st.src.value;
    st.src.replaceChildren();
    for (const s of srcs) { const o = document.createElement("option"); o.value = s; o.textContent = s.startsWith("@clip:") ? s + "  (rendered clip)" : s; st.src.appendChild(o); }
    if (keep && srcs.includes(keep)) st.src.value = keep;
    return srcs;
}

/* /view URL of a {filename, subfolder, type} ref, or "" */
function _viewURL(ref) {
    if (!ref || !ref.filename) return "";
    const q = "filename=" + encodeURIComponent(ref.filename) + "&subfolder=" + encodeURIComponent(ref.subfolder || "")
        + "&type=" + encodeURIComponent(ref.type || "output");
    try { return api.apiURL("/view?" + q); } catch (e) { return "/view?" + q; }
}

/* ------------------------------------------------------------ v1036 the clip lane */
/* the video track from the server: the clips on their places, the plan */
async function _stack(node) {
    const st = node._plsTl;
    if (!st) return;
    const m = _model(node);
    const w = _w(node, "timeline");
    const seq = (st.planSeq = (st.planSeq || 0) + 1);
    try {
        const r = await api.fetchApi("/pls/cine/edit", { method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ timeline: w ? w.value : "", edit: editOf(node) }) });
        const d = await r.json();
        if (seq !== st.planSeq) return;               // a newer edit is on its way
        st.planPending = false;
        if (d.error) { st.state.textContent = "video track: " + d.error; return; }
        st.plan = d;
        if (st.exact) { try { st.exact.pause(); st.exact.removeAttribute("src"); st.exact.load(); } catch (x) { /* gone */ } }
        st.exact = null;
        _playerLoad(node);
        _render(node);
        _playerDraw(node);
        _stripCheck(node, true);                       // v1157: drawn at width 0 -> drawn again once laid out
    } catch (e) { if (seq === st.planSeq) st.planPending = false; st.state.textContent = "video track unavailable: " + e.message; }
}
/* v1154 (audit): every piece op is built from the pieces the LAST plan drew; a second op inside one round trip was
 * built from the old list and overwrote the first. While an edit is on its way, a piece op waits for the plan */
function _planBusy(node) {
    const st = node._plsTl;
    if (!st || !st.planPending) return false;
    st.state.textContent = "one moment -- the last edit is still on its way (do it once more)";
    _render(node);
    return true;
}
function _saveEdit(node, e) {
    // v1198: a text edit's bracket still open (its preview on the way) is closed FIRST -- the frontend records
    // nothing while one is open, so this edit would ride on the text's step or, the bracket timing out, on none
    if (node._plsTl && node._plsTl.bracket) _end(node);
    if (node._plsTl) node._plsTl.planPending = true;
    node.properties = node.properties || {};
    node.properties[EDIT_KEY] = e;
    node.setDirtyCanvas?.(true, true);
    commitEdit();
    _stack(node);
}
/* v1037: the pieces into the edit (for the render kind on screen); the
 * v1036 form (cuts, end, slip, keep_tail) is folded into them from now on */
/* v1219: Alt held -- a cut may pass the repeated head on purpose (pointer events and keys both report it) */
let _altHeld = false;
try {
    const _alt = (ev) => { _altHeld = !!(ev && ev.altKey); };
    window.addEventListener("keydown", _alt, true); window.addEventListener("keyup", _alt, true);
    window.addEventListener("pointerdown", _alt, true); window.addEventListener("pointermove", _alt, true);
    window.addEventListener("blur", () => { _altHeld = false; });
} catch (e) { /* no window (node.js guard) */ }
/* v1219: the ends of the cut where a scene CARRIES into the next -> {k: {f, t}} (the carry and @clip:k:last follow
 * them); f = the last source frame of scene k's last piece, t = the take it was cut on */
export function carryEnds(m, pieces, takeOf) {
    const out = {}, sc = scenesOf(m);
    for (let k = 1; k < sc.length; k++) {
        if (!carryOf(m, k + 1)) continue;
        let last = null;
        for (const p of pieces || []) if (!isGap(p) && Number(p.k) === k) last = p;
        const t = takeOf(k);
        if (last && t) out[String(k)] = { f: Math.max(0, Math.trunc(last.b) - 1), t };
    }
    return out;
}
function _savePieces(node, pieces) {
    if (_planBusy(node)) return;
    const st = node._plsTl, e = editOf(node), m = _model(node), D = st && st.plan;
    e.pieces = e.pieces || {};
    // v1044: a deleted piece is a gap -- gaps merged, none at the end
    // v1088 (audit C9): the render kind of the MODEL now, not of the last plan (a switch
    // still on its way to the server put a cut under the old kind)
    const scenes = (m.render || "whole") !== "whole";
    let pcs = normLayer(pieces);
    if (scenes && D && D.plan) {
        // v1219 (the overlap): never into a motion context's repeated head, unless Alt; every piece keeps the take
        // and head it was cut on (a new take moves it to its own head), the list the scene count it was made for
        const g = guardHeads(pcs, D.plan.guard || {}, _altHeld);
        pcs = g.pieces;
        if (g.held && st.state) st.state.textContent = "held at the repeated head (the end of the scene before, rendered again) -- hold Alt to cut past it";
        const stamp = {};
        for (const p of D.plan.pieces || []) if (!isGap(p) && p.t && !(String(p.k) in stamp)) stamp[String(p.k)] = { t: p.t, h0: p.h0 || 0 };
        pcs = pcs.map((p) => (isGap(p) || p.t || !stamp[String(p.k)]) ? p : Object.assign({}, p, stamp[String(p.k)]));
        e.pieces_n = scenesOf(m).length;
        // v1219: where the cut ends a carried scene is NOT written into the timeline widget (a fine cut never
        // re-renders the graph, v1036) -- the queue hook hands it to the ONE run that carries from it (carryInject)
    }
    e.pieces[scenes ? "scenes" : "whole"] = pcs;
    for (const k of ["cuts", "end", "slip", "keep_tail"]) delete e[k];
    _saveEdit(node, e);
}
/* v1219: the cut list gone -- the film cuts where the prompt cuts; the cut ends of the carry go with it */
function _clearCuts(node, e) {
    for (const k of ["cuts", "end", "slip", "keep_tail", "pieces", "pieces_n"]) delete e[k];
    return e;
}
/* v1219: the timeline JSON a queued run carries -- with the cut ends THIS run reads (scene k-1's when scene k
 * carries; scene K's for an @clip:K:last anchor of the place it renders), taken from the edit's plan; null = the
 * prompt's ends (no cut list, or nothing carried). Only the queued prompt gets it, never the widget: a fine cut
 * changes the input of the run that depends on it, and of no other. */
export function carryInject(text, plan) {
    const P = plan && plan.plan;
    if (!P || P.source !== "edit") return null;
    const m = parseTimeline(text);
    if (m.render === "whole") return null;
    const sc = scenesOf(m), k = Math.max(1, Math.min(sc.length, Math.trunc(Number(m.active) || 1)));
    const need = new Set();
    if (carryOf(m, k)) need.add(k - 1);
    for (const a of m.anchors || []) { const mm = /^@clip:(\d{1,3}):last$/.exec(String(a.src || "").trim()); if (mm && Number(mm[1]) >= 1) need.add(Number(mm[1])); }
    if (!need.size) return null;
    const takeOf = (j) => { const c = (plan.clips || []).find((x) => x && x.k === j); return c && c.takes && c.path ? c.takes[c.take || 0] : null; };
    const all = carryEnds(Object.assign({}, m, { shots: m.shots.map((s, i) => (i && s.scene) ? Object.assign({}, s, { carry: true }) : s) }), P.pieces, takeOf);
    const ce = {};
    for (const j of need) if (all[String(j)]) ce[String(j)] = all[String(j)];
    if (!Object.keys(ce).length) return null;
    let d = {};
    try { d = JSON.parse(text); } catch (e) { return null; }
    d.carry_ends = ce;
    return JSON.stringify(d);
}
/* v1219 (Frank 24.10.: "die Clips wieder korrekt an die Schneidegrenze zuruecksetzen"): every clip with a known
 * place (V1's, a track's scene_<k> render file) back where the prompt cuts its scene -- the server measures each
 * file's own repeated head. which = "all", "base" or a track id */
async function _snap(node, which) {
    if (_planBusy(node)) return;
    const st = node._plsTl, w = _w(node, "timeline");
    let r = null;
    try { r = await _json("/pls/cine/snap", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ timeline: w ? w.value : "", edit: editOf(node), which }) }); }
    catch (x) { r = { error: x.message }; }
    if (!r || r.error) { st.state.textContent = "snap to scenes: " + ((r && r.error) || "no answer"); return; }
    let e = editOf(node);
    if (r.base) e = _clearCuts(node, e);
    for (const id of Object.keys(r.layers || {})) e = setLayer(e, id, r.layers[id]);
    _saveEdit(node, e);
    st.state.textContent = "snap to scenes -- " + ((r.lines || []).join("; ") || "nothing to move");
}
/* v1165 (Frank 18.10.: "dass man sehen kann, dass in der Timeline eine Datei ersetzt worden ist ... ein kleines
 * gruenes NEW"): V1 shows the newest run of its place and an EARLIER run's clip stood there before (replace
 * mode). The server decides which files belong to which run (the run history the register call now carries;
 * for a list from before v1165 an estimate) -- the extra files ONE run writes (raw, upscale, interpolated) are
 * no replacement. V1 pinned to an older take (new track mode: the later runs are tracks) is not marked. */
export function isReplaced(c) {
    return !!(c && c.replaced && Array.isArray(c.takes) && c.takes.length > 1 && (c.take || 0) === 0);
}
/* v1190 (Frank 20.10.: "zwei neue Clips sind hinzugekommen - die sollten als NEW markiert werden ... die drei anderen
 * laufen erst, sind aber trotzdem mit NEW markiert, das ist falsch"): NEW = the clip V1 shows came in with the LATEST
 * render (the server compares the run's sweep stamp with the newest of the project). A place the new render has
 * not reached yet shows its earlier clip without the mark; a first-time place is NEW too. V1 pinned to an older
 * take is never marked. */
export function isNew(c) {
    return !!(c && c.new && (c.take || 0) === 0);
}
function _clipLane(node, m, pct) {
    const st = node._plsTl;
    const lane = st.clips;
    // v1039: the heights first -- the boxes are drawn for exactly this lane
    const LH = _applyLanes(node);
    lane.replaceChildren();
    const D = st.plan, P = D && D.plan;
    if (!P) {
        lane.appendChild(_el("div", "position:absolute;left:6px;top:12px;font-size:10px;color:#666;", "video track: nothing rendered yet"));
        if (st.audio) st.audio.replaceChildren();
        _cutLayer(node, pct, null);
        _dividers(node);
        return;
    }
    const segH = LH.clip - 10;
    const pcs = P.pieces, st0 = starts(pcs), fK = (k) => framesOf(D, k);
    const dArr = P.seams.map((sm) => sm.d || 0);
    const clipOf = (k) => D.clips.find((c) => c && c.k === k) || {};
    const laneW = lane.clientWidth || 0;
    st.laneW = laneW;
    P.segments.forEach((sg, i) => {
        const c = clipOf(sg.k);
        const need = sg.film_out - sg.film_in;
        if (sg.gap) {
            // v1044: a deleted part -- empty, the layers below show through
            const g = _el("div", "position:absolute;top:0;height:" + segH + "px;box-sizing:border-box;border:1px dashed #333;border-radius:3px;pointer-events:none;"
                + "left:" + pct(sg.film_in / FPS) + ";width:" + pct(need / FPS) + ";");
            g.dataset.pls = "gap";
            g.appendChild(_el("div", "position:absolute;left:4px;bottom:2px;font-size:9px;color:#555;white-space:nowrap;", "empty"));
            lane.appendChild(g);
            return;
        }
        const picked = st.sel && st.sel.kind === "clip" && st.sel.i === i;
        const box = _el("div", "position:absolute;top:0;height:" + segH + "px;overflow:hidden;border-radius:3px;box-sizing:border-box;cursor:grab;background:#101010;"
            + "left:" + pct(sg.film_in / FPS) + ";width:" + pct(need / FPS) + ";"
            + (picked ? "outline:2px solid #f0c87a;" : "border:1px solid " + (sg.missing ? "#555;border-style:dashed;" : "#2c3e50;")));
        box.dataset.pls = "clip";
        // the pictures ride in one film layer (a slip drag moves it as a whole)
        const film = _el("div", "position:absolute;top:0;bottom:0;left:0;right:0;pointer-events:none;");
        box.appendChild(film);
        if (!sg.missing && c.strip && sg.frames) {
            // the box's INSIDE (its 1 px border off both sides) -- measured: the
            // outer height cropped the pictures to 1.79 for a 1.75 clip
            const boxW = laneW * (need / FPS) / (st.ax || 1) - 2, th = segH - 2;
            if (boxW > 0 && c.strip_tiles && c.w && c.h) {
                // v1038: the pictures in their aspect, as tall as the lane
                const url = _viewURL(c.strip);
                const srcAt = (x) => srcTime(sg, (sg.film_in + x / boxW * need) / FPS) * FPS;
                for (const tl of tilePlan(boxW, th, c.w / c.h, c.strip_tiles, c.strip_step || 1, srcAt)) {
                    const tile = _el("div", "position:absolute;top:0;bottom:0;left:" + tl.x.toFixed(1) + "px;width:" + tl.w.toFixed(1) + "px;background-repeat:no-repeat;opacity:0.92;"
                        + "border-right:1px solid #0008;box-sizing:border-box;background-image:url('" + url + "');background-size:" + (tl.tw * c.strip_tiles).toFixed(1) + "px " + th + "px;"
                        + "background-position:" + (-tl.idx * tl.tw).toFixed(1) + "px 0;");
                    tile.dataset.pls = "tile";
                    film.appendChild(tile);
                }
            } else {
                // not laid out yet: the strip stretched over the material (v1036)
                const shown = (need - sg.hold) * sg.speed;
                const wPct = 100 * sg.frames / Math.max(1, shown) * (need - sg.hold) / need;
                film.appendChild(_el("div", "position:absolute;top:0;bottom:0;background-repeat:no-repeat;background-size:100% 100%;opacity:0.9;"
                    + "left:" + (100 * sg.hold_head / need - wPct * sg.src_in / sg.frames) + "%;width:" + wPct + "%;background-image:url('" + _viewURL(c.strip) + "');"));
            }
        }
        if (sg.hold_tail) box.appendChild(_el("div", "position:absolute;top:0;bottom:0;right:0;width:" + (100 * sg.hold_tail / need) + "%;background:repeating-linear-gradient(90deg,#e5737355 0 2px,transparent 2px 5px);", ""));
        if (sg.hold_head) box.appendChild(_el("div", "position:absolute;top:0;bottom:0;left:0;width:" + (100 * sg.hold_head / need) + "%;background:repeating-linear-gradient(90deg,#e5737355 0 2px,transparent 2px 5px);", ""));
        const inW = i > 0 ? P.seams[i - 1].d : 0;
        const split = pcs.filter((q) => q.k === sg.k).length > 1;
        const lab = _el("div", "position:absolute;left:calc(" + (100 * Math.min(inW, need) / need) + "% + 5px);bottom:2px;font-size:9px;color:#fff;text-shadow:0 0 3px #000,0 0 2px #000;white-space:nowrap;pointer-events:none;",
            (sg.k ? "scene " + sg.k : "whole") + (split ? " \u00b7 " + (pcs.slice(0, i + 1).filter((q) => q.k === sg.k).length) : "")
            + (sg.speed !== 1 ? "  " + speedLabel(sg.speed) : "") + (sg.missing ? " -- not rendered" : ""));
        box.appendChild(lab);
        if (!sg.missing && isNew(c)) {
            // v1170 (Frank 18.10.: "das NEW unten rechts hinsetzen, damit es nicht von der CUT verdeckt wird"): the cut
            // chip sits on the seam at the top, the key diamond on the seam lower down -- NEW goes bottom right, clear of both
            const nw = _el("div", "position:absolute;right:12px;bottom:2px;padding:0 3px;border-radius:2px;font:bold 9px sans-serif;line-height:12px;"
                + "color:#0b1a0b;background:#4caf50;box-shadow:0 0 2px #000;pointer-events:none;", "NEW");
            nw.dataset.pls = "newmark";
            box.appendChild(nw);
        }
        // v1046 (Frank 30.09.): no take chip -- V1 shows the newest render; older
        // takes become tracks ("takes -> tracks" in the toolbar)
        box.title = (sg.k ? "scene " + sg.k : "whole run") + (split ? ", piece" : "") + (c.name ? ": " + c.name : "") + "\nfilm " + (sg.film_in / FPS).toFixed(3) + " - " + (sg.film_out / FPS).toFixed(3)
            + " s  <-  clip " + (sg.src_in / FPS).toFixed(3) + " - " + (sg.src_out / FPS).toFixed(3) + " s" + (sg.speed !== 1 ? "  at " + speedLabel(sg.speed) : "")
            + (sg.hold ? " (+" + sg.hold + " frames held)" : "")
            + (!sg.missing && isNew(c) ? "\nNEW: came in with the latest render" + (isReplaced(c) ? " (the earlier files stay on disk: \u29c9 takes \u2192 tracks)" : "") : "")
            + "\nclick: select + show its first frame  |  drag: slip (which part of the clip plays)  |  drag onto a track: a copy there (V1 keeps it)";
        // click = select + the player shows its first frame; drag = slip
        box.addEventListener("pointerdown", (ev) => {
            if (ev.button !== 0) return;
            ev.preventDefault(); ev.stopPropagation();
            box.setPointerCapture?.(ev.pointerId);
            const x0 = ev.clientX, y0 = ev.clientY, w = lane.getBoundingClientRect().width || 1;
            let moved = false, dFilm = 0, to = null;
            const mv = (e) => { if (Math.abs(e.clientX - x0) > 2 || Math.abs(e.clientY - y0) > 6) moved = true; if (!moved || sg.missing) return;
                dFilm = (e.clientX - x0) / w * st.ax * FPS;
                to = _laneUnder(node, e.clientX, e.clientY, BASE_ID);      // v1198: V1's piece over a track = a copy there
                film.style.transform = to ? "" : "translateX(" + ((e.clientX - x0) * _pxScale(st.clips)) + "px)"; };
            const up = (e) => { box.removeEventListener("pointermove", mv); box.removeEventListener("pointerup", up);
                _laneMark(node, null);
                if (moved && !sg.missing && to) {
                    const src = srcOfClip(c);
                    if (src) _moveAcross(node, src, pcs[i], Math.max(0, Math.round(st0[i] + dFilm)), to, null);
                    else node._plsTl.state.textContent = "this clip cannot be laid on a track (no file)";
                    return;
                }
                if (moved && !sg.missing) _savePieces(node, slipPiece(pcs, i, -dFilm * sg.speed, fK));
                else { st.sel = { kind: "clip", i }; _playerSeek(node, st0[i] / FPS); _render(node); } };
            box.addEventListener("pointermove", mv); box.addEventListener("pointerup", up);
        });
        lane.appendChild(box);
        // the material bar: the whole clip (hatched = not in the film)
        if (!sg.missing && sg.frames && sg.speed === 1) {
            const m0 = sg.film_in + sg.hold_head - sg.src_in;
            const bar = _el("div", "position:absolute;top:" + (segH + 3) + "px;height:5px;border-radius:2px;pointer-events:none;background:repeating-linear-gradient(135deg,#607d8b88 0 3px,transparent 3px 6px);"
                + "left:" + pct(m0 / FPS) + ";width:" + pct(sg.frames / FPS) + ";");
            bar.appendChild(_el("div", "position:absolute;top:0;bottom:0;background:#80cbc4;left:" + (100 * sg.src_in / sg.frames) + "%;width:" + (100 * Math.max(0, sg.src_out - sg.src_in) / sg.frames) + "%;"));
            lane.appendChild(bar);
        } else if (!sg.missing && sg.frames) {
            lane.appendChild(_el("div", "position:absolute;top:" + (segH + 3) + "px;height:5px;border-radius:2px;pointer-events:none;background:#ffb300;opacity:0.7;"
                + "left:" + pct(sg.film_in / FPS) + ";width:" + pct(need / FPS) + ";"));
        }
    });
    // the transitions: a chip on each cut (click = edit it), the blend's glow
    P.seams.forEach((sm, j) => {
        if (isGap(pcs[j]) || isGap(pcs[j + 1])) return;   // v1044: nothing to blend with
        const held = P.segments[j].hold_tail > 0 || P.segments[j + 1].hold_head > 0;
        const t = pcs[j + 1].tr && TRANSITION_NAMES.includes(pcs[j + 1].tr.kind) ? pcs[j + 1].tr : { kind: "cut", seconds: 0.5 };
        const chip = _el("div", "position:absolute;top:1px;height:14px;line-height:14px;transform:translateX(-50%);z-index:8;cursor:pointer;font-size:9px;color:#111;padding:0 5px;border-radius:7px;white-space:nowrap;"
            + "background:" + (held ? "#ef9a9a" : t.kind === "cut" ? "#b0bec5" : "#ffcc80") + ";left:" + pct(sm.at / FPS) + ";"
            + (st.sel && st.sel.kind === "transition" && st.sel.i === j ? "outline:2px solid #f0c87a;" : ""), transitionLabel(t));
        chip.dataset.pls = "seam";
        chip.title = "the transition at this cut (ffmpeg, after the run) -- click to edit"
            + (sm.d ? "\n" + (sm.d / FPS).toFixed(2) + " s, centred on the cut; without a held picture up to " + (sm.max_d / FPS).toFixed(2) + " s" : "");
        chip.addEventListener("pointerdown", (ev) => { ev.stopPropagation(); st.sel = { kind: "transition", i: j }; _render(node); });
        if (sm.d) lane.appendChild(_el("div", "position:absolute;top:0;height:" + segH + "px;pointer-events:none;z-index:3;background:linear-gradient(90deg,transparent,#ffcc8066,transparent);"
            + "left:" + pct((sm.at - sm.h1) / FPS) + ";width:" + pct(sm.d / FPS) + ";"));
        lane.appendChild(chip);
    });
    _audioLane(node, pct);
    st.pct = pct;
    st.cutCtx = { pcs, st0, dArr, segH };
    _cutLayer(node, pct, st.cutCtx);
    _dividers(node);
    // the razor and the plaster follow the playhead
    _cutButtons(node);
}
/* v1039: the lanes take their heights (node.properties) BEFORE anything is
 * drawn into them -- the boxes are drawn for exactly this height. Frank's
 * screenshot 29.09.: pictures 190 px tall over a 44 px lane, painted over the
 * sound lane and the rows below. One place sets the heights; the lanes clip
 * what reaches past them (overflow hidden), so it cannot happen again. */
function _applyLanes(node) {
    const st = node._plsTl, LH = laneHeights(node);
    if (st.clips.style.height !== LH.clip + "px") st.clips.style.height = LH.clip + "px";
    if (st.audio.style.height !== LH.audio + "px") st.audio.style.height = LH.audio + "px";
    _layout(node);
    return LH;
}
/* v1039: THE CUTS as one layer over the picture and the sound lane (Frank
 * 29.09.: "a real cut through the tracks"). A cut is a line through both;
 * its picture part (in the picture lane) and its sound part (the gap + the
 * sound lane) are its two grips. A/V linked: either part moves both, live.
 * Unlinked, or Alt for one drag: each part moves alone (J-/L-cut); a dashed
 * bracket in the gap ties the two parts together, "L +0.25 s" names it. The
 * amber diamond at the picture's bottom = the speed of the piece before (it
 * carries the line: the piece's end IS the cut). The film's end is a line
 * through both lanes with its diamond too. */
function _cutLayer(node, pct, cx) {
    const st = node._plsTl, L = st.cuts;
    if (!L) return;
    L.replaceChildren();
    const LH = laneHeights(node);
    const top = st.clips.offsetTop || LANES_TOP;
    const picH = LH.clip, sndTop = LH.clip, sndH = LANE_GAP + LH.audio;
    L.style.top = top + "px";
    L.style.height = (picH + sndH) + "px";
    const D = st.plan, P = D && D.plan;
    if (!P || !cx) return;
    const { pcs, st0, dArr, segH } = cx;
    const fK = (k) => framesOf(D, k), linked = isLinked(node), track = st.clips;
    const C = endsOf(pcs), A0 = [0].concat(P.seams.map((x) => (typeof x.a_at === "number" ? x.a_at : x.at)));
    const part = (y, h, f, col, tag) => {
        const g = _el("div", "position:absolute;top:" + y + "px;height:" + h + "px;width:12px;margin-left:-6px;cursor:ew-resize;pointer-events:auto;left:" + pct(f / FPS) + ";");
        g.appendChild(_el("div", "position:absolute;top:0;bottom:0;left:5px;width:2px;pointer-events:none;background:" + col + ";box-shadow:0 0 0 1px #000a;"));
        g.dataset.pls = tag;
        L.appendChild(g);
        return g;
    };
    P.seams.forEach((sm, j) => {
        const at = sm.at, aat = typeof sm.a_at === "number" ? sm.a_at : sm.at, ao = sm.ao || 0;
        const heals = pcs[j].k === pcs[j + 1].k && pcs[j].b === pcs[j + 1].a;
        const picked = st.sel && st.sel.kind === "transition" && st.sel.i === j;
        const col = picked ? "#f0c87a" : heals ? "#ffffff" : "#80cbc4";
        // where this sound's cut may go (the clips' material, the cuts beside it)
        const [lo, hi] = aoRange(pcs, C, dArr, fK, j + 1, A0[j]);
        const room = "the sound's cut can move " + (lo / FPS).toFixed(2) + " .. +" + (hi / FPS).toFixed(2) + " s from the picture's cut here (the clips' material)";
        let br = null, tag = null;
        if (ao) {
            br = _el("div", "position:absolute;top:" + picH + "px;height:0;border-top:2px dashed #ffcc80;pointer-events:none;"
                + "left:" + pct(Math.min(at, aat) / FPS) + ";width:" + pct(Math.abs(ao) / FPS) + ";");
            br.dataset.pls = "cutbracket";
            L.appendChild(br);
            tag = _el("div", "position:absolute;top:" + (sndTop + LANE_GAP + 1) + "px;font-size:9px;color:#ffcc80;text-shadow:0 0 3px #000;white-space:nowrap;pointer-events:none;"
                + (ao > 0 ? "left:calc(" + pct(aat / FPS) + " + 7px);" : "left:calc(" + pct(aat / FPS) + " - 7px);transform:translateX(-100%);"),
                (ao > 0 ? "L +" : "J -") + (Math.abs(ao) / FPS).toFixed(2) + " s");
            L.appendChild(tag);
        }
        const pic = part(0, picH, at, col, "cutgrip");
        const snd = part(sndTop, sndH, aat, ao ? "#ffcc80" : col, "soundcut");
        pic.title = (heals ? "a cut inside clip " + (pcs[j].k || "whole") : "the cut between two clips") + " at " + timeLabel(at / FPS) + " -- its picture part"
            + "\ndrag: move the cut (one piece gives, the other takes; the prompt stays) -- "
            + (linked ? "picture and sound together (Alt: the picture alone)" : "the picture alone, the sound's cut stays (Alt: both)")
            + (heals ? "\ndouble-click: heal (one piece again)" : "") + "\nclick: its transition  |  the amber diamond: the speed of the piece before";
        snd.title = "the sound's cut at " + timeLabel(aat / FPS) + (ao ? " (" + (ao > 0 ? "L-cut: the sound before runs on " : "J-cut: the sound after starts ") + (Math.abs(ao) / FPS).toFixed(2) + " s " + (ao > 0 ? "under the next picture" : "early") + ")" : " (with the picture)")
            + "\ndrag: " + (linked ? "picture and sound together (Alt: the sound alone -> J-/L-cut)" : "the sound alone -> J-/L-cut (Alt: both)")
            + (ao ? "  |  double-click: back to the picture's cut" : "") + "\n" + room;
        // live: linked moves both parts; unlinked (or Alt) only the part held
        const follow = (who, f, alt) => {
            const lk = linkedFor(node, alt);
            if (who === "pic") { pic.style.left = pct(f / FPS); if (lk) snd.style.left = pct((f + ao) / FPS); }
            else { snd.style.left = pct(f / FPS); if (lk) pic.style.left = pct((f - ao) / FPS); }
            if (br) br.style.display = lk ? "" : "none";
            if (tag) tag.style.display = lk ? "" : "none";
        };
        _drag(node, pic, track, "filmcut:" + j, (sec, done, alt) => {
            const f = Math.round(sec * FPS);
            follow("pic", f, alt);
            if (done) _savePieces(node, trimCut(pcs, j + 1, f, fK, linkedFor(node, alt), dArr));
        }, () => { if (heals) _savePieces(node, heal(pcs, j + 1)); },
        () => { st.sel = { kind: "transition", i: j }; _render(node); });
        _drag(node, snd, track, "soundcut:" + j, (sec, done, alt) => {
            const f = Math.round(sec * FPS);
            follow("snd", f, alt);
            if (!done) return;
            if (linkedFor(node, alt)) _savePieces(node, trimCut(pcs, j + 1, f - ao, fK, true, dArr));
            else {
                // a clamped drag says so -- the grip must not just jump back
                if (f - at < lo || f - at > hi) st.state.textContent = room;
                _savePieces(node, setSoundCut(pcs, j + 1, f, dArr, fK));
            }
        }, () => { if (ao) _savePieces(node, setSoundCut(pcs, j + 1, at, dArr, fK)); }, null);
        _diamond(node, L, pct, segH, pcs, st0, j, pic, snd, ao);
    });
    // the film's end: one line through both lanes (drag = the last piece plays
    // more or less of its clip) and the last piece's diamond
    const end = P.length;
    const ep = part(0, picH, end, "#b0bec5", "endgrip"), es = part(sndTop, sndH, end, "#b0bec5", "endline");
    ep.title = es.title = "the film ends at " + timeLabel(end / FPS) + " -- drag: the last piece plays more or less of its clip (picture and sound)";
    for (const g of [ep, es]) {
        _drag(node, g, track, "filmend:" + g.dataset.pls, (sec, done) => {
            ep.style.left = es.style.left = pct(sec);
            if (done) _savePieces(node, trimEnd(pcs, Math.round(sec * FPS), fK));
        }, null, null);
    }
    _diamond(node, L, pct, segH, pcs, st0, pcs.length - 1, ep, es, 0);
}
/* the speed of piece j: an amber diamond at the picture's bottom on the line
 * where the piece ends; drag = its length (slower / faster, picture and
 * sound, the pitch kept) -- the line goes along; double-click = 100% */
function _diamond(node, L, pct, segH, pcs, st0, j, pic, snd, ao) {
    const p = pcs[j];
    if (!p || isGap(p)) return;                   // v1044: a gap has no speed
    const end = st0[j] + pieceLen(p);
    const kn = _el("div", "position:absolute;top:" + (segH - 6) + "px;width:12px;height:12px;margin-left:-6px;cursor:col-resize;pointer-events:auto;"
        + "left:" + pct(end / FPS) + ";");
    kn.appendChild(_el("div", "position:absolute;left:1px;top:1px;width:9px;height:9px;transform:rotate(45deg);background:#ffb300;border:1px solid #000;pointer-events:none;"));
    const lab = _el("div", "position:absolute;right:14px;top:-1px;font-size:9px;line-height:12px;color:#ffb300;text-shadow:0 0 3px #000,0 0 2px #000;white-space:nowrap;pointer-events:none;",
        p.speed !== 1 ? speedLabel(p.speed) : "");
    kn.appendChild(lab);
    kn.dataset.pls = "stretch";
    kn.title = "speed of the piece before this line: " + speedLabel(p.speed) + " (picture and sound, pitch kept)"
        + "\ndrag right = slower (longer), left = faster (shorter); the pieces after it move  |  double-click: 100%";
    _drag(node, kn, node._plsTl.clips, "stretch:" + j, (sec, done) => {
        const f = Math.round(sec * FPS);
        const len = Math.max(SEG_MIN, f - st0[j]);
        kn.style.left = pic.style.left = pct((st0[j] + len) / FPS);
        if (snd) snd.style.left = pct((st0[j] + len + ao) / FPS);
        lab.textContent = speedLabel(Math.max(SPEED_MIN, Math.min(SPEED_MAX, (p.b - p.a) / len)));
        if (done) _savePieces(node, stretch(pcs, j, len));
    }, () => { const q = pcs.map((x) => Object.assign({}, x)); q[j].speed = 1; _savePieces(node, q); }, null);
    L.appendChild(kn);
}
/* v1039: the lanes' heights are pulled at the DIVIDER below each lane -- the
 * whole width of the gap, as in the editors (the corner grips crowded the
 * film's end: Frank's third screenshot). Double-click = the smallest. */
function _dividers(node) {
    const st = node._plsTl, box = st.tl || st.lanes;
    if (!box) return;
    for (const d of Array.from(box.querySelectorAll("[data-pls=clipheight],[data-pls=audioheight]"))) d.remove();
    st.divs = {};
    const mk = (which) => {
        const lane = which === "clip" ? st.clips : st.audio;
        // v1044 (Frank 30.09.: "no handle to make it smaller again"): a grip you
        // SEE -- a band over the whole width, a pill with the arrows, above the
        // cut layer (z 8 > 7), centred on the gap so it never hides under a picture
        const d = _el("div", "position:absolute;left:0;right:0;height:" + DIV_H + "px;cursor:ns-resize;z-index:8;background:#ffffff0d;pointer-events:auto;");
        const pill = _el("div", "position:absolute;left:50%;top:1px;width:64px;height:" + (DIV_H - 2) + "px;margin-left:-32px;border-radius:4px;background:#9e9e9e;"
            + "color:#111;font-size:9px;line-height:" + (DIV_H - 2) + "px;text-align:center;pointer-events:none;", "\u21d5");
        d.appendChild(pill);
        d.addEventListener("pointerenter", () => { pill.style.background = "#ffb300"; d.style.background = "#ffb30033"; });
        d.addEventListener("pointerleave", () => { pill.style.background = "#9e9e9e"; d.style.background = "#ffffff0d"; });
        d.dataset.pls = which === "clip" ? "clipheight" : "audioheight";
        d.title = (which === "clip" ? "drag down: a taller picture lane (bigger pictures)" : "drag down: a taller sound lane") + "  |  double-click: the smallest";
        d.addEventListener("pointerdown", (ev) => {
            ev.preventDefault(); ev.stopPropagation();
            if (isDouble(st, "lane:" + which)) { _setLaneH(node, which, which === "clip" ? CLIP_H_MIN : AUDIO_H_MIN, true); return; }
            try { d.setPointerCapture(ev.pointerId); } catch (e) { /* no capture */ }
            const h0 = laneHeights(node)[which], y0 = ev.clientY;
            const r = lane.getBoundingClientRect();
            const scale = (r.height || h0) / (lane.offsetHeight || h0 || 1);
            let raf = 0, last = h0;
            const up = () => { d.removeEventListener("pointermove", mv); d.removeEventListener("pointerup", up); d.removeEventListener("pointercancel", up);
                if (raf) cancelAnimationFrame(raf); _setLaneH(node, which, last, true); };
            const mv = (e) => {
                if (e.buttons === 0) { up(); return; }
                last = h0 + (e.clientY - y0) / (scale || 1);
                if (!raf) raf = requestAnimationFrame(() => { raf = 0; _setLaneH(node, which, last, false); });
            };
            d.addEventListener("pointermove", mv); d.addEventListener("pointerup", up); d.addEventListener("pointercancel", up);
        });
        st.divs[which] = d;
        box.appendChild(d);
    };
    mk("clip"); mk("audio");
    _placeDividers(node);
}
function _placeDividers(node) {
    const st = node._plsTl;
    if (!st.divs) return;
    const top = st.clips.offsetTop || LANES_TOP, LH = laneHeights(node);
    const c = Math.floor(LANE_GAP / 2 - DIV_H / 2);
    if (st.divs.clip) st.divs.clip.style.top = (top + LH.clip + c) + "px";
    if (st.divs.audio) st.divs.audio.style.top = (top + LH.clip + LANE_GAP + LH.audio + c) + "px";
}
function _setLaneH(node, which, h, done) {
    const st = node._plsTl;
    node.properties = node.properties || {};
    const cur = laneHeights(node);
    cur[which] = h;
    node.properties[LANE_KEY] = cur;
    node.properties[LANE_KEY] = laneHeights(node);          // clamped
    _applyLanes(node);
    _placeDividers(node);
    if (st.pct) _cutLayer(node, st.pct, done ? null : st.cutCtx);
    node.setDirtyCanvas?.(true, true);
    if (done) _render(node);
}
/* v1037: the audio lane -- every piece's sound where it plays: the clip's
 * waveform over its source window, the crossfades on the SOUND's cuts,
 * silence where the material ends. v1039: the cuts' lines and grips are on
 * the cut layer (one line through both lanes) */
function _audioLane(node, pct) {
    const st = node._plsTl, lane = st.audio;
    if (!lane) return;
    lane.replaceChildren();
    const D = st.plan, P = D && D.plan;
    const H = laneHeights(node).audio;
    if (!P) return;
    const AS = P.asegments || P.segments.map((s) => Object.assign({}, s, { pad_head: s.hold_head, pad_tail: s.hold_tail }));
    AS.forEach((sg, i) => {
        if (sg.gap) return;                       // v1044: silence
        const c = D.clips.find((x) => x && x.k === sg.k) || {};
        const need = sg.film_out - sg.film_in;
        const pad = (sg.pad_head || 0) + (sg.pad_tail || 0);
        const box = _el("div", "position:absolute;top:1px;height:" + (H - 2) + "px;overflow:hidden;border-radius:2px;box-sizing:border-box;background:#0d1a1f;border:1px solid #1f3a44;"
            + "left:" + pct(sg.film_in / FPS) + ";width:" + pct(need / FPS) + ";");
        box.dataset.pls = "audioclip";
        if (sg.missing || !c.audio) {
            box.appendChild(_el("div", "position:absolute;left:4px;top:5px;font-size:9px;color:#607d8b;white-space:nowrap;", sg.missing ? "" : "no audio"));
        } else if (c.wave && sg.frames) {
            const shown = (need - pad) * sg.speed;
            const wPct = 100 * sg.frames / Math.max(1, shown) * (need - pad) / need;
            box.appendChild(_el("div", "position:absolute;top:0;bottom:0;background-repeat:no-repeat;background-size:100% 100%;opacity:0.95;pointer-events:none;"
                + "left:" + (100 * (sg.pad_head || 0) / need - wPct * sg.src_in / sg.frames) + "%;width:" + wPct + "%;background-image:url('" + _viewURL(c.wave) + "');"));
        }
        if (sg.pad_tail) box.appendChild(_el("div", "position:absolute;top:0;bottom:0;right:0;background:#101418;width:" + (100 * sg.pad_tail / need) + "%;", ""));
        if (sg.pad_head) box.appendChild(_el("div", "position:absolute;top:0;bottom:0;left:0;background:#101418;width:" + (100 * sg.pad_head / need) + "%;", ""));
        box.title = (sg.k ? "scene " + sg.k : "whole run") + ": sound" + (sg.speed !== 1 ? " at " + speedLabel(sg.speed) + " (tempo, pitch kept)" : "")
            + (pad ? "; silent where the clip has no more material" : "");
        lane.appendChild(box);
    });
    P.seams.forEach((sm) => {
        const at = typeof sm.a_at === "number" ? sm.a_at : sm.at;
        if (sm.d) lane.appendChild(_el("div", "position:absolute;top:0;bottom:0;pointer-events:none;background:linear-gradient(90deg,transparent,#ffcc8055,transparent);"
            + "left:" + pct((at - sm.h1) / FPS) + ";width:" + pct(sm.d / FPS) + ";"));
    });
}
/* v1044: the chain the razor, the plaster and "delete" work on -- the ACTIVE
 * layer (the one last clicked; its head is lit), the render layer by default */
function _activeChain(node) { return _chainOf(node, node._plsTl.active || BASE_ID); }
function _chainOf(node, id) {
    const st = node._plsTl, D = st.plan;
    const base = { id: BASE_ID, pcs: D && D.plan ? D.plan.pieces : [], fK: (k) => framesOf(D, k), save: (q) => _savePieces(node, q),
        lock: !!(layerOf(editOf(node), BASE_ID) || {}).lock };
    if (id === BASE_ID || !D || !D.film) return base;
    const r = D.film.layers.find((x) => x.id === id);
    if (!r || !r.plan) return base;
    return { id, pcs: r.plan.pieces, fK: (k) => (Number(k) === GAP ? Infinity : ((clipK(r, k) || {}).frames || Infinity)), save: (q) => _saveLayerPieces(node, id, q),
        lock: !!(layerOf(editOf(node), id) || {}).lock, row: r };
}
/* v1198: the clip of file k of a layer row (k 0 = its src; more files in row.clips) */
export function clipK(row, k) {
    if (!row) return null;
    if (Array.isArray(row.clips)) return row.clips[Number(k)] || null;
    return Number(k) === 0 ? (row.clip || null) : null;
}
/* v1198: a render clip of V1 as a track's file -- output/ files by their place there, others where they lie */
export function srcOfClip(c) {
    if (!c) return null;
    const v = c.view;
    if (v && v.type === "output" && v.filename) return { kind: "render", path: (v.subfolder ? v.subfolder + "/" : "") + v.filename };
    if (c.path) return { kind: "file", path: c.path };
    return null;
}
/* v1198: the track lane under the pointer (not `self`, not locked) -> its id, marked; else null */
function _laneUnder(node, x, y, self) {
    let el = null;
    try { el = document.elementFromPoint(x, y); } catch (e) { /* no DOM */ }
    const lane = el && el.closest ? el.closest('[data-pls="layerpic"]') : null;
    const id = lane && lane.dataset.layer;
    const lay = id ? layerOf(editOf(node), id) : null;
    const ok = !!(lay && id !== self && id !== BASE_ID && !lay.lock);
    _laneMark(node, ok ? lane : null);
    return ok ? id : null;
}
function _laneMark(node, lane) {
    const st = node._plsTl;
    if (st._laneMarked && st._laneMarked !== lane) { st._laneMarked.style.outline = ""; st._laneMarked = null; }
    if (lane) { lane.style.outline = "2px dashed #f0c87a"; lane.style.outlineOffset = "-2px"; st._laneMarked = lane; }
}
/* v1198: a piece onto track `to` at film frame `at`; from = {id, j, pcs} moves it (its place on the old track
 * becomes empty), null = a copy. ONE save = one undo step. */
function _moveAcross(node, src, piece, at, to, from) {
    if (_planBusy(node)) return;
    let e = editOf(node);
    if (from) e = setLayer(e, from.id, { pieces: deletePiece(from.pcs, from.j) });
    e = dropOnto(e, to, src, { k: 0, a: piece.a, b: piece.b, speed: piece.speed || 1 }, at);
    const tl = layerOf(e, to);
    if (tl && !tl.name) e = setLayer(e, to, { name: String(src.path).split("/").pop() });
    _saveEdit(node, e);
    node._plsTl.active = to;
    node._plsTl.state.textContent = (from ? "moved" : "copied") + " onto " + ((tl && tl.name) || to) + " at " + timeLabel(at / FPS) + " -- it overwrites what lay there";
}
function _saveLayerPieces(node, id, pieces) {
    if (_planBusy(node)) return;
    _saveEdit(node, setLayer(editOf(node), id, { pieces: normLayer(pieces) }));
}
function _cutButtons(node) {
    const st = node._plsTl, D = st.plan, pl = st.player;
    if (!st.splitBtn || !pl) return;
    const ch = _activeChain(node), pcs = ch.pcs || [];
    const f = Math.round((pl.t || 0) * FPS);
    const j = pcs.length ? pieceAt(pcs, f) : -1;
    const can = !ch.lock && !!(pcs.length && j >= 0 && !isGap(pcs[j]) && splitAt(pcs, f));
    const hj = !ch.lock && pcs.length ? healableAt(pcs, f) : -1;
    const who = ch.id === BASE_ID ? "" : " (layer " + ((ch.row && ch.row.name) || ch.id) + ")";
    st.splitBtn.disabled = !can; st.splitBtn.style.opacity = can ? "1" : "0.5";
    st.splitBtn.title = can ? "cut at the playhead (" + timeLabel(f / FPS) + ") -- picture and sound" + who : (ch.lock ? "the active layer is locked" : "put the playhead inside a clip (not on a cut) to cut it" + who);
    st.healBtn.disabled = hj < 0; st.healBtn.style.opacity = hj >= 0 ? "1" : "0.5";
    st.healBtn.title = hj >= 0 ? "heal the cut at " + timeLabel(starts(pcs)[hj] / FPS) + " -- one piece again" + who : "put the playhead on a cut inside a clip (within 0.25 s) to heal it" + who;
    if (st.delBtn) {
        const sel = st.sel || {};
        const ok = _selPiece(node) !== null;
        st.delBtn.disabled = !ok; st.delBtn.style.opacity = ok ? "1" : "0.5";
        st.delBtn.title = ok ? "delete the selected piece -- it becomes empty, the layers below show through (Delete key)" : (sel.kind ? "select a piece (click it) to delete it" : "click a piece first");
    }
    _linkButton(node);
}
/* the selected piece as {ch, j} -- a piece of the render lane ("clip") or of
 * another layer ("lpiece"), unlocked, not a gap; else null */
function _selPiece(node) {
    const st = node._plsTl, sel = st.sel || {}, D = st.plan;
    if (!D || !D.plan) return null;
    if (sel.kind === "clip") {
        const ch = _chainOf(node, BASE_ID);
        const p = ch.pcs[sel.i];
        return p && !isGap(p) && !ch.lock ? { ch, j: sel.i } : null;
    }
    if (sel.kind === "lpiece") {
        const ch = _chainOf(node, sel.id);
        const p = ch.id === sel.id ? ch.pcs[sel.i] : null;
        return p && !isGap(p) && !ch.lock ? { ch, j: sel.i } : null;
    }
    return null;
}
function _deleteSel(node) {
    const x = _selPiece(node);
    if (!x) return false;
    node._plsTl.sel = { kind: "none" };
    x.ch.save(deletePiece(x.ch.pcs, x.j));
    return true;
}
/* v1038: A/V linked (default) / unlinked -- the NLE switch */
function _linkButton(node) {
    const st = node._plsTl, b = st.linkBtn;
    if (!b) return;
    const on = isLinked(node);
    // v1153: the order switch lit while the tracks stand by type
    // v1172: off = the tinted button of the row (BTN_BG), on = lit as before
    if (st.typeBtn) { const t = byType(node); st.typeBtn.style.color = t ? "#111" : "#e8eefb"; st.typeBtn.style.background = t ? "#9fc8ff" : BTN_BG; }
    b.textContent = on ? "A/V linked" : "A/V unlinked";
    b.style.color = on ? "#e8eefb" : "#111";
    b.style.background = on ? BTN_BG : "#ffcc80";
    b.title = on ? "picture and sound move together: a cut goes through both, a dragged cut moves both (Alt+drag: one alone)\nclick: unlink -- the sound's cut gets its own grip (J-/L-cut)"
        : "unlinked: dragging the picture's cut leaves the sound's cut where it is, the sound's grip moves the sound alone (J-/L-cut)\nclick: link again";
}
function _split(node) {
    const st = node._plsTl, D = st.plan;
    if (!D || !D.plan) return;
    const f = Math.round((st.player.t || 0) * FPS);
    const ch = _activeChain(node);
    if (ch.lock) return;
    if (ch.id !== BASE_ID) {
        const at = pieceAt(ch.pcs, f);
        if (at < 0 || isGap(ch.pcs[at])) { st.state.textContent = "cut: the playhead stands in an empty part of this track"; return; }
        const out = splitAt(ch.pcs, f); if (out) ch.save(out); return;
    }
    const j = pieceAt(D.plan.pieces, f);
    const out = splitAt(D.plan.pieces, f);
    // v1038: the new cut comes selected -- its line, its grip, its diamond
    if (out) { st.sel = { kind: "transition", i: j }; _savePieces(node, out); }
}
function _heal(node) {
    const st = node._plsTl, D = st.plan;
    if (!D || !D.plan) return;
    const ch = _activeChain(node);
    if (ch.lock) return;
    const j = healableAt(ch.pcs, Math.round((st.player.t || 0) * FPS));
    if (j >= 0) ch.save(heal(ch.pcs, j));
}

/* v1044: the lane sizes as a pick list -- the nearest size is shown */
export function sizeOf(h, col = 1) {
    // col 1 = a picture track's heights, 2 = a sound track's
    let best = SIZES[0][0], d = Infinity;
    for (const z of SIZES) { const x = Math.abs(z[col] - h); if (x < d) { d = x; best = z[0]; } }
    return best;
}
function _sizeSel(cur, title, onPick) {
    const sel = _el("select", INPUT + "height:16px;font-size:10px;padding:0 2px;flex:0 0 auto;");
    for (const [n] of SIZES) { const o = document.createElement("option"); o.value = n; o.textContent = n; sel.appendChild(o); }
    sel.value = cur; sel.title = title; sel.dataset.pls = "size";
    sel.addEventListener("pointerdown", (e) => e.stopPropagation());
    sel.addEventListener("change", () => { const z = SIZES.find((x) => x[0] === sel.value); if (z) onPick(z[1], z[2]); });
    return sel;
}
/* v1158 (Frank 17.10.: the clips that came in "not in the minimal view"): a NEW track (a run's clip, a drop, a file,
 * takes -> tracks, "+ new track") starts at the height V1/A1 have now -- the same pictures and waveform as the render,
 * not the smallest size (S). Stored like a size picked by hand; a track that has a height keeps it. */
function _seedTrackH(node, id) {
    node.properties = node.properties || {};
    const all = Object.assign({}, node.properties[LH_KEY] || {});
    if (all[id]) return;
    const v1 = laneHeights(node);
    all[id] = { clip: v1.clip, audio: v1.audio };
    node.properties[LH_KEY] = all;
    node.properties[LH_KEY][id] = layerHeights(node, id);       // clamped
}
function _setLayerSize(node, id, clip, audio) {
    node.properties = node.properties || {};
    const all = Object.assign({}, node.properties[LH_KEY] || {}), cur = layerHeights(node, id);
    all[id] = { clip: clip == null ? cur.clip : clip, audio: audio == null ? cur.audio : audio };
    node.properties[LH_KEY] = all;
    node.properties[LH_KEY][id] = layerHeights(node, id);
    node.setDirtyCanvas?.(true, true);
    _applyLanes(node);
    _render(node);
}

/* ------------------------------------------------------------ v1045 the track heads */
/* Premiere's track headers (Frank 30.09.): a flat icon button, one size for
 * all (the arrows as big as the rest); lit = a colour, off = dimmed */
const HB = "background:transparent;color:#c4c4c4;border:1px solid transparent;border-radius:3px;font-size:12px;line-height:18px;width:22px;height:20px;padding:0;cursor:pointer;flex:0 0 auto;text-align:center;";
function _hb(label, title, on, fn, tag) {
    const lit = on === "lit" ? "background:#e0a030;color:#111;" : on === "solo" ? "background:#3fa66b;color:#111;" : on === false ? "opacity:0.35;" : "";
    const b = _el("button", HB + lit, label);
    b.title = title; if (tag) b.dataset.pls = tag;
    b.addEventListener("pointerdown", (e) => e.stopPropagation());
    b.addEventListener("mouseenter", () => { if (!lit) b.style.borderColor = "#555"; });
    b.addEventListener("mouseleave", () => { b.style.borderColor = "transparent"; });
    b.addEventListener("click", (e) => { e.stopPropagation(); fn(); });
    return b;
}
/* the track's name chip: "V2" / "A2" -- blue when it is the ACTIVE track */
function _chip(text, act) {
    const c = _el("span", "flex:0 0 auto;width:24px;height:18px;line-height:18px;text-align:center;font-size:10px;font-weight:bold;border-radius:3px;"
        + (act ? "background:#2d6ec9;color:#fff;" : "background:#2a2a2a;color:#9a9a9a;"), text);
    c.dataset.pls = "chip";
    return c;
}
/* a picture track's head: lock, Vn, eye, up, down, reset, x, size */
function _vHead(node, el, lay, num, i, n, row) {
    const st = node._plsTl, base = lay.id === BASE_ID, e0 = () => editOf(node);
    const act = (st.active || BASE_ID) === lay.id;
    el.replaceChildren();
    el.dataset.pls = "layerhead"; el.dataset.layer = lay.id; el.dataset.kind = "v";
    el.style.background = act ? "#1f2733" : "#1c1c1c";
    const put = (patch) => _saveEdit(node, setLayer(e0(), lay.id, patch));
    // the render stays V1 at the bottom; the other tracks move among themselves
    const nb = n - 1;                                  // tracks above the render
    el.append(
        _hb(lay.lock ? "\u{1F512}" : "\u{1F513}", lay.lock ? "locked: no cut, move or delete -- click: unlock" : "click: lock this track (only look at it)", lay.lock ? "lit" : true, () => put({ lock: !lay.lock }), "lock"),
        _chip("V" + num, act),
        _hb("\u{1F441}", lay.video !== false ? "picture on -- click: hide it (the tracks below show)" : "picture off -- click: show it", lay.video !== false, () => put({ video: lay.video === false }), "eye"),
        _hb("\u25b2", base ? "the render stays V1" : "one track up (upper tracks win)", !base && i > 0, () => { if (!base && i > 0) _saveEdit(node, moveLayer(e0(), lay.id, -1)); }, "up"),
        _hb("\u25bc", base ? "the render stays V1" : "one track down", !base && i < nb - 1, () => { if (!base && i < nb - 1) _saveEdit(node, moveLayer(e0(), lay.id, 1)); }, "down"),
        _hb("\u21ba", base ? "reset: the render's cuts go (the film cuts where the prompt cuts)" : "reset: the file again, whole -- no cuts, no gaps", !lay.lock, () => _resetLayer(node, lay, row), "reset"),
        // v1219: this track's scene clips back to their scenes (V1: the same as reset)
        _hb("\u21e4", base ? "snap: the render's pieces back to the prompt's cuts" : "snap: this track's scene clips back to where their scenes start (scene_<k> in the name)", !lay.lock && (base || layerSrcs(lay).length > 0), () => {
            if (lay.lock) return;
            if (window.confirm(base ? "Snap V1 back to the prompt's cuts? Its cuts, trims and speeds go." : "Snap the track '" + (lay.name || lay.id) + "' back to its scenes? Its scene clips go to where their scenes start; other clips stay.")) _snap(node, base ? "base" : lay.id);
        }, "snap"),
        _hb("\u2715", base ? "the render track stays (hide it with the eye)" : "delete this track (V + A) from the timeline -- the file stays on disk", !base && !lay.lock, () => {
            if (base || lay.lock) return;
            if (window.confirm("Delete the track V" + num + " (" + (lay.name || lay.id) + ") from the timeline? The file stays on disk.")) {
                if (st.active === lay.id) st.active = BASE_ID;
                _saveEdit(node, removeLayer(e0(), lay.id));
            }
        }, "remove"));
    const hs = base ? laneHeights(node).clip : layerHeights(node, lay.id).clip;
    el.appendChild(_sizeSel(sizeOf(hs), "the height of this track (S / M / L / XL) -- or pull the grip under it",
        base ? (c, a2) => { _setLaneH(node, "clip", c, false); _setLaneH(node, "audio", a2, true); } : (c) => _setLayerSize(node, lay.id, c, null)));
    if (!base && lay.src && lay.src.kind === "empty") el.appendChild(_hb("\u2026", "file...: pick a clip on your computer (the native dialog) -- it is read where it lies", true, () => _pickInto(node, lay.id), "pick"));
    const nm = base ? "render" : (lay.name || (lay.src && lay.src.kind === "empty" ? "empty" : lay.id));
    let info = "";
    if (row && row.clip) { const c = row.clip; info = (Math.abs((c.fps || FPS) - FPS) > 0.01 ? Number(c.fps).toFixed(2).replace(/\.?0+$/, "") + " fps -> 24, " : "") + (c.w && c.h ? c.w + "x" + c.h : ""); }
    else if (row && row.error) info = row.error;
    el.title = "V" + num + ": " + nm + (info ? " (" + info + ")" : "") + (row && row.clip ? "\n" + row.clip.path : "") + "\nclick: the ACTIVE track (cut / heal / delete work on it)";
    el.onpointerdown = (ev) => { ev.stopPropagation(); if (st.active !== lay.id) { st.active = lay.id; _render(node); } };
}
/* a sound track's head: lock, An, M (mute), S (solo), size */
function _aHead(node, el, lay, num) {
    const st = node._plsTl, base = lay.id === BASE_ID, e0 = () => editOf(node);
    const act = (st.active || BASE_ID) === lay.id;
    el.replaceChildren();
    el.dataset.pls = "layerhead"; el.dataset.layer = lay.id; el.dataset.kind = "a";
    el.style.background = act ? "#1f2733" : "#1c1c1c";
    const put = (patch) => _saveEdit(node, setLayer(e0(), lay.id, patch));
    el.append(
        _hb(lay.lock ? "\u{1F512}" : "\u{1F513}", lay.lock ? "locked -- click: unlock" : "click: lock this track", lay.lock ? "lit" : true, () => put({ lock: !lay.lock }), "lock"),
        _chip("A" + num, act),
        _hb("M", lay.audio !== false ? "mute -- click: this track is silent (the sound below plays)" : "muted -- click: sound on", lay.audio === false ? "lit" : true, () => put({ audio: lay.audio === false }), "mute"),
        _hb("S", lay.solo ? "solo -- only the solo tracks sound; click: off" : "solo: hear ONLY the solo tracks", lay.solo ? "solo" : true, () => put({ solo: !lay.solo }), "solo"));
    const hs = base ? laneHeights(node).audio : layerHeights(node, lay.id).audio;
    el.appendChild(_sizeSel(sizeOf(hs, 2), "the height of this sound track -- or pull the grip under it",
        base ? (c, a2) => _setLaneH(node, "audio", a2, true) : (c, a2) => _setLayerSize(node, lay.id, null, a2)));
    el.title = "A" + num + ": the sound of " + (base ? "the render" : (lay.name || lay.id)) + "\nclick: the ACTIVE track";
    el.onpointerdown = (ev) => { ev.stopPropagation(); if (st.active !== lay.id) { st.active = lay.id; _render(node); } };
}
function _resetLayer(node, lay, row) {
    if (lay.lock) return;
    if (lay.id === BASE_ID) {
        // v1088 (audit C10): V1's reset asks like every other track's
        if (!window.confirm("Reset V1? The render's cuts, trims and speeds go -- the film cuts where the prompt cuts.")) return;
        _saveEdit(node, _clearCuts(node, editOf(node))); return;
    }
    if (!row || !row.plan) return;
    if (layerSrcs(lay).length > 1) {    // v1198: a track of several files -- each whole again where it was laid
        if (!window.confirm("Reset the track '" + (lay.name || lay.id) + "'? Its " + layerSrcs(lay).length + " files come back whole, each where it was laid -- cuts and gaps go.")) return;
        _saveLayerPieces(node, lay.id, resetMulti(lay, (k) => (clipK(row, k) || {}).frames));
        return;
    }
    if (!row.clip) return;
    if (!window.confirm("Reset the track '" + (lay.name || lay.id) + "'? The file comes back whole -- its cuts and gaps go.")) return;
    _saveLayerPieces(node, lay.id, resetPieces(row.plan.pieces, row.clip.frames));
}
/* every track: the render's heads (always), the other tracks' rows -- the
 * picture rows above V1 (top first), the sound rows below A1 (A2 first) */
function _stackRows(node, pct) {
    const st = node._plsTl, e = editOf(node), D = st.plan, film = D && D.film;
    for (const k of ["above", "below"]) { st[k].replaceChildren(); st[k].style.display = "none"; }
    const L = stackLayers(e);
    const bl = L.find((x) => x.id === BASE_ID) || {};
    st.clips.style.opacity = bl.video === false ? "0.35" : "1";
    st.audio.style.opacity = bl.audio === false || (L.some((x) => x.solo) && !bl.solo) ? "0.35" : "1";
    if (st.active && !L.some((x) => x.id === st.active)) st.active = BASE_ID;
    const rowOf = (id) => ((film && film.layers) || []).find((r) => r.id === id) || null;
    const others = L.filter((x) => x.id !== BASE_ID);
    st.layW = st.clips.clientWidth || 0;              // v1157: the width these rows' strips are laid out for
    _vHead(node, st.v1Head, bl, 1, others.length, others.length + 1, rowOf(BASE_ID));
    _aHead(node, st.a1Head, bl, 1);
    const anySolo = L.some((x) => x.solo);
    others.forEach((lay, i) => {
        const num = others.length - i + 1;            // the top track has the highest number
        const r = _layerRows(node, lay, num, i, others.length + 1, rowOf(lay.id), pct, anySolo);
        st.above.appendChild(r.v);
        if (byType(node)) st.below.insertBefore(r.a, st.below.firstChild);   // A2 right under A1
        else st.above.appendChild(r.a);                                        // v1153: the pair together
    });
    if (st.above.children.length) st.above.style.display = "flex";
    if (st.below.children.length) st.below.style.display = "flex";
    // v1153: the render's clamp (V1/A1 = the A/V switch)
    _bracket(node, st.v1Row, laneHeights(node).clip, laneHeights(node).audio, isLinked(node), () => {
        node.properties = node.properties || {}; node.properties[LINK_KEY] = !isLinked(node); node.setDirtyCanvas?.(true, true); _render(node); },
        "V1/A1: " + (isLinked(node) ? "linked -- a cut or a drag moves picture and sound; click: unlink (J-/L-cuts: each part alone)" : "unlinked -- each part alone (J-/L-cuts); click: link"));
}
/* v1153: the clamp at the left edge of a pair -- a bracket from the picture row down to the sound row */
function _bracket(node, row, hV, hA, linked, fn, title) {
    if (!row) return;
    let b = row.querySelector(':scope > [data-pls="clamp"]');
    if (!b) {
        b = _el("div", "position:absolute;left:1px;top:3px;width:6px;z-index:9;cursor:pointer;box-sizing:border-box;border:2px solid #777;border-right:none;border-radius:3px 0 0 3px;");
        b.dataset.pls = "clamp";
        b.addEventListener("pointerdown", (e) => e.stopPropagation());
        row.style.overflow = "visible";
        row.appendChild(b);
    }
    const on = byType(node) ? false : !!linked;
    b.style.height = (byType(node) ? hV - 6 : hV + LANE_GAP + hA - 6) + "px";
    b.style.borderColor = on ? "#ffb300" : "#555";
    b.style.borderStyle = on ? "solid" : "dashed";
    b.style.display = byType(node) ? "none" : "";
    b.title = title;
    b.onclick = (e) => { e.stopPropagation(); fn(); };
}
/* a track = its picture row (in the video block) and its sound row (in the
 * audio block); the same pieces, the same cuts on both */
function _layerRows(node, lay, num, i, n, row, pct, anySolo) {
    const st = node._plsTl, H = layerHeights(node, lay.id);
    const laneCss = "position:relative;flex:1 1 auto;min-width:0;overflow:hidden;";
    const vh = _el("div", GUT), ah = _el("div", GUT);
    _vHead(node, vh, lay, num, i, n, row);
    _aHead(node, ah, lay, num);
    const pic = _el("div", laneCss + "height:" + H.clip + "px;background:#141414;" + (lay.video === false ? "opacity:0.35;" : ""));
    const snd = _el("div", laneCss + "height:" + H.audio + "px;background:#101418;" + (lay.audio === false || (anySolo && !lay.solo) ? "opacity:0.35;" : ""));
    pic.dataset.pls = "layerpic"; snd.dataset.pls = "layersnd"; pic.dataset.layer = snd.dataset.layer = lay.id;
    const vRow = _row(vh, pic), aRow = _row(ah, snd);
    vRow.dataset.pls = "layer"; vRow.dataset.layer = lay.id; aRow.dataset.pls = "layeraudio"; aRow.dataset.layer = lay.id;
    vRow.style.position = "relative";
    const linked = layerLinked(node, lay), aoff = (row && row.aoff) || 0;
    _bracket(node, vRow, H.clip, H.audio, linked, () => _saveEdit(node, setLayer(editOf(node), lay.id, { link: !linked })),
        "V" + num + "/A" + num + ": " + (linked ? "linked -- a drag moves picture and sound; click: unlink (the sound slides alone)" : "unlinked -- drag the sound lane: the sound alone; drag a piece: the picture alone; click: link again")
        + (aoff ? "\nthe sound stands " + (aoff > 0 ? "+" : "") + (aoff / FPS).toFixed(2) + " s against the picture" : ""));
    const hasClip = !!(row && row.plan && row.plan.pieces && row.plan.pieces.length && !(lay.src && lay.src.kind === "empty"));
    if (!linked && !lay.lock && hasClip) {    // v1154 (audit): an empty track has no sound to slide
        snd.style.cursor = "grab";
        snd.title = "A" + num + " unlinked: drag = the sound alone, against the picture" + (aoff ? " (now " + (aoff > 0 ? "+" : "") + (aoff / FPS).toFixed(2) + " s)" : "") + "\ndouble-click: back on the picture";
        snd.addEventListener("pointerdown", (ev) => {
            if (ev.button !== 0) return;
            ev.preventDefault(); ev.stopPropagation();
            if (isDouble(st, "aoff:" + lay.id)) { _saveEdit(node, setLayer(editOf(node), lay.id, { aoff: 0 })); return; }
            try { snd.setPointerCapture(ev.pointerId); } catch (x) { /* no capture */ }
            const x0 = ev.clientX, w = st.clips.getBoundingClientRect().width || 1, ax = st.ax || 1;
            let dF = 0;
            const mv = (e) => { dF = Math.round((e.clientX - x0) / w * ax * FPS); snd.style.transform = "translateX(" + ((e.clientX - x0) * _pxScale(st.clips)) + "px)"; };
            const up = () => { snd.removeEventListener("pointermove", mv); snd.removeEventListener("pointerup", up); snd.style.transform = "";
                if (dF) _saveEdit(node, setLayer(editOf(node), lay.id, { aoff: aoff + dF })); };
            snd.addEventListener("pointermove", mv); snd.addEventListener("pointerup", up);
        });
    }
    if (!lay.lock && lay.src && lay.src.kind === "empty") _dropZone(node, pic, (dt, zone, at) => _dropInto(node, lay.id, dt, zone, at));
    const P = row && row.plan;
    if (!row || (lay.src && lay.src.kind === "empty") || !P || !P.pieces.length) {
        pic.appendChild(_el("div", "position:absolute;left:10px;top:50%;transform:translateY(-50%);font-size:10px;color:#6f6f6f;pointer-events:none;white-space:nowrap;",
            row && row.error ? row.error : (lay.src && lay.src.kind !== "empty")
                ? "every piece of " + (lay.name || "this clip") + " deleted -- \u21ba brings the whole clip back"
                : "empty track -- drop a video here (upper half: copy, lower half: read it where it lies) or \u2026 in the head"));
        _layerDivider(node, pic, lay, "clip");
        _layerDivider(node, snd, lay, "audio");
        return { v: vRow, a: aRow };
    }
    const pcs = P.pieces, st0 = starts(pcs);
    const fK = (k) => (Number(k) === GAP ? Infinity : ((clipK(row, k) || {}).frames || Infinity));   // v1198: per file
    const save = (q) => _saveLayerPieces(node, lay.id, q);
    const laneW = st.clips.clientWidth || 0, ax = st.ax || 1, segH = H.clip - 2;
    const multi = layerSrcs(lay).length > 1;
    P.segments.forEach((sg, j) => {
        if (sg.gap) return;
        const c = clipK(row, sg.k) || {}, url = c.strip ? _viewURL(c.strip) : "";
        const pname = multi ? (c.name || ("file " + (sg.k + 1))) : (lay.name || "clip");
        const need = sg.film_out - sg.film_in;
        const picked = st.sel && st.sel.kind === "lpiece" && st.sel.id === lay.id && st.sel.i === j;
        const b = _el("div", "position:absolute;top:1px;height:" + segH + "px;overflow:hidden;border-radius:2px;box-sizing:border-box;background:#2a2340;"
            + "cursor:" + (lay.lock ? "default" : "grab") + ";left:" + pct(sg.film_in / FPS) + ";width:" + pct(need / FPS) + ";"
            + (picked ? "outline:2px solid #f0c87a;" : "border:1px solid " + (sg.missing ? "#555;border-style:dashed;" : "#6c5aa0;")));
        b.dataset.pls = "lclip";
        const film = _el("div", "position:absolute;top:0;bottom:0;left:0;right:0;pointer-events:none;");
        b.appendChild(film);
        const boxW = laneW * (need / FPS) / ax - 2;
        if (url && c.strip_tiles && c.w && c.h && boxW > 0) {
            const nat = (c.fps || FPS) / FPS;            // strip tiles count the file's own frames
            const srcAt = (x) => srcTime(sg, (sg.film_in + x / boxW * need) / FPS) * FPS * nat;
            for (const tl of tilePlan(boxW, segH - 2, c.w / c.h, c.strip_tiles, c.strip_step || 1, srcAt)) {
                film.appendChild(_el("div", "position:absolute;top:0;bottom:0;left:" + tl.x.toFixed(1) + "px;width:" + tl.w.toFixed(1) + "px;background-repeat:no-repeat;opacity:0.92;"
                    + "border-right:1px solid #0008;box-sizing:border-box;background-image:url('" + url + "');background-size:" + (tl.tw * c.strip_tiles).toFixed(1) + "px " + (segH - 2) + "px;"
                    + "background-position:" + (-tl.idx * tl.tw).toFixed(1) + "px 0;"));
            }
        } else if (url && sg.frames) {
            // v1157: not laid out yet -- the strip stretched over the material, as V1 (v1036); never an empty
            // track. _stripCheck draws the pictures in their aspect as soon as the lanes have their width
            const hold = (sg.hold_head || 0) + (sg.hold_tail || 0), shown = (need - hold) * (sg.speed || 1);
            const wPct = 100 * sg.frames / Math.max(1, shown) * (need - hold) / need;
            film.appendChild(_el("div", "position:absolute;top:0;bottom:0;background-repeat:no-repeat;background-size:100% 100%;opacity:0.9;"
                + "left:" + (100 * (sg.hold_head || 0) / need - wPct * (sg.src_in || 0) / sg.frames) + "%;width:" + wPct + "%;background-image:url('" + url + "');"));
        }
        if (sg.hold_tail) b.appendChild(_el("div", "position:absolute;top:0;bottom:0;right:0;width:" + (100 * sg.hold_tail / need) + "%;background:repeating-linear-gradient(90deg,#e5737355 0 2px,transparent 2px 5px);", ""));
        // the clip's name on its top edge, as the editors do
        b.appendChild(_el("div", "position:absolute;left:0;right:0;top:0;height:12px;padding-left:4px;font-size:9px;line-height:12px;color:#fff;background:#00000080;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;pointer-events:none;",
            pname + (sg.speed !== 1 ? "  [" + speedLabel(sg.speed) + "]" : "")));
        b.title = (lay.name || lay.id) + (multi ? " -- " + pname : "") + "\nfilm " + (sg.film_in / FPS).toFixed(3) + " - " + (sg.film_out / FPS).toFixed(3) + " s  <-  clip " + (sg.src_in / FPS).toFixed(3) + " - " + (sg.src_out / FPS).toFixed(3) + " s"
            + (lay.lock ? "\nlocked" : "\nclick: select (Delete / \u{1F5D1} = empty it)  |  drag: move it along the track  |  Alt+drag: slip (which part of the clip plays)"
                + "\ndrag onto another track: move it there (Ctrl+drag: a copy) -- it overwrites what lies there");
        b.addEventListener("pointerdown", (ev) => {
            if (ev.button !== 0) return;
            ev.preventDefault(); ev.stopPropagation();
            try { b.setPointerCapture(ev.pointerId); } catch (x) { /* no capture */ }
            st.active = lay.id;
            const x0 = ev.clientX, y0 = ev.clientY, w = st.clips.getBoundingClientRect().width || 1, alt = !!ev.altKey;
            let moved = false, dF = 0, to = null;
            const mv = (e) => { if (Math.abs(e.clientX - x0) > 2 || Math.abs(e.clientY - y0) > 6) moved = true; if (!moved || lay.lock) return;
                dF = Math.round((e.clientX - x0) / w * ax * FPS);
                // v1198 (Frank 21.10.: "einen Clip eines Tracks in einen anderen Track reinziehen"): over another track's lane
                to = alt ? null : _laneUnder(node, e.clientX, e.clientY, lay.id);
                if (alt) film.style.transform = "translateX(" + ((e.clientX - x0) * _pxScale(st.clips)) + "px)";
                else b.style.transform = "translateX(" + ((e.clientX - x0) * _pxScale(st.clips)) + "px)"; };
            const up = (e) => { b.removeEventListener("pointermove", mv); b.removeEventListener("pointerup", up);
                _laneMark(node, null);
                if (moved && !lay.lock && to && e) {
                    const sx = layerSrcs(lay)[sg.k] || lay.src, copy = !!(e.ctrlKey || e.metaKey);
                    _moveAcross(node, { kind: sx.kind, path: sx.path, at: sx.at }, pcs[j], Math.max(0, st0[j] + dF), to, copy ? null : { id: lay.id, j, pcs });
                    return;
                }
                if (moved && !lay.lock) {
                    if (alt) save(slipPiece(pcs, j, -dF * sg.speed, fK));
                    else if (linked) save(movePiece(pcs, j, dF));
                    // v1153: unlinked -- the picture alone: the pieces move, the sound stays where it was
                    else if (!_planBusy(node)) {
                        // v1154 (audit): the sound keeps its place by the shift the picture REALLY made -- movePiece
                        // clamps at the gaps beside it (a +30 drag against a neighbour moved nothing, the sound 30)
                        const q = movePiece(pcs, j, dF), moved = movedStart(pcs, j, q) - st0[j];
                        if (moved) _saveEdit(node, setLayer(setLayer(editOf(node), lay.id, { pieces: q }), lay.id, { aoff: aoff - moved }));
                        else _render(node);
                    }
                }
                else { st.sel = { kind: "lpiece", id: lay.id, i: j }; try { st.root.focus({ preventScroll: true }); } catch (x) { /* no focus */ } _playerSeek(node, st0[j] / FPS); _render(node); } };
            b.addEventListener("pointermove", mv); b.addEventListener("pointerup", up);
        });
        pic.appendChild(b);
    });
    // the sound row: the waveform where each piece's sound plays
    (P.asegments || []).forEach((sg) => {
        if (sg.gap) return;
        const c = clipK(row, sg.k) || {};    // v1198: the file of this piece
        const need = sg.film_out - sg.film_in, pad = (sg.pad_head || 0) + (sg.pad_tail || 0);
        const ab = _el("div", "position:absolute;top:1px;bottom:1px;overflow:hidden;border-radius:2px;box-sizing:border-box;background:" + (c.audio ? "#16343c" : "#1a1d20") + ";border:1px solid " + (c.audio ? "#2f6a74" : "#2a2d30") + ";pointer-events:none;"
            + "left:" + pct(sg.film_in / FPS) + ";width:" + pct(need / FPS) + ";");
        if (c.audio && c.wave && sg.frames) {
            const shown = (need - pad) * sg.speed, wPct = 100 * sg.frames / Math.max(1, shown) * (need - pad) / need;
            ab.appendChild(_el("div", "position:absolute;top:0;bottom:0;background-repeat:no-repeat;background-size:100% 100%;opacity:0.95;"
                + "left:" + (100 * (sg.pad_head || 0) / need - wPct * sg.src_in / sg.frames) + "%;width:" + wPct + "%;background-image:url('" + _viewURL(c.wave) + "');"));
        } else if (!c.audio) ab.appendChild(_el("div", "position:absolute;left:4px;top:50%;transform:translateY(-50%);font-size:9px;color:#607d8b;white-space:nowrap;", "no sound -- the sound below plays"));
        snd.appendChild(ab);
    });
    // the grips: a line on every cut (trim), the amber diamond (speed), the
    // track's end -- on the picture row; the sound row shows the same cuts
    const C = endsOf(pcs);
    if (!lay.lock) {
        const mkGrip = (f, key, title, onDrag, onDouble, col) => {
            const g = _el("div", "position:absolute;top:0;bottom:0;width:12px;margin-left:-6px;cursor:ew-resize;z-index:5;left:" + pct(f / FPS) + ";");
            g.appendChild(_el("div", "position:absolute;top:0;bottom:0;left:5px;width:2px;pointer-events:none;background:" + col + ";box-shadow:0 0 0 1px #000a;"));
            g.dataset.pls = "lgrip"; g.title = title;
            _drag(node, g, st.clips, key, (sec, done) => { g.style.left = pct(sec); if (done) onDrag(Math.round(sec * FPS)); }, onDouble, () => { st.active = lay.id; _render(node); });
            pic.appendChild(g);
        };
        for (let j = 1; j < pcs.length; j++) {
            const heals = pcs[j - 1].k === pcs[j].k && pcs[j - 1].b === pcs[j].a && !isGap(pcs[j]);
            const edge = isGap(pcs[j - 1]) ? "the start of a piece" : isGap(pcs[j]) ? "the end of a piece" : "a cut";
            mkGrip(C[j - 1], "lcut:" + lay.id + ":" + j, edge + " at " + timeLabel(C[j - 1] / FPS) + " -- drag: trim (the gap or the piece beside gives or takes)" + (heals ? "\ndouble-click: heal" : ""),
                (f) => save(trimSeam(pcs, j, f, fK)), heals ? () => save(heal(pcs, j)) : null, heals ? "#ffffff" : "#b39ddb");
        }
        const end = C[C.length - 1];
        mkGrip(end, "lend:" + lay.id, "the track's material ends at " + timeLabel(end / FPS) + " -- drag: its last piece plays more or less of the clip",
            (f) => save(trimEnd(pcs, f, fK)), null, "#b0bec5");
        pcs.forEach((p, j) => {
            if (isGap(p)) return;
            const e1 = st0[j] + pieceLen(p);
            const kn = _el("div", "position:absolute;bottom:0;width:12px;height:12px;margin-left:-6px;cursor:col-resize;z-index:6;left:" + pct(e1 / FPS) + ";");
            kn.appendChild(_el("div", "position:absolute;left:1px;top:1px;width:9px;height:9px;transform:rotate(45deg);background:#ffb300;border:1px solid #000;pointer-events:none;"));
            kn.dataset.pls = "lstretch";
            kn.title = "speed of this piece: " + speedLabel(p.speed) + " -- drag right = slower, left = faster (the pieces after it move)  |  double-click: 100%";
            _drag(node, kn, st.clips, "lstretch:" + lay.id + ":" + j, (sec, done) => { kn.style.left = pct(sec); if (done) save(stretch(pcs, j, Math.max(SEG_MIN, Math.round(sec * FPS) - st0[j]))); },
                () => { const q = pcs.map((x) => Object.assign({}, x)); q[j].speed = 1; save(q); }, null);
            pic.appendChild(kn);
        });
    }
    for (let j = 1; j < pcs.length; j++) snd.appendChild(_el("div", "position:absolute;top:0;bottom:0;width:0;border-left:1px solid #b39ddb88;pointer-events:none;left:" + pct(C[j - 1] / FPS) + ";"));
    // past the edit span: dimmed, not exported
    const Lf = filmLen(st.plan);
    if (Lf && P.length > Lf) for (const el of [pic, snd]) el.appendChild(_el("div", "position:absolute;top:0;bottom:0;right:0;left:" + pct(Lf / FPS) + ";background:#000a;border-left:1px dashed #ffb300;pointer-events:none;z-index:4;"));
    _layerDivider(node, pic, lay, "clip");
    _layerDivider(node, snd, lay, "audio");
    return { v: vRow, a: aRow };
}
/* a track's height: pulled at its row's bottom edge; the rows below move */
function _layerDivider(node, laneEl, lay, which) {
    const st = node._plsTl;
    const d = _el("div", "position:absolute;left:0;right:0;bottom:0;height:6px;cursor:ns-resize;z-index:7;");
    const pill = _el("div", "position:absolute;left:50%;bottom:1px;width:48px;height:3px;margin-left:-24px;border-radius:2px;background:#6a6a6a;pointer-events:none;");
    d.appendChild(pill);
    d.addEventListener("pointerenter", () => { pill.style.background = "#ffb300"; d.style.background = "#ffb30033"; });
    d.addEventListener("pointerleave", () => { pill.style.background = "#6a6a6a"; d.style.background = ""; });
    d.dataset.pls = which === "clip" ? "layerheight" : "layeraheight";
    d.title = "drag: this track's height  |  double-click: the smallest";
    d.addEventListener("pointerdown", (ev) => {
        ev.preventDefault(); ev.stopPropagation();
        const set = (h, done) => {
            node.properties = node.properties || {};
            const all = Object.assign({}, node.properties[LH_KEY] || {});
            all[lay.id] = Object.assign({}, layerHeights(node, lay.id), { [which]: h });
            node.properties[LH_KEY] = all;
            node.properties[LH_KEY][lay.id] = layerHeights(node, lay.id);       // clamped
            laneEl.style.height = layerHeights(node, lay.id)[which] + "px";
            _applyLanes(node);
            if (done) { node.setDirtyCanvas?.(true, true); _render(node); }
        };
        if (isDouble(st, "lh:" + which + ":" + lay.id)) { set(which === "clip" ? LAYER_CLIP_H : LAYER_AUDIO_H, true); return; }
        try { d.setPointerCapture(ev.pointerId); } catch (e) { /* no capture */ }
        const h0 = layerHeights(node, lay.id)[which], y0 = ev.clientY;
        const r = laneEl.getBoundingClientRect(), scale = (r.height || 1) / (laneEl.offsetHeight || 1);
        let last = h0, raf = 0;
        const up = () => { d.removeEventListener("pointermove", mv); d.removeEventListener("pointerup", up); d.removeEventListener("pointercancel", up); if (raf) cancelAnimationFrame(raf); set(last, true); };
        const mv = (e) => { if (e.buttons === 0) { up(); return; } last = h0 + (e.clientY - y0) / (scale || 1); if (!raf) raf = requestAnimationFrame(() => { raf = 0; set(last, false); }); };
        d.addEventListener("pointermove", mv); d.addEventListener("pointerup", up); d.addEventListener("pointercancel", up);
    });
    laneEl.appendChild(d);
}

/* ------------------------------------------------------------ v1044 a clip into a layer */
/* the origin PATH of an OS drop, when the drag carries one (Firefox:
 * file:/// URIs, the desktop client: file.path) -- the Media Loader's way */
export function pathFromUri(c) {
    c = String(c || "").trim().replace(/^"|"$/g, "");
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
export function dropPathOf(dt) {
    const cand = [];
    try {
        for (const t of ["text/uri-list", "text/x-moz-url", "text/plain", "DownloadURL"]) {
            const v = (dt.getData && dt.getData(t)) || "";
            if (v) cand.push(...String(v).split(/[\r\n]+/));
        }
    } catch (e) { /* getData may throw during dragover */ }
    const f = dt.files && dt.files.length ? dt.files[0] : null;
    if (f && f.path) cand.push(f.path);
    for (const c of cand) { const p = pathFromUri(c); if (p && VIDEO_RE.test(p)) return p; }
    return "";
}
function _join(folder, file) { return folder + (String(folder).includes("\\") && !String(folder).includes("/") ? "\\" : "/") + file; }
/* the two zones on a layer's picture lane while a drag is over it: upper
 * half = copy into input/cine_media/<project>/, lower half = read it where it
 * lies. Stacked, not side by side: WHERE along the lane you let go is the time
 * the clip starts, so the choice cannot take the x. */
function _dropZone(node, pic, onDrop) {
    let zones = null;
    // the zones are at least 40 px tall (a thin lane -- the new-track row --
    // grows them downwards while a drag is over it)
    const zoneOf = (ev) => { const r = (zones || pic).getBoundingClientRect(); return ev.clientY - r.top < r.height / 2 ? "copy" : "pin"; };
    const show = (ev) => {
        if (!zones) {
            zones = _el("div", "position:absolute;left:0;right:0;top:0;height:" + Math.max(40, pic.offsetHeight || 0) + "px;display:flex;flex-direction:column;z-index:12;pointer-events:none;");
            if (getComputedStyle(pic).overflow === "hidden") pic.style.overflow = "visible";
            zones.dataset.pls = "dropzones";
            for (const [k, t] of [["copy", "\u2b07 copy into input/cine_media"], ["pin", "\u{1F4CC} read it where it lies"]]) {
                const z = _el("div", "flex:1 1 0;display:flex;align-items:center;justify-content:center;font-size:10px;color:#111;border:2px dashed #111;box-sizing:border-box;", t);
                z.dataset.zone = k; zones.appendChild(z);
            }
            pic.appendChild(zones);
        }
        const hot = zoneOf(ev);
        for (const z of zones.children) z.style.background = z.dataset.zone === hot ? "#ffcc80" : "#80cbc4aa";
        // where the clip will start: a line at the pointer
        let mark = zones.querySelector("[data-pls=dropat]");
        if (!mark) { mark = _el("div", "position:absolute;top:0;bottom:0;width:0;border-left:2px solid #ffb300;"); mark.dataset.pls = "dropat"; zones.appendChild(mark); }
        mark.style.left = (ev.clientX - pic.getBoundingClientRect().left) + "px";
    };
    const hide = () => { if (zones) { zones.remove(); zones = null; pic.style.overflow = pic.dataset.pls === "newtrack" ? "" : "hidden"; } };
    pic.addEventListener("dragover", (ev) => { ev.preventDefault(); ev.stopPropagation(); try { ev.dataTransfer.dropEffect = "copy"; } catch (x) { /* view only */ } show(ev); });
    pic.addEventListener("dragleave", (ev) => { if (!pic.contains(ev.relatedTarget)) hide(); });
    pic.addEventListener("drop", (ev) => {
        ev.preventDefault(); ev.stopPropagation();
        const zone = zoneOf(ev), at = Math.round(_secFromX(pic, ev.clientX, node._plsTl.ax || 1) * FPS);
        hide();
        Promise.resolve(onDrop(ev.dataTransfer, zone, at)).catch((e) => { node._plsTl.state.textContent = "drop failed: " + e.message; });
    });
}
/* v1045: a video dropped on the new-track row opens a track on top and loads it */
function _dropNewTrack(node, dt, zone, at) {
    // v1082: the track is opened by the load itself, once the clip is read --
    // a refused drop (no video, a failed copy or probe) leaves no empty layer
    return _dropInto(node, NEW_TRACK, dt, zone, at);
}
const NEW_TRACK = "+new";
async function _json(url, opt) { const r = await api.fetchApi(url, opt); return r.json(); }
async function _dropInto(node, id, dt, zone, at) {
    const st = node._plsTl, m = _model(node);
    let src = null, note = "";
    if (zone === "pin") {
        const path = dropPathOf(dt);
        if (path) {
            const r = await _json("/uls/media/resolve?path=" + encodeURIComponent(path));
            if (r && r.ok && r.kind === "video") { src = { kind: "file", path: _join(r.folder, r.file) }; note = "read where it lies -- nothing copied"; rememberFolder(st, r.folder); }
        } else if (dt.files && dt.files.length) {
            const f = dt.files[0];
            const r = await _json("/uls/media/locate", { method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name: f.name, size: f.size, mtime: Math.round((f.lastModified || 0) / 1000), folders: recentFolders(st.recent) }) });
            if (r && r.ok) { src = { kind: "file", path: _join(r.folder, r.file) }; note = "found in " + r.folder + " -- nothing copied"; rememberFolder(st, r.folder); }
            else note = (r && r.reason === "ambiguous" ? "the name and size match in several recent folders" : "your browser hides the file's path and it is in no recent folder")
                + " -- copied into input/cine_media (open its folder once in a Media Loader: the next drop reads it where it lies)";
        }
    }
    if (!src && dt.files && dt.files.length) {
        const f = Array.from(dt.files).find((x) => VIDEO_RE.test(x.name));
        if (!f) { st.state.textContent = "drop: no video file in it"; return; }
        const d = await _json("/pls/cine/media_dir?project=" + encodeURIComponent(m.project || "cine"));
        if (d.error) { st.state.textContent = "drop: " + d.error; return; }
        const fd = new FormData(); fd.append("file", f, f.name);
        st.state.textContent = "copying " + f.name + " ...";
        const r = await _json("/uls/media/upload?folder=" + encodeURIComponent(d.dir), { method: "POST", body: fd });
        if (!r || !r.ok || !r.names || !r.names.length) { st.state.textContent = "drop: the copy failed -- " + ((r && r.error) || "?"); return; }
        src = { kind: "file", path: d.rel + "/" + r.names[0] };
        note = (note ? note + "; " : "") + "copied into input/" + d.rel + "/";
    }
    if (!src) { st.state.textContent = "drop: nothing to load (no video file, no path)"; return; }
    await _loadInto(node, id, src, at, note);
}
async function _pickInto(node, id) {
    const st = node._plsTl;
    st.state.textContent = "file...: the dialog opens on the computer running ComfyUI";
    let d = null;
    try { d = await _json("/uls/media/native_pick"); } catch (e) { d = null; }
    if (!d || !d.ok || !d.file) { st.state.textContent = d && d.cancelled ? "file...: cancelled" : "file...: no native dialog here -- drop the clip on the layer instead"; return; }
    await _loadInto(node, id, { kind: "file", path: _join(d.path, d.file) }, Math.round((st.player.t || 0) * FPS), "read where it lies -- nothing copied");
}
async function _loadInto(node, id, src, at, note) {
    const st = node._plsTl;
    const r = await _json("/pls/cine/probe", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ src }) });
    if (!r || r.error) { st.state.textContent = "layer: " + ((r && r.error) || "the clip could not be read"); return; }
    let e = editOf(node);
    if (id === NEW_TRACK) { id = newLayerId(e); e = addLayer(e, id); _seedTrackH(node, id); }   // v1082 (v1158: V1's height): the track exists only with its clip (one undo step)
    st.active = id;
    _saveEdit(node, setLayer(e, id, { src, name: r.name, pieces: placeClip(r.frames, Math.max(0, at)), aoff: 0 }));   // v1154: a new clip starts on its picture
    st.state.textContent = r.name + " -> layer at " + timeLabel(Math.max(0, at) / FPS) + " (" + (r.frames / FPS).toFixed(2) + " s"
        + (Math.abs(r.fps - FPS) > 0.01 ? ", " + Number(r.fps).toFixed(2) + " fps played at 24" : "") + (r.audio ? "" : ", no sound") + ")" + (note ? " -- " + note : "");
}

/* ------------------------------------------------------------ v1046 E4 */
/* "new render -> new track": on = V1 keeps the clip it shows now (pinned as
 * its take) and every finished run opens a track on top; off = V1 follows the
 * newest run again (the pins go) */
/* v1080 (Frank 10.10.: "das soll ein Sammelbecken werden"): ON is the default.
 * The edit stores an explicit false for off; a missing key means on. */
export function autoOn(e) { return !(e && e[AUTO_KEY] === false); }
/* v1198 (Frank 21.10.: "die anderen Teile dann auch auf dem EINEN neuen Track, damit sie alle zusammenbleiben"):
 * the three choices -- "track" (every run its own track), "render" (every scene of ONE render on ONE new track),
 * "replace" (V1 follows the newest run) */
export function autoMode(e) { return !autoOn(e) ? "replace" : (e && e.auto_mode === "render" ? "render" : "track"); }
/* v1082: "restore on open" off -> the edit without its tracks, cuts and takes;
 * the settings (an explicit off of the switch, the film's span) stay */
export function wipeEdit(e) {
    const o = {};
    if (e && e[AUTO_KEY] === false) o[AUTO_KEY] = false;
    if (e && e.auto_mode === "render" && e[AUTO_KEY] !== false) o.auto_mode = "render";
    if (e && e.span != null) o.span = e.span;
    return o;
}
function _setAuto(node, mode) {
    const st = node._plsTl, e = editOf(node), D = st.plan;
    const on = mode === true || mode === "track" || mode === "render";
    if (mode === "render") e.auto_mode = "render"; else delete e.auto_mode;
    if (on) {
        e[AUTO_KEY] = true;
        e.takes = e.takes || {};
        for (const c of (D && D.clips) || []) if (c && c.takes && c.takes.length) e.takes[String(c.k)] = c.takes[c.take || 0];
    } else { e[AUTO_KEY] = false; delete e.takes; }
    _saveEdit(node, e);
}
/* where scene k starts on the film (0 = the whole run) */
function _sceneAt(node, k) {
    if (!k) return 0;
    const m = _model(node), sc = scenesOf(m);
    return sc[k - 1] ? Math.round(sceneSpan(m, k)[0] * FPS) : 0;
}
/* a render file (rel to output/) as a new track on top, at frame `at` */
async function _renderTrack(node, rel, at, e, name) {
    const r = await _json("/pls/cine/probe", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ src: { kind: "render", path: rel } }) });
    if (!r || r.error) throw new Error((r && r.error) || "the clip could not be read");
    const id = newLayerId(e, Date.now() + Math.floor(Math.random() * 1000));
    const e2 = addLayer(e, id, name || r.name);
    _seedTrackH(node, id);                                          // v1158: the new track as tall as V1
    return setLayer(e2, id, { src: { kind: "render", path: rel, at: Math.max(0, Math.trunc(at)) }, name: name || r.name, pieces: placeClip(r.frames, Math.max(0, at)) });
}
/* v1198: a render file laid onto an EXISTING track at frame `at` (one track per render) */
async function _renderInto(node, rel, at, e, id) {
    const r = await _json("/pls/cine/probe", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ src: { kind: "render", path: rel } }) });
    if (!r || r.error) throw new Error((r && r.error) || "the clip could not be read");
    return dropOnto(e, id, { kind: "render", path: rel, at: Math.max(0, Math.trunc(at)) }, { k: 0, a: 0, b: Math.max(1, Math.trunc(r.frames)), speed: 1 }, at);
}
/* v1198 (Frank 21.10.: 47 tracks from "takes -> tracks", "da brauchen wir auch noch eine sinnvolle Sortierung"):
 * the render's older takes as tracks -- ONE TRACK PER RENDER (its scenes, each at its place), the newest render on
 * top; per run only the clip it picked (a run's backup / raw saves stay out); what V1 shows or a track has is skipped */
export function renderStamp(sec) {
    try {
        const d = new Date(Number(sec) * 1000), p = (n) => String(n).padStart(2, "0");
        return p(d.getDate()) + "." + p(d.getMonth() + 1) + ". " + p(d.getHours()) + ":" + p(d.getMinutes()) + ":" + p(d.getSeconds());
    } catch (e) { return ""; }
}
/* which runs of which render still need a track: renders newest first in, OLDEST first out (so the newest lands on
 * top); `skip` = the files V1 shows or a track carries */
export function takePlan(renders, skip) {
    const out = [];
    for (const g of renders || []) {
        const runs = (g.runs || []).filter((r) => r && r.rel && !skip.has(r.rel));
        if (runs.length) out.push(Object.assign({}, g, { runs }));
    }
    return out.reverse();
}
async function _takesToTracks(node) {
    const st = node._plsTl, D = st.plan;
    if (!D) return;
    let e = editOf(node);
    const m = _model(node);
    const skip = new Set([].concat(...stackLayers(e).map((x) => layerPaths(x))).filter(Boolean));
    for (const c of D.clips || []) if (c && c.takes && c.takes.length) skip.add(c.takes[c.take || 0]);   // what V1 shows
    let R;
    try { R = await _json("/pls/cine/renders", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project: projectName(m), count: Math.max(1, scenesOf(m).length) }) }); }
    catch (x) { st.state.textContent = "takes -> tracks: " + x.message; return; }
    if (!R || R.error) { st.state.textContent = "takes -> tracks: " + ((R && R.error) || "no answer"); return; }
    const plan = takePlan(R.renders, skip);
    if (!plan.length) { st.state.textContent = "takes -> tracks: no older render (or all are on V1 / a track already)"; return; }
    let n = 0, clips = 0;
    try {
        for (const g of plan) {
            let id = null;
            for (const r of g.runs) {
                const at = _sceneAt(node, r.k);
                if (!id) {
                    e = await _renderTrack(node, r.rel, at, e, "render " + renderStamp(g.stamp) + (g.legacy ? " (by time)" : ""));
                    id = stackLayers(e)[0].id;
                    n++;
                } else e = await _renderInto(node, r.rel, at, e, id);
                clips++;
            }
            if (g.sweep) { const rt = Object.assign({}, e.render_tracks || {}); rt[String(g.sweep)] = id; e.render_tracks = rt; }
        }
    } catch (x) { st.state.textContent = "takes -> tracks: " + x.message; }
    if (n) {
        _saveEdit(node, e);      // ONE save = one undo step
        st.state.textContent = n + " older render(s) -> " + n + " track(s), newest on top (" + clips + " clip(s); "
            + (R.extras ? R.extras + " backup / raw save(s) of the runs left out" : "one clip per run") + ")";
    }
}
/* v1198: a control of the timeline that should give the focus back after a change (the frontend's undo / redo is
 * off while an INPUT holds it): ticks, sliders, radios, choices -- never a text field */
export function blurAfterChange(t) {
    if (!t || !t.tagName) return false;
    if (t.tagName === "SELECT") return true;
    return t.tagName === "INPUT" && /^(checkbox|radio|range|button|color)$/i.test(String(t.type || ""));
}
/* v1198: every track off the stack in one step (V1 and its pins stay; the files stay on disk) */
export function clearTracks(e) {
    const o = JSON.parse(JSON.stringify(e || {}));
    if (Array.isArray(o.layers)) { o.layers = o.layers.filter((x) => x && x.id === BASE_ID); if (!o.layers.length) delete o.layers; }
    delete o.render_tracks;
    return o;
}
function _clearTracks(node) {
    const e = editOf(node), n = stackLayers(e).length - 1;
    if (!n) { node._plsTl.state.textContent = "no track to clear -- V1 is the only one"; return; }
    if (!window.confirm("Remove all " + n + " track(s) above V1? V1 stays as it is; the files stay on disk (Ctrl+Z brings the tracks back).")) return;
    node._plsTl.active = BASE_ID;
    _saveEdit(node, clearTracks(e));
    node._plsTl.state.textContent = n + " track(s) removed -- V1 stays";
}
/* a finished run's clip, when the switch is on: a new track on top */
/* v1047: one after the other -- two runs ending close together (render all
 * scenes) each read the edit, add their track and save: in parallel the
 * second save dropped the first track (measured) */
function _autoTrack(node, rel, scene, nFiles, sweep) {
    const q = (node._plsAutoQ || Promise.resolve()).then(() => _autoTrack1(node, rel, scene, nFiles, sweep)).catch(() => { /* view only */ });
    node._plsAutoQ = q;
    return q;
}
async function _autoTrack1(node, rel, scene, nFiles, sweep) {
    const e = editOf(node);
    if (!autoOn(e)) return;
    const key = String(scene || 0);
    // v1082: a run that changed nothing is reported by ComfyUI like a new one
    // (a cached node still sends "executed") -- the clip V1 is pinned to, or
    // one already a track, is never added a second time
    if (e.takes && e.takes[key] === rel) { node._plsTl.state.textContent = "this run's clip is V1 of this place already: " + rel; return; }
    if (stackLayers(e).some((x) => layerPaths(x).includes(rel))) { node._plsTl.state.textContent = "this run's clip is on a track already: " + rel; return; }
    // v1080: the basin starts with V1 -- the FIRST run of a place is pinned
    // there (V1 would show it anyway; a track on top would show it twice);
    // every run after that lands in a new track while V1 keeps its clip
    if (!(e.takes && e.takes[key])) {
        e.takes = e.takes || {};
        e.takes[key] = rel;
        _saveEdit(node, e);
        node._plsTl.state.textContent = "first run of this place -> V1 (pinned): " + rel + " -- the next runs become tracks on top";
        return;
    }
    // v1198: one track per render -- the render's track (by its sweep stamp) takes every scene of it
    if (autoMode(e) === "render" && sweep) {
        const key = String(sweep), id = (e.render_tracks || {})[key], lay = id ? layerOf(e, id) : null;
        try {
            if (lay && lay.id !== BASE_ID) {
                _saveEdit(node, await _renderInto(node, rel, _sceneAt(node, scene), e, id));
                node._plsTl.state.textContent = "new render -> its track (" + (lay.name || id) + "): " + rel + (scene ? " at scene " + scene : "");
            } else {
                let stamp = "";
                try { stamp = new Date(Number(sweep) * 1000).toTimeString().slice(0, 8); } catch (x) { /* no clock */ }
                const e2 = await _renderTrack(node, rel, _sceneAt(node, scene), e, "render " + stamp);
                const nid = stackLayers(e2)[0].id, rt = Object.assign({}, e2.render_tracks || {});
                rt[key] = nid;
                const ks = Object.keys(rt);
                while (ks.length > 32) delete rt[ks.shift()];
                e2.render_tracks = rt;
                _saveEdit(node, e2);
                node._plsTl.state.textContent = "new render -> ONE new track for this render (render " + stamp + "): " + rel + " -- its other scenes join it";
            }
            setTimeout(() => { try { _stack(node); } catch (x) { /* view only */ } }, 1500);
        } catch (x) { node._plsTl.state.textContent = "new render -> its track: " + x.message; }
        return;
    }
    try { _saveEdit(node, await _renderTrack(node, rel, _sceneAt(node, scene), e));
        node._plsTl.state.textContent = "new render -> new track: " + rel + (nFiles > 1 ? "  (" + nFiles + " clips saved by this run -- the last one, or the one wired to clip_prefix, is the track)" : "");
        // v1153 (Frank: "after the second clip the timeline should refresh by itself"): the file may still be
        // closing when the first plan probes it -- a second look a moment later shows its pictures and sound
        setTimeout(() => { try { _stack(node); } catch (x) { /* view only */ } }, 1500); }
    catch (x) { node._plsTl.state.textContent = "new render -> new track: " + x.message; }
}

/* ------------------------------------------------------------ v1036 the player */
/* v1044: a clip's URL -- /view for output/input/temp, the Media Loader's
 * range route for a file read where it lies */
function _clipURL(c) {
    if (!c) return "";
    if (c.view) return _viewURL(c.view);
    if (c.file_url) {
        const q = "folder=" + encodeURIComponent(c.file_url.folder) + "&file=" + encodeURIComponent(c.file_url.file);
        try { return api.apiURL("/uls/media/file?" + q); } catch (e) { return "/uls/media/file?" + q; }
    }
    return "";
}
function _mkVideo(node, url, warmAt) {
    const pl = node._plsTl.player;
    const v = document.createElement("video");
    v.src = url; v.preload = "auto"; v.playsInline = true; v.crossOrigin = "anonymous";
    v.addEventListener("seeked", () => { if (!pl.playing) _playerDraw(node); });
    v.addEventListener("loadeddata", () => { if (!pl.playing) _playerDraw(node); });
    // v1172 (Frank 19.10.: "beim ersten Lauf flackern die Videos"): every clip is opened and its first picture
    // decoded when the plan comes, not when the playhead reaches it -- the first run finds them ready
    if (typeof warmAt === "number") v.addEventListener("loadedmetadata", () => { try { if (v.paused && v.currentTime < 1e-3) v.currentTime = warmAt; } catch (e) { /* not seekable yet */ } }, { once: true });
    return v;
}
/* v1155 (audit): layout px per screen px -- the node is drawn scaled with the canvas; a transform in the element's
 * own px moved the dragged clip by (screen dx x scale), ahead of or behind the pointer at canvas zoom != 1 */
function _pxScale(el) {
    try { const r = el.getBoundingClientRect(); return r.width > 0 && el.offsetWidth > 0 ? el.offsetWidth / r.width : 1; } catch (e) { return 1; }
}
/* v1154 (audit): every <video> the player holds -- the clips, the layer clips, the exact preview -- paused and let go
 * (a removed node or a replaced preview kept playing its sound) */
function _releaseVideos(st, keepExact) {
    const pl = st && st.player;
    const drop = (v) => { try { v.pause(); v.removeAttribute("src"); v.load(); } catch (e) { /* gone */ } };
    if (pl) {
        pl.playing = false;
        try { cancelAnimationFrame(pl.raf); } catch (e) { /* none */ }
        for (const v of (pl.videos || new Map()).values()) drop(v);
        for (const m of (pl.lvid || new Map()).values()) for (const v of m.values()) drop(v);
    }
    if (!keepExact && st && st.exact) { drop(st.exact); st.exact = null; }
}
function _playerLoad(node) {
    const st = node._plsTl, pl = st.player;
    for (const v of pl.videos.values()) { try { v.pause(); v.removeAttribute("src"); v.load(); } catch (e) { /* gone */ } }
    pl.videos.clear();
    for (const m of (pl.lvid || new Map()).values()) for (const v of m.values()) { try { v.pause(); v.removeAttribute("src"); v.load(); } catch (e) { /* gone */ } }
    pl.lvid = new Map();
    const D = st.plan;
    if (!D || !D.plan) return;
    D.plan.segments.forEach((sg, i) => {
        if (sg.gap) return;
        const c = D.clips.find((x) => x && x.k === sg.k);
        if (!c || !c.view) return;
        pl.videos.set(i, _mkVideo(node, _viewURL(c.view), srcTime(sg, sg.film_in / FPS)));
    });
    // v1044: every other layer's clip, one video per piece (a piece may play
    // another part of the same file at the same moment as its neighbour's blend)
    for (const r of (D.film && D.film.layers) || []) {
        if (r.base || !r.plan || !(r.clip || r.clips)) continue;
        const m = new Map();
        r.plan.segments.forEach((sg, i) => { if (sg.gap) return; const url = _clipURL(clipK(r, sg.k)); if (url) m.set(i, _mkVideo(node, url, srcTime(sg, sg.film_in / FPS))); });   // v1198: per file
        pl.lvid.set(r.id, m);
    }
    const c0 = D.clips.find((c) => c && c.w) || ((D.film && D.film.layers) || []).map((r) => r.clip).find((c) => c && c.w);
    if (c0) { pl.ar = c0.w / c0.h; pl.boxW = null; }   // v1163: the clip's aspect -> the monitor's box
    _monitorFit(node);
    pl.t = Math.min(pl.t || 0, filmLen(D) / FPS);
    _playerLabel(node);
}
function _playerLabel(node) {
    const st = node._plsTl, pl = st.player, L = filmLen(st.plan) / FPS;
    pl.time.textContent = (pl.t || 0).toFixed(2) + " / " + L.toFixed(2) + " s" + (st.exact ? "  (exact)" : "");
    if (st.tc) st.tc.textContent = timecode(pl.t || 0);
    pl.play.textContent = pl.playing ? "\u275a\u275a pause" : "\u25b6 play";
    if (st.head) { st.head.style.display = L ? "block" : "none"; st.head.style.left = (100 * (pl.t || 0) / (st.ax || 1)) + "%"; }
    if (st.headGrip) { st.headGrip.style.display = L ? "block" : "none"; st.headGrip.style.left = (100 * (pl.t || 0) / (st.ax || 1)) + "%"; }
    if (pl.playing) _follow(node);
    if (!pl.playing) _cutButtons(node);
}
/* v1044: who shows the picture / plays the sound at film time t --
 * {row, i} of D.film.layers (the stack), or the render alone without one */
function _owners(D, t) {
    if (!D || !D.film) return null;
    const f = Math.floor(t * FPS + 1e-6), fl = D.film.flat || {};
    const vi = ownerAt(fl.vruns, f), ai = ownerAt(fl.aruns, f);
    return { v: vi == null ? null : D.film.layers[vi], a: ai == null ? null : D.film.layers[ai] };
}
function _drawMix(ctx, W, H, videos, P, t, fit) {
    // one chain's picture at t: one piece, or the two of a blend (ffmpeg's formulas)
    const mx = mixAt({ plan: P }, t);
    const vid = (k) => mx.parts[k] && !(P.segments[mx.parts[k].i] || {}).gap ? videos.get(mx.parts[k].i) : null;
    const box = (v) => {
        if (!fit || !v || !v.videoWidth) return [0, 0, W, H];
        const sc = Math.min(W / v.videoWidth, H / v.videoHeight), w = v.videoWidth * sc, h = v.videoHeight * sc;
        return [(W - w) / 2, (H - h) / 2, w, h];
    };
    const draw = (v, x, a) => { if (v && v.readyState >= 2) { const b = box(v); ctx.globalAlpha = a; ctx.drawImage(v, b[0] + x, b[1], b[2], b[3]); ctx.globalAlpha = 1; } };
    if (mx.parts.length === 1) {
        const sg = P.segments[mx.parts[0].i];
        const v = vid(0);
        if (v) draw(v, 0, 1);
        else if (sg && !sg.gap && sg.missing) { ctx.fillStyle = "#777"; ctx.font = "14px sans-serif"; ctx.fillText("scene " + sg.k + " not rendered", 12, 24); }
    } else if (mx.parts.length === 2) {
        const a = vid(0), b = vid(1);
        for (const op of blendOps(mx.kind, mx.p, W)) {
            if (op[0] === "a") draw(a, op[1], op[2]);
            else if (op[0] === "b") draw(b, op[1], op[2]);
            else if (op[0] === "add") {
                const src = op[1] === "a" ? a : op[1] === "b" ? b : null;
                ctx.globalCompositeOperation = "lighter"; ctx.globalAlpha = Math.max(0, Math.min(1, op[2]));
                if (src && src.readyState >= 2) { const bx = box(src); ctx.drawImage(src, bx[0], bx[1], bx[2], bx[3]); }
                else if (!src) { ctx.fillStyle = op[1]; ctx.fillRect(0, 0, W, H); }
                ctx.globalCompositeOperation = "source-over"; ctx.globalAlpha = 1;
            }
            else if (op[0] === "bclip" && b && b.readyState >= 2) { ctx.save(); ctx.beginPath(); ctx.rect(op[1], 0, op[2] - op[1], H); ctx.clip(); const bx = box(b); ctx.drawImage(b, bx[0], bx[1], bx[2], bx[3]); ctx.restore(); }
        }
    }
}
/* v1172: the videos the picture needs at film time t (the chain that shows, its one or two pieces) */
function _pictureVideos(node, t) {
    const st = node._plsTl, pl = st.player, D = st.plan;
    if (!D || !D.plan) return [];
    const own = _owners(D, t);
    const vids = !own || (own.v && own.v.base) ? pl.videos : own.v ? ((pl.lvid && pl.lvid.get(own.v.id)) || new Map()) : null;
    const P = !own || (own.v && own.v.base) ? D.plan : own.v ? own.v.plan : null;
    if (!vids || !P) return [];
    return mixAt({ plan: P }, t).parts.map((q) => ({ v: (P.segments[q.i] || {}).gap ? null : vids.get(q.i), sg: P.segments[q.i] })).filter((x) => x.v);
}
/* v1172: can the picture at t be drawn as it is (every video it needs has a decoded picture and is not seeking)? */
export function pictureReady(vs) { return vs.every((x) => x && x.readyState >= 2 && !x.seeking); }
/* v1214: a playing piece whose video already shows a frame past its material (the clock has not reached the cut
 * yet) -- the monitor keeps its last picture instead of showing what the film leaves out */
export function pastOut(sg, v) { return !!(sg && v && !v.paused && v.currentTime * FPS >= sg.src_out); }
function _playerDraw(node) {
    const st = node._plsTl, pl = st.player, ctx = pl.canvas.getContext("2d");
    const W = pl.canvas.width, H = pl.canvas.height;
    // v1172 (Frank 19.10.: the first run flickers): a picture whose video is still seeking or loading is not drawn
    // as black -- the monitor keeps the last picture it showed until the new one is there
    if (!st.exact && pl.shown && st.plan && st.plan.plan) {
        const tq = Math.min(pl.t || 0, filmLen(st.plan) / FPS - 0.5 / FPS);
        const pv = _pictureVideos(node, tq);
        if (!pictureReady(pv.map((x) => x.v)) || (pv.length === 1 && pastOut(pv[0].sg, pv[0].v))) { _playerLabel(node); return; }
    }
    pl.shown = true;
    ctx.globalAlpha = 1; ctx.fillStyle = "#000"; ctx.fillRect(0, 0, W, H);
    if (st.exact) { try { const v = st.exact, sc = v.videoWidth ? Math.min(W / v.videoWidth, H / v.videoHeight) : 0;
        if (sc) ctx.drawImage(v, (W - v.videoWidth * sc) / 2, (H - v.videoHeight * sc) / 2, v.videoWidth * sc, v.videoHeight * sc); else ctx.drawImage(v, 0, 0, W, H); } catch (e) { /* not ready */ } _playerLabel(node); return; }
    const D = st.plan;
    if (!D || !D.plan) return;
    // at the very end the playhead stands AFTER the last frame -- show that frame
    const t = Math.min(pl.t || 0, filmLen(D) / FPS - 0.5 / FPS);
    const own = _owners(D, t);
    if (!own) _drawMix(ctx, W, H, pl.videos, D.plan, t, true);              // v1163: contain, never stretched
    else if (own.v && own.v.base) _drawMix(ctx, W, H, pl.videos, D.plan, t, true);
    else if (own.v) _drawMix(ctx, W, H, (pl.lvid && pl.lvid.get(own.v.id)) || new Map(), own.v.plan, t, true);
    _playerLabel(node);
}
/* put every clip where the film time says: the active ones seek (paused) or
 * play in step (playing), the rest pause; their sound follows the mix.
 * v1044: per layer -- only the layer that shows plays its picture, only the
 * layer that sounds is loud (the stack's owners) */
function _syncChain(videos, P, tt, playing, picOn, sndOn) {
    const mx = picOn ? mixAt({ plan: P }, tt) : { parts: [] };
    const amx = sndOn ? audioMixAt({ plan: P }, tt) : { parts: [] };
    const pic = new Map(mx.parts.map((q) => [q.i, q.w])), snd = new Map(amx.parts.map((q) => [q.i, q.w]));
    const AS = P.asegments;
    const f = tt * FPS;
    for (const [i, v] of videos) {
        if (!pic.has(i) && !snd.has(i)) {
            if (!v.paused) v.pause();
            // v1172 (Frank 19.10.: the videos "hackeln" at the transitions): the piece that comes next is put on
            // its first picture while the one before still plays -- at its cut it only has to start, not to seek
            const sg = P.segments[i];
            if (playing && sg && !sg.gap && sg.film_in > f && sg.film_in - f <= PREROLL * FPS) {
                const w0 = srcTime(sg, sg.film_in / FPS);
                if (!v.seeking && Math.abs(v.currentTime - w0) > 0.5 / FPS) v.currentTime = w0;
            }
            continue;
        }
        const sgi = P.segments[i];
        const want = pic.has(i) || !AS ? srcTime(sgi, tt) : asrcTime(AS[i], tt);
        v.volume = Math.max(0, Math.min(1, snd.get(i) || 0));
        const rate = Number(sgi.speed) || 1;
        if (v.playbackRate !== rate) v.playbackRate = rate;
        if (playing) {
            // v1172: a playing clip is the clock (see _masterTime) -- it is only put right when it is far off
            if (Math.abs(v.currentTime - want) > DRIFT_MAX * Math.max(1, rate) && !v.seeking) v.currentTime = want;
            if (v.paused) v.play().catch(() => {});
        } else {
            if (!v.paused) v.pause();
            if (Math.abs(v.currentTime - want) > 0.5 / FPS) v.currentTime = want;
        }
    }
}
export const PREROLL = 0.75, DRIFT_MAX = 0.25;      // v1172: seconds
/* v1172: while ONE clip shows and plays, the film's time is that clip's time (no wall clock racing ahead of a clip
 * that is still loading, no seek to pull it back); null when there is no such clip (a blend, a gap, loading) */
export function masterTime(sg, v) {
    if (!sg || !v || v.paused || v.seeking || v.readyState < 3) return null;
    const sp = Number(sg.speed) || 1;
    // v1214 (Frank 22.10.: a "Hacker" at a motion-carried seam, live only): a playing video shows frame
    // floor(currentTime * FPS) -- the clock counts that frame, not half a frame behind it. The half frame kept the film
    // in the old piece while its video already showed the frame after the cut (the grid overhang), and the next scene
    // then continued from the frame before: the same moment twice.
    const t = (sg.film_in + (sg.hold_head || 0) + (v.currentTime * FPS - sg.src_in) / sp) / FPS;
    if (!(t >= sg.film_in / FPS - 1 / FPS && t <= sg.film_out / FPS + 1 / FPS)) return null;
    return t;
}
function _playerSync(node, playing) {
    const st = node._plsTl, pl = st.player, D = st.plan;
    if (st.exact) {
        if (playing) { if (st.exact.paused) st.exact.play().catch(() => {}); }
        else { st.exact.pause(); st.exact.currentTime = pl.t || 0; }
        return;
    }
    if (!D || !D.plan) return;
    const tt = Math.min(pl.t || 0, filmLen(D) / FPS - 0.5 / FPS);
    // v1038: the picture's mix and the sound's mix (a J-/L-cut plays a
    // clip's sound while its picture is not on screen -- same source line)
    const own = _owners(D, tt);
    if (!own) { _syncChain(pl.videos, D.plan, tt, playing, true, true); return; }
    _syncChain(pl.videos, D.plan, tt, playing, !!(own.v && own.v.base), !!(own.a && own.a.base));
    for (const r of D.film.layers) {
        if (r.base || !r.plan) continue;
        const vids = (pl.lvid && pl.lvid.get(r.id)) || new Map();
        _syncChain(vids, r.plan, tt, playing, own.v === r, own.a === r);
    }
}
function _playerSeek(node, t) {
    const st = node._plsTl, pl = st.player, L = filmLen(st.plan) / FPS;
    pl.t = Math.max(0, Math.min(L, t));
    if (pl.playing) { pl.t0 = performance.now() / 1000 - pl.t; }
    _playerSync(node, pl.playing);
    _playerDraw(node);
}
/* v1038: the playhead is dragged (its grip on the ruler, or the ruler
 * itself): the player follows frame by frame; it snaps to the cuts (the
 * picture's and the sound's, the film's ends) within 6 px -- Shift = free.
 * Dragging pauses, as in every editor. */
function _headTargets(st) {
    const P = st.plan && st.plan.plan;
    if (!P) return [0];
    const t = [0, filmLen(st.plan) / FPS];
    for (const sm of P.seams) { t.push(sm.at / FPS); if (typeof sm.a_at === "number" && sm.a_at !== sm.at) t.push(sm.a_at / FPS); }
    // v1044: the cuts of every layer too
    for (const r of (st.plan.film && st.plan.film.layers) || []) if (!r.base && r.plan) for (const c of r.plan.cuts || []) t.push(c / FPS);
    return t;
}
function _scrub(node, ev) {
    const st = node._plsTl, pl = st.player, el = ev.currentTarget;
    if (ev.button !== 0) return;
    ev.preventDefault(); ev.stopPropagation();
    // a press without a real pointer (a script, a pen that left) must not throw
    try { el.setPointerCapture(ev.pointerId); } catch (e) { /* no capture: the moves still come while over the ruler */ }
    if (pl.playing) _playerToggle(node);
    st.scrubbing = true;
    const go = (e) => {
        const ax = st.ax || 1, W = st.ruler.getBoundingClientRect().width || 1;
        let t = _secFromX(st.ruler, e.clientX, ax);
        if (!e.shiftKey) t = snapHead(t, _headTargets(st), 6 / W * ax);
        _playerSeek(node, t);
    };
    go(ev);
    const up = () => { st.scrubbing = false; el.removeEventListener("pointermove", mv); el.removeEventListener("pointerup", up); el.removeEventListener("pointercancel", up); _cutButtons(node); };
    // no button held any more (the release happened elsewhere): the drag is over
    const mv = (e) => { e.stopPropagation(); if (e.buttons === 0) { up(); return; } go(e); };
    el.addEventListener("pointermove", mv); el.addEventListener("pointerup", up); el.addEventListener("pointercancel", up);
}
/* v1163 (Frank 18.10.: "bei aktiv selektierter Timeline-Node ... Space-Taste ... Play"): Space plays / pauses
 * the film of the ONE selected timeline node -- not while typing (a field, a select, a button keeps its own Space),
 * not with a modifier. Pure: which node Space belongs to, or null. */
export function spaceNode(ev, selected) {
    if (!ev || !(ev.code === "Space" || ev.key === " ") || ev.ctrlKey || ev.altKey || ev.metaKey || ev.shiftKey) return null;
    const t = ev.target, tag = t && t.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || tag === "BUTTON" || (t && t.isContentEditable)) return null;
    const tls = (selected || []).filter((n) => n && n.type === NODE && n._plsTl && (n.mode === 0 || n.mode === undefined));
    return tls.length === 1 ? tls[0] : null;
}
function _selectedNodes() {
    const s = app.canvas && app.canvas.selected_nodes;
    if (!s) return [];
    return s instanceof Map ? [...s.values()] : Array.isArray(s) ? s : Object.values(s);
}
function _onSpace(ev) {
    const n = spaceNode(ev, _selectedNodes());
    if (!n) return;
    ev.preventDefault(); ev.stopPropagation();          // the page does not scroll, the canvas does not pan
    if (!ev.repeat) { try { _playerToggle(n); } catch (x) { /* view only */ } }
}
/* the clip's picture inside the monitor's area: as large as fits, its own aspect (pure) */
export function monitorBox(areaW, areaH, ar) {
    const W = Math.max(1, Math.floor(Number(areaW) || 0)), H = Math.max(1, Math.floor(Number(areaH) || 0));
    const a = Number(ar) > 0 ? Number(ar) : 16 / 9;
    return W / H > a ? [Math.max(1, Math.round(H * a)), H] : [W, Math.max(1, Math.round(W / a))];
}
/* the monitor follows the clip's aspect and its column's width -- called when a clip comes (the plan) and on
 * every width check (a resize, a pan, coming on screen); cheap when nothing changed */
function _monitorFit(node) {
    const st = node && node._plsTl, pl = st && st.player;
    if (!pl || !pl.canvas) return;
    const col = st.monCol, aw = (col && col.clientWidth) || MON_W;
    if (!(aw > 0)) return;
    const [w, h] = monitorBox(aw, MON_AREA_H, pl.ar);
    if (pl.boxW === w && pl.boxH === h) return;
    pl.boxW = w; pl.boxH = h;
    const dpr = Math.min(2, (typeof window !== "undefined" && window.devicePixelRatio) || 1);
    pl.canvas.style.width = w + "px"; pl.canvas.style.height = h + "px";
    pl.canvas.width = Math.max(1, Math.round(w * dpr)); pl.canvas.height = Math.max(1, Math.round(h * dpr));
    try { _playerDraw(node); } catch (e) { /* not ready */ }
}
function _playerToggle(node) {
    const st = node._plsTl, pl = st.player, L = filmLen(st.plan) / FPS;
    if (!L) return;
    if (pl.playing) { pl.playing = false; cancelAnimationFrame(pl.raf); _playerSync(node, false); _playerDraw(node); return; }
    if ((pl.t || 0) >= L - 1 / FPS) pl.t = 0;
    pl.playing = true;
    pl.t0 = performance.now() / 1000 - (pl.t || 0);
    const tick = () => {
        if (!pl.playing) return;
        const now = performance.now() / 1000;
        if (st.exact) pl.t = st.exact.currentTime;
        else {
            // v1172: the clip on screen is the clock; while the picture is not ready the film waits for it
            const pv = _pictureVideos(node, Math.min(pl.t || 0, L - 0.5 / FPS));
            const mt = pv.length === 1 ? masterTime(pv[0].sg, pv[0].v) : null;
            if (mt !== null) { pl.t = Math.max(pl.t || 0, Math.min(mt, now - pl.t0 + 0.25)); pl.t0 = now - pl.t; }
            else if (!pictureReady(pv.map((x) => x.v))) pl.t0 = now - (pl.t || 0);
            else pl.t = now - pl.t0;
        }
        if (pl.t >= L) { pl.t = L; pl.playing = false; _playerSync(node, false); _playerDraw(node); return; }
        _playerSync(node, true);
        _playerDraw(node);
        pl.raf = requestAnimationFrame(tick);
    };
    _playerSync(node, true);
    pl.raf = requestAnimationFrame(tick);
}
/* exact preview (temp/, small) or the export (the day folder, PH_<hhmmss>_<project>_film; v1152) --
 * both ffmpeg on the same plan */
async function _export(node, preview) {
    const st = node._plsTl;
    const w = _w(node, "timeline");
    st.state.textContent = preview ? "rendering the exact preview ..." : "exporting the film ...";
    try {
        const r = await api.fetchApi("/pls/cine/export", { method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ timeline: w ? w.value : "", edit: editOf(node), preview: !!preview }) });
        const d = await r.json();
        if (d.error) { st.state.textContent = (preview ? "exact preview: " : "export: ") + d.error; return; }
        st.state.textContent = (d.lines || [])[0] || "done";
        if (d.view) {
            const v = document.createElement("video");
            v.src = _viewURL(d.view); v.preload = "auto"; v.playsInline = true; v.crossOrigin = "anonymous";
            v.addEventListener("seeked", () => _playerDraw(node));
            v.addEventListener("loadeddata", () => _playerDraw(node));
            if (st.player.playing) _playerToggle(node);
            if (st.exact) { try { st.exact.pause(); st.exact.removeAttribute("src"); st.exact.load(); } catch (x) { /* gone */ } }
            st.exact = v;
            _playerLabel(node);
        }
    } catch (e) { st.state.textContent = (preview ? "exact preview failed: " : "export failed: ") + e.message; }
}

async function _queue() {
    // the frontend's own queue call (the button's), so a run is a run
    if (typeof app.queuePrompt === "function") return app.queuePrompt(0, 1);
    throw new Error("app.queuePrompt missing");
}

async function _renderAll(node) {
    const st = node._plsTl;
    const m0 = _model(node);
    const n = scenesOf(m0).length;
    const keep = { render: m0.render, active: m0.active };
    const ticks = pickedOrder(n, 0, skipOf(node));               // v1163: the ticked scenes, in order
    if (!ticks.length) { st.state.textContent = "no scene is ticked -- tick the scenes to render on the Scenes row"; return; }
    let queued = 0;
    for (const k of ticks) {
        const mm = _model(node); mm.render = "scene"; mm.active = k; _save(node, mm, false);
        st.state.textContent = "queueing scene " + k + " of " + n + " ...";
        // v1082: the frontend's queue call returns false when the prompt was
        // refused (its own dialog shows why) -- the loop stops and says so
        let ok = true;
        try { ok = await _queue(); } catch (e) { ok = false; st.state.textContent = "queue failed at scene " + k + ": " + e.message; }
        if (ok === false) { break; }
        queued++;
    }
    const mm = _model(node); mm.render = keep.render; mm.active = keep.active; _save(node, mm, false);
    _render(node);
    if (queued < ticks.length) { const left = ticks.slice(queued); st.state.textContent = (st.state.textContent.startsWith("queue failed") ? st.state.textContent + " -- " : "scene " + left[0] + " was refused (see the dialog) -- ") + "scenes " + (left.every((x, i) => i === 0 || x === left[i - 1] + 1) ? left[0] + ".." + left[left.length - 1] : left.join(", ")) + " not queued (" + queued + " queued)"; }
    else st.state.textContent = queued + " scene run(s) queued (" + (ticks.length < n ? "scenes " + ticks.join(", ") : "all") + ") -> the stack of " + (m0.project || "cine");
}

/* v1030: a number field for seconds -- lands on the frame when applied */
function _secField(value, title, onApply) {
    const t = _el("input", INPUT + "width:66px;"); t.type = "number"; t.step = String(1 / FPS); t.min = "0";
    t.value = Number(value).toFixed(3); t.title = title;
    t.addEventListener("change", () => onApply(parseFloat(t.value)));
    t.addEventListener("keydown", (e) => { e.stopPropagation(); if (e.key === "Enter") t.blur(); });
    return t;
}
/* the shot's times: at (the cut) + length; for a cut also its transition */
function _timesRow(node, m, i) {
    const st = node._plsTl;
    const row = _el("div", "display:flex;gap:5px;align-items:center;font-size:11px;color:#ccc;flex-wrap:nowrap;");
    const put = (fn) => (v) => { if (!Number.isFinite(v)) { _render(node); return; } const mm = _model(node); fn(mm, v); _save(node, mm); _render(node); _stack(node); };
    const at = _secField(m.shots[i].at, i ? "the cut that starts this shot (lands on the frame, one frame from its neighbours)" : "shot 1 starts at 0 -- the official form has no time", put((mm, v) => moveCut(mm, i, v)));
    at.disabled = i === 0; at.dataset.pls = "at";
    const len = _secField(shotLength(m, i), i + 1 < m.shots.length ? "moves the next cut" : "the last shot's length is the clip length (the 17k+5 grid rounds it up)", put((mm, v) => setShotLength(mm, i, v)));
    len.dataset.pls = "length";
    const f0 = Math.round(m.shots[i].at * FPS), f1 = Math.round((m.shots[i].at + shotLength(m, i)) * FPS);
    row.append(_el("span", "", "at"), at, _el("span", "", "s   length"), len, _el("span", "", "s"),
        _el("span", "color:#777;font-size:10px;", "frames " + f0 + "-" + (f1 - 1)));
    if (i > 0) {
        const c = cutChoice(m, i);
        const k = _el("select", INPUT); k.dataset.pls = "transition";
        const grp = (label, xs) => { const g = document.createElement("optgroup"); g.label = label;
            for (const x of xs) { const o = document.createElement("option"); o.value = x; o.textContent = x; g.appendChild(o); } k.appendChild(g); };
        grp("inside the clip -- H3 paints it (one run)", [H3_CUT].concat(H3T_NAMES));
        // v1036: the ffmpeg transitions live on the clip lane's seams -- here
        // only what goes INTO the run. A scene cut shows its seam's name.
        if (!isH3(c.kind)) { const o = document.createElement("option"); o.value = c.kind; o.textContent = "scene cut: " + c.kind + " (on the clip lane)"; o.disabled = true; k.appendChild(o); }
        k.value = c.kind;
        k.title = "inside the clip: H3 cuts or blends within ONE clip (figure, light and sound carry over); "
            + "an H3 transition writes the formula and a sentence on each side of the cut into the prompt.\n"
            + "A scene cut (the tick box) is two clips: its transition is set on the seam of the clip lane below.";
        const h3 = H3T_NAMES.includes(c.kind);
        const d = _el("input", INPUT + "width:52px;"); d.type = "number"; d.step = h3 ? "0.25" : "0.05";
        d.min = h3 ? String(H3T_MIN) : "0.05"; d.max = h3 ? String(H3T_MAX) : "5";
        d.title = h3 ? "asked on each side of the cut -- H3 paints it and decides the exact length" : "the ffmpeg transition's length";
        d.value = Number(c.seconds).toFixed(2); d.disabled = c.kind === H3_CUT || c.kind === "cut";
        const apply = () => { const mm = _model(node); setCutChoice(mm, i, k.value, parseFloat(d.value)); _save(node, mm); _render(node); _stack(node); };
        k.addEventListener("change", apply); d.addEventListener("change", apply);
        row.append(_el("span", "margin-left:8px;color:" + (c.kind === H3_CUT ? "#f0c87a" : h3 ? H3T_COL : "#80cbc4") + ";", "at this cut"), k, d, _el("span", "", "s"));
        st.transSel = k;
    }
    return row;
}
/* v1035: the sentences of an H3 custom transition at cut i -- ONE line
 * (the edit area has a fixed height; the shot text gives up the room) */
export const CUSTOM_ROW_H = 22, TEXT_H = 246;   // v1150: the shot's text is the big field of the work area (was 64); v1153: 246, the sound beside it
function _customRow(node, m, i) {
    const t = m.shots[i].h3t;
    const row = _el("div", "display:flex;gap:5px;align-items:center;font-size:11px;height:" + CUSTOM_ROW_H + "px;color:" + H3T_COL + ";");
    const put = (part) => (e) => { const mm = _model(node); setH3Custom(mm, i, part, e.target.value); _save(node, mm, e.type !== "input"); };
    const f = _el("select", INPUT + "max-width:150px;");
    for (const x of H3T_FORMULAS) { const o = document.createElement("option"); o.value = x; o.textContent = x; f.appendChild(o); }
    f.value = t.formula || H3T_FORMULAS[0]; f.dataset.pls = "h3formula"; f.title = "the formula at the cut";
    f.addEventListener("change", (e) => { put("formula")(e); _render(node); });
    const box = (part, tip) => {
        const x = _el("input", INPUT + "flex:1;min-width:0;"); x.type = "text"; x.value = t[part] || ""; x.title = tip; x.dataset.pls = "h3" + part;
        x.placeholder = part === "out" ? "end of shot " + i : "start of shot " + (i + 1);
        x.addEventListener("input", put(part)); x.addEventListener("change", () => { _commit(node); _render(node); });
        x.addEventListener("keydown", (e) => e.stopPropagation());
        return x;
    };
    row.append(_el("span", "white-space:nowrap;", "H3 custom"), f,
        _el("span", "white-space:nowrap;", "end"), box("out", "what happens to the image as shot " + i + " ends -- \"During the last ... of the shot,\" is put in front"),
        _el("span", "white-space:nowrap;", "start"), box("in", "how the new image arrives in shot " + (i + 1) + " -- \"within ...\" is put behind"));
    return row;
}
/* scene k: start + end (the cuts at its edges; the last end = clip length) */
function _sceneRow(node, m, k) {
    const [t0, t1] = sceneSpan(m, k);
    const n = scenesOf(m).length;
    const row = _el("div", "display:flex;gap:5px;align-items:center;font-size:11px;color:#80cbc4;");
    const put = (edge) => (v) => { if (!Number.isFinite(v)) { _render(node); return; } const mm = _model(node); setSceneEdge(mm, k, edge, v); _save(node, mm); _render(node); _stack(node); };
    const a = _secField(t0, k > 1 ? "moves the cut that starts this scene" : "scene 1 starts at 0", put("start"));
    a.disabled = k === 1; a.dataset.pls = "start";
    const b = _secField(t1, k < n ? "moves the cut that ends this scene" : "the last scene ends with the clip (the grid rounds the length up)", put("end"));
    b.dataset.pls = "end";
    row.append(_el("span", "", "Scene " + k + "   start"), a, _el("span", "", "s   end"), b, _el("span", "", "s"),
        _el("span", "color:#777;font-size:10px;", "length " + (t1 - t0).toFixed(2) + " s"));
    if (k > 1) row.append(..._carryCtl(node, m, k));
    return row;
}
/* v1049: "begins on the last frame of scene k-1" + how the frame is used */
function _carryCtl(node, m, k) {
    const lab = _el("label", "display:flex;gap:3px;align-items:center;cursor:pointer;margin-left:8px;color:#ffd27a;white-space:nowrap;");
    const cb = document.createElement("input"); cb.type = "checkbox"; cb.checked = carryOf(m, k); cb.dataset.pls = "carry";
    cb.addEventListener("change", () => { const mm = _model(node); setCarry(mm, k, cb.checked); _save(node, mm); _render(node); });
    lab.append(cb, _el("span", "", "\u21e4 begins on the last frame of scene " + (k - 1)));
    lab.title = "scene " + k + " BEGINS where scene " + (k - 1) + " ended: the timeline reads the LAST frame of scene " + (k - 1) + "'s newest clip\n"
        + "(the project's stack) and hands it out as first_frame -- wire it to MiniMax Reference's or MiniMax Keyframes' first_frame.\n"
        + "For continuous action; after a camera cut it is wrong (the camera changed).\n"
        + "v1176 motion context: the last 22 film frames of scene " + (k - 1) + " and their sound go out as tail / tail_audio\n"
        + "-> MiniMax Continue (K2) -- the clip continues the motion and the sound; the film leaves the repeated frames out.";
    const sel = _el("select", INPUT);
    sel.dataset.pls = "carrymode";
    sel.title = "how the carried frame is used -- for every carrying scene of this timeline";
    for (const [v, t] of CARRY_MODES) { const o = document.createElement("option"); o.value = v; o.textContent = t; sel.appendChild(o); }
    sel.value = CARRY_MODES.some(([v]) => v === m.carry_mode) ? m.carry_mode : "pin";
    sel.disabled = !cb.checked;
    sel.addEventListener("change", () => { const mm = _model(node); mm.carry_mode = sel.value; _save(node, mm); _render(node); });
    // v1185: the head's length (carry_mode motion) -- 17k+5 windows, the timeline's choice, 22 as before
    const head = _el("select", INPUT);
    head.dataset.pls = "headframes";
    head.title = "motion context: how many film frames of the scene before are pinned as this scene's head (17k+5 windows: 5 = 0.21 s, 22 = 0.92 s, 39 = 1.63 s).\n"
        + "The scene renders longer by the head AND the grid's round-up (141 + 5 -> 158, + 22 -> 175, + 39 -> 192). Which length carries the motion is not measured yet.\n"
        + "MiniMax Continue's context_frames must be >= this (it caps the window).";
    for (const [v, t] of HEAD_FRAMES) { const o = document.createElement("option"); o.value = String(v); o.textContent = t; head.appendChild(o); }
    head.value = HEAD_FRAMES.some(([v]) => v === Number(m.head_frames)) ? String(Number(m.head_frames)) : "22";
    head.disabled = !cb.checked || sel.value !== "motion";
    head.addEventListener("change", () => { const mm = _model(node); mm.head_frames = Number(head.value); _save(node, mm); _render(node); });
    return [lab, sel, head];
}
/* a double-click asked for a field: focus it (and open the transition list) */
function _openRequest(st) {
    const o = st.open;
    st.open = null;
    if (!o) return;
    const f = st.edit.querySelector('[data-pls="' + o.focus + '"]:not([disabled])');
    if (f) { try { f.focus(); f.select?.(); } catch (e) { /* view only */ } }
    // the transition choice is shown lit, NOT popped open: a native popup
    // takes the keyboard, and the time typed next must reach `at` (measured
    // in the browser probe: with showPicker the typed 7 went nowhere)
    if (o.transition && st.transSel) st.transSel.style.outline = "2px solid #f0c87a";
}

/* ------------------------------------------------------------ v1047 film length + zoom */
/* the film's length = the edit span (empty = the render's); the view goes to "fit" */
function _setFilmLen(node, raw) {
    const st = node._plsTl, e = editOf(node), t = String(raw == null ? "" : raw).trim();
    if (!t) delete e.span;
    else {
        const v = parseLen(t);
        if (!(v > 0)) { st.state.textContent = "film length: \"" + t + "\" is not a length (5:00, 1:30.5, 300, 2m)"; _render(node); return; }
        e.span = Math.round(Math.min(SPAN_MAX, Math.max(0.25, v)) * FPS) / FPS;
    }
    node.properties = node.properties || {};
    node.properties[ZOOM_KEY] = 1;
    if (st.scroller) st.scroller.scrollLeft = 0;
    _saveEdit(node, e);
}
/* the zoomed lanes: wider than the view by the zoom, the heads stay put */
function _applyZoom(node) {
    const st = node._plsTl;
    if (!st || !st.lanes || !st.scroller) return;
    const z = zoomClamp(zoomOf(node), st.ax);
    const w = z === 1 ? "100%" : "calc(" + GUT_W + "px + " + z + " * (100% - " + GUT_W + "px))";
    if (st.lanes.style.width !== w) st.lanes.style.width = w;
    _navDraw(node);
}
/* the time at the middle of the view, and the view's window in seconds */
function _view(node) {
    const st = node._plsTl, sc = st.scroller, ax = st.ax || 1;
    const vis = Math.max(1, sc.clientWidth - GUT_W), full = Math.max(vis, sc.scrollWidth - GUT_W);
    const t0 = sc.scrollLeft / full * ax, t1 = Math.min(ax, (sc.scrollLeft + vis) / full * ax);
    return { t0, t1, vis, full, ax };
}
function _scrollTo(node, t0) {
    const st = node._plsTl, v = _view(node);
    st.scroller.scrollLeft = Math.max(0, Math.min(v.full - v.vis, t0 / v.ax * v.full));
    _navDraw(node);
}
/* zoom to z, time `at` stays where it is on screen (default: the playhead,
 * else the middle of the view) */
function _setZoom(node, z, at, screenFrac) {
    const st = node._plsTl;
    if (!st || !st.scroller) return;
    const v0 = _view(node);
    const t = at != null ? at : (st.player && st.player.t != null && st.player.t >= v0.t0 && st.player.t <= v0.t1 ? st.player.t : (v0.t0 + v0.t1) / 2);
    const frac = screenFrac != null ? screenFrac : (v0.t1 > v0.t0 ? (t - v0.t0) / (v0.t1 - v0.t0) : 0.5);
    node.properties = node.properties || {};
    node.properties[ZOOM_KEY] = zoomClamp(z, st.ax);
    node.setDirtyCanvas?.(true, true);
    // v1155 (audit): the positions are percent of the lanes -- zoom-invariant; only the film strips' tile count
    // follows the width. The lanes widen at once, the full render (new tiles) follows once the zoom rests
    // (every slider / wheel tick rebuilt every track: ~1300 tiles per segment at the 2-s window of a 5-min film)
    _applyZoom(node);
    if (st.zoomTimer) clearTimeout(st.zoomTimer);
    st.zoomTimer = setTimeout(() => { st.zoomTimer = null; try { _render(node); } catch (e) { /* view only */ } }, 180);
    const v = _view(node), win = v.vis / v.full * v.ax;
    _scrollTo(node, t - frac * win);
}
function _zoomBy(node, f, at, frac) { _setZoom(node, zoomOf(node) * f, at, frac); }
/* Ctrl + wheel: zoom around the mouse; Shift + wheel or a sideways wheel: scroll */
function _wheelLanes(node, ev) {
    const st = node._plsTl, sc = st.scroller;
    const r = sc.getBoundingClientRect(), x = ev.clientX - r.left - GUT_W;
    if (ev.ctrlKey || ev.metaKey) {
        ev.preventDefault();
        const v = _view(node), frac = Math.max(0, Math.min(1, x / v.vis));
        _zoomBy(node, ev.deltaY < 0 ? 1.25 : 0.8, v.t0 + frac * (v.t1 - v.t0), frac);
        return true;
    }
    const dx = Math.abs(ev.deltaX) > Math.abs(ev.deltaY) ? ev.deltaX : (ev.shiftKey ? ev.deltaY : 0);
    if (dx && zoomOf(node) > 1) { ev.preventDefault(); sc.scrollLeft += dx; _navDraw(node); return true; }
    return false;
}
/* Nodes 2.0 takes the wheel in a capture listener above the node (measured:
 * the field never saw it) -- so the timeline listens on the document, first */
let _wheelHooked = false;
function _wheelHook() {
    if (_wheelHooked || typeof document === "undefined") return;
    _wheelHooked = true;
    document.addEventListener("wheel", (ev) => {
        try {
            const sc = ev.target && ev.target.closest ? ev.target.closest('[data-pls="scroller"]') : null;
            if (sc && sc._plsNode && sc._plsNode._plsTl && _wheelLanes(sc._plsNode, ev)) ev.stopPropagation();
        } catch (e) { /* a view must never break the canvas */ }
    }, { capture: true, passive: false });
}
function _navDraw(node) {
    const st = node._plsTl;
    if (!st || !st.thumb || !st.scroller) return;
    const v = _view(node);
    const a = v.ax ? v.t0 / v.ax : 0, b = v.ax ? v.t1 / v.ax : 1;
    st.thumb.style.left = (100 * a).toFixed(3) + "%";
    st.thumb.style.width = Math.max(2, 100 * (b - a)).toFixed(3) + "%";
    if (st.viewInfo) st.viewInfo.textContent = zoomOf(node) > 1 ? fmtLen(v.t0) + "-" + fmtLen(v.t1) : "all " + fmtLen(v.ax);
    if (st.zSl && document.activeElement !== st.zSl) { const zmax = Math.max(1, (st.ax || 0) / 2); st.zSl.value = String(zmax > 1 ? Math.round(100 * Math.log(zoomClamp(zoomOf(node), st.ax)) / Math.log(zmax)) : 0); }
}
/* the scroll bar: drag the thumb, or click beside it (the view centres there) */
function _navDown(node, ev) {
    ev.preventDefault(); ev.stopPropagation();
    const st = node._plsTl, r = st.nav.getBoundingClientRect(), v = _view(node), win = v.t1 - v.t0;
    const tAt = (cx) => Math.max(0, Math.min(1, (cx - r.left) / r.width)) * v.ax;
    let grab = tAt(ev.clientX) - v.t0;
    if (grab < 0 || grab > win) { _scrollTo(node, tAt(ev.clientX) - win / 2); grab = win / 2; }
    const mv = (e) => _scrollTo(node, tAt(e.clientX) - grab);
    const up = () => { window.removeEventListener("pointermove", mv); window.removeEventListener("pointerup", up); };
    window.addEventListener("pointermove", mv); window.addEventListener("pointerup", up);
}
/* playing: the view follows the playhead when it runs out of the view */
function _follow(node) {
    const st = node._plsTl;
    if (!st || !st.scroller || zoomOf(node) <= 1 || st.scrubbing) return;
    const v = _view(node), t = (st.player && st.player.t) || 0;
    if (t < v.t0 || t > v.t1 - (v.t1 - v.t0) * 0.02) _scrollTo(node, t - (v.t1 - v.t0) * 0.1);
}

/* ---------------------------------------------------------- v1150 mode, form, refs, chips */
export const CTE_TYPE = "ULSCLIPTextEncode", CTE_FILL = "fill {pos_N}";
function _setMode(node, k) {
    const st = node._plsTl, mm = _model(node);
    if (k === "text" && mm.anchors.length) {
        st.state.textContent = "Text: remove the " + mm.anchors.length + " anchor(s) first -- they go into the run as long as they exist";
        return;
    }
    node.properties = node.properties || {};
    if (k !== "ref") node.properties[TAB_MODE_KEY] = k;
    const want = k === "ref" ? "ref2va" : "fl2va";
    if (mm.target !== want) { mm.target = want; _save(node, mm); }
    else node.setDirtyCanvas?.(true, true);
    _render(node);
}
/* the CLIP Text Encode whose pos_external is this timeline's prompt output (null = none) */
export function cteOf(node) {
    const g = node && (node.graph || app.graph);
    const out = node && node.outputs && node.outputs[0];
    if (!g || !out || !out.links) return null;
    for (const id of out.links) {
        const l = _link(node, id);
        const t = l && g.getNodeById ? g.getNodeById(l.target_id) : null;
        if (t && t.type === CTE_TYPE && t.inputs && t.inputs[l.target_slot] && t.inputs[l.target_slot].name === "pos_external") return t;
    }
    return null;
}
/* the segments that fill this timeline's form: the wired CTE's pos_1..pos_n while its mode is fill (null = none) */
export function fillOf(node) {
    const c = cteOf(node);
    if (!c || String(_w(c, "external_mode")?.value || "") !== CTE_FILL) return null;
    const n = Math.max(1, Math.min(6, Number(_w(c, "segments")?.value) || 1));
    // v1154 (audit): a segment (or the count) wired as a socket -- the server's form_fill reads nothing then; the
    // preview says so too instead of filling with the widget's stale text
    const wired = (name) => (c.inputs || []).some((x) => x && x.name === name && x.link != null);
    if (wired("segments")) return null;
    const out = [];
    for (let i = 1; i <= n; i++) { if (wired("pos_" + i)) return null; out.push(String(_w(c, "pos_" + i)?.value ?? "")); }
    return out;
}
const _isNote = (t) => !String(t || "").split("\n").some((ln) => ln.trim() && !ln.trim().startsWith("//"));
async function _toCte(node) {
    const st = node._plsTl, c = cteOf(node);
    if (!c) { st.state.textContent = "-> CTE: wire 'prompt' to a CLIP Text Encode's pos_external first"; return; }
    const w = _w(node, "timeline");
    let d;
    try {
        const r = await api.fetchApi("/pls/cine/to_form", { method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ timeline: w ? w.value : "" }) });
        d = await r.json();
    } catch (e) { st.state.textContent = "-> CTE failed: " + e.message; return; }
    if (d.error) { st.state.textContent = "-> CTE: " + d.error; return; }
    const segs = d.segments || [];
    if (!segs.length) { st.state.textContent = "-> CTE: Shot 1 holds nothing fixed to move (only steering rubrics)"; return; }
    const busy = [];
    for (let i = 1; i <= 6; i++) { const v = _w(c, "pos_" + i)?.value; if (!_isNote(v)) busy.push("pos_" + i); }
    if (busy.length && !window.confirm("The CLIP Text Encode already holds text in " + busy.join(", ") + ".\n\n"
        + "-> CTE writes " + segs.length + " segment(s) there (pos_1.." + segs.length + ") and sets external_mode to '" + CTE_FILL + "'. Overwrite?")) return;
    _begin(node);
    const setv = (name, v) => { const x = _w(c, name); if (x) { x.value = v; x.callback?.(x.value); } };
    setv("segments", segs.length);
    segs.forEach((t, i) => setv("pos_" + (i + 1), t));
    setv("external_mode", CTE_FILL);
    if (w) { w.value = d.timeline; w.callback?.(w.value); }
    c.setDirtyCanvas?.(true, true);
    st.state.textContent = "-> CTE: " + segs.length + " segment(s) moved; Shot 1 keeps " + segs.map((_, i) => "{pos_" + (i + 1) + "}").join(", ")
        + " where they stood -- the encoder sees the same text";
    _render(node); _preview(node);
}
/* the picture behind a wired IMAGE input: a node's preview, a Load Image file, a Media Loader item */
function _thumbOf(src) {
    if (!src) return "";
    try {
        if (src.imgs && src.imgs.length && src.imgs[0].src) return src.imgs[0].src;
        const iw = _w(src, "image");
        if (iw && typeof iw.value === "string" && iw.value) {
            const v = iw.value.replace(/\s*\[(input|output|temp)\]\s*$/, ""), typ = (/\[(output|temp)\]/.exec(iw.value) || [0, "input"])[1];
            const k = v.lastIndexOf("/");
            return _viewURL({ filename: k < 0 ? v : v.slice(k + 1), subfolder: k < 0 ? "" : v.slice(0, k), type: typ });
        }
        const mr = _w(src, "media_ref");
        if (mr && mr.value) {
            const j = JSON.parse(mr.value);
            if (j && j.folder && j.file) return api.apiURL("/uls/media/thumb?folder=" + encodeURIComponent(j.folder) + "&file=" + encodeURIComponent(j.file));
        }
    } catch (e) { /* a picture is a nicety */ }
    return "";
}
export function refItems(node) {
    const g = node && (node.graph || app.graph), out = [];
    if (!g) return out;
    const up = (name) => { const i = (node.inputs || []).find((x) => x && x.name === name); const l = i && i.link != null ? _link(node, i.link) : null;
        return l && g.getNodeById ? g.getNodeById(l.origin_id) : null; };
    const board = up("refs");
    const bw = board && _w(board, "board");
    if (bw) { try { for (const e of JSON.parse(bw.value || "[]")) if (e && e.kind === "image" && e.tag) out.push({ name: "@" + e.tag, url: _viewURL(Object.assign({ type: "input" }, e.file || {})) }); } catch (e) { /* not a board */ } }
    for (const sname of SLOTS) { const n = up(sname); if (n) out.push({ name: sname, url: _thumbOf(n), title: n.title || n.type }); }
    return out;
}
/* v1162 (Frank 18.10., the neon lane: three photos on MiniMax Reference, the timeline said "no reference
 * pictures"): the pictures may hang on the MiniMax Reference node the prompt flows into, not on this node.
 * Follow the outputs downstream (prompt -> CLIP Text Encode -> MiniMax Reference, a few hops) and list the
 * images wired THERE, in wired order -- the order the node numbers them <Picture 1..n>. */
export const REF_NODE = "ULSMiniMaxReference", DOWN_HOPS = 4;
export function refItemsDown(node) {
    const g = node && (node.graph || app.graph), out = [];
    if (!g || !g.getNodeById) return out;
    const seen = new Set([node.id]);
    let front = [node];
    for (let hop = 0; hop < DOWN_HOPS && front.length; hop++) {
        const next = [];
        for (const n of front) for (const o of (n.outputs || [])) for (const id of (o && o.links) || []) {
            const l = _link(n, id), t = l && g.getNodeById(l.target_id);
            if (!t || seen.has(t.id)) continue;
            seen.add(t.id);
            if (t.type === REF_NODE) {
                let k = 0;
                for (const inp of (t.inputs || [])) {
                    if (!inp || !/^image_\d+$/.test(inp.name) || inp.link == null) continue;
                    const il = _link(t, inp.link), src = il && g.getNodeById(il.origin_id);
                    if (!src) continue;
                    k += 1;
                    out.push({ name: "@" + inp.name, url: _thumbOf(src), title: "<Picture " + k + "> \u00b7 " + (src.title || src.type) });
                }
            } else next.push(t);
        }
        front = next;
    }
    return out;
}
function _refsRow(node, md) {
    const st = node._plsTl;
    if (!st || !st.refsRow || md === "text") return;
    let items = refItems(node);
    if (!items.length && md === "ref") items = refItemsDown(node);   // v1162: the pictures at MiniMax Reference count
    const sig = md + "|" + items.map((x) => x.name + "=" + x.url).join("|");
    if (st.refsSig === sig) return;
    st.refsSig = sig;
    st.refsRow.replaceChildren(_el("span", "color:#888;flex:0 0 auto;", md === "ref" ? "refs" : "pictures"));
    if (!items.length) {
        st.refsRow.appendChild(_el("span", "color:#777;font-size:11px;", md === "ref" ? "nothing wired -- a Reference Board on 'refs' or pictures on image_1..9" : "nothing wired -- pictures on image_1..9 (or a board on 'refs') become anchors"));
        // v1153 (Frank: "the node stands on Reference, but we have none?"): Reference without pictures is Text's job
        if (md === "ref") { const w = _el("span", "color:#ffb300;font-size:11px;", "\u2190 no reference pictures: a prompt-only run is Text (T2V) -- Reference needs a MiniMax Reference node and its pictures"); w.dataset.pls = "refhint"; st.refsRow.appendChild(w); }
        return;
    }
    for (const it of items) {
        const chip = _el("div", "display:inline-flex;gap:6px;align-items:center;background:#22222a;border:1px solid #3e3e4a;border-radius:16px;padding:2px 10px 2px 2px;flex:0 0 auto;cursor:default;");
        chip.dataset.pls = "refchip";
        const th = _el(it.url ? "img" : "div", "width:56px;height:32px;border-radius:12px;object-fit:cover;background:#333;display:block;");
        if (it.url) th.src = it.url;
        chip.append(th, _el("span", "color:#ddd;font-size:12px;white-space:nowrap;", it.name + (it.title ? " \u00b7 " + it.title : "")));
        chip.title = it.name + (it.url ? "" : " (no preview yet -- run the node that makes it once)");
        if (it.url) {
            chip.addEventListener("mouseenter", () => {
                const r = st.root.getBoundingClientRect(), c = chip.getBoundingClientRect(), z = r.width / (st.root.offsetWidth || r.width) || 1;
                st.popImg.src = it.url;
                st.pop.style.left = Math.max(0, (c.left - r.left) / z) + "px";
                st.pop.style.top = ((c.bottom - r.top) / z + 4) + "px";
                st.pop.style.display = "block";
            });
            chip.addEventListener("mouseleave", () => { st.pop.style.display = "none"; });
        }
        st.refsRow.appendChild(chip);
    }
}
/* the check as chips: errors red, warnings amber, else a green "H3 ok"; the form's state */
function _chips(node, d) {
    const st = node._plsTl;
    if (!st || !st.chips) return;
    const ch = (txt, kind, title) => {
        const col = kind === "err" ? "background:#4a1c1c;border:1px solid #a84040;color:#ff9a9a;" : kind === "warn" ? "background:#4a3410;border:1px solid #a87520;color:#f2c46a;"
            : kind === "form" ? "background:#1e2c40;border:1px solid #3d6ea8;color:#9fc8ff;" : "background:#173a22;border:1px solid #2f7a45;color:#8fe0a6;";
        const e = _el("span", col + "border-radius:12px;padding:2px 10px;white-space:nowrap;font-size:11px;", txt);
        e.title = title || txt;
        return e;
    };
    st.chips.replaceChildren();
    if (!d) return;
    if (d.errors && d.errors.length) { st.chips.appendChild(ch("does not compile: " + d.errors.length, "err", d.errors.join("\n"))); return; }
    const c = d.check || [], er = c.filter((x) => x.startsWith("ERROR")), wa = c.filter((x) => x.startsWith("WARN"));
    if (er.length) st.chips.appendChild(ch(er.length + " error(s)", "err", er.join("\n")));
    for (const x of wa.slice(0, 2)) st.chips.appendChild(ch(x.replace(/^WARN:\s*/, "").slice(0, 70) + (x.length > 76 ? "\u2026" : ""), "warn", x));
    if (wa.length > 2) st.chips.appendChild(ch("+" + (wa.length - 2) + " more", "warn", wa.slice(2).join("\n")));
    if (!er.length && !wa.length) st.chips.appendChild(ch("H3 format ok", "ok"));
    if (d.places && d.places.length) st.chips.appendChild(ch(d.places.map((x) => "{" + x + "}").join(" ") + (st.filled ? " \u2190 CTE" : " (no CTE fills them)"), st.filled ? "form" : "warn",
        st.filled ? "the CLIP Text Encode on 'prompt' fills these places (external_mode fill {pos_N})" : "wire 'prompt' to a CLIP Text Encode's pos_external and set its external_mode to fill {pos_N}"));
}

function _render(node) {
    const st = node._plsTl;
    if (!st) return;
    const m = _model(node);
    st.m = m;
    // v1150: the mode decides what is on screen (the sections, the anchors row and buttons, the refs)
    const md = modeOf(node, m);
    if (st.modeBtns) for (const [k, b] of Object.entries(st.modeBtns)) {
        const on = k === md;
        b.style.background = on ? "#3d6fd6" : "#24242c"; b.style.color = on ? "#fff" : "#bbb"; b.style.fontWeight = on ? "bold" : "normal";
    }
    if (st.anchorsRow) st.anchorsRow.style.display = md === "text" ? "none" : "flex";
    if (st.anchorBtns) st.anchorBtns.style.display = md === "text" ? "none" : "inline-flex";
    _refsRow(node, md);
    const n = framesFor(m.seconds), dur = n / FPS;
    if (document.activeElement !== st.len) st.len.value = Number(m.seconds).toFixed(2);
    st.lenInfo.textContent = "-> " + n + " frames = " + dur.toFixed(2) + " s" + (dur < 4 || dur > 15 ? "  (official 4-15 s)" : "");
    if (document.activeElement !== st.style) st.style.value = m.style || "";
    if (st.trig && document.activeElement !== st.trig) st.trig.value = m.triggers || "";
    _fitHead(node);
    if (st.tgt) st.tgt.value = TARGETS.some(([v]) => v === m.target) ? m.target : "fl2va";
    // v1036: one axis for every lane -- the prompt's grid, or the film when longer
    // v1039: room after the film's end -- its line and diamond sat on the
    // lanes' edge, crowded with the grips (Frank's third screenshot)
    const ax = viewAxis(axisOf(m, st.plan));
    st.ax = ax;
    _applyZoom(node);              // v1047: the lanes' width first -- the ruler measures it
    const pct = (s) => (100 * s / ax) + "%";

    // ruler: seconds + shaded 17-frame groups
    st.ruler.replaceChildren();
    for (let g = 0; g * 17 < n; g++) {
        if (g % 2) st.ruler.appendChild(_el("div", "position:absolute;top:0;bottom:0;background:#ffffff08;left:" + pct(g * 17 / FPS) + ";width:" + pct(Math.min(17, n - g * 17) / FPS)));
    }
    // v1047: the labelled step follows the scale (zoom, film length): >= 56 px apart
    const rw = st.ruler.clientWidth || Math.max(200, ((node.size && node.size[0]) || 900) - GUT_W - 24) * zoomOf(node);
    const stp = tickStep(rw / ax), minor = stp >= 5 && rw / ax * stp / 5 >= 10 ? stp / 5 : 0;
    if (minor) for (let s = minor; s <= ax; s += minor) {
        if (Math.abs(s / stp - Math.round(s / stp)) > 1e-6) st.ruler.appendChild(_el("div", "position:absolute;bottom:0;height:5px;border-left:1px solid #444;left:" + pct(s) + ";"));
    }
    for (let s = 0; s <= ax + 1e-6; s += stp) {
        const t = _el("div", "position:absolute;top:0;height:100%;border-left:1px solid #555;left:" + pct(s) + ";font-size:9px;color:#888;padding-left:2px;white-space:nowrap;",
            s >= 60 || ax >= 60 ? fmtLen(s) : s + "s");
        st.ruler.appendChild(t);
    }

    // scenes + transition chips
    const sc = scenesOf(m);
    st.scenes.replaceChildren();
    sc.forEach(([a, b], k) => {
        const t0 = m.shots[a].at, t1 = b < m.shots.length ? m.shots[b].at : dur;
        const active = m.render === "scene" && m.active === k + 1;
        const picked = st.sel && st.sel.kind === "scene" && st.sel.k === k + 1;
        // v1034: a scene after the first starts under half a transition chip --
        // its label keeps that room (Frank's screenshot: "Scene 2" was covered)
        const bar = _el("div", "position:absolute;top:2px;bottom:2px;border-radius:3px;overflow:hidden;white-space:nowrap;font-size:10px;color:#ddd;padding:2px 6px;box-sizing:border-box;cursor:pointer;"
            + "left:" + pct(t0) + ";width:" + pct(t1 - t0) + ";background:" + SCENE_COL[k % SCENE_COL.length] + ";"
            + (picked ? "outline:2px solid #f0c87a;" : active ? "outline:2px solid #80cbc4;" : "border:1px solid #333;"),
            (carryOf(m, k + 1) ? "\u21e4 " : "") + "Scene " + (k + 1) + "  " + (t1 - t0).toFixed(2) + " s" + (active ? "  (renders)" : ""));
        bar.title = "Scene " + (k + 1) + ": " + timeLabel(t0) + " - " + timeLabel(t1) + ", shots " + (a + 1) + "-" + b
            + "\nclick: select its first shot  |  double-click: its start and end";
        // v1163: the tick -- "render all scenes" / the all-scenes run queue the ticked scenes only
        if (sc.length > 1) {
            const on = !skipOf(node).has(k + 1);
            const tick = document.createElement("input"); tick.type = "checkbox"; tick.checked = on; tick.dataset.pls = "scenetick";
            tick.style.cssText = "margin:0 4px 0 0;vertical-align:middle;width:11px;height:11px;cursor:pointer;";
            tick.title = "ticked: 'render all scenes' renders scene " + (k + 1) + "  |  unticked: it is left out (its clips stay)";
            tick.addEventListener("pointerdown", (ev) => ev.stopPropagation());
            tick.addEventListener("click", (ev) => ev.stopPropagation());
            tick.addEventListener("change", () => { _setSkip(node, k + 1, tick.checked); _render(node); });
            bar.prepend(tick);
            if (!on) bar.style.opacity = "0.45";
        }
        bar.addEventListener("pointerdown", (ev) => {
            ev.stopPropagation();
            if (isDouble(st, "scene:" + (k + 1))) { ev.preventDefault(); st.sel = { kind: "scene", k: k + 1 }; st.open = { focus: "start" }; }
            else st.sel = { kind: "shot", i: a };
            _render(node);
        });
        st.scenes.appendChild(bar);
        // v1036: the transition chips moved to the clip lane's seams
    });

    // shots
    st.shots.replaceChildren();
    m.shots.forEach((s, i) => {
        const end = i + 1 < m.shots.length ? m.shots[i + 1].at : dur;
        const b = _el("div", "position:absolute;top:2px;bottom:2px;border-radius:4px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;font-size:12px;line-height:26px;color:#eee;padding:0 8px;box-sizing:border-box;cursor:pointer;"
            + "left:" + pct(s.at) + ";width:" + pct(end - s.at) + ";background:" + SHOT_COL[i % SHOT_COL.length] + ";"
            + (st.sel && st.sel.kind === "shot" && st.sel.i === i ? "outline:2px solid #f0c87a;" : ""),
            barText(i, s.text, 60));
        b.title = "[Shot " + (i + 1) + "] " + (i ? "At " + timeLabel(s.at) + (s.scene ? "  (starts scene)" : "") : "") + "\n" + (s.text || "(no text yet)")
            + "\ndouble-click: type its time and length";
        b.addEventListener("pointerdown", (ev) => {
            ev.stopPropagation();
            if (isDouble(st, "shot:" + i)) { ev.preventDefault(); st.open = { focus: i ? "at" : "length" }; }
            st.sel = { kind: "shot", i };
            _render(node);
        });
        st.shots.appendChild(b);
        if (i > 0) {
            // v1028: the handle you can SEE -- a grip on the cut, gold, with a
            // dark seam, the time on hover
            const ht = h3tOf(s);
            if (ht) {
                // v1035: the band of the asked length, on both sides of the cut
                const w = Math.max(H3T_MIN, Math.min(H3T_MAX, Number(ht.seconds) || H3T_DEFAULT));
                const a0 = Math.max(0, s.at - w), a1 = Math.min(dur, s.at + w);
                st.shots.appendChild(_el("div", "position:absolute;top:0;bottom:0;pointer-events:none;z-index:1;border-radius:3px;"
                    + "background:repeating-linear-gradient(135deg," + H3T_COL + "55 0 4px,transparent 4px 8px);left:" + pct(a0) + ";width:" + pct(a1 - a0) + ";"));
            }
            const h = _el("div", "position:absolute;top:0;bottom:0;width:12px;margin-left:-6px;cursor:ew-resize;z-index:3;left:" + pct(s.at) + ";"
                + "background:" + (s.scene ? "#80cbc4" : ht ? H3T_COL : "#f0c87a") + ";border-left:1px solid #000;border-right:1px solid #000;border-radius:2px;"
                + "display:flex;align-items:center;justify-content:center;color:#3a2a00;font-size:11px;font-weight:bold;user-select:none;", "\u2016");
            h.title = (s.scene ? "scene cut at " : ht ? ht.kind + " at " : "cut at ") + timeLabel(s.at) + " -- drag  |  double-click: time + transition";
            h.addEventListener("mouseenter", () => { h.style.filter = "brightness(1.25)"; });
            h.addEventListener("mouseleave", () => { h.style.filter = ""; });
            _drag(node, h, st.shots, "cut:" + i, (sec, done) => {
                const mm = _model(node); moveCut(mm, i, sec);
                h.style.left = pct(mm.shots[i].at);
                h.title = "cut at " + timeLabel(mm.shots[i].at);
                if (done) { _save(node, mm); _render(node); }
            }, () => {
                // v1030: double-click on a cut -> its time AND its transition
                st.sel = { kind: "shot", i }; st.open = { focus: "at", transition: true };
                _render(node);
            }, () => { st.sel = { kind: "shot", i }; _render(node); });
            st.shots.appendChild(h);
        }
    });

    // v1036: where the FILM cuts differ from the prompt -- ghost lines
    const P = st.plan && st.plan.plan;
    if (P && !st.plan.whole && P.place_cuts.length === P.cuts_prompt.length) P.place_cuts.forEach((f, j) => {
        if (f !== P.cuts_prompt[j]) st.shots.appendChild(_el("div", "position:absolute;top:0;bottom:0;width:0;border-left:2px dashed #80cbc4;pointer-events:none;z-index:4;left:" + pct(f / FPS) + ";"));
    });
    // v1044: the layer rows first -- they sit above the render lane and move it
    _stackRows(node, pct);
    // v1039: the lanes' heights are set in ONE place (_applyLanes, first thing in _clipLane)
    _clipLane(node, m, pct);
    if (st.span && document.activeElement !== st.span) {
        const e = editOf(node);
        st.span.value = e.span ? fmtLen(e.span) : "";
        st.span.placeholder = st.plan && st.plan.plan ? fmtLen(st.plan.plan.length / FPS) : fmtLen(dur);
    }
    if (st.filmInfo) {
        const e = editOf(node), rl = st.plan && st.plan.plan ? st.plan.plan.length / FPS : dur;
        st.filmInfo.textContent = e.span ? "render " + fmtLen(rl) + "  \u00b7  room " + fmtLen(e.span) : "= the render (" + fmtLen(rl) + ")";
    }
    _navDraw(node);

    // anchors
    st.anchors.replaceChildren();
    m.anchors.forEach((a, i) => {
        // v1219: the role in the run that renders it -- scene by scene, a boundary anchor is scene k's FIRST frame
        // (and with 'both' also the last of k-1), not a guide of the whole timeline
        const ri = anchorRoleIn(m, a), role = ri.role === "both" ? "first" : ri.role;
        const col = role === "guide" ? "#4fc3f7" : (ri.role === "both" ? "#ffd54f" : "#9ccc65");
        const p = _el("div", "position:absolute;top:3px;height:18px;min-width:14px;margin-left:-7px;cursor:ew-resize;font-size:10px;color:#111;white-space:nowrap;"
            + "padding:0 4px;border-radius:9px;border:1px solid #000;background:" + col + ";left:" + pct(a.at) + ";"
            + (st.sel && st.sel.kind === "anchor" && st.sel.i === i ? "outline:2px solid #f0c87a;" : ""), a.src || "?");
        p.title = a.src + " at " + timeLabel(a.at) + " -- " + (ri.role === "both"
            ? "a KEYFRAME of both scenes: the last frame of scene " + (ri.scene - 1) + " and the first of scene " + ri.scene
            : role === "guide" ? "a GUIDE -> Continue anchors (Core master)" + (ri.scene ? " (scene " + ri.scene + ")" : "")
            : "a KEYFRAME (" + role + " frame" + (ri.scene ? " of scene " + ri.scene : "") + ") -> Keyframes " + role + "_frame") + " -- drag";
        _drag(node, p, st.anchors, "anchor:" + i, (sec, done) => {
            const mm = _model(node); moveAnchor(mm, i, sec);
            p.style.left = pct(mm.anchors[i].at);
            st.sel = { kind: "anchor", i };
            if (done) { _save(node, mm); _render(node); }
        }, null, () => { st.sel = { kind: "anchor", i }; _render(node); });
        st.anchors.appendChild(p);
    });

    // source list for new anchors
    const srcs = _fillSources(node);
    if (st.snapSel && document.activeElement !== st.snapSel) st.snapSel.value = String(snapStep(node));
    if (st.autoChk) st.autoChk.value = autoMode(editOf(node));   // v1163: the choice mirrors the edit; v1198: three modes
    if (st.restChk) st.restChk.checked = !(node.properties && node.properties[RESTORE_KEY] === false);
    if (st.takesBtn) { const older = ((st.plan && st.plan.clips) || []).some((c) => c && (c.takes || []).length > 1);
        st.takesBtn.disabled = !older; st.takesBtn.style.opacity = older ? "1" : "0.5"; }

    // editor of the selection
    st.edit.replaceChildren();
    st.transSel = null;
    const sel = st.sel || { kind: "shot", i: 0 };
    if (sel.kind === "shot" && m.shots[sel.i]) {
        const s = m.shots[sel.i];
        const head = _el("div", "display:flex;gap:8px;align-items:center;font-size:13px;font-weight:bold;color:#f0c87a;min-height:26px;",
            "[Shot " + (sel.i + 1) + "]" + (sel.i ? "  At " + timeLabel(s.at) : "  (no time -- official form)"));
        if (sel.i > 0) {
            const c = _el("select", INPUT);
            for (const x of CUTS) { const o = document.createElement("option"); o.value = x; o.textContent = x; c.appendChild(o); }
            c.value = s.cut || CUTS[0];
            c.addEventListener("change", () => { const mm = _model(node); mm.shots[sel.i].cut = c.value; _save(node, mm); });
            const ht = h3tOf(s);
            if (ht) {
                // v1035: the preset sets the formula -- shown, locked
                const f = ht.kind === H3T_CUSTOM ? ht.formula : H3T_PRESETS[ht.kind].formula;
                if (!Array.from(c.options).some((o) => o.value === f)) { const o = document.createElement("option"); o.value = f; o.textContent = f; c.appendChild(o); }
                c.value = f; c.disabled = true; c.title = "set by " + ht.kind + " (at this cut)";
            }
            head.appendChild(c);
            const lab = _el("label", "display:flex;gap:3px;align-items:center;color:#80cbc4;cursor:pointer;margin-left:6px;");
            const cb = document.createElement("input"); cb.type = "checkbox"; cb.checked = !!s.scene;
            cb.addEventListener("change", () => { const mm = _model(node); setSceneStart(mm, sel.i, cb.checked); _save(node, mm); _render(node); _stack(node); });
            lab.append(cb, _el("span", "", "starts a new scene"));
            lab.title = "a scene is a clip of its own: rendered alone (render: scene k) and joined with a transition";
            head.appendChild(lab);
            if (s.scene) { const kk = scenesOf(m).findIndex(([a0]) => a0 === sel.i) + 1; if (kk > 1) head.append(..._carryCtl(node, m, kk)); }
        }
        const custom = sel.i > 0 && (h3tOf(s) || {}).kind === H3T_CUSTOM;
        const ta = _el("textarea", INPUT + "width:100%;height:" + (custom ? TEXT_H - CUSTOM_ROW_H - 3 : TEXT_H) + "px;resize:none;");
        ta.placeholder = "what happens in this shot -- camera, action, @tags from the board";
        ta.value = s.text || "";
        ta.addEventListener("input", () => { const mm = _model(node); mm.shots[sel.i].text = ta.value; _save(node, mm, false); });
        ta.addEventListener("change", () => { _commit(node); _render(node); });
        st.edit.append(head, _timesRow(node, m, sel.i));
        if (custom) st.edit.append(_customRow(node, m, sel.i));
        st.edit.append(ta);
    } else if (sel.kind === "scene" && sc[sel.k - 1]) {
        st.edit.append(_sceneRow(node, m, sel.k));
    } else if (sel.kind === "anchor" && m.anchors[sel.i]) {
        const a = m.anchors[sel.i];
        const row = _el("div", "display:flex;gap:6px;align-items:center;font-size:11px;color:#ccc;");
        const s = _el("select", INPUT);
        for (const x of Array.from(new Set(srcs.concat([a.src])))) { const o = document.createElement("option"); o.value = x; o.textContent = x; s.appendChild(o); }
        s.value = a.src;
        s.addEventListener("change", () => { const mm = _model(node); mm.anchors[sel.i].src = s.value; _save(node, mm); _render(node); });
        const t = _el("input", INPUT + "width:70px;"); t.type = "number"; t.step = String(1 / FPS); t.min = "0";
        t.value = Number(a.at).toFixed(3);
        t.addEventListener("change", () => { const mm = _model(node); moveAnchor(mm, sel.i, parseFloat(t.value) || 0); _save(node, mm); _render(node); });
        const ri = anchorRoleIn(m, a), role = ri.role === "both" ? "first" : ri.role;
        row.append(_el("span", "", "anchor"), s, _el("span", "", "at"), t, _el("span", "", "s"),
            _el("span", "color:" + (role === "guide" ? "#4fc3f7" : (ri.role === "both" ? "#ffd54f" : "#9ccc65")) + ";",
                ri.role === "both" ? "keyframe of BOTH scenes: last of scene " + (ri.scene - 1) + ", first of scene " + ri.scene + " -> Keyframes"
                : role === "guide" ? "guide -> Continue (Core master)" + (ri.scene ? ", scene " + ri.scene : "")
                : "keyframe " + role + (ri.scene ? " of scene " + ri.scene : "") + " -> Keyframes"));
        // v1219 (A -> B over a scene boundary): an anchor ON a boundary may serve both scenes -- the clip before ends
        // on it, the clip after starts on it; the two meet on one picture
        if (boundaryFrames(m).has(Math.round(Number(a.at) * FPS))) {
            const bl = _el("label", "display:flex;gap:3px;align-items:center;cursor:pointer;color:#ffd54f;");
            const bc = document.createElement("input"); bc.type = "checkbox"; bc.checked = !!a.both; bc.dataset.pls = "anchorboth";
            bc.addEventListener("change", () => { const mm = _model(node); if (!mm.anchors[sel.i]) return;
                if (bc.checked) mm.anchors[sel.i].both = true; else delete mm.anchors[sel.i].both; _save(node, mm); _render(node); });
            bl.title = "on a scene boundary: also the LAST frame of the scene before -- both clips meet on this picture (scene by scene renders)";
            bl.append(bc, _el("span", "", "both scenes"));
            row.append(bl);
        }
        st.edit.append(row);
    } else if (sel.kind === "transition" && st.plan && st.plan.plan && st.plan.plan.seams[sel.i]) {
        // v1037: the transition belongs to the seam of two pieces (the edit)
        const pcs = piecesOf(st.plan), q = pcs[sel.i + 1];
        const t = q.tr && TRANSITION_NAMES.includes(q.tr.kind) ? q.tr : { kind: "cut", seconds: 0.5 };
        const row = _el("div", "display:flex;gap:6px;align-items:center;font-size:11px;color:#ffcc80;");
        const k = _el("select", INPUT); k.dataset.pls = "seamkind";
        for (const x of TRANSITION_NAMES) { const o = document.createElement("option"); o.value = x; o.textContent = x; k.appendChild(o); }
        k.value = t.kind;
        const d = _el("input", INPUT + "width:60px;"); d.type = "number"; d.step = "0.05"; d.min = "0.05"; d.max = "5"; d.value = Number(t.seconds).toFixed(2);
        d.disabled = t.kind === "cut";
        const put = () => { const pp = piecesOf(st.plan); pp[sel.i + 1].tr = { kind: k.value, seconds: Math.max(0.05, Math.min(5, parseFloat(d.value) || 0.5)) };
            if (k.value === "cut") delete pp[sel.i + 1].tr; _savePieces(node, pp); };
        k.addEventListener("change", put); d.addEventListener("change", put);
        const sm = st.plan.plan.seams[sel.i];
        row.append(_el("span", "", "piece " + (sel.i + 1) + " -> " + (sel.i + 2)), k, d, _el("span", "", "s"),
            _el("span", "color:#888;", "centred on the film's cut at " + timeLabel(sm.at / FPS) + (t.kind === "cut" ? "" : "; without a held picture up to " + (sm.max_d / FPS).toFixed(2) + " s")));
        st.edit.append(row);
    } else if (sel.kind === "clip" && st.plan && st.plan.plan && st.plan.plan.segments[sel.i]) {
        // v1037: the selected piece -- its clip's take, where it starts in the
        // clip, its length and speed
        const D = st.plan, sg = D.plan.segments[sel.i], c = D.clips.find((x) => x && x.k === sg.k) || {};
        const pcs = piecesOf(D), pc = pcs[sel.i], fK = (k) => framesOf(D, k);
        const row = _el("div", "display:flex;gap:6px;align-items:center;font-size:11px;color:#80cbc4;flex-wrap:wrap;");
        row.append(_el("span", "", (sg.k ? "scene " + sg.k : "the whole run") + (pcs.filter((q) => q.k === sg.k).length > 1 ? ", piece " + (sel.i + 1) : "")));
        const from = _secField(pc.a / FPS, "where this piece starts in its clip (slip)", (v) => {
            if (Number.isFinite(v)) _savePieces(node, slipPiece(pcs, sel.i, Math.round(v * FPS) - pc.a, fK)); });
        from.dataset.pls = "slip";
        const len = _secField(pieceLen(pc) / FPS, "the piece's length on the film -- longer = slower, shorter = faster", (v) => {
            if (Number.isFinite(v)) _savePieces(node, stretch(pcs, sel.i, Math.round(v * FPS))); });
        len.dataset.pls = "plen";
        const spd = _el("input", INPUT + "width:56px;"); spd.type = "number"; spd.step = "0.05"; spd.min = String(SPEED_MIN); spd.max = String(SPEED_MAX);
        spd.value = Number(pc.speed).toFixed(2); spd.dataset.pls = "speed"; spd.title = "speed: 1 = as rendered, 0.5 = half as fast (twice as long)";
        spd.addEventListener("change", () => { const v = parseFloat(spd.value); if (!(v > 0)) return; const q = piecesOf(D); q[sel.i].speed = Math.max(SPEED_MIN, Math.min(SPEED_MAX, v)); _savePieces(node, q); });
        spd.addEventListener("keydown", (e) => { e.stopPropagation(); if (e.key === "Enter") spd.blur(); });
        row.append(_el("span", "", "from"), from, _el("span", "", "s  length"), len, _el("span", "", "s  speed"), spd,
            _el("span", "color:#888;font-size:10px;", (sg.speed < 1 ? "slower: the in-between frames are blended (ffmpeg)" : sg.speed > 1 ? "faster: frames are left out" : "")
                + (sg.hold ? (sg.speed !== 1 ? "; " : "") + "picture held " + (sg.hold / FPS).toFixed(2) + " s" : "")));
        st.edit.append(row);
    }
    else if (sel.kind === "lpiece") {
        // v1044: a piece of another layer -- where it starts in its file, its
        // length on the film, its speed (the numbers of the drags)
        const ch = _chainOf(node, sel.id), pc = ch.pcs[sel.i];
        if (pc && !isGap(pc)) {
            const row = _el("div", "display:flex;gap:6px;align-items:center;font-size:11px;color:#b39ddb;flex-wrap:wrap;");
            const name = (ch.row && ch.row.name) || sel.id;
            row.append(_el("span", "", "layer " + name + ", piece at " + timeLabel(starts(ch.pcs)[sel.i] / FPS) + (ch.lock ? " (locked)" : "")));
            const from = _secField(pc.a / FPS, "where this piece starts in its file (slip)", (v) => {
                if (Number.isFinite(v) && !ch.lock) ch.save(slipPiece(ch.pcs, sel.i, Math.round(v * FPS) - pc.a, ch.fK)); });
            from.dataset.pls = "lslip";
            const len = _secField(pieceLen(pc) / FPS, "the piece's length on the film -- longer = slower, shorter = faster", (v) => {
                if (Number.isFinite(v) && !ch.lock) ch.save(stretch(ch.pcs, sel.i, Math.round(v * FPS))); });
            len.dataset.pls = "llen";
            const spd = _el("input", INPUT + "width:56px;"); spd.type = "number"; spd.step = "0.05"; spd.min = String(SPEED_MIN); spd.max = String(SPEED_MAX);
            spd.value = Number(pc.speed).toFixed(2); spd.dataset.pls = "lspeed";
            spd.addEventListener("change", () => { const v = parseFloat(spd.value); if (!(v > 0) || ch.lock) return; const q = ch.pcs.map((x) => Object.assign({}, x)); q[sel.i].speed = Math.max(SPEED_MIN, Math.min(SPEED_MAX, v)); ch.save(q); });
            spd.addEventListener("keydown", (e) => { e.stopPropagation(); if (e.key === "Enter") spd.blur(); });
            for (const x of [from, len, spd]) x.disabled = !!ch.lock;
            row.append(_el("span", "", "from"), from, _el("span", "", "s  length"), len, _el("span", "", "s  speed"), spd);
            st.edit.append(row);
        }
    }
    _openRequest(st);
    if (document.activeElement !== st.sound) st.sound.value = m.soundscape || "";
    if (document.activeElement !== st.music) st.music.value = m.music || "";

    // render choice -- the user's, never guessed
    const want = m.render === "all" ? (sc.length > 1 ? "all" : "scene:1")
        : m.render === "scene" ? "scene:" + Math.min(sc.length, m.active) : "whole";
    st.mode.replaceChildren();
    const o0 = document.createElement("option"); o0.value = "whole"; o0.textContent = sc.length > 1 ? "whole timeline, one run (" + sc.length + " scenes as hard cuts)" : "whole timeline, one run"; st.mode.appendChild(o0);
    const nTick = pickedOrder(sc.length, 0, skipOf(node)).length;   // v1163
    if (sc.length > 1) { const oa = document.createElement("option"); oa.value = "all"; oa.textContent = nTick < sc.length ? "ticked scenes, one run each (" + nTick + " of " + sc.length + ")" : "all scenes, one run each (" + sc.length + " runs)"; st.mode.appendChild(oa); }
    sc.forEach((_, k) => { const o = document.createElement("option"); o.value = "scene:" + (k + 1); o.textContent = "scene " + (k + 1) + " only" + (sc.length === 1 ? " (= the whole)" : ""); st.mode.appendChild(o); });
    if (st.stackLabel) { const h = stackHint(m, sc.length); st.stackLabel.textContent = h; st.stackLabel.title = h; }
    st.mode.value = want;
    if (document.activeElement !== st.project) st.project.value = m.project || "";
    st.all.disabled = sc.length < 2; st.all.style.opacity = sc.length < 2 ? "0.5" : "1";
    st.all.style.display = sc.length < 2 ? "none" : "";   // v1153 (Frank: "what is render all scenes? do we have scenes?"): shown with 2+ scenes only
    st.all.textContent = nTick < sc.length ? "render ticked scenes (" + nTick + ")" : "render all scenes";   // v1163
    if (sc.length > 1 && nTick === 0) { st.all.disabled = true; st.all.style.opacity = "0.5"; }
    const ok = !!(st.plan && st.plan.plan && st.plan.plan.complete);
    for (const b of [st.join, st.exactBtn]) if (b) { b.disabled = !ok; b.style.opacity = ok ? "1" : "0.5"; }
    if (st.cutsBtn) { const P2 = st.plan && st.plan.plan; const diff = !!(P2 && !st.plan.whole && P2.place_cuts.length === P2.cuts_prompt.length && P2.place_cuts.some((f, j) => f !== P2.cuts_prompt[j]));
        st.cutsBtn.disabled = !diff; st.cutsBtn.style.opacity = diff ? "1" : "0.5"; }
    if (st.tail) { const P3 = st.plan && st.plan.plan; const lp = P3 && P3.pieces[P3.pieces.length - 1];
        st.tail.checked = !!(lp && P3.frames[String(lp.k)] != null && lp.b >= P3.frames[String(lp.k)]); }
    if (st.player) _playerLabel(node);
    if (st.prevBtn) st.prevBtn.textContent = (prevOpen(node) ? "\u25be" : "\u25b8") + " prompt the run will produce";
    _layout(node);
}

function _field(node) {
    if (_w(node, FIELD) || typeof node.addDOMWidget !== "function") return;
    const root = _el("div", "width:100%;height:" + fieldH(node, _model(node)) + "px;box-sizing:border-box;display:flex;flex-direction:column;gap:" + ROOT_GAP + "px;font-family:sans-serif;overflow:hidden;");
    const stop = (e) => e.stopPropagation();
    root.addEventListener("pointerdown", stop); root.addEventListener("wheel", stop);
    // v1150: the CLIP Text Encode's segments may have changed -- the next look at the timeline asks again
    root.addEventListener("pointerenter", () => { const st = node._plsTl; if (st && JSON.stringify(fillOf(node)) !== st.fillSig) _preview(node); });

    // v1150: the TOP BAR (mode, length, project, render choice, the runs) and the HEAD ROW (the old triggers /
    // style fields + "-> CTE", shown only while they hold text -- their place is the CLIP Text Encode now)
    const topBar = _el("div", "display:flex;gap:8px;align-items:center;font-size:12px;color:#aaa;flex:0 0 auto;overflow:hidden;");
    topBar.dataset.pls = "topbar";
    const headRow = _el("div", "display:flex;gap:6px;align-items:flex-start;font-size:12px;color:#aaa;line-height:19px;flex:0 0 auto;");
    headRow.dataset.pls = "headrow";
    const len = _el("input", INPUT + "width:68px;"); len.type = "number"; len.step = "0.25"; len.min = "0.5"; len.max = "30";
    const lenInfo = _el("span", "white-space:nowrap;");
    const HEADF = "flex:1 1 0;min-width:80px;height:" + HEAD_LINE_H + "px;resize:none;overflow:hidden;line-height:15px;white-space:pre-wrap;overflow-wrap:anywhere;font-family:sans-serif;";
    const style = _el("textarea", INPUT + HEADF); style.rows = 1; style.placeholder = "the look -- // LOOK at the head of [Shot 1]";
    // v1050 (Frank 01.10.: "two separate areas, one for the triggers, one for the
    // style" -- consistency over clips rendered one by one): both are the head
    // of [Shot 1] in EVERY run, each under its own // rubric
    const trig = _el("textarea", INPUT + HEADF + "color:#c5e1a5;"); trig.rows = 1; trig.placeholder = "LoRA triggers -- // TRIGGERS at the head of [Shot 1]";
    trig.dataset.pls = "triggers";
    trig.title = "the LoRA trigger words of this timeline: the head of [Shot 1] in EVERY run (// TRIGGERS), so scenes rendered one by one share them.\n"
        + "Name each one again at its place in a shot's text, next to its plain word (40 sections 9 + 10)";
    style.title = "the look of this timeline (// LOOK): after the triggers, at the head of [Shot 1] in every run";
    // v1051: which H3 model the run feeds -- decides what a first/last-frame anchor is
    const tgt = _el("select", INPUT + "width:auto;");
    tgt.dataset.pls = "target";
    for (const [v, t] of TARGETS) { const o = document.createElement("option"); o.value = v; o.textContent = t; tgt.appendChild(o); }
    tgt.title = "fl2va: first/last-frame anchors go to MiniMax Keyframes as <Picture n>, with the official alignment line.\n"
        + "ref2va: the first-frame anchor goes to MiniMax Reference first_frame (pinned, no alignment line); ref2va has no last frame.";
    tgt.addEventListener("change", () => { const mm = _model(node); mm.target = tgt.value; _save(node, mm); _render(node); });
    tgt.style.display = "none";            // v1150: the mode buttons set it (Reference = ref2va); kept as the target's carrier
    const modeSeg = _el("div", "display:flex;border:1px solid #4e4e5a;border-radius:5px;overflow:hidden;flex:0 0 auto;");
    modeSeg.dataset.pls = "modeseg";
    const modeBtns = {};
    for (const [k, label] of MODES) {
        const b = _el("button", "background:#24242c;color:#bbb;border:0;border-right:1px solid #4e4e5a;font-size:12px;padding:6px 12px;cursor:pointer;white-space:nowrap;", label);
        b.dataset.mode = k;
        b.title = k === "text" ? "prompt only: shots, cuts, sound -- no pictures (fl2va, no anchors)"
            : k === "keys" ? "pictures on the time axis: first / last frame (MiniMax Keyframes) and guides (Continue) -- fl2va"
            : "reference pictures (the board, image_1..9) and a pinned first frame -- ref2va (MiniMax Reference)";
        b.addEventListener("pointerdown", (e) => e.stopPropagation());
        b.addEventListener("click", (e) => { e.stopPropagation(); _setMode(node, k); });
        modeBtns[k] = b; modeSeg.appendChild(b);
    }
    const toCte = _el("button", BTN + "color:#c5e1a5;border-color:#5a7a3a;flex:0 0 auto;", "\u2192 CTE");
    toCte.dataset.pls = "tocte";
    toCte.title = "move what never changes -- the triggers, the look and every rubric of Shot 1 that does not steer the shot "
        + "(SCENE, LIGHT, SUBJECT, HAIR, WARDROBE, FACE, ...) -- into the CLIP Text Encode wired to 'prompt'. Shot 1 keeps "
        + "{pos_1}, {pos_2} ... where they stood; the CTE fills them (external_mode fill {pos_N}): the text the encoder sees stays "
        + "byte for byte the same";
    toCte.addEventListener("pointerdown", (e) => e.stopPropagation());
    toCte.addEventListener("click", (e) => { e.stopPropagation(); _toCte(node); });
    headRow.append(_el("span", "color:#c5e1a5;", "triggers"), trig, _el("span", "margin-left:6px", "style"), style, toCte);
    // v1150: the REFS row -- the wired pictures as thumbnails; the pointer over one shows it big, inside the node
    const refsRow = _el("div", "display:flex;gap:8px;align-items:center;font-size:12px;color:#aaa;flex:0 0 auto;overflow:hidden;");
    refsRow.dataset.pls = "refs";
    const pop = _el("div", "position:absolute;display:none;z-index:40;border:2px solid #f0c87a;border-radius:6px;background:#000;box-shadow:0 6px 20px #000a;pointer-events:none;overflow:hidden;");
    pop.dataset.pls = "refpop";
    const popImg = _el("img", "display:block;max-width:420px;max-height:300px;");
    pop.appendChild(popImg);
    // v1064: one line of words, wrapped -- Enter ends the edit, a pasted line break is a space
    for (const el of [trig, style]) {
        el.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); el.blur(); } });
        el.addEventListener("input", () => {
            if (/[\r\n]/.test(el.value)) { const at = el.selectionStart; el.value = el.value.replace(/\r?\n|\r/g, " "); try { el.setSelectionRange(at, at); } catch (x) { /* view only */ } }
            _fitHead(node);
        });
    }
    trig.addEventListener("input", () => { const mm = _model(node); mm.triggers = trig.value; _save(node, mm, false); });
    trig.addEventListener("change", () => _commit(node));
    len.addEventListener("change", () => {
        const mm = _model(node); setLength(mm, parseFloat(len.value));
        _save(node, mm); _render(node); _stack(node);
    });
    style.addEventListener("input", () => { const mm = _model(node); mm.style = style.value; _save(node, mm, false); });
    style.addEventListener("change", () => _commit(node));

    // v1045 (Frank 30.09., Premiere as the model): every row = a HEAD column
    // on the left + its lane; the time axis starts after the heads on every
    // row, so ruler, scenes, shots, tracks and anchors stay flush
    const lane = "position:relative;flex:1 1 auto;min-width:0;background:#161616;";
    const ruler = _el("div", lane + "height:" + RULER_H + "px;");
    const scenes = _el("div", lane + "height:" + SCENES_H + "px;");
    const shots = _el("div", lane + "height:" + SHOTS_H + "px;");
    const anchors = _el("div", lane + "height:" + ANCH_H + "px;");
    // v1036: the clip lane under the shots, the playhead over every lane
    const clips = _el("div", lane + "height:" + LANE_CLIP_H + "px;background:#121212;overflow:hidden;");
    // v1037: the audio lane under the clips (the waveforms, placed like the pictures)
    const audio = _el("div", lane + "height:" + LANE_AUDIO_H + "px;background:#101418;overflow:hidden;");
    const lanes = _el("div", "position:relative;display:flex;flex-direction:column;gap:4px;flex:0 0 auto;");
    // the overlay over the lanes' part (right of the heads): playhead, cuts, grips
    const tl = _el("div", "position:absolute;top:0;bottom:0;right:0;left:" + GUT_W + "px;pointer-events:none;z-index:6;");
    tl.dataset.pls = "timeaxis";
    const head = _el("div", "position:absolute;top:0;bottom:0;width:0;border-left:2px solid #ffb300;z-index:9;pointer-events:none;display:none;left:0;");
    // v1038: the playhead's grip -- a flag on the ruler (drag = scrub)
    const headGrip = _el("div", "position:absolute;top:0;height:16px;width:14px;margin-left:-6px;z-index:10;cursor:ew-resize;display:none;left:0;pointer-events:auto;");
    headGrip.appendChild(_el("div", "position:absolute;left:0;top:0;width:0;height:0;border-left:7px solid transparent;border-right:7px solid transparent;border-top:11px solid #ffb300;pointer-events:none;"));
    headGrip.dataset.pls = "headgrip";
    headGrip.title = "the playhead -- drag: the player follows; it snaps to the cuts (Shift: free)";
    headGrip.addEventListener("pointerdown", (ev) => _scrub(node, ev));
    // v1039: the cuts -- one layer over the picture and the sound lane
    const cuts = _el("div", "position:absolute;left:0;right:0;top:" + LANES_TOP + "px;height:" + (LANE_CLIP_H + LANE_GAP + LANE_AUDIO_H) + "px;pointer-events:none;z-index:7;");
    cuts.dataset.pls = "cutlayer";
    tl.append(head, headGrip, cuts);
    // v1044/v1045: the video tracks above the render's V1 (top first), the
    // audio tracks below its A1 (A2, A3, ...), the render's two heads
    const col = "display:none;flex-direction:column;gap:" + LANE_GAP + "px;flex:0 0 auto;";
    const above = _el("div", col), below = _el("div", col);
    above.dataset.pls = "above"; below.dataset.pls = "below";
    const gut = (text) => { const g = _el("div", GUT + "color:#8a8a8a;font-size:10px;padding-left:8px;", text || ""); return g; };
    const tc = gut("00:00:00:00"); tc.style.cssText += "font-family:monospace;font-size:11px;color:#5aa9ff;";
    tc.dataset.pls = "timecode"; tc.title = "the playhead (hours:minutes:seconds:frames at 24 fps)";
    const v1Head = _el("div", GUT), a1Head = _el("div", GUT);
    // the prominent way to a new track: a row of its own, a button in the head
    // and a lane you can drop a clip on (it opens the track AND loads the clip)
    const newLane = _el("div", lane + "height:" + NEWROW_H + "px;background:transparent;border:1px dashed #3a3a3a;box-sizing:border-box;display:flex;align-items:center;padding-left:10px;font-size:10px;color:#6f6f6f;",
        "drop a video here -> a new track (upper half: copy into input/cine_media, lower half: read it where it lies)");
    newLane.dataset.pls = "newtrack";
    const addBtn = _el("button", "margin-left:6px;background:#1f3550;color:#9fc8ff;border:1px solid #3d6ea8;border-radius:3px;font-size:11px;font-weight:bold;padding:1px 10px;cursor:pointer;", "+ new track");
    addBtn.dataset.pls = "addlayer";
    addBtn.title = "a new EMPTY track on top (V + A) -- drop a clip on it, or use its \u2026 button; upper tracks win";
    addBtn.addEventListener("pointerdown", (e) => e.stopPropagation());
    addBtn.addEventListener("click", (e) => { e.stopPropagation(); const ed = editOf(node), id = newLayerId(ed); node._plsTl.active = id; _seedTrackH(node, id); _saveEdit(node, addLayer(ed, id)); });
    const newHead = gut(); newHead.style.paddingLeft = "0"; newHead.appendChild(addBtn);
    _dropZone(node, newLane, (dt, zone, at) => _dropNewTrack(node, dt, zone, at));
    const anchorsRow = _row(gut("Anchors"), anchors);   // v1150: not in Text mode (no pictures there)
    anchorsRow.dataset.pls = "anchorsrow";
    const v1Row = _row(v1Head, clips); v1Row.style.position = "relative"; v1Row.dataset.pls = "v1row";
    lanes.append(_row(tc, ruler), _row(gut("Scenes"), scenes), _row(gut("Shots"), shots), _row(newHead, newLane), above,
        v1Row, _row(a1Head, audio), below, anchorsRow, tl);
    ruler.style.cursor = "ew-resize";
    ruler.title = "click or drag: the playhead follows (it snaps to the cuts; Shift: free)";
    ruler.addEventListener("pointerdown", (ev) => _scrub(node, ev));

    const bar = _el("div", "display:flex;gap:4px;align-items:center;");
    const src = _el("select", INPUT);
    src.addEventListener("pointerdown", () => { try { _fillSources(node); } catch (e) { /* view only */ } });
    src.addEventListener("focus", () => { try { _fillSources(node); } catch (e) { /* view only */ } });
    // v1030: drag snap -- a view setting in node.properties, never in the JSON
    const snapSel = _el("select", INPUT);
    snapSel.title = "where a dragged cut or anchor lands: every frame (1/24 s), or 0.25 s / 0.5 s steps";
    for (const [v, t] of [[0, "drag: frame"], [0.25, "drag: 0.25 s"], [0.5, "drag: 0.5 s"]]) { const o = document.createElement("option"); o.value = String(v); o.textContent = t; snapSel.appendChild(o); }
    snapSel.value = String(snapStep(node));
    snapSel.addEventListener("change", () => {
        node.properties = node.properties || {};
        node.properties.pls_tl_snap = SNAP_STEPS.includes(Number(snapSel.value)) ? Number(snapSel.value) : 0;
        node.setDirtyCanvas?.(true, true);
    });
    // v1039: a legend is ONE line (the rest in its tooltip) -- wrapped, it
    // pushed the node's rows past its frame on a narrow node
    const _legend = (t) => {
        const e = _el("span", "margin-left:auto;font-size:10px;color:#777;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0;flex:0 1 auto;", t);
        e.title = t;
        return e;
    };
    const mk = (label, fn, title) => { const b = _el("button", BTN, label); b.title = title || ""; b.addEventListener("click", (e) => { e.stopPropagation(); fn(); }); return b; };
    const anchorBtns = _el("span", "display:inline-flex;gap:6px;align-items:center;");
    anchorBtns.dataset.pls = "anchorbtns";
    bar.append(
        mk("+ shot", () => { const mm = _model(node); const i = (node._plsTl.sel && node._plsTl.sel.kind === "shot") ? node._plsTl.sel.i : mm.shots.length - 1; node._plsTl.sel = { kind: "shot", i: splitShot(mm, i) }; _save(node, mm); _render(node); }, "divide the selected shot in two (the prompt track; the film is cut with \u2702 cut)"),
        mk("- shot", () => { const mm = _model(node); const s = node._plsTl.sel; if (s && s.kind === "shot") { node._plsTl.sel = { kind: "shot", i: removeShot(mm, s.i) }; _save(node, mm); _render(node); _stack(node); } }, "remove the selected shot"),
        _el("span", "width:10px"),
        anchorBtns);
    // v1150: the anchor controls only where pictures belong (Keyframes / Reference)
    anchorBtns.append(mk("+ anchor", () => { const mm = _model(node); mm.anchors.push({ src: src.value || "image_1", at: freeSpot(mm) }); node._plsTl.sel = { kind: "anchor", i: mm.anchors.length - 1 }; _save(node, mm); _render(node); }, "first frame, else last frame, else the middle"),
        src,
        mk("- anchor", () => { const mm = _model(node); const s = node._plsTl.sel; if (s && s.kind === "anchor") { mm.anchors.splice(s.i, 1); node._plsTl.sel = { kind: "shot", i: 0 }; _save(node, mm); _render(node); } }, "remove the selected anchor"));
    bar.append(_el("span", "width:10px"),
        _legend("green = keyframe  blue = guide  teal = scene cut  |  double-click = type it  |  Ctrl+Z on the canvas = undo"));

    // v1150: the selected shot is the LEFT column of the work area: its rows, its text (TEXT_H), the sound under it
    const edit = _el("div", "display:flex;flex-direction:column;gap:4px;height:" + WORK_H + "px;flex:0 0 auto;overflow:hidden;");
    // v1153 (Frank 17.10.: "the box must be marked as the audio prompt -- one only sees it while it is empty"):
    // the sound column, each field under its name, in the room beside the monitor
    const sndRow = _el("div", "display:flex;flex-direction:column;gap:4px;flex:0 1 " + SND_W + "px;width:" + SND_W + "px;min-width:" + SND_MIN_W + "px;");
    sndRow.dataset.pls = "soundcol";
    const sndLab = _el("div", "font-size:11px;color:#80cbc4;font-weight:bold;flex:0 0 auto;", "overall_soundscape");
    sndLab.title = "the clip's sound: ambience and action sounds -- no dialogue, no diegetic music (H3 invents the sound when this is empty)";
    const musLab = _el("div", "font-size:11px;color:#80cbc4;font-weight:bold;flex:0 0 auto;", "non_diegetic_music");
    musLab.title = "music laid under the clip (not heard by the people in it); N/A = none";
    const sound = _el("textarea", INPUT + "flex:3 1 0;min-height:60px;resize:none;"); sound.placeholder = "ambience and action sounds, no dialogue, no diegetic music";
    const music = _el("textarea", INPUT + "flex:1 1 0;min-height:40px;resize:none;"); music.placeholder = "N/A";
    sound.addEventListener("input", () => { const mm = _model(node); mm.soundscape = sound.value; _save(node, mm, false); });
    music.addEventListener("input", () => { const mm = _model(node); mm.music = music.value; _save(node, mm, false); });
    sound.addEventListener("change", () => _commit(node)); music.addEventListener("change", () => _commit(node));
    sndRow.append(sndLab, sound, musLab, music);

    // v1028: the cutting room row -- render choice, project, the runs, the join
    const mode = _el("select", INPUT + "max-width:260px;");
    mode.title = "what the NEXT run renders -- your choice: the whole timeline in one run, or one scene alone";
    mode.addEventListener("change", () => {
        const mm = _model(node);
        if (mode.value === "whole") mm.render = "whole";
        else if (mode.value === "all") mm.render = "all";
        else { mm.render = "scene"; mm.active = parseInt(mode.value.split(":")[1], 10) || 1; }
        _save(node, mm); _render(node); _stack(node);      // v1037: whole <-> scenes changes the clip lane
    });
    const project = _el("input", INPUT + "width:120px;"); project.placeholder = "cine";
    project.title = "The project's name. Its stack LIST and anchor frames live in input/cine_media/<project>/stack/; an export goes to the day folder (output/<yyyy-MM-dd>/PH_<hhmmss>_<project>_film). The clips themselves stay where your Save node writes them: every video a run saves joins the stack by path (no copy, no second folder). Wiring clip_prefix to a Save's filename_prefix is optional and names that Save's files <day>/PH_<hhmmss>_<project>_whole | _scene_k.";
    project.addEventListener("change", () => { const mm = _model(node); mm.project = project.value; _save(node, mm); _render(node); _stack(node); });
    const target = _el("span", "font-size:11px;color:#888;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0;flex:0 1 auto;");
    const run = mk("\u25b6 Render", () => { _queue().catch((e) => { node._plsTl.state.textContent = "queue failed: " + e.message; }); }, "queue one run of what the choice says");
    run.style.cssText = BTN_GO; run.dataset.pls = "render";
    const all = mk("render all scenes", () => _renderAll(node), "queue one run per scene, in order (each run's saved clip joins the stack of its scene, wherever your Save writes it)");
    const join = mk("export", () => _export(node, false), "the film from the clip lane (clips, takes, cuts, transitions) -> the day folder, PH_<hhmmss>_<project>_film (ffmpeg)");
    const state = _el("div", "font-size:10px;color:#f0c87a;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-height:12px;flex:0 0 auto;");
    state.dataset.pls = "stateline";
    state.addEventListener("mouseenter", () => { state.title = state.textContent; });   // v1153: the whole line as the tooltip (it is cut at the width)
    // v1036: the player (the old stack tiles now live on the clip lane)
    // v1150: the monitor is the right column of the work area -- a 16:9 picture of MON_H px, the transport under it
    const stack = _el("div", "display:flex;flex-direction:column;gap:6px;flex:0 1 " + MON_W + "px;width:" + MON_W + "px;align-items:stretch;min-width:" + MON_MIN_W + "px;");
    // v1163: a box of MON_AREA_H px; the canvas inside takes the clip's aspect (_monitorFit), centred
    const monBox = _el("div", "height:" + MON_AREA_H + "px;flex:0 0 auto;display:flex;align-items:center;justify-content:center;background:#0b0b0b;border-radius:4px;overflow:hidden;min-width:0;");
    monBox.dataset.pls = "monitor-box";
    const canvas = _el("canvas", "height:" + MON_H + "px;background:#000;border-radius:4px;flex:0 0 auto;display:block;max-width:100%;");
    canvas.width = 480; canvas.height = 270;
    monBox.append(canvas);
    const ctl = _el("div", "display:flex;gap:6px;align-items:center;font-size:12px;color:#aaa;min-width:0;flex:0 0 auto;overflow:hidden;");
    const play = mk("\u25b6 play", () => _playerToggle(node), "play / pause the film as the clip lane cuts it (live mix; 'exact preview' = ffmpeg)");
    play.style.alignSelf = "flex-start"; play.style.minWidth = "84px";
    const ptime = _el("span", "font-family:monospace;color:#ddd;", "0.00 / 0.00 s");
    const exactBtn = mk("exact preview", () => _export(node, true), "ffmpeg renders the film small into temp/ -- exactly what export makes");
    const live = mk("live", () => { const s2 = node._plsTl; s2.exact = null; _playerLoad(node); _playerDraw(node); }, "back to the live mix of the clip lane");
    const cutsBtn = mk("cuts -> prompt", () => { const s2 = node._plsTl; if (!s2.plan) return; const mm = cutsIntoPrompt(_model(node), s2.plan); const e = editOf(node); for (const k of ["cuts", "pieces", "pieces_n"]) delete e[k];
        node.properties = node.properties || {}; node.properties[EDIT_KEY] = e; _save(node, mm); _render(node); _stack(node); }, "the film's cuts become the prompt's scene cuts (for the next run)");
    const resetBtn = mk("reset cuts", () => { _saveEdit(node, _clearCuts(node, editOf(node))); }, "the film cuts where the prompt cuts again: splits, trims and speeds go (takes stay)");
    // v1219: everything back to its scene -- V1 and every track's scene clips (the order the scenes say)
    const snapBtn = mk("\u21e4 snap to scenes", () => {
        if (window.confirm("Snap every clip back to its scene? V1 cuts where the prompt cuts again, and every track's scene clip goes to the start of its scene (repeated heads left out). Clips without a scene number stay where they are; locked tracks too.")) _snap(node, "all");
    }, "every clip back where its scene starts -- V1 and the tracks (scene_<k> in the file name says where it belongs)");
    snapBtn.dataset.pls = "snapall";
    const tailLab = _el("label", "display:flex;gap:3px;align-items:center;cursor:pointer;");
    const tail = document.createElement("input"); tail.type = "checkbox";
    tail.addEventListener("change", () => { const s2 = node._plsTl, D = s2.plan; if (!D || !D.plan) return;
        const pcs = piecesOf(D), j = pcs.length - 1, fK = (k) => framesOf(D, k);
        // v1082: no clip at the last place = no tail to keep (Infinity went in as 1e9 frames)
        if (tail.checked) { const fb = fK(pcs[j].k); if (!Number.isFinite(fb)) { tail.checked = false; s2.state.textContent = "keep tail: the last place has no clip yet"; return; } pcs[j].b = fb; _savePieces(node, pcs); }
        else _savePieces(node, trimEnd(pcs, Math.round(Number(_model(node).seconds) * FPS), fK)); });
    tailLab.append(tail, _el("span", "", "keep tail"));
    tailLab.title = "the last clip runs out to its own end (the film gets longer than the asked length)";
    // v1037: the razor and the plaster work at the playhead
    // v1038: ONE word for it -- "cut" (the button, the lines, the chips)
    const splitBtn = mk("\u2702 cut", () => _split(node), "cut at the playhead -- picture and sound");
    // v1044: the stack -- a new empty layer on top, and "delete" for the selected piece
    const delBtn = mk("\u{1F5D1} delete", () => _deleteSel(node), "delete the selected piece -- it becomes empty, the layers below show through");
    delBtn.dataset.pls = "delpiece";
    // v1047: the film's length (the edit span) has its own row now -- FILM,
    // over the tracks, with the zoom and the view's scroll bar
    const span = _el("input", INPUT + "width:72px;font-weight:bold;font-size:13px;color:#ffd27a;text-align:center;"); span.type = "text";
    span.dataset.pls = "span";
    span.title = "the FILM's length -- the editing room: what is played and exported. 5:00, 1:30.5, 300 or 2m; empty = the render's length.\n"
        + "Longer than the render: new renders and drops land at the front, move them where they belong. Tracks past the end are dimmed, not exported";
    span.addEventListener("keydown", (e) => { e.stopPropagation(); if (e.key === "Enter") span.blur(); if (e.key === "Escape") { span.value = ""; span.blur(); } });
    span.addEventListener("change", () => _setFilmLen(node, span.value));
    const preset = _el("select", INPUT);
    preset.dataset.pls = "filmpreset";
    preset.title = "a film length to start from";
    for (const v of [""].concat(FILM_PRESETS)) { const o = document.createElement("option"); o.value = String(v); o.textContent = v === "" ? "\u25be" : fmtLen(v); preset.appendChild(o); }
    preset.addEventListener("change", () => { if (preset.value) _setFilmLen(node, preset.value); preset.value = ""; });
    const filmInfo = _el("span", "font-size:10px;color:#8a8a8a;white-space:nowrap;");
    const zBtn = (t, title, fn, tag) => { const b = _el("button", BTN + "min-width:24px;padding:0 6px;", t); b.title = title; b.dataset.pls = tag;
        b.addEventListener("pointerdown", (e) => e.stopPropagation()); b.addEventListener("click", (e) => { e.stopPropagation(); fn(); }); return b; };
    const zOut = zBtn("\u2212", "zoom out (Ctrl + wheel over the tracks)", () => _zoomBy(node, 0.5), "zoomout");
    const zFit = zBtn("fit", "the whole film in view", () => _setZoom(node, 1), "zoomfit");
    const zIn = zBtn("+", "zoom in, around the playhead (Ctrl + wheel over the tracks: around the mouse)", () => _zoomBy(node, 2), "zoomin");
    const viewInfo = _el("span", "font-size:10px;color:#8a8a8a;white-space:nowrap;min-width:92px;font-family:monospace;");
    viewInfo.dataset.pls = "viewinfo";
    // v1153 (Frank: "a zoom slider"): the zoom as a slider, 1 .. the 2-second window, around the playhead
    const zSl = _el("input", "width:90px;height:12px;margin:0;accent-color:#ffd27a;cursor:pointer;"); zSl.type = "range"; zSl.min = "0"; zSl.max = "100"; zSl.step = "1"; zSl.value = "0";
    zSl.dataset.pls = "zoomslider"; zSl.title = "zoom: left = the whole film, right = a 2-second window (around the playhead)";
    zSl.addEventListener("pointerdown", (e) => e.stopPropagation());
    zSl.addEventListener("input", () => { const s2 = node._plsTl; const zmax = Math.max(1, (s2.ax || 0) / 2); _setZoom(node, Math.pow(zmax, Number(zSl.value) / 100)); });
    // the view's scroll bar: the thumb is the part of the film in view (drag it)
    const nav = _el("div", "position:relative;flex:1 1 auto;min-width:60px;height:12px;background:#121212;border:1px solid #2c2c2c;border-radius:6px;box-sizing:border-box;cursor:pointer;");
    nav.dataset.pls = "nav";
    nav.title = "the film -- the bright part is in view: drag it, or click where to look (Shift + wheel over the tracks scrolls too)";
    const thumb = _el("div", "position:absolute;top:1px;bottom:1px;left:0;width:100%;background:#4a5a70;border-radius:5px;cursor:grab;");
    thumb.dataset.pls = "navthumb";
    nav.appendChild(thumb);
    nav.addEventListener("pointerdown", (ev) => _navDown(node, ev));
    const film = _el("div", "display:flex;gap:6px;align-items:center;font-size:11px;color:#aaa;height:" + FILM_H + "px;flex:0 0 auto;"
        + "background:#1d1a14;border:1px solid #4a3d22;border-radius:3px;padding:0 6px;box-sizing:border-box;");
    film.dataset.pls = "film";
    film.append(_el("span", "font-weight:bold;color:#ffd27a;letter-spacing:1px;", "FILM"), span, preset, filmInfo,
        _el("span", "width:1px;align-self:stretch;background:#4a3d22;margin:0 4px;"), _el("span", "", "zoom"), zOut, zSl, zIn, zFit, viewInfo, nav);
    const healBtn = mk("\u271a heal", () => _heal(node), "heal the cut at the playhead");
    splitBtn.dataset.pls = "cut"; healBtn.dataset.pls = "heal";
    const linkBtn = mk("A/V linked", () => { node.properties = node.properties || {}; node.properties[LINK_KEY] = !isLinked(node);
        node.setDirtyCanvas?.(true, true); _render(node); }, "");
    linkBtn.dataset.pls = "link";
    // v1153: the tracks' order -- pairs (default) or by type
    const typeBtn = mk("V/A by type", () => { node.properties = node.properties || {}; node.properties[BYTYPE_KEY] = !byType(node); node.setDirtyCanvas?.(true, true); _render(node); },
        "off (default): every pair Vn/An together, its clamp at the left edge\non: every picture track above V1, every sound track below A1 (a view, the cut is the same)");
    typeBtn.dataset.pls = "bytype";
    // v1046 (Frank 30.09.: "the preview up, the tracks below -- with 20 tracks
    // the preview lies in nirvana; the cut buttons away from the preview, two
    // different areas"): the MONITOR (picture + transport) sits at the top; the
    // editing tools are the timeline's own TOOLBAR, right above the tracks
    const r1 = _el("div", "display:flex;gap:6px;align-items:center;"); r1.append(play, ptime);
    const r2 = _el("div", "display:flex;gap:6px;align-items:center;margin-left:auto;"); r2.append(exactBtn, live, join);
    ctl.append(r1, r2);
    stack.append(monBox, ctl);
    stack.dataset.pls = "monitor";
    // the E4 switches (Frank 30.09.): new renders into a new track; restore the
    // tracks when the workflow is opened again; the render's older takes as tracks
    const chk = (label, title, on, fn, tag) => {
        const l = _el("label", "display:flex;gap:3px;align-items:center;cursor:pointer;white-space:nowrap;");
        const c = document.createElement("input"); c.type = "checkbox"; c.checked = !!on; c.dataset.pls = tag;
        c.addEventListener("pointerdown", (e) => e.stopPropagation());
        c.addEventListener("change", () => fn(c.checked));
        l.append(c, _el("span", "", label)); l.title = title;
        return [l, c];
    };
    // v1163 (Frank 18.10.: "nicht new Track, sondern Replace"): the switch as an explicit choice, same two modes
    const autoLab = _el("label", "display:flex;gap:3px;align-items:center;white-space:nowrap;");
    const autoChk = document.createElement("select"); autoChk.dataset.pls = "auto";
    autoChk.style.cssText = "background:#222;color:#ddd;border:1px solid #444;border-radius:3px;font-size:11px;padding:0 2px;";
    for (const [v, lab] of AUTO_CHOICES) { const o = document.createElement("option"); o.value = v; o.textContent = lab; autoChk.appendChild(o); }
    autoChk.addEventListener("pointerdown", (e) => e.stopPropagation());
    autoChk.addEventListener("change", () => _setAuto(node, autoChk.value));
    autoLab.append(_el("span", "", "new render \u2192"), autoChk);
    autoLab.title = "new track (default): the first run of a place lands in V1, every run after it in a NEW track on top -- a basin of takes to cut from (V1 keeps the clip it shows now)\n"
        + "one track per render: like new track, but every scene of ONE render (one Run press over all scenes) lands on the SAME new track, each at its scene's place -- the render stays together\n"
        + "replace: a new run REPLACES the clip of its scene in V1 -- V1 shows the newest run of every scene (the old files stay on disk; tracks stay as they are)";
    const [restLab, restChk] = chk("restore on open", "on: the tracks and cuts come back when the workflow is opened again\noff: opened again, the timeline starts empty (V1 shows the newest render; the files stay on disk)",
        true, (v) => { node.properties = node.properties || {}; node.properties[RESTORE_KEY] = !!v; node.setDirtyCanvas?.(true, true); }, "restore");
    const takesBtn = mk("\u29c9 takes \u2192 tracks", () => _takesToTracks(node), "the OLDER renders as tracks -- ONE track per render (its scenes at their places), the newest on top; "
        + "per run only the clip it picked (backup / raw saves stay out); what V1 shows or a track has already is skipped");
    takesBtn.dataset.pls = "takes";
    // v1198: every track off in one step
    const clearBtn = mk("\u2715 tracks", () => _clearTracks(node), "remove ALL tracks above V1 in one step (asks first; Ctrl+Z brings them back) -- V1 stays, the files stay on disk");
    clearBtn.dataset.pls = "cleartracks";
    const sep = () => _el("span", "width:1px;align-self:stretch;background:#3a3a3a;margin:0 4px;");
    const tools = _el("div", "display:flex;gap:6px;align-items:center;flex-wrap:wrap;align-content:center;font-size:12px;color:#aaa;height:" + TOOLS_H + "px;flex:0 0 auto;overflow:hidden;"
        + "background:#1a1a1a;border:1px solid #2c2c2c;border-radius:3px;padding:3px 6px;box-sizing:border-box;");
    tools.dataset.pls = "tools";
    tools.title = "drag the playhead (it snaps to cuts); \u2702 cut / heal at it on the ACTIVE track; a cut is one line through picture and sound: drag = move it, "
        + "its amber \u25c6 = the speed of the clip before; drag a clip on V1 = slip, on the other tracks = move (Alt = slip); A/V unlinked or Alt+drag: picture or "
        + "sound part alone (J-/L-cut); the divider under a lane = its height; the prompt stays until 'cuts -> prompt'";
    tools.append(splitBtn, healBtn, delBtn, linkBtn, typeBtn, snapSel, sep(), resetBtn, snapBtn, cutsBtn, tailLab,
        sep(), autoLab, restLab, takesBtn, clearBtn);
    // v1034: this line says what the next Run does (stackHint) -- same height as before
    const stackLabel = _el("div", "font-size:10px;color:#9e9e9e;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:0 0 auto;min-height:12px;", "stack: the rendered scenes of the project's stack (hover plays)");

    // a FIXED height: Nodes 2.0 does not let a textarea take the flex space
    // (measured 26.09.: it collapsed to two lines there, classic was fine)
    // v1030: the height is FITTED to the text (_fitPreview), still in px
    const prev = _el("textarea", INPUT + "flex:0 0 auto;height:" + PREVIEW_DEF + "px;resize:none;overflow-y:hidden;font-family:monospace;font-size:11px;color:#cfd8dc;");
    prev.readOnly = true;
    prev.dataset.pls = "preview";
    // v1150: the STATUS row -- the check as chips, what the next run does, clip_prefix, the preview's switch
    const status = _el("div", "display:flex;gap:8px;align-items:center;font-size:11px;color:#9e9e9e;flex:0 0 auto;overflow:hidden;");
    status.dataset.pls = "status";
    const chips = _el("div", "display:flex;gap:6px;align-items:center;flex:0 0 auto;");
    chips.dataset.pls = "chips";
    const prevBtn = _el("button", BTN, "\u25b8 prompt the run will produce");
    prevBtn.dataset.pls = "prevtoggle";
    prevBtn.title = "open / close the whole prompt as the run encodes it (with the CLIP Text Encode's parts filled in) and the full check";
    prevBtn.addEventListener("pointerdown", (e) => e.stopPropagation());
    prevBtn.addEventListener("click", (e) => { e.stopPropagation(); node.properties = node.properties || {}; node.properties[PREV_OPEN_KEY] = !prevOpen(node);
        node.setDirtyCanvas?.(true, true); _render(node); if (prevOpen(node)) { _preview(node); _widthCheck(node, true); } });
    stackLabel.style.flex = "0 1 auto"; stackLabel.style.minWidth = "0";
    status.append(chips, prevBtn, stackLabel, target);
    // v1047: the lanes scroll sideways under their heads (zoom); the heads stick
    const scroller = _el("div", "position:relative;overflow:hidden;flex:0 0 auto;");
    scroller.dataset.pls = "scroller";
    scroller.appendChild(lanes);
    scroller._plsNode = node;          // the wheel: one capture listener on the document (_wheelHook)
    _wheelHook();
    topBar.append(modeSeg, tgt, _el("span", "margin-left:4px", "length"), len, _el("span", "", "s"), lenInfo,
        _el("span", "margin-left:8px", "project"), project, _el("span", "margin-left:8px", "render"), mode,
        _el("span", "flex:1 1 auto;"), all, run);
    const left = _el("div", "display:flex;flex-direction:column;gap:6px;flex:1 1 auto;min-width:" + LEFT_MIN_W + "px;");
    left.dataset.pls = "inspector";
    left.append(edit);
    const work = _el("div", "display:flex;gap:10px;flex:0 0 auto;align-items:stretch;height:" + WORK_H + "px;");
    work.dataset.pls = "work";
    work.append(left, sndRow, stack);
    state.style.height = STATE_H + "px";
    bar.style.flex = "0 0 auto";
    root.style.position = "relative";
    root.append(topBar, headRow, refsRow, work, state, tools, film, scroller, bar, status, prev, pop);
    try {
        const w = node.addDOMWidget(FIELD, "pls-timeline", root, { serialize: false });
        w.serialize = false;
        w.computeSize = (width) => { _widthCheck(node); return [width, fieldH(node)]; };
        // v1044: Delete / Backspace on a selected piece (the field has the focus
        // after a click on a piece) -- stopped here, never reaches the canvas
        root.tabIndex = -1; root.style.outline = "none";
        // v1198 (Frank 21.10.: Ctrl+Z / Ctrl+Y did nothing): the frontend skips undo / redo while an INPUT holds the
        // focus -- a ticked box, the zoom slider, a choice of the timeline kept it after the click. Such a control
        // gives the focus back once it has changed (text fields keep it: there Ctrl+Z is the text's own undo)
        root.addEventListener("change", (ev) => { const t = ev.target; if (blurAfterChange(t)) setTimeout(() => { try { t.blur(); } catch (x) { /* gone */ } }, 0); }, true);
        root.addEventListener("keydown", (ev) => {
            const t = ev.target && ev.target.tagName;
            if ((ev.key === "Delete" || ev.key === "Backspace") && t !== "INPUT" && t !== "TEXTAREA" && t !== "SELECT" && _deleteSel(node)) { ev.preventDefault(); ev.stopPropagation(); }
        });
        node._plsTl = { root, stackLabel, topBar, headRow, refsRow, work, status, chips, prevBtn, pop, popImg, modeBtns, modeSeg, toCte, m: _model(node),
            sec: { topbar: topBar, headrow: headRow, refs: refsRow, work, state, tools, film, scroller, bar, status, prev }, anchorsRow, anchorBtns, above, below, v1Row, tools, autoChk, restChk, takesBtn, v1Head, a1Head, tc, tl, newLane, addBtn, delBtn, span, filmInfo, viewInfo, zSl, nav, thumb, scroller, active: BASE_ID, recent: [], len, lenInfo, style, trig, tgt, ruler, scenes, shots, anchors, src, snapSel, edit, sound, music, prev, mode, project, target,
            all, join, state, stack, clips, audio, head, headGrip, cuts, lanes, divs: null, pct: null, cutCtx: null, plan: null, exact: null, ax: 0, exactBtn, cutsBtn, tail, splitBtn, healBtn, linkBtn, typeBtn, laneW: 0, scrubbing: false,
            player: { canvas, play, time: ptime, videos: new Map(), playing: false, t: 0, t0: 0, raf: 0, ar: 16 / 9, boxW: null, boxH: null }, monCol: stack, sel: { kind: "shot", i: 0 }, timer: null,
            io: null, prevH: PREVIEW_DEF, fitW: null, fitNodeW: null, fitQueued: false, bracket: false, bracketTimer: null, previewPending: false, fitting: false, lastDown: null, open: null, transSel: null };
        for (const [el, d] of [[state, "block"], [scroller, "block"], [prev, "block"]]) el._plsDisp = d;
        if (typeof IntersectionObserver === "function") {
            // v1157: the lanes too -- their film strips are drawn again once they come on screen with a width
            const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { _widthCheck(node, true); _stripCheck(node, true); } });
            io.observe(prev);
            io.observe(scroller);
            node._plsTl.io = io;
        }
    } catch (e) { /* a view must never break node creation */ }
}

app.registerExtension({
    name: "polyhedron.cine_timeline",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE) return;
        // v1163: the node's floor width -- LiteGraph clamps a drag against computeSize (no onResize hook)
        const _cs = nodeType.prototype.computeSize;
        nodeType.prototype.computeSize = function () {
            const size = _cs ? _cs.apply(this, arguments) : [TL_MIN_W, 80];
            size[0] = Math.max(size[0], TL_MIN_W);
            return size;
        };
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = created?.apply(this, arguments);
            setHidden(_w(this, "timeline"), true);
            _field(this);
            _render(this);
            // v1045: the track heads take GUT_W on the left -- a new node is wider
            if (this.size && this.size[0] < 680 + GUT_W) this.setSize([680 + GUT_W, this.size[1]]);
            refit(this);
            _preview(this);
            _stack(this);
            return r;
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = configured?.apply(this, arguments);
            // v1046 "restore on open" off: the FIRST load of this node in a page
            // session starts with an empty timeline (an undo later reloads the
            // graph in the same session -- that must never wipe the edit)
            try {
                // v1082: the wipe takes the tracks and cuts, not the settings
                // (the switch's explicit off and the film's span stay)
                // v1088 (audit B6): the memory is per WORKFLOW and node -- node ids are small
                // numbers per workflow, another workflow's timeline 7 is not this one
                const seenKey = seenKeyOf(this);
                if (this.properties && this.properties[RESTORE_KEY] === false && !_SEEN.has(seenKey)) this.properties[EDIT_KEY] = wipeEdit(this.properties[EDIT_KEY]);
                _SEEN.add(seenKey);
            } catch (e) { /* view only */ }
            setHidden(_w(this, "timeline"), true);
            requestAnimationFrame(() => requestAnimationFrame(() => { _render(this); _preview(this); _stack(this); }));
            return r;
        };
        const conn = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const r = conn?.apply(this, arguments);
            try { _render(this); _preview(this); } catch (e) { /* never throw from a link event */ }
            return r;
        };
        // v1031: the field comes on screen -> one width check (the first fit
        // of a node that was off screen when its text came). An
        // IntersectionObserver reports visibility, not size -- the fit sets
        // only a height, so it cannot start a loop (the ResizeObserver
        // lesson, v610). v1030 used the per-node draw hook for this, which
        // made the node count as hand-drawn (test_v921) -- it draws nothing.
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            try { this._plsTl && this._plsTl.io && this._plsTl.io.disconnect(); } catch (e) { /* gone anyway */ }
            try { if (this._plsTl) { _releaseVideos(this._plsTl); if (this._plsTl.zoomTimer) clearTimeout(this._plsTl.zoomTimer); } } catch (e) { /* gone anyway */ }
            // v1183: the pending preview and the undo bracket must not fire on a node that is gone
            try { if (this._plsTl) { clearTimeout(this._plsTl.timer); clearTimeout(this._plsTl.bracketTimer); this._plsTl.previewSeq = (this._plsTl.previewSeq || 0) + 1; } } catch (e) { /* gone anyway */ }
            return removed?.apply(this, arguments);
        };
        const exec = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            exec?.apply(this, arguments);
            const p = message && message.pls_timeline && message.pls_timeline[0];
            // a new storyboard was taken over by the run: it becomes the
            // timeline's state (the backend sends it once per new storyboard)
            if (p && p.adopt) {
                const w = _w(this, "timeline");
                if (w) { w.value = p.adopt; w.callback?.(w.value); }
                if (this._plsTl) this._plsTl.sel = { kind: "shot", i: 0 };
                try { _render(this); } catch (e) { /* the run's result stands */ }
            }
            if (p && this._plsTl) {
                this._plsTl.prev.style.color = "#cfd8dc"; this._plsTl.prev.value = p.prompt + "\n\n-- ran --\n" + p.info;
                _fitPreview(this);
                if (p.clip_prefix) this._plsTl.state.textContent = "ran: " + (p.scene ? "scene " + p.scene + " of " + p.scenes : "the whole timeline") + " -> output/" + String(p.clip_prefix).replace("%date:yyyy-MM-dd%", "<day>").replace("%date:hhmmss%", "<hhmmss>") + "_*.mp4";
            }
        };
    },
    async setup() {
        // v1163: Space = play / pause the selected timeline (one listener for the page, capture phase)
        if (!globalThis.__plsTlSpace && typeof document !== "undefined" && document.addEventListener) { globalThis.__plsTlSpace = true; document.addEventListener("keydown", _onSpace, true); }
        // v1031: a resize, a pan or a drag ends with the pointer let go --
        // then every timeline looks once whether its width changed
        window.addEventListener("pointerup", () => {
            for (const n of (app.graph && app.graph._nodes) || []) {
                if (n && n.type === NODE && n._plsTl) { try { _widthCheck(n, true); _stripCheck(n, true); } catch (e) { /* view only */ } }
            }
        }, true);
        // v1034: the export without a cable. A run's timeline says which scene
        // it was (its ui: clip_prefix + scene); the Save node downstream says
        // which file it wrote. Both arrive as "executed" events of the same
        // prompt -- together they list the clip in the project's stack.
        // v1047 (Frank 01.10.: "new render -> new track did not work"): a
        // timeline that comes from the CACHE sends no "executed" -- its run is
        // read from "execution_cached" (the node's last run, else its model).
        // And ONE track per run: the Saves' files are collected, at the run's
        // end the one wired to clip_prefix -- else the last -- becomes the track.
        // v1088 (the Timeline audit B3 / B9 / B12): ONE record per prompt with a run PER
        // timeline node (two timelines in a graph no longer overwrite each other); a Save's
        // files go to the timeline it hangs below in the graph (or the only timeline of
        // the prompt); a run read from "execution_cached" is PROVISIONAL -- its Saves are
        // held until the node's replayed "executed" says which scene it really was (with
        // --cache-lru the node's last run could be another scene), then flushed.
        const runs = new Map();
        const tlNodes = () => ((app.graph && app.graph._nodes) || []).filter((n) => n && n.type === NODE && n._plsTl);
        const tlById = (id) => tlNodes().find((n) => String(n.id) === String(id)) || null;
        const keep = (pid, rec) => { runs.set(pid, rec); if (runs.size > 64) runs.delete(runs.keys().next().value); };
        // v1198: the render a prompt belongs to, taken WHEN IT IS QUEUED (one Run press = one sweep stamp) -- read at
        // the run's end the node's stamp is already the next press's when the user pressed Run again meanwhile
        const sweepOfPid = new Map();
        if (api && typeof api.queuePrompt === "function" && !api.queuePrompt.__plsSweep) {
            const origQ = api.queuePrompt;
            const wrapQ = async function () {
                const sw = (tlNodes()[0] || {})._plsSweep || null;
                // v1219: the cut ends of the carry go into THIS run's prompt (not the widget, not the workflow)
                try {
                    const data = arguments[1], out = data && data.output;
                    if (out) for (const id of Object.keys(out)) {
                        const nd = out[id], n = tlById(id);
                        if (!nd || nd.class_type !== NODE || !n || !n._plsTl || typeof (nd.inputs || {}).timeline !== "string") continue;
                        const t2 = carryInject(nd.inputs.timeline, n._plsTl.plan);
                        if (t2) nd.inputs.timeline = t2;
                    }
                } catch (e) { /* the run goes as it was */ }
                const res = await origQ.apply(this, arguments);
                try {
                    const pid = res && res.prompt_id;
                    if (pid && sw) { sweepOfPid.set(pid, sw); if (sweepOfPid.size > 256) sweepOfPid.delete(sweepOfPid.keys().next().value); }
                } catch (e) { /* bookkeeping only */ }
                return res;
            };
            wrapQ.__plsSweep = true;
            api.queuePrompt = wrapQ;
        }
        const record = (pid) => { let r = runs.get(pid); if (!r) { r = { tls: new Map() }; keep(pid, r); } return r; };
        const say = (run, text) => { const n = tlById(run.node); if (n) n._plsTl.state.textContent = text; };
        const registerRef = (run, ref) => {
            // v1088 (audit C10): the slot is taken when the Save ran, filled when the answer
            // comes -- the pick ("the run's last video") follows the run, not the network
            const slot = run.rels.length;
            run.rels.push(null);
            run.pending.push(api.fetchApi("/pls/cine/register", { method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ project: run.project, scene: run.scene, file: ref }) })
                .then((r) => r.json()).then((j) => {
                    if (j && j.rel) run.rels[slot] = j.rel;
                    else if (j && j.error) say(run, "stack: " + j.error);
                }).catch(() => { /* the folder scan still sees wired clips */ }));
        };
        const flush = (run) => { const q = run.held.splice(0); run.provisional = false; for (const ref of q) registerRef(run, ref); };
        api.addEventListener("execution_cached", (ev) => {
            try {
                const d = ev && ev.detail;
                if (!d || !Array.isArray(d.nodes)) return;
                const ids = new Set(d.nodes.map(String));
                for (const n of tlNodes()) if (ids.has(String(n.id))) {
                    const rec = record(d.prompt_id);
                    if (!rec.tls.has(String(n.id))) rec.tls.set(String(n.id), freshRun(n._plsLastRun || runOf(_model(n)), n.id, true));
                }
            } catch (e) { /* a listener must never break a run */ }
        });
        api.addEventListener("executed", (ev) => {
            try {
                const d = ev && ev.detail;
                if (!d || !d.output) return;
                const t = d.output.pls_timeline && d.output.pls_timeline[0];
                const nid = String(d.display_node || d.node);
                if (t && t.clip_prefix) {
                    const r = { project: t.project ? String(t.project) : projectOf(t.clip_prefix), scene: Number(t.scene) || 0, clip_prefix: String(t.clip_prefix) };
                    const tn = tlById(nid) || tlById(d.node);
                    if (tn) tn._plsLastRun = r;
                    const rec = record(d.prompt_id), key = String(tn ? tn.id : nid), have = rec.tls.get(key);
                    // v1084: the cached node's REPLAYED "executed" (core: at the end, after the Saves)
                    // must not start its run over -- same run: keep it; v1088: a provisional run
                    // takes the real identity now and its held Saves are registered under it
                    if (have && have.provisional) { Object.assign(have, r); flush(have); return; }
                    if (have && sameRun(have, r)) return;
                    rec.tls.set(key, freshRun(r, key, false));
                    return;
                }
                const rec = runs.get(d.prompt_id);
                if (!rec || !rec.tls.size) return;
                const refs = videoRefs(d.output);
                if (!refs.length) return;
                const run = attributeSave(rec, nid, app.graph);
                if (!run) {
                    for (const r of rec.tls.values()) say(r, "two timelines ran in this prompt -- the clip of node " + nid
                        + " hangs below neither: wire it below one timeline (or run one at a time)");
                    return;
                }
                for (const ref of refs) {
                    const key = ref.subfolder + "/" + ref.filename;
                    if (run.sent.has(key)) continue;
                    run.sent.add(key);
                    if (run.provisional) run.held.push(ref); else registerRef(run, ref);
                }
            } catch (e) { /* a listener must never break a run */ }
        });
        // a finished run may have put a new clip on the stack
        api.addEventListener("execution_success", async (ev) => {
            const pid = ev && ev.detail && ev.detail.prompt_id, rec = pid ? runs.get(pid) : null;
            if (rec) {
                for (const run of rec.tls.values()) {
                    if (run.provisional && run.held.length) flush(run);   // an older server replays nothing: its best guess
                    if (!run.pending.length) continue;
                    try { await Promise.all(run.pending); } catch (e) { /* listed or not */ }
                    run.rels = run.rels.filter(Boolean);
                    const pick = pickTrackFile(run.rels, run.clip_prefix);
                    console.info("[PLS] Cine Timeline: run " + String(pid).slice(0, 8) + " node " + run.node + " -- " + run.rels.length
                        + " clip(s) registered for " + run.project + (run.scene ? " scene " + run.scene : " (whole)") + ", pick " + (pick || "none"));
                    // v1050: the SAME pick leads V1 -- the stack list marks it, the server orders the takes by it
                    if (pick) {
                        const cut = pick.lastIndexOf("/");
                        try {
                            await api.fetchApi("/pls/cine/register", { method: "POST", headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ project: run.project, scene: run.scene, primary: true, run_files: run.rels,   // v1165: the run history
                                    sweep: sweepOfPid.get(pid) || (tlById(run.node) || {})._plsSweep || null,                    // v1190: the render it belongs to (v1198: stamped at queue time)
                                    file: { filename: pick.slice(cut + 1), subfolder: cut >= 0 ? pick.slice(0, cut) : "", type: "output" } }) });
                        } catch (e) { /* the newest file still leads */ }
                        // v1046: "new render -> new track" -- v1088: on THIS run's timeline only
                        const n = tlById(run.node);
                        if (n && projectName(_model(n)) === run.project) {
                            try { await _autoTrack(n, pick, run.scene, run.rels.length, sweepOfPid.get(pid) || n._plsSweep || null); } catch (e) { /* view only */ }
                        }
                    }
                }
                runs.delete(pid);
            }
            for (const n of tlNodes()) _stack(n);
        });
        // v1034: "all scenes, one run each" -- ONE Run (the button, Ctrl+Enter,
        // the node's render) becomes one queued run per scene. The timeline
        // stays on "all"; each run carries active = k, awaited one by one so
        // every run is serialised with its own scene.
        const orig = app.queuePrompt;
        if (typeof orig === "function" && !orig.__plsSplit) {
            let splitting = false;
            const split = async function (number, batchCount) {
                let node = null;
                for (const n of (app.graph && app.graph._nodes) || []) {
                    if (n && n.type === NODE && n._plsTl && (n.mode === 0 || n.mode === undefined)) {
                        const m = _model(n);
                        if (m.render === "all" && scenesOf(m).length > 1) { node = n; break; }
                    }
                }
                // v1190: one render = one sweep stamp for every scene it queues (NEW = "came in with the latest render")
                if (!splitting) { const sw = Date.now() / 1000; for (const n of tlNodes()) n._plsSweep = sw; }   // seconds, like the server clock
                if (!node || splitting) return orig.apply(this, arguments);
                splitting = true;
                const m0 = _model(node), n = scenesOf(m0).length, keep = m0.active;
                let res;
                let done = 0, refused = 0;
                try {
                    const skip = skipOf(node);                       // v1163: the ticked scenes only
                    if (pickedOrder(n, number, skip).length === 0) { node._plsTl.state.textContent = "no scene is ticked -- nothing queued"; return false; }
                    for (const k of splitOrder(n, number)) {
                        if (skip.has(k)) continue;
                        const mm = _model(node); mm.active = k; _save(node, mm, false);
                        node._plsTl.state.textContent = "queueing scene " + k + " of " + n + " ...";
                        res = await orig.call(this, number, batchCount);
                        if (res === false) { refused = k; break; }   // v1082: refused = stop, say so
                        done++;
                    }
                    node._plsTl.state.textContent = refused ? "scene " + refused + " was refused (see the dialog) -- the remaining scenes are not queued (" + done + " queued)"
                        : done + " scene run(s) queued" + (done < n ? " (the ticked ones)" : "") + " -> the stack of " + (m0.project || "cine");
                } catch (x) {
                    node._plsTl.state.textContent = "queue failed after " + done + " scene run(s): " + x.message;
                    throw x;
                } finally {
                    const mm = _model(node); mm.active = keep; _save(node, mm, false);
                    splitting = false;
                    try { _render(node); } catch (e) { /* view only */ }
                }
                return res;
            };
            split.__plsSplit = true;
            app.queuePrompt = split;
        }
    },
});
