/**
 * Polyhedron World -- model files line (web/js/ph_world_weights.js, v1058)
 *
 * One line INSIDE the node (no floating window) that says whether the model
 * files are on disk and, when one is missing, offers the download:
 *   World Reconstruct / Depth : the DA3 file picked in `model`
 *   Image to Splat            : the four TripoSplat files
 * Nothing is downloaded unless the button is clicked; the server fetches from
 * the fixed Hugging Face address and checks the sha256 (ph_world_weights.py).
 *
 * v1059: install, progress and the amber notice live in ph_model_door.js --
 * the line and the notice are two handles on ONE download.
 *
 * House style: English for code, comments and UI text; ASCII only.
 */

import { app } from "../../scripts/app.js";
import { installModel as install, onProgress, noticeMissing, modelStatus, refreshModelLists, nodeActive, watchMode }
    from "./ph_model_door.js";
import { setHidden } from "./ph_widget_vis.js";

/* v1097: a PICK node shows the line for the model chosen in its combo -- Download only while that
   pick is not installed (Frank, 11.10.: "zuerst das (ungeladene) Modell auswaehlen und dann die
   Download-Moeglichkeit ... wie bei anderen Nodes in der Suite") */
// public build (v392): only the H3 latent upscaler and the Power Upscale's restore set are served here -- the other sets of the
// development tree belong to nodes that are not part of this release.
const PICK_NODES = {};
const SET_NODES = { ULSH3LatentUpscale: "h3_latent_upscale",                           // v1193: the H3 latent upscaler
                    ULSPowerUpscale: "restore" };                                      // v1194: the restore models (all optional)
/* v1204 (Frank 21.10.: "Das sollte dann erscheinen, sobald man es wirklich braucht ergo: nutzen moechte ... das muss
 * schaerfer sein ... immer auch in Hinblick auf den Malware-Download und unsere Redflags bei ComfyUI"): the line shows
 * ONLY what the node needs NOW -- and nothing else is ever offered:
 *   - a bypassed or muted node (also through its group) needs nothing: no line at all;
 *   - the set's REQUIRED files are needed while the node runs;
 *   - an OPTIONAL file is needed only while the option that uses it is chosen on the node (OPTION_NEEDS):
 *       pick   = a combo names the file (+ a companion it cannot run without, e.g. a model's own VAE);
 *       toggle = a switch turns on the path that loads these files.
 *   Nothing needed -> no line. Needed and missing -> orange + Download of exactly those files. Needed and there ->
 *   a quiet green check. No "optional download" button any more: nothing is offered that nobody asked for. */
const OPTION_NEEDS = {
    ULSPowerUpscale: { kind: "pick", widget: "restore_model", companion: /vae/i },          // v1199 restore models
};
/* the files the node needs now -> a list of status entries (pure: the guard drives it in node.js) */
export function neededModels(models, opt, value) {
    const req = models.filter((m) => !m.optional);
    if (!opt) return req;
    if (opt.kind === "toggle") return value ? req.concat(models.filter((m) => m.optional && (opt.files || []).includes(m.name))) : req;
    const name = stripLabel(value);
    if (!name || name === "off" || name.startsWith("(")) return [];
    const pick = models.filter((m) => m.name === name);
    if (!pick.length) return [];                     // a file on disk the registry does not know: nothing to fetch
    return pick.concat(opt.companion ? models.filter((m) => m.name !== name && opt.companion.test(m.name)) : []);
}
const LINE_H = 30;
const TAIL = /\s\u00b7\s[\d.,]+\s?(KB|MB|GB)(\s\u00b7\snot installed)?$/;

export function stripLabel(v) {
    return String(v == null ? "" : v).replace(TAIL, "").trim();
}

export function fmtBytes(n) {
    n = Number(n) || 0;
    if (n >= 1073741824) return (n / 1073741824).toFixed(1) + " GB";
    if (n >= 1048576) return Math.round(n / 1048576) + " MB";
    return Math.max(1, Math.round(n / 1024)) + " KB";
}

const fetchStatus = modelStatus;

// the line follows the file's progress for as long as it is fetched
function follow(line, name) {
    if (line._off) line._off();
    line._off = onProgress(name, (done, total, phase) => {
        if (phase === "start" || phase === "run") line.progress(done, total);
        else if (phase === "verify") line.progress(total, total, "checking sha256 (seconds for a big file) ...");
        else if (phase === "convert") line.progress(1, 1, "converting to safetensors (the console says what happens) ...");
    });
}

class Line {
    constructor(node) {
        this.node = node;
        const root = document.createElement("div");
        root.style.cssText = "display:flex;align-items:center;gap:6px;height:" + (LINE_H - 6) + "px;" +
            "font:11px sans-serif;color:#bbb;padding:0 4px;overflow:hidden;";
        this.text = document.createElement("span");
        this.text.style.cssText = "flex:1 1 auto;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
        this.bar = document.createElement("div");
        this.bar.style.cssText = "flex:0 0 70px;height:6px;border-radius:3px;background:#444;overflow:hidden;display:none;";
        this.fill = document.createElement("i");
        this.fill.style.cssText = "display:block;height:100%;width:0;background:#8fff8f;";
        this.bar.appendChild(this.fill);
        this.btn = document.createElement("button");
        this.btn.style.cssText = "flex:0 0 auto;padding:1px 8px;border-radius:4px;border:1px solid #8fff8f;" +
            "background:#8fff8f;color:#123;font:11px sans-serif;cursor:pointer;display:none;";
        root.appendChild(this.text);
        root.appendChild(this.bar);
        root.appendChild(this.btn);
        this.root = root;
        this.busy = false;
    }
    // v1204: a node that needs nothing shows no line at all (the widget is hidden, the node shrinks back)
    // the node grows / shrinks by exactly the line -- a size the user set stays his (no whole-node refit)
    _shift(dy) {
        const n = this.node;
        try {
            if (n && n.setSize && n.size) n.setSize([n.size[0], Math.max(40, n.size[1] + dy)]);
            n && n.setDirtyCanvas && n.setDirtyCanvas(true, true);
        } catch (e) { /* layout only */ }
    }
    hide() {
        if (this.hidden) return;
        this.hidden = true;
        this.root.style.display = "none";
        if (this.widget && setHidden(this.widget, true)) this._shift(-(LINE_H + 4));
    }
    reveal() {
        if (!this.hidden) return;
        this.hidden = false;
        this.root.style.display = "flex";
        if (this.widget && setHidden(this.widget, false)) this._shift(LINE_H + 4);
    }
    show(text, colour, btnLabel, onClick, title) {
        this.reveal();
        this.text.textContent = text;
        this.text.style.color = colour || "#bbb";
        this.root.title = title || text;
        this.bar.style.display = "none";
        if (btnLabel) {
            this.btn.textContent = btnLabel;
            this.btn.style.display = "";
            this.btn.onclick = (e) => { e.stopPropagation(); onClick(); };
        } else {
            this.btn.style.display = "none";
            this.btn.onclick = null;
        }
    }
    progress(done, total, note) {
        this.btn.style.display = "none";
        this.bar.style.display = "";
        const pct = total ? Math.min(100, done / total * 100) : 0;
        this.fill.style.width = pct.toFixed(1) + "%";
        this.text.textContent = note || (total ? (this.prefix || "") + fmtBytes(done) + " of " + fmtBytes(total)
                                                 : "downloading ...");
        this.text.style.color = "#ddd";
    }
}

function addLine(node) {
    if (typeof node.addDOMWidget !== "function") return null;
    const line = new Line(node);
    const w = node.addDOMWidget("pls_weights_line", "pls_weights", line.root, {
        serialize: false, hideOnZoom: false, getMinHeight: () => LINE_H,
    });
    w.computeSize = function (width) { return [width, LINE_H]; };
    w.serialize = false;              // a readout, not a value: widgets_values stay as they were
    line.widget = w;
    node.__plsWeights = line;
    return line;
}

const refreshLists = refreshModelLists;

/* the file picked in the node's model combo (World Reconstruct / Depth: DA3; Load Text Model) */
const pickOf = (node) => PICK_NODES[node.comfyClass || node.type];
async function updatePick(node) {
    const line = node.__plsWeights;
    const pk = pickOf(node);
    const w = pk && node.widgets && node.widgets.find((x) => x.name === pk.widget);
    if (!line || !w || line.busy) return;
    if (!nodeActive(node)) { line.hide(); return; }      // v1204: bypassed / muted -- nothing is needed
    const seq = line.seq = (line.seq || 0) + 1;          // v1204: only the LATEST check may draw
    const name = stripLabel(w.value);
    if (pk.set === "da3" && !name.toLowerCase().endsWith(".safetensors")) {
        line.show("DA3 package model -- installed by hand, read from models/depth_anything_3", "#999");
        return;
    }
    let models = [];
    try { models = await fetchStatus(pk.set); } catch (e) { if (seq === line.seq) line.show("model status unavailable", "#999"); return; }
    if (seq !== line.seq || line.busy) return;           // a stale answer draws nothing
    if (!nodeActive(node)) { line.hide(); return; }
    const base = name.split("/").pop();
    const m = models.find((x) => x.name === name || x.name === base);
    if (!m) { line.show("\u2713 " + name + " -- on disk", "#9c9"); return; }
    if (m.installed) {
        line.show("\u2713 " + m.title + " on disk (" + m.size + ")", "#9c9", null, null, m.path);
        return;
    }
    const after = () => { if (!line.busy) updatePick(node); };
    // v1100: the notice waits a tick and fires only if the pick STILL names the missing file -- a
    // loaded node is created with the combo's first entry (the recommended, uninstalled model) and
    // gets its saved pick in onConfigure right after; the toast then named a model nobody picked
    // (Frank, 11.10.: his qwen3vl_8b was on disk and selected, the toast asked for Qwen3.5 4B)
    setTimeout(() => {
        if (stripLabel(w.value) !== name || line.busy) return;
        noticeMissing(m, node.title || "World", after, node);
    }, 150);
    line.show(m.title + " not installed", "#e0a030", "Download " + m.size, async () => {
        line.busy = true;
        line.prefix = "";
        follow(line, m.name);
        line.progress(0, m.bytes);
        try {
            await install(m.name);
            line.busy = false;
            refreshLists();
            // the same file, now with its size and without the mark
            const fresh = (await fetchStatus(pk.set)).find((x) => x.name === m.name);
            if (fresh && w.options && Array.isArray(w.options.values)) {
                const vals = w.options.values;
                const at = vals.findIndex((v) => stripLabel(v) === m.name);
                if (at >= 0) vals[at] = fresh.label; else vals.unshift(fresh.label);
                w.value = fresh.label;
            }
            updatePick(node);
        } catch (e) {
            line.busy = false;
            line.show("Download failed -- hover for the reason and the manual way", "#ff6b6b", "Retry",
                      () => updatePick(node), String(e.message || e));
        }
        node.setDirtyCanvas && node.setDirtyCanvas(true, false);
    }, "Goes to " + m.dir + " -- from huggingface.co, sha256-checked. Nothing downloads unless you click.");
}

/* v1060: each set names itself and its source on the line (it said "TripoSplat" for every set) */
const SET_TEXT = {
    triposplat: { title: "TripoSplat", source: "huggingface.co/VAST-AI/TripoSplat" },
    music: { title: "Music3 DAV encoder", source: "huggingface.co/MiniMaxAI/MiniMax-Music3 (dav.pth, converted " +
             "once to minimax_music3_dav_full.safetensors, the download removed)" },
    music_stems: { title: "BS-RoFormer vocals (UVR / viperx, MIT)", source: "github.com/TRvlvr/model_repo (UVR's public "
                   + "models; model code lucidrains / ZFTurbo, MIT) -- the .ckpt converted once to safetensors into models/polyhedron_music" },
    vocal_clean: { title: "Vocal Clean models (MelBand-RoFormer karaoke + BS-RoFormer dereverb)",
                   source: "huggingface.co: Politrees/UVR_resources (UVR's karaoke release by aufr33 & viperx, byte-identical) "
                         + "and anvuew/deverb_bs_roformer (weights GPL-3.0) -- each .ckpt converted once to safetensors "
                         + "into models/polyhedron_music" },
    voice_swap: { title: "Voice Swap models (YingMusic-SVC + BigVGAN + Whisper-small + RMVPE + CAM++)",
                  source: "huggingface.co: GiantAILab/YingMusic-SVC (weights CC BY-NC 4.0, non-commercial), "
                        + "nvidia/bigvgan_v2_44khz_128band_512x, openai/whisper-small, lj1995/VoiceConversionWebUI, "
                        + "funasr/campplus -- each converted once to safetensors into models/polyhedron_music" },
    restore: { title: "Restoration models for the Power Upscale (Core's SeedVR2 family, Apache-2.0)",
               source: "huggingface.co/Comfy-Org/SeedVR2 -- the model into models/diffusion_models, its VAE into models/vae; "
                     + "pick the model in the node's restore_model (v1199: the node loads it with its VAE, next to the refine)" },
    parts: { title: "Parts checkpoints (SegFormer clothes + face parsing)",
             source: "huggingface.co: mattmdjaga/segformer_b2_clothes (MIT) and jonathandinu/face-parsing (non-commercial research) "
                   + "-- each as a folder (weights + the two configs) into models/segformer" },
    h3_latent_upscale: { title: "H3 latent upscaler (LBH-123-AI, MIT)",
                         source: "huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler (fp16) into models/latent_upscale_models" },
    semantic_bridge: { title: "Semantic Bridge adapters",
                       source: "huggingface.co: speach1sdef178/MiniMax-H3-Semantic-Bridge (MiniMax H3 Community License) and, optional, "
                             + "JOKER141/BUNNY_H3_Conditioning_Bridge V2 -- into models/semantic_bridge" },
    object_remove: { title: "big-lama (LaMa inpainting, Apache-2.0)",
                     source: "huggingface.co/biali/big-lama (the published generator as a safetensors state dict -- verified "
                           + "byte-identical to the TorchScript release's weights) into models/lama" },
    video_matte: { title: "MatAnyone (video matting, S-Lab 1.0 non-commercial)",
                   source: "huggingface.co/PeiqingYang/MatAnyone (the author's copy, already safetensors; the same weights as the "
                         + "GitHub release's matanyone.pth) into models/matanyone" },
    music_analyse: { title: "Music Analyse models (Demucs htdemucs_ft + music CLAP)",
                     source: "huggingface.co/adefossez/HTDemucs-ft (the Demucs author's repo, MIT) and laion/larger_clap_general "
                           + "(the .bin + its tokenizer data, converted once into the audio tower and the label bank) "
                           + "into models/polyhedron_music" },
};

/* Image to Splat: the four TripoSplat files; Music Audio Encode: the DAV encoder */
async function updateSet(node) {
    const line = node.__plsWeights;
    const set = SET_NODES[node.comfyClass || node.type];
    const txt = SET_TEXT[set] || { title: set, source: "huggingface.co" };
    if (!line || line.busy) return;
    if (!nodeActive(node)) { line.hide(); return; }          // v1204: bypassed / muted -- nothing is needed
    const seq = line.seq = (line.seq || 0) + 1;              // v1204: only the LATEST check may draw
    let models = [];
    try { models = await fetchStatus(set); } catch (e) { if (seq === line.seq) line.show("model status unavailable", "#999"); return; }
    // the node may have been bypassed (or re-checked) while the status was on its way -- a stale answer draws nothing
    if (seq !== line.seq || line.busy) return;
    if (!nodeActive(node)) { line.hide(); return; }
    // v1204: ONLY what the node needs now (required files + the files of the option chosen on it)
    const opt = OPTION_NEEDS[node.comfyClass || node.type];
    const ow = opt && node.widgets && node.widgets.find((x) => x.name === opt.widget);
    const need = neededModels(models, opt, ow ? ow.value : undefined);
    if (!need.length) { line.hide(); return; }               // nothing needed: no line, no offer
    const miss = need.filter((m) => !m.installed);
    const list = need.map((m) => (m.installed ? "\u2713 " : "- ") + m.name + "  (" + m.size + ", " + m.note +
                                   ")" + (m.outdated ? "  OUTDATED: " + m.outdated : "")).join("\n");
    if (!miss.length) {
        line.show("\u2713 " + txt.title + ": " + (need.length === 1 ? "on disk" : "all " + need.length + " files on disk"), "#9c9", null, null, list);
        return;
    }
    // v1092: a rebuild from kept files costs no download; an outdated file is named as such
    const total = miss.reduce((a, m) => a + (m.rebuild ? 0 : m.bytes), 0);
    const old = miss.filter((m) => m.outdated);
    const text = txt.title + ": " + (old.length ? "outdated -- " + old.map((m) => m.outdated).join("; ")
                                     : (need.length === 1 ? "not installed" : (need.length - miss.length) + " of " + need.length + " files"));
    line.show(text, "#e0a030", total ? "Download " + fmtBytes(total) : "Rebuild (no download)", async () => {
        line.busy = true;
        let i = 0;
        try {
            for (const m of miss) {
                i++;
                line.prefix = (miss.length > 1 ? i + "/" + miss.length + " " : "") + m.title + ": ";
                follow(line, m.name);
                line.progress(0, m.bytes);
                await install(m.name);
            }
            line.busy = false;
            refreshLists();
            // v1199: a pick names the file it just got -- with its size now, without the mark
            if (opt && opt.kind === "pick" && ow && ow.options && Array.isArray(ow.options.values)) {
                try {
                    const fresh = (await fetchStatus(set)).find((x) => x.name === stripLabel(ow.value));
                    if (fresh && fresh.installed && fresh.label) {
                        const vals = ow.options.values, at = vals.findIndex((v) => stripLabel(v) === fresh.name);
                        if (at >= 0) vals[at] = fresh.label;
                        ow.value = fresh.label;
                    }
                } catch (e) { /* the label only */ }
            }
            updateSet(node);
        } catch (e) {
            line.busy = false;
            line.show("Download failed -- hover for the reason and the manual way", "#ff6b6b", "Retry",
                      () => updateSet(node), String(e.message || e));
        }
    }, list + "\n\nFrom " + txt.source + ", sha256-checked, into the folders ComfyUI reports. Nothing downloads unless you click.");
}

function setup(node, kind) {
    const line = addLine(node);
    if (!line) return;
    const sz = node.computeSize();
    node.setSize([Math.max(node.size[0], sz[0]), Math.max(node.size[1], sz[1])]);
    if (kind === "pick") {
        const w = node.widgets && node.widgets.find((x) => x.name === pickOf(node).widget);
        if (w && !w._plsWeightsBound) {
            const orig = w.callback;
            w.callback = function () {
                const r = orig ? orig.apply(this, arguments) : undefined;
                try { updatePick(node); } catch (e) { /* never break a pick */ }
                return r;
            };
            w._plsWeightsBound = true;
        }
        updatePick(node);
    } else {
        // v1199 / v1204: a line that follows an option on the node re-reads it when the option changes
        const sp = OPTION_NEEDS[node.comfyClass || node.type];
        const w = sp && node.widgets && node.widgets.find((x) => x.name === sp.widget);
        if (w && !w._plsWeightsBound) {
            const orig = w.callback;
            w.callback = function () {
                const r = orig ? orig.apply(this, arguments) : undefined;
                try { updateSet(node); } catch (e) { /* never break a pick */ }
                return r;
            };
            w._plsWeightsBound = true;
        }
        updateSet(node);
    }
    // v1204: bypass / mute / back -- the line follows the node's mode (one quiet timer, any renderer)
    watchMode(node, (n) => { try { if (kind === "pick") updatePick(n); else updateSet(n); } catch (e) { /* never */ } });
}

app.registerExtension({
    name: "polyhedron.world_weights",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const name = nodeData && nodeData.name;
        const kind = PICK_NODES[name] ? "pick" : (SET_NODES[name] ? "set" : null);
        if (!kind) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = created ? created.apply(this, arguments) : undefined;
            try { setup(this, kind); } catch (e) { console.warn("[PLS] World weights line:", e); }
            return r;
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = configured ? configured.apply(this, arguments) : undefined;
            try {
                // v1059: a loaded node keeps its SAVED size, which was taken before the
                // line existed -- the line then hung below the node. Grow to fit.
                const sz = this.computeSize();
                if (this.size[1] < sz[1] || this.size[0] < sz[0])
                    this.setSize([Math.max(this.size[0], sz[0]), Math.max(this.size[1], sz[1])]);
                if (kind === "pick") updatePick(this); else updateSet(this);
            } catch (e) { /* ignore */ }
            return r;
        };
    },
});
