# -*- coding: ascii -*-
"""
ph_music.py -- MiniMax-Music3 in the suite: the Music Prompt and the Music
Save.

WHY: ComfyUI runs MiniMax Music 3 natively (core nodes MiniMaxMusic3TextEncode,
EmptyMiniMaxMusic3LatentAudio, VAEDecodeAudio[Tiled]), but as five loose
parts, with no place in a project and no label on the file. The Cine line
writes "non_diegetic_music: N/A" into every H3 prompt on purpose -- H3 makes
music per clip, and a film cut from several clips would change its music at
every cut. A score over the whole film comes from a music model: this one.

THE PROMPT (ULSMusicPrompt) takes the text encoder (CLIP, type "minimax") and
the two texts the model reads:
  caption  -- three sections: Global Metadata / Vocal Details / Arrangement
  lyrics   -- section tags drive the structure ([Intro] [Verse] [Chorus]
              [Bridge] [Instrumental] [Solo] [Outro] ...); the words only
              carry the mood
Both take "//" comment lines, as the suite's CLIP Text Encode does -- the
same cleaner (ph_clip_encode._clean), newlines kept (the lyrics need them).
The 8B language model inside the "text encoder" WRITES the music here (the
acoustic codes); the sampler then refines it. So the length the song really
got is known after this node: the empty latent goes out at exactly that
length, with the negative (zeroed, as the official template does).

THE SAVE (ULSMusicSave) decodes (decode_music below, tiled if asked)
or takes an AUDIO, and writes output/<project>/music/<name>_NNNNN_.<ext> --
next to the Cine clips of the same project -- plus a sidecar .json with the
caption, lyrics, seed and length. MiniMax-Music3 COMMUNITY LICENSE: machine-
generated content must be labelled, and a commercial product must show the
name "MiniMax-Music3" -- the file's metadata and the sidecar carry
"AI-generated with MiniMax-Music3", and both node names carry it.

THE WORKFLOW IN THE FILE (v1042): as the core's SaveAudio does, the Save takes
the hidden PROMPT and EXTRA_PNGINFO and writes them as the tags "prompt" and
"workflow" -- the frontend reads them back when a .flac / .mp3 is dropped on
the canvas (measured in 1.49.6: getFlacMetadata / getMp3Metadata; .wav is not
read). The JSON is ASCII-escaped: the mp3 reader scans the raw bytes as
latin-1. Two seeds make a song -- the Music Prompt's (the language model that
writes the codes) and the sampler's (the noise the DiT refines) -- so the
recipe tag "polyhedron_music" carries both, the sampler's found by walking
the executed graph back from the Save's 'samples'. The sidecar .json IS the
workflow then (the recipe under extra.polyhedron_music): dropped on the
canvas it loads too -- the way back for a .wav. --disable-metadata keeps all
of it out, as in the core.

NO CORE NODES (v1060, the pack rule since v1057: the suite calls the comfy.*
model library, never ComfyUI's node classes -- Core is GPL, the pack MIT): the
negative is zeroed here (zero_conditioning) and the decode is written here
(decode_music) instead of nodes.ConditioningZeroOut and
comfy_extras.nodes_audio.vae_decode_audio. Same results -- the Core-backed
guard half compares both against the core nodes.

AUDIO IN (v1060): the Music Prompt takes an optional `audio_latent` from the
Music Audio Encode (ph_music_dav.py). The language model still writes the
song; the latent that goes to the sampler is then the INPUT, cut to the
length the language model really wrote -- S0 measured that the part beyond
it breaks at high denoise (d097: 25 dB from the input past 42 s). Shorter
input: the rest stays empty (generated from the caption alone) and is said.

Sample rate: ComfyUI's DAV decoder gives 512 samples per latent frame and a
latent frame is 1/86.13 s -> 44100 Hz (comfy/sd.py sets 44100; measured from
latent_length() in comfy/ldm/minimax_music/dit.py). MiniMax's own pages say
32 kHz for their reference code -- the file carries what the decoder gives.
"""

import json
import os
import re

MODEL_LABEL = "MiniMax-Music3"
AI_LABEL = "AI-generated with MiniMax-Music3"
MUSIC_SUBFOLDER = "music"
FORMATS = ("flac", "mp3", "wav")
FPS_FALLBACK = 25            # comfy.ldm.minimax_music.ar.AUDIO_FRAMES_PER_SECOND
MAX_FRAMES_FALLBACK = 9000   # comfy.ldm.minimax_music.ar.MAX_AUDIO_FRAMES
SECTION_TAGS = ("[Intro]", "[Verse]", "[Pre-Chorus]", "[Chorus]", "[Post-Chorus]",
                "[Bridge]", "[Instrumental]", "[Solo]", "[Outro]")

# An example that runs as it is: an instrumental film score (the Cine case).
# Whether "no vocals" in Vocal Details plus [Instrumental] keeps a song
# entirely free of voice is NOT measured yet -- a field test on a GPU.
DEFAULT_CAPTION = (
    "// GLOBAL METADATA -- genre, tempo (BPM), key, the mood's arc, what it is for, production\n"
    "Global Metadata: Cinematic orchestral score with ambient textures. 84 BPM, D minor. "
    "Quiet and expectant at first, a slow build of warmth in the middle, resolving calmly at the end. "
    "Film soundtrack under a winter forest scene. Wide, clean concert-hall production with soft reverb.\n"
    "\n"
    "// VOCAL DETAILS -- voice, delivery, harmonies (a score: say there are none)\n"
    "Vocal Details: No vocals, purely instrumental.\n"
    "\n"
    "// ARRANGEMENT -- instruments, groove, and what each section does\n"
    "Arrangement: Sustained low strings and a soft felt piano as the bed, sparse harp figures, "
    "a solo cello carrying the melody. Intro: piano and string pad alone. Middle: horns and "
    "timpani swell in gently, the cello rises. Outro: everything thins out to a last piano chord.\n"
)
DEFAULT_LYRICS = (
    "// the tags are the structure; for a score use [Instrumental] / [Solo] sections only\n"
    "[Intro]\n"
    "[Instrumental]\n"
    "[Outro]\n"
)


# --------------------------------------------------------------- pure helpers
try:
    from . import ph_music_analyse_lib as AL          # v1069: the per-rubric caption merge (pure)
except ImportError:
    import ph_music_analyse_lib as AL


def clean_text(text):
    """'//' comment lines out, newlines kept -- the CLIP Text Encode's own
    cleaner (one implementation). Lines that end up empty are dropped."""
    try:
        from .ph_clip_encode import _clean
    except ImportError:                      # direct run / tests
        from ph_clip_encode import _clean
    return _clean(text or "", True, False)


def core_limits():
    """(frames per second, max frames) from the core -- constants as fallback."""
    try:
        from comfy.ldm.minimax_music.ar import AUDIO_FRAMES_PER_SECOND, MAX_AUDIO_FRAMES
        return int(AUDIO_FRAMES_PER_SECOND), int(MAX_AUDIO_FRAMES)
    except Exception:
        return FPS_FALLBACK, MAX_FRAMES_FALLBACK


def asked_frames(seconds, fps=FPS_FALLBACK, max_frames=MAX_FRAMES_FALLBACK):
    """Seconds -> the model's audio frames (as core's node rounds them)."""
    return min(int(max_frames), max(1, int(round(float(seconds) * fps))))


def want_seconds(seconds, seconds_in=None):
    """The wired seconds (the Cine Timeline's) win over the widget when > 0."""
    try:
        v = float(seconds_in)
        if v == v and v > 0:
            return v, "wired"
    except (TypeError, ValueError):
        pass
    return float(seconds), "widget"


_SECTIONS_SENTENCE = re.compile(r"Sections:\s*[^.\n]*\.")


def align_sections(caption, lyrics):
    """v1096: the caption's 'Sections: a; b; c.' (the Sheet's measured order) follows the lyrics'
    tags when the two differ -- the words decide the song's form and length (Frank, 11.10.: the
    measured Verse / Chorus of a short template held a 90 s song to 37 s). -> (caption, note)."""
    tags = [t.strip("[]").strip() for t in sections_of(lyrics)]
    m = _SECTIONS_SENTENCE.search(caption or "")
    if not tags or not m:
        return caption, ""
    new = "Sections: " + "; ".join(tags) + "."
    if " ".join(m.group(0).lower().split()) == " ".join(new.lower().split()):
        return caption, ""
    return caption[:m.start()] + new + caption[m.end():], "the caption's section order follows the lyrics (%s)" % " ".join(tags)


def own_lyrics_ignored(lyrics, lyrics_external):
    """v1096: the field has words of its own but the wired lyrics replace it -> a note, else ""."""
    if lyrics_external is None or not clean_text(lyrics_external).strip():
        return ""
    own = clean_text(lyrics)
    if not own.strip() or " ".join(own.split()) == " ".join(clean_text(lyrics_external).split()):
        return ""
    return "the lyrics typed in this node's own field are NOT sent -- lyrics_external is wired (edit them there)"


def sections_of(lyrics):
    """The section tags the lyrics use, in order (what drives the structure)."""
    return re.findall(r"\[[^\]\n]+\]", lyrics or "")


def safe_name(name, default="score"):
    s = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name or "")).strip("_")
    return s or default


def next_path(folder, name, ext):
    """<folder>/<name>_NNNNN_.<ext>, the next free counter (never overwrites)."""
    rx = re.compile(r"^" + re.escape(name) + r"_(\d{5})_\.[A-Za-z0-9]+$")
    n = 0
    if os.path.isdir(folder):
        for f in os.listdir(folder):
            m = rx.match(f)
            if m:
                n = max(n, int(m.group(1)))
    return os.path.join(folder, "%s_%05d_.%s" % (name, n + 1, ext))


def provenance(meta):
    """The label and the recipe that go into the file and the sidecar."""
    out = {"label": AI_LABEL, "model": MODEL_LABEL}
    for k in ("caption", "lyrics", "seed", "seconds", "asked_seconds", "cfg_scale", "top_k", "sample_rate",
              "audio_input", "caption_mix", "trigger_words"):
        if k in (meta or {}):
            out[k] = meta[k]
    return out


MODEL_EXTS = (".safetensors", ".sft", ".gguf", ".ckpt", ".pt", ".pth", ".bin")
SAMPLER_KEYS = ("seed", "noise_seed", "steps", "cfg", "sampler_name", "scheduler", "denoise")
RECIPE_TAG = "polyhedron_music"
READABLE_TAG = "polyhedron_recipe"


def _link_src(v):
    """[node_id, slot] of an API-prompt input, or None for a plain value."""
    if isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[1], int) and isinstance(v[0], (str, int)):
        return str(v[0])
    return None


def upstream_ids(prompt, start):
    """Every node the given node depends on (itself excluded), in the executed prompt."""
    seen, todo = set(), [str(start)]
    while todo:
        nid = todo.pop()
        node = (prompt or {}).get(nid) or {}
        for v in (node.get("inputs") or {}).values():
            src = _link_src(v)
            if src is not None and src not in seen and src in prompt:
                seen.add(src)
                todo.append(src)
    return seen


def _plain(prompt, v, depth=0):
    """A plain input value; a link to a node whose single value input is plain (a seed node) is followed once."""
    src = _link_src(v)
    if src is None:
        return v
    if depth > 2:
        return None
    ins = ((prompt or {}).get(src) or {}).get("inputs") or {}
    for k in ("seed", "noise_seed", "value", "int", "number"):
        if k in ins:
            return _plain(prompt, ins[k], depth + 1)
    return None


def sampler_recipe(prompt, save_id):
    """The sampler that made the Save's 'samples': class + seed/steps/cfg/sampler/scheduler/denoise."""
    node = (prompt or {}).get(str(save_id)) or {}
    src = _link_src((node.get("inputs") or {}).get("samples"))
    hops = 0
    while src is not None and hops < 8:
        n = (prompt or {}).get(src) or {}
        ins = n.get("inputs") or {}
        if any(k in ins for k in ("seed", "noise_seed")) and "steps" in ins:
            out = {"node": src, "class": n.get("class_type")}
            for k in SAMPLER_KEYS:
                if k in ins:
                    out["seed" if k == "noise_seed" else k] = _plain(prompt, ins[k])
            return out
        # a pass-through node (latent op) between: follow its first LATENT-ish link
        nxt = None
        for k in ("samples", "latent", "latent_image"):
            nxt = _link_src(ins.get(k))
            if nxt is not None:
                break
        src, hops = nxt, hops + 1
    return {}


def model_files(prompt, save_id):
    """The model files the song came from: every string input upstream that names one."""
    files = []
    for nid in sorted(upstream_ids(prompt, save_id), key=lambda x: (len(x), x)):
        for v in (((prompt or {}).get(nid) or {}).get("inputs") or {}).values():
            if isinstance(v, str) and v.lower().endswith(MODEL_EXTS) and v not in files:
                files.append(v)
    return files


def recipe(meta, prompt=None, save_id=None):
    """The whole recipe: the Music Prompt's part, the sampler's part, the files."""
    r = provenance(meta)
    if prompt and save_id is not None:
        smp = sampler_recipe(prompt, save_id)
        if smp:
            r["sampler"] = smp
        fl = model_files(prompt, save_id)
        if fl:
            r["files"] = fl
    return r


def readable(rec):
    """One line any player shows: the label first (the licence), then both seeds."""
    parts = [rec.get("label", AI_LABEL)]
    if "seed" in rec:
        parts.append("text seed %s" % rec["seed"])
    smp = rec.get("sampler") or {}
    if "seed" in smp:
        parts.append("sampler seed %s" % smp["seed"])
    bits = []
    for k, fmt in (("steps", "%s steps"), ("cfg", "cfg %s"), ("sampler_name", "%s"), ("scheduler", "%s")):
        if smp.get(k) is not None:
            bits.append(fmt % smp[k])
    if bits:
        parts.append(" ".join(bits))
    if "cfg_scale" in rec:
        parts.append("text cfg %s top_k %s" % (rec.get("cfg_scale"), rec.get("top_k")))
    if "seconds" in rec:
        parts.append("%.2f s" % float(rec["seconds"]))
    return " | ".join(parts)


def metadata_allowed():
    """The core's --disable-metadata, honoured."""
    try:
        from comfy.cli_args import args
        return not bool(getattr(args, "disable_metadata", False))
    except Exception:
        return True


def embed_tags(rec, prompt=None, extra_pnginfo=None, allowed=True):
    """The tags the file carries. The label always (the licence); the recipe,
    the prompt and the workflow unless --disable-metadata. JSON ASCII-escaped."""
    tags = {"comment": rec.get("label", AI_LABEL), "encoded_by": rec.get("model", MODEL_LABEL)}
    if not allowed:
        return tags
    tags[RECIPE_TAG] = json.dumps(rec, ensure_ascii=True)
    tags[READABLE_TAG] = readable(rec)
    if prompt is not None:
        tags["prompt"] = json.dumps(prompt, ensure_ascii=True)
    for k, v in (extra_pnginfo or {}).items():
        tags[str(k)] = json.dumps(v, ensure_ascii=True)
    return tags


def sidecar_doc(rec, extra_pnginfo=None, allowed=True):
    """The sidecar: the workflow itself (loadable by drop) with the recipe under
    extra.polyhedron_music -- or, with no workflow / metadata off, the recipe alone."""
    wf = (extra_pnginfo or {}).get("workflow") if allowed else None
    if isinstance(wf, dict) and "nodes" in wf:
        doc = json.loads(json.dumps(wf))
        extra = doc.get("extra")
        if not isinstance(extra, dict):
            extra = {}
            doc["extra"] = extra
        extra[RECIPE_TAG] = rec
        return doc
    return rec


def recipe_of(doc):
    """The recipe back from a sidecar of either shape."""
    if isinstance(doc, dict) and "nodes" in doc:
        return ((doc.get("extra") or {}).get(RECIPE_TAG)) or {}
    return doc if isinstance(doc, dict) else {}


# ------------------------------------------------- core-free building blocks (v1060)
ZERO_KEYS = ("pooled_output", "conditioning_lyrics", "conditioning_scale", "direct_context")


def zero_conditioning(conditioning):
    """The negative: every tensor of the conditioning zeroed -- the main one and
    the named extras (the set the core's zero-out node handles, master's
    direct_context included). Written here: no core node class."""
    import torch
    out = []
    for t in conditioning:
        d = dict(t[1])
        for k in ZERO_KEYS:
            if torch.is_tensor(d.get(k)):
                d[k] = torch.zeros_like(d[k])
        out.append([torch.zeros_like(t[0]), d])
    return out


def decode_music(vae, samples, tile=None, overlap=None):
    """Music latent -> AUDIO. vae.decode / vae.decode_tiled (the comfy.sd model
    library), channels first, and the loudness rule the core applies to every
    decoded audio: a std above 0.2 is scaled down to 0.2 (std*5 > 1 divides).
    The Core-backed guard checks this equals the core's audio decode."""
    import torch
    lat = samples["samples"]
    if getattr(lat, "is_nested", False):
        lat = lat.unbind()[-1]
    if tile is not None:
        wav = vae.decode_tiled(lat, tile_x=tile, tile_y=tile, overlap=overlap)
    else:
        wav = vae.decode(lat)
    wav = wav.movedim(-1, 1)
    scale = torch.clamp(torch.std(wav, dim=[1, 2], keepdim=True) * 5.0, min=1.0)
    wav = wav / scale
    sr = samples.get("sample_rate") or getattr(vae, "audio_sample_rate_output", None) \
        or getattr(vae, "audio_sample_rate", 44100)
    return {"waveform": wav, "sample_rate": int(sr)}


def fit_audio_latent(audio, frames_latent):
    """The input latent at exactly the song's latent length. -> (tensor, used_s,
    input_s, note). Longer input: cut (the part past the song breaks at high
    denoise, S0). Shorter: zeros after it -- generated from the caption alone."""
    import torch
    z = audio["samples"]
    if z.dim() == 2:
        z = z.unsqueeze(0)
    have = int(z.shape[-1])
    fps = 44100.0 / 512.0
    if have >= frames_latent:
        out = z[..., :frames_latent]
        note = ("input cut to the song: %.2f of %.2f s used" % (frames_latent / fps, have / fps)
                if have > frames_latent else "")
        used = frames_latent
    else:
        pad = torch.zeros(z.shape[:-1] + (frames_latent - have,), dtype=z.dtype, device=z.device)
        out = torch.cat([z, pad], dim=-1)
        used = have
        note = ("input shorter than the song: %.2f of %.2f s from the input, the rest from the caption alone "
                "-- wire the encoder's seconds into seconds_in to ask for the input's length"
                % (have / fps, frames_latent / fps))
    return out.contiguous(), used / fps, have / fps, note


def with_triggers(caption, trigger_words):
    """v1108 -- the trigger words of a LoRA go to the FRONT of the caption, the
    place the trainer put them. Comma-separated; a word already in the caption
    (case-insensitive, as a whole word) is not added twice. -> (caption, added)."""
    if trigger_words is None:
        return caption, []
    seen, words = set(), []
    for w in re.split(r"[,\n]+", str(trigger_words)):
        w = w.strip()
        if w and w.lower() not in seen:
            seen.add(w.lower())
            words.append(w)
    low = caption.lower()
    add = [w for w in words if not re.search(r"(?<![\w])%s(?![\w])" % re.escape(w.lower()), low)]
    if not add:
        return caption, []
    return ", ".join(add) + ", " + caption, add


def info_lines(meta):
    L = ["%s: asked %.2f s (%s) -> generated %.2f s (%d frames)" % (
        MODEL_LABEL, meta["asked_seconds"], meta.get("source", "widget"), meta["seconds"], meta["frames"]),
        "seed %s  cfg %.2f  top_k %d" % (meta["seed"], meta["cfg_scale"], meta["top_k"]),
        "sections: " + (" ".join(meta.get("sections") or []) or "(no tags -- the model decides the structure)")]
    if meta["seconds"] + 0.5 < meta["asked_seconds"]:
        L.append("note: the model ended the song %.2f s early -- the asked length is a ceiling: the song ends when "
                 "its words are sung (more lines = a longer song; Music Lyrics writes them for its seconds)"
                 % (meta["asked_seconds"] - meta["seconds"]))
    if meta.get("lyrics_source", "field") != "field":
        L.append("lyrics: %s" % meta["lyrics_source"])
    if meta.get("trigger_words"):              # v1108
        L.append("trigger words at the caption's front: " + ", ".join(meta["trigger_words"]))
    for n in meta.get("notes") or []:          # v1096
        L.append("note: " + n)
    cm = meta.get("caption_mix")
    if cm:
        try:
            from . import ph_music_analyse_lib as _AL
        except ImportError:
            import ph_music_analyse_lib as _AL
        L.append("caption: a wired draft, %s -> %s" % (cm.get("mode"), _AL.merge_words(cm.get("where") or {})))
    ai = meta.get("audio_input")
    if ai:
        L.append("audio input: %s | strength %s -> denoise %s | %.2f of %.2f s used%s" % (
            ai.get("source") or "(no source given)", ai.get("strength"), ai.get("denoise"),
            ai.get("used_seconds", 0), ai.get("input_seconds", 0),
            (" | " + ai["note"]) if ai.get("note") else ""))
    return L


# ------------------------------------------------------------------- the nodes
class ULSMusicPrompt:
    """Caption + lyrics -> the music (conditioning), its negative and latent."""

    @classmethod
    def INPUT_TYPES(cls):
        fps, mx = core_limits()
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "The MiniMax-Music3 text encoder, loaded as CLIP with type 'minimax' "
                                             "(minimax_music3_text_encoder_pruned_int8_convrot for 16 GB cards). "
                                             "It is an 8B language model: it writes the music here."}),
                # v1076 (Frank, 10.10.: "Normalerweise liegen alle Knoten oberhalb der Felder" -- the
                # frontend drew each field's input slot ON its top edge, under the field): the text
                # fields carry NO slot (socketless); text by wire comes in at the top --
                # caption_external, lyrics_external
                "caption": ("STRING", {"multiline": True, "default": DEFAULT_CAPTION, "socketless": True,
                                       "tooltip": "What the music is, in three sections: Global Metadata (genre, "
                                                  "BPM, key, mood arc, use, production), Vocal Details, Arrangement "
                                                  "(instruments and what each section does). The more concrete, "
                                                  "the closer. '//' lines are notes to yourself and are removed."}),
                "lyrics": ("STRING", {"multiline": True, "default": DEFAULT_LYRICS, "socketless": True,
                                      "tooltip": "Section tags drive the structure: [Intro] [Verse] [Pre-Chorus] "
                                                 "[Chorus] [Post-Chorus] [Bridge] [Instrumental] [Solo] [Outro]. "
                                                 "The words only carry the mood. For a score without voice use "
                                                 "[Instrumental] / [Solo] sections. '//' lines are removed."}),
                "seconds": ("FLOAT", {"default": 60.0, "min": 1.0, "max": float(mx) / fps, "step": 0.5,
                                      "tooltip": "How long the music may be (up to %d s). The model can end "
                                                 "earlier -- the latent goes out at the length it really got. "
                                                 "A wired 'seconds_in' wins over this." % (mx // fps)}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,
                                 "tooltip": "Same seed, same texts, same length -> the same music."}),
                "cfg_scale": ("FLOAT", {"default": 1.7, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01,
                                        "tooltip": "How strictly the language model follows caption and lyrics "
                                                   "while it writes (the official template: 1.7 -- free enough to "
                                                   "drift toward its own habits; 2.5-3.5 holds it to the words). "
                                                   "The Music Sheet's 'adherence' sets it: wire its cfg_scale "
                                                   "output here (convert this widget to an input)."}),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 1024,
                                  "tooltip": "How many candidate codes the language model picks from at each "
                                             "step (the official template: 50). Lower = safer, higher = wilder."}),
            },
            "optional": {
                "seconds_in": ("FLOAT", {"forceInput": True,
                                         "tooltip": "Wire the Cine Timeline's 'seconds' here: the music gets the "
                                                    "film's length. Wins over the 'seconds' widget when > 0."}),
                "audio_latent": ("LATENT", {"tooltip": "From the Music Audio Encode: the latent output is then "
                                                       "YOUR AUDIO, cut to the length the language model wrote "
                                                       "(the part past it breaks at high denoise). Wire the "
                                                       "encoder's denoise into the sampler."}),
                # v1069 (S2), appended (#577): the Music Analyse's draft by wire, merged per rubric
                "caption_external": ("STRING", {"forceInput": True,
                                                "tooltip": "A caption by wire: the Music Sheet's caption (with "
                                                           "caption_mix 'draft only') or the Music Analyse's draft. It is "
                                                           "merged with the caption above FIELD BY FIELD in MiniMax's own "
                                                           "caption schema (Global Metadata: Basic Attributes, ...; "
                                                           "Vocal Details: ...; Arrangement: Primary, ...) as "
                                                           "caption_mix says. A field that still says '(not measured "
                                                           "yet' is never sent to the composer."}),
                "caption_mix": (list(AL.MIX_MODES), {"default": AL.MIX_DEFAULT,
                                                     "tooltip": "How the wired draft and your caption meet, field by "
                                                                "field. 'your text wins, draft fills' (default): your "
                                                                "fields as written, the draft's where you wrote none "
                                                                "(Basic Attributes part by part: bpm, key, scale, genre). "
                                                                "'combine': your field AND the draft's clean measured "
                                                                "sentences appended -- uncertain leans and placeholders "
                                                                "never. 'draft wins, your text fills': the measured "
                                                                "draft, your text where it is incomplete. 'draft only' / "
                                                                "'your text only': no mixing. The template's section "
                                                                "times are dropped when the song's length differs by "
                                                                "more than 15 %. Without a wired draft the caption above "
                                                                "is used as is."}),
                # v1076, appended (#577 -- an input, not a widget): lyrics by wire, at the top
                "lyrics_external": ("STRING", {"forceInput": True,
                                               "tooltip": "Lyrics by wire (an LLM, a text file, a Show Text): when "
                                                          "wired and not empty they REPLACE the lyrics field. "
                                                          "'//' lines are removed."}),
                # v1108, appended (#577 -- an input, not a widget): a LoRA's trigger words
                "trigger_words": ("STRING", {"forceInput": True,
                                             "tooltip": "The trigger words of the LoRAs on this text encoder -- wire "
                                                        "the LoRA Stack's trigger_words here. They go to the FRONT of "
                                                        "the caption, where training put them; a word the caption "
                                                        "already has is not added twice."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "FLOAT", "STRING")
    RETURN_NAMES = ("positive", "negative", "latent", "seconds", "info")
    OUTPUT_TOOLTIPS = (
        "The music the language model wrote -- to the sampler's positive.",
        "Its zeroed twin -- to the sampler's negative (as the official template does).",
        "An empty music latent of exactly the length the song got -- to the sampler. With audio_latent: "
        "your audio at that length instead.",
        "The length the song really got, in seconds.",
        "What was asked and what came out; wire it into the Music Save for the file's label.",
    )
    FUNCTION = "encode"
    CATEGORY = "Polyhedron/Music"
    DESCRIPTION = (
        "MiniMax-Music3: caption + lyrics -> positive, negative and an empty latent of the length the "
        "song really got. The text encoder writes the music; the sampler refines it (30 steps, cfg 1.7, "
        "euler / simple in the official template); the Music Save decodes and files it. '//' lines are "
        "comments. Wire the Cine Timeline's seconds to score a whole film. With an audio_latent from the "
        "Music Audio Encode the latent is your audio at the song's length (audio to music). From a "
        "template: Music Analyse -> Music Sheet (you check and lock the fields) -> caption_external with "
        "caption_mix 'draft only' (any other mix merges it with the caption field by field) and "
        "lyrics_external -- the template reaches the composer as text."
    )

    @staticmethod
    def merged_caption(caption, caption_external, caption_mix, seconds=None):
        """The caption the model gets: the widget alone, or the field-by-field merge with a wired
        draft (v1073: the official schema; the template's section times only when the song is
        about as long -- seconds). -> (text, where) with where = None when nothing was wired."""
        own = clean_text(caption)
        if caption_external is None or not str(caption_external).strip():
            return own, None
        return AL.merge_captions(own, clean_text(caption_external), caption_mix or AL.MIX_DEFAULT, seconds=seconds)

    @staticmethod
    def chosen_lyrics(lyrics, lyrics_external):
        """v1076: the wired lyrics, when wired and not empty after cleaning, replace the field."""
        ext = clean_text(lyrics_external) if lyrics_external is not None else ""
        if ext.strip():
            return ext, "wired (lyrics_external)"
        return clean_text(lyrics), "field"

    def encode(self, clip, caption, lyrics, seconds, seed, cfg_scale, top_k, seconds_in=None, audio_latent=None,
               caption_external=None, caption_mix=None, lyrics_external=None, trigger_words=None):
        import torch
        import comfy.model_management as mm
        from comfy.ldm.minimax_music.dit import latent_length
        fps, mx = core_limits()
        tok = type(getattr(clip, "tokenizer", None)).__name__
        if "MiniMaxMusic3" not in tok:
            raise ValueError("This CLIP is not the MiniMax-Music3 text encoder (its tokenizer is %s). Load "
                             "minimax_music3_text_encoder_*.safetensors with type 'minimax'." % tok)
        lyr, lyr_src = self.chosen_lyrics(lyrics, lyrics_external)
        want, source = want_seconds(seconds, seconds_in)
        cap, where = self.merged_caption(caption, caption_external, caption_mix, seconds=want)
        cap, sec_note = align_sections(cap, lyr)                               # v1096
        cap, trig_added = with_triggers(cap, trigger_words)                   # v1108
        notes = [n for n in (own_lyrics_ignored(lyrics, lyrics_external), sec_note) if n]
        frames = asked_frames(want, fps, mx)
        # core's MiniMaxMusic3TextEncode.execute, the same calls (the guard
        # compares them with the core's source)
        tokens = clip.tokenize(cap, lyrics=lyr, seed=seed, max_audio_frames=frames, cfg_scale=cfg_scale, top_k=top_k)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        for cond in conditioning:
            hidden = cond[0]
            cond[1]["conditioning_scale"] = torch.ones((hidden.shape[0], 1, 1), device=hidden.device, dtype=hidden.dtype)
        got = int(conditioning[0][0].shape[1])
        # the negative: zeroed here (v1060 -- no core node class)
        negative = zero_conditioning(conditioning)
        # the latent: empty at the length the song got -- or, v1060, the audio input at that length
        n_lat = latent_length(got)
        audio_input = None
        if audio_latent is not None and audio_latent.get("samples") is not None:
            lat_t, used_s, in_s, note = fit_audio_latent(audio_latent, n_lat)
            latent = lat_t.to(device=mm.intermediate_device(), dtype=mm.intermediate_dtype())
            audio_input = dict(audio_latent.get("polyhedron_audio") or {})
            audio_input.update({"used_seconds": round(used_s, 2), "input_seconds": round(in_s, 2), "note": note})
        else:
            latent = torch.zeros((1, 128, n_lat), device=mm.intermediate_device(), dtype=mm.intermediate_dtype())
        meta = {"caption": cap, "lyrics": lyr, "seed": int(seed), "cfg_scale": float(cfg_scale), "top_k": int(top_k),
                "asked_seconds": frames / float(fps), "source": source, "seconds": got / float(fps), "frames": got,
                "sections": sections_of(lyr), "lyrics_source": lyr_src, "notes": notes}
        if audio_input:
            meta["audio_input"] = audio_input
        if where is not None:
            meta["caption_mix"] = {"mode": caption_mix or AL.MIX_DEFAULT, "where": where}
            for c in where.get("conflicts") or []:     # v1079: never quiet about a contradiction
                print("[PLS] Music Prompt: CONFLICT -- %s (mode '%s' decided; check the caption field)"
                      % (c, caption_mix or AL.MIX_DEFAULT))
        if trig_added:                              # v1108 (v1110: after the conflict print)
            meta["trigger_words"] = trig_added
        info = "\n".join(info_lines(meta)) + "\n" + json.dumps(provenance(meta))
        print("[PLS] Music Prompt: " + " | ".join(info_lines(meta)))
        return (conditioning, negative, {"samples": latent, "type": "audio", "downscale_ratio_temporal": 512},
                got / float(fps), info)


class ULSMusicSave:
    """Latent + VAE (or AUDIO) -> output/<project>/music/, labelled, with a player."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project": ("STRING", {"default": "cine",
                                       "tooltip": "output/<project>/music/ -- use the Cine Timeline's project, "
                                                  "and the score sits next to the film's clips."}),
                "name": ("STRING", {"default": "score", "tooltip": "The file name; a counter is added, nothing is overwritten."}),
                "format": (list(FORMATS), {"default": "flac",
                                           "tooltip": "flac: lossless; mp3: small -- both carry the label, the recipe "
                                                      "(both seeds) and the whole workflow: drop the file on the canvas "
                                                      "and the workflow is back. wav: plain PCM, nothing inside -- the "
                                                      "way back is the sidecar .json (drop that instead)."}),
                "tiled_decode": ("BOOLEAN", {"default": True,
                                             "tooltip": "Decode in overlapping tiles: far less VRAM for long music "
                                                        "(the official template's setting); off = one piece."}),
                "tile_size": ("INT", {"default": 1536, "min": 32, "max": 8192, "step": 8,
                                      "tooltip": "Latent frames per tile (86 frames = 1 s)."}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 8,
                                    "tooltip": "Latent frames the tiles share, against seams."}),
            },
            "optional": {
                "samples": ("LATENT", {"tooltip": "The sampler's music latent (with 'vae')."}),
                "vae": ("VAE", {"tooltip": "minimax_music3_dav -- the music decoder."}),
                "audio": ("AUDIO", {"tooltip": "Or an AUDIO that is already decoded (then samples/vae are not needed)."}),
                "info": ("STRING", {"forceInput": True,
                                    "tooltip": "The Music Prompt's info: caption, lyrics, seed and length go into "
                                               "the file's metadata and the sidecar .json."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "path")
    OUTPUT_TOOLTIPS = ("The music as AUDIO (e.g. into the Save's audio for a video).",
                       "Where the file went.")
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "Polyhedron/Music"
    DESCRIPTION = (
        "MiniMax-Music3: decodes the music latent (tiled if asked) or takes an AUDIO, and files it under "
        "output/<project>/music/ with a player in the node. The file (flac / mp3) carries the label "
        "'AI-generated with MiniMax-Music3' (the model's licence asks for it), the recipe with both seeds and "
        "the whole workflow -- drop it on the canvas to get the workflow back. The sidecar .json is the "
        "workflow too (the way back for a wav)."
    )

    def save(self, project, name, format, tiled_decode, tile_size, overlap,
             samples=None, vae=None, audio=None, info=None, prompt=None, extra_pnginfo=None, unique_id=None):
        if audio is None:
            if samples is None or vae is None:
                raise ValueError("Music Save needs either 'audio', or 'samples' AND 'vae' (the minimax_music3_dav).")
            audio = decode_music(vae, samples, tile_size if tiled_decode else None, overlap if tiled_decode else None)
        meta = self._meta(info)
        meta["sample_rate"] = int(audio["sample_rate"])
        import folder_paths
        try:
            from .cine_timeline import project_name
        except ImportError:
            from cine_timeline import project_name
        out_dir = folder_paths.get_output_directory()
        proj = project_name({"project": project})
        folder = os.path.join(out_dir, proj, MUSIC_SUBFOLDER)
        os.makedirs(folder, exist_ok=True)
        fmt = format if format in FORMATS else "flac"
        path = next_path(folder, safe_name(name), fmt)
        allowed = metadata_allowed()
        rec = recipe(meta, prompt, unique_id)
        tags = embed_tags(rec, prompt, extra_pnginfo, allowed)
        tags["title"] = os.path.splitext(os.path.basename(path))[0]
        written = self._write(path, audio, fmt, tags)
        with open(os.path.splitext(written)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump(sidecar_doc(rec, extra_pnginfo, allowed), f, indent=2, ensure_ascii=True)
        if allowed and "workflow" in (extra_pnginfo or {}):
            print("[PLS] Music Save: the workflow is in the %s%s -- drop it on the canvas to load it back (%s)"
                  % ("file and " if written.endswith((".flac", ".mp3")) else "", "sidecar .json", readable(rec)))
        wav = audio["waveform"]
        secs = wav.shape[-1] / float(audio["sample_rate"])
        rel = os.path.relpath(os.path.dirname(written), out_dir).replace(os.sep, "/")
        print("[PLS] Music Save: %s (%.2f s, %d Hz, %d ch) -- %s" % (os.path.join(rel, os.path.basename(written)),
                                                                   secs, meta["sample_rate"], wav.shape[-2], AI_LABEL))
        ui = {"pls_music": [{"filename": os.path.basename(written), "subfolder": rel, "type": "output",
                             "seconds": round(secs, 3), "sample_rate": meta["sample_rate"], "label": AI_LABEL}]}
        return {"ui": ui, "result": (audio, written)}

    @staticmethod
    def _meta(info):
        """The recipe from the Music Prompt's info (its last line is JSON)."""
        if not info:
            return {}
        for line in reversed(str(info).splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    d = json.loads(line)
                    return {k: v for k, v in d.items() if k not in ("label", "model")}
                except Exception:
                    return {}
        return {}

    @staticmethod
    def _write(path, audio, fmt, tags):
        """flac / mp3 via PyAV with the tags as metadata; wav via the stdlib.
        A failed PyAV write falls back to wav (and says so)."""
        import numpy as np
        wav = audio["waveform"]
        if hasattr(wav, "cpu"):
            wav = wav.float().cpu().numpy()
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim == 3:
            wav = wav[0]
        sr = int(audio["sample_rate"])
        if fmt in ("flac", "mp3"):
            try:
                import av
                c = av.open(path, mode="w")
                try:
                    for k, v in tags.items():
                        c.metadata[k] = v
                    layout = "stereo" if wav.shape[0] == 2 else "mono"
                    s = c.add_stream("flac" if fmt == "flac" else "libmp3lame", rate=sr, layout=layout)
                    if fmt == "mp3":
                        s.bit_rate = 320000
                    # planar float [channels, samples]; the encoder converts to its own format
                    frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(np.clip(wav, -1.0, 1.0)),
                                                       format="fltp", layout=layout)
                    frame.sample_rate = sr
                    for pkt in s.encode(frame):
                        c.mux(pkt)
                    for pkt in s.encode(None):
                        c.mux(pkt)
                finally:
                    c.close()
                return path
            except Exception as e:
                print("[PLS] Music Save: %s failed (%r) -- writing wav, the label is in the sidecar." % (fmt, e))
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                except Exception:
                    pass
                path = os.path.splitext(path)[0] + ".wav"
        import wave
        pcm = (np.clip(wav.T, -1.0, 1.0) * 32767.0).astype("<i2")
        with wave.open(path, "wb") as f:
            f.setnchannels(int(wav.shape[0]))
            f.setsampwidth(2)
            f.setframerate(sr)
            f.writeframes(pcm.tobytes())
        return path


class ULSMusicAudioEncode:
    """AUDIO -> a MiniMax-Music3 latent (audio to music), with the denoise the strength asks for."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The music to start from -- any sample rate, mono or stereo "
                                               "(e.g. from the Media Loader). Use material you may use: your own, "
                                               "public domain or a Creative Commons licence that allows it."}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001,
                                       "tooltip": "How far the caption may move away from your audio. 0 = your "
                                                  "audio stays, 0.5 = a true mix, 1 = the caption alone. Mapped "
                                                  "onto the narrow window where Music3 actually changes "
                                                  "(denoise ~0.80-0.96, scaled by your audio's own level); the "
                                                  "node shows the resulting denoise."}),
                "start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.5,
                                    "tooltip": "Seconds into the audio where the excerpt begins."}),
                "length": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 0.5,
                                     "tooltip": "Seconds of audio to use; 0 = to the end (at most 360 s, "
                                                "the longest song the language model writes)."}),
                "source": ("STRING", {"default": "",
                                      "tooltip": "Where the audio comes from and its licence (e.g. 'Freesound "
                                                 "#12345, CC0' or 'own recording'). Goes into the music file's "
                                                 "metadata next to the AI label."}),
            },
        }

    RETURN_TYPES = ("LATENT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("audio_latent", "denoise", "seconds", "info")
    OUTPUT_TOOLTIPS = (
        "Your audio as a Music3 latent -- into the Music Prompt's audio_latent.",
        "The denoise the strength asks for -- into the sampler's denoise.",
        "The length of the excerpt -- into the Music Prompt's seconds_in, so the song is asked for that length.",
        "What was encoded (length, level, the denoise, notes).",
    )
    FUNCTION = "encode"
    CATEGORY = "Polyhedron/Music"
    DESCRIPTION = (
        "MiniMax-Music3, audio to music: encodes your audio with the MiniMax-Music3 DAV encoder into the "
        "latent the sampler refines. Wire audio_latent into the Music Prompt, denoise into the sampler and "
        "seconds into the Music Prompt's seconds_in. strength 0 keeps your audio, 1 lets the caption take "
        "over. The encoder comes from MiniMax's dav.pth -- Download in the node fetches it once and converts "
        "it to safetensors; a run never downloads."
    )

    def encode(self, audio, strength, start, length, source=""):
        import torch
        try:
            from . import ph_music_dav as MD
        except ImportError:
            import ph_music_dav as MD
        path = MD.ensure_full(convert_user_file=True)
        if not path:
            raise RuntimeError(MD.missing_text())
        if audio is None or audio.get("waveform") is None:
            raise ValueError("Music Audio Encode: no audio on the input.")
        ln = float(length or 0)
        if ln <= 0 or ln > 360.0:
            ln = 360.0
        wav, notes = MD.prepare_waveform(audio["waveform"], audio.get("sample_rate", MD.SR), start, ln)
        try:
            import comfy.model_management as mm
            dev = mm.get_torch_device()
        except Exception:
            dev = torch.device("cpu")
        model = MD.load_encoder(path, dev)
        with torch.no_grad():
            z = MD.encode_chunked(model, wav.to(dev)).float().cpu()
        del model
        std = float(z.std())
        std_law, std_note = MD.clamp_std(std)
        if std_note:
            notes.append(std_note)
        dn = MD.strength_to_denoise(strength, std_law)
        secs = wav.shape[-1] / float(MD.SR)
        meta = {"source": str(source or "").strip(), "strength": round(float(strength), 3), "denoise": dn,
                "latent_std": round(std, 4), "law_std": round(std_law, 4), "seconds": round(secs, 2),
                "start": float(start or 0)}
        lines = ["audio -> Music3 latent: %.2f s (%d frames) from %.2f s in" % (secs, z.shape[-1], float(start or 0)),
                 "strength %.2f -> denoise %.3f (latent std %.2f; window measured with euler / simple)"
                 % (float(strength), dn, std),
                 "source: " + (meta["source"] or "(not given -- name it, the file's metadata carries it)")]
        lines += ["note: " + n for n in notes]
        info = "\n".join(lines)
        print("[PLS] Music Audio Encode: " + " | ".join(lines))
        latent = {"samples": z, "type": "audio", "downscale_ratio_temporal": MD.HOP, "polyhedron_audio": meta}
        ui = {"pls_audio_encode": [{"denoise": dn, "strength": float(strength), "seconds": round(secs, 2),
                                    "latent_std": round(std_law, 4)}]}
        return {"ui": ui, "result": (latent, dn, secs, info)}
