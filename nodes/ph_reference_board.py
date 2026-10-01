# -*- coding: ascii -*-
"""Polyhedron Reference Board (Cine C2, v1019).

Up to 9 images, 3 videos and 3 audios for MiniMax H3 -- as TILES inside one
node instead of 18 loose wires, each tile with a name, a role, a description,
a retention marker and (images) a megapixel budget. Research 26.09.: seven H3
reference managers exist, none joins a visual board, per-reference role /
description / budget AND a live prompt draft; and nobody renumbers safely.
We do the last part with @tags (v1017): the board hands its tags to the
Reference node, the Reference node puts the live labels in.

The board's state is ONE hidden JSON widget, `board` (web/js/ph_reference_
board.js draws and edits it). Files live in ComfyUI's input folder
(uploaded into input/pls_board by the board, or any file already there).

Outputs:
  refs         -> Polyhedron MiniMax Reference `refs`: the loaded media plus
                  their tags; the Reference node fills its free slots in board
                  order and learns the tags.
  definitions  -> a DRAFT of the official subject_definitions and
                  retention_analysis lines (ref-en.txt section 2 and 4), in
                  @tag form -- paste into the prompt or wire into a text
                  combine; the Reference node resolves the tags.
  info         -> what was loaded, with sizes and durations.
"""

import json
import os
import re

import numpy as np
import torch

from . import cine_clip as C
from . import h3_prompt as HP

N_IMAGES, N_VIDEOS, N_AUDIOS = 9, 3, 3
TAG_OK = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SLOT_NAME = re.compile(r"^(image|video|video_audio|audio)_\d+$|^(first|last)_frame$")
KINDS = ("image", "video", "audio")


def _base(kind):
    import folder_paths
    return {"input": folder_paths.get_input_directory(),
            "output": folder_paths.get_output_directory(),
            "temp": folder_paths.get_temp_directory()}.get(kind)


def file_path(ref):
    """Absolute path of a {filename, subfolder, type} ref; refuses anything
    that would leave its base folder (the name comes from the workflow)."""
    base = _base((ref or {}).get("type", "input"))
    if not base:
        raise ValueError("unknown folder type %r" % (ref or {}).get("type"))
    full = os.path.normpath(os.path.join(base, ref.get("subfolder", "") or "", ref.get("filename", "")))
    if not full.startswith(os.path.normpath(base) + os.sep):
        raise ValueError("board file %r leaves its folder -- refused" % ref.get("filename"))
    return full


def validate(entries):
    """Errors by name for a board state; [] when it can run. Pure."""
    errs = []
    counts = {k: 0 for k in KINDS}
    seen = {}
    for i, e in enumerate(entries, start=1):
        kind = e.get("kind")
        if kind not in KINDS:
            errs.append("tile %d has no kind (image/video/audio)" % i)
            continue
        counts[kind] += 1
        tag = str(e.get("tag", "")).strip()
        if not TAG_OK.match(tag):
            errs.append("tile %d: tag %r is not a name (letters, digits, _; not starting with a digit)" % (i, tag))
        elif SLOT_NAME.match(tag):
            errs.append("tile %d: tag %r is a slot name -- pick a speaking name" % (i, tag))
        elif tag in seen:
            errs.append("tile %d: tag @%s is already tile %d" % (i, tag, seen[tag]))
        else:
            seen[tag] = i
        if kind == "video" and e.get("sound") == "own":
            st = tag + "_sound"
            if st in seen:
                errs.append("tile %d: @%s (its soundtrack) is already taken" % (i, st))
            seen[st] = i
        if e.get("role") and e["role"] not in HP.ROLES[kind]:
            errs.append("tile %d: role %r is not one of %s" % (i, e["role"], ", ".join(HP.ROLES[kind])))
        ret = e.get("retention")
        pool = HP.RETENTION["audio" if kind == "audio" else "visual"]
        if ret and ret not in pool:
            errs.append("tile %d: retention %r is not one of %s" % (i, ret, ", ".join(pool)))
    for kind, lim in (("image", N_IMAGES), ("video", N_VIDEOS), ("audio", N_AUDIOS)):
        if counts[kind] > lim:
            errs.append("%d %s tiles -- MiniMax H3 takes at most %d" % (counts[kind], kind, lim))
    return errs


def _load_image(path):
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)[None]


def _audio_dict(path):
    a = C.load_audio(path)
    if a is None:
        return None
    return {"waveform": torch.from_numpy(a[0])[None], "sample_rate": int(a[1])}


def _video(path):
    from comfy_api.input_impl import VideoFromFile
    return VideoFromFile(path)


class ULSReferenceBoard:
    """Tiles for every MiniMax H3 reference, in one node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "board": ("STRING", {"default": "[]", "multiline": True,
                    "tooltip": "The board's state (JSON), edited by the tiles "
                               "above -- not by hand."}),
            },
        }

    RETURN_TYPES = ("PLS_REFS", "STRING", "STRING")
    RETURN_NAMES = ("refs", "definitions", "info")
    FUNCTION = "build"
    CATEGORY = "Polyhedron/Cine"
    DESCRIPTION = ("Up to 9 images, 3 videos and 3 audios for MiniMax H3 as "
                   "tiles in one node: a name (@tag), a role, a description, "
                   "the official retention marker and a megapixel budget per "
                   "tile, videos playing on mouse-over. `refs` feeds Polyhedron "
                   "MiniMax Reference, which fills its free slots in board order "
                   "and learns the tags; `definitions` drafts the official "
                   "subject_definitions and retention_analysis lines.")

    @classmethod
    def IS_CHANGED(cls, board):
        try:
            entries = json.loads(board or "[]")
            stamps = []
            for e in entries:
                p = file_path(e.get("file"))
                st = os.stat(p)
                stamps.append("%s|%d|%d" % (p, st.st_mtime_ns, st.st_size))
            return board + "||" + "|".join(stamps)
        except Exception:
            return board

    def build(self, board):
        try:
            entries = json.loads(board or "[]")
        except Exception as e:
            raise ValueError("[PLS] Reference Board: the board state is not "
                             "JSON (%s) -- re-add the tiles" % e)
        if not isinstance(entries, list):
            raise ValueError("[PLS] Reference Board: the board state is not a list")
        errs = validate(entries)
        if errs:
            raise ValueError("[PLS] Reference Board:\n  " + "\n  ".join(errs))

        refs = {"images": [], "videos": [], "audios": []}
        lines, draft = [], []
        for e in entries:
            path = file_path(e.get("file"))
            if not os.path.isfile(path):
                raise ValueError("[PLS] Reference Board: @%s -- file %s is gone"
                                 % (e["tag"], path))
            kind, tag = e["kind"], e["tag"]
            if kind == "image":
                img = _load_image(path)
                mp = float(e.get("mp", 0.0) or 0.0)
                refs["images"].append({"tag": tag, "image": img, "mp": mp,
                                       "role": e.get("role", ""), "desc": e.get("desc", "")})
                lines.append("@%s  image  %dx%d%s  %s" % (
                    tag, img.shape[2], img.shape[1],
                    ("  %.2f MP" % mp) if mp > 0 else "", os.path.basename(path)))
                draft.append(dict(e, sound_tag=None))
            elif kind == "video":
                sound = None
                if e.get("sound") == "own":
                    sound = _audio_dict(path)
                    if sound is None:
                        raise ValueError("[PLS] Reference Board: @%s asks for its own "
                                         "soundtrack, but %s has no audio track"
                                         % (tag, os.path.basename(path)))
                refs["videos"].append({"tag": tag, "video": _video(path),
                                       "sound": sound,
                                       "sound_tag": (tag + "_sound") if sound is not None else None,
                                       "role": e.get("role", ""), "desc": e.get("desc", "")})
                lines.append("@%s  video%s  %s" % (tag, "  + @%s_sound" % tag if sound is not None else "",
                                                     os.path.basename(path)))
                draft.append(dict(e, sound_tag=(tag + "_sound") if sound is not None else None))
            else:
                a = _audio_dict(path)
                if a is None:
                    raise ValueError("[PLS] Reference Board: @%s -- %s has no "
                                     "audio track" % (tag, os.path.basename(path)))
                refs["audios"].append({"tag": tag, "audio": a,
                                       "role": e.get("role", ""), "desc": e.get("desc", "")})
                lines.append("@%s  audio  %.2f s  %s" % (
                    tag, a["waveform"].shape[-1] / float(a["sample_rate"]), os.path.basename(path)))
                draft.append(dict(e, sound_tag=None))

        defs, keep = HP.definitions_block(draft)
        definitions = ""
        if defs:
            definitions = ("// subject_definitions\n" + "\n".join(defs)
                           + "\n// retention_analysis\n" + "\n".join(keep))
        info = "\n".join(["Reference Board: %d image(s), %d video(s), %d audio(s)"
                          % (len(refs["images"]), len(refs["videos"]), len(refs["audios"]))]
                         + ["  " + l for l in lines])
        print("[PLS] " + info.replace("\n", "\n[PLS] "))
        return (refs, definitions, info)


def apply_refs(refs, opt, mps):
    """Put a board's media into the Reference node's FREE slots, board order.

    opt: the node's optional kwargs (mutated: filled slots are set).
    mps: {slot_number: megapixels} (mutated for images with a board budget).
    Returns (aliases {tag: slot}, notes [str]). Raises by name when a kind
    has no free slot left. Pure but for the dict mutation -- the guard drives
    it without media.
    """
    aliases, notes = {}, []
    if not refs:
        return aliases, notes

    def free(prefix, count, also=None):
        for n in range(1, count + 1):
            if opt.get("%s_%d" % (prefix, n)) is None and (
                    also is None or opt.get("%s_%d" % (also, n)) is None):
                return n
        return None

    for it in refs.get("images", []):
        n = free("image", N_IMAGES)
        if n is None:
            raise ValueError("[PLS] MiniMax Reference: no free image slot for board @%s "
                             "(all %d taken by wires and the board)" % (it["tag"], N_IMAGES))
        opt["image_%d" % n] = it["image"]
        if it.get("mp", 0) > 0:
            mps[n] = it["mp"]
        aliases[it["tag"]] = "image_%d" % n
        notes.append("board @%s -> image_%d" % (it["tag"], n))
    for it in refs.get("videos", []):
        n = free("video", N_VIDEOS, "video_audio")
        if n is None:
            raise ValueError("[PLS] MiniMax Reference: no free video slot for board @%s"
                             % it["tag"])
        opt["video_%d" % n] = it["video"]
        aliases[it["tag"]] = "video_%d" % n
        if it.get("sound") is not None:
            opt["video_audio_%d" % n] = it["sound"]
            aliases[it["sound_tag"]] = "video_audio_%d" % n
        notes.append("board @%s -> video_%d%s" % (
            it["tag"], n, " (+ @%s -> video_audio_%d)" % (it["sound_tag"], n) if it.get("sound") is not None else ""))
    for it in refs.get("audios", []):
        n = free("audio", N_AUDIOS)
        if n is None:
            raise ValueError("[PLS] MiniMax Reference: no free audio slot for board @%s"
                             % it["tag"])
        opt["audio_%d" % n] = it["audio"]
        aliases[it["tag"]] = "audio_%d" % n
        notes.append("board @%s -> audio_%d" % (it["tag"], n))
    return aliases, notes


NODE_CLASS_MAPPINGS = {"ULSReferenceBoard": ULSReferenceBoard}
NODE_DISPLAY_NAME_MAPPINGS = {"ULSReferenceBoard": "\u2b21 Polyhedron Reference Board"}
