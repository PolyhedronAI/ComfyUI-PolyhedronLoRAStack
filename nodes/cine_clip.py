# -*- coding: ascii -*-
"""cine_clip -- the clip archive and the seam meter of the Cine chain (C3, v1018).

THE CLIP COMPANION. Every clip of a chain can carry a companion file next to
its video:  <stem>.plsclip.safetensors  holding the AV latent the sampler
produced (video + audio part, never decoded), the video file's md5, and a
small JSON header (size, frames, fps, recipe). The next clip reads its
predecessor's latent from there instead of re-encoding decoded pixels -- every
VAE round trip costs colour and sharpness, which is why every serious chain
tool keeps the latent (research 26.09.). The md5 ties the latent to exactly
one video file: a companion that no longer matches its video is reported, not
trusted.

THE SEAM METER. Our own measure from Cine_Feldbefunde_K1.md (20.09.): the mean
absolute luma difference between the last frame of clip n and the first frame
of clip n+1, read against the clip-internal band (frame 0 vs frame 1 of the new
clip). Field thresholds: <= 4 the anchor holds, >= 15 there was no anchor.
Plus the LEVEL STEP per channel over the first frames against where the
previous clip was heading (a line through its last frames) -- the brightness
bias at continuation seams that obvpm measured on H3 (median 2.87 against a
0.44 noise band, confined to ~12 frames). Measured here, corrected later (C4).

Pure numpy + safetensors + PyAV; the guard drives it without ComfyUI.
"""

import hashlib
import json
import os
import time
from collections import deque

import numpy as np

SUFFIX = ".plsclip.safetensors"
FORMAT = "plsclip/1"
SEAM_HOLDS = 4.0          # Cine_Feldbefunde_K1: anchor holds below this
SEAM_NO_ANCHOR = 15.0     # ... and there was no anchor above this
LEVEL_FRAMES = 12         # obvpm: the step is confined to ~12 frames
LEVEL_FIT = 8             # frames of the previous clip the trend is fitted on
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float64)


# --------------------------------------------------------------------------
# companion file
# --------------------------------------------------------------------------

def sidecar_path(video_path):
    return os.path.splitext(video_path)[0] + SUFFIX


def file_md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_sidecar(video_path, tensors, meta):
    """Write the companion atomically. tensors: {name: torch.Tensor}."""
    from safetensors.torch import save_file
    head = dict(meta)
    head["format"] = FORMAT
    head["video_file"] = os.path.basename(video_path)
    head["video_md5"] = file_md5(video_path) if os.path.isfile(video_path) else ""
    head["written"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    target = sidecar_path(video_path)
    tmp = target + ".tmp"
    save_file({k: v.detach().contiguous().cpu() for k, v in tensors.items()},
              tmp, metadata={"pls_meta": json.dumps(head, sort_keys=True)})
    os.replace(tmp, target)
    return target, head


def read_sidecar(video_path):
    """(tensors, meta, note) or (None, None, note). Never raises on a bad file."""
    target = sidecar_path(video_path)
    if not os.path.isfile(target):
        return None, None, "no companion %s" % os.path.basename(target)
    try:
        from safetensors import safe_open
        tensors = {}
        with safe_open(target, framework="pt") as f:
            meta = json.loads((f.metadata() or {}).get("pls_meta", "{}"))
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
    except Exception as e:
        return None, None, "companion unreadable (%s: %s)" % (type(e).__name__, e)
    if meta.get("format") != FORMAT:
        return None, None, "companion has format %r, expected %r" % (meta.get("format"), FORMAT)
    note = "companion %s" % os.path.basename(target)
    want = meta.get("video_md5", "")
    if want and os.path.isfile(video_path) and file_md5(video_path) != want:
        return tensors, meta, note + " -- MD5 MISMATCH: the video changed after the latent was stored; the latent is NOT this video's"
    return tensors, meta, note + " (md5 matches)"


# --------------------------------------------------------------------------
# tail of a clip, decoded without holding the whole clip
# --------------------------------------------------------------------------

def decode_tail(video_path, n_frames):
    """(frames uint8 [n,H,W,3], fps, total_frames, audio or None).

    Keeps a rolling window of the last n frames -- a 15 s clip at 1344x768 in
    float would be gigabytes; the tail is a few MB. audio: (waveform float32
    [C, L], sample_rate) of the WHOLE track (small), or None.
    """
    import av
    n_frames = max(1, int(n_frames))
    tail = deque(maxlen=n_frames)
    total = 0
    fps = 0.0
    audio = None
    with av.open(video_path) as c:
        vs = next((s for s in c.streams if s.type == "video"), None)
        if vs is None:
            raise ValueError("%s has no video stream" % video_path)
        fps = float(vs.average_rate or vs.guessed_rate or 0) or 0.0
        for fr in c.decode(vs):
            tail.append(fr.to_ndarray(format="rgb24"))
            total += 1
    audio = load_audio(video_path)
    if not tail:
        raise ValueError("%s decoded to zero frames" % video_path)
    return np.stack(list(tail), 0), fps, total, audio


def load_audio(path):
    """(waveform float32 [C, L], sample_rate) of a file's first audio stream,
    or None. One resampler to planar float: every codec's layout (packed s16,
    planar fltp, ...) arrives as [C, L]."""
    import av
    with av.open(path) as c:
        as_ = next((s for s in c.streams if s.type == "audio"), None)
        if as_ is None:
            return None
        rs = av.AudioResampler(format="fltp")
        chunks, sr = [], 0
        for fr in c.decode(as_):
            for out in rs.resample(fr):
                sr = int(out.sample_rate)
                chunks.append(out.to_ndarray().astype(np.float32))
        for out in rs.resample(None) or []:
            chunks.append(out.to_ndarray().astype(np.float32))
    if not chunks or sr <= 0:
        return None
    return np.concatenate(chunks, axis=1), sr


def audio_tail(audio, seconds):
    """Last `seconds` of an (wav [C,L], sr) pair, or None."""
    if audio is None:
        return None
    wav, sr = audio
    n = int(round(seconds * sr))
    if n <= 0:
        return None
    return wav[:, -n:] if wav.shape[1] > n else wav


# --------------------------------------------------------------------------
# seam meter
# --------------------------------------------------------------------------

def _as255(frames):
    """[N,H,W,C] -> float64 on 0..255 (float input in 0..1 is detected)."""
    f = np.asarray(frames, dtype=np.float64)[..., :3]
    if f.size and f.max() <= 1.0 + 1e-6:
        f = f * 255.0
    return f


def seam_metrics(prev_frames, next_frames):
    """Numbers for one seam. prev_frames: the END of clip n (>= 1 frame);
    next_frames: the START of clip n+1 (>= 1 frame). Same size required."""
    p = _as255(prev_frames)
    q = _as255(next_frames)
    if p.shape[1:3] != q.shape[1:3]:
        raise ValueError("seam: frame sizes differ (%s vs %s) -- measure on the "
                         "same canvas" % (p.shape[1:3], q.shape[1:3]))
    lum_p = p @ LUMA                      # [N,H,W]
    lum_q = q @ LUMA
    seam = float(np.abs(lum_p[-1] - lum_q[0]).mean())
    band = float(np.abs(lum_q[1] - lum_q[0]).mean()) if q.shape[0] > 1 else float("nan")
    cp = p.reshape(p.shape[0], -1, 3).mean(axis=1)   # channel means per frame
    cq = q.reshape(q.shape[0], -1, 3).mean(axis=1)
    # where clip n was HEADING: a line through its last frames, one step on --
    # not simply its last frame, or a clip walking into shade would be
    # "corrected" out of its own lighting change
    k = min(LEVEL_FIT, cp.shape[0])
    if k >= 2:
        x = np.arange(k, dtype=np.float64)
        pred = np.array([np.polyval(np.polyfit(x, cp[-k:, c], 1), k) for c in range(3)])
    else:
        pred = cp[-1]
    step0 = cq[0] - pred
    return {"seam": seam, "band": band,
            "level_step": [float(v) for v in step0],
            "level_step_max": float(np.abs(step0).max()),
            "frames_prev": int(p.shape[0]), "frames_next": int(q.shape[0])}


def seam_verdict(seam):
    if seam <= SEAM_HOLDS:
        return "seamless -- the anchor holds (<= %g)" % SEAM_HOLDS
    if seam < SEAM_NO_ANCHOR:
        return "soft -- visible step, between %g and %g" % (SEAM_HOLDS, SEAM_NO_ANCHOR)
    return "cut -- no anchor arrived (>= %g)" % SEAM_NO_ANCHOR


def seam_report(mx):
    r, g, b = mx["level_step"]
    return ["seam %.2f (clip-internal band %.2f) -> %s" % (mx["seam"], mx["band"], seam_verdict(mx["seam"])),
            "level step at the join R %+.2f G %+.2f B %+.2f (against the trend of the last %d frames of clip n)"
            % (r, g, b, min(LEVEL_FIT, mx["frames_prev"]))]


# --------------------------------------------------------------------------
# what the node SHOWS (v1018, Frank 26.09.: "Filmstreifen ... Zwischen-Videos
# einreihen ... bei Mouse-over abspielen -- so findet man sich besser zurueck
# als wenn alles nur lose ueber Tags laeuft")
# --------------------------------------------------------------------------

VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv", ".m4v")
STRIP_H = 72              # thumbnail height in px
REEL_LIMIT = 40           # newest N clips of a folder in the reel


def view_ref(path, bases):
    """{filename, subfolder, type} for ComfyUI's /view, or None when the file
    lies outside every served base. bases: [(type, dir), ...]."""
    ap = os.path.abspath(path)
    for kind, base in bases:
        if not base:
            continue
        b = os.path.abspath(base)
        if ap == b or not ap.startswith(b + os.sep):
            continue
        rel = os.path.relpath(ap, b)
        sub, name = os.path.split(rel)
        return {"filename": name, "subfolder": sub.replace(os.sep, "/"), "type": kind}
    return None


def reel_items(path, bases, limit=REEL_LIMIT):
    """The chain around `path`: every video in its folder, oldest first (the
    order clips were made in), newest `limit` kept. Pure but for os.listdir."""
    path = (path or "").strip().strip('"')
    folder = os.path.dirname(os.path.abspath(path)) if path else ""
    if not folder or not os.path.isdir(folder):
        return []
    cur = os.path.abspath(path)
    items = []
    for name in os.listdir(folder):
        if not name.lower().endswith(VIDEO_EXTS):
            continue
        full = os.path.join(folder, name)
        try:
            mt = os.path.getmtime(full)
        except OSError:
            continue
        items.append({"name": name, "path": full, "mtime": mt,
                      "latent": os.path.isfile(sidecar_path(full)),
                      "current": os.path.abspath(full) == cur,
                      "view": view_ref(full, bases)})
    items.sort(key=lambda d: (d["mtime"], d["name"]))
    return items[-int(limit):]


def write_previews(frames, out_dir, prefix, height=STRIP_H):
    """Small JPEGs of frames [N,H,W,3] (uint8 or float 0..1) for the strip.
    Returns the file names, in order. Never raises into a run."""
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    arr = np.asarray(frames)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * (255.0 if arr.max() <= 1.0 + 1e-6 else 1.0), 0, 255).astype(np.uint8)
    names = []
    for i, fr in enumerate(arr):
        im = Image.fromarray(fr[..., :3])
        w = max(1, int(round(im.width * height / float(im.height))))
        im = im.resize((w, height), Image.BILINEAR)
        name = "%s_%03d.jpg" % (prefix, i)
        im.save(os.path.join(out_dir, name), quality=82)
        names.append(name)
    return names


def poster_for(video_path, out_dir, height=STRIP_H):
    """A still for the reel tile (the clip's first frame), made once per
    (path, mtime, size) and kept in out_dir. Returns the file name or None.
    A tile must show SOMETHING before it plays -- and in a browser that cannot
    decode the codec it is all the tile can show."""
    try:
        st = os.stat(video_path)
        key = hashlib.md5(("%s|%d|%d" % (os.path.abspath(video_path), st.st_mtime_ns, st.st_size)).encode("utf-8")).hexdigest()[:16]
        name = "poster_%s.jpg" % key
        target = os.path.join(out_dir, name)
        if os.path.isfile(target):
            return name
        import av
        with av.open(video_path) as c:
            vs = next((s for s in c.streams if s.type == "video"), None)
            if vs is None:
                return None
            for fr in c.decode(vs):
                write_previews(fr.to_ndarray(format="rgb24")[None], out_dir, "poster_%s" % key, height)
                os.replace(os.path.join(out_dir, "poster_%s_000.jpg" % key), target)
                return name
    except Exception:
        return None
    return None


# --------------------------------------------------------------------------
# v1020 (Cine C4): the seam correction -- our own implementation of the idea
# obvpm measured and published (GPL-3; idea yes, code no): the continuation
# opens at a different LEVEL than the parent was heading to, confined to the
# first ~12 frames. Correct it multiplicatively per channel (a gain leaves
# black at black -- an offset would lift the blacks into haze) and DECAY the
# correction to identity by frame N, so a chain of fifty clips accumulates
# nothing (a constant per-clip grade would multiply down the chain).
# --------------------------------------------------------------------------

def _theil_sen_at0(y):
    """Value at x=0 of the Theil-Sen line through y[0..m-1]. Pure."""
    y = np.asarray(y, dtype=np.float64)
    x = np.arange(y.shape[0], dtype=np.float64)
    i, j = np.triu_indices(y.shape[0], 1)
    slope = float(np.median((y[j] - y[i]) / (x[j] - x[i])))
    return float(np.median(y - slope * x))


def level_gains(prev_frames, next_frames, n=LEVEL_FRAMES, fit=LEVEL_FIT):
    """[n, 3] per-frame, per-channel gains for the first n frames of next.

    target: where prev was heading (a line through its last `fit` frames,
            one step on); start: next's own opening, straightened (a line
            through its first n frames, value at 0) -- a single noisy first
            frame must not decide the correction. gain(f) = 1 + (target/start
            - 1) * (1 - f/n): full at the join, identity at frame n."""
    p = _as255(prev_frames)
    q = _as255(next_frames)
    cp = p.reshape(p.shape[0], -1, 3).mean(axis=1)
    cq = q.reshape(q.shape[0], -1, 3).mean(axis=1)
    k = min(fit, cp.shape[0])
    if k >= 2:
        x = np.arange(k, dtype=np.float64)
        target = np.array([np.polyval(np.polyfit(x, cp[-k:, c], 1), k) for c in range(3)])
    else:
        target = cp[-1]
    m = min(int(n), cq.shape[0])
    if m >= 2:
        # Theil-Sen (median of pairwise slopes, median intercept): a least-
        # squares line lets ONE flashed or black first frame drag the start
        # (measured: 200 in a 112 opening gave gain 0.72 instead of 0.89)
        start = np.array([_theil_sen_at0(cq[:m, c]) for c in range(3)])
    else:
        start = cq[0]
    ratio = np.where(start > 1e-3, target / np.maximum(start, 1e-3), 1.0)
    ramp = 1.0 - np.arange(m, dtype=np.float64) / float(max(1, m))
    return 1.0 + (ratio[None, :] - 1.0) * ramp[:, None]


def apply_gains(frames, gains):
    """frames [N,H,W,3] float 0..1 (numpy); gains [m,3] for the first m frames.
    Returns a new array, clipped to 0..1."""
    out = np.array(frames, dtype=np.float32, copy=True)
    m = min(gains.shape[0], out.shape[0])
    out[:m, ..., :3] *= gains[:m, None, None, :].astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def fade_in(wav, sr, ms):
    """Linear fade-in over `ms` milliseconds on [.., L] audio (a copy)."""
    n = int(round(sr * ms / 1000.0))
    out = np.array(wav, dtype=np.float32, copy=True)
    if n > 1:
        n = min(n, out.shape[-1])
        out[..., :n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
    return out


def concat_videos(paths, out_path):
    """Join clips of one chain into one file by PACKET COPY (no re-encode, no
    subprocess): every clip's packets are shifted by the running duration.
    All clips must share codec and size -- the chain's own clips do. Returns
    (frames_total_estimate, seconds). Raises by name on a mismatch."""
    import av
    if not paths:
        raise ValueError("nothing to join")
    with av.open(paths[0]) as first:
        v0 = next((s for s in first.streams if s.type == "video"), None)
        a0 = next((s for s in first.streams if s.type == "audio"), None)
        if v0 is None:
            raise ValueError("%s has no video stream" % os.path.basename(paths[0]))
        sig = (v0.codec_context.name, v0.codec_context.width, v0.codec_context.height)
    tmp = out_path + ".tmp.mp4"
    total_s = 0.0
    with av.open(tmp, "w") as out:
        with av.open(paths[0]) as first:
            vt = first.streams.video[0]
            ov = out.add_stream_from_template(vt) if hasattr(out, "add_stream_from_template") else out.add_stream(template=vt)
            oa = None
            if a0 is not None and first.streams.audio:
                at = first.streams.audio[0]
                oa = out.add_stream_from_template(at) if hasattr(out, "add_stream_from_template") else out.add_stream(template=at)
        off = {"v": 0.0, "a": 0.0}
        for p in paths:
            with av.open(p) as c:
                vs = c.streams.video[0]
                s2 = (vs.codec_context.name, vs.codec_context.width, vs.codec_context.height)
                if s2 != sig:
                    raise ValueError("%s is %s, the chain is %s -- only clips of one chain join"
                                     % (os.path.basename(p), s2, sig))
                dur = float(c.duration or 0) / 1e6 if c.duration else 0.0
                streams = [vs] + ([c.streams.audio[0]] if (oa is not None and c.streams.audio) else [])
                last = {"v": 0.0, "a": 0.0}
                for pkt in c.demux(*streams):
                    if pkt.dts is None:
                        continue
                    key = "v" if pkt.stream.type == "video" else "a"
                    tb = pkt.stream.time_base
                    shift = int(round(off[key] / tb))
                    pkt.pts = (pkt.pts or 0) + shift
                    pkt.dts = pkt.dts + shift
                    end = float((pkt.pts - shift + (pkt.duration or 0)) * tb)
                    last[key] = max(last[key], end)
                    pkt.stream = ov if key == "v" else oa
                    out.mux(pkt)
                step = max(last["v"], dur)
                off["v"] += step
                off["a"] += step
                total_s += step
    os.replace(tmp, out_path)
    return total_s
