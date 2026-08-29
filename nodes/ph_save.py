"""
⬡ Polyhedron Save (ULSSave) -- one unified OUTPUT node for the tail of the
Media Loader -> (Sampler -> VAEDecode) -> 3D/Relight/Upscale chain.

It takes our OWN native types -- IMAGE, the comfy_api VIDEO, native AUDIO, MASK --
and writes studio-quality output. It plays with the Media Loader directly because
it consumes the same contracts that node emits (get_components() on VIDEO, the
{waveform,sample_rate} AUDIO dict, the opaque->0 MASK convention) and it resolves
its output path the exact SaveImage way (folder_paths.get_save_image_path, so the
%date:...% / %Node.widget% filename tokens and the counter just work), the same as
our Sprite Sheet.

THREE write backends, chosen per preset (the routing table lives in the torch-free
ph_save_util so it is guard-tested):

  * native  -- H.264 MP4 delivery rides ComfyUI's OWN comfy_api VIDEO.save_to.
               Its code path is battle-tested and muxes the component audio for
               us: the everyday case uses the most robust path available.
  * pillow   -- animated GIF / WebP written by Pillow (no ffmpeg binary, no PyAV).
  * pyav     -- the pro masters native can't do (H.265 grain-tuned, ProRes 422 HQ,
               ProRes 4444 + alpha, FFV1 lossless, WebM VP9). ComfyUI bundles PyAV
               (`av`), so this adds NO external dependency -- the same library our
               Media Loader and Power Upscale already use. For 10-bit pix_fmts the
               frames are fed as a 16-bit intermediate (rgb48le) so the diffusion
               float precision reaches the encoder -> visibly less banding in
               skies / smoke / gradients. Intra presets set GOP=1 (every frame a
               keyframe) so WAN particle / grain content isn't smeared.

Design rule (Stabilität vor Funktionalität): every optional dependency is imported
lazily and every non-essential step (metadata embed, audio mux) is defensive -- a
failure there logs and degrades, it never loses the render.
"""

import os

import numpy as np

from . import ph_save_util as U

# The comfy_api VIDEO type + its save enums (optional, version-stable shims --
# the exact Media Loader / Power Upscale pattern). Absent -> the native backend
# and any wired VIDEO raise a clear error instead of crashing the pack.
try:
    from comfy_api.input_impl import VideoFromComponents
    from comfy_api.util import VideoComponents
    from fractions import Fraction
    _HAS_VIDEO_API = True
except Exception:  # pragma: no cover
    VideoFromComponents = VideoComponents = None
    _HAS_VIDEO_API = False

try:
    from comfy_api.util import VideoContainer, VideoCodec
    _HAS_VIDEO_ENUMS = True
except Exception:  # pragma: no cover
    VideoContainer = VideoCodec = None
    _HAS_VIDEO_ENUMS = False


# ── small torch-free-at-import helpers ───────────────────────────────────────
def _to_np(x):
    """ComfyUI hands tensors; accept either a tensor (.cpu()) or a numpy array
    without importing torch at module load."""
    if x is None:
        return None
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x, dtype=np.float32)


def _metadata_allowed(save_metadata):
    """save_metadata AND the server's own metadata switch must both allow it."""
    if not save_metadata:
        return False
    try:
        from comfy.cli_args import args as _cli
        return not bool(getattr(_cli, "disable_metadata", False))
    except Exception:
        return True


def _even_dims(width, height):
    """The largest EVEN canvas that fits inside (width, height).

    v591 - Frank's 2026-07-14 run died here, 18:36 in:

        av.error.ExternalError: [Errno 542398533] Generic error in an
        external library: 'avcodec_open2(libx264)'

    The canvas was 1075x1075 (768 * final_upscale_by 1.40 = 1075.2 -> 1075).
    H.264 subsamples chroma 2x2: an odd edge has no chroma sample to carry it,
    and libx264 refuses to even OPEN the codec - not on frame 500, on frame
    zero, after every second of the run was already spent. Reproduced in a
    sandbox: 1075x1075 raises, 1074x1074 writes, 1074x1075 raises. The encoder
    is not negotiable.

    CROP, never pad. One duplicated pixel row is a seam that moves with the
    image; one dropped row of 1075 is 0.09% of the frame and invisible. The
    user's dial is not touched - this is the encoder's floor, not a new canvas.
    """
    w, h = int(width), int(height)
    return (w - (w & 1), h - (h & 1))


def _needs_even(backend, pix_fmt):
    """Does this write path require even edges?

    native  -> comfy_api writes H.264/yuv420p. Always.
    pyav    -> only when the pixel format subsamples chroma (yuv420, yuv422,
               nv12...). yuv444 and the rgb/gbrp family carry a chroma sample
               per pixel and take odd edges without complaint.
    pillow  -> GIF/WebP have no chroma planes. Never.
    """
    b = str(backend or "").lower()
    if b == "pillow":
        return False
    if b == "native":
        return True
    f = str(pix_fmt or "").lower()
    return ("420" in f) or ("422" in f) or f.startswith("nv")


def _resolve_out(filename_prefix, width, height, save_output):
    """SaveImage-style resolution to (folder, stem, counter, subfolder, out_type).
    save_output False -> the temp dir + type 'temp' (preview-only, uncluttered)."""
    import folder_paths
    prefix = U.sanitize_prefix(filename_prefix, width, height)
    base = (folder_paths.get_output_directory() if save_output
            else folder_paths.get_temp_directory())
    full, stem, counter, subfolder, _ = folder_paths.get_save_image_path(
        prefix, base, width, height)
    os.makedirs(full, exist_ok=True)
    return full, stem, counter, subfolder, ("output" if save_output else "temp")


def _png_info(prompt, extra_pnginfo):
    """A PngInfo carrying the API prompt and every extra_pnginfo key (usually
    'workflow') -- the native SaveImage round-trip so the saved PNG can be
    dragged back into ComfyUI. Returns None on any trouble (never fatal)."""
    try:
        from PIL.PngImagePlugin import PngInfo
        import json
        meta = PngInfo()
        if prompt is not None:
            meta.add_text("prompt", json.dumps(prompt))
        if extra_pnginfo is not None:
            for k, v in extra_pnginfo.items():
                meta.add_text(k, json.dumps(v))
        return meta
    except Exception as e:  # pragma: no cover
        print(f"[PLS] Save: PNG metadata skipped ({e!r}).")
        return None


def _fps_of(video, frame_rate):
    """fps for the video: a wired VIDEO carries its own frame_rate; otherwise the
    frame_rate widget (>0), else a safe 24."""
    if video is not None and _HAS_VIDEO_API:
        try:
            fr = float(video.get_components().frame_rate)
            if fr > 0:
                return fr
        except Exception:
            pass
    fr = float(frame_rate or 0.0)
    return fr if fr > 0 else 24.0


# ── the node ─────────────────────────────────────────────────────────────────
class ULSSave:
    """⬡ Polyhedron Save -- unified image/video/audio writer (see module doc)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename_prefix": ("STRING", {
                    "default": "Polyhedron",
                    "tooltip": "Name under ComfyUI/output/. A '/' makes a subfolder. "
                               "Optional tokens are expanded here: %date:yyyy-MM-dd% / "
                               "%date:hhmmss%, %width%, %height% (illegal path characters "
                               "are stripped so a token can never break the path). A "
                               "counter keeps files unique.",
                }),
                "media_kind": (list(U.MEDIA_KINDS), {
                    "default": "auto",
                    "tooltip": "auto: infer from what's wired (VIDEO -> video; IMAGE+audio/"
                               "fps -> video; IMAGE alone -> image; AUDIO alone -> audio). "
                               "Or force 'image' / 'video'.",
                }),
                "image_format": (list(U.IMAGE_FORMATS), {
                    "default": "png",
                    "tooltip": "Still format. png = lossless + embedded workflow (drag back "
                               "to reload). webp = smaller / optional lossless. jpg = no alpha.",
                }),
                "image_quality": ("INT", {
                    "default": 100, "min": 1, "max": 100,
                    "tooltip": "webp / jpg quality (100 on webp = lossless). png ignores this "
                               "(it uses lossless compression).",
                }),
                "video_preset": (list(U.PRESET_NAMES), {
                    "default": U.DEFAULT_PRESET,
                    "tooltip": "Delivery (small, universal) vs Master (edit/archive). H.264 uses "
                               "ComfyUI's own tested writer; the masters use our bundled-PyAV "
                               "encoder (10-bit via a 16-bit intermediate, intra keyframes).",
                }),
                "quality": ("INT", {
                    "default": 17, "min": 0, "max": 100,
                    "tooltip": "crf-style quality for the lossy presets (lower = better/larger; "
                               "~17-20 is visually lossless). ProRes/FFV1 ignore it.",
                }),
                "frame_rate": ("FLOAT", {
                    "default": 24.0, "min": 0.0, "max": 240.0, "step": 0.01,
                    "tooltip": "fps when a video is built from an IMAGE batch. A wired VIDEO keeps its own "
                               "rate (this is ignored then). Convertible to an input -- wire Interpolate's fps"
                               " output here so interpolated frames keep their true duration.",
                }),
                "autoplay": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "In-node video preview: off = show first frame with play "
                               "controls (a single-frame clip never jumps). On = play "
                               "automatically (muted); loops only when the clip has >1 frame.",
                }),
                "pingpong": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Append the reversed frames (forward then back) for a seamless "
                               "boomerang loop.",
                }),
                "loop_count": ("INT", {
                    "default": 0, "min": 0, "max": 1000,
                    "tooltip": "GIF / animated-WebP loop count (0 = loop forever).",
                }),
                "trim_to_audio": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Match the clip length to the wired audio (trim the longer of the "
                               "two) so video and sound end together.",
                }),
                "save_metadata": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Embed the workflow + prompt (PNG text / MP4 metadata) so the file "
                               "can be dragged back into ComfyUI.",
                }),
                "save_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "On: write to ComfyUI/output/. Off: temp dir (preview only, keeps "
                               "output/ uncluttered while iterating).",
                }),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Still(s) or video frames [N,H,W,3|4]."}),
                "video": ("VIDEO", {"tooltip": "Native VIDEO (frames + audio + fps ride through). "
                                              "Takes the Media Loader's `video` or `video_audio` "
                                              "output alike -- a `video_audio` wire carries its "
                                              "trimmed companion track along and it is muxed into "
                                              "the written clip."}),
                "audio": ("AUDIO", {"tooltip": "Optional audio to mux with image frames, or to "
                                               "replace a wired video's audio."}),
                "mask": ("MASK", {"tooltip": "Optional alpha [N,H,W] (opaque->0). Wired ALONE, "
                                  "the mask itself is saved as grayscale stills. Kept for png/webp "
                                             "stills and alpha-capable video presets (ProRes 4444)."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("path", "fps")
    FUNCTION = "save"
    CATEGORY = "Polyhedron/IO"
    DESCRIPTION = ("One output node for images, video and audio. It picks the writer from "
                   "the media actually wired to it, embeds the workflow so the file can be "
                   "dragged back into ComfyUI, and when it writes a clip it drops a still "
                   "frame beside it as a control image - an .mp4 cannot carry a workflow, a "
                   "PNG can.")
    OUTPUT_NODE = True

    # -- dispatch -------------------------------------------------------------
    def save(self, filename_prefix, media_kind, image_format, image_quality,
             video_preset, quality, frame_rate, autoplay, pingpong, loop_count,
             trim_to_audio, save_metadata, save_output,
             image=None, video=None, audio=None, mask=None,
             prompt=None, extra_pnginfo=None):
        kind = U.resolve_media_kind(
            media_kind, image is not None, video is not None, audio is not None,
            n_frames=(int(image.shape[0]) if image is not None else 1),
            has_mask=(mask is not None))

        # v621: the resolved fps is exposed as a second output (Slot 1, appended --
        # the append-only output rule; 'path' stays Slot 0). A wired VIDEO carries
        # its own rate, else the frame_rate widget (>0), else 24 -- the same
        # resolution the writer itself uses (_fps_of).
        fps = _fps_of(video, frame_rate)

        if kind == "image":
            out = self._save_image(image, mask, image_format, image_quality,
                                   filename_prefix, save_output, save_metadata,
                                   prompt, extra_pnginfo)
        elif kind == "audio":
            out = self._save_audio(audio, filename_prefix, save_output)
        elif kind == "mask":
            out = self._save_mask(mask, image_format, image_quality,
                                  filename_prefix, save_output, save_metadata,
                                  prompt, extra_pnginfo)
        else:
            # v580: media_kind + the still settings ride along, so a clip written
            # under 'auto' can drop its control frame beside itself.
            out = self._save_video(image, video, audio, mask, video_preset, quality,
                                   frame_rate, autoplay, pingpong, loop_count, trim_to_audio,
                                   filename_prefix, save_output, save_metadata,
                                   prompt, extra_pnginfo,
                                   media_kind=media_kind, image_format=image_format,
                                   image_quality=image_quality)

        # append fps as the second result slot without disturbing the ui payload
        res = tuple(out.get("result", ())) if isinstance(out, dict) else tuple(out or ())
        if isinstance(out, dict):
            out["result"] = res + (float(fps),)
            return out
        return {"result": res + (float(fps),)}

    # -- image ----------------------------------------------------------------
    def _save_image(self, image, mask, image_format, image_quality,
                    filename_prefix, save_output, save_metadata, prompt, extra_pnginfo):
        from PIL import Image
        img = _to_np(image)
        if img is None or img.ndim != 4:
            raise ValueError("Save (image): expects an IMAGE batch [N,H,W,3].")
        msk = _to_np(mask)
        fmt = U.image_ext(image_format)
        # v648: alpha also flows from a 4-channel IMAGE (wired MASK keeps priority)
        alpha = U.still_uses_alpha(mask is not None, img.shape[-1], fmt)
        include_meta = _metadata_allowed(save_metadata)

        h, w = int(img.shape[1]), int(img.shape[2])
        full, stem, counter, subfolder, out_type = _resolve_out(
            filename_prefix, w, h, save_output)

        ui, last_path = [], ""
        for i in range(int(img.shape[0])):
            arr, pil_mode = U.compose_still(img[i], None if msk is None else msk[i], fmt)
            pil = Image.fromarray(arr, pil_mode)

            name = f"{stem}_{counter:05d}_.{fmt}"
            path = os.path.join(full, name)
            if fmt == "png":
                pil.save(path, pnginfo=(_png_info(prompt, extra_pnginfo) if include_meta else None),
                         compress_level=4)
            elif fmt == "webp":
                lossless = int(image_quality) >= 100
                pil.save(path, "WEBP", quality=int(image_quality), lossless=lossless)
                if include_meta:
                    self._embed_webp_workflow(path, extra_pnginfo, prompt)
            else:  # jpg
                pil.convert("RGB").save(path, "JPEG", quality=int(image_quality))
            ui.append({"filename": name, "subfolder": subfolder, "type": out_type})
            last_path = path
            counter += 1

        print(f"[PLS] Save: {int(img.shape[0])} {fmt.upper()} still(s) "
              f"({'RGBA' if alpha else 'RGB'}, meta={'on' if include_meta else 'off'}) -> {subfolder}")
        # v532: the preview channel carries EVERY saved still (ph_save.js renders
        # a flipbook with a live "i / N" counter for a multi-still run). The old
        # single-entry form (ui[-1] only) made an 11-still batch look like "one
        # image written" in the node -- the full list puts the saved count into
        # the preview itself. The return path stays the LAST file (unchanged).
        return {"ui": {"ph_save": [{"filename": e["filename"], "subfolder": subfolder,
                                    "type": out_type, "kind": "image"} for e in ui]},
                "result": (last_path,)}

    def _save_mask(self, mask, image_format, image_quality, filename_prefix,
                   save_output, save_metadata, prompt, extra_pnginfo):
        """v653: ONLY a mask wired -> the mask itself as GRAYSCALE stills
        (mode L; white = mask on). One file per batch entry, the same naming,
        metadata and preview machinery as the image path."""
        from PIL import Image
        msk = _to_np(mask)
        if msk is None or msk.ndim not in (2, 3):
            raise ValueError("Save (mask): expects a MASK [N,H,W].")
        if msk.ndim == 2:
            msk = msk[None]
        fmt = U.image_ext(image_format)
        include_meta = _metadata_allowed(save_metadata)
        h, w = int(msk.shape[1]), int(msk.shape[2])
        full, stem, counter, subfolder, out_type = _resolve_out(
            filename_prefix, w, h, save_output)
        ui, last_path = [], ""
        for i in range(int(msk.shape[0])):
            arr = (msk[i].clip(0.0, 1.0) * 255.0).round().astype("uint8")
            pil = Image.fromarray(arr, "L")
            name = f"{stem}_{counter:05d}_.{fmt}"
            path = os.path.join(full, name)
            if fmt == "png":
                pil.save(path, pnginfo=(_png_info(prompt, extra_pnginfo)
                                        if include_meta else None),
                         compress_level=4)
            elif fmt == "webp":
                lossless = int(image_quality) >= 100
                pil.save(path, "WEBP", quality=int(image_quality),
                         lossless=lossless)
                if include_meta:
                    self._embed_webp_workflow(path, extra_pnginfo, prompt)
            else:  # jpg
                pil.save(path, "JPEG", quality=int(image_quality))
            ui.append({"filename": name, "subfolder": subfolder, "type": out_type})
            last_path = path
            counter += 1
        print(f"[PLS] Save: {int(msk.shape[0])} {fmt.upper()} mask still(s) "
              f"(grayscale L, meta={'on' if include_meta else 'off'}) -> {subfolder}")
        return {"ui": {"ph_save": [{"filename": e["filename"], "subfolder": subfolder,
                                    "type": out_type, "kind": "image"} for e in ui]},
                "result": (last_path,)}

    def _embed_webp_workflow(self, path, extra_pnginfo, prompt):
        """Best-effort: write the workflow into WebP EXIF (ImageDescription) so
        ComfyUI can load it back. Never fatal.

        v577: the old code held Image.open(path) - a LAZY read handle - open
        while writing to that SAME path. On Windows that is a sharing violation,
        so the metadata was skipped every single time (the console said so; it
        looked like noise). Load the pixels, take a full in-memory copy, LET THE
        HANDLE GO, and only then write. Still-image path only - the animated
        WebP is written elsewhere and never re-saved (a re-save without
        save_all=True would flatten it to one frame).
        """
        try:
            from PIL import Image
            import json
            with Image.open(path) as src:
                src.load()                       # force the decode NOW
                exif = src.getexif()
                quality = src.info.get("quality", 95)
                lossless = src.info.get("lossless", False)
                im = src.copy()                  # pixels in RAM, no file behind them
            if extra_pnginfo and "workflow" in extra_pnginfo:
                exif[0x010e] = "Workflow:" + json.dumps(extra_pnginfo["workflow"])
            if prompt is not None:
                exif[0x010f] = "Prompt:" + json.dumps(prompt)
            im.save(path, "WEBP", exif=exif.tobytes(),
                    quality=quality, lossless=lossless)
        except Exception as e:  # pragma: no cover
            print(f"[PLS] Save: WebP metadata skipped ({e!r}).")

    # -- video ----------------------------------------------------------------
    def _save_video(self, image, video, audio, mask, video_preset, quality,
                    frame_rate, autoplay, pingpong, loop_count, trim_to_audio,
                    filename_prefix, save_output, save_metadata, prompt, extra_pnginfo,
                    media_kind="video", image_format="png", image_quality=95):
        p = U.preset(video_preset)
        fps = _fps_of(video, frame_rate)

        # Resolve frames (numpy) + audio, from a wired VIDEO or an IMAGE batch.
        if video is not None:
            if not _HAS_VIDEO_API:
                raise RuntimeError("Save: a VIDEO is wired but this ComfyUI build has no "
                                   "comfy_api VIDEO support -- feed frames into 'image' instead.")
            comps = video.get_components()
            frames = _to_np(comps.images)
            src_audio = comps.audio
            if image is not None:
                # v887: a wired VIDEO wins over a wired IMAGE -- and until now
                # it won in SILENCE. That is the one shape of this node that
                # can throw a whole stage's work away without a word: feed the
                # Interpolate's frames into `image` and the ORIGINAL clip into
                # `video` (the natural way to bring the audio along) and the
                # interpolated frames are discarded, at the original rate,
                # while the file still looks plausible. Saying which one won
                # costs one line; not saying it cost a field session.
                print("[PLS] Save: BOTH a video and an image are wired. The "
                      "VIDEO wins -- %d frame(s) at %.2f fps from the video; "
                      "the %d image frame(s) are NOT written, and the "
                      "frame_rate input is ignored because the video carries "
                      "its own rate. If you meant the image frames, unwire "
                      "the video and use the `audio` input for the sound -- "
                      "or feed a video that already carries them."
                      % (int(frames.shape[0]), fps, int(image.shape[0])))
        elif image is not None:
            frames = _to_np(image)
            src_audio = None
        else:
            raise ValueError("Save (video): connect a VIDEO or an IMAGE batch.")
        if frames is None or frames.ndim != 4 or int(frames.shape[0]) == 0:
            raise ValueError("Save (video): no frames to write.")

        audio_dict = audio if audio is not None else src_audio   # explicit audio overrides
        if trim_to_audio and audio_dict is not None:
            frames, audio_dict = self._trim_to_audio(frames, audio_dict, fps)
        frames = U.apply_pingpong(frames, bool(pingpong))

        alpha = U.wants_alpha(video_preset, mask is not None) and (video is None)
        # (alpha is only meaningful when we build frames from IMAGE+MASK ourselves)

        # v591: ONE seam, before the fork - frames and mask are still together
        # here, and no backend can slip past it. The encoder's floor is applied
        # exactly where the encoder is chosen, and it is SAID.
        _ow, _oh = int(frames.shape[2]), int(frames.shape[1])
        if _needs_even(p["backend"], p.get("pix_fmt")):
            _ew, _eh = _even_dims(_ow, _oh)
            if _ew < 2 or _eh < 2:
                raise ValueError(
                    f"Save (video): {_ow}x{_oh} is too small to encode "
                    f"({p['vcodec'] if p.get('vcodec') else p['backend']} "
                    f"needs at least 2x2 with even edges).")
            if (_ew, _eh) != (_ow, _oh):
                frames = frames[:, :_eh, :_ew, ...]
                if mask is not None:
                    _m = _to_np(mask)
                    mask = _m[:, :_eh, :_ew, ...] if _m is not None else None
                print(f"[PLS] Save: {_ow}x{_oh} has an odd edge - "
                      f"{p['vcodec'] or 'H.264'} subsamples chroma 2x2 and the "
                      f"encoder refuses to open on odd dimensions. Cropped to "
                      f"{_ew}x{_eh} ({_ow - _ew}px wide, {_oh - _eh}px tall). "
                      f"Your dial is untouched; the codec is not negotiable. "
                      f"A canvas that lands even avoids this entirely - on a "
                      f"{_ow}px source, nudging the upscale dial one notch "
                      f"usually does it.")

        h, w = int(frames.shape[1]), int(frames.shape[2])
        full, stem, counter, subfolder, out_type = _resolve_out(
            filename_prefix, w, h, save_output)
        name = f"{stem}_{counter:05d}_.{p['ext']}"
        path = os.path.join(full, name)

        backend = p["backend"]
        if backend == "native":
            self._write_native(frames, audio_dict, fps, path, save_metadata, prompt, extra_pnginfo)
        elif backend == "pillow":
            self._write_pillow_anim(frames, mask if alpha else None, p, fps, int(loop_count), path)
        else:
            self._write_pyav(frames, mask if alpha else None, audio_dict, p,
                             fps, int(quality), path)

        print(f"[PLS] Save: video '{video_preset}' [{backend}] {int(frames.shape[0])}f "
              f"{w}x{h} @ {fps:.2f}fps audio={'yes' if audio_dict is not None and U.audio_pass_supported(video_preset) else 'no'} "
              f"-> {os.path.join(subfolder, name)}")
        self._write_control_frame(frames, mask, path, subfolder, media_kind,
                                  image_format, image_quality,
                                  _metadata_allowed(save_metadata), prompt, extra_pnginfo)
        entry = {"filename": name, "subfolder": subfolder, "type": out_type}
        if backend == "pillow":
            # GIF / animated WebP are images -> an <img> animates them.
            entry["kind"] = "image"
            entry["format"] = "image/" + p["ext"]
        else:
            entry["kind"] = "video"
            entry["format"] = "video/" + p["ext"]
            entry["frame_rate"] = float(fps)
            entry["autoplay"] = bool(autoplay)
            entry["frames"] = int(frames.shape[0])   # JS loops only when >1
        # ONE preview channel; ph_save.js swaps a single <img>/<video> in place.
        return {"ui": {"ph_save": [entry]}, "result": (path,)}

    def _write_control_frame(self, frames, mask, video_path, subfolder, media_kind,
                             image_format, image_quality, include_meta,
                             prompt, extra_pnginfo):
        """v580: under media_kind='auto', a real clip also drops ONE still beside it.

        The three media_kinds now read as a sentence:
            'video' -> the clip, nothing else.
            'image' -> stills, nothing else (never reaches here).
            'auto'  -> let the wiring decide -- and if that turns out to be a
                       clip, leave a control frame next to it.
        A single frame under 'auto' is not a clip; it stays one image, as before.

        WHY THE MIDDLE FRAME, not the first: this is a CONTROL frame, not a
        poster. In a video-diffusion run frame 1 is the cleanest one the model
        will produce -- drift has not started yet -- so it is precisely the frame
        that hides the failure you opened the file to look for. n//2 is just as
        deterministic and it actually reports.

        WHY IT EARNS ITS PLACE: an .mp4 cannot carry a workflow. A PNG can, and
        this one does -- the native SaveImage round-trip, so dragging the still
        back into ComfyUI restores the whole graph. The clip is the delivery;
        the still is the receipt.

        Never fatal, and never silent about it: the video is already on disk by
        the time we get here, so a failure costs the still and nothing more --
        but it SAYS so (the v576 audit's B3 lesson: a swallowed exception is how
        'sometimes the workflow just isn't in the file' becomes a ghost story).
        """
        if str(media_kind) != "auto":
            return
        n = int(frames.shape[0])
        if n < 2:
            return
        try:
            from PIL import Image
            fmt = U.image_ext(image_format)
            idx = n // 2
            msk = _to_np(mask)
            arr, pil_mode = U.compose_still(
                frames[idx], None if msk is None else msk[idx], fmt)
            pil = Image.fromarray(arr, pil_mode)

            # Same stem as the clip, different extension: the pair stays together
            # in the folder, in the sort order, and in the eye.
            path = os.path.splitext(video_path)[0] + "." + fmt
            carries_workflow = bool(include_meta) and fmt != "jpg"
            if fmt == "png":
                pil.save(path, pnginfo=(_png_info(prompt, extra_pnginfo) if include_meta else None),
                         compress_level=4)
            elif fmt == "webp":
                pil.save(path, "WEBP", quality=int(image_quality),
                         lossless=int(image_quality) >= 100)
                if include_meta:
                    self._embed_webp_workflow(path, extra_pnginfo, prompt)
            else:
                pil.convert("RGB").save(path, "JPEG", quality=int(image_quality))

            print(f"[PLS] Save: + control frame {idx + 1}/{n} (middle) "
                  f"{'with workflow' if carries_workflow else 'no workflow (jpg)'} "
                  f"-> {os.path.join(subfolder, os.path.basename(path))}")
        except Exception as e:
            print(f"[PLS] Save: control frame FAILED ({type(e).__name__}: {e}). "
                  f"The video is written and intact -- only the still is missing.")

    def _trim_to_audio(self, frames, audio_dict, fps):
        """Trim the longer of {video, audio} so they end together."""
        try:
            import torch  # noqa: F401  (only to read shapes off an AUDIO tensor)
        except Exception:
            pass
        try:
            wav = audio_dict["waveform"]
            sr = int(audio_dict["sample_rate"])
            samples = int(wav.shape[-1])
            secs = samples / sr if sr else 0.0
            target = max(1, int(round(secs * fps)))
            n = int(frames.shape[0])
            if n > target:
                frames = frames[:target]
            elif target > n:
                keep = max(1, int(round(n / fps * sr)))
                audio_dict = {"waveform": wav[..., :keep], "sample_rate": sr}
        except Exception as e:  # pragma: no cover
            print(f"[PLS] Save: trim_to_audio skipped ({e!r}).")
        return frames, audio_dict

    def _write_native(self, frames, audio_dict, fps, path, save_metadata, prompt, extra_pnginfo):
        """H.264 MP4 via ComfyUI's own VideoFromComponents.save_to (audio muxed
        by the component)."""
        if not _HAS_VIDEO_API:
            raise RuntimeError("Save: H.264 preset needs comfy_api VIDEO support (update ComfyUI).")
        import torch
        img_t = torch.from_numpy(np.ascontiguousarray(np.clip(frames[..., :3], 0.0, 1.0)))
        vid = VideoFromComponents(VideoComponents(
            images=img_t, frame_rate=Fraction(fps).limit_denominator(100000), audio=audio_dict))
        meta = None
        if _metadata_allowed(save_metadata):
            meta = {}
            if prompt is not None:
                meta["prompt"] = prompt
            if extra_pnginfo is not None:
                meta.update(extra_pnginfo)
        try:
            if _HAS_VIDEO_ENUMS:
                vid.save_to(path, format=VideoContainer.MP4, codec=VideoCodec.H264, metadata=meta)
            else:
                vid.save_to(path, metadata=meta)     # defaults: auto -> mp4/h264
        except TypeError:  # older save_to without metadata kwarg
            vid.save_to(path)

    def _write_pillow_anim(self, frames, mask, p, fps, loop_count, path):
        """Animated GIF / WebP via Pillow -- no ffmpeg, no PyAV."""
        from PIL import Image
        msk = _to_np(mask)
        pil_frames = []
        for i in range(int(frames.shape[0])):
            if mask is not None:
                rgba = U.rgb_and_mask_to_rgba_uint8(frames[i:i + 1], None if msk is None else msk[i:i + 1])[0]
                pil_frames.append(Image.fromarray(rgba, "RGBA"))
            else:
                rgb = (np.clip(frames[i, ..., :3], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
                pil_frames.append(Image.fromarray(rgb, "RGB"))
        duration_ms = max(1, int(round(1000.0 / max(fps, 0.01))))
        save_kwargs = dict(save_all=True, append_images=pil_frames[1:],
                           duration=duration_ms, loop=int(loop_count), disposal=2)
        if p["ext"] == "webp":
            pil_frames[0].save(path, "WEBP", lossless=True, **save_kwargs)
        else:  # gif
            pil_frames[0].convert("RGBA" if mask is not None else "P").save(path, "GIF", **save_kwargs)

    def _write_pyav(self, frames, mask, audio_dict, p, fps, quality, path):
        """The pro-master encoder on ComfyUI's bundled PyAV. 10-bit pix_fmts get a
        16-bit intermediate; intra presets force GOP=1. Audio mux is defensive."""
        import av
        msk = _to_np(mask)
        alpha = mask is not None and p.get("alpha")
        ten = bool(p.get("ten_bit"))
        src_fmt = ("rgba64le" if ten else "rgba") if alpha else ("rgb48le" if ten else "rgb24")

        container = av.open(path, mode="w")
        try:
            vstream = container.add_stream(p["vcodec"], rate=Fraction(fps).limit_denominator(100000))
            vstream.width, vstream.height = int(frames.shape[2]), int(frames.shape[1])
            vstream.pix_fmt = p["pix_fmt"]
            opts = dict(p.get("opts") or {})
            opts.update(U.crf_option(p["vcodec"], quality))
            if p.get("intra"):
                opts["g"] = "1"                         # every frame a keyframe
            vstream.options = opts

            scale = 65535.0 if ten else 255.0
            dt = np.uint16 if ten else np.uint8
            for i in range(int(frames.shape[0])):
                if alpha:
                    if ten:
                        rgb = np.clip(frames[i, ..., :3], 0.0, 1.0)
                        a = (1.0 - (np.clip(msk[i], 0.0, 1.0) if msk is not None else 0.0)) if msk is not None else np.ones(rgb.shape[:2], np.float32)
                        arr = (np.concatenate([rgb, a[..., None]], -1) * scale + 0.5).astype(dt)
                    else:
                        arr = U.rgb_and_mask_to_rgba_uint8(frames[i:i + 1], None if msk is None else msk[i:i + 1])[0]
                else:
                    arr = (np.clip(frames[i, ..., :3], 0.0, 1.0) * scale + 0.5).astype(dt)
                vframe = av.VideoFrame.from_ndarray(np.ascontiguousarray(arr), format=src_fmt)
                vframe = vframe.reformat(format=p["pix_fmt"])
                for pkt in vstream.encode(vframe):
                    container.mux(pkt)
            for pkt in vstream.encode():                # flush video
                container.mux(pkt)

            if audio_dict is not None and p.get("acodec"):
                try:
                    self._mux_audio_pyav(container, audio_dict, p["acodec"])
                except Exception as e:  # pragma: no cover -- never lose the video
                    print(f"[PLS] Save: audio mux skipped ({e!r}); video written silent.")
        finally:
            # v577: the container closes even when the encode raises. Without
            # this, a rejected pix_fmt / codec option / full disk left the
            # handle OPEN - and on Windows that means the half-written file
            # stays LOCKED (undeletable, unoverwritable) until ComfyUI
            # restarts. The READ path (ph_media_loader) always did this right;
            # only the WRITE path - the one whose corpses stay on disk - did not.
            container.close()

    def _mux_audio_pyav(self, container, audio_dict, acodec):
        """Encode the AUDIO dict {waveform:[1,C,S],sample_rate} as a track and mux."""
        import av
        wav = audio_dict["waveform"]
        if hasattr(wav, "cpu"):
            wav = wav.cpu().numpy()
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim == 3:
            wav = wav[0]                            # [C,S]
        ch = int(wav.shape[0])
        sr = int(audio_dict["sample_rate"])
        astream = container.add_stream(acodec, rate=sr)
        layout = "stereo" if ch >= 2 else "mono"
        frame = av.AudioFrame.from_ndarray(
            np.ascontiguousarray(wav[:2] if ch > 2 else wav), format="fltp", layout=layout)
        frame.sample_rate = sr
        resampler = av.AudioResampler(format=astream.format, layout=astream.layout, rate=astream.rate)
        for rframe in resampler.resample(frame):
            for pkt in astream.encode(rframe):
                container.mux(pkt)
        for pkt in astream.encode():
            container.mux(pkt)

    # -- audio-only -----------------------------------------------------------
    def _save_audio(self, audio_dict, filename_prefix, save_output):
        """A lone AUDIO (Media Loader with no visual) -> lossless FLAC via PyAV,
        WAV via stdlib as a fallback."""
        full, stem, counter, subfolder, out_type = _resolve_out(
            filename_prefix, 0, 0, save_output)
        try:
            import av
            path = os.path.join(full, f"{stem}_{counter:05d}_.flac")
            container = av.open(path, mode="w")
            try:
                self._mux_audio_pyav(container, audio_dict, "flac")
            finally:
                container.close()          # v577: closes even when the mux raises
            fmt = "FLAC"
        except Exception as e:
            print(f"[PLS] Save: FLAC failed ({e!r}); writing WAV.")
            # v577: sweep the orphan. The old code left a half-written, still
            # LOCKED .flac next to the WAV - a file the user could not delete.
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except Exception as rm:
                print(f"[PLS] Save: could not remove the partial FLAC ({rm!r}).")
            path = os.path.join(full, f"{stem}_{counter:05d}_.wav")
            self._write_wav(path, audio_dict)
            fmt = "WAV"
        print(f"[PLS] Save: audio-only -> {fmt} {os.path.join(subfolder, os.path.basename(path))}")
        return {"ui": {}, "result": (path,)}

    def _write_wav(self, path, audio_dict):
        import wave
        wav = audio_dict["waveform"]
        if hasattr(wav, "cpu"):
            wav = wav.cpu().numpy()
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim == 3:
            wav = wav[0]
        ch = int(wav.shape[0])
        sr = int(audio_dict["sample_rate"])
        pcm = (np.clip(wav.T, -1.0, 1.0) * 32767.0).astype("<i2")
        with wave.open(path, "wb") as f:
            f.setnchannels(ch)
            f.setsampwidth(2)
            f.setframerate(sr)
            f.writeframes(pcm.tobytes())
