#!/usr/bin/env python3
"""v377 -- the media extension law lives in ONE place, and knows .avif.

Public issue #4 (14.09.2026) reported two things about the Media Loader:
no AVIF/WebP support, and a grid that only ever sorts newest-first. Measured
against the source before answering:

  * WebP was ALREADY supported -- .webp sat in both the loader's _IMAGE_EXTS
    and _BATCH_IMAGE_EXTS, and had for a long time.
  * AVIF was genuinely missing, and the reporter's reasoning was right: core's
    LoadImage filters by MIME (folder_paths.filter_files_content_types), and
    mimetypes.guess_type("x.avif") answers image/avif, so core lists it.
  * The reason a format could be half-supported at all is the real defect: the
    extension tuples existed TWICE, in ph_media_loader and in ph_media_routes, with
    a comment claiming they were "kept in lock-step". A comment is not a
    mechanism. This guard is the mechanism.

  L1  the law is defined in ph_media_util and nowhere else
  L2  the loader and the routes both READ it (no second literal tuple)
  L3  .avif is in the image law and in the batch-image law
  L4  .webp is still there (the thing the issue said was missing)
  L5  the gif split survives the move: video for the loader, image for the routes
  L6  decoding is a SEPARATE question from listing, asked in the one decode funnel
  L7  the refusal names the file and the cure, and is not a bare Pillow error

MUTATION PROBE (run 20.09.2026, each mutation applied and reverted):
  1. .avif removed from IMAGE_EXTS                     -> L3 caught
  2. .avif removed from BATCH_IMAGE_EXTS               -> L3 caught
  3. .webp removed from IMAGE_EXTS                     -> L4 caught
  4. literal tuple restored in ph_media_routes              -> L2 caught
  5. the avif decoder check deleted from the funnel    -> L6 caught
  6. AVIF_HINT emptied                                 -> L7 caught
  7. ".gif" appended to ph_media_util.VIDEO_EXTS       -> L5 caught
"""
import os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTIL = os.path.join(ROOT, "nodes", "ph_media_util.py")
LOADER = os.path.join(ROOT, "nodes", "ph_media_loader.py")
ROUTES = os.path.join(ROOT, "nodes", "ph_media_routes.py")
failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def _read(p):
    return open(p, encoding="utf-8").read()


def _strip(src):
    """Drop comments and docstrings so a mention in prose never satisfies a
    check -- the CODE has to carry it."""
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"^\s*#.*$", "", src, flags=re.M)
    return src


def main():
    print("v377: one extension law, and it knows AVIF")
    util, loader, routes = _read(UTIL), _read(LOADER), _read(ROUTES)
    util_c, loader_c, routes_c = _strip(util), _strip(loader), _strip(routes)

    # ── L1/L3/L4/L5: the law itself ────────────────────────────────────────
    sys.path.insert(0, os.path.join(ROOT, "nodes"))
    import ph_media_util as law

    check(".avif" in law.IMAGE_EXTS, "L3 .avif is an image extension")
    check(".avif" in law.BATCH_IMAGE_EXTS, "L3 .avif is eligible for the image batch")
    check(".webp" in law.IMAGE_EXTS and ".webp" in law.BATCH_IMAGE_EXTS,
          "L4 .webp is still in both lists (issue #4 claimed it was absent)")
    check(".gif" not in law.VIDEO_EXTS,
          "L5 the shared VIDEO law has no .gif -- the routes must tag a gif as an image")
    check(".gif" in law.IMAGE_EXTS, "L5 a gif is an image to the routes")
    for name in ("IMAGE_EXTS", "VIDEO_EXTS", "AUDIO_EXTS", "BATCH_IMAGE_EXTS"):
        check(isinstance(getattr(law, name, None), tuple),
              "L1 ph_media_util defines %s" % name)

    # ── L2: nobody keeps a second copy ─────────────────────────────────────
    # A literal extension tuple is any parenthesised run of at least three
    # quoted ".xyz" strings. Only MEDIA ones are the law's business: the 3D
    # cockpit's own _MESH_IMPORT_EXTS / _CONVERT_EXTS (.glb/.obj/.stl/...) are
    # a different domain and rightly stay local to the routes. This check
    # first fired on those -- a false alarm in my own guard, narrowed here
    # rather than waved through.
    MEDIA = set(law.IMAGE_EXTS) | set(law.VIDEO_EXTS) | set(law.AUDIO_EXTS)
    lit = re.compile(r'\(\s*"\.\w+"(?:\s*,\s*"\.\w+")+\s*,?\s*\)')

    def media_literals(src):
        out = []
        for m in lit.finditer(src):
            exts = set(re.findall(r'"(\.\w+)"', m.group(0)))
            if len(exts) >= 3 and exts & MEDIA:
                out.append(m.group(0))
        return out

    check(not media_literals(routes_c),
          "L2 ph_media_routes carries no literal MEDIA extension tuple of its own")
    check(not media_literals(loader_c),
          "L2 ph_media_loader carries no literal MEDIA extension tuple of its own")
    check(len(media_literals(util_c)) >= 3,
          "L1 ph_media_util is where the media literals live")
    check("from .ph_media_util import" in routes_c and "IMAGE_EXTS" in routes_c,
          "L2 ph_media_routes imports the law")
    check("IMAGE_EXTS as _LAW_IMAGE_EXTS" in loader_c,
          "L2 ph_media_loader imports the law")

    # the loader's own gif split is built FROM the law, not re-typed
    check(re.search(r'_VIDEO_EXTS\s*=\s*_LAW_VIDEO_EXTS\s*\+\s*\(\s*"\.gif"\s*,?\s*\)', loader_c)
          is not None,
          "L5 the loader derives its video list from the law plus .gif")

    # ── L6/L7: listing is not decoding ─────────────────────────────────────
    m = re.search(r"def _decode_image_rgba\(path: str\):(.*?)\ndef ", loader_c, re.S)
    body = m.group(1) if m else ""
    check(m is not None, "L6 the single decode funnel is still _decode_image_rgba")
    check('".avif"' in body and "avif_decoder_ready()" in body,
          "L6 the funnel asks whether THIS Pillow can decode avif")
    check("AVIF_HINT" in body,
          "L7 the refusal carries the hint rather than a bare Pillow error")
    check("os.path.basename(path)" in body,
          "L7 the refusal names the offending file")
    # the funnel is the ONE place both paths cross
    check(loader_c.count("_decode_image_rgba(") >= 2,
          "L6 both the single and the batch path go through that funnel")

    # the hint has to be useful: it must name the version and the package
    check("11.3" in law.AVIF_HINT and "pillow-avif-plugin" in law.AVIF_HINT,
          "L7 the hint names both cures (Pillow 11.3+, or the plugin package)")

    # ── the probe is a MEASUREMENT, not a version guess ────────────────────
    probe = re.search(r"def avif_decoder_ready\(\).*?(?=\n\n\nAVIF_HINT)", util_c, re.S)
    pbody = probe.group(0) if probe else ""
    check(probe is not None, "L6 avif_decoder_ready exists")
    check("features.check" in pbody and "registered_extensions" in pbody,
          "L6 the probe ASKS Pillow (feature registry, then extension registry)")
    check("__version__" not in pbody,
          "L6 the probe never guesses from a version string -- v376's lesson")
    check(isinstance(law.avif_decoder_ready(), bool),
          "L6 the probe answers a plain bool on this install")

    return 1 if failures else 0


if __name__ == "__main__":
    rc = main()
    print("v377 media formats: %s" % ("FAIL (%d)" % len(failures) if failures else "PASS"))
    sys.exit(rc)
