"""
Polyhedron LoRA Stack — Backend (v267)
═══════════════════════════════════════
Group-aware LoRA application for ComfyUI.

Two nodes:
  - UltimateLoraStack ("⬡ Polyhedron LoRA Stack")
      Group-ordered application with per-group merge mode.
      Outputs: MODEL, CLIP, debug_info, uls_config_out, trigger_words
  - ULSAccelerator   ("⬡ Polyhedron LoRA Engine")
      Flat list, single global merge mode for engine LoRAs.
      Outputs: MODEL, CLIP, debug_info

Design notes:
  - Universal: no model-type assumptions, works for FLUX / WAN / SDXL / SD1.5.
  - One node = one path. Two stacks side-by-side for dual-noise (HIGH+LOW).
  - Merge modes per group:
      SEQ      sequential native LoraLoader (cached) — default, always works
      CONCAT   rank-concatenation, mathematically identical to SEQ but
               travels through a different float path (potentially slightly
               different float-rounding behaviour worth comparing empirically)
      DARE     CONCAT + Bernoulli mask. Two variants per group:
                 channel — drop entire rank-channels (LoRA-aware)
                 element — drop individual tensor elements (classic paper)
  - Cleanup switches (modifiers beside the four modes, CONCAT/DARE only):
      Trim     drop the weakest rank-channels per LoRA by magnitude before
               merge (deterministic; against the "many quiet LoRAs → interference"
               pile-up). Output stays low-rank.
      Resolve  TIES sign-election across conflicting LoRAs, then a truncated
               SVD re-pack to low-rank so it rejoins the same hand-off.
               Composes after Trim; the DARE mask is skipped under Resolve.
  - DARE variant is per-group (Stack) or global (Engine).
  - DARE seed is deterministic across processes (hashlib.sha1).
  - DARE density auto-scales with group size: 1.0 - 0.05·(n-1), floor 0.5.
  - Schema: rows use `weight` for strength. Legacy `wLow`/`wHigh` are read
    as fallback (auto-migration on read).
"""

import os
import re
import json
import math
import time
import threading

import folder_paths

# Native ComfyUI LoraLoader — has built-in caching via self.loaded_lora.
# Going through this loader is what makes SEQ as efficient as Power LoRA Loader.
try:
    from nodes import LoraLoader as _NativeLoraLoader
except ImportError as e:
    raise ImportError(
        "[PLS] Could not import ComfyUI's native LoraLoader from `nodes`. "
        "This usually means ComfyUI is not on the Python path or the install "
        "is broken. Ensure this addon lives under ComfyUI/custom_nodes/."
    ) from e

import comfy.lora

from collections import OrderedDict

# v348: pure merge-math moved to uls_merge_math.py (no ComfyUI import there →
# unit-testable). Re-imported here so every existing internal reference and
# any `from .uls_stack_node import ...` in sibling modules keeps working.
from .uls_merge_math import (  # noqa: F401  (re-export)
    _TE_KEY_PREFIXES, _LORA_CONVENTIONS,
    _is_te_base, _dare_density, _trim_keep_fraction, _trim_channel_indices,
    _dare_seed, _resolve_pick_device, _resolve_sign_elect,
    _detect_convention, _collect_factor_keys, _has_mid_tensor, _dare_mask_apply,
)


# v265: optional interrupt hook — lets ComfyUI's red X (Cancel) abort a long
# merge promptly instead of only after it finishes. Resolved ONCE at import; if
# a future ComfyUI changes/removes the API, the check degrades to a no-op (no
# crash), consistent with the pack's isolated-failure design. It does NOT touch
# merge math: when not interrupted the call does nothing, so merges stay
# bit-identical. When the user cancels, comfy raises InterruptProcessingException
# which propagates up and ComfyUI handles it as a cancelled run.
class _NeverInterrupt(Exception):
    """Sentinel used as the 'interrupt' type when ComfyUI's API is absent, so
    `except INTERRUPT_EXC` clauses stay inert (this is never raised)."""
    pass

try:
    import comfy.model_management as _mm
    _throw_if_interrupted = _mm.throw_exception_if_processing_interrupted
    # The exception comfy raises on Cancel. Broad excepts in the merge/analysis
    # re-raise THIS so a cancel actually aborts instead of being swallowed into a
    # SEQ fallback or an "analysis failed" line.
    INTERRUPT_EXC = getattr(_mm, "InterruptProcessingException", _NeverInterrupt)

    def _check_interrupt():
        _throw_if_interrupted()
except Exception:
    INTERRUPT_EXC = _NeverInterrupt

    def _check_interrupt():
        pass


# Merge timing (v258): print a per-merge wall-time breakdown (load / trim /
# resolve / other) for CONCAT/DARE groups. ON by default — it only fires for an
# actual multi-LoRA merge (SEQ never reaches this path), and it answers "where
# do the seconds go" while calibrating Trim/Resolve. The clock reads never
# change a merge result; only the optional report is gated. Silence with
# PLS_TIMING=0 (or false/no/off).
_TIMING = os.environ.get("PLS_TIMING", "1").strip().lower() not in ("0", "false", "no", "off")


def _timing_bar(frac: float, width: int = 32) -> str:
    """Proportional ASCII bar for the merge-timing report (frac in 0..1)."""
    frac = 0.0 if frac < 0 else (1.0 if frac > 1 else frac)
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


# ─── CONCAT/DARE tensor-dict cache ─────────────────────────────────────────
# IS_CHANGED returns NaN (always re-execute), which is fine for SEQ because
# the native LoraLoader caches the file internally. CONCAT/DARE, however, read
# every safetensors file from disk via comfy.utils.load_torch_file on every
# queue run — expensive for the typical 5–8 LoRA DARE group. This bounded LRU
# caches the raw tensor dicts keyed by (path, mtime, size) so repeated runs
# with unchanged files skip the disk I/O entirely. Execution semantics are
# unchanged — the cache only avoids redundant file reads, it never affects the
# merge result. Bounded by BOTH an entry count AND a byte budget (v251): the
# count covers the realistic case, the byte budget guards the pathological one
# (e.g. many large LoRAs) so CPU-RAM stays actually capped, not just count-capped.
# Oldest entries are evicted first; at least one entry is always kept.
_TD_CACHE = OrderedDict()             # key -> (tensor_dict, nbytes)
_TD_CACHE_MAX = 32                    # hard cap on entries (primary, realistic case)
_TD_CACHE_MAX_BYTES = 4 * 1024 ** 3   # 4 GiB cap on cached CPU tensors (pathological guard)
_TD_CACHE_BYTES = 0                   # running total of cached bytes
# v254: guards the OrderedDict + running byte total so they stay consistent if
# ComfyUI ever executes graphs concurrently. Today's single execution worker is
# unaffected; the lock is uncontended and result-neutral. The slow disk load
# happens OUTSIDE the lock so loads never serialise behind one another.
_TD_CACHE_LOCK = threading.Lock()


def _td_nbytes(td) -> int:
    """Best-effort byte size of a tensor dict (sum of tensor storage sizes).
    Never raises — a value we can't measure simply counts as 0."""
    total = 0
    try:
        for v in td.values():
            try:
                total += v.numel() * v.element_size()
            except Exception:
                pass
    except Exception:
        pass
    return total


def _cached_load_torch_file(path: str):
    """Load a LoRA safetensors tensor dict with a small LRU cache keyed on
    (path, mtime, size). Falls back to a direct load on any stat/IO hiccup.
    Eviction is bounded by entry count and a byte budget (see note above).
    Cache mutations are guarded by _TD_CACHE_LOCK (v254); the disk load runs
    outside the lock so concurrent loads don't serialise behind it."""
    import comfy.utils
    global _TD_CACHE_BYTES
    try:
        st = os.stat(path)
        key = (path, int(st.st_mtime), int(st.st_size))
    except OSError:
        key = None

    # Fast path: serve from cache under the lock.
    if key is not None:
        with _TD_CACHE_LOCK:
            if key in _TD_CACHE:
                _TD_CACHE.move_to_end(key)
                return _TD_CACHE[key][0]

    # Slow path: load OUTSIDE the lock (disk I/O must not serialise behind it).
    td = comfy.utils.load_torch_file(path, safe_load=True)

    if key is not None and td:
        nb = _td_nbytes(td)
        with _TD_CACHE_LOCK:
            # Another thread may have inserted the same key while we loaded;
            # prefer the existing entry and drop our duplicate.
            if key in _TD_CACHE:
                _TD_CACHE.move_to_end(key)
                return _TD_CACHE[key][0]
            _TD_CACHE[key] = (td, nb)
            _TD_CACHE_BYTES += nb
            _TD_CACHE.move_to_end(key)
            # Evict oldest-first by entry count AND byte budget; always keep ≥1 entry
            # so a single oversized LoRA can still be served from cache.
            while len(_TD_CACHE) > _TD_CACHE_MAX or (
                    _TD_CACHE_BYTES > _TD_CACHE_MAX_BYTES and len(_TD_CACHE) > 1):
                _, (_ev_td, ev_nb) = _TD_CACHE.popitem(last=False)
                _TD_CACHE_BYTES -= ev_nb
    return td


# ─── Group Configuration ──────────────────────────────────────────────────
# Application order: broadest first, most specific last.
GROUP_ORDER = ["—", "acc", "style", "scene", "motion", "subject", "detail", "custom"]


# ─── Safetensors Metadata ──────────────────────────────────────────────────

def _read_meta(path: str) -> dict:
    """Read safetensors __metadata__ block. Returns {} on any error."""
    try:
        with open(path, "rb") as f:
            header_bytes = f.read(8)
            if len(header_bytes) < 8:
                return {}
            n = int.from_bytes(header_bytes, "little")
            if n <= 0 or n > 50 * 1024 * 1024:
                return {}
            raw = f.read(n)
            if len(raw) < n:
                return {}
            return json.loads(raw.decode("utf-8", errors="replace")).get("__metadata__", {})
    except Exception:
        return {}


# ── v579: ss_tag_frequency is NOT a trigger list ─────────────────────────────
# It is kohya's DATASET CAPTION STATISTICS - every tag that ever appeared in a
# training caption, with its count. Treating it as trigger words was a category
# error, and it was MEASURED in the field:
#
#   Frank's HIGH wire, 11 LoRAs whose headers carry tag frequencies:
#       TRIGGERS : 1710 / 512  (334.0%)   <- 20 caption fragments x 11 LoRAs
#   The SAME 11 LoRAs on the LOW wire, whose headers carry none, so the
#   FILENAME fallback spoke:
#       TRIGGERS :   38 / 512  (  7.4%)   <- "golden hour", "god rays", ...
#
# The tag soup BEAT the correct answer, because step 3 of the ladder
# short-circuited step 4. And the Inspector HID it: it printed triggers[0]
# truncated to the column, so a 20-tag soup showed up as the innocent word
# "hour". The Token Counter was the only node that told the truth.
#
# The discriminator is simple, because a trigger list is SHORT. Anything wider
# is caption statistics wearing a trigger's coat.
_TRIG_MAX_TAGS  = 6      # a real trigger list is a handful, not a table
_TRIG_MAX_CHARS = 96
_TRIG_HARD_TAGS = 20     # nothing may EVER escape the flattener above this
_TAG_SOUP_SEEN  = set()  # bounded by the LoRA library: say it once per file


def _cap_tags(s):
    """No way out of the flattener may be unbounded. Two of them were: a
    non-JSON string and a non-dict value came back VERBATIM, so a fat
    ss_tag_frequency blob became the 'trigger words' in full."""
    tags = [t.strip() for t in str(s or "").split(",") if t.strip()]
    return ", ".join(tags[:_TRIG_HARD_TAGS])


def _looks_like_trigger_list(s):
    """(is_a_trigger_list, n_tags) - pure, guard-executed."""
    tags = [t.strip() for t in str(s or "").split(",") if t.strip()]
    ok = (0 < len(tags) <= _TRIG_MAX_TAGS) and len(str(s or "")) <= _TRIG_MAX_CHARS
    return ok, len(tags)


def _warn_tag_soup_once(lora_name, n_tags):
    """Say it ONCE per LoRA per process, then be quiet. Silence is how the
    1710-token string got all the way into a 512-token budget unnoticed."""
    key = str(lora_name)
    if key in _TAG_SOUP_SEEN:
        return
    _TAG_SOUP_SEEN.add(key)
    print(f"[PLS] Trigger: '{os.path.basename(key)}' carries {n_tags} tags in its "
          f"ss_tag_frequency - that is a dataset caption table, not a trigger "
          f"list. Using the filename instead. To pin a real trigger, type it "
          f"into the row's trigger field (it wins over everything).")


def _flatten_tag_frequency(tw):
    """Normalize ss_tag_frequency into a comma-separated trigger string.

    Handles three real-world formats:
      1. dict of dicts: {"1_du8ne": {"du8ne": 12, "rare_tag": 1}, ...}
      2. flat dict:     {"trigger_a": 5, "trigger_b": 3, ...}
      3. JSON string:   '{"1_du8ne": {"du8ne": 12}}'  (kohya stores it as text)
      4. plain string:  "trigger_a, trigger_b"

    For nested dicts, the OUTER keys are concept-folder labels (e.g. "1_du8ne")
    — we discard those and use the INNER keys, which are the real tags learned
    during training. We pick the inner tag with the highest frequency per
    concept-folder, since that's the canonical trigger for that concept.
    """
    if not tw:
        return ""

    # Format 3: string that's actually JSON
    if isinstance(tw, str):
        s = tw.strip()
        if s.startswith("{"):
            try:
                tw = json.loads(s)
            except (ValueError, json.JSONDecodeError):
                return _cap_tags(s)   # v579: not valid JSON - capped, not verbatim
        else:
            return _cap_tags(s)   # v579: capped, not verbatim

    if not isinstance(tw, dict):
        return _cap_tags(tw)   # v579: capped, not verbatim

    # Detect nested format (dict of dicts)
    sample_val = next(iter(tw.values()), None)
    if isinstance(sample_val, dict):
        # Format 1: {"1_du8ne": {"du8ne": 12, ...}, ...}
        # Collect highest-frequency tag from each concept-folder.
        triggers = []
        for outer_key, inner in tw.items():
            if not isinstance(inner, dict) or not inner:
                continue
            # Sort inner tags by frequency, take the most frequent
            try:
                top_tag = max(inner.items(), key=lambda kv: kv[1])[0]
            except (TypeError, ValueError):
                top_tag = next(iter(inner.keys()))
            triggers.append(top_tag)
        return ", ".join(triggers[:20])

    # Format 2: flat dict {"trigger": frequency, ...}
    return ", ".join(list(tw.keys())[:20])


def _extract_lora_info(path: str) -> dict:
    meta = _read_meta(path)
    tw_raw = meta.get("ss_tag_frequency", meta.get("trigger_words", ""))
    tw = _flatten_tag_frequency(tw_raw)
    return {
        "trigger_words": tw,
        "base_model":    meta.get("ss_base_model_version",
                         meta.get("modelspec.architecture", "?")),
        "rank":          meta.get("ss_network_dim", "?"),
        "algo":          meta.get("ss_network_module", "lora").split(".")[-1],
        "description":   meta.get("modelspec.description", ""),
        "raw":           meta,
    }


def _path_within_loras(path: str) -> bool:
    """Defense-in-depth (v251): confirm a resolved path lives inside one of the
    configured 'loras' directories. On current ComfyUI, folder_paths.get_full_path
    already enforces containment, so this never fires in normal use — it only
    matters if an older/unpatched get_full_path is in play. Belt-and-suspenders,
    never the primary defense. Never raises; on any doubt it returns False."""
    try:
        rp = os.path.realpath(path)
        for d in folder_paths.get_folder_paths("loras"):
            base = os.path.realpath(d)
            if rp == base or rp.startswith(base + os.sep):
                return True
    except Exception:
        pass
    return False


def _find_preview(lora_name: str) -> dict:
    result = {}
    try:
        path = folder_paths.get_full_path("loras", lora_name)
        if not path or not _path_within_loras(path):
            return result
        base = os.path.splitext(path)[0]
        for ext in [".preview.png", ".preview.jpg", ".preview.jpeg",
                    ".jpg", ".jpeg", ".png"]:
            if os.path.isfile(base + ext):
                result["image"] = base + ext
                break
        for ext in [".preview.mp4", ".preview.gif", ".preview.webm",
                    ".mp4", ".gif", ".webm"]:
            if os.path.isfile(base + ext):
                result["video"] = base + ext
                break
    except Exception:
        pass
    return result


def _read_txt_trigger(lora_name: str) -> str:
    """Read trigger words from .txt file (read-only).
    Supports comma-separated and/or newline-separated values."""
    try:
        path = folder_paths.get_full_path("loras", lora_name)
        if not path or not _path_within_loras(path):
            return ""
        txt_path = os.path.splitext(path)[0] + ".txt"
        if os.path.isfile(txt_path):
            with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read().strip()
            parts = [p.strip() for p in re.split(r"[\n\r,]+", content) if p.strip()]
            return ", ".join(parts)
    except Exception:
        pass
    return ""


# ─── Helpers ───────────────────────────────────────────────────────────────

def _safe_weight(value, default: float = 1.0) -> float:
    try:
        w = float(value)
        if math.isnan(w) or math.isinf(w):
            return default
        return max(-10.0, min(10.0, w))
    except (TypeError, ValueError):
        return default


def _row_weight(row: dict, default: float = 1.0) -> float:
    """Read weight from a row dict, supporting both v097 (`weight`) and legacy
    (`wLow`/`wHigh`) schemas. Picks the first valid value found."""
    for key in ("weight", "wLow", "wHigh"):
        if key in row:
            v = _safe_weight(row.get(key), default=float("nan"))
            if not math.isnan(v):
                return v
    return default


def _row_clip_weight(row: dict, fallback: float) -> float:
    """v302: per-row CLIP strength. Reads optional `wClip`; anything missing or
    non-numeric falls back to the model weight — which makes every pre-v302
    workflow byte-identical (CLIP strength == model strength, as before)."""
    if "wClip" in row:
        v = _safe_weight(row.get("wClip"), default=float("nan"))
        if not math.isnan(v):
            return v
    return fallback


# v302: lora-key prefixes that target the text encoder (kohya `lora_te`,
# `lora_te1/2/3` for SDXL/SD3 duals, diffusers `text_encoder.`, cascade
# `lora_prior_te`). Used to pick the CLIP weight inside the merged build.
# A miss is graceful: an unrecognised TE layer is simply scaled with the
# model weight — i.e. exactly the pre-v302 behaviour, never corruption.




def _short_name(lora_name: str, n: int = 38) -> str:
    """Filename without extension, truncated. Cross-platform safe."""
    return os.path.basename(lora_name).replace(".safetensors", "")[:n]














# ═══ Group Apply Modes ════════════════════════════════════════════════════
#
# A group with 2+ LoRAs can be applied in three different ways:
#
#   SEQ      — sequentially patch each LoRA via the cached native loader.
#              Always correct. Default.
#
#   CONCAT   — concatenate the lora_A / lora_B factors of all LoRAs in the
#              group along the rank dimension, then apply ONCE. Mathematically:
#                  [B1·w1; B2·w2] @ [A1; A2]ᵀ  =  w1·B1·A1 + w2·B2·A2
#              Same delta as SEQ in exact arithmetic. Float rounding paths
#              differ, so empirical bit-for-bit equivalence is NOT guaranteed.
#
#   DARE     — like CONCAT but with a Bernoulli mask on B before concatenation.
#              Surviving entries are rescaled by 1/density to preserve the
#              per-LoRA expectation. Two variants:
#                  channel — drop entire rank-channels
#                            (LoRA-aware: reduces channel-overlap between
#                             concurrently-active LoRAs in the same group)
#                  element — drop individual tensor elements
#                            (classic DARE paper behaviour)
#
# Naming conventions handled (model-agnostic):
#   - Kohya / SD / SDXL:        lora_up.weight   / lora_down.weight
#   - WAN / FLUX / HunyuanVideo: lora_B.weight   / lora_A.weight
# ════════════════════════════════════════════════════════════════════════════









def _get_clip_model(clip):
    """Return the inner CLIP model object across ComfyUI versions, or None."""
    if clip is None:
        return None
    m = getattr(clip, "cond_stage_model", None)
    if m is not None:
        return m
    patcher = getattr(clip, "patcher", None)
    if patcher is not None:
        return getattr(patcher, "model", None)
    return None


# ─── Apply: SEQ ────────────────────────────────────────────────────────────

def _apply_seq(loader, model, clip, names: list, weights: list,
               clip_weights: list = None) -> tuple:
    """Sequentially apply each LoRA via the cached native loader.
    v302: optional per-LoRA CLIP strength (None → CLIP follows model weight,
    the pre-v302 behaviour). Returns (model, clip, [error_strings])."""
    if clip_weights is None:
        clip_weights = list(weights)
    m, c = model, clip
    errors = []
    for name, w, wc in zip(names, weights, clip_weights, strict=True):
        if (abs(w) < 1e-6 and abs(wc) < 1e-6) or not name or name == "None":
            continue
        path = folder_paths.get_full_path("loras", name)
        if not path:
            msg = f"⚠ LoRA not found: {name}"
            print(f"[PLS] {msg}")
            errors.append(msg)
            continue
        try:
            m, c = loader.load_lora(m, c, name, w, wc)
        except Exception as ex:
            short = _short_name(name)
            msg = f"✗ Skipped (incompatible): {short}"
            print(f"[PLS] {msg}: {ex}")
            errors.append(msg)
    return m, c, errors


# ─── Apply: CONCAT / DARE ──────────────────────────────────────────────────

def _apply_concat_or_dare(loader, model, clip, names: list, weights: list,
                          mode: str, dare_variant: str = "channel",
                          trim: bool = False, resolve: bool = False,
                          trim_amount: float = None,
                          force_resolve_device: str = None,
                          clip_weights: list = None) -> tuple:
    """
    Build a synthetic merged LoRA tensor dict by concatenating B/A factors
    along the rank dimension, then hand it to ComfyUI's standard load_lora()
    pipeline. CONCAT skips the mask; DARE applies a Bernoulli mask first.

    Cleanup switches (v250):
      trim    — drop the weakest rank-channels per LoRA by magnitude before
                concatenation (deterministic; reduces the background-noise
                pile-up of many concurrently-stacked LoRAs).
      resolve — TIES sign-election across conflicting LoRAs (v256). Elects the
                dominant sign per weight element and averages only the agreeing
                LoRAs, then re-packs the resolved delta as a low-rank LoRA via a
                truncated SVD so it uses the same hand-off. Composes after Trim;
                the DARE mask is not applied when resolve is on (resolve IS the
                merge). Falls back to SEQ on any layer it can't represent.

    Falls back to SEQ on any structural failure (LyCORIS / conv-mid /
    shape mismatch / empty merge).
    """
    import comfy.utils
    import torch

    # Merge-timing accumulators (v258). Reading the clock never changes the
    # merge result; only the optional report at the end is gated on _TIMING.
    _t_start = time.perf_counter()
    _t_load = _t_trim = _t_resolve = 0.0
    _resolve_layers = 0

    mode = mode.upper()
    dare_variant = (dare_variant or "channel").lower()
    if dare_variant not in ("channel", "element"):
        dare_variant = "channel"

    # resolve (TIES sign-election) is handled per base layer below, in place of
    # the plain rank-concat. See _resolve_sign_elect. Honest report after the loop.

    # --- Load all tensor dicts ---
    # v302: clip_weights rides along through the same validity filter so the
    # three lists stay index-aligned. None → CLIP follows the model weight.
    if clip_weights is None:
        clip_weights = list(weights)
    _t0 = time.perf_counter()
    raw, valid_names, valid_weights, valid_clip_weights = [], [], [], []
    for name, w, wc in zip(names, weights, clip_weights, strict=True):
        _check_interrupt()                     # v265: red X (Cancel) aborts during loading
        if (abs(w) < 1e-6 and abs(wc) < 1e-6) or not name or name == "None":
            continue
        path = folder_paths.get_full_path("loras", name)
        if not path:
            print(f"[PLS] ⚠ Not found: {name}")
            continue
        try:
            td = _cached_load_torch_file(path)
            if td:
                raw.append(td)
                valid_names.append(name)
                valid_weights.append(float(w))
                valid_clip_weights.append(float(wc))
        except Exception as ex:
            print(f"[PLS] ✗ Load failed: {name}: {ex}")
    _t_load += time.perf_counter() - _t0

    if len(raw) < 2:
        # 0 or 1 valid → fall back to SEQ
        return _apply_seq(loader, model, clip, valid_names, valid_weights, valid_clip_weights)

    # --- Detect convention per LoRA ---
    convs = [_detect_convention(td) for td in raw]
    unrecognised = [valid_names[i] for i, c in enumerate(convs) if c is None]
    if unrecognised:
        print(f"[PLS] ⚠ {mode}: {len(unrecognised)} LoRA(s) use non-standard "
              f"format (LyCORIS/LoHA/LoKr?), falling back to SEQ:")
        for n in unrecognised:
            print(f"[PLS]      - {_short_name(n)}")
        return _apply_seq(loader, model, clip, valid_names, valid_weights, valid_clip_weights)

    # Use the first LoRA's convention as the output naming.
    out_up_suffix, out_down_suffix = convs[0]

    # All LoRAs in ONE merge group must share ONE naming convention. Mixing
    # kohya (.lora_up/.lora_down) with WAN/FLUX (.lora_B/.lora_A) here would
    # re-suffix bases collected under one convention with the OTHER's output
    # suffix → unmappable keys. The dangerous case is partial mapping: if some
    # keys still resolve, the non-matching LoRAs get silently dropped from the
    # merge while the report claims success. SEQ applies each LoRA under its
    # own convention, so it is the correct, safe path for a mixed group.
    if len({c for c in convs}) > 1:
        print(f"[PLS] ⚠ {mode}: group mixes LoRA naming conventions "
              f"(kohya vs WAN/FLUX) — falling back to SEQ so each LoRA is "
              f"applied correctly under its own convention.")
        return _apply_seq(loader, model, clip, valid_names, valid_weights, valid_clip_weights)

    # --- Conv/LoCon CP-decomposition guard (v253) ---
    # A LoRA carrying a 'mid' tensor has layer delta up · mid · down, but the
    # concat path reconstructs up @ down only. Concatenating up/down while
    # dropping mid would silently produce a wrong delta for those layers — and
    # convention detection still matches on lora_up/lora_B, so the group would
    # NOT otherwise fall back. Route the whole group to SEQ (native loader is
    # mid-aware). Result-neutral for linear LoRAs (no mid → guard never fires).
    mid_loras = [valid_names[i] for i, td in enumerate(raw) if _has_mid_tensor(td)]
    if mid_loras:
        print(f"[PLS] ⚠ {mode}: {len(mid_loras)} LoRA(s) carry a conv 'mid' "
              f"tensor (LoCon/CP) the concat path can't represent — falling "
              f"back to SEQ so each is applied correctly:")
        for n in mid_loras:
            print(f"[PLS]      - {_short_name(n)}")
        return _apply_seq(loader, model, clip, valid_names, valid_weights, valid_clip_weights)

    # --- Per-LoRA: enumerate (base, up_key, down_key, alpha_key) ---
    per_lora_keys = [_collect_factor_keys(td, conv) for td, conv in zip(raw, convs, strict=True)]

    # --- Group keys by their base name across LoRAs ---
    base_to_sources = {}   # base_name → [(lora_idx, base, up_key, down_key, alpha_key), …]
    for li, triples in enumerate(per_lora_keys):
        for base, uk, dk, ak in triples:
            base_to_sources.setdefault(base, []).append((li, base, uk, dk, ak))

    if not base_to_sources:
        print(f"[PLS] ⚠ {mode}: no factor keys found, falling back to SEQ")
        return _apply_seq(loader, model, clip, valid_names, valid_weights, valid_clip_weights)

    # --- Build synthetic merged tensor dict ---
    # resolve takes over the merge (sign-election), so the DARE mask is NOT
    # applied when resolve is on — resolve IS the merge strategy.
    use_mask = (mode == "DARE") and (not resolve)
    n_active = len(valid_names)
    density  = _dare_density(n_active)
    # v261: per-group Trim strength. `trim_amount` is a kept-fraction (0.5–1.0).
    # When set it OVERRIDES the auto group-size formula; None keeps the v260 auto
    # behaviour (bit-identical). Clamped so a stray value can't gut the LoRA.
    if not trim:
        trim_keep = 1.0
    elif trim_amount is not None:
        trim_keep = float(max(0.5, min(1.0, trim_amount)))
    else:
        trim_keep = _trim_keep_fraction(n_active)
    resolve_seed = _dare_seed(valid_names, valid_weights) if resolve else 0
    # v259: pick the RESOLVE compute device once per merge so the path is
    # stable (stable path -> reproducible result). CPU fallback is rare + loud.
    _resolve_dev = "cpu"
    _resolve_fp16 = False
    if resolve:
        _resolve_dev = force_resolve_device or _resolve_pick_device()
        _resolve_fp16 = (_resolve_dev == "cuda")
        _dev_note = (" (fp16 matmuls, fp32 elect+SVD)" if _resolve_fp16
                     else (" (fp32, CPU fallback)" if force_resolve_device
                           else " (fp32)"))
        print(f"[PLS]   RESOLVE device: {_resolve_dev}{_dev_note}")
    rng = None
    if use_mask:
        seed = _dare_seed(valid_names, valid_weights)
        rng = torch.Generator(device="cpu").manual_seed(seed)
        print(f"[PLS]   DARE: variant={dare_variant}  density={density:.3f}  "
              f"n={n_active}  seed={seed}")
    if trim:
        print(f"[PLS]   TRIM: keep_fraction={trim_keep:.3f}  n={n_active}  "
              f"(dropping weakest rank-channels, deterministic)")

    merged_td = {}
    skipped_shape_mismatch = 0
    alpha_missing_count = 0
    trim_channels_kept = 0
    trim_channels_total = 0
    _n_bases = len(base_to_sources)   # v260: denominator for the live RESOLVE progress line

    for base, sources in base_to_sources.items():
        _check_interrupt()                     # v265: red X (Cancel) aborts during the merge
        bs, as_ = [], []
        out_dim = None
        in_dim_flat = None
        ref_dtype = None

        for (li, _, uk, dk, ak) in sources:
            td = raw[li]
            try:
                B = td[uk].cpu().contiguous()    # [out, rank, ...]
                A = td[dk].cpu().contiguous()    # [rank, in,  ...]
            except Exception:
                continue

            if ref_dtype is None:
                ref_dtype = B.dtype

            B_out_dim = B.shape[0]
            A_in_dim_flat = 1
            for s in A.shape[1:]:
                A_in_dim_flat *= s

            # Cross-LoRA shape check — out_dim and in_dim must match,
            # only the rank may differ (which is the whole point of concat).
            if out_dim is None:
                out_dim = B_out_dim
                in_dim_flat = A_in_dim_flat
            elif B_out_dim != out_dim or A_in_dim_flat != in_dim_flat:
                skipped_shape_mismatch += 1
                continue

            # alpha / rank scale
            rank = A.shape[0]
            alpha_val = None
            if ak is not None:
                try:
                    alpha_val = td[ak].item() if hasattr(td[ak], "item") else float(td[ak])
                except Exception:
                    pass
            if alpha_val is None:
                alpha_missing_count += 1
                scale = 1.0
            else:
                scale = float(alpha_val / rank) if rank > 0 else 1.0

            # v302: text-encoder layers are scaled with the per-LoRA CLIP
            # weight; everything else with the model weight. With wClip unset
            # both are equal → bit-identical to pre-v302. The DARE/RESOLVE
            # seed stays derived from the model weights only, so existing
            # WAN workflows keep their exact masks.
            w = valid_clip_weights[li] if _is_te_base(base) else valid_weights[li]

            # float32 for arithmetic, back to original dtype at end.
            B_f = B.float()
            A_f = A.float()

            # Fold (w * scale) into B once: delta = (w·scale·B) @ A
            B_f = B_f * (w * scale)

            # TRIM (v250): deterministically drop the weakest rank-channels.
            # Done in factor space, BEFORE the (random) DARE mask, so the two
            # switches compose cleanly: trim keeps the strongest channels,
            # DARE may then still thin the survivors. Output stays low-rank.
            trim_channels_total += rank
            if trim:
                _t1 = time.perf_counter()
                keep_idx = _trim_channel_indices(B_f, A_f, trim_keep)
                if keep_idx is not None and keep_idx.numel() < rank:
                    B_f = B_f.index_select(1, keep_idx).contiguous()
                    A_f = A_f.index_select(0, keep_idx).contiguous()
                    rank = A_f.shape[0]   # downstream DARE mask uses trimmed rank
                _t_trim += time.perf_counter() - _t1
            trim_channels_kept += rank

            # DARE mask (v348: extracted to uls_merge_math._dare_mask_apply)
            if use_mask:
                B_f = _dare_mask_apply(B_f, density, dare_variant, rng)

            bs.append(B_f.to(ref_dtype))
            as_.append(A_f.to(ref_dtype))

        if not bs:
            continue

        # Combine the per-LoRA factors into the stored low-rank pair.
        if resolve:
            # RESOLVE (v256): TIES sign-election + disjoint merge, re-packed
            # low-rank. v259: runs on _resolve_dev (cuda/fp16 when available).
            try:
                _t2 = time.perf_counter()
                res = _resolve_sign_elect(bs, as_, out_dim, in_dim_flat,
                                          seed=resolve_seed,
                                          device=_resolve_dev, use_fp16=_resolve_fp16)
                _dt = time.perf_counter() - _t2          # was inline; numerically identical
                _t_resolve += _dt
                _resolve_layers += 1
                # v260: throttled live progress so a long merge is visibly moving and a
                # CPU fallback is spotted immediately. Diagnostic ONLY - no tensor math is
                # touched, so the merge result stays bit-identical to v259. flush=True
                # forces the line out DURING the loop instead of buffering it to the end.
                if _resolve_layers == 1 or _resolve_layers % 25 == 0:
                    _vram = ""
                    if _resolve_dev == "cuda":
                        try:
                            _free, _ = torch.cuda.mem_get_info()
                            _vram = f"  free={_free / (1024 ** 3):.1f}G"
                        except Exception:
                            pass
                    print(f"[PLS]   RESOLVE {_resolve_layers}/{_n_bases}  "
                          f"{_resolve_dev}{'/fp16' if _resolve_fp16 else ''}  "
                          f"layer={_dt:.2f}s  cum={_t_resolve:.1f}s{_vram}", flush=True)
            except RuntimeError as ex:
                if _resolve_dev == "cuda" and "out of memory" in str(ex).lower():
                    print(f"[PLS] ⚠ RESOLVE: CUDA out of memory on layer "
                          f"'{base}' - retrying the WHOLE merge on CPU (slower; "
                          f"result is the CPU variant, NOT identical to GPU runs).")
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    return _apply_concat_or_dare(loader, model, clip, names, weights,
                                                 mode, dare_variant, trim, resolve,
                                                 trim_amount=trim_amount,
                                                 force_resolve_device="cpu",
                                                 clip_weights=clip_weights)
                print(f"[PLS] ⚠ RESOLVE: layer '{base}' could not be sign-elected "
                      f"({ex}) - falling back to SEQ for the whole group.")
                return _apply_seq(loader, model, clip, valid_names, valid_weights, valid_clip_weights)
            except Exception as ex:
                print(f"[PLS] ⚠ RESOLVE: layer '{base}' could not be sign-elected "
                      f"({ex}) - falling back to SEQ for the whole group.")
                return _apply_seq(loader, model, clip, valid_names, valid_weights, valid_clip_weights)
            if res is None:
                continue   # full cancellation at this layer → no patch
            B_concat, A_concat = res
        else:
            # Concatenate along the RANK dimension (CONCAT / DARE).
            try:
                B_concat = torch.cat(bs,  dim=1)
                A_concat = torch.cat(as_, dim=0)
            except Exception as ex:
                print(f"[PLS]   skip {base}: concat failed ({ex})")
                continue

        merged_td[base + out_up_suffix]   = B_concat
        merged_td[base + out_down_suffix] = A_concat
        # alpha = rank → scale = 1.0 (we already folded scale + weight into B).
        # Explicit float32 so add_patches treats this as a clean scalar.
        merged_td[base + ".alpha"]        = torch.tensor(float(B_concat.shape[1]),
                                                          dtype=torch.float32)

    if not merged_td:
        print(f"[PLS] ⚠ {mode}: merged dict empty, falling back to SEQ")
        return _apply_seq(loader, model, clip, valid_names, valid_weights, valid_clip_weights)

    if skipped_shape_mismatch:
        print(f"[PLS]   {mode}: skipped {skipped_shape_mismatch} shape-mismatched layer source(s)")
    if alpha_missing_count:
        print(f"[PLS]   {mode}: {alpha_missing_count} layer(s) had no alpha → assumed scale=1.0")
    if trim and trim_channels_total > 0:
        dropped = trim_channels_total - trim_channels_kept
        print(f"[PLS]   TRIM: kept {trim_channels_kept}/{trim_channels_total} "
              f"rank-channels (dropped {dropped} weakest across all sources)")
    if resolve:
        print(f"[PLS]   RESOLVE: sign-election + disjoint merge over {n_active} "
              f"LoRAs{' (after trim)' if trim else ''}; resolved delta re-packed "
              f"low-rank per layer")

    # --- Hand off to ComfyUI's standard LoRA pipeline ---
    try:
        model_keymap = comfy.lora.model_lora_keys_unet(model.model, {})

        clip_model = _get_clip_model(clip)
        clip_keymap = (comfy.lora.model_lora_keys_clip(clip_model, {})
                       if clip_model is not None else {})

        full_keymap = {**model_keymap, **clip_keymap}

        loaded = comfy.lora.load_lora(merged_td, full_keymap)
        if not loaded:
            print(f"[PLS] ⚠ {mode}: ComfyUI mapped 0 patches, falling back to SEQ")
            return _apply_seq(loader, model, clip, valid_names, valid_weights, valid_clip_weights)

        new_model = model.clone()
        new_model.add_patches(loaded, 1.0, 1.0)

        new_clip = clip
        if clip is not None and clip_keymap:
            clip_target_keys = set(clip_keymap.values())
            clip_loaded = {k: v for k, v in loaded.items() if k in clip_target_keys}
            if clip_loaded:
                new_clip = clip.clone()
                new_clip.add_patches(clip_loaded, 1.0, 1.0)

        shorts = [_short_name(n, 18) for n in valid_names]
        mode_tag = mode + (" +TRIM" if trim else "") + (" +RESOLVE" if resolve else "")
        print(f"[PLS] ✓ {mode_tag} merged {n_active} LoRAs [{', '.join(shorts)}]  "
              f"layers={len(merged_td)//3}  patches={len(loaded)}")
        if _TIMING:
            _t_total = time.perf_counter() - _t_start
            _t_other = _t_total - _t_load - _t_trim - _t_resolve
            if _t_other < 0:
                _t_other = 0.0
            _layers = len(merged_td) // 3
            _denom = _t_total if _t_total > 1e-9 else 1.0
            _rows = [
                ("load",    _t_load,    "safetensors read (cached on re-run)"),
                ("trim",    _t_trim,    "magnitude top-k" if trim else ""),
                ("resolve", _t_resolve, (f"TIES + SVD x{_resolve_layers} layers [{_resolve_dev}{'/fp16' if _resolve_fp16 else ''}]" if resolve else "")),
                ("other",   _t_other,   "concat + DARE mask + hand-off"),
            ]
            print(f"[PLS]   ⏱ merge timing  {mode_tag}  n={n_active}, {_layers} layers "
                  f"— CPU, once (before the sampler)")
            for _label, _sec, _note in _rows:
                _frac = _sec / _denom
                _suffix = f"   {_note}" if _note else ""
                print(f"[PLS]       {_label:<8}{_sec:7.2f}s  {_timing_bar(_frac)}  "
                      f"{_frac * 100:4.0f} %{_suffix}")
            print(f"[PLS]       {'─' * 52}")
            print(f"[PLS]       {'total':<8}{_t_total:7.2f}s")
        return new_model, new_clip, []

    except INTERRUPT_EXC:
        raise                          # v265: let a Cancel (red X) abort; don't swallow into SEQ
    except Exception as ex:
        print(f"[PLS] ✗ {mode} apply failed ({ex}), falling back to SEQ")
        import traceback; traceback.print_exc()
        return _apply_seq(loader, model, clip, valid_names, valid_weights, valid_clip_weights)


# ─── Unified Apply Helper ──────────────────────────────────────────────────

def apply_lora_set(loader, model, clip, names: list, weights: list,
                   mode: str = "SEQ", dare_variant: str = "channel",
                   trim: bool = False, resolve: bool = False,
                   trim_amount: float = None,
                   clip_weights: list = None) -> tuple:
    """
    THE unified apply helper. Used by both Stack (per group) and Engine.

    - mode: "SEQ" | "CONCAT" | "DARE"  (case-insensitive, unknown→SEQ)
    - dare_variant: "channel" | "element"  (only used when mode=DARE)
    - trim / resolve: cleanup switches (v250). Only meaningful for CONCAT/DARE —
      SEQ never sees them (LoRAs are never side-by-side under SEQ), which is
      exactly why the UI greys them out for SEQ.
    - Single LoRA always uses SEQ regardless of mode (mode would be a no-op).

    Returns (model, clip, [error_strings]).
    """
    if not names:
        return model, clip, []

    mode = (mode or "SEQ").upper()
    if mode not in ("SEQ", "CONCAT", "DARE"):
        mode = "SEQ"

    # v302: clip strengths ride along (None → CLIP follows model weight).
    if clip_weights is None:
        clip_weights = list(weights)

    # Filter out empties / zero weights up front. A row survives if EITHER
    # strength is non-zero (model 0 + clip 0.8 is a valid CLIP-only row).
    triples = [(n, w, wc) for n, w, wc in zip(names, weights, clip_weights, strict=True)
               if n and n != "None"
               and (abs(float(w)) >= 1e-6 or abs(float(wc)) >= 1e-6)]
    if not triples:
        return model, clip, []

    f_names   = [t[0] for t in triples]
    f_weights = [t[1] for t in triples]
    f_clip    = [t[2] for t in triples]

    if len(f_names) == 1 or mode == "SEQ":
        return _apply_seq(loader, model, clip, f_names, f_weights, f_clip)

    return _apply_concat_or_dare(loader, model, clip, f_names, f_weights,
                                  mode=mode, dare_variant=dare_variant,
                                  trim=trim, resolve=resolve, trim_amount=trim_amount,
                                  clip_weights=f_clip)


# ─── Trigger Words ─────────────────────────────────────────────────────────

# ─── .uls-meta.json — canonical location + legacy migration ────────────────
#
# Companion metadata (user-curated trigger words, civitai ids, …) lives in a
# JSON file next to the .safetensors. Historically TWO different paths were
# written by different code paths, which silently desynced the data:
#
#   canonical : foo.uls-meta.json            (matches the .txt / .jpg companion
#                                              convention — splitext base)
#   legacy    : foo.safetensors.uls-meta.json (older Civitai-fetch builds)
#
# The frontend overlay's "Save triggers" wrote the canonical name, but the
# Stack backend only ever read the legacy name — so overlay-saved triggers
# were invisible at generation time. These helpers are the single source of
# truth: read tolerates both (canonical wins), write always emits canonical
# and folds-in + removes any legacy file so the two locations converge.

def _uls_meta_path_canonical(full_path: str) -> str:
    """foo.safetensors → foo.uls-meta.json (companion-file convention)."""
    return os.path.splitext(full_path)[0] + ".uls-meta.json"


def _uls_meta_path_legacy(full_path: str) -> str:
    """foo.safetensors → foo.safetensors.uls-meta.json (older builds)."""
    return full_path + ".uls-meta.json"


def _uls_meta_read(full_path: str) -> dict:
    """Read companion metadata, tolerating both historical locations.
    Canonical overrides legacy on key conflicts. Returns {} on any error."""
    data = {}
    if not full_path:
        return data
    # Legacy first, then canonical, so canonical values win on .update().
    for p in (_uls_meta_path_legacy(full_path), _uls_meta_path_canonical(full_path)):
        try:
            if os.path.isfile(p):
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    d = json.load(f)
                if isinstance(d, dict):
                    data.update(d)
        except Exception:
            pass
    return data


def _uls_meta_write(full_path: str, updates: dict) -> str:
    """Merge `updates` into the existing companion metadata and write the
    result to the canonical path. Any legacy file is folded in first, then
    removed, so the two historical locations converge to one. Returns the
    path written, or "" on failure."""
    if not full_path:
        return ""
    canonical = _uls_meta_path_canonical(full_path)
    legacy    = _uls_meta_path_legacy(full_path)
    merged = _uls_meta_read(full_path)   # existing canonical + legacy, merged
    merged.update(updates or {})
    try:
        with open(canonical, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[PLS] ⚠ Failed to write {os.path.basename(canonical)}: {e}")
        return ""
    # Retire the legacy file now that its content is safely in canonical.
    if legacy != canonical and os.path.isfile(legacy):
        try:
            os.remove(legacy)
        except Exception:
            pass
    return canonical


def _read_uls_meta_trigger(lora_name: str) -> str:
    """Read trigger words from .uls-meta.json next to the LoRA file.
    This is where Save-Triggers writes user-curated trigger words from the
    frontend overlay — highest priority because it's user intent."""
    try:
        path = folder_paths.get_full_path("loras", lora_name)
        if not path:
            return ""
        tw = _uls_meta_read(path).get("trigger_words", "")
        if isinstance(tw, str):
            return tw.strip()
    except Exception:
        pass
    return ""


_STAMP_RE = re.compile(r'_(high|low|hd|ld)_noise$', re.IGNORECASE)
_SIDE_RE  = re.compile(r'_(high|low)$',             re.IGNORECASE)
# A "coined" token carries BOTH letters and digits: du8ne, lac8e, oxyge8n,
# rai8n, r8ing, fer8n. That is a deliberate anti-collision spelling -- a word
# a base model has never seen -- so when one is present it IS the trigger.
_COINED_RE = re.compile(r'^(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9]+$')


def _trigger_from_filename(lora_name: str) -> str:
    """Stage 4. Pure, guard-executed against Frank's 459 real filenames.

    v580 -- THIS WAS BROKEN, AND THE SCAN PROVED IT. Two wounds, both measured:

      1. THE STAMP REPEATS. Half the WAN library carries it twice, one file
         three times:
             polyhedron_wan2.2_golden_hour_low_noise_low_noise
             polyhedron_wan2.2_rings_low_noise__low_noise_low_noise_r8ing
         The old regex was anchored at `$` and ran ONCE, so it peeled one stamp
         and left `..._low_noise` standing. Now it peels in a loop.

      2. IT RETURNED parts[-1] -- the LAST WORD. `golden_hour` yielded "hour",
         `god_rays` yielded "rays", `apple_tree` yielded "tree", and
         `1_fern_fer8n_2` yielded "2". The whole phrase IS the trigger; the
         last word is a fragment of it.

    Order matters: a coined token wins over the phrase, because when Frank
    spells a trigger `lac8e` he means `lac8e` and not "lace lac8e".

    This RETURNS A GUESS. The caller tags it `name` and the Inspector prints
    that tag, so a guess never passes itself off as a fact. To pin a real one,
    type it into the row's trigger field -- it beats every rung of this ladder.
    """
    base = os.path.basename(lora_name).replace(".safetensors", "")
    s = base
    for _ in range(6):                      # bounded: no runaway on odd names
        nxt = _SIDE_RE.sub('', _STAMP_RE.sub('', s)).rstrip('_')
        if nxt == s:
            break
        s = nxt
    s = re.sub(r'wan\d+[\._]\d+_?', '', s, flags=re.IGNORECASE)
    s = re.sub(r'polyhedron_?',     '', s, flags=re.IGNORECASE)
    s = re.sub(r'v\d+$',            '', s, flags=re.IGNORECASE)
    s = re.sub(r'_+', '_', s).strip('_')
    parts = [p for p in s.split('_') if p]
    if not parts:
        return ""
    coined = [p for p in parts if _COINED_RE.match(p) and not p.isdigit()]
    if coined:
        return coined[-1]
    words = [p for p in parts if not p.isdigit()]
    return " ".join(words) if words else ""


def _warn_hand_trigger_once(lora_name, tw, where):
    """A companion file is INTENT -- the trigger law says intent is not capped,
    and v580 keeps that. But 26 of Frank's 290 `.txt` files hold whole image
    captions (up to 480 chars), and stage 2 never looked. So: take it, and say
    it once. The value still ships; the surprise does not."""
    ok, n_tags = _looks_like_trigger_list(tw)
    if ok:
        return
    key = "hand:" + str(lora_name)
    if key in _TAG_SOUP_SEEN:
        return
    _TAG_SOUP_SEEN.add(key)
    print(f"[PLS] Trigger: '{os.path.basename(str(lora_name))}' takes its trigger from "
          f"{where} -- {n_tags} tags, {len(tw)} chars. That is long for a trigger "
          f"list, and it goes into the prompt VERBATIM (your file, your call). If "
          f"that was not the plan, edit the file or pin the row's trigger field.")


def _get_trigger(lora_name: str) -> tuple:
    """(trigger, source) -- source is 'meta' | 'txt' | 'header' | 'name' | ''.

    v580: the source travels WITH the value. Stage 4 guesses; before, the guess
    arrived looking exactly like a fact, and the Inspector printed it as one.
    Priority: .uls-meta.json (curated) -> .txt -> safetensors header -> filename.
    """
    # 1. User-curated trigger words from the frontend overlay
    tw = _read_uls_meta_trigger(lora_name)
    if tw:
        _warn_hand_trigger_once(lora_name, tw, ".uls-meta.json")
        return tw, "meta"
    # 2. Companion .txt file (often shipped with Civitai LoRAs)
    tw = _read_txt_trigger(lora_name)
    if tw:
        _warn_hand_trigger_once(lora_name, tw, ".txt")
        return tw, "txt"
    # 3. Embedded safetensors metadata (ss_tag_frequency / trigger_words)
    #    v579: ACCEPTED ONLY IF IT LOOKS LIKE A TRIGGER LIST. ss_tag_frequency is
    #    a dataset caption table, and a soup of it used to SHORT-CIRCUIT step 4
    #    below - which, for a name like "polyhedron_wan2.2_golden_hour_low_noise",
    #    yields exactly "golden hour". The garbage was beating the right answer.
    #    Measured: 1710 trigger tokens against a 512 budget on the wire whose
    #    headers had tag tables; 38 on the wire whose headers had none.
    path = folder_paths.get_full_path("loras", lora_name)
    if path:
        info = _extract_lora_info(path)
        tw = info.get("trigger_words", "")
        if tw:
            ok, n_tags = _looks_like_trigger_list(tw)
            if ok:
                return tw, "header"
            _warn_tag_soup_once(lora_name, n_tags)   # and fall through to 4
    # 4. Last resort — derive from filename. A GUESS, and it says so.
    guess = _trigger_from_filename(lora_name)
    return guess, ("name" if guess else "")


# ─── Group Sorting ─────────────────────────────────────────────────────────

def _sort_active_rows(rows: list, flat_mode: bool = False,
                      custom_order: dict = None):
    """Filter active rows, bucket by group, sort by order.

    flat_mode=True  → skip group bucketing entirely, return rows in list order
                      as a single virtual group "—". Useful for simple sequential
                      stacking without any group logic.
    custom_order    → dict mapping group name to int priority (lower = first).
                      Groups not in the dict fall back to GROUP_ORDER index.
                      Only used when flat_mode=False.

    Returns [(group, [row, ...], [weight, ...]), …]."""

    active = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if not row.get("enabled", True):
            continue
        name = row.get("name", "None")
        if not name or name == "None":
            continue
        active.append(row)

    if flat_mode:
        # Simple sequential: all rows in list order, one virtual group
        if not active:
            return []
        weights = [round(_row_weight(r, default=1.0), 4) for r in active]
        return [("—", active, weights)]

    # Group bucketing
    groups = {g: ([], []) for g in GROUP_ORDER}
    for row in active:
        group = str(row.get("group", "—"))
        if group not in groups:
            group = "custom"
        w = _row_weight(row, default=1.0)
        groups[group][0].append(row)
        groups[group][1].append(round(w, 4))

    # Determine sort key for each group
    def _sort_key(group):
        if custom_order and group in custom_order:
            try:
                return (0, int(custom_order[group]))
            except (ValueError, TypeError):
                pass
        # Fallback: standard GROUP_ORDER index
        try:
            return (1, GROUP_ORDER.index(group))
        except ValueError:
            return (1, 999)

    sorted_groups = sorted(
        [g for g in GROUP_ORDER if groups[g][0]],
        key=_sort_key
    )

    return [(g, groups[g][0], groups[g][1]) for g in sorted_groups]


# ═══ Stack Node ══════════════════════════════════════════════════════════

class UltimateLoraStack:
    """
    Polyhedron LoRA Stack — applies multiple LoRAs to MODEL (and optionally CLIP)
    in a deterministic group-based order. For dual-noise architectures (WAN 2.x):
    use TWO instances side-by-side, one per model.

    Universal: works for FLUX / WAN / SDXL / SD1.5 — no model-type assumptions.
    """

    def __init__(self):
        # Per-instance native loader holds the LoRA cache (self.loaded_lora).
        # Persisting it across executions avoids re-reading safetensors files.
        self._loader = _NativeLoraLoader()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "clip": ("CLIP",),
                # MUST live in 'optional' (NOT 'hidden') — ComfyUI silently
                # discards custom STRING widgets in 'hidden'. Frontend hides
                # this widget visually via _ulsHideConfigWidget.
                "uls_config": ("STRING", {
                    "default": '{"rows":[],"mult":1.0}',
                    "multiline": False,
                }),
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES  = ("MODEL", "CLIP", "STRING", "STRING", "STRING")
    RETURN_NAMES  = ("MODEL", "CLIP", "debug_info", "uls_config_out", "trigger_words")
    FUNCTION      = "apply"
    CATEGORY      = "Polyhedron/Loaders"
    DESCRIPTION = ("Applies many LoRAs to MODEL (and optionally CLIP) in a deterministic, "
                   "group-ordered sequence, with a merge mode chosen per group. Groups are "
                   "the point: stacking fifteen or more LoRAs sequentially produces "
                   "multi-LoRA interference, and a rank-concatenated or sign-elected group "
                   "holds its members apart instead of letting them average each other "
                   "away. Emits its collected trigger words and its own config as text. For "
                   "dual-noise architectures, run two side by side, one per expert.")
    OUTPUT_NODE   = False

    def apply(self, model, clip=None, uls_config='{"rows":[],"mult":1.0}', node_id=None):
        # v540 fail-loud: a filename STRING on 'model' (⬡ Select Model Switch wired
        # directly instead of through a loader) poisoned every consumer downstream
        # until an unrelated node crashed. Name the mistake here, at the source.
        if isinstance(model, str):
            raise ValueError(
                f"[PLS] LoRA Stack: 'model' received the STRING '{model}' -- a filename, "
                f"not a MODEL. \u2b21 Select Model Switch feeds a LOADER's combo input "
                f"(unet_name); wire that loader's MODEL output here."
            )
        if not uls_config or not uls_config.strip():
            uls_config = '{"rows":[],"mult":1.0}'
        try:
            cfg = json.loads(uls_config)
        except json.JSONDecodeError as e:
            print(f"[PLS] ⚠ uls_config JSON invalid: {e}")
            cfg = {"rows": [], "mult": 1.0}

        rows = cfg.get("rows", []) if isinstance(cfg.get("rows"), list) else []

        # Per-group apply modes from frontend: {"scene": "DARE", "detail": "CONCAT", ...}
        group_modes = cfg.get("group_modes", {}) if isinstance(cfg.get("group_modes"), dict) else {}

        # Per-group DARE variants (v098): {"detail": "channel", "scene": "element", ...}
        # Legacy global dare_variant key is used as fallback for old workflows.
        group_dare = cfg.get("group_dare", {}) if isinstance(cfg.get("group_dare"), dict) else {}
        legacy_dare_variant = str(cfg.get("dare_variant", "channel")).lower()
        if legacy_dare_variant not in ("channel", "element"):
            legacy_dare_variant = "channel"

        # Per-group cleanup switches (v250): {"subject": true, ...}. Absent → off,
        # so old workflows are bit-identical. Only act on CONCAT/DARE groups.
        group_trim    = cfg.get("group_trim", {})    if isinstance(cfg.get("group_trim"), dict)    else {}
        group_resolve = cfg.get("group_resolve", {}) if isinstance(cfg.get("group_resolve"), dict) else {}
        # v261: per-group Trim strength (kept-fraction). Absent group → Auto formula.
        group_trim_amount = cfg.get("group_trim_amount", {}) if isinstance(cfg.get("group_trim_amount"), dict) else {}

        # v105: flat_mode disables group sorting — rows applied in list order.
        flat_mode = bool(cfg.get("flatMode", False))

        # v105: custom group order — {"subject": 1, "detail": 2, "scene": 3, ...}
        custom_order = cfg.get("groupOrder", {}) if isinstance(cfg.get("groupOrder"), dict) else {}

        ordered      = _sort_active_rows(rows, flat_mode=flat_mode,
                                         custom_order=custom_order or None)
        total_active = sum(len(grp_rows) for _, grp_rows, _ in ordered)

        model_out = model
        clip_out  = clip
        all_errors = []

        sort_mode_label = "FLAT" if flat_mode else ("CUSTOM ORDER" if custom_order else "GROUP ORDER")
        lines = [
            "═══ Polyhedron LoRA Stack ═══",
            f"  CLIP        : {'connected' if clip is not None else 'not connected'}",
            f"  Rows in     : {len(rows)}  (active: {total_active})",
            f"  Sort mode   : {sort_mode_label}",
            f"  Groups      : {len(ordered)}",
            "───────────────────────────────",
        ]
        if not rows:
            lines.append("  ⚠ No rows received from frontend!")
            lines.append(f"  uls_config: {uls_config[:80]}")

        for group, grp_rows, grp_weights in ordered:
            n = len(grp_rows)
            grp_label = f"[{group}]" if group != "—" else "[—]"
            mode = (group_modes.get(group) or "SEQ").upper()
            if mode not in ("SEQ", "CONCAT", "DARE"):
                mode = "SEQ"

            # Per-group DARE variant, fallback to legacy global setting
            dare_variant = str(group_dare.get(group, legacy_dare_variant)).lower()
            if dare_variant not in ("channel", "element"):
                dare_variant = "channel"

            # Per-group cleanup switches (v250). Only meaningful for CONCAT/DARE.
            trim    = bool(group_trim.get(group, False))    and mode != "SEQ"
            resolve = bool(group_resolve.get(group, False)) and mode != "SEQ"
            # v261: optional Trim strength override (only a real number counts;
            # anything else → None → auto group-size formula).
            trim_amount = None
            if trim:
                _ta = group_trim_amount.get(group, None)
                if isinstance(_ta, (int, float)):
                    trim_amount = float(_ta)

            names = [r.get("name", "None") for r in grp_rows]
            # v302: per-row CLIP strength (defaults to the model weight)
            grp_clip = [round(_row_clip_weight(r, w), 4)
                        for r, w in zip(grp_rows, grp_weights, strict=True)]

            if n == 1:
                short = _short_name(names[0])
                lines.append(f"  {grp_label} {short}  ×{grp_weights[0]}")
            else:
                dare_suffix = f" [{dare_variant[:4].upper()}]" if mode == "DARE" else ""
                clean_suffix = (" +TRIM" if trim else "") + (" +RESOLVE" if resolve else "")
                lines.append(f"  {grp_label} {mode}{dare_suffix}{clean_suffix} ({n} LoRAs):")

            model_out, clip_out, errs = apply_lora_set(
                self._loader, model_out, clip_out,
                names, grp_weights, mode=mode, dare_variant=dare_variant,
                trim=trim, resolve=resolve, trim_amount=trim_amount,
                clip_weights=grp_clip
            )
            all_errors.extend(errs)

            if n >= 2:
                err_set = set(errs)
                for row, w in zip(grp_rows, grp_weights, strict=True):
                    short = _short_name(row.get("name", ""), 35)
                    if any(short in e for e in err_set):
                        lines.append(f"    ⚠ {short}  skipped")
                    else:
                        lines.append(f"    • {short}  ×{w}")
            elif errs:
                # n==1 with error
                for e in errs:
                    lines.append(f"    {e}")

        if all_errors:
            lines.append("───────────────────────────────")
            lines.append(f"  ⚠ {len(all_errors)} LoRA(s) skipped (incompatible model)")
        lines.append("───────────────────────────────")
        debug = "\n".join(lines)
        print(f"\n[PLS]\n{debug}\n")

        # Collect trigger words + build lora_info for Inspector
        triggers = []
        lora_info = []  # [{name, weight, group, trigger_words}, ...]
        for group, grp_rows, grp_weights in ordered:
            for row, w in zip(grp_rows, grp_weights, strict=True):
                name = row.get("name", "")
                tw, src = _get_trigger(name)     # v580: the source rides along
                if tw and tw not in triggers:
                    triggers.append(tw)
                lora_info.append({
                    "name":          os.path.basename(name).replace(".safetensors", ""),
                    "weight":        w,
                    "group":         group,
                    "trigger_words": tw,
                    "trigger_src":   src,
                })
        trigger_words = ", ".join(triggers)

        # Attach lora_info to uls_config_out so Inspector can read it
        try:
            cfg_out = json.loads(uls_config)
        except Exception:
            cfg_out = {}
        cfg_out["lora_info"] = lora_info
        uls_config_out = json.dumps(cfg_out)

        return (model_out, clip_out, debug, uls_config_out, trigger_words)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force re-execution every time — model identity may change
        # without uls_config changing. Cached loader avoids real I/O cost.
        # **kwargs accepts any combination of args ComfyUI throws at us
        # (model, clip, uls_config, node_id) even when some are missing
        # during workflow validation.
        return float("nan")


# ═══ Engine Node ══════════════════════════════════════════════════════════

class ULSAccelerator:
    """
    Polyhedron LoRA Engine — applies engine-class LoRAs (Lightning, Turbo,
    LCM, FusionX, LightXT2V, CausVid, …) before the main creative stack.
    These modify HOW the model computes (inference trajectory) rather than
    WHAT it depicts.

    Flat list (no groups), single global merge mode.
    Same universal behaviour as the Stack — no model-type assumptions.

    Class name kept as `ULSAccelerator` for workflow back-compat.
    """

    def __init__(self):
        self._loader = _NativeLoraLoader()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "clip": ("CLIP",),
                "engine_config": ("STRING", {
                    "default": '{"rows":[],"mode":"SEQ"}',
                    "multiline": False,
                }),
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES  = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES  = ("MODEL", "CLIP", "debug_info")
    FUNCTION      = "apply"
    CATEGORY      = "Polyhedron/Loaders"
    DESCRIPTION = ("Applies engine-class LoRAs (Lightning, Turbo, LCM, FusionX, CausVid and "
                   "the like) ahead of the creative stack. These change HOW the model "
                   "computes - the inference trajectory - rather than WHAT it depicts, "
                   "which is why they get their own node: a flat list, one global merge "
                   "mode, no groups to reason about.")
    OUTPUT_NODE   = False

    def apply(self, model, clip=None, engine_config='{"rows":[],"mode":"SEQ"}', node_id=None):
        # v540 fail-loud: see LoRA Stack guard above -- same filename-STRING trap.
        if isinstance(model, str):
            raise ValueError(
                f"[PLS] LoRA Engine: 'model' received the STRING '{model}' -- a filename, "
                f"not a MODEL. \u2b21 Select Model Switch feeds a LOADER's combo input "
                f"(unet_name); wire that loader's MODEL output here."
            )
        if not engine_config or not engine_config.strip():
            engine_config = '{"rows":[],"mode":"SEQ"}'
        try:
            cfg = json.loads(engine_config)
        except json.JSONDecodeError as e:
            print(f"[Engine] ⚠ engine_config JSON invalid: {e}")
            cfg = {"rows": [], "mode": "SEQ"}

        rows = cfg.get("rows", []) if isinstance(cfg.get("rows"), list) else []
        mode = (cfg.get("mode") or "SEQ").upper()
        if mode not in ("SEQ", "CONCAT", "DARE"):
            mode = "SEQ"

        dare_variant = str(cfg.get("dare_variant", "channel")).lower()
        if dare_variant not in ("channel", "element"):
            dare_variant = "channel"

        # Optional global cleanup switches (v250). No dedicated Engine UI yet;
        # default off. Only meaningful for CONCAT/DARE.
        trim    = bool(cfg.get("trim", False))    and mode != "SEQ"
        resolve = bool(cfg.get("resolve", False)) and mode != "SEQ"

        # Filter active rows. Engine uses `weight`; tolerate legacy too via _row_weight.
        active_names, active_weights, active_clip = [], [], []
        for row in rows:
            if not isinstance(row, dict):           continue
            if not row.get("enabled", True):        continue
            name = row.get("name", "None")
            if not name or name == "None":          continue
            w = _row_weight(row, default=1.0)
            active_names.append(name)
            active_weights.append(round(w, 4))
            active_clip.append(round(_row_clip_weight(row, w), 4))

        n = len(active_names)
        lines = [
            "═══ Polyhedron LoRA Engine ═══",
            f"  CLIP        : {'connected' if clip is not None else 'not connected'}",
            f"  Active      : {n} engine LoRA(s)",
            f"  Mode        : {mode}",
        ]
        if mode == "DARE":
            lines.append(f"  DARE variant: {dare_variant}")
        if trim or resolve:
            _cleanup = []
            if trim:    _cleanup.append("TRIM (magnitude)")
            if resolve: _cleanup.append("RESOLVE (TIES)")
            lines.append("  Cleanup     : " + " + ".join(_cleanup))
        lines.append("──────────────────────────────")

        if n == 0:
            lines.append("  (no engine LoRAs active — pass-through)")
            debug = "\n".join(lines)
            print(f"\n[Engine]\n{debug}\n")
            return (model, clip, debug)

        model_out, clip_out, errs = apply_lora_set(
            self._loader, model, clip,
            active_names, active_weights,
            mode=mode, dare_variant=dare_variant,
            trim=trim, resolve=resolve,
            clip_weights=active_clip,
        )

        err_set = set(errs)
        for name, w in zip(active_names, active_weights, strict=True):
            short = _short_name(name)
            if any(short in e for e in err_set):
                lines.append(f"  ⚠ {short}  skipped")
            else:
                lines.append(f"  • {short}  ×{w}")

        if errs:
            lines.append("──────────────────────────────")
            lines.append(f"  ⚠ {len(errs)} LoRA(s) skipped (incompatible model)")
        lines.append("──────────────────────────────")
        debug = "\n".join(lines)
        print(f"\n[Engine]\n{debug}\n")
        return (model_out, clip_out, debug)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # See ULS Stack IS_CHANGED — same reasoning. **kwargs makes the
        # method robust against any arg combination ComfyUI may pass
        # (e.g. missing model during validation phase).
        return float("nan")


# ═══ Inspector Node ═══════════════════════════════════════════════════════════

class ULSInspector:
    """
    Polyhedron LoRA Inspector — passive consistency-check node.

    Reads active LoRAs + their trigger words from uls_config_out (Stack output),
    then checks whether each trigger word appears in the supplied prompt string.
    Outputs a formatted report as STRING.

    No model patching — purely informational.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "uls_config_out": ("STRING", {
                    "default": '{"rows":[]}',
                    "multiline": False,
                    "forceInput": True,
                }),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "forceInput": True,
                }),
            },
        }

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("inspector_report",)
    FUNCTION      = "inspect"
    CATEGORY      = "Polyhedron/Utils"
    OUTPUT_NODE   = False

    def inspect(self, uls_config_out: str, prompt: str) -> tuple:
        # Parse config
        try:
            cfg = json.loads(uls_config_out)
        except Exception:
            cfg = {}

        lora_info = cfg.get("lora_info", [])

        if not lora_info:
            report = "⬡ Polyhedron LoRA Inspector\n  (no lora_info — connect uls_config_out from Stack v119+)"
            return (report,)

        # Build prompt token map: word → explicit weight or "plain"
        # Matches (word:1.2) syntax and plain words
        prompt_weights = {}
        for m in re.finditer(r'\(([^():,]+?):([\d.+-]+)\)', prompt):
            word = m.group(1).strip().lower()
            try:
                prompt_weights[word] = float(m.group(2))
            except ValueError:
                prompt_weights[word] = None

        prompt_lower = prompt.lower()

        # Build report
        lines = [
            "═══ Polyhedron LoRA Inspector ═══",
            f"  LoRAs active : {len(lora_info)}",
            "─────────────────────────────────",
        ]

        col_name = 32
        col_lora = 8
        col_trig = 24
        col_src  = 6

        # v580: WHERE a trigger came from is as important as what it says. Six
        # of Frank's WAN LoRAs carry no trigger anywhere, so stage 4 GUESSES one
        # off the filename -- and the guess used to print in the same column,
        # same font, same confidence as a curated one. It cannot any more.
        _SRC_LABEL = {"meta": "meta", "txt": "txt", "header": "hdr",
                      "name": "name?", "": "—"}

        header = (f"  {'LoRA':<{col_name}} {'Weight':>{col_lora}}   "
                  f"{'Trigger':<{col_trig}} {'From':<{col_src}} {'In Prompt'}")
        lines.append(header)
        lines.append("  " + "─" * (col_name + col_lora + col_trig + col_src + 20))

        found_count = 0
        missing_triggers = []
        guessed_count = 0

        for entry in lora_info:
            name    = entry.get("name", "?")[:col_name]
            weight  = entry.get("weight", 0.0)
            tw_raw  = entry.get("trigger_words", "")
            src     = _SRC_LABEL.get(str(entry.get("trigger_src", "")), "—")
            if src == "name?":
                guessed_count += 1

            if not tw_raw:
                # No trigger words known for this LoRA
                lora_col  = f"×{weight:.2f}"
                trig_col  = "(none)"
                match_col = "—"
                lines.append(f"  {name:<{col_name}} {lora_col:>{col_lora}}   "
                             f"{trig_col:<{col_trig}} {src:<{col_src}} {match_col}")
                continue

            # Split trigger words and check each
            triggers = [t.strip() for t in tw_raw.split(",") if t.strip()]
            best_match = None
            best_weight_str = ""

            for t in triggers:
                t_lower = t.lower()
                # Word-boundary match — avoids false positives like
                # "ring" matching inside "stinging" or "string"
                pattern = r'\b' + re.escape(t_lower) + r'\b'
                if t_lower in prompt_weights:
                    w = prompt_weights[t_lower]
                    best_match = t
                    best_weight_str = f"({t}:{w})" if w is not None else f"({t}:?)"
                    break
                elif re.search(pattern, prompt_lower):
                    best_match = t
                    best_weight_str = f"{t}  plain"
                    break

            lora_col = f"×{weight:.2f}"
            # v579: SAY how many there are. The old line printed triggers[0]
            # truncated to the column, so a 20-tag caption soup showed up as the
            # innocent word "hour" - and the report that exists to surface
            # trigger problems was the one hiding this one.
            shown = best_match or triggers[0]
            if len(triggers) > 1:
                shown = f"{shown} (+{len(triggers) - 1})"
            trig_col = shown[:col_trig]

            if best_match:
                found_count += 1
                match_col = f"✓  {best_weight_str}"
            else:
                missing_triggers.append(name)
                match_col = "✗  NOT IN PROMPT"

            lines.append(f"  {name:<{col_name}} {lora_col:>{col_lora}}   "
                         f"{trig_col:<{col_trig}} {src:<{col_src}} {match_col}")

        lines.append("  " + "─" * (col_name + col_lora + col_trig + col_src + 20))
        missing = len(lora_info) - found_count
        no_trigger = sum(1 for e in lora_info if not e.get("trigger_words"))
        lines.append(f"  ✓ {found_count} matched   ✗ {missing - no_trigger} missing   — {no_trigger} no trigger defined")
        lines.append("  From: meta=.uls-meta.json · txt=companion file · hdr=safetensors header")
        if guessed_count:
            lines.append(f"        name? = GUESSED from the filename ({guessed_count} here) — "
                         f"no trigger is stored anywhere for these.")
            lines.append("        To pin one, type it into the row's trigger field: it beats every rung.")

        if missing_triggers:
            lines.append("  Missing:")
            for n in missing_triggers:
                lines.append(f"    • {n}")

        lines.append("─────────────────────────────────")
        report = "\n".join(lines)
        print(f"\n[PLS Inspector]\n{report}\n")
        return (report,)


# ─── Token Counter ─────────────────────────────────────────────────────────

# Optional: try to use the real UMT5-XXL tokenizer for exact counts. Falls
# back to a heuristic if transformers / tokenizer files are not available.
# We import lazily inside the count function so import-time stays cheap and
# the node also loads on systems without `transformers` installed.
# v580 -- THE WIRE THAT WAS THERE AND STILL WRONG.
#
# Frank's HIGH counter read TRIGGERS: 1710 / 512 (334%). His LOW counter, same
# eleven LoRAs, read 38. For three rounds that number was treated as evidence:
# a "tag soup" was diagnosed, a fix was built for it, a guard was written, a law
# was written into the handover. All of it was fiction.
#
# The workflow JSON settled it in one read:
#     #155 Token Counter . trigger_words  <-  #256 Stack HIGH . uls_config_out
#     #156 Token Counter . trigger_words  <-  #257 Stack LOW  . trigger_words
#
# The HIGH counter was measuring the serialised Stack config -- 15 rows of JSON,
# ~6000 chars. STRING fits into STRING, so LiteGraph passed it without a word.
#
# The v568 wire law guards "unwired". Nothing guarded "wired, but to the wrong
# socket" -- and a socket NAME is a label, not a contract. So the counter now
# looks at what it was handed. A trigger list is a handful of short words. A
# config object announces itself in the first character.
_TRIG_SANE_CHARS = 600      # 11 LoRAs of real triggers measured 129 chars


def _wrap_note(text, width):
    """Soft-wrap on word boundaries. Pure; no textwrap import for four lines."""
    out, line = [], ""
    for word in str(text).split():
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


def _mis_wired_trigger_input(s, n_tokens):
    """'' if plausible, else the reason. Pure -- the guard executes this."""
    t = str(s or "").strip()
    if not t:
        return ""
    head = t[:1]
    if head in "{[":
        return (f"the 'trigger_words' input is being handed JSON, not triggers "
                f"({n_tokens} tokens, {len(t)} chars, starts with '{head}'). That is "
                f"almost certainly the Stack's 'uls_config_out' socket on the wrong "
                f"input -- rewire it to 'trigger_words'. The count below is meaningless.")
    for probe in ('"rows"', '"lora_info"', '"uls_config"', '"widgets_values"'):
        if probe in t:
            return (f"the 'trigger_words' input contains {probe} -- that is a config "
                    f"object, not a trigger list. Rewire it to the Stack's "
                    f"'trigger_words' socket. The count below is meaningless.")
    if len(t) > _TRIG_SANE_CHARS:
        return (f"the 'trigger_words' input is {len(t)} chars / {n_tokens} tokens long. "
                f"A real trigger list is a handful of words (Frank's 11-LoRA stack "
                f"measures 129 chars). Check what is wired here, and check the "
                f"companion .txt of the LoRAs in the stack.")
    return ""


_UMT5_TOKENIZER = None  # cached after first successful load
_UMT5_LOAD_ATTEMPTED = False


def _try_load_umt5_tokenizer():
    """Lazy-load UMT5-XXL tokenizer once. Returns tokenizer or None."""
    global _UMT5_TOKENIZER, _UMT5_LOAD_ATTEMPTED
    if _UMT5_LOAD_ATTEMPTED:
        return _UMT5_TOKENIZER
    _UMT5_LOAD_ATTEMPTED = True
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("[PLS Tokens] transformers not installed — using heuristic estimator")
        return None
    # v267 (audit A-4): genuinely offline-friendly two-stage attempt — first
    # the LOCAL cache only (no network round-trip, no online etag check when
    # cached), then the regular online fetch (which caches for next time).
    # Name matches WAN's config.
    for name in ("google/umt5-xxl", "google/umt5-base"):
        for _local_only in (True, False):
            try:
                _UMT5_TOKENIZER = AutoTokenizer.from_pretrained(
                    name, legacy=False, local_files_only=_local_only)
                _src = "local cache" if _local_only else "online (now cached)"
                print(f"[PLS Tokens] ✓ Loaded {name} tokenizer (exact counts, {_src})")
                return _UMT5_TOKENIZER
            except Exception:
                continue
    print("[PLS Tokens] UMT5 tokenizer not in local cache — using heuristic estimator")
    print("[PLS Tokens]   For exact counts: pip install transformers + first online run will cache it")
    return None


def _heuristic_token_count(text: str) -> int:
    """
    SentencePiece-style heuristic for UMT5-XXL token count.

    Calibrated against the empirical observation that the user's v163 crash
    prompt of 2712 characters / 348 words produced exactly 635 tokens (the
    crash log said 'negative dimension -123', i.e. 512 - actual = -123 →
    actual = 635). The calibration yields ~5% accuracy on prompts of this
    style; mileage on heavily non-English or symbol-dense prompts may vary.

    Two-term formula:
      base       = chars / 4.3   (UMT5-XXL English-mix average)
      digit_bonus = 1.5 per digit-mid-word (stor8m, ne8ttle, w3tcl0, ...)
                    — these trigger fragment splits in SentencePiece, each
                    costing roughly 2 extra tokens vs the surface form.

    For exact counts the node prefers the real UMT5-XXL tokenizer via
    `transformers`; this heuristic is the always-available fallback.
    """
    if not text:
        return 0
    # Base: UMT5-XXL averages ~4.3 chars per token for English-mix text
    # (validated against transformers UMT5-XXL output on the v163 prompt)
    base = len(text) / 4.3
    # Correction: digit-in-the-middle words like "stor8m" or "ne8ttle" get
    # aggressively fragmented by SentencePiece. Each adds ~1.5 extra tokens
    # over the chars/4.3 baseline.
    digit_words = re.findall(r"\b\w*\d\w+\b", text)
    digit_bonus = len(digit_words) * 1.5
    return max(1, round(base + digit_bonus))


# ---------------------------------------------------------------------------
# v905: ask the LIVE encoder instead of guessing.
#
# Until v904 the count came from a UMT5-XXL tokenizer we loaded ourselves,
# next to the one ComfyUI had already loaded. Two places computing the same
# thing drift -- and this one drifted twice over:
#
#   * WRONG TOKENIZER for anything that is not WAN. MiniMax H3 encodes with
#     Qwen3-VL, Flux2-Klein with Qwen3-4B. Measured against the core
#     tokenizers, UMT5 reads 10-20% high on prose and low on weighted syntax.
#   * WRONG LIMIT. `min_length` is a PAD FLOOR, not a cap. In core v0.32.0
#     every modern family carries `max_length=99999999, pad_to_max_length=
#     False`; umt5xxl (WAN) has min_length=512, qwen3_4b 512, t5xxl 256.
#     They pad UP to that number and never truncate. The 512 the node has
#     been reporting as "the limit" is the floor, and comfy/ldm/wan/model.py
#     assigns self.text_len = 512 and then never reads it again.
#
# MEASURED, not assumed: KleinTokenizer (qwen3_4b, min_length=512) on a
# 13-word sentence returns 512 tokens by default and 29 with min_length=0.
# Counting a live tokenizer WITHOUT passing min_length=0 therefore reports
# the floor as if it were the prompt. That is why the kwarg is mandatory
# below and why a tokenizer that refuses it is not used at all.
# ---------------------------------------------------------------------------

# Core writes a literal 99999999 for "no cap". Read it as a threshold rather
# than an equality so a future core that writes 2**31 still means the same.
_NO_CAP_AT_OR_ABOVE = 1_000_000


# ---------------------------------------------------------------------------
# v907 -- WHAT "TOO LONG" MEANS ON MINIMAX H3.
#
# It does not mean truncation. Measured in core v0.33.4,
# comfy/text_encoders/qwen3vl.py: max_length=99999999, pad_to_max_length=False.
# Nothing is cut, ever. A report that warns "over 512" on H3 is inventing a
# limit, and Frank spent a whole prompt session trimming against one.
#
# What DOES happen is structural, and it comes out of the model itself.
# H3 is a single-stream packed-token transformer: text, video and audio share
# ONE sequence and ONE position axis (comfy/ldm/minimax/model.py, PackedLayout):
#
#     segments = [("text", text_len)]
#     g[:, 0] = torch.arange(text_len)   # each text token costs 1.0 on t
#     cursor  = text_len                 # the video starts BEHIND the text
#
# A latent frame costs FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] on that same
# axis -- 1.67 or 6.67, not 1. So the text does not merely sit next to the
# video, it PUSHES it along the time axis, and a long prompt pushes it far:
# 378 text tokens against a 22-frame clip put the video at t=378..415, while
# the clip itself spans 36.7. The prompt occupies ten times the video's own
# extent.
#
# Two consequences follow, and only these two are claimed here:
#   * RoPE encodes DISTANCE. The first token of a 378-token prompt sits 378
#     units from the video; the last sits next to it. Early parts of a long
#     prompt therefore pull measurably weaker than late ones.
#   * The video's absolute start position grows with prompt length, which is
#     the same class of fault as v900 (a coordinate outside anything training
#     saw) -- there the still's 0.0, here a start at t=378.
#
# The 512 is NOT invented either, but it is not the encoder's: ai-toolkit's H3
# trainer defaults `max_text_length` to 512 and says in the same breath that
# "the released stack has no limit". So 512 is the span LoRAs are trained
# within -- a training convention worth knowing, not a cliff to fall off.
#
# These helpers are pure and mirror the model's own constants. If core ever
# changes FRAME_PER_TOKEN or FRAME_RESCALE, this is the one place to follow.
# ---------------------------------------------------------------------------

H3_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
H3_FRAME_RESCALE = 5.0 / 3.0
H3_TRAIN_SPAN = 512          # ai-toolkit max_text_length default


def _any_encoder_truncates(clip) -> bool:
    """True only if some live encoder really has a cap.

    No clip wired -> True: without knowing the encoder we must not promise
    that nothing is cut. Silence about a real risk is worse than a caveat.
    """
    facts = []
    try:
        facts = _encoder_facts(clip)
    except Exception:
        return True
    if not facts:
        return True
    return any(f.get("cap") is not None for f in facts)


def _encoder_label(clip):
    """The live encoder's name for the toast, or None."""
    try:
        facts = _encoder_facts(clip)
    except Exception:
        return None
    return facts[0].get("name") if facts else None


def _is_h3_encoder(clip) -> bool:
    """True only for a MiniMax H3 text encoder.

    Read off the live tokenizer via the v905 facts helper, never guessed from
    the model or the latent: the H3 section below is only true for H3, and
    printing it elsewhere would be exactly the kind of invented number this
    whole block exists to remove.
    """
    # v908 -- MEASURED CORRECTION. v907 looked for "minimax" in the inner
    # tokenizer's NAME and never fired, because H3's inner tokenizer is called
    # `qwen3vl_32b` -- the same name a plain Qwen3-VL model uses
    # (comfy/text_encoders/minimax.py: MiniMaxH3Tokenizer passes
    # embedding_key="qwen3vl_32b"). The name cannot identify H3 and never
    # could. The OUTER class can: MiniMaxH3Tokenizer is the thing that builds
    # the packed sequence this whole report is about.
    try:
        tk = getattr(clip, "tokenizer", None)
        if tk is not None and "minimax" in type(tk).__name__.lower():
            return True
    except Exception:
        pass
    return False


def h3_video_span(latent_t):
    """The extent one video of `latent_t` latent frames occupies on the shared
    time axis. DECLARED MIRROR of _video_t_spans in comfy/ldm/minimax/model.py.
    """
    try:
        n = int(latent_t)
    except (TypeError, ValueError):
        return 0.0
    if n <= 0:
        return 0.0
    return sum(H3_FRAME_RESCALE * H3_FRAME_PER_TOKEN[k % 5] for k in range(n))


def h3_reach(text_tokens, latent_t):
    """How the prompt sits on the time axis relative to the clip.

    Returns {"text": float, "video": float, "ratio": float|None,
             "start": float, "end": float} -- ratio None when there is no
    video to compare against (a still, or no latent wired). Ratio is the
    number that carries the meaning: 1.0 means the prompt occupies as much of
    the axis as the clip does.
    """
    try:
        t = float(int(text_tokens))
    except (TypeError, ValueError):
        t = 0.0
    v = h3_video_span(latent_t)
    return {
        "text": t,
        "video": v,
        "ratio": (t / v) if v > 0 else None,
        "start": t,
        "end": t + v,
    }


def h3_latent_frames(latent):
    """Latent-frame count out of a LATENT dict, joint or plain.

    Reuses the split rule the VAE node already declares: a joint AV latent is
    nested, and the VIDEO half is part 0. Returns 0 when anything is missing --
    the caller must survive that, because this input is optional by design.
    """
    if not isinstance(latent, dict):
        return 0
    sam = latent.get("samples", None)
    if sam is None:
        return 0
    if bool(getattr(sam, "is_nested", False)):
        try:
            parts = sam.unbind()
        except Exception:
            return 0
        if not parts:
            return 0
        sam = parts[0]
    shape = getattr(sam, "shape", None)
    if shape is None or len(shape) < 5:
        return 0
    try:
        return int(shape[2])          # (B, C, T, H, W)
    except (TypeError, ValueError, IndexError):
        return 0


def _encoder_facts(clip) -> list:
    """
    Read the REAL tokenizer limits off a live CLIP object.

    Returns one dict per inner tokenizer:
        {"name": str, "cap": int|None, "pad_floor": int, "pads_to_cap": bool}
    `cap` is None when the encoder does not truncate at all. Returns [] if
    anything is missing or unreadable -- every caller must work without it.
    """
    tk = getattr(clip, "tokenizer", None)
    if tk is None:
        return []
    facts = []
    try:
        members = vars(tk)
    except TypeError:
        return []
    for attr, inner in members.items():
        # An inner SDTokenizer is identified by carrying BOTH limits; that is
        # what we are here to read, so it is also the right duck test.
        if not (hasattr(inner, "max_length") and hasattr(inner, "min_length")):
            continue
        try:
            cap = int(getattr(inner, "max_length"))
            floor = int(getattr(inner, "min_length") or 0)
            pads = bool(getattr(inner, "pad_to_max_length", False))
        except (TypeError, ValueError):
            continue
        facts.append({
            "name": str(getattr(inner, "embedding_key", attr) or attr),
            "cap": None if cap >= _NO_CAP_AT_OR_ABOVE else cap,
            "pad_floor": floor,
            "pads_to_cap": pads,
        })
    return facts


def _count_with_clip(clip, text: str):
    """
    Exact count from the encoder that will actually run. Returns
    (count, label) or (None, "") when this path is not usable.

    min_length=0 is NOT optional -- see the module note above. A tokenizer
    that will not take it would report its pad floor as the prompt length,
    so we decline instead of reporting a number we do not believe.
    """
    try:
        out = clip.tokenize(text, min_length=0)
    except Exception as e:
        print("[PLS Tokens] live tokenizer declined min_length=0 (%r) -- "
              "falling back rather than reporting a padded count." % (e,))
        return (None, "")
    if not isinstance(out, dict) or not out:
        return (None, "")
    best_n, best_name = -1, ""
    for key, chunks in out.items():
        try:
            n = sum(len(c) for c in chunks)
        except TypeError:
            continue
        if n > best_n:
            best_n, best_name = n, str(key)
    if best_n < 0:
        return (None, "")
    return (best_n, best_name)


def _count_tokens(text: str, clip=None) -> tuple:
    """
    Returns (count, method) where method is "exact", "heuristic", or the
    live encoder name when a CLIP object was handed in.

    `clip` is optional and appended, so every existing caller keeps its
    exact v904 behaviour. With a clip the count comes from the tokenizer
    that will actually encode the prompt -- correct for every family, no
    download, no `transformers`, and no second tokenizer to drift against.
    """
    if not text:
        return (0, "exact")
    if clip is not None:
        n, name = _count_with_clip(clip, text)
        if n is not None:
            return (n, name)
    tok = _try_load_umt5_tokenizer()
    if tok is not None:
        try:
            return (len(tok.encode(text, add_special_tokens=True)), "exact")
        except Exception as e:
            print(f"[PLS Tokens] tokenizer failed ({e}) — falling back to heuristic")
    return (_heuristic_token_count(text), "heuristic")


def _make_bar(used: int, total: int, width: int = 32) -> str:
    """ASCII progress bar."""
    if total <= 0:
        return "[" + "?" * width + "]"
    ratio = min(1.5, used / total)  # cap at 150% for visual
    filled = int(round(ratio * width))
    if filled <= width:
        bar = "█" * filled + "░" * (width - filled)
    else:
        overflow = filled - width
        bar = "█" * width + "▓" * min(overflow, 8) + "!" * max(0, overflow - 8)
    return f"[{bar}]"


class ULSTokenCounter:
    """
    Polyhedron Token Counter — diagnostic node for prompt budgets.

    v905 CORRECTION, measured against core v0.32.0. This node used to
    report 512 as "the model limit" for WAN. That number is a PAD FLOOR,
    not a cap: comfy/text_encoders/wan.py builds umt5xxl with
    `max_length=99999999, pad_to_max_length=False, min_length=512`, and
    comfy/ldm/wan/model.py assigns `self.text_len = 512` and then never
    reads it. Short prompts are padded UP to 512; long ones are not cut.
    The same holds for every modern family (t5xxl 256, qwen3_4b 512,
    qwen3vl_32b 1) -- none of them truncates.

    Where 512 IS real: kijai's WanVideoWrapper, whose fixed 512-wide
    buffer produced the crash this node was originally built for
    ("RuntimeError: Trying to create tensor with negative dimension").
    That path is still worth guarding, which is why `model_limit` stays --
    but it is now labelled as YOUR budget, not as the encoder's cap.

    Wire the optional `clip` input and the count comes from the tokenizer
    that will actually run: exact for every family, no download, no
    `transformers`. Without it the node behaves exactly as in v904.

    No model patching — purely informational.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_limit": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 8192,
                    "step": 64,
                    "tooltip": "YOUR token budget -- a self-set guide rail, not "
                               "the encoder's cap. Measured against core "
                               "v0.32.0, no modern text encoder truncates: 512 "
                               "on WAN is a pad FLOOR. Wire `clip` and the "
                               "report names the real limits. 512 is still the "
                               "right number for kijai's WanVideoWrapper, whose "
                               "buffer really is fixed. Old guidance: SDXL = 75 per CLIP "
                               "chunk. Leave at 512 for WAN.",
                }),
                "warn_threshold": ("FLOAT", {
                    "default": 0.90,
                    "min": 0.50,
                    "max": 1.00,
                    "step": 0.05,
                    "tooltip": "Fraction of the limit at which to warn. 0.90 "
                               "means warn at ≥460 of 512 tokens.",
                }),
            },
            "optional": {
                "positive_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "Positive prompt to count.",
                }),
                "negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "Negative prompt to count.",
                }),
                "trigger_words": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "Optional: wire the Stack's trigger_words output here "
                               "to see how much of the budget the auto-collected "
                               "triggers alone consume. Diagnostic only — these are "
                               "normally already part of the positive prompt, so they "
                               "are NOT added to over_limit.",
                }),
                # v905, APPENDED (#577): a socket, not a widget -- widget_values
                # serialises by index and the widget baseline stays
                # "model_limit,warn_threshold" untouched.
                "clip": ("CLIP", {
                    "tooltip": "Optional but recommended: the same CLIP that "
                               "encodes this prompt. The count then comes from "
                               "the tokenizer that will actually run -- exact "
                               "for WAN, MiniMax H3, Flux2 and everything else "
                               "-- and the report names that encoder's REAL "
                               "limits instead of assuming WAN's. Without it "
                               "the node falls back to UMT5-XXL, which is only "
                               "correct for WAN.",
                }),
                # v907, APPENDED (#577): a socket like `clip`, so the widget
                # baseline stays "model_limit,warn_threshold" untouched.
                "latent": ("LATENT", {
                    "tooltip": "Optional, MiniMax H3 only: wire the same latent "
                               "the sampler gets. H3 puts text and video on ONE "
                               "position axis, so the prompt's length decides "
                               "where the video sits on it. With the latent the "
                               "report says how far the prompt pushes the clip "
                               "-- the only 'too long' that means anything on "
                               "H3, since its encoder never truncates.",
                }),
            },
        }

    RETURN_TYPES  = ("STRING", "INT", "INT", "BOOLEAN", "INT")
    RETURN_NAMES  = ("report", "positive_tokens", "negative_tokens", "over_limit", "trigger_tokens")
    FUNCTION      = "count"
    CATEGORY      = "Polyhedron/Utils"
    DESCRIPTION = ("Counts the prompt against the text encoder's hard token limit and says "
                   "how close it is. Over the limit, the native path truncates in silence: "
                   "no crash, the end of the prompt simply never reaches the model. It also "
                   "distrusts its own trigger_words input - if that socket is handed a "
                   "serialised config instead of triggers, it says so on the number itself "
                   "rather than printing a count it does not believe.")
    OUTPUT_NODE   = False

    def count(self,
              model_limit: int,
              warn_threshold: float,
              positive_prompt: str = "",
              negative_prompt: str = "",
              trigger_words: str = "",
              clip=None,
              latent=None) -> tuple:

        pos_count, pos_method = _count_tokens(positive_prompt, clip)
        neg_count, neg_method = _count_tokens(negative_prompt, clip)
        facts = _encoder_facts(clip) if clip is not None else []
        # Trigger words are diagnostic only — see the report note and the
        # RETURN comment. Always defined so the trigger_tokens output is stable.
        trig_count = 0
        trig_alarm = ""
        if trigger_words and trigger_words.strip():
            trig_count, _ = _count_tokens(trigger_words, clip)
            trig_alarm = _mis_wired_trigger_input(trigger_words, trig_count)
            if trig_alarm:
                print(f"[PLS] Token Counter: {trig_alarm}")

        # Use the same method label if both are exact, otherwise show mixed
        # v905: only prompts that actually carry text get a vote. An empty
        # negative returns "exact" for zero tokens and used to drag the label
        # to "mixed" next to a perfectly exact positive count. Seen while
        # rendering the report, not while reading it.
        _voted = [mth for mth, txt in ((pos_method, positive_prompt),
                                       (neg_method, negative_prompt)) if txt]
        if not _voted:
            method = "exact"
        elif len(set(_voted)) == 1:
            method = _voted[0]
        else:
            method = "mixed"
        method_label = {
            "exact":     "UMT5-XXL tokenizer (exact) — assumes WAN",
            "heuristic": "heuristic estimator (±5%) — assumes WAN",
            "mixed":     "mixed (some exact, some heuristic)",
        }.get(method, "%s — the live encoder (exact)" % method)

        pos_pct = (pos_count / model_limit * 100) if model_limit else 0.0
        neg_pct = (neg_count / model_limit * 100) if model_limit else 0.0
        warn_at = int(model_limit * warn_threshold)

        pos_status = self._status(pos_count, model_limit, warn_at)
        neg_status = self._status(neg_count, model_limit, warn_at)

        over_limit = (pos_count > model_limit) or (neg_count > model_limit)

        # Build report
        lines = []
        if over_limit:
            # The loud alert is now the native ComfyUI toast (uls_token_toast.js,
            # visible anywhere). Keep a compact one-line marker in the report for
            # anyone reading the text output directly.
            lines.append(">>> ⚠ OVER YOUR BUDGET — see details below <<<")
        lines += [
            "═══ Polyhedron Token Counter ═══",
            f"  Your budget : {model_limit} tokens  (self-set, not the encoder's cap)",
            f"  Warn at     : {warn_at} tokens ({int(warn_threshold * 100)}%)",
            f"  Method      : {method_label}",
        ]
        # v905: what the ENCODER actually does, read off the live object.
        # Without a clip we say so instead of implying we know.
        if facts:
            for f in facts:
                cap_txt = ("no cap (never truncates)" if f["cap"] is None
                           else "CUTS at %d" % f["cap"])
                floor_txt = (" · pads up to %d" % f["pad_floor"]) if f["pad_floor"] > 1 else ""
                lines.append(f"  Encoder     : {f['name']} · {cap_txt}{floor_txt}")
        else:
            lines.append("  Encoder     : unknown — wire `clip` for the real limits")
        lines += [
            "─────────────────────────────────",
            f"  POSITIVE    : {pos_count:>4} / {model_limit}  ({pos_pct:5.1f}%)  {pos_status}",
            f"                {_make_bar(pos_count, model_limit)}",
        ]
        if pos_count > 0:
            words = len(positive_prompt.split())
            ratio = pos_count / max(1, words)
            lines.append(f"                {words} words → {ratio:.2f} tokens/word")
        lines.append("")
        lines.append(f"  NEGATIVE    : {neg_count:>4} / {model_limit}  ({neg_pct:5.1f}%)  {neg_status}")
        lines.append(f"                {_make_bar(neg_count, model_limit)}")
        if neg_count > 0:
            words = len(negative_prompt.split())
            ratio = neg_count / max(1, words)
            lines.append(f"                {words} words → {ratio:.2f} tokens/word")
        # Trigger words (optional 3rd input) — diagnostic only. These are
        # typically ALREADY part of the positive prompt (wired via JoinStrings),
        # so they are deliberately NOT folded into over_limit; showing them
        # separately answers "how much of my budget do the auto-collected
        # triggers eat?" (open point #2 from the roadmap).
        if trig_count > 0:
            trig_pct = (trig_count / model_limit * 100) if model_limit else 0.0
            lines.append("")
            lines.append(f"  TRIGGERS    : {trig_count:>4} / {model_limit}  ({trig_pct:5.1f}%)  (auto-collected)")
            lines.append(f"                {_make_bar(trig_count, model_limit)}")
            if trig_alarm:
                # v580: the alarm sits ON the number it distrusts. Frank's 1710
                # was believed for three rounds because nothing next to it ever
                # asked whether a trigger list could look like that.
                lines.append("                ⚠ THIS NUMBER IS NOT A TRIGGER COUNT:")
                for chunk in _wrap_note(trig_alarm, 60):
                    lines.append(f"                  {chunk}")
            else:
                lines.append("                ℹ usually already inside POSITIVE — informational, not added to over-limit")

        # --- v907: on H3 the only meaningful "too long" ---------------------
        # Printed only when the encoder really is H3. On every other model the
        # numbers below have no meaning, and a figure without meaning is worse
        # than no figure at all.
        if _is_h3_encoder(clip):
            lat_t = h3_latent_frames(latent)
            reach = h3_reach(pos_count, lat_t)
            lines.append("─────────────────────────────────")
            lines.append("  MiniMax H3 — position axis")
            lines.append("    Nothing is truncated here: this encoder has no cap.")
            if reach["ratio"] is None:
                lines.append("    Wire `latent` to see how far the prompt pushes")
                lines.append("    the clip along the shared time axis.")
            else:
                lines.append(f"    prompt spans {reach['text']:.0f} · clip spans "
                             f"{reach['video']:.1f} → prompt is "
                             f"{reach['ratio']:.1f}x the clip")
                lines.append(f"    the video sits at t={reach['start']:.0f}"
                             f"..{reach['end']:.1f}, not at 0")
                if reach["ratio"] >= 8.0:
                    lines.append("    ⚠ the prompt's opening is far from the video;")
                    lines.append("      early lines pull weaker than late ones.")
                    lines.append("      Put what matters most LAST.")
            if pos_count > H3_TRAIN_SPAN:
                lines.append(f"    ⚠ over {H3_TRAIN_SPAN} tokens — not a cap, but the span")
                lines.append("      LoRAs are trained within (ai-toolkit default).")
        lines.append("─────────────────────────────────")

        # Actionable hints
        hints = []
        if over_limit:
            over_by = max(pos_count, neg_count) - model_limit
            hints.append(f"⚠ OVER YOUR BUDGET by {over_by} token(s).")
            # v905: say what actually happens, and only where it is true.
            # The old text promised silent truncation for every path. Measured
            # against core v0.32.0 that is false for the core encoders; it is
            # kijai's fixed 512-wide buffer that really breaks.
            real_cap = next((f["cap"] for f in facts if f["cap"] is not None), None)
            if real_cap is not None:
                hints.append(f"  The encoder DOES cut at {real_cap} tokens — past that the")
                hints.append("  tail of the prompt never reaches the model.")
            elif facts:
                hints.append("  This encoder has NO cap — nothing is truncated, so this is")
                hints.append("  your own guide rail talking, not a failure. Long prompts")
                hints.append("  still cost attention and tend to dilute the subject.")
            else:
                hints.append("  Whether anything is actually cut depends on the encoder;")
                hints.append("  wire `clip` and this line will say so instead of guessing.")
            hints.append("  Where 512 is REAL: kijai's WanVideoWrapper has a fixed")
            hints.append("    512-wide buffer → 'RuntimeError: Trying to create tensor")
            hints.append("    with negative dimension'. On that path, stay under it.")
            hints.append("  • Shorten the prompt — drop redundant tags and remove any")
            hints.append("    (word:1.x) weight syntax: WAN/UMT5 ignores the weighting,")
            hints.append("    and so does MiniMax H3, but both still spend tokens on the")
            hints.append("    digits and parentheses (measured: 8 words → 35 tokens).")
        elif (pos_count >= warn_at) or (neg_count >= warn_at):
            pct = int(round(warn_threshold * 100))
            hints.append(f"⚠ Approaching limit — at or above the {pct}% warn threshold")
            hints.append(f"  ({warn_at}/{model_limit} tokens). Quality tends to degrade as the")
            hints.append("  budget fills (motion slows, 'grid' patterns appear in output")
            hints.append("  — kijai issue #1781). Consider trimming before you hit the cap.")
        if method == "heuristic":
            hints.append("ℹ For exact counts: wire the `clip` input — no download, no")
            hints.append("  `transformers`, and correct for whichever encoder you run.")
            hints.append("  (The old route still works: `pip install transformers` plus a")
            hints.append("  cached UMT5-XXL, but that is only right for WAN.)")

        if hints:
            for h in hints:
                lines.append(f"  {h}")
            lines.append("─────────────────────────────────")

        report = "\n".join(lines)
        print(f"\n[PLS Tokens]\n{report}\n")

        # v318: also hand the frontend structured numbers via the UI channel so
        # an onExecuted hook can raise a native ComfyUI toast on over-limit
        # WITHOUT parsing the report text. (Backend can't toast directly.)
        ui = {"pls_tokens": [{
            "over_limit":   bool(over_limit),
            "pos":          int(pos_count),
            "neg":          int(neg_count),
            "limit":        int(model_limit),
            "warn_at":      int(warn_at),
            "near_limit":   bool((pos_count >= warn_at) or (neg_count >= warn_at)),
            # v908: the toast used to promise truncation and a kijai crash for
            # EVERY over-budget run. On an encoder that cannot truncate, that
            # is a lie loud enough to act on -- and Frank did act on it. The
            # frontend now gets the two facts it needs to say something true.
            "can_truncate": bool(_any_encoder_truncates(clip)),
            "encoder":      str(_encoder_label(clip) or ""),
        }]}
        return {"ui": ui,
                "result": (report, pos_count, neg_count, over_limit, trig_count)}

    @staticmethod
    def _status(count: int, limit: int, warn_at: int) -> str:
        if count == 0:
            return "(empty)"
        if count > limit:
            return f"✗ OVER by {count - limit}"
        if count >= warn_at:
            return "⚠ near limit"
        return "✓ ok"
