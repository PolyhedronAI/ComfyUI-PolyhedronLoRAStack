"""
Polyhedron progress  (ph_progress)
==================================
v1010: ONE instrument for every long-running node of the suite.

Frank, 25.09.2026: "dann fehlen mir fuer unsere Nodes, die ein wenig laenger
brauchen: Interpolate oder VAE decode ... und andere dieser gruene
Progress-Balken oder auch in der Konsole, dass man sieht, was gerade laeuft
... wo man noch etwas on the fly oder eben im zweiten Laufen messen und
anzeigen kann".

Two grips on one time model:

  NodeProgress  -- work that COUNTS (frames, tasks, chunks, layers). The
                   node calls tick(n); the green node bar follows, the console
                   speaks at most every few seconds (count, percent, elapsed,
                   rate, ETA). Before the first tick the ETA comes from the
                   rate learned on earlier runs; after it, from this run.
  blocking      -- ONE call that cannot report (a whole-clip VAE decode, a
                   model load, a file finalise). A heartbeat thread speaks
                   every 15 s; the bar follows elapsed / learned estimate and
                   stops at 95 % until the call returns, so it never claims a
                   finish it has not seen. Peak VRAM is measured per call.

Both learn: seconds per unit of work (a frame, a megapixel-frame, a MB) are
folded in by EMA and kept in <user>/polyhedron/rates.json, one SECTION per
node path and model class, so the SECOND run of a size class opens with a
real estimate. Nothing here can break a run: every bar, print and file
access is wrapped. The v1007/v1008 Power Upscale clock (the heartbeat
_Phase, _phase_plan, _memory_note, _peak_line and the rates store) moved
here unchanged in substance; ph_power_upscale re-exports it.
"""
import json
import os
import threading
import time

try:
    import torch
except Exception:  # pragma: no cover - a comfy-less import (tools)
    torch = None
try:
    import folder_paths
except Exception:  # pragma: no cover
    folder_paths = None
try:
    from .ph_runclock import _fmt_clock
except ImportError:  # pragma: no cover - direct-run fallback
    from ph_runclock import _fmt_clock

_HUD_EVENT = "polyhedron.pu_tile"   # the Power Upscale process pane (sent only with a node id)


# ── v1007/v1008: the clock -- no phase is silent, on either path ───────────
# Frank's 24.09. run: "final pass done" ... 126 s of nothing ... "h3 refine
# begin" ... a 60 s step with no tick ... 80 s of nothing ... "done". Console,
# node HUD and progress bar all went dark for minutes at a time. Three
# griffs, one time model, now for the tile refine as well:
#   1. RATES are learned per phase in seconds per megapixel-frame (w*h*n/1e6)
#      and remembered in the user directory (they survive a restart), one
#      SECTION per path and model class ('joint:MiniMaxH3', 'tile:WAN22'), so
#      the SECOND run of a size class opens with a real plan;
#   2. a PLAN names the phases with their estimates before the first one starts
#      (or says honestly that nothing is learned yet);
#   3. a HEARTBEAT thread ticks every 15 s inside every phase that has no
#      callback of its own (the encode, each sampling step, the decode) --
#      the same line to the console, the node HUD and the progress bar.
# v1008 adds the MEMORY half: each phase measures its peak VRAM (torch's own
# allocator counter), the refine prints one line with the peaks against the
# card, and the section remembers the last peak with the size it was measured
# at -- so a larger run can be told BEFORE it starts what the last one used.
_HEARTBEAT_S = 15.0
_RATES_EMA = 0.5
_RATES = {}            # section -> {"enc","step","dec","runs","when","peak_gb","peak_mpf"}
_RATES_LOADED = False
_RATES_LEGACY = "joint:legacy"   # v1007's h3_rates.json, read once


def _rates_path():
    try:
        return os.path.join(folder_paths.get_user_directory(), "polyhedron", "rates.json")
    except Exception:
        return None


def _rates_v1008_path():
    """v1008/v1009 kept the sections in pu_rates.json (Power Upscale only)."""
    try:
        return os.path.join(folder_paths.get_user_directory(), "polyhedron", "pu_rates.json")
    except Exception:
        return None


def _rates_legacy_path():
    try:
        return os.path.join(folder_paths.get_user_directory(), "polyhedron", "h3_rates.json")
    except Exception:
        return None


def _rates_blank():
    return {"enc": None, "step": None, "dec": None, "runs": 0, "when": "",
            "peak_gb": None, "peak_mpf": None}


def _rates_clean(d):
    r = _rates_blank()
    if not isinstance(d, dict):
        return r
    for k in ("enc", "step", "dec", "peak_gb", "peak_mpf"):
        v = d.get(k)
        r[k] = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 else None
    try:
        r["runs"] = int(d.get("runs", 0) or 0)
    except Exception:
        r["runs"] = 0
    r["when"] = str(d.get("when", "") or "")
    return r


def _rates_section(kind, model):
    """'joint:<model class>' / 'tile:<model class>' -- rates are a property of
    the path AND the model (an SD1 step is not a Wan step)."""
    inner = getattr(model, "model", model)
    return "%s:%s" % (kind, type(inner).__name__)


def _rates_load(section):
    """The rates of one section (a live dict -- _rates_learn updates it). The
    file is read ONCE per session; v1007's flat h3_rates.json seeds any joint
    section that has not learned yet, so no measurement is lost."""
    global _RATES_LOADED
    if not _RATES_LOADED:
        _RATES_LOADED = True
        path = _rates_path()
        if not (path and os.path.isfile(path)):
            path = _rates_v1008_path()   # v1010: the v1008/v1009 file seeds the shared one
        try:
            if path and os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as fh:
                    d = json.load(fh)
                for k, v in (d.get("sections", {}) or {}).items():
                    _RATES[str(k)] = _rates_clean(v)
        except Exception:
            pass
        lp = _rates_legacy_path()
        try:
            if lp and os.path.isfile(lp) and _RATES_LEGACY not in _RATES:
                with open(lp, "r", encoding="utf-8") as fh:
                    _RATES[_RATES_LEGACY] = _rates_clean(json.load(fh))
        except Exception:
            pass
    if section not in _RATES:
        seed = _RATES.get(_RATES_LEGACY) if str(section).startswith("joint:") else None
        _RATES[section] = dict(seed) if seed else _rates_blank()
    return _RATES[section]


def _rates_learn(section, mpf, enc_s, step_s, dec_s, peak_gb=None):
    """Fold one measurement into the section's rates (EMA) and remember them.
    A phase that did not run (cache hit: enc_s None) leaves its rate alone. The
    peak is the LAST measured one with the size it belongs to (a projection
    needs a pair, not an average)."""
    r = _rates_load(section)
    mpf = max(1e-6, float(mpf))
    for k, sec in (("enc", enc_s), ("step", step_s), ("dec", dec_s)):
        if sec is None or sec <= 0:
            continue
        rate = float(sec) / mpf
        r[k] = rate if r[k] is None else (_RATES_EMA * rate + (1.0 - _RATES_EMA) * r[k])
    if peak_gb is not None and peak_gb > 0:
        r["peak_gb"], r["peak_mpf"] = float(peak_gb), float(mpf)
    r["runs"] = int(r.get("runs", 0)) + 1
    r["when"] = time.strftime("%Y-%m-%d %H:%M")
    path = _rates_path()
    try:
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"sections": {k: v for k, v in _RATES.items()
                                        if k != _RATES_LEGACY}}, fh)
    except Exception:
        pass
    return r


def _phase_plan(mpf, steps_run, cached, rates):
    """(lines, est) -- the banner lines and the per-phase estimates in seconds
    (None where nothing is learned). Pure."""
    est = {}
    for k in ("enc", "step", "dec"):
        est[k] = (rates.get(k) * mpf) if rates.get(k) else None
    if cached:
        est["enc"] = 0.0
    total = None
    parts = [est["enc"], (est["step"] * steps_run) if est["step"] is not None else None, est["dec"]]
    if all(v is not None for v in parts):
        total = sum(parts)
    fmt = lambda v: ("~" + _fmt_clock(v)) if v is not None else "no rate yet"
    lines = [f"phase 1/3 encode   {'skipped (cache)' if cached else fmt(est['enc'])}",
             f"phase 2/3 sample   {steps_run} step{'s' if steps_run != 1 else ''} "
             f"{('~' + _fmt_clock(est['step']) + '/step') if est['step'] is not None else 'no rate yet'}",
             f"phase 3/3 decode   {fmt(est['dec'])}",
             (f"total {fmt(total)} -- rates learned from {rates.get('runs', 0)} run(s), "
              f"last {rates.get('when') or '?'}; the clock corrects as each phase measures")
             if total is not None else
             "total: no rate learned yet -- this run measures them; a heartbeat ticks every "
             f"{int(_HEARTBEAT_S)} s so nothing is silent"]
    return lines, est


def _vram_total_gb():
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1e9
    except Exception:
        pass
    return None


def _vram_peak_reset():
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _vram_peak_gb():
    try:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass
    return None


def _memory_note(rates, mpf, total_gb, levers):
    """ONE line BEFORE a refine, from the LAST measured peak of this section --
    or None when there is nothing measured to say. Never a prediction dressed
    as a fact: it names the pair it knows and the size ratio. Pure."""
    pk, pm = rates.get("peak_gb"), rates.get("peak_mpf")
    if not pk or not pm or not total_gb:
        return None
    ratio = float(mpf) / float(pm)
    tight = pk >= 0.85 * float(total_gb)
    if ratio <= 1.05 and not tight:
        return None
    head = ("NOTE - memory: the last run of this path peaked at %.1f GB of the card's "
            "%.1f GB at %.0f megapixel-frames; this one is %.0f (x%.2f)."
            % (pk, float(total_gb), pm, float(mpf), ratio))
    head += (" A peak at the card's limit means Comfy offloads weights (slower) "
             "or the run OOMs. Levers, cheapest first: %s." % levers)
    return head


class _Phase:
    """A silent phase made audible: prints a heartbeat every period with
    elapsed / left-of-estimate, mirrors it to the node HUD (its stage, the
    phase name, no picture) and pushes the progress bar -- and measures the
    phase's peak VRAM (peak_gb after exit). A context manager; the thread is a
    daemon and stops on exit. Nothing here can break the run: every tick is
    wrapped. announce=False keeps the ENTRY tick off the console (the tile
    path prints its own line per tile and step; the HUD still hears it)."""

    def __init__(self, name, est_s, clock, node_id=None, canvas=(0, 0),
                 step=None, steps=None, period=_HEARTBEAT_S, stage="joint",
                 label="joint", tile=1, tiles=1, rect=None, announce=True,
                 prefix="Power Upscale"):
        self.name, self.est, self.clock = name, est_s, clock
        self.node_id, self.canvas = node_id, canvas
        self.step, self.steps = step, steps
        self.period = period
        self.stage, self.label = stage, label
        self.tile, self.tiles = tile, tiles
        self.rect = rect
        self.announce = announce
        self.prefix = prefix
        self.peak_gb = None
        self._stop = threading.Event()
        self._t0 = None
        self._thread = None

    def _line(self, el):
        left = (max(0.0, self.est - el)) if self.est is not None else None
        tail = (f" (~{_fmt_clock(left)} left of ~{_fmt_clock(self.est)})"
                if left is not None else " (no estimate yet -- measuring)")
        stp = f" step {self.step}/{self.steps}" if self.step is not None else ""
        return f"[PLS] {self.prefix}:   {self.label} {self.name}{stp} ... {_fmt_clock(el)} elapsed{tail}"

    def _tick(self, console=True):
        el = time.monotonic() - self._t0
        if console:
            try:
                print(self._line(el))
            except Exception:
                pass
        try:
            if self.clock is not None:
                self.clock.push()
        except Exception:
            pass
        try:
            if self.node_id is not None:
                from server import PromptServer
                left = (max(0.0, self.est - el)) if self.est is not None else None
                rect = self.rect or (0, 0, int(self.canvas[0]), int(self.canvas[1]))
                PromptServer.instance.send_sync(_HUD_EVENT, {
                    "node": str(self.node_id), "stage": self.stage, "phase": self.name,
                    "elapsed": int(self.clock.elapsed()) if self.clock is not None else int(el),
                    "eta": (None if self.clock is None else
                            (None if self.clock.eta() is None else int(self.clock.eta()))),
                    "phase_elapsed": int(el), "phase_left": (None if left is None else int(left)),
                    "tile": int(self.tile), "tiles": int(self.tiles),
                    "step": int(self.step or 1), "steps": int(self.steps or 1),
                    "rect": [int(v) for v in rect],
                    "canvas": [int(self.canvas[0]), int(self.canvas[1])],
                })
        except Exception:
            pass

    def _run(self):
        while not self._stop.wait(self.period):
            self._tick()

    def __enter__(self):
        self._t0 = time.monotonic()
        _vram_peak_reset()
        self._tick(console=self.announce)   # the phase announces itself at once
        self._thread = threading.Thread(target=self._run, name="pls-pu-heartbeat", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.peak_gb = _vram_peak_gb()
        return False


def _peak_line(label, peaks, total_gb):
    """'VRAM peak: encode 3.1 GB · sample 11.8 GB · decode 4.0 GB of 16.0 GB'
    -- or None when nothing was measured (CPU run). Pure."""
    parts = [f"{k} {v:.1f} GB" for k, v in peaks if v is not None]
    if not parts:
        return None
    top = max(v for _k, v in peaks if v is not None)
    tail = f" of {total_gb:.1f} GB" if total_gb else ""
    warn = (" -- at the card's limit: Comfy is offloading (slower) or close to an OOM"
            if total_gb and top >= 0.92 * total_gb else "")
    return f"[PLS] Power Upscale: {label} VRAM peak: " + " · ".join(parts) + tail + warn


# ── v1010: the general grips ───────────────────────────────────────────────────────
_BAR_SCALE = 1000        # the bar counts per-mille: time-weighted, never raw units
_SAY_EVERY_S = 3.0       # console cadence of a counting run
_BLOCK_CAP = 0.95        # a blocking bar never claims the finish it has not seen


def _bar(total=_BAR_SCALE):
    """A Core ProgressBar (the green node bar), or None outside ComfyUI."""
    try:
        import comfy.utils
        return comfy.utils.ProgressBar(total)
    except Exception:
        return None


def _bar_set(bar, frac):
    if bar is None:
        return
    try:
        bar.update_absolute(int(max(0.0, min(1.0, float(frac))) * _BAR_SCALE), _BAR_SCALE)
    except Exception:
        pass


def model_key(obj):
    """A short class name for a rates section: the wrapped model's class for
    a patcher (MODEL, VAE, UPSCALE_MODEL), the object's own class otherwise,
    or the string itself."""
    if obj is None:
        return "none"
    if isinstance(obj, str):
        return obj
    for attr in ("model", "first_stage_model"):
        inner = getattr(obj, attr, None)
        if inner is not None and not isinstance(inner, (str, int, float)):
            return type(inner).__name__
    return type(obj).__name__


def _eta_text(left):
    return ("~" + _fmt_clock(left)) if left is not None else "?"


class NodeProgress:
    """Counting work: tick(n) per finished unit. Usage::

        with NodeProgress("Save", "save:h264", total=n, unit="frame", size=w*h/1e6) as p:
            for frame in frames:
                ...
                p.tick()

    `size` is the cost of ONE unit in the rate's own measure (e.g. megapixels
    per frame); the learned rate is seconds per (unit x size), so a 1080p
    clip learned from a 720p one scales honestly. `section` defaults to the
    kind. A run that ends early (an exception, an interrupt) learns nothing."""

    def __init__(self, label, section, total, unit="frame", size=1.0,
                 say_every=_SAY_EVERY_S, quiet=False, bar=True):
        self.label, self.section = str(label), str(section)
        self.total = max(0, int(total))
        self.unit = unit
        self.size = max(1e-9, float(size))
        self.say_every = float(say_every)
        self.quiet = bool(quiet)
        self.use_bar = bool(bar)   # False where a Core call inside drives the bar itself
        self.done = 0
        self.bar = None
        self._t0 = None
        self._said = 0.0
        self._ok = False
        self.est_total = None

    def _plural(self, n):
        return self.unit if n == 1 else self.unit + "s"

    def __enter__(self):
        self._t0 = time.monotonic()
        self._said = self._t0
        self.bar = _bar() if self.use_bar else None
        _bar_set(self.bar, 0.0)
        try:
            r = _rates_load(self.section)
            if r.get("step"):
                self.est_total = r["step"] * self.size * self.total
        except Exception:
            self.est_total = None
        if not self.quiet and self.total > 0:
            est = (f"~{_fmt_clock(self.est_total)} (learned)" if self.est_total is not None
                   else "no rate learned yet -- this run measures it")
            print(f"[PLS] {self.label}: {self.total} {self._plural(self.total)} -- estimate {est}")
        return self

    def rate_left(self):
        """(units per second, seconds left) from THIS run, or the learned
        estimate before the first unit is done."""
        el = time.monotonic() - self._t0
        if self.done > 0 and el > 0:
            per = el / self.done
            return 1.0 / per, per * max(0, self.total - self.done)
        if self.est_total is not None:
            return None, max(0.0, self.est_total - el)
        return None, None

    def tick(self, n=1):
        self.done = min(self.total, self.done + int(n)) if self.total else self.done + int(n)
        if self.total:
            _bar_set(self.bar, self.done / float(self.total))
        now = time.monotonic()
        if self.quiet or self.total == 0:
            return
        if (now - self._said) >= self.say_every and self.done < self.total:
            self._said = now
            ups, left = self.rate_left()
            rate = (f"{ups:.2f} {self.unit}/s" if ups is not None and ups >= 1.0 else
                    (f"{1.0 / ups:.1f} s/{self.unit}" if ups else "?"))
            print(f"[PLS] {self.label}:   {self.done}/{self.total} {self._plural(self.total)} "
                  f"({100.0 * self.done / self.total:.0f} %) {_fmt_clock(now - self._t0)} elapsed, "
                  f"{rate}, {_eta_text(left)} left")

    def finish(self):
        """Mark the run complete (the with-block does it on a clean exit)."""
        self._ok = True

    def __exit__(self, exc_type, *exc):
        el = time.monotonic() - self._t0
        if exc_type is None:
            self._ok = True
        if self._ok and self.done > 0:
            _bar_set(self.bar, 1.0)
            try:
                r = _rates_learn(self.section, self.size * self.done, None, el, None)
                runs = r.get("runs", 0)
            except Exception:
                runs = 0
            if not self.quiet:
                per = el / max(1, self.done)
                print(f"[PLS] {self.label}: {self.done} {self._plural(self.done)} in "
                      f"{_fmt_clock(el)} ({per:.2f} s/{self.unit}"
                      + (f", planned ~{_fmt_clock(self.est_total)}" if self.est_total is not None else "")
                      + f") -- rate learned ({runs} run(s))")
        return False


class blocking:
    """ONE call that cannot report. Usage::

        with blocking("VAE", "vae:decode:WanVAE", size=mpf, what="decode") as b:
            images = vae.decode(latent)

    A heartbeat thread prints every `period` seconds (elapsed, and what is
    left of the learned estimate); the bar follows elapsed / estimate up to
    95 %. On exit the call's seconds per `size` unit are learned, the peak
    VRAM is measured (b.peak_gb) and one done line is printed."""

    def __init__(self, label, section, size=1.0, what="work", period=_HEARTBEAT_S,
                 quiet=False, unit="unit", bar=True):
        self.label, self.section = str(label), str(section)
        self.size = max(1e-9, float(size))
        self.what = what
        self.period = float(period)
        self.quiet = bool(quiet)
        self.unit = unit
        self.use_bar = bool(bar)   # False where Core drives the bar itself (a 2-D tiled VAE pass)
        self.est = None
        self.peak_gb = None
        self.seconds = None
        self.bar = None
        self._stop = threading.Event()
        self._t0 = None
        self._thread = None

    def _tick(self):
        el = time.monotonic() - self._t0
        if self.est:
            _bar_set(self.bar, min(_BLOCK_CAP, el / self.est))
        if self.quiet:
            return
        try:
            tail = (f" (~{_fmt_clock(max(0.0, self.est - el))} left of ~{_fmt_clock(self.est)})"
                    if self.est else " (no estimate yet -- measuring)")
            print(f"[PLS] {self.label}:   {self.what} ... {_fmt_clock(el)} elapsed{tail}")
        except Exception:
            pass

    def _run(self):
        while not self._stop.wait(self.period):
            self._tick()

    def __enter__(self):
        self._t0 = time.monotonic()
        self.bar = _bar() if self.use_bar else None
        _bar_set(self.bar, 0.0)
        try:
            r = _rates_load(self.section)
            if r.get("step"):
                self.est = r["step"] * self.size
        except Exception:
            self.est = None
        if not self.quiet:
            print(f"[PLS] {self.label}: {self.what} begin -- "
                  + (f"estimate ~{_fmt_clock(self.est)} (learned)" if self.est
                     else "no rate learned yet -- this run measures it; a heartbeat "
                          f"ticks every {int(self.period)} s"))
        _vram_peak_reset()
        self._thread = threading.Thread(target=self._run, name="pls-blocking-heartbeat", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.seconds = time.monotonic() - self._t0
        self.peak_gb = _vram_peak_gb()
        if exc_type is None:
            _bar_set(self.bar, 1.0)
            try:
                r = _rates_learn(self.section, self.size, None, self.seconds, None,
                                 peak_gb=self.peak_gb)
                runs = r.get("runs", 0)
            except Exception:
                runs = 0
            if not self.quiet:
                tot = _vram_total_gb()
                pk = (f", VRAM peak {self.peak_gb:.1f} GB" + (f" of {tot:.1f} GB" if tot else "")
                      if self.peak_gb else "")
                print(f"[PLS] {self.label}: {self.what} done in {_fmt_clock(self.seconds)}"
                      + (f" (planned ~{_fmt_clock(self.est)})" if self.est else "")
                      + f"{pk} -- rate learned ({runs} run(s))")
        return False
