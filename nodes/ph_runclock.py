"""
Polyhedron run clock  (ph_runclock)
===================================
v576: the clock moved out of ph_power_upscale.py into this shared module so
the Polyhedron Sampler can tell time too. The code is the v567 clock,
byte-for-byte -- only the address changed. ph_power_upscale re-exports both
names (house pattern: uls_stack_node -> uls_merge_math), so every existing
caller and the probe/HUD wiring keep working unchanged.

Deliberately dependency-free (import time, nothing else): the guard suite
execs THIS WHOLE FILE as the window -- structure pinned instead of the two
regex windows that used to carve the class out of ph_power_upscale (and
whose anchors were the fragile part, ledger lesson #3).
"""
import time


def _fmt_clock(seconds):
    """m:ss / h:mm:ss - the ONE time format for console, bar and HUD."""
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class _RunClock:
    """v567: ONE clock for the console, the bar and the HUD.

    THE MEASURED WOUND: Frank's 11:38 run had 26 bar units - 18 pixel chunks
    at ~62 ms and 8 sampler steps at 43-93 s. Half a second in, the bar stood
    at 9/26 = "35%" while the truth was 0.07%; it then crawled through the
    remaining 65% for eleven minutes. Ticks are not time.

    The clock keeps POSTS (key = "kind:stage", e.g. "step:low") with units,
    a WEIGHT (tile pixel area - the honest cross-stage proportionality; no
    card-specific constants anywhere), and a RATE learned live as an EMA of
    measured seconds. Estimation ladder, most-informed first:
      1. a post that measured itself   -> its own EMA
      2. same KIND measured elsewhere  -> that rate, weight-scaled (linear;
         attention makes big tiles super-linear, so the EMA corrects the
         moment the class measures itself - the estimate carries '~')
      3. nothing of that kind measured -> total spent seconds / total done
         weight, times the open weight (a whole-run extrapolation)
      4. nothing measured at all       -> eta None; the bar shows ~1% and
         says so, instead of inventing a number.
    value on the bar IS the wall clock (deciseconds, monotonic); total is
    elapsed + eta and may breathe as the estimate improves - like every
    honest download bar. `now` is injectable so test_v567 runs the whole
    ladder deterministically."""

    def __init__(self, pbar=None, now=time.monotonic):
        self.now = now
        self.t0 = now()
        self.pbar = pbar
        self.posts = {}
        self._cursor = self.t0
        self._preview = None   # v885: ONE-SHOT preview for the node's own slot

    def offer_preview(self, image):
        """v885 -- hand the node's progress bar a picture.

        MEASURED at Core (comfy/utils.py ProgressBar.update_absolute): a
        non-None preview BYPASSES Core's own throttle
        (PROGRESS_THROTTLE_MIN_INTERVAL / _MIN_PERCENT) and is sent at once.
        Attaching one to every push would therefore flood the socket, which is
        exactly the failure the throttle exists to prevent.

        So this is a ONE-SHOT slot: whoever produces a frame OFFERS it, the
        next push CONSUMES it and clears it. The rate is then the producer's
        own (the probes are throttled by _PROBE_MIN_INTERVAL), never the
        clock's tick rate. `image` is Core's PreviewImageTuple --
        (format, PIL.Image, max_edge) -- the same shape latent_preview hands
        the stock KSampler; anything else is the caller's error, not ours.
        """
        self._preview = image

    def tick(self):
        """v577: seconds since the last tick - ONE cursor per RUN.

        THE MEASURED WOUND (v576, caught by audit before it ever ran): the
        sampler kept its time cursor inside each CALLBACK's closure, seeded at
        the callback's BUILD time. The MoE chain builds cb_high AND cb_low
        BEFORE phase 1 - so cb_low's cursor sat at t0, and the first LOW step
        measured everything since the start of the run. Simulated on Frank's
        rates (HIGH 43.1 s, LOW 93.4 s): LOW step 1 reported 222.7 s, the EMA
        swallowed it, and the run eta DOUBLED at the handoff. The sum of the
        posts (725.6 s) exceeded the wall clock (596.3 s) - physically
        impossible, and the cleanest self-betrayal a clock can offer.

        The cursor belongs to the RUN, so it lives on the clock the stages
        SHARE. Then LOW step 1's dt = t(low_1) - t(high_last) = its own
        duration plus the lazy model swap - exactly, and finally truthfully,
        what the docstring promises. Invariant, guard-executed: sum(spent) can
        never exceed elapsed().
        """
        t = self.now()
        dt = t - self._cursor
        self._cursor = t
        return dt

    def post(self, key, units, weight):
        self.posts[key] = {"kind": key.split(":", 1)[0], "units": int(units),
                           "done": 0, "weight": float(max(1e-9, weight)),
                           "rate": None, "spent": 0.0}

    def resize(self, key, units):
        """The pixel pass may clamp its own chunk count for VRAM (v565); the
        true count is only known once it runs. Correct the plan, keep truth."""
        if key in self.posts:
            p = self.posts[key]
            p["units"] = max(int(units), p["done"])

    def measure(self, key, seconds):
        """Book `seconds` against a post. Returns True if it counted as a UNIT.

        v580 -- THE TAIL CALLBACK (measured, not theorised): some samplers fire
        the step callback ONE MORE TIME after the last step. RES4LYF's `res_2s`
        does; ComfyUI's `dpmpp_2m` does not (Frank's PU posts 3/3, his sampler
        posts 3/3 AND a fourth). The tail carries real seconds -- the final
        preview decode -- and it grows with the frame count:

            13 frames -> 5.7 s / 5.4 s      65 frames -> 37.8 s / 34.4 s

        v576 clamped `done` at `units`, so the tail was never miscounted as a
        step. But it still fed `spent` AND the rate. The EMA ate it, the rate
        collapsed, and the run-ETA at the MoE handoff went 108 s optimistic on
        a 65-frame run (said 4:22, LOW took 6:10). The console said `step 3/3`
        twice, with two different numbers.

        The split is: the SECONDS are real -> they stay in `spent`, so the
        invariant sum(spent) == elapsed still holds and the clock cannot lie
        about the wall. The STEP is not real -> it must not move `done` and must
        not touch `rate`. The caller gets False and stays quiet.
        """
        p = self.posts[key]
        seconds = float(seconds)
        p["spent"] += seconds                 # the wall clock is the wall clock
        if p["done"] >= p["units"]:
            self.push()
            return False                      # a tail, not a step
        p["done"] += 1
        p["rate"] = (seconds if p["rate"] is None
                     else 0.5 * p["rate"] + 0.5 * seconds)
        self.push()
        return True

    def _rate(self, key):
        p = self.posts[key]
        if p["rate"] is not None:
            return p["rate"]
        for q in self.posts.values():             # rung 2: same kind, scaled
            if q["kind"] == p["kind"] and q["rate"] is not None:
                return q["rate"] * (p["weight"] / q["weight"])
        return None

    def elapsed(self):
        return self.now() - self.t0

    def eta(self, tag=None):
        """Remaining seconds - for one stage (tag) or the whole run (None)."""
        rem, unresolved = 0.0, 0.0
        for key, p in self.posts.items():
            if tag is not None and not key.endswith(":" + tag):
                continue
            left = p["units"] - p["done"]
            if left <= 0:
                continue
            r = self._rate(key)
            if r is None:
                unresolved += left * p["weight"]
            else:
                rem += left * r
        if unresolved > 0.0:
            wd = sum(q["done"] * q["weight"] for q in self.posts.values())
            sd = sum(q["spent"] for q in self.posts.values())
            if wd > 0.0:
                rem += unresolved * (sd / wd)     # rung 3: extrapolate
            else:
                return None                       # rung 4: say so
        return rem

    def push(self):
        """The bar IS a time bar: value = elapsed deciseconds (monotonic),
        total = elapsed + eta. Before anything is measured the total is held
        far out (~1%) - honest ignorance beats a confident lie."""
        if self.pbar is None:
            return
        el = self.elapsed()
        eta = self.eta()
        total = el + (eta if eta is not None else max(el, 1.0) * 99.0)
        v = int(el * 10)
        # v885: consume the one-shot preview (see offer_preview). None when no
        # producer offered one since the last push -- which is the normal case
        # and keeps Core's throttle in charge of plain progress updates.
        img, self._preview = self._preview, None
        self.pbar.update_absolute(v, max(int(total * 10), v + 1), img)
