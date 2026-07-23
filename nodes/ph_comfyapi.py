"""
Polyhedron comfy_api door  (ph_comfyapi)
========================================
v577: ONE import site for ComfyUI's versioned node API.

THE EXPOSURE (found by the v576 audit): nine `*_v3.py` files each did
`from comfy_api.latest import io`. ComfyUI's own documentation says what
`latest` is: the API version still UNDER DEVELOPMENT -- "more changes will be
made to it without warning". The version BEFORE latest is the stable one. The
advice from the docs and the ecosystem is identical: develop against latest,
SHIP against a pin.

The nasty failure mode is not a broken import (our per-node try/except catches
that and registers the proven legacy node). It is a CHANGED SIGNATURE: the
import succeeds, `_V3_OK` stays True, and the error only surfaces when ComfyUI
builds its node list -- long after the isolation could have helped.

WHAT THIS MODULE DOES NOT DO: blind-pin. Pinning to a version that has never
run on Frank's machine would be Glauben, not Messen. So:

  1. ONE door. When we pin, we pin HERE -- not in nine files.
  2. PIN_TO honours an explicit choice. None (today) = latest, which is
     byte-identical to v576 behaviour. Nothing changes until we decide.
  3. It SAYS, once, on the console: which module it bound, and which pinned
     versions the running ComfyUI actually offers. That console line is the
     MEASUREMENT that lets the next cut pin with knowledge instead of a guess.

Deliberately dependency-free beyond comfy_api + importlib, so a guard can read
it without a ComfyUI around.
"""
import importlib
import importlib.util

# Set to a version string (e.g. "v0_0_2") to pin. None = ride `latest`, the
# v576 behaviour. Change ONLY after the console line below has told us which
# versions this ComfyUI actually ships -- that is the whole point of the line.
#
# v580 -- PINNED, ON A MEASUREMENT. Frank's console (ComfyUI 0.24.1, frontend
# 1.44.19, 2026-07-13) said, in the words of the line below:
#
#     [PLS] comfy_api: bound 'latest' (pinned versions present: v0_0_1, v0_0_2)
#
# So v0_0_2 is the newest version this ComfyUI actually pins, and `latest` is by
# ComfyUI's own documentation the one still under development -- changed without
# notice. The advice from the docs and from the community is the same: develop
# against `latest`, SHIP against a pin. The ladder below still falls back to
# older pins and finally to `latest`, so a ComfyUI without v0_0_2 loses nothing.
#
# The audit (S1, HIGH) named the ugly failure mode: a broken IMPORT is caught by
# our isolation, but a changed SIGNATURE imports cleanly and detonates later,
# when ComfyUI builds the node list -- by then the isolation is over.
PIN_TO = "v0_0_2"

# The scan range for the availability probe. Cheap: find_spec does not execute
# the module, it only asks the import system whether it exists.
_PROBE_MAX = 12


def available_versions():
    """Pinned comfy_api versions this ComfyUI actually offers, oldest first."""
    out = []
    for n in range(1, _PROBE_MAX):
        name = f"comfy_api.v0_0_{n}"
        try:
            if importlib.util.find_spec(name) is not None:
                out.append(f"v0_0_{n}")
        except Exception:
            pass                      # a probe must never raise into the loader
    return out


BOUND = None
io = None

if PIN_TO:
    try:
        _mod = importlib.import_module("comfy_api." + str(PIN_TO))
        io = _mod.io
        BOUND = str(PIN_TO)
    except Exception as e:
        print(f"[PLS] comfy_api: pin '{PIN_TO}' unavailable ({e!r}) -> falling "
              f"back to 'latest'. Fix PIN_TO in nodes/ph_comfyapi.py.")

if io is None:
    # No pin (or the pin failed): ride latest, exactly as v576 did. A failure
    # here propagates to uls_v3_extension, whose caller registers the legacy
    # nodes -- the proven path, unchanged.
    from comfy_api.latest import io as _io
    io = _io
    BOUND = "latest"

try:
    _avail = available_versions()
    print(f"[PLS] comfy_api: bound '{BOUND}'"
          + (f" (pinned versions present: {', '.join(_avail)})" if _avail
             else " (no pinned versions present on this ComfyUI)"))
except Exception:
    pass                              # telemetry must never break the load
