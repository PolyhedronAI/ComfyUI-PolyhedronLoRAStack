# -*- coding: ascii -*-
"""Polyhedron Rough Cut Gate (Rough Cut V2 + V3, v1027) -- look before the
render finishes, and stop it if the coarse things are wrong.

The 1-step rough cut Frank asked for (27.09.) IS the sampler's x0 prediction
after step 1; the steps that follow only refine it. So no second sampler:
the gate makes the sampler PAUSE after step k, decode the whole clip through
the preview decoder (taeh3: every latent frame, not the 8 the running
preview shows), show it in THIS node next to the previous clip's tail, and
wait for a decision -- continue, or abort the run. No decision within the
timeout = continue, so a queue never hangs at night.

How the gate reaches the sampler without a widget on it: the gate node sits
BETWEEN Empty Latent and the sampler on the LATENT wire and tags the latent
dict (`pls_gate` = {after_step, timeout, node}); Core passes the dict
through untouched, the Polyhedron Sampler reads the tag when its run starts
(uls_sampler._sample_impl) and its preview callback fires the gate. The
sampler's own 20 serialised widgets stay exactly as they are (#577).

    Empty Latent --latent--> Rough Cut Gate --latent--> Polyhedron Sampler
    Clip Chain Source --tail--> Rough Cut Gate   (optional: the handover)

The decision arrives through POST /pls/cine/gate {node_id, action}. The
state lives in this module (per gate node id); the sampler side imports
`fire()` lazily and never fails the run because of the gate: any error here
is printed once and the run continues as if there were no gate.
"""

import threading
import time

_LOCK = threading.Lock()
_DECISION = {}          # gate node id -> "continue" | "abort"
_ACTIVE = {}            # gate node id -> {"step", "total", "since", "timeout"}
GATE_EVENT = "polyhedron.rough_cut_gate"
DONE_EVENT = "polyhedron.rough_cut_gate_done"
POLL_S = 0.25
TAIL_SHOW = 8
GATE_MAX_FRAMES = 256   # every latent frame of a clip (H3: 31 for 124 px frames)


# ------------------------------------------------------------- state (pure)
def tag(latent, after_step, timeout, node_id):
    """A copy of the latent dict carrying the gate. after_step <= 0 = no gate."""
    out = dict(latent)
    if int(after_step) <= 0:
        out.pop("pls_gate", None)
        return out
    out["pls_gate"] = {"after_step": int(after_step), "timeout": float(timeout), "node": str(node_id)}
    return out


def gate_of(latent):
    """The gate a latent carries, or None."""
    g = latent.get("pls_gate") if isinstance(latent, dict) else None
    if not isinstance(g, dict) or int(g.get("after_step", 0)) <= 0:
        return None
    return g


def decide(node_id, action):
    """The user's answer. -> True when a gate was waiting for it."""
    if action not in ("continue", "abort"):
        return False
    with _LOCK:
        if str(node_id) not in _ACTIVE:
            return False
        _DECISION[str(node_id)] = action
    return True


def active():
    with _LOCK:
        return {k: dict(v) for k, v in _ACTIVE.items()}


def wait(node_id, timeout, poll=POLL_S, clock=time.time, sleep=time.sleep, interrupted=None):
    """Block until a decision, the timeout, or Core's own interrupt.
    -> "continue" | "abort" | "timeout". The node is marked active for the
    whole wait, so decide() knows whom it answers."""
    key = str(node_id)
    t0 = clock()
    with _LOCK:
        _DECISION.pop(key, None)
        _ACTIVE.setdefault(key, {})["since"] = t0
    try:
        while True:
            with _LOCK:
                d = _DECISION.pop(key, None)
            if d is not None:
                return d
            if interrupted is not None and interrupted():
                return "abort"
            if clock() - t0 >= float(timeout):
                return "timeout"
            sleep(poll)
    finally:
        with _LOCK:
            _ACTIVE.pop(key, None)


# ------------------------------------------------------------- the sampler side
def _send(event, payload):
    try:
        from server import PromptServer
        PromptServer.instance.send_sync(event, payload)
    except Exception:
        pass


def fire(gate, previewer, x0, step_done, total_steps, fps=6):
    """Called by the sampler's preview callback after step `step_done`
    (1-based). Decodes the whole clip, shows it, waits, and either returns
    (continue) or raises Core's InterruptProcessingException (abort).
    Anything else that goes wrong is printed once and the run continues."""
    node = str(gate.get("node"))
    timeout = float(gate.get("timeout", 60))
    with _LOCK:
        _ACTIVE[node] = {"step": int(step_done), "total": int(total_steps), "since": time.time(),
                         "timeout": timeout}
    try:
        keep = getattr(previewer, "max_frames", None)
        frames = []
        try:
            if keep is not None:
                previewer.max_frames = GATE_MAX_FRAMES
            frames = previewer.frames_b64(x0) if previewer is not None else []
        finally:
            if keep is not None:
                previewer.max_frames = keep
        _send(GATE_EVENT, {"node": node, "frames": frames, "fps": fps, "step": int(step_done),
                           "total": int(total_steps), "timeout": timeout})
        print("[PLS GATE] rough cut after step %d/%d: %d frame(s) shown, waiting up to %.0f s "
              "for continue / abort" % (step_done, total_steps, len(frames), timeout))
    except Exception as e:
        with _LOCK:
            _ACTIVE.pop(node, None)
        print("[PLS GATE] could not show the rough cut (%s: %s) -- the run continues" % (type(e).__name__, e))
        return "continue"
    try:
        import comfy.model_management as mm
        d = wait(node, timeout, interrupted=mm.processing_interrupted)
    except ImportError:
        d = wait(node, timeout)
        mm = None
    _send(DONE_EVENT, {"node": node, "decision": d, "step": int(step_done)})
    if d == "abort":
        print("[PLS GATE] aborted at the gate after step %d -- the remaining %d step(s) are not run"
              % (step_done, max(0, int(total_steps) - int(step_done))))
        if mm is not None:
            mm.interrupt_current_processing()
            mm.throw_exception_if_processing_interrupted()
        return "abort"
    print("[PLS GATE] %s -- the run continues" % ("continue" if d == "continue" else "no answer in %.0f s" % timeout))
    return "continue"


# ------------------------------------------------------------- the node
def _thumbs(frames, n=TAIL_SHOW, height=72):
    """The last n frames as base64 JPEG thumbnails (for the field)."""
    import base64
    import io
    from PIL import Image
    out = []
    arr = frames[-n:]
    for f in arr:
        a = (f.detach().cpu().numpy() * 255.0).round().clip(0, 255).astype("uint8")
        im = Image.fromarray(a)
        w = max(8, int(round(im.width * height / float(im.height))))
        im = im.resize((w, height), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=80)
        out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return out


class ULSRoughCutGate:
    """Pause the sampler after step k, show the whole rough cut, continue or abort."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "From Empty Latent (or the chain's start latent); "
                    "goes on to the Polyhedron Sampler carrying the gate."}),
                "after_step": ("INT", {"default": 1, "min": 0, "max": 100,
                    "tooltip": "Pause after this step and show the rough cut (the x0 prediction "
                               "of the whole clip). 0 = no gate, the latent passes through."}),
                "timeout": ("INT", {"default": 90, "min": 5, "max": 3600, "step": 5,
                    "tooltip": "Seconds to wait for continue / abort; then the run continues "
                               "by itself (a queue never hangs)."}),
            },
            "optional": {
                "tail": ("IMAGE", {"tooltip": "The END of the previous clip (Clip Chain Source "
                    "`tail`): shown left of the rough cut, so the handover is seen before "
                    "the render is done."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "info")
    FUNCTION = "arm"
    OUTPUT_NODE = True
    CATEGORY = "Polyhedron/Cine"
    DESCRIPTION = ("Sits between Empty Latent and the Polyhedron Sampler. The sampler "
                   "pauses after `after_step`, decodes the WHOLE clip through the "
                   "preview decoder (taeh3) and shows it here -- the x0 prediction "
                   "after step 1 is the 1-step rough cut -- next to the previous "
                   "clip's tail. Continue, or abort the run and save the remaining "
                   "steps. No answer within `timeout` = continue. Needs the sampler's "
                   "preview_mode on a video mode (taeh3 for H3).")

    def arm(self, latent, after_step, timeout, tail=None, unique_id=None):
        node = str(unique_id) if unique_id is not None else "?"
        out = tag(latent, after_step, timeout, node)
        payload = {"kind": "armed", "after_step": int(after_step), "timeout": int(timeout)}
        lines = ["Rough Cut Gate: %s" % ("no gate (after_step 0) -- the latent passes through"
                                          if int(after_step) <= 0 else
                                          "the sampler pauses after step %d, waits up to %d s" % (after_step, timeout))]
        if tail is not None:
            try:
                payload["tail"] = _thumbs(tail)
                lines.append("  handover: the last %d of %d tail frame(s) shown left of the rough cut"
                             % (len(payload["tail"]), int(tail.shape[0])))
            except Exception as e:
                lines.append("  tail not shown (%s)" % e)
        info = "\n".join(lines)
        print("[PLS] " + info.replace("\n", "\n[PLS] "))
        return {"ui": {"pls_gate": [payload]}, "result": (out, info)}


# ------------------------------------------------------------- the route
async def handle_gate(request):
    """POST {node_id, action: continue|abort} -> {ok, waiting}."""
    from aiohttp import web
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "bad json"}, status=400)
    node_id = body.get("node_id")
    action = body.get("action")
    if node_id is None or action not in ("continue", "abort"):
        return web.json_response({"ok": False, "error": "node_id and action (continue|abort) required"}, status=400)
    ok = decide(node_id, action)
    return web.json_response({"ok": ok, "node_id": str(node_id), "action": action,
                              "error": None if ok else "no gate is waiting on node %s" % node_id})


async def handle_gate_status(request):
    from aiohttp import web
    return web.json_response({"active": active()})


def register_gate_routes():
    try:
        from server import PromptServer
        app = PromptServer.instance.app
    except Exception as e:
        print("[PLS] Rough Cut Gate route not registered: %s" % e)
        return 0
    n = 0
    for p, fn in (("/pls/cine/gate", handle_gate), ("/api/pls/cine/gate", handle_gate)):
        try:
            app.router.add_route("POST", p, fn)
            n += 1
        except Exception as e:
            print("[PLS] could not add route POST %s: %s" % (p, e))
    for p in ("/pls/cine/gate", "/api/pls/cine/gate"):
        try:
            app.router.add_route("GET", p, handle_gate_status)
            n += 1
        except Exception as e:
            print("[PLS] could not add route GET %s: %s" % (p, e))
    return n


NODE_CLASS_MAPPINGS = {"ULSRoughCutGate": ULSRoughCutGate}
NODE_DISPLAY_NAME_MAPPINGS = {"ULSRoughCutGate": "\u2b21 Polyhedron Rough Cut Gate"}
