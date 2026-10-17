"""Polyhedron Flux2 helpers (v1062) -- the empty latent and the sigma schedule.

Written here instead of calling the core's EmptyFlux2LatentImage and
Flux2Scheduler node classes (Frank: the suite calls no Core node classes; Core
is GPL, the pack MIT). Both are small published rules:

  empty latent  zeros [batch, 128, height // 16, width // 16] on the model
                library's intermediate device
  schedule      steps+1 times from 1 to 0, shifted by
                exp(mu) / (exp(mu) + (1/t - 1)), mu read off a straight-line
                fit in the image's token count (width*height/256) and the
                step count -- two lines, one for 10 and one for 200 steps,
                interpolated by steps; above 4300 tokens only the 200 line.

test_v1062 compares both with the core nodes on both cores (bit for bit).

Pure ASCII.
"""
import math

import torch

LATENT_CH = 128
LATENT_DIV = 16

# the two lines mu(tokens) = a * tokens + b, measured for 10 and for 200 steps
_LINE_10 = (8.73809524e-05, 1.89833333)
_LINE_200 = (0.00016927, 0.45666666)
_LONG_SEQ = 4300


def empty_latent(width, height, batch_size=1):
    """-> (latent dict,) -- the node-call shape the callers use."""
    try:
        import comfy.model_management as mm
        device = mm.intermediate_device()
    except Exception:
        device = "cpu"
    z = torch.zeros([int(batch_size), LATENT_CH, int(height) // LATENT_DIV, int(width) // LATENT_DIV],
                    device=device)
    return ({"samples": z},)


def shift_mu(tokens, steps):
    """The schedule's shift for an image of `tokens` latent tokens at `steps`."""
    a200, b200 = _LINE_200
    if tokens > _LONG_SEQ:
        return float(a200 * tokens + b200)
    a10, b10 = _LINE_10
    at200 = a200 * tokens + b200
    at10 = a10 * tokens + b10
    slope = (at200 - at10) / 190.0
    return float(slope * steps + (at200 - 200.0 * slope))


def scheduler_sigmas(steps, width, height):
    """-> (sigmas,) -- steps+1 values from 1.0 down to 0.0."""
    tokens = round(int(width) * int(height) / float(LATENT_DIV * LATENT_DIV))
    mu = shift_mu(tokens, int(steps))
    t = torch.linspace(1, 0, int(steps) + 1)
    e = math.exp(mu)
    return (e / (e + (1 / t - 1) ** 1.0),)
