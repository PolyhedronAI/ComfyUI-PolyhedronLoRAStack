"""
uls_noise.py -- reproducible latent-noise generation for ⬡ Polyhedron Empty
Latent.

REPRODUCIBILITY IS THE POINT. Every tensor is drawn from a torch.Generator on
the CPU seeded by (noise_seed) -- exactly ComfyUI's own comfy.sample.prepare_noise
approach. CPU generation is deterministic ACROSS machines and driver versions
(GPU RNG is not), so the same seed + type + shape always yields the same tensor,
whether generated now or after the workflow is re-loaded from PNG metadata. The
node then moves the finished tensor to the intermediate device.

WHY TYPES (the concrete-utility question): the sampler forms its start as
x = latent_image + noise * sigma[0]. For flow-matching models (Wan, Flux) sigma[0]
= 1.0, so the latent's own content enters at FULL weight; and for denoise < 1.0
(refine passes) sigma[0] is small, so the latent content DOMINATES. There, the
SPECTRAL character of the injected noise measurably shifts the result: low-freq
(brown) biases composition, high-freq (blue) biases fine detail. For SD1.x/SDXL
at denoise 1.0 (sigma[0] ~ 14) the effect is a small perturbation -- documented,
not hidden.

Types (each spectrally distinct -- proven in the guard by radial power spectrum):
  zeros     all-zero  (sampler owns all noise; the correct "empty" for denoise 1)
  gaussian  flat white (torch.randn parity -- what other empty-latent nodes offer)
  pink      1/f      power (beta 1)  -- natural balance
  brown     1/f^2    power (beta 2)  -- low-freq heavy -> composition bias
  blue      f^1      power (beta -1) -- high-freq heavy -> detail bias
  fractal   summed multi-octave value noise -- coherent "cloud" structure, the
            strongest compositional bias and the one the live preview shows best
  offset    white plus a per-channel DC term (the community "offset noise"
            trick) -- near-white spectrum, widens the value range the model
            reaches (darker darks, brighter lights)
  pyramid   full-res white plus decaying coarser octaves (multi-res noise,
            discount 0.7) -- the tame cousin of fractal: a mild low-frequency
            lift instead of cloud structure

CHARACTER (v833): every shaped type takes a `character` dial in 0..1.
1.0 (the default) is the pure type -- and for the pre-v833 types it is
STREAM-BIT-IDENTICAL to before, because the shaped tensor is drawn FIRST
and the blending white only afterwards (and not at all at 1.0). Below 1.0
the field is cross-faded with white drawn from the same generator:
normalize(sqrt(1-c^2)*white + c*shaped). WHY: the model is trained on
white gaussian; the spectrally distant types read as image content at
full strength ("interesting but off"). The dial turns a binary excursion
into a dosed bias. offset scales its DC term by the dial instead
(character 0 == gaussian, bit-identical stream). gaussian/zeros ignore it.

The spectral recipe (beta exponents, radial 1/f weighting) is mirrored bit-for-
bit by a numpy twin in the guard, which proves determinism AND the power-spectrum
ordering; this torch module is the faithful runtime transcription.
"""

# Canonical type list (single source of truth; the node's combo + the JS preview
# both derive from this order).
# v833 (Frank's Go, decisions delegated): offset and pyramid APPENDED --
# combo options serialise by string, so growth at the end is free (#747).
NOISE_TYPES = ["zeros", "gaussian", "pink", "brown", "blue", "fractal",
               "offset", "pyramid"]

# Power-spectrum exponent per colored type: power(f) ~ f^(-BETA).  pink=1,
# brown=2 (low-freq heavy), blue=-1 (high-freq heavy).  Amplitude weight is
# f^(-BETA/2).  These constants are asserted against the numpy twin in the guard.
COLORED_BETA = {"pink": 1.0, "brown": 2.0, "blue": -1.0}

# Types whose seed/strength do nothing (used by the JS to grey those widgets).
SEEDLESS_TYPES = ["zeros"]

_FRACTAL_OCTAVES = 5
_FRACTAL_PERSISTENCE = 0.5

# v833: pyramid = full-res white + this many coarser octaves at this decay
# (the Kohya multi-res recipe); offset's DC amplitude at character 1.0.
_PYRAMID_OCTAVES = 4
_PYRAMID_DISCOUNT = 0.7
_OFFSET_MAX = 0.15


def _generator(seed):
    import torch
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
    return g


def _normalize(t):
    """Zero-mean, unit-std (per whole tensor). Guards a zero-std edge case."""
    m = t.mean()
    s = t.std()
    if float(s) < 1e-8:
        return t - m
    return (t - m) / s


def _colored(shape, beta, g):
    """Isotropic colored noise via radial 1/f weighting of a white field.
    Batched FFT over the last two (spatial) dims -- every channel/frame is
    filtered independently in one shot, no python loop."""
    import torch
    white = torch.randn(shape, generator=g)
    H, W = shape[-2], shape[-1]
    # Frequency grids matching rfft2 output (H x (W//2+1)).
    fy = torch.fft.fftfreq(H).reshape(H, 1)
    fx = torch.fft.rfftfreq(W).reshape(1, W // 2 + 1)
    radial = torch.sqrt(fy * fy + fx * fx)
    weight = torch.zeros_like(radial)
    nz = radial > 0
    weight[nz] = radial[nz] ** (-beta / 2.0)  # amplitude weight -> power f^-beta
    weight[~nz] = 0.0                          # kill DC (mean handled by _normalize)
    spec = torch.fft.rfft2(white, dim=(-2, -1))
    spec = spec * weight  # broadcasts over leading (batch/channel/frame) dims
    out = torch.fft.irfft2(spec, s=(H, W), dim=(-2, -1))
    return _normalize(out)


def _fractal(shape, g):
    """Coherent multi-octave value noise (fBm). Each octave is a low-res white
    grid bilinearly upsampled to full size; octaves sum with geometric decay.
    Fully batched via a (N, 1, gh, gw) interpolate per octave."""
    import torch
    import torch.nn.functional as F
    H, W = shape[-2], shape[-1]
    lead = 1
    for d in shape[:-2]:
        lead *= int(d)
    acc = torch.zeros((lead, 1, H, W))
    amp = 1.0
    total = 0.0
    for o in range(_FRACTAL_OCTAVES):
        # coarse (o=0) -> fine: grid resolution doubles each octave.
        div = 1 << (_FRACTAL_OCTAVES - 1 - o)
        gh = max(2, H // div)
        gw = max(2, W // div)
        grid = torch.randn((lead, 1, gh, gw), generator=g)
        up = F.interpolate(grid, size=(H, W), mode="bilinear", align_corners=False)
        acc = acc + amp * up
        total += amp
        amp *= _FRACTAL_PERSISTENCE
    acc = acc / total
    return _normalize(acc.reshape(shape))


def _pyramid(shape, g):
    """v833: multi-resolution gaussian sum (the multi-res / pyramid-noise
    recipe). UNLIKE _fractal this KEEPS a full-resolution white base and only
    ADDS decaying coarser octaves -- so the spectrum stays near-white with a
    mild low-frequency lift instead of coherent cloud structure. That is the
    difference between "works" and "interesting but off" in the field."""
    import torch
    import torch.nn.functional as F
    H, W = shape[-2], shape[-1]
    lead = 1
    for d in shape[:-2]:
        lead *= int(d)
    acc = torch.randn((lead, 1, H, W), generator=g)
    amp = 1.0
    total = 1.0
    for o in range(1, _PYRAMID_OCTAVES + 1):
        gh = max(2, H >> o)
        gw = max(2, W >> o)
        grid = torch.randn((lead, 1, gh, gw), generator=g)
        up = F.interpolate(grid, size=(H, W), mode="bilinear",
                           align_corners=False)
        amp *= _PYRAMID_DISCOUNT
        acc = acc + amp * up
        total += amp
    acc = acc / total
    return _normalize(acc.reshape(shape))


def make_noise(noise_type, shape, seed, strength, character=1.0):
    """Return a CPU float32 tensor of `shape` for the given noise type.

    zeros -> exact zeros (strength/seed irrelevant). Every other type is
    normalized to unit std then scaled by `strength`, so strength is expressed in
    units of standard latent noise (strength 1.0 == torch.randn scale).

    `character` (v833, 0..1, default 1.0): how much of the type's character
    survives. STREAM LAW, guard-pinned: the shaped tensor is drawn FIRST,
    the blending white only when character < 1.0 -- so 1.0 is bit-identical
    to the pre-dial builds and a saved workflow reproduces. offset scales
    its DC term by the dial instead; gaussian and zeros ignore it.
    """
    import torch
    shape = tuple(int(x) for x in shape)
    try:
        c = min(1.0, max(0.0, float(character)))
    except Exception:
        c = 1.0
    if noise_type == "zeros":
        return torch.zeros(shape, dtype=torch.float32)
    g = _generator(seed)
    if noise_type == "gaussian":
        out = torch.randn(shape, generator=g)
    elif noise_type == "offset":
        # White first -- at character 0 this IS the gaussian stream,
        # bit-identical, because the DC draw never happens.
        out = torch.randn(shape, generator=g)
        if c > 0.0:
            # v834 (audit A1): the DC term anchors on the LEADING batch/
            # channel axes and BROADCASTS over everything behind them.
            # v833 drew it as shape[:-2]+(1,1), which on a 5D video latent
            # (B,C,T,H,W) meant an INDEPENDENT offset per frame -- measured
            # frame-mean spread 0.13-0.22 across T against 0.02-0.03 of
            # pure white jitter: brightness flicker. With `lead` capped at
            # 2, the 4D/3D/2D shapes keep their exact v833 dc shape (and
            # stream), only 5D+ changes -- which is the fix.
            lead = min(2, max(0, len(shape) - 2))
            dc = torch.randn(tuple(shape[:lead])
                             + (1,) * (len(shape) - lead), generator=g)
            out = _normalize(out + (_OFFSET_MAX * c) * dc)
    else:
        if noise_type in COLORED_BETA:
            out = _colored(shape, COLORED_BETA[noise_type], g)
        elif noise_type == "fractal":
            out = _fractal(shape, g)
        elif noise_type == "pyramid":
            out = _pyramid(shape, g)
        else:
            # Unknown type -> safest fallback is flat white (never crash
            # a render).
            out = torch.randn(shape, generator=g)
        if c < 1.0:
            white = torch.randn(shape, generator=g)
            import math
            out = _normalize(math.sqrt(1.0 - c * c) * white + c * out)
    return (out * float(strength)).to(torch.float32)


# ---------------------------------------------------------------------------
# v685 -- NOISE source object (Core's own contract)
# ---------------------------------------------------------------------------

class ULSNoiseSource:
    """A NOISE object in ComfyUI's own shape: `.seed` plus
    `.generate_noise(latent) -> CPU tensor`.

    WHY AN OBJECT AND NOT A TENSOR: the geometry is not known where the seed
    is chosen -- it belongs to the latent, which the sampler holds. Core
    solved this the same way (Noise_RandomNoise), so speaking the same
    contract means our noise can drive Core's samplers and Core's RandomNoise
    can drive ours.

    BIT-IDENTITY: gaussian at strength 1.0 delegates to
    comfy.sample.prepare_noise -- the exact call the sampler made before this
    existed, batch_index included. So the default output of a wired noise pin
    is indistinguishable from an unwired one, and every other type is a
    deliberate departure from it.
    """

    def __init__(self, seed, noise_type="gaussian", strength=1.0,
                 character=1.0):
        self.seed = int(seed)
        self.noise_type = str(noise_type)
        self.strength = float(strength)
        try:
            self.character = min(1.0, max(0.0, float(character)))
        except Exception:
            self.character = 1.0

    def is_default(self):
        """True when this source reproduces prepare_noise exactly."""
        return self.noise_type == "gaussian" and abs(self.strength - 1.0) < 1e-9

    def generate_noise(self, input_latent):
        samples = input_latent["samples"]
        if self.is_default():
            import comfy.sample
            batch_inds = input_latent.get("batch_index", None)
            return comfy.sample.prepare_noise(samples, self.seed, batch_inds)
        # v872: a JOINT latent (MiniMax H3: NestedTensor((video, audio))) needs
        # one noise tensor PER PART, re-wrapped -- exactly what Core's
        # comfy.sample.prepare_noise does for the default path. Without this the
        # line below read `.shape`, which delegates to tensors[0], and returned
        # a flat VIDEO-shaped tensor against a nested latent. NestedTensor has
        # no __radd__, so the run died far from the cause -- and only a Seed
        # node left at gaussian/1.0 (which takes the branch above) ever worked.
        if getattr(samples, "is_nested", False):
            return self._shaped_noise_joint(samples)
        # Shaped noise. batch_index is NOT honoured here: it exists so a
        # re-run of one batch member reproduces its slice of a bigger noise
        # tensor, and that only has meaning for the plain randn layout.
        return make_noise(self.noise_type, tuple(samples.shape),
                          self.seed, self.strength, self.character)

    def _shaped_noise_joint(self, samples):
        """Thin seat on the module-level rule -- see shaped_noise_joint()."""
        return shaped_noise_joint(samples, self.noise_type, self.seed,
                                  self.strength, self.character)


def shaped_noise_joint(samples, noise_type, seed, strength, character=1.0):
    """Shaped noise for a joint latent: SHAPE THE VIDEO HALF ONLY.

        DECIDED, not defaulted (v872): offset, pyramid and blue noise are
        SPATIAL constructions -- pyramid noise builds an octave stack over H and
        W, offset noise adds a per-channel constant across the picture plane. An
        audio latent is [B, 32, 2, t]: it has no picture plane, so 'pyramid
        noise' there is not a weaker version of the effect, it is a different
        operation wearing the same name. The audio half therefore gets Core's
        plain gaussian, and the node SAYS which half got what -- a silent
        difference between the two halves is exactly the kind of thing that
        gets debugged for an hour six months later.
        """
    import comfy.nested_tensor
    import comfy.sample
    parts = list(samples.unbind())
    noises = [make_noise(noise_type, tuple(parts[0].shape),
                         seed, strength, character)]
    for t in parts[1:]:
        noises.append(comfy.sample.prepare_noise(t, seed, None))
    print("[PLS] joint latent -> '%s' noise (strength %.2f, character %.2f) "
          "on the VIDEO half %s; plain gaussian on %d further half/halves %s. "
          "Shaped noise is spatial and has no meaning on an audio latent."
          % (noise_type, strength, character, tuple(parts[0].shape),
             len(parts) - 1, [tuple(t.shape) for t in parts[1:]]))
    return comfy.nested_tensor.NestedTensor(noises)

    def __repr__(self):
        return ("ULSNoiseSource(seed=%d, type=%s, strength=%.3f, "
                "character=%.2f)"
                % (self.seed, self.noise_type, self.strength,
                   self.character))
