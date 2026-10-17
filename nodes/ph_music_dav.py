# -*- coding: ascii -*-
"""
ph_music_dav.py -- audio INTO MiniMax-Music3: the DAV encoder, the one-time
safetensors conversion of MiniMax's dav.pth, and the strength -> denoise law.

WHY (Frank, 07.10.2026): "ob wir bei der Minimax H3 Music auch Input-Musik
reingeben koennen ... wie wir das bei I2V machen ... Musik analysieren, dann
als Latent, dann mit rein in die Music-Workflow". Core's DAV is decoder-only
(comfy/sd.py: "MiniMax Music3 DAV cannot encode audio"); MiniMax's original
dav.pth carries the encoder. This module is the encoder, written for this pack
from the published layout (DAC-style: Snake, weight-normalised convolutions,
strides 2/4/8/8 = hop 512 at 44.1 kHz, 1024 -> mean_proj -> 64 per channel,
two channels = the 128-channel latent the Music3 DiT samples).

MEASURED in S0 (07.10., Frank's GPU + the sandbox), the numbers the code
relies on:
  * dav.pth = encoder 22.4 M + mean_proj + logs_proj + decoder 54.2 M (bit-equal
    to Comfy's minimax_music3_dav.safetensors) + flow.flows 46.3 M; 491817450
    bytes, sha256 52adde6c...
  * round trip input -> encoder -> Comfy's decoder: log-mel 1.54 dB, SNR 29 dB,
    loudness +0.2 dB, stereo width kept;
  * the DiT returns our latent at denoise 0.2 with 0.97 dB log-mel -- the
    encoder lands in the DiT's space (gen latent std 2.71 / ours 3.25, same
    mean, same channel picture);
  * chunked encoding (20 s windows, 8192 samples overlap) == whole (1e-6);
  * the cliff: input at denoise <= 0.88, a true mix at 0.91, the caption alone
    at >= 0.94 (half at 0.89-0.92 in chroma, log-mel and loudness envelope).
    Music3 latents are ~3x louder than unit noise, so 80 percent of the
    denoise range does nothing. STRENGTH therefore maps onto the remaining
    signal-to-noise ratio, scaled by the INPUT's own latent std (see
    strength_to_denoise) -- calibrated with euler / simple, the template's
    sampler, on ONE input (an ambient drone); a second input is the open
    field test.

THE FILE (Frank, 07.10.: "Alles in Savetensors konvertieren"): the door
fetches MiniMax's dav.pth (pinned), this module converts ALL of it once into
minimax_music3_dav_full.safetensors (tensor-for-tensor verified) and the
download is deleted; a dav.pth the user placed himself is converted too but
never deleted. Loading reads only the safetensors file afterwards.
torch.load runs with weights_only=True on that one pinned file.

No ComfyUI node class is used here; only torch, safetensors and (optionally)
comfy.model_management for the device.
"""

import math
import os

SR = 44100
HOP = 512
LATENT_CHANNELS = 128
FOLDER = "audio_encoders"
FULL_FILE = "minimax_music3_dav_full.safetensors"
ORIG_FILE = "minimax_music3_dav_original.pth"       # the door's download name
USER_PTH_NAMES = (ORIG_FILE, "dav.pth")              # what a user may have placed
SOURCE_URL = "https://huggingface.co/MiniMaxAI/MiniMax-Music3/resolve/main/dav.pth"
SOURCE_BYTES = 491817450
SOURCE_SHA256 = "52adde6c6c52cca872f549449cd7b608677d1e69f4a98eee22da3403e63b56e8"
ENC_PREFIXES = ("encoder.", "mean_proj.")

# strength -> denoise (S0, see the header): the remaining SNR runs
# geometrically from SNR_AT_0 (input kept) to SNR_AT_1 (caption only); the
# denoise that leaves that SNR for an input of latent std s is s / (s + SNR).
# For the S0 input (std 3.25): strength 0 -> 0.80, 0.5 -> 0.91, 1 -> 0.96.
SNR_AT_0 = 0.80
SNR_AT_1 = 0.13
STD_FALLBACK = 3.25
# The law is calibrated at ONE level (3.25, the S0 input; a generated song sits at
# 2.71). Outside a sane band it would be a guess -- the level that goes into the
# law is held inside it and the node says so (clamp_std).
STD_MIN, STD_MAX = 1.5, 6.0


def clamp_std(std):
    """-> (the level the law uses, note or '')."""
    try:
        v = float(std)
    except (TypeError, ValueError):
        return STD_FALLBACK, "level unknown -- the measured %.2f is used" % STD_FALLBACK
    if v < STD_MIN or v > STD_MAX:
        c = min(STD_MAX, max(STD_MIN, v))
        return c, ("latent level %.2f is outside the measured band %.1f-%.1f -- the law uses %.2f"
                   % (v, STD_MIN, STD_MAX, c))
    return v, ""


def strength_to_denoise(strength, latent_std=STD_FALLBACK):
    s = min(1.0, max(0.0, float(strength)))
    std = float(latent_std) if latent_std and latent_std > 0 else STD_FALLBACK
    snr = math.exp(math.log(SNR_AT_0) + s * (math.log(SNR_AT_1) - math.log(SNR_AT_0)))
    return round(std / (std + snr), 4)


# --------------------------------------------------------------- the encoder
def _torch():
    import torch
    return torch


def build_encoder():
    """The DAV encoder as plain modules; weights come from load_encoder_state."""
    torch = _torch()
    nn = torch.nn
    F = torch.nn.functional

    class Snake1d(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.alpha = nn.Parameter(torch.ones(1, ch, 1))

        def forward(self, x):
            a = self.alpha
            return x + (a + 1e-9).reciprocal() * torch.sin(a * x).pow(2)

    class WNConv1d(nn.Module):
        """Stored as weight_g / weight_v (torch's old weight_norm over dim 0);
        folded into a plain weight once at load (fold())."""

        def __init__(self, cin, cout, k, stride=1, padding=0, dilation=1):
            super().__init__()
            self.weight_g = nn.Parameter(torch.ones(cout, 1, 1))
            self.weight_v = nn.Parameter(torch.zeros(cout, cin, k))
            self.bias = nn.Parameter(torch.zeros(cout))
            self.stride, self.padding, self.dilation = stride, padding, dilation
            self._w = None

        def fold(self):
            v = self.weight_v
            self._w = (self.weight_g * v / v.pow(2).sum(dim=(1, 2), keepdim=True).sqrt()).detach()

        def forward(self, x):
            w = self._w if self._w is not None else \
                self.weight_g * self.weight_v / self.weight_v.pow(2).sum(dim=(1, 2), keepdim=True).sqrt()
            return F.conv1d(x, w.to(x.dtype), self.bias.to(x.dtype), self.stride, self.padding, self.dilation)

    class Residual(nn.Module):
        def __init__(self, dim, dilation):
            super().__init__()
            self.block = nn.Sequential(Snake1d(dim), WNConv1d(dim, dim, 7, dilation=dilation, padding=3 * dilation),
                                       Snake1d(dim), WNConv1d(dim, dim, 1))

        def forward(self, x):
            r = self.block(x)
            if r.shape[-1] != x.shape[-1]:
                p = (x.shape[-1] - r.shape[-1]) // 2
                x = x[..., p:x.shape[-1] - p]
            return x + r

    class Down(nn.Module):
        def __init__(self, dim, stride):
            super().__init__()
            self.block = nn.Sequential(Residual(dim // 2, 1), Residual(dim // 2, 3), Residual(dim // 2, 9),
                                       Snake1d(dim // 2),
                                       WNConv1d(dim // 2, dim, 2 * stride, stride=stride,
                                                padding=math.ceil(stride / 2)))

        def forward(self, x):
            return self.block(x)

    class Body(nn.Module):
        def __init__(self, dim=64, rates=(2, 4, 8, 8), latent_dim=1024):
            super().__init__()
            layers = [WNConv1d(1, dim, 7, padding=3)]
            for s in rates:
                dim *= 2
                layers.append(Down(dim, s))
            layers += [Snake1d(dim), WNConv1d(dim, latent_dim, 3, padding=1)]
            self.block = nn.Sequential(*layers)

        def forward(self, x):
            return self.block(x)

    class DAVEncoder(nn.Module):
        """[B, 2, samples] at 44.1 kHz -> [B, 128, samples / 512]."""

        def __init__(self):
            super().__init__()
            self.encoder = Body()
            self.mean_proj = nn.Conv1d(1024, 64, kernel_size=1)

        def fold(self):
            for m in self.modules():
                if isinstance(m, WNConv1d):
                    m.fold()
            return self

        def forward(self, wav):
            b, c, n = wav.shape
            h = self.encoder(wav.reshape(b * c, 1, n))
            return self.mean_proj(h).reshape(b, c * 64, -1)

    return DAVEncoder()


def encoder_state(sd):
    """The encoder's tensors out of a full DAV state dict."""
    return {k: v for k, v in sd.items() if k.startswith(ENC_PREFIXES)}


def load_encoder(path, device=None):
    """Build + load + fold the encoder from the full safetensors file."""
    torch = _torch()
    from safetensors.torch import load_file
    sd = encoder_state(load_file(path))
    m = build_encoder()
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError("Music Audio Encode: %s does not hold the MiniMax-Music3 DAV encoder "
                           "(missing %d, unexpected %d keys)" % (os.path.basename(path), len(missing), len(unexpected)))
    m.eval()
    m.requires_grad_(False)
    m.to(device or torch.device("cpu"))
    return m.fold()          # after .to(): the folded weight is a plain tensor and would not move


def encode_chunked(model, wav, chunk_s=20.0, overlap=8192):
    """Long audio in windows; the overlap is cut away after encoding so every
    kept frame saw context on both sides (S0: == whole, max |d| ~1e-6)."""
    torch = _torch()
    n = wav.shape[-1]
    chunk = max(HOP, int(chunk_s * SR) // HOP * HOP)
    out, start = [], 0
    while start < n:
        end = min(n, start + chunk)
        a, b = max(0, start - overlap), min(n, end + overlap)
        z = model(wav[..., a:b])
        z0 = (start - a) // HOP
        out.append(z[..., z0:z0 + (end - start) // HOP])
        start = end
    return torch.cat(out, dim=-1)


def prepare_waveform(waveform, sample_rate, start=0.0, length=0.0):
    """AUDIO waveform [B, C, N] -> [1, 2, N'] float32 at 44.1 kHz, N' a multiple
    of 512. Mono is doubled, more than two channels keep the first two; only
    the first item of a batch is used. -> (wav, notes)."""
    torch = _torch()
    notes = []
    w = waveform
    if w.dim() == 2:
        w = w.unsqueeze(0)
    if w.shape[0] > 1:
        notes.append("batch of %d: the first item is used" % w.shape[0])
    w = w[:1].float()
    if w.shape[1] == 1:
        w = w.repeat(1, 2, 1)
        notes.append("mono doubled to stereo")
    elif w.shape[1] > 2:
        notes.append("%d channels: the first two are used" % w.shape[1])
        w = w[:, :2]
    sr = int(sample_rate)
    a = max(0, int(round(float(start or 0) * sr)))
    b = w.shape[-1] if not length or length <= 0 else min(w.shape[-1], a + int(round(float(length) * sr)))
    w = w[..., a:b]
    if sr != SR:
        try:
            import torchaudio.functional as AF
            w = AF.resample(w, sr, SR)
        except Exception:
            n = int(round(w.shape[-1] * SR / float(sr)))
            w = torch.nn.functional.interpolate(w, size=n, mode="linear", align_corners=False)
            notes.append("resampled linearly (torchaudio missing)")
        notes.append("resampled %d -> %d Hz" % (sr, SR))
    n = w.shape[-1] // HOP * HOP
    if n <= 0:
        raise ValueError("Music Audio Encode: the audio is shorter than %d samples (%.3f s)" % (HOP, HOP / float(SR)))
    return w[..., :n].contiguous(), notes


# --------------------------------------------------------------- the file
def _unwrap(obj):
    for key in ("state_dict", "model", "generator"):
        if isinstance(obj, dict) and isinstance(obj.get(key), dict):
            return _unwrap(obj[key])
    return obj


def convert_pth(src, dst):
    """MiniMax's dav.pth -> ONE safetensors file with every tensor (encoder,
    decoder, flow), verified tensor for tensor. -> report dict."""
    torch = _torch()
    from safetensors.torch import save_file, load_file
    try:
        raw = torch.load(src, map_location="cpu", weights_only=True, mmap=True)
    except Exception:
        raw = torch.load(src, map_location="cpu", weights_only=True)
    sd = {k: v.detach().contiguous().clone() for k, v in _unwrap(raw).items() if torch.is_tensor(v)}
    if not encoder_state(sd):
        raise RuntimeError("%s holds no DAV encoder (encoder.* / mean_proj.*) -- not MiniMax's dav.pth"
                           % os.path.basename(src))
    part = dst + ".part"
    save_file(sd, part, metadata={"source": SOURCE_URL, "source_sha256": SOURCE_SHA256,
                                  "converted_by": "Polyhedron Suite (ph_music_dav)"})
    back = load_file(part)
    bad = [k for k in sd if k not in back or back[k].shape != sd[k].shape or not torch.equal(back[k], sd[k])]
    if bad or len(back) != len(sd):
        os.remove(part)
        raise RuntimeError("safetensors conversion did not verify (%d tensors differ)" % len(bad))
    os.replace(part, dst)
    return {"tensors": len(sd), "params_M": round(sum(v.numel() for v in sd.values()) / 1e6, 2),
            "encoder_tensors": len(encoder_state(sd)), "bytes": os.path.getsize(dst)}


def _pw():
    try:
        from . import ph_weights as PW
    except ImportError:
        import ph_weights as PW
    return PW


def find_full():
    """The converted file on disk, or None."""
    return _pw().find_file(FOLDER, FULL_FILE)


def find_user_pth():
    """A dav.pth the user placed (audio_encoders or vae), or None."""
    PW = _pw()
    for folder in (FOLDER, "vae"):
        for name in USER_PTH_NAMES:
            p = PW.find_file(folder, name)
            if p:
                return p
    return None


def ensure_full(convert_user_file=True):
    """The safetensors path, converting a user-placed dav.pth on the way (local
    work only -- this never downloads). None when there is nothing to use."""
    p = find_full()
    if p:
        return p
    src = find_user_pth() if convert_user_file else None
    if not src:
        return None
    dst = os.path.join(os.path.dirname(src), FULL_FILE)
    rep = convert_pth(src, dst)
    print("[PLS] Music Audio Encode: converted %s -> %s (%d tensors, %.1f M params, verified); "
          "your %s stays where it is" % (os.path.basename(src), FULL_FILE, rep["tensors"], rep["params_M"],
                                          os.path.basename(src)))
    return dst


def after_download(pth_path):
    """The door's post-step: convert the downloaded .pth, delete it."""
    dst = os.path.join(os.path.dirname(pth_path), FULL_FILE)
    rep = convert_pth(pth_path, dst)
    try:
        os.remove(pth_path)
    except OSError:
        pass
    print("[PLS] Music weights: %s -> %s (%d tensors, verified), the .pth download removed"
          % (ORIG_FILE, FULL_FILE, rep["tensors"]))
    return dst


def missing_text():
    PW = _pw()
    return ("[PLS] Music Audio Encode: the MiniMax-Music3 DAV encoder is not installed. Click Download in the "
            "node (MiniMax's dav.pth, 469 MB, converted once into %s) -- or put dav.pth from %s into %s; "
            "it is converted on the next run." % (FULL_FILE, SOURCE_URL, PW.primary_dir(FOLDER)))
