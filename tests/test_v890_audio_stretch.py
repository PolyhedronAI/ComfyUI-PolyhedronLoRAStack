"""
test_v890_audio_stretch.py -- promises of the audio retime line.

WHAT IS PINNED, and why it is DRIVEN rather than read:

  A1  the plan arithmetic against HAND computation -- a wrong tempo does not
      crash, it detunes the soundtrack, the silent failure class.
  A2  the atempo chain: every stage inside [0.5, 2.0] and the PRODUCT equals
      the tempo. A chain whose product drifts IS the detune bug.
  A3  the machinery is RUN: length and pitch measured off a real 440 Hz probe
      through the real PyAV graph, both pitch modes. This is the 28.08.
      sandbox measurement, promoted to a guard so it cannot rot.
  A4  STEREO STAYS STEREO with channel identity intact -- the wound this
      build's own first probe caught (atempo hands back PACKED flt; read
      naively, one channel silently vanished into doubled length).
  A5  the interpolate coupling: audio_mode sits at the END of the canon
      (#577), the historical outputs keep their slots, `audio` is appended
      LAST (the v887 twin rule), and the tempo the node uses comes from the
      SHARED plan, not a re-derivation.
  A6  keep stays keep: mode "keep" must hand the SAME object through -- the
      v887 behaviour is a promise, not a default that may quietly change.
  A7  failure falls back to the ORIGINAL audio, out loud -- a soundtrack
      problem must never cost a finished interpolation.
"""

import ast
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nodes"))

FAILED = []


def check(cond, label):
    if cond:
        print("  PASS  " + label)
    else:
        print("  FAIL  " + label)
        FAILED.append(label)


def _read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def _hz(x, sr):
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1.0 / sr)[int(np.argmax(spec))])


def main():
    import torch
    from uls_audio_math import (atempo_chain, stretch_plan, TEMPO_EPS,
                                ATEMPO_MIN, ATEMPO_MAX)
    import ph_audio_stretch as PAS

    print("A1 - the plan against hand computation")
    p = stretch_plan(65, 16.0, 129, 16.0)          # 2x slow motion
    check(p["action"] == "stretch", "65f@16 -> 129f@16 is a stretch")
    check(abs(p["tempo"] - 0.5) < 1e-12,
          "slow-motion tempo is EXACTLY 0.5 - the span formula, hand-checked: "
          "16*(65-1)/(16*(129-1)) (got %.6f)" % p["tempo"])
    p = stretch_plan(2, 16.0, 5, 16.0)             # 4x slow, tiny clip
    check(abs(p["tempo"] - 0.25) < 1e-12,
          "4x slow on a 2-frame clip is exactly 0.25 (span, not duration)")
    p = stretch_plan(1, 16.0, 1, 16.0)
    check(p["action"] == "trim", "a single frame has no span - trim, no math")
    p = stretch_plan(65, 16.0, 129, 32.0)          # the interpolate standard
    check(p["action"] == "trim",
          "the standard case (fps doubles too) is TRIM, not resynthesis")
    check(abs(p["d_out"] - 129 / 32.0) < 1e-12, "trim target is the output duration")
    p = stretch_plan(65, 16.0, 129, 16000.0)
    check(p["action"] == "refuse", "an insane tempo is refused")
    check("wiring mistake" in p["note"], "and the refusal says why")
    p = stretch_plan(0, 16.0, 129, 16.0)
    check(p["action"] == "refuse", "zero frames refuse instead of dividing")

    print("A2 - the chain: banded stages, exact product")
    for tempo in (0.3, 0.5, 0.9, 1.0, 1.5, 2.0, 5.0, 0.0625, 16.0):
        ch = atempo_chain(tempo)
        prod = 1.0
        for s in ch:
            prod *= s
        ok_band = all(ATEMPO_MIN - 1e-9 <= s <= ATEMPO_MAX + 1e-9 for s in ch)
        check(ok_band and abs(prod - tempo) < 1e-9,
              "chain(%.4f) = %s, product exact, all in band" % (tempo, ch))

    print("A3 - the machinery is RUN (length + pitch off a real probe)")
    sr = 48000
    t = torch.arange(int(sr * 2.0)) / sr
    sig = (0.6 * torch.sin(2 * math.pi * 440 * t)).float()
    mono = {"waveform": sig.view(1, 1, -1), "sample_rate": sr}

    out = PAS.stretch_audio(mono, 0.5, "preserve", d_target=4.0)
    d = PAS.audio_duration(out)
    check(3.9 <= d <= 4.0, "preserve 2x slow: %.3fs (window 3.9..4.0)" % d)
    hz = _hz(out["waveform"][0, 0].numpy(), sr)
    check(abs(hz - 440.0) < 2.0, "preserve keeps pitch (%.1f Hz)" % hz)

    out = PAS.stretch_audio(mono, 0.5, "follow speed", d_target=4.0)
    d = PAS.audio_duration(out)
    check(abs(d - 4.0) < 0.01, "follow-speed 2x slow: %.3fs" % d)
    hz = _hz(out["waveform"][0, 0].numpy(), sr)
    check(abs(hz - 220.0) < 2.0,
          "follow-speed halves pitch (%.1f Hz, soll 220)" % hz)

    out = PAS.stretch_audio(mono, 2.0, "preserve", d_target=1.0)
    d = PAS.audio_duration(out)
    check(0.98 <= d <= 1.0, "preserve 2x fast: %.3fs" % d)

    out = PAS.trim_audio(mono, 1.5)
    check(abs(PAS.audio_duration(out) - 1.5) < 1e-6, "trim cuts exactly")
    out = PAS.trim_audio(mono, 5.0)
    check(PAS.audio_duration(out) == 2.0, "trim never pads")

    print("A4 - stereo stays stereo, channels stay themselves")
    import numpy as np
    L = sig
    R = 0.5 * sig
    stereo = {"waveform": torch.stack([L, R]).view(1, 2, -1), "sample_rate": sr}
    out = PAS.stretch_audio(stereo, 0.5, "preserve", d_target=4.0)
    w = out["waveform"]
    check(w.shape[1] == 2, "two channels in, two channels out (got %s)"
          % (tuple(w.shape),))
    if w.shape[1] == 2:
        l = w[0, 0].numpy()
        r = w[0, 1].numpy()
        mask = np.abs(l) > 0.2
        ratio = float(np.median(np.abs(r)[mask] / np.abs(l)[mask]))
        check(abs(ratio - 0.5) < 0.05,
              "channel identity: R/L level %.3f (soll 0.5) - the packed-flt "
              "wound stays closed" % ratio)
        check(abs(_hz(l, sr) - 440) < 2 and abs(_hz(r, sr) - 440) < 2,
              "both channels keep pitch")
    src_pin = _read("nodes", "ph_audio_stretch.py")
    check('graph.add("aformat"' in src_pin,
          "the aformat pin exists (planar output is FORCED, not assumed)")

    print("A5 - the interpolate coupling")
    interp = _read("nodes", "ph_interpolate.py")
    tree = ast.parse(interp)
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and "Interpolate" in n.name)
    # canon order out of INPUT_TYPES required, by AST
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
              and n.name == "INPUT_TYPES")
    req_keys = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value == "required" \
                        and isinstance(v, ast.Dict):
                    req_keys = [kk.value for kk in v.keys
                                if isinstance(kk, ast.Constant)]
    check(bool(req_keys) and req_keys[-1] == "audio_mode",
          "audio_mode is the LAST required widget (#577), canon tail: %s"
          % req_keys[-3:])
    rt = re.search(r'RETURN_TYPES = \(([^)]*)\)', interp).group(1)
    slots = [s.strip().strip('"') for s in rt.split(",") if s.strip()]
    check(slots[:5] == ["IMAGE", "INT", "FLOAT", "STRING", "VIDEO"],
          "the five historical output slots kept their places")
    check(slots[-1] == "AUDIO", "audio is appended LAST (v887 twin rule)")
    check("stretch_plan(" in interp and "from" in interp,
          "the tempo comes from the SHARED plan")
    check(not re.search(r"^def stretch_plan", interp, re.M),
          "and is NOT re-implemented in the interpolate")

    print("A6 - keep stays keep")
    src = interp
    m = re.search(r"def _retime_audio.*?(?=\n\ndef )", src, re.S)
    check(m is not None, "_retime_audio is liftable")
    ns = {}
    exec(compile(m.group(0), "<retime>", "exec"), ns)
    # The probe must be a REAL audio dict: the first version passed object(),
    # and a mutation that copied dicts on the keep path sailed past its own
    # isinstance gate unseen -- a stand-in that never enters the mutated
    # branch checks nothing (the v888 lesson, shape two).
    same = {"waveform": "x", "sample_rate": 48000}
    got, note = ns["_retime_audio"](same, "keep", 65, 16.0, 129, 32.0)
    check(got is same, "mode keep hands the SAME dict through, not a copy")
    got, note = ns["_retime_audio"](None, "stretch to output", 65, 16, 129, 16)
    check(got is None, "no audio in, no audio out, no crash")
    got, note = ns["_retime_audio"](same, "mute", 65, 16.0, 129, 32.0)
    check(got is None and "mute" in note, "mute is honoured and named")

    print("A7 - failure falls back to the original, out loud")
    junk = {"waveform": "not a tensor", "sample_rate": 48000}
    out = PAS.stretch_audio(junk, 0.5, "preserve")
    check(out is junk, "junk audio comes back unchanged instead of raising")
    got, note = ns["_retime_audio"](junk, "stretch to output", 65, 16, 129, 16)
    check(got is junk or isinstance(got, dict),
          "the interpolate path survives junk audio")

    print("")
    if FAILED:
        print("FAIL -- %d broken promise(s)" % len(FAILED))
        for f in FAILED:
            print("   * " + f)
        return 1
    print("PASS -- v890 audio retime holds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
