# -*- coding: ascii -*-
"""
uls_sched_loras.py
==================
v989 -- LoRAs that only work on their OWN trained sigma schedule.

A step-distilled LoRA such as HyperFlow 8-step for MiniMax H3 was trained on a
fixed grid of sigmas. On any other schedule (the sampler's scheduler widget,
a curve node) it still runs -- and gives the soft, noisy or burnt result that
makes people think the LoRA is broken. The grid lives in the Sigma List's
preset table (wan_sigma_schedule.SIGMA_PRESETS); the Stack and the Engine
know which LoRAs are active; the Sampler is the only place that sees both
the model and the schedule. So:

  * the Stack/Engine attach a small record to the model they return when such
    a LoRA is active (ATTACH_KEY) -- the same way a PDD head bank rides along;
  * the Sampler reads it and, like the PDD guard (pdd_plan), REFUSES a run
    that has no external SIGMAS at all, with one numbered message saying what
    to connect; with SIGMAS connected it only compares them to the trained
    grid and says so when they differ (a hand-made grid may be deliberate).

Recognition is by the file name, the same contract the PDD trunk pairing uses
(uls_pdd_apply.name_trunk): the table below is the one place to add another
schedule-bound LoRA. Pure: no torch, no comfy.
"""

ATTACH_KEY = "pls_schedule_lora"

# token (lower-case, found in the file's base name) -> the recipe
SCHEDULE_LORAS = (
    {"token": "hyperflow", "label": "HyperFlow 8-step",
     "preset": "hyperflow_8step_h3_raw", "sampler": "euler", "cfg": 1.0},
)

GRID_TOL = 1e-3     # a connected grid within this of the preset IS the preset


def _base(name):
    s = str(name or "").replace("\\", "/").rsplit("/", 1)[-1].lower()
    return s[:-len(".safetensors")] if s.endswith(".safetensors") else s


def recognise(names_weights):
    """[(entry, file name)] for every active schedule-bound LoRA.
    names_weights: iterable of (name, weight); weight 0 counts as inactive."""
    out = []
    for name, w in names_weights:
        try:
            if abs(float(w)) < 1e-6:
                continue
        except (TypeError, ValueError):
            continue
        b = _base(name)
        for e in SCHEDULE_LORAS:
            if e["token"] in b:
                out.append((e, str(name)))
    return out


def record(found, previous=None):
    """The attachment for the model: previous records (an earlier Stack or
    Engine in the chain) plus the new ones, one per preset."""
    rec = dict(previous or {})
    for e, name in found:
        cur = rec.get(e["preset"])
        names = list(cur["names"]) if cur else []
        if name not in names:
            names.append(name)
        rec[e["preset"]] = {"label": e["label"], "preset": e["preset"],
                            "sampler": e["sampler"], "cfg": e["cfg"], "names": names}
    return rec


def _short(name):
    b = str(name).replace("\\", "/").rsplit("/", 1)[-1]
    return b[:-len(".safetensors")] if b.endswith(".safetensors") else b


def plan(rec, sigmas_list, sampler_name, cfg, preset_grid):
    """Decide for one run. rec: the attachment (or None). sigmas_list: the
    external SIGMAS as floats, or None when nothing is connected.
    preset_grid(name) -> the resolved grid (floats) of a Sigma List preset.

    Returns notes (list of str) for a run that may proceed; raises ValueError
    with ONE numbered message when it may not (no external SIGMAS)."""
    notes = []
    if not rec:
        return notes
    missing = []
    for preset, r in sorted(rec.items()):
        who = ", ".join(_short(n) for n in r["names"])
        if sigmas_list is None:
            missing.append((r, who))
            continue
        try:
            grid = list(preset_grid(preset))
        except Exception:
            grid = None
        if grid is not None and sigmas_list:     # [] = connected elsewhere (High+Low pair)
            same = (len(grid) == len(sigmas_list)
                    and all(abs(a - b) <= GRID_TOL for a, b in zip(grid, sigmas_list)))
            if not same:
                notes.append(
                    "%s LoRA (%s) is active, but the connected SIGMAS are not its "
                    "trained grid (%d steps connected, the preset '%s' has %d). "
                    "Deliberate? Then fine -- otherwise pick that preset on the "
                    "Sigma List." % (r["label"], who, max(0, len(sigmas_list) - 1),
                                     preset, max(0, len(grid) - 1)))
        if str(sampler_name) != r["sampler"]:
            notes.append("%s was trained with sampler '%s' (running '%s')"
                         % (r["label"], r["sampler"], sampler_name))
        try:
            if abs(float(cfg) - float(r["cfg"])) > 1e-6:
                notes.append("%s was trained for cfg %.1f (running %.2f)"
                             % (r["label"], r["cfg"], float(cfg)))
        except (TypeError, ValueError):
            pass
    if missing:
        lines = ["[PLS] Sampler: %d schedule-bound LoRA%s active, but nothing is "
                 "connected to 'sigmas':" % (len(missing), "" if len(missing) == 1 else "s")]
        for i, (r, who) in enumerate(missing, 1):
            lines.append(
                "  %d. %s (%s) only works on its trained sigma grid -- the "
                "scheduler widget would give it another one. Connect a "
                "\u2b21 Polyhedron Sigma List with preset '%s' to the Sampler's "
                "'sigmas' input (recipe: sampler %s, cfg %.1f)."
                % (i, r["label"], who, r["preset"], r["sampler"], r["cfg"]))
        lines.append("  Or switch the LoRA off for this run.")
        raise ValueError("\n".join(lines))
    return notes
