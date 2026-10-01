# -*- coding: ascii -*-
"""h3_prompt -- the MiniMax H3 prompt compiler and checker (Cine C1, v1017).

Pure Python, no torch, no ComfyUI: the guard drives every rule with strings.

1. TAGS INSTEAD OF NUMBERS
   A prompt names its references by NAME -- `@fox`, `@forest`, `@image_2`,
   `@video_1` -- never by number. The node that owns the references (Reference,
   Keyframes) knows which slots are wired and in which order the tokenizer
   will number them, and resolves every tag to the live `<Picture i>` /
   `<Video k>` / `<Audio j>` right before the encode. Rewire, unwire a slot,
   reorder -- the prompt stays right, because it never held a number.

   Why `@` and not `{fox}`: both prompt widgets run through ComfyUI's dynamic
   prompts, which resolve `{a|b}` in the FRONTEND before the backend ever sees
   the text -- `{fox}` would arrive as `fox`. `@` has no meaning there, none in
   the H3 grammar (`<Picture n>`, `<d>`, `[Shot n]`, `(S1)`) and none in the
   CLIP Text Encode.

   Every slot name is a tag by itself (`@image_3`, `@video_audio_1`,
   `@first_frame`). An ALIAS table gives slots speaking names:
       fox    = image_1
       forest = image_3
   An unknown tag, or an alias pointing at an unwired slot, is an ERROR that
   stops the run before sampling: a stray `@fox` reaching the model is a
   wasted run, and the node can see it coming.

2. THE CHECK
   Rules from 40_H3_REGELWERK.md that a string can prove, each with its
   paragraph. Three levels: ERROR (the prompt refers to something that is not
   there), WARN (the model will read it differently than meant), HINT (outside
   the measured/official band, maybe on purpose). Nothing here edits the
   prompt except the tag resolution -- the check REPORTS, it never rewrites
   (Blatt 80: no silent change).
"""

import re

TAG_RX = re.compile(r"(?<![\w@])@([A-Za-z_][A-Za-z0-9_]*)")
LABEL_RX = re.compile(r"<(Picture|Video|Audio)\s+(\d+)>")
SHOT_RX = re.compile(r"\[Shot\s+(\d+)\](\s+At\s+(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?)?")


def shot_markers(text):
    """v1023: the [Shot N] MARKERS of a prompt. The official text also REFERS
    to shots -- "(from [Shot 1])" in the alignment line, "(appears in
    [Shot 1], [Shot 2])" in retention_analysis -- always inside parentheses,
    and those are not markers (they made the check report "[Shot 1] follows
    [Shot 1]"). Line breaks prove nothing: the CLIP Text Encode strips them."""
    out = []
    for m in SHOT_RX.finditer(text or ""):
        before = text[:m.start()]
        if before.count("(") - before.count(")") > 0:
            continue
        out.append(m)
    return out
WEIGHT_RX = re.compile(r"\([^()\n]*:\s*-?\d+(?:\.\d+)?\s*\)")
ALIAS_RX = re.compile(r"^\s*@?([A-Za-z_][A-Za-z0-9_]*)\s*[=:]\s*@?([A-Za-z_][A-Za-z0-9_]*)\s*$")

WORDS_OFFICIAL = (350, 500)     # 40_H3_REGELWERK section 7
TOKENS_SOFT_CAP = 700
TOKENS_PER_WORD = 1.35          # English prose, rough -- labelled as estimate

ERROR, WARN, HINT = "ERROR", "WARN", "HINT"


def parse_aliases(text):
    """'name = slot' lines -> ({alias: slot}, [errors]). '#' and '//' comment
    lines and blank lines are ignored."""
    aliases, errors = {}, []
    for n, raw in enumerate((text or "").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        m = ALIAS_RX.match(line)
        if not m:
            errors.append("tags line %d is not 'name = slot': %r" % (n, line[:60]))
            continue
        name, slot = m.group(1), m.group(2)
        if name in aliases and aliases[name] != slot:
            errors.append("tags line %d: @%s is already %s" % (n, name, aliases[name]))
            continue
        aliases[name] = slot
    return aliases, errors


def resolve(text, slot_tags, all_slots, aliases=None):
    """Replace every @tag with its live label.

    slot_tags: {slot_name: "<Picture 2>", ...} for the WIRED slots.
    all_slots: every slot name the node has (wired or not) -- so '@image_4'
               with image_4 unwired says 'not wired' instead of 'unknown'.
    aliases:   {alias: slot_name}.
    Returns (resolved_text, used [(tag, label)], errors [str]).
    """
    aliases = aliases or {}
    used, errors, seen = [], [], set()
    for alias, slot in aliases.items():
        if alias in all_slots and slot != alias:
            errors.append("alias @%s shadows the slot name %s" % (alias, alias))
        if slot not in all_slots:
            errors.append("alias @%s points at %r, which is not a slot of this "
                          "node (slots: %s)" % (alias, slot, ", ".join(all_slots)))

    def sub(m):
        name = m.group(1)
        slot = aliases.get(name, name)
        if slot not in all_slots:
            if name not in seen:
                seen.add(name)
                errors.append("unknown tag @%s -- known: %s"
                              % (name, ", ".join(["@" + s for s in all_slots]
                                                 + ["@" + a for a in aliases])))
            return m.group(0)
        label = slot_tags.get(slot)
        if label is None:
            if name not in seen:
                seen.add(name)
                errors.append("@%s is %s, which is not wired -- it would refer "
                              "to nothing" % (name, slot))
            return m.group(0)
        if name not in seen:
            seen.add(name)
            used.append(("@" + name, label))
        return label

    return TAG_RX.sub(sub, text or ""), used, errors


def _shot_seconds(m):
    if not m.group(2):
        return None
    ms = (m.group(5) or "0").ljust(3, "0")
    return int(m.group(3)) * 60 + int(m.group(4)) + int(ms) / 1000.0


def check(text, live, duration_s=None):
    """[(level, message)] for a RESOLVED prompt.

    live: {"Picture": n, "Video": k, "Audio": j} -- what is wired.
    duration_s: clip length, for the shot-time rule (None = skip).
    """
    out = []
    t = text or ""

    # tags that survived resolution (only if resolve() was bypassed)
    for m in TAG_RX.finditer(t):
        out.append((ERROR, "unresolved tag @%s reaches the model" % m.group(1)))

    # section 5/refs: labels must point at something wired, wired refs should be named
    named = {"Picture": set(), "Video": set(), "Audio": set()}
    for m in LABEL_RX.finditer(t):
        kind, n = m.group(1), int(m.group(2))
        named[kind].add(n)
        if n < 1 or n > int(live.get(kind, 0)):
            out.append((ERROR, "<%s %d> refers to nothing -- %d %s reference(s) "
                        "wired" % (kind, n, int(live.get(kind, 0)), kind.lower())))
    # v1023: the official FL2VA text names its pictures WITHOUT brackets --
    # "Picture 1 (from Shot 1) aligns with ...", "established by Picture 2"
    # (10_H3_OFFIZIELL section 2, 60_VORLAGEN A3). That counts as named.
    for m in re.finditer(r"(?<![<\w])Picture (\d+)\b", t):
        named["Picture"].add(int(m.group(1)))
    for kind in ("Picture", "Video", "Audio"):
        for n in range(1, int(live.get(kind, 0)) + 1):
            if n not in named[kind]:
                out.append((HINT, "<%s %d> is wired but never named in the "
                            "prompt -- H3 binds a reference through its label"
                            % (kind, n)))

    # section 11: weight brackets are not parsed
    for m in WEIGHT_RX.finditer(t):
        out.append((WARN, "weight bracket %r is not parsed by H3 -- it reaches "
                    "the model as characters (40 section 11: use the LoRA "
                    "strength instead)" % m.group(0)[:40]))

    # section 36: rubric comments
    lines = t.splitlines()
    for n, line in enumerate(lines, start=1):
        s = line.lstrip()
        if s.startswith("//"):
            out.append((WARN, "line %d is a // rubric and reaches the model "
                        "here -- route the prompt through the CLIP Text Encode "
                        "(strip_comments) or remove it" % n))
            break
    for n, line in enumerate(lines, start=1):
        s = line.lstrip()
        if s.startswith("/") and not s.startswith("//"):
            out.append((WARN, "line %d starts with a single '/' -- a broken "
                        "rubric head leaks its whole line (40 section 36)" % n))

    # section 3: shots
    shots = shot_markers(t)
    if shots:
        want, last_t = 1, -1.0
        for m in shots:
            num = int(m.group(1))
            if num != want:
                out.append((WARN, "[Shot %d] follows [Shot %d] -- shots are "
                            "numbered 1, 2, 3 ... without gaps" % (num, want - 1)))
            want = num + 1
            sec = _shot_seconds(m)
            if num == 1 and sec is not None:
                out.append((WARN, "[Shot 1] carries a time -- the official form "
                            "is '[Shot 1]' alone, times start at [Shot 2]"))
            if num > 1 and sec is None:
                out.append((WARN, "[Shot %d] has no 'At MM:SS.mmm' -- every shot "
                            "after the first needs its start time" % num))
            if sec is not None:
                if sec <= last_t:
                    out.append((WARN, "[Shot %d] At %.3f s is not after the "
                                "previous shot (%.3f s)" % (num, sec, last_t)))
                if duration_s is not None and sec >= duration_s:
                    out.append((WARN, "[Shot %d] At %.3f s starts after the clip "
                                "ends (%.3f s)" % (num, sec, duration_s)))
                last_t = sec

    # section 5: dialogue
    opens, closes = t.count("<d>"), t.count("</d>")
    if opens != closes:
        out.append((WARN, "%d <d> against %d </d> -- a dialogue block is not "
                    "closed" % (opens, closes)))
    for m in re.finditer(r"<d>(.*?)</d>", t, re.S):
        if '"' in m.group(1):
            out.append((HINT, "double quotes inside <d>...</d> -- they are "
                        "reserved for visible on-screen text (40 section 5)"))
            break

    # section 7: length
    words = len(re.findall(r"\S+", t))
    est = int(round(words * TOKENS_PER_WORD))
    if est > TOKENS_SOFT_CAP:
        out.append((WARN, "~%d tokens (estimate, %d words) -- beyond ~%d you "
                    "work outside the official band on purpose (40 section 7)"
                    % (est, words, TOKENS_SOFT_CAP)))
    elif words > WORDS_OFFICIAL[1]:
        out.append((HINT, "%d words -- the official norm is %d-%d"
                    % (words, WORDS_OFFICIAL[0], WORDS_OFFICIAL[1])))
    return out


def report(used, findings):
    """Report lines for the info output / console."""
    lines = []
    if used:
        lines.append("tags: " + ", ".join("%s -> %s" % u for u in used))
    e = sum(1 for f in findings if f[0] == ERROR)
    w = sum(1 for f in findings if f[0] == WARN)
    h = sum(1 for f in findings if f[0] == HINT)
    lines.append("prompt check: %d error(s), %d warning(s), %d hint(s)" % (e, w, h))
    for level, msg in findings:
        lines.append("  %s: %s" % (level, msg))
    return lines


# --------------------------------------------------------------------------
# v1019 (Cine C2): the definition block the Reference Board drafts
# --------------------------------------------------------------------------

# Roles the board offers, per kind. "subject" roles become a <Subject k>
# (content reused in the target video); "picture" roles describe a concrete
# frame; video and audio roles name what the clip or sound provides.
ROLES = {
    "image": ["subject", "scene", "style", "first frame", "last frame",
              "keyframe", "storyboard"],
    "video": ["continuation", "motion", "structure", "edit source"],
    "audio": ["voice", "music", "ambience", "sound effect"],
}
SUBJECT_ROLES = ("subject", "scene", "style")
RETENTION = {
    "visual": ["fully_preserved", "partially_preserved", "attribute_transfer",
               "weak_reference"],
    "audio": ["fully_copy", "partially_copy", "reference", "weak_reference"],
}
_PICTURE_PHRASE = {
    "first frame": "the first frame of [Shot 1]",
    "last frame": "the final frame of the video",
    "keyframe": "a keyframe of the target video",
    "storyboard": "a storyboard reference for the shot order and viewpoints",
}
_VIDEO_PHRASE = {
    "continuation": "the source video whose ending the target video continues",
    "motion": "the source of the motion to follow",
    "structure": "the source of the cut and pacing structure",
    "edit source": "the source video that is edited",
}


def _clean(s):
    return " ".join(str(s or "").split()).rstrip(".")


def definitions_block(entries):
    """Draft subject_definitions + retention_analysis lines from board entries.

    entries: [{"kind", "tag", "role", "desc", "retention", "sound_tag"?}] in
    board order. References are written as @tags -- the Reference node turns
    them into the live <Picture i>/<Video k>/<Audio j> -- while <Subject k> is
    numbered here, in board order, because subjects are the prompt's own
    labels, not slots. Returns (definitions_lines, retention_lines).
    """
    defs, keep = [], []
    k = 0
    for e in entries:
        tag = "@" + e["tag"]
        desc = _clean(e.get("desc")) or "(describe it)"
        role = e.get("role", "")
        ret = e.get("retention") or ("reference" if e["kind"] == "audio" else "fully_preserved")
        if e["kind"] == "image" and role in SUBJECT_ROLES:
            k += 1
            what = {"subject": "", "scene": "the environment ", "style": "the visual style "}[role]
            defs.append("<Subject %d> is %s%s, as shown in %s." % (k, what, desc, tag))
            keep.append("<Subject %d>: %s - %s" % (k, ret, desc))
        elif e["kind"] == "image":
            defs.append("%s is %s, showing %s." % (tag, _PICTURE_PHRASE.get(role, "a reference frame"), desc))
            keep.append("%s (%s): %s - %s" % (tag, role or "picture", ret, desc))
        elif e["kind"] == "video":
            defs.append("%s is %s: %s." % (tag, _VIDEO_PHRASE.get(role, "a reference video"), desc))
            keep.append("%s (%s): %s - %s" % (tag, role or "video", ret, desc))
            if e.get("sound_tag"):
                stag = "@" + e["sound_tag"]
                defs.append("%s is the soundtrack of %s." % (stag, tag))
                keep.append("%s: %s - the soundtrack of %s" % (stag, "reference", tag))
        else:
            defs.append("%s is %s (%s)." % (tag, desc, role or "audio"))
            keep.append("%s: %s - %s" % (tag, ret, desc))
    return defs, keep
