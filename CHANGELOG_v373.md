# v373 — an invented token limit, removed

Version triple `3.73.0` / banner `Polyhedron Suite  v373` / `PLUGIN_VERSION v373`.
**37 nodes, unchanged.** No new node, no new route, no widget.

Carries the internal cuts **v907 + v908**. `nodes/uls_routes.py` is still
`bcc4d8c4` — eleventh release running.

## What was wrong

The Token Counter reported `412 / 512` on MiniMax H3 and called 512 a limit.
It is not one. Measured in core v0.33.4,
`comfy/text_encoders/qwen3vl.py`: `max_length=99999999`,
`pad_to_max_length=False`. **That encoder never truncates.**

Worse, a sticky toast said `Token limit exceeded — may be silently truncated
or crash kijai's WanVideoSampler` on every over-budget run, on every encoder.
On an H3 graph both halves are false and there is no kijai sampler in sight.
The report and the toast contradicted each other on one screen, and the toast
is the one that appears unasked.

## What is true instead, and what the node now says

H3 is a single-stream packed-token transformer: text, video and audio share
ONE sequence and ONE position axis (`comfy/ldm/minimax/model.py`,
`PackedLayout`). One text token costs 1.0 on that axis; one latent frame costs
1.67 or 6.67. **The prompt pushes the clip along the axis.** A real run: 378
tokens against a 22-frame clip puts the video at `t=378..414.7` while the clip
itself spans `36.7` — the prompt occupies 10.3x the video's own extent. RoPE
encodes distance, so the opening of a long prompt pulls weaker than its end.

The node now reports that ratio for H3, with the actionable half: **put what
matters most LAST**. Wire the `latent` (a new optional SOCKET, appended after
`clip`; the widget baseline is byte-identical, #577 holds) and it appears.
Without it the ratio is `None` and the report says so rather than inventing a
figure.

The 512 is not invented either — it is ai-toolkit's `max_text_length` default
for training, whose own comment reads "the released stack has no limit". The
report now calls it what it is: the span LoRAs are trained within, **not a
cap**.

The toast asks the backend whether any live encoder can truncate at all. Where
one can, the old wording survives verbatim — it is true for kijai's fixed 512
buffer. Where it cannot, it says so. With no `clip` wired the caveat stays: an
unknown encoder is not a safe encoder.

## Two errors of mine, both caught in the field

The first H3 detector looked for "minimax" in the encoder's NAME and could
never fire: `MiniMaxH3Tokenizer` passes `embedding_key="qwen3vl_32b"`, the
same name a plain Qwen3-VL uses. Written from an assumption I never measured.
It keys on the tokenizer CLASS now, and the guard drives it against both.

And a guard of mine rejected correct text, searching for the word "truncated"
in a branch that legitimately reads "Nothing is truncated". Re-anchored on the
claim.

## Guards

`test_v907_h3_reach.py` (27 promises, 8 mutations red) and
`test_v908_toast_truth.py` (14 promises, 3 mutations red), both adopted here
and running honestly green — neither weakened for the public tree.

Suite **128 -> 130**.
