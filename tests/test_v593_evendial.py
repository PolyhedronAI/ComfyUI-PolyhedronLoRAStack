"""v593 guard: A SUGGESTION THE USER CANNOT TYPE IS NOT A SUGGESTION.

v591 added a note at `final pass begin` warning that an odd canvas would be
cropped at the save, and offering a dial value that would avoid it. On Frank's
2026-07-14 10:16 run it printed:

    NOTE - 1075x1075 has an odd edge. ... To land even, nudge
    final_upscale_by: on 768x768, 1.40 gives 1076px.

His dial was already at 1.40. 1.40 gives 1075. The note recommended the number
he already had and promised him a different result.

The bug is small and the lesson is not. The line computed 1076/768 = 1.40104,
printed it rounded to two decimals (the shape the widget has) and printed the
pixel count from the UNROUNDED value. Two numbers, one derivation, and the
rounding sat between them. It also re-derived the canvas arithmetic beside
`_final_canvas` instead of asking it - and a copy of a calculation always
drifts from the calculation, with the copy doing the lying.

The law, in two halves:

  1. The value is searched ON the grid the dial actually has (0.01, two
     decimals). What is printed must be typeable, exactly as printed.
  2. The canvas it claims is obtained from `_final_canvas` - the SAME function
     the pass just ran - never from a parallel formula.

This guard pins the INVARIANT, not the search (lesson 4): whatever `_even_dial`
returns, feeding it back through `_final_canvas` must land even. The search may
be rewritten; the promise may not be broken. And when it returns None, the
guard proves no value on the grid would have worked - a silent shrug is only
honest if the shelf is really empty.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PU = ROOT / "nodes" / "ph_power_upscale.py"

_KERNEL = "lanczos (cpu)"


def _fail(msg):
    print(f"[test_v593_evendial] FAIL: {msg}")
    sys.exit(1)


def _pure(src, name, ns):
    """Lift a module-level pure function into `ns` - ph_power_upscale imports
    torch and comfy, which no guard may need."""
    m = re.search(r"^def " + name + r"\(.*?(?=\n(?:def |class )|\Z)",
                  src, re.S | re.M)
    if not m:
        _fail(f"{name}() is gone")
    exec(m.group(0), ns)  # noqa: S102 - our own source, measured not believed
    return ns[name]


def main():
    src = PU.read_text(encoding="utf-8")
    ns = {"_NO_RESIZE": "none"}
    canvas = _pure(src, "_final_canvas", ns)
    dial = _pure(src, "_even_dial", ns)

    # ---- 0: the size law still produces Frank's field number ----------------
    # If this drifts, every claim below is about a different machine.
    if tuple(canvas(768, 768, 4.0, _KERNEL, 1.40)[:2]) != (1075, 1075):
        _fail("_final_canvas no longer reproduces the field: 768x768 @ 1.40 "
              "must be 1075x1075 (Frank's 2026-07-14 log). The guard is "
              "measuring a machine that is not the one that shipped.")

    # ---- 1: the case that caused this cut -----------------------------------
    # 1.39 -> 1068, NOT 1.41 -> 1082. _final_canvas ROUNDS; it does not floor.
    # The v593 author hand-computed the suggestion with int() while writing this
    # very fix, published "1.41 gives 1082" to Frank, and was caught by section
    # 2 below - which asks the size law instead of trusting a person. round(768
    # * 1.41) = 1083. Odd. He would have typed it and hit the identical crop.
    # At 1.40 int() and round() happen to agree (1075), so the field check
    # passed and the mistake stayed invisible. That is the whole lesson of this
    # file, and it bit the file's own author on the way in.
    got = dial(768, 768, 4.0, _KERNEL, 1.40)
    if got is None:
        _fail("no even dial found for 768x768 @ 1.40 - 1.39 lands 1068x1068")
    if got != (1.39, 1068, 1068):
        _fail(f"_even_dial(768,768,4.0,{_KERNEL!r},1.40) = {got}, expected "
              f"(1.39, 1068, 1068) - the nearest typeable notch that lands "
              f"even THROUGH _final_canvas (which rounds: 1.41 -> 1083, odd)")
    if abs(got[0] - 1.40) < 1e-9:
        _fail("the note suggests the dial the user ALREADY HAS - that is the "
              "v591 bug, verbatim")

    # ---- 2: THE INVARIANT. Whatever it returns must survive the size law. ---
    # This is the pin. The search may be rewritten; the promise may not break.
    checked = 0
    for w, h in [(768, 768), (832, 480), (1024, 576), (960, 544), (1104, 832),
                 (720, 1280), (513, 513), (1080, 1080)]:
        for scale in (2.0, 4.0):
            for d100 in range(25, 301, 7):       # 0.25 .. 3.00 across the grid
                dv = round(d100 / 100.0, 2)
                sug = dial(w, h, scale, _KERNEL, dv)
                if sug is None:
                    continue
                v, cw, ch = sug
                # (a) it must be typeable: two decimals, inside the widget range
                if round(v, 2) != v or not (0.25 <= v <= 8.0):
                    _fail(f"_even_dial({w},{h},{scale},{dv}) suggests {v!r} - "
                          f"not a value the 0.25..8.0 / 0.01 widget can hold")
                # (b) it must be TRUE: the pass's own law must land it even
                rw, rh = canvas(w, h, scale, _KERNEL, v)[:2]
                if (rw, rh) != (cw, ch):
                    _fail(f"_even_dial promises {cw}x{ch} at {v}, but "
                          f"_final_canvas produces {rw}x{rh} - the note is "
                          f"deriving the canvas beside the size law again "
                          f"(that IS the v591 bug)")
                if (int(rw) & 1) or (int(rh) & 1):
                    _fail(f"_even_dial({w},{h},{scale},{dv}) suggests {v} -> "
                          f"{rw}x{rh}, which is STILL ODD. The user would type "
                          f"it and hit the identical crop.")
                checked += 1
    if checked < 200:
        _fail(f"only {checked} suggestions exercised - the sweep is not "
              f"touching the code it claims to guard")

    # ---- 3: None must mean NONE (an honest shrug, not a lazy one) -----------
    # 'none' ignores the dial entirely: no notch can move that canvas. If the
    # function ever returns a value there, it is inventing one.
    if dial(768, 768, 4.0, "none", 1.40) is not None:
        _fail("_even_dial invents a suggestion under resize_method='none', "
              "where the dial is ignored by the size law - a confident answer "
              "to a question the code cannot answer")

    # ---- 4: the note actually USES it ---------------------------------------
    if "_even_dial(" not in src.split("def _even_dial", 1)[1]:
        _fail("_even_dial is defined but never called - the note is still "
              "printing a number of its own making")
    if re.search(r"gives \{int\(fw \* \(\(_et \+ 2\)", src):
        _fail("the v591 formula is still in the note - it prints a rounded "
              "dial next to unrounded pixels")

    print(f"PASS: v593 -- {checked} suggestions checked: every one is typeable "
          f"on the 0.01 grid AND lands even through _final_canvas itself; "
          f"768@1.40 -> 1.39 (1068x1068); 'none' honestly returns nothing")


if __name__ == "__main__":
    main()
