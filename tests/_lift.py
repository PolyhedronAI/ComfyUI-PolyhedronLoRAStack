# -*- coding: ascii -*-
"""Lifting source out of a module, CLOSED.

Several guards cannot import the module they test -- uls_sampler.py needs the
whole comfy stack -- so they lift the piece under test out of the file as text
and run it against stubs. That is the right instrument: it tests the shipped
source, not a restatement of it.

It has one failure mode, and the tree has now paid for it twice. A guard's lift
list is written by hand, so it records the dependencies the seam had ON THE DAY
IT WAS WRITTEN. When a later cut gives the seam a new helper, the list is not
updated, the exec raises NameError, and the guard goes red for a reason that has
nothing to do with its promise:

    v685  _initial_noise gained _latent_parts       (v870)  -> NameError
    v839  the apply block gained _is_ragged_latent  (v870)  -> NameError

Both stood red for many versions and made a healthy tree look broken. Worse, the
v893 handover reasoned from "green in the public build, red internally" that the
wound must be IN THE TREE -- the public build is green precisely because it does
not have the v870 construction, so its stale lift list still fits. A red guard
is evidence about the guard until the tree has been asked directly.

So the list is not written by hand any more. `close_over` starts from the names
a guard actually wants and pulls in, transitively, every TOP-LEVEL name of the
module those pieces reference. What the harness provides itself (torch, a comfy
stub, injected values) is declared; anything left over is reported by NAME, so
the failure says "the lift is short of X", never "X is not defined".

ASCII only, standard library only, no ComfyUI imports.
"""

import ast
import builtins


def _segment(text, node, lines):
    """Source of one top-level statement, decorators included."""
    start = min([node.lineno] + [d.lineno for d in
                                 getattr(node, "decorator_list", [])]) - 1
    end = node.end_lineno
    return "\n".join(lines[start:end])


def top_level(text):
    """Map every top-level name of a module to its source segment.

    Functions, classes and simple assignments -- the three shapes a seam is
    ever built out of. Order is source order, which matters: a lifted block is
    executed, so definitions must arrive in an order that runs.
    """
    lines = text.split("\n")
    out = []
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            out.append((node.name, _segment(text, node, lines)))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target])
            seg = _segment(text, node, lines)
            for t in targets:
                if isinstance(t, ast.Name):
                    out.append((t.id, seg))
    return out


def free_names(code, provided=()):
    """Names a piece of code LOADS but never binds, minus builtins and
    `provided`. Read off the AST on purpose: a regex counts `if` and `and` out
    of the prose and reports them as free names (measured, v894).
    """
    tree = ast.parse(code)
    bound, loaded = set(), set()

    def walk(node, local):
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                if isinstance(child.ctx, (ast.Store, ast.Del)):
                    local.add(child.id)
                else:
                    loaded.add(child.id)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                    ast.ClassDef)):
                local.add(child.name)
            elif isinstance(child, ast.arg):
                local.add(child.arg)
            elif isinstance(child, ast.alias):
                local.add((child.asname or child.name).split(".")[0])
            elif isinstance(child, ast.ExceptHandler) and child.name:
                local.add(child.name)

    walk(tree, bound)
    return sorted(n for n in loaded
                  if n not in bound and n not in provided
                  and not hasattr(builtins, n))


def close_over(text, wanted, provided=()):
    """Lift `wanted` out of `text` together with everything they need.

    Returns (source, missing). `source` carries the requested segments plus,
    transitively, every top-level name of the module they reference, in source
    order. `missing` names what neither the module nor `provided` supplies --
    empty means the lift is closed and will not raise NameError.

    Callers report `missing` themselves, through their own failure channel: a
    first draft of this module also offered a `lift()` that raised on a short
    lift, and the v896 mutation round showed nobody called it. A convenience
    with no caller is a second way to do the same thing, kept in step by hope.
    """
    segments = top_level(text)
    by_name = {}
    order = {}
    for i, (name, seg) in enumerate(segments):
        by_name.setdefault(name, seg)
        order.setdefault(name, i)

    provided = set(provided)
    take, missing, queue = set(), set(), list(wanted)
    while queue:
        name = queue.pop()
        if name in take or name in provided:
            continue
        if name not in by_name:
            missing.add(name)
            continue
        take.add(name)
        for dep in free_names(by_name[name], provided):
            if dep not in take:
                queue.append(dep)

    picked = sorted(take, key=lambda n: order[n])
    seen, chunks = set(), []
    for name in picked:
        seg = by_name[name]
        if seg in seen:          # one statement can bind several names
            continue
        seen.add(seg)
        chunks.append(seg)
    return "\n\n\n".join(chunks) + "\n", sorted(missing)
