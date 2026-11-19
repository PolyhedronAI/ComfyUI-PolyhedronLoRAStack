"""ph_model_path.py -- find a model file again after it moved into a sub folder (v1326). Library, no node.

A workflow stores a model as the name the list offered when it was saved ("model.safetensors"). Sorting the models
folder into sub folders ("Family/model.safetensors") changes that name, and the saved one no longer resolves.

relocate() answers ONE question: the stored name is gone -- is there exactly one file of the same FILE NAME in the
same kind of folder? Then that is the file. It never guesses: no match keeps the stored name (the caller's own error
names it), and two or more matches are refused by name, because picking one would be the silent-wrong-model bug.

The frontend mirror is web/js/ph_model_relocate.js (it moves the widget value on load, so the node shows the file
it will use). This module is what makes a queued API prompt and an unopened workflow behave the same way.
"""

import re

_SEP = re.compile(r"[\\/]+")


def file_name(name):
    """The last path part of a list entry, lower case ('Fam\\\\A.safetensors' -> 'a.safetensors')."""
    return _SEP.split(str(name).strip())[-1].lower()


def _norm(name):
    return "/".join(p for p in _SEP.split(str(name).strip()) if p).lower()


def candidates(names, name):
    """Entries of `names` with the same file name as `name`, one per distinct path, list order kept."""
    want = file_name(name)
    out, seen = [], set()
    for n in names:
        if file_name(n) != want:
            continue
        key = _norm(n)
        if key in seen:
            continue
        seen.add(key)
        out.append(n)
    return out


def _say(text):
    print(text, flush=True)  # seen at once, also when the console is redirected to a file


def relocate(folders, name, who="", folder_paths=None, log=_say):
    """Return the list name to load for a stored `name`.

    folders: the folder types that count as the same kind (("diffusion_models", "unet"), ("vae",) ...).
    Unchanged when the stored name still resolves, or when nothing of that file name exists.
    The one match when exactly one exists. ValueError when several do.
    """
    if folder_paths is None:
        import folder_paths as _fp
        folder_paths = _fp
    if isinstance(folders, str):
        folders = (folders,)
    name = str(name)
    names = []
    for folder in folders:
        try:
            if folder_paths.get_full_path(folder, name):
                return name
        except Exception:
            pass
        try:
            names.extend(folder_paths.get_filename_list(folder))
        except Exception:
            pass
    found = candidates(names, name)
    if not found:
        return name
    if len(found) > 1:
        raise ValueError(
            "[PLS] %s'%s' is not at its saved place and %d files carry that file name: %s -- pick the one you mean."
            % ((who + ": ") if who else "", name, len(found), ", ".join(found)))
    if log:
        log("[PLS] %s'%s' moved -> using '%s'" % ((who + ": ") if who else "", name, found[0]))
    return found[0]
