# v379 -- back on the Registry

## Why this release exists

Every version from **3.72.0 to 3.78.0** was marked `NodeVersionStatusFlagged`
by the Comfy Registry's scanner. A flagged version is listed but **not
distributed**: anyone installing through ComfyUI-Manager or
`comfy node install polyhedron-lora-stack` has been getting **3.70.0** (08.08.)
ever since.

The one high-severity finding was `install.py`:

    install.py:28  subprocess.check_call([sys.executable, "-m", "pip", "install", ...])
    rule: subprocess-pip-install   tags: any-code-execute, system-modification

A runtime `pip install` through `subprocess` is one of the three practices the
Registry standards prohibit outright. The file had been there since the first
public release (04.06.) and 3.70.0 still carries it; the Registry tightened its
scanning between 08.08. and 29.08.

## What changed

- **`install.py` removed.** It installed Pillow and requests -- both already in
  ComfyUI's own requirements, both listed in `requirements.txt` beside it (which
  the Manager reads anyway), neither needed by the nodes themselves, and no code
  called the script. Nothing is lost.
- **`.comfyignore` added**, excluding `tests/` and `tools/` from the Registry
  package. The suite runs node.js harnesses and lifts functions out of source
  with `exec()` -- legitimate in a test, ~100 files of `subprocess`/`exec` a
  ComfyUI user never needs. They stay in this repository for anyone who wants
  to verify the pack. What the Registry package still contains are six
  deliberate, shell-free calls: `open`/`xdg-open` behind "open folder" and
  `ffmpeg` in the optional `uls_preview_gen.py` CLI.
- **Version triple made consistent.** The startup banner still read
  `Polyhedron Suite  v376` through 3.77 and 3.78; it now reads `v379`, matching
  `pyproject` 3.79.0 and `PLUGIN_VERSION` v379. The three guards that pin the
  triple (v348, v351, v352) had been red since v377 and are pulled along.

No node, widget or behaviour changed.
