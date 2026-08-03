"""ph_weights (v754) -- ONE folder resolution and ONE provisioner for every
model this pack loads.

Until now `models_dir()`, `_https()`, `_sha256_file()` and `ensure_weights()`
existed three times, once per engine, in nearly identical copies. Three
truths for one thing: every fix had to be made three times, and every new
model made it four. This module is that one truth; the engines keep only
their registry and their loading code.

Two things changed while merging, both measured, not assumed:

1. FOLDER RESOLUTION NOW ASKS ComfyUI. The old copies took
   `folder_paths.models_dir` -- the main root -- and appended their own
   subfolder. A user who moved models to another drive with
   `extra_model_paths.yaml` was therefore NOT found, even though ComfyUI
   knew the path. Resolution now goes through `get_folder_paths()`, which
   returns every configured location, and the folders are registered with
   `add_model_folder_path()` so ComfyUI knows about ours too.

2. READ FROM ALL ROOTS, WRITE TO ONE. A weight file is looked for in every
   configured folder. A download still lands in the primary one, and still
   only when someone explicitly asked for it.

The `ULS_*_HOME` overrides keep working and keep winning: the guards and the
sandbox depend on them, and an explicit environment variable should beat a
discovered path anyway.
"""
import hashlib
import os
import urllib.request

HTTP_TIMEOUT = 60
UA = {"User-Agent": "PolyhedronLoRAStack-provision"}
WEIGHT_EXTS = [".pt", ".pth", ".safetensors", ".bin"]

# folder name under models/ -> environment override. One table, so a new
# model type is one line here instead of a new copy of this file.
FOLDERS = {
    "birefnet": "ULS_BGR_HOME",
    "sam2": "ULS_SAM_HOME",
    "groundingdino": "ULS_DINO_HOME",
    "sam3": "ULS_SAM3_HOME",
    # v836 (audit B2): the RIFE checkpoints now come through this door too.
    "vfi": "ULS_VFI_HOME",
}

_registered = set()


def _fp():
    try:
        import folder_paths
        return folder_paths
    except Exception:
        return None


def register_folders():
    """Tell ComfyUI about our model folders, so `extra_model_paths.yaml` and
    any other configuration applies to them like it does to checkpoints.

    Defensive by design: this runs at import time inside someone else's
    application, and an API that moved must never take the pack down with
    it. Registering is a nicety -- resolution below works without it.
    """
    fp = _fp()
    if fp is None:
        return
    for folder in FOLDERS:
        if folder in _registered:
            continue
        try:
            base = os.path.join(fp.models_dir, folder)
            fp.add_model_folder_path(folder, base)
            _registered.add(folder)
        except Exception:
            pass          # older/newer API, or the name is already taken


def model_dirs(folder):
    """Every configured folder for this model type, primary FIRST.

    Order of authority: the environment override wins outright, then what
    ComfyUI reports, then the historical fallback (models root + subfolder,
    and finally a folder inside the pack for a sandbox with no ComfyUI at
    all).
    """
    env = FOLDERS.get(folder)
    ov = os.environ.get(env) if env else None
    if ov:
        return [ov]
    out = []
    fp = _fp()
    if fp is not None:
        try:
            out = [p for p in (fp.get_folder_paths(folder) or []) if p]
        except Exception:
            out = []
        if not out:
            try:
                base = getattr(fp, "models_dir", None)
                if base:
                    out = [os.path.join(base, folder)]
            except Exception:
                out = []
    if not out:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = [os.path.join(here, "models", folder)]
    return out


def primary_dir(folder):
    """Where a download goes. Reading happens everywhere, writing here."""
    return model_dirs(folder)[0]


def find_file(folder, filename):
    """The first configured folder that actually holds `filename`, or None.
    Subfolders count: people file their models in trees."""
    for root in model_dirs(folder):
        direct = os.path.join(root, filename)
        if os.path.isfile(direct):
            return direct
        try:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                if filename in filenames:
                    return os.path.join(dirpath, filename)
        except OSError:
            continue
    return None


def https(url):
    """Refuse anything that is not https -- a weight file is code that will
    be loaded, and the pin is only worth something over a verified channel."""
    if not isinstance(url, str) or not url.lower().startswith("https://"):
        raise RuntimeError("provision: refusing non-https URL: %r" % (url,))
    return url


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ensure_weights(folder, name, registry, tag="provision", _fetch=None,
                   mismatch_text=None):
    """Return the local path for `name`, downloading + verifying on first
    use. This is the ONLY place in the pack that writes a weight file.

    `_fetch` is the injection point the guards use to bomb the network and
    the download route uses to report progress -- it is deliberately kept.
    `mismatch_text` lets an engine explain a specific sha mismatch (SAM 2.1
    knows the hash of a superseded release, for instance).
    """
    if name not in registry:
        raise RuntimeError("%s: unknown model %r" % (tag, name))
    spec = registry[name]
    found = find_file(folder, spec["file"])
    if found:
        return found
    home = primary_dir(folder)
    os.makedirs(home, exist_ok=True)
    dest = os.path.join(home, spec["file"])
    part = dest + ".part"
    try:
        if _fetch is not None:
            _fetch(spec["url"], part)
        else:
            req = urllib.request.Request(https(spec["url"]), headers=dict(UA))
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp, \
                    open(part, "wb") as out:
                total = 0
                try:
                    total = int(resp.headers.get("Content-Length") or 0)
                except (TypeError, ValueError):
                    total = 0
                got = 0
                next_mark = 0
                print("[PLS] %s: downloading '%s' (%s)%s"
                      % (tag, name, spec["file"],
                         " ~%d MB" % (total >> 20) if total else ""))
                while True:
                    b = resp.read(1 << 20)
                    if not b:
                        break
                    out.write(b)
                    got += len(b)
                    if got >= next_mark:
                        print("[PLS] %s: %d%s MB"
                              % (tag, got >> 20,
                                 "/%d" % (total >> 20) if total else ""))
                        next_mark = got + (32 << 20)
                print("[PLS] %s: download done (%d MB), verifying sha256"
                      % (tag, got >> 20))
        got_hash = sha256_file(part)
        if got_hash != spec["sha256"]:
            extra = ""
            if mismatch_text is not None:
                try:
                    extra = mismatch_text(name, spec, got_hash) or ""
                except Exception:
                    extra = ""
            raise RuntimeError(
                "%s: sha256 mismatch for %s (got %s, want %s)%s"
                % (tag, spec["file"], got_hash, spec["sha256"], extra))
        os.replace(part, dest)
        return dest
    except Exception as e:
        if os.path.exists(part):
            try:
                os.remove(part)
            except OSError:
                pass
        raise RuntimeError(
            "%s failed for '%s': %s\n"
            "Manual fix: download\n  %s\nand place it at\n  %s"
            % (tag, name, e, spec["url"], dest)) from e


register_folders()
