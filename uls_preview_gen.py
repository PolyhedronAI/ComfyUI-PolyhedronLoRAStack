#!/usr/bin/env python3
"""
ULS Preview & Metadata Fetcher
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Versorgt alle LoRAs im ComfyUI loras-Ordner mit:
  • Preview-Bild  (.preview.png)
  • Preview-Video (.preview.mp4, falls verfügbar)
  • Trigger Words + Metadata (.uls-meta.json)

Quellen in dieser Priorität:
  1. Manuell hinterlegte Dateien (werden NIEMALS überschrieben)
  2. Civitai API (via SHA256-Hash der .safetensors Datei)
  3. Eingebettetes Thumbnail im safetensors-Header
  4. Info-Card aus Metadata (Pillow, als Fallback)

Manuell hinterlegen:
  Lege einfach neben deine .safetensors Datei:
    mein_lora.safetensors
    mein_lora.preview.png      ← eigenes Vorschaubild
    mein_lora.preview.mp4      ← eigenes Vorschau-Video
    mein_lora.uls-meta.json    ← eigene Trigger Words / Notizen

  Format der .uls-meta.json:
    {
      "trigger_words": "mein_trigger, weiterer trigger",
      "description": "Eigene Notiz",
      "base_model": "WAN 2.2"
    }

Verwendung:
  # Alle LoRAs im Ordner verarbeiten
  python uls_preview_gen.py --lora_dir "C:/AI/ComfyUI/models/loras"

  # Nur Civitai-Daten holen (kein Fallback-Card)
  python uls_preview_gen.py --lora_dir "..." --civitai_only

  # Bestehende Previews überschreiben (außer manuellen)
  python uls_preview_gen.py --lora_dir "..." --force

  # Einzelne Datei
  python uls_preview_gen.py --single "models/loras/mein_lora.safetensors"

  # Mit Civitai API-Key (höhere Rate-Limits)
  python uls_preview_gen.py --lora_dir "..." --api_key "dein_key"

Abhängigkeiten:
  pip install Pillow requests
  (optional: ffmpeg im PATH für Video-Thumbnails)
"""

import os
import sys
import json
import struct
import hashlib
import argparse
import time
from pathlib import Path

# ─── Optionale Imports ────────────────────────────────────────────────────

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[ULS] ⚠ Pillow nicht installiert – Info-Cards übersprungen")
    print("      pip install Pillow")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("[ULS] ⚠ requests nicht installiert – Civitai-API nicht verfügbar")
    print("      pip install requests")


# ─── Konstanten ────────────────────────────────────────────────────────────

CIVITAI_BY_HASH = "https://civitai.com/api/v1/model-versions/by-hash/{hash}"
CIVITAI_TIMEOUT = 15
CIVITAI_DELAY   = 1.5   # Sekunden zwischen Requests (Rate-Limit-Schutz)

PREVIEW_IMAGE_EXTS = [".preview.png", ".preview.jpg", ".preview.jpeg", ".png", ".jpg"]
PREVIEW_VIDEO_EXTS = [".preview.mp4", ".preview.gif", ".preview.webm", ".mp4"]
META_EXT           = ".uls-meta.json"

# Manuelle Dateinamen-Endungen die NIEMALS überschrieben werden
MANUAL_MARKERS = [".preview.png", ".preview.jpg", ".preview.jpeg",
                  ".preview.mp4", ".preview.gif", ".preview.webm",
                  ".uls-meta.json"]


# ─── Hilfsfunktionen ──────────────────────────────────────────────────────

def sha256_of_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Berechnet SHA256-Hash einer Datei (chunk-weise für große Dateien)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def read_safetensors_meta(path: str) -> dict:
    """Liest __metadata__ aus dem safetensors-Header."""
    try:
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            if n > 50 * 1024 * 1024:
                return {}
            raw = f.read(n)
            return json.loads(raw.decode("utf-8", errors="replace")).get("__metadata__", {})
    except Exception:
        return {}


def extract_embedded_thumbnail(meta: dict) -> bytes | None:
    """Extrahiert base64-Thumbnail aus safetensors-Metadata (Civitai-Format)."""
    import base64
    for key in ["modelspec.thumbnail", "thumbnail", "preview"]:
        thumb = meta.get(key, "")
        if not thumb:
            continue
        if "," in thumb:
            thumb = thumb.split(",", 1)[1]
        try:
            data = base64.b64decode(thumb)
            if len(data) > 100:  # Muss echtes Bild sein
                return data
        except Exception:
            continue
    return None


def has_manual_file(base: str, ext_list: list) -> str | None:
    """Prüft ob eine manuell hinterlegte Datei existiert. Gibt Pfad zurück."""
    for ext in ext_list:
        p = base + ext
        if os.path.isfile(p):
            return p
    return None


def save_bytes_as_image(data: bytes, out_path: str) -> bool:
    """Speichert Bytes als Bilddatei, konvertiert zu PNG wenn nötig."""
    try:
        if HAS_PIL:
            from io import BytesIO
            img = Image.open(BytesIO(data))
            img.save(out_path, "PNG")
        else:
            with open(out_path, "wb") as f:
                f.write(data)
        return True
    except Exception as e:
        print(f"    [!] Bildspeicherung fehlgeschlagen: {e}")
        return False


def extract_video_thumbnail(video_path: str, out_path: str) -> bool:
    """Extrahiert erstes Frame aus Video via ffmpeg."""
    import subprocess
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-vframes", "1", "-q:v", "2", out_path],
            capture_output=True, timeout=20
        )
        return r.returncode == 0 and os.path.isfile(out_path)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def create_info_card(lora_name: str, meta: dict, uls_meta: dict, out_path: str) -> bool:
    """Erstellt ein Info-Card PNG als Fallback-Preview."""
    if not HAS_PIL:
        return False
    try:
        W, H = 320, 200
        img  = Image.new("RGB", (W, H), (20, 18, 35))
        draw = ImageDraw.Draw(img)

        # Rahmen + Header
        draw.rounded_rectangle([0, 0, W-1, H-1], radius=10, outline=(80, 70, 140), width=2)
        draw.rounded_rectangle([0, 0, W-1, 42], radius=10, fill=(30, 26, 55))
        draw.line([(0, 42), (W, 42)], fill=(60, 50, 100))

        # Titel
        name_display = Path(lora_name).stem[:32]
        draw.text((10, 8),  "⚡ ULS — Kein Preview",  fill=(120, 100, 210))
        draw.text((10, 24), name_display,              fill=(200, 195, 230))

        # Metadata-Zeilen (kombiniert aus safetensors + uls-meta)
        combined = {**meta, **uls_meta}
        rows = [
            ("Base",    combined.get("ss_base_model_version",
                        combined.get("base_model", combined.get("modelspec.architecture", "")))),
            ("Rank",    combined.get("ss_network_dim", combined.get("rank", ""))),
            ("Algo",    combined.get("ss_network_module", "").split(".")[-1]),
        ]
        y = 54
        for label, value in rows:
            if value and str(value) not in ("?", "", "unknown"):
                draw.text((10, y), f"{label}:", fill=(120, 115, 170))
                draw.text((80, y), str(value)[:30], fill=(200, 195, 230))
                y += 18

        # Trigger Words
        tw = uls_meta.get("trigger_words", "") or meta.get("trigger_words", "")
        if isinstance(tw, dict):
            tw = ", ".join(list(tw.keys())[:8])
        if tw:
            draw.text((10, y + 4), "Trigger:", fill=(120, 115, 170))
            y += 20
            # Wörter umbrechen
            words = str(tw)[:120]
            draw.text((10, y), words, fill=(144, 180, 249))

        # Description
        desc = uls_meta.get("description", "") or combined.get("modelspec.description", "")
        if desc:
            draw.text((10, H - 20), str(desc)[:48], fill=(100, 95, 140))

        img.save(out_path, "PNG")
        return True
    except Exception as e:
        print(f"    [!] Info-Card Fehler: {e}")
        return False


# ─── uls-meta.json Handling ───────────────────────────────────────────────

def load_uls_meta(base: str) -> dict:
    """Lädt eine bestehende .uls-meta.json falls vorhanden."""
    path = base + META_EXT
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_uls_meta(base: str, data: dict, force: bool = False) -> bool:
    """
    Schreibt .uls-meta.json.
    Wenn schon eine existiert: nur fehlende Felder ergänzen (manuell = Vorrang).
    """
    path = base + META_EXT
    existing = load_uls_meta(base)

    if existing and not force:
        # Nur neue Felder aus 'data' ergänzen, bestehende NICHT überschreiben
        merged = {**data, **existing}  # existing hat Vorrang
    else:
        merged = data

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"    [!] meta.json Schreiben fehlgeschlagen: {e}")
        return False


# ─── Civitai API ──────────────────────────────────────────────────────────

def query_civitai(file_hash: str, api_key: str = "") -> dict | None:
    """
    Fragt Civitai nach Modell-Daten anhand des SHA256-Hash.
    Gibt None zurück wenn nicht gefunden oder API nicht verfügbar.
    """
    if not HAS_REQUESTS:
        return None

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = CIVITAI_BY_HASH.format(hash=file_hash)
    try:
        r = requests.get(url, headers=headers, timeout=CIVITAI_TIMEOUT)
        if r.status_code == 404:
            return None  # Nicht auf Civitai
        if r.status_code == 429:
            print("    [!] Civitai Rate-Limit — warte 30s...")
            time.sleep(30)
            r = requests.get(url, headers=headers, timeout=CIVITAI_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f"    [!] Civitai-Fehler: {e}")
        return None


def download_file(url: str, out_path: str, label: str = "",
                  max_bytes: int = 32 * 1024 * 1024) -> bool:
    """Lädt eine Datei von URL herunter.
    v267 (audit A-3): derselbe 32-MB-Cap wie die In-App-Route — Content-Length
    vorab geprüft UND mid-stream erzwungen, damit eine chunked/lügende Antwort
    den RAM/Disk nicht fluten kann. Teilfile wird bei Abbruch entfernt."""
    if not HAS_REQUESTS:
        return False
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        clen = r.headers.get("Content-Length")
        if clen and str(clen).isdigit() and int(clen) > max_bytes:
            print(f"    [!] Download übersprungen ({label}): {clen} Bytes > {max_bytes}-Cap")
            return False
        total = 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                total += len(chunk)
                if total > max_bytes:
                    print(f"    [!] Download abgebrochen ({label}): {max_bytes}-Cap mid-stream überschritten")
                    f.close()
                    os.unlink(out_path)
                    return False
                f.write(chunk)
        size_kb = os.path.getsize(out_path) // 1024
        print(f"    ↓ {label} ({size_kb} KB)")
        return True
    except Exception as e:
        print(f"    [!] Download fehlgeschlagen ({label}): {e}")
        if os.path.isfile(out_path):
            os.unlink(out_path)
        return False


def parse_civitai_response(data: dict) -> dict:
    """
    Extrahiert relevante Felder aus der Civitai API-Antwort.
    Gibt ein dict zurück das direkt in .uls-meta.json geschrieben werden kann.
    """
    result = {}

    # Trigger Words
    trigger_words = data.get("trainedWords", [])
    if isinstance(trigger_words, list):
        result["trigger_words"] = ", ".join(trigger_words)
    elif isinstance(trigger_words, str):
        result["trigger_words"] = trigger_words

    # Basis-Modell
    result["base_model"] = data.get("baseModel", "")

    # Name
    model_info = data.get("model", {})
    result["civitai_name"] = model_info.get("name", "")
    result["civitai_type"] = model_info.get("type", "")
    result["civitai_id"]   = data.get("modelId", "")
    result["civitai_version_id"] = data.get("id", "")

    # Beschreibung (aus model.description, HTML-Tags entfernen)
    desc = model_info.get("description", "") or data.get("description", "")
    if desc:
        import re
        desc = re.sub(r"<[^>]+>", "", desc)[:200]
        result["description"] = desc.strip()

    # Preview URLs (Bild + Video)
    images = data.get("images", [])
    result["_preview_images"] = []
    result["_preview_videos"] = []

    for img in images:
        url  = img.get("url", "")
        nsfq = img.get("nsfwLevel", 0)
        # v267 (N-1): Schwelle vereinheitlicht mit der In-App-Route (≤ 2).
        # Civitai-Level: 1=None / 2=Soft / 4=Mature / 8+=X — Level 3 existiert
        # nicht, das alte "> 3" war also effektiv identisch, nur unklar.
        if nsfq > 2:  # NSFW überspringen
            continue
        mtype = img.get("type", "image")
        if mtype == "video":
            result["_preview_videos"].append(url)
        else:
            result["_preview_images"].append(url)

    return result


# ─── Haupt-Verarbeitungslogik ─────────────────────────────────────────────

def process_lora(
    lora_path: str,
    force: bool = False,
    civitai_only: bool = False,
    api_key: str = "",
    civitai_delay: float = CIVITAI_DELAY,
) -> dict:
    """
    Verarbeitet eine einzelne .safetensors Datei.
    Gibt Status-Dict zurück: {name, preview_image, preview_video, meta, source}
    """
    base   = os.path.splitext(lora_path)[0]
    name   = Path(lora_path).name
    status = {"name": name, "source": "none", "preview_image": None,
              "preview_video": None, "trigger_words": ""}

    print(f"\n  📦 {name}")

    # ── 0. Manuell hinterlegte Dateien prüfen ──────────────────────────────
    manual_img = has_manual_file(base, [".preview.png", ".preview.jpg", ".preview.jpeg"])
    manual_vid = has_manual_file(base, [".preview.mp4", ".preview.gif", ".preview.webm"])
    manual_meta = load_uls_meta(base)

    has_manual_img  = manual_img is not None
    has_manual_vid  = manual_vid is not None
    has_manual_meta = bool(manual_meta.get("trigger_words"))

    if has_manual_img:
        print(f"    ✋ Manuelles Bild vorhanden: {Path(manual_img).name} — wird nicht überschrieben")
        status["preview_image"] = manual_img
    if has_manual_vid:
        print(f"    ✋ Manuelles Video vorhanden: {Path(manual_vid).name} — wird nicht überschrieben")
        status["preview_video"] = manual_vid
    if has_manual_meta:
        print(f"    ✋ Manuelle Trigger Words: {manual_meta['trigger_words'][:60]}")
        status["trigger_words"] = manual_meta["trigger_words"]

    needs_image  = not has_manual_img  and (force or not has_manual_file(base, PREVIEW_IMAGE_EXTS))
    needs_video  = not has_manual_vid
    needs_meta   = not has_manual_meta

    if not needs_image and not needs_meta:
        print("    ✓ Alles vorhanden — überspringe")
        status["source"] = "manual"
        return status

    # ── 1. safetensors-Header lesen ────────────────────────────────────────
    sf_meta = read_safetensors_meta(lora_path)

    # ── 2. Civitai API ─────────────────────────────────────────────────────
    civitai_data = None
    if HAS_REQUESTS and (needs_image or needs_meta):
        print("    🔍 SHA256 berechnen...")
        file_hash = sha256_of_file(lora_path)
        print(f"    🌐 Civitai-Abfrage... ({file_hash[:12]}...)")
        time.sleep(civitai_delay)
        civitai_data = query_civitai(file_hash, api_key)

        if civitai_data:
            parsed = parse_civitai_response(civitai_data)
            print(f"    ✓ Gefunden: {parsed.get('civitai_name', '?')}")

            # Trigger Words
            tw = parsed.get("trigger_words", "")
            if tw:
                print(f"    🏷  Trigger: {tw[:80]}")
                status["trigger_words"] = tw

            # Meta speichern (nur neue Felder, manuell hat Vorrang)
            meta_to_save = {k: v for k, v in parsed.items() if not k.startswith("_")}
            save_uls_meta(base, meta_to_save, force=False)

            # Preview-Bild herunterladen
            if needs_image and parsed["_preview_images"]:
                img_url = parsed["_preview_images"][0]
                img_out = base + ".preview.png"
                if download_file(img_url, img_out, "Preview-Bild"):
                    status["preview_image"] = img_out
                    status["source"] = "civitai"
                    needs_image = False

            # Preview-Video herunterladen
            if needs_video and parsed["_preview_videos"]:
                vid_url = parsed["_preview_videos"][0]
                vid_out = base + ".preview.mp4"
                if download_file(vid_url, vid_out, "Preview-Video"):
                    status["preview_video"] = vid_out

        else:
            print("    – Nicht auf Civitai gefunden")

    if civitai_only:
        return status

    # ── 3. Eingebettetes Thumbnail aus safetensors ─────────────────────────
    if needs_image:
        thumb_bytes = extract_embedded_thumbnail(sf_meta)
        if thumb_bytes:
            img_out = base + ".preview.png"
            if save_bytes_as_image(thumb_bytes, img_out):
                print("    🖼  Eingebettetes Thumbnail extrahiert")
                status["preview_image"] = img_out
                status["source"] = "embedded"
                needs_image = False

    # ── 4. Video-Thumbnail als Bild ────────────────────────────────────────
    if needs_image:
        existing_vid = has_manual_file(base, PREVIEW_VIDEO_EXTS)
        if existing_vid:
            img_out = base + ".preview.png"
            if extract_video_thumbnail(existing_vid, img_out):
                print("    🎬 Video-Thumbnail extrahiert")
                status["preview_image"] = img_out
                status["source"] = "video_thumb"
                needs_image = False

    # ── 5. Info-Card als letzter Fallback ──────────────────────────────────
    if needs_image:
        img_out = base + ".preview.png"
        uls_meta_current = load_uls_meta(base)
        if create_info_card(name, sf_meta, uls_meta_current, img_out):
            print("    📋 Info-Card erstellt")
            status["preview_image"] = img_out
            status["source"] = "info_card"

    # ── 6. Trigger aus safetensors-Header als letzter Meta-Fallback ────────
    if needs_meta and not status["trigger_words"]:
        tw = sf_meta.get("ss_tag_frequency", sf_meta.get("trigger_words", ""))
        if isinstance(tw, dict):
            tw = ", ".join(list(tw.keys())[:15])
        if tw:
            save_uls_meta(base, {"trigger_words": str(tw)})
            status["trigger_words"] = str(tw)
            print(f"    🏷  Trigger aus Header: {str(tw)[:60]}")

    return status


# ─── Verzeichnis-Scan ─────────────────────────────────────────────────────

def scan_directory(
    lora_dir: str,
    force: bool = False,
    civitai_only: bool = False,
    api_key: str = "",
) -> None:
    lora_dir = Path(lora_dir)
    if not lora_dir.is_dir():
        print(f"[!] Verzeichnis nicht gefunden: {lora_dir}")
        sys.exit(1)

    loras = sorted(lora_dir.rglob("*.safetensors"))
    total = len(loras)
    print(f"\n[ULS] Gefunden: {total} LoRA-Dateien in {lora_dir}")
    print(f"      Civitai-API: {'✓' if HAS_REQUESTS else '✗ (pip install requests)'}")
    print(f"      Pillow:      {'✓' if HAS_PIL else '✗ (pip install Pillow)'}")
    print(f"      Force:       {'ja' if force else 'nein'}")
    print()

    results = {"civitai": 0, "embedded": 0, "video_thumb": 0,
               "info_card": 0, "manual": 0, "none": 0, "with_triggers": 0}

    for i, lora_path in enumerate(loras, 1):
        print(f"[{i:3d}/{total}]", end="")
        st = process_lora(str(lora_path), force=force,
                         civitai_only=civitai_only, api_key=api_key)
        src = st.get("source", "none")
        results[src] = results.get(src, 0) + 1
        if st.get("trigger_words"):
            results["with_triggers"] += 1

    # Zusammenfassung
    print(f"""
╔══════════════════════════════════════╗
║   ⚡ ULS Preview-Generator — Fertig  ║
╠══════════════════════════════════════╣
║  Civitai:        {results['civitai']:4d}                  ║
║  Eingebettet:    {results['embedded']:4d}                  ║
║  Video-Thumb:    {results['video_thumb']:4d}                  ║
║  Info-Card:      {results['info_card']:4d}                  ║
║  Manuell (✋):   {results['manual']:4d}                  ║
║  Mit Triggern:   {results['with_triggers']:4d}                  ║
╚══════════════════════════════════════╝
""")


# ─── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ULS Preview & Metadata Fetcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python uls_preview_gen.py --lora_dir "C:/AI/ComfyUI/models/loras"
  python uls_preview_gen.py --lora_dir "C:/AI/ComfyUI/models/loras" --force
  python uls_preview_gen.py --lora_dir "..." --civitai_only
  python uls_preview_gen.py --lora_dir "..." --api_key "mein_civitai_key"
  python uls_preview_gen.py --single "models/loras/mein_lora.safetensors"

Manuell hinterlegen (werden nie überschrieben):
  mein_lora.preview.png       Eigenes Vorschaubild
  mein_lora.preview.mp4       Eigenes Video
  mein_lora.uls-meta.json     {"trigger_words": "mein trigger"}
        """
    )
    parser.add_argument("--lora_dir",     type=str, help="Pfad zum loras-Ordner")
    parser.add_argument("--single",       type=str, help="Einzelne .safetensors Datei")
    parser.add_argument("--force",        action="store_true",
                        help="Bestehende Previews überschreiben (außer manuelle)")
    parser.add_argument("--civitai_only", action="store_true",
                        help="Nur Civitai-API, kein Fallback")
    parser.add_argument("--api_key",      type=str, default="",
                        help="Civitai API-Key (optional, für höhere Rate-Limits)")
    args = parser.parse_args()

    if args.single:
        st = process_lora(args.single, force=args.force,
                         civitai_only=args.civitai_only, api_key=args.api_key)
        print(f"\n  Quelle:   {st['source']}")
        print(f"  Bild:     {st['preview_image'] or '—'}")
        print(f"  Video:    {st['preview_video'] or '—'}")
        print(f"  Trigger:  {st['trigger_words'] or '—'}")
    elif args.lora_dir:
        scan_directory(args.lora_dir, force=args.force,
                      civitai_only=args.civitai_only, api_key=args.api_key)
    else:
        parser.print_help()
