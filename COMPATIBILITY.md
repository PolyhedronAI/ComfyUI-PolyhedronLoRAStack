# Kompatibilität — Polyhedron Suite (v257)

Konsolidierter Kompatibilitäts-Vertrag dieses Packs. Faktisch am Code verankert
(siehe `AUDIT_v256_full.md` für die ausführliche Analyse, `AUDIT_v253_full.md`
für die vorausgegangene). Zweck: ein klarer, solider Stand — was wo läuft, was
bei ComfyUI-/kijai-Updates passiert, und wo der eine offene Frontpunkt liegt.

## Laufzeitumgebung
- **Python:** ≥ 3.10 (`requires-python`). Keine 3.10+-Sprachfeatures im Code —
  der Floor ist konservativ, nicht ausgereizt.
- **Abhängigkeiten:** Die geladenen Nodes brauchen **weder Pillow noch
  requests** (graceful degradation; der Civitai-Fetch zur Laufzeit nutzt
  aiohttp, das ComfyUI ohnehin mitbringt). Beide sind ein **optionales Extra**
  (`pip install .[cli]`) nur für das eigenständige CLI `uls_preview_gen.py`,
  das vollständig vom Node-Laden entkoppelt ist. Der ComfyUI-Manager installiert
  sie weiterhin über `requirements.txt`/`install.py`.

## Frontend-Renderer (der entscheidende Punkt)
- Die **Stack-** und **Engine**-UI wird vollständig auf dem **LiteGraph-Canvas**
  gezeichnet (`onDrawForeground` + manuelles Hit-Testing). Unter ComfyUIs neuem
  **Vue-Renderer („Nodes 2.0")** wird `onDrawForeground` nicht aufgerufen → die
  UI erscheint dort **ohne Fehler nicht** (leerer Node).
- **Erkennung:** `uls_compat.js` meldet genau diesen Fall verhaltensbasiert
  (über `node._ulsDrawFired`, nicht über Versions-Strings) — eine einmalige,
  abschaltbare Warnung. Inspizierbar in der Browser-Konsole via
  `window.__POLYHEDRON_COMPAT__`. Abschalten: Settings → Polyhedron.
- **Workaround heute:** In den ComfyUI-Settings auf **LiteGraph-Rendering**
  schalten (Vue/Nodes-2.0 aus), dann erscheint die volle Polyhedron-UI.
- **Renderer-agnostisch (kein Problem):** alle Popups sind **DOM-Overlays**
  (`document.body`); die **Backend-Nodes** (Bridge, Sigma, Frame Inflate, Token
  Counter, Inspector, Model Switch) nutzen Standard-Widgets und laufen unter
  beiden Renderern; die **Persistenz** läuft über ein verstecktes
  `uls_config`/`engine_config`-Widget — ein gespeicherter Workflow lädt auch
  nach einem Renderer-Wechsel korrekt (nur das Zeichnen fehlte ggf.).
- **Offener Frontpunkt (eigene Roadmap):** der volle Umbau des Node-Körpers von
  Canvas-Zeichnen auf `addDOMWidget` (renderer-agnostisch). Bewusst **nicht** in
  einem Rutsch — gestaffeltes Eigenprojekt; die DOM-Overlay-Popups sind die
  Blaupause. Pflicht, sobald ComfyUI ein LiteGraph-Abkündigungs-Datum nennt.

## Bridge ↔ kijai WanVideoWrapper
- Die Bridge patcht **kijai-interne** Funktionen (`load_weights`,
  `set_lora_params`, `remove_lora_from_module`), gefunden per `sys.modules`-Scan
  nach „WanVideoWrapper". Alle Patches sind **idempotent**, marker-basiert
  (`_BRIDGE_SKIP_MARKER`) und werden bei Bedarf erneut versucht.
- **Bei kijai-Updates:** Fehlt ein erwartetes Symbol (z. B. nach einem Rename in
  kijais Code), meldet die Bridge das jetzt mit **einer lauten Ein-Mal-Warnung**
  (`_warn_symbol_missing_once`, seit v254) statt still nichts zu tun. Bei
  merkwürdigem/doppeltem WAN-LoRA-Verhalten zuerst auf ein kijai-Update prüfen.
- **Verbose-Logs:** standardmäßig **aus**; mit `PLS_BRIDGE_VERBOSE=1` (bzw.
  `true`/`yes`/`on`) einschalten.
- **Hand-off-Pfad** (x-Unpack, t→timestep, y-Concat, Return-Shape, Sampler):
  unverändert seit v248; v254 hat nur additive Diagnose danebengelegt.

## Robustheit beim Laden (seit v254)
- Jede Node-Gruppe importiert **unabhängig**. Ein brechender ComfyUI-Core-Wechsel
  in einer Gruppe (z. B. `comfy.lora`, das nur Stack/Engine nutzen) reißt **nicht**
  das ganze Pack mit — die betroffene Gruppe meldet klar und wird übersprungen,
  der Rest registriert weiter.
- Der Tensor-Cache (CONCAT/DARE) ist **thread-sicher** gelockt (zukunftssicher
  gegen parallele Graph-Exekution; beim heutigen Single-Worker unkontestiert).

## API-Wachposten (was bei Updates zuerst bricht)
Niedrige Stabilität (beobachten): Canvas-Render-Pfad (Vue/Nodes-2.0),
kijai-WanVideoWrapper-Interna, `comfy.lora.*`. Hohe Stabilität: `folder_paths.*`,
das versteckte Persistenz-Widget, `api.fetchApi`/`api.apiURL` (Routen sind unter
nacktem **und** `/api`-Pfad registriert).

## Vendored third-party code

`nodes/vfi/rife_arch.py` — the IFNet frame-interpolation network from
**Practical-RIFE** (hzwer), by way of **ComfyUI-Frame-Interpolation**
(Fannovel16). MIT licence, copied verbatim, not modified. Attribution and
reasoning in `nodes/vfi/NOTICE.md`. Used by ⬡ Polyhedron Interpolate.
