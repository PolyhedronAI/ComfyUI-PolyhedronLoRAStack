# v371 — the workflow essentials

Thirteen nodes join the published pack, taking it from **20 to 33**. They are
the pieces a real Polyhedron graph is wired *from*, and until now a workflow
built with the suite could not be opened by anyone who installed it from here.

| Node | node_id |
|---|---|
| Polyhedron Load Model | `ULSLoadModel` |
| Polyhedron Load CLIP | `ULSLoadCLIP` |
| Polyhedron Load VAE | `ULSLoadVAE` |
| Polyhedron Load Upscale Model | `ULSLoadUpscaleModel` |
| Polyhedron VAE Codec | `ULSVAE` |
| Polyhedron Seed | `ULSSeed` |
| Polyhedron Int | `ULSInt` |
| Polyhedron Empty Latent | `ULSEmptyLatent` |
| Polyhedron Switch | `ULSAnySwitch` |
| Polyhedron Switch Inverse | `ULSAnySwitchInv` |
| Polyhedron Media Info | `ULSMediaInfo` |
| Polyhedron MiniMax Reference | `ULSMiniMaxReference` |
| Polyhedron Note | `ULSNote` |

## The build principle holds, for the tenth release running

`nodes/uls_routes.py` is still byte-identical to upstream's file
(`bcc4d8c47e500c892619c22d51f7cbc6`). **Not one of the thirteen opens a server
route** — measured before the cut: no `routes.get`/`routes.post` in any of the
nine carrier modules, and no `fetch()` in any of their frontends. So no new
route module was needed, `ph_media_routes.py` and `ph_sampler_routes.py` are
untouched, and the console still reports the same **28 media paths and 6
sampler paths** as v370. That promise is now pinned by
`test_v365_public_build.py`.

## What came along

Nine carrier modules — `ph_basics`, `ph_switch`, `ph_int`, `ph_empty_latent`,
`ph_media_info`, `ph_minimax_ref`, `ph_note`, `ph_vae`, `ph_upscale_loader` —
plus four house modules they pull in: `uls_latent_math`, `uls_noise`,
`ph_te_detect`, `ph_joint_probe`. Four further carriers were already here
(`ph_logmute`, `ph_power_upscale`, `ph_runclock`, `ph_weights`), and the
coupling was verified at symbol level: `ph_upscale_loader` borrows
`_model_card` from the Power Upscale node, `_sized_list`/`_strip_size` come
from `ph_basics`, and every referenced symbol exists here.

Frontend: `ph_basics.js`, `ph_switch.js`, `ph_int.js`, `ph_empty_latent.js`,
`ph_media_info.js`, `ph_note.js`, `ph_seed.js`, and the three modules they
import — `ph_widget_vis.js`, `ph_widget_hydrate.js`, `ph_noise_field.js`.
`ph_palette.js` is taken along as well, so the published nodes wear the same
colours as the internal build.

**A gap caught by the guards, not by me:** my first file list missed
`ph_seed.js`. Seed lives in `ph_basics.py` but has its own frontend, so the
node would have shipped without its noise preview. Three guards fell on the
missing file; the JS import chain was then closed iteratively to a fixed
point, which pulled in two more modules I had also missed.

## The identity gate arrives

`NODE_IDS_baseline_v371.txt` is the side build's **first** node-id baseline —
until now it had the `WIDGET_ORDER` gate but not the identity gate, which is
the one that protects other people's saved workflows. 33 ids pinned, guarded
by `test_v578_node_ids.py`.

`WIDGET_ORDER_baseline_v370.txt` → `v371`: 33 classes (30 static, 3 dynamic),
**11 rows added, none removed, none reordered** — no saved workflow shifts.

## Guards: 86 → 111

Forty-two internal candidates read the new files. **Twenty-nine were adopted
because they ran honestly green here. Not one was weakened.** Thirteen were
not taken, each for a stated reason:

| Guard | why not |
|---|---|
| `test_v686_hydrate_preview` | reads `ph_reference.js` — that node is not in this cut |
| `test_v830_tae_status` | expects the TAE routes in `uls_routes.py`; here they live in `ph_sampler_routes.py` |
| `test_v833_noise_character` | reads content from `uls_routes.py` — the one file the build principle freezes |
| `test_v835_weights_only` | reads `ph_dino_engine.py` |
| `test_v836_rife_door` | reads `ph_bgr_engine.py` |
| `test_v869_warning_panel` | imports `ph_vectorize` |
| `test_v890_audio_stretch` | imports `uls_audio_math` |
| `test_v876_encode_gate` | pins a CLIP Text Encode feature newer than the published node |
| `test_v877_engine_fit` | pins a `uls_node.js` newer than the published one |
| `test_v880_upscale_joint_model`, `test_v889_vae_joint` | pin Power Upscale features newer than the published node |
| `test_v887_interp_audio` | pins an Interpolate feature newer than the published node |
| `test_v888_hide_inactive` | pins a Sampler frontend newer than the published one |

## One declared re-grounding

`test_v538_seed` pinned the exact **spelling** of an import line
(`from .nodes.ph_basics import ULSSeed`). This build imports the four
`ph_basics` nodes in one grouped statement — the same fact written
differently. Pinning spelling instead of the invariant is precisely the wound
`test_v352` was decoupled from in v364, so the check now asserts the
INVARIANT: `ULSSeed` is imported from `ph_basics`, registered, and has its
display name. Mutation-tested — removing `ULSSeed` from the import turns it
red.

## Version

`3.71.0` / banner `Polyhedron Suite  v371` / `PLUGIN_VERSION v371`.
`CHANGELOG_v370.md` moves to `docs/changelog-archive/`.

The documentation PDF is unchanged in this release and still reads v369; the
new nodes get their own documentation pass, as in v367 and v370.
