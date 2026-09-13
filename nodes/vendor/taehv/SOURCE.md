# nodes/vendor/taehv -- provenance

- Upstream: https://github.com/madebyollin/taehv (MIT, (c) 2025 Ollin Boer Bohan)
- File: `taehv.py`, fetched 2026-09-05 from `main`, verbatim
  (md5 see the v918 guard). GitHub's commit API was rate-limited in the
  build sandbox that day, so the pin is the file hash, not a commit sha.
- Weights: `safetensors/taeh3.safetensors`, 22,709,752 bytes,
  sha256 4fd022bfcab08772fe0536b17ea1a3bbb5625be11e397868d1c5d891863d4c13
  (pinned in uls_sampler.TAE_REGISTRY, fetched through the pack's one
  download door).
- Why vendored: ComfyUI Core 0.33.4 (rev 7a131a3a, 2026-08-24) has no H3
  branch in comfy/taesd/taehv.py -- `TAEHV(latent_channels=24)` there
  builds patch_size 1 and cannot load taeh3. Core master (2026-09) has it;
  once Frank's Core is on that, `Still · ComfyUI` with `--preview-method
  taesd` also works and this vendor is a belt to Core's braces.
