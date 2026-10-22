# Third-party code in nodes/ph_h3_quality.py (v1048)

| What | From | Licence | Changes |
|---|---|---|---|
| The 3D H3 latent upscaler network, its weight-format detection, the H3 latent mean/std | [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) | MIT, Copyright (c) 2026 LBH-123-AI (LBH-123-AI_..._LICENSE_MIT.txt) | unused attention block and einops removed; fp32 only; the H3 AV latent split around it |
| The fp32 rule, the AV handling, the Semantic Bridge math | [AIMixer/ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director) | Apache-2.0 (AIMixer_..._LICENSE_Apache-2.0.txt) | re-written as stand-alone nodes; no Director runtime |

The Semantic Bridge formula is from speach1sdef178/MiniMax-H3-Semantic-Bridge; no
weights are shipped (models/semantic_bridge, downloaded by the user). No upscaler
weights are shipped either (models/latent_upscale_models).
