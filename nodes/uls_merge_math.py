# -*- coding: utf-8 -*-
"""
uls_merge_math.py
═════════════════
Pure-math core of the LoRA merge, factored out of uls_stack_node.py (v348) so it
carries NO ComfyUI import and is therefore unit-testable in isolation (see
tests/test_v348_merge_math.py). These functions were MOVED verbatim — same code,
new home — and re-imported back into uls_stack_node, so behaviour is unchanged.

Contents: convention detection / factor-key collection, DARE density+seed+mask
helpers, deterministic Trim channel selection, and the TIES RESOLVE sign-election
(_resolve_sign_elect). torch is imported lazily inside the functions, exactly as
before, so importing this module stays cheap and dependency-light.
"""
import hashlib

# ── module constants ──────────────────────────────────────────────────────
_TE_KEY_PREFIXES = ("lora_te", "text_encoder", "lora_prior_te")
_LORA_CONVENTIONS = [
    (".lora_up.weight",   ".lora_down.weight"),   # Kohya / SD / SDXL
    (".lora_B.weight",    ".lora_A.weight"),       # WAN / FLUX / HunyuanVideo
]

# ── pure-math helpers (moved verbatim from uls_stack_node.py) ─────────────
def _is_te_base(base: str) -> bool:
    """True if a lora base key targets the text encoder (CLIP) rather than
    the diffusion model."""
    return str(base).startswith(_TE_KEY_PREFIXES)

def _dare_density(n: int) -> float:
    """Auto-scale density with group size: 2 LoRAs → 0.95, 5 → 0.80, 10 → 0.55,
    floor at 0.5. Empirically reasonable; not a hard rule."""
    if n <= 1:
        return 1.0
    return max(0.5, 1.0 - 0.05 * (n - 1))

def _trim_keep_fraction(n: int) -> float:
    """Fraction of rank-channels to KEEP for the deterministic magnitude trim
    (the 'Trim' cleanup switch). Gentle, group-scaled — the more LoRAs are
    stacked concurrently, the more aggressively the weak tail is pruned to keep
    background-noise interference out of the merge:
        2 LoRAs → ~0.95, 10 → ~0.73, 15+ → floor 0.60.
    Empirically reasonable starting point; calibrate against real runs."""
    if n <= 1:
        return 1.0
    return max(0.60, min(0.95, 1.0 - 0.03 * (n - 1)))

def _trim_channel_indices(B_f, A_f, keep_fraction: float):
    """Deterministic counterpart to the DARE channel mask: instead of dropping
    *random* rank-channels, drop the *weakest* ones by contribution magnitude.

    The contribution of rank-channel r is the rank-1 outer product
    B[:, r] ⊗ A[r, :]; its Frobenius norm is ‖B[:, r]‖·‖A[r, :]‖. We keep the
    top `keep_fraction` channels by that norm and drop the rest. Operates purely
    in factor space — no full-delta reconstruction, so it is cheap and the
    output stays low-rank (same hand-off path as plain CONCAT/DARE).

    Returns a sorted LongTensor of channel indices to keep, or None when there
    is nothing to trim (keep all)."""
    import torch
    rank = A_f.shape[0]
    if rank <= 1:
        return None
    keep_count = max(1, int(round(rank * keep_fraction)))
    if keep_count >= rank:
        return None
    # B_f: [out, rank, ...] → channel dim is 1 ; A_f: [rank, in, ...] → dim 0.
    b = B_f.transpose(0, 1).reshape(rank, -1)   # [rank, out·…]
    a = A_f.reshape(rank, -1)                    # [rank, in·…]
    mag = b.norm(dim=1) * a.norm(dim=1)          # [rank]
    keep = torch.topk(mag, keep_count, largest=True).indices
    # Sort so concat order is stable and reproducible run-to-run.
    keep, _ = torch.sort(keep)
    return keep.to(torch.long)

def _dare_seed(names: list, weights: list) -> int:
    """Deterministic seed across processes — must NOT use Python's hash()
    because PYTHONHASHSEED is randomised per process by default."""
    h = hashlib.sha1()
    for name, w in zip(names, weights, strict=True):
        h.update(name.encode("utf-8", errors="replace"))
        h.update(b"|")
        h.update(f"{round(float(w), 4):.4f}".encode("ascii"))
        h.update(b";")
    # 31-bit positive int, fits torch generator
    return int.from_bytes(h.digest()[:4], "big") & 0x7FFFFFFF

def _resolve_pick_device(min_free_gb: float = 3.0) -> str:
    """Pick the device for the RESOLVE math: CUDA when it is available AND has at
    least `min_free_gb` free, otherwise CPU. At merge time the diffusion model is
    not loaded yet, so the GPU is normally idle and CUDA is chosen; the choice is
    therefore stable run-to-run, which keeps a workflow reproducible."""
    import torch
    if not torch.cuda.is_available():
        return "cpu"
    try:
        free, _total = torch.cuda.mem_get_info()
        if free >= int(min_free_gb * (1024 ** 3)):
            return "cuda"
    except Exception:
        pass
    return "cpu"

def _resolve_sign_elect(bs, as_, out_dim: int, in_dim_flat: int,
                        seed: int = 0, device: str = "cpu", use_fp16: bool = False):
    """RESOLVE (v256; GPU path added v259) - TIES sign-election + disjoint merge
    for ONE base layer.

    Each (B_f, A_f) pair is already weight/scale-folded (and, if Trim is on,
    already trimmed). Three steps:
      1. Elect a sign per weight element from the SUM of all per-LoRA deltas.
      2. Disjoint mean: average ONLY the LoRAs whose sign agrees with the elected
         one at that element; the sign-fighting contributions are dropped (this is
         what cuts the "many LoRAs cancel each other out" effect).
      3. Re-pack the resolved full delta as a LOW-RANK LoRA via a truncated
         randomized SVD (rank = sum of input ranks, i.e. CONCAT footprint), so the
         result rejoins the standard load_lora hand-off completely unchanged.

    v259 - device / dtype:
      * `device` selects where the heavy math runs ("cuda" or "cpu").
      * On CUDA the per-source delta matmuls (B @ A) run in fp16 (tensor cores)
        when `use_fp16`; the sign SUM, the disjoint MEAN and the SVD always run in
        fp32 for stability. The sign election only needs the sign, so the fp16
        matmul error is irrelevant there.
      * The resolved low-rank pair is moved back to CPU before return and every
        device temporary is freed, so VRAM stays flat across all layers.
      * CUDA OOM is intentionally NOT caught here - it propagates so the caller
        can retry the whole merge on CPU.

    v259 - single pass: each source delta is built ONCE and reused for both the
    sign election and the disjoint mean (v258 rebuilt it twice to cap memory). On
    the CPU-fp32 path this is bit-identical to v258 (same ops, same order, same
    seed), so CPU stays a byte-exact fallback for the GPU path. The SVD is seeded
    deterministically (RNG state saved/restored) so the merge is reproducible
    run-to-run on a given device, like the DARE path.

    Returns (B_merged, A_merged) on CPU, reshaped to the layer's factor shapes,
    None on full cancellation, or raises on a shape it cannot represent (caller
    falls back to SEQ) or on CUDA OOM (caller retries on CPU).
    """
    import torch

    tgt = torch.device(device)
    mm_dtype = torch.float16 if (use_fp16 and tgt.type == "cuda") else torch.float32

    n = len(bs)
    ranks = [b.shape[1] for b in bs]
    sum_rank = int(sum(ranks))
    b_trail = list(bs[0].shape[2:])     # up trailing dims (linear: []; conv: [1,1])
    a_trail = list(as_[0].shape[1:])    # down trailing dims ([in] or [in, kh, kw])

    # Build each source delta ONCE, on the target device, in the matmul dtype.
    deltas = []
    for i in range(n):
        B2 = bs[i].reshape(out_dim, ranks[i]).to(tgt, dtype=mm_dtype)
        A2 = as_[i].reshape(ranks[i], in_dim_flat).to(tgt, dtype=mm_dtype)
        deltas.append(B2 @ A2)          # [out, in_flat] in mm_dtype on tgt
        del B2, A2

    # Pass 1 - elected sign from the running sum (fp32 accumulation).
    S = torch.zeros(out_dim, in_dim_flat, dtype=torch.float32, device=tgt)
    for W in deltas:
        S += W.float()
    gamma = torch.sign(S)               # {-1, 0, +1}, fp32
    del S

    # Pass 2 - disjoint mean of the sign-agreeing contributions (fp32).
    num = torch.zeros(out_dim, in_dim_flat, dtype=torch.float32, device=tgt)
    den = torch.zeros(out_dim, in_dim_flat, dtype=torch.float32, device=tgt)
    for W in deltas:
        Wf = W.float()
        agree = (torch.sign(Wf) == gamma) & (gamma != 0)
        num += torch.where(agree, Wf, torch.zeros_like(Wf))
        den += agree.to(torch.float32)
        del Wf, agree
    deltas.clear()
    del gamma
    W_merged = num / den.clamp(min=1.0)
    del num, den

    if (not torch.isfinite(W_merged).all()) or float(W_merged.abs().max()) == 0.0:
        del W_merged
        return None                     # full cancellation -> no patch for this base

    # Re-pack low-rank: rank = sum of input ranks (CONCAT parity), capped by dims.
    min_dim = min(out_dim, in_dim_flat)
    r = max(1, min(sum_rank, min_dim))
    q = min(min_dim, r + 8)             # small oversample for SVD accuracy
    cpu_state = torch.get_rng_state()   # keep the merge reproducible
    # v267 (N-7): manual_seed below reseeds CPU *and all CUDA devices*. Save/
    # restore the CUDA RNG state whenever CUDA is initialised — also on the
    # CPU path of a CUDA machine (the OOM→CPU-retry case) — so the merge
    # never perturbs the global CUDA RNG. Output-identical: state save and
    # restore only, no tensor op changes (test_v259 [A] stays bit-identical).
    cuda_state = (torch.cuda.get_rng_state_all()
                  if (torch.cuda.is_available() and torch.cuda.is_initialized())
                  else None)
    try:
        torch.manual_seed(seed & 0x7FFFFFFF)   # seeds CPU + all CUDA devices
        U, Sv, V = torch.svd_lowrank(W_merged, q=q, niter=4)
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    del W_merged

    U = U[:, :r].contiguous()
    Sv = Sv[:r]
    V = V[:, :r].contiguous()
    B_merged = (U * Sv.unsqueeze(0)).reshape([out_dim, r] + b_trail).contiguous()
    A_merged = V.transpose(0, 1).reshape([r] + a_trail).contiguous()

    # Back to CPU for the standard hand-off; free the device copies.
    B_cpu = B_merged.to("cpu", dtype=torch.float32).contiguous()
    A_cpu = A_merged.to("cpu", dtype=torch.float32).contiguous()
    del B_merged, A_merged, U, Sv, V
    return B_cpu, A_cpu

def _detect_convention(td: dict):
    """Return (up_suffix, down_suffix) of the dominant convention, or None
    if the tensor dict matches no known LoRA layout (e.g. LoHA / LoKr)."""
    best = None
    best_count = 0
    for conv in _LORA_CONVENTIONS:
        up_suffix = conv[0]
        cnt = sum(1 for k in td if k.endswith(up_suffix))
        if cnt > best_count:
            best_count = cnt
            best = conv
    return best

def _collect_factor_keys(td: dict, conv: tuple) -> list:
    """Triples (base, up_key, down_key, alpha_key_or_None) for this LoRA."""
    up_suffix, down_suffix = conv
    triples = []
    for k in td:
        if not k.endswith(up_suffix):
            continue
        base = k[: -len(up_suffix)]
        dk = base + down_suffix
        if dk not in td:
            continue
        ak_found = None
        for ak in [base + ".alpha", base + ".lora_alpha"]:
            if ak in td:
                ak_found = ak
                break
        triples.append((base, k, dk, ak_found))
    return triples

def _has_mid_tensor(td: dict) -> bool:
    """True if the LoRA carries a CP/Tucker 'mid' tensor (LoCon / conv LoRA),
    i.e. a '<base>.lora_mid.weight' key. Our CONCAT/DARE path reconstructs the
    layer delta as up @ down only; a present mid makes the true delta
    up · mid · down, which this path does NOT represent — and convention
    detection still matches on lora_up/lora_B, so without this guard the group
    would NOT fall back. A group containing any such LoRA is routed to SEQ,
    whose native loader applies the mid correctly. Detection is by key suffix
    only (no tensor is read), so it is cheap and never touches a merge result.
    (This is exactly the mid key ComfyUI's own load_lora maps, so SEQ handles
    it natively.)"""
    return any(k.endswith(".lora_mid.weight") for k in td)


def _dare_mask_apply(B_f, density, dare_variant, rng):
    """Bernoulli DARE mask + density rescale, identical to the inline merge path
    (v348 extraction). 'channel' drops whole rank channels; 'element' is classic
    element-wise DARE. No-op when density is outside (0, 1). rank is read from
    B_f.shape[1], which equals the (post-trim) rank the caller would pass."""
    import torch
    if not (0.0 < density < 1.0):
        return B_f
    rank = B_f.shape[1]
    if dare_variant == "channel":
        # mask shape [1, rank, 1, 1, ...] — drops entire rank channels
        mask_1d = torch.bernoulli(torch.full((rank,), density), generator=rng)
        mask_shape = [1, rank] + [1] * (B_f.dim() - 2)
        mask = mask_1d.view(mask_shape)
    else:
        # element-wise (classic DARE)
        mask = torch.bernoulli(torch.full_like(B_f, density), generator=rng)
    return (B_f * mask) / density
