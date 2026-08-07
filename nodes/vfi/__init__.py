"""Vendored VFI engines.

rife_arch.py is the MIT IFNet from Practical-RIFE (hzwer), by way of
ComfyUI-Frame-Interpolation (Fannovel16) -- copied byte-for-byte, on purpose.
It is the exact network that produced Frank's 129 frames on 2026-07-14, which
makes a byte-identical comparison against that run a meaningful anchor rather
than a hopeful one.

Nothing in here is ours and nothing in here gets "improved". Every correction
this pack makes to frame interpolation lives OUTSIDE the engine, in
nodes/ph_interpolate.py, where it can be guarded.
"""
