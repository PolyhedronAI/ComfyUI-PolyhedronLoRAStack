# Third-party code in this package

`rife_arch.py` is **not our code**. It is the IFNet frame-interpolation
network from **Practical-RIFE** by hzwer, taken by way of
**ComfyUI-Frame-Interpolation** by Fannovel16, and copied byte-for-byte on
purpose.

| | |
|---|---|
| Upstream | https://github.com/hzwer/Practical-RIFE |
| Via | https://github.com/Fannovel16/ComfyUI-Frame-Interpolation |
| Related | https://github.com/HolyWu/vs-rife |
| Licence | MIT |

Copying it verbatim is deliberate, not laziness. It is the exact network
that produced our reference run, which makes a byte-identical comparison
against that run a meaningful anchor rather than a hopeful one.

**Nothing in this folder gets "improved".** Every correction this pack makes
to frame interpolation lives outside the engine, in `nodes/ph_interpolate.py`,
where it can be guarded. If you are looking for our work, look there.

The MIT licence permits redistribution and requires attribution; this file is
that attribution. The upstream copyright notice remains with the upstream
projects listed above.
