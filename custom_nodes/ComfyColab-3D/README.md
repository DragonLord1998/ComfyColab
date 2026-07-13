# ComfyColab 3D facade nodes

This pack exposes two normal-search ComfyUI V3 nodes:

- **ComfyColab TRELLIS.2 — Image to 3D**
- **ComfyColab UltraShape — Refine Geometry**

The TRELLIS node expands into the pinned modular TRELLIS.2 nodes. The
UltraShape node launches the file-only worker under `worker/ultrashape` with
the cached `trellis2-nodes` interpreter. All adapters are registered as
development-only nodes so the complete upstream TRELLIS suite remains
available without cluttering normal search.

Both public outputs are native string-path-backed `FILE_3D_GLB` values that
connect directly to ComfyUI's Preview 3D and Save GLB nodes. Result cache data
is temporary and defaults to `/content/.comfycolab/cache/3d`; final assets are
published under `/content/ComfyUI/output/3d`.

See the repository [README](../../README.md), the two examples in
[`workflows/`](../../workflows), and [`docs/3d-validation.md`](../../docs/3d-validation.md).
