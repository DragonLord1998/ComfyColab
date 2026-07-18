# ComfyColab 3D facade nodes

This pack exposes seven normal-search ComfyUI V3 nodes:

- **ComfyColab TRELLIS.2 — Image to 3D**
- **ComfyColab TRELLIS2MV — Multi-View to 3D**
- **ComfyColab UltraShape — Refine Geometry**
- **ComfyColab Pixal3D — Image to 3D**
- **ComfyColab Pixal3DMV (Experimental) — Multi-View to 3D**
- **ComfyColab SkinTokens — Auto Rig 3D**
- **ComfyColab CubePart — Segment 3D Parts**

The TRELLIS node expands into the pinned modular TRELLIS.2 nodes. The
UltraShape node launches the file-only worker under `worker/ultrashape` with
the cached `trellis2-nodes` interpreter. The Pixal3D node launches an isolated
hidden worker for the official pinned `TencentARC/Pixal3D` source revision
`cdbb2bbffbf4e6f298b5f2af3d1d76a8d823d2af`. TRELLIS2MV uses the pinned
community multiview sampler with one-time-per-run spatial weight computation.
Pixal3DMV is an explicitly experimental ReconViaGen-inspired projection-feature
adapter, not official Pixal3D multiview support. SkinTokens and CubePart run in
isolated workers. All graph/worker adapters are development-only, so the
complete upstream TRELLIS suite remains available without cluttering search.

All public outputs are native string-path-backed `FILE_3D_GLB` values that
connect directly to ComfyUI's Preview 3D and Save GLB nodes. Result cache data
is temporary and defaults to `/content/.comfycolab/cache/3d`; final assets are
published under `/content/ComfyUI/output/3d`.

The TRELLIS facade keeps the visible wrapper updated with native ComfyUI stage
text and progress while its hidden expansion runs. If its output is connected
to Preview 3D, the same viewer receives a neutral untextured mesh after the
shape-processing stage, then the final textured GLB when the full graph ends.
Use the facade's expand control when individual upstream nodes need inspection.
The expansion applies the pinned upstream remesh settings and inserts
development-only semantic gates at the raw, processed, and final geometry
boundaries. These gates use intrinsic PCA rank, not a single axis thickness,
so rotated planes are rejected while genuinely rank-3 thin meshes remain
valid. Current result-cache keys are schema-versioned and cached GLBs are
revalidated before reuse.

UltraShape keeps `Fast` at 512 and adds `Conservative` as the public 24-step,
512 default. `Detailed` and `Ultra` retain experimental 1024 behavior for
saved-workflow compatibility. Its worker rejects planar input
before provisioning models and translates an empty adaptive decode into the
actionable `NoDecodableSurface` domain error without retaining partial output.
The live validation runner listens for the facade's native text/progress events
and requires all five transitions, an early geometry-preview event, the final
preview event, and an explicitly reported textured SaveGLB artifact. This is a
release verifier; it does not turn local contract tests into live Colab proof.

The original Pixal3D facade stays single-image-only and retains its exact
`1024 — Stable` / `1536 — Experimental` contract. Pixal3DMV is a separate node
with required front/back/left/right views and an optional top/bottom pair; it
never routes through a contact sheet. `keep_worker_loaded` keeps the isolated
worker warm between requests. Every Pixal3D multiview live G4 gate remains
pending until the validation runner records actual GPU evidence.

SkinTokens returns a rigged GLB containing a skeleton and skin weights. Its
texture-preserving transfer mode is enabled by default. CubePart requires a
non-empty ordered part schema and explicit research-license acceptance before
provisioning; it returns a combined GLB, per-part output directory, and JSON
manifest rather than claiming unlabeled automatic segmentation.

See the repository [README](../../README.md), the examples in
[`workflows/`](../../workflows), and [`docs/3d-validation.md`](../../docs/3d-validation.md).
