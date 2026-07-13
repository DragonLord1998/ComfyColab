# ComfyColab 3D adversarial QA

This matrix records hostile local scenarios separately from the live G4 model
gates. It is intentionally focused on failures that can corrupt output, retain
GPU work, cross a filesystem boundary, or make a cache claim untrustworthy.

## Scenario matrix

| Category | Scenario | Expected result | Local evidence |
| --- | --- | --- | --- |
| GLB input | Truncated or malformed GLB | Reject before worker/model use | `test_glb_validation_and_string_backed_file3d` |
| GLB input | Non-finite position or out-of-range index | Reject with an explicit validation error | `test_glb_validation_rejects_nonfinite_vertices_and_invalid_indices` |
| GLB input | Wrong accessor type, non-triangle mode, or invalid triangle count | Reject before cache or preview use | `test_glb_validation_enforces_triangle_accessor_semantics` |
| GLB output | Textured cache entry lacks material, UV, base-color texture, or embedded image | Treat as corrupt and recompute | deep `validate_glb` flags used by both textured facade caches |
| Cache | Valid unchanged TRELLIS request | Return native File3D without graph expansion or inference | `test_trellis_public_cache_hit_skips_graph_and_inference` |
| Cache | Valid unchanged UltraShape + texture request | Skip worker and texture inference | `test_ultrashape_texture_cache_hit_skips_worker_and_texture_inference` |
| Cache | Valid unchanged geometry-only UltraShape request | Skip BiRefNet and UltraShape; expand only deterministic mesh postprocessing | `test_ultrashape_geometry_cache_hit_skips_birefnet_and_worker_models` |
| Cache | Refresh, disable, and corrupt TRELLIS entries | Recompute; disable performs no result-cache read/write | `test_trellis_refresh_disable_and_corruption_expand_instead_of_hitting_cache` |
| Worker | Process exits zero but omits the machine result sentinel | Fail and remove partial output | `test_worker_rejects_zero_exit_without_machine_result` |
| Worker | Cancellation during a running process | SIGTERM the process group, escalate to SIGKILL on timeout, clean partials | `test_worker_cancellation_terminates_process_group_and_cleans_partials` |
| Filesystem | A dev-only cleanup input points at an arbitrary `refined.glb` | Never remove its parent | `test_temporary_cleanup_cannot_delete_an_arbitrary_parent` |
| Download | Early EOF, ignored Range, bad checksum, corrupted published artifact, or stalled stream | Resume when verifiable; otherwise restart safely; never promote corrupt bytes | `test_ultrashape_downloads.py` and `test_bootstrap.py` |
| Archive | Cache archive contains traversal, escaping hard link, symlink escape, or special file | Reject before extraction | archive safety tests in `test_bootstrap.py` |
| Dependency | Required pinned TRELLIS node ID is unavailable | Raise an actionable refresh error instead of a broken expansion | `test_missing_upstream_nodes_raise_actionable_dependency_error` |
| Resolution | Pinned 1536 cascade exceeds `max_tokens` | Raise with required token count; never decrement resolution | revision-checked patch test and upstream patch application probe |

Prompt-injection testing is not applicable to these two facades: neither
accepts a prompt, URL, shell fragment, repository name, or model identifier.
User-controlled inputs are typed ComfyUI image/File3D values and bounded widget
settings. Supply-chain inputs are immutable source revisions and checksum-pinned
artifacts.

## Local QA cycles

### Cycle 1

- Found that final cache lookup was downstream of inference. Moved cache checks
  to the public facades and split the result cache into `trellis`,
  `ultrashape`, and `texture` stages.
- Found that temporary result directories could remain after cache-disabled
  runs. Added owned-temp cleanup with a strict path boundary.
- Replaced large accessor list allocation in deep GLB validation with streaming
  iteration.
- Added mandatory worker result sentinels and per-step diffusion progress.

Result: the full repository suite passes locally under Python 3.12. Live CUDA,
VRAM-release, Preview3D rendering, and model-quality scenarios remain in
[`3d-validation.md`](3d-validation.md).

### Cycle 2

- Added the required V3 expansion flag and exact flattened DynamicCombo inputs.
- Threaded one seeded NumPy generator through pinned UltraShape surface sampling
  and seeded Torch before voxel conditioning.
- Made `geometry.glb`, `transform.json`, and their checksum record one atomic
  cache unit with rollback on refresh failure.
- Pinned the BiRefNet model and trusted remote code revision and included it in
  TRELLIS cache keys and runtime provenance.
- Added a real SM120 cubvh kernel probe and reran the original TRELLIS and
  GeometryPack CUDA probes after overlay installation.

Result: `scripts/check.sh` passes 100 local tests. Live G4 gates remain separate.
