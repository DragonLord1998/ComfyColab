# ComfyColab 3D validation record

This record separates local contract tests from live G4 evidence. A local green
suite does not prove model inference, peak VRAM, or output quality.

The machine-readable release gate is [`3d-validation.json`](3d-validation.json).
It deliberately remains `pending` until every gate below has live evidence and
all benchmark metrics are recorded. `scripts/build_3d_cache.py` refuses to mark
the combined environment manifest `ready` unless that JSON record is `passed`,
matches the exact cache profile, sources, and patches, and contains evidence for
every required gate. The ready manifest embeds the record's SHA-256, size, run
ID, completion time, and passed-gate list for auditability.

## Pinned runtime

| Component | Revision/profile |
| --- | --- |
| ComfyUI | `8b099de36acd81acd1afa3b5442951dc847e0a52` |
| ComfyUI-TRELLIS2 | `9b878516f2dc2fd873f4f6cceadba403dd12d83e` |
| GeometryPack | `c67199de05705642258e727fa118f412877b4ebf` |
| UltraShape source | `5e8dcef05df101ab00ab6cd5fdd0ed0c74fbca66` |
| UltraShape model | `5aeb21a7185d39f042d02b2695802f125a6f5159` |
| DINOv2 Large | `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c` |
| cubvh | `757b913bfbf19ed65e3a379d159391a8e29efa0f` |
| BiRefNet | `e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4` |
| G4 base cache | `g4-linux64-py31213-torch2110-cu128-sm120-glibc235-v1` |
| Combined cache | See `cache/3d-g4-v2.json`; `awaiting-build` is not release-ready |

## Local contract gates

- [x] Exactly two public ComfyColab 3D nodes; all adapters are dev-only.
- [x] Import does not load torch, trimesh, NumPy, Pillow, or initialize CUDA.
- [x] TRELLIS facade expands through the pinned modular node IDs and never uses
      `Trellis2ExportGLB`.
- [x] The facade emits all five ordered text/progress transitions and constructs
      an early neutral Preview3D branch before texture generation.
- [x] The live runner's WebSocket verifier rejects missing/reordered stages or
      previews and separately validates the early geometry and final SaveGLB.
- [x] File3D outputs are string-path-backed and validate as GLB before return.
- [x] Strict 1536 raises when its token count reaches the cap and never retries
      at a lower resolution.
- [x] Worker cancellation kills its process group and removes partial files.
- [x] Cache use, refresh, disable, corruption recovery, and atomic writes pass.
- [x] Asymmetric Y-up/Z-up and normalization round trips preserve handedness,
      orientation, position, and scale.
- [x] XY and rotated planes fail the semantic gate; a very thin rank-3 box
      passes it. Cache keys and UltraShape geometry records carry bumped schema
      versions, and legacy planar results are not reusable.
- [x] Pinned TRELLIS processing uses remesh on, band 1, and inner-face removal;
      GLB export flips texture V exactly once.
- [x] UltraShape empty adaptive decoding becomes `NoDecodableSurface` and
      cleans all partial outputs. `Conservative` resolves to 512; `Detailed`
      and `Ultra` retain explicit experimental 1024 behavior.
- [x] `scripts/check.sh` passes: 163 tests on 2026-07-15.

## Live G4 benchmark table

Do not fill a row from requested settings alone. Record the actual shape
resolution and token count emitted by the patched runtime, validate the GLB,
and retain only the metrics/JSON record—not benchmark GLBs—in Git. UltraShape
rows require matching machine-readable resolved-preset and worker settings;
the 512 row sends `octree_resolution=0` so it proves the `Conservative` preset
path rather than a manual 512 override.

| Pipeline | Actual resolution | Tokens | Runtime | Peak VRAM | GLB bytes | Faces | Texture | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TRELLIS fast | 512 | 3,964 | 29.06 s | 4,190,109,696 B | 13,623,892 | 195,999 | 1024 | prior structural pass; semantic rerun pending |
| TRELLIS quality cascade | 1024 | 18,945 | 96.49 s | 5,320,474,624 B | 37,959,968 | 488,148 | 2048 | prior structural pass; semantic rerun pending |
| TRELLIS maximum cascade | incomplete | 50,145 observed at default cap | n/a | n/a | n/a | n/a | n/a | Colab backend disappeared during the genuine 1536 retry; no GLB recorded |
| UltraShape 384 smoke | 384 requested | n/a | pending | pending | pending | pending | geometry only | pending live G4 |
| UltraShape 512 smoke | 512 requested | n/a | pending | pending | pending | pending | geometry only | pending live G4 |
| UltraShape 1024 run 1 | 1024 requested | n/a | pending | pending | pending | pending | geometry only | release gate |
| UltraShape 1024 run 2 | 1024 requested | n/a | pending | pending | pending | pending | geometry only | release gate |

`Conservative` is the new public 24-step 512 tier. `Fast` remains 512, while
`Detailed` and `Ultra` preserve their existing 1024 semantics and remain
experimental until both 1024 rows pass without OOM or empty-surface decoding.
The 384 override and 512 rows also remain live release gates; selecting a 512
default is not recorded as live proof.

The supplied failure used TRELLIS 512 with 41 steps, 25,000 target faces,
50,193 max tokens, background removal on, and cache use. Its fresh regression
matrix must include those exact values plus preset defaults with cache disabled
and refreshed. Each raw, processed, and final artifact must record bounds,
intrinsic rank/singular ratios, connected components, nondegenerate-face ratio,
and surface area. The original dog source image is required for that live gate;
the screenshot alone is not a lossless replacement.

The default 1536 path separately passed the strict no-downgrade gate: this
input required 50,145 tokens with a 49,152 cap, and returned an actionable error
recommending at least 50,146 tokens or manual 1024 selection. It did not retry
at a lower resolution. The later genuine-1536 attempt progressed through the
high-density tiled sparse-convolution path, but the Colab backend became
unavailable before completion, so it is not counted as a passed benchmark.

Also proven on the same live G4 run:

- PyTorch 2.11.0 + CUDA 12.8 on SM120, cubvh's CUDA distance kernel,
  UltraShape imports, TRELLIS surface loading, and bootstrap regression probes.
- The preserved advanced modular TRELLIS workflow previously produced a
  structurally validated textured GLB after its category move; it now needs a
  fresh semantic-geometry rerun.
- TRELLIS facade and advanced-workflow GLBs connected directly to Preview 3D
  and Save 3D.

UltraShape model refinement, the five full chained workflows, cancellation,
cache-hit inference suppression, and the combined-cache build were deliberately
stopped before live release proof. The machine-readable record therefore
remains `pending`, and bootstrap continues to use the rollback TRELLIS cache
plus the UltraShape inference overlay.

The five-stage/dual-preview verifier is locally covered but has not yet been
executed against a new Colab runtime. A future live run must capture its
WebSocket proof before this UI behavior is counted as release evidence.

After deploying this repair, old result keys are intentionally unreachable.
`Use cache` revalidates current entries semantically; `Refresh this node` is the
recommended first diagnostic rerun and `Disable cache` is required for release
evidence.

## Publishing the combined environment cache

After the live table and workflow gates pass, update `3d-validation.json` with
the actual metrics, one non-empty evidence reference per gate, the Colab run ID,
completion timestamp, and `status: passed`. Then build with:

```bash
python scripts/build_3d_cache.py \
  --validation-record docs/3d-validation.json \
  --install-overlay
```

The builder validates the record before installing packages, creating an
archive, or changing the cache manifest. Do not hand-edit the combined manifest
to `ready`; the builder-generated live-validation digest is part of its release
provenance.

## Full workflow gates

- [ ] Hard-surface input
- [ ] Organic input
- [ ] Thin input
- [ ] Holed input
- [ ] Transparent/background-removal input
- [ ] Unchanged rerun proves cache hits and performs no model inference
- [ ] Cancellation leaves no child process, partial GLB, or retained allocation
- [ ] Existing advanced TRELLIS workflow passes the new semantic geometry gate
      (its prior structural load/execute check remains recorded)
- [ ] Both facade outputs connect directly to Preview 3D and Save 3D Model
