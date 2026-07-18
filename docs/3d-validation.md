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
| TripoSplat model | `VAST-AI/TripoSplat@de3b99ab2627d565a8d5fc40f2db52557b82b974` |
| Pixal3D source | `cdbb2bbffbf4e6f298b5f2af3d1d76a8d823d2af` |
| Pixal3D model | `0b31f9160aa400719af409098bff7936a932f726` |
| Pixal3D nvdiffrast | `NVlabs/nvdiffrast@253ac4fcea7de5f396371124af597e6cc957bfae` |
| VGGT-Ω source | `facebookresearch/vggt-omega@39a0cb8af88554f15ddcb5354cd52bde588fa014` |
| VGGT-Ω model | `facebook/VGGT-Omega@05654241adc2f218dfb089c373a011f8a7040576` (gated) |
| VGGT-Ω fallback | `1kaiser/vggt-omega-jax@a8c3a718e0cf78e9e4c6847229efea793d37f060` |
| VGGT-Ω checkpoint SHA-256 | `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934` |
| Pixal3D worker profile | `g4-linux64-py31213-torch2110-cu128-sm120-pixal3d-v3` |
| SkinTokens source/model | `273b691d35989d71cd17ff2895fdc735097b92d1` / `VAST-AI/SkinTokens@79736cad0fd84de384d5eede659b4ebd24effe33` |
| SkinTokens worker profile | `g4-linux64-py31115-torch270-cu128-bpy4222-skintokens-v2` |
| CubePart source/model | `3c6d06ddbef3160a1e1950cb13ab63dd12a61e50` / `Roblox/cubepart@28431d124e77040fcaf34c0a71623ff61d35a6c0` |
| DINOv2 Large | `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c` |
| cubvh | `757b913bfbf19ed65e3a379d159391a8e29efa0f` |
| BiRefNet | `e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4` |
| G4 base cache | `g4-linux64-py31213-torch2110-cu128-sm120-glibc235-v1` |
| Combined cache | See `cache/3d-g4-v2.json`; `awaiting-build` is not release-ready |

## Local contract gates

- [x] Exactly eight public ComfyColab 3D nodes across the mesh and TripoSplat
      packs; graph and worker adapters are dev-only.
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
- [x] Pixal3D local workflow and prompt construction are single-image-only,
      expose `1024 — Stable` and `1536 — Experimental`, and include no
      `mode` or `num_views` input. This is local contract coverage only.
- [x] TRELLIS2MV and experimental Pixal3DMV expose ordered four-view inputs
      with optional paired top/bottom views. Pixal3DMV serializes real
      view-aligned projection fusion and never uses a contact sheet.
- [x] Advanced Pixal3DMV pins the official VGGT-Ω source/model revisions plus
      a digest-verified public retrieval fallback, preserves exact Pixal
      cameras and global tokens, applies depth/confidence weights only to
      projection features, and labels strict/fallback behavior without
      claiming a trained residual/register adapter.
- [x] SkinTokens uses a measured Python 3.11.15 environment, validates real
      `JOINTS_0`/`WEIGHTS_0` payloads, and applies deterministic bounded retries
      only to malformed autoregressive skeleton/skin sequences. Cancellation
      still cleans partial outputs. CubePart gates provisioning on explicit
      research license acceptance and validates ordered per-part manifests.
- [x] `scripts/check.sh` passes: 385 tests on 2026-07-19 (four optional tests
      skipped in the minimal local environment).

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
| TripoSplat fast 65K | 65,536 Gaussians requested | n/a | pending | pending | pending PLY bytes/digest | pending Gaussian count | PLY / FILE_3D | pending live G4 |
| Pixal3D cold 1024 | 1024 requested | n/a | pending | pending | pending | pending | 2048 requested | pending live G4 |
| Pixal3D object auto 1024 | 1024 requested | n/a | pending | pending | pending | pending | 2048 requested | pending live G4 |
| Pixal3D transparent 1024 | 1024 requested | n/a | pending | pending | pending | pending | 2048 requested | pending live G4 |
| Pixal3D worker reuse 1024 | 1024 requested | n/a | pending | pending | pending | pending | 2048 requested | pending live G4 |
| Pixal3D preview/save GLB reader | 1024 requested | n/a | pending | pending | pending | pending | 2048 requested | pending live G4 |
| Pixal3D 1536 experimental | 1536 requested | n/a | pending | pending | pending | pending | 4096 requested | pending live G4 |
| TRELLIS2MV four-view / FLUX.2 Klein 9B | 512 | n/a | 182.76 s | not captured | 9,023,296 | 149,099 | 1024 | passed live G4; textured, rank 3 |
| Pixal3DMV four-view experimental / FLUX.2 Klein 9B | 1024 | 15,808 | 128.88 s worker / 232.58 s workflow | 57,337,375,616 B | 6,295,492 | 149,162 | 1024 | passed live G4; textured, rank 3 |
| Pixal3DMV Advanced / VGGT-Ω strict | 1024 | 15,416 | 74.86 s worker / 101.25 s workflow | 57,337,113,280 B worker / 63,888,687,104 B workflow | 6,062,468 | 143,940 | 1024 | passed live G4; textured, rank 3; Sim(3) normalized RMS 0.0655 |
| SkinTokens auto-rig | n/a | n/a | 141.39 s | 8,027,897,856 B | 8,972,012 | 149,162 | 2 embedded textures | passed after retry: 1 skin, 20 joints, 131,757 weighted vertices |
| CubePart schema decomposition | n/a | n/a | pending | pending | pending combined/per-part GLBs | pending part count | colored parts | pending live G4 |

The multiview probe used one camera-faithful four-view toy-van set generated
live by FLUX.2 Klein 9B (front, back, left, right). The same four files were
fed unchanged to both reconstruction nodes. The FLUX contact sheet SHA-256 is
`2c54479647c7b13782ef6e9faf9a3fcdb0ddca4f825a338882fec2a9f4199ba8`.
TRELLIS2MV prompt `8035c2d3-3978-4d5d-8ca6-29af1069cdce` produced GLB
`d7face8d0a9c4c6975c7d091a6f252717598eab51297a582d43f34cbe0fd03c4`;
Pixal3DMV prompt `7d5bd370-331c-4581-a586-c0c5d97c8fe5` produced GLB
`1941019ca59d07a9936ab5a73a90f3e1c041aa93f365c647180bed70214aa951`.
Both outputs passed exact volumetric validation with no collapse reasons.

The Advanced facade still prefers the official gated checkpoint. If that
download fails, it may retrieve the byte-identical checkpoint from the pinned
`1kaiser/vggt-omega-jax` mirror. Both sources must resolve to the exact
4,576,706,117-byte file and SHA-256 recorded above. A weighted Pixal3D fallback
is not counted as Advanced-node validation. If the Hub Xet client rejects its
public-token request, the mirror downloader may use the same pinned revision's
immutable direct `resolve` URL, still gated by the exact size and SHA-256.

Strict run `g4-7aa0844849334f72`, prompt
`fc9dbe5b-5378-4cc5-9c7d-434ea2781d95`, used the exact-digest mirror path in an
unauthenticated Colab runtime. It recorded VGGT-Ω depth/confidence inference,
valid sequence-level Sim(3) alignment with normalized RMS
`0.06550437211564838`, exact labeled Pixal camera policy, no register-token
injection, and GLB SHA-256
`ac89840c533a22a4a1af0b27fca571540e45272bb124d5e37ac44a0a40ad00b0`.
The downloaded artifact independently passed material, texture, UV, exact
rank-3, surface-area, and noncollapse validation.

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

SkinTokens passed its repository-native live case on run
`g4-2fe26f88b4b04666` (prompt
`64b6a922-b740-4971-bc2a-9948e7f0a285`). The G4 output SHA-256 is
`04b871e54542480c72f95b9ffcea9d8725f8d370beee23a254bd39d3bf36e8c9`;
the durable validator read the actual `JOINTS_0` and `WEIGHTS_0` buffers and
confirmed one skin, 20 joints, one inverse-bind-matrix accessor, one skinned
primitive, and normalized weights for all 131,757 vertices. It also verified
the preserved embedded textures, native Preview3D/SaveGLB compatibility, and
non-collapsed rank-3 geometry. The worker attested Python 3.11.15, PyTorch
2.7.0+cu128, NumPy 1.26.4, bpy 4.2.22, transformers 4.57.3, diffusers 0.37.1,
and flash-attn 2.8.3.post1 from the active environment marker. The exact-source
G4 run exercised the recovery path: attempts one through three (seeds 550593027
through 550593029) produced incomplete per-joint skin tokens, while attempt four
used the conservative sampling profile with seed 550593030 and completed.
Evidence:
`live-g4:g4-2fe26f88b4b04666:skintokens_auto_rig:4a74198dd1ccadc824ce7b604e3be500b892f12d8ce41dc9ad2738b2a137f779`.

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

TripoSplat live GPU execution is not locally proven. The `triposplat_fast_65k`
case is now independently runnable and must use the public
`ComfyColabTripoSplatImageToGaussianSplat` facade with one image, `Fast — 65K`,
background removal on, sampling preview on, PLY output, native Preview3D, and
the generic SaveGLB File3D saver. Its release evidence must include the PLY SHA-256,
non-zero byte size, binary little-endian 3DGS property validation, Gaussian
count, runtime, peak VRAM, and model revision. The gate remains pending until
that evidence comes from an actual live G4 run.

Pixal3D's experimental four-view projection-fusion path now has one live G4
proof from the FLUX.2 Klein 9B van set. Its separate cold/single-view,
cache-hit, worker-reuse, cancellation, preview-reader, and 1536 gates remain
pending, so the worker cache is not release-ready.

TRELLIS2MV likewise has one genuine four-view textured GLB. Its six-view gate
remains pending. Advanced Pixal3DMV now has a strict verified VGGT-Ω run through
the exact-digest mirror retrieval path; future official-source runs must resolve
to the same checkpoint digest. Schema loading or the explicit weighted
Pixal3D fallback cannot pass that gate. SkinTokens now has separate live G4
proof for a textured rigged GLB with valid skins/joints and normalized vertex
weights. CubePart still needs a combined scene plus ordered per-part GLBs.
Local protocol/schema tests do not satisfy those remaining gates.

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
- [ ] All eight public 3D outputs connect to their native preview/save nodes
- [ ] TripoSplat fast 65K produces a structurally valid binary little-endian
      PLY/3DGS FILE_3D artifact with digest, bytes, Gaussian count, runtime,
      peak VRAM, revision, Preview3D, and save-node proof
- [ ] Pixal3D cold 1024 first run completes from official pinned sources
- [ ] Pixal3D worker reuse with `keep_worker_loaded=true` avoids relaunch
- [ ] Pixal3D cache hit performs no worker inference
- [ ] Pixal3D cancellation leaves no worker process, partial GLB, or retained allocation
- [ ] Pixal3D 1536 experimental completes without silent downgrade
- [x] TRELLIS2MV four-view FLUX.2 Klein 9B run produces a textured GLB
- [x] Pixal3DMV four-view FLUX.2 Klein 9B run produces a textured GLB without a contact sheet
- [ ] TRELLIS2MV six-view run produces a textured GLB
- [x] Advanced Pixal3DMV strict VGGT-Ω run records alignment/depth guidance and produces a textured GLB
- [x] SkinTokens produces a textured rigged GLB with 1 skin, 20 joints, 1
      skinned primitive, and normalized weights for all 131,757 vertices
- [ ] CubePart produces a combined GLB, ordered per-part GLBs, and matching manifest
