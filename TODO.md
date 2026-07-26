# ComfyColab TODO

## Next week — Pixal3DMV Advanced face detail

Status: Planned
Source plan: `/Users/philipkavalam/Desktop/pixal3dmv_advanced_face_detail_implementation_plan.md`

Target: Extend `ComfyColabPixal3DMVAdvanced` with optional front, left-profile,
and right-profile face detail while preserving the existing Advanced behavior
when the input is disconnected.

### Guardrails

- [ ] Keep face crops tied to their existing canonical cameras.
- [ ] Do not pass face crops into VGGT-Ω.
- [ ] Do not use enhanced crops for sparse structure or shape generation.
- [ ] Do not modify global DINO tokens.
- [ ] Limit the MVP to texture-stage conditioning and UV/RGB base-color transfer.
- [ ] Keep the first release experimental, 1024-only, and texture-first.
- [ ] Report skipped, weakened, rejected, and applied views truthfully.

### Phase 0 — Baseline

- [ ] Capture the current Advanced node schema.
- [ ] Capture a deterministic no-face worker request and cache key.
- [ ] Capture current metadata and local test results.
- [ ] Capture one live 1024 G4 baseline when the runtime is available.

### Phase 1 — Contracts and node wiring

- [ ] Add `COMFYCOLAB_FACE_CROP_META`.
- [ ] Add `COMFYCOLAB_FACE_DETAIL_SET`.
- [ ] Add `ComfyColabFaceDetailCropSet`.
- [ ] Add `ComfyColabFaceDetailSet`.
- [ ] Add the optional face-detail socket and controls to
      `ComfyColabPixal3DMVAdvanced`.
- [ ] Bump the internal Pixal3D worker protocol to version 2.
- [ ] Preserve the existing no-face execution path.

### Phase 2 — Crop transforms and validation

- [ ] Store deterministic full-image-to-crop and crop-to-full transforms.
- [ ] Support automatic and manual normalized crop boxes.
- [ ] Validate labels, transforms, dimensions, masks, and score ranges.
- [ ] Support strict failure and skip-invalid-view policies.
- [ ] Add validation reports and crop previews.

### Phase 3 — UV/RGB transfer

- [ ] Add crop-aware UV-space projection.
- [ ] Add depth visibility and surface-normal weighting.
- [ ] Implement high-frequency transfer as the default.
- [ ] Implement optional full-RGB transfer with color matching.
- [ ] Preserve alpha, metallic, and roughness channels.
- [ ] Add UV coverage and per-view projection diagnostics.

### Phase 4 — Localized DINO conditioning

- [ ] Add `worker/pixal3d/local_face_detail.py`.
- [ ] Extract and project local DINOv3 features through canonical cameras.
- [ ] Blend only supported projected texture-conditioning keys.
- [ ] Prove global tokens and shape latents remain unchanged.
- [ ] Add explicit NAF-compatible or no-NAF behavior.

### Phase 5 — Hardening

- [ ] Include all semantic face-detail inputs in cache identity.
- [ ] Add cancellation and failure cleanup.
- [ ] Add bounded-memory and sequential-view processing.
- [ ] Add optional debug masks, weights, UV coverage, and texture previews.
- [ ] Add the example face-detail workflow and documentation.

### Verification and release gate

- [ ] Run contract, cache, projection, UV-transfer, protocol, and node-schema tests.
- [ ] Prove the no-face path remains backward-compatible.
- [ ] Prove left/right profile detail does not leak across the mesh.
- [ ] Validate the final textured GLB, materials, UVs, and textures.
- [ ] Run front-only, profiles, full-set, invalid-skip, and strict-invalid cases
      through the cached-base Colab CLI G4 session.
- [ ] Inspect real generated output and runtime logs before packaging or publishing.
- [ ] Keep 1536 and weak shape conditioning behind separate future gates.
