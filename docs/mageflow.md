# ComfyColab Mage-Flow

ComfyColab exposes four Mage-Flow image facades plus a native empty-latent
utility:

| Workflow | Public node | Purpose |
| --- | --- | --- |
| `workflows/comfycolab_mageflow.json` | `ComfyColabMageFlow` | standard text-to-image |
| `workflows/comfycolab_mageflow_turbo.json` | `ComfyColabMageFlowTurbo` | low-step text-to-image |
| `workflows/comfycolab_mageflow_edit.json` | `ComfyColabMageFlowEdit` | standard image edit |
| `workflows/comfycolab_mageflow_edit_turbo.json` | `ComfyColabMageFlowEditTurbo` | low-step image edit |
| custom sampler graph | `ComfyColabMageFlowEmptyLatent` | 128-channel Mage text-to-image latent |

No `Base` Mage-Flow classes are public ComfyUI nodes. Internal worker and
component helpers remain hidden.

## Exposed model components

Every Mage-Flow facade returns four outputs:

1. `IMAGE` — the facade's complete default Mage-Flow result.
2. `MODEL` — a worker-backed rectified-flow model compatible with ComfyUI
   samplers.
3. `CLIP` — the Mage text encoder interface. Connect it to the standard
   `CLIP Text Encode` node.
4. `VAE` — the Mage-VAE encoder/decoder interface.

To choose your own sampler, connect `MODEL` and the positive/negative
conditioning from `CLIP Text Encode` to `KSampler` or the advanced sampler
nodes. Use `ComfyColab Mage-Flow — Empty Latent` for text-to-image; it creates
the required `[B, 128, H/16, W/16]` latent. Decode the sampled latent through
the exposed `VAE`. The VAE preserves ComfyUI image/latent batches and supports
the standard tiled encode/decode nodes.

For an Edit facade, the exported `MODEL` retains the facade's connected source
image as reference conditioning. Use that model with an empty target latent,
the facade's text encoder, and your chosen sampler. Exactly one reference image
is supported per Edit model instance.

The component objects proxy tensor operations to the same persistent isolated
worker used by the facade. This preserves Mage's pinned dependency environment
while allowing ComfyUI to own sampler selection, CFG, and scheduling.
Direct `CLIP Text Encode` conditioning is supported; numeric conditioning
interpolation/averaging is rejected because the worker needs the original
prompt text.

## Policy and noise contract

This personal-project integration does not include prompt screening, image
screening, refusal placeholder generation, moderation toggles, safety-checker
inputs, or any content-screening configuration.

Gaussian-Shading watermarking is also removed. The graph should use ordinary
seeded Gaussian noise controlled by the public `seed` input. There are no
watermark keys, watermark toggles, Gaussian-Shading tensors, or watermark
configuration fields in the node schemas or bundled workflows.

## Workflow shape

The bundled quick-start text-to-image workflows contain one Mage-Flow facade
connected directly to native `PreviewImage` and `SaveImage` nodes. Advanced
graphs can instead use the facade's component outputs with native sampler
nodes.

The edit workflows add one `LoadImage` node and connect it to the facade's
`image` input before previewing and saving the edited `IMAGE` output.

The example defaults are intentionally simple:

- `seed = 0` for reproducible local smoke runs.
- standard variants use more steps and higher guidance.
- turbo variants use low-step defaults suitable for quick Colab iteration.

Live model quality, VRAM, and latency still require real Colab validation. Local
tests only prove schema, workflow, and graph-contract behavior.
