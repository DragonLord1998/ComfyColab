# ComfyColab Mage-Flow

ComfyColab exposes Mage-Flow as four public image facades:

| Workflow | Public node | Purpose |
| --- | --- | --- |
| `workflows/comfycolab_mageflow.json` | `ComfyColabMageFlow` | standard text-to-image |
| `workflows/comfycolab_mageflow_turbo.json` | `ComfyColabMageFlowTurbo` | low-step text-to-image |
| `workflows/comfycolab_mageflow_edit.json` | `ComfyColabMageFlowEdit` | standard image edit |
| `workflows/comfycolab_mageflow_edit_turbo.json` | `ComfyColabMageFlowEditTurbo` | low-step image edit |

No `Base` Mage-Flow classes are public ComfyUI nodes. Internal helpers may exist
inside the pack, but the extension surface is intentionally limited to the four
facades above.

## Policy and noise contract

This personal-project integration does not include prompt screening, image
screening, refusal placeholder generation, moderation toggles, safety-checker
inputs, or any content-screening configuration.

Gaussian-Shading watermarking is also removed. The graph should use ordinary
seeded Gaussian noise controlled by the public `seed` input. There are no
watermark keys, watermark toggles, Gaussian-Shading tensors, or watermark
configuration fields in the node schemas or bundled workflows.

## Workflow shape

The text-to-image workflows contain one Mage-Flow facade connected directly to
native `PreviewImage` and `SaveImage` nodes.

The edit workflows add one `LoadImage` node and connect it to the facade's
`image` input before previewing and saving the edited `IMAGE` output.

The example defaults are intentionally simple:

- `seed = 0` for reproducible local smoke runs.
- standard variants use more steps and higher guidance.
- turbo variants use low-step defaults suitable for quick Colab iteration.

Live model quality, VRAM, and latency still require real Colab validation. Local
tests only prove schema, workflow, and graph-contract behavior.
