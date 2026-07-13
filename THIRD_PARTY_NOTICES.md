# Third-party 3D notices

ComfyColab downloads and connects third-party projects at runtime. It does not
store their model weights in this Git repository. Review the upstream terms
before enabling the 3D nodes, especially when deploying a hosted service.

## UltraShape 1.0

- Source: <https://github.com/PKU-YuanGroup/UltraShape-1.0>
- Pinned revision: `5e8dcef05df101ab00ab6cd5fdd0ed0c74fbca66`
- Model: <https://huggingface.co/infinith/UltraShape>
- Pinned model revision: `5aeb21a7185d39f042d02b2695802f125a6f5159`

The pinned source repository includes the Tencent Hunyuan 3D 2.1 Community
License Agreement and its acceptable-use policy. That agreement expressly
excludes use in the European Union, United Kingdom, and South Korea from its
defined territory, contains distribution/hosted-service conditions, and has
additional terms for products above its stated monthly-active-user threshold.
The Hugging Face model repository separately declares Apache-2.0 metadata.
These notices are not legal advice; the upstream license files control.

ComfyColab applies a revision-checked inference patch that removes eager imports
of UltraShape training and unused mesh-postprocessing modules and threads a
seeded generator through surface sampling. It does not alter the model
architecture or weights.

## DINOv2 Large

- Model: <https://huggingface.co/facebook/dinov2-large>
- Pinned revision: `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`
- Declared model-repository license: Apache-2.0

Only the configuration, safetensors weights, and image preprocessor files
needed by UltraShape inference are downloaded.

## cubvh

- Source: <https://github.com/ashawkey/cubvh>
- Pinned revision: `757b913bfbf19ed65e3a379d159391a8e29efa0f`
- License: MIT, with an additional upstream NVIDIA notice for bundled code

The CUDA extension is compiled for the G4 runtime's SM120 architecture and is
stored only in the checksum-pinned environment cache release.

## TRELLIS.2 and ComfyUI integrations

- TRELLIS.2 ComfyUI wrapper: <https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2>
- GeometryPack: <https://github.com/PozzettiAndrea/ComfyUI-GeometryPack>
- Microsoft TRELLIS.2 model: <https://huggingface.co/microsoft/TRELLIS.2-4B>

The wrapper and model licenses, conditioning-model terms, and all transitive
CUDA/package licenses remain applicable. ComfyColab preserves upstream node
IDs and contracts and keeps the complete advanced node suites installed.

The TRELLIS wrapper's default background-removal path uses
`ZhengPeng7/BiRefNet`. ComfyColab pins both its weights and trusted remote code
to revision `e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4`.
