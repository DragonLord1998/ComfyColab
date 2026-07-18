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

## TripoSplat

- Official source: <https://github.com/VAST-AI-Research/TripoSplat>
- Official model: <https://huggingface.co/VAST-AI/TripoSplat>
- License: MIT
- Native ComfyUI support: ComfyUI v0.23.0+ and the ComfyUI revision pinned by
  ComfyColab

ComfyColab uses the native TripoSplat and Gaussian-splat nodes from pinned
ComfyUI. It does not redistribute TripoSplat source files or weights in this
Git repository. The first uncached run downloads about 3.78 GB of public model
assets from the official Hugging Face repository into the active runtime's
ComfyUI model folders.

## Pixal3D

- Official source: <https://github.com/TencentARC/Pixal3D>
- Pinned source revision: `cdbb2bbffbf4e6f298b5f2af3d1d76a8d823d2af`
- Hugging Face model: <https://huggingface.co/TencentARC/Pixal3D>
- Pinned model revision: `0b31f9160aa400719af409098bff7936a932f726`
- DINOv3 companion model: <https://huggingface.co/camenduru/dinov3-vitl16-pretrain-lvd1689m>
- Pinned DINOv3 revision: `3c276edd87d6f6e569ff0c4400e086807d0f3881`
- MoGe companion model: <https://huggingface.co/Ruicheng/moge-2-vitl>
- Pinned MoGe model revision: `39c4d5e957afe587e04eec59dc2bcc3be5ecd968`
- Pinned NAF source revision: `37f2dfc180f2de53d98bd601109c0da0dd6b0f43`

Pixal3D is installed at runtime in an isolated worker environment. ComfyColab
does not redistribute Pixal3D weights, DINOv3 weights, MoGe weights, NAF
assets, `utils3d`, or NATTEN wheels in this Git repository. The first uncached
runtime may download/build several multi-GB components. Review the upstream
project, model, and package licenses before using Pixal3D commercially or in a
hosted service. The current Hugging Face model metadata also flags access as
disallowed in the EU; ComfyColab does not bypass repository gating or regional
terms, and users must comply with the model repository's current access rules.

The separate experimental Pixal3DMV node is a ComfyColab adapter. It is not an
official TencentARC Pixal3D feature and uses no separately released multiview
weights. Its projection-fusion design is inspired by the methodology in
ReconViaGen:

- Paper: <https://arxiv.org/abs/2510.23306>
- Official source: <https://github.com/GAP-LAB-CUHK-SZ/ReconViaGen>

ReconViaGen's public implementation targets its own VGGT/TRELLIS pipeline; this
notice does not imply compatibility, endorsement, or official Pixal3D support.

## SkinTokens / TokenRig

- Official source: <https://github.com/VAST-AI-Research/SkinTokens>
- Pinned source revision: `273b691d35989d71cd17ff2895fdc735097b92d1`
- Model repository: <https://huggingface.co/VAST-AI/SkinTokens>
- Pinned model revision: `79736cad0fd84de384d5eede659b4ebd24effe33`
- Qwen3-0.6B configuration repository: <https://huggingface.co/Qwen/Qwen3-0.6B>
- Pinned Qwen revision: `c1899de289a04d12100db370d81485cdf75e47ca`
- Source and model license metadata: MIT

ComfyColab downloads the source, checkpoints, and Qwen configuration at runtime
into temporary Colab storage. The upstream runtime also starts Blender for GLB
skin export and requires its transitive CUDA, FlashAttention, and package
licenses.

## CubePart

- Official source: <https://github.com/Roblox/cube/tree/main/cubepart>
- Pinned source revision: `3c6d06ddbef3160a1e1950cb13ab63dd12a61e50`
- Model repository: <https://huggingface.co/Roblox/cubepart>
- Pinned model revision: `28431d124e77040fcaf34c0a71623ff61d35a6c0`
- Code terms: Cube3D Research-Only RAIL-MS repository license
- Model metadata: OpenRAIL, subject to the upstream repository terms

The CubePart node is disabled until the user explicitly enables
`accept_research_license`. ComfyColab does not redistribute its approximately
10 GB checkpoint set and does not convert the research-only terms into a
commercial license. The upstream method is schema-conditioned part
decomposition; ComfyColab does not represent it as unlabeled segmentation.

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
- Pinned wrapper revision: `9b878516f2dc2fd873f4f6cceadba403dd12d83e`
- GeometryPack: <https://github.com/PozzettiAndrea/ComfyUI-GeometryPack>
- Microsoft TRELLIS.2 model: <https://huggingface.co/microsoft/TRELLIS.2-4B>

The wrapper and model licenses, conditioning-model terms, and all transitive
CUDA/package licenses remain applicable. ComfyColab preserves upstream node
IDs and contracts and keeps the complete advanced node suites installed.

The public TRELLIS2MV facade calls the wrapper's multiview node. ComfyColab's
revision-checked patch only caches its spatial camera-blend weights once per
sampler run; it does not change the active views or blending equation.

The TRELLIS wrapper's default background-removal path uses
`ZhengPeng7/BiRefNet`. ComfyColab pins both its weights and trusted remote code
to revision `e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4`.
