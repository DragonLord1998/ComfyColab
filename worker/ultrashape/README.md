# UltraShape worker

This directory is executed with the cached `trellis2-nodes` interpreter. The
ComfyUI process never imports UltraShape, torch, cubvh, or its CUDA extensions.
The boundary consists only of a GLB, PNG, JSON metadata, and newline-delimited
machine-readable progress.

The worker bakes source scene transforms into one canonical mesh, records the
same center/scale normalization used by pinned UltraShape, applies the inverse
to the refined geometry, validates the result, and atomically publishes a GLB
in the original glTF Y-up world space.
