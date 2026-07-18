# Compatibility baseline

`baseline-v1.json` freezes the public ComfyColab surface at commit
`e618d4fd8aafde1ea4a2278e9984b26bac53f698` before repository extraction.

The inventory is intentionally limited to public node IDs, display names,
categories, workflow filenames, pinned sources, and readiness keys. Existing
pack tests remain the authoritative input/output-schema contracts and move with
their owning packs.

Local test success does not replace a pack-specific live Colab/GPU validation
gate.
