# Repository modularization status

## Implemented locally

- Frozen compatibility inventory at `compatibility/baseline-v1.json`.
- Generic `PackRefV1`, `PackManifestV1`, and `ComfyColabLockV1` contracts.
- Deterministic conflict detection and canonical lock hashing.
- Authenticated standard-library stage 0 and generic stage-1 runtime.
- Core-only profile and pinned ComfyUI engine registry.
- Canonical Python CLI, thin shell shim, versioned runtime state, and lock
  digest verification.
- Canonical public-core URL defaults and a managed Python 3.12+ installer.
- Clean lock-owned runtime-root rebuilds for non-reused locks, with stale-node,
  dependency, environment, and pack-state removal.
- Local runtime-capability preflight before any Colab session allocation.
- Pack list/resolve/update/doctor commands.
- Deterministic two-cell notebook rendering from the same stage-0 config.
- Version- and checksum-pinned cloudflared fallback assets for Linux amd64 and
  arm64.
- Enforced post-start node probes and trusted-pack health commands.
- Five independently testable daughter repositories published from sibling Git
  roots.
- Immutable public daughter commits and raw manifest digests recorded in
  `registry/published-packs.json`.
- An explicit `legacy-full` notebook compatibility runtime that verifies the
  four node-bearing daughter commits, links their declared node roots, and
  reuses the previously validated full dependency bootstrap without promoting
  them into the official generic-runtime registry.

## Local repository layout

The local filesystem split is complete. Each sibling is an independent, clean
Git repository whose local `main` matches its public remote:

- `../ComfyColab-Image` -> `929040f97a53785706fa6372546efe23034babee`
- `../ComfyColab-Video` -> `828d5f6543472901f797e028f7cd20750f472e38`
- `../ComfyColab-3D` -> `92a7a2b4b48f3ab144e122b262a406f1ce8c4aba`
- `../ComfyColab-3DGS` -> `788e1fd5b27af5007f6fe3263bed8dcc57a967d0`
- `../ComfyColab-WorldModels` -> public repository `ComfyColab-WM` at
  `96329dd9bb46f2755fbfefbbef40648917911d05`

The local World Model directory keeps its extraction-era name for compatibility
with current sibling-checkout tests; its public repository and display name use
the singular World Model terminology.

## Fresh local verification

- Core `scripts/check.sh`: 360 tests run against exact published daughter
  checkouts (358 pass and 2 optional Pillow cases skip), including validation
  of all five manifests from their sibling checkouts.
- Daughter suites: 18 Image, 15 Video, 140 3D, 27 3DGS, and 4 World Model
  tests pass from their independent repository roots.
- Shell syntax, Python compilation, offline pack doctors, and whitespace checks
  pass.

The 360-test core checkout still includes the preserved legacy domain regression
suite. It proves migration compatibility, not that legacy source is ready to be
deleted.

## Daughter repositories

| Repository | Local contract status | Live/release status |
| --- | --- | --- |
| ComfyColab-Image | 18 tests pass; pinned upstream requirements are declared | Exact-lock Colab install, startup, node discovery, inference, and rollback remain |
| ComfyColab-Video | 15 tests pass; pinned upstream requirements are declared | Exact-lock Colab install, startup, node discovery, inference, and rollback remain |
| ComfyColab-3D | 140 tests pass | Generic environment-TOML installation, cache restore, and live G4 gates remain |
| ComfyColab-3DGS | 27 tests pass | Resolved model-path ownership and live splat generation remain |
| ComfyColab-WM | 4 tests pass, zero capabilities | Intentional World Model contract skeleton; no node or model claim |

These local checks prove source layout, manifest contracts, public-node
inventory, workflows, and offline doctors. They do not prove CUDA environment
installation or model inference.

## Trust and readiness boundaries

- Pack commits and manifests are authenticated before execution. Official pack
  hooks are therefore trusted code.
- The Python audit guard blocks accidental undeclared writes, subprocesses, and
  network use, and hooks receive a credential-minimized environment. It is not
  an OS-level sandbox against hostile native code. Public third-party hook
  execution remains out of scope for manifest API v1.
- Manifest `readiness` declarations are surfaced as `reserved-metadata`; they
  do not claim those pack-specific fields have been produced until a future
  schema defines their value contract.
- Python and apt package resolution is not yet a fully hashed supply-chain lock;
  official runtime promotion remains gated on clean-lock installation evidence.

## Intentional transition boundaries

- `registry/published-packs.json` pins all public daughter commits for contract
  verification; no mutable `main` reference is accepted.
- The official pack registry remains empty until each candidate is runnable,
  discoverable in ComfyUI, accelerator-validated, and rollback-safe.
- `profiles/legacy-full.json` is published only for the explicit notebook
  compatibility runtime. It is not an official generic-runtime promotion and
  excludes World Model because that repository has no nodes.
- Production domain source remains in core until the matching daughter
  pre-release, pack integration, live smoke, and lock rollback are proven.
- The legacy `comfy-env==0.3.89` timeout patch remains in the old bootstrap
  because the exact upstream source artifact was unavailable for trustworthy
  before/after hashes during extraction.
- Existing cache release URLs remain owned by their historical core release;
  new cache generations will be published from ComfyColab-3D.

## Remaining promotion gates

The repositories are public and immutably recorded. Runtime promotion now
requires:

1. run the exact-ref multi-repository CI lane against
   `registry/published-packs.json`;
2. generate and verify exact combined and individual locks;
3. run core-only install/start/stop in a clean Colab runtime;
4. run Image and Video clean-lock startup, node-discovery, representative
   inference, refresh, update, and rollback gates;
5. resolve the 3DGS model-path ownership mismatch and prove cold/cached splat
   generation;
6. implement generic 3D environment-TOML installation and cache restoration,
   then run every required live G4 path;
7. promote only passing packs into `registry/official-packs.json`;
8. replace the `legacy-full` compatibility runtime with the generic pack
   runtime only after the combined profile passes;
9. remove matching legacy source and regression tests from core one pack at a
   time, retaining the old bootstrap until its remaining behavior is covered.
