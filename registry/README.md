# Registries

`engine.json` pins the ComfyUI engine revision.

`published-packs.json` records every public daughter repository at an immutable
commit and binds that commit to the exact bytes of its pack manifest. It is the
source for multi-repository contract CI and does not make a runtime-readiness
claim.

`official-packs.json` is the user-selectable runtime registry. A daughter pack
must pass clean-lock installation, ComfyUI startup, node discovery, rollback,
and its required accelerator smoke gate before it is promoted from the
published-contract registry into the official registry.

The separation is intentional: a public repository with passing offline tests
is not automatically a runnable Colab pack.
