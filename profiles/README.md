# Pack profiles

Profiles select pack references; resolved lock files pin installations.

- `core.json` selects no daughter packs.
- `legacy-full.json` selects the four node-bearing daughter packs for the
  explicit notebook-only `--legacy-full` compatibility runtime. It excludes
  World Model because that repository has no nodes. This profile is not an
  official generic-runtime promotion: the current 3D development manifest
  still lacks a supported generic environment/cache installation path.

`comfycolab start` defaults to the core-only profile.
