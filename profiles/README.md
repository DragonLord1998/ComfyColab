# Pack profiles

Profiles select pack references; resolved lock files pin installations.

- `core.json` selects no daughter packs.
- `legacy-full.json` will be added only after all five immutable published packs
  pass individual and combined clean-lock Colab gates. Public commits alone are
  not sufficient: the current 3D development manifest still lacks a supported
  generic environment/cache installation path.

`comfycolab start` defaults to the core-only profile.
