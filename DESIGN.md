# Design

## Source of truth
- Status: Active
- Last refreshed: 2026-07-15
- Primary product surfaces: Google Colab launcher notebook and the ComfyUI workflow canvas.
- Evidence reviewed: the user-provided running-workflow screenshot; `custom_nodes/ComfyColab-3D/nodes.py`; `custom_nodes/ComfyColab-3D/graph.py`; `workflows/comfycolab_trellis_image_to_3d.json`; `README.md`.

## Brand
- Personality: direct, dependable, technically honest.
- Trust signals: explicit stage names, visible progress, early previews, actionable errors, and no silent quality downgrade.
- Avoid: unexplained spinners, hidden long-running work, false precision, and controls that imply live output before an artifact exists.

## Product goals
- Goals: make long Colab generations visibly active; expose the current 3D stage; show geometry as soon as a valid mesh exists; preserve simple facade nodes.
- Non-goals: replace ComfyUI's queue, build a second frontend, or claim that TRELLIS produces a continuously deforming mesh.
- Success signals: the facade reads as active during expanded execution; users can name the current stage; Preview 3D shows neutral geometry before the textured GLB completes.

## Personas and jobs
- Primary personas: creators running ComfyUI remotely on temporary Colab GPUs.
- User jobs: confirm the job has not stalled, estimate where it is in the pipeline, inspect shape quality early, and cancel bad generations before texturing finishes.
- Key contexts of use: long remote runs, variable network speed, and a compact workflow that hides advanced TRELLIS internals.

## Information architecture
- Primary navigation: the existing ComfyUI canvas, queue panel, and node details panel.
- Core routes/screens: two-cell notebook launcher and ComfyUI graph.
- Content hierarchy: current stage first, progress second, early geometry preview third, advanced internals on demand.

## Design principles
- Show real state: status changes only at completed execution boundaries.
- Preview the earliest valid artifact: show an untextured mesh after shape processing, then replace it with the final textured model.
- Keep complexity collapsible: retain one public facade while allowing ComfyUI's built-in expansion for troubleshooting.
- Tradeoffs: stage progress is intentionally coarse because upstream model loading and sampling do not expose stable cross-node percentages.

## Visual language
- Color: inherit ComfyUI status, progress, and selection colors.
- Typography: inherit ComfyUI node text and monospace terminal output.
- Spacing/layout rhythm: no new layout system; keep the existing compact node.
- Shape/radius/elevation: inherit ComfyUI.
- Motion: use native progress updates; no decorative animation.
- Imagery/iconography: use the native Preview 3D viewer and the source image.

## Components
- Existing components to reuse: expandable V3 facade node, ComfyUI progress text, native progress state, Preview 3D, Save GLB, and queue cancellation.
- New/changed components: internal progress checkpoints and an early neutral-geometry Preview 3D branch.
- Variants and states: preparing, generating shape, building preview, generating texture, baking final model, complete, cached, error, cancelled.
- Token/component ownership: ComfyUI owns all visual tokens and widgets; ComfyColab owns stage names and graph boundaries.

## Accessibility
- Target standard: inherit ComfyUI accessibility behavior and avoid color-only status.
- Keyboard/focus behavior: no custom focus model.
- Contrast/readability: status is plain text in addition to native highlighting.
- Screen-reader semantics: rely on native node text and progress events.
- Reduced motion and sensory considerations: no added animation.

## Responsive behavior
- Supported breakpoints/devices: desktop browsers supported by ComfyUI; Colab proxy behavior is unchanged.
- Layout adaptations: stage text remains inside the facade; Preview 3D may be resized independently.
- Touch/hover differences: no new hover-only information.

## Interaction states
- Loading: show the active stage on the visible facade.
- Empty: Preview 3D shows its normal grid until valid geometry exists.
- Error: preserve ComfyUI's failing-node and error-report behavior.
- Success: show `Complete - 3D model ready` and the textured model.
- Disabled: unchanged.
- Offline/slow network: the last completed stage remains visible; the queue still provides cancel control.

## Content voice
- Tone: short and operational.
- Terminology: `Preparing`, `Generating shape`, `Geometry preview ready`, `Generating texture`, `Baking final model`, `Complete`.
- Microcopy rules: state what is happening now; label the early mesh as untextured; never show an ETA without measured evidence.

## Implementation constraints
- Framework/styling system: ComfyUI V3 Python nodes at the pinned core revision; no custom frontend framework.
- Design-token constraints: use native ComfyUI UI events and widgets.
- Performance constraints: preview generation must reuse the already-produced mesh and must not invoke another model.
- Compatibility constraints: preserve the facade's existing `model_3d` output and saved workflows.
- Test/screenshot expectations: structural graph tests, full repository checks, then a live Colab run for final UI proof.

## Open questions
- [ ] Confirm the exact frontend rendering of progress text and early Preview 3D through the Colab proxy on the next live G4 run / Codex / release evidence.
