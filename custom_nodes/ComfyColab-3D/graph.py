from __future__ import annotations

import importlib
from typing import Any

from .cache import trellis_cache_key
from .presets import TrellisSettings


FORBIDDEN_TRELLIS_NODE = "Trellis2ExportGLB"
COMFYUI_REF = "8b099de36acd81acd1afa3b5442951dc847e0a52"
TRELLIS_WRAPPER_REF = "9b878516f2dc2fd873f4f6cceadba403dd12d83e"
TRELLIS_PATCH_ID = "trellis2-strict-1536-birefnet-pin-metrics-v4"
BIREFNET_MODEL_REF = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"


def _upstream_remesh_inputs() -> dict[str, Any]:
    """Pinned TRELLIS workflow parity for generated shape post-processing."""
    return {
        "remesh": "on",
        "remesh.remesh_band": 1.0,
        "remesh.remove_inner_faces": True,
    }


def _builder():
    return importlib.import_module("comfy_execution.graph_utils").GraphBuilder()


def _finish(graph, link):
    io = importlib.import_module("comfy_api.latest").io
    return io.NodeOutput(link, expand=graph.finalize())


def _progress_checkpoint(
    graph,
    value,
    *,
    progress_node_id: str | None,
    completed: int,
    total: int,
    status: str,
    wait_for=None,
):
    if not progress_node_id:
        return value
    inputs = {
        "value": value,
        "progress_node_id": progress_node_id,
        "completed": completed,
        "total": total,
        "status": status,
    }
    if wait_for is not None:
        inputs["wait_for"] = wait_for
    checkpoint = graph.node("ComfyColab3DProgressCheckpoint", **inputs)
    return checkpoint.out(0)


def _early_preview(graph, model_file, preview_target: dict[str, Any] | None):
    if not preview_target:
        return None
    class_type = str(preview_target["class_type"])
    original_inputs = dict(preview_target.get("inputs", {}))
    if class_type == "Preview3DAdvanced":
        allowed = {
            "viewport_state", "model_3d_info", "camera_info", "width", "height",
        }
        inputs = {name: value for name, value in original_inputs.items() if name in allowed}
        inputs["model_3d"] = model_file
    elif class_type == "Preview3D":
        allowed = {"camera_info", "bg_image"}
        inputs = {name: value for name, value in original_inputs.items() if name in allowed}
        inputs["model_file"] = model_file
    else:
        raise ValueError(f"Unsupported 3D preview target: {class_type}")
    preview = graph.node(class_type, **inputs)
    preview.set_override_display_id(str(preview_target["node_id"]))
    return preview


def build_trellis_graph(
    image: Any,
    settings: TrellisSettings,
    *,
    seed: int,
    remove_background: str,
    cache_mode: str,
    cache_key: str | None = None,
    progress_node_id: str | None = None,
    preview_target: dict[str, Any] | None = None,
):
    graph = _builder()
    models = graph.node("LoadTrellis2Models", resolution=settings.resolution)
    if remove_background == "Off":
        mask = graph.node("ComfyColab3DImageOpaqueMask", image=image)
        prepared_image, prepared_mask = image, mask.out(0)
    else:
        background = graph.node("Trellis2RemoveBackground", image=image, low_vram=True)
        prepared_image, prepared_mask = background.out(0), background.out(1)
    conditioning = graph.node(
        "Trellis2GetConditioning",
        model_config=models.out(0),
        image=prepared_image,
        mask=prepared_mask,
        background_color="black",
    )
    shape_conditioning = _progress_checkpoint(
        graph,
        conditioning.out(0),
        progress_node_id=progress_node_id,
        completed=1,
        total=5,
        status="Stage 2/5 - Generating 3D shape...",
    )
    shape = graph.node(
        "Trellis2ImageToShape",
        model_config=models.out(0),
        conditioning=shape_conditioning,
        seed=seed,
        ss_sampling_steps=settings.sampling_steps,
        shape_sampling_steps=settings.sampling_steps,
        max_tokens=settings.max_tokens,
    )
    preview_mesh = _progress_checkpoint(
        graph,
        shape.out(0),
        progress_node_id=progress_node_id,
        completed=2,
        total=5,
        status="Stage 3/5 - Building geometry preview...",
    )
    raw_mesh = graph.node(
        "ComfyColab3DValidateMesh",
        trimesh=preview_mesh,
        stage="TRELLIS raw shape",
        analysis_mode="raw",
    )
    processed = graph.node(
        "Trellis2ProcessMesh",
        trimesh=raw_mesh.out(0),
        target_face_count=settings.target_face_count,
        floater_threshold=0.001,
        weld_vertices=True,
        **_upstream_remesh_inputs(),
    )
    processed_mesh = graph.node(
        "ComfyColab3DValidateMesh",
        trimesh=processed.out(0),
        stage="TRELLIS processed mesh",
    )
    early_file = None
    if preview_target:
        early_file = graph.node("ComfyColab3DNeutralMeshToFile3D", trimesh=processed_mesh.out(0))
    if early_file is not None:
        _early_preview(graph, early_file.out(0), preview_target)
    texture_shape = _progress_checkpoint(
        graph,
        shape.out(1),
        progress_node_id=progress_node_id,
        completed=3,
        total=5,
        status="Stage 4/5 - Geometry preview ready; generating texture...",
        wait_for=early_file.out(0) if early_file is not None else processed_mesh.out(0),
    )
    texture = graph.node(
        "Trellis2ShapeToTexturedMesh",
        model_config=models.out(0),
        conditioning=conditioning.out(0),
        shape_slat=texture_shape,
        subs=shape.out(2),
        seed=seed,
        tex_sampling_steps=settings.sampling_steps,
    )
    texture_voxelgrid = _progress_checkpoint(
        graph,
        texture.out(0),
        progress_node_id=progress_node_id,
        completed=4,
        total=5,
        status="Stage 5/5 - Baking PBR material and final GLB...",
    )
    rasterized = graph.node(
        "Trellis2RasterizePBR",
        trimesh=processed_mesh.out(0),
        voxelgrid=texture_voxelgrid,
        texture_size=settings.texture_size,
        original_mesh=raw_mesh.out(0),
    )
    final_mesh = graph.node(
        "ComfyColab3DValidateMesh",
        trimesh=rasterized.out(0),
        stage="TRELLIS rasterized mesh",
    )
    key = cache_key or trellis_cache_key(
        image,
        settings=settings,
        seed=seed,
        remove_background=remove_background,
        comfyui_ref=COMFYUI_REF,
        trellis_ref=TRELLIS_WRAPPER_REF,
        trellis_patch_id=TRELLIS_PATCH_ID,
        birefnet_ref=BIREFNET_MODEL_REF,
    )
    file_node = graph.node(
        "ComfyColab3DTrimeshToFile3D",
        trimesh=final_mesh.out(0),
        cache_stage="trellis",
        cache_key=key,
        cache_mode=cache_mode,
    )
    final_file = _progress_checkpoint(
        graph,
        file_node.out(0),
        progress_node_id=progress_node_id,
        completed=5,
        total=5,
        status="Complete - 3D model ready",
    )
    finalized = graph.finalize()
    if FORBIDDEN_TRELLIS_NODE in str(finalized):
        raise AssertionError(f"Incompatible node {FORBIDDEN_TRELLIS_NODE} entered the facade graph")
    io = importlib.import_module("comfy_api.latest").io
    return io.NodeOutput(final_file, expand=finalized)


def build_ultrashape_graph(
    model_3d: Any,
    reference_image: Any,
    *,
    detail: str,
    seed: int,
    retexture: bool,
    steps: int,
    num_latents: int,
    octree_resolution: int,
    decode_chunk_size: int,
    target_face_count: int,
    texture_size: int,
    low_vram: str,
    cache_mode: str,
    geometry_cache_key: str | None = None,
):
    graph = _builder()
    background = graph.node("Trellis2RemoveBackground", image=reference_image, low_vram=True)
    worker = graph.node(
        "ComfyColab3DUltraShapeWorker",
        model_3d=model_3d,
        reference_image=background.out(0),
        reference_mask=background.out(1),
        detail=detail,
        seed=seed,
        steps=steps,
        num_latents=num_latents,
        octree_resolution=octree_resolution,
        decode_chunk_size=decode_chunk_size,
        low_vram=low_vram,
        cache_mode=cache_mode,
        geometry_cache_key=geometry_cache_key or "",
    )
    loaded = graph.node(
        "ComfyColab3DGLBToTrellisMesh",
        glb_path=worker.out(0),
        delete_source=cache_mode == "Disable cache",
    )
    processed = graph.node(
        "Trellis2ProcessMesh",
        trimesh=loaded.out(0),
        target_face_count=target_face_count,
        floater_threshold=0.001,
        weld_vertices=True,
        **{
            "remesh": "off",
            "remesh.fill_holes": True,
            "remesh.fill_holes_perimeter": 0.03,
        },
    )
    if not retexture:
        final = graph.node(
            "ComfyColab3DNeutralMeshToFile3D",
            trimesh=processed.out(0),
        )
        return _finish(graph, final.out(0))

    models = graph.node("LoadTrellis2Models", resolution="1024_cascade")
    conditioning = graph.node(
        "Trellis2GetConditioning",
        model_config=models.out(0), image=background.out(0), mask=background.out(1), background_color="black",
    )
    encoded = graph.node(
        "Trellis2EncodeMesh", model_config=models.out(0), mesh=loaded.out(0), resolution=1024,
    )
    encoded_mesh = graph.node("ComfyColab3DEncodedMeshToTrimesh", shape_latent=encoded.out(0))
    texture = graph.node(
        "Trellis2TextureMesh",
        model_config=models.out(0),
        conditioning=conditioning.out(0),
        shape_latent=encoded.out(0),
        seed=seed,
        tex_sampling_steps=12,
    )
    textured_processed = graph.node(
        "Trellis2ProcessMesh",
        trimesh=encoded_mesh.out(0),
        target_face_count=target_face_count,
        floater_threshold=0.001,
        weld_vertices=True,
        **{
            "remesh": "off",
            "remesh.fill_holes": True,
            "remesh.fill_holes_perimeter": 0.03,
        },
    )
    rasterized = graph.node(
        "Trellis2RasterizePBR",
        trimesh=textured_processed.out(0),
        voxelgrid=texture.out(0),
        texture_size=texture_size,
        original_mesh=encoded_mesh.out(0),
    )
    restored = graph.node(
        "ComfyColab3DRestoreMeshTransform", trimesh=rasterized.out(0), transform=loaded.out(1),
    )
    final = graph.node(
        "ComfyColab3DTextureToFile3D",
        trimesh=restored.out(0),
        reference_image=reference_image,
        refined_geometry_digest=loaded.out(2),
        seed=seed,
        target_face_count=target_face_count,
        texture_size=texture_size,
        texture_sampling_steps=12,
        cache_mode=cache_mode,
    )
    return _finish(graph, final.out(0))


def build_ultrashape_cached_geometry_graph(
    glb_path: str,
    *,
    target_face_count: int,
):
    """Postprocess a cached refinement without loading BiRefNet or UltraShape."""

    graph = _builder()
    loaded = graph.node(
        "ComfyColab3DGLBToTrellisMesh",
        glb_path=glb_path,
        delete_source=False,
    )
    processed = graph.node(
        "Trellis2ProcessMesh",
        trimesh=loaded.out(0),
        target_face_count=target_face_count,
        floater_threshold=0.001,
        weld_vertices=True,
        **{
            "remesh": "off",
            "remesh.fill_holes": True,
            "remesh.fill_holes_perimeter": 0.03,
        },
    )
    final = graph.node("ComfyColab3DNeutralMeshToFile3D", trimesh=processed.out(0))
    return _finish(graph, final.out(0))
