from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


MESHFLOW_SOURCE_REPO = "https://github.com/facebookresearch/meshflow.git"
MESHFLOW_SOURCE_REF = "55f56f60e1bbf98d1c1991670ac998094d5f59ae"
MESHFLOW_MODEL_REPO = "facebook/meshflow"
MESHFLOW_MODEL_REF = "9249a90e4997e105e533d6f502453fa3b344676f"
MESHFLOW_MODEL_SUBDIR = "meshflow"
MESHFLOW_ARTIFACT_SCHEMA = "comfycolab-meshflow-artifacts-v1"
DINOV3_MODEL_REPO = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
DINOV3_MODEL_REF = "3c276edd87d6f6e569ff0c4400e086807d0f3881"
DINOV3_ARTIFACT_SCHEMA = "comfycolab-meshflow-dinov3-artifacts-v1"


@dataclass(frozen=True)
class MeshFlowArtifacts:
    checkpoint_dir: Path
    dinov3_dir: Path | None = None


def _ensure_dinov3_artifacts(
    root: Path,
    *,
    progress: Callable[[dict], None],
) -> Path:
    destination = root / f"dinov3-{DINOV3_MODEL_REF[:12]}"
    marker = destination / ".comfycolab-artifact.json"
    config = destination / "config.json"
    model = destination / "model.safetensors"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    if (
        payload.get("schema") == DINOV3_ARTIFACT_SCHEMA
        and payload.get("repo") == DINOV3_MODEL_REPO
        and payload.get("revision") == DINOV3_MODEL_REF
        and config.is_file()
        and model.is_file()
        and config.stat().st_size > 0
        and model.stat().st_size > 0
    ):
        return destination

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("huggingface_hub is required to download DINOv3") from error

    destination.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    progress(
        {
            "stage": "meshflow_dinov3",
            "current": 0,
            "total": 1,
            "repo": DINOV3_MODEL_REPO,
        }
    )
    snapshot_download(
        repo_id=DINOV3_MODEL_REPO,
        revision=DINOV3_MODEL_REF,
        local_dir=str(destination),
        allow_patterns=["config.json", "model.safetensors"],
        token=token,
    )
    if not config.is_file() or not model.is_file():
        raise RuntimeError("Pinned DINOv3 conditioning bundle is incomplete")
    marker.write_text(
        json.dumps(
            {
                "schema": DINOV3_ARTIFACT_SCHEMA,
                "repo": DINOV3_MODEL_REPO,
                "revision": DINOV3_MODEL_REF,
                "config_bytes": config.stat().st_size,
                "model_bytes": model.stat().st_size,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    progress({"stage": "meshflow_dinov3", "current": 1, "total": 1})
    return destination


def ensure_meshflow_artifacts(
    root: str | Path,
    *,
    include_dinov3: bool = False,
    progress: Callable[[dict], None] = lambda _event: None,
) -> MeshFlowArtifacts:
    root = Path(root)
    destination = root / f"meshflow-{MESHFLOW_MODEL_REF[:12]}"
    checkpoint_dir = destination / MESHFLOW_MODEL_SUBDIR
    marker = destination / ".comfycolab-artifact.json"
    config = checkpoint_dir / "config.yaml"
    model = checkpoint_dir / "model.pth"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    if (
        payload.get("schema") == MESHFLOW_ARTIFACT_SCHEMA
        and payload.get("repo") == MESHFLOW_MODEL_REPO
        and payload.get("revision") == MESHFLOW_MODEL_REF
        and config.is_file()
        and model.is_file()
        and config.stat().st_size > 0
        and model.stat().st_size > 0
    ):
        dinov3_dir = (
            _ensure_dinov3_artifacts(root, progress=progress)
            if include_dinov3
            else None
        )
        return MeshFlowArtifacts(
            checkpoint_dir=checkpoint_dir,
            dinov3_dir=dinov3_dir,
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("huggingface_hub is required to download MeshFlow") from error

    destination.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    progress(
        {
            "stage": "meshflow_checkpoint",
            "current": 0,
            "total": 1,
            "repo": MESHFLOW_MODEL_REPO,
        }
    )
    try:
        snapshot_download(
            repo_id=MESHFLOW_MODEL_REPO,
            revision=MESHFLOW_MODEL_REF,
            local_dir=str(destination),
            allow_patterns=[
                f"{MESHFLOW_MODEL_SUBDIR}/config.yaml",
                f"{MESHFLOW_MODEL_SUBDIR}/model.pth",
            ],
            token=token,
        )
    except Exception as error:
        raise RuntimeError(
            "MeshFlow checkpoint download failed. Accept the facebook/meshflow "
            "research license on Hugging Face and provide HF_TOKEN."
        ) from error
    if not config.is_file() or not model.is_file():
        raise RuntimeError("Pinned MeshFlow checkpoint bundle is incomplete")
    marker.write_text(
        json.dumps(
            {
                "schema": MESHFLOW_ARTIFACT_SCHEMA,
                "repo": MESHFLOW_MODEL_REPO,
                "revision": MESHFLOW_MODEL_REF,
                "config_bytes": config.stat().st_size,
                "model_bytes": model.stat().st_size,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    progress({"stage": "meshflow_checkpoint", "current": 1, "total": 1})
    dinov3_dir = (
        _ensure_dinov3_artifacts(root, progress=progress)
        if include_dinov3
        else None
    )
    return MeshFlowArtifacts(
        checkpoint_dir=checkpoint_dir,
        dinov3_dir=dinov3_dir,
    )
