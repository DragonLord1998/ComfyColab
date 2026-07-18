#!/usr/bin/env python3
"""Persistent process-isolated runner for the pinned official SkinTokens pipeline."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any


READY_PREFIX = "COMFYCOLAB_SKINTOKENS_READY="
PROGRESS_PREFIX = "COMFYCOLAB_SKINTOKENS_PROGRESS="
RESULT_PREFIX = "COMFYCOLAB_SKINTOKENS_RESULT="
PROTOCOL_VERSION = 1
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
NODE_PACK = REPO_ROOT / "custom_nodes" / "ComfyColab-3D"


def _emit(prefix: str, payload: dict[str, Any]) -> None:
    print(prefix + json.dumps(payload, sort_keys=True), flush=True)


def emit_progress(request_id: str, stage: str, current: int, total: int, **details: Any) -> None:
    _emit(
        PROGRESS_PREFIX,
        {"request_id": request_id, "stage": stage, "current": current, "total": total, **details},
    )


def emit_result(**details: Any) -> None:
    _emit(RESULT_PREFIX, details)


def _load_comfycolab_file3d():
    package_name = "comfycolab_skintokens_contract"
    if package_name not in sys.modules:
        specification = importlib.util.spec_from_file_location(
            package_name,
            NODE_PACK / "__init__.py",
            submodule_search_locations=[str(NODE_PACK)],
        )
        if specification is None or specification.loader is None:
            raise RuntimeError("Unable to load the ComfyColab GLB contract")
        package = importlib.util.module_from_spec(specification)
        sys.modules[package_name] = package
        specification.loader.exec_module(package)
    return importlib.import_module(f"{package_name}.file3d")


def _load_official_demo(source_dir: Path) -> ModuleType:
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    path = source_dir / "demo.py"
    if not path.is_file():
        raise FileNotFoundError(f"Pinned SkinTokens demo entrypoint is missing: {path}")
    name = "comfycolab_official_skintokens_demo"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    previous_cwd = Path.cwd()
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Unable to load official SkinTokens demo from {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        os.chdir(source_dir)
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    finally:
        os.chdir(previous_cwd)
    return module


def _validate_request(request: dict[str, Any]) -> None:
    if int(request.get("protocol", -1)) != PROTOCOL_VERSION:
        raise ValueError("Unsupported SkinTokens worker protocol")
    for name in ("input_glb", "output_glb", "metadata_output", "request_id"):
        if not str(request.get(name, "")):
            raise ValueError(f"SkinTokens request omitted {name}")
    input_glb = Path(str(request["input_glb"]))
    if input_glb.suffix.lower() != ".glb" or not input_glb.is_file():
        raise FileNotFoundError(f"SkinTokens input GLB does not exist: {input_glb}")
    if int(request.get("top_k", 0)) < 1:
        raise ValueError("top_k must be positive")
    if not 0.0 < float(request.get("top_p", 0.0)) <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if float(request.get("temperature", 0.0)) <= 0.0:
        raise ValueError("temperature must be positive")
    if int(request.get("num_beams", 0)) < 1:
        raise ValueError("num_beams must be positive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _snapshot_revision(path: Path) -> str:
    try:
        marker = json.loads((path / ".comfycolab-artifact.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Pinned SkinTokens model snapshot marker is invalid: {path}") from error
    revision = str(marker.get("revision", ""))
    if not revision:
        raise RuntimeError(f"Pinned SkinTokens model snapshot marker omitted revision: {path}")
    return revision


def _validate_rig_contract(document: dict[str, Any]) -> dict[str, int]:
    skins = document.get("skins") or []
    nodes = document.get("nodes") or []
    meshes = document.get("meshes") or []
    if not skins:
        raise ValueError("SkinTokens output GLB does not contain a glTF skin")
    joint_count = 0
    for skin in skins:
        joints = skin.get("joints") or []
        if not joints:
            raise ValueError("SkinTokens output GLB skin has no joints")
        for joint in joints:
            if not isinstance(joint, int) or joint < 0 or joint >= len(nodes):
                raise ValueError("SkinTokens output GLB skin references an invalid joint node")
        joint_count += len(joints)
    skinned_nodes = [
        node for node in nodes if isinstance(node, dict) and isinstance(node.get("skin"), int)
    ]
    if not skinned_nodes:
        raise ValueError("SkinTokens output GLB has no mesh node bound to a skin")
    for node in skinned_nodes:
        mesh_index = node.get("mesh")
        skin_index = node.get("skin")
        if not isinstance(mesh_index, int) or mesh_index < 0 or mesh_index >= len(meshes):
            raise ValueError("SkinTokens output GLB skinned node has no valid mesh")
        if not isinstance(skin_index, int) or skin_index < 0 or skin_index >= len(skins):
            raise ValueError("SkinTokens output GLB skinned node has no valid skin")
    return {"skins": len(skins), "joints": joint_count, "skinned_nodes": len(skinned_nodes)}


class SkinTokensRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.demo: ModuleType | None = None
        self.bpy_process = None
        self.model_load_count = 0

    @staticmethod
    def _ensure_directory_link(link: Path, target: Path) -> None:
        if not target.is_dir():
            raise FileNotFoundError(f"Pinned SkinTokens runtime directory is missing: {target}")
        if link.is_symlink():
            if link.resolve() == target.resolve():
                return
            link.unlink()
        elif link.exists():
            raise RuntimeError(
                f"SkinTokens source contains an unexpected runtime artifact directory: {link}"
            )
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=True)

    def ensure_runtime_artifact_links(self) -> None:
        self._ensure_directory_link(
            self.args.source_dir / "experiments",
            self.args.model_dir / "experiments",
        )
        self._ensure_directory_link(
            self.args.source_dir / "models" / "Qwen3-0.6B",
            self.args.qwen_dir,
        )

    def resolved_revisions(self, request: dict[str, Any]) -> dict[str, str]:
        actual = {
            "source": _git_revision(self.args.source_dir),
            "model": _snapshot_revision(self.args.model_dir),
            "qwen": _snapshot_revision(self.args.qwen_dir),
            "environment": os.environ.get("COMFYCOLAB_SKINTOKENS_ENVIRONMENT_REF", ""),
        }
        requested = request.get("revisions")
        if not isinstance(requested, dict):
            raise RuntimeError("SkinTokens request omitted pinned revision claims")
        for name, actual_value in actual.items():
            if not actual_value or actual_value != str(requested.get(name, "")):
                raise RuntimeError(
                    f"SkinTokens {name} revision mismatch: requested "
                    f"{requested.get(name)!r}, resolved {actual_value!r}"
                )
        return actual

    def ensure_loaded(self, request_id: str) -> ModuleType:
        if self.demo is not None:
            return self.demo
        emit_progress(request_id, "load_model", 0, 2)
        os.environ.setdefault("XFORMERS_IGNORE_FLASH_VERSION_CHECK", "1")
        self.ensure_runtime_artifact_links()
        self.demo = _load_official_demo(self.args.source_dir)
        previous_cwd = Path.cwd()
        try:
            os.chdir(self.args.source_dir)
            self.bpy_process = self.demo.start_bpy_server()
            self.demo.wait_for_bpy_server(timeout=90)
            emit_progress(request_id, "load_model", 1, 2)
            self.demo.load_model(str(self.args.checkpoint), None)
        finally:
            os.chdir(previous_cwd)
        self.model_load_count += 1
        emit_progress(request_id, "load_model", 2, 2, model_load_count=self.model_load_count)
        return self.demo

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        _validate_request(request)
        request_id = str(request["request_id"])
        started = time.monotonic()
        input_glb = Path(str(request["input_glb"])).resolve()
        output = Path(str(request["output_glb"])).resolve()
        metadata_output = Path(str(request["metadata_output"])).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        partial_output = output.with_name(f".{output.stem}.{request_id}.partial.glb")
        partial_metadata = metadata_output.with_name(f".{metadata_output.stem}.{request_id}.partial.json")
        for path in (output, metadata_output, partial_output, partial_metadata):
            path.unlink(missing_ok=True)
        try:
            file3d = _load_comfycolab_file3d()
            input_validation = file3d.validate_glb(input_glb)
            revisions = self.resolved_revisions(request)
            demo = self.ensure_loaded(request_id)
            emit_progress(request_id, "rig", 0, 1)
            previous_cwd = Path.cwd()
            try:
                os.chdir(self.args.source_dir)
                demo.run_rig(
                    [input_glb],
                    int(request["top_k"]),
                    float(request["top_p"]),
                    float(request["temperature"]),
                    float(request["repetition_penalty"]),
                    int(request["num_beams"]),
                    bool(request["use_skeleton"]),
                    bool(request["use_transfer"]),
                    bool(request["use_postprocess"]),
                    [partial_output],
                    str(self.args.checkpoint),
                    None,
                )
            finally:
                os.chdir(previous_cwd)
            document = file3d.validate_glb(
                partial_output,
                require_material=bool(request["preserve_texture"]),
                require_texture=bool(request["preserve_texture"]),
                require_uv=bool(request["preserve_texture"]),
            )
            rig_contract = _validate_rig_contract(document)
            metadata = {
                "schema": "comfycolab-skintokens-worker-result-v1",
                "request_id": request_id,
                "input_glb": str(input_glb),
                "output_glb": str(output),
                "settings": {
                    "preserve_texture": bool(request["preserve_texture"]),
                    "use_transfer": bool(request["use_transfer"]),
                    "use_skeleton": bool(request["use_skeleton"]),
                    "use_postprocess": bool(request["use_postprocess"]),
                    "top_k": int(request["top_k"]),
                    "top_p": float(request["top_p"]),
                    "temperature": float(request["temperature"]),
                    "repetition_penalty": float(request["repetition_penalty"]),
                    "num_beams": int(request["num_beams"]),
                },
                "texture_preservation": {
                    "requested": bool(request["preserve_texture"]),
                    "transfer_enabled": bool(request["use_transfer"]),
                },
                "input_validation": {"meshes": len(input_validation.get("meshes") or [])},
                "rig_contract": rig_contract,
                "revisions": revisions,
                "runtime_seconds": time.monotonic() - started,
                "worker_pid": os.getpid(),
                "model_load_count": self.model_load_count,
                "bytes": partial_output.stat().st_size,
                "sha256": _sha256(partial_output),
            }
            partial_metadata.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(partial_output, output)
            os.replace(partial_metadata, metadata_output)
            emit_progress(request_id, "rig", 1, 1, bytes=output.stat().st_size)
            return metadata
        except BaseException:
            for path in (output, metadata_output, partial_output, partial_metadata):
                path.unlink(missing_ok=True)
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--one-shot", action="store_true")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--qwen-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--request-id", default="")
    parser.add_argument("--input-glb", type=Path)
    parser.add_argument("--output-glb", type=Path)
    parser.add_argument("--metadata-output", type=Path)
    return parser


def _request_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_VERSION,
        "request_id": args.request_id,
        "input_glb": str(args.input_glb),
        "output_glb": str(args.output_glb),
        "metadata_output": str(args.metadata_output),
        "preserve_texture": True,
        "use_transfer": True,
        "use_skeleton": False,
        "use_postprocess": False,
        "top_k": 5,
        "top_p": 0.95,
        "temperature": 1.0,
        "repetition_penalty": 2.0,
        "num_beams": 10,
        "revisions": {},
    }


def _handle(runtime: SkinTokensRuntime, request: dict[str, Any]) -> bool:
    if request.get("command") == "shutdown":
        return False
    request_id = str(request.get("request_id", "unknown"))
    try:
        metadata = runtime.run(request)
        emit_result(
            **{
                **metadata,
                "request_id": request_id,
                "status": "ok",
                "output_glb": str(Path(request["output_glb"]).resolve()),
                "metadata_output": str(Path(request["metadata_output"]).resolve()),
            }
        )
    except BaseException as error:
        emit_result(
            request_id=request_id,
            status="error",
            error=str(error),
            error_type=type(error).__name__,
            traceback="".join(traceback.format_exception(error))[-8000:],
        )
    return True


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime = SkinTokensRuntime(args)
    if args.one_shot:
        return 0 if _handle(runtime, _request_from_args(args)) else 0
    if not args.server:
        raise SystemExit("Use --server or --one-shot")
    _emit(READY_PREFIX, {"protocol": PROTOCOL_VERSION, "pid": os.getpid()})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("SkinTokens protocol requests must be JSON objects")
        except BaseException as error:
            emit_result(
                request_id="unknown",
                status="error",
                error=str(error),
                error_type=type(error).__name__,
            )
            continue
        if not _handle(runtime, request):
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
