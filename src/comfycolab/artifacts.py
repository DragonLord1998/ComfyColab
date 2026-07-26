from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable


@dataclass(frozen=True)
class RemoteArtifact:
    filename: str
    subfolder: str
    kind: str

    @property
    def identity(self) -> str:
        return f"{self.kind}:{self.subfolder}/{self.filename}"


def _safe_artifact(value: dict[str, Any]) -> RemoteArtifact | None:
    filename = value.get("filename")
    subfolder = value.get("subfolder", "")
    kind = value.get("type", "output")
    if not all(isinstance(item, str) for item in (filename, subfolder, kind)):
        return None
    filename_path = PurePosixPath(filename)
    subfolder_path = PurePosixPath(subfolder)
    if (
        not filename
        or filename_path.name != filename
        or filename_path.is_absolute()
        or subfolder_path.is_absolute()
        or any(part in {"", ".", ".."} for part in subfolder_path.parts)
        or kind not in {"output", "temp"}
        or Path(filename).suffix.lower() != ".glb"
    ):
        return None
    return RemoteArtifact(filename, subfolder, kind)


def discover_glb_artifacts(history: Any) -> list[RemoteArtifact]:
    discovered: dict[str, RemoteArtifact] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            artifact = _safe_artifact(value)
            if artifact is not None:
                discovered[artifact.identity] = artifact
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(history)
    return [discovered[key] for key in sorted(discovered)]


def _read_json(url: str, opener: Callable[..., Any]) -> Any:
    with opener(url, timeout=60) as response:
        return json.loads(response.read())


def _download(url: str, destination: Path, opener: Callable[..., Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    try:
        with opener(url, timeout=300) as response, partial.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        if partial.stat().st_size <= 0:
            raise RuntimeError(f"Downloaded artifact is empty: {destination.name}")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def sync_glb_artifacts(
    comfy_url: str,
    output_dir: str | Path,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> list[Path]:
    base = comfy_url.rstrip("/")
    output_dir = Path(output_dir).expanduser()
    state_path = output_dir / ".comfycolab-artifacts.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        state = {"schema": 1, "downloaded": []}
    downloaded = set(state.get("downloaded", []))
    history = _read_json(base + "/history", opener)
    saved: list[Path] = []
    for artifact in discover_glb_artifacts(history):
        if artifact.identity in downloaded:
            continue
        query = urllib.parse.urlencode(
            {
                "filename": artifact.filename,
                "subfolder": artifact.subfolder,
                "type": artifact.kind,
            }
        )
        destination = output_dir
        if artifact.subfolder:
            destination = destination.joinpath(*PurePosixPath(artifact.subfolder).parts)
        destination = destination / artifact.filename
        _download(base + "/view?" + query, destination, opener)
        downloaded.add(artifact.identity)
        saved.append(destination)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {"schema": 1, "downloaded": sorted(downloaded)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return saved


def watch_glb_artifacts(
    comfy_url: str,
    output_dir: str | Path,
    *,
    interval: float = 3.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
    on_saved: Callable[[Path], None] = lambda _path: None,
    on_error: Callable[[Exception], None] = lambda _error: None,
) -> None:
    while True:
        try:
            saved = sync_glb_artifacts(comfy_url, output_dir, opener=opener)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            on_error(error)
        else:
            for path in saved:
                on_saved(path)
        time.sleep(max(0.5, float(interval)))
