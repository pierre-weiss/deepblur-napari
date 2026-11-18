r"""
Download and local cache of the deepblur weights, as in ``deepblur.weights``.

The weights are served by the MAMBO server (https://mambo-utoulouse.fr/weights),
together with a manifest giving the sha256 of every file. They are cached in
``~/.cache/deepblur/weights`` (or ``$DEEPBLUR_CACHE/weights``), shared with the
deepblur package and the gradio app, and downloaded again only when the
manifest says they changed. Without network, the cached files are used. The
identification models are also on the Hugging Face hub, which serves as a
fallback when the server can not provide them.
"""

import json
import os
import urllib.request
from functools import lru_cache
from pathlib import Path

WEIGHTS_URL = os.getenv(
    "DEEPBLUR_WEIGHTS_URL", "https://mambo-utoulouse.fr/weights"
)
CACHE_DIR = (
    Path(os.getenv("DEEPBLUR_CACHE", Path.home() / ".cache" / "deepblur"))
    / "weights"
)
HF_REPO_ID = "mhnguyen712/deepblur"
MANIFEST = "manifest.json"


class WeightsUnavailableError(RuntimeError):
    r"""Raised when a weights file can neither be downloaded nor found in the cache."""


@lru_cache(maxsize=1)
def _get_manifest() -> dict | None:
    r"""
    Returns the server manifest ``{path: {"sha256": ..., "size": ...}}``, or
    None if the server is unreachable. It is fetched once per session.
    """
    try:
        with urllib.request.urlopen(
            f"{WEIGHTS_URL}/{MANIFEST}", timeout=10
        ) as response:
            return json.load(response)["files"]
    except (OSError, ValueError, KeyError) as e:
        print(f"Weights server {WEIGHTS_URL} unreachable: {e}")
        return None


def get_weights(path: str) -> Path:
    r"""
    Returns the local path of a weights file, downloading it from the MAMBO
    server if it is not cached or out of date.

    :param str path: path relative to the weights root, e.g. ``"RAM/checkpoint_rgb.pth"``.
    :raises WeightsUnavailableError: if the file can not be downloaded and is not cached.
    """
    local = CACHE_DIR / path
    manifest = _get_manifest()
    if manifest is None or path not in manifest:
        if local.exists():
            return local
        raise WeightsUnavailableError(
            f"{path} is not available from {WEIGHTS_URL}"
        )

    # The sha256 of the cached version is stored next to it, to avoid hashing
    # large files at every load
    sha256 = manifest[path]["sha256"]
    stamp = local.with_name(local.name + ".sha256")
    if local.exists() and stamp.exists() and stamp.read_text() == sha256:
        return local

    from torch.hub import download_url_to_file

    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        download_url_to_file(
            f"{WEIGHTS_URL}/{path}", str(local), hash_prefix=sha256
        )
    except (OSError, RuntimeError) as e:
        raise WeightsUnavailableError(f"Could not download {path}: {e}") from e
    stamp.write_text(sha256)
    return local


def get_weights_files(paths: list[str]) -> tuple[list[Path], str]:
    r"""
    Returns the local paths of weights files that must come from the same
    source, like a checkpoint and its configs: the MAMBO server, or the
    Hugging Face hub if the server can not provide all of them. The hub may
    have older versions of the weights.

    :param list[str] paths: paths relative to the weights root.
    :return: the local paths, and their source: ``"mambo"`` or ``"huggingface"``.
    """
    try:
        return [get_weights(path) for path in paths], "mambo"
    except WeightsUnavailableError as e:
        print(f"{e}. Falling back to the Hugging Face hub ({HF_REPO_ID}).")
        from huggingface_hub import hf_hub_download

        return [
            Path(hf_hub_download(repo_id=HF_REPO_ID, filename=path))
            for path in paths
        ], "huggingface"
