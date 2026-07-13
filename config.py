"""Resolve the ViTime checkpoint from a local path or Hugging Face Hub."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download


VITIME_HF_REPO_ID = os.getenv("VITIME_HF_REPO_ID", "IkeYEUNG/ViTime")
VITIME_HF_FILENAME = os.getenv("VITIME_HF_FILENAME", "ViTime_Model.pth")
VITIME_HF_REVISION = os.getenv("VITIME_HF_REVISION", "v1.0.0")

# Backward-compatible programmatic override. Prefer passing ``model_path`` to
# ViTimePrediction or setting the VITIME_MODEL_PATH environment variable.
VITIME_MODEL_PATH: str | None = None


def resolve_model_path(model_path: str | None = None) -> str:
    """Return a validated local checkpoint path, downloading it if needed.

    Resolution order:
    1. The explicit ``model_path`` argument.
    2. The ``VITIME_MODEL_PATH`` environment variable.
    3. The backward-compatible ``config.VITIME_MODEL_PATH`` value.
    4. The versioned public checkpoint on Hugging Face Hub.
    """

    override = model_path or os.getenv("VITIME_MODEL_PATH") or VITIME_MODEL_PATH
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"ViTime checkpoint does not exist: {path}")
        return str(path.resolve())

    try:
        return hf_hub_download(
            repo_id=VITIME_HF_REPO_ID,
            filename=VITIME_HF_FILENAME,
            revision=VITIME_HF_REVISION,
        )
    except Exception as exc:
        raise RuntimeError(
            "Unable to download the ViTime checkpoint from Hugging Face Hub. "
            "Check your network connection, or set VITIME_MODEL_PATH to a "
            "local ViTime_Model.pth file."
        ) from exc
