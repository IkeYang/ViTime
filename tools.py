
from __future__ import annotations

import numpy as np

# Underlying implementations
from local_model_predictor import (
    InferenceInterface,
)

import config



class ViTimePredictor:
    """Thin wrapper around the underlying inference interface.

    Resolves model weights from a local override or Hugging Face Hub and
    exposes a callable that maps a time series and `future_length` to
    predictions.
    """


    def __init__(
        self,
        device: str = 'cuda:0',
        model_name: str = 'MAE',
        tempature=1,
        model_path: str | None = None,
    ) -> None:
        resolved_model_path = config.resolve_model_path(model_path)
        self.tempature = tempature
        self._iface = InferenceInterface(
            resolved_model_path,
            model_name=model_name,
            device=device,
        )

    def __call__(self, time_series, future_length,sampleNumber) -> np.ndarray:
      
        pred = self._iface.inference(
            np.asarray(time_series),
            future_length,
            sampleNumber,
            tempature=self.tempature
        )
        return pred
