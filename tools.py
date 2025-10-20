
from __future__ import annotations

import numpy as np

# Underlying implementations
from local_model_predictor import (
    InferenceInterface,
)

import config



class ViTimePredictor:
    """Thin wrapper around the underlying inference interface.

    Initializes model weights from `config.VITIME_MODEL_PATH` and exposes
    a callable that maps a time series and `future_length` to predictions.
    """


    def __init__(
        self,
        device: str = 'cuda:0',
        model_name: str = 'MAE',

    ) -> None:
        
        model_path_env = config.VITIME_MODEL_PATH  
    
        self._iface = InferenceInterface(model_path_env,  model_name=model_name, device=device)

    def __call__(self, time_series, future_length) -> np.ndarray:
      
        pred = self._iface.inference(
            np.asarray(time_series),
            future_length
        )
        return pred.flatten()
