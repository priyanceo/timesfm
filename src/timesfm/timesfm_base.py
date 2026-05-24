"""Base class for TimesFM forecasting models.

This module provides the abstract base class that defines the interface
for all TimesFM model implementations.
"""

import abc
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass
class TimesFmHparams:
    """Hyperparameters for TimesFM models.

    Attributes:
        context_len: Number of time steps used as input context.
        horizon_len: Number of time steps to forecast.
        input_patch_len: Length of each input patch.
        output_patch_len: Length of each output patch.
        num_layers: Number of transformer layers.
        model_dims: Dimensionality of the model.
        backend: Computation backend ('cpu', 'gpu', 'tpu').
        per_core_batch_size: Batch size per compute core.
        quantiles: List of quantile levels to predict.
    """

    context_len: int = 512
    horizon_len: int = 128
    input_patch_len: int = 32
    output_patch_len: int = 128
    num_layers: int = 20
    model_dims: int = 1280
    backend: str = "cpu"
    per_core_batch_size: int = 32
    quantiles: list[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )

    def __post_init__(self):
        if self.context_len <= 0:
            raise ValueError(f"context_len must be positive, got {self.context_len}")
        if self.horizon_len <= 0:
            raise ValueError(f"horizon_len must be positive, got {self.horizon_len}")
        if self.backend not in ("cpu", "gpu", "tpu"):
            raise ValueError(
                f"backend must be one of 'cpu', 'gpu', 'tpu', got {self.backend}"
            )
        for q in self.quantiles:
            if not 0.0 < q < 1.0:
                raise ValueError(
                    f"All quantiles must be in (0, 1), got {q}"
                )


class TimesFmBase(abc.ABC):
    """Abstract base class for TimesFM forecasting models.

    Subclasses must implement `_forecast` to provide model-specific
    inference logic.
    """

    def __init__(self, hparams: TimesFmHparams, checkpoint: Any):
        """Initializes the base model.

        Args:
            hparams: Model hyperparameters.
            checkpoint: Checkpoint specification (path, HuggingFace repo, etc.).
        """
        self.hparams = hparams
        self.checkpoint = checkpoint
        self._model_loaded = False

    @abc.abstractmethod
    def load_from_checkpoint(self, checkpoint: Any) -> None:
        """Loads model weights from a checkpoint.

        Args:
            checkpoint: Checkpoint specification compatible with the backend.
        """

    @abc.abstractmethod
    def _forecast(
        self,
        inputs: Sequence[np.ndarray],
        freq: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Core forecasting method to be implemented by subclasses.

        Args:
            inputs: List of 1-D numpy arrays, each representing a time series.
            freq: List of frequency indicators (0=high, 1=medium, 2=low).

        Returns:
            A tuple of (point_forecasts, quantile_forecasts) where:
              - point_forecasts has shape (batch, horizon_len)
              - quantile_forecasts has shape (batch, horizon_len, num_quantiles)
        """

    def forecast(
        self,
        inputs: Sequence[np.ndarray],
        freq: Sequence[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generates forecasts for the provided time series.

        Args:
            inputs: List of 1-D numpy arrays, each representing a time series.
            freq: Optional list of frequency indicators per series.
                  Defaults to 0 (high frequency) for all series.

        Returns:
            A tuple of (point_forecasts, quantile_forecasts).

        Raises:
            RuntimeError: If the model has not been loaded yet.
        """
        if not self._model_loaded:
            raise RuntimeError(
                "Model weights have not been loaded. "
                "Call load_from_checkpoint() before forecasting."
            )

        if freq is None:
            freq = [0] * len(inputs)

        if len(inputs) != len(freq):
            raise ValueError(
                f"Length of inputs ({len(inputs)}) must match "
                f"length of freq ({len(freq)})."
            )

        return self._forecast(inputs, freq)
