# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Patched time-series decoder implementation for TimesFM.

This module implements the core patched decoder architecture used by TimesFM
for time-series forecasting. Input sequences are divided into non-overlapping
patches which are then processed by a transformer-based decoder.
"""

from typing import Optional

import numpy as np


def moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute a simple moving average over the last axis.

    Args:
        arr: Input array of shape (..., T).
        window: Size of the averaging window.

    Returns:
        Smoothed array of the same shape as ``arr``.
    """
    if window <= 1:
        return arr
    kernel = np.ones(window) / window
    # Apply along the last axis for each leading dimension.
    shape = arr.shape
    flat = arr.reshape(-1, shape[-1])
    smoothed = np.array(
        [np.convolve(row, kernel, mode="same") for row in flat]
    )
    return smoothed.reshape(shape)


def create_patches(
    time_series: np.ndarray,
    patch_len: int,
    stride: Optional[int] = None,
) -> np.ndarray:
    """Divide a time series into non-overlapping (or strided) patches.

    Args:
        time_series: Array of shape (batch, time) or (time,).
        patch_len: Length of each patch.
        stride: Step between consecutive patches. Defaults to ``patch_len``
            (non-overlapping patches).

    Returns:
        Array of shape (batch, num_patches, patch_len) or
        (num_patches, patch_len) when the input is 1-D.
    """
    if stride is None:
        stride = patch_len

    squeeze = time_series.ndim == 1
    if squeeze:
        time_series = time_series[np.newaxis, :]

    batch, time = time_series.shape
    num_patches = max(0, (time - patch_len) // stride + 1)

    patches = np.stack(
        [time_series[:, i * stride : i * stride + patch_len] for i in range(num_patches)],
        axis=1,
    )  # (batch, num_patches, patch_len)

    if squeeze:
        patches = patches[0]  # (num_patches, patch_len)
    return patches


def pad_to_patch_multiple(
    time_series: np.ndarray,
    patch_len: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Pad a time series along the time axis so its length is a multiple of patch_len.

    Args:
        time_series: Array of shape (batch, time) or (time,).
        patch_len: Desired patch length.
        pad_value: Constant value used for padding.

    Returns:
        Padded array whose time dimension is divisible by ``patch_len``.
    """
    squeeze = time_series.ndim == 1
    if squeeze:
        time_series = time_series[np.newaxis, :]

    batch, time = time_series.shape
    remainder = time % patch_len
    if remainder != 0:
        pad_width = patch_len - remainder
        padding = np.full((batch, pad_width), pad_value, dtype=time_series.dtype)
        time_series = np.concatenate([time_series, padding], axis=1)

    if squeeze:
        time_series = time_series[0]
    return time_series


def normalize_patches(patches: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Instance-normalize patches by subtracting the mean and dividing by the std.

    Args:
        patches: Array of shape (batch, num_patches, patch_len).
        eps: Small constant for numerical stability.

    Returns:
        A tuple ``(normalized, means, stds)`` where ``means`` and ``stds``
        have shape (batch, num_patches, 1).
    """
    means = patches.mean(axis=-1, keepdims=True)
    stds = patches.std(axis=-1, keepdims=True) + eps
    normalized = (patches - means) / stds
    return normalized, means, stds
