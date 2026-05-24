# Copyright 2024 The TimesFM Authors.
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

"""TimesFM: A time series foundation model for forecasting.

This package provides a pretrained time series foundation model capable of
zero-shot forecasting across a wide range of time series tasks.

Basic usage:
    >>> import timesfm
    >>> tfm = timesfm.TimesFm(
    ...     context_len=512,
    ...     horizon_len=96,
    ...     backend="cpu",
    ... )
    >>> tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")
    >>> forecast = tfm.forecast(inputs, freq=[0])

Note (personal): For GPU usage, set backend="gpu" and ensure JAX is installed
with CUDA support. The 200m model works well for most daily/hourly data.
"""

from timesfm.timesfm_base import TimesFmBase
from timesfm.timesfm_base import TimesFmHparams

try:
    from timesfm._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"

__all__ = [
    "TimesFmBase",
    "TimesFmHparams",
    "__version__",
]
