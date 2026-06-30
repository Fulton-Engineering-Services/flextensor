# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single source for FlexTensor's installed package version."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("flextensor")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
