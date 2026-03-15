<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# OffloadManager

The `OffloadManager` class provides full control over the offloading lifecycle.
Obtain an instance with [`get_offload_manager()`](simplified.md#flextensor.offload_manager.get_offload_manager)
rather than constructing one directly.

::: flextensor.offload_manager.OffloadManager
    options:
      members:
        - init
        - set_config
        - offload
        - offload_block
        - get_gpu_memory_usage
        - save_profile
        - load_profile
        - release
