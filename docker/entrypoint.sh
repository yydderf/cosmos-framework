#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


# Docker entrypoint script.

set -e

uv pip install --no-deps -e . || true

mkdir -p /opt/nsight/bin && ln -sf /opt/nsight/target-linux-x64/nsys /opt/nsight/bin/nsys
export PATH=/opt/nsight/bin:$PATH

exec "$@"
