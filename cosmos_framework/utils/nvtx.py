# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Opt-in NVTX phase ranges for profiling inference under Nsight Systems.

Ranges are off unless ``COSMOS_NVTX=1`` is exported, so the instrumentation in
shared model code costs nothing on training runs and in production serving.

Placement rule: the MoT decoder blocks and five of the ``Cosmos3VFMNetwork``
encode/decode heads run under ``torch.compile(fullgraph=True)`` (see
``mot/parallelize_unified_mot.py`` and ``mot/parallelize_vfm_network.py``), and a
range pushed inside a compiled region is traced away rather than emitted. Wrap
the eager *call site* instead of the callee body.

Kept separate from ``cosmos_framework.utils.misc`` so that the low-level network
modules can import it without pulling in that module's dependencies.
"""

import contextlib
import os

import torch

# Resolved once at import: the env var is a profiling-session switch, not something
# that should be re-read per range in a hot loop.
_ENABLED: bool = os.environ.get("COSMOS_NVTX", "0").strip().lower() not in ("0", "", "false", "no")

# ``nullcontext`` holds no state, so a single instance is safe to share across
# nested and concurrent ``with`` blocks. This keeps the disabled path allocation-free.
_NULL: contextlib.nullcontext = contextlib.nullcontext()


def nvtx_enabled() -> bool:
    """Whether ``COSMOS_NVTX`` selected NVTX range emission at import time."""
    return _ENABLED


def nvtx_phase(name: str):
    """Return a context manager that marks ``name`` as an NVTX range.

    A no-op unless ``COSMOS_NVTX=1``. Unlike
    :class:`cosmos_framework.utils.misc.NVTXRangeContext` this never calls
    ``torch.cuda.synchronize()``: synchronizing on range boundaries serializes the
    CPU and GPU and manufactures exactly the idle gaps that profiling is meant to
    find.
    """
    if not _ENABLED:
        return _NULL
    return torch.cuda.nvtx.range(name)


def nvtx_push(name: str) -> None:
    """Open a range that :func:`nvtx_pop` closes.

    Prefer :func:`nvtx_phase`. This pair exists for spans too long to wrap in a
    ``with`` without reindenting the whole body, and is only safe on straight-line
    code: an early ``return`` or a raised exception between the two calls leaves the
    range open.
    """
    if _ENABLED:
        torch.cuda.nvtx.range_push(name)


def nvtx_pop() -> None:
    """Close the range most recently opened by :func:`nvtx_push`."""
    if _ENABLED:
        torch.cuda.nvtx.range_pop()
