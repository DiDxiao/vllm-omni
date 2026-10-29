# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-agnostic storage for completed video-generation session snapshots.

Purpose
-------
Streaming diffusion generation (e.g. Helios) is chunked: each chunk's denoised
latents are accumulated into a growing ``history_latents`` buffer so subsequent
chunks can condition on the already-generated frames. If the user is unhappy
with the tail of a video, we can regenerate it from some frame *without*
re-denosing the prefix by seeding the new generation with the saved prefix
latents.

This module provides the *model-agnostic* half of that feature:

* :class:`GenerationSessionSnapshot` -- an opaque, model-specific ``payload``
  (usually on-device latents + metadata) associated with a ``session_id``.
* :class:`GenerationSessionStore` -- an in-process store keyed by
  ``session_id``, with LRU + max-entries eviction. The latents are kept
  on the accelerator device (no CPU copy) so seeding a regeneration is a cheap
  device-to-device operation; device memory is reclaimed on eviction.

The store deliberately knows nothing about the model: the model-specific
"snapshot" and "seed from snapshot" logic lives in each pipeline (e.g.
``HeliosDiffusionPipeline.snapshot_generation_state`` /
``seed_generation_from_snapshot``). Keeping this store at the diffusion layer
makes it reusable by any future streaming-video model.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any


class GenerationSessionSnapshot:
    """An immutable, model-agnostic snapshot of a finished generation session.

    ``payload`` is a model-private dict produced by a pipeline's
    ``snapshot_generation_state()`` hook. For Helios it contains the CPU
    ``history_latents`` plus the metadata needed to resume from an arbitrary
    frame (see ``pipeline_helios.py``).
    """

    __slots__ = (
        "session_id",
        "num_frames",
        "width",
        "height",
        "fps",
        "model",
        "payload",
        "created_at",
        "last_access",
    )

    def __init__(
        self,
        *,
        session_id: str,
        num_frames: int,
        width: int,
        height: int,
        fps: int,
        model: str,
        payload: dict[str, Any],
    ) -> None:
        self.session_id = session_id
        self.num_frames = num_frames
        self.width = width
        self.height = height
        self.fps = fps
        self.model = model
        self.payload = payload
        self.created_at = time.monotonic()
        self.last_access = self.created_at


class GenerationSessionStore:
    """In-process store of generation snapshots.

    The snapshot latents are kept on the accelerator device. For a single-worker
    server the regenerated request is routed to the same worker that holds the
    snapshot; for multi-worker deployments you must ensure session->worker
    affinity (out of scope here).

    Eviction policy
    ---------------
    When the number of live entries exceeds ``max_entries``, the least
    recently accessed entry is evicted (LRU).
    """

    def __init__(self, max_entries: int = 32) -> None:
        self._max_entries = max(1, int(max_entries))
        self._snapshots: dict[str, GenerationSessionSnapshot] = {}
        # session_id in most-recently-used -> least-recently-used order.
        self._lru: "OrderedDict[str, None]" = OrderedDict()

    # -- query API ---------------------------------------------------------

    def get(self, session_id: str) -> GenerationSessionSnapshot | None:
        snapshot = self._snapshots.get(session_id)
        if snapshot is None:
            return None
        # Touch LRU order.
        self._lru.pop(session_id, None)
        self._lru[session_id] = None
        snapshot.last_access = time.monotonic()
        return snapshot

    def put(self, snapshot: GenerationSessionSnapshot) -> None:
        self._snapshots[snapshot.session_id] = snapshot
        self._lru[snapshot.session_id] = None
        self._prune()

    def __contains__(self, session_id: str) -> bool:
        return session_id in self._snapshots

    def __len__(self) -> int:
        return len(self._snapshots)

    # -- internals ---------------------------------------------------------

    def _remove(self, session_id: str) -> None:
        self._snapshots.pop(session_id, None)
        self._lru.pop(session_id, None)

    def _prune(self) -> None:
        # Enforce LRU cap.
        while len(self._snapshots) > self._max_entries and self._lru:
            oldest = next(iter(self._lru))
            self._remove(oldest)
