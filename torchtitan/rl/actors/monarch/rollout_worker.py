# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import functools
import os
import socket
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from monarch.actor import concurrent_endpoint, current_rank

# TODO: load the `grain` and `datasets` packages only where they are used, to save ~450 MiB per rollout worker.
#   This import runs `torchtitan/components/data/__init__.py`, which loads both. The renderer and
#   `rl/rollout/data.py` import them too. First check that a worker doesn't need them.
from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.components.renderer import RendererConfig

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.observability import structured_logger as sl
from torchtitan.rl.actors.monarch.actor import Actor
from torchtitan.rl.rollout.rollouter import BaseRolloutWorker
from torchtitan.rl.rollout.types import RolloutGroup
from torchtitan.tools.garbage_collector import GarbageCollector


class RolloutWorkerActor(Actor):
    """Monarch adapter that owns a configurable standalone rollout worker."""

    def __init__(
        self,
        *,
        worker_config: BaseRolloutWorker.Config,
        num_threads: int,
        structured_logger_init_fn: functools.partial[None] | None,
        first_worker_index: int,
    ) -> None:
        if structured_logger_init_fn is not None:
            # Each host's workers form their own proc mesh, so `current_rank()` restarts at 0 on
            # every host. The offset keeps each worker's log rank unique.
            structured_logger_init_fn(
                source="rl_rollout_worker",
                rank=first_worker_index + current_rank().rank,
            )
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=num_threads)
        )
        self._worker_config = worker_config
        # Built in `setup_async`: a build error there, e.g. a Verifiers env that fails to load,
        # reaches the caller as an endpoint error. Raised here, it would kill the actor.
        self._worker: BaseRolloutWorker

    @concurrent_endpoint
    async def setup_async(
        self,
        *,
        tokenizer_config: HuggingFaceTokenizer.Config,
        renderer_config: RendererConfig,
        hf_assets_path: str,
        paths_on_every_host: list[str],
        vision_processor_config: VisionProcessor.Config | None = None,
    ) -> None:
        # The training dataset's paths, e.g. its image dir. A host without one fails the job here, at
        # start, instead of failing each rollout that reads it.
        for path in paths_on_every_host:
            if not os.path.exists(path):
                raise RuntimeError(
                    f"{os.path.abspath(path)} is missing on host {socket.gethostname()}. The "
                    "training dataset's samples point there (rollouter.training_dataloader.dataset). "
                    "Please make it available at this path on every host that runs rollout workers, "
                    "best in a shared folder, or keep every rollout worker on the controller's host: "
                    "set rollouter.worker_roles.generator=None and "
                    "rollouter.worker_roles.controller=True."
                )
        self._worker = self._worker_config.build()
        await self._worker.setup_async(
            tokenizer_config=tokenizer_config,
            renderer_config=renderer_config,
            hf_assets_path=hf_assets_path,
            vision_processor_config=vision_processor_config,
        )

        # Setup objects live until exit: freeze them, so later full GCs skip them.
        GarbageCollector.freeze(
            "Full GC before freezing the rollout worker's setup objects"
        )

        # Start measuring after setup and the freeze: both hold the loop.
        self._worker.start_process_metrics()

    @concurrent_endpoint
    async def run_group(
        self,
        *,
        generate_fn: Any,
        release_session_fn: Any,
        sample: object,
        group_id: int,
        group_size: int,
        sampling: Any,
    ) -> RolloutGroup:
        group = await self._worker.run_group(
            generate_fn=generate_fn,
            release_session_fn=release_session_fn,
            sample=sample,
            group_id=group_id,
            group_size=group_size,
            sampling=sampling,
        )
        group.metrics += self._worker.flush_process_metrics(group_id)
        return group

    @concurrent_endpoint
    async def close(self) -> None:
        # The controller closes every host's workers, also when `setup_async` failed before it
        # built this one, e.g. on a missing path. Then there is nothing to close.
        if hasattr(self, "_worker"):
            await self._worker.close()

    @concurrent_endpoint
    async def sync_log_step(self, step: int) -> None:
        sl.set_step(step)
