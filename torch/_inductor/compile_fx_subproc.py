from __future__ import annotations

import atexit
import functools
import os
from typing import Optional, TYPE_CHECKING
<<<<<<< HEAD
from typing_extensions import override
=======
from typing_extensions import final, override
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
import torch.fx
from torch._inductor.compile_worker.subproc_pool import (
    AnyPool,
    SubprocKind,
    SubprocPool,
)
<<<<<<< HEAD
from torch._inductor.utils import clear_inductor_caches
=======
from torch._inductor.utils import clear_caches
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

from .compile_fx_ext import (
    _OutOfProcessFxCompile,
    _WireProtocolPickledInput,
    _WireProtocolPickledOutput,
)
from .output_code import complex_memory_overlap as complex_memory_overlap  # noqa: F401


if TYPE_CHECKING:
    from collections.abc import Mapping
<<<<<<< HEAD


class _SubprocessFxCompile(_OutOfProcessFxCompile):
    @override
    def _send_to_child(
        self, input: _WireProtocolPickledInput
    ) -> _WireProtocolPickledOutput:
=======
    from concurrent.futures import Future


@final
class _SubprocessFxCompile(_OutOfProcessFxCompile):
    @override
    def _send_to_child_async(
        self, input: _WireProtocolPickledInput
    ) -> Future[_WireProtocolPickledOutput]:
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        # TODO: Do we need to copy across some kind of logging IDs? (ChromiumEventLogger)

        pool = self.process_pool()

        # TODO: This is probably the wrong thing to do long-term - but for now
        # let's share the cache so we can identify tests broken by this later.
        env_vars = ["TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"]
        extra_env = {v: os.environ[v] for v in env_vars if v in os.environ}

<<<<<<< HEAD
        f = pool.submit(_SubprocessFxCompile._run_in_child_subprocess, input, extra_env)

        # For debugging: If we want to print status updates...
        # last = time.time()
        # while not f.done():
        #     print("tick...")
        #     time.sleep(0.125)
        #     now = time.time()
        #     if now - last > 1:
        #         last = now

        output = f.result()

        return output
=======
        return pool.submit(
            _SubprocessFxCompile._run_in_child_subprocess, input, extra_env
        )
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

    @staticmethod
    @functools.cache
    def process_pool() -> AnyPool:
        pool = SubprocPool(
            # TODO: Consider raising this limit if we start using async w/
            # subprocess and want to compile multiple graphs in parallel.
            1,
            kind=SubprocKind.SPAWN,
        )

        atexit.register(pool.shutdown)

        return pool

    @classmethod
    def _run_in_child_subprocess(
        cls,
        pickled_input: _WireProtocolPickledInput,
        extra_env: Optional[Mapping[str, str]],
    ) -> _WireProtocolPickledOutput:
        # TODO: In subprocess mode we need to clear the inductor caches.
        # The problem:
        #   1. We compile in worker A which fills stuff in tmpdir
        #   2. parent clears inductor caches which deletes tmpdirs and tells
        #      cpp_prefix_path() to clear its LRU cache
        #   3. We compile a second time in subproc A - but since we never told
        #      cpp_prefix_path() in worker A to clear its LRU it thinks the
        #      tmpdir still exists and fails to compile.
        #
        # TODO: We probably should be using a separate tmpdir in the worker
<<<<<<< HEAD
        # anyway... but we should probably still respect clear_inductor_caches()
=======
        # anyway... but we should probably still respect clear_caches()
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        # in the parent... maybe?
        #
        # TODO: We could be less aggressive by keeping a clock which gets
        # incremented when we clear the cache, send the clock to the worker and
        # only clear caches if the clock changed since last time.
        #
<<<<<<< HEAD
        clear_inductor_caches()
=======
        clear_caches()
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        torch._inductor.metrics.reset()

        # TODO: turn off config.fx_graph_async_compile

        result = cls._run_in_child(pickled_input, extra_env)
        return result
