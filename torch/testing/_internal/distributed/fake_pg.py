# mypy: allow-untyped-defs

import torch.distributed as dist
<<<<<<< HEAD

from torch._C._distributed_c10d import (
    FakeProcessGroup,
)
=======
from torch._C._distributed_c10d import FakeProcessGroup
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))


class FakeStore(dist.Store):
    """
    A fake store is a fake Key-Value store simply for initialization usage
    the of fake process group, one can either use FakeStore or HashStore.
    """


def _create_fake_pg(prefix_store, rank, world_size, timeout):
    """
    A fake process group (not related to FakeTensor) is a process group which
    doesn't actually do any communication, it just hallucinates some
    communication.  You can run a single rank with a fake process group
    without needing multiple processes (simulates per-rank behavior)

    NOTE: This is not a real process group, and it would produce wrong results
<<<<<<< HEAD
    for every collective. It should be used as a convinient tool when playing
=======
    for every collective. It should be used as a convenient tool when playing
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    with distributed but don't care about the actual data.
    """
    return FakeProcessGroup(rank, world_size)


<<<<<<< HEAD
dist.Backend.register_backend("fake", _create_fake_pg, devices=['cpu', 'cuda'])
=======
dist.Backend.register_backend("fake", _create_fake_pg, devices=["cpu", "cuda", "hpu"])
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
