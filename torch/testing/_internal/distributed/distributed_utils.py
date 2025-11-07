# mypy: allow-untyped-defs

from contextlib import contextmanager
from datetime import timedelta
<<<<<<< HEAD
from functools import (
    partial,
    wraps,
)
=======
from functools import partial, wraps
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

<<<<<<< HEAD
class MockProcessGroup(dist.ProcessGroup):

=======

class MockProcessGroup(dist.ProcessGroup):
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    def __init__(self, rank, world):
        super().__init__(rank, world)

    def getBackendName(self):
        return "mock_process_group"

<<<<<<< HEAD
def create_mock_pg(prefix_store, rank, world_size, timeout):
    return MockProcessGroup(rank, world_size)

dist.Backend.register_backend('mock_process_group', create_mock_pg)
=======

def create_mock_pg(prefix_store, rank, world_size, timeout):
    return MockProcessGroup(rank, world_size)


dist.Backend.register_backend("mock_process_group", create_mock_pg)

>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

def mock_init_dist(rank, world_size):
    # !!! WARNING !!!
    # Kids don't try this at home, this is a cute pile of hacks that
    # depends on a small mountain of c10d internals
    assert not dist.is_initialized()
    store = dist.HashStore()
    # Trick _store_based_barrier into believing everyone else already checked-in
    # Zero is the group index
    store.add(f"{c10d.STORE_BASED_BARRIER_PREFIX}:0", world_size - 1)
    dist.init_process_group(
        backend="mock_process_group",
        rank=rank,
        world_size=world_size,
        store=store,
        group_name="fake",
<<<<<<< HEAD
        timeout=timedelta(seconds=1))
=======
        timeout=timedelta(seconds=1),
    )

>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

@contextmanager
def with_dist(rank=0, world_size=2):
    """
    Context manager that initializer c10d with a fake process group.
    """
    mock_init_dist(rank=rank, world_size=world_size)
    try:
        yield
    finally:
        dist.destroy_process_group()

<<<<<<< HEAD
=======

>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
def with_fake_comms(func=None, rank=0, world_size=2):
    """
    Function wrapper that inits a fake process group designed for testing.
    Right now only querying for world size is available
    """
    if func is None:
        return partial(with_fake_comms, rank=rank, world_size=world_size)

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with with_dist(rank, world_size):
            func(self, *args, **kwargs)
<<<<<<< HEAD
=======

>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    return wrapper
