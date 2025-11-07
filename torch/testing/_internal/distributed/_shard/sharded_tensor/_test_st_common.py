# mypy: allow-untyped-defs

import copy
import random
<<<<<<< HEAD
import torch
from torch.distributed._shard import sharded_tensor

from torch.distributed._shard.sharding_spec import (
    ChunkShardingSpec,
)
=======

import torch
from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec

>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

PLACEMENTS = [
    "rank:0/cuda:0",
    "rank:1/cuda:1",
    "rank:2/cuda:2",
    "rank:3/cuda:3",
]

DEFAULT_GPU_NUM = 4


def _chunk_sharding_specs_list_for_test(sharding_dims, seed=0):
    spec_list = []
    for i in range(len(sharding_dims)):
        random.Random(seed + i).shuffle(PLACEMENTS)
        spec_list.append(
            ChunkShardingSpec(
                dim=sharding_dims[i],
                placements=copy.deepcopy(PLACEMENTS),
            )
        )
    return spec_list

<<<<<<< HEAD
class MyShardedModel2(torch.nn.Module):
    def __init__(
        self,
        spec=None,
        group=None,
        init_rrefs=True
    ) -> None:
=======

class MyShardedModel2(torch.nn.Module):
    def __init__(self, spec=None, group=None, init_rrefs=True) -> None:
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        super().__init__()
        if spec is not None:
            self.sharded_tensor2 = sharded_tensor.rand(
                spec, 10, 20, process_group=group, init_rrefs=init_rrefs
            )
        else:
            self.sharded_tensor2 = None
        self.random_tensor2 = torch.nn.Parameter(torch.rand(2, 2))


class MyShardedModel1(torch.nn.Module):
<<<<<<< HEAD
    def __init__(
        self,
        spec=None,
        group=None,
        init_rrefs=True
    ) -> None:
=======
    def __init__(self, spec=None, group=None, init_rrefs=True) -> None:
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        super().__init__()
        if spec is not None:
            self.sharded_tensor1 = sharded_tensor.rand(
                spec, 10, 20, process_group=group, init_rrefs=init_rrefs
            )
        else:
            self.sharded_tensor1 = None
        self.random_tensor1 = torch.nn.Parameter(torch.rand(2, 2))
        self.submodule = MyShardedModel2(spec, group, init_rrefs)
