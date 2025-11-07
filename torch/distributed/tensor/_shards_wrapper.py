# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import (
    TensorWriteData,
    WriteItem,
    WriteItemType,
)


<<<<<<< HEAD
aten = (
    torch.ops.aten
)  # pyre-ignore[5]: Globally accessible variable `aten` has no type specified.


class LocalShardsWrapper(torch.Tensor):  # pyre-ignore[13]: pyre is bad at __new__
=======
aten = torch.ops.aten


class LocalShardsWrapper(torch.Tensor):
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    """
    A wrapper class to hold local shards of a DTensor.
    This class is used largely for checkpointing purposes and implicity subtypes
    the _Checkpointable protocol.
    """

    __slots__ = ["_local_shards", "_storage_meta"]
    _local_shards: list[torch.Tensor]
    _storage_meta: TensorStorageMetadata

    @staticmethod
    def __new__(
        cls, local_shards: list[torch.Tensor], local_offsets: list[tuple[int, ...]]
    ) -> "LocalShardsWrapper":
<<<<<<< HEAD
        assert len(local_shards) > 0
        assert len(local_shards) == len(local_offsets)
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        assert all(
            tensor.device == local_shards[0].device for tensor in local_shards[1:]
        )

<<<<<<< HEAD
        # we calculate the total tensor size by "concat" on second tensor dimension
        cat_tensor_shape = list(local_shards[0].size())
        if len(local_shards) > 1:  # column-wise sharding
            for shard in local_shards[1:]:
                cat_tensor_shape[1] += shard.size()[1]

=======
        # if empty shard, we create a empty tensor
        if len(local_shards) == 0:
            r = torch.Tensor._make_wrapper_subclass(
                cls,
                torch.Size([0, 0]),
            )
            r._local_shards = []
            r._storage_meta = TensorStorageMetadata(
                properties=TensorProperties(),
                size=torch.Size([0, 0]),
                chunks=[
                    ChunkStorageMetadata(
                        offsets=torch.Size([0, 0]), sizes=torch.Size([0, 0])
                    )
                ],
            )
            return r

        # we calculate the total tensor size by "concat" on second tensor dimension
        cat_tensor_shape = list(local_shards[0].size())
        if len(local_shards) > 1 and local_shards[0].ndim == 2:  # column-wise sharding
            for shard in local_shards[1:]:
                cat_tensor_shape[1] += shard.size()[1]

        # in cases of sharding optimizer rowwise, we calculate total tensor size by "concat" on first tensor dimension
        if len(local_shards) > 1 and local_shards[0].ndim == 1:  # column-wise sharding
            for shard in local_shards[1:]:
                cat_tensor_shape[0] += shard.size()[0]

>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        wrapper_properties = TensorProperties.create_from_tensor(local_shards[0])
        wrapper_shape = torch.Size(cat_tensor_shape)
        chunks_meta = [
            ChunkStorageMetadata(
                offsets=torch.Size(offset),
                sizes=shard.size(),
            )
            for shard, offset in zip(local_shards, local_offsets)
        ]

<<<<<<< HEAD
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
=======
        r = torch.Tensor._make_wrapper_subclass(
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
            cls,
            torch.Size(cat_tensor_shape),
        )
        r._local_shards = local_shards
        r._storage_meta = TensorStorageMetadata(
            properties=wrapper_properties,
            size=wrapper_shape,
            chunks=chunks_meta,
        )

        return r

    # necessary for ops dispatching from this subclass to its local shards
    @classmethod
<<<<<<< HEAD
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
=======
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore[override]
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        kwargs = kwargs or {}

        dispatcher = {
            torch.ops._c10d_functional.all_gather_into_tensor.default: cls.handle_all_gather_into_tensor,
            torch.ops._c10d_functional.wait_tensor.default: cls.handle_wait_tensor,
            aten._to_copy.default: cls.handle_to_copy,
            aten.view.default: cls.handle_view,
            aten.equal.default: cls.handle_equal,
            aten.detach.default: cls.handle_detach,
            aten.clone.default: cls.handle_clone,
<<<<<<< HEAD
        }

        if func in dispatcher:
            return dispatcher[func](
                args, kwargs
            )  # pyre-ignore [29] - `Variable[_VT]` is not a function.
=======
            aten.new_empty.default: cls.handle_new_empty,
        }

        if func in dispatcher:
            return dispatcher[func](args, kwargs)
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        else:
            raise NotImplementedError(
                f"{func} is not supported for LocalShardsWrapper!"
            )

    @staticmethod
<<<<<<< HEAD
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def handle_all_gather_into_tensor(args, kwargs):
=======
    def handle_all_gather_into_tensor(args, kwargs) -> torch.Tensor:
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        dim = args[0].local_sizes()[0][1]
        cat_tensor = torch.cat(
            [t.view(-1) for t in args[0].local_shards()], dim=0
        ).view(-1, dim)
        return torch.ops._c10d_functional.all_gather_into_tensor.default(
            cat_tensor, *args[1:], **kwargs
        )

    @staticmethod
<<<<<<< HEAD
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def handle_wait_tensor(args, kwargs):
        return torch.ops._c10d_functional.wait_tensor(args[0])

    @staticmethod
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def handle_to_copy(args, kwargs):
=======
    def handle_wait_tensor(args, kwargs) -> torch.Tensor:
        return torch.ops._c10d_functional.wait_tensor(args[0])

    @staticmethod
    def handle_to_copy(args, kwargs) -> torch.Tensor:
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        res_shards_list = [
            aten._to_copy.default(shard, *args[1:], **kwargs)
            for shard in args[0].local_shards()
        ]
        return LocalShardsWrapper(res_shards_list, args[0].local_offsets())

    @staticmethod
<<<<<<< HEAD
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def handle_view(args, kwargs):
        # TODO, do we need to change the shape of associated offsets?
        res_shards_list = [
            aten.view.default(shard, args[1], **kwargs)
            for shard in args[0].local_shards()
        ]
        return LocalShardsWrapper(res_shards_list, args[0].local_offsets())

    @staticmethod
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def handle_equal(args, kwargs):
=======
    def handle_view(args, kwargs) -> "LocalShardsWrapper":
        view_shape = args[1]
        res_shards_list = []
        if len(args[0].local_shards()) > 1:
            if args[0].local_shards()[0].ndim == 2:
                assert (
                    args[0].storage_metadata().size[0] == view_shape[0]
                    and args[0].storage_metadata().size[1] == view_shape[1]
                )
                # This accounts for a DTensor quirk, when multiple shards are present on a rank, DTensor on
                # init calls view_as() on the global tensor shape
                # will fail because the view shape is not applicable to individual shards.
                res_shards_list = [
                    aten.view.default(shard, shard.shape, **kwargs)
                    for shard in args[0].local_shards()
                ]
            elif args[0].local_shards()[0].ndim == 1:
                assert args[0].storage_metadata().size[0] == view_shape[0]
                # This case is for optimizer sharding as regardles of sharding type, optimizer state is row wise sharded
                res_shards_list = [
                    aten.view.default(shard, shard.shape, **kwargs)
                    for shard in args[0].local_shards()
                ]
            else:
                raise NotImplementedError("No support for view on tensors ndim > 2")
        else:
            # view is called per shard
            res_shards_list = [
                aten.view.default(shard, args[1], **kwargs)
                for shard in args[0].local_shards()
            ]
        return LocalShardsWrapper(res_shards_list, args[0].local_offsets())

    @staticmethod
    def handle_equal(args, kwargs) -> bool:
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        """
        LocalShardsWrapper equal impl also checks for equality of storage metadata
        and the order of shards
        """
        a, b = args[0], args[1]
        if len(a.local_shards()) != len(b.local_shards()):
            return False
        if not all(
            aten.equal.default(x, y) for x, y in zip(a.local_shards(), b.local_shards())
        ):
            return False
        if not a.storage_metadata() == b.storage_metadata():
            return False
        return True

    @staticmethod
<<<<<<< HEAD
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def handle_detach(args, kwargs):
=======
    def handle_detach(args, kwargs) -> "LocalShardsWrapper":
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        self_ls = args[0]
        deatched_local_shards = [
            aten.detach.default(shard) for shard in self_ls.local_shards()
        ]
        self_ls._local_shards = deatched_local_shards
        self_ls._storage_meta.properties.requires_grad = False
        return self_ls

    @staticmethod
<<<<<<< HEAD
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def handle_clone(args, kwargs):
=======
    def handle_clone(args, kwargs) -> "LocalShardsWrapper":
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        self_ls = args[0]
        desired_memory_format = kwargs.get("memory_format", None)
        if desired_memory_format and desired_memory_format != torch.preserve_format:
            raise NotImplementedError(
                f"{desired_memory_format} is not supported for LocalShardsWrapper!"
            )
        cloned_local_shards = [
            shard.clone(memory_format=desired_memory_format)
            for shard in self_ls._local_shards
        ]
        return LocalShardsWrapper(cloned_local_shards, self_ls.local_offsets())

<<<<<<< HEAD
    @property
    def device(self) -> torch._C.device:  # type: ignore[override]
        return self._local_shards[0].device

    @property
    def is_meta(self) -> bool:  # type: ignore[override]
        return self._local_shards[0].is_meta

    # pyre-ignore[14]
    def is_pinned(self) -> bool:  # type: ignore[override]
        return self._storage_meta.properties.pin_memory

    # pyre-ignore[14]
=======
    @staticmethod
    def handle_new_empty(args, kwargs) -> "LocalShardsWrapper":
        self_ls = args[0]
        return LocalShardsWrapper(
            [torch.empty_like(shard) for shard in self_ls._local_shards],
            self_ls.local_offsets(),
        )

    @property
    def device(self) -> torch._C.device:  # type: ignore[override]
        return (
            self._local_shards[0].device if self._local_shards else torch.device("meta")
        )

    @property
    def is_meta(self) -> bool:  # type: ignore[override]
        return self._local_shards[0].is_meta if self._local_shards else True

    def is_pinned(self) -> bool:  # type: ignore[override]
        return self._storage_meta.properties.pin_memory

>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    def requires_grad_(self, requires_grad: bool = True) -> "LocalShardsWrapper":
        self._storage_meta.properties.requires_grad = requires_grad
        [shard.requires_grad_(requires_grad) for shard in self._local_shards]
        return self

    def local_shards(self) -> list[torch.Tensor]:
        """
        Returns a list of :class:`torch.Tensor' corresponding to the
        local shards for this rank. Returns an empty list if the current rank
        does not host any shards for this Tensor.
        """
        return self._local_shards

    def local_sizes(self) -> list[torch.Size]:
        """
        Returns a list of :class:`torch.Size' corresponding to the
        local sizes for the shards on this rank. Returns an empty list if the current rank
        does not host any shards for this Tensor.
        """
        return [chunk.sizes for chunk in self._storage_meta.chunks]

    def local_offsets(self) -> list[torch.Size]:
        """
        Returns a list of :class:`torch.Size' corresponding to the
        local offsets for the shards on this rank. Returns an empty list if the current rank
        does not host any shards for this Tensor.
        """
        return [chunk.offsets for chunk in self._storage_meta.chunks]

    @property
    def local_chunks(self) -> list[ChunkStorageMetadata]:
        """
<<<<<<< HEAD
        Returns a :class:`List[ChunkStorageMetadata]` object corresponding to the
=======
        Returns a :class:`list[ChunkStorageMetadata]` object corresponding to the
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        metadata for each tensor shard
        """
        return self._storage_meta.chunks

    def storage_metadata(self) -> TensorStorageMetadata:
        """
        Returns a :class:`TensorStorageMetadata` object corresponding to the
        metadata for the local tensor on current rank
        """
        return self._storage_meta

<<<<<<< HEAD
    def __create_write_items__(
        self, fqn: str, object: Any
    ) -> list[WriteItem]:  # pyre-ignore[2]
=======
    def is_empty_shard(self) -> bool:
        """
        Returns a :class:`bool` object indicating if the local tensor on current rank
        is an empty tensor
        """
        return self._storage_meta.size[0] == 0 and self._storage_meta.size[1] == 0

    def __create_write_items__(self, fqn: str, object: Any) -> list[WriteItem]:
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        """
        For compatibility with DCP, we support creation of WriteItems
        such that they can be saved properly.
        """
        return [
            WriteItem(
                index=MetadataIndex(fqn, chunks.offsets),
                type=WriteItemType.SHARD,
                tensor_data=TensorWriteData(
                    chunk=ChunkStorageMetadata(
                        offsets=chunks.offsets,
                        sizes=chunks.sizes,
                    ),
                    properties=self._storage_meta.properties,
                    size=object.size(),
                ),
            )
            for tensor, chunks in zip(self.local_shards(), self.local_chunks)
        ]

    def __create_chunk_list__(self) -> list[ChunkStorageMetadata]:
        """
        For compatibility with DCP, we support creation of chunk lists
        such that they can be saved properly.
        """
        return self._storage_meta.chunks

    def __get_tensor_shard__(self, index: MetadataIndex) -> torch.Tensor:
        """
        For compatibility with DCP, we support finding shard based on index
        Return a 'torch.Tensor' shard based on 'MetadataIndex'.
        """
        # Fast lookup path
        if index.index is not None:
            if (
                len(self._local_shards) > index.index
                and self._storage_meta.chunks[index.index].offsets == index.offset
            ):
                return self._local_shards[index.index]

        if index.offset is not None:
            for shard, chunk in zip(self._local_shards, self._storage_meta.chunks):
                if chunk.offsets == index.offset:
                    return shard

<<<<<<< HEAD
=======
        # Empty shard case
        if len(self._local_shards) == 0 and self._storage_meta.chunks[
            0
        ].sizes == torch.Size([0, 0]):
            return torch.empty(0)

>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        raise ValueError(
            f"Could not find shard at '{index.offset}' for FQN: '{index.fqn}'"
        )

    def _get_tensor_size_bytes(self) -> int:
        object_size = 0
        for shard in self.local_shards():
            object_size += shard.nelement() * shard.element_size()
        return object_size

<<<<<<< HEAD
    # pyre-fixme[3]: Return type must be annotated.
    def __hash__(self):
        return id(self)

    # pyre-fixme[14]: `__repr__` overrides method defined in `torch._tensor.Tensor` inconsistently.
    # pyre-fixme[3]: Return type must be annotated.
=======
    def __hash__(self) -> int:
        return id(self)

>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    def __repr__(self) -> str:  # type: ignore[override]
        return f"LocalShardsWrapper:{self._local_shards} {self._storage_meta}"

    def __str__(self) -> str:
        return f"LocalShardsWrapper:{self._local_shards} {self._storage_meta}"
