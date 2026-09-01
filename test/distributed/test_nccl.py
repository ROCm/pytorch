# Owner(s): ["oncall: distributed"]

import gc
import os
import sys
import threading
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar
from unittest import SkipTest

import torch
import torch.cuda
import torch.cuda.nccl as nccl
import torch.distributed as c10d
import torch.distributed._symmetric_memory as symm_mem
from torch.cuda._utils import (
    _check_cuda_bindings,
    _cuda_bindings_driver as _drv,
    _HAS_CUDA_BINDINGS,
)
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    MultiProcessTestCase,
    PLATFORM_SUPPORTS_SYMM_MEM,
    requires_nccl,
    requires_nccl_shrink,
    requires_nccl_version,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    getRocmVersion,
    instantiate_parametrized_tests,
    IS_WINDOWS,
    load_tests,
    NoTest,
    parametrize,
    requires_cuda_p2p_access,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_ROCM,
    TestCase,
)


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127

nGPUs = torch.cuda.device_count()
if not TEST_CUDA:
    print("CUDA not available, skipping tests", file=sys.stderr)
    TestCase = NoTest


datatypes = [torch.float]
if (
    TEST_CUDA and c10d.is_nccl_available() and nccl.version() >= (2, 10)
) or TEST_WITH_ROCM:
    datatypes.append(torch.bfloat16)

# Broadcast (and alltoall) support float8, while reduce and allreduce do not support float8 currently
broadcast_dtypes = (
    datatypes + [torch.float8_e4m3fnuz, torch.float8_e5m2fnuz]
    if TEST_WITH_ROCM
    else [torch.float8_e4m3fn, torch.float8_e5m2]
)


class TestNCCL(TestCase):
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    def test_unique_id(self, device):
        uid = nccl.unique_id()
        self.assertIsInstance(uid, bytes)
        self.assertGreater(len(uid), 1)

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "only one GPU detected")
    @dtypes(*broadcast_dtypes)
    def test_broadcast(self, device, dtype):
        expected = torch.zeros(128).uniform_().to(dtype=dtype)
        tensors = [expected.cuda()]
        for device in range(1, torch.cuda.device_count()):
            tensors.append(torch.zeros(128, dtype=dtype, device=device))

        nccl.broadcast(tensors)
        for i in range(torch.cuda.device_count()):
            self.assertEqual(tensors[i], expected)

        # Test with tuple
        tensors = [expected.cuda()]
        for device in range(1, torch.cuda.device_count()):
            tensors.append(torch.zeros(128, dtype=dtype, device=device))

        nccl.broadcast(tuple(tensors))
        for i in range(torch.cuda.device_count()):
            self.assertEqual(tensors[i], expected)

        # Test with a non-zero root (regression test for #179908)
        root = nGPUs - 1
        expected = torch.zeros(128).uniform_().to(dtype=dtype)
        tensors = [
            expected.cuda(device)
            if device == root
            else torch.zeros(128, dtype=dtype, device=device)
            for device in range(nGPUs)
        ]

        nccl.broadcast(tensors, root=root)
        for i in range(nGPUs):
            self.assertEqual(tensors[i], expected)

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "only one GPU detected")
    @dtypes(*datatypes)
    def test_reduce(self, device, dtype):
        cpu_tensors = [
            torch.zeros(128).uniform_().to(dtype=dtype) for i in range(nGPUs)
        ]
        expected = torch.zeros(128, dtype=dtype)
        for t in cpu_tensors:
            expected.add_(t)

        tensors = [cpu_tensors[i].cuda(i) for i in range(nGPUs)]
        nccl.reduce(tensors)

        self.assertEqual(tensors[0], expected)

        # Test with tuple
        tensors = [cpu_tensors[i].cuda(i) for i in range(nGPUs)]
        nccl.reduce(tuple(tensors))

        self.assertEqual(tensors[0], expected)

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "only one GPU detected")
    @dtypes(*datatypes)
    def test_all_reduce(self, device, dtype):
        cpu_tensors = [
            torch.zeros(128).uniform_().to(dtype=dtype) for i in range(nGPUs)
        ]
        expected = torch.zeros(128, dtype=dtype)
        for t in cpu_tensors:
            expected.add_(t)

        tensors = [cpu_tensors[i].cuda(i) for i in range(nGPUs)]
        nccl.all_reduce(tensors)

        for tensor in tensors:
            self.assertEqual(tensor, expected)

        # Test with tuple.
        tensors = tuple(cpu_tensors[i].cuda(i) for i in range(nGPUs))
        nccl.all_reduce(tensors)

        for tensor in tensors:
            self.assertEqual(tensor, expected)

        # Test with set.
        tensors = {cpu_tensors[i].cuda(i) for i in range(nGPUs)}
        nccl.all_reduce(tensors)

        for tensor in tensors:
            self.assertEqual(tensor, expected)

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    def test_collective_errors(self, device):
        t = torch.rand(10).cuda(0)
        with self.assertRaisesRegex(
            TypeError, "Inputs should be a collection of tensors"
        ):
            nccl.all_reduce(t)

        with self.assertRaisesRegex(
            TypeError, "Inputs should be a collection of tensors"
        ):
            nccl.reduce(t)

        with self.assertRaisesRegex(
            TypeError, "Inputs should be a collection of tensors"
        ):
            nccl.broadcast(t)

        with self.assertRaisesRegex(
            TypeError, "Inputs should be a collection of tensors"
        ):
            nccl.all_gather(t, t)

        with self.assertRaisesRegex(
            TypeError, "Inputs should be a collection of tensors"
        ):
            nccl.reduce_scatter(t, t)

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "only one GPU detected")
    @dtypes(*datatypes)
    def test_all_gather(self, device, dtype):
        cpu_inputs = [torch.zeros(128).uniform_().to(dtype=dtype) for i in range(nGPUs)]
        expected = torch.cat(cpu_inputs, 0)

        inputs = [cpu_inputs[i].cuda(i) for i in range(nGPUs)]
        outputs = [
            torch.zeros(128 * nGPUs, device=i, dtype=dtype) for i in range(nGPUs)
        ]
        nccl.all_gather(inputs, outputs)

        for tensor in outputs:
            self.assertEqual(tensor, expected)

        # Test with tuple.
        inputs = [cpu_inputs[i].cuda(i) for i in range(nGPUs)]
        outputs = [
            torch.zeros(128 * nGPUs, device=i, dtype=dtype) for i in range(nGPUs)
        ]
        nccl.all_gather(tuple(inputs), tuple(outputs))

        for tensor in outputs:
            self.assertEqual(tensor, expected)

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "only one GPU detected")
    @dtypes(*datatypes)
    def test_reduce_scatter(self, device, dtype):
        in_size = 32 * nGPUs
        out_size = 32

        cpu_inputs = [
            torch.zeros(in_size).uniform_().to(dtype=dtype) for i in range(nGPUs)
        ]
        expected = torch.zeros(in_size, dtype=dtype)
        for t in cpu_inputs:
            expected.add_(t)
        expected = expected.view(nGPUs, 32)

        inputs = [cpu_inputs[i].cuda(i) for i in range(nGPUs)]
        outputs = [torch.zeros(out_size, device=i, dtype=dtype) for i in range(nGPUs)]
        nccl.reduce_scatter(inputs, outputs)

        for i in range(nGPUs):
            self.assertEqual(outputs[i], expected[i])

        # Test with tuple
        inputs = [cpu_inputs[i].cuda(i) for i in range(nGPUs)]
        outputs = [torch.zeros(out_size, device=i, dtype=dtype) for i in range(nGPUs)]
        nccl.reduce_scatter(tuple(inputs), tuple(outputs))

        for i in range(nGPUs):
            self.assertEqual(outputs[i], expected[i])


@instantiate_parametrized_tests
@requires_cuda_p2p_access()
@skip_but_pass_in_sandcastle_if(
    TEST_WITH_ROCM and nccl.version() < (2, 30, 4),
    "RCCL host device APIs require RCCL 2.30.4 or newer",
)
@skip_but_pass_in_sandcastle_if(
    TEST_WITH_ROCM and getRocmVersion() == (7, 14),
    "RCCL CUMEM/P2P communicator init fails on ROCm 7.14 (p2p_tmp.cc:358)",
)
class NCCLSymmetricMemoryTest(MultiProcContinuousTest):
    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file):
        # Eager NCCL communicator init via device_id, so symm_mem rendezvous
        # does not require a separate warm-up collective.
        if rdvz_file is None:
            raise AssertionError("Expected rdvz_file to not be None")
        os.environ["LOCAL_RANK"] = str(rank)
        if TEST_WITH_ROCM:
            # RCCL symmetric-memory windows require VMM (cuMem) and window
            # registration. setdefault so an explicit override (e.g.
            # NCCL_CUMEM_ENABLE=0 to exercise the fail-fast path in the NCCL
            # symmetric-memory backend) is respected; the backend raises an
            # actionable error at rendezvous if either is disabled.
            os.environ.setdefault("NCCL_CUMEM_ENABLE", "1")
            os.environ.setdefault("NCCL_WIN_ENABLE", "1")
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        store = c10d.FileStore(rdvz_file, world_size)
        c10d.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            store=store,
            timeout=cls.timeout,
            device_id=device,
        )
        cls.pg = c10d.distributed_c10d._get_default_group()

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7) if TEST_WITH_ROCM else (2, 27),
        "NCCL/RCCL symmetric-memory API version requirement",
    )
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_alloc(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        # Need this all_reduce to initialize NCCL communicator. Otherwise, the
        # test will hang.  TODO: investigate how NCCLSymmetricMemory can
        # initialize NCCL communicator.
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        dtype = torch.float
        numel = 1024

        def foo():
            inp = symm_mem.empty(numel, dtype=dtype, device=self.device)
            symm_mem.rendezvous(inp, group=group_name)

        foo()

        out = symm_mem.empty(numel, dtype=dtype, device=self.device)
        symm_mem.rendezvous(out, group=group_name)

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7) if TEST_WITH_ROCM else (2, 27),
        "NCCL/RCCL symmetric-memory API version requirement",
    )
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_rendezvous_many_allocations(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        tensors = [
            symm_mem.empty(1, dtype=torch.float, device=self.device) for _ in range(256)
        ]

        # Rendezvous a subset twice so the repeated lookup path is covered
        # while many allocations are still live.
        sampled_tensors = tensors[::16]
        for tensor in sampled_tensors:
            handle = symm_mem.rendezvous(tensor, group=group_name)
            self.assertEqual(handle.rank, self.rank)
            self.assertEqual(handle.world_size, self.world_size)
        for tensor in sampled_tensors:
            symm_mem.rendezvous(tensor, group=group_name)

        result = torch.ops.symm_mem.one_shot_all_reduce(
            tensors[-1].fill_(self.rank), "sum", group_name
        )
        self.assertEqual(
            result, torch.full_like(result, (self.world_size - 1) * self.world_size / 2)
        )

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7) if TEST_WITH_ROCM else (2, 27),
        "NCCL/RCCL symmetric-memory API version requirement",
    )
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_rendezvous_world(self):
        symm_mem.set_backend("NCCL")
        group_name = c10d.group.WORLD.group_name

        t = symm_mem.empty(64, device=self.device)
        handle = symm_mem.rendezvous(t, group=group_name)

        self.assertEqual(handle.world_size, self.world_size)
        self.assertEqual(handle.rank, self.rank)

        t.fill_(self.rank)
        c10d.barrier()

        peer_rank = (self.rank + 1) % self.world_size
        buf = handle.get_buffer(peer_rank, (64,), torch.float32)
        self.assertTrue(buf.eq(peer_rank).all())

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7) if TEST_WITH_ROCM else (2, 27),
        "NCCL/RCCL symmetric-memory API version requirement",
    )
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_barrier(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        numel = 64
        t = symm_mem.empty(numel, dtype=torch.float32, device=self.device).fill_(
            self.rank
        )
        handle = symm_mem.rendezvous(t, group=group_name)
        self.assertEqual(handle.rank, self.rank)
        self.assertEqual(handle.world_size, self.world_size)

        handle.barrier()
        for peer in range(self.world_size):
            buf = handle.get_buffer(peer, (numel,), torch.float32)
            self.assertTrue(buf.eq(peer).all())
        handle.barrier()

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7) if TEST_WITH_ROCM else (2, 28),
        "NCCL/RCCL symmetric-memory API version requirement",
    )
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_barrier_channel_out_of_bounds(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        t = symm_mem.empty(64, dtype=torch.float32, device=self.device)
        handle = symm_mem.rendezvous(t, group=group_name)

        num_slots = handle.signal_pad_size // 4
        max_channel = num_slots // self.world_size

        # check_channel() is shared with the CUDA backend; an over-capacity
        # channel must be rejected host-side before the kernel launch (#191618).
        with self.assertRaisesRegex(RuntimeError, "maximum supported channel"):
            handle.barrier(channel=max_channel)
        handle.barrier(channel=max_channel - 1)

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version((2, 29), "NCCL one-sided host API support from nccl 2.29")
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_signal_rank_out_of_bounds(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        t = symm_mem.empty(64, dtype=torch.float32, device=self.device)
        handle = symm_mem.rendezvous(t, group=group_name)

        for bad_rank in (-1, self.world_size):
            with self.assertRaisesRegex(RuntimeError, r"must be in \[0"):
                handle.put_signal(dst_rank=bad_rank)
            with self.assertRaisesRegex(RuntimeError, r"must be in \[0"):
                handle.wait_signal(src_rank=bad_rank)

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7) if TEST_WITH_ROCM else (2, 28),
        "NCCL/RCCL symmetric-memory API version requirement",
    )
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_rendezvous_subgroup(self):
        symm_mem.set_backend("NCCL")

        subgroup = c10d.new_group(list(range(self.world_size)))

        t = symm_mem.empty(64, device=self.device)
        handle = symm_mem.rendezvous(t, group=subgroup)

        self.assertEqual(handle.world_size, self.world_size)
        self.assertEqual(handle.rank, self.rank)

        t.fill_(self.rank)
        c10d.barrier(group=subgroup)

        peer_rank = (self.rank + 1) % self.world_size
        buf = handle.get_buffer(peer_rank, (64,), torch.float32)
        self.assertTrue(buf.eq(peer_rank).all())

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7) if TEST_WITH_ROCM else (2, 28),
        "NCCL/RCCL symmetric-memory API version requirement",
    )
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_collective(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        # Need this all_reduce to initialize NCCL communicator. Otherwise, the
        # test will hang.  TODO: investigate how NCCLSymmetricMemory can
        # initialize NCCL communicator.
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        dtype = torch.float
        numel = 1024

        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        symm_mem.rendezvous(out, group=group_name)
        c10d.all_reduce(out)
        torch.cuda.synchronize()
        self.assertEqual(
            out, torch.full_like(out, (self.world_size - 1) * self.world_size / 2)
        )

        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        symm_mem.rendezvous(inp, group=group_name)
        res = torch.ops.symm_mem.one_shot_all_reduce(inp, "sum", group_name)
        self.assertEqual(out, res)

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7) if TEST_WITH_ROCM else (2, 28),
        "NCCL/RCCL symmetric-memory API version requirement",
    )
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_collective_cuda_graph(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        # Need this all_reduce to initialize NCCL communicator. Otherwise, the
        # test will hang.  TODO: investigate how NCCLSymmetricMemory can
        # initialize NCCL communicator.
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        dtype = torch.float
        numel = 1024

        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        symm_mem.rendezvous(out, group=group_name)
        graph_all_reduce = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_all_reduce):
            c10d.all_reduce(out)
        graph_all_reduce.replay()
        self.assertEqual(
            out, torch.full_like(out, (self.world_size - 1) * self.world_size / 2)
        )

        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        symm_mem.rendezvous(inp, group=group_name)
        graph_one_shot_all_reduce = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_one_shot_all_reduce):
            res = torch.ops.symm_mem.one_shot_all_reduce(inp, "sum", group_name)
        graph_one_shot_all_reduce.replay()
        self.assertEqual(out, res)

        for repeat in range(3):
            offset = 13 + repeat
            inp.fill_(self.rank + offset)
            out.fill_(self.rank + offset)
            res.fill_(0.0)
            expected_sum = float(
                self.world_size * offset + self.world_size * (self.world_size - 1) / 2
            )
            graph_all_reduce.replay()
            graph_one_shot_all_reduce.replay()
            self.assertEqual(out, torch.full_like(out, expected_sum))
            self.assertEqual(res, out)

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7) if TEST_WITH_ROCM else (2, 28),
        "NCCL/RCCL symmetric-memory API version requirement",
    )
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_tensor_creation_and_collective_cuda_graph(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        dtype = torch.float
        numel = 1024
        capture_offset = torch.tensor(1, dtype=dtype, device=self.device)
        capture_stream = torch.cuda.Stream(device=self.device)

        # Warm up on the same stream that will be captured. This establishes
        # NCCL window metadata for the symmetric block that capture reuses.
        with torch.cuda.stream(capture_stream):
            inp = symm_mem.empty(numel, dtype=dtype, device=self.device)
            out = torch.empty(numel, dtype=dtype, device=self.device)
            for warmup_idx in range(3):
                offset = 1 + warmup_idx
                capture_offset.fill_(self.rank + offset)
                inp = symm_mem.empty(numel, dtype=dtype, device=self.device)
                out = torch.empty(numel, dtype=dtype, device=self.device)
                inp.fill_(capture_offset)
                out.fill_(0.0)
                torch.ops.symm_mem.one_shot_all_reduce_out(inp, "sum", group_name, out)
                expected_sum = float(
                    self.world_size * offset
                    + self.world_size * (self.world_size - 1) / 2
                )
                self.assertEqual(out, torch.full_like(out, expected_sum))
            del inp, out
        capture_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        offset = 13
        capture_offset.fill_(self.rank + offset)
        with torch.cuda.graph(graph, stream=capture_stream):
            inp = symm_mem.empty(numel, dtype=dtype, device=self.device)
            out = torch.empty(numel, dtype=dtype, device=self.device)
            inp.fill_(capture_offset)
            out.fill_(0.0)
            torch.ops.symm_mem.one_shot_all_reduce_out(inp, "sum", group_name, out)

        graph.replay()
        expected_sum = float(
            self.world_size * offset + self.world_size * (self.world_size - 1) / 2
        )
        self.assertEqual(out, torch.full_like(out, expected_sum))

        for repeat in range(3):
            offset = 20 + repeat
            capture_offset.fill_(self.rank + offset)
            graph.replay()
            expected_sum = float(
                self.world_size * offset + self.world_size * (self.world_size - 1) / 2
            )
            self.assertEqual(out, torch.full_like(out, expected_sum))

    @skip_but_pass_in_sandcastle_if(
        not TEST_WITH_ROCM, "ROCm-specific HIP graph allocation behavior"
    )
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7), "ROCm LSA symmetric-memory support from RCCL 2.29.7"
    )
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_cuda_graph_allocation_requires_warmup(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))

        graph = torch.cuda.CUDAGraph()
        with self.assertRaisesRegex(RuntimeError, "without a clean cached block"):
            with torch.cuda.graph(graph):
                # This exact byte size is intentionally not warmed up.
                symm_mem.empty(123457, dtype=torch.float32, device=self.device)

    @skip_but_pass_in_sandcastle_if(
        not TEST_WITH_ROCM, "ROCm-specific HIP graph allocation behavior"
    )
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7), "ROCm LSA symmetric-memory support from RCCL 2.29.7"
    )
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_cached_alloc_on_other_thread_during_capture(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name
        numel, dtype = 1024, torch.float

        # Seed an exact-size cached block whose zero-completion event can be
        # queried by an allocation on another thread.
        warm = symm_mem.empty(numel, dtype=dtype, device=self.device)
        warm_handle = symm_mem.rendezvous(warm, group=group_name)
        warm_ptr = warm.data_ptr()
        del warm_handle, warm
        gc.collect()
        torch.cuda.synchronize(self.device)

        release = threading.Event()
        done = threading.Event()
        reused_tensors = []
        reused_ptrs = []
        errors = []

        def worker():
            torch.cuda.set_device(self.rank)
            release.wait()
            try:
                reused = symm_mem.empty(numel, dtype=dtype, device=self.device)
                reused_ptrs.append(reused.data_ptr())
                # Keep the allocation alive until the main-thread capture ends;
                # freeing it here would test a different event-record path.
                reused_tensors.append(reused)
            except BaseException as exc:
                errors.append(exc)
            finally:
                done.set()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream(device=self.device)
        worker_completed_during_capture = False
        try:
            with torch.cuda.graph(graph, stream=capture_stream):
                # Keep a real node in the graph while the worker reuses the
                # cached block and queries its pre-capture completion event.
                torch.cuda._sleep(1)
                release.set()
                worker_completed_during_capture = done.wait(timeout=30)
        finally:
            release.set()
            thread.join(timeout=30)

        self.assertTrue(
            worker_completed_during_capture,
            "worker allocation did not complete during graph capture",
        )
        self.assertFalse(thread.is_alive(), "worker allocation thread did not exit")
        if errors:
            raise errors[0]
        self.assertEqual(reused_ptrs, [warm_ptr])

        graph.replay()
        capture_stream.synchronize()
        reused_tensors.clear()

    @skip_but_pass_in_sandcastle_if(
        not TEST_WITH_ROCM, "ROCm-specific HIP graph allocation behavior"
    )
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7), "ROCm LSA symmetric-memory support from RCCL 2.29.7"
    )
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_cuda_graph_cache_keys_exact_size(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        # Both payloads fit in the same 4096-byte-aligned NCCL window size.
        # Warming them independently verifies that cache identity uses the exact
        # user size instead of only the rounded allocation size.
        for numel in (100, 101):
            tensor = symm_mem.empty(numel, dtype=torch.float32, device=self.device)
            handle = symm_mem.rendezvous(tensor, group=group_name)
            tensor.fill_(self.rank)
            del handle, tensor
        torch.cuda.synchronize(self.device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            tensor_a = symm_mem.empty(100, dtype=torch.float32, device=self.device)
            handle_a = symm_mem.rendezvous(tensor_a, group=group_name)
            tensor_a.fill_(self.rank + 1)
            tensor_b = symm_mem.empty(101, dtype=torch.float32, device=self.device)
            handle_b = symm_mem.rendezvous(tensor_b, group=group_name)
            tensor_b.fill_(self.rank + 2)

        graph.replay()
        self.assertEqual(tensor_a, torch.full_like(tensor_a, self.rank + 1))
        self.assertEqual(tensor_b, torch.full_like(tensor_b, self.rank + 2))
        del handle_a, handle_b

    @skip_but_pass_in_sandcastle_if(
        not TEST_WITH_ROCM, "ROCm-specific free-block reuse behavior"
    )
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7), "ROCm LSA symmetric-memory support from RCCL 2.29.7"
    )
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_free_then_reuse_rendezvous(self):
        # Freeing a symmetric tensor must erase its cached rendezvous handle
        # from the allocator's symm_mems_ map, while keeping the block's window
        # registration reusable: the freed block comes back from the free cache
        # with the same pointer, and a fresh rendezvous on it builds a new
        # handle without re-registering.
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        tensor = symm_mem.empty(4096, dtype=torch.float32, device=self.device)
        handle = symm_mem.rendezvous(tensor, group=group_name)
        handle.barrier()
        ptr = tensor.data_ptr()

        # Free the tensor but keep the old handle alive. free() must drop the
        # cached entry even though this Python handle still references the
        # SymmetricMemory object.
        del tensor
        torch.cuda.synchronize(self.device)

        reused = symm_mem.empty(4096, dtype=torch.float32, device=self.device)
        # The same block returns from the free cache.
        self.assertEqual(reused.data_ptr(), ptr)
        new_handle = symm_mem.rendezvous(reused, group=group_name)
        # Because free() erased symm_mems_ for this pointer, rendezvous rebuilt
        # a distinct handle rather than returning the stale cached object. The
        # binding preserves Python identity for a live handle, so a cache hit on
        # the reused pointer would have returned `handle` itself.
        self.assertIsNot(new_handle, handle)

        # The recycled block still works end to end across peers.
        reused.fill_(self.rank)
        torch.cuda.synchronize(self.device)
        new_handle.barrier()
        peer_rank = (self.rank + 1) % self.world_size
        buf = new_handle.get_buffer(peer_rank, (4096,), torch.float32)
        self.assertTrue(buf.eq(peer_rank).all())
        new_handle.barrier()
        del handle, new_handle

    @skip_but_pass_in_sandcastle_if(
        not TEST_WITH_ROCM, "ROCm-only: targets the symm-mem owned devcomm cache"
    )
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7), "nccl_reduce_scatter_offset requires nccl 2.29.7"
    )
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_devcomm_group_destroy_recreate(self):
        # Lifetime smoke test for the ROCm owned devcomm cache: repeatedly
        # create a group, build+use a device communicator via
        # reduce_scatter_offset, then destroy the group. Exercises the
        # get_or_create (fresh create) and release(comm) (owner-matched erase)
        # paths across create/destroy cycles with real communicators. (On CUDA
        # the device communicators live in NCCLDevCommManager, so this is ROCm
        # only.)
        #
        # It does NOT pin the same-group-name restart path (stale-owner rebuild
        # in get_or_create when owner != comm, or the leave-successor release):
        # new_group cannot reuse a group name because _hash_ranks_to_str salts
        # the name with the monotonic _world.group_count. That branch is a
        # pointer-identity mirror of the reviewed NCCLDevCommManager
        # unregister_comm(name, comm) pattern; reproducing it would require
        # default-group abort/re-init, which is not worth the fragility here.
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))

        ranks = list(range(self.world_size))
        rows, cols = 64, 32
        n_experts = self.world_size  # experts_per_rank = 1: each rank owns one
        rank_sum = float(sum(r + 1 for r in range(self.world_size)))

        for _ in range(3):
            pg = c10d.new_group(ranks)
            group_name = pg.group_name

            buf = symm_mem.empty(
                n_experts * rows, cols, dtype=torch.float, device=self.device
            )
            for i in range(n_experts):
                buf[i * rows : (i + 1) * rows, :] = float((self.rank + 1) * (i + 1))
            symm_mem.rendezvous(buf, group=group_name)

            dst_ranks = [i % self.world_size for i in range(n_experts)]
            out = [torch.zeros(rows, cols, dtype=torch.float, device=self.device)]
            offsets = [i * rows for i in range(1, n_experts + 1)]
            symm_mem.reduce_scatter_offset(
                buf, out, group_name, dim=0, offsets=offsets, dst_ranks=dst_ranks
            )
            torch.cuda.synchronize()

            # We own expert `self.rank` (j = 0): (expert_idx + 1) * rank_sum.
            expected = float(self.rank + 1) * rank_sum
            self.assertEqual(out[0], torch.full_like(out[0], expected))

            c10d.destroy_process_group(pg)
            del pg, buf, out
            gc.collect()

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7) if TEST_WITH_ROCM else (2, 28),
        "NCCL/RCCL symmetric-memory API version requirement",
    )
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_put(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        # Need this all_reduce to initialize NCCL communicator. Otherwise, the
        # test will hang.  TODO: investigate how NCCLSymmetricMemory can
        # initialize NCCL communicator.
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        dtype = torch.float
        numel = 1024
        tensor = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        # This is needed to make sure we don't get blocked the second time we call rendezvous
        # for the same tensor because it will be cached by that moment.
        symm_mem.rendezvous(tensor, group=group_name)
        signal_val = 5
        c10d.barrier()

        if self.rank == 1:
            torch.ops.symm_mem.nccl_put_with_signal(tensor, signal_val, 0)
        elif self.rank == 0:
            torch.ops.symm_mem.nccl_wait_for_signal(tensor, signal_val)
            torch.testing.assert_close(
                tensor, torch.ones(numel, dtype=dtype, device=self.device)
            )
        c10d.barrier()
        if self.rank == 1:
            tensor *= 2
            torch.ops.symm_mem.nccl_put(tensor, 0)
            c10d.barrier()
        else:
            c10d.barrier()
        if self.rank == 0:
            torch.testing.assert_close(
                tensor, torch.ones(numel, dtype=dtype, device=self.device) * 2
            )

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version((2, 29), "NCCL one-sided host API support from nccl 2.29")
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_handle_signal(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        # need this all_reduce to initialize NCCL communicator. Otherwise, the
        # test will hang.  TODO: investigate how NCCLSymmetricMemory can
        # initialize NCCL communicator.
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        dtype = torch.float
        numel = 1024
        tensor = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        handle = symm_mem.rendezvous(tensor, group=group_name)

        channel = 0
        world_size = handle.world_size

        c10d.barrier()

        # Pair up ranks: odd ranks send to even ranks
        # This allows the test to work with any number of GPUs
        if self.rank % 2 == 1:
            # Odd rank: send signal to previous even rank
            dst_rank = self.rank - 1
            handle.put_signal(dst_rank=dst_rank, channel=channel)
            torch.cuda.synchronize()
        elif self.rank % 2 == 0 and self.rank + 1 < world_size:
            # Even rank: wait for signal from next odd rank (if it exists)
            src_rank = self.rank + 1
            # wait_signal blocks until the signal arrives
            # If this completes without hanging, the test passes
            handle.wait_signal(src_rank=src_rank, channel=channel)
            torch.cuda.synchronize()

        c10d.barrier()

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7) if TEST_WITH_ROCM else (2, 28),
        "NCCL/RCCL symmetric-memory API version requirement",
    )
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_get(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        # Need this all_reduce to initialize NCCL communicator. Otherwise, the
        # test will hang.  TODO: investigate how NCCLSymmetricMemory can
        # initialize NCCL communicator.
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        dtype = torch.float
        numel = 1024
        tensor = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        # This is needed to make sure we don't get blocked the second time we call rendezvous
        # for the same tensor because it will be cached by that moment.
        symm_mem.rendezvous(tensor, group=group_name)
        c10d.barrier()
        if self.rank == 0:
            torch.ops.symm_mem.nccl_get(tensor, 1)
            # TODO: remove after we have wait_signal
            c10d.barrier()
            torch.testing.assert_close(
                tensor, torch.ones(numel, dtype=dtype, device=self.device)
            )
        else:
            # handle.wait_signal(src_rank=0)
            # TODO: remove after we have wait_signal
            c10d.barrier()

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7) if TEST_WITH_ROCM else (2, 28),
        "NCCL/RCCL symmetric-memory API version requirement",
    )
    @skip_if_lt_x_gpu(2)
    def test_get(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        dtype = torch.float
        numel = 1024

        # Full-buffer get from a peer's allocation.
        src = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        hdl = symm_mem.rendezvous(src, group=group_name)
        c10d.barrier()

        if self.rank == 0:
            dst = torch.empty_like(src)
            symm_mem.get(dst, hdl, peer=1)
            torch.testing.assert_close(dst, torch.ones_like(dst))

        c10d.barrier()

        # Offset get: copy a sub-region of the peer's allocation.
        src_base = symm_mem.empty(2 * numel, dtype=dtype, device=self.device)
        src_base.copy_(
            torch.arange(2 * numel, dtype=dtype, device=self.device)
            + self.rank * 2 * numel
        )
        hdl = symm_mem.rendezvous(src_base, group=group_name)
        c10d.barrier()

        if self.rank == 0:
            offset = numel // 2
            dst = torch.empty(numel, dtype=dtype, device=self.device)
            symm_mem.get(dst, hdl, peer=1, offset=offset)
            expected = (
                torch.arange(offset, offset + numel, dtype=dtype, device=self.device)
                + 2 * numel
            )
            torch.testing.assert_close(dst, expected)

            # Filling a sub-region: pass a view; the rest of dst is untouched.
            larger_dst = torch.full((numel + 1,), -1, dtype=dtype, device=self.device)
            symm_mem.get(larger_dst[:numel], hdl, peer=1, offset=offset)
            self.assertEqual(larger_dst[:numel], expected)
            self.assertEqual(larger_dst[numel], -1)

            noncontig_dst = torch.empty(2 * numel, dtype=dtype, device=self.device)[::2]
            with self.assertRaisesRegex(ValueError, "contiguous"):
                symm_mem.get(noncontig_dst, hdl, peer=1)

            with self.assertRaisesRegex(ValueError, "non-negative"):
                symm_mem.get(
                    torch.empty(numel, dtype=dtype, device=self.device),
                    hdl,
                    peer=1,
                    offset=-1,
                )

            with self.assertRaisesRegex(ValueError, "exceeds"):
                symm_mem.get(
                    torch.empty(1, dtype=dtype, device=self.device),
                    hdl,
                    peer=1,
                    offset=hdl.buffer_size // dst.element_size(),
                )

            with self.assertRaisesRegex(ValueError, "invalid peer"):
                symm_mem.get(
                    torch.empty(numel, dtype=dtype, device=self.device),
                    hdl,
                    peer=hdl.world_size,
                )

        c10d.barrier()

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7), "nccl_reduce_scatter_offset requires nccl 2.29.7"
    )
    @skip_if_lt_x_gpu(2)
    @parametrize("experts_per_rank", [1, 2])
    @parametrize("dim", [0, 1])
    def test_reduce_scatter_offset(self, experts_per_rank: int, dim: int):
        """reduce_scatter_offset: each expert gradient is reduced to its
        destination rank and written to a separate contiguous tensor; the source
        Grouped GEMM buffer is left unmodified."""
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        rows, cols = 64, 32
        n_experts = experts_per_rank * self.world_size

        # dim=1: experts laid out as column blocks [rows, n_experts * cols]
        # dim=0: experts laid out as row blocks    [n_experts * rows, cols]
        if dim == 1:
            buf = symm_mem.empty(
                rows, n_experts * cols, dtype=torch.float, device=self.device
            )
            for i in range(n_experts):
                buf[:, i * cols : (i + 1) * cols] = float((self.rank + 1) * (i + 1))
        else:
            buf = symm_mem.empty(
                n_experts * rows, cols, dtype=torch.float, device=self.device
            )
            for i in range(n_experts):
                buf[i * rows : (i + 1) * rows, :] = float((self.rank + 1) * (i + 1))
        symm_mem.rendezvous(buf, group=group_name)

        # Round-robin: expert i is reduced to rank i % world_size.
        dst_ranks = [i % self.world_size for i in range(n_experts)]
        n_owned = sum(r == self.rank for r in dst_ranks)
        out = [
            torch.zeros(rows, cols, dtype=torch.float, device=self.device)
            for _ in range(n_owned)
        ]
        block_size = cols if dim == 1 else rows
        offsets = [i * block_size for i in range(1, n_experts + 1)]

        symm_mem.reduce_scatter_offset(
            buf, out, group_name, dim=dim, offsets=offsets, dst_ranks=dst_ranks
        )
        torch.cuda.synchronize()

        # out[j] corresponds to expert (rank + j * world_size); expected value is
        # (expert_idx + 1) * sum(r + 1 for r in range(world_size)).
        rank_sum = float(sum(r + 1 for r in range(self.world_size)))
        for j in range(n_owned):
            expert_idx = self.rank + j * self.world_size
            expected = float(expert_idx + 1) * rank_sum
            self.assertEqual(
                out[j],
                torch.full_like(out[j], expected),
                msg=lambda msg: f"{msg}\nrank {self.rank}: out[{j}] should contain the reduced sum",
            )
        # Source buffer must be unmodified.
        for i in range(n_experts):
            if dim == 1:
                src_slice = buf[:, i * cols : (i + 1) * cols]
            else:
                src_slice = buf[i * rows : (i + 1) * rows, :]
            self.assertEqual(
                src_slice,
                torch.full(
                    (rows, cols),
                    float((self.rank + 1) * (i + 1)),
                    dtype=torch.float,
                    device=self.device,
                ),
                msg=lambda msg: f"{msg}\nrank {self.rank}: source buffer block {i} should be unchanged",
            )

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7), "nccl_reduce_scatter_offset requires nccl 2.29.7"
    )
    @skip_if_lt_x_gpu(2)
    @parametrize("dim", [0, 1])
    def test_reduce_scatter_offset_uneven(self, dim: int):
        """reduce_scatter_offset with uneven block sizes: j=0 and j=1 own blocks
        of different sizes, verifying that out[j] shapes differ across j."""
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        rows, cols = 64, 32
        # j=0 blocks have size_0 along dim; j=1 blocks have size_1 along dim.
        # Arrange blocks as [size_0] * world_size + [size_1] * world_size so
        # that round-robin assigns each rank exactly one block of each size.
        size_0, size_1 = 16, 48
        block_sizes = [size_0] * self.world_size + [size_1] * self.world_size
        offsets = []
        total = 0
        for sz in block_sizes:
            total += sz
            offsets.append(total)

        n_experts = 2 * self.world_size
        if dim == 1:
            buf = symm_mem.empty(rows, total, dtype=torch.float, device=self.device)
            pos = 0
            for i, sz in enumerate(block_sizes):
                buf[:, pos : pos + sz] = float((self.rank + 1) * (i + 1))
                pos += sz
        else:
            buf = symm_mem.empty(total, cols, dtype=torch.float, device=self.device)
            pos = 0
            for i, sz in enumerate(block_sizes):
                buf[pos : pos + sz, :] = float((self.rank + 1) * (i + 1))
                pos += sz
        symm_mem.rendezvous(buf, group=group_name)

        dst_ranks = [i % self.world_size for i in range(n_experts)]
        if dim == 1:
            out = [
                torch.zeros(rows, size_0, dtype=torch.float, device=self.device),
                torch.zeros(rows, size_1, dtype=torch.float, device=self.device),
            ]
        else:
            out = [
                torch.zeros(size_0, cols, dtype=torch.float, device=self.device),
                torch.zeros(size_1, cols, dtype=torch.float, device=self.device),
            ]

        symm_mem.reduce_scatter_offset(
            buf, out, group_name, dim=dim, offsets=offsets, dst_ranks=dst_ranks
        )
        torch.cuda.synchronize()

        rank_sum = float(sum(r + 1 for r in range(self.world_size)))
        for j in range(2):
            expert_idx = self.rank + j * self.world_size
            expected = float(expert_idx + 1) * rank_sum
            self.assertEqual(
                out[j],
                torch.full_like(out[j], expected),
                msg=lambda msg: f"{msg}\nrank {self.rank}: out[{j}] should contain the reduced sum",
            )

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7) if TEST_WITH_ROCM else (2, 28, 0),
        "NCCL/RCCL symmetric-memory API version requirement",
    )
    @skip_if_lt_x_gpu(2)
    @parametrize(
        "scatter_gather,out_2d,input_3d",
        [
            ((1, 0), False, False),
            ((1, 0), True, False),
            ((1, 0), False, True),
            ((1, 0), True, True),
            ((0, 1), False, False),
            ((0, 1), True, False),
            ((0, 1), False, True),
            ((0, 1), True, True),
        ],
    )
    def test_all_to_all_nd(self, scatter_gather, out_2d, input_3d):
        """all_to_all_nd: (1,0)/(0,1); 3-D input [rows,G,loc] or [G,loc,cols] where supported."""
        scatter_dim, gather_dim = scatter_gather
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        p = self.world_size
        dtype = torch.float

        if scatter_dim == 1 and gather_dim == 0:
            local_cols = 4
            rows = 8
            if input_3d:
                buf = symm_mem.empty(
                    rows, p, local_cols, dtype=dtype, device=self.device
                ).fill_(float(self.rank))
            else:
                buf = symm_mem.empty(
                    rows, p * local_cols, dtype=dtype, device=self.device
                ).fill_(float(self.rank))
            symm_mem.rendezvous(buf, group=group_name)
            if out_2d:
                out = torch.empty(p * rows, local_cols, dtype=dtype, device=self.device)
            else:
                out = torch.empty(p, rows, local_cols, dtype=dtype, device=self.device)
            symm_mem.all_to_all_nd(
                buf,
                out,
                scatter_dim=scatter_dim,
                gather_dim=gather_dim,
                group=group_name,
            )
            torch.cuda.synchronize()
            out_view = out.view(p, rows, local_cols) if out_2d else out
            for j in range(p):
                self.assertEqual(
                    out_view[j],
                    torch.full(
                        (rows, local_cols),
                        float(j),
                        dtype=dtype,
                        device=self.device,
                    ),
                    msg=f"rank {self.rank}: out[{j}] should be peer {j}'s column block",
                )
        else:
            local_rows = 4
            cols = 4
            if input_3d:
                buf = symm_mem.empty(
                    p, local_rows, cols, dtype=dtype, device=self.device
                ).fill_(float(self.rank))
            else:
                buf = symm_mem.empty(
                    p * local_rows, cols, dtype=dtype, device=self.device
                ).fill_(float(self.rank))
            symm_mem.rendezvous(buf, group=group_name)
            if out_2d:
                out = torch.empty(local_rows, p * cols, dtype=dtype, device=self.device)
            else:
                out = torch.empty(local_rows, p, cols, dtype=dtype, device=self.device)
            symm_mem.all_to_all_nd(
                buf,
                out,
                scatter_dim=scatter_dim,
                gather_dim=gather_dim,
                group=group_name,
            )
            torch.cuda.synchronize()
            out_view = out.view(local_rows, p, cols) if out_2d else out
            for j in range(p):
                self.assertEqual(
                    out_view[:, j, :],
                    torch.full(
                        (local_rows, cols),
                        float(j),
                        dtype=dtype,
                        device=self.device,
                    ),
                    msg=f"rank {self.rank}: out[:, {j}, :] should be peer {j}'s row block",
                )

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version((2, 29), "nccl_all_gather_offset requires nccl 2.29")
    @skip_if_lt_x_gpu(2)
    # Shard sizes are multiples of 8 so every slice stays 16-byte aligned for
    # both float (4B) and bfloat16 (2B), as the kernel requires.  `out` is a
    # symmetric tensor; the multimem (NVLink SHARP) path is used when multicast
    # is available, otherwise the LSA push fallback (e.g. NCCL_NVLS_ENABLE=0).
    @parametrize("split_sizes", [[64], [32, 96], [16, 48, 32]])
    @parametrize("explicit_offsets", [False, True])
    @parametrize("dtype", [torch.float, torch.bfloat16])
    def test_all_gather_offset(self, split_sizes, explicit_offsets, dtype):
        """all_gather_offset: gather each rank's parameter shards into a
        parameter-contiguous symmetric output, fusing the copy-out reorder.
        Covers both the default (derived) and an explicitly-passed
        split_offsets, and multiple dtypes."""
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        n_params = len(split_sizes)
        offsets = []
        acc = 0
        for sz in split_sizes:
            offsets.append(acc)
            acc += sz
        numel = acc

        # Distinct per-(rank, param) constant, kept small so it is exactly
        # representable in bfloat16.
        def value(rank, i):
            return float((i + 1) * 10 + (rank + 1))

        inp = symm_mem.empty(numel, dtype=dtype, device=self.device)
        for i in range(n_params):
            inp[offsets[i] : offsets[i] + split_sizes[i]] = value(self.rank, i)
        symm_mem.rendezvous(inp, group=group_name)

        out = symm_mem.empty(
            numel * self.world_size, dtype=dtype, device=self.device
        ).fill_(0)
        symm_mem.rendezvous(out, group=group_name)
        symm_mem.all_gather_offset(
            inp,
            out,
            group_name,
            split_sizes,
            split_offsets=offsets if explicit_offsets else None,
        )
        torch.cuda.synchronize()

        # Parameter i occupies out[off*W : (off+size)*W], with source rank r
        # at sub-offset r*size.
        for i in range(n_params):
            base = offsets[i] * self.world_size
            for r in range(self.world_size):
                region = out[
                    base + r * split_sizes[i] : base + (r + 1) * split_sizes[i]
                ]
                self.assertEqual(
                    region,
                    torch.full_like(region, value(r, i)),
                    msg=f"rank {self.rank}: param {i} from src rank {r} mismatch",
                )

        # Source buffer must be unmodified.
        for i in range(n_params):
            src_slice = inp[offsets[i] : offsets[i] + split_sizes[i]]
            self.assertEqual(
                src_slice,
                torch.full_like(src_slice, value(self.rank, i)),
                msg=f"rank {self.rank}: source param {i} should be unchanged",
            )

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version((2, 29), "NCCL one-sided host API support from nccl 2.29")
    @skip_if_lt_x_gpu(2)
    def test_put_wait_signal(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        # Use this barrier to make sure all ranks are initialized.
        c10d.barrier()
        group_name = c10d.group.WORLD.group_name

        dtype = torch.float
        numel = 1024
        src = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        dst = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(src, group=group_name)
        hdl = symm_mem.rendezvous(dst, group=group_name)

        # Pair ranks: odd ranks send to previous even ranks.
        if self.rank % 2 == 1:
            dst_rank = self.rank - 1
            symm_mem.put_signal(src, hdl, dst_rank)
        elif self.rank % 2 == 0 and self.rank + 1 < self.world_size:
            src_rank = self.rank + 1
            symm_mem.wait_signal(hdl, src_rank)
            self.assertEqual(dst, torch.full_like(dst, float(src_rank)))

        c10d.barrier()

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7) if TEST_WITH_ROCM else (2, 27),
        "NCCL/RCCL symmetric-memory API version requirement",
    )
    @skip_if_lt_x_gpu(2)
    def test_mempool_tensor_factory(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        # Need this all_reduce to initialize NCCL communicator. Otherwise, the
        # test will hang.  TODO: investigate how NCCLSymmetricMemory can
        # initialize NCCL communicator.
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        dtype = torch.float
        numel = 1024

        mempool = symm_mem.get_mem_pool(self.device)

        with torch.cuda.use_mem_pool(mempool):
            tensor = torch.arange(numel, dtype=dtype, device=self.device)

        # Rendezvous should not error out
        symm_mem.rendezvous(tensor, group=group_name)
        tensor = torch.ops.symm_mem.one_shot_all_reduce(tensor, "sum", group_name)
        expected = (
            torch.arange(numel, dtype=dtype, device=self.device) * self.world_size
        )
        self.assertEqual(tensor, expected)

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7) if TEST_WITH_ROCM else (2, 27),
        "NCCL/RCCL symmetric-memory API version requirement",
    )
    @skip_if_lt_x_gpu(2)
    def test_mempool_compute_ops(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        # Need this all_reduce to initialize NCCL communicator. Otherwise, the
        # test will hang.  TODO: investigate how NCCLSymmetricMemory can
        # initialize NCCL communicator.
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        dtype = torch.float
        dim = 1024
        w = torch.ones(dim, dim, dtype=dtype, device=self.device)
        x = torch.ones(1, dim, dtype=dtype, device=self.device)

        mempool = symm_mem.get_mem_pool(self.device)

        with torch.cuda.use_mem_pool(mempool):
            y = torch.mm(x, w)

        # One-shot all-reduce should not error out
        y = torch.ops.symm_mem.one_shot_all_reduce(y, "sum", group_name)
        expected = torch.mm(x, w) * self.world_size
        self.assertEqual(y, expected)

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7) if TEST_WITH_ROCM else (2, 27),
        "NCCL/RCCL symmetric-memory API version requirement",
    )
    @skip_if_lt_x_gpu(2)
    def test_mempool_recycled_barrier(self):
        # Regression test for the signal-pad pollution bug: ncclMemAlloc does
        # not zero memory, so a MemPool-recycled NCCL symmetric allocation could
        # start the CAS-based barrier() protocol from a non-zero signal pad and
        # deadlock. alloc() zeros the pad up front. Allocate from the SymmMem
        # MemPool, run a barrier / buffer round-trip, free, then allocate the
        # same size again (recycling the freed block) and confirm the round-trip
        # still works on the recycled region.
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name
        mempool = symm_mem.get_mem_pool(self.device)
        numel, dtype = 1024, torch.float

        def barrier_roundtrip():
            with torch.cuda.use_mem_pool(mempool):
                t = torch.empty(numel, dtype=dtype, device=self.device)
            hdl = symm_mem.rendezvous(t, group=group_name)
            t.fill_(self.rank)
            # Bounded barriers so a polluted-pad regression fails cleanly
            # instead of hanging.
            hdl.barrier(timeout_ms=60000)
            for peer in range(self.world_size):
                buf = hdl.get_buffer(peer, (numel,), dtype)
                self.assertTrue(buf.eq(peer).all())
            hdl.barrier(timeout_ms=60000)
            return t, hdl

        t1, hdl1 = barrier_roundtrip()
        del hdl1, t1
        t2, hdl2 = barrier_roundtrip()
        del hdl2, t2

    @skip_but_pass_in_sandcastle_if(
        not TEST_WITH_ROCM, "ROCm-only: RCCL free-cache cleanup ordering"
    )
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7), "ROCm LSA symmetric-memory support from RCCL 2.29.7"
    )
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_reuse_waits_for_free_stream_zero(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name
        numel, dtype = 1024, torch.float

        t = symm_mem.empty(numel, dtype=dtype, device=self.device)
        hdl = symm_mem.rendezvous(t, group=group_name)
        ptr = t.data_ptr()
        dirty_pad = hdl.get_signal_pad(self.rank, (1,), dtype=torch.uint32)
        dirty_pad.fill_(123)
        torch.cuda.synchronize()

        free_stream = torch.cuda.Stream(device=self.device)
        reuse_stream = torch.cuda.Stream(device=self.device)

        with torch.cuda.stream(free_stream):
            torch.cuda._sleep(100_000_000)
            del dirty_pad, hdl, t
            gc.collect()

        with torch.cuda.stream(reuse_stream):
            reused = symm_mem.empty(numel, dtype=dtype, device=self.device)
            self.assertEqual(reused.data_ptr(), ptr)
            reused_hdl = symm_mem.rendezvous(reused, group=group_name)
            reused_pad = reused_hdl.get_signal_pad(self.rank, (1,), dtype=torch.uint32)
            reused_pad_value = reused_pad.clone()

        reuse_stream.synchronize()
        self.assertEqual(reused_pad_value.item(), 0)
        del reused_pad, reused_hdl, reused

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7) if TEST_WITH_ROCM else (2, 29),
        "NCCL/RCCL symmetric-memory API version requirement",
    )
    @skip_but_pass_in_sandcastle_if(
        os.environ.get("NCCL_NVLS_ENABLE", "1") == "0",
        "NCCL_NVLS_ENABLE=0",
    )
    @skip_if_lt_x_gpu(2)
    def test_multicast_ptr(self) -> None:
        """
        Get the multicast pointer
        """
        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _SymmetricMemory

        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        tensor = symm_mem.empty(1, device=self.device)
        handle = symm_mem.rendezvous(tensor, group_name)
        if _SymmetricMemory.has_multicast_support(DeviceType.CUDA, self.device.index):
            self.assertNotEqual(handle.multicast_ptr, 0)
        else:
            self.assertEqual(handle.multicast_ptr, 0)


@requires_cuda_p2p_access()
class NCCLSymmetricMemoryNccl2Test(MultiProcContinuousTest):
    """NCCL symmetric memory over an nccl2-backed process group.

    Same flow as NCCLSymmetricMemoryTest, but the process group uses the in-tree
    torchcomms "nccl2" backend. Regression test for nccl2 publishing its host
    ncclComm_t into NCCLDevCommManager (via the comm-registration hook) so that
    NCCLSymmetricMemory can resolve it by group name -- before that, rendezvous
    on an nccl2 group raised "NCCL host communicator for group ... not found".
    """

    backend_name = "nccl2"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file):
        # Eager communicator init via device_id, mirroring
        # NCCLSymmetricMemoryTest.
        if rdvz_file is None:
            raise AssertionError("Expected rdvz_file to not be None")
        os.environ["LOCAL_RANK"] = str(rank)
        if TEST_WITH_ROCM:
            os.environ.setdefault("NCCL_CUMEM_ENABLE", "1")
            os.environ.setdefault("NCCL_WIN_ENABLE", "1")
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        store = c10d.FileStore(rdvz_file, world_size)
        c10d.init_process_group(
            backend=cls.backend_name,
            world_size=world_size,
            rank=rank,
            store=store,
            timeout=cls.timeout,
            device_id=device,
        )
        cls.pg = c10d.distributed_c10d._get_default_group()

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 30, 4) if TEST_WITH_ROCM else (2, 27),
        "NCCL/RCCL symmetric-memory API version requirement",
    )
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_rendezvous(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        # Confirm the intended path: a torchcomms PG + the NCCL symm-mem backend.
        self.assertEqual(c10d.get_backend(c10d.group.WORLD), self.backend_name)
        self.assertEqual(symm_mem.get_backend(self.device), "NCCL")
        # Publish the communicator before rendezvous looks it up.
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        t = symm_mem.empty(64, dtype=torch.float, device=self.device).fill_(self.rank)
        handle = symm_mem.rendezvous(t, group=group_name)
        self.assertEqual(handle.rank, self.rank)
        self.assertEqual(handle.world_size, self.world_size)

        # Exercise an actual symm-mem collective end to end.
        result = torch.ops.symm_mem.one_shot_all_reduce(t, "sum", group_name)
        self.assertEqual(
            result, torch.full_like(result, (self.world_size - 1) * self.world_size / 2)
        )


class NCCLSymmetricMemoryNcclLazyTest(NCCLSymmetricMemoryNccl2Test):
    backend_name = "nccl-lazy"


@requires_cuda_p2p_access()
@skip_but_pass_in_sandcastle_if(
    TEST_WITH_ROCM and getRocmVersion() == (7, 14),
    "RCCL CUMEM/P2P communicator init fails on ROCm 7.14 (p2p_tmp.cc:358)",
)
class NCCLSymmetricMemoryWinDisabledTest(MultiProcContinuousTest):
    """RCCL symmetric-memory precondition fail-fast.

    The process group is created with NCCL_WIN_ENABLE=0, so RCCL builds the comm
    without window support. Rendezvous must fail fast with the documented error,
    driven by the snapshot recorded at comm init -- not a re-read of the (since
    restored) environment.
    """

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file):
        if rdvz_file is None:
            raise AssertionError("Expected rdvz_file to not be None")
        os.environ["LOCAL_RANK"] = str(rank)
        # Force NCCL_WIN_ENABLE=0 only across this comm's creation -- RCCL samples
        # it inside init_process_group. Afterwards set it back to "1" so the live
        # environment is healthy at rendezvous: the fail-fast then can only come
        # from the init-time snapshot (a regression to reading the current env
        # would pass). Leaving it "1" also avoids polluting sibling classes.
        if TEST_WITH_ROCM:
            os.environ["NCCL_CUMEM_ENABLE"] = "1"
            os.environ["NCCL_WIN_ENABLE"] = "0"
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        store = c10d.FileStore(rdvz_file, world_size)
        try:
            c10d.init_process_group(
                backend="nccl",
                world_size=world_size,
                rank=rank,
                store=store,
                timeout=cls.timeout,
                device_id=device,
            )
        finally:
            if TEST_WITH_ROCM:
                os.environ["NCCL_WIN_ENABLE"] = "1"
        cls.pg = c10d.distributed_c10d._get_default_group()

    @skip_but_pass_in_sandcastle_if(
        not TEST_WITH_ROCM, "ROCm-only: RCCL window precondition"
    )
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7), "ROCm LSA symmetric-memory support from RCCL 2.29.7"
    )
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_precondition_fail_fast(self):
        # _init_pg created the comm with NCCL_WIN_ENABLE=0 then set it back to
        # "1", so the live environment is healthy here. Rendezvous must still
        # raise from the init-time snapshot -- proving the check reflects what
        # RCCL saw at comm creation, not the current environment.
        self.assertEqual(os.environ.get("NCCL_WIN_ENABLE"), "1")
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        tensor = symm_mem.empty(64, dtype=torch.float32, device=self.device)
        with self.assertRaisesRegex(
            RuntimeError, r"NCCL_CUMEM_ENABLE=1 and NCCL_WIN_ENABLE=1"
        ):
            symm_mem.rendezvous(tensor, group=group_name)


@requires_cuda_p2p_access()
@skip_but_pass_in_sandcastle_if(
    not TEST_WITH_ROCM, "NCCL symmetric-memory shrink test is ROCm-only"
)
@skip_but_pass_in_sandcastle_if(
    nGPUs < 3, "NCCL symmetric-memory shrink test requires 3+ GPUs"
)
@skip_but_pass_in_sandcastle_if(
    TEST_WITH_ROCM and getRocmVersion() == (7, 14),
    "RCCL CUMEM/P2P communicator init fails on ROCm 7.14 (p2p_tmp.cc:358)",
)
class NCCLSymmetricMemoryShrinkTest(MultiProcContinuousTest):
    @property
    def world_size(self):
        return 3

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file):
        if rdvz_file is None:
            raise AssertionError("Expected rdvz_file to not be None")
        os.environ["LOCAL_RANK"] = str(rank)
        if TEST_WITH_ROCM:
            os.environ.setdefault("NCCL_CUMEM_ENABLE", "1")
            os.environ.setdefault("NCCL_WIN_ENABLE", "1")
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        store = c10d.FileStore(rdvz_file, world_size)
        c10d.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            store=store,
            timeout=cls.timeout,
            device_id=device,
        )
        cls.pg = c10d.distributed_c10d._get_default_group()

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_shrink()
    @requires_nccl_version(
        (2, 29, 7) if TEST_WITH_ROCM else (2, 27),
        "NCCL/RCCL symmetric-memory API version requirement",
    )
    @skip_if_lt_x_gpu(3)
    @skip_but_pass_in_sandcastle_if(
        not PLATFORM_SUPPORTS_SYMM_MEM, "SymmMem is not supported on this platform"
    )
    def test_shrink_group_symm_mem_registry(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        default_name = c10d.group.WORLD.group_name

        default_tensor = symm_mem.empty(
            64, dtype=torch.float32, device=self.device
        ).fill_(self.rank)
        default_handle = symm_mem.rendezvous(default_tensor, group=default_name)
        default_handle.barrier()

        subgroup = c10d.new_group(list(range(self.world_size)))
        c10d.all_reduce(torch.ones(1, device=self.device), group=subgroup)
        # shrink_flags=0 requires the parent communicator to have no pending
        # work. Ensure the nccl2 work queue observes this warm-up as complete.
        torch.cuda.synchronize(self.device)
        ranks_to_exclude = [self.world_size - 1]
        shrunk_pg = None

        if self.rank in ranks_to_exclude:
            c10d.destroy_process_group(subgroup)
        else:
            shrunk_pg = c10d.shrink_group(ranks_to_exclude, group=subgroup)
            shrunk_tensor = symm_mem.empty(
                64, dtype=torch.float32, device=self.device
            ).fill_(self.rank)
            shrunk_handle = symm_mem.rendezvous(
                shrunk_tensor, group=shrunk_pg.group_name
            )
            shrunk_handle.barrier()
            for peer in range(shrunk_pg.size()):
                buf = shrunk_handle.get_buffer(peer, (64,), torch.float32)
                self.assertTrue(buf.eq(peer).all())

        c10d.barrier()
        live_default = symm_mem.empty(
            64, dtype=torch.float32, device=self.device
        ).fill_(self.rank)
        live_default_handle = symm_mem.rendezvous(live_default, group=default_name)
        live_default_handle.barrier()
        for peer in range(self.world_size):
            buf = live_default_handle.get_buffer(peer, (64,), torch.float32)
            self.assertTrue(buf.eq(peer).all())

        if shrunk_pg is not None:
            c10d.destroy_process_group(shrunk_pg)

        c10d.barrier()
        final_default = symm_mem.empty(
            64, dtype=torch.float32, device=self.device
        ).fill_(self.rank + 10)
        final_default_handle = symm_mem.rendezvous(final_default, group=default_name)
        final_default_handle.barrier()
        for peer in range(self.world_size):
            buf = final_default_handle.get_buffer(peer, (64,), torch.float32)
            self.assertTrue(buf.eq(peer + 10).all())


@requires_cuda_p2p_access()
@skip_but_pass_in_sandcastle_if(
    not TEST_WITH_ROCM,
    "ROCm-only: the free-block cache that retains peer alloc infos is USE_ROCM",
)
@skip_but_pass_in_sandcastle_if(
    TEST_WITH_ROCM and getRocmVersion() == (7, 14),
    "RCCL CUMEM/P2P communicator init fails on ROCm 7.14 (p2p_tmp.cc:358)",
)
@skip_but_pass_in_sandcastle_if(
    TEST_WITH_ROCM and nccl.version() < (2, 30, 4),
    "RCCL host device APIs require RCCL 2.30.4 or newer",
)
class NCCLSymmetricMemoryRestartTest(MultiProcContinuousTest):
    """ROCm cached allocations must not retain PAIs for dead communicators.

    The ncclMemAlloc block itself is communicator-independent and may be
    recycled, but its old window and peer tables must be invalidated whether
    the tensor is freed before or after communicator teardown. A same-name
    successor must build a fresh PAI. Isolated in its own class because it
    tears down and re-inits the default PG.
    """

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file):
        if rdvz_file is None:
            raise AssertionError("Expected rdvz_file to not be None")
        # Remember the rendezvous file so the test can derive a second,
        # rank-agreed store for the mid-test re-init.
        cls.rdvz_file = rdvz_file
        os.environ["LOCAL_RANK"] = str(rank)
        if TEST_WITH_ROCM:
            os.environ.setdefault("NCCL_CUMEM_ENABLE", "1")
            os.environ.setdefault("NCCL_WIN_ENABLE", "1")
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        store = c10d.FileStore(rdvz_file, world_size)
        c10d.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            store=store,
            timeout=cls.timeout,
            device_id=device,
        )
        cls.pg = c10d.distributed_c10d._get_default_group()

    def _run_restart(self, free_before_destroy):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        numel = 4096
        tensor = symm_mem.empty(numel, dtype=torch.float32, device=self.device)
        old_handle = symm_mem.rendezvous(tensor, group=group_name)
        old_handle.barrier()
        unretained_tensor = None
        if free_before_destroy:
            # The block is already in the free cache when teardown invalidates
            # and removes its communicator-scoped PAI.
            del tensor
            torch.cuda.synchronize(self.device)
        else:
            # Exercise teardown with an enqueued kernel whose PAI has no
            # externally retained handle. Invalidation may destroy its device
            # pointer tables, so shutdown must synchronize first.
            unretained_tensor = symm_mem.empty(
                numel + 1, dtype=torch.float32, device=self.device
            )
            unretained_handle = symm_mem.rendezvous(unretained_tensor, group=group_name)
            unretained_handle.barrier()
            del unretained_handle

        c10d.destroy_process_group()
        with self.assertRaisesRegex(RuntimeError, "destroyed communicator"):
            old_handle.barrier()
        if not free_before_destroy:
            # Teardown invalidates only the live block's old PAI. The allocation
            # itself remains safe to cache after free.
            del unretained_tensor
            del tensor
            torch.cuda.synchronize(self.device)

        # Re-init the default group under a fresh store with the same rank
        # layout and the same finalized process-group name.
        order = "free_first" if free_before_destroy else "destroy_first"
        restart_file = type(self).rdvz_file + f"_symmem_restart_{order}"
        store = c10d.FileStore(restart_file, self.world_size)
        c10d.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            timeout=type(self).timeout,
            device_id=self.device,
        )
        type(self).pg = c10d.distributed_c10d._get_default_group()
        c10d.all_reduce(torch.ones(1, device=self.device))
        new_group_name = c10d.group.WORLD.group_name

        # The allocation may be recycled, but rendezvous must create peer
        # metadata for the successor communicator.
        fresh = symm_mem.empty(numel, dtype=torch.float32, device=self.device)
        handle = symm_mem.rendezvous(fresh, group=new_group_name)
        # A same-name successor must not make the predecessor handle usable.
        with self.assertRaisesRegex(RuntimeError, "destroyed communicator"):
            old_handle.barrier()
        fresh.fill_(self.rank)
        torch.cuda.synchronize(self.device)
        handle.barrier()
        peer_rank = (self.rank + 1) % self.world_size
        buf = handle.get_buffer(peer_rank, (numel,), torch.float32)
        self.assertTrue(buf.eq(peer_rank).all())
        del handle
        del old_handle

    def _run_reduce_scatter_offset(self, group_name):
        rows, cols = 64, 32
        n_experts = self.world_size
        rank_sum = float(sum(r + 1 for r in range(self.world_size)))

        buf = symm_mem.empty(
            n_experts * rows, cols, dtype=torch.float, device=self.device
        )
        for i in range(n_experts):
            buf[i * rows : (i + 1) * rows, :] = float((self.rank + 1) * (i + 1))
        handle = symm_mem.rendezvous(buf, group=group_name)

        dst_ranks = [i % self.world_size for i in range(n_experts)]
        out = [torch.zeros(rows, cols, dtype=torch.float, device=self.device)]
        offsets = [i * rows for i in range(1, n_experts + 1)]
        symm_mem.reduce_scatter_offset(
            buf, out, group_name, dim=0, offsets=offsets, dst_ranks=dst_ranks
        )
        torch.cuda.synchronize(self.device)

        expected = float(self.rank + 1) * rank_sum
        self.assertEqual(out[0], torch.full_like(out[0], expected))
        return buf, handle

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7), "ROCm LSA symmetric-memory support from RCCL 2.29.7"
    )
    @skip_if_lt_x_gpu(2)
    def test_free_cache_pai_refreshed_on_pg_restart(self):
        self._run_restart(free_before_destroy=True)

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7), "ROCm LSA symmetric-memory support from RCCL 2.29.7"
    )
    @skip_if_lt_x_gpu(2)
    def test_live_allocation_pai_refreshed_on_pg_restart(self):
        self._run_restart(free_before_destroy=False)

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7), "ROCm LSA symmetric-memory support from RCCL 2.29.7"
    )
    @skip_if_lt_x_gpu(2)
    def test_abort_restart_rebuilds_reduce_scatter_offset_devcomm(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        old_buf, old_handle = self._run_reduce_scatter_offset(group_name)
        backend = type(self).pg._get_backend(self.device)
        backend.abort()
        c10d.destroy_process_group()

        with self.assertRaisesRegex(RuntimeError, "destroyed communicator"):
            old_handle.barrier()
        del old_buf
        torch.cuda.synchronize(self.device)

        restart_file = type(self).rdvz_file + "_symmem_abort_restart"
        store = c10d.FileStore(restart_file, self.world_size)
        c10d.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            timeout=type(self).timeout,
            device_id=self.device,
        )
        type(self).pg = c10d.distributed_c10d._get_default_group()
        c10d.all_reduce(torch.ones(1, device=self.device))

        new_group_name = c10d.group.WORLD.group_name
        self.assertEqual(new_group_name, group_name)
        fresh_buf, fresh_handle = self._run_reduce_scatter_offset(new_group_name)
        with self.assertRaisesRegex(RuntimeError, "destroyed communicator"):
            old_handle.barrier()
        del fresh_handle
        del fresh_buf
        del old_handle

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7), "ROCm LSA symmetric-memory support from RCCL 2.29.7"
    )
    @skip_if_lt_x_gpu(2)
    def test_retained_handle_invalidated_on_cache_eviction(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        # Four distinct cache keys total 134 MiB, exceeding the 128 MiB
        # per-device budget and evicting the oldest allocation.
        sizes_mib = (32, 33, 34, 35)
        tensor = symm_mem.empty(
            sizes_mib[0] * 1024 * 1024 // 4,
            dtype=torch.float32,
            device=self.device,
        )
        old_handle = symm_mem.rendezvous(tensor, group=group_name)
        old_handle.barrier()
        torch.cuda.synchronize(self.device)
        del tensor

        for size_mib in sizes_mib[1:]:
            tensor = symm_mem.empty(
                size_mib * 1024 * 1024 // 4,
                dtype=torch.float32,
                device=self.device,
            )
            del tensor
        torch.cuda.synchronize(self.device)

        with self.assertRaisesRegex(RuntimeError, "freed backing allocation"):
            old_handle.barrier()


def _host_cft_unsupported_reason() -> str | None:
    """Python mirror of NCCL's runtime gate for CFT logical endpoints
    (ncclGpuCftSupport in cft_dev_runtime.cc): a Blackwell-class GPU whose
    driver reports CUDA >= 13.3 and sets both logical-endpoint device
    attributes. One NCCL requirement stays invisible here: libnccl itself
    must be built with CUDA >= 13.3.
    """
    if torch.cuda.get_device_capability() < (10, 0):
        return "host-side CFT requires Blackwell (sm_100+)"
    if os.environ.get("NCCL_CFT_ENABLE", "1") == "0":
        return "host-side CFT disabled via NCCL_CFT_ENABLE=0"
    if not _HAS_CUDA_BINDINGS:
        return "cuda-bindings is required to probe CFT support"
    try:
        _check_cuda_bindings(_drv.cuInit(0))
        if _check_cuda_bindings(_drv.cuDriverGetVersion()) < 13030:
            return "host-side CFT requires a driver reporting CUDA >= 13.3"
        for name, raw in (
            ("CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_SUPPORTED", 153),
            ("CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_MULTICAST_SUPPORTED", 154),
        ):
            # The enum member only exists in cuda-bindings >= 13.3; the raw
            # value is part of the stable driver ABI.
            attr = getattr(_drv.CUdevice_attribute, name, raw)
            if not _check_cuda_bindings(_drv.cuDeviceGetAttribute(attr, 0)):
                return f"device does not report {name}"
    except Exception as e:
        return f"failed to probe CFT support: {e}"
    return None


_P = ParamSpec("_P")
_T = TypeVar("_T")


def requires_cft_support() -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """Skip unless the GPU/driver stack can create CFT logical endpoints.

    Like requires_nvls in test_nvshmem.py, but evaluated lazily inside the
    wrapper so the CUDA probing runs in the spawned child process rather
    than at decoration time.
    """

    def decorator(func: Callable[_P, _T]) -> Callable[_P, _T]:
        @wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            reason = _host_cft_unsupported_reason()
            if reason is not None:
                raise SkipTest(reason)
            return func(*args, **kwargs)

        return wrapper

    return decorator


@skip_but_pass_in_sandcastle_if(
    TEST_WITH_ROCM, "NCCL symmetric memory is not supported on ROCm"
)
class SymmMemCftHandleTest(MultiProcessTestCase):
    """Host-side NCCL CFT logical-endpoint handles exposed on _SymmetricMemory.

    These are the `(le_id, le_offset)` coordinates a custom kernel feeds to the
    device-side `ncclCft` put/get/red family, so the handle only means anything
    for the group it was rendezvoused with.
    """

    def setUp(self) -> None:
        super().setUp()
        # The backend has to be chosen before the child processes start.
        # `set_backend` cannot do it: by the time a test body runs the
        # allocator has already been handed out once and it refuses to swap.
        # TORCH_SYMMMEM is read at static-init time instead, and spawn passes
        # the environment down.
        backend = "CUDA" if self._testMethodName.endswith("wrong_backend") else "NCCL"
        os.environ["TORCH_SYMMMEM"] = backend
        self.addCleanup(os.environ.pop, "TORCH_SYMMMEM", None)
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        return 2

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _init_process(self):
        if not PLATFORM_SUPPORTS_SYMM_MEM:
            raise SkipTest("Test requires SymmMem support")
        for peer in range(self.world_size):
            if peer == self.rank:
                continue
            if not torch._C._cuda_canDeviceAccessPeer(self.rank, peer):
                raise SkipTest("Test requires p2p access")

        torch.cuda.set_device(self.device)
        pg_opts = c10d.ProcessGroupNCCL.Options()
        # ncclHostCftFallback: create the CFT logical endpoints if the hardware
        # allows, otherwise silently proceed without them. The endpoints are
        # made during window registration, i.e. inside rendezvous, so this has
        # to be set before the communicator exists.
        pg_opts.config.host_cft_mode = 3
        c10d.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=c10d.FileStore(self.file_name, self.world_size),
            pg_options=pg_opts,
            device_id=self.device,
        )
        self.addCleanup(c10d.destroy_process_group)
        # The NCCL backend rendezvous looks the communicator up rather than
        # creating it, so force it into existence first.
        c10d.all_reduce(torch.ones(1, device=self.device))
        t = symm_mem.empty(1024, dtype=torch.float32, device=self.device)
        return symm_mem.rendezvous(t, group=c10d.group.WORLD.group_name)

    @requires_nccl()
    @requires_nccl_version((2, 31, 2), "Need NCCL 2.31.2+ for host-side CFT")
    @skip_if_lt_x_gpu(2)
    @requires_cft_support()
    def test_get_cft_handle(self) -> None:
        hdl = self._init_process()
        self.assertEqual(symm_mem.get_backend(self.device), "NCCL")
        _, self_le_offset = hdl.get_peer_cft_handle(self.rank)

        le_ids = set()
        for peer in range(self.world_size):
            le_id, le_offset = hdl.get_peer_cft_handle(peer)
            self.assertNotIn(le_id, le_ids, f"peer {peer} reuses le_id {le_id}")
            le_ids.add(le_id)
            # Every rank maps the buffer at the same offset in the symmetric
            # space, so only the endpoint varies from peer to peer.
            self.assertEqual(le_offset, self_le_offset)

        with self.assertRaisesRegex(RuntimeError, "invalid peer"):
            hdl.get_peer_cft_handle(self.world_size)
        with self.assertRaisesRegex(RuntimeError, "invalid peer"):
            hdl.get_peer_cft_handle(-1)

        try:
            # Collective on first call unless the endpoint was created eagerly
            # at window registration, so every rank has to reach this.
            _, mc_le_offset = hdl.get_multimem_cft_handle()
            self.assertEqual(mc_le_offset, self_le_offset)
        except RuntimeError:
            # NCCL disables CFT multicast when NVLS is unavailable, uniformly
            # across ranks, so nobody is left waiting in the collective above.
            pass

    @requires_nccl()
    @requires_nccl_version((2, 31, 2), "Need NCCL 2.31.2+ for host-side CFT")
    @skip_if_lt_x_gpu(2)
    def test_get_cft_handle_wrong_backend(self) -> None:
        hdl = self._init_process()
        self.assertEqual(symm_mem.get_backend(self.device), "CUDA")
        with self.assertRaisesRegex(RuntimeError, "only available on the NCCL"):
            hdl.get_peer_cft_handle(0)
        with self.assertRaisesRegex(RuntimeError, "only available on the NCCL"):
            hdl.get_multimem_cft_handle()


@requires_cuda_p2p_access()
class NCCLSymmMemWatchdogTest(MultiProcContinuousTest):
    """Test watchdog integration with real NCCL symmetric memory ops."""

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file):
        if rdvz_file is None:
            raise AssertionError("Expected rdvz_file to not be None")
        os.environ["LOCAL_RANK"] = str(rank)
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        store = c10d.FileStore(rdvz_file, world_size)
        c10d.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            store=store,
            timeout=cls.timeout,
            device_id=device,
        )
        cls.pg = c10d.distributed_c10d._get_default_group()

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version((2, 27), "NCCL Symmetric Memory support from nccl 2.27")
    @skip_if_lt_x_gpu(2)
    def test_rendezvous_with_watchdog(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        symm_mem.set_watchdog_timeout(30.0)
        try:
            group_name = c10d.group.WORLD.group_name
            t = symm_mem.empty(1024, dtype=torch.float, device=self.device)
            hdl = symm_mem.rendezvous(t, group=group_name)
            self.assertEqual(hdl.rank, self.rank)
            self.assertEqual(hdl.world_size, self.world_size)
        finally:
            symm_mem.set_watchdog_timeout(None)

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version((2, 29), "NCCL one-sided host API support from nccl 2.29")
    @skip_if_lt_x_gpu(2)
    def test_put_wait_signal_with_watchdog(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        symm_mem.set_watchdog_timeout(30.0)
        try:
            group_name = c10d.group.WORLD.group_name
            t = symm_mem.empty(1024, dtype=torch.float, device=self.device).fill_(
                self.rank
            )
            handle = symm_mem.rendezvous(t, group=group_name)
            c10d.barrier()

            if self.rank % 2 == 1:
                dst_rank = self.rank - 1
                symm_mem.put_signal(t, handle, dst_rank)
                torch.cuda.synchronize()
            elif self.rank % 2 == 0 and self.rank + 1 < handle.world_size:
                src_rank = self.rank + 1
                symm_mem.wait_signal(handle, src_rank)
                torch.cuda.synchronize()

            c10d.barrier()
        finally:
            symm_mem.set_watchdog_timeout(None)

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version((2, 27), "NCCL Symmetric Memory support from nccl 2.27")
    @skip_if_lt_x_gpu(2)
    def test_rendezvous_watchdog_cancel_on_exception(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        symm_mem.set_watchdog_timeout(30.0)
        try:
            non_symm_tensor = torch.ones(64, device=self.device)
            with self.assertRaises(RuntimeError):
                symm_mem.rendezvous(non_symm_tensor, group=c10d.group.WORLD.group_name)
        finally:
            symm_mem.set_watchdog_timeout(None)


instantiate_device_type_tests(TestNCCL, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
