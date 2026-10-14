# Owner(s): ["oncall: distributed"]

import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch.distributed.distributed_c10d import _TORCHCOMM_AVAILABLE
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import C10dTorchCommsTestBase
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    subtest,
    TestCase,
)


@unittest.skipIf(not _TORCHCOMM_AVAILABLE, "TorchComms is not installed")
class TestC10dTorchCommsBasic(C10dTorchCommsTestBase):
    REDUCE_OPS = [
        subtest(dist.ReduceOp.SUM, name="SUM"),
        subtest(dist.ReduceOp.AVG, name="AVG"),
        subtest(dist.ReduceOp.MIN, name="MIN"),
        subtest(dist.ReduceOp.MAX, name="MAX"),
        subtest(dist.ReduceOp.PRODUCT, name="PRODUCT"),
    ]

    @property
    def _rank_value(self):
        return self.rank + 1

    def _skip_if_product_overflows(self, op):
        if op == dist.ReduceOp.PRODUCT and self.world_size > 12:
            self.skipTest(
                f"world_size={self.world_size} > 12: PRODUCT is world_size! "
                "and only up to 12! is exactly representable in float32"
            )

    def _expected_reduce_result(self, op):
        """Return the expected scalar result for a rank+1 input reduced across all ranks."""
        total = sum(range(1, self.world_size + 1))
        if op == dist.ReduceOp.SUM:
            return total
        elif op == dist.ReduceOp.AVG:
            return total / self.world_size
        elif op == dist.ReduceOp.MIN:
            return 1
        elif op == dist.ReduceOp.MAX:
            return self.world_size
        elif op == dist.ReduceOp.PRODUCT:
            result = 1
            for i in range(1, self.world_size + 1):
                result *= i
            return result
        raise ValueError(f"Unsupported op: {op}")

    @parametrize("op", REDUCE_OPS)
    def test_allreduce(self, op):
        self._skip_if_product_overflows(op)
        tensor = torch.tensor([self._rank_value], dtype=torch.float32)
        dist.all_reduce(tensor, op=op, group=self.pg)
        self.assertEqual(tensor.item(), self._expected_reduce_result(op))

    def test_all_gather(self):
        input_tensor = torch.tensor([self._rank_value], dtype=torch.float32)
        gather_list = [torch.empty_like(input_tensor) for _ in range(self.world_size)]
        dist.all_gather(gather_list, input_tensor, group=self.pg)
        expected = list(range(1, self.world_size + 1))
        self.assertEqual([t.item() for t in gather_list], expected)

    def test_all_gather_into_tensor(self):
        input_tensor = torch.tensor([self._rank_value], dtype=torch.float32)
        output_tensor = torch.empty(self.world_size, dtype=torch.float32)
        dist.all_gather_single(output_tensor, input_tensor, group=self.pg)
        expected = list(range(1, self.world_size + 1))
        self.assertEqual([t.item() for t in output_tensor], expected)

    def test_broadcast(self):
        tensor = torch.tensor([self._rank_value], dtype=torch.float32)
        dist.broadcast(tensor, src=0, group=self.pg)
        self.assertEqual(tensor.item(), 1)

    def test_gather(self):
        tensor = torch.tensor([self._rank_value], dtype=torch.float32)
        gather_list = None
        if self.rank == 0:
            gather_list = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.gather(tensor, gather_list=gather_list, dst=0, group=self.pg)
        if self.rank == 0:
            expected = list(range(1, self.world_size + 1))
            self.assertEqual([t.item() for t in gather_list], expected)

    def test_scatter(self):
        if self.rank == 0:
            scatter_list = [
                torch.tensor([i], dtype=torch.float32) for i in range(self.world_size)
            ]
        else:
            scatter_list = None
        tensor = torch.empty(1, dtype=torch.float32)
        dist.scatter(tensor, scatter_list=scatter_list, src=0, group=self.pg)
        self.assertEqual(tensor.item(), self.rank)

    @parametrize("op", REDUCE_OPS)
    def test_reduce(self, op):
        self._skip_if_product_overflows(op)
        input_tensor = torch.tensor([self._rank_value], dtype=torch.float32)
        dist.reduce(input_tensor, dst=0, op=op, group=self.pg)
        if self.rank == 0:
            self.assertEqual(input_tensor.item(), self._expected_reduce_result(op))

    @parametrize("op", REDUCE_OPS)
    def test_reduce_scatter(self, op):
        self._skip_if_product_overflows(op)
        input_tensor = [
            torch.tensor([self._rank_value], dtype=torch.float32)
            for _ in range(self.world_size)
        ]
        output_tensor = torch.empty(1, dtype=torch.float32)
        dist.reduce_scatter(output_tensor, input_tensor, op=op, group=self.pg)
        self.assertEqual(output_tensor.item(), self._expected_reduce_result(op))

    @parametrize("op", REDUCE_OPS)
    def test_reduce_scatter_tensor(self, op):
        self._skip_if_product_overflows(op)
        input_tensor = torch.full(
            (self.world_size,), self._rank_value, dtype=torch.float32
        )
        output_tensor = torch.empty(1, dtype=torch.float32)
        dist.reduce_scatter_single(output_tensor, input_tensor, op=op, group=self.pg)
        self.assertEqual(output_tensor.item(), self._expected_reduce_result(op))

    def test_all_to_all(self):
        input_tensor = [
            torch.tensor([self._rank_value], dtype=torch.float32)
            for _ in range(self.world_size)
        ]
        output_tensor = [
            torch.empty(1, dtype=torch.float32) for _ in range(self.world_size)
        ]
        dist.all_to_all(output_tensor, input_tensor, group=self.pg)
        expected = list(range(1, self.world_size + 1))
        self.assertEqual([t.item() for t in output_tensor], expected)

    def test_all_to_all_single(self):
        input_tensor = torch.full(
            (self.world_size,), self._rank_value, dtype=torch.float32
        )
        output_tensor = torch.empty([self.world_size], dtype=torch.float32)
        dist.all_to_all_single(output_tensor, input_tensor, group=self.pg)
        expected = list(range(1, self.world_size + 1))
        self.assertEqual([t.item() for t in output_tensor], expected)

    def test_all_to_all_single_with_split_sizes(self):
        # Each rank sends (rank + 1) elements to every other rank,
        # so rank r's input_split_sizes are all (rank + 1).
        input_split_sizes = [self._rank_value] * self.world_size
        # Rank r receives (sender_rank + 1) elements from each sender,
        # so output_split_sizes[i] = i + 1.
        output_split_sizes = [i + 1 for i in range(self.world_size)]

        input_tensor = torch.empty(sum(input_split_sizes), dtype=torch.float32)
        offset = 0
        for dst in range(self.world_size):
            input_tensor[offset : offset + input_split_sizes[dst]].fill_(
                self.rank + dst
            )
            offset += input_split_sizes[dst]

        output_tensor = torch.empty(sum(output_split_sizes), dtype=torch.float32)
        dist.all_to_all_single(
            output_tensor,
            input_tensor,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=self.pg,
        )

        # Verify: section from sender i should contain value (i + rank)
        offset = 0
        for src in range(self.world_size):
            section = output_tensor[offset : offset + output_split_sizes[src]]
            expected = torch.full_like(section, src + self.rank)
            self.assertTrue(
                torch.equal(section, expected),
                f"Mismatch in section from rank {src}: got {section}, expected {expected}",
            )
            offset += output_split_sizes[src]

    def test_send_recv(self):
        send_rank = (self.rank + 1) % self.world_size
        recv_rank = (self.rank + self.world_size - 1) % self.world_size
        send_tensor = torch.tensor([self.rank], dtype=torch.float32)
        recv_tensor = torch.empty(1, dtype=torch.float32)
        if self.rank % 2 == 0:
            # Even ranks: send first, then receive
            dist.send(send_tensor, dst=send_rank, group=self.pg)
            dist.recv(recv_tensor, src=recv_rank, group=self.pg)
        else:
            # Odd ranks: receive first, then send
            dist.recv(recv_tensor, src=recv_rank, group=self.pg)
            dist.send(send_tensor, dst=send_rank, group=self.pg)
        # Each rank receives the rank number of the sender
        self.assertEqual(recv_tensor.item(), recv_rank)

    def test_barrier(self):
        dist.barrier(group=self.pg)
        # If we reach this point, the barrier succeeded without deadlock
        self.assertTrue(True)

    def test_new_group_delegates_to_split_group(self):
        # Under torchcomms, `new_group` routes through `split_group`. The
        # resulting subgroup must contain the requested ranks and be usable
        # for collectives.
        subg_ranks = list(range(self.world_size // 2))
        ng = dist.new_group(ranks=subg_ranks)

        if self.rank in subg_ranks:
            self.assertEqual(dist.get_process_group_ranks(ng), subg_ranks)
            tensor = torch.tensor([self._rank_value], dtype=torch.float32)
            dist.all_reduce(tensor, group=ng)
            self.assertEqual(tensor.item(), sum(r + 1 for r in subg_ranks))
        else:
            self.assertIs(ng, dist.GroupMember.NON_GROUP_MEMBER)

    def test_new_group_via_split_group_raises_on_unsupported_args(self):
        # `split_group` has a narrower surface than `new_group`; under
        # torchcomms the delegation must surface that mismatch instead of
        # silently falling back to the legacy path.
        ranks = list(range(self.world_size))
        with self.assertRaisesRegex(NotImplementedError, "use_local_synchronization"):
            dist.new_group(ranks=ranks, use_local_synchronization=True)
        with self.assertRaisesRegex(NotImplementedError, "sort_ranks"):
            dist.new_group(ranks=ranks, sort_ranks=False)


devices = ["cpu", "cuda", "xpu"]
instantiate_device_type_tests(
    TestC10dTorchCommsBasic, globals(), only_for=devices, allow_xpu=True
)


class _FakeBackendWrapper:
    def __init__(self, comm):
        self._comm = comm

    def get_comm(self):
        return self._comm


class _FakeOtherBackend:
    """A c10d backend that is not a _BackendWrapper."""


class TestC10dTorchCommsDestroyDedup(TestCase):
    _WORLD_ATTRS = (
        "pg_map",
        "pg_names",
        "pg_group_ranks",
        "pg_backend_config",
        "pg_to_tag",
        "tags_to_pg",
        "pg_coalesce_state",
        "comms",
    )

    def setUp(self):
        super().setUp()
        self._saved_world = {
            attr: getattr(c10d._world, attr).copy() for attr in self._WORLD_ATTRS
        }
        self._saved_default_pg = c10d._world.default_pg
        self._saved_group_count = c10d._world.group_count

    def tearDown(self):
        for attr, value in self._saved_world.items():
            container = getattr(c10d._world, attr)
            container.clear()
            if isinstance(value, dict):
                container.update(value)
            else:
                container.extend(value)
        c10d._world.default_pg = self._saved_default_pg
        c10d._world.group_count = self._saved_group_count
        super().tearDown()

    def _register_pg(self, device_backends, *, name=None):
        pg = mock.MagicMock()
        pg._device_types = list(device_backends)
        pg._get_backend.side_effect = lambda dev: device_backends[dev]
        group_name = c10d.GroupName(self.id() if name is None else name)
        pg.group_name = group_name

        c10d._world.pg_map[pg] = (None, None)
        c10d._world.pg_names[pg] = group_name
        c10d._world.pg_group_ranks[pg] = {}
        c10d._world.pg_backend_config[pg] = ""
        c10d._world.pg_to_tag[pg] = None
        return pg

    def test_destroy_shared_comm_finalized_once(self):
        comm = mock.Mock()
        wrapper = _FakeBackendWrapper(comm)
        pg = self._register_pg({"cuda": wrapper, "cpu": wrapper})
        c10d._world.comms.append(comm)

        with mock.patch.multiple(
            c10d,
            _TORCHCOMM_AVAILABLE=True,
            _BackendWrapper=_FakeBackendWrapper,
            _release_nvshmem_team_pool=lambda group_name: None,
            _unregister_process_group=lambda group_name: None,
            create=True,
        ):
            c10d.destroy_process_group(pg)

        comm.finalize.assert_called_once()
        self.assertEqual(c10d._world.comms, [])

    def _register_abortable_pg(self, calls, comm, name):
        backend = _FakeOtherBackend() if comm is None else _FakeBackendWrapper(comm)
        pg = self._register_pg({"cuda": backend}, name=name)
        pg._get_backend.side_effect = lambda dev: backend
        calls.attach_mock(pg.abort, f"abort_{name}")
        return pg

    def test_abort_finalizes_comms_and_releases_nvshmem_teams(self):
        calls = mock.Mock()
        comm_a, comm_b = calls.comm_a, calls.comm_b
        world = self._register_abortable_pg(calls, None, "world")
        group_a = self._register_abortable_pg(calls, comm_a, "a")
        self._register_abortable_pg(calls, comm_b, "b")
        c10d._world.default_pg = world
        c10d._world.comms.extend([comm_a, comm_b])
        call = mock.call

        with mock.patch.multiple(
            c10d,
            _TORCHCOMM_AVAILABLE=True,
            _BackendWrapper=_FakeBackendWrapper,
            _release_nvshmem_team_pool=calls.release,
            _unregister_process_group=lambda group_name: None,
            _unregister_all_process_groups=lambda: None,
            _update_default_pg=lambda pg: setattr(c10d._world, "default_pg", pg),
            create=True,
        ):
            c10d._abort_process_group(group_a)
            group_a_calls = [call.abort_a(), call.comm_a.finalize(), call.release("a")]
            self.assertListEqual(calls.mock_calls, group_a_calls)
            self.assertEqual(c10d._world.comms, [comm_b])

            calls.reset_mock()
            c10d._abort_process_group()
            aborts = [call.abort_world(), call.abort_b()]
            freed = [call.comm_b.finalize(), call.release("world"), call.release("b")]
            self.assertCountEqual(calls.mock_calls[:2], aborts)
            self.assertCountEqual(calls.mock_calls[2:], freed)
            self.assertEqual(c10d._world.comms, [])

    def test_abort_cleanup_continues_when_finalize_raises(self):
        calls = mock.Mock()
        comm_a, comm_b = calls.comm_a, calls.comm_b
        world = self._register_abortable_pg(calls, None, "world")
        group_a = self._register_abortable_pg(calls, comm_a, "a")
        group_b = self._register_abortable_pg(calls, comm_b, "b")
        c10d._world.default_pg = world
        c10d._world.comms.extend([comm_a, comm_b])

        with mock.patch.multiple(
            c10d,
            _TORCHCOMM_AVAILABLE=True,
            _BackendWrapper=_FakeBackendWrapper,
            _release_nvshmem_team_pool=calls.release,
            _unregister_process_group=lambda group_name: None,
            _unregister_all_process_groups=lambda: None,
            _update_default_pg=lambda pg: setattr(c10d._world, "default_pg", pg),
            create=True,
        ):
            comm_a.finalize.side_effect = RuntimeError("Work timed out during finalize")
            with self.assertLogs(c10d.logger, level="WARNING") as logs:
                c10d._abort_process_group(group_a)
            self.assertIn(
                "Failed to finalize TorchComms communicator after abort",
                "\n".join(logs.output),
            )
            calls.release.assert_called_once_with("a")
            self.assertEqual(c10d._world.comms, [comm_b])
            self.assertNotIn(group_a, c10d._world.pg_map)
            self.assertIn(group_b, c10d._world.pg_map)

            calls.reset_mock()
            comm_b.finalize.side_effect = RuntimeError("Work timed out during finalize")
            with self.assertLogs(c10d.logger, level="WARNING") as logs:
                c10d._abort_process_group()
            self.assertIn(
                "Failed to finalize TorchComms communicator after abort",
                "\n".join(logs.output),
            )
            calls.release.assert_has_calls(
                [mock.call("world"), mock.call("b")], any_order=True
            )
            self.assertEqual(c10d._world.comms, [])
            self.assertEqual(c10d._world.pg_map, {})
            self.assertEqual(c10d._world.pg_names, {})
            self.assertEqual(c10d._world.pg_group_ranks, {})
            self.assertIsNone(c10d._world.default_pg)


if __name__ == "__main__":
    run_tests()
