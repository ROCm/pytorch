# Owner(s): ["oncall: distributed"]

from unittest import mock

import torch.distributed.distributed_c10d as c10d
from torch.testing._internal.common_utils import run_tests, TestCase


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
