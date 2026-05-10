# Owner(s): ["oncall: profiler"]
import os

import torch
from torch._C._profiler import _rpd_available
from torch.profiler import rpd_profile
from torch.testing._internal.common_utils import run_tests, TestCase


def _rpd_loaded():
    """Check if librpd_tracer.so is already loaded without triggering dlopen."""
    return os.environ.get("RPDT_LOADED") == "1"


class TestRpdProfilerNoTracer(TestCase):
    """Tests that work without librpd_tracer.so loaded."""

    def setUp(self):
        if _rpd_loaded():
            self.skipTest("librpd_tracer.so is loaded; these tests require it absent")

    def test_context_manager_without_rpd(self):
        """rpd_profile enters and exits cleanly even when rpd is unavailable."""
        with rpd_profile() as p:
            x = torch.randn(10, 10)
            _ = x + x

        events = p.events()
        self.assertIsInstance(events, list)

    def test_key_averages_without_rpd(self):
        with rpd_profile() as p:
            _ = torch.randn(5)

        table = p.key_averages().table(sort_by="self_cpu_time_total")
        self.assertIsInstance(table, str)

    def test_schedule_without_rpd(self):
        """Scheduling state machine works without rpd."""
        from torch.profiler import schedule

        trace_count = [0]

        def on_trace(prof):
            trace_count[0] += 1

        with rpd_profile(
            schedule=schedule(wait=1, warmup=1, active=2),
            on_trace_ready=on_trace,
        ) as p:
            for _ in range(5):
                _ = torch.randn(10)
                p.step()

        self.assertEqual(trace_count[0], 1)

    def test_trace_file_path(self):
        with rpd_profile() as p:
            pass
        path = p.trace_file_path()
        self.assertIsInstance(path, str)
        self.assertTrue(path.endswith(".rpd"))

    def test_available_returns_bool(self):
        result = _rpd_available()
        self.assertIsInstance(result, bool)


class TestRpdProfilerWithTracer(TestCase):
    """Tests that require librpd_tracer.so to be loaded (via LD_PRELOAD)."""

    def setUp(self):
        if not _rpd_available():
            self.skipTest("librpd_tracer.so not loaded")

    def test_cpu_events_collected(self):
        with rpd_profile() as p:
            x = torch.randn(10, 10)
            _ = x @ x

        events = p.events()
        names = [e.name for e in events]
        self.assertTrue(
            any("randn" in n for n in names),
            f"Expected aten::randn in events, got: {names[:20]}",
        )

    def test_key_averages_table(self):
        with rpd_profile() as p:
            x = torch.randn(10, 10)
            _ = x @ x

        table = p.key_averages().table(sort_by="self_cpu_time_total")
        self.assertIn("Self CPU", table)

    def test_events_have_timing(self):
        with rpd_profile() as p:
            x = torch.randn(100, 100)
            _ = x + x

        events = p.events()
        cpu_events = [e for e in events if "randn" in e.name]
        self.assertGreater(len(cpu_events), 0)
        for e in cpu_events:
            self.assertGreater(e.time_range.end - e.time_range.start, 0)

    def test_schedule_collects_events(self):
        from torch.profiler import schedule

        collected = [None]

        def on_trace(prof):
            collected[0] = prof.events()

        with rpd_profile(
            schedule=schedule(wait=0, warmup=1, active=1, repeat=1),
            on_trace_ready=on_trace,
        ) as p:
            for _ in range(3):
                _ = torch.randn(10, 10)
                p.step()

        self.assertIsNotNone(collected[0])
        self.assertGreater(len(collected[0]), 0)


if __name__ == "__main__":
    run_tests()
