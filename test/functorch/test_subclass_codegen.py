# Owner(s): ["module: functorch"]

import logging
from contextlib import contextmanager

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.two_tensor import TwoTensor


trace_log = logging.getLogger("torch.__trace")


class TestSubclassCodegen(TestCase):
    @contextmanager
    def _capture_wrapper_source(self):
        """Capture subclass_wrapper artifacts from the structured trace log."""
        captured: list[str] = []

        class _ArtifactHandler(logging.Handler):
            def emit(self, record):
                metadata = getattr(record, "metadata", {})
                if (
                    "artifact" in metadata
                    and metadata["artifact"].get("name") == "subclass_wrapper"
                ):
                    payload = getattr(record, "payload", None)
                    if payload is not None:
                        captured.append(payload)

        handler = _ArtifactHandler()
        handler.setLevel(logging.DEBUG)
        old_level = trace_log.level
        trace_log.setLevel(logging.DEBUG)
        trace_log.addHandler(handler)
        try:
            yield captured
        finally:
            trace_log.removeHandler(handler)
            trace_log.setLevel(old_level)

    def test_compile_simple(self):
        """torch.compile with TwoTensor produces correct wrapper and output."""
        with self._capture_wrapper_source() as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x * 2

            a = torch.randn(4)
            b = torch.randn(4)
            tt = TwoTensor(a, b)
            out = f(tt)

        self.assertIsInstance(out, TwoTensor)
        self.assertEqual(out.a, a * 2)
        self.assertEqual(out.b, b * 2)

        self.assertEqual(len(captured), 1)
        self.assertExpectedInline(
            captured[0],
            """\
def inner_fn(args):
    unwrapped_args = []
    _inp_0 = args[0]
    assert type(_inp_0) is _expected_type_1, f'expected {_expected_type_1}, got {type(_inp_0)}'
    unwrapped_args.append(_inp_0.a)
    unwrapped_args.append(_inp_0.b)
    args.clear()
    unwrapped_outs = compiled_fn(unwrapped_args)
    wrapped_outs = []
    _out_inner_2 = {'a': unwrapped_outs[0], 'b': unwrapped_outs[1]}
    _out_5 = _subclass_type_3.__tensor_unflatten__(_out_inner_2, _meta_4, (4,), (1,))
    wrapped_outs.append(_out_5)
    return tuple(wrapped_outs)""",
        )

    def test_compile_nested_subclass(self):
        """torch.compile with nested TwoTensor produces correct recursive wrapper."""
        with self._capture_wrapper_source() as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x.sin().cos()

            a1 = torch.randn(4)
            a2 = torch.randn(4)
            a3 = torch.randn(4)
            a4 = torch.randn(4)
            inner_a = TwoTensor(a1, a2)
            inner_b = TwoTensor(a3, a4)
            tt = TwoTensor(inner_a, inner_b)
            out = f(tt)

        self.assertIsInstance(out, TwoTensor)
        self.assertIsInstance(out.a, TwoTensor)
        self.assertIsInstance(out.b, TwoTensor)
        self.assertEqual(out.a.a, inner_a.a.sin().cos())
        self.assertEqual(out.b.b, inner_b.b.sin().cos())

        self.assertEqual(len(captured), 1)
        self.assertExpectedInline(
            captured[0],
            """\
def inner_fn(args):
    unwrapped_args = []
    _inp_0 = args[0]
    assert type(_inp_0) is _expected_type_1, f'expected {_expected_type_1}, got {type(_inp_0)}'
    _inner_2 = _inp_0.a
    unwrapped_args.append(_inner_2.a)
    unwrapped_args.append(_inner_2.b)
    _inner_3 = _inp_0.b
    unwrapped_args.append(_inner_3.a)
    unwrapped_args.append(_inner_3.b)
    args.clear()
    unwrapped_outs = compiled_fn(unwrapped_args)
    wrapped_outs = []
    _out_inner_5 = {'a': unwrapped_outs[0], 'b': unwrapped_outs[1]}
    _out_8 = _subclass_type_6.__tensor_unflatten__(_out_inner_5, _meta_7, (4,), (1,))
    _out_inner_9 = {'a': unwrapped_outs[2], 'b': unwrapped_outs[3]}
    _out_12 = _subclass_type_10.__tensor_unflatten__(_out_inner_9, _meta_11, (4,), (1,))
    _out_inner_4 = {'a': _out_8, 'b': _out_12}
    _out_15 = _subclass_type_13.__tensor_unflatten__(_out_inner_4, _meta_14, (4,), (1,))
    wrapped_outs.append(_out_15)
    return tuple(wrapped_outs)""",
        )


if __name__ == "__main__":
    run_tests()
