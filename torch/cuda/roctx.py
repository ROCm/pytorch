# mypy: allow-untyped-defs
r"""This package adds support for AMD ROCTX (ROCm Tools Extension) used in profiling.

Mirrors the API of :mod:`torch.cuda.nvtx` so that the same code can be used
on ROCm with ROCTX markers visible to rocprof, rocprofv3, etc.
"""

from contextlib import contextmanager


try:
    from torch._C import _roctx
except ImportError:

    class _ROCTXStub:
        @staticmethod
        def _fail(*args, **kwargs):
            raise RuntimeError(
                "ROCTX functions not installed. Are you sure you have a ROCm build?"
            )

        rangePushA = _fail
        rangePop = _fail
        markA = _fail
        rangeStartA = _fail
        rangeEnd = _fail
        deviceRangeStart = _fail
        deviceRangeEnd = _fail

    _roctx = _ROCTXStub()  # type: ignore[assignment]


__all__ = ["range_push", "range_pop", "range_start", "range_end", "mark", "range"]


def range_push(msg):
    """
    Push a range onto a stack of nested range span.  Returns zero-based depth of the range that is started.

    Args:
        msg (str): ASCII message to associate with range
    """
    return _roctx.rangePushA(msg)


def range_pop():
    """Pop a range off of a stack of nested range spans.  Returns the zero-based depth of the range that is ended."""
    return _roctx.rangePop()


def range_start(msg) -> int:
    """
    Mark the start of a range with string message. It returns an unique handle
    for this range to pass to the corresponding call to range_end().

    A key difference between this and range_push/range_pop is that the
    range_start/range_end version supports range across threads (start on one
    thread and end on another thread).

    Returns: A range handle (uint64_t) that can be passed to range_end().

    Args:
        msg (str): ASCII message to associate with the range.
    """
    return _roctx.rangeStartA(msg)


def range_end(range_id) -> None:
    """
    Mark the end of a range for a given range_id.

    Args:
        range_id (int): an unique handle for the start range.
    """
    _roctx.rangeEnd(range_id)


def _device_range_start(msg: str, stream: int = 0) -> object:
    """
    Marks the start of a range with string message.
    It returns an opaque heap-allocated handle for this range
    to pass to the corresponding call to _device_range_end().

    On ROCm, ROCTX has no stream-callback API; this is a no-op and returns None.

    Args:
        msg (str): ASCII message to associate with the range.
        stream (int): HIP stream id.
    """
    return _roctx.deviceRangeStart(msg, stream)


def _device_range_end(range_handle: object, stream: int = 0) -> None:
    """
    Mark the end of a range for a given range_handle.
    On ROCm, ROCTX has no stream-callback API; this is a no-op.

    Args:
        range_handle: an unique handle for the start range.
        stream (int): HIP stream id.
    """
    _roctx.deviceRangeEnd(range_handle, stream)


def mark(msg):
    """
    Describe an instantaneous event that occurred at some point.

    Args:
        msg (str): ASCII message to associate with the event.
    """
    return _roctx.markA(msg)


@contextmanager
def range(msg, *args, **kwargs):
    """
    Context manager / decorator that pushes a ROCTX range at the beginning
    of its scope, and pops it at the end. If extra arguments are given,
    they are passed as arguments to msg.format().

    Args:
        msg (str): message to associate with the range
    """
    range_push(msg.format(*args, **kwargs))
    try:
        yield
    finally:
        range_pop()
