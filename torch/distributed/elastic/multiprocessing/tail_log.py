#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import time
from concurrent.futures.thread import ThreadPoolExecutor
from threading import Event
<<<<<<< HEAD
from typing import Callable, Optional, TextIO, TYPE_CHECKING, Union
=======
from typing import Optional, TextIO, TYPE_CHECKING
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))


if TYPE_CHECKING:
    from concurrent.futures._base import Future
<<<<<<< HEAD
    from io import TextIOWrapper
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

__all__ = ["tail_logfile", "TailLog"]

logger = logging.getLogger(__name__)


def tail_logfile(
<<<<<<< HEAD
    header: str,
    file: str,
    dst: TextIO,
    finished: Event,
    interval_sec: float,
    log_line_filter: Optional[Callable[[str], bool]] = None,
=======
    header: str, file: str, dst: TextIO, finished: Event, interval_sec: float
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
):
    while not os.path.exists(file):
        if finished.is_set():
            return
        time.sleep(interval_sec)

    with open(file, errors="replace") as fp:
        while True:
            line = fp.readline()

            if line:
<<<<<<< HEAD
                if log_line_filter and log_line_filter(line):
                    dst.write(f"{header}{line}")
=======
                dst.write(f"{header}{line}")
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
            else:  # reached EOF
                if finished.is_set():
                    # log line producer is finished
                    break
                else:
                    # log line producer is still going
                    # wait for a bit before looping again
                    time.sleep(interval_sec)


class TailLog:
    """
    Tail the given log files.

    The log files do not have to exist when the ``start()`` method is called. The tail-er will gracefully wait until
    the log files are created by the producer and will tail the contents of the
    log files until the ``stop()`` method is called.

    .. warning:: ``TailLog`` will wait indefinitely for the log file to be created!

    Each log file's line will be suffixed with a header of the form: ``[{name}{idx}]:``,
    where the ``name`` is user-provided and ``idx`` is the index of the log file
    in the ``log_files`` mapping. ``log_line_prefixes`` can be used to override the
    header for each log file.

    Usage:

    ::

     log_files = {0: "/tmp/0_stdout.log", 1: "/tmp/1_stdout.log"}
     tailer = TailLog("trainer", log_files, sys.stdout).start()
     # actually run the trainers to produce 0_stdout.log and 1_stdout.log
     run_trainers()
     tailer.stop()

     # once run_trainers() start writing the ##_stdout.log files
     # the tailer will print to sys.stdout:
     # >>> [trainer0]:log_line1
     # >>> [trainer1]:log_line1
     # >>> [trainer0]:log_line2
     # >>> [trainer0]:log_line3
     # >>> [trainer1]:log_line2

    .. note:: Due to buffering log lines between files may not necessarily
              be printed out in order. You should configure your application's
              logger to suffix each log line with a proper timestamp.

    """

    def __init__(
        self,
        name: str,
        log_files: dict[int, str],
<<<<<<< HEAD
        dst: Union[TextIO, str],
        log_line_prefixes: Optional[dict[int, str]] = None,
        interval_sec: float = 0.1,
        log_line_filter: Callable[[str], bool] = (lambda _: True),
=======
        dst: TextIO,
        log_line_prefixes: Optional[dict[int, str]] = None,
        interval_sec: float = 0.1,
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    ):
        n = len(log_files)
        self._threadpool = None
        if n > 0:
<<<<<<< HEAD
            # pyrefly: ignore [bad-assignment]
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
            self._threadpool = ThreadPoolExecutor(
                max_workers=n,
                thread_name_prefix=f"{self.__class__.__qualname__}_{name}",
            )

        self._name = name
<<<<<<< HEAD
        self._dst_file: Optional[TextIOWrapper] = None
        self._dst: Optional[Union[TextIO, TextIOWrapper]] = None
        if isinstance(dst, str):
            try:
                self._dst_file = open(dst, mode="w", errors="replace")
                self._dst = self._dst_file
            except Exception:
                logger.exception("error opening dst file %s.", dst)
                self._dst = None
                self._dst_file = None

        else:
            self._dst = dst
        self._log_files = log_files
        self._log_line_prefixes = log_line_prefixes
        self._log_line_filter = log_line_filter
=======
        self._dst = dst
        self._log_files = log_files
        self._log_line_prefixes = log_line_prefixes
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        self._finished_events: dict[int, Event] = {
            local_rank: Event() for local_rank in log_files.keys()
        }
        self._futs: list[Future] = []
        self._interval_sec = interval_sec
        self._stopped = False

    def start(self) -> "TailLog":
<<<<<<< HEAD
        if not self._threadpool or not self._dst:
=======
        if not self._threadpool:
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
            return self

        for local_rank, file in self._log_files.items():
            header = f"[{self._name}{local_rank}]:"
            if self._log_line_prefixes and local_rank in self._log_line_prefixes:
                header = self._log_line_prefixes[local_rank]
            self._futs.append(
                self._threadpool.submit(
                    tail_logfile,
                    header=header,
                    file=file,
                    dst=self._dst,
                    finished=self._finished_events[local_rank],
                    interval_sec=self._interval_sec,
<<<<<<< HEAD
                    log_line_filter=self._log_line_filter,
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
                )
            )
        return self

    def stop(self) -> None:
        for finished in self._finished_events.values():
            finished.set()

        for local_rank, f in enumerate(self._futs):
            try:
                f.result()
            except Exception as e:
<<<<<<< HEAD
                logger.exception(
                    "error in log tailor for %s%s. %s",
                    self._name,
                    local_rank,
                    e.__class__.__qualname__,
=======
                logger.error(
                    "error in log tailor for %s%s. %s: %s",
                    self._name,
                    local_rank,
                    e.__class__.__qualname__,
                    e,
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
                )

        if self._threadpool:
            self._threadpool.shutdown(wait=True)

<<<<<<< HEAD
        if self._dst_file:
            self._dst_file.close()

=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        self._stopped = True

    def stopped(self) -> bool:
        return self._stopped
