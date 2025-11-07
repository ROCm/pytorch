# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import signal
import subprocess
<<<<<<< HEAD
from collections.abc import Generator
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
from contextlib import contextmanager


@contextmanager
<<<<<<< HEAD
def magic_trace(
    output: str = "trace.fxt", magic_trace_cache: str = "/tmp/magic-trace"
) -> Generator[None, None, None]:
=======
def magic_trace(output="trace.fxt", magic_trace_cache="/tmp/magic-trace"):
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    pid = os.getpid()
    if not os.path.exists(magic_trace_cache):
        print(f"Downloading magic_trace to: {magic_trace_cache}")
        subprocess.run(
            [
                "wget",
                "-O",
                magic_trace_cache,
                "-q",
                "https://github.com/janestreet/magic-trace/releases/download/v1.0.2/magic-trace",
            ]
        )
        subprocess.run(["chmod", "+x", magic_trace_cache])
    args = [magic_trace_cache, "attach", "-pid", str(pid), "-o", output]
    p = subprocess.Popen(args, stderr=subprocess.PIPE, encoding="utf-8")
<<<<<<< HEAD
    assert p.stderr is not None
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    while True:
        x = p.stderr.readline()
        print(x)
        if "Attached" in x:
            break
    try:
        yield
    finally:
        p.send_signal(signal.SIGINT)
        r = p.wait()
<<<<<<< HEAD
        if p.stderr is not None:
            print(p.stderr.read())
            p.stderr.close()
=======
        print(p.stderr.read())
        p.stderr.close()
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        if r != 0:
            raise ValueError(f"magic_trace exited abnormally: {r}")
