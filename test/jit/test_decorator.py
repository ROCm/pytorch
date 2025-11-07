# Owner(s): ["oncall: jit"]
<<<<<<< HEAD
# flake8: noqa

import sys
import unittest
from enum import Enum
from typing import List, Optional
=======

>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

import torch
from jit.myfunction_a import my_function_a
from torch.testing._internal.jit_utils import JitTestCase


class TestDecorator(JitTestCase):
    def test_decorator(self):
        # Note: JitTestCase.checkScript() does not work with decorators
        # self.checkScript(my_function_a, (1.0,))
        # Error:
        #   RuntimeError: expected def but found '@' here:
        #   @my_decorator
        #   ~ <--- HERE
        #   def my_function_a(x: float) -> float:
        # Do a simple torch.jit.script() test instead
        fn = my_function_a
        fx = torch.jit.script(fn)
        self.assertEqual(fn(1.0), fx(1.0))
<<<<<<< HEAD
=======


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
