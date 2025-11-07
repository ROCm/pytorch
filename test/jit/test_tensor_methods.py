# Owner(s): ["oncall: jit"]

import os
import sys

import torch


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing import FileCheck
<<<<<<< HEAD
from torch.testing._internal.jit_utils import JitTestCase


if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


=======
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
class TestTensorMethods(JitTestCase):
    def test_getitem(self):
        def tensor_getitem(inp: torch.Tensor):
            indices = torch.tensor([0, 2], dtype=torch.long)
            return inp.__getitem__(indices)

        inp = torch.rand(3, 4)
        self.checkScript(tensor_getitem, (inp,))

        scripted = torch.jit.script(tensor_getitem)
        FileCheck().check("aten::index").run(scripted.graph)

    def test_getitem_invalid(self):
        def tensor_getitem_invalid(inp: torch.Tensor):
            return inp.__getitem__()

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "expected exactly 1 argument", "inp.__getitem__"
        ):
            torch.jit.script(tensor_getitem_invalid)
<<<<<<< HEAD
=======


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
