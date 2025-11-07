<<<<<<< HEAD
import copy
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar, Union
=======
# mypy: allow-untyped-defs
import copy
import warnings
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

from torch.utils.data.datapipes.datapipe import MapDataPipe


<<<<<<< HEAD
_T = TypeVar("_T")

__all__ = ["SequenceWrapperMapDataPipe"]


class SequenceWrapperMapDataPipe(MapDataPipe[_T]):
=======
__all__ = ["SequenceWrapperMapDataPipe"]


class SequenceWrapperMapDataPipe(MapDataPipe):
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    r"""
    Wraps a sequence object into a MapDataPipe.

    Args:
        sequence: Sequence object to be wrapped into an MapDataPipe
        deepcopy: Option to deepcopy input sequence object

    .. note::
      If ``deepcopy`` is set to False explicitly, users should ensure
      that data pipeline doesn't contain any in-place operations over
      the iterable instance, in order to prevent data inconsistency
      across iterations.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp = SequenceWrapper(range(10))
        >>> list(dp)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
<<<<<<< HEAD
        >>> dp = SequenceWrapper({"a": 100, "b": 200, "c": 300, "d": 400})
        >>> dp["a"]
        100
    """

    sequence: Union[Sequence[_T], Mapping[Any, _T]]

    def __init__(
        self, sequence: Union[Sequence[_T], Mapping[Any, _T]], deepcopy: bool = True
    ) -> None:
=======
        >>> dp = SequenceWrapper({'a': 100, 'b': 200, 'c': 300, 'd': 400})
        >>> dp['a']
        100
    """

    def __init__(self, sequence, deepcopy=True):
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        if deepcopy:
            try:
                self.sequence = copy.deepcopy(sequence)
            except TypeError:
                warnings.warn(
                    "The input sequence can not be deepcopied, "
<<<<<<< HEAD
                    "please be aware of in-place modification would affect source data",
                    stacklevel=2,
=======
                    "please be aware of in-place modification would affect source data"
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
                )
                self.sequence = sequence
        else:
            self.sequence = sequence

<<<<<<< HEAD
    def __getitem__(self, index: int) -> _T:
        return self.sequence[index]

    def __len__(self) -> int:
=======
    def __getitem__(self, index):
        return self.sequence[index]

    def __len__(self):
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        return len(self.sequence)
