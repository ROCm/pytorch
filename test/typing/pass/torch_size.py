<<<<<<< HEAD
from typing_extensions import assert_type
=======
# mypy: enable-error-code=unused-ignore

from typing_extensions import assert_type, Never
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

from torch import Size


<<<<<<< HEAD
s1 = Size([1, 2, 3])
s2 = Size([1, 2, 3])


=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
class ZeroIndex:
    def __index__(self) -> int:
        return 0


<<<<<<< HEAD
# __getitem__
assert_type(s1[0], int)
assert_type(s1[ZeroIndex()], int)
assert_type(s1[:2], Size)
# __add__
assert_type(s1 + s2, Size)
assert_type(s1 + (1, 2), Size)
# Size has no __radd__, so tuple.__add__(right, left) is called
assert_type((1, 2) + s1, tuple[int, ...])
# __mul__
assert_type(s1 * 3, Size)
assert_type(s1 * ZeroIndex(), Size)
assert_type(3 * s1, Size)
assert_type(ZeroIndex() * s1, Size)
=======
tup0: tuple[()] = ()
tup1: tuple[int] = (1,)
tup2: tuple[int, int] = (1, 2)
tupN: tuple[int, int, int] = (1, 2, 3)
tupX: tuple[Never, ...] = tuple()
s = Size([1, 2, 3])

# assignability to tuple
t: tuple[int, ...] = s

# __getitem__
assert_type(s[0], int)
assert_type(s[ZeroIndex()], int)
assert_type(s[:2], Size)
# __add__
assert_type(s + s, Size)
assert_type(s + tup0, Size)
assert_type(s + tup1, Size)
assert_type(s + tup2, Size)
assert_type(s + tupN, Size)
assert_type(s + tupX, Size)
# __radd__
# NOTE: currently incorrect inference, see: https://github.com/python/mypy/issues/19006
assert_type(tup0 + s, Size)  # type: ignore[assert-type]
assert_type(tup1 + s, Size)  # type: ignore[assert-type]
assert_type(tup2 + s, Size)  # type: ignore[assert-type]
assert_type(tupN + s, Size)  # type: ignore[assert-type]
assert_type(tupX + s, Size)  # type: ignore[assert-type]
# __mul__
assert_type(s * 3, Size)
assert_type(s * ZeroIndex(), Size)
assert_type(3 * s, Size)
assert_type(ZeroIndex() * s, Size)
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
