# mypy: allow-untyped-defs
"""Adds docstrings to Storage functions"""

import torch._C
from torch._C import _add_docstr as add_docstr


storage_classes = ["StorageBase"]


def add_docstr_all(method, docstr):
    for cls_name in storage_classes:
        cls = getattr(torch._C, cls_name)
        try:
            add_docstr(getattr(cls, method), docstr)
        except AttributeError:
            pass


add_docstr_all(
    "from_file",
    """
<<<<<<< HEAD
from_file(filename, shared=False, size=0) -> Storage
=======
from_file(filename, shared=False, nbytes=0) -> Storage
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

Creates a CPU storage backed by a memory-mapped file.

If ``shared`` is ``True``, then memory is shared between all processes.
All changes are written to the file. If ``shared`` is ``False``, then the changes on
the storage do not affect the file.

<<<<<<< HEAD
``size`` is the number of elements in the storage. If ``shared`` is ``False``,
then the file must contain at least ``size * sizeof(Type)`` bytes
(``Type`` is the type of storage, in the case of an ``UnTypedStorage`` the file must contain at
least ``size`` bytes). If ``shared`` is ``True`` the file will be created if needed.
=======
``nbytes`` is the number of bytes of storage. If ``shared`` is ``False``,
then the file must contain at least ``nbytes`` bytes. If ``shared`` is
``True`` the file will be created if needed. (Note that for ``UntypedStorage``
this argument differs from that of ``TypedStorage.from_file``)
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

Args:
    filename (str): file name to map
    shared (bool): whether to share memory (whether ``MAP_SHARED`` or ``MAP_PRIVATE`` is passed to the
                    underlying `mmap(2) call <https://man7.org/linux/man-pages/man2/mmap.2.html>`_)
<<<<<<< HEAD
    size (int): number of elements in the storage
=======
    nbytes (int): number of bytes of storage
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
""",
)
