"""Hook system for Dynamo's guard functionality.

This module provides a way to register callback functions that are triggered during
guard-related operations.

The Hooks class manages two types of hook functions:
- guard_export_fn: Called when guards need to be exported, taking a GuardsSet as input
- guard_fail_fn: Called when a guard check fails, taking a GuardFail object as input
<<<<<<< HEAD

=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
These hooks enable customization of guard export and failure handling behaviors.
"""

import dataclasses
from typing import Callable, Optional

from torch._guards import GuardsSet

<<<<<<< HEAD
from .types import GuardFail
=======
from .types import GuardFail, GuardFilterEntry
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))


@dataclasses.dataclass
class Hooks:
    guard_export_fn: Optional[Callable[[GuardsSet], None]] = None
    guard_fail_fn: Optional[Callable[[GuardFail], None]] = None
<<<<<<< HEAD
=======
    guard_filter_fn: Optional[Callable[[list[GuardFilterEntry]], list[bool]]] = None
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
