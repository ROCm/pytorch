# Defined in torch/csrc/monitor/python_init.cpp

import datetime
from enum import Enum
from types import TracebackType
<<<<<<< HEAD
from typing import Callable, Optional
=======
from typing import Callable
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))

class Aggregation(Enum):
    VALUE = ...
    MEAN = ...
    COUNT = ...
    SUM = ...
    MAX = ...
    MIN = ...

class Stat:
    name: str
    count: int
    def __init__(
        self,
        name: str,
        aggregations: list[Aggregation],
        window_size: int,
        max_samples: int = -1,
    ) -> None: ...
    def add(self, v: float) -> None: ...
    def get(self) -> dict[Aggregation, float]: ...

class Event:
    name: str
    timestamp: datetime.datetime
    data: dict[str, int | float | bool | str]
    def __init__(
        self,
        name: str,
        timestamp: datetime.datetime,
        data: dict[str, int | float | bool | str],
    ) -> None: ...

def log_event(e: Event) -> None: ...

class EventHandlerHandle: ...

def register_event_handler(handler: Callable[[Event], None]) -> EventHandlerHandle: ...
def unregister_event_handler(handle: EventHandlerHandle) -> None: ...

class _WaitCounterTracker:
    def __enter__(self) -> None: ...
    def __exit__(
        self,
<<<<<<< HEAD
        exec_type: Optional[type[BaseException]] = None,
        exec_value: Optional[BaseException] = None,
        traceback: Optional[TracebackType] = None,
=======
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    ) -> None: ...

class _WaitCounter:
    def __init__(self, key: str) -> None: ...
    def guard(self) -> _WaitCounterTracker: ...
