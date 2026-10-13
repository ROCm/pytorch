"""Caller-supplied Triton configuration candidates for codegen kernels.

A caller who has measured their own workload can hand Inductor a short list of
configurations per kernel category, which the compiler benchmarks alongside the ones it
generates. Nothing here asserts that a configuration is best for a kernel, only that it is
worth trying: the compiler re-benchmarks the list on the machine it is compiling for.

See README.md for the table format and the precedence rules.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from collections.abc import Callable

    from triton import Config


log = logging.getLogger(__name__)

__all__ = ["apply", "lookup"]

# Block sizes are passed positionally to the config constructors, in this order, so a
# supplied entry names them rather than relying on the caller to order them. "r0_" covers
# the reduction extent under either of the names size_hints uses for it.
BLOCK_ARGS: dict[str, tuple[str, ...]] = {
    "pointwise": ("XBLOCK", "YBLOCK", "ZBLOCK"),
    "reduction": ("XBLOCK", "R0_BLOCK"),
    "persistent_reduction": ("XBLOCK", "R0_BLOCK"),
}

# Everything else a supplied entry may carry. These reach the constructor as keywords, so
# the set is deliberately narrow: an unknown key is a caller mistake, not a stale config.
KNOB_ARGS: dict[str, frozenset[str]] = {
    "pointwise": frozenset(
        {"num_warps", "num_stages", "num_elements_per_warp", "waves_per_eu", "kpack"}
    ),
    "reduction": frozenset({"num_warps", "num_stages", "waves_per_eu"}),
    "persistent_reduction": frozenset({"num_warps", "num_stages", "waves_per_eu"}),
}

MODES = frozenset({"append", "replace"})


class InvalidTuningList(ValueError):
    """A supplied table is malformed.

    Raised rather than warned about, because a key that is not a knob or a block size that
    is not an integer is a mistake in the caller's own table and is true of every kernel.
    A configuration that merely cannot be built for one kernel's shape is dropped instead.
    """


def _device_name(device: Any) -> str | None:
    """The device key a table may be specialised on, or None if it cannot be determined."""
    name = getattr(device, "cc", None)
    if isinstance(name, str) and name:
        return name
    try:
        return torch.cuda.get_device_properties().gcnArchName
    except Exception:  # noqa: BLE001 - absent or uninitialised device, not an error here
        return None


def lookup(
    category: str, size_hints: dict[str, int], device: Any = None
) -> dict[str, Any] | None:
    """Find the entry for this category and arity, device-specific key first."""
    table = torch._inductor.config.tuning_list.table
    if not table:
        return None

    suffix = f"{category}/{len(size_hints)}"
    name = _device_name(device)
    for key in (f"{name}+{suffix}" if name else None, suffix):
        if key is not None and key in table:
            entry = table[key]
            if not isinstance(entry, dict) or "configs" not in entry:
                raise InvalidTuningList(
                    f"tuning_list entry {key!r} must be a dict carrying 'configs', got "
                    f"{type(entry).__name__}"
                )
            mode = entry.get("mode", "append")
            if mode not in MODES:
                raise InvalidTuningList(
                    f"tuning_list entry {key!r} has mode {mode!r}, expected one of "
                    f"{sorted(MODES)}"
                )
            return {"key": key, "mode": mode, "configs": entry["configs"]}
    return None


def _split(category: str, supplied: Any, key: str) -> tuple[list[int], dict[str, Any]]:
    """Separate a supplied entry into positional block sizes and keyword knobs.

    Raises on anything the table itself got wrong. Block sizes must be positive integers
    and must be contiguous from the first axis, because that is how the constructors take
    them; a gap would silently shift a value onto the wrong axis.
    """
    if not isinstance(supplied, dict):
        raise InvalidTuningList(
            f"tuning_list entry {key!r} holds {type(supplied).__name__}, expected a dict "
            "of block sizes and knobs"
        )

    names = BLOCK_ARGS[category]
    unknown = set(supplied) - set(names) - KNOB_ARGS[category]
    if unknown:
        raise InvalidTuningList(
            f"tuning_list entry {key!r} carries {sorted(unknown)}, which {category} does "
            f"not accept. Block sizes are {list(names)}; knobs are "
            f"{sorted(KNOB_ARGS[category])}"
        )

    blocks = []
    for name in names:
        if name not in supplied:
            break
        value = supplied[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise InvalidTuningList(
                f"tuning_list entry {key!r} has {name}={value!r}, expected a positive int"
            )
        blocks.append(value)
    if not blocks:
        raise InvalidTuningList(
            f"tuning_list entry {key!r} names no block size; expected at least {names[0]}"
        )
    missing = [n for n in names[len(blocks) :] if n in supplied]
    if missing:
        raise InvalidTuningList(
            f"tuning_list entry {key!r} gives {missing} without the block sizes before "
            f"them; {category} takes {list(names)} in order"
        )

    return blocks, {k: v for k, v in supplied.items() if k not in names}


def apply(
    category: str,
    size_hints: dict[str, int],
    configs: list[Config],
    make_config: Callable[..., Config],
    device: Any = None,
    kernel_name: str | None = None,
) -> list[Config]:
    """Fold a supplied list into the candidates the heuristic generated.

    Called after the heuristic has returned, so a supplied list reaches kernels the
    heuristic gave a single candidate to -- which is the case it has most to offer, and
    the one every injection point inside the heuristics misses.
    """
    entry = lookup(category, size_hints, device)
    if entry is None:
        return configs

    supplied = []
    for one in entry["configs"]:
        blocks, knobs = _split(category, one, entry["key"])
        try:
            supplied.append(make_config(size_hints, *blocks, **knobs))
        except Exception as exc:  # noqa: BLE001 - a failure here is about this kernel's shape
            # A list fitted on one corpus meets kernels it was not fitted on, so a config
            # that will not build for this shape is expected rather than exceptional.
            log.warning(
                "tuning_list: dropping %s from %r for %s: %s",
                one,
                entry["key"],
                kernel_name or "this kernel",
                exc,
            )

    if not supplied:
        return configs
    return supplied if entry["mode"] == "replace" else [*configs, *supplied]
