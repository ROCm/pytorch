import difflib
import re


def _is_rocm_job_prefix(prefix):
    prefix_lower = prefix.lower()
    return "rocm" in prefix_lower and "cuda" not in prefix_lower


def _candidate_job_prefixes(names, test_config):
    """Count prefixes from '<prefix> / <kind> (<config>, <idx>, <total>, ...)'."""
    pattern = re.compile(
        r"^(.+?) / \S+ \(" + re.escape(test_config) + r", \d+, \d+"
    )
    candidates = {}
    for name in names:
        match = pattern.match(name)
        if match:
            prefix = match.group(1)
            candidates[prefix] = candidates.get(prefix, 0) + 1
    return candidates


def choose_fuzzy_job_prefix(names, test_config, configured_prefix):
    """Choose the closest scoped ROCm prefix, preserving an exact match."""
    candidates = {
        prefix: count
        for prefix, count in _candidate_job_prefixes(names, test_config).items()
        if _is_rocm_job_prefix(prefix)
    }
    if not candidates or configured_prefix in candidates:
        return configured_prefix
    return max(
        candidates,
        key=lambda prefix: (
            difflib.SequenceMatcher(
                None, prefix, configured_prefix
            ).ratio(),
            candidates[prefix],
        ),
    )
