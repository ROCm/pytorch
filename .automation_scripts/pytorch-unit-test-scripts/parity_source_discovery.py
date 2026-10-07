"""Resolve parity sources from the jobs that actually ran for a commit."""

import collections
import datetime
import re
from urllib.parse import urlparse


_TEST_JOB = re.compile(
    r"^(?P<prefix>.+) / (?P<kind>test(?:-osdc)?) "
    r"\((?P<config>[^,]+), (?P<shard>\d+), (?P<total>\d+)"
    r"(?:, (?P<runner>[^)]+))?\)"
)
_RUN_ID = re.compile(r"/actions/runs/(?P<run_id>\d+)(?:/|$)")
_VARIANT_MARKERS = ("rerun_disabled_tests", "mem_leak_check")
_SPECIAL_FAMILIES = ("build-only", "debug", "no-ops", "slow-gradcheck", "smoke")
_CONFIGS = ("default", "distributed", "inductor")
_ROCM_CONFIGS = ("default", "distributed", "distributed_4gpu", "inductor")


class AmbiguousSourceError(RuntimeError):
    """Raised when two different workflow sources are equally suitable."""


def _run_id(check):
    match = _RUN_ID.search(check.get("details_url") or "")
    return int(match.group("run_id")) if match else None


def _platform(prefix):
    normalized = prefix.lower()
    if "rocm" in normalized and "cuda" not in normalized:
        return "rocm"
    if "cuda" in normalized and "rocm" not in normalized:
        return "cuda"
    return None


def _arch(text, parity_config):
    normalized = text.lower()
    # Preview job names also contain mi350, so exact preview aliases win.
    ordered = sorted(
        parity_config["rocm"].items(),
        key=lambda item: item[0] != "preview",
    )
    for arch, config in ordered:
        aliases = config.get("arch_aliases") or [arch]
        if any(alias.lower() in normalized for alias in aliases):
            return arch
    return None


def _created_epoch(value):
    if not value:
        return 0
    try:
        return int(
            datetime.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        )
    except (TypeError, ValueError):
        return 0


def discover_families(check_runs, run_metadata, parity_config):
    """Group check-runs into complete, run-scoped semantic test families."""
    runs = {int(run["id"]): run for run in run_metadata}
    groups = collections.defaultdict(list)
    for check in check_runs:
        name = check.get("name") or ""
        if any(marker in name for marker in _VARIANT_MARKERS):
            continue
        match = _TEST_JOB.match(name)
        run_id = _run_id(check)
        if not match or run_id is None or run_id not in runs:
            continue
        prefix = match.group("prefix")
        platform = _platform(prefix)
        if platform is None:
            continue
        runner = match.group("runner") or ""
        arch = _arch(f"{prefix} {runner}", parity_config) if platform == "rocm" else None
        if platform == "rocm" and arch is None:
            continue
        key = (
            run_id,
            platform,
            arch,
            match.group("config"),
            prefix,
            match.group("kind"),
            int(match.group("total")),
        )
        groups[key].append(check)

    families = []
    for key, checks in groups.items():
        run_id, platform, arch, config, prefix, kind, total = key
        shards = {
            int(_TEST_JOB.match(check["name"]).group("shard"))
            for check in checks
        }
        families.append(
            {
                "run_id": run_id,
                "workflow_path": runs[run_id].get("path") or "",
                "event": runs[run_id].get("event") or "",
                "created_at": runs[run_id].get("created_at") or "",
                "head_sha": runs[run_id].get("head_sha") or "",
                "run_attempt": runs[run_id].get("run_attempt") or 1,
                "platform": platform,
                "arch": arch,
                "config": config,
                "prefix": prefix,
                "kind": kind,
                "total": total,
                "job_ids": sorted(
                    int(check["id"]) for check in checks if check.get("id") is not None
                ),
                "complete": shards == set(range(1, total + 1)),
                "completed": all(check.get("status") == "completed" for check in checks),
                "conclusions": sorted(
                    {check.get("conclusion") for check in checks if check.get("conclusion")}
                ),
            }
        )
    return families


def _hints(parity_config, platform, arch, config):
    source = (
        parity_config["cuda"]
        if platform == "cuda"
        else parity_config["rocm"].get(arch, {})
    )
    return source.get(config) or []


def _workflow_basename(path):
    parsed = urlparse(path).path
    name = parsed.rsplit("/", 1)[-1]
    return name.removesuffix(".yml").removesuffix(".yaml")


def _hint_score(family, hints):
    for index, hint in enumerate(hints):
        workflow_match = _workflow_basename(family["workflow_path"]) == hint["workflow"]
        prefix_match = family["prefix"] == hint["job_prefix"]
        if workflow_match and prefix_match:
            return 2_000 - index
        if workflow_match or prefix_match:
            return 1_000 - index
    return 0


def _event_score(event):
    return {"push": 3, "schedule": 2, "workflow_dispatch": 1}.get(event, 0)


def _bundle_sizes(families):
    bundles = collections.Counter(
        (f["run_id"], f["platform"], f["arch"], f["prefix"])
        for f in families
        if f["complete"] and f["config"] in _CONFIGS
    )
    return bundles


def _select_family(candidates, hints, bundle_sizes):
    if not candidates:
        return None

    def semantic_score(family):
        bundle_key = (
            family["run_id"],
            family["platform"],
            family["arch"],
            family["prefix"],
        )
        normalized = family["prefix"].lower()
        return (
            family["complete"],
            family["completed"],
            bundle_sizes[bundle_key],
            not any(marker in normalized for marker in _SPECIAL_FAMILIES),
            _hint_score(family, hints),
            _event_score(family["event"]),
        )

    best_score = max(semantic_score(family) for family in candidates)
    best = [family for family in candidates if semantic_score(family) == best_score]
    if len(best) == 1:
        return best[0]

    # A rerun of the same workflow is one physical source; prefer the newest.
    paths = {family["workflow_path"] for family in best}
    prefixes = {family["prefix"] for family in best}
    if paths != {""} and len(paths) == 1 and len(prefixes) == 1:
        return max(best, key=lambda family: _created_epoch(family["created_at"]))

    choices = ", ".join(
        f"run {family['run_id']} {family['workflow_path']} {family['prefix']}"
        for family in best
    )
    raise AmbiguousSourceError(f"equally suitable parity sources: {choices}")


def resolve_topology(check_runs, run_metadata, parity_config, archs=None):
    """Return the selected physical source for each semantic parity role."""
    families = discover_families(check_runs, run_metadata, parity_config)
    bundles = _bundle_sizes(families)
    selected = {"version": 1, "cuda": {}, "rocm": {}, "missing": []}

    for config in _CONFIGS:
        candidates = [
            family
            for family in families
            if family["platform"] == "cuda" and family["config"] == config
        ]
        family = _select_family(
            candidates, _hints(parity_config, "cuda", None, config), bundles
        )
        if family is None:
            selected["missing"].append(f"cuda/{config}")
        else:
            selected["cuda"][config] = family

    target_archs = archs or tuple(parity_config["rocm"])
    for arch in target_archs:
        selected["rocm"][arch] = {}
        arch_config = parity_config["rocm"][arch]
        for config in _ROCM_CONFIGS:
            if config not in arch_config:
                continue
            candidates = [
                family
                for family in families
                if family["platform"] == "rocm"
                and family["arch"] == arch
                and family["config"] == config
            ]
            family = _select_family(
                candidates, _hints(parity_config, "rocm", arch, config), bundles
            )
            if family is None:
                selected["missing"].append(f"rocm/{arch}/{config}")
            else:
                selected["rocm"][arch][config] = family

    return selected
