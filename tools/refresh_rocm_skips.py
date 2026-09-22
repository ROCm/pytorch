#!/usr/bin/env python3
"""Reconcile the rocm-skipped-tests issue set against the current test tree.

Dry run by default: it performs no mutations and prints a tally. Mutations
require --apply.

Detection and the GitHub client are imported from find_rocm_skips.py so the
scan stays byte-identical to the script that created the issues.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


_FRS_PATH = Path(__file__).resolve().parent / "find_rocm_skips.py"
_spec = importlib.util.spec_from_file_location("find_rocm_skips", _FRS_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load {_FRS_PATH}")
_frs = importlib.util.module_from_spec(_spec)
sys.modules["find_rocm_skips"] = _frs
_spec.loader.exec_module(_frs)

GitHubClient = _frs.GitHubClient
build_sub_issue_body = _frs.build_sub_issue_body
build_issue_spec = _frs.build_issue_spec
child_issue_key = _frs.child_issue_key
extract_issue_marker = _frs.extract_issue_marker
gather_results = _frs.gather_results
parse_issue_repo = _frs.parse_issue_repo


LABEL = "rocm-skipped-tests"
NEW_ISSUE_LABELS = ["rocm-skipped-tests", "triaged", "module: rocm"]
VERSION_FLOOR_RE = re.compile(
    r"skipIfRocmVersionLessThan|skipIfRocmVersionAtLeast|ROCM_VERSION|"
    r"torch\.version\.hip\s*[<>=]|_get_torch_rocm_version|getRocmVersion"
)
META_FLAG_RE = re.compile(r"IS_FBCODE|IS_REMOTE_GPU|IS_SANDCASTLE")
CONTEXT_END_RE = re.compile(r"^\s*$")


CHECK_RE = re.compile(r"^- \[( |x|X)\]\s+`?([A-Za-z0-9_]+)`?")


def parse_checklist(body: str | None) -> dict[str, bool]:
    """Return {test_name: done} from an existing parent body."""
    out: dict[str, bool] = {}
    for line in (body or "").splitlines():
        m = CHECK_RE.match(line.strip())
        if m:
            out[m.group(2)] = m.group(1).lower() == "x"
    return out


def parent_body_with_checklist(spec, issue_repo: str, existing_body: str | None) -> str:
    """Parent body: context + a checklist of skipped tests.

    A test that was listed before and is no longer detected stays on the list as
    checked, so the issue shows progress instead of silently shrinking.
    """
    detected = {t.name: t.lineno for t in spec.tests}
    previous = parse_checklist(existing_body)
    names = sorted(set(detected) | set(previous))
    lines = [
        f"<!-- ROCM-SKIP:{spec.key} -->",
        "",
        "## Context",
        f"- Source file: `{spec.file_path}`",
        f"- Test class: `{spec.class_name}`",
        f"- ROCm-skipped tests: {len(detected)}",
    ]
    if issue_repo:
        lines.append(
            f"- Code reference: https://github.com/{issue_repo}/blob/main/{spec.file_path}"
        )
    lines += ["", "### Skipped tests", ""]
    for name in names:
        if name in detected:
            lines.append(f"- [ ] `{name}` (line {detected[name]})")
        else:
            lines.append(f"- [x] `{name}` (no longer skipped)")
    return "\n".join(lines).rstrip() + "\n"


def suffix_after_context(body: str | None) -> str:
    """Everything the script does not own (cc-bot appendix, human notes)."""
    if not body:
        return ""
    marker = "\n### Skipped tests"
    if marker in body:
        tail = body.split(marker, 1)[1]
        rest = [
            l for l in tail.splitlines() if l.strip() and not CHECK_RE.match(l.strip())
        ]
        return "\n".join(rest)
    return ""


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def tree_guard(repo_root: Path) -> str:
    """Warn only, per the plan."""
    warnings: list[str] = []
    head = run(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    dirty = run(["git", "-C", str(repo_root), "status", "--porcelain"])
    behind = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "merge-base",
            "--is-ancestor",
            "HEAD",
            "origin/main",
        ],
        capture_output=True,
    ).returncode
    upstream = run(["git", "-C", str(repo_root), "rev-parse", "origin/main"])
    if head != upstream:
        warnings.append(f"HEAD {head[:9]} is not origin/main {upstream[:9]}")
    if behind != 0:
        warnings.append("HEAD is not an ancestor of origin/main (diverged)")
    if dirty:
        warnings.append(f"working tree is dirty ({len(dirty.splitlines())} paths)")
    for w in warnings:
        print(f"WARNING: {w}")
    return head


def scan(repo_root: Path, test_root: Path, issue_repo: str):
    """Return (parents, children) keyed by ROCM-SKIP marker key."""
    results = gather_results(test_root, include_third_party=False)
    parents: dict[str, Any] = {}
    children: dict[str, tuple[Any, Any]] = {}
    for file_result in results:
        for class_result in file_result.classes:
            if not class_result.tests:
                continue
            spec = build_issue_spec(issue_repo, repo_root, file_result, class_result)
            parents[spec.key] = spec
            for test in class_result.tests:
                children[child_issue_key(spec.key, test.name)] = (spec, test)
    return parents, children


def fetch_marked_issues(client: GitHubClient, owner: str, repo: str, verbose: bool):
    """Issues carrying the label AND a ROCM-SKIP marker, keyed by marker."""
    by_key: dict[str, dict[str, Any]] = {}
    unmarked = 0
    total = 0
    params = {"labels": LABEL, "state": "all", "per_page": 100}
    data, headers = client._rest("GET", f"/repos/{owner}/{repo}/issues", params=params)
    while True:
        if isinstance(data, list):
            for issue in data:
                if not isinstance(issue, dict) or "pull_request" in issue:
                    continue
                total += 1
                marker = extract_issue_marker(issue.get("body"))
                if not marker:
                    unmarked += 1
                    continue
                by_key.setdefault(marker.strip(), issue)
        next_link = client._extract_next_link(headers)
        if not next_link:
            break
        data, headers = client._rest_full_url("GET", next_link)
    if verbose:
        print(
            f"[verbose] {total} labelled issues, {len(by_key)} marked, {unmarked} unmarked (ignored)"
        )
    return by_key, total, unmarked


def source_for_key(repo_root: Path, key: str) -> str | None:
    path = key.split("::", 1)[0]
    f = repo_root / path
    if not f.exists():
        return None
    return f.read_text(errors="replace")


def close_triage(repo_root: Path, key: str) -> tuple[str, str]:
    """Return (action, reason) where action is CLOSE or REVIEW."""
    parts = key.split("::")
    path = parts[0]
    src = source_for_key(repo_root, key)
    if src is None:
        return "CLOSE", "source file no longer exists"
    class_name = parts[1] if len(parts) > 1 else None
    test_name = parts[2] if len(parts) > 2 else None
    if class_name and f"class {class_name}" not in src:
        return "CLOSE", f"class {class_name} no longer exists in {path}"
    if test_name and f"def {test_name}(" not in src:
        return "CLOSE", f"test {test_name} no longer exists in {path}"
    # The test still exists but the scan no longer reports a ROCm skip for it.
    # Re-inspect the decorator/body region for what is left.
    region = src
    if test_name:
        idx = src.find(f"def {test_name}(")
        start = max(0, src.rfind("\n\n", 0, max(0, idx - 400)))
        region = src[start : idx + 400]
    if not re.search(r"[Rr]ocm|ROCM|hip", region):
        return "CLOSE", "no ROCm-conditional signal remains"
    if VERSION_FLOOR_RE.search(region):
        return "CLOSE", "remaining ROCm condition is a version floor"
    if META_FLAG_RE.search(region):
        return "CLOSE", "remaining ROCm condition also requires a Meta-internal CI flag"
    return "REVIEW", "ROCm-mentioning condition remains; needs human review"


def context_drifted(issue: dict[str, Any], expected_body: str) -> bool:
    body = issue.get("body") or ""
    exp_ctx = expected_body.split("## Context", 1)
    cur_ctx = body.split("## Context", 1)
    if len(exp_ctx) < 2 or len(cur_ctx) < 2:
        return True

    def ctx_lines(text: str) -> list[str]:
        out = []
        for line in text.splitlines():
            if line.startswith("- "):
                out.append(line.strip())
            elif out and not line.strip():
                break
        return out

    return ctx_lines(exp_ctx[1]) != ctx_lines(cur_ctx[1])


def rename_pairs(
    vanished: dict[str, dict[str, Any]], unmatched: list[str]
) -> dict[str, str]:
    """Pair vanished child keys with new child keys (file move or class rename)."""
    pairs: dict[str, str] = {}
    by_tail: dict[str, list[str]] = defaultdict(list)
    by_file_test: dict[str, list[str]] = defaultdict(list)
    for key in unmatched:
        parts = key.split("::")
        if len(parts) != 3:
            continue
        by_tail[f"{parts[1]}::{parts[2]}"].append(key)
        by_file_test[f"{parts[0]}::{parts[2]}"].append(key)
    old_by_tail: dict[str, list[str]] = defaultdict(list)
    old_by_file_test: dict[str, list[str]] = defaultdict(list)
    for key in vanished:
        parts = key.split("::")
        if len(parts) != 3:
            continue
        old_by_tail[f"{parts[1]}::{parts[2]}"].append(key)
        old_by_file_test[f"{parts[0]}::{parts[2]}"].append(key)
    for tail, olds in old_by_tail.items():
        news = by_tail.get(tail, [])
        if len(olds) == 1 and len(news) == 1 and olds[0] != news[0]:
            pairs[olds[0]] = news[0]
    for ft, olds in old_by_file_test.items():
        news = by_file_test.get(ft, [])
        if len(olds) == 1 and len(news) == 1 and olds[0] != news[0]:
            pairs.setdefault(olds[0], news[0])
    # keep only 1:1
    seen_new: dict[str, int] = defaultdict(int)
    for new in pairs.values():
        seen_new[new] += 1
    return {o: n for o, n in pairs.items() if seen_new[n] == 1}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--apply", action="store_true", help="perform mutations (default: dry run)"
    )
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--test-root", default="test")
    ap.add_argument("--issue-repo", default="pytorch/pytorch")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--max-creations", type=int, default=None)
    ap.add_argument(
        "--with-children",
        action="store_true",
        help="also manage per-test child issues (legacy; default is parents-only)",
    )
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    test_root = (repo_root / args.test_root).resolve()
    owner, repo = parse_issue_repo(args.issue_repo)

    head = tree_guard(repo_root)
    token = run(["gh", "auth", "token"])
    if not token:
        print("ERROR: no GitHub token (gh auth token failed)")
        return 2
    client = GitHubClient(token=token, dry_run=not args.apply, verbose=args.verbose)

    parents, children = scan(repo_root, test_root, args.issue_repo)
    print(
        f"scan: {len(parents)} parent keys + {len(children)} child keys = {len(parents) + len(children)}"
    )

    issues, labelled_total, unmarked = fetch_marked_issues(
        client, owner, repo, args.verbose
    )
    open_issues = {k: i for k, i in issues.items() if i.get("state") == "open"}
    closed_issues = {k: i for k, i in issues.items() if i.get("state") != "open"}
    print(
        f"issues: {labelled_total} labelled, {len(issues)} marker-managed "
        f"({len(open_issues)} open, {len(closed_issues)} closed), {unmarked} unmarked (never touched)"
    )

    if args.with_children:
        scan_keys = set(parents) | set(children)
    else:
        scan_keys = set(parents)
        print("parents-only mode: per-test child issues are not created or reopened")
    create_keys = [k for k in scan_keys if k not in issues]
    vanished = {k: i for k, i in open_issues.items() if k not in scan_keys}
    reopen_keys = [k for k in closed_issues if k in scan_keys]

    renames = rename_pairs(vanished, [k for k in create_keys if k.count("::") == 2])
    for old in renames:
        vanished.pop(old, None)
    create_keys = [k for k in create_keys if k not in set(renames.values())]

    closes: list[tuple[str, str]] = []
    reviews: list[tuple[str, str]] = []
    consolidate: list[str] = []
    if not args.with_children:
        # Pre-existing per-test issues are not stale: their tests are still
        # skipped, they are just no longer the tracking unit. Report them
        # separately instead of triaging them as vanished keys.
        for key in [k for k in sorted(vanished) if k.count("::") == 2]:
            if True:
                consolidate.append(key)
                vanished.pop(key, None)
    for key in sorted(vanished):
        action, reason = close_triage(repo_root, key)
        (closes if action == "CLOSE" else reviews).append((key, reason))

    refreshes: list[str] = []
    for key, issue in sorted(open_issues.items()):
        if key not in scan_keys:
            continue
        if key in parents:
            spec = parents[key]
            expected = (
                spec.body
                if args.with_children
                else parent_body_with_checklist(
                    spec, args.issue_repo, issue.get("body")
                )
            )
        else:
            spec, test = children[key]
            parent_issue = issues.get(spec.key)
            parent_num = parent_issue.get("number") if parent_issue else None
            expected = build_sub_issue_body(spec, test, args.issue_repo, parent_num)
        if context_drifted(issue, expected):
            refreshes.append(key)

    create_parents = [k for k in create_keys if k.count("::") == 1]
    create_children = [k for k in create_keys if k.count("::") == 2]

    if args.verbose:
        for k in sorted(create_keys):
            print(f"CREATE {k}")
        for k, why in closes:
            print(f"CLOSE  {k} [#{vanished_num(issues, k)}] {why}")
        for k, why in reviews:
            print(f"REVIEW {k} [#{vanished_num(issues, k)}] {why}")
        for k in reopen_keys:
            print(f"REOPEN {k} [#{vanished_num(issues, k)}]")
        for k in refreshes:
            print(f"REFRESH {k} [#{vanished_num(issues, k)}]")
        for old, new in renames.items():
            print(f"RENAME {old} -> {new} [#{vanished_num(issues, old)}]")

    print()
    print(
        f"=== refresh tally at {head[:9]} ({'APPLY' if args.apply else 'dry run'}) ==="
    )
    print(
        f"create: {len(create_keys)} ({len(create_parents)} parents + {len(create_children)} children)"
    )
    print(f"close: {len(closes)}")
    print(f"reopen: {len(reopen_keys)}")
    print(f"refresh: {len(refreshes)}")
    print(f"rename-updates: {len(renames)}")
    print(f"needs-human-review: {len(reviews)}")
    if not args.with_children:
        still_skipped = [k for k in consolidate if k in children]
        print(
            f"consolidate: {len(consolidate)} open per-test issues superseded by their parent "
            f"({len(still_skipped)} whose test is still skipped)"
        )
    print(f">>> Warn oncall: {len(create_keys)} new issues will be created.")

    if not args.apply:
        print("(dry run: no mutations performed)")
        return 0

    print(
        "ERROR: --apply is not implemented in this rebuild; the original apply path was lost."
    )
    return 3


def vanished_num(issues: dict[str, dict[str, Any]], key: str) -> Any:
    issue = issues.get(key)
    return issue.get("number") if issue else "?"


if __name__ == "__main__":
    raise SystemExit(main())
