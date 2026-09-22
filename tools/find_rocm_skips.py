#!/usr/bin/env python3
"""
Collect tests that are skipped on ROCm within the PyTorch test suite.

The script scans Python test files (default: `pytorch/test`) for a variety of
patterns that cause tests to be disabled on ROCm. The output is written to a
JSON file containing the results grouped by file and class, along with summary
statistics.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import http.client
import json
import os
import re
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


ROCM_DETAIL_PATTERN = re.compile(
    r"(rocm|hip|amd|mi\d{2,3}|amdsmi|miopen|gfx\d{2,3})", re.IGNORECASE
)

ROCM_CONDITION_NAMES = (
    "TEST_WITH_ROCM",
    "torch.version.hip",
    "torch.version.rocm",
)

ISSUE_MARKER_PREFIX = "ROCM-SKIP"
ISSUE_MARKER_RE = re.compile(r"<!--\s*ROCM-SKIP:([^>]+)\s*-->")
ROCM_SKIP_COUNT_LINE_RE = re.compile(
    r"^-\s*ROCm-skipped tests:\s*\d+\s*$", re.MULTILINE
)

ROCM_SKIP_DECORATOR_SUFFIXES = {
    "skipifrocm",
    "skipifrocmarch",
    "skipifrocmmultiprocess",
    "skipifrocmarchmultiprocess",
    "skipcudaifrocm",
    "skiponrocm",
    "skiprocmiiftorchinductor",
}
ROCM_SKIP_CALL_SUFFIXES = ROCM_SKIP_DECORATOR_SUFFIXES | {
    "skiptest",
}
NON_ROCM_SKIP_DECORATOR_SUFFIXES = {
    "skipcudaifnotrocm",
    "skipcudaifnohipdnn",
    "runonrocm",
    "skipifrocmversionlessthan",
    "skipifrocmverlessthanmultiprocess",
    "skipcudaifrocmversionlessthan",
}


def expr_to_text(node: ast.AST) -> str:
    try:
        return ast.unparse(node)  # type: ignore[attr-defined]
    except Exception:
        return ""


def contains_rocm_text(text: str) -> bool:
    return bool(text and ROCM_DETAIL_PATTERN.search(text))


def expr_contains_rocm(node: ast.AST) -> bool:
    return contains_rocm_text(expr_to_text(node))


def get_full_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = get_full_name(node.value)
        if base:
            return f"{base}.{node.attr}"
        return node.attr
    if isinstance(node, ast.Call):
        return get_full_name(node.func)
    if isinstance(node, ast.Subscript):
        return get_full_name(node.value)
    return None


def normalize_name(name: Optional[str]) -> str:
    if not name:
        return ""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def matches_suffix(normalized_name: str, suffixes: Sequence[str]) -> bool:
    return any(normalized_name.endswith(suffix) for suffix in suffixes)


def is_false_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is False


def is_true_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def is_none_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _is_rocm_indicator(node: ast.AST) -> Optional[str]:
    """Return the indicator name if *node* is a known ROCm condition, else None."""
    name = get_full_name(node)
    if name is None:
        return None
    for indicator in ROCM_CONDITION_NAMES:
        if name == indicator:
            return indicator
    return None


def classify_condition_node(node: ast.AST) -> Tuple[Optional[str], Optional[str]]:
    """
    Classify an AST condition expression for ROCm relevance by walking
    the boolean structure.

    Returns (classification, indicator) where classification is one of:
        - "positive": condition is True on ROCm regardless of other variables
        - "negative": condition is False on ROCm regardless of other variables
        - None: unable to classify (compound / runtime-dependent)

    Key semantics:
        - Or(a, b):  "positive" if ANY operand is "positive"
        - And(a, b): "positive" only if ALL operands are "positive"
        - Not(x):    flips "positive" <-> "negative"
        - Leaf:      "positive" if it's a ROCm indicator name, else None

    This ensures that compound conditions like
        TEST_WITH_ROCM and (not torch.cuda.has_magma)
    return None (not sufficient), while
        TEST_WITH_ROCM or IS_WINDOWS
    returns "positive" (ROCm alone is sufficient).
    """
    indicator = _is_rocm_indicator(node)
    if indicator is not None:
        return "positive", indicator

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        inner_cls, inner_ind = classify_condition_node(node.operand)
        if inner_cls == "positive":
            return "negative", inner_ind
        if inner_cls == "negative":
            return "positive", inner_ind
        return None, None

    if isinstance(node, ast.BoolOp):
        results = [classify_condition_node(v) for v in node.values]
        if isinstance(node.op, ast.Or):
            for cls, ind in results:
                if cls == "positive":
                    return "positive", ind
            for cls, ind in results:
                if cls == "negative":
                    return "negative", ind
            return None, None
        if isinstance(node.op, ast.And):
            all_positive = all(cls == "positive" for cls, _ in results)
            if all_positive and results:
                return "positive", results[0][1]
            any_negative = any(cls == "negative" for cls, _ in results)
            if any_negative:
                return "negative", next(
                    ind for cls, ind in results if cls == "negative"
                )
            return None, None

    if isinstance(node, ast.Compare):
        left_name = get_full_name(node.left)
        if left_name:
            for indicator in ROCM_CONDITION_NAMES:
                if left_name != indicator:
                    continue
                if len(node.ops) == 1 and len(node.comparators) == 1:
                    op = node.ops[0]
                    comp = node.comparators[0]
                    if isinstance(op, (ast.Is, ast.Eq)):
                        if is_false_constant(comp) or is_none_constant(comp):
                            return "negative", indicator
                        if is_true_constant(comp):
                            return "positive", indicator
                    elif isinstance(op, (ast.IsNot, ast.NotEq)):
                        if is_true_constant(comp):
                            return "negative", indicator
                        if is_false_constant(comp):
                            return "positive", indicator
                return None, None

    return None, None


def classify_condition_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Fallback: parse a condition string and classify via AST.

    Used only when the caller has text but no AST node (e.g. visit_If
    where the condition text is already extracted).
    """
    if not text:
        return None, None
    try:
        tree = ast.parse(text, mode="eval")
        return classify_condition_node(tree.body)
    except SyntaxError:
        return None, None


def block_has_skip_behavior(statements: Sequence[ast.stmt]) -> bool:
    for stmt in statements:
        if isinstance(stmt, ast.Return):
            return True
        if isinstance(stmt, ast.Raise):
            exc = stmt.exc
            if exc is None:
                continue
            if expr_contains_rocm(exc):
                return True
            if isinstance(exc, ast.Call):
                name = get_full_name(exc.func)
                if name and "skiptest" in name.lower():
                    return True
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call_name = get_full_name(stmt.value.func)
            if call_name and "skip" in call_name.lower():
                return True
    return False


def merge_category_maps(
    target: Dict[str, List[Dict[str, Optional[str]]]],
    source: Dict[str, List[Dict[str, Optional[str]]]],
) -> None:
    for name, occurrences in source.items():
        target.setdefault(name, [])
        for occurrence in occurrences:
            if occurrence not in target[name]:
                target[name].append(occurrence)


def add_category(
    categories: Dict[str, List[Dict[str, Optional[str]]]],
    name: str,
    node: ast.AST,
    detail: Optional[str] = None,
    skip_on_rocm: Optional[bool] = None,
) -> None:
    if skip_on_rocm is False:
        return
    occurrence: Dict[str, Optional[str]] = {
        "line": getattr(node, "lineno", None),
    }
    if detail:
        occurrence["detail"] = detail
    if skip_on_rocm is not None:
        occurrence["skip_on_rocm"] = skip_on_rocm
    categories.setdefault(name, [])
    if occurrence not in categories[name]:
        categories[name].append(occurrence)


def detect_decorator_categories(
    decorators: Sequence[ast.expr],
) -> Dict[str, List[Dict[str, Optional[str]]]]:
    categories: Dict[str, List[Dict[str, Optional[str]]]] = {}
    for deco in decorators:
        if isinstance(deco, ast.Call):
            target = deco.func
            args = list(deco.args) + [kw.value for kw in deco.keywords]
        else:
            target = deco
            args = []

        name = get_full_name(target)
        normalized = normalize_name(name)
        detail = expr_to_text(deco)

        skip_on_rocm: Optional[bool] = None

        if normalized.startswith("unittestskip"):
            is_skip_unless = "unless" in normalized
            condition_node = args[0] if args else None
            if condition_node is None:
                if not contains_rocm_text(detail):
                    continue
                skip_on_rocm = True
            else:
                classification, _ = classify_condition_node(condition_node)
                if is_skip_unless:
                    if classification != "negative":
                        continue
                else:
                    if classification != "positive":
                        continue
                skip_on_rocm = True
            add_category(
                categories,
                "decorator:unittest.skip_rocm",
                deco,
                detail,
                skip_on_rocm=skip_on_rocm,
            )
            continue

        if not ("skip" in normalized or "runonrocm" in normalized):
            continue

        if matches_suffix(normalized, NON_ROCM_SKIP_DECORATOR_SUFFIXES):
            continue

        if matches_suffix(
            normalized, ROCM_SKIP_DECORATOR_SUFFIXES
        ) or normalized.endswith("runonrocmarch"):
            skip_on_rocm = True
        else:
            condition_node = args[0] if args else None
            if condition_node is None:
                if not contains_rocm_text(detail):
                    continue
                skip_on_rocm = True
            else:
                classification, _ = classify_condition_node(condition_node)
                if classification != "positive":
                    continue
                skip_on_rocm = True

        add_category(
            categories,
            f"decorator:{name or 'unknown'}",
            deco,
            detail,
            skip_on_rocm=skip_on_rocm,
        )

    return categories


class FunctionBodyAnalyzer(ast.NodeVisitor):
    def __init__(self) -> None:
        self.categories: Dict[str, List[Dict[str, Optional[str]]]] = {}
        self._rocm_branch_stack: List[bool] = []

    def _in_rocm_branch(self) -> bool:
        return any(self._rocm_branch_stack)

    def _visit_block(
        self, block: Sequence[ast.stmt], rocm_context: bool = False
    ) -> None:
        if rocm_context:
            self._rocm_branch_stack.append(True)
        for stmt in block:
            self.visit(stmt)
        if rocm_context:
            self._rocm_branch_stack.pop()

    def visit_If(self, node: ast.If) -> None:
        condition_text = expr_to_text(node.test)
        classification, indicator = classify_condition_node(node.test)

        if classification == "positive":
            if block_has_skip_behavior(node.body):
                guard_name = f"guard:{indicator or 'ROCM'}"
                add_category(
                    self.categories,
                    guard_name,
                    node,
                    condition_text,
                    skip_on_rocm=True,
                )
            self._visit_block(node.body, rocm_context=True)
            self._visit_block(node.orelse, rocm_context=False)
        elif classification == "negative":
            self._visit_block(node.body, rocm_context=False)
            if node.orelse:
                if block_has_skip_behavior(node.orelse):
                    guard_name = f"guard:{indicator or 'ROCM'}"
                    add_category(
                        self.categories,
                        guard_name,
                        node,
                        f"else branch via: {condition_text}",
                        skip_on_rocm=True,
                    )
                self._visit_block(node.orelse, rocm_context=True)
        else:
            self._visit_block(node.body, rocm_context=False)
            self._visit_block(node.orelse, rocm_context=False)

    def visit_Call(self, node: ast.Call) -> None:
        name = get_full_name(node.func)
        normalized = normalize_name(name)
        detail = expr_to_text(node)
        args = list(node.args) + [kw.value for kw in node.keywords]

        in_rocm_branch = self._in_rocm_branch()

        if normalized.endswith("skiptest") or (
            name and name.lower().endswith(".skiptest")
        ):
            skip_on_rocm = in_rocm_branch or any(
                expr_contains_rocm(arg) for arg in args
            )
            if skip_on_rocm:
                category_name = (
                    "call:self.skipTest"
                    if name and "self." in name
                    else "call:skipTest"
                )
                add_category(
                    self.categories, category_name, node, detail, skip_on_rocm=True
                )
            return

        if not ("skip" in normalized or "runonrocm" in normalized):
            return

        if matches_suffix(normalized, NON_ROCM_SKIP_DECORATOR_SUFFIXES):
            return

        skip_on_rocm: Optional[bool] = None

        if matches_suffix(normalized, ROCM_SKIP_CALL_SUFFIXES) or normalized.endswith(
            "runonrocmarch"
        ):
            skip_on_rocm = True
        else:
            if args:
                classification, _ = classify_condition_node(args[0])
                if classification == "positive":
                    skip_on_rocm = True
                else:
                    skip_on_rocm = False
            else:
                skip_on_rocm = in_rocm_branch or contains_rocm_text(detail)

        if skip_on_rocm:
            add_category(
                self.categories,
                f"call:{name or 'unknown'}",
                node,
                detail,
                skip_on_rocm=True,
            )

        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        if node.exc:
            detail = expr_to_text(node.exc)
            name = None
            if isinstance(node.exc, ast.Call):
                name = get_full_name(node.exc.func)
            elif isinstance(node.exc, ast.Name):
                name = node.exc.id

            if (
                (name and "skiptest" in name.lower())
                or expr_contains_rocm(node.exc)
                or self._in_rocm_branch()
            ):
                add_category(
                    self.categories,
                    "raise:SkipTest",
                    node,
                    detail,
                    skip_on_rocm=self._in_rocm_branch(),
                )
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if self._in_rocm_branch():
            add_category(
                self.categories,
                "return:rocm_guard",
                node,
                expr_to_text(node),
                skip_on_rocm=True,
            )
        self.generic_visit(node)


def analyze_function(node: ast.AST) -> Dict[str, List[Dict[str, Optional[str]]]]:
    analyzer = FunctionBodyAnalyzer()
    analyzer.visit(node)
    return analyzer.categories


def categories_to_list(
    categories: Dict[str, List[Dict[str, Optional[str]]]],
) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for name in sorted(categories.keys()):
        entry: Dict[str, object] = {"name": name, "occurrences": categories[name]}
        output.append(entry)
    return output


@dataclass
class TestResult:
    name: str
    lineno: int
    categories: List[Dict[str, object]]


@dataclass
class ClassResult:
    name: str
    lineno: int
    class_categories: List[Dict[str, object]]
    tests: List[TestResult]


@dataclass
class FileResult:
    path: str
    classes: List[ClassResult]


@dataclass
class IssueSpec:
    key: str
    title: str
    body: str
    file_path: str
    class_name: str
    tests: List[TestResult]
    include_checkboxes: bool = False


def process_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    inherited_categories: Dict[str, List[Dict[str, Optional[str]]]],
) -> Optional[TestResult]:
    if not node.name.startswith("test"):
        return None

    own_categories = detect_decorator_categories(node.decorator_list)
    body_categories = analyze_function(node)

    merge_category_maps(own_categories, body_categories)
    if inherited_categories:
        merge_category_maps(own_categories, inherited_categories)

    if not own_categories:
        return None

    if not categories_have_rocm_skip(own_categories):
        return None

    return TestResult(
        name=node.name,
        lineno=node.lineno,
        categories=categories_to_list(own_categories),
    )


def process_class(node: ast.ClassDef) -> Optional[ClassResult]:
    class_categories = detect_decorator_categories(node.decorator_list)
    tests: List[TestResult] = []

    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result = process_function(child, class_categories)
            if result:
                tests.append(result)

    if not tests:
        return None

    return ClassResult(
        name=node.name,
        lineno=node.lineno,
        class_categories=categories_to_list(class_categories),
        tests=tests,
    )


def process_module_level_tests(
    module: ast.Module,
) -> Optional[ClassResult]:
    tests: List[TestResult] = []
    class_categories: Dict[str, List[Dict[str, Optional[str]]]] = {}

    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result = process_function(node, {})
            if result:
                tests.append(result)

    if not tests:
        return None

    return ClassResult(
        name="__module__",
        lineno=0,
        class_categories=categories_to_list(class_categories),
        tests=tests,
    )


def process_file(path: Path) -> Optional[FileResult]:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    try:
        module = ast.parse(source, filename=str(path))
    except SyntaxError:
        return None

    classes: List[ClassResult] = []

    for node in module.body:
        if isinstance(node, ast.ClassDef):
            class_result = process_class(node)
            if class_result:
                classes.append(class_result)

    module_tests = process_module_level_tests(module)
    if module_tests:
        classes.append(module_tests)

    if not classes:
        return None

    return FileResult(path=str(path), classes=classes)


def gather_results(search_root: Path, include_third_party: bool) -> List[FileResult]:
    results: List[FileResult] = []
    for file_path in sorted(search_root.rglob("*.py")):
        if file_path.name == "__init__.py":
            continue
        if not include_third_party and "third_party" in file_path.parts:
            continue
        file_result = process_file(file_path)
        if file_result:
            results.append(file_result)
    return results


def build_stats(results: Iterable[FileResult]) -> Tuple[int, Counter]:
    total_tests = 0
    category_counter: Counter = Counter()

    for file_result in results:
        for class_result in file_result.classes:
            for test in class_result.tests:
                total_tests += 1
                for category in test.categories:
                    name = category["name"]
                    occurrences = category.get("occurrences") or []
                    count = 0
                    if isinstance(occurrences, list):
                        for occurrence in occurrences:
                            if isinstance(occurrence, dict):
                                if occurrence.get("skip_on_rocm", True):
                                    count += 1
                    if count:
                        category_counter[name] += count

    return total_tests, category_counter


def _relative_repo_path(path_str: str, repo_root: Path) -> str:
    path = Path(path_str)
    try:
        rel = path.resolve().relative_to(repo_root)
        return rel.as_posix()
    except Exception:
        normalized = path.as_posix()
        # If --test-root is absolute (possibly from another checkout), keep paths
        # stable for issue markers by trimming to the test subtree when possible.
        match = re.search(r"(?:^|/)test/.*", normalized)
        if match:
            return match.group(0).lstrip("/")
        return normalized


def _compact_detail(detail: Optional[str], width: int = 160) -> Optional[str]:
    if not detail:
        return None
    clean = " ".join(detail.split())
    if len(clean) <= width:
        return clean
    return clean[: max(width - 1, 1)] + "…"


def _format_category_occurrences(
    categories: List[Dict[str, object]],
) -> List[str]:
    lines: List[str] = []
    for category in categories:
        name = category.get("name", "unknown")
        occurrences = category.get("occurrences") or []
        if not isinstance(occurrences, list):
            occurrences = [occurrences]
        if not occurrences:
            lines.append(f"- {name}")
            continue
        for occ in occurrences:
            if not isinstance(occ, dict):
                lines.append(f"- {name}")
                continue
            detail = _compact_detail(occ.get("detail"))
            lineno = occ.get("line")
            fragments: List[str] = []
            if detail:
                fragments.append(f"`{detail}`")
            if lineno is not None:
                fragments.append(f"line {lineno}")
            suffix = f" ({', '.join(fragments)})" if fragments else ""
            lines.append(f"- {name}{suffix}")
    return lines


def _format_checkbox_lines(tests: List[TestResult]) -> List[str]:
    lines: List[str] = []
    for test in tests:
        category_names = ", ".join(category["name"] for category in test.categories)
        header = f"- [ ] `{test.name}`"
        if category_names:
            header += f" - {category_names}"
        lines.append(header)
        for category in test.categories:
            occurrences = category.get("occurrences") or []
            if not isinstance(occurrences, list):
                occurrences = [occurrences]
            for occ in occurrences:
                if not isinstance(occ, dict):
                    continue
                detail = _compact_detail(occ.get("detail"))
                lineno = occ.get("line")
                fragments: List[str] = []
                if detail:
                    fragments.append(f"`{detail}`")
                if lineno is not None:
                    fragments.append(f"line {lineno}")
                if fragments:
                    lines.append(f"  - {category['name']}: {', '.join(fragments)}")
    return lines


def _aggregate_category_counts(
    class_result: ClassResult,
) -> Counter:
    counter: Counter = Counter()
    for test in class_result.tests:
        for category in test.categories:
            name = category.get("name", "unknown")
            occurrences = category.get("occurrences") or []
            count = 0
            if isinstance(occurrences, list):
                for occurrence in occurrences:
                    if isinstance(occurrence, dict):
                        if occurrence.get("skip_on_rocm", True):
                            count += 1
            if count == 0:
                continue
            counter[name] += count
    return counter


def categories_have_rocm_skip(
    categories: Dict[str, List[Dict[str, Optional[str]]]],
) -> bool:
    for occurrences in categories.values():
        for occurrence in occurrences:
            if occurrence.get("skip_on_rocm"):
                return True
    return False


def build_issue_spec(
    issue_repo: Optional[str],
    repo_root: Path,
    file_result: FileResult,
    class_result: ClassResult,
) -> IssueSpec:
    relative_path = _relative_repo_path(file_result.path, repo_root)
    key = f"{relative_path}::{class_result.name}"
    marker = f"<!-- {ISSUE_MARKER_PREFIX}:{key} -->"

    lines: List[str] = [marker, ""]
    lines.append("## Context")
    lines.append(f"- Source file: `{relative_path}`")
    lines.append(f"- Test class: `{class_result.name}`")
    lines.append(f"- ROCm-skipped tests: {len(class_result.tests)}")

    if issue_repo:
        code_url = f"https://github.com/{issue_repo}/blob/main/{relative_path}"
        lines.append(f"- Code reference: {code_url}")

    body = "\n".join(lines).rstrip() + "\n"
    title = f"TestModule: {relative_path}::{class_result.name}"
    return IssueSpec(
        key=key,
        title=title,
        body=body,
        file_path=relative_path,
        class_name=class_result.name,
        tests=class_result.tests,
    )


def update_parent_skip_count_line(
    body: Optional[str],
    skip_count: int,
) -> Tuple[str, bool]:
    desired_line = f"- ROCm-skipped tests: {skip_count}"

    if body is None:
        return f"## Context\n{desired_line}\n", True

    line_ending = "\r\n" if "\r\n" in body else "\n"
    normalized = body.replace("\r\n", "\n")

    match = ROCM_SKIP_COUNT_LINE_RE.search(normalized)
    if match:
        current_line = match.group(0)
        if current_line == desired_line:
            return body, False
        updated = normalized[: match.start()] + desired_line + normalized[match.end() :]
        if not updated.endswith("\n"):
            updated += "\n"
        if line_ending == "\r\n":
            updated = updated.replace("\n", "\r\n")
        return updated, True

    lines = normalized.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]

    context_header_index: Optional[int] = None
    for idx, line in enumerate(lines):
        if line.strip() == "## Context":
            context_header_index = idx
            break

    if context_header_index is not None:
        section_end = len(lines)
        for idx in range(context_header_index + 1, len(lines)):
            if lines[idx].startswith("## "):
                section_end = idx
                break
        insert_at = section_end
        while (
            insert_at > context_header_index + 1 and lines[insert_at - 1].strip() == ""
        ):
            insert_at -= 1
        lines.insert(insert_at, desired_line)
    else:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append("## Context")
        lines.append(desired_line)

    updated = "\n".join(lines).rstrip("\n") + "\n"
    if line_ending == "\r\n":
        updated = updated.replace("\n", "\r\n")
    return updated, True


def build_sub_issue_body(
    parent_spec: IssueSpec,
    test: TestResult,
    issue_repo: Optional[str],
    parent_issue_number: Optional[int],
) -> str:
    child_marker = f"<!-- {ISSUE_MARKER_PREFIX}:{parent_spec.key}::{test.name} -->"
    lines: List[str] = [child_marker, ""]
    lines.append("## Context")
    lines.append(f"- Source file: `{parent_spec.file_path}`")
    lines.append(f"- Test class: `{parent_spec.class_name}`")
    lines.append(f"- Test name: `{test.name}`")
    lines.append(f"- Defined at line: {test.lineno}")
    if parent_issue_number is not None:
        lines.append(f"- Parent issue: #{parent_issue_number}")
    if issue_repo:
        code_url = f"https://github.com/{issue_repo}/blob/main/{parent_spec.file_path}"
        lines.append(f"- Code reference: {code_url}")
    return "\n".join(lines).rstrip() + "\n"


def child_issue_key(parent_key: str, test_name: str) -> str:
    return f"{parent_key}::{test_name}"


def create_sub_issues_for_parent(
    client: GitHubClient,
    owner: str,
    repo: str,
    parent_issue: Dict[str, Any],
    parent_spec: IssueSpec,
    issue_repo: Optional[str],
    issue_labels: Optional[List[str]],
    issue_assignees: Optional[List[str]],
    existing_markers: Dict[str, Dict[str, Any]],
    project_id: Optional[str],
) -> None:
    parent_number = parent_issue.get("number")
    if parent_number is None and not client.dry_run:
        print("Unable to determine parent issue number; skipping sub-issue creation.")
        return

    parent_node_id = parent_issue.get("node_id")
    if project_id and parent_node_id:
        try:
            client.add_issue_to_project(project_id, parent_node_id)
        except GitHubAPIError as exc:
            print(f"    Failed to add parent #{parent_number} to project: {exc}")

    for test in parent_spec.tests:
        key = child_issue_key(parent_spec.key, test.name)
        existing_child = existing_markers.get(key)
        child_issue: Optional[Dict[str, Any]] = None

        if existing_child:
            child_issue = existing_child
            child_number = child_issue.get("number")
            print(
                f"Sub-issue already exists for {key} "
                f"(#{child_number if child_number is not None else 'unknown'}); skipping creation."
            )

        else:
            title = (
                f"Test: {parent_spec.class_name}.{test.name}"
                if parent_spec.class_name
                else f"Test: {test.name}"
            )
            body = build_sub_issue_body(
                parent_spec,
                test,
                issue_repo,
                parent_number,
            )
            child_issue = client.create_issue(
                owner,
                repo,
                title,
                body,
                labels=issue_labels,
                assignees=issue_assignees,
            )
            if child_issue is None:
                continue
            existing_markers[key] = child_issue
            child_number = child_issue.get("number")
            print(
                f"  Created sub-issue #{child_number if child_number is not None else 'unknown'} "
                f"for {parent_spec.class_name}.{test.name}"
            )

        if child_issue is None:
            continue

        child_number = child_issue.get("number")
        child_node_id = child_issue.get("node_id")
        child_issue_id = child_issue.get("id")

        if project_id and child_node_id:
            try:
                client.add_issue_to_project(project_id, child_node_id)
            except GitHubAPIError as exc:
                print(f"    Failed to add sub-issue #{child_number} to project: {exc}")

        if child_issue_id is not None and existing_child is None:
            try:
                client.link_sub_issue(owner, repo, parent_number, child_issue_id)
            except GitHubAPIError as exc:
                print(f"    Failed to link sub-issue #{child_number}: {exc}")


def generate_issue_specs(
    results: Iterable[FileResult],
    repo_root: Path,
    issue_repo: Optional[str],
) -> List[IssueSpec]:
    specs: List[IssueSpec] = []
    for file_result in results:
        for class_result in file_result.classes:
            if not class_result.tests:
                continue
            specs.append(
                build_issue_spec(
                    issue_repo,
                    repo_root,
                    file_result,
                    class_result,
                )
            )
    return specs


class GitHubAPIError(RuntimeError):
    pass


class GitHubClient:
    def __init__(
        self, token: str, dry_run: bool = False, verbose: bool = False
    ) -> None:
        self.token = token
        self.dry_run = dry_run
        self.verbose = verbose
        self.api_base = "https://api.github.com"
        self.graphql_url = f"{self.api_base}/graphql"

    def _request(
        self,
        method: str,
        url: str,
        data: Optional[bytes] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[bytes, Dict[str, str]]:
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Accept", "application/vnd.github+json")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if extra_headers:
            for key, value in extra_headers.items():
                req.add_header(key, value)
        try:
            with urllib.request.urlopen(req) as response:
                if self.verbose:
                    print(f"[verbose] {method} {url}")
                try:
                    payload = response.read()
                except http.client.IncompleteRead as err:
                    if self.verbose:
                        print(
                            "[verbose] Received incomplete read from GitHub; "
                            "continuing with partial payload."
                        )
                    payload = err.partial
                headers = {k: v for k, v in response.getheaders()}
                return payload, headers
        except urllib.error.HTTPError as error:
            message = error.read().decode("utf-8", errors="ignore")
            raise GitHubAPIError(
                f"GitHub API {method} {url} failed: {error.code} {error.reason}\n{message}"
            ) from error

    def _json_from_bytes(self, payload: bytes) -> Any:
        if not payload:
            return None
        return json.loads(payload.decode("utf-8"))

    def _rest(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, object]] = None,
        data: Optional[Dict[str, object]] = None,
    ) -> Tuple[Any, Dict[str, str]]:
        url = f"{self.api_base}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        body: Optional[bytes] = None
        if data is not None:
            body = json.dumps(data).encode("utf-8")
        payload, headers = self._request(method, url, body)
        return self._json_from_bytes(payload), headers

    def _rest_full_url(
        self,
        method: str,
        url: str,
    ) -> Tuple[Any, Dict[str, str]]:
        payload, headers = self._request(method, url)
        return self._json_from_bytes(payload), headers

    @staticmethod
    def _extract_next_link(headers: Dict[str, str]) -> Optional[str]:
        link_header = headers.get("Link")
        if not link_header:
            return None
        for part in link_header.split(","):
            segment = part.strip()
            if 'rel="next"' in segment:
                start = segment.find("<")
                end = segment.find(">")
                if start != -1 and end != -1 and end > start:
                    return segment[start + 1 : end]
        return None

    def list_repo_issues(
        self, owner: str, repo: str, state: str = "all"
    ) -> Iterator[Dict[str, Any]]:
        params = {"state": state, "per_page": 100}
        page = 1
        data, headers = self._rest(
            "GET", f"/repos/{owner}/{repo}/issues", params=params
        )
        while True:
            if isinstance(data, list):
                if self.verbose:
                    print(
                        f"[verbose] Retrieved {len(data)} issue(s) from page {page} "
                        f"of {owner}/{repo}."
                    )
                for issue in data:
                    if isinstance(issue, dict) and "pull_request" not in issue:
                        yield issue
            next_link = self._extract_next_link(headers)
            if not next_link:
                break
            page += 1
            data, headers = self._rest_full_url("GET", next_link)

    def list_project_issues(
        self,
        project_id: str,
        repo_full_name: str,
        page_size: int = 50,
    ) -> Iterator[Dict[str, Any]]:
        query = """
            query($projectId: ID!, $pageSize: Int!, $cursor: String) {
              node(id: $projectId) {
                ... on ProjectV2 {
                  items(first: $pageSize, after: $cursor) {
                    pageInfo {
                      hasNextPage
                      endCursor
                    }
                    nodes {
                      content {
                        __typename
                        ... on Issue {
                          id
                          databaseId
                          number
                          title
                          body
                          url
                          repository {
                            nameWithOwner
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
        """
        cursor: Optional[str] = None
        page = 1
        while True:
            variables = {
                "projectId": project_id,
                "pageSize": page_size,
                "cursor": cursor,
            }
            data = self.graphql(query, variables)
            node = data.get("node") if isinstance(data, dict) else None
            project = node if isinstance(node, dict) else None
            items = project.get("items") if project else None
            if not items:
                break
            nodes = items.get("nodes") or []
            if self.verbose:
                print(
                    f"[verbose] Retrieved {len(nodes)} project item(s) from page {page}."
                )
            for item in nodes:
                content = item.get("content") if isinstance(item, dict) else None
                if not isinstance(content, dict):
                    continue
                if content.get("__typename") != "Issue":
                    continue
                repo_info = content.get("repository")
                if (
                    isinstance(repo_info, dict)
                    and repo_info.get("nameWithOwner") != repo_full_name
                ):
                    continue
                issue = {
                    "number": content.get("number"),
                    "body": content.get("body"),
                    "node_id": content.get("id"),
                    "id": content.get("databaseId"),
                    "title": content.get("title"),
                    "url": content.get("url"),
                }
                yield issue
            page_info = items.get("pageInfo") if isinstance(items, dict) else None
            if not page_info or not page_info.get("hasNextPage"):
                break
            cursor = page_info.get("endCursor")
            page += 1

    def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str,
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        payload: Dict[str, Any] = {"title": title, "body": body}
        if labels:
            payload["labels"] = labels
        if assignees:
            payload["assignees"] = assignees
        if self.dry_run:
            print(f"[dry-run] Would create issue '{title}' in {owner}/{repo}")
            return None
        response, _ = self._rest("POST", f"/repos/{owner}/{repo}/issues", data=payload)
        if not isinstance(response, dict):
            raise GitHubAPIError("Unexpected response when creating issue")
        return response

    def graphql(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Any:
        payload = json.dumps({"query": query, "variables": variables or {}}).encode(
            "utf-8"
        )
        response_bytes, _ = self._request(
            "POST", self.graphql_url, payload, {"Content-Type": "application/json"}
        )
        response = self._json_from_bytes(response_bytes)
        if not isinstance(response, dict):
            raise GitHubAPIError("Invalid GraphQL response")
        if response.get("errors"):
            raise GitHubAPIError(f"GraphQL error: {response['errors']}")
        return response.get("data")

    def fetch_project_id(
        self, owner: str, number: int, project_type: str = "user"
    ) -> str:
        if project_type not in {"user", "org"}:
            raise ValueError("project_type must be 'user' or 'org'")
        if project_type == "user":
            query = """
                query($login: String!, $number: Int!) {
                  user(login: $login) {
                    projectV2(number: $number) {
                      id
                    }
                  }
                }
            """
        else:
            query = """
                query($login: String!, $number: Int!) {
                  organization(login: $login) {
                    projectV2(number: $number) {
                      id
                    }
                  }
                }
            """
        data = self.graphql(query, {"login": owner, "number": number})
        container = (
            data.get("user") if project_type == "user" else data.get("organization")
        )
        project = container.get("projectV2") if isinstance(container, dict) else None
        if not project or "id" not in project:
            raise GitHubAPIError(f"Unable to locate project {owner} #{number}")
        return project["id"]

    def add_issue_to_project(self, project_id: str, content_id: str) -> None:
        if self.dry_run:
            print(f"[dry-run] Would add content {content_id} to project {project_id}")
            return
        mutation = """
            mutation($projectId: ID!, $contentId: ID!) {
              addProjectV2ItemById(input: {projectId: $projectId, contentId: $contentId}) {
                item { id }
              }
            }
        """
        self.graphql(mutation, {"projectId": project_id, "contentId": content_id})

    def update_issue(
        self,
        owner: str,
        repo: str,
        number: int,
        *,
        title: Optional[str] = None,
        body: Optional[str] = None,
    ) -> None:
        payload: Dict[str, Any] = {}
        if title is not None:
            payload["title"] = title
        if body is not None:
            payload["body"] = body
        if not payload:
            return
        if self.dry_run:
            print(f"[dry-run] Would update issue #{number} in {owner}/{repo}")
            return
        self._rest(
            "PATCH",
            f"/repos/{owner}/{repo}/issues/{number}",
            data=payload,
        )

    def link_sub_issue(
        self,
        owner: str,
        repo: str,
        parent_number: int,
        sub_issue_id: int,
    ) -> None:
        if self.dry_run:
            print(
                f"[dry-run] Would link sub-issue id {sub_issue_id} to issue #{parent_number}"
            )
            return
        data = {"sub_issue_id": sub_issue_id}
        self._rest(
            "POST",
            f"/repos/{owner}/{repo}/issues/{parent_number}/sub_issues",
            data=data,
        )


def parse_issue_repo(repo: str) -> Tuple[str, str]:
    if "/" not in repo:
        raise ValueError(f"Expected owner/repo format, received '{repo}'")
    owner, name = repo.split("/", 1)
    if not owner or not name:
        raise ValueError(f"Invalid repository identifier: '{repo}'")
    return owner, name


def extract_issue_marker(body: Optional[str]) -> Optional[str]:
    if not body:
        return None
    match = ISSUE_MARKER_RE.search(body)
    if match:
        return match.group(1).strip()
    return None


def collect_existing_issue_markers(
    client: GitHubClient,
    owner: str,
    repo: str,
    *,
    project_id: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    markers: Dict[str, Dict[str, Any]] = {}
    repo_full_name = f"{owner}/{repo}"
    if project_id:
        iterator: Iterable[Dict[str, Any]] = client.list_project_issues(
            project_id, repo_full_name
        )
    else:
        iterator = client.list_repo_issues(owner, repo, state="all")
    for issue in iterator:
        marker = extract_issue_marker(issue.get("body"))
        if marker and marker not in markers:
            markers[marker] = issue
    return markers


def make_output_payload(
    results: List[FileResult],
    search_root: Path,
    include_third_party: bool,
) -> Dict[str, object]:
    total_tests, category_counter = build_stats(results)

    payload = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "search_root": str(search_root),
        "include_third_party": include_third_party,
        "total_tests": total_tests,
        "stats_by_category": dict(sorted(category_counter.items())),
        "files": [
            {
                "path": file_result.path,
                "classes": [
                    {
                        "name": class_result.name,
                        "lineno": class_result.lineno,
                        "class_categories": class_result.class_categories,
                        "tests": [
                            {
                                "name": test.name,
                                "lineno": test.lineno,
                                "categories": test.categories,
                            }
                            for test in class_result.tests
                        ],
                    }
                    for class_result in file_result.classes
                ],
            }
            for file_result in results
        ],
    }

    return payload


def print_stats(payload: Dict[str, object]) -> None:
    total_tests = payload.get("total_tests", 0)
    stats = payload.get("stats_by_category", {})
    print(f"Total skipped tests detected: {total_tests}")
    if not stats:
        print("No categories identified.")
        return
    print("Breakdown by category:")
    for name, count in sorted(stats.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {name}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect tests skipped on ROCm across the PyTorch test suite.",
    )
    parser.add_argument(
        "--test-root",
        default="test",
        help="Path (relative to repo root) containing tests to scan.",
    )
    parser.add_argument(
        "--output",
        default="rocm_skipped_tests.json",
        help="Destination JSON file (relative to repo root unless absolute).",
    )
    parser.add_argument(
        "--include-third-party",
        action="store_true",
        help="Include files under directories named 'third_party'.",
    )
    parser.add_argument(
        "--issue-repo",
        help="owner/repo where issues should be created (e.g. pytorch/pytorch).",
    )
    parser.add_argument(
        "--project-owner",
        help="Login for the project that should receive created issues.",
    )
    parser.add_argument(
        "--project-number",
        type=int,
        help="Number of the target GitHub project (for projectv2).",
    )
    parser.add_argument(
        "--project-type",
        choices=["user", "org"],
        default="user",
        help="Project owner type (user or org).",
    )
    parser.add_argument(
        "--issue-label",
        dest="issue_labels",
        action="append",
        help="Label to apply to created issues (repeatable).",
    )
    parser.add_argument(
        "--issue-assignee",
        dest="issue_assignees",
        action="append",
        help="Assign user(s) to created issues (repeatable).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview issue creation without making API mutations.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information while running.",
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        help="Process at most this many issue candidates (useful for testing).",
    )
    parser.add_argument(
        "--category-limit",
        type=int,
        default=20,
        help="Maximum number of category summary entries per issue.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    search_root = (repo_root / args.test_root).resolve()

    results = gather_results(search_root, include_third_party=args.include_third_party)
    payload = make_output_payload(results, search_root, args.include_third_party)

    if Path(args.output).is_absolute():
        output_path = Path(args.output)
    else:
        output_path = repo_root / args.output

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote ROCm skip report to: {output_path}")
    print_stats(payload)

    if args.issue_repo:
        token = os.environ.get("PROJECT_ACCESS_TOKEN")
        if not token and not args.dry_run:
            raise SystemExit(
                "PROJECT_ACCESS_TOKEN environment variable is required when --issue-repo is provided."
            )

        if bool(args.project_owner) ^ bool(args.project_number):
            raise SystemExit(
                "Both --project-owner and --project-number must be supplied together when targeting a project."
            )

        owner, repo = parse_issue_repo(args.issue_repo)
        issue_specs = generate_issue_specs(
            results,
            repo_root,
            args.issue_repo,
        )

        if args.max_issues is not None:
            issue_specs = issue_specs[: args.max_issues]

        if not issue_specs:
            print("No ROCm-skipped tests detected; skipping issue creation.")
            return

        client: Optional[GitHubClient] = None
        existing_markers: Dict[str, Dict[str, Any]] = {}
        project_id: Optional[str] = None

        if token:
            client = GitHubClient(
                token=token, dry_run=args.dry_run, verbose=args.verbose
            )

            if args.project_owner and args.project_number is not None:
                project_id = client.fetch_project_id(
                    args.project_owner,
                    args.project_number,
                    project_type=args.project_type,
                )
                print(
                    f"Resolved project ID {project_id} for {args.project_owner} project #{args.project_number}."
                )

            if args.verbose:
                scope = (
                    f"project {project_id}"
                    if project_id
                    else f"repository {owner}/{repo}"
                )
                print(f"[verbose] Collecting existing issue markers from {scope}.")

            existing_markers = collect_existing_issue_markers(
                client, owner, repo, project_id=project_id
            )
            print(
                f"Found {len(existing_markers)} existing ROCm skip issue markers in "
                f"{owner}/{repo}{' within project scope' if project_id else ''}."
            )
        else:
            print(
                "[dry-run] PROJECT_ACCESS_TOKEN not set; skipping GitHub API calls and duplicate detection."
            )

        if client is None:
            for spec in issue_specs:
                print(f"[dry-run] Would create issue '{spec.title}':")
                print(spec.body)
                print("-" * 60)
            print(
                f"[dry-run] Previewed {len(issue_specs)} issue(s) without contacting GitHub."
            )
            return

        planned = 0
        created = 0
        skipped_existing = 0

        for spec in issue_specs:
            existing_issue = existing_markers.get(spec.key)
            if existing_issue:
                skipped_existing += 1
                issue_number = existing_issue.get("number")
                print(
                    f"Skipping existing issue for {spec.key} "
                    f"(#{issue_number if issue_number is not None else 'unknown'})"
                )
                updated_body, body_changed = update_parent_skip_count_line(
                    existing_issue.get("body"),
                    len(spec.tests),
                )
                if body_changed:
                    if issue_number is None:
                        print(
                            "  Unable to update parent issue count line (missing issue number)."
                        )
                    else:
                        try:
                            client.update_issue(
                                owner,
                                repo,
                                issue_number,
                                body=updated_body,
                            )
                            existing_issue["body"] = updated_body
                            print(
                                f"  Updated parent issue #{issue_number} ROCm-skipped tests count to {len(spec.tests)}."
                            )
                        except GitHubAPIError as exc:
                            print(f"  Failed to update parent issue count line: {exc}")
                create_sub_issues_for_parent(
                    client,
                    owner,
                    repo,
                    existing_issue,
                    spec,
                    args.issue_repo,
                    args.issue_labels,
                    args.issue_assignees,
                    existing_markers,
                    project_id,
                )
                continue

            planned += 1
            print(f"Preparing issue for {spec.key}")
            response = client.create_issue(
                owner,
                repo,
                spec.title,
                spec.body,
                labels=args.issue_labels,
                assignees=args.issue_assignees,
            )
            if response is not None:
                created += 1
                existing_markers[spec.key] = response
                issue_number = response.get("number")
                node_id = response.get("node_id")
                if project_id and node_id:
                    client.add_issue_to_project(project_id, node_id)
                    print(f"  Added issue #{issue_number} to project.")
                create_sub_issues_for_parent(
                    client,
                    owner,
                    repo,
                    response,
                    spec,
                    args.issue_repo,
                    args.issue_labels,
                    args.issue_assignees,
                    existing_markers,
                    project_id,
                )
            elif args.dry_run:
                stub = {"number": None, "node_id": None, "id": None}
                create_sub_issues_for_parent(
                    client,
                    owner,
                    repo,
                    stub,
                    spec,
                    args.issue_repo,
                    args.issue_labels,
                    args.issue_assignees,
                    existing_markers,
                    project_id,
                )

        if args.dry_run:
            print(
                f"[dry-run] Would create {planned} issues; skipped {skipped_existing} existing issues."
            )
        else:
            print(
                f"Issue creation summary: created {created} new issues; skipped {skipped_existing} existing issues."
            )


if __name__ == "__main__":
    main()
