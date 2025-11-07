from __future__ import annotations

<<<<<<< HEAD
import sys
import token
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING


_PARENT = Path(__file__).parent.absolute()
_PATH = [Path(p).absolute() for p in sys.path]

if TYPE_CHECKING or _PARENT not in _PATH:
=======
import itertools
import json
import sys
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING


_FILE = Path(__file__).absolute()
_PATH = [Path(p).absolute() for p in sys.path]

if TYPE_CHECKING or _FILE.parent not in _PATH:
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    from . import _linter
else:
    import _linter

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
<<<<<<< HEAD
    from tokenize import TokenInfo


MAX_LINES = {"class": 100, "def": 80}

MIN_DOCSTRING = 16  # docstrings shorter than this are ignored
IGNORE_PROTECTED = True  # If True, ignore classes and files whose names start with _.

ERROR_FMT = "Every {type} with more than {length} lines needs a docstring"

DESCRIPTION = """`docstring_linter` reports on long functions, methods or classes
without docstrings"""

# How many top violations to report?
REPORT_TOP_RESULTS = 3


def _is_def(t: TokenInfo) -> bool:
    return t.type == token.NAME and t.string in ("class", "def")
=======


GRANDFATHER_LIST = _FILE.parent / "docstring_linter-grandfather.json"

# We tolerate a 10% increase in block size before demanding a docstring
TOLERANCE_PERCENT = 10

MAX_LINES = {"class": 100, "def": 80}

MIN_DOCSTRING = 50  # docstrings shorter than this are too short

DESCRIPTION = """
`docstring_linter` reports on long functions, methods or classes without docstrings
""".strip()

METHOD_OVERRIDE_HINT = (
    "If the method overrides a method on a parent class, adding the"
    " `@typing_extensions.override` decorator will make this error"
    " go away."
)
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))


class DocstringLinter(_linter.FileLinter):
    linter_name = "docstring_linter"
    description = DESCRIPTION
    is_fixer = False
<<<<<<< HEAD
    results: dict[str, list[tuple[int, Path, str]]]

    def __init__(self, argv: list[str] | None = None) -> None:
        super().__init__(argv)
        self.results = {}

        help = "Maximum number of lines for an undocumented class"
        self.parser.add_argument(
            "--max-class", "-c", default=MAX_LINES["class"], type=int, help=help
        )

        help = "Maximum number of lines for an undocumented function"
        self.parser.add_argument(
            "--max-def", "-d", default=MAX_LINES["def"], type=int, help=help
        )

        help = "Minimum number of characters for a docstring"
        self.parser.add_argument(
            "--min-docstring", "-m", default=MIN_DOCSTRING, type=int, help=help
        )

        help = "Lint functions, methods and classes that start with _"
        self.parser.add_argument(
            "--lint-protected", "-p", action="store_true", help=help
        )

    @cached_property
    def max_lines(self) -> dict[str, int]:
        return {"class": self.args.max_class, "def": self.args.max_def}

    def lint_all(self) -> bool:
        success = super().lint_all()
        if not self.args.lintrunner and self.results:
            self._report_results()
        return success

    def _lint(self, pf: _linter.PythonFile) -> Iterator[_linter.LintResult]:
        tokens = pf.tokens
        indents = indent_to_dedent(tokens)
        defs = [i for i, t in enumerate(tokens) if _is_def(t)]

        def next_token(start: int, token_type: int, error: str) -> int:  # type: ignore[return]
            for i in range(start, len(tokens)):
                if tokens[i].type == token_type:
                    return i
            _linter.ParseError.check(False, tokens[-1], error)

        for i in defs:
            name = next_token(i + 1, token.NAME, "Definition with no name")
            if not self.args.lint_protected and tokens[name].string.startswith("_"):
                continue

            indent = next_token(name + 1, token.INDENT, "Definition with no indent")
            dedent = indents[indent]

            lines = tokens[dedent].start[0] - tokens[indent].start[0]
            max_lines = self.max_lines[tokens[i].string]
            if lines <= max_lines:
                continue

            # Now search for a docstring
            docstring_len = -1
            for k in range(indent + 1, len(tokens)):
                tk = tokens[k]
                if tk.type == token.STRING:
                    docstring_len = len(tk.string)
                    break
                if tk.type not in _linter.EMPTY_TOKENS:
                    break

            if docstring_len >= self.args.min_docstring:
                continue

            # Now check if it's omitted
            if pf.omitted(pf.tokens[i:indent]):
                continue

            t = tokens[i]
            def_name = "function" if t.string == "def" else t.string
            tname = tokens[name].string
            msg = f"docstring found for {def_name} '{tname}' ({lines} lines)"
            if docstring_len < 0:
                msg = "No " + msg
            else:
                msg = msg + f" was too short ({docstring_len} characters)"
            yield _linter.LintResult(msg, *t.start)
            if pf.path is not None:
                self.results.setdefault(def_name, []).append((lines, pf.path, tname))

    def _report_results(self) -> None:
        print()
        for i, (k, v) in enumerate(sorted(self.results.items())):
            if i:
                print()
            top = sorted(v, reverse=True)[:REPORT_TOP_RESULTS]
            if len(top) == 1:
                s = ""
                t = f"{len(top)} "
            else:
                s = "es" if k.endswith("s") else "s"
                t = ""
            print(f"Top {t}undocumented {k}{s}:")
            for lines, path, tname in top:
                print(f"    {lines} lines: {path}:{tname}")


def indent_to_dedent(tokens: Sequence[TokenInfo]) -> dict[int, int]:
    indent_to_dedent: dict[int, int] = {}
    stack: list[int] = []

    for i, t in enumerate(tokens):
        if t.type == token.INDENT:
            stack.append(i)
        elif t.type == token.DEDENT:
            assert stack
            indent_to_dedent[stack.pop()] = i

    assert not stack
    # Can't happen: the tokenization process would already have failed on a bad indent

    return indent_to_dedent
=======

    path_to_blocks: dict[str, list[dict[str, Any]]]
    path_to_errors: dict[str, list[dict[str, Any]]]

    def __init__(self, argv: Sequence[str] | None = None) -> None:
        super().__init__(argv)
        add_arguments(self.parser.add_argument)
        self.path_to_blocks = {}
        self.path_to_errors = {}

    def lint_all(self) -> bool:
        success = super().lint_all()
        self._report()
        self._write_grandfather()
        return success

    def _lint(self, pf: _linter.PythonFile) -> Iterator[_linter.LintResult]:
        if (p := str(pf.path)) in self.path_to_blocks:
            print("Repeated file", p, file=sys.stderr)
            return

        blocks = pf.blocks
        bad = {b for b in blocks if self._is_bad_block(b, pf)}
        bad = self._dont_require_constructor_and_class_docs(blocks, bad)
        gf = self._grandfathered(pf.path, bad)

        yield from (self._block_result(b, pf) for b in sorted(bad - gf))

        def as_data(b: _linter.Block) -> dict[str, Any]:
            status = "grandfather" if b in gf else "bad" if b in bad else "good"
            return {"status": status, **b.as_data()}

        self.path_to_blocks[p] = [as_data(b) for b in blocks]

    def _error(self, pf: _linter.PythonFile, result: _linter.LintResult) -> None:
        self.path_to_errors[str(pf.path)] = [{str(result.line): result.name}]

    @cached_property
    def _grandfather(self) -> dict[str, dict[str, Any]]:
        try:
            with open(self.args.grandfather) as fp:
                return json.load(fp)  # type: ignore[no-any-return]
        except FileNotFoundError:
            return {}
        except Exception as e:
            print("ERROR:", e, "in", GRANDFATHER_LIST, file=sys.stderr)
            raise

    @cached_property
    def _max_lines(self) -> dict[str, int]:
        return {"class": self.args.max_class, "def": self.args.max_def}

    def _grandfathered(
        self, path: Path | None, bad: set[_linter.Block]
    ) -> set[_linter.Block]:
        if path is None or self.args.no_grandfather or self.args.write_grandfather:
            return set()

        grand: dict[str, int] = self._grandfather.get(str(path), {})
        tolerance_ratio = 1 + self.args.grandfather_tolerance / 100.0

        def grandfathered(b: _linter.Block) -> bool:
            lines = int(grand.get(b.display_name, 0) * tolerance_ratio)
            return b.line_count <= lines

        return {b for b in bad if grandfathered(b)}

    def _block_result(
        self, b: _linter.Block, pf: _linter.PythonFile
    ) -> _linter.LintResult:
        def_name = "function" if b.category == "def" else "class"
        msg = f"docstring found for {def_name} '{b.name}' ({b.line_count} lines)"
        if len(b.docstring):
            s = "" if len(b.docstring) == 1 else "s"
            needed = f"needed {self.args.min_docstring}"
            msg = f"{msg} was too short ({len(b.docstring)} character{s}, {needed})"
        else:
            msg = f"No {msg}"
            if b.is_method:
                msg = f"{msg}. {METHOD_OVERRIDE_HINT}"
        return _linter.LintResult(msg, *pf.tokens[b.begin].start)

    def _display(
        self, pf: _linter.PythonFile, results: list[_linter.LintResult]
    ) -> Iterator[str]:
        if not self.args.report:
            yield from super()._display(pf, results)

    def _dont_require_constructor_and_class_docs(
        self, blocks: Sequence[_linter.Block], bad: set[_linter.Block]
    ) -> set[_linter.Block]:
        if self.args.lint_init:
            return bad

        good = {b for b in blocks if len(b.docstring) >= self.args.min_docstring}

        def has_class_init_doc(b: _linter.Block) -> bool:
            if b.is_class:
                # Is it a class whose constructor is documented?
                children = (blocks[i] for i in b.children)
                return any(b.is_init and b in good for b in children)

            # Is it a constructor whose class is documented?
            return b.is_init and b.parent is not None and blocks[b.parent] in good

        return {b for b in bad if not has_class_init_doc(b)}

    def _is_bad_block(self, b: _linter.Block, pf: _linter.PythonFile) -> bool:
        max_lines = self._max_lines[b.category]
        return (
            not pf.omitted(pf.tokens, b.begin, b.dedent)
            and b.line_count > max_lines
            and len(b.docstring) < self.args.min_docstring
            and (self.args.lint_local or not b.is_local)
            and (self.args.lint_protected or not b.name.startswith("_"))
        )

    def _report(self) -> None:
        if not self.args.lintrunner and self.path_to_blocks and self.args.report:
            report = {
                k: s for k, v in self.path_to_blocks.items() if (s := file_summary(v))
            } | self.path_to_errors
            print(json.dumps(report, sort_keys=True, indent=2))

    def _write_grandfather(self) -> None:
        if self.args.write_grandfather:
            results: dict[str, dict[str, int]] = {}

            for path, blocks in self.path_to_blocks.items():
                for block in blocks:
                    if block["status"] == "bad":
                        d = results.setdefault(path, {})
                        d[block["display_name"]] = block["line_count"]

            with open(self.args.grandfather, "w") as fp:
                json.dump(results, fp, sort_keys=True, indent=2)


def make_recursive(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def rec(i: int) -> dict[str, Any]:
        d = dict(blocks[i])
        d["children"] = [rec(c) for c in d["children"]]
        return d

    return [rec(i) for i, b in enumerate(blocks) if b["parent"] is None]


def make_terse(
    blocks: Sequence[dict[str, Any]],
    index_by_line: bool = True,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    max_line = max(b["start_line"] for b in blocks) if blocks else 0
    line_field_width = len(str(max_line))

    for b in blocks:
        root = f"{b['category']} {b['full_name']}"
        for i in itertools.count():
            name = root + bool(i) * f"[{i + 1}]"
            if name not in result:
                break

        d = {
            "docstring_len": len(b["docstring"]),
            "lines": b["line_count"],
            "status": b.get("status", "good"),
        }

        start_line = b["start_line"]
        if index_by_line:
            d["name"] = name
            result[f"{start_line:>{line_field_width}}"] = d
        else:
            d["line"] = start_line
            result[name] = d

        if kids := b["children"]:
            if not all(isinstance(k, int) for k in kids):
                assert all(isinstance(k, dict) for k in kids)
                d["children"] = make_terse(kids)

    return result


def file_summary(
    blocks: Sequence[dict[str, Any]], report_all: bool = False
) -> dict[str, str]:
    def to_line(v: dict[str, Any]) -> str | None:
        if (status := v["status"]) == "good":
            if not report_all:
                return None
            fail = ""
        elif status == "grandfather":
            fail = ": (grandfathered)"
        else:
            assert status == "bad"
            fail = ": FAIL"
        name = v["name"]
        lines = v["lines"]
        docs = v["docstring_len"]
        parens = "()" if name.startswith("def ") else ""
        return f"{name}{parens}: {lines=}, {docs=}{fail}"

    t = make_terse(blocks)
    r = {k: line for k, v in t.items() if (line := to_line(v))}
    while r and all(k.startswith(" ") for k in r):
        r = {k[1:]: v for k, v in r.items()}
    return r


def add_arguments(add: Callable[..., Any]) -> None:
    h = "Set the grandfather list"
    add("--grandfather", "-g", default=str(GRANDFATHER_LIST), type=str, help=h)

    h = "Tolerance for grandfather sizes, in percent"
    add("--grandfather-tolerance", "-t", default=TOLERANCE_PERCENT, type=float, help=h)

    h = "Lint __init__ and class separately"
    add("--lint-init", "-i", action="store_true", help=h)

    h = "Lint definitions inside other functions"
    add("--lint-local", "-o", action="store_true", help=h)

    h = "Lint functions, methods and classes that start with _"
    add("--lint-protected", "-p", action="store_true", help=h)

    h = "Maximum number of lines for an undocumented class"
    add("--max-class", "-c", default=MAX_LINES["class"], type=int, help=h)

    h = "Maximum number of lines for an undocumented function"
    add("--max-def", "-d", default=MAX_LINES["def"], type=int, help=h)

    h = "Minimum number of characters for a docstring"
    add("--min-docstring", "-s", default=MIN_DOCSTRING, type=int, help=h)

    h = "Disable the grandfather list"
    add("--no-grandfather", "-n", action="store_true", help=h)

    h = "Print a report on all classes and defs"
    add("--report", "-r", action="store_true", help=h)

    h = "Rewrite the grandfather list"
    add("--write-grandfather", "-w", action="store_true", help=h)
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))


if __name__ == "__main__":
    DocstringLinter.run()
