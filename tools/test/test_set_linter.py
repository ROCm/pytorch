# mypy: ignore-errors
from __future__ import annotations

import sys
from pathlib import Path
from token import NAME
from tokenize import TokenInfo

<<<<<<< HEAD
from tools.linter.adapters._linter import PythonFile
from tools.linter.adapters.set_linter import PythonLines, SetLinter
=======
from tools.linter.adapters.set_linter import SetLinter
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))


_PARENT = Path(__file__).parent.absolute()
_PATH = [Path(p).absolute() for p in sys.path]

if _PARENT in _PATH:
    from linter_test_case import LinterTestCase
else:
    from .linter_test_case import LinterTestCase


TESTDATA = Path("tools/test/set_linter_testdata")

TESTFILE = TESTDATA / "python_code.py.txt"
INCLUDES_FILE = TESTDATA / "includes.py.txt"
INCLUDES_FILE2 = TESTDATA / "includes_doesnt_change.py.txt"
FILES = TESTFILE, INCLUDES_FILE, INCLUDES_FILE2


<<<<<<< HEAD
def python_lines(p: str | Path) -> PythonLines:
    pf = PythonFile.make(SetLinter.linter_name, p)
    return PythonLines(pf)


=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
class TestSetLinter(LinterTestCase):
    maxDiff = 10000000
    LinterClass = SetLinter

    def test_get_all_tokens(self) -> None:
<<<<<<< HEAD
        self.assertEqual(EXPECTED_SETS, python_lines(TESTFILE).sets)

    def test_omitted_lines(self) -> None:
        actual = sorted(python_lines(TESTFILE).omitted.omitted)
        expected = [3, 13]
=======
        self.assertEqual(EXPECTED_SETS, SetLinter.make_file(TESTFILE).sets)

    def test_omitted_lines(self) -> None:
        actual = sorted(SetLinter.make_file(TESTFILE).omitted.omitted)
        expected = [6, 16]
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
        self.assertEqual(expected, actual)

    def test_linting(self) -> None:
        for path in (TESTFILE, INCLUDES_FILE, INCLUDES_FILE2):
            with self.subTest(path):
                r = self.lint_fix_test(path, [])
                self.assertEqual(r.name, "Suggested fixes for set_linter")

    def test_bracket_pairs(self) -> None:
        TESTS: tuple[tuple[str, dict[int, int]], ...] = (
            ("", {}),
            ("{}", {0: 1}),
            ("{1}", {0: 2}),
            ("{1, 2}", {0: 4}),
            ("{1: 2}", {0: 4}),
            ("{One()}", {0: 4, 2: 3}),
            (
                "{One({1: [2], 2: {3}, 3: {4: 5}})}",
                {0: 25, 2: 24, 3: 23, 6: 8, 12: 14, 18: 22},
            ),
<<<<<<< HEAD
        )
        for s, expected in TESTS:
            pl = python_lines(s)
            if s:
                actual = pl.token_lines[0].bracket_pairs
            else:
                self.assertEqual(pl.token_lines, [])
=======
            ("f'{a}'", {}),
        )
        for s, expected in TESTS:
            pf = SetLinter.make_file(s)
            if s:
                actual = pf._lines_with_sets[0].bracket_pairs
            else:
                self.assertEqual(pf._lines_with_sets, [])
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
                actual = {}
            self.assertEqual(actual, expected)

    def test_match_braced_sets(self) -> None:
        TESTS: tuple[tuple[str, int], ...] = (
            ("{cast(int, inst.offset): inst for inst in instructions}", 0),
            ("", 0),
            ("{}", 0),
            ("{1: 0}", 0),
            ("{1}", 1),
            ("{i for i in range(2, 3)}", 1),
            ("{1, 2}", 1),
            ("{One({'a': 1}), Two([{}, {2}, {1, 2}])}", 3),
        )
        for s, expected in TESTS:
<<<<<<< HEAD
            pl = python_lines(s)
            actual = pl.token_lines and pl.token_lines[0].braced_sets
=======
            pf = SetLinter.make_file(s)
            actual = pf._lines_with_sets and pf._lines_with_sets[0].braced_sets
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
            self.assertEqual(len(actual), expected)


EXPECTED_SETS = [
<<<<<<< HEAD
    TokenInfo(NAME, "set", (4, 4), (4, 7), "a = set()\n"),
    TokenInfo(NAME, "set", (6, 4), (6, 7), "c = set\n"),
    TokenInfo(NAME, "set", (9, 3), (9, 6), "   set(\n"),
=======
    TokenInfo(NAME, "set", (7, 4), (7, 7), "a = set()\n"),
    TokenInfo(NAME, "set", (9, 4), (9, 7), "c = set\n"),
    TokenInfo(NAME, "set", (12, 3), (12, 6), "   set(\n"),
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
]
