"""
Generic linter that greps for a pattern and optionally suggests replacements.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from enum import Enum
from typing import NamedTuple


IS_WINDOWS: bool = os.name == "nt"
<<<<<<< HEAD
MAX_FILE_SIZE: int = 1024 * 1024 * 1024  # 1GB in bytes
MAX_MATCHES_PER_FILE: int = 100  # Maximum number of matches to report per file
MAX_ORIGINAL_SIZE: int = (
    512 * 1024
)  # 512KB - don't compute replacement if original is larger
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


<<<<<<< HEAD
LINTER_NAME: str = ""
ERROR_DESCRIPTION: str | None = None


=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
class LintMessage(NamedTuple):
    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None


def as_posix(name: str) -> str:
    return name.replace("\\", "/") if IS_WINDOWS else name


def run_command(
    args: list[str],
) -> subprocess.CompletedProcess[bytes]:
    logging.debug("$ %s", " ".join(args))
    start_time = time.monotonic()
    try:
        return subprocess.run(
            args,
            capture_output=True,
        )
    finally:
        end_time = time.monotonic()
        logging.debug("took %dms", (end_time - start_time) * 1000)


<<<<<<< HEAD
def print_lint_message(
    name: str,
    severity: LintSeverity = LintSeverity.ERROR,
    path: str | None = None,
    line: int | None = None,
    original: str | None = None,
    replacement: str | None = None,
    description: str | None = None,
) -> None:
    """
    Create a LintMessage and print it as JSON.

    Accepts the same arguments as LintMessage constructor.
    """
    char = None
    code = LINTER_NAME
    description = description or ERROR_DESCRIPTION
    lint_message = LintMessage(
        path, line, char, code, severity, name, original, replacement, description
    )
    print(json.dumps(lint_message._asdict()), flush=True)


def group_lines_by_file(lines: list[str]) -> dict[str, list[str]]:
    """
    Group matching lines by filename.

    Args:
        lines: List of grep output lines in format "filename:line:content"

    Returns:
        Dictionary mapping filename to list of line remainders (without filename prefix)
    """
    grouped: dict[str, list[str]] = {}
    for line in lines:
        if not line:
            continue
        # Extract filename and remainder from "filename:line:content" format
        parts = line.split(":", 1)
        filename = parts[0]
        remainder = parts[1] if len(parts) > 1 else ""
        if filename not in grouped:
            grouped[filename] = []
        grouped[filename].append(remainder)
    return grouped


def check_allowlist(
    filename: str,
    allowlist_pattern: str,
) -> bool:
    """
    Check if a file matches the allowlist pattern.

    Args:
        filename: Path to the file to check
        allowlist_pattern: Pattern to grep for in the file

    Returns:
        True if the file should be skipped (allowlist pattern matched), False otherwise.
        Prints error message and returns False if there was an error running grep.
    """
    if not allowlist_pattern:
        return False

    try:
        proc = run_command(["grep", "-nEHI", allowlist_pattern, filename])
    except Exception as err:
        print_lint_message(
            name="command-failed",
            description=(
                f"Failed due to {err.__class__.__name__}:\n{err}"
                if not isinstance(err, subprocess.CalledProcessError)
                else (
                    "COMMAND (exit code {returncode})\n"
                    "{command}\n\n"
                    "STDERR\n{stderr}\n\n"
                    "STDOUT\n{stdout}"
                ).format(
                    returncode=err.returncode,
                    command=" ".join(as_posix(x) for x in err.cmd),
                    stderr=err.stderr.decode("utf-8").strip() or "(empty)",
                    stdout=err.stdout.decode("utf-8").strip() or "(empty)",
                )
            ),
        )
        return False

    # allowlist pattern was found, abort lint
    if proc.returncode == 0:
        return True

    return False


def lint_file(
    filename: str,
    line_remainders: list[str],
    allowlist_pattern: str,
    replace_pattern: str,
    error_name: str,
) -> None:
    """
    Lint a file with one or more pattern matches, printing LintMessages as they're created.

    Args:
        filename: Path to the file being linted
        line_remainders: List of line remainders (format: "line:content" without filename prefix)
        allowlist_pattern: Pattern to check for allowlisting
        replace_pattern: Pattern for sed replacement
        error_name: Human-readable error name
    """
    if not line_remainders:
        return

    should_skip = check_allowlist(filename, allowlist_pattern)
    if should_skip:
        return

    # Check if file is too large to compute replacement
    file_size = os.path.getsize(filename)
    compute_replacement = replace_pattern and file_size <= MAX_ORIGINAL_SIZE

    # Apply replacement to entire file if pattern is specified and file is not too large
    original = None
    replacement = None
    if compute_replacement:
        # When we have a replacement, report a single message with line=None
        try:
            with open(filename) as f:
                original = f.read()

            proc = run_command(["sed", "-r", replace_pattern, filename])
            replacement = proc.stdout.decode("utf-8")
        except Exception as err:
            print_lint_message(
                name="command-failed",
=======
def lint_file(
    matching_line: str,
    allowlist_pattern: str,
    replace_pattern: str,
    linter_name: str,
    error_name: str,
    error_description: str,
) -> LintMessage | None:
    # matching_line looks like:
    #   tools/linter/clangtidy_linter.py:13:import foo.bar.baz
    split = matching_line.split(":")
    filename = split[0]

    if allowlist_pattern:
        try:
            proc = run_command(["grep", "-nEHI", allowlist_pattern, filename])
        except Exception as err:
            return LintMessage(
                path=None,
                line=None,
                char=None,
                code=linter_name,
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
                description=(
                    f"Failed due to {err.__class__.__name__}:\n{err}"
                    if not isinstance(err, subprocess.CalledProcessError)
                    else (
                        "COMMAND (exit code {returncode})\n"
                        "{command}\n\n"
                        "STDERR\n{stderr}\n\n"
                        "STDOUT\n{stdout}"
                    ).format(
                        returncode=err.returncode,
                        command=" ".join(as_posix(x) for x in err.cmd),
                        stderr=err.stderr.decode("utf-8").strip() or "(empty)",
                        stdout=err.stdout.decode("utf-8").strip() or "(empty)",
                    )
                ),
            )
<<<<<<< HEAD
            return

        print_lint_message(
            path=filename,
            name=error_name,
            original=original,
            replacement=replacement,
        )
    else:
        # When no replacement, report each matching line (up to MAX_MATCHES_PER_FILE)
        total_matches = len(line_remainders)
        matches_to_report = min(total_matches, MAX_MATCHES_PER_FILE)

        for line_remainder in line_remainders[:matches_to_report]:
            # line_remainder format: "line_number:content"
            split = line_remainder.split(":", 1)
            line_number = int(split[0]) if split[0] else None
            print_lint_message(
                path=filename,
                line=line_number,
                name=error_name,
            )

        # If there are more matches than the limit, print an error
        if total_matches > MAX_MATCHES_PER_FILE:
            print_lint_message(
                path=filename,
                name="too-many-matches",
                description=f"File has {total_matches} matches, only showing first {MAX_MATCHES_PER_FILE}",
            )
=======

        # allowlist pattern was found, abort lint
        if proc.returncode == 0:
            return None

    original = None
    replacement = None
    if replace_pattern:
        with open(filename) as f:
            original = f.read()

        try:
            proc = run_command(["sed", "-r", replace_pattern, filename])
            replacement = proc.stdout.decode("utf-8")
        except Exception as err:
            return LintMessage(
                path=None,
                line=None,
                char=None,
                code=linter_name,
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=(
                    f"Failed due to {err.__class__.__name__}:\n{err}"
                    if not isinstance(err, subprocess.CalledProcessError)
                    else (
                        "COMMAND (exit code {returncode})\n"
                        "{command}\n\n"
                        "STDERR\n{stderr}\n\n"
                        "STDOUT\n{stdout}"
                    ).format(
                        returncode=err.returncode,
                        command=" ".join(as_posix(x) for x in err.cmd),
                        stderr=err.stderr.decode("utf-8").strip() or "(empty)",
                        stdout=err.stdout.decode("utf-8").strip() or "(empty)",
                    )
                ),
            )

    return LintMessage(
        path=split[0],
        line=int(split[1]) if len(split) > 1 else None,
        char=None,
        code=linter_name,
        severity=LintSeverity.ERROR,
        name=error_name,
        original=original,
        replacement=replacement,
        description=error_description,
    )
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="grep wrapper linter.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--pattern",
        required=True,
        help="pattern to grep for",
    )
    parser.add_argument(
        "--allowlist-pattern",
        help="if this pattern is true in the file, we don't grep for pattern",
    )
    parser.add_argument(
        "--linter-name",
        required=True,
        help="name of the linter",
    )
    parser.add_argument(
        "--match-first-only",
        action="store_true",
        help="only match the first hit in the file",
    )
    parser.add_argument(
        "--error-name",
        required=True,
        help="human-readable description of what the error is",
    )
    parser.add_argument(
        "--error-description",
        required=True,
        help="message to display when the pattern is found",
    )
    parser.add_argument(
        "--replace-pattern",
        help=(
            "the form of a pattern passed to `sed -r`. "
            "If specified, this will become proposed replacement text."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
<<<<<<< HEAD

    # Check for duplicate arguments before parsing
    seen_args = set()
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            arg_name = arg.split("=")[0]
            if arg_name in seen_args:
                parser.error(
                    f"argument {arg_name}: not allowed to be specified multiple times"
                )
            seen_args.add(arg_name)

    args = parser.parse_args()

    global LINTER_NAME, ERROR_DESCRIPTION
    LINTER_NAME = args.linter_name
    ERROR_DESCRIPTION = args.error_description

=======
    args = parser.parse_args()

>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET
        if args.verbose
        else logging.DEBUG
        if len(args.filenames) < 1000
        else logging.INFO,
        stream=sys.stderr,
    )

<<<<<<< HEAD
    # Filter out files that are too large before running grep
    filtered_filenames = []
    for filename in args.filenames:
        try:
            file_size = os.path.getsize(filename)
            if file_size > MAX_FILE_SIZE:
                print_lint_message(
                    path=filename,
                    severity=LintSeverity.WARNING,
                    name="file-too-large",
                    description=f"File size ({file_size} bytes) exceeds {MAX_FILE_SIZE} bytes limit, skipping",
                )
            else:
                filtered_filenames.append(filename)
        except OSError as err:
            print_lint_message(
                path=filename,
                name="file-access-error",
                description=f"Failed to get file size: {err}",
            )

    # If all files were filtered out, nothing to do
    if not filtered_filenames:
        return

=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    files_with_matches = []
    if args.match_first_only:
        files_with_matches = ["--files-with-matches"]

    lines = []
    try:
        # Split the grep command into multiple batches to avoid hitting the
        # command line length limit of ~1M on my machine
<<<<<<< HEAD
        arg_length = sum(len(x) for x in filtered_filenames)
        batches = arg_length // 750000 + 1
        batch_size = len(filtered_filenames) // batches
        for i in range(0, len(filtered_filenames), batch_size):
=======
        arg_length = sum(len(x) for x in args.filenames)
        batches = arg_length // 750000 + 1
        batch_size = len(args.filenames) // batches
        for i in range(0, len(args.filenames), batch_size):
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
            proc = run_command(
                [
                    "grep",
                    "-nEHI",
                    *files_with_matches,
                    args.pattern,
<<<<<<< HEAD
                    *filtered_filenames[i : i + batch_size],
=======
                    *args.filenames[i : i + batch_size],
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
                ]
            )
            lines.extend(proc.stdout.decode().splitlines())
    except Exception as err:
<<<<<<< HEAD
        print_lint_message(
            name="command-failed",
=======
        err_msg = LintMessage(
            path=None,
            line=None,
            char=None,
            code=args.linter_name,
            severity=LintSeverity.ERROR,
            name="command-failed",
            original=None,
            replacement=None,
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
            description=(
                f"Failed due to {err.__class__.__name__}:\n{err}"
                if not isinstance(err, subprocess.CalledProcessError)
                else (
                    "COMMAND (exit code {returncode})\n"
                    "{command}\n\n"
                    "STDERR\n{stderr}\n\n"
                    "STDOUT\n{stdout}"
                ).format(
                    returncode=err.returncode,
                    command=" ".join(as_posix(x) for x in err.cmd),
                    stderr=err.stderr.decode("utf-8").strip() or "(empty)",
                    stdout=err.stdout.decode("utf-8").strip() or "(empty)",
                )
            ),
        )
<<<<<<< HEAD
        sys.exit(0)

    # Group lines by file to call lint_file once per file
    grouped_lines = group_lines_by_file(lines)

    for filename, line_remainders in grouped_lines.items():
        lint_file(
            filename,
            line_remainders,
            args.allowlist_pattern,
            args.replace_pattern,
            args.error_name,
        )
=======
        print(json.dumps(err_msg._asdict()), flush=True)
        sys.exit(0)

    for line in lines:
        lint_message = lint_file(
            line,
            args.allowlist_pattern,
            args.replace_pattern,
            args.linter_name,
            args.error_name,
            args.error_description,
        )
        if lint_message is not None:
            print(json.dumps(lint_message._asdict()), flush=True)
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))


if __name__ == "__main__":
    main()
