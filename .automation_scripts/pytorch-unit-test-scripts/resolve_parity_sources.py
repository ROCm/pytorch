#!/usr/bin/env python3
"""Resolve a commit's physical parity sources from GitHub API inventories."""

import argparse
import json
import sys

from parity_source_discovery import AmbiguousSourceError, resolve_topology


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-runs", required=True)
    parser.add_argument("--workflow-runs", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--arch", action="append", dest="archs")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.check_runs) as check_file:
        checks = json.load(check_file)
    with open(args.workflow_runs) as run_file:
        runs = json.load(run_file)
    with open(args.config) as config_file:
        config = json.load(config_file)
    try:
        topology = resolve_topology(
            checks, runs, config, args.archs, sha=args.sha
        )
    except AmbiguousSourceError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    json.dump(topology, sys.stdout, sort_keys=True, separators=(",", ":"))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
