#!/usr/bin/env python3

import argparse
import csv
import os
import sys


TEST_CONFIGS = ['default', 'distributed', 'inductor']
TEST_CONFIG_DISPLAY = {
    'default': 'TEST DEFAULT',
    'distributed': 'TEST DISTRIBUTED',
    'inductor': 'TEST INDUCTOR',
}
MAX_DIAGNOSTIC_FIELD_CHARS = 20_000
DIAGNOSTIC_FIELDS = {
    'comments',
    'message_cuda',
    'message_rocm',
    'message_set1',
    'message_set2',
    'reason',
    'skip_reason',
}


def _configure_csv_field_limit():
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


_configure_csv_field_limit()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate a parity summary from per-architecture test status CSVs'
    )
    parser.add_argument(
        '--csv', nargs='+', required=True,
        help='CSV file(s) to summarize (one per architecture, same order as --arch)'
    )
    parser.add_argument(
        '--arch', nargs='+', required=True,
        help='Architecture labels matching --csv order (e.g. mi200 mi300 mi355)'
    )
    parser.add_argument('--sha', type=str, default='', help='Commit SHA')
    parser.add_argument('--pr_id', type=str, default='', help='Pull request ID')
    parser.add_argument(
        '--set1_name', type=str, default='set1',
        help='Name used for set1 in CSV column headers (default: set1)'
    )
    parser.add_argument(
        '--set2_name', type=str, default='set2',
        help='Name used for set2 in CSV column headers (default: set2)'
    )
    parser.add_argument(
        '--output', type=str, default='parity_summary',
        help='Output path prefix (produces .csv and .md)'
    )
    parser.add_argument(
        '--log-failures', nargs='*', default=[],
        help='CSV file(s) from detect_log_failures.py to include in summary'
    )
    return parser.parse_args()


def load_csv(filepath):
    with open(filepath, newline='') as f:
        return [_truncate_diagnostic_fields(row) for row in csv.DictReader(f)]


def _truncate_diagnostic_fields(row):
    for field in DIAGNOSTIC_FIELDS:
        value = row.get(field, '')
        if len(value) > MAX_DIAGNOSTIC_FIELD_CHARS:
            omitted = len(value) - MAX_DIAGNOSTIC_FIELD_CHARS
            row[field] = (
                value[:MAX_DIAGNOSTIC_FIELD_CHARS]
                + f'\n...[truncated {omitted:,} chars by generate_summary.py]'
            )
    return row


def detect_columns(headers, set1_name, set2_name):
    s1_status = f'status_{set1_name}'
    s2_status = f'status_{set2_name}'
    s1_time = f'running_time_{set1_name}'
    s2_time = f'running_time_{set2_name}'
    if s1_status not in headers:
        s1_status = 'status_set1'
        s2_status = 'status_set2'
        s1_time = 'running_time_set1'
        s2_time = 'running_time_set2'
    return s1_status, s2_status, s1_time, s2_time


def test_config_stats_keys(s1_name, s2_name, has_set2=True):
    s1 = s1_name.upper()
    s2 = s2_name.upper()
    if not has_set2:
        return [
            f'PASSED ({s1_name})',
            f'SKIPPED ({s1_name})',
            f'FAILED ({s1_name})',
            f'MISSED ({s1_name})',
            f'TOTAL {s1}',
        ]
    return [
        f'SKIPPED (on {s1_name}, but not on {s2_name})',
        f'SKIPPED (on {s1_name})',
        f'SKIPPED (on {s2_name})',
        f'MISSED (MISSED on {s1_name}, NOT SKIPPED on {s2_name})',
        f'{s1}ONLY (PASSED on {s1}, NOT PASSED on {s2})',
        s2,
        s1,
        'SKIPPED + MISSED',
        f'{s2} - (SKIPPED + MISSED)',
        f'DISAGREE [(SKIPPED+MISSED)/{s2}] %',
    ]


def compute_test_config_stats(rows, s1_col, s2_col, s1_name, s2_name, has_set2=True):
    s1 = s1_name.upper()
    s2 = s2_name.upper()

    if not has_set2:
        vals = {}
        keys = test_config_stats_keys(s1_name, s2_name, has_set2=False)
        vals[keys[0]] = sum(1 for r in rows if r[s1_col] == 'PASSED')
        vals[keys[1]] = sum(1 for r in rows if r[s1_col] == 'SKIPPED')
        vals[keys[2]] = sum(1 for r in rows if r[s1_col] == 'FAILED')
        vals[keys[3]] = sum(1 for r in rows if r[s1_col] == 'MISSED')
        vals[keys[4]] = sum(1 for r in rows if r[s1_col].strip())
        return vals

    s1_skip_not_s2 = sum(
        1 for r in rows
        if r[s1_col] == 'SKIPPED' and r[s2_col] != 'SKIPPED'
    )
    s1_skip = sum(1 for r in rows if r[s1_col] == 'SKIPPED')
    s2_skip = sum(1 for r in rows if r[s2_col] == 'SKIPPED')
    s1_miss_not_s2_skip = sum(
        1 for r in rows
        if r[s1_col] == 'MISSED' and r[s2_col] != 'SKIPPED'
    )
    only_s1 = sum(
        1 for r in rows
        if r[s1_col] == 'PASSED' and r[s2_col] != 'PASSED'
    )
    total_s2 = sum(1 for r in rows if r[s2_col].strip() and r[s2_col].strip() != 'MISSED')
    total_s1 = sum(1 for r in rows if r[s1_col].strip() and r[s1_col].strip() != 'MISSED')

    skip_miss = s1_skip_not_s2 + s1_miss_not_s2_skip
    s2_minus = total_s2 - skip_miss
    pct = (skip_miss / total_s2 * 100) if total_s2 else 0

    vals = {}
    keys = test_config_stats_keys(s1_name, s2_name)
    vals[keys[0]] = s1_skip_not_s2
    vals[keys[1]] = s1_skip
    vals[keys[2]] = s2_skip
    vals[keys[3]] = s1_miss_not_s2_skip
    vals[keys[4]] = only_s1
    vals[keys[5]] = total_s2
    vals[keys[6]] = total_s1
    vals[keys[7]] = skip_miss
    vals[keys[8]] = s2_minus
    vals[keys[9]] = f'{pct:.2f}%'
    return vals


def overall_stats_keys(s1_name, s2_name, has_set2=True):
    s1 = s1_name.upper()
    s2 = s2_name.upper()
    if not has_set2:
        keys = []
        for status in ['PASSED', 'SKIPPED', 'FAILED', 'XFAILED']:
            keys.append(f'{status}({s1_name})')
        keys += [
            f'TOTAL {s1}',
            f'TOTAL {s1} RUNNING TIME',
        ]
        return keys
    keys = [
        'Overall DISAGREE%',
        'Overall AGREE%',
    ]
    for status in ['PASSED', 'SKIPPED', 'FAILED', 'XFAILED']:
        keys.append(f'{status}({s1_name})')
        keys.append(f'{status}({s2_name})')
    keys += [
        f'TOTAL {s2}',
        f'TOTAL {s1}',
        f'TOTAL {s1} RUNNING TIME',
        f'TOTAL {s2} RUNNING TIME',
    ]
    return keys


def compute_overall_stats(rows, s1_col, s2_col, s1_time_col, s2_time_col, s1_name, s2_name, has_set2=True):
    s1 = s1_name.upper()
    s2 = s2_name.upper()

    def safe_float(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.0

    if not has_set2:
        vals = {}
        keys = overall_stats_keys(s1_name, s2_name, has_set2=False)
        idx = 0
        for status in ['PASSED', 'SKIPPED', 'FAILED', 'XFAILED']:
            vals[keys[idx]] = sum(1 for r in rows if r[s1_col] == status)
            idx += 1
        vals[keys[idx]] = sum(1 for r in rows if r[s1_col].strip())
        idx += 1
        vals[keys[idx]] = f'{sum(safe_float(r[s1_time_col]) for r in rows):.2f}'
        return vals

    total_disagree = 0
    total_s2 = 0
    for wf in TEST_CONFIGS:
        wf_rows = [r for r in rows if r['test_config'] == wf]
        s1_skip_not_s2 = sum(
            1 for r in wf_rows
            if r[s1_col] == 'SKIPPED' and r[s2_col] != 'SKIPPED'
        )
        s1_miss_not_s2_skip = sum(
            1 for r in wf_rows
            if r[s1_col] == 'MISSED' and r[s2_col] != 'SKIPPED'
        )
        total_disagree += s1_skip_not_s2 + s1_miss_not_s2_skip
        total_s2 += sum(1 for r in wf_rows if r[s2_col].strip() and r[s2_col].strip() != 'MISSED')

    disagree_pct = (total_disagree / total_s2 * 100) if total_s2 else 0
    agree_pct = 100 - disagree_pct

    vals = {}
    keys = overall_stats_keys(s1_name, s2_name)
    vals[keys[0]] = f'{disagree_pct:.2f}%'
    vals[keys[1]] = f'{agree_pct:.2f}%'

    idx = 2
    for status in ['PASSED', 'SKIPPED', 'FAILED', 'XFAILED']:
        vals[keys[idx]] = sum(1 for r in rows if r[s1_col] == status)
        vals[keys[idx + 1]] = sum(1 for r in rows if r[s2_col] == status)
        idx += 2

    vals[keys[idx]] = sum(1 for r in rows if r[s2_col].strip() and r[s2_col].strip() != 'MISSED')
    idx += 1
    vals[keys[idx]] = sum(1 for r in rows if r[s1_col].strip() and r[s1_col].strip() != 'MISSED')
    idx += 1

    vals[keys[idx]] = f'{sum(safe_float(r[s1_time_col]) for r in rows):.2f}'
    idx += 1
    vals[keys[idx]] = f'{sum(safe_float(r[s2_time_col]) for r in rows):.2f}'
    return vals


def collect_failed_tests(arch_data, archs, s1_name, s2_name):
    """Return a list of failed test rows across all architectures.

    Only collects tests where s1 (ROCm) is FAILED. Each entry records shards
    for both s1 and s2 so the reviewer can look up the failure in either CI
    job. 'also_failing_in' is populated later once log failures are known so
    CUDA log-only failures can be included.
    """
    failed = []
    for arch in archs:
        d = arch_data[arch]
        s1_col, s2_col, _, _ = d['cols']
        has_set2 = d.get('has_set2', True)
        for r in d['rows']:
            s1 = r[s1_col].strip()
            s2 = r[s2_col].strip() if has_set2 else ''
            if s1 == 'FAILED':
                entry = {
                    'arch': arch,
                    'test_file': r.get('test_file', ''),
                    'test_class': r.get('test_class', ''),
                    'test_name': r.get('test_name', ''),
                    'test_config': r.get('test_config', ''),
                    f'shard_{s1_name}': r.get(f'shard_{s1_name}', ''),
                    f'status_{s1_name}': s1,
                }
                if has_set2:
                    entry[f'shard_{s2_name}'] = r.get(f'shard_{s2_name}', '')
                    entry[f'status_{s2_name}'] = s2
                failed.append(entry)

    return failed


def _add_cross_arch_info(failed_tests, log_failures, s2_name):
    """Populate 'also_failing_in' for each entry.

    Matches across other ROCm architectures (from XML-based failures) and also
    includes s2 (CUDA) if a log failure is recorded for the same test tuple.
    """
    from collections import defaultdict
    by_tuple = defaultdict(set)
    for t in failed_tests:
        key = (t['test_file'], t['test_class'], t['test_name'])
        by_tuple[key].add(t['arch'])

    cuda_log_tuples = set()
    for lf in log_failures or []:
        if lf.get('platform', '') == s2_name:
            test_class, test_name = _parse_log_failure_names(lf)
            cuda_log_tuples.add((lf.get('test_file', ''), test_class, test_name))

    for t in failed_tests:
        key = (t['test_file'], t['test_class'], t['test_name'])
        others = sorted(a for a in by_tuple[key] if a != t['arch'])
        if key in cuda_log_tuples and s2_name not in others:
            others.append(s2_name)
        t['also_failing_in'] = ', '.join(others)


def _add_log_failure_cross_arch(log_failures, failed_tests, s1_name, s2_name):
    """Populate 'also_failing_in' for each log failure entry.

    Cross-references: other archs that have the same test failing (either as
    a log failure or as an XML-based failure), plus s2 (CUDA) if it appears
    in log failures for the same test tuple.
    """
    from collections import defaultdict
    by_tuple_archs = defaultdict(set)

    for lf in log_failures or []:
        if lf.get('platform', '') == s1_name:
            test_class, test_name = _parse_log_failure_names(lf)
            key = (lf.get('test_file', ''), test_class, test_name)
            by_tuple_archs[key].add(lf.get('arch', ''))
    for t in failed_tests or []:
        key = (t['test_file'], t['test_class'], t['test_name'])
        by_tuple_archs[key].add(t['arch'])

    cuda_log_tuples = set()
    for lf in log_failures or []:
        if lf.get('platform', '') == s2_name:
            test_class, test_name = _parse_log_failure_names(lf)
            cuda_log_tuples.add((lf.get('test_file', ''), test_class, test_name))

    for lf in log_failures or []:
        test_class, test_name = _parse_log_failure_names(lf)
        key = (lf.get('test_file', ''), test_class, test_name)
        arch = lf.get('arch', '')
        others = sorted(a for a in by_tuple_archs[key] if a and a != arch)
        if key in cuda_log_tuples and s2_name not in others:
            others.append(s2_name)
        lf['also_failing_in'] = ', '.join(others)


def load_log_failures(filepaths):
    """Load log failure CSVs from detect_log_failures.py.

    Extracts the architecture from the filename (e.g. log_failures_mi355.csv -> mi355).
    """
    entries = []
    for fp in filepaths:
        if not os.path.isfile(fp):
            continue
        basename = os.path.basename(fp)
        arch = ''
        if basename.startswith('log_failures_') and basename.endswith('.csv'):
            arch = basename[len('log_failures_'):-len('.csv')]
        with open(fp, newline='') as f:
            for row in csv.DictReader(f):
                row['arch'] = arch
                entries.append(row)
    return entries


def load_flaky_tests_as_log_failures(filepaths):
    """Load flaky_tests_<arch>.csv and return entries shaped like log-failure rows.

    Each returned dict has the same schema as the entries produced by
    load_log_failures, with category='FLAKY' and reason='<test_class>::<test_name>',
    so they can be appended to the log_failures list and surfaced in the
    LOG-BASED FAILURES table alongside crashes/timeouts/etc.
    """
    entries = []
    for fp in filepaths or []:
        if not fp:
            continue
        basename = os.path.basename(fp)
        if not (basename.startswith('log_failures_') and basename.endswith('.csv')):
            continue
        arch = basename[len('log_failures_'):-len('.csv')]
        flaky_path = os.path.join(
            os.path.dirname(fp),
            'flaky_tests_' + basename[len('log_failures_'):],
        )
        if not os.path.isfile(flaky_path):
            continue
        with open(flaky_path, newline='') as f:
            for row in csv.DictReader(f):
                test_class = row.get('test_class', '')
                test_name = row.get('test_name', '')
                entries.append({
                    'arch': arch,
                    'log_file': row.get('log_file', ''),
                    'platform': row.get('platform', ''),
                    'test_config': row.get('test_config', ''),
                    'test_file': row.get('test_file', ''),
                    'job_shard': row.get('job_shard', ''),
                    'test_shard': row.get('test_shard', ''),
                    'status': 'FLAKY',
                    'category': 'FLAKY',
                    'reason': f'{test_class}::{test_name}' if test_class else test_name,
                    'exit_codes': '',
                })
    return entries


def load_log_shards(filepaths):
    """Load log shard inventory CSVs written alongside log_failures files.

    For each log_failures_<arch>.csv, looks for a sibling log_shards_<arch>.csv
    and returns a lookup dict:
        (arch, platform, test_config, job_shard, normalized_test_file) -> test_shards_str

    The CSV is produced by detect_log_failures.py and records every
    (test_file, test_shard) pair observed per job-level shard. If an XML-based
    failure's key matches, we can back-fill the test-level shard value.
    """
    lookup = {}
    for fp in filepaths:
        if not fp:
            continue
        basename = os.path.basename(fp)
        arch = ''
        if basename.startswith('log_failures_') and basename.endswith('.csv'):
            arch = basename[len('log_failures_'):-len('.csv')]
            shards_path = os.path.join(
                os.path.dirname(fp),
                'log_shards_' + basename[len('log_failures_'):],
            )
        else:
            continue
        if not os.path.isfile(shards_path):
            continue
        with open(shards_path, newline='') as f:
            for row in csv.DictReader(f):
                key = (arch, row.get('platform', ''), row.get('test_config', ''),
                       row.get('job_shard', ''),
                       _norm_test_file(row.get('test_file', '')))
                lookup[key] = row.get('test_shards', '')
    return lookup


def _format_test_shards(shards_str):
    """Collapse a test_shards inventory string into a compact display value.

    - '' -> ''
    - '1/1' -> '1/1'
    - '3/14' -> '3/14'
    - '1/14,6/14,12/14' -> '1,6,12/14' (multiple test-level shards observed)
    - mixed totals fall back to the raw string."""
    if not shards_str:
        return ''
    parts = [p for p in shards_str.split(',') if p]
    if len(parts) == 1:
        return parts[0]
    totals = set()
    nums = []
    for p in parts:
        if '/' not in p:
            return shards_str
        a, b = p.split('/', 1)
        totals.add(b)
        nums.append(a)
    if len(totals) == 1:
        return f"{','.join(nums)}/{totals.pop()}"
    return shards_str


def fmt_val(v):
    if isinstance(v, int):
        return f'{v:,}'
    return str(v)


def build_rows(args, archs, arch_data):
    """Return a list of (label, val_per_arch...) tuples and section markers."""
    out = []
    any_has_set2 = any(d.get('has_set2', True) for d in arch_data.values())

    if args.sha:
        out.append(('__header__', f'Commit SHA: {args.sha}'))
    if args.pr_id:
        out.append(('__header__', f'PR ID: {args.pr_id}'))

    wf_keys = test_config_stats_keys(args.set1_name, args.set2_name, has_set2=any_has_set2)
    for wf in TEST_CONFIGS:
        out.append(('__section__', TEST_CONFIG_DISPLAY[wf]))
        arch_stats = {}
        for arch in archs:
            d = arch_data[arch]
            s1_col, s2_col, _, _ = d['cols']
            has_set2 = d.get('has_set2', True)
            wf_rows = [r for r in d['rows'] if r['test_config'] == wf]
            arch_stats[arch] = compute_test_config_stats(
                wf_rows, s1_col, s2_col, args.set1_name, args.set2_name,
                has_set2=has_set2,
            )
        for key in wf_keys:
            out.append((key, [arch_stats[a].get(key, 0) for a in archs]))

    out.append(('__section__', 'OVERALL'))
    ov_keys = overall_stats_keys(args.set1_name, args.set2_name, has_set2=any_has_set2)
    arch_overall = {}
    for arch in archs:
        d = arch_data[arch]
        s1_col, s2_col, s1_time, s2_time = d['cols']
        has_set2 = d.get('has_set2', True)
        arch_overall[arch] = compute_overall_stats(
            d['rows'], s1_col, s2_col, s1_time, s2_time,
            args.set1_name, args.set2_name, has_set2=has_set2,
        )
    for key in ov_keys:
        out.append((key, [arch_overall[a].get(key, 0) for a in archs]))
    return out


def _norm_test_file(path):
    """Normalize a test_file string so XML-sourced ('a.b.c') and log-sourced
    ('a/b/c') forms compare equal. Also strips a trailing .py if present."""
    if not path:
        return ''
    s = path.replace('/', '.')
    if s.endswith('.py'):
        s = s[:-3]
    return s


def _parse_log_failure_names(lf):
    """Extract test_class and test_name from a log failure's reason field.

    Handles formats like 'TestClass::test_method' and
    'TestClass::test_method | extra reason text'.
    """
    reason = lf.get('reason', '')
    if '::' not in reason:
        return '', ''
    test_part = reason.split(' | ', 1)[0] if ' | ' in reason else reason
    parts = test_part.split('::', 1)
    return parts[0], parts[1]


def write_csv(rows, archs, output_path, failed_tests=None, s1_name='set1', s2_name='set2', has_set2=True, log_failures=None, shard_lookup=None):
    csv_rows = []
    csv_rows.append([''] + list(archs))
    for label, vals in rows:
        if label == '__header__':
            csv_rows.append([vals])
        elif label == '__section__':
            csv_rows.append([])
            csv_rows.append([vals])
        else:
            csv_rows.append([label] + list(vals))
    csv_rows.append([])

    s1_failed = [t for t in (failed_tests or []) if t.get(f'status_{s1_name}') == 'FAILED']

    shard_lookup = shard_lookup or {}

    def _xml_test_shard(t, platform):
        key = (t.get('arch', ''), platform, t.get('test_config', ''),
               t.get(f'shard_{platform}', ''),
               _norm_test_file(t.get('test_file', '')))
        return _format_test_shards(shard_lookup.get(key, ''))

    if s1_failed:
        csv_rows.append(['FAILED TESTS'])
        header = ['Arch', 'Test Config', 'Test File', 'Test Class',
                  'Test Name',
                  f'Job-Level Shard ({s1_name})',
                  f'Test-Level Shard ({s1_name})']
        if has_set2:
            header.append(f'Job-Level Shard ({s2_name})')
            header.append(f'Test-Level Shard ({s2_name})')
        header.append(f'Status ({s1_name})')
        if has_set2:
            header.append(f'Status ({s2_name})')
        header.append('Also Failing In')
        csv_rows.append(header)
        for t in s1_failed:
            row = [t['arch'], t['test_config'], t['test_file'],
                   t['test_class'], t['test_name'],
                   t.get(f'shard_{s1_name}', ''),
                   _xml_test_shard(t, s1_name)]
            if has_set2:
                row.append(t.get(f'shard_{s2_name}', ''))
                row.append(_xml_test_shard(t, s2_name))
            row.append(t[f'status_{s1_name}'])
            if has_set2:
                row.append(t.get(f'status_{s2_name}', ''))
            row.append(t.get('also_failing_in', ''))
            csv_rows.append(row)
        csv_rows.append([])

    if log_failures:
        xml_failed_keys = {
            (t['arch'], _norm_test_file(t['test_file']), t['test_class'], t['test_name'])
            for t in (failed_tests or [])
        }
        rocm_log_failures = []
        for lf in log_failures:
            if lf.get('platform', '') != s1_name:
                continue
            test_class, test_name = _parse_log_failure_names(lf)
            key = (lf.get('arch', ''), _norm_test_file(lf.get('test_file', '')),
                   test_class, test_name)
            # Skip entries already present in the XML-based FAILED TESTS table
            # to avoid double-counting the same failure, except for FLAKY
            # entries which represent an independent signal (a rerun passed).
            if key in xml_failed_keys and lf.get('category', '') != 'FLAKY':
                continue
            rocm_log_failures.append(lf)
        if rocm_log_failures:
            csv_rows.append(['LOG-BASED FAILURES (not in XML)'])
            csv_rows.append(['Arch', 'Platform', 'Test Config', 'Test File', 'Test Class',
                             'Test Name', 'Job-Level Shard', 'Test-Level Shard',
                             'Category', 'Also Failing In', 'Log File'])
            for lf in rocm_log_failures:
                test_class, test_name = _parse_log_failure_names(lf)
                csv_rows.append([
                    lf.get('arch', ''), lf.get('platform', ''), lf.get('test_config', ''),
                    lf.get('test_file', ''), test_class, test_name,
                    lf.get('job_shard', ''),
                    lf.get('test_shard', lf.get('shard', '')),
                    lf.get('category', ''),
                    lf.get('also_failing_in', ''),
                    lf.get('log_file', ''),
                ])
            csv_rows.append([])

    with open(output_path, 'w', newline='') as f:
        csv.writer(f).writerows(csv_rows)
    print(f'CSV written to {output_path}')


def write_markdown(rows, archs, output_path, failed_tests=None, s1_name='set1', s2_name='set2', has_set2=True, log_failures=None, shard_lookup=None):
    lines = []
    current_section = []

    def flush_table():
        if not current_section:
            return
        header = '| Metric | ' + ' | '.join(archs) + ' |'
        sep = '| :--- | ' + ' | '.join(['---:'] * len(archs)) + ' |'
        lines.append(header)
        lines.append(sep)
        for label, vals in current_section:
            formatted = [fmt_val(v) for v in vals]
            lines.append(f'| {label} | ' + ' | '.join(formatted) + ' |')
        lines.append('')
        current_section.clear()

    for label, vals in rows:
        if label == '__header__':
            flush_table()
            lines.append(f'**{vals}**')
            lines.append('')
        elif label == '__section__':
            flush_table()
            lines.append(f'### {vals}')
            lines.append('')
        else:
            current_section.append((label, vals))

    flush_table()

    s1_failed = [t for t in (failed_tests or []) if t.get(f'status_{s1_name}') == 'FAILED']

    shard_lookup = shard_lookup or {}

    def _xml_test_shard(t, platform):
        key = (t.get('arch', ''), platform, t.get('test_config', ''),
               t.get(f'shard_{platform}', ''),
               _norm_test_file(t.get('test_file', '')))
        return _format_test_shards(shard_lookup.get(key, ''))

    cols = ['Arch', 'Test Config', 'Test File', 'Test Class', 'Test Name',
            f'Job-Level Shard ({s1_name})',
            f'Test-Level Shard ({s1_name})']
    if has_set2:
        cols.append(f'Job-Level Shard ({s2_name})')
        cols.append(f'Test-Level Shard ({s2_name})')
    cols.append(f'Status ({s1_name})')
    if has_set2:
        cols.append(f'Status ({s2_name})')
    cols.append('Also Failing In')

    if s1_failed:
        lines.append(f'### FAILED TESTS ({len(s1_failed)})')
        lines.append('')
        lines.append('| ' + ' | '.join(cols) + ' |')
        lines.append('| ' + ' | '.join(['---'] * len(cols)) + ' |')
        for t in s1_failed:
            line = (f"| {t['arch']} | {t['test_config']} | {t['test_file']} "
                    f"| {t['test_class']} | {t['test_name']} "
                    f"| {t.get(f'shard_{s1_name}', '')} "
                    f"| {_xml_test_shard(t, s1_name)}")
            if has_set2:
                line += f" | {t.get(f'shard_{s2_name}', '')}"
                line += f" | {_xml_test_shard(t, s2_name)}"
            line += f" | {t[f'status_{s1_name}']}"
            if has_set2:
                line += f" | {t.get(f'status_{s2_name}', '')}"
            line += f" | {t.get('also_failing_in', '')} |"
            lines.append(line)
        lines.append('')
    else:
        lines.append('### FAILED TESTS')
        lines.append('')
        lines.append('No failed tests found.')
        lines.append('')

    if log_failures:
        xml_failed_keys = {
            (t['arch'], _norm_test_file(t['test_file']), t['test_class'], t['test_name'])
            for t in (failed_tests or [])
        }
        rocm_log_failures = []
        for lf in log_failures:
            if lf.get('platform', '') != s1_name:
                continue
            test_class, test_name = _parse_log_failure_names(lf)
            key = (lf.get('arch', ''), _norm_test_file(lf.get('test_file', '')),
                   test_class, test_name)
            if key in xml_failed_keys and lf.get('category', '') != 'FLAKY':
                continue
            rocm_log_failures.append(lf)
        if rocm_log_failures:
            lines.append(f'### LOG-BASED FAILURES (not in XML) ({len(rocm_log_failures)})')
            lines.append('')
            lines.append('These test failures were detected from CI log files but have no XML report')
            lines.append('(typically due to timeouts, crashes, or process kills).')
            lines.append('')
            lines.append('| Arch | Platform | Test Config | Test File | Test Class | Test Name | Job-Level Shard | Test-Level Shard | Category | Also Failing In |')
            lines.append('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')
            for lf in rocm_log_failures:
                test_class, test_name = _parse_log_failure_names(lf)
                lines.append(
                    f"| {lf.get('arch', '')} | {lf.get('platform', '')} | {lf.get('test_config', '')} "
                    f"| {lf.get('test_file', '')} | {test_class} "
                    f"| {test_name} "
                    f"| {lf.get('job_shard', '')} "
                    f"| {lf.get('test_shard', lf.get('shard', ''))} "
                    f"| {lf.get('category', '')} "
                    f"| {lf.get('also_failing_in', '')} |"
                )
            lines.append('')

    md = '\n'.join(lines)
    with open(output_path, 'w') as f:
        f.write(md)
    print(f'Markdown written to {output_path}')
    return md


def main():
    args = parse_args()

    if len(args.csv) != len(args.arch):
        print('Error: --csv and --arch must have the same number of values')
        sys.exit(1)

    archs = args.arch
    arch_data = {}
    for csv_path, arch in zip(args.csv, archs):
        rows = load_csv(csv_path)
        headers = set(rows[0].keys()) if rows else set()
        cols = detect_columns(headers, args.set1_name, args.set2_name)
        s2_col = cols[1]
        has_set2 = any(r.get(s2_col, '').strip() for r in rows)
        arch_data[arch] = {'rows': rows, 'cols': cols, 'has_set2': has_set2}

    data_rows = build_rows(args, archs, arch_data)
    failed = collect_failed_tests(arch_data, archs, args.set1_name, args.set2_name)
    any_has_set2 = any(d.get('has_set2', True) for d in arch_data.values())
    log_failures = load_log_failures(args.log_failures) if args.log_failures else []
    if args.log_failures:
        log_failures.extend(load_flaky_tests_as_log_failures(args.log_failures))
    shard_lookup = load_log_shards(args.log_failures) if args.log_failures else {}

    _add_cross_arch_info(failed, log_failures, args.set2_name)
    _add_log_failure_cross_arch(log_failures, failed, args.set1_name, args.set2_name)

    output_base = args.output
    if output_base.endswith('.csv') or output_base.endswith('.md'):
        output_base = output_base.rsplit('.', 1)[0]

    write_csv(data_rows, archs, f'{output_base}.csv', failed, args.set1_name, args.set2_name, has_set2=any_has_set2, log_failures=log_failures, shard_lookup=shard_lookup)
    write_markdown(data_rows, archs, f'{output_base}.md', failed, args.set1_name, args.set2_name, has_set2=any_has_set2, log_failures=log_failures, shard_lookup=shard_lookup)


if __name__ == '__main__':
    main()
