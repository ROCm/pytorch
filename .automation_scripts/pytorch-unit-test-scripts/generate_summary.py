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
        return list(csv.DictReader(f))


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
    """Return a list of failed test rows across all architectures."""
    failed = []
    for arch in archs:
        d = arch_data[arch]
        s1_col, s2_col, _, _ = d['cols']
        has_set2 = d.get('has_set2', True)
        for r in d['rows']:
            s1 = r[s1_col].strip()
            s2 = r[s2_col].strip() if has_set2 else ''
            if s1 == 'FAILED' or s2 == 'FAILED':
                shard = r.get(f'shard_{s1_name}', '') if s1 == 'FAILED' else r.get(f'shard_{s2_name}', '')
                entry = {
                    'arch': arch,
                    'test_file': r.get('test_file', ''),
                    'test_class': r.get('test_class', ''),
                    'test_name': r.get('test_name', ''),
                    'test_config': r.get('test_config', ''),
                    'shard': shard,
                    f'status_{s1_name}': s1,
                }
                if has_set2:
                    entry[f'status_{s2_name}'] = s2
                failed.append(entry)
    return failed


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
            out.append((key, [arch_stats[a][key] for a in archs]))

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
        out.append((key, [arch_overall[a][key] for a in archs]))
    return out


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


def write_csv(rows, archs, output_path, failed_tests=None, s1_name='set1', s2_name='set2', has_set2=True, log_failures=None):
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

    if failed_tests:
        csv_rows.append(['FAILED TESTS'])
        header = ['Arch', 'Test Config', 'Test File', 'Test Class',
                  'Test Name', 'Shard', f'Status ({s1_name})']
        if has_set2:
            header.append(f'Status ({s2_name})')
        csv_rows.append(header)
        for t in failed_tests:
            row = [t['arch'], t['test_config'], t['test_file'],
                   t['test_class'], t['test_name'], t.get('shard', ''),
                   t[f'status_{s1_name}']]
            if has_set2:
                row.append(t.get(f'status_{s2_name}', ''))
            csv_rows.append(row)
        csv_rows.append([])

    if log_failures:
        csv_rows.append(['LOG-BASED FAILURES (not in XML)'])
        csv_rows.append(['Arch', 'Platform', 'Test Config', 'Test File', 'Test Class', 'Test Name', 'Shard', 'Category', 'Log File'])
        for lf in log_failures:
            test_class, test_name = _parse_log_failure_names(lf)
            csv_rows.append([
                lf.get('arch', ''), lf.get('platform', ''), lf.get('test_config', ''),
                lf.get('test_file', ''), test_class, test_name,
                lf.get('shard', ''), lf.get('category', ''),
                lf.get('log_file', ''),
            ])
        csv_rows.append([])

    with open(output_path, 'w', newline='') as f:
        csv.writer(f).writerows(csv_rows)
    print(f'CSV written to {output_path}')


def write_markdown(rows, archs, output_path, failed_tests=None, s1_name='set1', s2_name='set2', has_set2=True, log_failures=None):
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

    if failed_tests:
        lines.append('### FAILED TESTS')
        lines.append('')
        cols = ['Arch', 'Test Config', 'Test File', 'Test Class', 'Test Name',
                'Shard', f'Status ({s1_name})']
        if has_set2:
            cols.append(f'Status ({s2_name})')
        lines.append('| ' + ' | '.join(cols) + ' |')
        lines.append('| ' + ' | '.join(['---'] * len(cols)) + ' |')
        for t in failed_tests:
            line = (f"| {t['arch']} | {t['test_config']} | {t['test_file']} "
                    f"| {t['test_class']} | {t['test_name']} "
                    f"| {t.get('shard', '')} | {t[f'status_{s1_name}']}")
            if has_set2:
                line += f" | {t.get(f'status_{s2_name}', '')}"
            line += ' |'
            lines.append(line)
        lines.append('')
    else:
        lines.append('### FAILED TESTS')
        lines.append('')
        lines.append('No failed tests found.')
        lines.append('')

    if log_failures:
        lines.append('### LOG-BASED FAILURES (not in XML)')
        lines.append('')
        lines.append('These test failures were detected from CI log files but have no XML report')
        lines.append('(typically due to timeouts, crashes, or process kills).')
        lines.append('')
        lines.append('| Arch | Platform | Test Config | Test File | Test Class | Test Name | Shard | Category |')
        lines.append('| --- | --- | --- | --- | --- | --- | --- | --- |')
        for lf in log_failures:
            test_class, test_name = _parse_log_failure_names(lf)
            lines.append(
                f"| {lf.get('arch', '')} | {lf.get('platform', '')} | {lf.get('test_config', '')} "
                f"| {lf.get('test_file', '')} | {test_class} "
                f"| {test_name} | {lf.get('shard', '')} "
                f"| {lf.get('category', '')} |"
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

    output_base = args.output
    if output_base.endswith('.csv') or output_base.endswith('.md'):
        output_base = output_base.rsplit('.', 1)[0]

    write_csv(data_rows, archs, f'{output_base}.csv', failed, args.set1_name, args.set2_name, has_set2=any_has_set2, log_failures=log_failures)
    write_markdown(data_rows, archs, f'{output_base}.md', failed, args.set1_name, args.set2_name, has_set2=any_has_set2, log_failures=log_failures)


if __name__ == '__main__':
    main()
