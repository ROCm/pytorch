#!/usr/bin/env python3

import argparse
import csv
import json
import html
import sys
from collections import Counter, defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate an interactive HTML dashboard from parity CSVs'
    )
    parser.add_argument(
        '--csv', nargs='+', required=True,
        help='CSV file(s) (one per architecture, same order as --arch)'
    )
    parser.add_argument(
        '--arch', nargs='+', required=True,
        help='Architecture labels matching --csv order'
    )
    parser.add_argument('--sha', type=str, default='', help='Commit SHA')
    parser.add_argument('--pr_id', type=str, default='', help='Pull request ID')
    parser.add_argument('--set1_name', type=str, default='rocm')
    parser.add_argument('--set2_name', type=str, default='cuda')
    parser.add_argument(
        '--prev_csv', nargs='*', default=[],
        help='Previous week CSV file(s) (one per architecture, same order as --arch)'
    )
    parser.add_argument('--output', type=str, default='parity_dashboard.html')
    return parser.parse_args()


def load_csv(filepath):
    with open(filepath, newline='') as f:
        return list(csv.DictReader(f))


WORKFLOWS = ['default', 'distributed', 'inductor']
WORKFLOW_DISPLAY = {
    'default': 'Default',
    'distributed': 'Distributed',
    'inductor': 'Inductor',
}


def compute_stats(rows, s1_col, s2_col, s1_time_col, s2_time_col):
    stats = {}
    for wf in WORKFLOWS:
        wf_rows = [r for r in rows if r.get('work_flow_name') == wf]
        s1_skip_not_s2 = sum(1 for r in wf_rows if r[s1_col] == 'SKIPPED' and r[s2_col] != 'SKIPPED')
        s1_miss_not_s2 = sum(1 for r in wf_rows if r[s1_col] == 'MISSED' and r[s2_col] != 'SKIPPED')
        total_s2 = sum(1 for r in wf_rows if r[s2_col].strip())
        total_s1 = sum(1 for r in wf_rows if r[s1_col].strip())
        skip_miss = s1_skip_not_s2 + s1_miss_not_s2
        disagree_pct = (skip_miss / total_s2 * 100) if total_s2 else 0
        stats[wf] = {
            'skipped': s1_skip_not_s2,
            'missed': s1_miss_not_s2,
            'total_s1': total_s1,
            'total_s2': total_s2,
            'skip_miss': skip_miss,
            'disagree_pct': round(disagree_pct, 2),
        }

    total_disagree = sum(s['skip_miss'] for s in stats.values())
    total_s2 = sum(s['total_s2'] for s in stats.values())
    overall_disagree = (total_disagree / total_s2 * 100) if total_s2 else 0

    status_counts = Counter()
    for r in rows:
        for col, prefix in [(s1_col, 'rocm_'), (s2_col, 'cuda_')]:
            status_counts[prefix + r[col].strip()] += 1

    def safe_float(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.0

    stats['overall'] = {
        'disagree_pct': round(overall_disagree, 2),
        'agree_pct': round(100 - overall_disagree, 2),
        'total_s1': sum(1 for r in rows if r[s1_col].strip()),
        'total_s2': total_s2,
        'passed_s1': status_counts.get('rocm_PASSED', 0),
        'passed_s2': status_counts.get('cuda_PASSED', 0),
        'skipped_s1': status_counts.get('rocm_SKIPPED', 0),
        'skipped_s2': status_counts.get('cuda_SKIPPED', 0),
        'failed_s1': status_counts.get('rocm_FAILED', 0),
        'failed_s2': status_counts.get('cuda_FAILED', 0),
        'xfailed_s1': status_counts.get('rocm_XFAILED', 0),
        'xfailed_s2': status_counts.get('cuda_XFAILED', 0),
        'time_s1': round(sum(safe_float(r.get(s1_time_col, 0)) for r in rows), 2),
        'time_s2': round(sum(safe_float(r.get(s2_time_col, 0)) for r in rows), 2),
        'new_tests': sum(1 for r in rows if r.get('existed_last_week', '') == 'no'),
    }
    return stats


def get_skip_reason_counts(rows, s1_col, s2_col):
    by_workflow = defaultdict(lambda: Counter())
    for r in rows:
        reason = r.get('skip_reason', '').strip()
        if not reason:
            s1 = r[s1_col].strip()
            s2 = r[s2_col].strip()
            if s1 == 'SKIPPED' and s2 != 'SKIPPED':
                reason = '(unlabeled skip)'
            elif s1 == 'MISSED' and s2 != 'SKIPPED':
                reason = '(unlabeled miss)'
            else:
                continue
        wf = r.get('work_flow_name', 'unknown')
        by_workflow[wf][reason] += 1
    return by_workflow


def get_failed_tests(rows, arch, s1_col, s2_col, s1_name, s2_name):
    failed = []
    for r in rows:
        s1 = r[s1_col].strip()
        s2 = r[s2_col].strip()
        if s1 == 'FAILED' or s2 == 'FAILED':
            failed.append({
                'arch': arch,
                'test_file': r.get('test_file', ''),
                'test_class': r.get('test_class', ''),
                'test_name': r.get('test_name', ''),
                'workflow': r.get('work_flow_name', ''),
                f'status_{s1_name}': s1,
                f'status_{s2_name}': s2,
            })
    return failed


def build_html(args, all_data, prev_data=None):
    arch_stats = {}
    arch_skip_reasons = {}
    arch_prev_skip_reasons = {}
    all_failed = []
    all_tests_json = []

    for arch, data in all_data.items():
        rows = data['rows']
        s1_col = data['s1_col']
        s2_col = data['s2_col']
        s1_time = data['s1_time']
        s2_time = data['s2_time']

        arch_stats[arch] = compute_stats(rows, s1_col, s2_col, s1_time, s2_time)
        arch_skip_reasons[arch] = get_skip_reason_counts(rows, s1_col, s2_col)
        all_failed.extend(get_failed_tests(rows, arch, s1_col, s2_col, args.set1_name, args.set2_name))

        if prev_data and arch in prev_data:
            pdata = prev_data[arch]
            arch_prev_skip_reasons[arch] = get_skip_reason_counts(
                pdata['rows'], pdata['s1_col'], pdata['s2_col']
            )

        for r in rows:
            s1 = r[s1_col].strip()
            s2 = r[s2_col].strip()
            reason = r.get('skip_reason', '').strip()
            is_new = r.get('existed_last_week', '') == 'no'
            interesting = (
                s1 in ('SKIPPED', 'MISSED', 'FAILED')
                or s2 in ('SKIPPED', 'MISSED', 'FAILED')
                or reason
                or is_new
                or (s1 != s2)
            )
            if interesting:
                all_tests_json.append({
                    'arch': arch,
                    'tf': r.get('test_file', ''),
                    'tc': r.get('test_class', ''),
                    'tn': r.get('test_name', ''),
                    'wf': r.get('work_flow_name', ''),
                    'sr': reason,
                    's1': s1,
                    's2': s2,
                    'new': is_new,
                })

    skip_reasons_json = {}
    for arch, wf_counts in arch_skip_reasons.items():
        skip_reasons_json[arch] = {}
        for wf, counts in wf_counts.items():
            skip_reasons_json[arch][wf] = dict(counts.most_common())

    prev_skip_reasons_json = {}
    for arch, wf_counts in arch_prev_skip_reasons.items():
        prev_skip_reasons_json[arch] = {}
        for wf, counts in wf_counts.items():
            prev_skip_reasons_json[arch][wf] = dict(counts.most_common())

    return HTML_TEMPLATE.replace(
        '/*DATA_PLACEHOLDER*/',
        f"""
        const ARCHS = {json.dumps(args.arch)};
        const SHA = {json.dumps(args.sha)};
        const PR_ID = {json.dumps(args.pr_id)};
        const SET1 = {json.dumps(args.set1_name)};
        const SET2 = {json.dumps(args.set2_name)};
        const STATS = {json.dumps(arch_stats)};
        const SKIP_REASONS = {json.dumps(skip_reasons_json)};
        const PREV_SKIP_REASONS = {json.dumps(prev_skip_reasons_json)};
        const FAILED = {json.dumps(all_failed)};
        const WORKFLOWS = {json.dumps(WORKFLOWS)};
        const WF_DISPLAY = {json.dumps(WORKFLOW_DISPLAY)};
        const ALL_TESTS = {json.dumps(all_tests_json)};
        """
    )


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Parity Dashboard</title>
<style>
:root {
  --bg: #0d1117;
  --surface: #161b22;
  --border: #30363d;
  --text: #e6edf3;
  --text-dim: #8b949e;
  --accent: #58a6ff;
  --green: #3fb950;
  --red: #f85149;
  --yellow: #d29922;
  --orange: #db6d28;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  background: var(--bg); color: var(--text);
  line-height: 1.5; padding: 24px;
}
h1 { font-size: 28px; font-weight: 600; margin-bottom: 8px; }
.meta { color: var(--text-dim); font-size: 14px; margin-bottom: 24px; }
.meta code { background: var(--surface); padding: 2px 6px; border-radius: 4px; font-size: 13px; color: var(--accent); }
.tabs { display: flex; gap: 4px; margin-bottom: 20px; border-bottom: 1px solid var(--border); }
.tab { padding: 8px 16px; cursor: pointer; border: none; background: none; color: var(--text-dim);
  font-size: 14px; border-bottom: 2px solid transparent; transition: all 0.15s; }
.tab:hover { color: var(--text); }
.tab.active { color: var(--text); border-bottom-color: var(--accent); }
.panel { display: none; }
.panel.active { display: block; }

.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; margin-bottom: 24px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px; }
.card h3 { font-size: 16px; font-weight: 600; margin-bottom: 12px; color: var(--accent); }
.stat-row { display: flex; justify-content: space-between; padding: 4px 0; font-size: 14px; }
.stat-row .label { color: var(--text-dim); }
.stat-row .value { font-weight: 600; font-variant-numeric: tabular-nums; }
.pct-good { color: var(--green); }
.pct-bad { color: var(--red); }
.pct-warn { color: var(--yellow); }

.arch-tabs { display: flex; gap: 8px; margin-bottom: 16px; }
.arch-tab { padding: 6px 14px; cursor: pointer; border: 1px solid var(--border); background: var(--surface);
  color: var(--text-dim); border-radius: 20px; font-size: 13px; transition: all 0.15s; }
.arch-tab:hover { border-color: var(--accent); color: var(--text); }
.arch-tab.active { background: var(--accent); color: #000; border-color: var(--accent); }

table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; padding: 8px 12px; background: var(--surface); border-bottom: 2px solid var(--border);
  font-weight: 600; color: var(--text-dim); position: sticky; top: 0; }
td { padding: 6px 12px; border-bottom: 1px solid var(--border); }
tr:hover td { background: rgba(88, 166, 255, 0.05); }
.status-PASSED { color: var(--green); }
.status-SKIPPED { color: var(--yellow); }
.status-FAILED { color: var(--red); }
.status-MISSED { color: var(--orange); }
.status-XFAILED { color: var(--text-dim); }
.new-badge { background: var(--accent); color: #000; font-size: 10px; padding: 1px 5px;
  border-radius: 8px; font-weight: 600; margin-left: 4px; }

.skip-table { margin-bottom: 16px; }
.skip-table th:last-child, .skip-table td:last-child { text-align: right; }
.reason-bar { display: inline-block; height: 8px; background: var(--accent); border-radius: 4px;
  margin-right: 8px; vertical-align: middle; min-width: 2px; }

.delta-up { color: var(--red); font-size: 11px; margin-left: 4px; }
.delta-down { color: var(--green); font-size: 11px; margin-left: 4px; }
.delta-same { color: var(--text-dim); font-size: 11px; margin-left: 4px; }

.search { display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; align-items: center; }
.search input, .search select { background: var(--surface); border: 1px solid var(--border);
  color: var(--text); padding: 8px 12px; border-radius: 6px; font-size: 13px; }
.search input { flex: 1; min-width: 200px; }
.search select { min-width: 140px; }
.search .count { color: var(--text-dim); font-size: 13px; padding: 8px 0; }

.table-wrap { max-height: 600px; overflow-y: auto; border: 1px solid var(--border); border-radius: 8px; }
</style>
</head>
<body>

<h1>Parity Dashboard</h1>
<div class="meta" id="meta"></div>

<div class="tabs" id="main-tabs">
  <button class="tab active" data-panel="summary">Summary</button>
  <button class="tab" data-panel="skip-reasons">Skip Reasons</button>
  <button class="tab" data-panel="failed">Failed Tests</button>
  <button class="tab" data-panel="all-tests">All Tests</button>
</div>

<div class="panel active" id="panel-summary"></div>
<div class="panel" id="panel-skip-reasons"></div>
<div class="panel" id="panel-failed"></div>
<div class="panel" id="panel-all-tests"></div>

<script>
/*DATA_PLACEHOLDER*/

document.getElementById('main-tabs').addEventListener('click', e => {
  if (!e.target.classList.contains('tab')) return;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  e.target.classList.add('active');
  document.getElementById('panel-' + e.target.dataset.panel).classList.add('active');
});

const meta = document.getElementById('meta');
let parts = [];
if (SHA) parts.push('Commit: <code>' + SHA + '</code>');
if (PR_ID) parts.push('PR: <code>#' + PR_ID + '</code>');
parts.push('Architectures: ' + ARCHS.map(a => '<code>' + a.toUpperCase() + '</code>').join(', '));
meta.innerHTML = parts.join(' &middot; ');

function fmt(n) { return typeof n === 'number' ? n.toLocaleString() : n; }
function pctClass(p) { return p <= 1 ? 'pct-good' : p <= 2 ? 'pct-warn' : 'pct-bad'; }

// Summary panel
function renderSummary() {
  const el = document.getElementById('panel-summary');
  let h = '<div class="arch-tabs" id="summary-arch-tabs">';
  ARCHS.forEach((a, i) => h += `<button class="arch-tab ${i===0?'active':''}" data-arch="${a}">${a.toUpperCase()}</button>`);
  h += '</div>';
  ARCHS.forEach((a, i) => {
    const s = STATS[a];
    h += `<div class="arch-panel" data-arch="${a}" style="${i>0?'display:none':''}">`;
    h += '<div class="cards">';
    WORKFLOWS.forEach(wf => {
      const ws = s[wf];
      h += `<div class="card"><h3>${WF_DISPLAY[wf]}</h3>
        <div class="stat-row"><span class="label">SKIPPED (${SET1} not ${SET2})</span><span class="value">${fmt(ws.skipped)}</span></div>
        <div class="stat-row"><span class="label">MISSED</span><span class="value">${fmt(ws.missed)}</span></div>
        <div class="stat-row"><span class="label">Total ${SET1.toUpperCase()}</span><span class="value">${fmt(ws.total_s1)}</span></div>
        <div class="stat-row"><span class="label">Total ${SET2.toUpperCase()}</span><span class="value">${fmt(ws.total_s2)}</span></div>
        <div class="stat-row"><span class="label">DISAGREE %</span><span class="value ${pctClass(ws.disagree_pct)}">${ws.disagree_pct}%</span></div>
      </div>`;
    });
    const o = s.overall;
    h += `<div class="card"><h3>Overall</h3>
      <div class="stat-row"><span class="label">AGREE %</span><span class="value pct-good">${o.agree_pct}%</span></div>
      <div class="stat-row"><span class="label">DISAGREE %</span><span class="value ${pctClass(o.disagree_pct)}">${o.disagree_pct}%</span></div>
      <div class="stat-row"><span class="label">PASSED (${SET1})</span><span class="value">${fmt(o.passed_s1)}</span></div>
      <div class="stat-row"><span class="label">PASSED (${SET2})</span><span class="value">${fmt(o.passed_s2)}</span></div>
      <div class="stat-row"><span class="label">SKIPPED (${SET1})</span><span class="value">${fmt(o.skipped_s1)}</span></div>
      <div class="stat-row"><span class="label">SKIPPED (${SET2})</span><span class="value">${fmt(o.skipped_s2)}</span></div>
      <div class="stat-row"><span class="label">FAILED (${SET1})</span><span class="value ${o.failed_s1>0?'pct-bad':''}">${fmt(o.failed_s1)}</span></div>
      <div class="stat-row"><span class="label">FAILED (${SET2})</span><span class="value ${o.failed_s2>0?'pct-bad':''}">${fmt(o.failed_s2)}</span></div>
      <div class="stat-row"><span class="label">New tests this week</span><span class="value">${fmt(o.new_tests)}</span></div>
      <div class="stat-row"><span class="label">${SET1.toUpperCase()} running time</span><span class="value">${fmt(o.time_s1)}s</span></div>
      <div class="stat-row"><span class="label">${SET2.toUpperCase()} running time</span><span class="value">${fmt(o.time_s2)}s</span></div>
    </div>`;
    h += '</div></div>';
  });
  el.innerHTML = h;
  document.getElementById('summary-arch-tabs').addEventListener('click', e => {
    if (!e.target.classList.contains('arch-tab')) return;
    const arch = e.target.dataset.arch;
    el.querySelectorAll('.arch-tab').forEach(t => t.classList.toggle('active', t.dataset.arch === arch));
    el.querySelectorAll('.arch-panel').forEach(p => p.style.display = p.dataset.arch === arch ? '' : 'none');
  });
}

// Skip reasons panel
function renderSkipReasons() {
  const el = document.getElementById('panel-skip-reasons');
  const hasPrev = Object.keys(PREV_SKIP_REASONS).length > 0;
  let h = '<div class="arch-tabs" id="skip-arch-tabs">';
  ARCHS.forEach((a, i) => h += `<button class="arch-tab ${i===0?'active':''}" data-arch="${a}">${a.toUpperCase()}</button>`);
  h += '</div>';
  ARCHS.forEach((a, i) => {
    const archData = SKIP_REASONS[a] || {};
    const prevArchData = PREV_SKIP_REASONS[a] || {};
    h += `<div class="arch-panel" data-arch="${a}" style="${i>0?'display:none':''}">`;
    if (hasPrev) {
      const summaryParts = [];
      WORKFLOWS.forEach(wf => {
        const counts = archData[wf] || {};
        const prevCounts = prevArchData[wf] || {};
        if (Object.keys(counts).length === 0) return;
        const allReasons = new Set([...Object.keys(counts), ...Object.keys(prevCounts)]);
        const ups = [], downs = [];
        allReasons.forEach(r => {
          const d = (counts[r] || 0) - (prevCounts[r] || 0);
          if (d > 0) ups.push(`${r} +${fmt(d)}`);
          else if (d < 0) downs.push(`${r} ${fmt(d)}`);
        });
        const parts = [...ups, ...downs];
        if (parts.length === 0) {
          summaryParts.push(`<b>${WF_DISPLAY[wf]}:</b> All categories flat (no changes).`);
        } else {
          summaryParts.push(`<b>${WF_DISPLAY[wf]}:</b> ${parts.join(', ')}; everything else flat.`);
        }
      });
      h += `<div class="card" style="margin-bottom:16px;font-size:13px;line-height:1.6;color:var(--text-dim)">${summaryParts.join(' ')}</div>`;
    }
    WORKFLOWS.forEach(wf => {
      const counts = archData[wf];
      if (!counts || Object.keys(counts).length === 0) return;
      const prevCounts = prevArchData[wf] || {};
      const allReasons = new Set([...Object.keys(counts), ...Object.keys(prevCounts)]);
      const sorted = [...allReasons].map(r => [r, counts[r] || 0, prevCounts[r] || 0]).sort((a, b) => b[1] - a[1]);
      const max = Math.max(...sorted.map(x => x[1]), 1);
      const total = sorted.reduce((s, [, v]) => s + v, 0);
      const prevTotal = sorted.reduce((s, [,, v]) => s + v, 0);
      let totalDelta = '';
      if (hasPrev) {
        const d = total - prevTotal;
        if (d > 0) totalDelta = `<span class="delta-up"> +${fmt(d)}</span>`;
        else if (d < 0) totalDelta = `<span class="delta-down"> ${fmt(d)}</span>`;
        else totalDelta = `<span class="delta-same"> +0</span>`;
      }
      h += `<h3 style="margin: 16px 0 8px; color: var(--accent)">${WF_DISPLAY[wf]} <span style="color:var(--text-dim);font-weight:400">(${fmt(total)} total${totalDelta})</span></h3>`;
      h += '<div class="table-wrap" style="max-height:400px"><table class="skip-table"><thead><tr><th>Skip Reason</th>';
      h += '<th>This Week</th>';
      if (hasPrev) h += '<th>Last Week</th><th>Delta</th>';
      h += '</tr></thead><tbody>';
      sorted.forEach(([reason, count, prev]) => {
        const w = Math.max(2, (count / max) * 120);
        h += `<tr><td><span class="reason-bar" style="width:${w}px"></span>${esc(reason)}</td>`;
        h += `<td>${fmt(count)}</td>`;
        if (hasPrev) {
          h += `<td>${fmt(prev)}</td>`;
          const d = count - prev;
          if (d > 0) h += `<td class="delta-up">+${fmt(d)}</td>`;
          else if (d < 0) h += `<td class="delta-down">${fmt(d)}</td>`;
          else h += `<td class="delta-same">-</td>`;
        }
        h += '</tr>';
      });
      h += '</tbody></table></div>';
    });
    h += '</div>';
  });
  el.innerHTML = h;
  document.getElementById('skip-arch-tabs').addEventListener('click', e => {
    if (!e.target.classList.contains('arch-tab')) return;
    const arch = e.target.dataset.arch;
    el.querySelectorAll('.arch-tab').forEach(t => t.classList.toggle('active', t.dataset.arch === arch));
    el.querySelectorAll('.arch-panel').forEach(p => p.style.display = p.dataset.arch === arch ? '' : 'none');
  });
}

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

// Failed tests panel
function renderFailed() {
  const el = document.getElementById('panel-failed');
  if (FAILED.length === 0) {
    el.innerHTML = '<div class="card"><h3>No failed tests</h3></div>';
    return;
  }
  let h = `<p style="margin-bottom:12px;color:var(--text-dim)">${FAILED.length} failed test(s)</p>`;
  h += '<div class="table-wrap"><table><thead><tr><th>Arch</th><th>Workflow</th><th>Test File</th><th>Test Class</th><th>Test Name</th><th>' + SET1 + '</th><th>' + SET2 + '</th></tr></thead><tbody>';
  FAILED.forEach(t => {
    const s1 = t['status_' + SET1];
    const s2 = t['status_' + SET2];
    h += `<tr><td>${esc(t.arch)}</td><td>${esc(t.workflow)}</td><td>${esc(t.test_file)}</td><td>${esc(t.test_class)}</td><td>${esc(t.test_name)}</td>
      <td class="status-${s1}">${s1}</td><td class="status-${s2}">${s2}</td></tr>`;
  });
  h += '</tbody></table></div>';
  el.innerHTML = h;
}

// All tests panel with filtering
let currentPage = 0;
const PAGE_SIZE = 200;
let filteredTests = [];

function renderAllTests() {
  const el = document.getElementById('panel-all-tests');

  const reasons = new Set();
  ALL_TESTS.forEach(t => { if (t.sr) reasons.add(t.sr); });
  const reasonOpts = Array.from(reasons).sort();

  let h = '<div class="search">';
  h += '<input type="text" id="test-search" placeholder="Search test file, class, or name...">';
  h += '<select id="filter-arch"><option value="">All archs</option>';
  ARCHS.forEach(a => h += `<option value="${a}">${a.toUpperCase()}</option>`);
  h += '</select>';
  h += '<select id="filter-wf"><option value="">All workflows</option>';
  WORKFLOWS.forEach(wf => h += `<option value="${wf}">${WF_DISPLAY[wf]}</option>`);
  h += '</select>';
  h += '<select id="filter-status-s1"><option value="">All ' + SET1 + '</option>';
  ['PASSED','SKIPPED','FAILED','MISSED','XFAILED'].forEach(s => h += `<option value="${s}">${s}</option>`);
  h += '</select>';
  h += '<select id="filter-status-s2"><option value="">All ' + SET2 + '</option>';
  ['PASSED','SKIPPED','FAILED','MISSED','XFAILED'].forEach(s => h += `<option value="${s}">${s}</option>`);
  h += '</select>';
  h += '<select id="filter-reason"><option value="">All skip reasons</option><option value="__empty__">(no reason)</option>';
  reasonOpts.forEach(r => h += `<option value="${esc(r)}">${esc(r)}</option>`);
  h += '</select>';
  h += '<label style="font-size:13px;color:var(--text-dim)"><input type="checkbox" id="filter-new"> New only</label>';
  h += '<span class="count" id="test-count"></span>';
  h += '</div>';
  h += '<div class="table-wrap" id="test-table-wrap"></div>';
  h += '<div style="margin-top:12px;display:flex;gap:8px;align-items:center" id="pagination"></div>';
  el.innerHTML = h;

  const search = document.getElementById('test-search');
  const fArch = document.getElementById('filter-arch');
  const fWf = document.getElementById('filter-wf');
  const fStatusS1 = document.getElementById('filter-status-s1');
  const fStatusS2 = document.getElementById('filter-status-s2');
  const fReason = document.getElementById('filter-reason');
  const fNew = document.getElementById('filter-new');

  function applyFilters() {
    const q = search.value.toLowerCase();
    const arch = fArch.value;
    const wf = fWf.value;
    const statusS1 = fStatusS1.value;
    const statusS2 = fStatusS2.value;
    const reason = fReason.value;
    const newOnly = fNew.checked;

    filteredTests = ALL_TESTS.filter(t => {
      if (arch && t.arch !== arch) return false;
      if (wf && t.wf !== wf) return false;
      if (statusS1 && t.s1 !== statusS1) return false;
      if (statusS2 && t.s2 !== statusS2) return false;
      if (reason === '__empty__' && t.sr) return false;
      if (reason && reason !== '__empty__' && t.sr !== reason) return false;
      if (newOnly && !t.new) return false;
      if (q && !(t.tf + ' ' + t.tc + ' ' + t.tn).toLowerCase().includes(q)) return false;
      return true;
    });
    currentPage = 0;
    renderPage();
  }

  function renderPage() {
    const start = currentPage * PAGE_SIZE;
    const page = filteredTests.slice(start, start + PAGE_SIZE);
    const total = filteredTests.length;
    document.getElementById('test-count').textContent = fmt(total) + ' tests';

    let h = '<table><thead><tr><th>Arch</th><th>Workflow</th><th>Test File</th><th>Test Class</th><th>Test Name</th><th>Skip Reason</th><th>' + SET1 + '</th><th>' + SET2 + '</th></tr></thead><tbody>';
    page.forEach(t => {
      h += `<tr><td>${esc(t.arch)}</td><td>${esc(t.wf)}</td><td>${esc(t.tf)}</td><td>${esc(t.tc)}</td>
        <td>${esc(t.tn)}${t.new ? '<span class="new-badge">NEW</span>' : ''}</td>
        <td>${esc(t.sr)}</td>
        <td class="status-${t.s1}">${t.s1}</td><td class="status-${t.s2}">${t.s2}</td></tr>`;
    });
    h += '</tbody></table>';
    document.getElementById('test-table-wrap').innerHTML = h;

    const totalPages = Math.ceil(total / PAGE_SIZE);
    let ph = '';
    if (totalPages > 1) {
      ph += `<button class="arch-tab ${currentPage===0?'':'active'}" onclick="prevPage()" ${currentPage===0?'disabled':''}>Prev</button>`;
      ph += `<span style="color:var(--text-dim);font-size:13px">Page ${currentPage+1} of ${totalPages}</span>`;
      ph += `<button class="arch-tab ${currentPage>=totalPages-1?'':'active'}" onclick="nextPage()" ${currentPage>=totalPages-1?'disabled':''}>Next</button>`;
    }
    document.getElementById('pagination').innerHTML = ph;
  }

  window.prevPage = () => { if (currentPage > 0) { currentPage--; renderPage(); } };
  window.nextPage = () => { if ((currentPage + 1) * PAGE_SIZE < filteredTests.length) { currentPage++; renderPage(); } };

  [search, fArch, fWf, fStatusS1, fStatusS2, fReason, fNew].forEach(el => {
    el.addEventListener(el.type === 'checkbox' ? 'change' : 'input', applyFilters);
  });

  filteredTests = ALL_TESTS;
  renderPage();
}

renderSummary();
renderSkipReasons();
renderFailed();
renderAllTests();
</script>
</body>
</html>"""


def main():
    args = parse_args()
    if len(args.csv) != len(args.arch):
        print('Error: --csv and --arch must have the same number of values')
        sys.exit(1)

    all_data = {}
    for csv_path, arch in zip(args.csv, args.arch):
        rows = load_csv(csv_path)
        if not rows:
            print(f'Warning: No rows in {csv_path}')
            continue
        headers = set(rows[0].keys())
        s1_col = f'status_{args.set1_name}' if f'status_{args.set1_name}' in headers else 'status_set1'
        s2_col = f'status_{args.set2_name}' if f'status_{args.set2_name}' in headers else 'status_set2'
        s1_time = f'running_time_{args.set1_name}' if f'running_time_{args.set1_name}' in headers else 'running_time_set1'
        s2_time = f'running_time_{args.set2_name}' if f'running_time_{args.set2_name}' in headers else 'running_time_set2'
        all_data[arch] = {'rows': rows, 's1_col': s1_col, 's2_col': s2_col, 's1_time': s1_time, 's2_time': s2_time}

    prev_data = {}
    if args.prev_csv:
        for csv_path, arch in zip(args.prev_csv, args.arch):
            if not csv_path or csv_path == 'NONE':
                continue
            try:
                rows = load_csv(csv_path)
                if not rows:
                    continue
                headers = set(rows[0].keys())
                s1_col = f'status_{args.set1_name}' if f'status_{args.set1_name}' in headers else 'status_set1'
                s2_col = f'status_{args.set2_name}' if f'status_{args.set2_name}' in headers else 'status_set2'
                prev_data[arch] = {'rows': rows, 's1_col': s1_col, 's2_col': s2_col}
                print(f'Loaded previous CSV for {arch}: {csv_path} ({len(rows)} rows)')
            except Exception as e:
                print(f'Warning: Could not load prev CSV {csv_path}: {e}')

    html_content = build_html(args, all_data, prev_data if prev_data else None)
    output = args.output if args.output.endswith('.html') else args.output + '.html'
    with open(output, 'w') as f:
        f.write(html_content)
    print(f'Dashboard written to {output}')


if __name__ == '__main__':
    main()
