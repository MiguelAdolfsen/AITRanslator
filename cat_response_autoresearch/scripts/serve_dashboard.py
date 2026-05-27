from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CAT Autoresearch</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f4f1;
      --panel: #fff;
      --ink: #242326;
      --muted: #79706b;
      --line: #e7e0dc;
      --accent: #b7795f;
      --accent-2: #5b8ea3;
      --bad: #a94442;
      --good: #497d55;
      --shadow: 0 10px 24px rgba(36, 35, 38, 0.06);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 44px 28px 64px;
    }
    .topbar {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 18px;
      margin-bottom: 28px;
    }
    h1 {
      margin: 0 0 4px;
      font-size: 32px;
      line-height: 1.05;
      letter-spacing: 0;
    }
    .subtitle {
      color: var(--muted);
      font-size: 14px;
    }
    .live {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px;
      border-radius: 999px;
      color: #fff;
      background: #b43d3d;
      font-size: 11px;
      font-weight: 800;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    .live.idle { background: #7d7774; }
    .dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: currentColor;
      box-shadow: 0 0 0 4px rgba(255,255,255,.18);
    }
    .cards {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 16px;
      margin-bottom: 26px;
    }
    .card, .panel {
      background: var(--panel);
      border: 1px solid rgba(231,224,220,.7);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .card {
      min-height: 112px;
      padding: 19px 20px;
    }
    .label {
      color: var(--muted);
      font-size: 11px;
      line-height: 1.2;
      font-weight: 800;
      letter-spacing: .14em;
      text-transform: uppercase;
    }
    .value {
      margin-top: 12px;
      font-size: 34px;
      line-height: 1;
      font-weight: 760;
    }
    .value.accent { color: var(--accent); }
    .value.good { color: var(--good); }
    .value.bad { color: var(--bad); }
    .small {
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) minmax(340px, .9fr);
      gap: 16px;
      margin-bottom: 16px;
    }
    .panel {
      padding: 18px;
      overflow: hidden;
    }
    .panel h2 {
      margin: 0 0 16px;
      font-size: 13px;
      letter-spacing: .14em;
      text-transform: uppercase;
      color: var(--muted);
    }
    canvas {
      width: 100%;
      height: 270px;
      display: block;
    }
    .progress-shell {
      margin-top: 12px;
      height: 10px;
      background: #eee8e4;
      border-radius: 999px;
      overflow: hidden;
    }
    .progress-fill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
      transition: width .2s ease;
    }
    .kv {
      display: grid;
      grid-template-columns: 142px 1fr;
      gap: 8px 12px;
      font-size: 13px;
    }
    .kv b { color: var(--muted); font-weight: 700; }
    code {
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      padding: 9px 8px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-size: 11px;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    .status-pill {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      background: #ebe5e0;
      color: var(--muted);
    }
    .status-pill.good { background: #e5f0e6; color: var(--good); }
    .status-pill.bad { background: #f4e4e2; color: var(--bad); }
    @media (max-width: 860px) {
      .cards, .grid { grid-template-columns: 1fr; }
      main { padding: 30px 16px 48px; }
    }
  </style>
</head>
<body>
<main>
  <div class="topbar">
    <div>
      <h1>CAT Autoresearch <span id="liveBadge" class="live idle"><span class="dot"></span><span id="liveText">Idle</span></span></h1>
      <div id="subtitle" class="subtitle">Loading...</div>
    </div>
  </div>

  <section class="cards">
    <div class="card">
      <div class="label">Kept Best</div>
      <div id="bestApproval" class="value accent">--</div>
      <div id="bestRun" class="small">No kept run yet</div>
    </div>
    <div class="card">
      <div class="label">Main Approval</div>
      <div id="latestApproval" class="value">--</div>
      <div id="latestRun" class="small">No main runs yet</div>
    </div>
    <div class="card">
      <div class="label">Main Quality</div>
      <div id="latestScore" class="value">--</div>
      <div id="latestFailure" class="small">Hard failure: --</div>
    </div>
    <div class="card">
      <div class="label">Runs / Kept</div>
      <div id="runsKept" class="value">0/0</div>
      <div id="lastUpdated" class="small">Last update: --</div>
    </div>
  </section>

  <section class="grid">
    <div class="panel">
      <h2>Approval Trend</h2>
      <canvas id="approvalChart" width="900" height="300"></canvas>
    </div>
    <div class="panel">
      <h2>Active Run</h2>
      <div class="kv">
        <b>Status</b><span id="activeStatus">--</span>
        <b>Run</b><code id="activeRun">--</code>
        <b>Benchmark</b><span id="activeBenchmark">--</span>
        <b>Progress</b><span id="activeProgress">--</span>
        <b>Current Case</b><code id="activeCase">--</code>
        <b>Profile</b><code id="activeProfile">--</code>
        <b>Blind Report</b><span id="activeBlind">--</span>
      </div>
      <div class="progress-shell"><div id="progressFill" class="progress-fill"></div></div>
    </div>
  </section>

  <section class="grid">
    <div class="panel">
      <h2>Current Prompt / Settings</h2>
      <div id="currentPrompt" class="kv"></div>
    </div>
    <div class="panel">
      <h2>Category Health</h2>
      <table>
        <thead><tr><th>Category</th><th>Approval</th><th>Evaluations</th></tr></thead>
        <tbody id="categoryRows"><tr><td colspan="3">No summary yet.</td></tr></tbody>
      </table>
    </div>
  </section>

  <section class="panel">
    <h2>Recent Runs</h2>
    <table>
      <thead><tr><th>Run</th><th>Benchmark</th><th>Approval</th><th>Quality</th><th>Repeats</th><th>Low Categories</th><th>Status</th></tr></thead>
      <tbody id="runRows"><tr><td colspan="7">No runs yet.</td></tr></tbody>
    </table>
  </section>
</main>

<script>
const fmtPct = value => Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : '--';
const fmtNum = value => Number.isFinite(Number(value)) ? Number(value).toFixed(3) : '--';
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function benchmarkRole(name) {
  if (name === 'real_mined') return {label: 'private real_mined', role: 'MAIN', cls: 'good'};
  if (name === 'synthetic') return {label: 'synthetic', role: 'REGRESSION', cls: ''};
  if (String(name || '').includes('holdout')) return {label: name || 'holdout', role: 'HOLDOUT', cls: 'bad'};
  return {label: name || '--', role: 'UNKNOWN', cls: ''};
}

function drawChart(canvas, rows) {
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, width, height);
  const pad = {left: 48, right: 18, top: 16, bottom: 34};
  ctx.strokeStyle = '#eee5e0';
  ctx.lineWidth = 1;
  ctx.fillStyle = '#8a817c';
  ctx.font = '12px system-ui';
  for (let i = 0; i <= 5; i++) {
    const y = pad.top + (height - pad.top - pad.bottom) * i / 5;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    const value = Math.round((1 - i / 5) * 100);
    ctx.fillText(String(value), 12, y + 4);
  }
  if (!rows.length) return;
  const xFor = index => rows.length === 1 ? pad.left : pad.left + (width - pad.left - pad.right) * index / (rows.length - 1);
  const yFor = value => pad.top + (height - pad.top - pad.bottom) * (1 - Math.max(0, Math.min(1, Number(value) || 0)));
  ctx.strokeStyle = '#b7795f';
  ctx.lineWidth = 3;
  ctx.beginPath();
  rows.forEach((row, index) => {
    const x = xFor(index);
    const y = yFor(row.cat_approval_rate);
    if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  rows.forEach((row, index) => {
    const x = xFor(index);
    const y = yFor(row.cat_approval_rate);
    ctx.fillStyle = row.hard_failure === 'True' || row.hard_failure === true ? '#a94442' : '#b7795f';
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fill();
  });
}

function renderState(state) {
  const rows = state.results || [];
  const latest = state.latest_main_result || state.latest_result || {};
  const best = state.best || {};
  const active = state.active_progress || {};
  const running = active.status === 'running';
  const latestBench = benchmarkRole(latest.benchmark_set);
  const bestBench = benchmarkRole(state.best_summary?.benchmark_set || best.benchmark_set);
  const activeBench = benchmarkRole(active.benchmark_set);
  const anyLatest = state.latest_result || {};
  const regression = state.latest_synthetic_result || {};
  const holdout = state.latest_holdout_result || {};
  document.getElementById('liveBadge').className = running ? 'live' : 'live idle';
  document.getElementById('liveText').textContent = running ? 'Live' : 'Idle';
  document.getElementById('subtitle').textContent = `CAT response optimization - focus: private real_mined - synthetic is regression only - ${rows.length} runs - last: ${anyLatest.run_id || '--'}`;

  document.getElementById('bestApproval').textContent = fmtPct(best.cat_approval_rate);
  document.getElementById('bestRun').textContent = best.best_run_id ? `${best.best_run_id} - ${bestBench.role.toLowerCase()} - quality ${fmtNum(best.best_quality_score ?? best.best_score)}` : 'No kept run yet';
  document.getElementById('latestApproval').textContent = fmtPct(latest.cat_approval_rate);
  document.getElementById('latestRun').textContent = latest.run_id ? `${latest.run_id} - ${latestBench.role.toLowerCase()}` : 'No runs yet';
  document.getElementById('latestScore').textContent = fmtNum(latest.cat_quality_score ?? latest.cat_response_score);
  document.getElementById('latestFailure').textContent = `Hard failure: ${latest.hard_failure ?? '--'}${state.latest_main_summary?.blind_report ? ' - blind report' : ''} | synthetic ${fmtPct(regression.cat_approval_rate)} | holdout ${fmtPct(holdout.cat_approval_rate)}`;
  const kept = rows.filter(row => String(row.kept).toLowerCase() === 'true').length;
  document.getElementById('runsKept').textContent = `${rows.length}/${kept}`;
  document.getElementById('lastUpdated').textContent = `Last update: ${state.last_updated || '--'}`;

  document.getElementById('activeStatus').innerHTML = `<span class="status-pill ${running ? 'bad' : 'good'}">${esc(active.status || 'idle')}</span>`;
  document.getElementById('activeRun').textContent = active.run_id || '--';
  document.getElementById('activeBenchmark').innerHTML = `<span class="status-pill ${activeBench.cls}">${esc(activeBench.role)}</span> ${esc(activeBench.label)}`;
  document.getElementById('activeProgress').textContent = active.evaluations_total ? `${active.completed_evaluations}/${active.evaluations_total} (${fmtPct(active.progress_rate)})` : '--';
  document.getElementById('activeCase').textContent = active.current_case_id || '--';
  document.getElementById('activeProfile').textContent = active.profile_name ? `${active.profile_name} ${active.profile_hash || ''}` : '--';
  document.getElementById('activeBlind').innerHTML = `<span class="status-pill ${active.blind_report ? 'good' : ''}">${active.blind_report ? 'enabled' : 'off'}</span>`;
  document.getElementById('progressFill').style.width = `${Math.max(0, Math.min(100, Number(active.progress_rate || 0) * 100))}%`;

  const promptSummary = running && state.active_summary?.profile ? state.active_summary : (state.latest_main_summary?.profile ? state.latest_main_summary : (state.latest_summary?.profile ? state.latest_summary : state.best_summary || {}));
  const profile = promptSummary.profile || {};
  const production = promptSummary.production_cat || {};
  const promptSource = running && state.active_summary?.profile ? 'active run' : (state.latest_main_summary?.profile ? 'latest main run' : (state.latest_summary?.profile ? 'latest run' : (state.best_summary?.profile ? 'best run' : '--')));
  const currentPrompt = document.getElementById('currentPrompt');
  const promptRows = [
    ['Source', promptSource],
    ['Run', promptSummary.run_id || '--'],
    ['Profile', profile.name || '--'],
    ['Prompt Version', production.cat_prompt_version || '--'],
    ['Validation Version', production.cat_validation_version || '--'],
    ['Prompt', profile.prompt_template || '--'],
    ['Retry', profile.retry_prompt_template || '--'],
    ['System', profile.system_prompt || '--'],
    ['Settings', `num_predict=${profile.num_predict ?? '--'}, temp=${profile.temperature ?? '--'}, top_p=${profile.top_p ?? '--'}, top_k=${profile.top_k ?? '--'}`],
  ];
  currentPrompt.innerHTML = promptRows.map(([k, v]) => `<b>${esc(k)}</b><code>${esc(v)}</code>`).join('');

  const categorySummary = running && state.active_summary?.metrics ? state.active_summary : (state.latest_main_summary?.metrics ? state.latest_main_summary : state.latest_summary || {});
  const categories = categorySummary?.metrics?.category_metrics || {};
  const categoryRows = document.getElementById('categoryRows');
  const entries = Object.entries(categories).sort((a, b) => Number(a[1].approval_rate) - Number(b[1].approval_rate));
  categoryRows.innerHTML = entries.length ? entries.map(([name, data]) => {
    const cls = Number(data.approval_rate) < 0.8 ? 'bad' : 'good';
    return `<tr><td>${esc(name)}</td><td><span class="status-pill ${cls}">${fmtPct(data.approval_rate)}</span></td><td>${esc(data.evaluations)}</td></tr>`;
  }).join('') : '<tr><td colspan="3">No summary yet.</td></tr>';

  const runRows = document.getElementById('runRows');
  const recent = [...rows].slice(-10).reverse();
  runRows.innerHTML = recent.length ? recent.map(row => {
    const hard = String(row.hard_failure).toLowerCase() === 'true';
    const bench = benchmarkRole(row.benchmark_set);
    return `<tr>
      <td><code>${esc(row.run_id)}</code></td>
      <td><span class="status-pill ${bench.cls}">${esc(bench.role)}</span><div class="small">${esc(bench.label)}</div></td>
      <td>${fmtPct(row.cat_approval_rate)}</td>
      <td>${fmtNum(row.cat_quality_score ?? row.cat_response_score)}</td>
      <td>${esc(row.repeat_count || '')}</td>
      <td>${esc(row.low_categories || 'none')}</td>
      <td><span class="status-pill ${hard ? 'bad' : 'good'}">${hard ? 'hard fail' : 'ok'}</span></td>
    </tr>`;
  }).join('') : '<tr><td colspan="7">No runs yet.</td></tr>';

  drawChart(document.getElementById('approvalChart'), rows);
}

async function refresh() {
  try {
    const response = await fetch('/api/state', {cache: 'no-store'});
    renderState(await response.json());
  } catch (error) {
    document.getElementById('subtitle').textContent = `Dashboard error: ${error}`;
  }
}

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def read_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle, delimiter="\t") if row.get("run_id")]
    return rows


def file_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def newest_json(paths: list[Path]) -> tuple[Path | None, dict[str, Any]]:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None, {}
    newest = max(existing, key=file_mtime)
    return newest, read_json(newest)


def find_run_files(runs_dir: Path, filename: str) -> list[Path]:
    if not runs_dir.exists():
        return []
    return sorted(runs_dir.glob(f"*/{filename}"), key=file_mtime)


def row_from_summary(summary_path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    progress = read_json(summary_path.parent / "progress.json")
    return {
        "run_id": str(summary.get("run_id") or summary_path.parent.name),
        "timestamp": str(progress.get("updated_at") or progress.get("started_at") or ""),
        "benchmark_set": str(summary.get("benchmark_set") or ""),
        "cat_response_score": str(summary.get("cat_response_score", metrics.get("cat_response_score", ""))),
        "cat_quality_score": str(summary.get("cat_quality_score", metrics.get("cat_quality_score", ""))),
        "cat_latency_score": str(summary.get("cat_latency_score", metrics.get("cat_latency_score", ""))),
        "cat_approval_rate": str(summary.get("cat_approval_rate", metrics.get("cat_approval_rate", ""))),
        "cases_total": str(metrics.get("cases_total", "")),
        "repeat_count": str(metrics.get("repeat_count", "")),
        "evaluations_total": str(metrics.get("evaluations_total", "")),
        "retried_count": str(metrics.get("retried_count", "")),
        "retry_rate": str(metrics.get("retry_rate", "")),
        "primary_rejected_count": str(metrics.get("primary_rejected_count", "")),
        "primary_rejected_rate": str(metrics.get("primary_rejected_rate", "")),
        "low_categories": str(metrics.get("low_categories", "")),
        "hard_failure": str(summary.get("hard_failure", metrics.get("hard_failure", ""))),
        "kept": "False",
        "notes": f"profile={summary.get('profile', {}).get('name', '')} hash={summary.get('profile_hash', '')}",
    }


def merge_summary_rows(rows: list[dict[str, Any]], summary_files: list[Path]) -> list[dict[str, Any]]:
    merged = list(rows)
    known_run_ids = {row.get("run_id") for row in rows}
    for summary_path in summary_files:
        summary = read_json(summary_path)
        run_id = str(summary.get("run_id") or summary_path.parent.name)
        if not run_id or run_id in known_run_ids:
            continue
        merged.append(row_from_summary(summary_path, summary))
        known_run_ids.add(run_id)
    return merged


def latest_row_for_benchmark(rows: list[dict[str, Any]], benchmark_set: str) -> dict[str, Any]:
    for row in reversed(rows):
        if row.get("benchmark_set") == benchmark_set:
            return row
    return {}


def latest_holdout_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in reversed(rows):
        if "holdout" in str(row.get("benchmark_set", "")):
            return row
    return {}


def summary_for_row(runs_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    run_id = row.get("run_id")
    if not run_id:
        return {}
    return read_json(runs_dir / str(run_id) / "summary.json")


def dashboard_state(results_path: Path, best_path: Path, runs_dir: Path) -> dict[str, Any]:
    base_rows = read_results(results_path)
    summary_files = find_run_files(runs_dir, "summary.json")
    rows = merge_summary_rows(base_rows, summary_files)
    latest_result = rows[-1] if rows else {}
    latest_main_result = latest_row_for_benchmark(rows, "real_mined")
    latest_synthetic_result = latest_row_for_benchmark(rows, "synthetic")
    latest_holdout_result = latest_holdout_row(rows)
    best = read_json(best_path)
    progress_files = find_run_files(runs_dir, "progress.json")
    latest_summary = summary_for_row(runs_dir, latest_result)
    latest_main_summary = summary_for_row(runs_dir, latest_main_result)
    latest_synthetic_summary = summary_for_row(runs_dir, latest_synthetic_result)
    latest_holdout_summary = summary_for_row(runs_dir, latest_holdout_result)
    if not latest_summary:
        _, latest_summary = newest_json(summary_files)
    _, active_progress = newest_json(progress_files)
    if active_progress.get("status") != "running" and progress_files:
        # Keep the most recent completed progress visible, but the badge stays idle.
        active_progress = active_progress or {}

    best_summary: dict[str, Any] = {}
    best_run_id = best.get("best_run_id")
    if best_run_id:
        best_summary = read_json(runs_dir / str(best_run_id) / "summary.json")
    active_summary: dict[str, Any] = {}
    active_run_id = active_progress.get("run_id")
    if active_run_id:
        active_summary = read_json(runs_dir / str(active_run_id) / "summary.json")

    last_updated = ""
    latest_times = [file_mtime(path) for path in [results_path, best_path, *(summary_files[-1:] or []), *(progress_files[-1:] or [])]]
    if latest_times:
        last_updated = datetime.fromtimestamp(max(latest_times), timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "schema_version": 1,
        "last_updated": last_updated,
        "results_path": str(results_path),
        "runs_dir": str(runs_dir),
        "best": best,
        "results": rows,
        "latest_result": latest_result,
        "latest_main_result": latest_main_result,
        "latest_synthetic_result": latest_synthetic_result,
        "latest_holdout_result": latest_holdout_result,
        "latest_summary": latest_summary,
        "latest_main_summary": latest_main_summary,
        "latest_synthetic_summary": latest_synthetic_summary,
        "latest_holdout_summary": latest_holdout_summary,
        "best_summary": best_summary,
        "active_summary": active_summary,
        "active_progress": active_progress,
    }


class DashboardHandler(BaseHTTPRequestHandler):
    results_path: Path
    best_path: Path
    runs_dir: Path

    def log_message(self, format: str, *args: Any) -> None:
        return

    def send_bytes(self, content: bytes, *, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self.send_bytes(DASHBOARD_HTML.encode("utf-8"), content_type="text/html; charset=utf-8")
            return
        if parsed.path == "/api/state":
            payload = dashboard_state(self.results_path, self.best_path, self.runs_dir)
            self.send_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"), content_type="application/json; charset=utf-8")
            return
        content_type = mimetypes.guess_type(parsed.path)[0] or "text/plain"
        self.send_bytes(f"Not found: {parsed.path}".encode("utf-8"), content_type=content_type, status=404)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve a local CAT response autoresearch dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "cat_response_autoresearch" / "results" / "results.tsv")
    parser.add_argument("--best", type=Path, default=REPO_ROOT / "cat_response_autoresearch" / "results" / "best.json")
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "cat_response_autoresearch" / "runs")
    args = parser.parse_args(argv)

    DashboardHandler.results_path = args.results
    DashboardHandler.best_path = args.best
    DashboardHandler.runs_dir = args.runs
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"CAT autoresearch dashboard: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
