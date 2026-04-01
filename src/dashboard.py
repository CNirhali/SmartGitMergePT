from flask import Flask, render_template_string, g
from git_utils import GitUtils
from predictor import ConflictPredictor
import argparse
from datetime import datetime, timezone
import secrets
import shlex
from collections import Counter

app = Flask(__name__)

# 🛡️ Sentinel: Register shquote filter to safely generate shell commands
@app.template_filter('shquote')
def shquote_filter(s):
    return shlex.quote(s)

# BOLT: Singleton predictor and git_utils to keep in-memory cache alive across refreshes
repo_path = "demo/conflict_scenarios/demo-repo"
git_utils = GitUtils(repo_path)
predictor = ConflictPredictor(repo_path)

@app.before_request
def generate_nonce():
    g.nonce = secrets.token_urlsafe(16)

template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SmartGitMergePT Dashboard</title>
    <style nonce="{{ nonce }}">
        html { scroll-behavior: smooth; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; margin: 2em; line-height: 1.5; color: #24292f; }
        table { border-collapse: collapse; width: 100%; max-width: 900px; margin-top: 1em; }
        th, td { border: 1px solid #d0d7de; padding: 12px; text-align: left; }
        th { background: #f6f8fa; font-weight: 600; position: sticky; top: 0; z-index: 10; box-shadow: inset 0 -1px 0 #d0d7de; }
        tr:hover { background-color: #f6f8fa; }
        .present { color: #cf222e; font-weight: bold; }
        .absent { color: #1a7f37; }
        .refresh-btn { margin-bottom: 1em; padding: 6px 12px; cursor: pointer; background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px; font-weight: 600; transition: background-color 0.2s; user-select: none; }
        .refresh-btn:hover { background: #f3f4f6; }
        .refresh-btn:active { background: #ebecf0; }
        .refresh-btn:disabled { opacity: 0.6; cursor: not-allowed; }
        .refresh-btn:focus-visible { outline: 2px solid #0969da; outline-offset: 2px; }
        tr { transition: background-color 0.2s, border-color 0.2s; }
        code { background: #afb8c133; padding: 0.2em 0.4em; border-radius: 6px; font-size: 85%; transition: background-color 0.2s, color 0.2s; }
        .timestamp { color: #57606a; font-size: 0.9em; margin-bottom: 1em; }
        .empty-state { padding: 20px; text-align: center; background: #f6f8fa; border: 1px dashed #d0d7de; border-radius: 6px; color: #57606a; }
        .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: 600; border: 1px solid transparent; }
        .badge-success { background: #dafbe1; color: #1a7f37; }
        .badge-error { background: #ffebe9; color: #cf222e; }
        .interactive-badge { cursor: pointer; user-select: none; transition: all 0.2s; }
        .interactive-badge:hover { background: #ffebe9; border-color: #cf222e; }
        .interactive-badge:active { transform: translateY(1px); }
        .interactive-badge.active-filter { background: #cf222e; color: white; box-shadow: 0 0 0 2px #fff, 0 0 0 4px #cf222e; }
        .branch-tag { background: #ddf4ff; color: #0969da; padding: 2px 6px; border-radius: 6px; font-family: ui-monospace, monospace; font-size: 0.85em; text-decoration: none; cursor: pointer; border: 1px solid transparent; transition: all 0.2s; position: relative; display: inline-block; white-space: nowrap; user-select: none; }
        .branch-tag:hover { background: #cfeeff; border-color: #0969da; }
        .branch-tag:focus-visible { outline: 2px solid #0969da; outline-offset: 2px; }
        .branch-tag:active { background: #cfeeff; transform: translateY(1px); }
        .branch-tag.highlight { background: #0969da; color: white; border-color: #0969da; }
        .branch-tag.highlight-secondary { background: #ddf4ff; color: #0969da; border-color: #0969da; }
        .branch-tag.active-filter { background: #0969da; color: white; box-shadow: 0 0 0 2px #fff, 0 0 0 4px #0969da; }
        .branch-tag .conflict-count {
            background: #cf222e;
            color: white;
            border-radius: 10px;
            padding: 0 5px;
            font-size: 10px;
            margin-left: 4px;
            vertical-align: middle;
            display: inline-block;
            line-height: 1.4;
            min-width: 10px;
            text-align: center;
            font-weight: bold;
        }
        .branch-tag.base-branch { border-style: dashed; border-color: #57606a; background-color: #f6f8fa; color: #24292f; }
        .branch-tag.base-branch small { color: #57606a; font-weight: normal; margin-left: 2px; }
        .copy-diff-btn {
            display: none;
            margin-left: 8px;
            padding: 2px 6px;
            font-size: 11px;
            background: #f6f8fa;
            border: 1px solid #d0d7de;
            border-radius: 4px;
            cursor: pointer;
            color: #57606a;
            vertical-align: middle;
            transition: all 0.2s;
        }
        .copy-diff-btn:hover { background: #ebecf0; color: #24292f; border-color: #afb8c1; }
        tr:hover .copy-diff-btn, tr:focus-within .copy-diff-btn { display: inline-block; }
        .file-tag { cursor: pointer; transition: background-color 0.2s, transform 0.1s; position: relative; display: inline-block; white-space: nowrap; user-select: none; }
        .file-tag:hover { background: #afb8c166; }
        .file-tag:focus-visible { outline: 2px solid #0969da; outline-offset: 2px; }
        .file-tag:active { background: #afb8c188; transform: translateY(1px); }
        .file-tag.highlight { background: #0969da; color: white; }
        .copy-success { background-color: #dafbe1 !important; color: #1a7f37 !important; transition: background-color 0.2s; }
        tr.highlight { background-color: #ddf4ff; border-left: 2px solid #0969da; }
        .copy-tooltip { position: absolute; bottom: 125%; left: 50%; transform: translateX(-50%); background: #24292f; color: white; padding: 4px 8px; border-radius: 6px; font-size: 12px; opacity: 0; pointer-events: none; transition: opacity 0.2s; white-space: nowrap; z-index: 1000; }
        .copy-tooltip.show { opacity: 1; }
        kbd { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 3px; box-shadow: inset 0 -1px 0 #d0d7de; color: #24292f; font-family: ui-monospace, monospace; font-size: 11px; padding: 3px 5px; margin-left: 4px; }
        .summary { display: flex; gap: 1em; margin: 1.5em 0; }
        .summary-item { background: #f6f8fa; padding: 12px 20px; border-radius: 8px; border: 1px solid #d0d7de; }
        .conflict-yes { color: #cf222e; font-weight: 600; }
        .conflict-no { color: #57606a; }
        .skip-link {
            position: absolute;
            top: -40px;
            left: 0;
            background: #0969da;
            color: white;
            padding: 8px;
            z-index: 100;
            text-decoration: none;
            border-bottom-right-radius: 6px;
        }
        .skip-link:focus {
            top: 0;
            outline: 2px solid #0969da;
            outline-offset: 2px;
        }
        .filter-container { margin: 1.5em 0; }
        .filter-input {
            width: 100%;
            max-width: 400px;
            padding: 8px 12px;
            font-size: 14px;
            border: 1px solid #d0d7de;
            border-radius: 6px;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        .filter-input:focus {
            border-color: #0969da;
            box-shadow: 0 0 0 3px rgba(9, 105, 218, 0.3);
        }
        .no-results { display: none; text-align: center; padding: 20px; color: #57606a; background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px; margin-top: 1em; }
        .no-results button { margin-left: 8px; margin-bottom: 0; }
        .filter-container { display: flex; align-items: center; gap: 8px; }
        #clear-filter {
            margin-bottom: 0;
            padding: 4px 8px;
            display: none;
            opacity: 0;
            transition: opacity 0.2s;
        }
        #clear-filter.visible { display: inline-block; opacity: 1; }
        .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); border: 0; }
        #monitored-branches-list { margin-bottom: 2em; list-style: none; padding: 0; display: flex; flex-wrap: wrap; gap: 8px; }
        .scenario-list { list-style: none; padding-left: 0; }
        .branch-sep { color: #57606a; margin: 0 4px; }
        #filter-results-count { font-size: 0.9em; color: #57606a; }
        .scenario-types li { padding: 4px 8px; border-radius: 6px; transition: background-color 0.2s, transform 0.1s; cursor: pointer; border: 1px solid transparent; }
        .scenario-types li:hover { background-color: #f6f8fa; }
        .scenario-types li:focus-within { background-color: #f6f8fa; outline: 2px solid #0969da; outline-offset: -2px; }
        .scenario-types li.active-filter { background-color: #ddf4ff; border-color: #0969da; border-left: 4px solid #0969da; }
        .scenario-types li.highlight-secondary { background-color: #ddf4ff; border-color: #0969da; }
        @media (max-width: 600px) {
            body { margin: 1em; }
            .summary { flex-direction: column; }
            .summary-item { width: auto; }
            .filter-input { max-width: none; }
        }
    </style>
</head>
<body>
    <a href="#main-content" class="skip-link">Skip to main content</a>
    <button class="refresh-btn" aria-label="Refresh conflict predictions (Press 'r')">Refresh<kbd>r</kbd></button>
    <div class="timestamp">Last updated: <time id="last-updated" datetime="{{ now.isoformat() }}">{{ now.strftime('%Y-%m-%d %H:%M:%S') }} UTC</time> <span id="relative-time" style="font-size: 0.85em; margin-left: 4px;"></span></div>
    <h1 id="main-content" tabindex="-1">SmartGitMergePT Dashboard</h1>

    <div class="filter-container">
        <label for="filter-input" class="sr-only">Filter branches or conflicts</label>
        <input type="text" id="filter-input" class="filter-input" placeholder="Filter branches or conflicts... (Press '/' to focus)" aria-label="Filter branches or conflicts">
        <button id="clear-filter" class="refresh-btn" aria-label="Clear filter (Press 'Esc')">Clear<kbd>Esc</kbd></button>
        <span id="filter-results-count" aria-live="polite"></span>
    </div>
    <div id="announcer" class="sr-only" aria-live="polite"></div>

    <div class="summary">
        <div class="summary-item">
            <strong>Branches</strong>: {{ branches|length }}
        </div>
        <div class="summary-item">
            <strong>Conflict Pairs</strong>:
            <span class="badge {{ 'badge-error' if predictions else 'badge-success' }}" aria-label="{{ predictions|length }} conflicts detected">
                {{ predictions|length }}
            </span>
        </div>
    </div>

    <h2>Monitored Branches</h2>
    <ul id="monitored-branches-list">
    {% for branch in branches %}
        {% set conflict_count = conflicting_branches.get(branch, 0) %}
        {% set is_base = branch == main_branch %}
        <li><span class="branch-tag {{ 'base-branch' if is_base }}" role="button" tabindex="0" aria-pressed="false" aria-label="Filter and copy branch name: {{ branch }}{{ ' (base branch)' if is_base }}{{ ' (' ~ conflict_count ~ ' predicted conflicts)' if conflict_count > 0 }}" title="Filter and copy" data-branch="{{ branch }}">{{ branch }}{% if is_base %} <small aria-hidden="true">(base)</small>{% endif %}{% if conflict_count > 0 %}<span class="conflict-count" title="{{ conflict_count }} conflicts">{{ conflict_count }}</span>{% endif %}</span></li>
    {% endfor %}
    </ul>

    <h2>Scenario Types</h2>
    <ul class="scenario-types" style="list-style: none; padding-left: 0;">
        <li class="{{ 'present' if scenario_types['file_overlap'] else 'absent' }}" role="button" tabindex="0" aria-pressed="false" data-filter="File Overlap" data-scenario="file_overlap" aria-label="Filter by File Overlap: {{ scenario_types['file_overlap'] }} found">
            {% if scenario_types['file_overlap'] %}<span role="img" aria-label="Warning" title="Detected">⚠️</span>{% else %}<span role="img" aria-label="Clear" title="Not detected">✅</span>{% endif %}
            <strong>File Overlap</strong> ({{ scenario_types['file_overlap'] }}): Both branches modify the same file(s).
        </li>
        <li class="{{ 'present' if scenario_types['line_overlap'] else 'absent' }}" role="button" tabindex="0" aria-pressed="false" data-filter="Line Overlap" data-scenario="line_overlap" aria-label="Filter by Line Overlap: {{ scenario_types['line_overlap'] }} found">
            {% if scenario_types['line_overlap'] %}<span role="img" aria-label="Warning" title="Detected">⚠️</span>{% else %}<span role="img" aria-label="Clear" title="Not detected">✅</span>{% endif %}
            <strong>Line Overlap</strong> ({{ scenario_types['line_overlap'] }}): Both branches change the same or similar lines in a file.
        </li>
        <li class="{{ 'present' if scenario_types['semantic_conflict'] else 'absent' }}" role="button" tabindex="0" aria-pressed="false" data-filter="Semantic Conflict" data-scenario="semantic_conflict" aria-label="Filter by Semantic Conflict: {{ scenario_types['semantic_conflict'] }} found">
            {% if scenario_types['semantic_conflict'] %}<span role="img" aria-label="Warning" title="Detected">⚠️</span>{% else %}<span role="img" aria-label="Clear" title="Not detected">✅</span>{% endif %}
            <strong>Semantic Conflict</strong> ({{ scenario_types['semantic_conflict'] }}): Changes are different but may cause logical or functional conflicts.
        </li>
    </ul>

    <h2>Predicted Conflicts</h2>
    {% if predictions %}
    <table aria-label="Predicted merge conflicts">
        <thead>
            <tr>
                <th scope="col">Branches</th>
                <th scope="col">Files</th>
                <th scope="col">Line Overlap</th>
                <th scope="col">Semantic Conflict</th>
            </tr>
        </thead>
        <tbody>
            {% for pred in predictions %}
            {% set is_base_a = pred['branches'][0] == main_branch %}
            {% set is_base_b = pred['branches'][1] == main_branch %}
            <tr data-branch-a="{{ pred['branches'][0] }}" data-branch-b="{{ pred['branches'][1] }}" data-scenarios="{{ 'file_overlap' if pred.get('files') }} {{ 'line_overlap' if pred.get('line_conflicts') }} {{ 'semantic_conflict' if pred.get('semantic_conflict') }}">
                <td>
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div>
                            <span class="branch-tag {{ 'base-branch' if is_base_a }}" role="button" tabindex="0" aria-pressed="false" aria-label="Filter and copy branch name: {{ pred['branches'][0] }}{{ ' (base branch)' if is_base_a }}" title="Filter and copy" data-branch="{{ pred['branches'][0] }}">{{ pred['branches'][0] }}{% if is_base_a %} <small aria-hidden="true">(base)</small>{% endif %}</span>
                            <span class="branch-sep" aria-hidden="true">↔</span>
                            <span class="branch-tag {{ 'base-branch' if is_base_b }}" role="button" tabindex="0" aria-pressed="false" aria-label="Filter and copy branch name: {{ pred['branches'][1] }}{{ ' (base branch)' if is_base_b }}" title="Filter and copy" data-branch="{{ pred['branches'][1] }}">{{ pred['branches'][1] }}{% if is_base_b %} <small aria-hidden="true">(base)</small>{% endif %}</span>
                        </div>
                        {% if is_base_a %}
                            {% set diff_cmd = "git diff " ~ (pred['branches'][0]|shquote) ~ "..." ~ (pred['branches'][1]|shquote) %}
                        {% elif is_base_b %}
                            {% set diff_cmd = "git diff " ~ (pred['branches'][1]|shquote) ~ "..." ~ (pred['branches'][0]|shquote) %}
                        {% else %}
                            {% set diff_cmd = "git diff " ~ (pred['branches'][0]|shquote) ~ ".." ~ (pred['branches'][1]|shquote) %}
                        {% endif %}
                        <button class="copy-diff-btn" aria-label="Copy git diff command for {{ pred['branches'][0] }} and {{ pred['branches'][1] }}" title="Copy: {{ diff_cmd }}" data-diff="{{ diff_cmd }}">📋 diff</button>
                    </div>
                </td>
                <td>
                    {% if pred['files'] %}
                        <span class="sr-only">File Overlap</span>
                        {% for file in pred['files'] %}
                            <code class="file-tag" role="button" tabindex="0" aria-label="Click to copy and filter by file path: {{ file }}" title="Click to copy and filter" data-copy="{{ file }}">{{ file }}</code>{% if not loop.last %}, {% endif %}
                        {% endfor %}
                    {% else %}
                        <span class="absent" aria-label="No files overlap">—</span>
                    {% endif %}
                </td>
                <td>
                    {% if pred['line_conflicts'] %}
                        <span class="badge badge-error interactive-badge" role="button" tabindex="0" aria-pressed="false" aria-label="Filter by Line Overlap: Line overlap detected" title="Line Overlap: Both branches change the same or similar lines. Click to filter." data-filter="Line Overlap" data-scenario="line_overlap"><span class="sr-only">Line Overlap</span>⚠️ Yes</span>
                    {% else %}
                        <span class="absent" aria-label="No line overlap detected">—</span>
                    {% endif %}
                </td>
                <td>
                    {% if pred['semantic_conflict'] %}
                        <span class="badge badge-error interactive-badge" role="button" tabindex="0" aria-pressed="false" aria-label="Filter by Semantic Conflict: Semantic conflict detected" title="Semantic Conflict: Changes are different but may cause logical or functional conflicts. Click to filter." data-filter="Semantic Conflict" data-scenario="semantic_conflict"><span class="sr-only">Semantic Conflict</span>⚠️ Yes</span>
                    {% else %}
                        <span class="absent" aria-label="No semantic conflict detected">—</span>
                    {% endif %}
                </td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    {% else %}
    <div class="empty-state">No likely conflicts detected between branches. Everything looks clear! ✨</div>
    {% endif %}

    <script nonce="{{ nonce }}">
        const tableRows = document.querySelectorAll('tbody tr');
        const filterInput = document.getElementById('filter-input');
        if (filterInput) {
            filterInput.onfocus = () => filterInput.select();
        }
        const clearFilterBtn = document.getElementById('clear-filter');
        const refreshBtn = document.querySelector('.refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', refresh);
        }
        function refresh() {
            refreshBtn.textContent = 'Refreshing...';
            refreshBtn.disabled = true;
            window.location.reload();
        }

        function applyGlobalFilter(filterValue) {
            // PALETTE: Additive filtering - toggle term in space-separated query
            const terms = filterInput.value.split(/\\s+/).filter(t => t !== '');
            const lowerFilter = filterValue.toLowerCase();
            const index = terms.findIndex(t => t.toLowerCase() === lowerFilter);

            if (index > -1) {
                terms.splice(index, 1);
            } else {
                terms.push(filterValue);
            }
            filterInput.value = terms.join(' ');
            filterInput.dispatchEvent(new Event('input'));
            filterInput.focus();
            // Smooth scroll to table if not in view
            const table = document.querySelector('table');
            if (table && window.getComputedStyle(table).display !== 'none' && filterInput.value !== '') {
                table.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }

        // PALETTE: Select text on focus for easier overwriting
        filterInput.addEventListener('focus', () => {
            // Using setTimeout to ensure selection happens after browser focus behavior
            setTimeout(() => {
                filterInput.select();
            }, 0);
        });

        function copyToClipboard(element, attr = 'data-branch', successElement = null, updateText = false) {
            const text = element.getAttribute(attr);
            if (element.querySelector('.copy-tooltip')) return;
            navigator.clipboard.writeText(text).then(() => {
                const tooltip = document.createElement('div');
                tooltip.className = 'copy-tooltip show';
                tooltip.textContent = 'Copied!';
                element.appendChild(tooltip);

                const el = successElement || element;
                el.classList.add('copy-success');

                let originalHTML = '';
                if (updateText) {
                    originalHTML = element.innerHTML;
                    element.textContent = 'Copied!';
                }

                const announcer = document.getElementById('announcer');
                if (announcer) {
                    announcer.textContent = `Copied ${text} to clipboard`;
                }

                setTimeout(() => {
                    tooltip.remove();
                    el.classList.remove('copy-success');
                    if (updateText) {
                        element.innerHTML = originalHTML;
                    }
                    if (announcer) announcer.textContent = '';
                }, 1000);
            });
        }

        function toggleHighlight(branch, active) {
            document.querySelectorAll(`.branch-tag[data-branch="${branch}"]`).forEach(tag => {
                tag.classList.toggle('highlight', active);
            });
            document.querySelectorAll(`tr[data-branch-a="${branch}"], tr[data-branch-b="${branch}"]`).forEach(row => {
                row.classList.toggle('highlight', active);
                // PALETTE: Highlight the "partner" branch in the monitored list
                const partner = row.getAttribute('data-branch-a') === branch ? row.getAttribute('data-branch-b') : row.getAttribute('data-branch-a');
                document.querySelectorAll(`#monitored-branches-list .branch-tag[data-branch="${partner}"]`).forEach(tag => {
                    tag.classList.toggle('highlight-secondary', active);
                });
            });
        }

        function toggleFileHighlight(file, active) {
            document.querySelectorAll(`.file-tag[data-copy="${file}"]`).forEach(tag => {
                tag.classList.toggle('highlight', active);
                // PALETTE: Highlight the entire conflict row when a file is hovered
                const row = tag.closest('tr');
                if (row) {
                    row.classList.toggle('highlight', active);
                    // PALETTE: Highlight branches involved in this specific conflict row
                    const bA = row.getAttribute('data-branch-a');
                    const bB = row.getAttribute('data-branch-b');
                    document.querySelectorAll(`#monitored-branches-list .branch-tag[data-branch="${bA}"], #monitored-branches-list .branch-tag[data-branch="${bB}"]`).forEach(tag => {
                        tag.classList.toggle('highlight-secondary', active);
                    });
                    // PALETTE: Highlight matching scenario types
                    const scenarios = (row.getAttribute('data-scenarios') || '').split(' ');
                    scenarios.forEach(s => {
                        if (s.trim()) {
                            document.querySelectorAll(`.scenario-types li[data-scenario="${s.trim()}"]`).forEach(item => {
                                item.classList.toggle('highlight-secondary', active);
                            });
                        }
                    });
                }
            });
        }

        function toggleScenarioHighlight(scenario, active) {
            document.querySelectorAll(`.scenario-types li[data-scenario="${scenario}"], .interactive-badge[data-scenario="${scenario}"]`).forEach(item => {
                item.classList.toggle('highlight-secondary', active);
            });
            document.querySelectorAll(`tr[data-scenarios]`).forEach(row => {
                const scenarios = (row.getAttribute('data-scenarios') || '').split(' ');
                if (scenarios.includes(scenario)) {
                    row.classList.toggle('highlight', active);
                    // PALETTE: Also highlight the branches for these rows
                    const bA = row.getAttribute('data-branch-a');
                    const bB = row.getAttribute('data-branch-b');
                    document.querySelectorAll(`#monitored-branches-list .branch-tag[data-branch="${bA}"], #monitored-branches-list .branch-tag[data-branch="${bB}"]`).forEach(tag => {
                        tag.classList.toggle('highlight-secondary', active);
                    });
                }
            });
        }

        document.querySelectorAll('.scenario-types li, .interactive-badge').forEach(item => {
            const scenario = item.getAttribute('data-scenario');
            item.addEventListener('mouseenter', () => toggleScenarioHighlight(scenario, true));
            item.addEventListener('mouseleave', () => toggleScenarioHighlight(scenario, false));
            item.addEventListener('focusin', () => toggleScenarioHighlight(scenario, true));
            item.addEventListener('focusout', () => toggleScenarioHighlight(scenario, false));
            item.addEventListener('click', (e) => {
                e.stopPropagation(); // Avoid triggering row hover or other click handlers
                applyGlobalFilter(item.getAttribute('data-filter'));
            });
            item.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    e.stopPropagation();
                    applyGlobalFilter(item.getAttribute('data-filter'));
                }
            });
        });

        document.querySelectorAll('.copy-diff-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                copyToClipboard(btn, 'data-diff', btn.closest('tr'), true);
            });
        });

        document.querySelectorAll('.branch-tag, .file-tag').forEach(tag => {
            const isFile = tag.classList.contains('file-tag');
            const attr = isFile ? 'data-copy' : 'data-branch';
            tag.addEventListener('click', (e) => {
                e.stopPropagation();
                copyToClipboard(tag, attr);
                // PALETTE: Universal filtering for ALL tags (branches and files)
                const filterVal = isFile ? tag.getAttribute('data-copy') : tag.getAttribute('data-branch');
                applyGlobalFilter(filterVal);
            });

            if (isFile) {
                tag.addEventListener('mouseenter', () => toggleFileHighlight(tag.getAttribute('data-copy'), true));
                tag.addEventListener('mouseleave', () => toggleFileHighlight(tag.getAttribute('data-copy'), false));
                tag.addEventListener('focusin', () => toggleFileHighlight(tag.getAttribute('data-copy'), true));
                tag.addEventListener('focusout', () => toggleFileHighlight(tag.getAttribute('data-copy'), false));
            } else {
                tag.addEventListener('mouseenter', () => toggleHighlight(tag.getAttribute('data-branch'), true));
                tag.addEventListener('mouseleave', () => toggleHighlight(tag.getAttribute('data-branch'), false));
                tag.addEventListener('focusin', () => toggleHighlight(tag.getAttribute('data-branch'), true));
                tag.addEventListener('focusout', () => toggleHighlight(tag.getAttribute('data-branch'), false));
            }

            tag.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    copyToClipboard(tag, attr);
                    // PALETTE: Universal filtering for ALL tags via keyboard
                    const filterVal = isFile ? tag.getAttribute('data-copy') : tag.getAttribute('data-branch');
                    applyGlobalFilter(filterVal);
                }
            });
        });

        const resultsCount = document.getElementById('filter-results-count');
        const noResults = document.createElement('div');
        noResults.className = 'no-results';
        noResults.textContent = 'No matching branches or conflicts found. ';
        const clearBtn = document.createElement('button');
        clearBtn.className = 'refresh-btn';
        clearBtn.setAttribute('aria-label', "Clear filter (Press 'Esc')");
        clearBtn.textContent = 'Clear filter ';
        const kbd = document.createElement('kbd');
        kbd.textContent = 'Esc';
        clearBtn.appendChild(kbd);
        noResults.appendChild(clearBtn);
        clearBtn.addEventListener('click', () => {
            filterInput.value = '';
            filterInput.dispatchEvent(new Event('input'));
            filterInput.focus();
        });

        const table = document.querySelector('table');
        if (table) {
            table.parentNode.insertBefore(noResults, table.nextSibling);
        } else {
            // PALETTE: If no table (no conflicts at all), append after the monitored list
            const monitoredList = document.getElementById('monitored-branches-list');
            if (monitoredList) {
                monitoredList.parentNode.insertBefore(noResults, monitoredList.nextSibling);
            }
        }

        function updatePageTitle(count) {
            const baseTitle = "SmartGitMergePT Dashboard";
            document.title = (count > 0 ? `(${count}) ` : "") + baseTitle;
        }

        // Initial title update
        updatePageTitle(tableRows.length);

        // PALETTE: Restore filter from localStorage
        try {
            const savedFilter = localStorage.getItem('smartgit_dashboard_filter');
            if (savedFilter) {
                filterInput.value = savedFilter;
                // Delay execution slightly to ensure all elements are ready if needed,
                // though here we can call it immediately since DOM is loaded.
                setTimeout(() => {
                    filterInput.dispatchEvent(new Event('input'));
                }, 0);
            }
        } catch (e) {
            console.warn('Could not restore filter from localStorage:', e);
        }

        filterInput.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase();
            const terms = query.split(/\\s+/).filter(t => t !== '');
            // PALETTE: Save filter to localStorage
            try {
                localStorage.setItem('smartgit_dashboard_filter', query);
            } catch (e) {
                console.warn('Could not save filter to localStorage:', e);
            }

            let visibleRowsCount = 0;
            const visibleBranchesInTable = new Set();
            if (query) {
                clearFilterBtn.classList.add('visible');
            } else {
                clearFilterBtn.classList.remove('visible');
            }

            // PALETTE: Capture rows within the handler for robustness against DOM changes
            const currentTableRows = document.querySelectorAll('tbody tr');

            // PALETTE: First pass - determine visible rows and collect branches from them
            currentTableRows.forEach(row => {
                const text = row.textContent.toLowerCase();
                // PALETTE: AND filtering - all terms must match
                const isVisible = terms.every(term => text.includes(term));
                row.style.display = isVisible ? '' : 'none';
                if (isVisible) {
                    visibleRowsCount++;
                    visibleBranchesInTable.add(row.getAttribute('data-branch-a'));
                    visibleBranchesInTable.add(row.getAttribute('data-branch-b'));
                }
            });

            // Update all branch tags (monitored list + table) for highlighting
            document.querySelectorAll('.branch-tag').forEach(tag => {
                const branchName = tag.getAttribute('data-branch');
                const text = branchName.toLowerCase();
                // PALETTE: Active if this specific value is one of the filter terms
                const isActive = terms.some(term => text === term);
                // PALETTE: Intelligent filtering - show if name matches ALL terms OR if branch is in a visible conflict
                if (tag.closest('#monitored-branches-list')) {
                    const matchesQuery = terms.every(term => text.includes(term));
                    const isInVisibleConflict = visibleBranchesInTable.has(branchName);
                    tag.style.display = (matchesQuery || isInVisibleConflict) ? 'inline-block' : 'none';
                }
                tag.classList.toggle('active-filter', isActive);
                // PALETTE: Sync ARIA state
                tag.setAttribute('aria-pressed', isActive ? 'true' : 'false');
            });

            document.querySelectorAll('.scenario-types li, .interactive-badge').forEach(el => {
                const text = el.getAttribute('data-filter').toLowerCase();
                const isActive = terms.some(term => text === term);
                el.classList.toggle('active-filter', isActive);
                // PALETTE: Sync ARIA state
                el.setAttribute('aria-pressed', isActive ? 'true' : 'false');
            });

            if (query === '') {
                resultsCount.textContent = '';
            } else {
                resultsCount.textContent = `Showing ${visibleRowsCount} matching conflict${visibleRowsCount !== 1 ? 's' : ''}`;
            }

            if (table) {
                table.style.display = visibleRowsCount > 0 ? '' : 'none';
            }
            // PALETTE: Check monitored branches list as well for "No results"
            let anyMonitoredVisible = false;
            document.querySelectorAll('#monitored-branches-list .branch-tag').forEach(tag => {
                if (tag.style.display !== 'none') anyMonitoredVisible = true;
            });
            noResults.style.display = ((visibleRowsCount === 0 && !anyMonitoredVisible) && query !== '') ? 'block' : 'none';
            updatePageTitle(visibleRowsCount);
        });

        clearFilterBtn.addEventListener('click', () => {
            filterInput.value = '';
            filterInput.dispatchEvent(new Event('input'));
            filterInput.focus();
        });

        tableRows.forEach(row => {
            const highlightRow = (active) => {
                toggleHighlight(row.getAttribute('data-branch-a'), active);
                toggleHighlight(row.getAttribute('data-branch-b'), active);
                // PALETTE: Highlight matching scenario types
                const scenarios = (row.getAttribute('data-scenarios') || '').split(' ');
                scenarios.forEach(s => {
                    if (s.trim()) {
                        document.querySelectorAll(`.scenario-types li[data-scenario="${s.trim()}"]`).forEach(item => {
                            item.classList.toggle('highlight-secondary', active);
                        });
                    }
                });
            };
            row.addEventListener('mouseenter', () => highlightRow(true));
            row.addEventListener('mouseleave', () => highlightRow(false));
            row.addEventListener('focusin', () => highlightRow(true));
            row.addEventListener('focusout', () => highlightRow(false));
        });

        function updateRelativeTime() {
            const timeEl = document.getElementById('last-updated');
            const relativeEl = document.getElementById('relative-time');
            if (!timeEl || !relativeEl) return;
            const updated = new Date(timeEl.getAttribute('datetime'));
            relativeEl.title = updated.toLocaleString();
            const diffSeconds = Math.floor((Date.now() - updated.getTime()) / 1000);
            let text = '';
            if (isNaN(diffSeconds)) {
                text = '';
            } else if (diffSeconds < 60) {
                text = '(just now)';
            } else if (diffSeconds < 3600) {
                text = `(${Math.floor(diffSeconds / 60)}m ago)`;
            } else if (diffSeconds < 86400) {
                text = `(${Math.floor(diffSeconds / 3600)}h ago)`;
            } else {
                text = `(${Math.floor(diffSeconds / 86400)}d ago)`;
            }
            relativeEl.textContent = text;
        }
        setInterval(updateRelativeTime, 30000);
        updateRelativeTime();

        document.addEventListener('keydown', (e) => {
            if (e.key.toLowerCase() === 'r' && !e.ctrlKey && !e.metaKey && !e.altKey && document.activeElement !== filterInput) {
                refresh();
            }
            if (e.key === '/' && document.activeElement !== filterInput) {
                e.preventDefault();
                filterInput.focus();
                // PALETTE: Select text on focus for easier overwriting
                filterInput.select();
            }
            if (e.key === 'Escape') {
                if (filterInput.value !== '') {
                    filterInput.value = '';
                    filterInput.dispatchEvent(new Event('input'));
                }
                if (document.activeElement === filterInput) {
                    filterInput.blur();
                }
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def dashboard():
    branches = git_utils.list_branches()
    # PALETTE: Determine main branch for visualization
    if 'main' in branches:
        main_branch = 'main'
    elif 'master' in branches:
        main_branch = 'master'
    else:
        try:
            main_branch = git_utils.repo.active_branch.name
        except:
            main_branch = 'main'

    predictions = predictor.predict_conflicts(branches)

    # BOLT: Calculate scenario types in a single pass over predictions
    file_overlap_count = 0
    line_overlap_count = 0
    semantic_conflict_count = 0

    conflicting_branches = Counter()
    for pred in predictions:
        if pred.get('files'):
            file_overlap_count += 1
        if pred.get('line_conflicts'):
            line_overlap_count += 1
        if pred.get('semantic_conflict'):
            semantic_conflict_count += 1

        # PALETTE: Count conflicts per branch for more informative status
        for b in pred['branches']:
            conflicting_branches[b] += 1

    # PALETTE: Sort branches by (is_not_base, -conflict_count, name)
    # This puts the base branch first, then those with most conflicts.
    branches.sort(key=lambda b: (b != main_branch, -conflicting_branches[b], b))

    scenario_types = {
        'file_overlap': file_overlap_count,
        'line_overlap': line_overlap_count,
        'semantic_conflict': semantic_conflict_count
    }
    return render_template_string(
        template,
        nonce=g.nonce,
        branches=branches,
        main_branch=main_branch,
        predictions=predictions,
        conflicting_branches=conflicting_branches,
        scenario_types=scenario_types,
        now=datetime.now(timezone.utc)
    )

@app.after_request
def add_security_headers(response):
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
    response.headers['Cross-Origin-Resource-Policy'] = 'same-origin'
    response.headers['Permissions-Policy'] = 'camera=(), microphone=(), geolocation=()'
    csp = (
        "default-src 'self'; "
        f"style-src 'self' 'nonce-{g.get('nonce', '')}'; "
        f"script-src 'self' 'nonce-{g.get('nonce', '')}'; "
        "img-src 'self'; "
        "connect-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "frame-ancestors 'none'"
    )
    response.headers['Content-Security-Policy'] = csp
    return response

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SmartGitMergePT Dashboard')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the dashboard on')
    args = parser.parse_args()
    # Explicitly binding to 127.0.0.1 for security
    app.run(debug=False, port=args.port, host='127.0.0.1')
