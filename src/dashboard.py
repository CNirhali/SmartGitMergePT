from flask import Flask, render_template_string
from git_utils import GitUtils
from predictor import ConflictPredictor
import argparse
from datetime import datetime, timezone

app = Flask(__name__)

template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SmartGitMergePT Dashboard</title>
    <style>
        html { scroll-behavior: smooth; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; margin: 2em; line-height: 1.5; color: #24292f; }
        table { border-collapse: collapse; width: 100%; max-width: 900px; margin-top: 1em; }
        th, td { border: 1px solid #d0d7de; padding: 12px; text-align: left; }
        th { background: #f6f8fa; font-weight: 600; }
        tr:hover { background-color: #f6f8fa; }
        .present { color: #cf222e; font-weight: bold; }
        .absent { color: #1a7f37; }
        .refresh-btn { margin-bottom: 1em; padding: 6px 12px; cursor: pointer; background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px; font-weight: 600; transition: background-color 0.2s; }
        .refresh-btn:hover { background: #f3f4f6; }
        .refresh-btn:active { background: #ebecf0; }
        tr { transition: background-color 0.1s; }
        code { background: #afb8c133; padding: 0.2em 0.4em; border-radius: 6px; font-size: 85%; }
        .timestamp { color: #57606a; font-size: 0.9em; margin-bottom: 1em; }
        .empty-state { padding: 20px; text-align: center; background: #f6f8fa; border: 1px dashed #d0d7de; border-radius: 6px; color: #57606a; }
        .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.85em; font-weight: 600; }
        .badge-success { background: #dafbe1; color: #1a7f37; }
        .badge-error { background: #ffebe9; color: #cf222e; }
        .branch-tag { background: #ddf4ff; color: #0969da; padding: 2px 6px; border-radius: 6px; font-family: ui-monospace, monospace; font-size: 0.85em; text-decoration: none; cursor: pointer; border: 1px solid transparent; transition: all 0.2s; position: relative; }
        .branch-tag:hover { background: #cfeeff; border-color: #0969da; }
        .branch-tag:focus { outline: 2px solid #0969da; outline-offset: 2px; }
        .branch-tag:active { background: #cfeeff; transform: translateY(1px); }
        .branch-tag.highlight { background: #0969da; color: white; border-color: #0969da; }
        .file-tag { cursor: pointer; transition: background-color 0.2s, transform 0.1s; position: relative; }
        .file-tag:hover { background: #afb8c166; }
        .file-tag:focus { outline: 2px solid #0969da; outline-offset: 2px; }
        .file-tag:active { background: #afb8c188; transform: translateY(1px); }
        .file-tag.highlight { background: #0969da; color: white; }
        tr.highlight { background-color: #ddf4ff; }
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
    </style>
</head>
<body>
    <a href="#main-content" class="skip-link">Skip to main content</a>
    <button class="refresh-btn" onclick="refresh()" aria-label="Refresh conflict predictions (Press 'r')">Refresh<kbd>r</kbd></button>
    <div class="timestamp">Last updated: <time datetime="{{ now.isoformat() }}">{{ now.strftime('%Y-%m-%d %H:%M:%S') }} UTC</time></div>
    <h1 id="main-content">SmartGitMergePT Dashboard</h1>

    <div class="filter-container" style="display: flex; align-items: center; gap: 8px;">
        <input type="text" id="filter-input" class="filter-input" placeholder="Filter branches or conflicts... (Press '/' to focus)" aria-label="Filter branches or conflicts">
        <button id="clear-filter" class="refresh-btn" style="margin-bottom: 0; padding: 4px 8px; display: none;" aria-label="Clear filter (Press 'Esc')">Clear<kbd>Esc</kbd></button>
        <span id="filter-results-count" style="font-size: 0.9em; color: #57606a;" aria-live="polite"></span>
    </div>
    <div id="announcer" class="sr-only" aria-live="polite" style="position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); border: 0;"></div>

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
    <div id="monitored-branches-list" style="margin-bottom: 2em;">
    {% for branch in branches %}
        <span class="branch-tag" role="button" tabindex="0" aria-label="Click to copy branch name: {{ branch }}" data-branch="{{ branch }}">{{ branch }}</span>
    {% endfor %}
    </div>

    <h2>Scenario Types</h2>
    <ul style="list-style: none; padding-left: 0;">
        <li class="{{ 'present' if scenario_types['file_overlap'] else 'absent' }}">
            {% if scenario_types['file_overlap'] %}<span role="img" aria-label="Warning">⚠️</span>{% else %}<span role="img" aria-label="Clear">✅</span>{% endif %}
            <strong>File Overlap</strong>: Both branches modify the same file(s).
        </li>
        <li class="{{ 'present' if scenario_types['line_overlap'] else 'absent' }}">
            {% if scenario_types['line_overlap'] %}<span role="img" aria-label="Warning">⚠️</span>{% else %}<span role="img" aria-label="Clear">✅</span>{% endif %}
            <strong>Line Overlap</strong>: Both branches change the same or similar lines in a file.
        </li>
        <li class="{{ 'present' if scenario_types['semantic_conflict'] else 'absent' }}">
            {% if scenario_types['semantic_conflict'] %}<span role="img" aria-label="Warning">⚠️</span>{% else %}<span role="img" aria-label="Clear">✅</span>{% endif %}
            <strong>Semantic Conflict</strong>: Changes are different but may cause logical or functional conflicts.
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
            <tr data-branch-a="{{ pred['branches'][0] }}" data-branch-b="{{ pred['branches'][1] }}">
                <td>
                    <span class="branch-tag" role="button" tabindex="0" aria-label="Click to copy branch name: {{ pred['branches'][0] }}" data-branch="{{ pred['branches'][0] }}">{{ pred['branches'][0] }}</span>
                    <span style="color: #57606a; margin: 0 4px;">↔</span>
                    <span class="branch-tag" role="button" tabindex="0" aria-label="Click to copy branch name: {{ pred['branches'][1] }}" data-branch="{{ pred['branches'][1] }}">{{ pred['branches'][1] }}</span>
                </td>
                <td>
                    {% if pred['files'] %}
                        {% for file in pred['files'] %}
                            <code class="file-tag" role="button" tabindex="0" aria-label="Click to copy file path: {{ file }}" data-copy="{{ file }}">{{ file }}</code>{% if not loop.last %}, {% endif %}
                        {% endfor %}
                    {% else %}
                        <span class="absent" aria-label="No files overlap">—</span>
                    {% endif %}
                </td>
                <td>
                    {% if pred['line_conflicts'] %}
                        <span class="badge badge-error" aria-label="Line overlap detected">⚠️ Yes</span>
                    {% else %}
                        <span class="absent" aria-label="No line overlap detected">—</span>
                    {% endif %}
                </td>
                <td>
                    {% if pred['semantic_conflict'] %}
                        <span class="badge badge-error" aria-label="Semantic conflict detected">⚠️ Yes</span>
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

    <script>
        const refreshBtn = document.querySelector('.refresh-btn');
        function refresh() {
            refreshBtn.textContent = 'Refreshing...';
            refreshBtn.disabled = true;
            window.location.reload();
        }

        function copyToClipboard(element, attr = 'data-branch') {
            const text = element.getAttribute(attr);
            if (element.querySelector('.copy-tooltip')) return;
            navigator.clipboard.writeText(text).then(() => {
                const tooltip = document.createElement('div');
                tooltip.className = 'copy-tooltip show';
                tooltip.textContent = 'Copied!';
                element.appendChild(tooltip);

                const announcer = document.getElementById('announcer');
                if (announcer) {
                    announcer.textContent = `Copied ${text} to clipboard`;
                }

                setTimeout(() => {
                    tooltip.remove();
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
            });
        }

        function toggleFileHighlight(file, active) {
            document.querySelectorAll(`.file-tag[data-copy="${file}"]`).forEach(tag => {
                tag.classList.toggle('highlight', active);
            });
        }

        document.querySelectorAll('.branch-tag, .file-tag').forEach(tag => {
            const isFile = tag.classList.contains('file-tag');
            const attr = isFile ? 'data-copy' : 'data-branch';
            tag.addEventListener('click', () => copyToClipboard(tag, attr));

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
                }
            });
        });

        const filterInput = document.getElementById('filter-input');
        const clearFilterBtn = document.getElementById('clear-filter');
        const resultsCount = document.getElementById('filter-results-count');
        const monitoredBranchTags = document.querySelectorAll('#monitored-branches-list .branch-tag');
        const tableRows = document.querySelectorAll('tbody tr');
        const noResults = document.createElement('div');
        noResults.className = 'no-results';
        noResults.textContent = 'No matching branches or conflicts found.';

        const table = document.querySelector('table');
        if (table) {
            table.parentNode.insertBefore(noResults, table.nextSibling);
        }

        filterInput.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase();
            let visibleRowsCount = 0;
            clearFilterBtn.style.display = query ? 'inline-block' : 'none';

            monitoredBranchTags.forEach(tag => {
                const text = tag.getAttribute('data-branch').toLowerCase();
                tag.style.display = text.includes(query) ? 'inline-block' : 'none';
            });

            tableRows.forEach(row => {
                const text = row.textContent.toLowerCase();
                const isVisible = text.includes(query);
                row.style.display = isVisible ? '' : 'none';
                if (isVisible) visibleRowsCount++;
            });

            if (query === '') {
                resultsCount.textContent = '';
            } else {
                resultsCount.textContent = `Showing ${visibleRowsCount} matching conflict${visibleRowsCount !== 1 ? 's' : ''}`;
            }

            if (table) {
                table.style.display = visibleRowsCount > 0 ? '' : 'none';
                noResults.style.display = (visibleRowsCount === 0 && query !== '') ? 'block' : 'none';
            }
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
            };
            row.addEventListener('mouseenter', () => highlightRow(true));
            row.addEventListener('mouseleave', () => highlightRow(false));
            row.addEventListener('focusin', () => highlightRow(true));
            row.addEventListener('focusout', () => highlightRow(false));
        });

        document.addEventListener('keydown', (e) => {
            if (e.key.toLowerCase() === 'r' && !e.ctrlKey && !e.metaKey && !e.altKey && document.activeElement !== filterInput) {
                refresh();
            }
            if (e.key === '/' && document.activeElement !== filterInput) {
                e.preventDefault();
                filterInput.focus();
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
    repo_path = "demo/conflict_scenarios/demo-repo"
    git_utils = GitUtils(repo_path)
    predictor = ConflictPredictor(repo_path)
    branches = git_utils.list_branches()
    predictions = predictor.predict_conflicts(branches)
    # Determine which scenario types are present
    scenario_types = {
        'file_overlap': any(pred['files'] for pred in predictions),
        'line_overlap': any(pred['line_conflicts'] for pred in predictions),
        'semantic_conflict': any(pred['semantic_conflict'] for pred in predictions)
    }
    return render_template_string(
        template,
        branches=branches,
        predictions=predictions,
        scenario_types=scenario_types,
        now=datetime.now(timezone.utc)
    )

@app.after_request
def add_security_headers(response):
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Content-Security-Policy'] = "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'"
    return response

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SmartGitMergePT Dashboard')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the dashboard on')
    args = parser.parse_args()
    app.run(debug=False, port=args.port)
