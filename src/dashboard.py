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
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; margin: 2em; line-height: 1.5; color: #24292f; }
        table { border-collapse: collapse; width: 100%; max-width: 900px; margin-top: 1em; }
        th, td { border: 1px solid #d0d7de; padding: 12px; text-align: left; }
        th { background: #f6f8fa; font-weight: 600; }
        tr:hover { background-color: #f6f8fa; }
        .present { color: #1a7f37; font-weight: bold; }
        .absent { color: #57606a; }
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
        .branch-tag { background: #ddf4ff; color: #0969da; padding: 2px 6px; border-radius: 6px; font-family: ui-monospace, monospace; font-size: 0.85em; text-decoration: none; }
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
    </style>
</head>
<body>
    <a href="#main-content" class="skip-link">Skip to main content</a>
    <button class="refresh-btn" onclick="refresh()" aria-label="Refresh conflict predictions (Press 'r')">Refresh<kbd>r</kbd></button>
    <div class="timestamp">Last updated: {{ now.strftime('%Y-%m-%d %H:%M:%S') }}</div>
    <h1 id="main-content">SmartGitMergePT Dashboard</h1>

    <div class="summary">
        <div class="summary-item">
            <strong>Branches</strong>: {{ branches|length }}
        </div>
        <div class="summary-item">
            <strong>Conflict Pairs</strong>:
            <span class="badge {{ 'badge-error' if predictions else 'badge-success' }}">
                {{ predictions|length }}
            </span>
        </div>
    </div>

    <h2>Monitored Branches</h2>
    <div style="margin-bottom: 2em;">
    {% for branch in branches %}
        <span class="branch-tag">{{ branch }}</span>
    {% endfor %}
    </div>

    <h2>Scenario Types</h2>
    <ul>
        <li class="{{ 'present' if scenario_types['file_overlap'] else 'absent' }}">
            <strong>File Overlap</strong>: Both branches modify the same file(s).
        </li>
        <li class="{{ 'present' if scenario_types['line_overlap'] else 'absent' }}">
            <strong>Line Overlap</strong>: Both branches change the same or similar lines in a file.
        </li>
        <li class="{{ 'present' if scenario_types['semantic_conflict'] else 'absent' }}">
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
            <tr>
                <td>
                    <span class="branch-tag">{{ pred['branches'][0] }}</span>
                    <span style="color: #57606a; margin: 0 4px;">↔</span>
                    <span class="branch-tag">{{ pred['branches'][1] }}</span>
                </td>
                <td>
                    {% if pred['files'] %}
                        {% for file in pred['files'] %}
                            <code>{{ file }}</code>{% if not loop.last %}, {% endif %}
                        {% endfor %}
                    {% else %}
                        <span class="absent">—</span>
                    {% endif %}
                </td>
                <td>
                    {% if pred['line_conflicts'] %}
                        <span class="badge badge-error">⚠️ Yes</span>
                    {% else %}
                        <span class="absent">—</span>
                    {% endif %}
                </td>
                <td>
                    {% if pred['semantic_conflict'] %}
                        <span class="badge badge-error">⚠️ Yes</span>
                    {% else %}
                        <span class="absent">—</span>
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
        document.addEventListener('keydown', (e) => {
            if (e.key.toLowerCase() === 'r' && !e.ctrlKey && !e.metaKey && !e.altKey) {
                refresh();
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
