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
        .absent { color: #6e7781; }
        .refresh-btn { margin-bottom: 1em; padding: 6px 12px; cursor: pointer; background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px; font-weight: 600; }
        .refresh-btn:hover { background: #f3f4f6; }
        code { background: #afb8c133; padding: 0.2em 0.4em; border-radius: 6px; font-size: 85%; }
        .timestamp { color: #6e7781; font-size: 0.9em; margin-bottom: 2em; }
        .empty-state { padding: 20px; text-align: center; background: #f6f8fa; border: 1px dashed #d0d7de; border-radius: 6px; color: #6e7781; }
        .conflict-yes { color: #cf222e; font-weight: 600; }
        .conflict-no { color: #6e7781; }
    </style>
</head>
<body>
    <button class="refresh-btn" onclick="window.location.reload()" aria-label="Refresh conflict predictions">Refresh</button>
    <div class="timestamp">Last updated: {{ now.strftime('%Y-%m-%d %H:%M:%S') }}</div>
    <h1>SmartGitMergePT Dashboard</h1>

    <h2>Branches</h2>
    <ul>
    {% for branch in branches %}
        <li><code>{{ branch }}</code></li>
    {% endfor %}
    </ul>

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
            <tr><th>Branches</th><th>Files</th><th>Line Overlap</th><th>Semantic Conflict</th></tr>
        </thead>
        <tbody>
            {% for pred in predictions %}
            <tr>
                <td><code>{{ pred['branches'][0] }}</code> ↔ <code>{{ pred['branches'][1] }}</code></td>
                <td>
                    {% if pred['files'] %}
                        {% for file in pred['files'] %}
                            <code>{{ file }}</code>{% if not loop.last %}, {% endif %}
                        {% endfor %}
                    {% else %}
                        —
                    {% endif %}
                </td>
                <td class="{{ 'conflict-yes' if pred['line_conflicts'] else 'conflict-no' }}">
                    {{ '⚠️ Yes' if pred['line_conflicts'] else '—' }}
                </td>
                <td class="{{ 'conflict-yes' if pred['semantic_conflict'] else 'conflict-no' }}">
                    {{ '⚠️ Yes' if pred['semantic_conflict'] else '—' }}
                </td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    {% else %}
    <div class="empty-state">No likely conflicts detected between branches. Everything looks clear! ✨</div>
    {% endif %}
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
