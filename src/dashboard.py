from flask import Flask, render_template_string
from git_utils import GitUtils
from predictor import ConflictPredictor
import argparse

app = Flask(__name__)

template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SmartGitMergePT Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 2em; }
        table { border-collapse: collapse; width: 80%; }
        th, td { border: 1px solid #ccc; padding: 8px; }
        th { background: #eee; }
        tr:hover { background-color: #f5f5f5; }
        .present { color: green; font-weight: bold; }
        .absent { color: #767676; }
        .refresh-btn { margin-bottom: 1em; padding: 5px 10px; cursor: pointer; }
    </style>
</head>
<body>
    <button class="refresh-btn" onclick="window.location.reload()" aria-label="Refresh conflict predictions">Refresh</button>
    <h1>SmartGitMergePT Dashboard</h1>
    <h2>Branches</h2>
    <ul>
    {% for branch in branches %}
        <li>{{ branch }}</li>
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
            <strong>Semantic Conflict</strong>: Changes are different but may cause logical or functional conflicts (e.g., renaming a function in one branch and using the old name in another).
        </li>
    </ul>
    <h2>Predicted Conflicts</h2>
    <table aria-label="Predicted merge conflicts">
        <tr><th>Branches</th><th>Files</th><th>Line Overlap</th><th>Semantic Conflict</th></tr>
        {% for pred in predictions %}
        <tr>
            <td>{{ pred['branches'] }}</td>
            <td>{{ pred['files'] }}</td>
            <td>{{ pred['line_conflicts'] }}</td>
            <td>{{ pred['semantic_conflict'] }}</td>
        </tr>
        {% endfor %}
    </table>
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
    return render_template_string(template, branches=branches, predictions=predictions, scenario_types=scenario_types)

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
