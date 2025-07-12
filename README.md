# SmartGitMergePT

## Overview
SmartGitMergePT is an LLM-powered tool that predicts, detects, and resolves Git merge conflicts for teams with many contributors. It scans codebases, predicts potential conflicts before they occur, and uses a large language model to suggest or auto-resolve conflicts, improving collaboration and reducing merge pain.

## Features
- **Conflict Prediction:** Scans branches and PRs to predict likely merge conflicts early.
- **Conflict Detection:** Detects actual conflicts during merges.
- **LLM-based Resolution:** Uses an LLM to suggest or auto-resolve merge conflicts.
- **Team Dashboard/CLI:** Shows predicted and actual conflicts for all contributors.
- **Demo & Test Suite:** Includes scripts to simulate repo conflicts and automated tests.

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Run the demo: `bash demo/repo_simulation.sh`
3. Use the CLI: `python src/main.py --help`

## Demo
- The `demo/` folder contains scripts and scenarios to simulate real-world repo conflicts with multiple contributors.

## Testing
- Run tests with: `pytest src/tests/`

## License
MIT