import argparse
from git_utils import GitUtils
from predictor import ConflictPredictor
from llm_resolver import resolve_conflict_with_mistral

# Placeholder imports for future modules
# from .predictor import predict_conflicts
# from .llm_resolver import resolve_conflicts
# from .dashboard import show_dashboard

def main():
    parser = argparse.ArgumentParser(description='SmartGitMergePT: LLM-based Git merge conflict resolver and predictor')
    subparsers = parser.add_subparsers(dest='command')

    subparsers.add_parser('predict', help='Predict potential merge conflicts')
    subparsers.add_parser('detect', help='Detect actual merge conflicts')
    subparsers.add_parser('resolve', help='Resolve merge conflicts using LLM')
    subparsers.add_parser('dashboard', help='Show team conflict dashboard')

    args = parser.parse_args()
    repo_path = "demo/conflict_scenarios/demo-repo"
    git_utils = GitUtils(repo_path)
    predictor = ConflictPredictor(repo_path)

    if args.command == 'predict':
        branches = git_utils.list_branches()
        predictions = predictor.predict_conflicts(branches)
        if not predictions:
            print("No likely conflicts detected between branches.")
        else:
            print("Predicted conflicts:")
            for pred in predictions:
                print(f"Branches: {pred['branches']}, Files: {pred['files']}")

    elif args.command == 'detect':
        branches = git_utils.list_branches()
        for i, branch_a in enumerate(branches):
            for branch_b in branches[i+1:]:
                ok, msg = git_utils.simulate_merge(branch_a, branch_b)
                if not ok:
                    print(f"Conflict detected merging {branch_a} into {branch_b}: {msg}")
                else:
                    print(f"No conflict merging {branch_a} into {branch_b}.")

    elif args.command == 'resolve':
        print("Paste the merge conflict block (end with EOF / Ctrl-D):")
        import sys
        conflict_block = sys.stdin.read()
        resolved = resolve_conflict_with_mistral(conflict_block)
        print("\nResolved block:\n")
        print(resolved)

    elif args.command == 'dashboard':
        branches = git_utils.list_branches()
        predictions = predictor.predict_conflicts(branches)
        print("=== Team Conflict Dashboard ===")
        print(f"Branches: {branches}")
        if not predictions:
            print("No likely conflicts detected.")
        else:
            print("Predicted conflicts:")
            for pred in predictions:
                print(f"Branches: {pred['branches']}, Files: {pred['files']}")
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 