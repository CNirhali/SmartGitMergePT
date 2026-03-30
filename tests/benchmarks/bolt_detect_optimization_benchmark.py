import os
import shutil
import git
import time
import sys
from pathlib import Path
import tempfile
from concurrent.futures import ThreadPoolExecutor

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from git_utils import GitUtils
from predictor import ConflictPredictor

def setup_benchmark_repo(tmp_path, num_branches=50):
    repo_dir = Path(tmp_path) / "detect_opt_repo"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.mkdir(parents=True)
    repo = git.Repo.init(str(repo_dir))

    # Initial commit
    file_path = repo_dir / "common.txt"
    file_path.write_text("common base\n")
    repo.index.add([str(file_path)])
    repo.index.commit("initial")

    # Determine the default branch name
    try:
        default_branch = repo.active_branch.name
    except Exception:
        default_branch = 'master'

    # Create branches
    branches = []
    for i in range(num_branches):
        branch_name = f"branch_{i}"
        repo.git.checkout(default_branch)
        repo.git.checkout('-b', branch_name)

        # Modify a unique file for each branch to avoid conflicts by default
        f = repo_dir / f"file_{i}.txt"
        f.write_text(f"content {i}\n")
        repo.index.add([str(f)])

        # Add one shared conflict between branch_0 and branch_1
        if i == 0:
            shared = repo_dir / "shared.txt"
            shared.write_text("branch 0 content\n")
            repo.index.add([str(shared)])
        elif i == 1:
            shared = repo_dir / "shared.txt"
            shared.write_text("branch 1 content\n")
            repo.index.add([str(shared)])

        repo.index.commit(f"commit {i}")
        branches.append(branch_name)

    return str(repo_dir), branches

def run_detect_logic_v1(git_utils, branches):
    """Current O(N^2) implementation logic from main.py"""
    pairs = []
    for i, branch_a in enumerate(branches):
        for branch_b in branches[i+1:]:
            pairs.append((branch_a, branch_b))

    def check_pair(pair):
        branch_a, branch_b = pair
        ok, msg = git_utils.simulate_merge(branch_a, branch_b)
        return (branch_a, branch_b, ok, msg)

    results = []
    with ThreadPoolExecutor() as executor:
        for res in executor.map(check_pair, pairs):
            results.append(res)
    return results

def run_detect_logic_v2(git_utils, predictor, branches):
    """Optimized implementation using ConflictPredictor to pre-filter"""
    # 1. Get predictions to find pairs with file overlap
    predictions = predictor.predict_conflicts(branches)

    # Map predictions to sets of pairs for O(1) lookup or just use them as the pair list
    candidate_pairs = [p['branches'] for p in predictions]

    def check_pair(pair):
        branch_a, branch_b = pair
        ok, msg = git_utils.simulate_merge(branch_a, branch_b)
        return (branch_a, branch_b, ok, msg)

    results = []
    # We still want to show "No conflict" for non-overlapping pairs if we want to maintain EXACT same output,
    # BUT the task is optimization. If we only run merge-tree for candidates, we skip the rest.
    # To maintain same output behavior, we'd need to know the full set of pairs.

    # For benchmarking the CORE logic speedup:
    with ThreadPoolExecutor() as executor:
        for res in executor.map(check_pair, candidate_pairs):
            results.append(res)

    return results

def main():
    num_branches = 50
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_path, branches = setup_benchmark_repo(tmp_dir, num_branches=num_branches)
        git_utils = GitUtils(repo_path)
        predictor = ConflictPredictor(repo_path)

        print(f"Benchmarking 'detect' with {num_branches} branches ({num_branches*(num_branches-1)//2} pairs)...")

        start = time.perf_counter()
        results_v1 = run_detect_logic_v1(git_utils, branches)
        dur_v1 = time.perf_counter() - start
        print(f"V1 (Sequential O(N^2) calls): {dur_v1:.4f}s")

        start = time.perf_counter()
        results_v2 = run_detect_logic_v2(git_utils, predictor, branches)
        dur_v2 = time.perf_counter() - start
        print(f"V2 (Filtered candidates): {dur_v2:.4f}s")

        print(f"Speedup: {dur_v1/dur_v2:.2f}x")

        # Verify correctness: branch_0 and branch_1 should be in both
        conflicts_v1 = [r for r in results_v1 if not r[2]]
        conflicts_v2 = [r for r in results_v2 if not r[2]]

        print(f"Conflicts found V1: {len(conflicts_v1)}")
        print(f"Conflicts found V2: {len(conflicts_v2)}")

if __name__ == "__main__":
    main()
