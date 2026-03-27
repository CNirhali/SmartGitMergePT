import os
import shutil
import git
import time
import sys
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from git_utils import GitUtils
from concurrent.futures import ThreadPoolExecutor

def setup_benchmark_repo(tmp_path, num_branches=10):
    repo_dir = tmp_path / "detect_benchmark_repo"
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

        # Modify a file
        f = repo_dir / f"file_{i}.txt"
        f.write_text(f"content {i}\n")

        # Occasionally modify the same file to ensure some overlap/conflict
        if i % 5 == 0:
            common = repo_dir / "shared.txt"
            with open(common, "a") as cf:
                cf.write(f"branch {i} change\n")
            repo.index.add([str(common)])

        repo.index.add([str(f)])
        repo.index.commit(f"commit {i}")
        branches.append(branch_name)

    return str(repo_dir), branches

def benchmark_sequential(git_utils, branches):
    print(f"Benchmarking sequential detection with {len(branches)} branches...")
    start_time = time.perf_counter()
    count = 0
    for i, branch_a in enumerate(branches):
        for branch_b in branches[i+1:]:
            git_utils.simulate_merge(branch_a, branch_b)
            count += 1
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"Sequential took: {duration:.4f} seconds for {count} pairs")
    return duration

def benchmark_parallel(git_utils, branches):
    print(f"Benchmarking parallel detection with {len(branches)} branches...")
    start_time = time.perf_counter()
    pairs = []
    for i, branch_a in enumerate(branches):
        for branch_b in branches[i+1:]:
            pairs.append((branch_a, branch_b))

    def check_pair(pair):
        branch_a, branch_b = pair
        return git_utils.simulate_merge(branch_a, branch_b)

    with ThreadPoolExecutor() as executor:
        list(executor.map(check_pair, pairs))

    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"Parallel took: {duration:.4f} seconds for {len(pairs)} pairs")
    return duration

import tempfile

def main():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        num_branches = 15 # results in 105 pairs
        repo_path, branches = setup_benchmark_repo(tmp_path, num_branches=num_branches)
        git_utils = GitUtils(repo_path)

        seq_dur = benchmark_sequential(git_utils, branches)
        par_dur = benchmark_parallel(git_utils, branches)

        speedup = (seq_dur / par_dur) if par_dur > 0 else 0
        print(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    main()
