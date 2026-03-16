import os
import shutil
import git
import time
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from predictor import ConflictPredictor
from git_utils import GitUtils

def setup_large_diff_repo(tmp_path, num_branches=5, lines_per_file=500):
    # Resolve to absolute path to avoid confusion for GitPython
    repo_dir = Path(tmp_path).resolve() / "large_diff_repo"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.mkdir(parents=True, exist_ok=True)
    repo = git.Repo.init(str(repo_dir))

    # Initial commit with a large file
    file_path = repo_dir / "large_file.txt"
    content = "\n".join([f"Line {i}" for i in range(lines_per_file)])
    file_path.write_text(content + "\n")

    # Add using relative path from repo root or absolute path
    repo.index.add(["large_file.txt"])
    repo.index.commit("initial")

    branches = []
    for i in range(num_branches):
        branch_name = f"branch_{i}"
        # Ensure we start from 'master' or 'main'
        main_name = 'master' if 'master' in [h.name for h in repo.heads] else 'main'
        if main_name not in [h.name for h in repo.heads]:
            # If neither exists, we're likely on the first branch created by init
            main_name = repo.active_branch.name

        repo.git.checkout(main_name)
        repo.git.checkout('-b', branch_name)

        # Modify one line in the middle
        lines = file_path.read_text().splitlines()
        lines[lines_per_file // 2 + i] = f"Modified by {branch_name}"
        file_path.write_text("\n".join(lines) + "\n")

        repo.index.add(["large_file.txt"])
        repo.index.commit(f"commit {i}")
        branches.append(branch_name)

    return str(repo_dir), branches

def run_benchmark(repo_path, branches):
    git_utils = GitUtils(repo_path)
    predictor = ConflictPredictor(repo_path)

    # Disable cache to measure raw performance
    predictor.cache.clear()

    start_time = time.perf_counter()
    # Force determination of main branch so it doesn't vary
    try:
        main_branch = 'master' if 'master' in git_utils.list_branches() else 'main'
    except:
        main_branch = branches[0]

    predictor.predict_conflicts([main_branch] + branches)
    end_time = time.perf_counter()

    return end_time - start_time

def main():
    # Use a local path to avoid potential permission issues in /tmp
    tmp_path = Path("./tmp_bolt_bench").resolve()
    tmp_path.mkdir(exist_ok=True, parents=True)

    num_branches = 5
    lines_per_file = 500
    repo_path, branches = setup_large_diff_repo(tmp_path, num_branches=num_branches, lines_per_file=lines_per_file)

    print(f"Benchmarking with {num_branches} branches and {lines_per_file} lines per file...")

    # Warm up
    run_benchmark(repo_path, branches)

    times = []
    for _ in range(5):
        t = run_benchmark(repo_path, branches)
        times.append(t)
        print(f"Run: {t:.4f}s")

    avg_time = sum(times) / len(times)
    print(f"Average time: {avg_time:.4f}s")

    # Cleanup
    shutil.rmtree(repo_path)

if __name__ == "__main__":
    main()
