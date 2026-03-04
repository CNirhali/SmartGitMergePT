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

def setup_large_diff_repo(tmp_path, num_branches=10, lines_per_file=1000):
    repo_dir = tmp_path / "large_diff_repo"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.mkdir()
    repo = git.Repo.init(str(repo_dir))

    # Initial commit with a large file
    file_path = repo_dir / "large_file.txt"
    content = "\n".join([f"Line {i}" for i in range(lines_per_file)])
    file_path.write_text(content + "\n")
    repo.index.add([str(file_path)])
    repo.index.commit("initial")

    branches = []
    for i in range(num_branches):
        branch_name = f"branch_{i}"
        repo.git.checkout('master' if 'master' in repo.heads else repo.active_branch.name)
        repo.git.checkout('-b', branch_name)

        # Modify one line in the middle
        lines = file_path.read_text().splitlines()
        lines[lines_per_file // 2 + i] = f"Modified by {branch_name}"
        file_path.write_text("\n".join(lines) + "\n")

        repo.index.add([str(file_path)])
        repo.index.commit(f"commit {i}")
        branches.append(branch_name)

    return str(repo_dir), branches

def run_benchmark(repo_path, branches, unified_value=3):
    git_utils = GitUtils(repo_path)
    predictor = ConflictPredictor(repo_path)

    # Monkeypatch git_utils to use specific unified value
    original_get_diff = git_utils.get_diff_between_branches
    def patched_get_diff(a, b, unified=None):
        u = unified if unified is not None else unified_value
        return git_utils.repo.git.diff(f'{a}..{b}', unified=u)

    # We need to reach into predictor and change its git_utils
    predictor.git_utils = git_utils
    git_utils.get_diff_between_branches = patched_get_diff

    start_time = time.perf_counter()
    # Force determination of main branch so it doesn't vary
    try:
        main_branch = 'master' if 'master' in git_utils.list_branches() else 'main'
    except:
        main_branch = branches[0]

    predictions = predictor.predict_conflicts([main_branch] + branches)
    end_time = time.perf_counter()

    return end_time - start_time, len(predictions)

def main():
    tmp_path = Path("/tmp/bolt_unified_bench")
    tmp_path.mkdir(exist_ok=True)

    num_branches = 20
    lines_per_file = 5000
    repo_path, branches = setup_large_diff_repo(tmp_path, num_branches=num_branches, lines_per_file=lines_per_file)

    print(f"Benchmarking with {num_branches} branches and {lines_per_file} lines per file...")

    # Warm up
    run_benchmark(repo_path, branches, unified_value=3)

    time_u3, pred_count = run_benchmark(repo_path, branches, unified_value=3)
    print(f"Unified=3 (default): {time_u3:.4f} seconds")

    time_u0, _ = run_benchmark(repo_path, branches, unified_value=0)
    print(f"Unified=0:           {time_u0:.4f} seconds")

    improvement = (time_u3 - time_u0) / time_u3 * 100
    print(f"Improvement: {improvement:.2f}%")

    # Cleanup
    shutil.rmtree(repo_path)

if __name__ == "__main__":
    main()
