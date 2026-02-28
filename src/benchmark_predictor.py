import os
import shutil
import git
import time
from src.predictor import ConflictPredictor
from src.git_utils import GitUtils
from pathlib import Path

def setup_benchmark_repo(tmp_path, num_branches=10):
    repo_dir = tmp_path / "benchmark_repo"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.mkdir()
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

def main():
    tmp_path = Path("/tmp/bolt_bench")
    tmp_path.mkdir(exist_ok=True)
    repo_path, branches = setup_benchmark_repo(tmp_path, num_branches=100)

    predictor = ConflictPredictor(repo_path)

    print(f"Benchmarking with {len(branches)} branches...")

    start_time = time.perf_counter()
    predictions = predictor.predict_conflicts(branches)
    end_time = time.perf_counter()

    duration = end_time - start_time
    print(f"Time taken: {duration:.4f} seconds")
    print(f"Number of predictions: {len(predictions)}")

    # Cleanup
    shutil.rmtree(repo_path)

if __name__ == "__main__":
    main()
