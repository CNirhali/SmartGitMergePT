import os
import shutil
import git
import time
from src.predictor import ConflictPredictor
from src.git_utils import GitUtils
from pathlib import Path

def setup_benchmark_repo(tmp_path, num_branches=5, files_per_branch=500):
    repo_dir = tmp_path / "benchmark_mini_repo"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.mkdir(parents=True)

    repo_dir_abs = str(repo_dir.absolute())
    repo = git.Repo.init(repo_dir_abs)

    file_path = repo_dir / "common.txt"
    file_path.write_text("common base\n")
    repo.index.add([str(file_path.absolute())])
    repo.index.commit("initial")

    try:
        repo.git.branch('-M', 'main')
        default_branch = 'main'
    except Exception:
        default_branch = 'master'

    branches = []
    for i in range(num_branches):
        branch_name = f"branch_{i}"
        repo.git.checkout(default_branch)
        repo.git.checkout('-b', branch_name)

        for j in range(files_per_branch):
            f = repo_dir / f"branch_{i}_file_{j}.txt"
            f.write_text(f"content {i} {j}\n" * 10)
            repo.index.add([str(f.absolute())])

        shared = repo_dir / "shared_file.txt"
        with open(shared, "a") as sf:
            sf.write(f"branch {i} change\n")
        repo.index.add([str(shared.absolute())])

        repo.index.commit(f"commit {i}")
        branches.append(branch_name)

    return repo_dir_abs, branches

def main():
    tmp_path = Path("/tmp/bolt_bench_mini")
    repo_path, branches = setup_benchmark_repo(tmp_path)

    predictor = ConflictPredictor(repo_path)

    print(f"Benchmarking with {len(branches)} branches, each modifying 500 unique files + 1 shared file...")

    start_time = time.perf_counter()
    predictions = predictor.predict_conflicts(branches)
    end_time = time.perf_counter()

    duration = end_time - start_time
    print(f"Time taken: {duration:.4f} seconds")
    print(f"Number of predictions: {len(predictions)}")

    shutil.rmtree(repo_path)

if __name__ == "__main__":
    main()
