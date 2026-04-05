import os
import shutil
import git
import time
import collections
from src.predictor import ConflictPredictor
from src.git_utils import GitUtils
from pathlib import Path
import hashlib

def setup_benchmark_repo(tmp_path, num_branches=50):
    repo_dir = tmp_path / "bolt_bench_repo"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.mkdir(parents=True)
    repo = git.Repo.init(str(repo_dir))

    # Configure user for commits
    with repo.config_writer() as config:
        config.set_value("user", "name", "Bolt")
        config.set_value("user", "email", "bolt@example.com")

    # Initial commit
    file_path = repo_dir / "common.txt"
    file_path.write_text("common base\n" * 100)
    repo.index.add(["common.txt"])
    repo.index.commit("initial")

    default_branch = 'master'
    try:
        default_branch = repo.active_branch.name
    except:
        pass

    # Create branches with significant file overlap but no line conflict
    branches = []
    # 5 shared files that most branches will modify
    shared_files = [f"shared_{i}.txt" for i in range(5)]
    for sf in shared_files:
        (repo_dir / sf).write_text("initial content\n")
        repo.index.add([sf])
    repo.index.commit("setup shared files")

    for i in range(num_branches):
        branch_name = f"branch_{i}"
        repo.git.checkout(default_branch)
        repo.git.checkout('-b', branch_name)

        # Each branch modifies 2 shared files at different lines
        # This ensures high 'overlap' count but zero 'line_conflicts'
        sf_idx1 = i % 5
        sf_idx2 = (i + 1) % 5

        for sf in [shared_files[sf_idx1], shared_files[sf_idx2]]:
            with open(repo_dir / sf, "a") as f:
                f.write(f"branch {i} unique line\n")
            repo.index.add([sf])

        # Also modify a unique file
        unique_f = f"unique_{i}.txt"
        (repo_dir / unique_f).write_text(f"unique content {i}\n")
        repo.index.add([unique_f])

        repo.index.commit(f"branch {i} work")
        branches.append(branch_name)

    return str(repo_dir), branches

def main():
    tmp_path = Path("/tmp/bolt_bench_intensive")
    tmp_path.mkdir(exist_ok=True)

    num_branches = 50
    repo_path, branches = setup_benchmark_repo(tmp_path, num_branches=num_branches)

    # Warm up Git and caches
    predictor = ConflictPredictor(repo_path)
    predictor.predict_conflicts(branches[:5])

    print(f"--- BOLT INTENSIVE BENCHMARK ({num_branches} branches) ---")

    # Measure predict_conflicts
    start_time = time.perf_counter()
    predictions = predictor.predict_conflicts(branches)
    end_time = time.perf_counter()

    duration = end_time - start_time
    print(f"predict_conflicts total time: {duration:.4f} seconds")
    print(f"Number of branch pairs with overlap: {len(predictions)}")

    # Measure memory usage if possible
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        print(f"Process memory usage: {mem_mb:.2f} MB")
    except ImportError:
        pass

    # Cleanup
    shutil.rmtree(repo_path)

if __name__ == "__main__":
    main()
