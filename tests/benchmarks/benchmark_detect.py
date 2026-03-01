import os
import shutil
import git
import time
from src.git_utils import GitUtils
from pathlib import Path

def setup_benchmark_repo(tmp_path, num_branches=10):
    repo_dir = tmp_path / "benchmark_detect_repo"
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
    tmp_path = Path("/tmp/bolt_bench_detect")
    tmp_path.mkdir(exist_ok=True, parents=True)
    # 20 branches means ~190 merges.
    num_branches = 20
    repo_path, branches = setup_benchmark_repo(tmp_path, num_branches=num_branches)

    git_utils = GitUtils(repo_path)

    print(f"Benchmarking detect with {len(branches)} branches (~{len(branches)*(len(branches)-1)//2} merges)...")

    start_time = time.perf_counter()
    results = []
    for i, branch_a in enumerate(branches):
        for branch_b in branches[i+1:]:
            ok, msg = git_utils.simulate_merge(branch_a, branch_b)
            results.append((branch_a, branch_b, ok))
    end_time = time.perf_counter()

    duration = end_time - start_time
    print(f"Time taken: {duration:.4f} seconds")
    print(f"Number of merges simulated: {len(results)}")

    # Cleanup
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)

if __name__ == "__main__":
    main()
