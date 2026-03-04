import os
import shutil
import git
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from predictor import ConflictPredictor

def setup_debug_repo(tmp_path):
    repo_dir = tmp_path / "debug_repo"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.mkdir()
    repo = git.Repo.init(str(repo_dir))

    # Initial commit
    file_path = repo_dir / "common.txt"
    file_path.write_text("common base\n")
    repo.index.add([str(file_path)])
    repo.index.commit("initial")

    # Create 3 branches
    branches = []
    for i in range(3):
        branch_name = f"branch_{i}"
        repo.git.checkout('master' if 'master' in repo.heads else repo.active_branch.name)
        repo.git.checkout('-b', branch_name)
        f = repo_dir / f"file_{i}.txt"
        f.write_text(f"content {i}\n")
        repo.index.add([str(f)])
        repo.index.commit(f"commit {i}")
        branches.append(branch_name)

    # Create a fourth branch that overlaps with branch_0
    repo.git.checkout('master' if 'master' in repo.heads else repo.active_branch.name)
    repo.git.checkout('-b', 'branch_overlap')
    f = repo_dir / "file_0.txt"
    f.write_text("overlapping content\n")
    repo.index.add([str(f)])
    repo.index.commit("commit overlap")
    branches.append('branch_overlap')

    return str(repo_dir), ['master'] + branches

def main():
    tmp_path = Path("/tmp/bolt_debug")
    tmp_path.mkdir(exist_ok=True)
    repo_path, branches = setup_debug_repo(tmp_path)

    predictor = ConflictPredictor(repo_path)
    # Mocking active branch to be a feature branch
    git_repo = git.Repo(repo_path)
    git_repo.git.checkout('branch_2')

    predictions = predictor.predict_conflicts(branches)

    print(f"Active branch: {git_repo.active_branch.name}")
    print(f"Branches: {branches}")
    print(f"Number of predictions: {len(predictions)}")
    for p in predictions:
        print(f"Conflict between {p['branches']}: files={p['files']}")

    # Cleanup
    shutil.rmtree(repo_path)

if __name__ == "__main__":
    main()
