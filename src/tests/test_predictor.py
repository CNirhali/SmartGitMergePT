import os
import shutil
import git
import pytest
from src.predictor import ConflictPredictor
from src.git_utils import GitUtils

def setup_demo_repo(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    repo = git.Repo.init(str(repo_dir))
    file_path = repo_dir / "file.txt"
    file_path.write_text("base\n")
    repo.index.add([str(file_path)])
    repo.index.commit("init")
    # Ensure current branch is 'main'
    try:
        repo.git.branch('-M', 'main')
    except:
        pass
    repo.git.checkout('-b', 'feature/a')
    file_path.write_text("base\na\n")
    repo.index.add([str(file_path)])
    repo.index.commit("a change")

    # Use the active branch (likely master or main)
    try:
        repo.git.checkout('main')
    except:
        repo.git.checkout('master')

    repo.git.checkout('-b', 'feature/b')
    file_path.write_text("base\nb\n")
    repo.index.add([str(file_path)])
    repo.index.commit("b change")
    return str(repo_dir)

def test_predictor_detects_conflict(tmp_path):
    repo_path = setup_demo_repo(tmp_path)
    predictor = ConflictPredictor(repo_path)
    git_utils = GitUtils(repo_path)
    branches = git_utils.list_branches()
    predictions = predictor.predict_conflicts(branches)
    assert any(p['conflict_likely'] for p in predictions)
