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

    default_branch = repo.active_branch.name
    repo.git.checkout(default_branch)
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

def test_predictor_no_false_positive_different_files(tmp_path):
    repo_dir = tmp_path / "repo_fp"
    repo_dir.mkdir()
    repo = git.Repo.init(str(repo_dir))

    # Initial commit
    file_common = repo_dir / "common.txt"
    file_common.write_text("common base\n")
    repo.index.add([str(file_common)])
    repo.index.commit("initial")

    try:
        repo.git.branch('-M', 'main')
        main_name = 'main'
    except:
        main_name = 'master'

    # Branch A modifies file_a.txt
    repo.git.checkout('-b', 'branch_a')
    file_a = repo_dir / "file_a.txt"
    file_a.write_text("same line\n")
    repo.index.add([str(file_a)])
    repo.index.commit("add file_a")

    # Branch B modifies file_b.txt (different file, same content)
    repo.git.checkout(main_name)
    repo.git.checkout('-b', 'branch_b')
    file_b = repo_dir / "file_b.txt"
    file_b.write_text("same line\n")
    repo.index.add([str(file_b)])
    repo.index.commit("add file_b")

    repo.git.checkout(main_name)

    predictor = ConflictPredictor(str(repo_dir))
    predictions = predictor.predict_conflicts([main_name, 'branch_a', 'branch_b'])

    # There should be NO predictions between branch_a and branch_b
    # because they modify different files.
    assert len(predictions) == 0
