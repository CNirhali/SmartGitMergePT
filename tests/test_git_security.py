import pytest
import os
import git
from git_utils import GitUtils

@pytest.fixture
def temp_repo(tmp_path):
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    repo = git.Repo.init(repo_path)

    # Create initial commit
    file_path = repo_path / "file.txt"
    file_path.write_text("initial content")
    repo.index.add([str(file_path)])
    repo.index.commit("initial commit")

    # Create a branch
    repo.git.branch("feature")

    return str(repo_path)

def test_get_diff_between_branches_vulnerable_branch(temp_repo):
    utils = GitUtils(temp_repo)
    with pytest.raises(ValueError, match="Invalid branch name"):
        utils.get_diff_between_branches("--version", "master")
    with pytest.raises(ValueError, match="Invalid branch name"):
        utils.get_diff_between_branches("master", "--version")

    # 🛡️ Sentinel: Test for shell metacharacters (Social Engineering injection)
    # Also test for space and colon which were recently added
    dangerous_branches = ["main;touch_file", "main&whoami", "branch$(calc)", "branch`id`", "branch with space", "branch:with:colon"]
    for db in dangerous_branches:
        with pytest.raises(ValueError, match="Invalid branch name"):
            utils.get_diff_between_branches(db, "master")

def test_simulate_merge_vulnerable_branch(temp_repo):
    utils = GitUtils(temp_repo)
    with pytest.raises(ValueError, match="Invalid branch name"):
        utils.simulate_merge("--version", "master")
    with pytest.raises(ValueError, match="Invalid branch name"):
        utils.simulate_merge("master", "--version")

def test_normal_branches_work(temp_repo):
    utils = GitUtils(temp_repo)
    # Should not raise ValueError
    try:
        # Note: 'master' might not exist depending on git version, but init usually creates it or 'main'
        branches = utils.list_branches()
        if branches:
            utils.get_diff_between_branches(branches[0], branches[0])
            utils.simulate_merge(branches[0], branches[0])
    except ValueError:
        pytest.fail("ValueError raised for normal branch name")
    except Exception:
        # Other exceptions are fine for this test, we only care about ValueError from validation
        pass
