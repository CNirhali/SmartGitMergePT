import git
from src.predictor import ConflictPredictor
from src.git_utils import GitUtils

def setup_demo_repo(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    repo = git.Repo.init(str(repo_dir))
    file_path = repo_dir / "file.txt"
    file_path.write_text("line 1\nline 2\n")
    repo.index.add([str(file_path)])
    repo.index.commit("init")

    repo.git.checkout('-b', 'feature/a')
    file_path.write_text("line 1 changed in a\nline 2\n")
    repo.index.add([str(file_path)])
    repo.index.commit("a change")

    # Use the active branch (likely master or main)
    try:
        repo.git.checkout('main')
    except:
        repo.git.checkout('master')

    default_branch = 'master' if 'master' in [h.name for h in repo.heads] else 'main'
    repo.git.checkout(default_branch)
    repo.git.checkout('-b', 'feature/b')
    # Force a conflict by modifying the same line
    file_path.write_text("line 1 changed in b\nline 2\n")
    repo.index.add([str(file_path)])
    repo.index.commit("b change")
    return str(repo_dir)

def test_integration_predict_and_detect(tmp_path):
    repo_path = setup_demo_repo(tmp_path)
    predictor = ConflictPredictor(repo_path)
    git_utils = GitUtils(repo_path)
    branches = git_utils.list_branches()
    predictions = predictor.predict_conflicts(branches)
    assert any(p['conflict_likely'] for p in predictions)
    # Simulate merge
    ok, msg = git_utils.simulate_merge('feature/a', 'feature/b')
    assert not ok
