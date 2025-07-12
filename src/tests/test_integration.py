import git
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
    repo.git.checkout('-b', 'feature/a')
    file_path.write_text("base\na\n")
    repo.index.add([str(file_path)])
    repo.index.commit("a change")
    repo.git.checkout('main')
    repo.git.checkout('-b', 'feature/b')
    file_path.write_text("base\nb\n")
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