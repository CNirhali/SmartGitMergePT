import tempfile
import os
import git
from src.predictor import ConflictPredictor
from src.git_utils import GitUtils

def setup_demo_repo():
    tmpdir = tempfile.TemporaryDirectory()
    repo_dir = tmpdir.name
    repo = git.Repo.init(repo_dir)
    file_path = os.path.join(repo_dir, "file.txt")
    with open(file_path, 'w') as f:
        f.write("base\n")
    repo.index.add([file_path])
    repo.index.commit("init")
    repo.git.checkout('-b', 'feature/a')
    with open(file_path, 'a') as f:
        f.write("a\n")
    repo.index.add([file_path])
    repo.index.commit("a change")
    repo.git.checkout('main')
    repo.git.checkout('-b', 'feature/b')
    with open(file_path, 'a') as f:
        f.write("b\n")
    repo.index.add([file_path])
    repo.index.commit("b change")
    return repo_dir, tmpdir

def main():
    repo_dir, tmpdir = setup_demo_repo()
    predictor = ConflictPredictor(repo_dir)
    git_utils = GitUtils(repo_dir)
    branches = git_utils.list_branches()
    print("Branches:", branches)
    predictions = predictor.predict_conflicts(branches)
    print("Predicted Conflicts:")
    for pred in predictions:
        print(pred)
    tmpdir.cleanup()

if __name__ == "__main__":
    main() 