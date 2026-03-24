import pytest
from predictor import ConflictPredictor

def test_get_diff_metadata_basic():
    predictor = ConflictPredictor()
    diff = """diff --git a/file1.txt b/file1.txt
index 1234567..89abcde 100644
--- a/file1.txt
+++ b/file1.txt
@@ -1,1 +1,1 @@
-old line
+new line
diff --git a/file2.txt b/file2.txt
--- a/file2.txt
+++ b/file2.txt
-removed line
+added line"""
    files, lines = predictor._get_diff_metadata(diff)
    assert "file1.txt" in files
    assert "file2.txt" in files
    assert "new line" in lines["file1.txt"]
    assert "old line" in lines["file1.txt"]
    assert "added line" in lines["file2.txt"]
    assert "removed line" in lines["file2.txt"]
    assert "--- a/file1.txt" not in lines["file1.txt"]
    assert "+++ b/file1.txt" not in lines["file1.txt"]

def test_semantic_similarity_basic():
    predictor = ConflictPredictor()
    s1 = "import os\nimport sys\ndef hello():\n    print('hello world')\n"
    s2 = "import os\nimport sys\ndef hello():\n    print('hello world')\n"
    assert predictor._semantic_similarity(s1, s2) is True

    s3 = "def goodbye():\n    pass"
    assert predictor._semantic_similarity(s1, s3) is False

    s4 = "import os\nimport sys\ndef hello_world():\n    print('hello world!')\n"
    assert predictor._semantic_similarity(s1, s4) is True

def test_predict_conflicts_with_busy_master(tmp_path):
    import git
    # Create a real repo to test triple-dot diff behavior in predictor
    repo_dir = tmp_path / "repo"
    repo = git.Repo.init(repo_dir)
    with repo.config_writer() as config:
        config.set_value("user", "name", "Test")
        config.set_value("user", "email", "test@example.com")

    base_file = repo_dir / "base.txt"
    base_file.write_text("initial\n")
    repo.index.add(["base.txt"])
    repo.index.commit("initial")

    # Feature branch diverges here
    repo.git.checkout("-b", "feature")
    feat_file = repo_dir / "feature.txt"
    feat_file.write_text("feature change\n")
    repo.index.add(["feature.txt"])
    repo.index.commit("feature change")

    # Master progresses with changes that don't conflict with feature branch's NEW changes
    repo.git.checkout("master")
    base_file.write_text("initial\nmaster change\n")
    repo.index.add(["base.txt"])
    repo.index.commit("master change")

    from predictor import ConflictPredictor
    predictor = ConflictPredictor(str(repo_dir))

    # Predict conflicts between master and feature
    # Using '..' (double-dot) would show base.txt as modified in BOTH if we compared master..feature
    # But '...' (triple-dot) only shows feature.txt as modified in feature.
    predictions = predictor.predict_conflicts(["master", "feature"])

    # There should be no conflicts because feature only added feature.txt
    # and base.txt changes in master should NOT be attributed to the feature branch.
    assert len(predictions) == 0
