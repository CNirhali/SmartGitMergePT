import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from predictor import ConflictPredictor

def test_empty_line_diff():
    predictor = ConflictPredictor()
    # Diff with a file header and then a single '+' which represents an added empty line
    diff = "diff --git a/file.txt b/file.txt\n--- a/file.txt\n+++ b/file.txt\n+"
    try:
        files, lines = predictor._get_diff_metadata(diff)
        print("Success: _get_diff_metadata handled empty line diff")
        assert "file.txt" in files
        assert "" in lines["file.txt"]
    except IndexError:
        print("Failure: IndexError raised")
        sys.exit(1)
    except Exception as e:
        print(f"Failure: {e} raised")
        sys.exit(1)

if __name__ == "__main__":
    test_empty_line_diff()
