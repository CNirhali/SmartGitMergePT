## 2026-02-27 - [Path Traversal in Code Reviewer]
**Vulnerability:** Path traversal vulnerability in `MistralCodeReviewer._review_file` allowed reading arbitrary files outside the repository root.
**Learning:** Using `os.path.join(self.repo_path, file_path)` is insufficient to prevent path traversal if `file_path` contains `..` segments.
**Prevention:** Always resolve the absolute path and verify it starts with the expected base directory using `os.path.commonpath`.
