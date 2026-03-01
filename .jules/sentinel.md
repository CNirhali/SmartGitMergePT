## 2026-02-27 - [Path Traversal in Code Reviewer]
**Vulnerability:** Path traversal vulnerability in `MistralCodeReviewer._review_file` allowed reading arbitrary files outside the repository root.
**Learning:** Using `os.path.join(self.repo_path, file_path)` is insufficient to prevent path traversal if `file_path` contains `..` segments.
**Prevention:** Always resolve the absolute path and verify it starts with the expected base directory using `os.path.commonpath`.

## 2026-02-28 - [Insecure Deserialization in SmartCache]
**Vulnerability:** Insecure deserialization using `pickle.load` in `SmartCache._load_from_disk` allowed for potential Remote Code Execution (RCE) via malicious cache files.
**Learning:** `pickle` is inherently unsafe for data that persists to disk where it could be tampered with. Even internal caches should use safer formats.
**Prevention:** Use `json` for serialization of persistent data. For non-JSON serializable types like `set`, implement custom encoders and decoders.

## 2025-05-15 - [Path Traversal Bypass in InputValidator]
**Vulnerability:** The `InputValidator.validate_path` function used a weak `startswith` check on normalized paths, which could be bypassed by sibling directories with matching prefixes (e.g., `/base/path-secrets` matching `/base/path`).
**Learning:** Simple string prefix matching is insufficient for path validation even after normalization. Directory boundaries must be explicitly respected.
**Prevention:** Use `os.path.commonpath` with absolute paths to ensure the validated path is strictly a descendant of the base directory.
