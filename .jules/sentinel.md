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

## 2026-03-02 - [Stack Trace Leakage in Error Summary]
**Vulnerability:** The `ErrorHandler.get_error_summary` method included full Python stack traces (via `traceback.format_exc()`) in the public-facing health status report.
**Learning:** Error details collected for internal debugging can inadvertently be exposed through status/summary APIs if not explicitly sanitized at the egress point.
**Prevention:** Always scrub sensitive internal technical details like stack traces from data structures before returning them through public-facing or cross-boundary APIs (CWE-209). Use shallow copies to sanitize output without affecting internal debug logs.

## 2026-03-03 - [Blocking Event Loop in Async Health Checks]
**Vulnerability:** Use of `psutil.cpu_percent(interval=1)` in an `async` health check method blocked the entire `asyncio` event loop for 1 second per check, creating a potential DoS vector and degrading performance.
**Learning:** Blocking calls in `async` functions negate the benefits of concurrency. `psutil.cpu_percent` should use `interval=None` for non-blocking checks after an initial call.
**Prevention:** Avoid blocking synchronous calls in `async` methods. For `psutil`, initialize the counter in `__init__` and use `interval=None` in the check method.

## 2026-03-04 - [SSRF Vulnerability in URL Validator]
**Vulnerability:** `InputValidator.validate_url` allowed Server-Side Request Forgery (SSRF) bypasses via shorthand IP notation (e.g., `127.1`), non-standard loopback addresses (e.g., `127.0.0.2`), and cloud metadata/private IP ranges.
**Learning:** Simple string matching against a few hostnames is insufficient for URL security. Attackers can use various IP representations to target internal resources.
**Prevention:** Use the `ipaddress` module for comprehensive CIDR-based validation of loopback and private ranges. Normalize shorthand IP notations using `socket.inet_aton` before performing IP-level checks.

## 2026-03-05 - [Sensitive Data Exposure in Agentic Tracker]
**Vulnerability:** The `AgenticTracker` was capturing and saving raw webcam images of developers and storing session activity in a local database without proper Git exclusion, creating a high risk of sensitive biometric and activity data being accidentally committed and exposed.
**Learning:** Automated tracking systems that capture personal data must have robust privacy controls integrated from the start, including automatic sanitization (e.g., face blurring) and strict repository hygiene (e.g., `.gitignore`).
**Prevention:** Implement automatic face blurring for biometric data at the point of capture and ensure all tracking artifacts, logs, and local databases are explicitly ignored by version control.

## 2026-03-06 - [Insecure Password-Based Encryption]
**Vulnerability:** `DataEncryption` used a hardcoded static salt and only 100,000 PBKDF2 iterations for password-derived keys, making it vulnerable to precomputed rainbow tables and efficient brute-force attacks.
**Learning:** Static salts negate the benefits of salting against rainbow tables when the same salt is used across all users/installations. Iteration counts should be periodically reviewed against modern recommendations (e.g., OWASP).
**Prevention:** Use a unique, random salt (minimum 16 bytes) for every encryption operation. Store the salt alongside the ciphertext (e.g., using a versioned prefix format like `[version][salt][ciphertext]`) to ensure backward compatibility while allowing for future cryptographic upgrades.

## 2026-03-07 - [Path Traversal via Symlinks in Agentic Tracker]
**Vulnerability:** `AgenticTracker.register_developer` was vulnerable to path traversal, allowing arbitrary file reads. Initial fix using `os.path.abspath` was insufficient as it didn't resolve symbolic links.
**Learning:** `os.path.abspath` only resolves `..` segments but leaves symlinks intact, potentially allowing traversal if a symlink in the repo points outside. `os.path.realpath` is required for robust validation.
**Prevention:** Always use `os.path.realpath` before checking if a path is within the intended base directory using `os.path.commonpath`.
