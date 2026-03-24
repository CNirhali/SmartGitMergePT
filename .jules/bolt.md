## 2025-05-15 - [Conflict Prediction Algorithm Complexity]
**Learning:** Identifying $O(N^2)$ patterns that involve external process calls (like `git diff`) is a high-impact optimization target. In this codebase, predicting conflicts between $N$ branches was triggering $N^2$ git calls, which is extremely slow as $N$ grows.
**Action:** Always check if metadata needed for pairwise comparisons can be pre-calculated and cached once per item ($O(N)$), reducing the inner loop to simple in-memory operations.

## 2025-05-16 - [Accelerating Diff Comparisons with Early Rejection]
**Learning:** `difflib.SequenceMatcher.ratio()` is $O(L^2)$ in the worst case and extremely slow. Using `real_quick_ratio()` and `quick_ratio()` as early filters can skip the full computation for the majority of non-matching pairs.
**Action:** Always use hierarchical similarity checks (quick filters -> full ratio) when performing batch string comparisons.

## 2025-05-20 - [Non-blocking CPU monitoring and Cache Data Integrity]
**Learning:** Using 'psutil.cpu_percent(interval=1)' in high-frequency paths (like decorators) introduces a mandatory 1-second block per call, drastically killing performance. Cache compression should be transparently decompressed during 'get()' regardless of current config to avoid returning internal markers.
**Action:** Always initialize 'psutil' counters in '__init__' and use 'interval=None' for non-blocking checks. Ensure cache retrieval always attempts decompression for transparency and data integrity.

## 2025-05-22 - [File-Scoped Line Conflict Detection]
**Learning:** Performing line-level conflict detection across global sets of changed lines (all files merged) causes false positives and unnecessary set intersections. Grouping changes by file allows for more precise and efficient overlap checks.
**Action:** When analyzing changes across multiple files, maintain file boundaries in metadata structures to avoid cross-file false positives and skip line-level comparisons for non-overlapping files.

## 2025-05-25 - [Parallelizing Git Subprocesses]
**Learning:** Fetching Git diffs for multiple branches is I/O-bound due to external process overhead. Sequential execution creates significant latency that scales linearly with the number of branches.
**Action:** Use `ThreadPoolExecutor` to parallelize I/O-bound `git` command calls. This can yield ~50% performance gains in data preparation phases for large branch sets.

## 2025-05-30 - [Targeted Pairwise Comparison via Inverted Index]
**Learning:** Naive pairwise comparison of $N$ branches is $O(N^2)$, which is devastating for performance in large repositories. Using an inverted index (mapping files to branches that modified them) reduces the search space to only those pairs that actually share modified files.
**Action:** When performing exhaustive pairwise checks on objects with shared attributes, always build an inverted index first to isolate relevant pairs.

## 2025-05-30 - [Context-Aware Semantic Similarity]
**Learning:** Running `difflib.SequenceMatcher` on entire diff strings is wasteful and slow, especially when only a small portion of the diff (the overlapping files) is relevant to the conflict prediction. Trimming the input to only overlapping file content significantly improves execution speed and heuristic focus.
**Action:** Always filter and minimize input size for expensive string comparison algorithms like `SequenceMatcher` to focus on the specific segments that matter.

## 2025-06-05 - [Lazy Evaluation of Comparison Metadata]
**Learning:** Eagerly calculating comparison metadata (like sorted line sets for similarity checks) during the initial O(N) data gathering phase is wasteful. For repositories where most files don't overlap between branches, thousands of expensive operations are performed for data that is never used.
**Action:** Always use lazy evaluation and memoization for expensive transformation of per-item metadata that is only needed during targeted pairwise comparisons.

## 2025-06-10 - [Regex Pattern Combination and Fast-Pathing]
**Learning:** Calling `re.search` multiple times in a loop (e.g., for security validation) is significantly slower than combining patterns into a single grouped regex. Additionally, for "clean" inputs, expensive regex substitutions in sanitization can be entirely bypassed with $O(N)$ string checks (like `in` or `str.contains`).
**Action:** Combine multiple independent regex patterns into a single pre-compiled regex for initial "all-clear" checks. Implement $O(N)$ fast-paths to skip expensive processing for common safe inputs.

## 2025-06-12 - [Hierarchical Fast-Path for String Sanitization]
**Learning:** Initializing expensive operations like 'text.lower()' in a fast-path can still be a significant bottleneck for large inputs, even if it avoids regex. In this codebase, 'InputValidator._sanitize_html' was calling 'lower()' eagerly for every string without tags.
**Action:** Use hierarchical checks: first check for simple character markers (like '<' or ':') before calling 'lower()' or starting regex searches. Pre-calculate 'lower()' once if multiple substring checks are needed to avoid redundant (N)$ passes.

## 2025-06-15 - [Process-Specific Resource Monitoring]
**Learning:** Using system-wide metrics like `psutil.virtual_memory().used` in a `ResourceManager` causes the application to throttle or trigger garbage collection due to unrelated "noisy neighbor" processes.
**Action:** Always use process-specific metrics via `psutil.Process()` (e.g., `memory_info().rss` and `cpu_percent(interval=None)`) to ensure optimization logic is grounded in the application's actual resource footprint.

## 2025-06-20 - [Redundant Hashing in Cache Operations]
**Learning:** When using `@cached_function`, the arguments are hashed to create a cache key, which is then re-hashed by `SmartCache`. On a cache miss, this results in triple hashing (once during key generation, once during `get`, and once during `set`).
**Action:** Introduce an `is_hash` parameter to cache methods to bypass redundant internal hashing when the key is already hashed. Combine this with single-pass `dict.get()` lookups for maximum efficiency in hot paths like conflict prediction.

## 2025-06-25 - [Memoizing Pairwise Content Comparisons]
**Learning:** In applications performing O(N^2) pairwise comparisons of items (like git branches) containing shared sub-items (like files), memoizing the comparison of those sub-items provides exponential speedups. Content-based memoization using '@functools.lru_cache' on the similarity function reduced comparison time from ~0.77s to ~0.009s in benchmarks.
**Action:** Always look for opportunities to memoize granular comparison logic in pairwise algorithms, especially when the input space has high overlap.

## 2025-06-25 - [Static vs Dynamic String Checks in Hot Loops]
**Learning:** Dynamic string multiplication (e.g., 'c0 * 3') and slicing in a hot loop (like diff parsing) can be measurably slower than static prefix checks. Replacing dynamic header detection with explicit 'startswith' for '+++' and '---' reduces overhead in O(L) diff processing where L is total lines across all branch diffs.
**Action:** Prefer static constant comparisons over dynamic string construction in performance-critical parsing loops.

## 2025-07-05 - [Hierarchical Fail-Fast for Diff Similarity and Header Detection]
**Learning:** In high-frequency diff parsing and similarity checking, `real_quick_ratio()` is an essential first-stage filter for `difflib.SequenceMatcher`. Additionally, manual indexing (`line[1] == c0`) is measurably faster than `startswith` in Python loops as it avoids method dispatch and argument string overhead.
**Action:** Use `real_quick_ratio()` as the first stage of any `SequenceMatcher` hierarchy. Replace `startswith` with indexing in hot loops where the match pattern is a single character repetition (like diff headers).

## 2025-07-10 - [Batching Git Metadata Retrieval]
**Learning:** Spawning Git subprocesses is expensive. For metadata retrieval across many branches (e.g., commit hashes), performing $N$ individual `git rev-parse` calls (or using `repo.commit(branch).hexsha`) introduces significant overhead that scales linearly with $N$.
**Action:** Batch Git metadata retrieval whenever possible (e.g., `git rev-parse branch1 branch2 ...`) to reduce subprocess spawning overhead. Always include a fallback for the batch call to handle individual failures gracefully.
