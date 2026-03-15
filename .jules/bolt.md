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
**Learning:** Naive pairwise comparison of $ branches is (N^2)$, which is devastating for performance in large repositories. Using an inverted index (mapping files to branches that modified them) reduces the search space to only those pairs that actually share modified files.
**Action:** When performing exhaustive pairwise checks on objects with shared attributes, always build an inverted index first to isolate relevant pairs.

## 2025-05-30 - [Context-Aware Semantic Similarity]
**Learning:** Running  on entire diff strings is wasteful and slow, especially when only a small portion of the diff (the overlapping files) is relevant to the conflict prediction. Trimming the input to only overlapping file content significantly improves execution speed and heuristic focus.
**Action:** Always filter and minimize input size for expensive string comparison algorithms like  to focus on the specific segments that matter.

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
