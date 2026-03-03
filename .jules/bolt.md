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
