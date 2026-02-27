## 2025-05-15 - [Conflict Prediction Algorithm Complexity]
**Learning:** Identifying $O(N^2)$ patterns that involve external process calls (like `git diff`) is a high-impact optimization target. In this codebase, predicting conflicts between $N$ branches was triggering $N^2$ git calls, which is extremely slow as $N$ grows.
**Action:** Always check if metadata needed for pairwise comparisons can be pre-calculated and cached once per item ($O(N)$), reducing the inner loop to simple in-memory operations.
