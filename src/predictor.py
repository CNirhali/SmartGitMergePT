from git_utils import GitUtils
from optimizer import SmartCache, CacheConfig, CacheStrategy
from typing import List, Dict, Set, Tuple
from concurrent.futures import ThreadPoolExecutor
import difflib
import collections
import hashlib
import functools

class ConflictPredictor:
    def __init__(self, repo_path: str = "."):
        self.git_utils = GitUtils(repo_path)
        # BOLT: Initialize cache for branch metadata and semantic similarity
        # BOLT: Enabled persistence to allow results to survive between runs/refreshes
        self.cache = SmartCache(CacheConfig(
            strategy=CacheStrategy.TTL,
            max_size=2000,
            ttl_seconds=600,  # 10 minutes cache
            enable_persistence=True
        ))

    def _fetch_branch_data_task(self, branch: str, main_branch: str, main_commit: str) -> Tuple[str, Dict]:
        """BOLT: Pre-calculate diffs and metadata for a branch"""
        try:
            commit_hash = self.git_utils.repo.commit(branch).hexsha
        except:
            commit_hash = "unknown"

        # BOLT: Include main_commit hash in cache key to avoid stale results
        cache_key_raw = f"metadata:{branch}:{commit_hash}:{main_branch}:{main_commit}"
        # BOLT: Pre-calculate hash to avoid triple-hashing (get + set + internal _get_cache_key)
        cache_key = hashlib.md5(cache_key_raw.encode()).hexdigest()

        cached_data = self.cache.get(cache_key, is_hash=True)
        if cached_data:
            return branch, cached_data

        if branch == main_branch:
            diff = ""
        else:
            try:
                # BOLT: Using unified=0 to reduce diff size as context is not needed for overlap check
                diff = self.git_utils.get_diff_between_branches(main_branch, branch, unified=0)
            except:
                diff = ""

        files, lines = self._get_diff_metadata(diff)
        data = {
            # BOLT: Removed large 'diff' string to save memory and I/O
            'files': files,
            'lines': lines,
            # BOLT: Removed eager 'sorted_lines' calculation to improve O(N) phase
            'commit': commit_hash
        }
        self.cache.set(cache_key, data, is_hash=True)
        return branch, data

    def _get_lazy_sorted_lines(self, data: Dict, file: str) -> str:
        """BOLT: Lazily calculate and memoize sorted lines in branch_data"""
        if 'sorted_lines' not in data:
            data['sorted_lines'] = {}

        if file not in data['sorted_lines']:
            data['sorted_lines'][file] = "\n".join(sorted(data['lines'].get(file, [])))

        return data['sorted_lines'][file]

    def predict_conflicts(self, branches: List[str]) -> List[Dict]:
        predictions = []
        # BOLT: Improve main_branch detection to prioritize 'main' or 'master' from input list
        # This prevents comparing everything against an active feature branch.
        if 'main' in branches:
            main_branch = 'main'
        elif 'master' in branches:
            main_branch = 'master'
        else:
            try:
                main_branch = self.git_utils.repo.active_branch.name
            except:
                main_branch = 'main'

        # BOLT: Get main branch commit hash to ensure cache validity
        try:
            main_commit = self.git_utils.repo.commit(main_branch).hexsha
        except:
            main_commit = "unknown"

        # BOLT OPTIMIZATION: Pre-calculate diffs and metadata for each branch in parallel
        # This reduces Git operations from O(N^2) to O(N) where N is number of branches
        # Parallelization handles I/O-bound git process calls efficiently.
        with ThreadPoolExecutor() as executor:
            # BOLT: Using lambda to pass additional context without re-defining function
            results = list(executor.map(lambda b: self._fetch_branch_data_task(b, main_branch, main_commit), branches))

        branch_data = dict(results)

        # BOLT: Build a file-to-branches index to optimize pairwise comparison
        # This transforms the O(N^2) search into a more efficient targeted check
        # BOLT: Using defaultdict for O(1) initialization of list entries
        file_to_branches = collections.defaultdict(list)
        for branch, data in branch_data.items():
            if branch == main_branch:
                continue
            for file in data['files']:
                file_to_branches[file].append(branch)

        # BOLT: Only check pairs of branches that actually share at least one modified file
        seen_pairs = set()
        for common_file, sharers in file_to_branches.items():
            for i, branch_a in enumerate(sharers):
                for branch_b in sharers[i+1:]:
                    # BOLT: Faster pair creation without sorted()
                    pair = (branch_a, branch_b) if branch_a < branch_b else (branch_b, branch_a)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)

                    data_a = branch_data[branch_a]
                    data_b = branch_data[branch_b]

                    # Recalculate full overlap for this specific pair
                    overlap = data_a['files'] & data_b['files']

                    # BOLT: Using pre-calculated line sets per file
                    # Only check line-level overlap for files that both branches modified
                    # BOLT: Use isdisjoint() for O(min(len_a, len_b)) and early exit
                    line_conflicts = False
                    for common_f in overlap:
                        if not data_a['lines'][common_f].isdisjoint(data_b['lines'][common_f]):
                            line_conflicts = True
                            break

                    # BOLT: Semantic similarity is slow, only check if there's no line overlap
                    # but files still overlap. Optimized to only compare overlapping file content.
                    semantic_conflict = False
                    if not line_conflicts:
                        # Use commit hashes in cache key for semantic similarity
                        sim_key_raw = f"sim:{data_a['commit']}:{data_b['commit']}"
                        # BOLT: Pre-calculate hash to avoid triple-hashing
                        sim_key = hashlib.md5(sim_key_raw.encode()).hexdigest()

                        cached_sim = self.cache.get(sim_key, is_hash=True)
                        if cached_sim is not None:
                            semantic_conflict = cached_sim
                        else:
                            # BOLT: Check similarity file-by-file for early exit and smaller SequenceMatcher inputs
                            # This provides a massive win for large multi-file diffs.
                            for common_f in overlap:
                                content_a = self._get_lazy_sorted_lines(data_a, common_f)
                                content_b = self._get_lazy_sorted_lines(data_b, common_f)
                                if self._semantic_similarity(content_a, content_b):
                                    semantic_conflict = True
                                    break

                            self.cache.set(sim_key, semantic_conflict, is_hash=True)

                    if overlap or line_conflicts or semantic_conflict:
                        predictions.append({
                            'branches': pair,
                            'files': list(overlap),
                            'line_conflicts': line_conflicts,
                            'semantic_conflict': semantic_conflict,
                            'conflict_likely': True
                        })
        return predictions

    def _get_diff_metadata(self, diff: str) -> Tuple[Set[str], Dict[str, Set[str]]]:
        """BOLT: Optimized extraction of changed files and lines"""
        files = set()
        lines_by_file = {}
        current_lines = None

        # BOLT: Using splitlines() for robustness with universal newlines (LF, CRLF)
        for line in diff.splitlines():
            if not line:
                continue

            c0 = line[0]
            if c0 == '+' or c0 == '-':
                if current_lines is not None:
                    # BOLT: Optimized header skip check: check 2nd and 3rd chars
                    # Using indexing is faster than startswith('+++')/startswith('---')
                    # because it avoids a method call and argument string creation.
                    if len(line) >= 3 and line[1] == c0 and line[2] == c0:
                        continue
                    # BOLT: Slicing line[1:] creates a new string but is necessary to extract content.
                    current_lines.add(line[1:])
            elif c0 == 'd':  # diff --git ...
                if line.startswith('diff --git'):
                    # Extract file path after ' b/' to avoid multiple splits
                    b_idx = line.find(' b/')
                    if b_idx != -1:
                        current_file = line[b_idx + 3:]
                        files.add(current_file)
                        current_lines = set()
                        lines_by_file[current_file] = current_lines
                    else:
                        current_lines = None
        return files, lines_by_file

    def _extract_changed_files(self, diff: str) -> List[str]:
        # Maintained for backward compatibility but using _get_diff_metadata internally is better
        files, _ = self._get_diff_metadata(diff)
        return list(files)

    def _line_level_overlap(self, diff_a: str, diff_b: str) -> bool:
        # Maintained for backward compatibility
        files_a, lines_by_file_a = self._get_diff_metadata(diff_a)
        files_b, lines_by_file_b = self._get_diff_metadata(diff_b)
        overlap = files_a & files_b
        for common_file in overlap:
            if not lines_by_file_a[common_file].isdisjoint(lines_by_file_b[common_file]):
                return True
        return False

    @functools.lru_cache(maxsize=1024)
    def _semantic_similarity(self, diff_a: str, diff_b: str) -> bool:
        # BOLT: SequenceMatcher is expensive. Use quick_ratio for early rejection.
        if not diff_a or not diff_b:
            return False

        # BOLT: Identity check for fast-path
        if diff_a == diff_b:
            return True

        # BOLT: Length-based early rejection
        # If the strings have significantly different lengths, the similarity ratio
        # cannot physically exceed 0.7. Mathematical upper bound: 2*min(L1, L2)/(L1+L2)
        # This is O(1) and replaces the more expensive seq.real_quick_ratio() check.
        len_a, len_b = len(diff_a), len(diff_b)
        if 2.0 * min(len_a, len_b) / (len_a + len_b) < 0.7:
            return False

        seq = difflib.SequenceMatcher(None, diff_a, diff_b)

        # BOLT: Fast early rejection using a hierarchy of checks
        # real_quick_ratio() is the fastest, then quick_ratio(), then ratio()
        if seq.real_quick_ratio() < 0.7:
            return False

        if seq.quick_ratio() < 0.7:
            return False

        return seq.ratio() > 0.7
