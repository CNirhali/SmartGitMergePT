from git_utils import GitUtils
from typing import List, Dict, Set, Tuple
from concurrent.futures import ThreadPoolExecutor
import difflib

class ConflictPredictor:
    def __init__(self, repo_path: str = "."):
        self.git_utils = GitUtils(repo_path)

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

        # BOLT OPTIMIZATION: Pre-calculate diffs and metadata for each branch in parallel
        # This reduces Git operations from O(N^2) to O(N) where N is number of branches
        # Parallelization handles I/O-bound git process calls efficiently.
        def _fetch_branch_data(branch):
            if branch == main_branch:
                diff = ""
            else:
                try:
                    # BOLT: Using unified=0 to reduce diff size as context is not needed for overlap check
                    diff = self.git_utils.get_diff_between_branches(main_branch, branch, unified=0)
                except:
                    diff = ""
            files, lines = self._get_diff_metadata(diff)
            return branch, {
                'diff': diff,
                'files': files,
                'lines': lines
            }

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(_fetch_branch_data, branches))

        branch_data = dict(results)

        for i, branch_a in enumerate(branches):
            if branch_a == main_branch:
                continue
            data_a = branch_data[branch_a]
            for branch_b in branches[i+1:]:
                if branch_b == main_branch:
                    continue
                data_b = branch_data[branch_b]

                overlap = data_a['files'] & data_b['files']

                # BOLT: Using pre-calculated line sets per file
                # Only check line-level overlap for files that both branches modified
                line_conflicts = False
                for common_file in overlap:
                    if data_a['lines'].get(common_file) & data_b['lines'].get(common_file):
                        line_conflicts = True
                        break

                # BOLT: Semantic similarity is slow, only check if there's no overlap already
                # or if some files overlap to warrant deeper check
                semantic_conflict = False
                if not line_conflicts and overlap:
                    semantic_conflict = self._semantic_similarity(data_a['diff'], data_b['diff'])

                if overlap or line_conflicts or semantic_conflict:
                    predictions.append({
                        'branches': (branch_a, branch_b),
                        'files': list(overlap),
                        'line_conflicts': line_conflicts,
                        'semantic_conflict': semantic_conflict,
                        'conflict_likely': bool(overlap or line_conflicts or semantic_conflict)
                    })
        return predictions

    def _get_diff_metadata(self, diff: str) -> Tuple[Set[str], Dict[str, Set[str]]]:
        """BOLT: Optimized extraction of changed files and lines"""
        files = set()
        lines_by_file = {}
        current_lines = None

        for line in diff.splitlines():
            if not line:
                continue

            c0 = line[0]
            if c0 == 'd':  # diff --git ...
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
            elif (c0 == '+' or c0 == '-') and current_lines is not None:
                # Skip +++ or --- headers
                if len(line) >= 3 and line[1] == c0 and line[2] == c0:
                    continue
                current_lines.add(line[1:])
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
            if lines_by_file_a.get(common_file) & lines_by_file_b.get(common_file):
                return True
        return False

    def _semantic_similarity(self, diff_a: str, diff_b: str) -> bool:
        # BOLT: SequenceMatcher is expensive. Use quick_ratio for early rejection.
        if not diff_a or not diff_b:
            return False

        seq = difflib.SequenceMatcher(None, diff_a, diff_b)

        # Fast early rejection
        if seq.real_quick_ratio() < 0.7:
            return False
        if seq.quick_ratio() < 0.7:
            return False

        return seq.ratio() > 0.7
