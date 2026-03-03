from git_utils import GitUtils
from typing import List, Dict, Set, Tuple
import difflib
from functools import lru_cache

class ConflictPredictor:
    def __init__(self, repo_path: str = "."):
        self.git_utils = GitUtils(repo_path)

    def predict_conflicts(self, branches: List[str]) -> List[Dict]:
        predictions = []
        # Try to determine the default branch
        try:
            main_branch = self.git_utils.repo.active_branch.name
        except:
            # Fallback to checking for main/master in branches list
            if 'main' in branches:
                main_branch = 'main'
            elif 'master' in branches:
                main_branch = 'master'
            else:
                main_branch = 'main'

        # BOLT OPTIMIZATION: Pre-calculate diffs and metadata for each branch
        # This reduces Git operations from O(N^2) to O(N) where N is number of branches
        branch_data = {}
        for branch in branches:
            if branch == main_branch:
                diff = ""
            else:
                try:
                    diff = self.git_utils.get_diff_between_branches(main_branch, branch)
                except:
                    # If we fail to get diff, use empty diff
                    diff = ""
            files, lines = self._get_diff_metadata(diff)
            branch_data[branch] = {
                'diff': diff,
                'files': files,
                'lines': lines
            }

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

    @lru_cache(maxsize=128)
    def _get_diff_metadata(self, diff: str) -> Tuple[Set[str], Dict[str, Set[str]]]:
        """BOLT: Extract changed files and lines grouped by file in a single pass with caching"""
        files = set()
        lines_by_file = {}
        current_file = None
        for line in diff.splitlines():
            if line.startswith('diff --git'):
                parts = line.split(' ')
                if len(parts) > 2:
                    current_file = parts[2][2:]  # Remove the a/ prefix
                    files.add(current_file)
                    if current_file not in lines_by_file:
                        lines_by_file[current_file] = set()
            elif line.startswith('+') or line.startswith('-'):
                # Avoid capturing the diff header lines as changes
                if not (line.startswith('+++') or line.startswith('---')) and current_file:
                    lines_by_file[current_file].add(line[1:])
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
