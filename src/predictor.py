from git_utils import GitUtils
from typing import List, Dict
import difflib

class ConflictPredictor:
    def __init__(self, repo_path: str = "."):
        self.git_utils = GitUtils(repo_path)

    def predict_conflicts(self, branches: List[str]) -> List[Dict]:
        predictions = []
        # Try to determine the default branch
        try:
            main_branch = self.git_utils.repo.active_branch.name
        except:
            main_branch = 'main'

        # BOLT OPTIMIZATION: Pre-calculate diffs and metadata for each branch
        # This reduces Git operations from O(N^2) to O(N) where N is number of branches
        branch_data = {}
        for branch in branches:
            diff = self.git_utils.get_diff_between_branches(main_branch, branch)
            branch_data[branch] = {
                'diff': diff,
                'files': set(self._extract_changed_files(diff))
            }

        for i, branch_a in enumerate(branches):
            data_a = branch_data[branch_a]
            for branch_b in branches[i+1:]:
                data_b = branch_data[branch_b]

                overlap = data_a['files'] & data_b['files']
                line_conflicts = self._line_level_overlap(data_a['diff'], data_b['diff'])
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

    def _extract_changed_files(self, diff: str) -> List[str]:
        files = []
        for line in diff.splitlines():
            if line.startswith('diff --git'):
                parts = line.split(' ')
                if len(parts) > 2:
                    files.append(parts[2][2:])  # Remove the a/ prefix
        return files

    def _line_level_overlap(self, diff_a: str, diff_b: str) -> bool:
        # Naive approach: check if any added/removed lines are the same
        lines_a = set([l[1:] for l in diff_a.splitlines() if l.startswith('+') or l.startswith('-')])
        lines_b = set([l[1:] for l in diff_b.splitlines() if l.startswith('+') or l.startswith('-')])
        return bool(lines_a & lines_b)

    def _semantic_similarity(self, diff_a: str, diff_b: str) -> bool:
        # Placeholder: in real use, use embeddings or LLM for semantic similarity
        # Here, use difflib for a rough similarity check
        seq = difflib.SequenceMatcher(None, diff_a, diff_b)
        return seq.ratio() > 0.7 