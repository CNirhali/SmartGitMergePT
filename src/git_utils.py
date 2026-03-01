import git
from typing import List, Tuple

class GitUtils:
    def __init__(self, repo_path: str = "."):
        self.repo = git.Repo(repo_path)

    def list_branches(self) -> List[str]:
        return [head.name for head in self.repo.heads]

    def get_diff_between_branches(self, branch_a: str, branch_b: str) -> str:
        return self.repo.git.diff(f'{branch_a}..{branch_b}')

    def simulate_merge(self, source_branch: str, target_branch: str) -> Tuple[bool, str]:
        """BOLT: Perform in-memory merge simulation using git merge-tree for speed"""
        try:
            # git merge-tree <branch1> <branch2> performs an in-memory merge
            # It returns non-zero exit code if there are conflicts
            output = self.repo.git.merge_tree(source_branch, target_branch)
            if 'CONFLICT' in output:
                return False, output
            return True, "No conflicts detected."
        except git.exc.GitCommandError as e:
            # If git merge-tree returns non-zero, it usually means there were conflicts
            return False, str(e)