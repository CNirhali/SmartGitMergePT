import git
from typing import List, Tuple

class GitUtils:
    def __init__(self, repo_path: str = "."):
        self.repo = git.Repo(repo_path)

    def list_branches(self) -> List[str]:
        return [head.name for head in self.repo.heads]

    def _validate_branch_name(self, branch_name: str):
        """🛡️ Sentinel: Validate branch name to prevent argument and shell injection"""
        if not branch_name:
            raise ValueError("Branch name cannot be empty.")

        if branch_name.startswith('-'):
            raise ValueError(f"Invalid branch name: {branch_name}. Branch names cannot start with a hyphen.")

        # 🛡️ Sentinel: Block dangerous shell metacharacters to prevent command injection via social engineering
        # These characters are either not allowed in git branch names or pose a risk if pasted into a shell
        dangerous_chars = {';', '&', '|', '$', '(', ')', '`', '>', '<', '\\', "'", '"', '*', '?', '[', ']', '!', '{', '}', '\n', '\r'}
        for char in dangerous_chars:
            if char in branch_name:
                raise ValueError(f"Invalid branch name: {branch_name}. Branch names cannot contain shell metacharacters like '{char}'.")

    def get_diff_between_branches(self, branch_a: str, branch_b: str, unified: int = 3) -> str:
        self._validate_branch_name(branch_a)
        self._validate_branch_name(branch_b)
        return self.repo.git.diff(f'{branch_a}..{branch_b}', unified=unified)

    def simulate_merge(self, source_branch: str, target_branch: str) -> Tuple[bool, str]:
        """BOLT: Perform in-memory merge simulation using git merge-tree for speed"""
        self._validate_branch_name(source_branch)
        self._validate_branch_name(target_branch)
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