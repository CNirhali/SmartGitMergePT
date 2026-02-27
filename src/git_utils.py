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
        # Checkout target branch
        self.repo.git.checkout(target_branch)
        try:
            self.repo.git.merge(source_branch, '--no-commit', '--no-ff')
            # If merge succeeds, abort to keep repo clean
            try:
                self.repo.git.merge('--abort')
            except:
                pass
            return True, "No conflicts detected."
        except git.exc.GitCommandError as e:
            try:
                self.repo.git.merge('--abort')
            except:
                pass
            return False, str(e) 