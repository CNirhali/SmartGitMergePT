import pytest
from unittest.mock import patch
from src.llm_resolver import resolve_conflict_with_mistral

def test_resolve_conflict_with_mistral():
    conflict_block = '<<<<<<< HEAD\nfoo\n=======\nbar\n>>>>>>> branch\n'
    expected = 'foo-bar-resolved'
    with patch('subprocess.run') as mock_run:
        mock_run.return_value.stdout = expected.encode()
        mock_run.return_value.returncode = 0
        result = resolve_conflict_with_mistral(conflict_block)
        assert result == expected 