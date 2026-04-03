import os
import pytest
import stat
from pathlib import Path
from guardrails import ensure_private_file, ensure_private_dir

def test_ensure_private_file_permissions(tmp_path):
    test_file = tmp_path / "test_private.txt"
    ensure_private_file(test_file)

    assert test_file.exists()
    mode = os.stat(test_file).st_mode
    # Check for 0o600 (rw-------)
    assert stat.S_IMODE(mode) == 0o600

def test_ensure_private_file_rejects_symlink(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("secret")

    link = tmp_path / "link.txt"
    os.symlink(target, link)

    with pytest.raises(ValueError, match="Security error:.*is a symbolic link"):
        ensure_private_file(link)

def test_ensure_private_dir_permissions(tmp_path):
    test_dir = tmp_path / "test_private_dir"
    ensure_private_dir(test_dir)

    assert test_dir.is_dir()
    mode = os.stat(test_dir).st_mode
    # Check for 0o700 (rwx------)
    assert stat.S_IMODE(mode) == 0o700

def test_ensure_private_dir_rejects_symlink(tmp_path):
    target = tmp_path / "target_dir"
    target.mkdir()

    link = tmp_path / "link_dir"
    os.symlink(target, link)

    with pytest.raises(ValueError, match="Security error:.*is a symbolic link"):
        ensure_private_dir(link)

def test_ensure_private_file_race_condition_safety(tmp_path):
    # This test verifies that we can create a file that doesn't exist
    # and it gets 0o600 from the start.
    test_file = tmp_path / "new_secure_file.txt"
    ensure_private_file(test_file)
    mode = os.stat(test_file).st_mode
    assert stat.S_IMODE(mode) == 0o600
