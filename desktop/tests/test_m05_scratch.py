"""Use the real OS temporary root, including macOS's real /var alias."""
from pathlib import Path
import sys
import tempfile

import pytest

from ptb_desktop.file_provider import FileAccessError, checked_path
from ptb_desktop.m05_bridge import _scratch_directory


def test_actual_system_scratch_is_canonical_and_admitted(record_property):
    original = Path(tempfile.gettempdir())
    canonical = original.resolve(strict=True)
    record_property('system_temp', str(original))
    record_property('canonical_system_temp', str(canonical))
    if sys.platform == 'darwin':
        # Read the existing system alias directly. No mocked temp root or
        # synthetic link replaces the path that caused the real regression.
        assert Path('/var').is_symlink()
        assert Path('/var/tmp').samefile('/private/var/tmp')
        with pytest.raises(FileAccessError, match='链接或重解析点'):
            checked_path(Path('/var/tmp'))
    with _scratch_directory('ptb-m05-owned-regression-') as name:
        scratch = Path(name)
        assert scratch.parent == canonical
        assert checked_path(scratch) == scratch
        media = scratch / 'owned-source.media'
        media.write_bytes(b'owned m05 scratch fixture')
        assert checked_path(media) == media
        assert media.read_bytes() == b'owned m05 scratch fixture'
    assert not scratch.exists()
