"""P16: atomic no-clobber publication and owned-directory cleanup."""
import os
import pytest
from ptb_desktop.directory_io import pin_directory
from ptb_desktop.file_provider import FileAccessError


def test_publish_never_overwrites_a_concurrently_created_destination(tmp_path):
    with pin_directory(tmp_path) as directory:
        with directory.open('candidate.part','xb') as f:f.write(b'new')
        (tmp_path/'result.txt').write_bytes(b'external')
        with pytest.raises(FileExistsError):directory.publish('candidate.part','result.txt')
    assert (tmp_path/'result.txt').read_bytes()==b'external'
    assert (tmp_path/'candidate.part').read_bytes()==b'new'


def test_publish_and_replace_in_pinned_directory(tmp_path):
    with pin_directory(tmp_path) as directory:
        with directory.open('one.part','xb') as f:f.write(b'first')
        directory.publish('one.part','output.txt')
        with directory.open('two.part','xb') as f:f.write(b'second')
        directory.publish('two.part','output.txt',replace=True)
        with directory.open('output.txt','rb') as f:assert f.read()==b'second'
        for name in ('../escape','sub/file','C:\\escape'):
            with pytest.raises(FileAccessError):directory.open(name,'xb')


@pytest.mark.skipif(os.name!='posix',reason='POSIX filesystem behavior requires a POSIX host')
def test_renamed_root_cannot_redirect_publication_or_cleanup(tmp_path):
    root=tmp_path/'root';root.mkdir();moved=tmp_path/'moved'
    with pin_directory(root) as directory:
        with directory.open('owned.part','xb') as f:f.write(b'owned')
        root.rename(moved);root.mkdir();(root/'owned.part').write_bytes(b'external')
        with pytest.raises(FileAccessError):directory.publish('owned.part','out')
        directory.unlink('owned.part')
    assert (root/'owned.part').read_bytes()==b'external'
    assert not (moved/'owned.part').exists()


@pytest.mark.skipif(os.name!='posix',reason='POSIX symlink behavior requires a POSIX host')
def test_symlinks_and_hardlinks_are_not_followed(tmp_path):
    outside=tmp_path/'outside';outside.write_bytes(b'private')
    root=tmp_path/'root';root.mkdir();(root/'link').symlink_to(outside)
    os.link(outside,root/'hardlink')
    with pin_directory(root) as directory:
        for name in ('link','hardlink'):
            with pytest.raises((OSError,FileAccessError)):directory.open(name,'rb')
