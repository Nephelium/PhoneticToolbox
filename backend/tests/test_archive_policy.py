import io
import struct
import zipfile
import pytest
from ptb_worker.archives import inspect_zip, entry_name, archive as write_archive
from ptb_api.quota import StorageError


def archive(name='sound.wav', data=b'RIFF-test'):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(name, data)
    stream.seek(0)
    return stream


def test_exact_safe_content_and_bounded_directory():
    stream = archive()
    inspect_zip(stream, len(stream.getvalue()))
    with zipfile.ZipFile(stream) as z:
        assert entry_name(z.infolist()[0]) == 'sound.wav'
        assert z.read('sound.wav') == b'RIFF-test'
    oversized = bytearray(stream.getvalue())
    offset = oversized.rfind(b'PK\x05\x06')
    struct.pack_into('<I', oversized, offset+12, 10_000_000)
    with pytest.raises(StorageError):
        inspect_zip(io.BytesIO(oversized), len(oversized))


@pytest.mark.parametrize('name', ['../escape','/root','C:drive','a\\b','a/../b','a//b'])
def test_entry_path_rejection(name):
    with pytest.raises(StorageError):
        entry_name(zipfile.ZipInfo(name))


def test_links_encryption_and_unsupported_method_rejected():
    for change in ('link','encrypted','method'):
        item = zipfile.ZipInfo('file')
        if change == 'link': item.external_attr = 0o120777 << 16
        if change == 'encrypted': item.flag_bits = 1
        if change == 'method': item.compress_type = zipfile.ZIP_LZMA
        with pytest.raises(StorageError): entry_name(item)


def test_unsupported_large_zip_rejected_before_output_creation():
    class NoOutput:
        def output(self,*args): raise AssertionError('Must reject before output creation')
    with pytest.raises(StorageError,match='archive_rejected'):
        write_archive(NoOutput(),None,[{'name':'large.bin','size_bytes':zipfile.ZIP64_LIMIT+1}],None)
