"""Actual Linux parameter decoding; synthetic in-memory SQLite, no service DDL."""
import hashlib
import os
import sqlite3
import sys

import pytest

from ptb_worker.native.process import run_fixed_module
from ptb_worker.io.limits import Limits
from ptb_worker.parameter_preview import render
from ptb_worker.spectrogram_preview import PreviewError

REAL = pytest.mark.skipif(sys.platform != 'linux' or os.environ.get('PTB_P11_SYSTEMD_TESTS') != '1',
                          reason='requires explicitly authorized Linux user systemd')


def test_arbitrary_entry_rejected_before_process(tmp_path):
    with pytest.raises(ValueError, match='Unsupported fixed stdio entry'):
        run_fixed_module('os', b'', tmp_path, Limits())


@REAL
def test_linux_table_retains_numeric_null_and_chinese_ipa():
    with sqlite3.connect(':memory:') as connection:
        connection.execute('CREATE TABLE params(Time_s REAL, pF0 REAL, TextGrid TEXT)')
        connection.executemany('INSERT INTO params VALUES(?,?,?)', [(0., 100., '阴平 ə'), (.01, None, '上声 ɕ')])
        connection.commit()
        raw = connection.serialize()
    result = render(raw, '参数.ptb.sqlite')
    assert result['rows'] == [[0., 100., '阴平 ə'], [.01, None, '上声 ɕ']]
    assert result['sha256'] == hashlib.sha256(raw).hexdigest()
    assert result['kinds'] == ['number', 'number', 'text']


@REAL
def test_linux_invalid_table_and_timeout_are_failures():
    with pytest.raises(PreviewError):
        render(b'SQLite format 3\0', 'broken.ptb.sqlite')
    with pytest.raises(PreviewError, match='parameter_read_timeout'):
        render(b'SQLite format 3\0', 'broken.ptb.sqlite', timeout=.0001)
