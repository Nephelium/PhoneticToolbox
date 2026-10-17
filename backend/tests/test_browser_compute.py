"""Browser orchestration keeps the original readers and native runtime guard."""
import hashlib
import sqlite3
import pytest
from ptb_worker.browser_compute import execute


def table(schema='CREATE TABLE params(Time_s REAL,F0_Hz REAL)'):
    c=sqlite3.connect(':memory:');c.execute(schema)
    if 'params' in schema:c.executemany('INSERT INTO params VALUES(?,?)',[(0.,150.),(.01,None)])
    c.commit();value=c.serialize();c.close();return value


def test_browser_table_retains_exact_schema_missing_values_and_input_identity():
    raw=table();value=execute('parameters',raw,{'name':'fixture.ptb.sqlite'})
    assert value==dict(sha256=hashlib.sha256(raw).hexdigest(),columns=['Time_s','F0_Hz'],kinds=['number','number'],rows=[[0.,150.],[.01,None]])
    with pytest.raises(ValueError):execute('parameters',table('CREATE TABLE other(value TEXT)'),{'name':'fixture.ptb.sqlite'})


def test_praat_native_session_still_requires_its_original_fingerprint(monkeypatch):
    from ptb_worker import egg_runtime
    from ptb_worker.egg_interactive_child import Session
    def denied():raise RuntimeError('original-runtime-guard')
    monkeypatch.setattr(egg_runtime,'fingerprint',denied)
    with pytest.raises(RuntimeError,match='original-runtime-guard'):Session(b'not-a-wave')


def test_half_integer_model_prediction_requires_canonical_server_rounding(monkeypatch):
    from ptb_worker import spec2wav_child
    monkeypatch.setattr(spec2wav_child,'preview',lambda *args:dict(prediction=dict(predicted_window_ms=72.50001)))
    with pytest.raises(ValueError,match='canonical rounding'):execute('prediction',b'png',{})


def test_browser_reconstruction_rejects_unported_neural_backend(monkeypatch):
    from ptb_worker import spec2wav_child
    def unexpected(*args):raise AssertionError('Neural computation must stay server-side')
    monkeypatch.setattr(spec2wav_child,'prepare',unexpected)
    with pytest.raises(ValueError,match='Griffin-Lim'):execute('reconstruction',b'png',dict(algorithm='neural_pipeline'))


def test_browser_worker_accepts_only_fixed_operations():
    with pytest.raises(ValueError,match='Unsupported browser operation'):execute('eval',b'',{})


def test_lpc_native_path_keeps_original_runtime_fingerprint(monkeypatch):
    from ptb_worker import lpc_runtime
    from ptb_worker.lpc_child import prepare
    def denied():raise RuntimeError('original-lpc-runtime-guard')
    monkeypatch.setattr(lpc_runtime,'fingerprint',denied)
    with pytest.raises(RuntimeError,match='original-lpc-runtime-guard'):prepare(b'bad',{})


def test_browser_synthesis_does_not_replace_native_extraction():
    with pytest.raises(ValueError,match='extraction is unavailable'):execute('speech_synthesis',b'',dict(action='extract'))


def test_phonology_validates_the_same_server_contract():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):execute('phonology',b'csv',dict(name='调查.csv',config=dict(action='export')))
