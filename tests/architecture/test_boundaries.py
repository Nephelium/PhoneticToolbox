import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('boundaries', ROOT / 'scripts/check_architecture.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_forbidden_python_imports():
    for code in ['import PyQt6.QtCore', 'from fastapi import FastAPI',
                 'from . import fastapi', 'import importlib; importlib.import_module("ptb_api")',
                 '__import__("phonetic_toolbox")', 'import socket', 'import pydantic']:
        assert module.check_python(code, 'core'), code


def test_valid_core_and_backend():
    assert not module.check_python('from importlib.metadata import version\nimport math', 'core')
    assert not module.check_python('from phonetic_core import __version__\nimport fastapi', 'backend')
    assert module.check_python('import ptb_desktop', 'backend')
    assert module.check_python('import ptb_api', 'desktop')


def test_frontend_import_and_legacy_paths():
    for code in ['import fs from "node:fs"', 'import x from "../../backend/x"',
                 'const x = import("child_process")', 'const x = require("electron")',
                 'const x = "D:\\\\PhoneticToolbox\\\\PhoneticToolbox_v2"']:
        assert module.check_frontend(code), code
    assert not module.check_frontend('import type { components } from "../../contracts/generated/api"')


def test_unregistered_resource(tmp_path):
    asset = tmp_path / 'frontend/public/new.bin'
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b'unregistered')
    assert module.check_assets(tmp_path, {'resources': []}, set())
