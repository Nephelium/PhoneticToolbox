"""Verify separately linked macOS images against their local build receipt."""
import hashlib
import json
import platform
from pathlib import Path

VTL_SOURCE_SHA256='aa696bf4cf5c44cbd433304943aebb241e5ea3064b949daf5ea09f96567cff9b'


def validate_pair(synthesis, analysis):
    synthesis,analysis=Path(synthesis),Path(analysis)
    try:
        report=json.loads((synthesis.parent/'native-pair.json').read_text('utf-8'))
        if (report.get('schema')!='ptb-vtl-native-pair/1' or report.get('platform')!='darwin'
                or report.get('architecture')!=platform.machine().lower()
                or report.get('same_source_sha256')!=VTL_SOURCE_SHA256):
            raise ValueError('Native pair provenance unavailable')
        for path in (synthesis,analysis):
            if report['files'].get(path.name)!=hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError('Native pair identity mismatch')
    except (OSError,ValueError,KeyError,TypeError) as error:
        raise RuntimeError('Analysis library does not match its native build receipt') from error
