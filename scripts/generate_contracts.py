"""Generate review snapshots without adding duplicate handwritten wire definitions."""
import argparse
import json
from pathlib import Path

from ptb_api.main import create_app
from ptb_api.models import Audio, Selection, Track, Viewport
from ptb_api.acoustic_models import ACOUSTIC_SCHEMAS
from ptb_api.job_models import ResultManifestEnvelope

ROOT = Path(__file__).resolve().parents[1]


def snapshots():
    result = {'contracts/openapi.json': create_app().openapi()}
    for model in (Audio, Selection, Track, Viewport,*ACOUSTIC_SCHEMAS,ResultManifestEnvelope):
        schema = model.model_json_schema()
        schema['$schema'] = 'https://json-schema.org/draft/2020-12/schema'
        result[f'contracts/schemas/{model.__name__.lower()}.json'] = schema
    return {path: json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n'
            for path, value in result.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    drift = []
    for name, content in snapshots().items():
        path = ROOT / name
        if args.check:
            if not path.is_file() or path.read_text('utf-8') != content:
                drift.append(name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8', newline='\n')
    print(json.dumps({'schema_drift': drift}))
    raise SystemExit(bool(drift))


if __name__ == '__main__':
    main()
