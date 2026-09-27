"""Real optional-runtime probe using a newly generated public input."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests/support'))
from m11_fixture import make_fixture
from phonetic_core.transcription.mfa_name_codec import encode_fs_name
from ptb_worker.mfa.runtime import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--syllables', type=int, default=4)
    args = parser.parse_args()
    root = ROOT / 'output/validation/m11' / ('alignment-' + uuid4().hex)
    audio, text = make_fixture(root / 'public', syllables=args.syllables)
    corpus = root / 'corpus'
    corpus.mkdir()
    for p in (audio, text):
        shutil.copyfile(p, corpus / encode_fs_name(p.name))
    dictionary = root / 'dictionary.dict'
    dictionary.write_text('a\ta˥˥\n', encoding='utf8')
    model = root / 'acoustic.zip'
    shutil.copyfile(args.model, model)
    evidence = {}
    try:
        result = run(args.runtime, root, dict(action='align', model=str(model), dictionary=str(dictionary),
                    config=dict(beam=10,retry_beam=40),expected_files=1), evidence=evidence)
    except Exception as exc:
        result = dict(success=False,error=str(exc))
    (root / 'report.json').write_text(json.dumps(dict(result=result,resources=evidence), ensure_ascii=False, indent=2),encoding='utf8')
    print(json.dumps(dict(success=result['success'],error=result.get('error'),root=str(root),resources=evidence), ensure_ascii=False))


if __name__ == '__main__':
    main()
