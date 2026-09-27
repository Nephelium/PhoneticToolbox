"""Reproducible local offline import at a short path, no rebuild/publication."""
import argparse,json,os,time
from pathlib import Path
from uuid import uuid4
from ptb_worker.mfa.components import ComponentManager,atomic_json
from ptb_worker.mfa.probe import register,publish_registration

def main():
    p=argparse.ArgumentParser()
    for name in ('archive','manifest','model','dictionary'):p.add_argument('--'+name,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]/'output'/('m11c-'+uuid4().hex[:8]);root.mkdir()
    print(root,flush=True);os.environ['PTB_M11_COMPONENT_ROOT']=str(root)
    trusted=json.loads(Path(a.manifest).read_text(encoding='utf8'));prepared={};report=dict(success=False);started=time.monotonic()
    def check(target):
        prepared.update(register(target,a.model,a.dictionary,publish=False))
        return dict(success=True,receipt=prepared['receipt'])
    try:
        target=ComponentManager(root).import_archive(a.archive,trusted,check)
        publish_registration(prepared,root)
        report.update(success=True,target=str(target),resources=prepared['receipt']['resources'])
    finally:
        report['elapsed_seconds']=time.monotonic()-started;atomic_json(root/'import-report.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
