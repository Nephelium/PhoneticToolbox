"""M01-C Windows synthetic native + formats + independent scientific goldens.

Writes new ignored evidence directories only. Never changes goldens or any user
database. No API/IAB/Qt process is started. Run using the isolated m01-io Python.
"""
from pathlib import Path
import hashlib
import json
import pickle
import sqlite3
import sys
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests/support'))
from baseline_support import RECIPES,create_fixture,load_json,compare
from m01_science import associations,pack
from phonetic_core.models.acoustic import AcousticConfig
from phonetic_core.models.associations import AcousticAssociations
from phonetic_core.ports.acoustic import AcousticBackends
from phonetic_core.services.acoustic import analyze_audio
from phonetic_core.acoustic.catalog import PARAMETER_MAPPING
from ptb_worker.io.audio import decode_wav
from ptb_worker.io.annotations import decode_textgrid
from ptb_worker.io.lip import convert_local_legacy_lip,decode_lip
from ptb_worker.io.scratch import Scratch
from ptb_worker.io.parameter_exports import export_analysis,verify_pair,table_from_frame
from ptb_worker.io.limits import Limits
from ptb_worker.native.reaper import Reaper,REAPER_SHA256


def short_grid(tiers):
    quote=lambda text:'"'+text.replace('"','""')+'"'
    lines=['File type = "ooTextFile short"','"TextGrid"','0','.8','<exists>',str(len(tiers))]
    for tier in tiers:
        lines.extend(['"IntervalTier"',quote(tier.name),'0','.8',str(len(tier.intervals))])
        for interval in tier.intervals:lines.extend([str(interval.xmin),str(interval.xmax),quote(interval.text)])
    return '\n'.join(lines).encode('utf-8')


def main():
    folder=ROOT/'output/validation/m01'/('native-io-'+uuid4().hex);folder.mkdir(parents=True)
    rows=[]
    for case in ('ASSOCIATED','FORMULA-EXPORT'):
        target=folder/case;target.mkdir()
        old=load_json(ROOT/'tests/fixtures/m01'/f'{case}.json.gz')['scientific']
        wav=create_fixture(target,RECIPES[0]);before=hashlib.sha256(wav.read_bytes()).hexdigest()
        source=associations(case=='FORMULA-EXPORT')
        safe=convert_local_legacy_lip(pickle.dumps(source.lip))
        (target/'lip.json').write_bytes(safe)
        tg=short_grid(source.tiers);(target/'labels.TextGrid').write_bytes(tg)
        assoc=AcousticAssociations(lip=decode_lip(safe),tiers=decode_textgrid(tg))
        config=AcousticConfig(**{k:v for k,v in old['config'].items() if k!='reaper_bin_path'})
        with Scratch(target,4_000_000) as scratch:
            native=Reaper(ROOT/'phonetic_toolbox/core/acoustic/reaper.exe',scratch)
            result=analyze_audio(decode_wav(wav.read_bytes()),config,assoc,AcousticBackends(reaper=native))
            frame=result.to_dataframe()
            assert list(frame.columns)==old['columns']
            assert result.time_axis.tolist()==old['time_axis']
            differences=compare(old['tracks'],{str(c):pack(frame[c].to_numpy()) for c in frame})
            assert not differences,differences
            assert result.sampling_rate==44100 and any(e['actual']=='native_reaper' for e in result.backend_events)
            pair=export_analysis(result,scratch)
            display=frame.rename(columns=PARAMETER_MAPPING)
            assert list(pair.columns)==old['export']['display_columns']
            verify_pair(pair,table_from_frame(display,Limits()))
            assert scratch.used==0
            # New synthetic evidence artifacts only, after bounded pair preparation.
            (target/'result.xlsx').write_bytes(pair.xlsx)
            database=target/'result.ptb.sqlite';database.write_bytes(pair.sqlite)
            conn=sqlite3.connect(database.as_uri()+'?mode=ro',uri=True)
            try:assert conn.execute('SELECT COUNT(*) FROM params').fetchone()[0]==len(frame)
            finally:conn.close()
        assert hashlib.sha256(wav.read_bytes()).hexdigest()==before
        rows.append({'case':case,'rows':len(frame),'columns':len(frame.columns),'strict_golden_differences':differences,
            'xlsx_bytes':len(pair.xlsx),'sqlite_bytes':len(pair.sqlite),'native_est_bytes':native.last_output_bytes,
            'xlsx_sha256':hashlib.sha256(pair.xlsx).hexdigest(),'sqlite_sha256':hashlib.sha256(pair.sqlite).hexdigest(),
            'formula_text_preserved':case=='FORMULA-EXPORT','backend_events':result.backend_events})
    assert not any(name.startswith('phonetic_toolbox') for name in sys.modules)
    report={'task':'M01-C','platform':'Windows','native_sha256':REAPER_SHA256,'cases':rows,
            'scope':'Bounded adapters and synthetic paired artifact preparation; durable publication is M01-F'}
    (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'report':str((folder/'report.json').relative_to(ROOT)),'cases':rows},ensure_ascii=False))


if __name__=='__main__':main()
