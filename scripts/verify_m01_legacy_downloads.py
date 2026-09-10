"""Independent readback of user-associated old tables, downloaded through UI."""
import hashlib
import json
import math
import sqlite3
import numpy as np
from scipy.io import wavfile


def verify_legacy_downloads(out):
    files=[f for f in json.loads((out/'web-downloads.json').read_text('utf-8')) if f['path'].startswith('legacy-')]
    assert len(files)==14
    before=json.loads((out/'legacy-fixtures.json').read_text('utf-8'))['original_hashes']
    assert all(hashlib.sha256((out/'upload'/n).read_bytes()).hexdigest()==h for n,h in before.items())
    rate,samples=wavfile.read(next((out/'upload').glob('声调_*.wav')))
    count=0
    for folder,name in [('legacy-xlsx-downloads','历史参数.xlsx'),('legacy-sqlite-downloads','历史参数.ptb.sqlite')]:
        group=[f for f in files if f['path'].startswith(folder+'/') or f['path'].startswith(folder+'\\')]
        for file in group:assert hashlib.sha256((out/file['path']).read_bytes()).hexdigest()==file['sha256']
        manifest=json.loads((out/next(f['path'] for f in group if f['name']=='segments.ptb.json')).read_text('utf-8'))
        assert manifest['parent_result_sha256'] is None and manifest['reestimated'] is False
        assert manifest['legacy_result']==dict(name=name,sha256=before[name],provenance='user_associated_unverified')
        for segment in manifest['segments']:
            entries=[f for f in group if any(e['name']==f['name'] and e['segment_index']==segment['interval_index'] for e in manifest['files'])]
            actual_rate,actual=wavfile.read(out/next(f['path'] for f in entries if f['name'].endswith('.wav')))
            assert actual_rate==rate;np.testing.assert_array_equal(actual,samples[segment['first_sample']:segment['last_sample']])
            db=(out/next(f['path'] for f in entries if f['name'].endswith('.ptb.sqlite'))).resolve()
            with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as conn:
                assert [r[1] for r in conn.execute('PRAGMA table_info(params)')]==['Time_s','Source_Time_s','pF0','textgrid_音节']
                rows=conn.execute('SELECT * FROM params ORDER BY rowid').fetchall()
                expected=[(0.,0.,120.,'ɑ̃˥'),(.1,.1,None,'')] if segment['interval_index']==0 else [(0.,.4,math.inf,'ʔ'),(.6-.4,.6,-math.inf,'β'),(.79-.4,.79,140.,'上声')]
                assert rows==expected
            from openpyxl import load_workbook
            wb=load_workbook(out/next(f['path'] for f in entries if f['name'].endswith('.xlsx')),read_only=True,data_only=False)
            try:
                restored=list(wb.active)
                assert [c.value for c in restored[0]]==['Time_s','Source_Time_s','pF0','textgrid_音节']
                assert len(restored)==len(expected)+1
                for cells,row in zip(restored[1:],expected):
                    for c,v in zip(cells,row):
                        assert c.data_type!='f'
                        target=('inf' if v>0 else '-inf') if isinstance(v,float) and math.isinf(v) else None if v=='' else v
                        assert c.value==target or isinstance(target,float) and isinstance(c.value,float) and math.isclose(c.value,target,rel_tol=1e-12,abs_tol=1e-12)
            finally:wb.close()
            count+=1
    (out/'legacy-readback.json').write_text(json.dumps(dict(downloads=14,parameter_pairs=count,values_frames_and_samples_exact=True,
        original_v2_writer_inputs_preserved=True,provenance_explicit=True),indent=2),encoding='utf-8')
