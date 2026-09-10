"""M01-G independent readback of bytes downloaded by the real browser.

Only generated test files are passed to the read-only original v2 I/O functions.
No v2 import is introduced into the application or any production package.
"""
import hashlib
import json
import math
from pathlib import Path
import runpy
import sqlite3
from openpyxl import load_workbook
import numpy as np
from scipy.io import wavfile

ROOT=Path(__file__).resolve().parents[1]


def verify_downloads(out):
    files=json.loads((out/'web-downloads.json').read_text('utf-8'))
    assert len(files)==10
    for file in files:
        path=out/file['path'];assert path.resolve().is_relative_to(out.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest()==file['sha256']
    groups={op:[f for f in files if f['operation']==op] for op in ('acoustic_analysis','textgrid_segment')}
    analysis=groups['acoustic_analysis'];cuts=groups['textgrid_segment']
    wire=json.loads((out/next(f['path'] for f in analysis if f['name']=='result.ptb.json')).read_text('utf-8'))
    numeric={c['key']:c for c in wire['numeric']};text={c['key']:c for c in wire['text']}
    keys=wire['column_order'];columns=[numeric[k]['label'] if k in numeric else k for k in keys]
    rows=[]
    for i,time in enumerate(wire['times_s']):
        row=[]
        for key in keys:
            if key=='Time_s':row.append(time)
            elif key in text:row.append(text[key]['values'][i])
            else:
                col=numeric[key];row.append(col['values'][i] if col['nonfinite'][i]==0 else (None,float('inf'),float('-inf'))[col['nonfinite'][i]-1])
        rows.append(row)
    assert len(wire['metadata']['config']['selection']['keys'])==80 and len(rows)>0
    assert len(numeric)==76  # Four lip metrics require a lip association; no fabricated columns.
    assert set(wire['metadata']['config']['selection']['keys'])-set(numeric)=={'LipOpen','LipCirc','LipWidth','LipArea'}
    # Executed only in this test: actual read-only source, no package imports or writes.
    source=ROOT.parent/'PhoneticToolbox_v2/phonetic_toolbox/services/io/excel.py'
    original=runpy.run_path(str(source));source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    counts=[]
    def equal(expected,actual,excel=False):
        if expected is None:return actual is None or isinstance(actual,float) and math.isnan(actual)
        if isinstance(expected,str):return (expected or '')==(actual or '')
        if math.isinf(expected):return actual==('inf' if expected>0 else '-inf') if excel else actual==expected
        return isinstance(actual,(int,float)) and math.isclose(expected,actual,rel_tol=1e-12,abs_tol=1e-12)
    def pair(entries,cols,expected):
        xlsx=out/next(f['path'] for f in entries if f['name'].endswith('.xlsx'))
        db=out/next(f['path'] for f in entries if f['name'].endswith('.ptb.sqlite'))
        wb=load_workbook(xlsx,read_only=True,data_only=False)
        try:
            sheet=list(wb.active);assert [c.value for c in sheet[0]]==cols
            assert len(sheet)==len(expected)+1
            for cells,row in zip(sheet[1:],expected):
                assert all(c.data_type!='f' and equal(v,c.value,True) for v,c in zip(row,cells))
        finally:wb.close()
        with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as conn:
            assert conn.execute('PRAGMA integrity_check').fetchone()==('ok',)
            assert [x[1] for x in conn.execute('PRAGMA table_info(params)')]==cols
            assert any(x[1]=='idx_params_time' for x in conn.execute('PRAGMA index_list(params)'))
            restored=conn.execute('SELECT * FROM params ORDER BY rowid').fetchall()
            assert len(restored)==len(expected)
            assert all(all(equal(a,b) for a,b in zip(row,actual)) for row,actual in zip(expected,restored))
        old_excel=original['load_excel'](xlsx);old_cols=original['load_fastdb_columns'](db)
        assert list(old_excel)==old_cols==cols
        old_window=original['load_fastdb_window'](db,expected[0][0],expected[-1][0],cols)
        assert old_window is not None and len(old_window)==len(expected)
        for index,row in enumerate(expected):
            assert all(equal(value,old_window.iloc[index,ci]) for ci,value in enumerate(row))
            # pandas restores spreadsheet inf and NaN; compare their scientific meaning.
            assert all(equal(value,old_excel[col][index]) for col,value in zip(cols,row))
        counts.append(dict(rows=len(expected),columns=len(cols),xlsx_sha256=hashlib.sha256(xlsx.read_bytes()).hexdigest(),sqlite_sha256=hashlib.sha256(db.read_bytes()).hexdigest()))
    pair(analysis,columns,rows)
    manifest=json.loads((out/next(f['path'] for f in cuts if f['name']=='segments.ptb.json')).read_text('utf-8'))
    source_wav=next((out/'upload').glob('声调_*.wav'));rate,samples=wavfile.read(source_wav)
    for segment in manifest['segments']:
        index=segment['interval_index'];entries=[f for f in cuts if any(e['name']==f['name'] and e['segment_index']==index for e in manifest['files'])]
        path=out/next(f['path'] for f in entries if f['name'].endswith('.wav'));actual_rate,actual=wavfile.read(path)
        assert actual_rate==rate;np.testing.assert_array_equal(actual,samples[segment['first_sample']:segment['last_sample']])
        offset=segment['first_sample']/rate;end=segment['last_sample']/rate
        selected=[[row[0]-offset,row[0],*row[1:]] for row in rows if offset<=row[0]<end]
        assert len(selected)==segment['parameter_rows']
        pair(entries,['Time_s','Source_Time_s',*columns[1:]],selected)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
    assert all(hashlib.sha256((out/f['path']).read_bytes()).hexdigest()==f['sha256'] for f in files)
    report=dict(downloads=len(files),hashes_match=True,pairs=counts,original_v2_reader_sha256=source_hash,
        v2_reader_columns_values_and_time_window=True,cut_samples_exact=True,formula_labels_literal=True,
        scope='Generated v3 downloads read by original v2 source; not arbitrary historical input import')
    (out/'web-readback.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
