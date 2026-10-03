"""Limited Linux checks using existing runtime. No GUI or job availability claim."""
import hashlib
import json
import struct
from pathlib import Path
import numpy as np
from ptb_worker.egg_exports import f0_axis_range

root=Path(__file__).resolve().parents[1]
checks=[]
for values,expected in [([35,45],(30,50)),([600],(588,612)),([35,45,600,900],(0,987)),([0,np.nan,np.inf,-1],None)]:
    assert f0_axis_range(values)==expected
    checks.append('F0 display range '+str(expected))
for name,fields in [('egginversedata',['full_egg_values','sample_rate_hz','lp_order','gci_count','fixed_window_crossings']),('eggpreviewdata',['suggested_db_range'])]:
    schema=json.loads((root/f'contracts/schemas/{name}.json').read_text('utf-8'))
    assert all(f in schema['properties'] and f not in schema.get('required',[]) for f in fields)
    checks.append(name+' new fields remain optional')
manifest=json.loads((root/'output/m03-r3/qt-png-hashes.json').read_text('utf-8'))
for relative,expected in manifest.items():
    raw=(root/relative).read_bytes();assert hashlib.sha256(raw).hexdigest()==expected
    assert raw[:8]==b'\x89PNG\r\n\x1a\n'
    width,height=struct.unpack('>II',raw[16:24]);assert width>1000 and height>500
    offset=8;phys=None
    while offset<len(raw):
        size=struct.unpack('>I',raw[offset:offset+4])[0]
        if raw[offset+4:offset+8]==b'pHYs':phys=struct.unpack('>IIB',raw[offset+8:offset+17])
        offset+=size+12
    assert phys and phys[2]==1 and abs(phys[0]*.0254-300)<1
    checks.append('unchanged Windows PNG readable: '+Path(relative).name)
report={'platform':'WSL NInfer','checks':checks,'limits':'Display helper / schema / saved PNG only; no Linux M03 job, MKL parity or GUI validation.'}
(root/'output/m03-r3/wsl-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
