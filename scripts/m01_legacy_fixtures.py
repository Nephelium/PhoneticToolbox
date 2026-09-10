"""Test-only old writer use. Only new synthetic files in the caller-owned folder."""
import hashlib
from pathlib import Path
import pickle
import runpy
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def create_legacy_fixtures(folder):
    source=ROOT.parent/'PhoneticToolbox_v2/phonetic_toolbox/services/io/excel.py'
    before=hashlib.sha256(source.read_bytes()).hexdigest();writer=runpy.run_path(str(source))
    data={'Time_s':[0.,.1,.4,.6,.79],'pF0':[120.,np.nan,np.inf,-np.inf,140.],'textgrid_音节':['ɑ̃˥','','ʔ','β','上声']}
    xlsx=folder/'历史参数.xlsx'
    if xlsx.exists() or xlsx.with_suffix('.ptb.sqlite').exists():raise ValueError('Test fixture already exists')
    writer['save_excel'](xlsx,data);writer['save_fast_parameter_db'](xlsx,pd.DataFrame(data))
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    # Exact numeric shapes emitted by v2 lip_gui: list(np.float64) times and
    # per-frame float32 landmark arrays, primitive metric vectors and metadata.
    lip=folder/'旧唇形.pkl';companion=folder/'旧唇形_timestamps.pkl'
    payload={'absolute_timestamps':list(np.array([100.,100.25,100.5,100.75])),
        'relative_times':list(np.array([0.,.25,.5,.75])),
        'landmarks':[np.zeros((478,3),np.float32) for _ in range(4)],
        'metadata':{'audio_first_frame_time':None,'lip_manual_offset':np.float64(.025)},
        **{key:[1.,2.,3.,4.] for key in ('open','outer_width','area','circularity')}}
    with lip.open('xb') as handle:pickle.dump(payload,handle)
    with companion.open('xb') as handle:pickle.dump({'start_time':100.,'frame_timestamps':[100.]},handle)
    paths=[xlsx,xlsx.with_suffix('.ptb.sqlite'),lip,companion]
    return {'writer_sha256':before,'original_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
