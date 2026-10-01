"""Read back P17 real outputs without modifying sources or databases."""
import csv,hashlib,json,sqlite3,sys,math
from pathlib import Path
import numpy as np
from scipy.io import wavfile
sys.path.append(str(Path(__file__).resolve().parents[1]/'.venv/m03-compatible/Lib/site-packages'))
from PIL import Image,ImageStat
from openpyxl import load_workbook
out=Path(sys.argv[1]).resolve(); saved=out/'saved';checks=[]
short=next(p for p in (out/'inputs').glob('*.wav') if not p.name.startswith('EGG-'));fs,raw=wavfile.read(short)
segments=json.loads((saved/'segments.ptb.json').read_text('utf8'))
for seg in segments['segments']:
 f=next(f for f in segments['files'] if f['format']=='wav' and f['segment_index']==seg['interval_index'])
 rate,data=wavfile.read(saved/f['name']);assert rate==fs and np.array_equal(data,raw[seg['first_sample']:seg['last_sample']]);checks.append({'segment':seg['label'],'samples':len(data),'exact':True})
for f in saved.glob('*.ptb.sqlite'):
 conn=sqlite3.connect('file:'+str(f)+'?mode=ro',uri=True);cur=conn.execute('select * from params');columns=tuple(x[0] for x in cur.description);db=cur.fetchall();conn.close();xlsx=f.with_name(f.name.replace('.ptb.sqlite','.xlsx'));wb=load_workbook(xlsx,read_only=True,data_only=True);rows=list(wb.active.values);wb.close();assert rows[0]==columns and len(rows)-1==len(db)
 count=0
 for a,b in zip(rows[1:],db):
  for x,y in zip(a,b):
   if isinstance(x,(int,float)) and isinstance(y,(int,float)):assert math.isclose(x,y,rel_tol=2e-14,abs_tol=1e-14)
   else:assert x==y or x is None and y==''
   count+=1
 checks.append({'table':f.name,'rows':len(db),'columns':len(columns),'cells':count,'xlsx_sqlite_equal':True})
j=json.loads((saved/(short.stem+'.ptb.json')).read_text('utf8'));assert j['metadata']['computation_revision']=='acoustic/2';checks.append({'acoustic_revision':j['metadata']['computation_revision']})
for f in list(saved.glob('*.png'))+list(out.glob('M02-*.png'))+[out/'M03-inverse-four.png']:
 if not f.exists():continue
 im=Image.open(f).convert('RGB');assert min(im.size)>100 and max(ImageStat.Stat(im).stddev)>1;checks.append({'image':f.name,'size':im.size,'nonblank':True})
lpc=json.loads(next(saved.glob('LPC*.json')).read_text('utf8'));rate,a=wavfile.read(next(saved.glob('LPC*.wav')));mono=raw.astype(np.float64)/32768;expected=mono[lpc['selection']['start_sample']:lpc['selection']['end_sample']];assert rate==fs and np.array_equal(a,expected);checks.append({'lpc_roi_wav_exact':True,'samples':len(a)})
orig=next(saved.glob('*_ORIG.wav'));inv=next(saved.glob('*_IF.wav'));r1,a1=wavfile.read(orig);r2,a2=wavfile.read(inv);assert r1==r2==44100 and len(a1)==len(a2)==22050 and np.isfinite(a1).all() and np.isfinite(a2).all();checks.append({'egg_if_wavs':len(a1),'finite':True})
with next(saved.glob('*_DATA.csv')).open(encoding='utf-8-sig',newline='') as f:
 rows=list(csv.reader(f));assert len(rows)>2 and len(rows[0])>=5;checks.append({'egg_csv_rows':len(rows)-1,'columns':rows[0]})
checks.append({'originals':json.loads((out/'originals-unchanged.json').read_text('utf8'))});assert checks[-1]['originals']['unchanged']
(out/'readback.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(checks,ensure_ascii=False,indent=2))
