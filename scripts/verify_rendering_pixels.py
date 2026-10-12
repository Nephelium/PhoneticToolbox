"""Check actual captured pixels at sidebar and scrolled IPA glyph locations."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image

def main():
 parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
 root=args.directory.resolve();report=json.loads((root/'report.json').read_text(encoding='utf-8'))
 assert report['success'],'DOM/GPU scenario failed'
 checks=[]
 for frame in report['frames']:
  with Image.open(root/frame['file']) as image:pixels=np.asarray(image.convert('RGB'))
  scale=pixels.shape[1]/frame['viewport'][0];counts=[]
  for item in frame['boxes']:
   x,y,w,h=item['box'];x0,y0,x1,y1=[round(v*scale) for v in (x,y,x+w,y+h)]
   crop=pixels[y0:y1,x0:x1].astype(float)
   count=int((np.max(np.abs(crop-np.array(item['color'])),axis=2)<38).sum());counts.append(count)
  checks.append({'file':frame['file'],'text_pixels':counts,'success':len(counts)>=9 and min(counts)>=8})
 result={'success':all(c['success'] for c in checks),'frames':len(checks),'checks':checks,'scope':'Chromium screenshots from visible default-GPU final EXE; native desktop inspection recorded separately'}
 (root/'pixels.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 assert result['success'],[c for c in checks if not c['success']]
 print(json.dumps({'success':True,'frames':len(checks),'minimumGlyphPixels':min(min(c['text_pixels']) for c in checks)}))

if __name__=='__main__':main()
