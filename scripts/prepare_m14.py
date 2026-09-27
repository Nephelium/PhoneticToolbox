"""Public synthetic inputs only; never opens research data."""
import csv
import json
from pathlib import Path
from openpyxl import Workbook
import xlwt

ROOT = Path(__file__).resolve().parents[1]
out = ROOT/'tests/fixtures/m14'; out.mkdir(exist_ok=True)
rows = [['汉字','音标','备注'],['妈','ma55',''],['麻(文)','ma35','文'],['马','ma214','1'],['怕','pʰa51',''],['巴','pa55',''],['鼻','pã35','鼻化'],['资','tsɿ55',''],['知','ʈʂʅ51',''],['嗯','ŋ̍35',''],['儿','ɻ̍35',''],['鱼','y35',''],['合音','[ ãː５５ ]','2'],['树','t̠ʃʰu51',''],['字','ȶi0',''],['妈','ma55',''],['缺','', '缺失'],['','ma55','缺字'],['阿','a0',''],['检','NA','默认缺失值']]
(out/'recipe.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
for suffix,sep in [('csv',','),('tsv','\t'),('txt','\t')]:
 with (out/('public.'+suffix)).open('w',encoding='utf-8-sig',newline='') as f: csv.writer(f,delimiter=sep).writerows(rows)
w=Workbook();s=w.active
for row in rows:s.append(row)
w.save(out/'public.xlsx')
w=xlwt.Workbook();s=w.add_sheet('公开合成')
for i,row in enumerate(rows):
 for j,v in enumerate(row):s.write(i,j,v)
w.save(str(out/'public.xls'))
(out/'empty.txt').write_text('',encoding='utf-8')
(out/'missing.tsv').write_text('字\n妈\n',encoding='utf-8')
(out/'broken.xlsx').write_bytes(b'not an xlsx')
print(out)
