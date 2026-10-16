"""User-authorized table example, real native workflow, no source workbook edits."""
import base64, hashlib, json
from capture_website_modules import Capture, ROOT, verification_run, Path

def main():
    source=Path(r'C:\Users\13680\Desktop\PhoneticToolbox\测试.xlsx')
    original=hashlib.sha256(source.read_bytes()).hexdigest()
    with verification_run('website','phonology','User-supplied XLSX; fictional tone names; actual native results at 2560x1440 and 150%') as (out,scratch):
        c=Capture(out,scratch)
        try:
            c.until("!!document.querySelector('.home-page')")
            c.palette('gruvbox','light');c.module('音系归纳')
            raw=base64.b64encode(source.read_bytes()).decode()
            name='测试（声调仅为举例，并非真实调类）.xlsx'
            c.js('(()=>{const d=new DataTransfer();d.items.add(new File([Uint8Array.from(atob('+json.dumps(raw)+'),c=>c.charCodeAt(0))],'+json.dumps(name)+'));const e=document.querySelector(".phonology-page input[type=file]");e.files=d.files;e.dispatchEvent(new Event("change",{bubbles:true}));})()')
            c.until('!!document.querySelector("table[aria-label=原始表格样例]")')
            c.report['source_sample']=c.js('document.querySelector("table[aria-label=原始表格样例]").innerText')
            print(c.report['source_sample'][:700],flush=True)
            c.js('(()=>{const e=document.querySelector("[aria-label=数据开始行]");e.value="1";e.dispatchEvent(new Event("input",{bubbles:true}));e.dispatchEvent(new Event("change",{bubbles:true}));})()')
            c.until('document.querySelector(".phonology-page").getAttribute("aria-busy")==="false"')
            c.click('确认导入并继续')
            c.until('document.querySelectorAll(".tone-scroll input").length===6')
            tones={'1':'阴平','2':'阳平','5':'上声','6':'去声','7':'阴入','8':'阳入'}
            actual=c.js('[...document.querySelectorAll(".tone-scroll input")].map(e=>e.getAttribute("aria-label").slice(3))')
            assert set(actual)==set(tones),actual
            for tone,name in tones.items():
                c.js('(()=>{const e=document.querySelector('+json.dumps('input[aria-label="调类 '+tone+'"]')+');e.value='+json.dumps(name)+';e.dispatchEvent(new Event("input",{bubbles:true}));})()')
            c.click('保存调类设置并继续');c.click('保存声韵设置并继续')
            c.click('生成三份结果')
            c.until('document.querySelector(".phonology-page").innerText.includes("三份结果已完整生成")',240)
            c.report['tone_mapping']=tones;c.report['result_summary']=c.js('document.querySelector(".phonology-page").innerText')
            c.click('2 · 调类与调值')
            c.snapshot('phonology-tones-gruvbox-light','M14','User XLSX; six illustrative tone classes, explicitly fictional; actual tone ordering/grouping editor')
            c.click('4 · 结果');c.click('声母 → 韵母DOCX')
            c.js('document.querySelector(".word-sheet .summary").open=false')
            c.snapshot('phonology-initials-gruvbox-light','M14','Completed initial-to-final grouping from user XLSX; original rows preserved; tone labels are examples only')
            c.click('二维声韵表XLSX')
            c.snapshot('phonology-matrix-gruvbox-light','M14','Completed two-dimensional initial/final matrix from user XLSX; tone labels are examples only')
            c.palette('tokyo-night','dark');c.click('外观')
            c.snapshot('appearance-tokyo-night-dark','settings-appearance','Appearance settings, dark palette, 150%')
            c.palette('dracula','light');c.click('图标')
            c.snapshot('icons-dracula-light','settings-icons','Current icon choices, light palette, 150%')
            c.palette('temple','dark');c.click('使用说明')
            c.until('document.body.innerText.includes("第一次打开工作台")',90)
            c.snapshot('manual-temple-dark','manual','Actual built-in usage guide, dark palette, 150%')
            assert hashlib.sha256(source.read_bytes()).hexdigest()==original
            c.report['source_sha256']=original;c.report['source_unchanged']=True;c.report['success']=True
            p=ROOT/'temporary-site/assets/screens/provenance.json';data=json.loads(p.read_text('utf-8'))
            names={r['file'] for r in c.report['captures']};data['captures']=[r for r in data['captures'] if r['file'] not in names]+c.report['captures']
            p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n','utf-8')
        except Exception as e:
            c.report['failure']=str(e);c.w.view.grab().save(str(out/'failure.png'));raise
        finally:
            assert hashlib.sha256(source.read_bytes()).hexdigest()==original
            c.close();print(str(out),flush=True)

if __name__=='__main__':main()
