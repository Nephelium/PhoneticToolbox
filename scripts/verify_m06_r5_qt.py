"""R5 actual Qt source loading, explicit extraction, clipboard and layout."""
import json
import sqlite3
from PyQt6.QtCore import QMimeData
from PyQt6.QtWidgets import QApplication


def verify(window,out,db,js,click,until,pause,report):
    def capture(name):
        window.view.update();pause(700);window.view.grab().save(str(out/name))
    def close_rules():
        js('document.querySelector("[aria-label=关闭对话框]")?.click()')
        until('!document.querySelector(".vowel-rules")');pause(200)
    def theme_mode(theme):
        click('设置');until('!!document.querySelector(".mode-choices")');pause(150)
        click(theme);until('[...document.querySelectorAll(".mode-choices button")].some(e=>e.textContent==='+json.dumps(theme)+'&&e.getAttribute("aria-pressed")==="true")')
        pause(200);click('语音合成');until('!!document.querySelector(".m06-page")?.offsetParent');pause(300)
    def count():
        with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as conn:return conn.execute('select count(*) from jobs').fetchone()[0]
    def selected(index):
        js('(()=>{const e=document.querySelector(".source-picker");e.selectedIndex='+str(index)+';e.dispatchEvent(new Event("change",{bubbles:true}))})()')
    disabled='[...document.querySelectorAll(".m06-page button")].find(e=>e.textContent==="提取参数").disabled'
    assert js(disabled)
    assert not js('!!document.querySelector("[aria-label=合成方法]")')
    original=count();duration=js('document.querySelector("[aria-label=总时长]").value')
    click('打开音频目录');until('document.querySelector(".source-picker").options.length>1');selected(1)
    until('document.body.innerText.includes("音频已加载")')
    assert count()==original and not js(disabled)
    assert js('document.querySelector("[aria-label=总时长]").value')==duration
    click('播放选区');pause(80);click('停止')
    selected(0);pause(100);assert js(disabled)
    selected(1);until('document.body.innerText.includes("音频已加载")');assert count()==original
    click('提取参数');until('document.body.innerText.includes("参数提取完成")');assert count()==original+1
    report['checks'].append('R5 actual Qt: no-source disabled, load/clear/reload do not submit jobs; explicit extraction submits exactly one')
    clipboard=QApplication.clipboard();saved=QMimeData()
    current=clipboard.mimeData()
    if current:
        for mime in current.formats():saved.setData(mime,current.data(mime))
    try:
        theme_mode('浅色')
        click('元音规则');until('!!document.querySelector(".vowel-rules")')
        symbols=js('[...document.querySelectorAll(".vowel-rules button")].map(e=>e.textContent)')
        for symbol in symbols:
            js('[...document.querySelectorAll(".vowel-rules button")].find(e=>e.textContent==='+json.dumps(symbol)+').click()')
            until('document.querySelector(".copy-feedback").textContent==='+json.dumps('已复制 '+symbol))
            assert clipboard.text()==symbol,(symbol,clipboard.text())
        assert not js('document.body.innerText.includes("Write permission denied")')
        report['clipboard_symbols']=symbols
        rows=js('[...document.querySelectorAll(".vowel-rules tbody tr")].map(e=>e.getBoundingClientRect().height)')
        assert max(rows)<=36,rows
        capture('r5-qt-vowels-light.png')
        close_rules()
    finally:
        clipboard.setMimeData(saved)
    report['checks'].append('R5 actual Qt clipboard: all vowel symbols copied via native bridge and read back; previous clipboard MIME data restored')
    report['r5_layouts']=[]
    for width,height in [(1920,1000),(1440,900),(900,700)]:
        for theme in ['浅色','深色']:
            window.resize(width,height);theme_mode(theme)
            assert not js('!!document.querySelector(".vowel-rules")')
            m=js('''(()=>{const r=document.querySelector('.m06-page'),b=e=>{const q=e.getBoundingClientRect();return [q.x,q.y,q.width,q.height,q.bottom]};return {first:r.querySelector('.workbench-left').firstElementChild.classList.contains('source-section'),root:b(r),bar:b(r.querySelector('.m06-transport')),source:b(r.querySelector('.source-section')),overflow:r.scrollWidth-r.clientWidth}})()''')
            assert m['first'] and m['overflow']<=1 and m['bar'][4]<=m['root'][4]+1,m
            m['background']=js('getComputedStyle(document.documentElement).getPropertyValue("--app").trim()')
            if theme=='深色':assert m['background']!=report['r5_layouts'][-1]['measurement']['background']
            report['r5_layouts'].append(dict(width=width,height=height,theme=theme,measurement=m))
            capture(f'r5-qt-{width}-{theme}.png')
    window.resize(1920,1000);pause(150);click('元音规则');pause(100)
    capture('r5-qt-vowels-dark.png');close_rules()
    click('合成音频');until('document.body.innerText.includes("合成完成")');click('导出音频');until('document.body.innerText.includes("已导出合成结果")')
    meta=json.loads((out/'saved/m06.ptb.json').read_text('utf8'))
    assert meta['computation_revision']=='klatt/2' and 'render' not in meta['config']
    assert set(p.name for p in (out/'saved').iterdir())=={'synthesis.wav','parameters.csv','m06.ptb.json'}
    report['checks'].append('R5 actual Qt: 6 layouts, source selector first; Klatt three-file native save with unchanged calculation revision')
