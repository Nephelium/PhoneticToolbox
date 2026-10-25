"""Visible, owned Qt acceptance of the actual M19 frontend and local resources."""
import argparse
import hashlib
import json
import math
import os
from datetime import datetime
import faulthandler
from pathlib import Path
import sys
import time
import traceback
import csv
import base64
import re
import shutil
from collections import Counter
import xml.etree.ElementTree as ET
from verification_artifacts import verification_run
from workbench_source import configure

ROOT=Path(__file__).resolve().parents[1]

def verify_datav_regions(data_root,js,click,fill,until,pause,export,shot,report):
    admins=json.loads((data_root/'map-index.json').read_text('utf-8'))
    if not any(a.get('source','').startswith('datav-') for a in admins):return
    index={a['id']:a for a in admins};hainan=index['province:460000'];taiwan=index['province:710000']
    assert hainan['extent'][1]<4
    def snapshot():return json.loads(export('导出工程','.json').read_text('utf-8'))
    def frame(project,extent):
        size=js('(()=>{const e=document.querySelector(".map-target");return [e.clientWidth,e.clientHeight]})()')
        radius=6378137;resolution=156543.03392804097/2**project['view']['zoom']
        def merc(lon,lat):return radius*lon*math.pi/180,radius*math.log(math.tan(math.pi/4+lat*math.pi/360))
        center=merc(*project['view']['center']);corners=[]
        for lon,lat in [(extent[0],extent[1]),(extent[2],extent[3])]:
            x,y=merc(lon,lat);point=[(x-center[0])/resolution+size[0]/2,(center[1]-y)/resolution+size[1]/2]
            assert 0<=point[0]<=size[0] and 0<=point[1]<=size[1],(project['view'],size,extent,point)
            corners.append(point)
        return dict(view=project['view'],size=size,corners=corners,extent=extent)
    click('语言现象');fill('.map-heading label select','1','change');click('中国');pause(600)
    report['datavChinaFrame']=frame(snapshot(),hainan['extent'])
    fill('[aria-label="省级地区"]',hainan['id'],'change');pause(600)
    report['datavHainanFrame']=frame(snapshot(),hainan['extent']);shot('datav-hainan-zengmu-fit')
    report['datavRegions']=[]
    for province,region in [('province:110000','district:110101'),('province:810000','district:810001'),('province:410000','city:419001'),('province:420000','city:429004'),('province:460000','city:469024'),('province:650000','city:659010'),('province:440000','city:441900')]:
        a=index[region];fill('[aria-label="省级地区"]',province,'change');fill('[aria-label="下一级地区"]',region,'change');pause(350)
        assert js('document.querySelectorAll("[aria-label=县级地区]").length===0')
        saved=snapshot();assert saved['settings']['region']==region
        report['datavRegions'].append(dict(id=region,level=a['level'],parent=a['parent'],view=saved['view']))
    fill('[aria-label="省级地区"]',taiwan['id'],'change');pause(600)
    assert js('document.querySelectorAll("[aria-label=下一级地区],[aria-label=县级地区]").length===0')
    assert js('document.body.innerText.includes("台湾省整体查看")')
    report['datavTaiwanFrame']=frame(snapshot(),taiwan['extent'])
    fill('.geography-toolbar label select','province','change');until('!document.querySelector(".map-loading")');pause(350)
    svg_path=export('SVG','.svg');svg_text=svg_path.read_text('utf-8');svg=ET.fromstring(svg_text);ns={'s':'http://www.w3.org/2000/svg'}
    paths=svg.find('s:g',ns).findall('s:path',ns)
    layers=[json.loads((data_root/'maps'/f'{name}.json').read_text('utf-8')) for name in ['world','territory','province']]
    def flatten(geometry):return [g for child in geometry['geometries'] for g in flatten(child)] if geometry['type']=='GeometryCollection' else [geometry]
    geometries=[g for layer in layers for feature in layer['features'] if feature.get('properties',{}).get('boundaryType')!='省界' for g in flatten(feature['geometry'])]
    assert len(paths)==len(geometries)
    # OpenLayers returns its spatial index in random order, not GeoJSON order.
    lines=[path for path in paths if path.get('d') and 'Z' not in path.get('d')]
    areas=[path for path in paths if 'Z' in path.get('d')]
    assert len(lines)==sum(g['type'] in ('LineString','MultiLineString') for g in geometries)
    assert len(areas)==sum(g['type'] in ('Polygon','MultiPolygon') for g in geometries)
    for path in lines:assert path.get('fill')=='none'
    face_order=[i for i,p in enumerate(paths) if p.get('fill')=='#f8f8f1']
    line_order=[i for i,p in enumerate(paths) if p in lines]
    symbol_order=[i for i,p in enumerate(paths) if p.get('fill')=='#526c78']
    assert len(face_order)==1 and len(line_order)==4 and len(symbol_order)==10
    assert max(face_order)<min(line_order) and max(line_order)<min(symbol_order)
    assert sum(p.get('stroke-dasharray')=='4 3' for p in lines)==1
    report['datavPainterPriority']=dict(area=face_order,lines=line_order,symbols=symbol_order,disputedDashPaths=1)
    parts=next(f for f in layers[2]['features'] if f['id']==taiwan['id'])['geometry']['coordinates']
    rings=sum(len(part) for part in parts);vertices=sum(len(ring) for part in parts for ring in part)
    matches=[path for path in paths if path.get('d').count('M')==rings and len(re.findall('[ML]',path.get('d')))==vertices]
    assert len(parts)==14 and len(matches)==1
    frame_rect=svg.find('.//s:clipPath/s:rect',ns);bounds=[float(frame_rect.get('width')),float(frame_rect.get('height'))]
    assert all(0<=value<=bounds[i%2] for i,value in enumerate(map(float,re.findall(r'-?[\d.]+',matches[0].get('d')))))
    assert 'DataV GeoAtlas / 高德开放平台' in svg_text
    report['datavTaiwanParts']=14;report['datavOpenLinePaths']=sum(g['type'] in ('LineString','MultiLineString') for g in geometries);shot('datav-taiwan-province')
    fill('.geography-toolbar label select','auto','change');fill('[aria-label="省级地区"]','','change');pause(500)
    report['checks'].append('Reviewed real GeoAtlas data: China/Hainan fit south to Zengmu; direct district/source-city choices; Taiwan province with 14 actual SVG components and no fabricated children; all actual open boundary SVG paths have fill=none')

def run(out,scratch,bundle=None,client_check=False,layout_check=False,frontend_dist=None,data_directory=None,jiyuan_check=False,renderer_check=False,geography_check=False,interaction_check=False):
    if bundle is None:configure()
    bundle=bundle or ROOT
    from PyQt6.QtCore import QEventLoop,QTimer,Qt,QPoint,QCoreApplication,QEvent
    from PyQt6.QtWidgets import QApplication,QFileDialog
    from PyQt6.QtTest import QTest
    from PyQt6.QtGui import QImage
    from ptb_desktop.host import Workbench,register_scheme
    register_scheme()
    app=QApplication(['M19-owned-QA'])
    frontend=Path(frontend_dist).resolve() if frontend_dist is not None else bundle/'frontend/dist'
    explicit_data=Path(data_directory).resolve() if data_directory is not None else None
    window=Workbench(frontend,test=True,jobs_path=None,vocal_profile=scratch/'vocal')
    if explicit_data is not None:window.assets.geolinguistics_root=explicit_data
    window.setWindowTitle('PhoneticToolbox 3.0 · 自动验证（测试窗口）')
    def mark_test_window(ok):
        if ok:window.page.runJavaScript('''(()=>{document.title='PhoneticToolbox · 自动验证（测试窗口）';if(!window.__qaClicks){window.__qaClicks=[];document.addEventListener('click',event=>{const el=event.target.closest?.('button,label,input,select');if(el){window.__qaClicks.push({time:Date.now(),trusted:event.isTrusted,tag:el.tagName,text:(el.textContent||el.getAttribute('aria-label')||'').trim().slice(0,80)});window.__qaClicks=window.__qaClicks.slice(-40);}},true);}let badge=document.getElementById('ptb-automation-badge');if(!badge){badge=document.createElement('div');badge.id='ptb-automation-badge';document.body.appendChild(badge);}badge.textContent='自动验证 · 测试窗口';badge.style.cssText='position:fixed;right:10px;bottom:4px;z-index:2147483647;background:#9b3a18;color:#fff;padding:4px 10px;border-radius:4px;font:13px sans-serif;pointer-events:none';})()''')
    window.page.loadFinished.connect(mark_test_window)
    window.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating,True)
    window.resize(1440,900);window.show()
    report={'success':False,'checks':[],'scope':'visible actual Qt, isolated profile; '+('frozen EXE' if getattr(sys,'frozen',False) else 'current source; no final EXE'),
            'inputs':{'frontend':str(frontend),'frontendIndexSha256':hashlib.sha256((frontend/'index.html').read_bytes()).hexdigest()}}
    if explicit_data is not None:report['inputs'].update(dataDirectory=str(explicit_data),dataManifestSha256=hashlib.sha256((explicit_data/'data-manifest.json').read_bytes()).hexdigest())
    trace_stream=(out/'python-stacks.log').open('w',encoding='utf-8')
    faulthandler.enable(trace_stream);faulthandler.dump_traceback_later(30,repeat=True,file=trace_stream)
    def phase(stage,**values):
        with (out/'phases.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps({'time':time.time(),'localTime':datetime.now().astimezone().isoformat(timespec='milliseconds'),'pid':os.getpid(),'stage':stage,**values},ensure_ascii=False)+'\n')
    phase('owned-window-created',title=window.windowTitle(),visible=window.isVisible())
    report['ownedWindow']={'pid':os.getpid(),'title':window.windowTitle(),'createdAt':datetime.now().astimezone().isoformat()}
    save_dialog=QFileDialog.getSaveFileName
    chosen=[]
    def save_destination(*args,**kwargs):
        # Each export must create fresh bytes, including repeated filenames.
        path=scratch/(str(len(chosen)+1)+'-'+Path(args[2]).name);chosen.append(path);return str(path),''
    QFileDialog.getSaveFileName=save_destination
    def download_diagnostic(item):
        phase('download-requested',name=item.suggestedFileName())
        item.stateChanged.connect(lambda state:phase('download-state',name=item.suggestedFileName(),state=str(state),reason=item.interruptReasonString()))
    window.profile.downloadRequested.connect(download_diagnostic)
    def pause(ms=120):
        loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()
    def js(code):
        started=time.monotonic();phase('javascript-start',code=code[:240])
        loop=QEventLoop();result=[]
        window.page.runJavaScript(code,lambda value:(result.append(value),loop.quit()))
        QTimer.singleShot(10000,loop.quit);loop.exec()
        phase('javascript-end',elapsed=time.monotonic()-started,returned=bool(result))
        assert result,'JavaScript timeout'
        return result[0]
    def until(code,seconds=40):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if js(code):return
            pause()
        raise AssertionError(code+'\n'+str(js('document.body.innerText.slice(-2500)')))
    def click(text):
        result=js('(()=>{const label='+json.dumps(text)+';const scope=["语音","词汇","语法"].includes(label)?".volume-tabs button":"button";const buttons=[...document.querySelectorAll(scope)].filter(e=>e.offsetParent&&!e.disabled);const e=buttons.find(e=>e.textContent.trim()===label||e.getAttribute("aria-label")===label);if(!e)return null;const chosen={label:e.textContent.trim(),className:e.className};e.click();return chosen})()')
        assert result,text
        phase('clicked-control',requested=text,actual=result)
        pause()
    def fill(selector,value,event='input'):
        assert js('(()=>{const e=document.querySelector('+json.dumps(selector)+');if(!e)return false;e.value='+json.dumps(value)+';e.dispatchEvent(new Event('+json.dumps(event)+',{bubbles:true}));return true})()'),selector
        pause()
    def shot(name):
        phase('screenshot-start',name=name)
        pause(700);window.grab().save(str(out/(name+'.png')))
        phase('screenshot-end',name=name)
    def check_buttons():
        sizes=js('JSON.stringify([...document.querySelectorAll(".language-atlas button,.language-atlas .file-button")].filter(e=>e.offsetParent&&!e.closest(".ol-control")&&e.textContent.trim()).map(e=>({label:e.textContent.trim().slice(0,50),size:getComputedStyle(e).fontSize,weight:getComputedStyle(e).fontWeight,source:e.classList.contains("map-sources-toggle")})))')
        values=json.loads(sizes)
        tools=[v for v in values if not v['source']]
        sources=[v for v in values if v['source']]
        assert tools and len({v['size'] for v in tools})==1,tools
        assert len({v['weight'] for v in tools})==1,tools
        assert len(sources)==js('document.querySelectorAll(".geo-map").length'),sources
        assert sources and all(v['size']=='11px' and v['weight']=='400' for v in sources),sources
        report.setdefault('buttonTypography',[]).append({'tools':tools,'sources':sources})
    def export(label,suffix):
        before=len(chosen);click(label)
        if label in ('SVG','PNG'):
            # Existing scientific checks compare every legend entry. Select that
            # content explicitly; separate options acceptance checks the defaults.
            js('(()=>{const e=document.querySelector(".export-options>label input");e.checked=true;e.dispatchEvent(new Event("change",{bubbles:true}));})()')
            js('(()=>{const label=[...document.querySelectorAll(".geography-toolbar label")].find(e=>e.textContent.trim()==="地名");if(label?.querySelector("input")?.checked){const e=document.querySelector("[aria-label=导出地名]");e.value="all";e.dispatchEvent(new Event("change",{bubbles:true}));}})()')
            click('导出图片')
        end=time.monotonic()+30
        while time.monotonic()<end:
            if len(chosen)>before and chosen[-1].is_file() and chosen[-1].stat().st_size>0:
                pause(800);assert chosen[-1].suffix==suffix;return chosen[-1]
            pause()
        phase('export-failed',minimized=window.isMinimized(),active=window.isActiveWindow(),diagnostic=js('JSON.stringify({visibility:document.visibilityState,fonts:document.fonts.status,error:document.querySelector(".atlas-error")?.textContent,body:document.body.innerText.slice(0,1400),canvas:[...document.querySelectorAll("canvas")].map(e=>[e.width,e.height])})'))
        raise AssertionError('Actual download missing: '+label+' / '+str(js('document.body.innerText.slice(-1200)')))
    def import_file(selector,name,text):
        js('(()=>{const e=document.querySelector('+json.dumps(selector)+'),dt=new DataTransfer();dt.items.add(new File(['+json.dumps(text)+'],'+json.dumps(name)+',{type:"text/plain"}));e.files=dt.files;e.dispatchEvent(new Event("change",{bubbles:true}));})()');pause(500)
    def choose_feature(feature_id):
        fill('[aria-label="检索语言特征"]','')
        click({'P':'语音','L':'词汇','S':'语法'}[feature_id[-1]])
        fill('[aria-label="检索语言特征"]',feature_id)
        until('document.querySelectorAll(".feature-list button").length===1')
        assert js('(()=>{const e=document.querySelector(".feature-list button");if(!e)return false;e.click();return true})()')
        pause()
    def check_broad_exports(feature_id,site_ids=None,overrides=None):
        """Independent expected: source cells and explicit memberships, never UI grouping helpers."""
        f=next(f for f in source_catalog if f['id']==feature_id)
        members={code:group for group in f.get('groups',[]) for code in group['codes']}
        counts=Counter();expected_rows={};legend_items={};pending_count=0
        for index,site in enumerate(source_sites):
            if site_ids is not None and site['id'] not in site_ids:continue
            original=source_observations[feature_id]['values'][index]
            value=(overrides or {}).get(site['id'],original)
            raw=source_observations[feature_id]['raw'].get(str(index))
            keys=[]
            if not value or not value.strip():
                keys=['待核'];legend_items['待核']=('原值待核','原值待核','原值待核')
            else:
                exception=f.get('groupExceptions',{}).get(site['id'])
                exceptions=exception if isinstance(exception,list) else [exception] if exception else []
                for code in value.split('#'):
                    if any(item['code']==code for item in exceptions):
                        key='@source-conflict'
                        if key not in keys:keys.append(key)
                        legend_items[key]=('原图对应待核','原图对应待核','原图对应待核')
                        continue
                    group=members.get(code)
                    key='group:'+group['id'] if group else '@unmapped-broad'
                    if key not in keys:keys.append(key)
                    legend_items[key]=(group['label'],group['label'],group['label']) if group else ('此记录的大类尚待对应','此记录的大类尚待对应','此记录的大类尚待对应')
            counts.update(key for key in keys if key.startswith('group:'))
            pending_count+=any(not key.startswith('group:') for key in keys)
            expected_rows[site['id']]=(value or '','；'.join(legend_items[key][0] for key in keys),str(raw['raw']) if raw is not None and 'raw' in raw else original or '')
        actual=json.loads(js('JSON.stringify([...document.querySelectorAll(".legend button")].map(e=>({label:e.querySelector(".legend-word").textContent.trim(),title:e.title,count:Number(e.querySelector(".legend-count").textContent.replace(/\\D/g,""))})))'))
        expected=sorted((legend_items[key][0],legend_items[key][1],count) for key,count in counts.items())
        assert sorted((item['label'],item['title'],item['count']) for item in actual)==expected,(feature_id,actual,expected)
        assert all(not re.search(r'\b[A-Z]\d+\b',item['label']+' '+item['title']) for item in actual)
        assert js('!document.querySelector(".table-scroll .category-code,.atlas-details>p .category-code")')
        table=export('导出表格','.csv')
        with table.open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
        assert len(rows)==len(expected_rows) and len({row['地点编号'] for row in rows})==len(rows)
        for row in rows:
            assert (row['类别编码'],row['具体含义'],row['原始值'])==expected_rows[row['地点编号']],(feature_id,row,expected_rows[row['地点编号']])
            assert row['显示层级']=='大类' and row['原书PDF页']==str(f['page'])
        svg=ET.fromstring(export('SVG','.svg').read_text(encoding='utf-8'))
        texts=''.join(''.join(e.itertext()) for e in svg.findall('.//{http://www.w3.org/2000/svg}text'))
        assert '大类分布' in texts and f'PDF 第 {f["page"]} 页' in texts
        assert not any((e.text or '')=='北京' for e in svg.findall('.//{http://www.w3.org/2000/svg}text')), 'Selected location label must follow the place-name switch'
        assert not re.search(r'\b[A-Z]\d+\b|（大类 [A-Z]）',texts),texts[-2000:]
        for key,count in counts.items():assert legend_items[key][2]+f' · {count} 个地点' in texts,(feature_id,key,count,texts[-2000:])
        assert '尚未对应的大类' not in texts
        if pending_count:
            assert f'{pending_count} 个地点含待核记录' in texts
            assert js('document.querySelector(".pending-records")?.textContent.includes('+json.dumps(str(pending_count)+' 个地点')+')')
        report.setdefault('broadChecks',[]).append({'feature':feature_id,'sites':len(rows),'legend':actual,'csvRawCellsAndMeanings':True,'svgLegendMatches':True})
        return rows,actual
    try:
        until('document.querySelector(".host-badge")?.textContent==="本地桌面"')
        if client_check:
            from verify_compact_pages import verify as verify_clients
            report['scope']='current-source precheck of corrected compact client acceptance'
            verify_clients(window,scratch,report)
            report['success']=True
            return
        click('语言地图')
        until('document.querySelector(".atlas-catalog")?.textContent.includes("3666")&&document.querySelectorAll(".ol-layer canvas").length>=1')
        assert js('document.querySelector(".mode-tabs button").textContent==="方言分布"&&document.querySelector(".mode-tabs button").getAttribute("aria-selected")==="true"')
        if interaction_check:
            assert explicit_data is not None
            report['scope']='Current shared source, actual Windows Qt WebEngine; real trial2 scale/exports and native map fullscreen. No final EXE or Mac claim.'
            js('[...document.querySelectorAll("dialog button")].find(b=>b.textContent.trim()==="知道了")?.click()')
            for theme in ['light','dark']:
                js('document.documentElement.dataset.theme='+json.dumps(theme))
                for level,size in [('province',34),('city',370),('district',2875)]:
                    fill('.geography-toolbar label select',level,'change');until('!document.querySelector(".map-loading")');pause(350)
                    svg=ET.fromstring(export('SVG','.svg').read_text('utf-8'));ns={'s':'http://www.w3.org/2000/svg'}
                    paths=svg.findall('.//s:g[@clip-path]/s:path',ns)
                    assert len(paths)==size+14,(level,len(paths))
                    assert not any(p.get('fill')=='#edf0ed' for p in paths)
                    assert sum(p.get('fill')=='#526c78' for p in paths)==10
                    report['checks'].append(dict(theme=theme,level=level,paths=len(paths),singleScale=True))
                shot('single-scale-'+theme)
            click('语言现象');choose_feature('F120_P')
            def native_click(label):
                position=js('(()=>{const e=[...document.querySelectorAll("button")].find(b=>b.offsetParent&&b.textContent.trim()==='+json.dumps(label)+');if(!e)return null;const r=e.getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2]})()')
                assert position,label
                QTest.mouseClick(window.view.focusProxy() or window.view,Qt.MouseButton.LeftButton,pos=QPoint(round(position[0]),round(position[1])))
                pause(250)
            for count in [1,2,4]:
                fill('.map-heading label select',str(count),'change');until('!document.querySelector(".map-loading")');pause(350)
                before=json.loads(export('导出工程','.json').read_text('utf-8'))['view']
                original_size=(window.width(),window.height());native_click('全屏地图')
                until('document.fullscreenElement?.classList.contains("atlas-center")===true');pause(500)
                assert window.isFullScreen()
                assert js('document.querySelectorAll(":fullscreen .geo-map").length')==count
                shot('fullscreen-'+str(count))
                if count==2:QTest.keyClick(window.view.focusProxy() or window.view,Qt.Key.Key_Escape)
                else:native_click('退出全屏')
                until('!document.fullscreenElement');pause(500)
                assert not window.isFullScreen()
                assert (window.width(),window.height())==original_size
                after=json.loads(export('导出工程','.json').read_text('utf-8'))['view']
                assert before==after,(before,after)
                report['checks'].append(dict(fullscreenMaps=count,nativeExit='Escape' if count==2 else 'button',windowAndViewRestored=True))
            report['success']=True;return
        if geography_check:
            from ptb_desktop.startup_cache import external_executable
            from ptb_desktop.web_assets import geolinguistics_directory
            audit_data=explicit_data or geolinguistics_directory(frontend,external_executable(),frozen=getattr(sys,'frozen',False))
            assert audit_data is not None
            report['scope']='Explicit historical geographic UI audit; excluded from default full and current renderer acceptance'
            verify_datav_regions(audit_data,js,click,fill,until,pause,export,shot,report)
            report['success']=True;return
        if renderer_check:
            click('语言现象');choose_feature('F1_P')
            fill('.geography-toolbar label select','auto','change')
            report['rendererCases']=[]
            export('导出工程','.json');pause(300)
            for theme in ['light','dark']:
                js('document.documentElement.dataset.theme='+json.dumps(theme))
                for step,(level,size) in enumerate([('province',34),('city',370),('district',2875),('province',34)]):
                    for _ in range(12):
                        saved=json.loads(export('导出工程','.json').read_text('utf-8'));zoom=saved['view']['zoom']
                        actual='province' if zoom<6 else 'city' if zoom<8 else 'district'
                        if actual==level:break
                        lower={'province':0,'city':6,'district':8}[level]
                        js('document.querySelector('+json.dumps('.ol-zoom-in' if zoom<lower else '.ol-zoom-out')+').click()');pause(400)
                    assert actual==level and saved['settings']['level']=='auto'
                    until('!document.querySelector(".map-loading")');pause(400)
                    svg_file=export('SVG','.svg');svg=ET.fromstring(svg_file.read_text('utf-8'));ns={'s':'http://www.w3.org/2000/svg'};paths=svg.find('s:g',ns).findall('s:path',ns)
                    lines=[p for p in paths if p.get('d') and 'Z' not in p.get('d')]
                    assert len(paths)==14+size and len(lines)==4
                    assert sum(p.get('stroke')=='#9aa9aa' for p in paths)==size
                    assert sum(p.get('fill')=='#526c78' for p in paths)==10
                    assert sum(p.get('stroke-dasharray')=='4 3' for p in lines)==1
                    # Structural renderer fingerprints from frozen trial2. No
                    # coordinate, source seam or geographic-content comparison.
                    fingerprints={'国界':(14,6392),'争议':(1,105),'港澳':(2,52),'海洋':(362,6267),'省界':(94,18724)}
                    presence={name:sum(p.get('d').count('M')==segments and len(re.findall('[ML]',p.get('d')))==vertices for p in lines) for name,(segments,vertices) in fingerprints.items()}
                    assert presence=={'国界':1,'争议':1,'港澳':1,'海洋':1,'省界':0},presence
                    row=dict(theme=theme,step=step,automaticLevel=level,zoom=zoom,adminFeatures=size,svgPaths=len(paths),boundaryPresence=presence,jdSymbols=10)
                    if level=='district':
                        image=QImage(str(export('PNG','.png')));assert not image.isNull() and image.width()==2*float(svg.get('width')) and image.height()==2*float(svg.get('height'));row['pngSize']=[image.width(),image.height()]
                    shot(f'renderer-{theme}-{level}-{step}');report['rendererCases'].append(row)
            report['checks'].append('Actual Qt automatic zoom province/city/district and back: one admin set, fixed territory province omitted,4other boundary classes and10JD retained, light/dark screen/SVG and PNG exports; renderer counts only')
            report['success']=True;return
        if jiyuan_check:
            assert explicit_data is not None
            click('语言现象');choose_feature('F1_P')
            fill('[aria-label="省级地区"]','province:410000','change')
            fill('[aria-label="下一级地区"]','city:419001','change')
            fill('.geography-toolbar label select','district','change')
            until('!document.querySelector(".map-loading")');pause(500)
            export('导出工程','.json');pause(250)
            for _ in range(8):
                scale=js('document.querySelector(".ol-scale-line-inner").textContent').strip()
                number,unit=re.fullmatch(r'([\d.]+)\s*(km|m)',scale).groups()
                meters=float(number)*(1000 if unit=='km' else 1)
                if meters==10000:break
                js('document.querySelector('+json.dumps('.ol-zoom-in' if meters>10000 else '.ol-zoom-out')+').click()');pause(500)
            assert meters==10000,scale
            until('!document.querySelector(".map-loading")');pause(500)
            project_file=export('导出工程','.json');saved=json.loads(project_file.read_text('utf-8'))
            assert saved['settings']['region']=='city:419001' and saved['settings']['level']=='district'
            native_size=js('(()=>{const e=document.querySelector(".map-target");return [e.clientWidth,e.clientHeight]})()')
            xml_file=export('SVG','.svg');xml=ET.fromstring(xml_file.read_text('utf-8'));ns={'s':'http://www.w3.org/2000/svg'}
            paths=xml.find('s:g',ns).findall('s:path',ns)
            layers={name:json.loads((explicit_data/'maps'/f'{name}.json').read_text('utf-8')) for name in ['world','territory','district']}
            line_evidence=[]
            for f in layers['territory']['features']:
                if f['geometry']['type']!='MultiLineString':continue
                coordinates=f['geometry']['coordinates'];segments=len(coordinates);vertices=sum(map(len,coordinates))
                matches=[p for p in paths if p.get('d').count('M')==segments and len(re.findall('[ML]',p.get('d')))==vertices and 'Z' not in p.get('d')]
                assert len(matches)==1,(f['id'],len(matches))
                p=matches[0];assert p.get('fill')=='none'
                line_evidence.append(dict(id=f['id'],boundaryType=f['properties'].get('boundaryType'),renderRole=f['properties'].get('renderRole'),line=f['properties'].get('line'),segments=segments,vertices=vertices,actualSvgMatches=1,stroke=p.get('stroke'),dash=p.get('stroke-dasharray')))
            assert len(line_evidence)==5
            fallback=next(f for f in layers['district']['features'] if f.get('id')=='city:419001')
            coords=[p for part in fallback['geometry']['coordinates'] for ring in part for p in ring]
            merc=lambda lon,lat:(6378137*lon*math.pi/180,6378137*math.log(math.tan(math.pi/4+lat*math.pi/360)))
            cx,cy=merc(*saved['view']['center']);resolution=156543.03392804097/2**saved['view']['zoom'];factor=1200/native_size[0]
            projected=[((merc(*p)[0]-cx)/resolution*factor+600,(cy-merc(*p)[1])/resolution*factor+native_size[1]*factor/2) for p in coords]
            expected=[min(p[0] for p in projected),min(p[1] for p in projected),max(p[0] for p in projected),max(p[1] for p in projected)]
            def bounds(p):
                nums=list(map(float,re.findall(r'-?[\d.]+',p.get('d'))));return [min(nums[::2]),min(nums[1::2]),max(nums[::2]),max(nums[1::2])]
            candidates=[p for p in paths if 'Z' in p.get('d') and all(abs(a-b)<=.02 for a,b in zip(bounds(p),expected))]
            assert len(candidates)==1 and candidates[0].get('stroke')=='#9aa9aa',(len(candidates),expected)
            assert len(paths)==sum(len(layer['features']) for layer in layers.values())
            copied=[]
            for name,p in [('project',project_file),('map-svg',xml_file),('map-png',export('PNG','.png'))]:
                target=out/('jiyuan-district-10km-'+name+p.suffix);shutil.copyfile(p,target)
                copied.append(dict(path=target.name,bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
            actual_theme=js('document.documentElement.dataset.theme')
            shot('jiyuan-district-10km-'+actual_theme)
            report['jiyuan']=dict(region='city:419001',level='district',scale=scale,theme=actual_theme,view=saved['view'],nativeMapSize=native_size,territoryLineFeatures=line_evidence,
                adminLayer={'name':'district','features':len(layers['district']['features']),'jiyuanSource':fallback['properties'],'actualSvgMatches':1,'actualSvgBounds':bounds(candidates[0]),'expectedSvgBounds':expected},
                layers={k:len(v['features']) for k,v in layers.items()},actualSvgPathCount=len(paths),exports=copied)
            report['checks'].append('Actual Qt Jiyuan district fallback at10km, all5territory line classes and1admin layer independently matched in real SVG; native screen and SVG/PNG/project retained')
            report['success']=True;return
        if layout_check:
            from ptb_desktop.startup_cache import external_executable
            from ptb_desktop.web_assets import geolinguistics_directory
            layout_data=explicit_data or geolinguistics_directory(frontend,external_executable(),frozen=getattr(sys,'frozen',False))
            assert layout_data is not None
            datav_layout=any(a.get('source','').startswith('datav-') for a in json.loads((layout_data/'map-index.json').read_text('utf-8')))
            click('语言现象');choose_feature('F86_S')
            long_title=js('document.querySelector(".map-heading strong").textContent')
            assert len(long_title)>30 and js('document.querySelectorAll(".map-panel-heading").length===0')
            # Establish the export status row before comparing viewport-dependent SVG/PNG dimensions.
            export('导出工程','.json');pause(300)
            report['layouts']=[]
            for theme in ['light','dark']:
                js('document.documentElement.dataset.theme='+json.dumps(theme))
                for count in [2,4]:
                    fill('.map-heading label select',str(count),'change');fill('[aria-label="对照图 2"]','F89_S','change')
                    until('document.querySelectorAll(".map-cell canvas").length>='+str(count));pause(500)
                    sizes=js('(()=>[...document.querySelectorAll(".map-cell")].map(cell=>{const rect=selector=>{const r=cell.querySelector(selector).getBoundingClientRect();return {width:r.width,height:r.height,bottom:r.bottom};};return {gridBottom:cell.parentElement.getBoundingClientRect().bottom,heading:rect(".map-panel-heading"),map:rect(".map-target"),title:cell.querySelector(".map-panel-title,select").title};}))()')
                    assert len(sizes)==count and long_title in sizes[0]['title']
                    for item in sizes:
                        assert item['heading']['height']>=30 and abs(item['heading']['height']-sizes[0]['heading']['height'])<1
                        assert abs(item['map']['width']-sizes[0]['map']['width'])<=1 and abs(item['map']['height']-sizes[0]['map']['height'])<=1 and item['map']['height']>=100 and item['map']['bottom']<=item['gridBottom']+1
                    svg=ET.fromstring(export('SVG','.svg').read_text(encoding='utf-8'));ns={'s':'http://www.w3.org/2000/svg'}
                    panels=svg.findall('s:svg',ns);frames=svg.findall('.//s:clipPath/s:rect',ns)
                    assert len(panels)==count and len(frames)==count
                    for panel in panels:assert [panel.get(k) for k in ['width','height','viewBox']]==[panels[0].get(k) for k in ['width','height','viewBox']]
                    for frame in frames:assert frame.attrib==frames[0].attrib
                    assert long_title in ''.join(''.join(t.itertext()) for t in svg.findall('.//s:text',ns))
                    png=QImage(str(export('PNG','.png')))
                    phase('layout-export-sizes',theme=theme,count=count,svg=[svg.get('width'),svg.get('height')],png=[png.width(),png.height()])
                    assert not png.isNull() and png.width()==2*float(svg.get('width')) and png.height()==2*float(svg.get('height')),(png.width(),png.height(),svg.attrib)
                    report['layouts'].append({'theme':theme,'count':count,'sizes':sizes,'svg':[float(svg.get('width')),float(svg.get('height'))],'png':[png.width(),png.height()]})
                    png.scaledToWidth(1600).save(str(out/f'layout-export-{theme}-{count}.png'));shot(f'equal-panels-{theme}-{count}')
            check_buttons()
            # Send native Qt key events through the actual WebEngine focus widget.
            assert js('document.querySelectorAll(".map-sources-toggle").length===4&&[...document.querySelectorAll(".map-sources-toggle")].every(b=>b.getAttribute("aria-expanded")==="false")')
            def source_key(index,key,link=False):
                window.view.setFocus()
                js('(()=>{const b=document.querySelectorAll(".map-sources-toggle")['+str(index)+'];const e='+('document.getElementById(b.getAttribute("aria-controls")).querySelector("a")' if link else 'b')+';e.focus();})()')
                QTest.keyClick(window.view.focusProxy() or window.view,key);pause(200)
            source_key(0,Qt.Key.Key_Return)
            until('document.querySelectorAll(".map-sources-toggle")[0].getAttribute("aria-expanded")==="true"')
            expected_sources=['DataV GeoAtlas / 高德开放平台','内政部国土测绘中心','Natural Earth'] if datav_layout else ['© OpenStreetMap contributors','内政部国土测绘中心','© 香港特别行政区政府','Huang 等','Natural Earth']
            assert js('(()=>{const b=document.querySelectorAll(".map-sources-toggle")[0],e=document.getElementById(b.getAttribute("aria-controls")),r=e.getBoundingClientRect(),sources='+json.dumps(expected_sources)+';return !e.closest(".geo-map")&&e.querySelectorAll("a").length===sources.length&&sources.every(s=>e.textContent.includes(s))&&r.left>=0&&r.right<=innerWidth&&r.top>=0&&r.bottom<=innerHeight;})()')
            source_key(1,Qt.Key.Key_Space)
            until('document.querySelectorAll(".map-sources-toggle")[1].getAttribute("aria-expanded")==="true"')
            source_key(1,Qt.Key.Key_Escape,True)
            assert js('(()=>{const b=document.querySelectorAll(".map-sources-toggle");return b[0].getAttribute("aria-expanded")==="true"&&b[1].getAttribute("aria-expanded")==="false"&&b[1]===document.activeElement;})()')
            source_key(0,Qt.Key.Key_Escape,True)
            assert js('(()=>{const b=document.querySelectorAll(".map-sources-toggle")[0];return b.getAttribute("aria-expanded")==="false"&&b===document.activeElement;})()')
            report['checks'].append('Actual Qt Enter/Space/Escape source controls, separate popups, source links and focus restoration')
            fill('.map-heading label select','1','change');assert js('document.querySelectorAll(".map-panel-heading").length===0&&document.querySelector(".map-heading strong").textContent==='+json.dumps(long_title))
            report['checks'].append('Double/four panels have equal title bars and map rectangles in light/dark; long title preserved; SVG and PNG panel/frame sizes match; single title unchanged')
            report['success']=True;return
        shot('default-county-classification')
        click('使用方法');until('!!document.querySelector("dialog .help-text")')
        click('打开完整使用说明')
        until('document.querySelector(".manual-document")?.dataset.manualChapter==="m19"&&document.querySelector(".manual-reader")?.getBoundingClientRect().height>200')
        assert js('document.querySelector("#main-content").classList.contains("pane-manual")&&document.querySelector(".language-atlas").getClientRects().length===0&&!document.querySelector(".manual-load-error")')
        assert js('(()=>{const e=document.querySelector(".manual-reading-scroll"),r=e.getBoundingClientRect();return !!document.elementFromPoint(r.left+r.width/2,r.top+Math.min(120,r.height/2))?.closest(".manual-reader")})()')
        # Inspect this chapter's actual resources, including release-compressed
        # derivatives. Count comes from the rendered chapter, never a fixed gate.
        manual_images=js('document.querySelectorAll(".manual-document .manual-media-image").length')
        report['manualImageCount']=manual_images
        for image_index in range(manual_images):
            expression='document.querySelectorAll(".manual-document .manual-media-image")['+str(image_index)+']'
            js(expression+'.scrollIntoView({block:"center"})')
            until('(()=>{const i='+expression+'.querySelector("img");return i?.complete&&i.naturalWidth>0&&i.naturalHeight>0})()')
            js('window.__m19ManualImageDecoded=false;'+expression+'.querySelector("img").decode().then(()=>window.__m19ManualImageDecoded=true)')
            until('window.__m19ManualImageDecoded')
            assert js('!'+expression+'.querySelector(".manual-media-error")')
        if manual_images:
            report['checks'].append({'workflow':'actual M19 manual image decode in current resource format','images':manual_images})
        js('document.querySelector(".manual-reading-scroll").scrollTop=0')
        shot('manual-visible-map-hidden')
        click('返回 语言地图')
        until('document.querySelector(".language-atlas").getBoundingClientRect().height>200&&document.querySelector(".manual-reader").getClientRects().length===0')
        click('首页');until('document.querySelector(".home-page")?.getBoundingClientRect().height>100')
        assert js('document.querySelector("#main-content").classList.contains("pane-home")&&document.querySelector(".language-atlas").getClientRects().length===0&&document.querySelector(".manual-reader").getClientRects().length===0')
        shot('home-visible-map-hidden');click('语言地图')
        until('document.querySelector(".language-atlas").getBoundingClientRect().height>200&&document.querySelectorAll(".ol-layer canvas").length>=1')
        assert js('document.querySelector(".atlas-catalog").textContent.includes("3666")&&document.querySelector(".mode-tabs button").getAttribute("aria-selected")==="true"')
        shot('map-visible-after-home');report['checks'].append('M19 help opens actual m19 manual chapter without map overlay; return-to-tool, home and return preserve map visibility and state')
        from ptb_desktop.startup_cache import external_executable
        from ptb_desktop.web_assets import geolinguistics_directory
        data_root=explicit_data or geolinguistics_directory(frontend,external_executable(),frozen=getattr(sys,'frozen',False))
        assert data_root is not None,'Local geolinguistics data directory is missing'
        if getattr(sys,'frozen',False):
            manifest=json.loads((data_root/'data-manifest.json').read_text('utf-8'))
            assert manifest['profile']=='interactive-data' and len(manifest['files'])==16
            assert not (data_root/'book').exists() and not list(data_root.rglob('*.jpg'))
        source_observations=json.loads((data_root/'observations.json').read_text('utf-8'))
        source_catalog=json.loads((data_root/'catalog.json').read_text('utf-8'))['features']
        source_sites=json.loads((data_root/'sites.json').read_text('utf-8'))
        click('语言现象')
        until('!!document.querySelector("[aria-label=分类层级]")&&document.querySelectorAll(".ol-layer canvas").length>=1')
        assert js('document.querySelector("[aria-label=分类层级]").value==="broad"&&document.querySelector("[aria-label=语言现象范围]").value==="reviewed"')
        readable=[f for f in source_catalog if f.get('groups')]
        assert len(readable)>10
        assert js('document.querySelector("[aria-label=语言现象范围]").textContent.includes('+json.dumps(f'有大类释义 · {len(readable)} 项')+')')
        assert js('document.querySelectorAll('+json.dumps('.atlas-toolbar input[type=file][accept=".json"]')+').length===1&&document.querySelector("[aria-label=选择语言地图资料夹]").hasAttribute("webkitdirectory")')
        for name,volume in [('语音','P'),('词汇','L'),('语法','S')]:
            click(name);total=sum(f['volume']==volume for f in readable)
            until(f'document.querySelectorAll(".feature-list button").length==={total}')
        click('语音');shot('broad-default-reviewed')
        fill('[aria-label="语言现象范围"]','all','change')
        choose_feature('F120_P')
        f120=next(f for f in source_catalog if f['id']=='F120_P')
        assert [(g['id'],g['label']) for g in f120['groups']]==list(zip('ABCDEF',['有[-m]','只有[-n -ŋ]','有[-n -ŋ ã/-0]','只有[-n]或[-ŋ]','有[-n ã/-0]或[-ŋ ã/-0]','无鼻尾']))
        assert f120['groupReview']['complete'] and f120['groupReview']['verifiedCodes']==57
        until('document.querySelectorAll(".legend button").length===6')
        assert js('document.querySelector(".category-coverage").textContent.includes("已对应 6 个大类")&&document.querySelector(".atlas-details details").textContent.includes("57 / 57")')
        check_broad_exports('F120_P');shot('broad-f120-six-groups')
        fill('[aria-label="分类层级"]','fine','change')
        until('document.querySelector("[aria-label=分类层级]").value==="fine"&&document.querySelectorAll(".legend button").length>6')
        assert js('document.querySelector(".category-coverage").textContent.includes("6 / 57")&&document.querySelector(".legend").textContent.includes("咸[m/n]")')
        fill('[aria-label="分类层级"]','broad','change')
        until('document.querySelectorAll(".legend button").length===6')
        choose_feature('F120_P')
        fill('[aria-label="检索调查点"]','S49');until('document.querySelectorAll(".table-scroll tbody tr").length===1')
        js('document.querySelector(".table-scroll tbody tr").click()');pause()
        click('编辑当前特征');fill('dialog .full-input','QA1#QA2');click('保存校订')
        rows,items=check_broad_exports('F120_P',{'S49'},{'S49':'QA1#QA2'})
        assert not items and rows[0]['具体含义']=='此记录的大类尚待对应'
        click('查看待核地点');until('document.querySelectorAll(".table-scroll tbody tr").length===1')
        shot('pending-location-without-grey-category')
        click('编辑当前特征');fill('dialog .full-input','A1#QA1#QA2');click('保存校订')
        rows,items=check_broad_exports('F120_P',{'S49'},{'S49':'A1#QA1#QA2'})
        assert rows[0]['具体含义']=='有[-m]；此记录的大类尚待对应' and len(items)==1 and items[0]['count']==1
        shot('known-group-with-partial-record')
        click('恢复原值');until('document.querySelectorAll(".table-scroll tbody tr").length===0')
        click('全部类别');fill('[aria-label="检索调查点"]','')
        # Confirm user-supplied rare glyphs survive the actual UI and native exports.
        for fid,glyphs in [('F77_L','屄䏘姕閪鳖'),('F90_L','煠㷙罩'),('F73_L','脧𡳞'),('F126_L','噘𧮙'),('F148_L','揾𢶀攞'),('F178_L','𩟗'),('F179_L','𣤩'),('F182_L','𢜶')]:
            choose_feature(fid)
            rows,items=check_broad_exports(fid)
            labels=''.join(item['label'] for item in items)
            assert all(glyph in labels for glyph in glyphs),(fid,glyphs,labels)
            shot('verified-glyphs-'+fid)
            if fid=='F178_L':
                rare_png=export('PNG','.png')
                from PyQt6.QtGui import QImage
                bitmap=QImage(str(rare_png));assert not bitmap.isNull() and bitmap.width()>=2400
                bitmap.scaledToWidth(1800).save(str(out/'rare-glyph-export-preview.png'))
        report['checks'].append('User-confirmed lexicon glyphs in real legend, CSV and SVG; supplementary-plane characters and actual PNG captured for visual verification')
        choose_feature('F35_P')
        f35=next(f for f in source_catalog if f['id']=='F35_P')
        assert f35['groupReview']['complete'] and f35['groupReview']['verifiedGroups']==5
        rows,items=check_broad_exports('F35_P')
        assert len(items)==5 and all(item['label']!='尚未对应的大类' and not re.fullmatch(r'[A-Z]\d*',item['label']) for item in items)
        assert js('document.querySelector(".category-coverage").textContent.includes("已对应 5 个大类")')
        js('(()=>{const label=[...document.querySelectorAll(".geography-toolbar label")].find(e=>e.textContent.trim()==="地名"),input=label.querySelector("input");input.checked=true;input.dispatchEvent(new Event("change",{bubbles:true}));})()');pause()
        labelled_svg=ET.fromstring(export('SVG','.svg').read_text(encoding='utf-8'))
        assert any((e.text or '')=='北京' for e in labelled_svg.findall('.//{http://www.w3.org/2000/svg}text'))
        selected_circles=[e for e in labelled_svg.findall('.//{http://www.w3.org/2000/svg}circle') if e.get('stroke')=='#172c40']
        assert selected_circles and all(float(e.get('r'))<=5.5 and float(e.get('stroke-width'))<=1.5 for e in selected_circles), 'Selection must not become an oversized statistical marker'
        js('(()=>{const label=[...document.querySelectorAll(".geography-toolbar label")].find(e=>e.textContent.trim()==="地名"),input=label.querySelector("input");input.checked=false;input.dispatchEvent(new Event("change",{bubbles:true}));})()');pause()
        shot('broad-f35-five-groups')
        fill('[aria-label="检索调查点"]','S279')
        until('document.querySelectorAll(".table-scroll tbody tr").length===1')
        rows,items=check_broad_exports('F35_P',{'S279'})
        assert rows[0]['类别编码']=='D1#D2' and rows[0]['具体含义']=='读如清去或清上'
        assert len(items)==1 and items[0]['count']==1
        shot('broad-same-site-deduplicated');fill('[aria-label="检索调查点"]','')
        for feature_id,site_id,raw in [('F43_S','S49','A'),('F45_S','S151','A'),('F48_S','S49','B'),('F80_S','S339','C1#A1')]:
            choose_feature(feature_id);fill('[aria-label="检索调查点"]',site_id)
            until('document.querySelectorAll(".table-scroll tbody tr").length===1')
            assert js('document.querySelector(".table-scroll tbody tr td:nth-child(3)").textContent.trim()==="原图对应待核"')
            rows,items=check_broad_exports(feature_id,{site_id})
            assert rows[0]['具体含义']=='原图对应待核' and rows[0]['类别编码']==raw
            assert not items  # A source conflict is a record state, not a linguistic class.
            shot('broad-source-conflict-'+feature_id);fill('[aria-label="检索调查点"]','')
        choose_feature('F43_S');fill('[aria-label="检索调查点"]','S49')
        until('document.querySelectorAll(".table-scroll tbody tr").length===1')
        js('document.querySelector(".table-scroll tbody tr").click()');pause()
        click('编辑当前特征');fill('dialog .full-input','B');click('保存校订')
        until('!document.querySelector(".table-scroll").textContent.includes("原图对应待核")')
        rows,items=check_broad_exports('F43_S',{'S49'},{'S49':'B'})
        assert rows[0]['具体含义']=='无“阿”前缀' and rows[0]['原始值']=='A'
        click('编辑当前特征');fill('dialog .full-input','A#B');click('保存校订')
        rows,items=check_broad_exports('F43_S',{'S49'},{'S49':'A#B'})
        assert rows[0]['具体含义']=='原图对应待核；无“阿”前缀' and len(items)==1 and items[0]['label']=='无“阿”前缀'
        shot('broad-conflict-keeps-other-variant')
        click('恢复原值');check_broad_exports('F43_S',{'S49'})
        fill('[aria-label="检索调查点"]','')
        report['checks'].append('broad default and reviewed catalogs; F120 six complete groups; fine/broad switch; per-site same-group deduplication; UI/CSV/SVG count and meaning identity; source-conflict cells keep raw codes without forced meanings')
        report['checks'].append('source-conflict check is per original atomic code: user replacement B displays its group, A#B keeps the valid B variant, restore retains original A')
        report['checks'].append('broad legends/tooltips/table/details/SVG hide fine codes and group IDs; pending records have hollow/dashed locations and a deduplicated pending-site status without a synthetic category; mixed records retain known memberships; CSV keeps raw source columns')
        # Restore the original ten-phenomenon fine-category starting point; all legacy acceptance follows unchanged.
        choose_feature('F1_P');fill('[aria-label="检索语言特征"]','')
        fill('[aria-label="分类层级"]','fine','change');fill('[aria-label="语言现象范围"]','reviewed','change')
        until('document.querySelectorAll(".feature-list button").length===2&&document.querySelectorAll(".ol-layer canvas").length>=1')
        assert js('document.querySelector("[aria-label=语言现象范围]").textContent.includes("10 项")&&!document.querySelector(".original-map,.legend-preview")')
        click('收藏本图');js('document.querySelector(".bookmark-filter input").click()');pause()
        assert js('document.querySelectorAll(".feature-list button").length===1')
        js('document.querySelector(".bookmark-filter input").click()');pause()
        for name,total in [('词汇',5),('语法',3),('语音',2)]:
            click(name);until(f'document.querySelectorAll(".feature-list button").length==={total}')
            assert js('!document.querySelector(".legend").textContent.includes("释义待核")')
        pause(1000);js('window.__geoReloadPending=true;location.reload()');until('typeof window.__geoReloadPending==="undefined"&&!!document.querySelector(".nav-item")');click('语言地图')
        until('document.querySelectorAll(".feature-list button").length===2')
        assert js('[...document.querySelectorAll(".map-heading button")].some(e=>e.textContent==="取消收藏")')
        assert js('!performance.getEntriesByType("resource").some(r=>r.name.includes("/book/"))')
        legend_geometry=json.loads(js('JSON.stringify([...document.querySelectorAll(".legend button")].map(e=>({height:e.getBoundingClientRect().height,code:e.querySelector(".category-code")?.textContent??null,title:e.title,lineHeight:getComputedStyle(e).lineHeight})))'))
        assert len(legend_geometry)==11 and all(r['height']>=40 and r['code'] is None and r['title'].startswith('分类编码：') for r in legend_geometry),legend_geometry
        report['legendGeometry']=legend_geometry
        shot('readable-language-phenomena')
        report['checks'].append('county default, ten fully decoded phenomena across three volumes, interactive map and source page references, bookmarks survive reload, no scanned-page requests')
        fill('[aria-label="语言现象范围"]','all','change')
        until('document.querySelectorAll(".feature-list button").length===205')
        assert js('document.querySelector(".map-heading").textContent.includes("调类的数目")')
        assert js('document.querySelectorAll(".feature-list button")[2].textContent.trim()==="003上声的分化"')
        assert js('![...document.querySelectorAll(".feature-list button")].some(e=>/Division|Tone|Merger|�/.test(e.textContent))')
        fill('[aria-label="检索语言特征"]','tʂʰ')
        assert js('document.querySelectorAll(".feature-list button").length===1&&document.querySelector(".feature-list button").textContent.includes("[tʂ tʂʰ ʂ]")')
        fill('[aria-label="检索语言特征"]','')
        check_buttons();js('document.documentElement.dataset.theme="light"')
        shot('features-light');report['checks'].append('930 sites, 205 phonology feature controls, offline OpenLayers canvas')
        for name,total in [('词汇',203),('语法',102)]:
            click(name);until(f'document.querySelectorAll(".feature-list button").length==={total}')
            assert js('![...document.querySelectorAll(".feature-list button")].some(e=>/Division|Initials|Finals|Merger|�|[语音词汇法]图 \\d/.test(e.textContent))')
        click('语音');report['checks'].append('510 source-reviewed Chinese titles, no OCR subtitles/placeholders; searchable exact IPA superscript')
        fill('[aria-label="检索语言特征"]','撮口呼')
        assert js('document.querySelectorAll(".feature-list button").length>=1')
        js('document.querySelector(".feature-list button").click()');pause()
        assert js('document.querySelector(".map-heading").textContent.includes("撮口呼")')
        assert js('document.querySelector(".legend").textContent.includes("只有[y]韵")&&document.querySelector(".legend").textContent.includes("无撮口呼韵母")')
        assert js('document.querySelector(".category-coverage").textContent.includes("4 / 4")')
        assert js('[...document.querySelectorAll(".legend .ipa")].some(e=>e.textContent==="[y-]"&&getComputedStyle(e).fontFamily.includes("PTB-Doulos"))')
        semantic_svg=export('SVG','.svg').read_text(encoding='utf-8')
        semantic_text=''.join(e.text or '' for e in ET.fromstring(semantic_svg).findall('.//{http://www.w3.org/2000/svg}text'))
        assert '有[y]韵和[y-]介音韵（A）' in semantic_text and '617 个地点' in semantic_text and 'PDF 第 453 页' in semantic_text
        semantic_csv=export('导出表格','.csv')
        with semantic_csv.open(encoding='utf-8-sig',newline='') as stream: semantic_rows=list(csv.DictReader(stream))
        beijing=next(row for row in semantic_rows if row['地点编号']=='S49')
        assert beijing['类别编码']=='A' and beijing['具体含义']=='有[y]韵和[y-]介音韵' and beijing['原始值']=='A'
        shot('reviewed-meanings')
        report['checks'].append('reviewed P117 meanings in actual UI/IPA font, SVG with code/site count/page, CSV readable meanings and original codes')
        assert js('document.querySelector(".atlas-details").textContent.includes("453")&&!document.querySelector(".legend-preview")')
        click('方言分布');until('document.querySelector(".atlas-catalog").textContent.includes("3666")')
        check_buttons()
        fill('[aria-label="省级地区"]','province:430000','change')
        assert js('document.querySelector(".atlas-catalog .summary").textContent.includes("17")')
        click('查看计入的分类');until('document.querySelectorAll(".path-list button").length===17');shot('hunan-paths');click('关闭对话框')
        click('中国');fill('.atlas-catalog select','islands','change');until('document.querySelector(".table-toolbar").textContent.includes("356")');shot('islands')
        report['checks'].append('independent curated 3666 rows, Hunan 17 full slice paths, 356 island rows')
        click('方言相似性');until('document.querySelectorAll(".mds-plot circle").length===930')
        js('document.querySelector(".table-scroll tbody tr").click()');until('document.querySelectorAll(".nearby").length===15');shot('paper-results')
        # Main is already the default selector value: restoration must load it explicitly.
        pause(1000);js('window.__geoReloadPending=true;location.reload()');until('typeof window.__geoReloadPending==="undefined"&&!!document.querySelector(".nav-item")');click('语言地图')
        until('document.querySelectorAll(".nearby").length===15')
        assert js('[...document.querySelectorAll(".mode-tabs button")].some(e=>e.textContent==="方言相似性"&&e.getAttribute("aria-selected")==="true")')
        paper_project=export('导出工程','.json').read_text(encoding='utf-8')
        assert json.loads(paper_project)['settings']['matrixName']=='main'
        import_file('.atlas-toolbar input[type=file][accept=".json"]','论文主结果.ptb-map.json',paper_project)
        until('!!document.querySelector("dialog .dialog-actions")')
        js('[...document.querySelectorAll("dialog .dialog-actions button")].find(e=>e.textContent==="打开工程").click()')
        until('document.querySelectorAll(".nearby").length===15');shot('paper-main-restored')
        report['checks'].append('paper/main save, reload and project reimport restore 15 neighbours without changing matrix selector')
        # Compare a fixed pair against raw source columns, not the UI comparison helper.
        js('document.querySelector(".nearby button:last-child").click()');until('!!document.querySelector(".comparison-summary")')
        fill('[aria-label="对照地点 1"]','S49','change');fill('[aria-label="对照地点 2"]','S151','change')
        source_observations=json.loads((data_root/'observations.json').read_text('utf-8'))
        source_catalog=json.loads((data_root/'catalog.json').read_text('utf-8'))['features']
        expected={'same':0,'different':0,'missing':0}
        for f in source_catalog:
            first,second=(source_observations[f['id']]['values'][i] for i in [48,150])
            expected['missing' if first is None or second is None else 'same' if first==second else 'different']+=1
        summary=js('document.querySelector(".comparison-summary").textContent')
        for key,label in [('same','记录相同'),('different','记录不同'),('missing','含缺值')]:assert f'{label} {expected[key]} 项' in summary,summary
        comparison_started=time.monotonic()
        fill('[aria-label="比较项目范围"]','P','change');fill('[aria-label="比较记录关系"]','','change')
        assert js('document.querySelector(".comparison-summary").textContent.startsWith("205 项")&&document.querySelectorAll(".comparison-scroll tbody tr").length===60')
        first_page=js('document.querySelector(".comparison-scroll tbody").textContent')
        for _ in range(3):click('地点对照下一页')
        assert js('document.querySelectorAll(".comparison-scroll tbody tr").length===25&&document.querySelector("[aria-label=地点对照下一页]").disabled')
        click('地点对照上一页')
        assert js('document.querySelectorAll(".comparison-scroll tbody tr").length===60')
        fill('[aria-label="比较记录关系"]','same','change')
        assert js('document.querySelector("[aria-label=地点对照上一页]").disabled')
        fill('[aria-label="比较记录关系"]','','change')
        assert js('document.querySelector(".comparison-scroll tbody").textContent')==first_page
        report['comparisonPagingSeconds']=round(time.monotonic()-comparison_started,3)
        compare_csv=export('导出对照表','.csv')
        with compare_csv.open(encoding='utf-8-sig',newline='') as stream: compared=list(csv.DictReader(stream))
        assert len(compared)==205 and all(row['卷']=='语音' for row in compared)
        for f,row in zip([f for f in source_catalog if f['volume']=='P'],compared):
            codes=[v for k,v in row.items() if k.endswith(' · 编码')]
            assert codes==[source_observations[f['id']]['values'][i] or '' for i in [48,150]]
        shot('two-place-comparison');click('关闭对话框')
        fill('.atlas-catalog select','sharp','change');fill('.atlas-catalog input[type=range]','7');pause()
        assert js('document.querySelector(".paper-details").textContent.includes("7 类")&&document.querySelector(".legend").textContent.length>0')
        shot('paper-clusters')
        report['checks'].append({'workflow':'MDS, original distance neighbours, pair compare, 205 phonology CSV raw-cell identity, seven-cluster map','pair':['S49','S151'],'expected':expected})
        click('我的研究地图');click('输入地点坐标');fill('dialog .edit-grid input','测试珠海');fill('dialog .edit-grid label:nth-child(4) input','ŋ̍˨˩');click('保存地点')
        until('document.querySelector(".table-scroll").textContent.includes("测试珠海")');click('撤销');until('!document.querySelector(".table-scroll").textContent.includes("测试珠海")');click('重做');until('document.querySelector(".table-scroll").textContent.includes("测试珠海")')
        click('保存到本机');pause(1000)
        js('window.__geoReloadPending=true;location.reload()');until('typeof window.__geoReloadPending==="undefined"&&!!document.querySelector(".nav-item")');click('语言地图');until('document.querySelector(".table-scroll")?.textContent.includes("测试珠海")')
        assert js('document.querySelector(".table-scroll").textContent.includes("ŋ̍˨˩")');report['checks'].append('Unicode custom point, undo/redo, IndexedDB save and reload')
        import_file('.atlas-catalog input[type=file]','IPA地点.csv','地点,经度,纬度,类别,备注\n测试音标,113.6,22.4,tʂʰ ã n̩ ʔ,原样保留\n')
        until('!!document.querySelector("dialog .comparison-scroll")');click('导入全部地点');until('document.querySelector(".table-scroll").textContent.includes("测试音标")')
        csv_path=export('导出表格','.csv')
        with csv_path.open(encoding='utf-8-sig',newline='') as stream: exported=list(csv.reader(stream))
        assert any(row[3]=='tʂʰ ã n̩ ʔ' for row in exported[1:]),exported
        project_path=export('导出工程','.json');saved_project=json.loads(project_path.read_text(encoding='utf-8'))
        assert len(saved_project['customSites'])==2
        import_file('.atlas-toolbar input[type=file][accept=".json"]','恢复.ptb-map.json',project_path.read_text(encoding='utf-8'));until('!!document.querySelector("dialog .dialog-actions")')
        js('[...document.querySelectorAll("dialog .dialog-actions button")].find(e=>e.textContent==="打开工程").click()');pause()
        assert js('document.querySelector(".table-scroll").textContent.includes("测试音标")')
        report['checks'].append('actual CSV import, native blob download and project re-import preserve IPA code points')
        legacy_project={key:value for key,value in saved_project.items() if key!='featureDetail'}
        import_file('.atlas-toolbar input[type=file][accept=".json"]','旧格式.ptb-map.json',json.dumps(legacy_project,ensure_ascii=False));until('!!document.querySelector("dialog .dialog-actions")')
        js('[...document.querySelectorAll("dialog .dialog-actions button")].find(e=>e.textContent==="打开工程").click()');pause()
        click('语言现象');assert js('document.querySelector("[aria-label=分类层级]").value==="fine"')
        report['checks'].append('legacy project without featureDetail retains fine-category presentation')
        click('语言现象');fill('[aria-label="检索调查点"]','北京');until('document.querySelectorAll(".table-scroll tbody tr").length===1');js('document.querySelector(".table-scroll tbody tr").click()');pause();click('编辑当前特征');fill('dialog .full-input','QA1#QA2');click('保存校订');until('document.querySelector(".table-scroll").textContent.includes("QA1")');click('恢复原值');until('!document.querySelector(".table-scroll").textContent.includes("QA1")');report['checks'].append('observation override and restore preserve original')
        fill('[aria-label="检索调查点"]','');click('中国');pause(400)
        assert not js('[...document.querySelectorAll(".geography-toolbar button")].some(e=>e.textContent.trim()==="世界")')
        fill('.map-heading label select','2','change');until('document.querySelectorAll(".map-cell canvas").length>=2');pause(700)
        china_project=json.loads(export('导出工程','.json').read_text(encoding='utf-8'))
        assert abs(china_project['view']['center'][0]-104)<1e-3 and 25<china_project['view']['center'][1]<40,china_project['view']
        report['checks'].append('China overview remains centered after creating a second map; world entry is absent')
        click('中国');fill('.map-heading label select','4','change');until('document.querySelectorAll(".map-cell").length===4');shot('four-maps')
        svg_path=export('SVG','.svg');svg=ET.fromstring(svg_path.read_text(encoding='utf-8'));ns={'s':'http://www.w3.org/2000/svg'}
        assert len(svg.findall('s:svg',ns))==4
        assert len(svg.findall('.//s:clipPath',ns))==4 and len({e.attrib['id'] for e in svg.findall('.//s:clipPath',ns)})==4
        assert 'PTB-Map-IPA' in svg_path.read_text(encoding='utf-8')
        svg_text=''.join(e.text or '' for e in svg.findall('.//s:text',ns))
        assert '有[y]韵和[y-]介音韵' in svg_text and '手' in svg_text and '蜀' in svg_text and '释义待核' not in svg_text
        assert all(not (e.text or '').startswith(('\u0303','\u0329')) for e in svg.findall('.//s:text',ns))
        png_path=export('PNG','.png');bitmap=QImage(str(png_path));assert not bitmap.isNull() and bitmap.width()>1000
        assert float(svg.attrib['width']) == 2420 and bitmap.width() == 2*float(svg.attrib['width']) and bitmap.height()/bitmap.width()<2
        assert bitmap.scaled(1600,1600,Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation).save(str(out/'export-four-maps-preview.png'))
        report['checks'].append('actual 4-panel SVG/PNG downloads, unique clipping paths, full legends and embedded licensed IPA font')
        click('方言分布');fill('[aria-label="省级地区"]','province:430000','change');fill('.atlas-catalog select','combined','change');pause()
        class_svg=export('SVG','.svg').read_text(encoding='utf-8');assert '井韶子整理' in class_svg and '官话' in class_svg and '13调' not in class_svg
        click('我的研究地图');click('保存到本机');pause(800);js('window.__geoReloadPending=true;location.reload()');until('typeof window.__geoReloadPending==="undefined"&&!!document.querySelector(".nav-item")');click('语言地图');until('!!document.querySelector(".atlas-workbench")');click('方言分布')
        assert js('document.querySelector("[aria-label=省级地区]").value==="province:430000"')
        assert js('document.querySelector(".atlas-catalog .summary").textContent.includes("17")')
        report['checks'].append('classification export uses classification legend; region and sheet survive reopening')
        click('语言现象');click('中国')
        js('document.documentElement.dataset.theme="dark"');pause();shot('features-dark')
        check_buttons()
        for width,height in [(1280,800),(1920,1080)]:
            window.resize(width,height);pause(800);shot(f'viewport-{width}')
        report['checks'].append('four maps with shared view; light/dark and 1280/1440/1920 windows')
        report['success']=True
    except Exception:
        report['error']=traceback.format_exc()
        try:report['recentClicks']=js('window.__qaClicks??[]')
        except Exception:pass
        shot('failure');raise
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        faulthandler.cancel_dump_traceback_later();faulthandler.disable();trace_stream.close()
        QFileDialog.getSaveFileName=save_dialog
        phase('owned-window-close-start',reason='verification finally cleanup',success=report['success'])
        window.closing=True;window.close();phase('owned-window-close-returned');pause(1000)
        window.page.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete);pause(300)
        window.profile.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete);pause(700)
        window.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete);app.quit();phase('owned-window-cleanup-complete')


def verify(bundle,root):
    with verification_run('geolinguistics','exe','Final EXE M19 actual map/edit/export/IPA acceptance',root=root) as (out,scratch):
        (root/'m19-run-path.txt').write_text(str(out),encoding='utf-8')
        run(out,scratch,bundle=bundle)
    return 0

def verify_renderer(bundle,root):
    with verification_run('geolinguistics','exe-renderer','Final frozen renderer policy/counts only; automatic admin levels and screen/SVG agreement',root=root) as (out,scratch):
        (root/'m19-renderer-run-path.txt').write_text(str(out),encoding='utf-8')
        run(out,scratch,bundle=bundle,renderer_check=True)
    return 0

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frontend-dist',type=Path)
    parser.add_argument('--data-directory',type=Path)
    parser.add_argument('--compact-client',action='store_true')
    parser.add_argument('--layout-only',action='store_true')
    parser.add_argument('--jiyuan-only',action='store_true')
    parser.add_argument('--renderer-only',action='store_true')
    parser.add_argument('--interaction-only',action='store_true')
    parser.add_argument('--geography-audit-only',action='store_true',help='Explicit historical geographic-content UI probe; excluded from default full and renderer checks')
    args=parser.parse_args()
    with verification_run('geolinguistics','qt','M19 actual offline data, map canvas, curated classification, scientific results, edit and persistence') as (out,scratch):
        print(out,flush=True);run(out,scratch,client_check=args.compact_client,layout_check=args.layout_only,frontend_dist=args.frontend_dist,data_directory=args.data_directory,jiyuan_check=args.jiyuan_only,renderer_check=args.renderer_only,geography_check=args.geography_audit_only,interaction_check=args.interaction_only)
