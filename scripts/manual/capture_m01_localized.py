"""Real native M01 analysis, EGG, segmentation and restored-result workflows."""
from __future__ import annotations
import argparse,hashlib,json,re,shutil,sqlite3,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from verification_artifacts import verification_run
from manual.capture_a_support import Capture


def run(args,out,scratch):
    c=Capture(args,out,scratch,'m01','parameter-estimation',reaper_binary=ROOT/'phonetic_toolbox/core/acoustic/reaper.exe')
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QFileDialog
    from PyQt6 import sip
    from ptb_desktop.host import Workbench
    import soundfile as sf
    config=json.loads((ROOT/'local-data/manual-authoring/sources.json').read_text(encoding='utf-8'))
    original_hashes={};prepared=[];batch_ids=[]
    folders={name:scratch/name for name in ('batch','textgrid','egg','recovered')}
    for folder in folders.values():folder.mkdir()
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    def copy(key,group,name,grid=False):
        source=Path(config[key]);original_hashes[str(source)]=sha(source)
        destination=folders[group]/(name+'.wav');shutil.copyfile(source,destination)
        info=sf.info(destination);prepared.append({'key':key,'name':destination.name,'sha256':sha(destination),'sampleRate':info.samplerate,'channels':info.channels,'duration':info.duration})
        if grid:
            sidecar=source.with_suffix('.TextGrid');assert sidecar.is_file();original_hashes[str(sidecar)]=sha(sidecar)
            shutil.copyfile(sidecar,destination.with_suffix('.TextGrid'))
    copy('single','batch','single-syllable');copy('sentence','batch','speech-sentence',True);copy('target','batch','target-syllable')
    copy('single','textgrid','single-syllable');copy('sentence','textgrid','speech-sentence',True);copy('egg','egg','egg-stereo')
    choice={'folder':folders['batch']}
    QFileDialog.getExistingDirectory=lambda *a,**k:str(choice['folder'])
    def observe():
        original=c.window.service.request
        def request(path,method='GET',body=None):
            result=original(path,method,body)
            if path=='/api/v1/jobs/batches/create':batch_ids.append(result['id'])
            return result
        c.window.service.request=request
    observe()
    def text(key,values=None):return c.ui.key('modules.parameter-estimation.'+key,values)
    def includes(value):return 'document.querySelector(".m01-page").innerText.includes('+json.dumps(value,ensure_ascii=False)+')'
    def hit(selector):
        assert c.js('(()=>{const e=document.querySelector('+json.dumps(selector)+');if(!e||e.disabled)return false;e.click();return true})()'),selector
        c.pause()
    def checked(selector,value):
        assert c.js('(()=>{const e=document.querySelector('+json.dumps(selector)+');if(!e||e.disabled)return false;if(e.checked!=='+json.dumps(value)+')e.click();return true})()'),selector
        c.pause()
    def label_check(chinese,value):
        label=c.ui.label(chinese)
        assert c.js('(()=>{const e=[...document.querySelectorAll("label")].find(e=>e.textContent.includes('+json.dumps(label,ensure_ascii=False)+'))?.querySelector("input[type=checkbox]");if(!e||e.disabled)return false;if(e.checked!=='+json.dumps(value)+')e.click();return true})()'),chinese
        c.pause()
    def aria(chinese):return '[aria-label='+json.dumps(c.ui.label(chinese),ensure_ascii=False)+']'
    def select(chinese,value):c.fill('select'+aria(chinese),value,'change')
    def mark_only(names):
        checked('.m01-select-all input',True);checked('.m01-select-all input',False)
        for name in names:
            assert c.js('(()=>{const row=[...document.querySelectorAll(".m01-file-entry")].find(e=>e.querySelector("button").textContent.includes('+json.dumps(name)+'));if(!row)return false;row.querySelector("input").click();return true})()')
            c.pause()
        assert c.js('document.querySelectorAll(".m01-file-entry input:checked").length')==len(names)
    def choose(group,count):
        choice['folder']=folders[group];c.click('打开音频目录');c.until('document.querySelectorAll(".m01-file-entry").length==='+str(count))
    def audio(name):
        assert c.js('(()=>{const e=[...document.querySelectorAll(".m01-file-list button")].find(e=>e.textContent.includes('+json.dumps(name)+'));if(!e)return false;e.click();return true})()')
        c.until('!!document.querySelector(".m01-page .wave-track svg")&&!document.querySelector(".m01-page .audio-loading")&&document.querySelector(".signal-heading h2").textContent==='+json.dumps(name),100)
    def scroll(left=0,right=0):
        for selector,value in (('.workbench-left',left),('.workbench-right-body',right),('.workbench-center',0)):
            c.js('document.querySelector(".m01-page '+selector+'").scrollTop='+str(value))
        c.pause(200)
    def saved(count):return text('saved_value_result_files_existing_files_with_the_same',{'v0':count})
    def settle(previous,expected=None,success=True,seconds=420):
        deadline=time.monotonic()+seconds
        while len(batch_ids)<=previous and time.monotonic()<deadline:c.pause()
        assert len(batch_ids)>previous,'No native batch submitted'
        while time.monotonic()<deadline:
            batch=c.window.service.get('/api/v1/jobs/batches/'+batch_ids[-1])
            if batch['summary']['closed']:break
            c.pause(180)
        else:raise AssertionError('Actual native batch did not close')
        if success:
            assert batch['summary']['complete'],batch
            if batch['operation']=='acoustic_analysis':c.until(includes(saved(batch['summary']['total']*2)),120)
            else:
                pattern=re.escape(saved('__COUNT__')).replace('__COUNT__',r'\d+')
                c.until('new RegExp('+json.dumps(pattern,ensure_ascii=False)+').test(document.querySelector(".m01-page").innerText)',120)
        else:
            assert batch['summary']['counts']['failed']>=1,batch
            c.until('!!document.querySelector(".m01-result-items small.danger-text")')
        if expected is not None:assert batch['audio_names']==expected,batch['audio_names']
        c.report.setdefault('batches',[]).append(batch);c.pause(250);return batch
    def shot(name,regions=None):c.shot(name,regions)
    try:
        c.until('!!document.querySelector(".m01-page")');c.prepare();choose('batch',3);audio('single-syllable.wav')
        shot('m01-overview-light')
        c.click('选择输出参数');assert c.js('document.querySelectorAll(".parameter-grid input:checked").length')==80
        shot('m01-parameters');shot('m01-parameters-all80-light')
        c.fill(aria('搜索参数'),'H1H2');assert c.js('document.querySelectorAll(".parameter-grid input").length')==4;shot('m01-parameters-search-light')
        c.fill(aria('搜索参数'),'');c.click('全不选');assert c.js('document.querySelector("dialog[open] footer button.primary").disabled');shot('m01-parameters-empty-light')
        for key in ('pF0','pF1','pF2'):checked('.parameter-grid input[value='+key+']',True)
        c.click('应用到草稿');c.click('编辑14项设置');assert c.js('document.querySelectorAll(".m01-settings .setting-row").length')==12;shot('m01-settings')
        c.fill('input'+aria('最小基频'),900);c.click('应用设置');c.until('!!document.querySelector("dialog[open] [role=alert]")')
        shot('m01-settings-f0-invalid-light',{'settings':'.m01-settings','minimum':'.m01-settings .setting-row:nth-last-child(2)','maximum':'.m01-settings .setting-row:last-child','error':'dialog[open] [role=alert]'})
        c.fill('input'+aria('最小基频'),30);hit('dialog[open] .parameter-toolbar button:nth-child(2)');assert c.js('document.querySelectorAll(".m01-settings input[type=checkbox]").length')==2
        shot('m01-reaper',{'settings':'.m01-settings','hilbert':'.m01-settings .setting-row:first-child','highpass':'.m01-settings .setting-row:last-child'});c.click('取消')
        mark_only([]);assert c.js('document.querySelector(".m01-batch-bar button:nth-of-type(2)").disabled');shot('m01-selected-empty-light')
        mark_only(['single-syllable.wav']);previous=len(batch_ids);c.click('分析选中文件');settle(previous,['single-syllable.wav']);shot('m01-result-flow-r3-light')
        assert not (folders['batch']/'target-syllable.xlsx').exists() and not (folders['batch']/'speech-sentence.xlsx').exists()
        previous=len(batch_ids);c.click('开始全列表分析');batch=settle(previous);assert batch['summary']['total']==3
        c.until(includes(saved(6)));scroll(right=100);shot('m01-all-list-completed-light')
        choose('textgrid',2);audio('speech-sentence.wav');c.until('document.querySelectorAll(".textgrid-tier").length===2');scroll(10000);shot('m01-textgrid-linked-light')
        assert c.js('(()=>{const e=[...document.querySelectorAll(".textgrid-interval")].find(e=>e.textContent.trim()==="zhe4");if(!e)return false;e.click();return true})()')
        label_check('显示语谱图',True);c.until('document.querySelector(".spectrogram-canvas")?.getAttribute("aria-busy")==="false"&&document.querySelector(".spectrogram-canvas canvas")?.style.visibility==="visible"',100);shot('m01-spectrogram-selected-interval-light');label_check('显示语谱图',False)
        mark_only(['single-syllable.wav','speech-sentence.wav']);previous=len(batch_ids);c.click('保存当前层切分音频');c.until(includes('single-syllable.wav')+'&&'+includes(text('has_no_associated_textgrid')));assert len(batch_ids)==previous;shot('m01-textgrid-missing-block-light')
        mark_only(['speech-sentence.wav']);previous=len(batch_ids);c.click('分析选中文件');settle(previous,['speech-sentence.wav']);label_check('同时切分参数结果',True);select('切分参数来源','recent');scroll(10000);shot('m01-slice-recent-source-light')
        select('切分参数来源','legacy');previous=len(batch_ids);c.click('保存当前层切分音频');c.until(includes(text('has_no_explicitly_associated_historical_parameter_table_select_one')));assert len(batch_ids)==previous;shot('m01-slice-legacy-unassigned-light')
        select('切分参数来源','recent');previous=len(batch_ids);c.click('保存当前层切分音频');batch=settle(previous);assert batch['operation']=='textgrid_segment'
        sliced=[p for p in folders['textgrid'].glob('*.ptb.sqlite') if p.name!='speech-sentence.ptb.sqlite'];assert sliced
        with sqlite3.connect(sliced[0]) as connection:assert 'Source_Time_s' in [row[1] for row in connection.execute('PRAGMA table_info(params)')]
        scroll(right=100);shot('m01-textgrid-slices-saved-light')
        choose('egg',1);audio('egg-stereo.wav');label_check('联合计算 EGG',True);select('默认分析声道','1');select('EGG参数保存方式','aligned');scroll(500);shot('m01-egg-aligned-roles-light')
        select('EGG参数保存方式','cycles');hit('.m01-joint summary');label_check('计算 GCI F0 派生声学参数',True);scroll(10000)
        shot('m01-egg-cycles-settings-light',{'roles':'.m01-joint','channel':'.m01-joint>label:first-of-type','storage':'.m01-joint>label:has(select[aria-label='+json.dumps(c.ui.label('EGG参数保存方式'),ensure_ascii=False)+'])'})
        c.fill('.m01-joint details label:nth-of-type(2) input',30000);mark_only(['egg-stereo.wav']);previous=len(batch_ids);c.click('分析选中文件');batch=settle(previous,success=False)
        assert batch['summary']['items'][0]['error_code']=='m01_egg_filter_range'
        c.until(includes(text('the_egg_low_pass_cutoff_must_be_below_half')));scroll(650,10000);shot('m01-egg-filter-failed-light')
        c.fill('.m01-joint details label:nth-of-type(2) input',2000);previous=len(batch_ids);c.click('分析选中文件')
        running='(()=>{const p=document.querySelector(".m01-file-progress progress");return p&&p.value>0&&p.value<1})()';c.until(running,180);scroll(500);assert c.js(running);shot('m01-egg-running-progress-light');settle(previous)
        sqlite_file=folders['egg']/'egg-stereo.ptb.sqlite'
        with sqlite3.connect(sqlite_file) as connection:
            tables={r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")};assert {'params','egg_cycles','ptb_metadata'}<=tables
            columns=[r[1] for r in connection.execute('PRAGMA table_info(params)')];assert any('(gF0)' in col for col in columns) and 'CQ' not in columns
            assert connection.execute('SELECT COUNT(*) FROM egg_cycles').fetchone()[0]>0
        original_cycle_sha=sha(sqlite_file);hit('.m01-joint summary');scroll(500,100);shot('m01-egg-cycles-saved-light')
        select('EGG参数保存方式','aligned');previous=len(batch_ids);c.click('分析选中文件');settle(previous)
        with sqlite3.connect(folders['egg']/'egg-stereo (2).ptb.sqlite') as connection:
            columns=[r[1] for r in connection.execute('PRAGMA table_info(params)')];assert 'CQ' in columns and 'SQ' in columns
        assert sha(sqlite_file)==original_cycle_sha
        # A new real host restores history but gets no previous directory grant.
        old=c.window;db,cache=old.service.jobs_path,old.service.local_files_root;old.closing=True;old.close();c.app.processEvents();sip.delete(old);c.app.processEvents()
        c.window=Workbench(args.frontend_dist,test=True,jobs_path=db,local_files_root=cache,reaper_binary=ROOT/'phonetic_toolbox/core/acoustic/reaper.exe',vocal_profile=scratch/'vocal',start_module='M01')
        c.window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen);c.window.resize(c.window.screen().geometry().width(),c.window.screen().geometry().height());c.window.showMaximized();c.window.view.setZoomFactor(1);c.window.page.setAudioMuted(True);observe()
        c.until('!!document.querySelector(".m01-page")');c.prepare();c.until('!!document.querySelector(".m01-result-actions button.primary:not(:disabled)")');c.click('保存已完成结果');c.until(includes(text('after_reopening_select_a_results_folder_before_saving_completed')));shot('m01-save-directory-recovery-dark')
        checked('.m01-output-row input',False);choice['folder']=folders['recovered'];c.click('选择结果目录');previous=len(batch_ids);c.click('保存已完成结果');c.until(includes(saved(2)));assert len(batch_ids)==previous and len(list(folders['recovered'].iterdir()))==2
        c.report.update(success=True,prepared=prepared,originalSourceHashes=original_hashes,originalSourcesUnchanged=all(sha(Path(path))==digest for path,digest in original_hashes.items()),checks=['Actual selected and all-list native analyses, paired XLSX/SQLite saves, original input hashes retained','Actual authorized TextGrid read/interval spectrogram, missing association rejection, same-source parameter segmentation and Source_Time_s readback','Actual invalid EGG cutoff failure, translated reason, raw-cycle and aligned outputs with gF0-derived columns, non-overwriting group suffix','New native host restores history without directory capability; explicit new grant saves same completed result without another scientific submission'])
        assert c.report['originalSourcesUnchanged']
    finally:
        if not c.report['success']:
            c.report['failedVisibleText']=c.js('document.body.innerText');c.window.view.grab().save(str(out/'failed.png'))
        c.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--locale',choices=('en','ja'),required=True);parser.add_argument('--frontend-dist',type=Path,default=ROOT/'output/validation/app-i18n-root/m01-capture-build');args=parser.parse_args();args.frontend_dist=args.frontend_dist.resolve(strict=True)
    with verification_run('m01-localized',args.locale,'Actual native Windows/M01 scientific workflows and authorized sample copies; no user profile or source writes') as (out,scratch):run(args,out,scratch)
