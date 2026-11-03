"""Real localized M07 analysis/edit/nine-step/save with registered inputs unchanged."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from verification_artifacts import verification_run
from manual.capture_a_support import Capture

def run(args):
 with verification_run('m07-manual',args.locale,'Actual maximized Qt, registered source/target WAV read-only, original Parselmouth 50–600Hz/1ms/21 points, actual nine-step continuum and complete save; muted output')as(out,scratch):
    c=Capture(args,out,scratch,'m07','phonation-synthesis',reaper_binary=ROOT/'phonetic_toolbox/core/acoustic/reaper.exe');print(out,flush=True)
    from PyQt6.QtWidgets import QFileDialog
    saved=scratch/'saved';saved.mkdir();inputs=ROOT/'manual/assets/software-only'
    input_hashes={name:hashlib.sha256((inputs/name).read_bytes()).hexdigest()for name in ('single-original.wav','single-target.wav')}
    choices={'directory':inputs};QFileDialog.getExistingDirectory=lambda *a,**k:str(choices['directory'])
    QFileDialog.getSaveFileName=lambda *a,**k:(str(saved/Path(a[2]).name),'')
    report=c.report;report['inputHashes']=input_hashes;report['settings']={'f0_backend':'parselmouth','min_f0_hz':50,'max_f0_hz':600,'f0_frame_interval_ms':1,'point_count':21,'alignment':'normalize','continuum_type':2,'reverse_direction':False,'step_count':9,'energy_match':True,'normalize_to_source':True,'output_peak_limit':.98}
    js,until,click,pause=c.js,c.until,c.click,c.pause
    def fill(label,value):c.fill('[aria-label='+json.dumps(c.ui.label(label),ensure_ascii=False)+']',value)
    def select(label,name):
        selector='select[aria-label='+json.dumps(c.ui.label(label),ensure_ascii=False)+']'
        until('!![...document.querySelector('+json.dumps(selector)+').options].find(x=>x.textContent==='+json.dumps(name)+')')
        js('(()=>{const e=document.querySelector('+json.dumps(selector)+');e.value=[...e.options].find(x=>x.textContent==='+json.dumps(name)+').value;e.dispatchEvent(new Event("change",{bubbles:true}));})()');pause(300)
    def notice(key):until('document.body.innerText.includes('+json.dumps(c.ui.key('modules.phonation-synthesis.'+key))+')')
    def idle():until('document.querySelector(".m07-page")?.getAttribute("aria-busy")==="false"',180)
    def capture(name,caption,state,settle=True):
        regions={'table':'.table-scroll thead','rows':'.table-scroll tbody tr:nth-child(2)','source':'.table-scroll tbody tr:first-child td:nth-child(2) input','target':'.table-scroll tbody tr:first-child td:nth-child(3) input'}if name=='m07-f0-edit-pending-light'else{}
        if args.details_only and name!='m07-f0-edit-pending-light':return
        c.shot(name,regions);report['captures'][-1].update(caption=caption,state=state)
    try:
        until('!!document.querySelector(".m07-page")');c.prepare()
        click('打开音频目录')
        select('源音频', 'single-original.wav')
        select('目标音频', 'single-target.wav')
        until('[...document.querySelectorAll("button")].some(b=>b.textContent.trim()==='+json.dumps(c.ui.label('提取 F0'))+'&&!b.disabled)')
        assert js('document.querySelectorAll(".table-scroll tbody tr").length') == 0
        capture('m07-input-loaded-light', '源与目标 WAV 已载入，两幅实际波形可见；F0 控制表和合成结果尚未建立。', 'two-inputs-loaded-no-analysis')

        fill('最高 F0 (Hz)', 600)
        click('提取 F0')
        idle()
        notice('analysis_complete_control_points_are_unchanged')
        assert js('document.querySelectorAll(".table-scroll tbody tr").length') == 21
        assert js('document.querySelectorAll(".f0-plot [data-curve=synthesis]").length') == 0
        capture('m07-f0-extracted-light', '提取 F0 后出现源与目标控制曲线及 21 点数值表；合成音频仍为空。示例最高 F0 为 600 Hz。', 'analysis-complete-no-generation')

        js('document.querySelector(".m07-page details").open=true')
        pause(150)
        capture('m07-advanced-parameters-light', '展开高级分析参数，核对帧长、帧移、LPC 阶数、预加重和脉冲设置；改变这些值需要重新分析。', 'advanced-analysis-expanded')
        js('document.querySelector(".m07-page details").open=false')
        c.fill('.table-scroll tbody tr:first-child td:nth-child(2) input',110)
        notice('control_points_or_alignment_have_unapplied_changes_click_apply')
        assert js('[...document.querySelectorAll("button")].find(x=>x.textContent.trim()==='+json.dumps(c.ui.label('生成当前'))+').disabled')
        capture('m07-f0-edit-pending-light', '修改第一项源 F0 后，控制表与图面立即更新，右栏提示未应用编辑，生成按钮暂时禁用。', 'f0-first-point-edited-pending-apply')
        if args.details_only:report['success']=True;return
        click('应用编辑')
        idle()
        notice('f0_edits_applied_lpc_residuals_and_pulses_are_reused')
        capture('m07-f0-edit-applied-light', '应用编辑成功后形成新的 F0 快照，提示 LPC、残差和脉冲继续复用，生成按钮恢复可用。', 'f0-edit-applied')

        # Reanalyze the unchanged example inputs before the demonstration group.
        click('提取 F0')
        idle()
        notice('analysis_complete_control_points_are_unchanged')
        click('生成当前')
        if js('document.querySelector(".m07-page").getAttribute("aria-busy")==="true"'):
            capture('m07-generation-task-light', '当前九步连续统已提交，任务状态与批次完成数显示在右栏；成功组才会提供试听和保存。', 'actual-generation-in-progress', settle=False)
        idle()
        until('document.querySelectorAll(".result-audios button").length===10&&document.querySelectorAll(".f0-plot [data-curve=synthesis]").length===9')
        capture('m07-nine-step-result-light', '源到目标、仅 F0 变化的九步结果已完整成功，整组波形、各步生成控制 F0、任务描述和保存入口可见。', 'nine-step-group-complete')
        assert js('(()=>{const b=[...document.querySelectorAll(".result-audios button")].find(b=>b.textContent.trim()==="step05");if(!b||b.disabled)return false;b.click();return true})()');pause(200)
        until('document.querySelectorAll(".f0-plot [data-curve=synthesis]").length===1')
        pause(700)
        capture('m07-step05-comparison-light', '选择 step05 后显示该步波形及生成控制 F0；当前试听信息明确标出同一组的第五步。', 'selected-step05-single-wave-and-curve')
        # Select the complete group without introducing fake playback positions.
        click('整组试听')
        until('document.querySelectorAll(".f0-plot [data-curve=synthesis]").length===9')
        assert js('(()=>{const e=document.querySelector(".synthesized-plot .audio-transport button:nth-child(2)");e.click();return true})()')
        choices['directory'] = saved
        click('输出位置')
        click('保存完整组')
        notice('saved_all_files_and_parameter_manifest_for_this_group')
        groups=[p for p in saved.iterdir()if p.is_dir()]
        report['savedDirectories']=[p.name for p in groups]
        assert len(groups)==1,report['savedDirectories']
        group=groups[0]
        files = list(group.iterdir())
        assert len(files) == 12
        capture('m07-save-complete-light', '完整九步组已保存，右栏显示保存成功，顶部输出位置供核对；PNG 仍需独立导出。', 'complete-group-saved')
        report['savedFiles'] = [{'name': f.name, 'sizeBytes': f.stat().st_size,
                                 'sha256': hashlib.sha256(f.read_bytes()).hexdigest()} for f in files]
        report['checks'] += ['Two registered original inputs read without modifying them',
                             'Actual analysis, edit/apply, reanalysis, nine-step generation and complete-group save',
                             'Distinct UI state evidence retained for every capture',
                             'All screenshots captured from the maximized actual Qt window']
        assert input_hashes == {name: hashlib.sha256((inputs / name).read_bytes()).hexdigest() for name in input_hashes}
        assert not report.get('terminations')
        report['success'] = True
    except BaseException as exc:
        report['error']=repr(exc);report['failureDom']=js('({lang:document.documentElement.lang,text:document.body.innerText,selects:[...document.querySelectorAll(\"select\")].map(e=>({aria:e.getAttribute(\"aria-label\"),html:e.outerHTML}))})');c.shot('failure-diagnostic');raise
    finally:c.close()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--locale',choices=('en','ja'),required=True);p.add_argument('--details-only',action='store_true');p.add_argument('--frontend-dist',type=Path,default=ROOT/'frontend/dist');run(p.parse_args())
