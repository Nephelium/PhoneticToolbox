"""R18 additions to the owned Qt regression, no separate server or user profile."""
import json
import subprocess

def exercise(local,click,until,settled,capture,w,target,report,drag_handle):
    click('#tab-organs');click('#sagittalButton');click('[data-preset=a]')
    anchor=local('return win.qaViewer.state.contours.lower_cover[10]')
    before=local('return win.qaViewer.state.tube_areas')
    def arrow(handle,key,count=1):
        for _ in range(count):
            local('d.querySelector("[data-handle='+handle+']").dispatchEvent(new win.KeyboardEvent("keydown",{key:'+json.dumps(key)+',bubbles:true}));return true;');settled()
    drag_handle('surface',0,-18)
    assert local('return win.qaViewer.state.tongue_shape[1]')>0
    drag_handle('belly',-12,0)
    assert local('return win.qaViewer.state.tongue_shape[2]')<0
    arrow('surface','ArrowUp',4)
    shape=local('return win.qaViewer.state.tongue_shape')
    assert shape[1]>.1 and local('return win.qaViewer.state.tube_areas')!=before
    arrow('belly','ArrowDown',3)
    assert local('return win.qaViewer.state.tongue_shape[3]')<0
    assert local('return win.qaViewer.state.contours.lower_cover[10]')==anchor
    after=local('return win.qaViewer.state.tongue_shape')
    click('#undoButton');assert local('return win.qaViewer.state.tongue_shape')!=after
    click('#redoButton');assert local('return win.qaViewer.state.tongue_shape')==after
    capture('r18-local-controls');click('#threeButton');capture('r18-local-controls-3d')
    assert local('return win.qaViewer.handles().filter(h=>["surface","belly"].includes(h.name)).every(h=>h.p.every(Number.isFinite))')
    click('#savePreset');click('#presetInputButton')
    local('d.querySelector("#presetInput").value="qa-local";return true;');click('#presetInputSave')
    until('!d.querySelector("#presetDialog").open')
    click('#customPreset');assert local('return !d.querySelector("[data-ipa=a]").disabled')
    click('#presetFavorites');click('[data-common-name="qa-local"]')
    until('[...d.querySelectorAll("#presets button")].some(b=>b.textContent==="qa-local")')
    assert local('return d.querySelector("[data-common-name=qa-local]").getAttribute("aria-pressed")==="true"')
    click('#presetFavorites')
    assert local('return win.getComputedStyle(d.querySelector("[data-ipa=qa-local]"),"::after").content.includes("★")')
    click('#presetFavorites')
    click('[data-common-id=\"a\"]');assert not local('return !!d.querySelector("#presets [data-preset=a]")')
    click('[data-common-id=\"a\"]')
    capture('r18-common-presets');click('#presetCancel')
    custom=w.vocal.invoke('presets/load',{});favorite=w.vocal.invoke('favorites/load',{})
    assert custom[-1]['id'] in favorite and custom[-1]['tongue_shape']==after
    click('[data-preset=a]')
    local('[...d.querySelectorAll("#presets button")].find(b=>b.textContent==="qa-local").click();return true;');settled()
    assert local('return win.qaViewer.state.tongue_shape')==after
    click('#tab-motion')
    if local('return d.querySelectorAll("#keyframeList .pose-card").length'):click('#clearFrames')
    else:assert local('return d.querySelector("#clearFrames").disabled')
    click('#captureFrame');click('#captureAspiration')
    assert local('return d.querySelectorAll("#keyframeList .pose-card").length')==2
    assert local('return d.querySelectorAll("#keyframeList .pose-card")[1].querySelector(".pose-name").value')=='ʰ'
    local('const e=d.querySelectorAll("#keyframeList .pose-card")[1].querySelector(".pose-name");e.value="ʰ（短）";e.dispatchEvent(new Event("change"));return true;')
    local('const e=d.querySelectorAll("#keyframeList .pose-card")[1].querySelector("input[type=number]");e.value="0.01";e.dispatchEvent(new Event("change"));return true;')
    click('#tab-organs');click('[data-preset=i]');click('#tab-motion');click('#captureFrame')
    click('#exportFrames');until('d.querySelector("#frameStatus").textContent.includes("已导出")')
    doc=json.loads(target.read_text('utf-8'));assert doc['version']==5 and doc['frames'][1]=={'aspiration':True,'duration':.01,'name':'ʰ（短）'}
    assert doc['frames'][0]['tongue_shape']==after
    click('#importFrames');until('d.querySelector("#frameStatus").textContent.includes("已导入")')
    click('#playFrames');until('!d.body.classList.contains("motion-busy")')
    assert '失败' not in local('return d.querySelector("#frameStatus").textContent')
    capture('r18-aspiration-frames')
    video=target.with_name('aspiration.mp4')
    w.bridge.test_vocal_picker=lambda op:str(video) if op=='video/begin' else str(target)
    click('#exportVideo');click('#videoStart')
    until('!d.querySelector("#videoStart").disabled',seconds=60)
    assert local('return d.querySelector("#videoStatus").textContent').startswith('已保存')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)]))
    assert {s['codec_name'] for s in probe['streams']}=={'h264','aac'}
    v=next(s for s in probe['streams'] if s['codec_type']=='video')
    assert int(v['nb_frames'])==13 and abs(float(probe['format']['duration'])-.41)<.04
    # At 30 fps, frame 6 samples exactly the 0.20 s aspiration start.
    picture=video.parent.parent/'r18-aspiration-video.png'
    subprocess.run(['ffmpeg','-v','error','-i',str(video),'-vf','select=eq(n\\,6)','-frames:v','1',str(picture)],check=True)
    report['checks'].append({'aspiration_video':True,'codecs':['h264','aac'],'frames':13,'duration':probe['format']['duration'],'frame_at_aspiration':6})
    click('#videoClose')
    report['checks'].append({'r18_actual_pointer_drag':True,'r18_local_handles':True,'native_acoustics_changed':True,'fixed_anchor':True,'undo_redo':True,'favorite_custom_roundtrip':True,'builtin_lit':True,'v5_roundtrip':True,'aspiration_default_superscript_and_custom_name':True,'aspiration_10ms_playback':True,'shape':after})
