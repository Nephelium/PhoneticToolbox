import {canvasFont} from './fonts.js';
import {sectionView} from './section-view.mjs';
import {areaReadout} from './area-view.mjs';
import {tint} from './theme.js';
import {request,response,ipaChart} from './platform.js';
import {VocalTractViewer} from './viewer.js';
import {setupMonitor} from './monitor.js';
import {nearestSection} from './geometry.mjs';
import {setupPresets,builtinPose} from './presets.js';
import {setupKeyframes} from './keyframes.js';
import {moveControl,PoseHistory,ORGAN_PARAMS} from './controls.mjs';

const $=id=>document.getElementById(id);
const accountSettingsKey=new URLSearchParams(location.search).get('settings');
const preferenceKey=accountSettingsKey&&/^web-settings\.v1\.[^:]+\.M10$/.test(accountSettingsKey)?'ptb.v3.'+accountSettingsKey+'.frame':null;
let retainedSettings={};
const preferencesReady=preferenceKey?new Promise((resolve,reject)=>{
 const timer=setTimeout(()=>{window.removeEventListener('message',receive);reject(Error('账号设置读取超时，请重新打开此模块。'));},15000);
 function receive(event){if(event.source!==parent||event.origin!==location.origin||event.data?.type!=='m10-settings-value'||event.data.key!==accountSettingsKey)return;clearTimeout(timer);window.removeEventListener('message',receive);retainedSettings=event.data.value??{};resolve();}
 window.addEventListener('message',receive);parent.postMessage({type:'m10-settings-read'},location.origin);
}):Promise.resolve();
const modelSettingKeys=['f0','source','keepVowel','pose','perturbation'];
function retainSettings(values){if(!preferenceKey)return;const speaker=meta?.speaker||'JD2',models={...(retainedSettings.models||{})};models[speaker]={...(models[speaker]||{}),...Object.fromEntries(modelSettingKeys.filter(k=>k in retainedSettings).map(k=>[k,retainedSettings[k]])),...Object.fromEntries(modelSettingKeys.filter(k=>k in values).map(k=>[k,values[k]]))};retainedSettings={...retainedSettings,...values,models};parent.postMessage({type:'m10-settings-write',body:retainedSettings},location.origin);}
let perturbation=null,keyframeEditor,perturbationQueue=Promise.resolve(),unsupportedPerturbation=false;
let sourceSettings,manualRoot=false,presetLibrary,larynxHeight=0,tongueShape=[0,0,0,0],lipShape=[0,0,0,0];
let meta,params,snapshot,token,revision=0,dirty=false,inFlight=false,live=false,closed=false,lastGeometry='';
let toastTimer,selectedPreset='a',rendering=false,lipWidth=1,motionBusy=false,playbackError='';
let keyframeBusy=false,vowelBusy=false;
function applyPoseLock(){
  const wasBusy=motionBusy;motionBusy=keyframeBusy||vowelBusy;viewer.playing=motionBusy;
  document.body.classList.toggle('motion-busy',motionBusy);document.body.dataset.vowelBusy=String(vowelBusy);
  document.querySelectorAll('#parameterGroups input,.lip-width-panel input,.source-panel button,.source-panel input,.source-panel select,.edit-panel button,#presets button,#portRange,#portToggle,#keepVowel,#sliceRange,#sectionCount,#speakerSelect,#customPreset,#savePreset,#manualRoot').forEach(e=>e.disabled=motionBusy);
  if(meta?.host==='web')document.querySelectorAll('#oneSecondButton,#testToneButton,#volumeRange,#outputDevice,#gainSelect').forEach(e=>e.disabled=motionBusy||!!meta.audio_locked);
  $('liveButton').disabled=motionBusy||!!meta?.audio_locked||(meta?.host==='web'&&!meta?.continuous_audio);
  if(unsupportedPerturbation){$('oneSecondButton').disabled=true;for(const id of ['jitterRange','shimmerRange','perturbationSeed'])$(id).disabled=true;}
  updateControls();updateHistory();if(wasBusy&&!motionBusy){dirty=true;queuePose();}
}
function setVowelBusy(value){if(vowelBusy!==value){vowelBusy=value;applyPoseLock();}}
const inputs=new Map();
const history=new PoseHistory();let selectedSide=2,lastOpenArea=.7;
const currentEdit=()=>({params:[...params],preset:selectedPreset,name:presetLibrary?.getName(selectedPreset)||'',manual_root:manualRoot,larynx_height:larynxHeight,tongue_shape:[...tongueShape],lip_shape:[...lipShape],lip_width:lipWidth,f0:+$('f0Range').value,source:{...sourceSettings}});
function beginEdit(){if(params)history.begin(currentEdit());}
function endEdit(){if(params)history.commit(currentEdit());updateHistory();}
function updateHistory(){$('undoButton').disabled=motionBusy||!history.past.length;$('redoButton').disabled=motionBusy||!history.future.length;}
function restoreEdit(state){if(!state)return;manualRoot=state.manual_root??false;larynxHeight=state.larynx_height??0;tongueShape=[...(state.tongue_shape??[0,0,0,0])];lipShape=[...(state.lip_shape??[0,0,0,0])];params=[...state.params];sourceSettings={...(state.source||meta.source_presets.voiced)};selectedPreset=state.preset||'';lipWidth=state.lip_width??1;$('f0Range').value=state.f0??meta.default_f0??120;$('f0Output').textContent=$('f0Range').value+' Hz';updateControls();dirty=true;queuePose();updateHistory();}
function saveView(){const view={mode:viewer.mode,teeth:viewer.showTeeth,head:viewer.showHead,nose:viewer.showNose,labels:viewer.labels,fullModel:viewer.fullModel};if(preferenceKey){retainSettings({view});return;}try{localStorage.setItem('vtl-view-v2',JSON.stringify(view));}catch{}}
function selectInspector(organ){
  const labels={tongue:'舌头',lips:'唇与下颌',velum:'软腭与咽部'};
  document.querySelectorAll('[data-select-organ]').forEach(b=>{b.classList.toggle('selected',b.dataset.selectOrgan===organ);b.setAttribute('aria-pressed',b.dataset.selectOrgan===organ);});
  $('resetOrgan').textContent='重置'+labels[organ];$('editHint').textContent=organ==='tongue'?'拖动舌背、舌面、舌叶、舌尖与舌底；侧缘在横截面与三维中调整，左右联动。':organ==='lips'?'分别拖动上唇或下唇调整前后、高低；正面圆点调整唇宽。':'软腭抬起关闭，下降开放。使用面积滑块微调。';
  document.querySelectorAll('#parameterGroups .parameter-group').forEach(d=>{d.open=true;d.hidden=false;});
  document.querySelector('.lip-width-panel').hidden=false;$('velumControls').hidden=false;
  document.dispatchEvent(new CustomEvent('inspector-organ')); 
  if(snapshot)drawCharts();
}
function toast(message){$('toast').textContent=message;$('toast').classList.add('show');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').classList.remove('show'),4500);}
async function post(path,body={}){
  const r=await response(path,body);
  if(!r.ok){const e=await r.json();if(r.status===409)revision=Math.max(revision,e.revision);throw Error(e.error||r.statusText);}
  if(path==='audio/settings')retainSettings({audio:{...retainedSettings.audio,...body}});
  if(path==='pose'||path==='source/update')retainSettings({sectionCount:body.section_count,f0:body.f0,source:body.source,keepVowel:body.keep_vowel,pose:{params:body.params,manualRoot:body.manual_root,larynxHeight:body.larynx_height,tongueShape:body.tongue_shape,lipShape:body.lip_shape,lipWidth:body.lip_width}});
  return r;
}
function markChanged(){dirty=true;selectedPreset='';updateControls();queuePose();}
const geometryKey=()=>JSON.stringify([params,manualRoot,larynxHeight,tongueShape,lipShape,lipWidth,+$('sectionCount').value,$('keepVowel').checked]);
async function queuePose(){
  if(inFlight||!dirty||closed||!params||motionBusy)return;
  inFlight=true;document.body.dataset.posePending='true';dirty=false;const rev=++revision,geometry=geometryKey(),sourceOnly=!!(meta.source_updates&&snapshot&&geometry===lastGeometry);
  try{
    const r=await post(sourceOnly?'source/update':'pose',{params:[...params],manual_root:manualRoot,larynx_height:larynxHeight,tongue_shape:[...tongueShape],lip_shape:[...lipShape],revision:rev,section_count:+$('sectionCount').value,section:+$('sliceRange').value,f0:+$('f0Range').value,lip_width:lipWidth,source:{...sourceSettings},keep_vowel:$('keepVowel').checked});
    const value=await r.json(),state=sourceOnly?{...snapshot,...value,compute_ms:0}:value;
    if(state.revision===rev){
      snapshot=state;try{localStorage.setItem(meta.host==='web'?'m10-web-section-count':'m10-section-count-r19',String(state.section_count));}catch{}
      if(!dirty){params=[...state.params];lipWidth=state.lip_width;larynxHeight=state.larynx_height??0;tongueShape=[...(state.tongue_shape??[0,0,0,0])];lipShape=[...(state.lip_shape??[0,0,0,0])];if(!manualRoot)for(const key of ['TRX','TRY']){const i=meta.parameters.findIndex(p=>p.name===key);params[i]=state.limited[i];}updateControls();}
      if(!sourceOnly){lastGeometry=dirty?geometry:geometryKey();updateModel(state);drawCharts();}
      $('sceneMetrics').textContent=state.uvula_contact_lift>1e-5?'小舌已接触舌面':'软腭 · 同一三维模型截面';document.body.dataset.computeMs=String(Math.round(state.compute_ms));
      const limited=state.nasal_constraint;$('portConstraint').classList.toggle('limited',!!limited);$('portConstraint').textContent=limited?`该舌位下开口已限制为 ${limited.accepted_area.toFixed(2)} cm²，避免新产生口腔闭塞。`:'软腭下降开放，抬起关闭。';
      if(!sourceOnly&&meta.parameters.some((p,i)=>ORGAN_PARAMS[viewer.selected].includes(p.name)&&Math.abs(state.params[i]-state.limited[i])>.04))$('editHint').textContent='已达到器官约束，显示的是引擎允许的构形。';
    }
  }catch(e){toast(e.message);}finally{inFlight=false;document.body.dataset.poseRevision=String(rev);document.body.dataset.posePending=String(dirty);if(dirty)setTimeout(queuePose,0);}
}
$('sectionCount').onchange=()=>{const old=snapshot?.centerline.length??129,count=+$('sectionCount').value;$('sliceRange').value=Math.round(+$('sliceRange').value*(count-1)/(old-1));dirty=true;queuePose();};
function updateControls(){
  $('manualRoot').checked=manualRoot;
  if($('larynxHeight')){$('larynxHeight').value=larynxHeight;$('larynxHeightValue').textContent=larynxHeight.toFixed(2)+' cm';}
  for(let i=0;i<4;i++){const input=$('lip-shape-'+i);if(input){input.value=lipShape[i];$('lip-shape-value-'+i).textContent=lipShape[i].toFixed(2)+' cm';}}
  updateSourceControls();
  for(const [name,{input,out,index}] of inputs){input.value=params[index];out.textContent=params[index].toFixed(2);}
  document.querySelectorAll('#presets button').forEach(b=>b.classList.toggle('selected',b.dataset.preset===selectedPreset));
  presetLibrary?.select(selectedPreset);
  const area=Math.max(0,params[meta.parameters.findIndex(p=>p.name==='VO')]);if(area>0)lastOpenArea=area;
  $('lipWidthRange').value=Math.round(lipWidth*100);$('lipWidthValue').textContent=Math.round(lipWidth*100)+'%';
  $('portRange').value=area;$('portValue').textContent=area.toFixed(2)+' cm²';$('portToggle').setAttribute('aria-checked',area>0);$('portToggle').textContent=area>0?'已开放':'已关闭';
}
function choosePreset(name){beginEdit();restoreEdit(builtinPose(meta,name));endEdit();}
function updateSourceControls(){
  if(!sourceSettings)return;
  $('sourceMode').value=sourceSettings.mode;
  for(const [id,key,out,digits] of [['pressureRange','pressure_pa','pressureValue',0],['openingRange','opening_mm','openingValue',2],['chinkRange','posterior_gap_mm2','chinkValue',1]]){$(id).value=sourceSettings[key];$(out).textContent=sourceSettings[key].toFixed(digits);}
  $('f0Range').disabled=motionBusy||sourceSettings.vibration===0;
  $('sourceHint').textContent=sourceSettings.vibration===0?'周期振动已关闭；减小开度使声门更内收。F0 在此模式不生效。':'减小声门半宽或后部缝隙表示更内收；气流速度由气压和声道阻力共同决定。';
  if(sourceSettings.mode==='whisper')$('sourceHint').textContent+=` 耳语试听补偿 ${sourceSettings.audition_gain_db??24} dB，播放与视频一致。`;
  if(sourceSettings.mode==='voiceless')$('sourceHint').textContent+=' 舌尖擦音需保留中央窄缝并抬高前侧缘；声音过轻可调整弱音增益。';
}
function setupControls(){
  sourceSettings={...meta.source_presets.voiced};
  const pose=preferenceKey?retainedSettings.pose:null;
  if(pose&&Array.isArray(pose.params)&&pose.params.length===meta.parameters.length&&pose.params.every(Number.isFinite)){params=[...pose.params];manualRoot=pose.manualRoot===true;if(Number.isFinite(pose.larynxHeight))larynxHeight=pose.larynxHeight;if(Array.isArray(pose.tongueShape)&&pose.tongueShape.length===4&&pose.tongueShape.every(Number.isFinite))tongueShape=[...pose.tongueShape];if(Array.isArray(pose.lipShape)&&pose.lipShape.length===4&&pose.lipShape.every(Number.isFinite))lipShape=[...pose.lipShape];if(Number.isFinite(pose.lipWidth))lipWidth=pose.lipWidth;}
  if(preferenceKey){for(const key of Object.keys(sourceSettings))if(typeof retainedSettings.source?.[key]===typeof sourceSettings[key])sourceSettings[key]=retainedSettings.source[key];if(Number.isFinite(retainedSettings.f0))$('f0Range').value=retainedSettings.f0;if(typeof retainedSettings.keepVowel==='boolean')$('keepVowel').checked=retainedSettings.keepVowel;}
  $('sourceMode').onchange=()=>{if(!$('sourceMode').value||$('sourceMode').value==='transition')return;beginEdit();sourceSettings={...meta.source_presets[$('sourceMode').value],pressure_pa:sourceSettings.pressure_pa};dirty=true;updateSourceControls();queuePose();endEdit();};
  for(const [id,key] of [['pressureRange','pressure_pa'],['openingRange','opening_mm'],['chinkRange','posterior_gap_mm2']]){
    $(id).onpointerdown=beginEdit;$(id).onkeydown=beginEdit;$(id).onchange=endEdit;
    $(id).oninput=()=>{beginEdit();sourceSettings[key]=+$(id).value;dirty=true;updateSourceControls();queuePose();};
  }
  $('oneSecondButton').textContent=meta.host==='web'?'▶ 测试发声':'▶ 发声 1 秒';
  $('oneSecondButton').onclick=()=>preview(false,meta.host==='web'?.5:1);
  $('gainSelect').value=meta.audio_settings.gain_db||0;
  $('gainSelect').onchange=async()=>{try{updateAudio(await(await post('audio/settings',{gain_db:+$('gainSelect').value})).json());}catch(e){toast(e.message);}};

  const symbols={E:'ɛ','2':'ø'};
  presetLibrary=setupPresets({initial:meta.custom_presets,initialLibrary:meta.preset_library,builtins:Object.keys(meta.presets).map(id=>({...builtinPose(meta,id),duration:.2,id,name:symbols[id]||id,builtin:true})),favorites:meta.favorites,chooseBuiltin:choosePreset,chart:meta.ipa_chart,getPose:currentEdit,post,isReady:()=>!motionBusy&&!dirty&&!inFlight,onSaved:p=>{selectedPreset=p.id;updateControls();},onDeleted:id=>{if(selectedPreset===id)selectedPreset='';updateControls();},notify:toast,apply:p=>{beginEdit();restoreEdit({...p,preset:p.id,source:p.source||sourceSettings});endEdit();}});
  const groups=[['嘴唇与下颌',['JA','LP','LD'],false,'lips'],['舌头',['TCX','TCY','TBX','TBY','TTX','TTY'],true,'tongue'],['舌根',['TRX','TRY'],true,'tongue'],['舌骨',['HX','HY'],true,'tongue'],['软腭',['VS'],false,'velum'],['舌侧缘',['TS1','TS2','TS3'],true,'tongue']];
  for(const [label,names,open,organ] of groups){const details=document.createElement('section');details.className='parameter-group';details.dataset.organ=organ;const summary=document.createElement('h3');summary.className='group-title';summary.textContent=label;details.append(summary);
    for(const name of names){const index=meta.parameters.findIndex(x=>x.name===name),p=meta.parameters[index],row=document.createElement('div');row.className='parameter';
      const label=document.createElement('label');label.htmlFor='p-'+name;label.textContent=p.label;const small=document.createElement('span');small.textContent=name;label.append(small);
      const out=document.createElement('output');out.htmlFor='p-'+name;const input=document.createElement('input');Object.assign(input,{id:'p-'+name,type:'range',min:p.min,max:p.max,step:.01,value:params[index]});
      input.addEventListener('pointerdown',beginEdit);input.addEventListener('keydown',beginEdit);input.addEventListener('change',endEdit);input.addEventListener('blur',endEdit);
      input.addEventListener('input',()=>{beginEdit();if(['TRX','TRY'].includes(name))manualRoot=true;if(name==='JA'&&$('jawLink').checked){const j=meta.parameters.findIndex(p=>p.name==='LD'),p=meta.parameters[j];params[j]=Math.max(p.min,Math.min(p.max,params[j]-(+input.value-params[index])*.22));}params[index]=+input.value;markChanged();});row.append(label,out,input);details.append(row);inputs.set(name,{input,out,index});
    }$('parameterGroups').append(details);
  }
  const lipPanel=document.createElement('details');lipPanel.className='lip-independent';lipPanel.innerHTML='<summary>上下唇独立调整</summary>';
  for(const [i,label] of ['上唇前后','上唇高低','下唇前后','下唇高低'].entries()){const row=document.createElement('div');row.className='parameter';const title=document.createElement('label');title.htmlFor='lip-shape-'+i;title.textContent=label;const output=document.createElement('output');output.id='lip-shape-value-'+i;const input=document.createElement('input');Object.assign(input,{id:'lip-shape-'+i,type:'range',min:-.8,max:.8,step:.01,value:0});input.onpointerdown=beginEdit;input.onkeydown=beginEdit;input.onchange=endEdit;input.oninput=()=>{beginEdit();lipShape[i]=+input.value;markChanged();};row.append(title,output,input);lipPanel.append(row);}
  $('parameterGroups').querySelector('[data-organ="lips"]').append(lipPanel);
  const larynxPanel=document.createElement('section');larynxPanel.className='parameter-group';
  larynxPanel.innerHTML='<h3 class="group-title">喉部</h3><div class="parameter" id="larynxControl"><label for="larynxHeight">声门高低<span>相对舌骨</span></label><output id="larynxHeightValue" for="larynxHeight"></output><input id="larynxHeight" type="range" min="-1" max="1" step="0.01" value="0"></div>';
  $('parameterGroups').insertBefore(larynxPanel,$('parameterGroups').firstChild);
  $('larynxHeight').onpointerdown=beginEdit;$('larynxHeight').onkeydown=beginEdit;$('larynxHeight').onchange=endEdit;
  $('larynxHeight').oninput=()=>{beginEdit();larynxHeight=+$('larynxHeight').value;markChanged();};
  $('manualRoot').onchange=()=>{beginEdit();manualRoot=$('manualRoot').checked;markChanged();endEdit();};
  $('undoButton').onclick=()=>restoreEdit(history.undo(currentEdit()));$('redoButton').onclick=()=>restoreEdit(history.redo(currentEdit()));
  $('resetOrgan').onclick=()=>{beginEdit();if(viewer.selected==='tongue')tongueShape=[0,0,0,0];if(viewer.selected==='lips'){lipWidth=1;lipShape=[0,0,0,0];}if(viewer.selected==='velum')larynxHeight=0;for(const name of ORGAN_PARAMS[viewer.selected]){const i=meta.parameters.findIndex(p=>p.name===name);params[i]=meta.presets.a[i];}markChanged();endEdit();};
  document.querySelectorAll('[data-select-organ]').forEach(b=>b.onclick=()=>viewer.select(b.dataset.selectOrgan));
  const portIndex=meta.parameters.findIndex(p=>p.name==='VO');$('portRange').max=meta.parameters[portIndex].max;
  $('portRange').onpointerdown=beginEdit;$('portRange').onkeydown=beginEdit;$('portRange').onchange=endEdit;
  $('portRange').oninput=()=>{beginEdit();params[portIndex]=+$('portRange').value||meta.parameters[portIndex].min;markChanged();};
  $('portToggle').onclick=()=>{beginEdit();params[portIndex]=params[portIndex]>0?meta.parameters[portIndex].min:Math.min(lastOpenArea,meta.parameters[portIndex].max);markChanged();endEdit();};
  $('portFocus').onclick=()=>viewer.select('velum');$('keepVowel').onchange=()=>{dirty=true;queuePose();};
  $('lipWidthRange').onpointerdown=beginEdit;$('lipWidthRange').onkeydown=beginEdit;$('lipWidthRange').onchange=endEdit;$('lipWidthRange').oninput=()=>{beginEdit();lipWidth=+$('lipWidthRange').value/100;markChanged();};
  document.addEventListener('keydown',e=>{if(motionBusy||e.target.matches('input,select,textarea'))return;if((e.ctrlKey||e.metaKey)&&['z','y'].includes(e.key.toLowerCase())){e.preventDefault();restoreEdit((e.key.toLowerCase()==='y'||e.shiftKey)?history.redo(currentEdit()):history.undo(currentEdit()));}});
  $('f0Range').oninput=()=>{$('f0Output').textContent=$('f0Range').value+' Hz';dirty=true;queuePose();};
$('sliceRange').oninput=()=>selectSection(+$('sliceRange').value);
  $('liveButton').onclick=async()=>{if(meta.host==='web'&&!meta.continuous_audio)return;try{if(!live&&(dirty||inFlight)){toast('正在更新构形，请稍后开始发声');return;}const s=await(await post('live',{active:!live})).json();updateAudio(s);}catch(e){toast(e.message);}};
  if(meta.host==='web'&&!meta.continuous_audio){
    const button=$('liveButton'),hint='由于服务器性能限制，网页版暂不支持持续发声，请下载桌面版。';
    button.disabled=true;button.title=hint;const wrapper=document.createElement('span');wrapper.className='web-disabled-action';wrapper.tabIndex=0;wrapper.title=hint;wrapper.setAttribute('aria-label',hint);button.before(wrapper);wrapper.append(button);
  }
  for(const d of meta.output_devices){const option=document.createElement('option');option.value=d.id;option.textContent=d.name;$('outputDevice').append(option);}
  $('outputDevice').value=meta.audio_settings.device;$('volumeRange').value=Math.round(meta.audio_settings.volume*100);$('volumeOutput').textContent=$('volumeRange').value+'%';
  $('outputDevice').onchange=async()=>{try{updateAudio(await(await post('audio/settings',{device:$('outputDevice').value})).json());toast('输出设备已切换');}catch(e){toast(e.message);}};
  let volumeTimer;$('volumeRange').oninput=()=>{$('volumeOutput').textContent=$('volumeRange').value+'%';clearTimeout(volumeTimer);volumeTimer=setTimeout(async()=>{try{await post('audio/settings',{volume:+$('volumeRange').value/100});}catch(e){toast(e.message);}},100);};
  $('testToneButton').onclick=async()=>{try{updateAudio(await(await post('audio/test')).json());}catch(e){toast(e.message);}};
  $('aboutButton').onclick=()=>$('aboutDialog').showModal();$('sharedReferences').onclick=()=>parent.postMessage({type:'m10-references'},'*');$('sharedHelp').onclick=()=>parent.postMessage({type:'m10-help'},'*');$('closeAbout').onclick=()=>$('aboutDialog').close();
  $('sourceState').textContent=meta.audio_locked?'静音测试':'手动播放';
  if(meta.audio_locked){document.body.classList.add('audio-locked');document.querySelectorAll('#oneSecondButton,#liveButton,#testToneButton,#volumeRange,#outputDevice,#gainSelect').forEach(e=>e.disabled=true);$('audioStatus').textContent='静音建模已锁定，后端不允许音频输出。';}
  keyframeEditor=setupKeyframes({metadata:meta,getCatalog:()=>[...Object.entries(meta.presets).map(([id,p])=>({...builtinPose(meta,id),duration:.2,name:({E:'ɛ','2':'ø'}[id]||id)})),...presetLibrary.getCustom()],initialGlobals:meta.saved_globals,initialFeatures:meta.saved_features,initialFrames:meta.saved_frames,initialCurve:meta.saved_curve,initialPressureCurve:meta.saved_pressure,getPerturbation:()=>perturbation,setPerturbation,canSynthesize:()=>!unsupportedPerturbation,getPose:currentEdit,applyPose:(state,animated=false)=>{if(animated){manualRoot=state.manual_root??false;larynxHeight=state.larynx_height??0;tongueShape=[...(state.tongue_shape??[0,0,0,0])];lipShape=[...(state.lip_shape??[0,0,0,0])];params=[...state.params];sourceSettings={...(state.source||meta.source_presets.voiced)};lipWidth=state.lip_width;selectedPreset='';snapshot=state;$('f0Range').value=state.f0;$('f0Output').textContent=Math.round(state.f0)+' Hz';updateControls();updateModel(state);drawCharts();}else{beginEdit();restoreEdit(state);endEdit();}},post,setBusy:value=>{keyframeBusy=value;applyPoseLock();},maxDuration:meta.max_animation_duration??60,isReady:()=>!motionBusy&&!dirty&&!inFlight&&!rendering,isSilent:meta.audio_locked,keepVowel:()=>$('keepVowel').checked,viewer});
  updateControls();updateHistory();
}
async function setPerturbation(value){
  const next=value?{...value}:null;
  const unsupported=next!==null&&!meta.source_perturbation;
  if(!unsupported&&(next!==null||meta.source_perturbation))await post('source/perturbation',{perturbation:next});
  perturbation=next;unsupportedPerturbation=unsupported;retainSettings({perturbation:next});updatePerturbationControls();applyPoseLock();keyframeEditor?.refresh();
}
function updatePerturbationControls(){
  for(const [id,key] of [['jitterRange','jitter'],['shimmerRange','shimmer']]){$(id).value=perturbation?.[key]??(key==='jitter'?10:0);$(key+'Value').textContent=$(id).value;}
  $('perturbationSeed').value=perturbation?.seed??1;
  $('perturbationHint').textContent=perturbation?'自定义强度 0–100 · 同一种子可重复合成；持续发声中改姿态保留随机序列。':'模型默认 · jitter 10，shimmer 0。调整后使用固定随机种子。';
  if(unsupportedPerturbation)$('perturbationHint').textContent='浏览器声源未能载入，当前服务器声源不支持这组抖动设置。可重新打开本模块重试，或恢复模型默认后发声。';
  $('perturbationControls').hidden=!meta.source_perturbation&&!unsupportedPerturbation;
}
function setupPerturbationControls(){
  $('perturbationControls').hidden=!meta.source_perturbation&&!unsupportedPerturbation;
  const save=value=>{perturbationQueue=perturbationQueue.catch(()=>{}).then(async()=>{await setPerturbation(value);await keyframeEditor?.save();}).catch(e=>{toast(e.message);updatePerturbationControls();throw e;});perturbationQueue.catch(()=>{});};
  for(const id of ['jitterRange','shimmerRange','perturbationSeed']){$(id).oninput=()=>{if(id!=='perturbationSeed')$(id==='jitterRange'?'jitterValue':'shimmerValue').textContent=$(id).value;};$(id).onchange=()=>save({jitter:+$('jitterRange').value,shimmer:+$('shimmerRange').value,seed:+$('perturbationSeed').value});}
  $('perturbationReset').onclick=()=>save(null);updatePerturbationControls();
}
function setLive(value){live=value;const b=$('liveButton'),active=live&&!!(meta?.host!=='web'||meta?.continuous_audio);b.classList.toggle('active',active);b.lastElementChild.textContent=active?'停止播放':'持续发声';b.firstElementChild.textContent=active?'■':'▶';}
function updateAudio(s){document.dispatchEvent(new CustomEvent('m10-output-state',{detail:{active:s.active,generation:s.playback_generation}}));if(s.audio_locked){$('audioStatus').textContent='静音建模已锁定，后端不允许音频输出。';return;}setLive(s.active);const db=20*Math.log10(Math.max(s.audio.rms||0,1e-5));$('levelFill').style.width=Math.max(0,Math.min(100,(db+60)/60*100))+'%';$('levelText').textContent=s.active?db.toFixed(0)+' dBFS':'静音';$('levelMeter').dataset.rms=s.audio.rms||0;
  if(vowelBusy&&!rendering&&!s.active)setVowelBusy(false);
  if(s.audio_error){setVowelBusy(false);void post('deactivate').catch(()=>{});$('audioStatus').textContent='音频未启动：'+s.audio_error;return;}
  if(s.active)$('audioStatus').textContent=s.buffering?'正在准备声音，请稍候…':`${s.mode==='live'?'持续发声':s.mode==='test'?'测试音':s.mode==='animation'?'关键帧播放':'元音试听'} · ${s.audio.device}${s.audio.underflows?' · 欠载 '+s.audio.underflows:''}`;
  else if(!rendering)$('audioStatus').textContent=playbackError||'声音将从上方选定的设备播放。';
}
async function preview(glide,duration=1.2){
  if(unsupportedPerturbation){toast('请先重试浏览器声源，或恢复模型默认后发声。');return;}
  if(motionBusy||rendering)return;
  rendering=true;playbackError='';$('audioStatus').textContent='正在准备试听…';
  const buttons=[$('oneSecondButton'),$('liveButton')];buttons.forEach(b=>b.disabled=true);
  try{
    // Finish the last drag before capturing the pose, without dropping its click.
    const deadline=performance.now()+10000;
    while(dirty||inFlight){if(performance.now()>deadline)throw Error('构形尚未更新完成，请检查当前参数后重试。');await new Promise(resolve=>setTimeout(resolve,25));}
    await perturbationQueue;
    if(meta.host==='web')setVowelBusy(true);$('audioStatus').textContent='正在合成…';
    const r=await post('preview',{params:[...params],lip_width:lipWidth,larynx_height:larynxHeight,tongue_shape:[...tongueShape],lip_shape:[...lipShape],duration:glide?2.4:duration,sequence:glide?'a-i-u':null});updateAudio(await r.json());
  }catch(e){setLive(false);setVowelBusy(false);void post('deactivate').catch(()=>{});toast(e.message);playbackError='试听失败：'+e.message;$('audioStatus').textContent=playbackError;}finally{rendering=false;if(meta.host==='web'){if(!live)setVowelBusy(false);applyPoseLock();}else buttons.forEach(b=>b.disabled=!!meta.audio_locked);}
}

const viewport=$('viewport');
export const viewer=new VocalTractViewer(viewport,(name,dx,dy,baseline)=>{
  if(motionBusy)return;if(name==='larynx'){larynxHeight=Math.max(-1,Math.min(1,(baseline.larynx_height??0)+dy));markChanged();return;}if(name==='root')manualRoot=true;const moved=moveControl(meta,baseline,name,dx,dy);params=moved.params;if(moved.tongue_shape)tongueShape=moved.tongue_shape;if(moved.lip_shape)lipShape=moved.lip_shape;markChanged();if(moved.limited)$('editHint').textContent='已达到该参数的移动边界。';
},{onStart:beginEdit,onEnd:endEdit,onSelect:selectInspector});
function updateModel(state){state.section=Math.max(0,Math.min(state.centerline.length-1,+$('sliceRange').value));state.upper=state.airway_sections[state.section].upper;state.lower=state.airway_sections[state.section].lower;if(sideDrag){const h=state.lower[sideDrag.sample],p=state.params[sideDrag.index];if(h!==null){const dh=h-sideDrag.height,dp=p-sideDrag.parameter;if(Math.abs(dh)>.002&&dp/dh>0)sideDrag.gain=Math.max(.05,Math.min(3,dp/dh));sideDrag.height=h;sideDrag.parameter=p;sideDrag.params=[...state.params];}}viewer.update(state);drawLipFront(viewer.state);$('loadIndicator').hidden=true;$('slicePosition').textContent=state.centerline[state.section][2].toFixed(1)+' cm';const area=state.nasal.port_area;$('nasalPortStatus').textContent=area>0?'口鼻气腔连通':'软腭封闭入口';$('nasalPortStatus').classList.toggle('open',area>0);}
let widthDrag=null;
function drawLipFront(state){
  const svg=$('lipFront'),upper=state.meshes.find(m=>m.name==='upper_lip'),lower=state.meshes.find(m=>m.name==='lower_lip');
  const edge=m=>state.lipEdges[m.name].toSorted((a,b)=>a[2]-b[2]);
  const up=edge(upper),lo=edge(lower),cy=(Math.max(...up.map(p=>p[1]))+Math.min(...lo.map(p=>p[1])))/2;
  const pts=[...up,...lo.toReversed()],d='M'+pts.map(p=>p[2]+','+(cy-p[1])).join('L')+'Z';
  svg.replaceChildren();const NS='http://www.w3.org/2000/svg',path=document.createElementNS(NS,'path');path.setAttribute('d',d);path.setAttribute('fill','#fffaf2');path.setAttribute('stroke','#bf8089');path.setAttribute('stroke-width','.17');svg.append(path);
  for(const sign of [-1,1]){const z=sign*Math.max(...pts.map(p=>Math.abs(p[2]))),g=document.createElementNS(NS,'g');g.dataset.widthSide=sign;g.setAttribute('transform',`translate(${z} 0)`);g.setAttribute('role','slider');g.setAttribute('tabindex','0');g.setAttribute('aria-label',sign<0?'左唇角宽度':'右唇角宽度');g.setAttribute('aria-valuemin','55');g.setAttribute('aria-valuemax','160');g.setAttribute('aria-valuenow',Math.round(lipWidth*100));g.innerHTML='<circle r=".3" fill="transparent"/><circle r=".12" fill="#fffdf6" stroke="#237768" stroke-width=".04"/>';g.onkeydown=e=>{if(motionBusy||!['ArrowLeft','ArrowRight'].includes(e.key))return;e.preventDefault();beginEdit();lipWidth=Math.max(.55,Math.min(1.6,lipWidth+(e.key==='ArrowRight'?1:-1)*sign*.02));markChanged();endEdit();};svg.append(g);}
}
$('lipFront').onpointerdown=e=>{const target=e.target.closest('[data-width-side]');if(!target||motionBusy)return;beginEdit();widthDrag={x:e.clientX,width:lipWidth,sign:+target.dataset.widthSide};$('lipFront').setPointerCapture(e.pointerId);e.preventDefault();};
$('lipFront').onpointermove=e=>{if(!widthDrag)return;lipWidth=Math.max(.55,Math.min(1.6,widthDrag.width+(e.clientX-widthDrag.x)*widthDrag.sign*.01));markChanged();};
for(const name of ['pointerup','pointercancel','lostpointercapture'])$('lipFront').addEventListener(name,()=>{if(widthDrag){widthDrag=null;endEdit();}});
function setView(value){viewer.setView(value);$('sagittalButton').classList.toggle('selected',value);$('threeButton').classList.toggle('selected',!value);$('viewDescription').textContent=value?'空白拖动平移，滚轮缩放；拖动圆点调整器官。':'拖动绿色点控制器官；空白处旋转，滚轮缩放。';}
$('sagittalButton').onclick=()=>setView(true);$('threeButton').onclick=()=>setView(false);$('resetView').onclick=()=>viewer.reset();
$('focusView').onclick=()=>{viewer.setOptions({focused:!viewer.focused});$('focusView').textContent=viewer.focused?'查看整头':'聚焦声道';};
for(const [id,key] of [['headToggle','head'],['noseToggle','nose'],['labelsToggle','labels'],['teethToggle','teeth'],['fullModelToggle','fullModel']])$(id).onchange=()=>{viewer.setOptions({[key]:$(id).checked});saveView();};
function setMode(mode){viewer.setOptions({mode});document.querySelectorAll('[data-mode]').forEach(b=>{b.classList.toggle('selected',b.dataset.mode===mode);b.setAttribute('aria-pressed',b.dataset.mode===mode);});$('modeLegend').textContent=mode==='organs'?'粉红：软组织 · 淡线：气腔参考':mode==='airway'?'青绿：空气占据的空间 · 粉红：软腭阀门':'粉红：软组织 · 青绿：气腔叠加';saveView();}
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>setMode(b.dataset.mode));
let sideDrag=null,sectionScale=1;
$('sectionHandles').addEventListener('pointerdown',e=>{const t=e.target.closest('[data-side]');if(!t||motionBusy)return;beginEdit();const svg=$('sectionHandles'),sample=+t.dataset.sample,index=meta.parameters.findIndex(p=>p.name==='TS'+selectedSide),height=snapshot.lower[sample];sideDrag={start:e.clientY,params:[...params],index,sample,origin:height,height,parameter:params[index],gain:[1.2,.36,.3][selectedSide-1],scale:sectionScale*svg.getBoundingClientRect().height/svg.viewBox.baseVal.height};svg.setPointerCapture(e.pointerId);e.preventDefault();});
$('sectionHandles').addEventListener('pointermove',e=>{if(!sideDrag)return;const d=sideDrag,error=d.origin+(d.start-e.clientY)/d.scale-d.height,p=meta.parameters[d.index];params=[...d.params];let base=d.parameter;if(selectedSide<3)base=error<0?Math.min(.3,base):Math.max(-.15,base);params[d.index]=Math.max(p.min,Math.min(p.max,base+error*d.gain));markChanged();});
for(const event of ['pointerup','pointercancel','lostpointercapture'])$('sectionHandles').addEventListener(event,()=>{if(sideDrag){sideDrag=null;endEdit();}});
document.querySelectorAll('[data-side-region]').forEach(b=>b.onclick=()=>{selectedSide=+b.dataset.sideRegion;document.querySelectorAll('[data-side-region]').forEach(t=>t.classList.toggle('selected',t===b));if(snapshot){const target=viewer.handles().find(h=>h.name==='side'+selectedSide).p;let best=3;for(let i=4;i<snapshot.centerline.length-3;i++)if(Math.hypot(snapshot.centerline[i][0]-target[0],snapshot.centerline[i][1]-target[1])<Math.hypot(snapshot.centerline[best][0]-target[0],snapshot.centerline[best][1]-target[1]))best=i;selectSection(best);}});
function drawSideHandles(s,w,h,x,y,scale){
  sectionScale=scale;const svg=$('sectionHandles'),focus=svg.contains(document.activeElement);svg.setAttribute('viewBox',`0 0 ${w} ${h}`);svg.replaceChildren();svg.hidden=viewer.selected!=='tongue';if(svg.hidden)return;
  const valid=s.lower.map((v,i)=>v===null?null:i).filter(i=>i!==null);if(valid.length<2)return;
  for(const fraction of [.15,.85]){const i=valid[Math.round((valid.length-1)*fraction)],g=document.createElementNS('http://www.w3.org/2000/svg','g');g.dataset.side=selectedSide;g.setAttribute('tabindex','0');g.setAttribute('role','slider');g.setAttribute('aria-label','拖动'+['后','中','前'][selectedSide-1]+'侧缘');const p=meta.parameters.find(p=>p.name==='TS'+selectedSide);g.setAttribute('aria-valuemin',p.min);g.setAttribute('aria-valuemax',p.max);g.setAttribute('aria-valuenow',params[meta.parameters.indexOf(p)]);
    g.dataset.sample=i;g.setAttribute('transform',`translate(${x(i)} ${y(s.lower[i])})`);g.innerHTML='<circle r="14" fill="transparent"/><circle r="5" fill="#fffdf6" stroke="#2c7b6a" stroke-width="1.4"/><circle r="2" fill="#2c7b6a"/>';g.addEventListener('keydown',e=>{if(motionBusy||!['ArrowUp','ArrowDown'].includes(e.key))return;e.preventDefault();beginEdit();params=moveControl(meta,params,'side'+selectedSide,0,e.key==='ArrowUp'?.05:-.05).params;markChanged();endEdit();});svg.append(g);
  }if(focus)svg.firstElementChild.focus({preventScroll:true});
}
function chart(id,xmax,ymin,ymax,xlabel){
  const canvas=$(id),w=canvas.clientWidth,h=canvas.clientHeight,dpr=Math.min(devicePixelRatio,2);canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);const pad={l:27,r:7,t:7,b:17};const x=v=>pad.l+v/xmax*(w-pad.l-pad.r),y=v=>h-pad.b-(v-ymin)/(ymax-ymin)*(h-pad.t-pad.b);
  ctx.font=canvasFont(9);ctx.fillStyle=tint('#8a9489');ctx.strokeStyle=tint('#e5e7dd');ctx.lineWidth=.7;
  for(let i=0;i<3;i++){const v=ymin+(ymax-ymin)*i/2;ctx.beginPath();ctx.moveTo(pad.l,y(v));ctx.lineTo(w-pad.r,y(v));ctx.stroke();ctx.fillText(Math.round(v),2,y(v)+3);}
  for(let i=0;i<3;i++){const v=xmax*i/2;ctx.textAlign=i===2?'right':i===0?'left':'center';ctx.fillText(xlabel?xlabel(v):v.toFixed(0),x(v),h-2);}ctx.textAlign='left';
  return {ctx,w,h,x,y,pad};
}
function drawCharts(){if(!snapshot)return;const s=snapshot;
  $('areaChart').setAttribute('aria-valuenow',s.centerline[s.section][2].toFixed(2));$('areaChart').setAttribute('aria-valuetext',s.centerline[s.section][2].toFixed(2)+' 厘米');
  const readouts=s.airway_sections.map((_,i)=>areaReadout(s,i)),selected=readouts[s.section];
  const total=s.tube_lengths.reduce((a,b)=>a+b,0);$('tractLength').textContent='声道长度 '+total.toFixed(1)+' cm';const maxArea=Math.max(8,Math.ceil(Math.max(...s.tube_areas,...readouts.map(a=>a.geometric))/4)*4);let c=chart('areaChart',total,0,maxArea),pos=0;
  c.ctx.beginPath();c.ctx.moveTo(c.x(0),c.y(0));readouts.forEach(a=>c.ctx.lineTo(c.x(a.position),c.y(a.closed?0:a.geometric)));c.ctx.lineTo(c.x(total),c.y(0));c.ctx.closePath();c.ctx.fillStyle=tint('#39877920');c.ctx.fill();c.ctx.strokeStyle=tint('#297c6b');c.ctx.lineWidth=1.35;c.ctx.stroke();
  c.ctx.beginPath();c.ctx.setLineDash([3,3]);s.tube_areas.forEach((a,i)=>{const value=a<=.00011?0:a;if(i)c.ctx.lineTo(c.x(pos),c.y(value));else c.ctx.moveTo(c.x(pos),c.y(value));pos+=s.tube_lengths[i];c.ctx.lineTo(c.x(pos),c.y(value));});c.ctx.strokeStyle=tint('#b48774');c.ctx.stroke();
  c.ctx.beginPath();const sx=c.x(selected.position);c.ctx.moveTo(sx,c.y(0));c.ctx.lineTo(sx,c.y(maxArea));c.ctx.strokeStyle=tint('#8a9088');c.ctx.stroke();c.ctx.setLineDash([]);c.ctx.beginPath();c.ctx.arc(sx,c.y(selected.closed?0:selected.geometric),2.7,0,2*Math.PI);c.ctx.fillStyle=tint('#297c6b');c.ctx.fill();
  $('areaChart').title='实线：当前几何截面面积；虚线：用于合成的声学修正面积';$('areaChart').dataset.geometricArea=String(selected.geometric);$('areaChart').dataset.acousticArea=String(selected.acoustic);
  const ymax=Math.ceil(Math.max(...s.transfer_db)/10)*10,ymin=Math.floor(Math.max(-80,Math.min(...s.transfer_db))/10)*10;c=chart('spectrumChart',6000,ymin,ymax,v=>v>=1000?(v/1000)+'k':v);
  c.ctx.beginPath();s.frequency.forEach((f,i)=>{const yy=c.y(s.transfer_db[i]);if(i)c.ctx.lineTo(c.x(f),yy);else c.ctx.moveTo(c.x(f),yy);});c.ctx.strokeStyle=tint('#b17869');c.ctx.lineWidth=1.3;c.ctx.stroke();
  $('resonances').replaceChildren(...s.resonances.slice(0,3).map((v,i)=>{const el=document.createElement('span');el.textContent=`${s.nasal.port_area>.01?'峰':'F'}${i+1} ${Math.round(v)}`;return el;}));
  const canvas=$('sectionChart'),w=canvas.clientWidth,h=canvas.clientHeight;if(!w||!h)return;const dpr=Math.min(devicePixelRatio,2);canvas.width=w*dpr;canvas.height=h*dpr;const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);
  const {scale,x,y}=sectionView(w,h);canvas.dataset.pixelsPerCm=String(scale);if(!scale)return;
  ctx.strokeStyle=tint('#e5e7dd');ctx.beginPath();ctx.moveTo(w/2,3);ctx.lineTo(w/2,h-3);ctx.stroke();
  let i=0;while(i<96){while(i<96&&s.upper[i]===null)i++;const start=i;while(i<96&&s.upper[i]!==null)i++;const end=i;if(end<=start)continue;ctx.beginPath();for(let j=start;j<end;j++){if(j===start)ctx.moveTo(x(j),y(s.upper[j]));else ctx.lineTo(x(j),y(s.upper[j]));}for(let j=end-1;j>=start;j--)ctx.lineTo(x(j),y(s.lower[j]));ctx.closePath();ctx.fillStyle=tint('#62918227');ctx.fill();ctx.strokeStyle=tint('#518575');ctx.lineWidth=1.2;ctx.stroke();}
  drawSideHandles(s,w,h,x,y,scale);
  const open=s.upper.some((v,i)=>v!==null&&v-s.lower[i]>1e-6);
  $('sectionStatus').textContent=(open?'当前截面 '+selected.geometric.toFixed(3)+' cm²':'当前截面闭合 · 0 cm²')+' · 腭咽口 '+s.nasal.port_area.toFixed(2)+' cm²';
  if(open&&selected.midlineClosed)$('sectionStatus').textContent+=' · 正中接触，侧面仍有通路';
  if(!open&&selected.acoustic<=.00011)$('sectionStatus').textContent+=' · 计算保留 0.0001 cm² 数值下限';
  if(s.oral_min_area<=0.00011&&s.nasal.port_area<=0.0001)$('sectionStatus').textContent+=' · 口鼻闭塞，稳态应近静音';
  ctx.strokeStyle=tint('#92a08e');ctx.fillStyle=tint('#7b897b');ctx.font=canvasFont(8);ctx.beginPath();ctx.moveTo(5,h-9);ctx.lineTo(5+scale,h-9);ctx.stroke();ctx.fillText('1 cm',8+scale,h-6);
}
function selectSection(index){
  if(!snapshot)return;
  index=Math.max(0,Math.min(snapshot.centerline.length-1,index));$('sliceRange').value=index;snapshot.section=index;
  snapshot.upper=snapshot.airway_sections[index].upper;snapshot.lower=snapshot.airway_sections[index].lower;
  Object.assign(viewer.state,{section:index,upper:snapshot.upper,lower:snapshot.lower});viewer.draw2D();$('slicePosition').textContent=snapshot.centerline[index][2].toFixed(1)+' cm';drawCharts();
}
function selectArea(e){if(!snapshot)return;const r=$('areaChart').getBoundingClientRect(),total=snapshot.tube_lengths.reduce((a,b)=>a+b,0);selectSection(nearestSection(snapshot,Math.max(0,Math.min(1,(e.clientX-r.left-27)/(r.width-34)))*total));}
let areaDrag=false;
$('areaChart').onpointerdown=e=>{e.preventDefault();areaDrag=true;$('areaChart').focus();$('areaChart').setPointerCapture(e.pointerId);selectArea(e);};
$('areaChart').onpointermove=e=>{if(areaDrag)selectArea(e);};
for(const event of ['pointerup','pointercancel','lostpointercapture'])$('areaChart').addEventListener(event,()=>areaDrag=false);
$('areaChart').onkeydown=e=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(e.key)){e.preventDefault();selectSection(e.key==='Home'?0:e.key==='End'?snapshot.centerline.length-1:+$('sliceRange').value+(e.key==='ArrowRight'?1:-1));}};
new ResizeObserver(()=>drawCharts()).observe(document.querySelector('.charts'));
async function presentation(){
  const response=await fetch('presentation-metadata.json');if(!response.ok)throw Error('设置预览无法读取');
  meta=await response.json();const defaults=await(await fetch('default-presets.json')).json();
  meta.default_poses=meta.speaker==='W02'?{}:Object.fromEntries(Object.entries(defaults.presets).map(([key,pose])=>[key,{...pose,f0:meta.default_f0}]));for(const name of defaults.excluded)delete meta.presets[name];
  Object.assign(meta,{custom_presets:[],favorites:[],ipa_chart:await ipaChart,saved_frames:[],saved_curve:[],output_devices:[],audio_settings:{device:'',volume:Number($('volumeRange').value)/100,gain_db:0},audio_locked:false});
  params=[...meta.presets.a];viewer.metadata=meta;setupControls();selectInspector('tongue');
  $('loadIndicator').hidden=true;$('audioStatus').textContent='网页版暂未接通';
  document.body.inert=true;document.body.dataset.engineState='presentation';
}
async function boot(){try{
  await preferencesReady;
  if(new URLSearchParams(location.search).get('presentation')==='1'){await presentation();return;}

  meta=await request('meta');
  let desired=new URLSearchParams(location.search).get('speaker')||retainedSettings.speaker;try{desired||=localStorage.getItem('m10-speaker');}catch{}
  if(['JD2','W02'].includes(desired)&&desired!==meta.speaker){await request('speaker/select',{speaker:desired});meta=await request('meta');}
  if(preferenceKey){const model=retainedSettings.models?.[meta.speaker];if(model||meta.speaker!=='JD2'){for(const key of modelSettingKeys)delete retainedSettings[key];Object.assign(retainedSettings,model||{});}}
  $('speakerSelect').value=meta.speaker||'JD2';$('f0Range').value=meta.default_f0??120;
  $('speakerSelect').onchange=async()=>{const speaker=$('speakerSelect').value;if(motionBusy||keyframeEditor?.isBusy()){toast('请先完成当前序列编辑或播放');$('speakerSelect').value=meta.speaker;return;}try{await keyframeEditor?.flush();await post('deactivate');retainSettings({speaker});try{localStorage.setItem('m10-speaker',speaker);}catch{}const url=new URL(location.href);url.searchParams.set('speaker',speaker);location.replace(url.href);}catch(e){toast(e.message);$('speakerSelect').value=meta.speaker;}};
  const defaultResponse=await fetch('default-presets.json');if(!defaultResponse.ok)throw Error('默认构形无法读取');const defaults=await defaultResponse.json();meta.default_poses=meta.speaker==='W02'?{}:Object.fromEntries(Object.entries(defaults.presets).map(([key,pose])=>[key,{...pose,f0:meta.default_f0}]));for(const name of defaults.excluded)delete meta.presets[name];revision=(await request('status')).revision;params=[...meta.presets.a];viewer.metadata=meta;
  const web=meta.host==='web';parent.postMessage({type:'m10-capabilities',execution:meta.execution},location.origin);
  document.body.dataset.execution=meta.execution||'native';
  if(web){
    $('videoStart').textContent='生成并下载';
    $('sectionCount').replaceChildren(...[33,65,129,257].map(n=>{const o=document.createElement('option');o.value=n;o.textContent=n+' 截面'+(n===meta.section_count?' · 默认':'');o.disabled=!meta.section_options.includes(n);return o;}));
    if(!meta.continuous_audio){const note=document.createElement('small');note.className='web-sampling-note';note.append('129、257 截面请使用 ');const link=document.createElement('a');link.href='https://www.phonetictoolbox.com/';link.target='_blank';link.rel='noopener noreferrer';link.textContent='下载桌面版';note.append(link);$('sectionCount').closest('section').querySelector('.sampling-control').after(note);}
  }
  if(preferenceKey&&retainedSettings.audio){const audio={};for(const k of ['volume','gain_db'])if(Number.isFinite(retainedSettings.audio[k]))audio[k]=retainedSettings.audio[k];if(typeof retainedSettings.audio.device==='string')audio.device=retainedSettings.audio.device;meta.audio_settings={...meta.audio_settings,...audio};await post('audio/settings',audio);}
  let count=web?(meta.section_count??33):65;if(preferenceKey&&meta.section_options.includes(retainedSettings.sectionCount))count=retainedSettings.sectionCount;try{const current=preferenceKey?null:localStorage.getItem(web?'m10-web-section-count':'m10-section-count-r19'),saved=Number(current??(!web&&!preferenceKey?localStorage.getItem('m10-section-count'):null));if(meta.section_options.includes(saved)&&(web||current!==null||saved!==129))count=saved;}catch{}$('sectionCount').value=count;$('sliceRange').value=Math.floor(count/2);const savedResponse=await response('keyframes/load');if(!savedResponse.ok)throw Error('关键帧配置无法读取，请重新打开工作区');const sequence=await savedResponse.json();meta.preset_library=await request('library/load');meta.custom_presets=await request('presets/load');meta.favorites=await request('favorites/load');meta.ipa_chart=await ipaChart;meta.saved_frames=sequence.frames;meta.saved_curve=sequence.pitch_curve||[];meta.saved_pressure=sequence.pressure_curve||[];meta.saved_globals=sequence.global_controls||{};meta.saved_features=sequence.features||[];setupControls();await setPerturbation(sequence.perturbation??retainedSettings.perturbation??null);setupPerturbationControls();setupMonitor(post,()=>closed);await viewer.load();let saved=preferenceKey?(retainedSettings.view??{}):{};try{if(!preferenceKey)saved=JSON.parse(localStorage.getItem('vtl-view-v2')||'{}');}catch{}for(const [id,key] of [['headToggle','head'],['noseToggle','nose'],['labelsToggle','labels'],['teethToggle','teeth'],['fullModelToggle','fullModel']]){if(typeof saved[key]==='boolean')$(id).checked=saved[key];viewer.setOptions({[key]:$(id).checked});}setMode(['organs','airway','overlay'].includes(saved.mode)?saved.mode:'organs');dirty=true;await queuePose();selectInspector('tongue');document.body.dataset.engineState='ready';
  let heartbeatPending=false;setInterval(async()=>{if(closed||heartbeatPending)return;heartbeatPending=true;try{const s=await(await post('heartbeat')).json();updateAudio(s);document.body.dataset.engineState='ready';}catch(e){setLive(false);document.body.dataset.engineState='offline';if(!$('loadIndicator').hidden)$('loadIndicator').textContent='服务已离线，请重新打开生理参数合成';}finally{heartbeatPending=false;}},meta.execution==='browser-wasm'?100:meta.host==='web'?500:250);
  window.addEventListener('pagehide',()=>{request('deactivate').catch(()=>{});});
}catch(e){$('loadIndicator').textContent='载入失败：'+e.message;document.body.dataset.engineState='failed';if(e.message.includes('恢复副本')){const recovery=document.createElement('button');recovery.textContent='导出恢复副本';recovery.onclick=()=>request('profile/recovery').catch(error=>toast(error.message));$('loadIndicator').append(recovery);}console.error(e);}}
boot();

document.addEventListener("m10-theme",()=>{drawCharts();viewer.draw2D();viewer.style3D();});
