import {createApp,h,markRaw,nextTick,reactive,ref} from 'vue';
import ParameterFigure from '../../src/modules/parameter-display/ParameterFigure.vue';
import RecordingWave from '../../src/modules/recording/RecordingWave.vue';
import PdfReader from '../../src/modules/paper-reading/PdfReader.vue';
import PitchCurve from '../../src/modules/pitch-manipulation/PitchCurve.vue';
import HistoryPlot from '../../src/modules/pitch-manipulation/HistoryPlot.vue';
import F0Comparison from '../../src/modules/phonation-synthesis/F0Comparison.vue';
import ParameterEstimationPage from '../../src/modules/parameter-estimation/ParameterEstimationPage.vue';
import EggAnalysisPage from '../../src/modules/egg-analysis/EggAnalysisPage.vue';
import LpcSpectrumPage from '../../src/modules/lpc-spectrum/LpcSpectrumPage.vue';
import PitchManipulationPage from '../../src/modules/pitch-manipulation/PitchManipulationPage.vue';
import Spec2WavPage from '../../src/modules/spectrogram-to-audio/Spec2WavPage.vue';
import RecordingPage from '../../src/modules/recording/RecordingPage.vue';
import {getRecordingPort,installRecordingPort} from '../../src/platform/recording.ts';
import type {Workspace} from '../../src/state/workspace.ts';
import type {ParameterTable} from '../../src/platform/research.ts';
import type {Paper,PapersClient} from '../../src/platform/papers.ts';
import type {Preview} from '../../src/modules/recording/types.ts';
import type {Result} from '../../src/modules/pitch-manipulation/port.ts';

export function mountRedundantWork(){
  const qa=(window as any).__workAudit;
  const shown=ref(true),availableWidth=ref(1100),selection=ref<[number,number]>([0,0]),activeStep=ref('source');
  const times=Array.from({length:12000},(_,i)=>i/400),values=times.map(t=>150+30*Math.sin(t*5));
  const wave=reactive<Workspace>({asset:markRaw({id:'qa',name:'synthetic.wav',duration:30,sampleRate:16000,channels:[new Float32Array(480000)]}) as Workspace['asset'],start:0,end:1,channel:0,parameters:[],dirty:false,error:'',loading:false,zoom:1,offset:0});
  const table=ref(markRaw({columns:['Time_s','F0','intensity'],kinds:['number','number','number'],rows:times.map((t,i)=>[t,values[i],60+Math.sin(t)]),sha256:'qa'} as ParameterTable));
  const group={id:1,title:'QA parameters',parameters:['F0','intensity']};
  const roles=['microphone','egg'] as const;
  const preview=ref(markRaw({wave:[0,1].map(()=>Array.from({length:1200},(_,i)=>[-.6*Math.abs(Math.sin(i)),.6*Math.abs(Math.sin(i))])),start_frame:0,window_frames:480000,frames:480000,sample_rate:16000,spectrum:{rows:Array.from({length:100},()=>Array(64).fill(-30)),pixels:Array.from({length:100},()=>Array(64).fill(120)),frequencies:Array.from({length:64},(_,i)=>i*80),times:Array.from({length:100},(_,i)=>i*.3)}} as Preview));
  const points={axis:times,source:values,target:values.map(v=>v*1.05)};
  const results=markRaw([{id:'qa',name:'QA',times,original_f0:values}] as Result[]);
  const paper={id:'qa',title:'Synthetic PDF',local:true,downloaded:true} as Paper;
  const image='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aWQAAAABJRU5ErkJggg==';
  const port={
    async render(_id:string,language:string,page:number,width:number,signal:AbortSignal){
      qa.renders.push({language,page,width,hidden:!shown.value});
      await new Promise(resolve=>setTimeout(resolve,15));if(signal.aborted)throw new DOMException('Aborted','AbortError');
      return {pages:4,page,image,imageWidth:width,imageHeight:width*1.4,ratio:1.4,ratios:[1.4,1.4,1.4,1.4],text:'QA',lines:[]};
    },
    async annotations(){return {revision:0,items:[]};}
  } as unknown as PapersClient;
  const host=document.createElement('div');host.style.cssText='position:absolute;inset:0;z-index:99999;background:var(--app)';document.body.append(host);
  const pane=(name:string,node:unknown)=>h('div',{'data-audit-pane':name,style:{display:shown.value?'':'none',width:availableWidth.value+'px',height:name==='pdf'?'700px':'360px'}},[node as any]);
  const app=createApp({render:()=>h('div',{},[
    pane('parameters',h(ParameterFigure,{table:table.value,group,wave,waveform:()=>undefined})),
    pane('recording',h(RecordingWave,{preview:preview.value,selection:selection.value,roles:[...roles],spectrum:true,disabled:false,scale:1,allChannels:true})),
    pane('pitch',h(PitchCurve,{times,original:values,modified:values,start:0,end:30,min:50,max:300,references:[]})),
    pane('history',h(HistoryPlot,{results,min:50,max:300})),
    pane('f0',h(F0Comparison,{points,alignment:'normalize',activeStep:activeStep.value})),
    pane('pdf',h(PdfReader,{paper,language:'original',initialPage:1,active:shown.value,port}))
  ])});app.mount(host);
  return {
    async toggle(value:boolean){shown.value=value;await nextTick();},
    async select(i:number){wave.start=i*.02;wave.end=1+i*.02;selection.value=[i*100,(i+1)*100];activeStep.value=i%2?'source':'target';await nextTick();},
    async resize(width:number){availableWidth.value=width;await nextTick();},
    async newData(){table.value=markRaw({...table.value,rows:table.value.rows.map(row=>[row[0],Number(row[1])+20,row[2]])});preview.value=markRaw({...preview.value,wave:preview.value.wave.map(channel=>channel.map(([a,b])=>[a*.5,b*.5]))});await nextTick();},
    dispose(){app.unmount();host.remove();}
  };
}

export function mountPollingAudit(){
  const qa=(window as any).__workAudit,shown=ref(true);let live=false;
  const count=(key:string)=>{qa.polls[key]=(qa.polls[key]??0)+1;};
  const jobs=()=>live?[{id:'qa-running',state:'running',source_id:'qa',results:[],created_at:1,progress:0}]:[];
  const project={schema_version:'ptb-recording/1',id:'qa',revision:0,label:'QA',tasks:[],takes:[],selected:{},recoveries:[]};
  const previous=getRecordingPort();
  installRecordingPort({choose:async()=>null,request:async(body:any)=>{
    if(body.op==='capabilities')return {available:true,devices:[],device_error:'',project,local_only:true} as any;
    if(body.op==='status'){count('recording');return {capture_active:false,playback:{playing:false,frame:0,error:''},job:live?{state:'running',progress:0}:null} as any;}
    return project as any;
  }});
  const tasks={list:async()=>{count('batches');return [];},eggJobs:async()=>{count('egg');return jobs();},lpcJobs:async()=>{count('lpc');return jobs();},reconstructions:async()=>{count('reconstruction');return jobs();},vocoderStatus:async()=>{count('vocoder');return {available:false,ready:false,progress:{stage:'idle'}};}};
  const m08={jobs:async()=>{count('pitch');return jobs();}};
  const context={key:'qa-poll',files:{kind:'preview',list:async()=>[],tasks,m08}} as any;
  const host=document.createElement('div');document.body.append(host);
  const app=createApp({render:()=>h('div',{style:{display:shown.value?'':'none'}},[
    h(ParameterEstimationPage,{context,active:shown.value}),h(EggAnalysisPage,{context,stateKey:'qa-poll-egg',active:shown.value}),h(LpcSpectrumPage,{context,stateKey:'qa-poll-lpc',active:shown.value}),
    h(PitchManipulationPage,{context,stateKey:'qa-poll-pitch',active:shown.value}),h(Spec2WavPage,{context,stateKey:'qa-poll-reconstruction',active:shown.value}),h(RecordingPage,{stateKey:'qa-poll-recording',active:shown.value})
  ])});app.mount(host);
  return {async toggle(value:boolean){shown.value=value;await nextTick();},setLive(value:boolean){live=value;},dispose(){app.unmount();host.remove();if(previous)installRecordingPort(previous);}};
}

export async function mountMonitorAudit(){
  const qa=(window as any).__workAudit;qa.monitorCalls=0;let closed=false;
  const host=document.createElement('div');host.style.width='900px';
  host.innerHTML='<div class="analysis"><button data-analysis="acoustics">Acoustics</button><button data-analysis="monitor">Monitor</button><div id="monitorHome"><div id="monitorPanel" style="width:800px;height:400px"><canvas id="waveCanvas" style="width:800px;height:160px"></canvas><canvas id="spectrogramCanvas" style="width:800px;height:160px"></canvas></div></div></div><dialog id="monitorDialog"><div class="monitor-dialog-body"></div></dialog>'+['monitorCollapse','monitorExpand','monitorClose','monitorStop','stopFrames'].map(id=>`<button id="${id}"></button>`).join('')+['monitorSeconds','monitorAmplitude','monitorWindow','monitorHop'].map((id,i)=>`<input id="${id}" value="${[3,'auto',10,10][i]}">`).join('')+'<span id="waveLabel"></span><span id="monitorStatus"></span>';
  document.body.append(host);
  // Native module loading matches the embedded page; public assets bypass Vite.
  await new Promise<void>((resolve,reject)=>{qa.monitorReady=resolve;const script=document.createElement('script');script.type='module';script.textContent='import {setupMonitor} from "/vocal-tract/monitor.js"; window.__workAudit.setupMonitor=setupMonitor; window.__workAudit.monitorReady();';script.onerror=reject;document.head.append(script);});
  const setupMonitor=qa.setupMonitor;
  setupMonitor(async()=>{qa.monitorCalls++;return {json:async()=>({generation:1,end:1,seconds:3,window_ms:10,resolution_hz:100,hop_seconds:.01,sample_rate:48000,waveform:[],spectrogram:[]})};},()=>closed);
  host.querySelector<HTMLButtonElement>('[data-analysis=monitor]')!.click();
  return {toggle(value:boolean){host.style.display=value?'':'none';},dispose(){closed=true;host.remove();}};
}
