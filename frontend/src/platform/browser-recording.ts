import type {RecordingPort} from './recording.ts';
import type {ResearchContext} from './research.ts';
import type {Config,Project,Take,Task,Device,Preview,Meter,Spectrum} from '../modules/recording/types.ts';
import {floatWave} from './recording-wav.ts';
import {serialRequests} from './serial-requests.ts';
import {BrowserRecordingProject} from './browser-recording-project.ts';

const clone=<T>(value:T):T=>JSON.parse(JSON.stringify(value));
/** Device meters only; processing still uses the shared scientific worker. */
export function recordingMeter(blocks:Float32Array[][],gain:number,all=false):Meter{
 const selected=all?blocks:blocks.slice(-1),channels=selected[0]?.length??0,length=selected.reduce((n,b)=>n+b[0].length,0),peak:number[]=[],rms:number[]=[],dc:number[]=[],clip:number[]=[],digital:number[]=[],near:number[]=[];
 for(let c=0;c<channels;c++){let max=0,sum=0,square=0,raw=0,dig=0,n=0;for(const b of selected)for(const x of b[c]){max=Math.max(max,Math.abs(x));sum+=x;square+=x*x;if(Math.abs(x)>=1)raw++;if(Math.abs(x*10**(gain/20))>1)dig++;if(Math.abs(x)>=.98)n++;}peak.push(max);rms.push(Math.sqrt(square/Math.max(1,length)));dc.push(sum/Math.max(1,length));clip.push(raw);digital.push(dig);near.push(n);}
 let correlation:number|null=null;if(channels===2&&length>2){let a=0,b=0,cross=0;for(const block of selected)for(let i=0;i<block[0].length;i++){const x=block[0][i]-dc[0],y=block[1][i]-dc[1];a+=x*x;b+=y*y;cross+=x*y;}if(Math.sqrt(a/length)>1e-8&&Math.sqrt(b/length)>1e-8)correlation=Math.max(-1,Math.min(1,cross/Math.sqrt(a*b)));}
 return {peak,rms,dc,raw_clip:clip,digital_clip:digital,near_full_scale:near,correlation};
}
/** Browser device adapter for the same RecordingPage. No native file grants. */
export function browserRecordingPort(research:ResearchContext):RecordingPort {
 const project:Project={schema_version:'ptb-recording/1',id:crypto.randomUUID(),revision:0,label:research.label,tasks:[],takes:[],selected:{},recoveries:[]};
 if(!research.files.recording)throw Error('录音文件接口不可用，请重新登录。');
 const manager=new BrowserRecordingProject(project,research.files.recording);
 let importFiles:Map<string,File>|undefined;
 let disposed=false,stream:MediaStream|undefined,audio:AudioContext|undefined,node:AudioWorkletNode|undefined,input:MediaStreamAudioSourceNode|undefined;
 let recording=false,probing=false,blocks:Float32Array[][]=[],frames=0,config:Config,task:Task|null=null,problem='',ended=false,ack:Promise<void>|undefined,resolveAck:(()=>void)|undefined;
 let captureId='',checkpointBlock=0,checkpointIndex=0,recoveryError='',recoveryQueue:Promise<void>=Promise.resolve();
 let browserInput:Record<string,unknown>={};
 let liveEpoch=0,livePending=false,liveAt=0,liveChannel=-1,liveError='';
 let liveSpectrum:{value:Spectrum;start:number;frames:number}|undefined;
 function resetLive(){liveEpoch++;liveSpectrum=undefined;liveError='';liveAt=0;liveChannel=-1;}
 function updateLive(tail:Float32Array[][],count:number,channel:number){
  if(channel!==liveChannel){resetLive();liveChannel=channel;}
  if(livePending||Date.now()-liveAt<750)return;
  livePending=true;liveAt=Date.now();const ticket=liveEpoch,start=frames-count;
  const file=new File([floatWave(tail,config.sample_rate,config.channels)],'录音预览.wav');
  void research.files.recording!.live(file,count,channel).then(value=>{
   if(!disposed&&ticket===liveEpoch&&(recording||probing)){liveSpectrum={value,start,frames:count};liveError='';}
  },e=>{if(!disposed&&ticket===liveEpoch)liveError=e instanceof Error?e.message:String(e);}).finally(()=>{livePending=false;});
 }
 function checkpoint(final=false){if(!recording||checkpointBlock===blocks.length)return;const part=blocks.slice(checkpointBlock),count=part.reduce((n,b)=>n+b[0].length,0);if(!final&&count<131072)return;checkpointBlock=blocks.length;const index=checkpointIndex++,file=new File([floatWave(part,config.sample_rate,config.channels)],'录音恢复-'+captureId+'-'+index+'.wav',{type:'audio/wav'}),id=captureId;recoveryQueue=recoveryQueue.catch(()=>{}).then(()=>manager.checkpoint(id,index,file,count)).catch(e=>{recoveryError=e instanceof Error?e.message:String(e);});}
 let playback:AudioContext|undefined,player:AudioBufferSourceNode|undefined,playStart=0,playOffset=0,playRate=48000;
 function stopPlay(){try{player?.stop();}catch{/* A source that failed before start has no playback to stop. */}player=undefined;if(playback)void playback.close();playback=undefined;}
 async function release(){input?.disconnect();node?.disconnect();stream?.getTracks().forEach(t=>{t.onended=null;t.stop();});input=undefined;node=undefined;stream=undefined;const current=audio;audio=undefined;if(current&&current.state!=='closed')await current.close().catch(()=>{});}
 async function devices():Promise<Device[]>{
  const list=await navigator.mediaDevices?.enumerateDevices()??[];
  return [{id:'default',index:0,name:'系统默认麦克风 / 扬声器',hostapi:'浏览器',inputs:8,outputs:2,default_rate:48000},...list.filter(d=>d.kind!=='videoinput'&&d.deviceId&&d.deviceId!=='default').map((d,i)=>({id:d.deviceId,index:i+1,name:d.label||(d.kind==='audioinput'?'麦克风':'扬声器'),hostapi:'浏览器',inputs:d.kind==='audioinput'?8:0,outputs:d.kind==='audiooutput'?2:0,default_rate:48000}))];
 }
 function preview(source:Float32Array[][],rate:number,channel:number,start:number,end:number,width=900):Preview{
  const total=source.reduce((n,b)=>n+b[0].length,0);start=Math.max(0,Math.min(total,Math.round(start)));end=Math.max(start,Math.min(total,Math.round(end)));
  const count=source[0]?.length??0,bins=Math.min(Math.max(1,Math.floor(width)),Math.max(1,end-start));
  const wave=Array.from({length:count},()=>Array.from({length:bins},()=>[Infinity,-Infinity]));let offset=0;
  for(const block of source){for(let i=Math.max(0,start-offset);i<Math.min(block[0].length,end-offset);i++){const bin=Math.min(bins-1,Math.floor((offset+i-start)*bins/Math.max(1,end-start)));for(let c=0;c<count;c++){wave[c][bin][0]=Math.min(wave[c][bin][0],block[c][i]);wave[c][bin][1]=Math.max(wave[c][bin][1],block[c][i]);}}offset+=block[0].length;}
  for(const lane of wave)for(const bin of lane)if(!Number.isFinite(bin[0]))bin[0]=bin[1]=0;
  return {wave,window_frames:end-start,frames:total,start_frame:start,sample_rate:rate,spectrum:null};
 }
 function meter(gain:number,all=false):Meter{
  return recordingMeter(blocks,gain,all);
 }
 async function start(body:Record<string,unknown>){
  if(recording||probing)throw Error('请先停止当前采集。');
  if(!window.isSecureContext||!navigator.mediaDevices?.getUserMedia)throw Error('请通过 HTTPS 或本机测试入口录音，并允许麦克风权限。');
  const requested=clone(body.config as Config);
  if(!Number.isInteger(requested.channels)||requested.channels<1||requested.channels>8||requested.roles.length!==requested.channels)throw Error('请设置1–8个通道及对应角色。');
  for(const id of manager.cache.keys())if(!manager.pending.has(id))manager.cache.delete(id);
  const used=[...manager.cache.values()].reduce((n,s)=>n+s.file.size,0),remaining=100_000_000-used;
  if(remaining<requested.sample_rate*requested.channels*4)throw Error('本页录音已达 100 MB。请先保存或导出，再关闭并重新打开录音页。');
  stopPlay();problem='';ended=false;
  try{
   stream=await navigator.mediaDevices.getUserMedia({audio:{deviceId:requested.device==='default'?undefined:{exact:requested.device},channelCount:{exact:requested.channels},echoCancellation:false,noiseSuppression:false,autoGainControl:false},video:false});
   if(disposed)throw Error('录音页已关闭。');
   const settings=stream.getAudioTracks()[0].getSettings();
   browserInput=Object.fromEntries(['channelCount','sampleRate','echoCancellation','noiseSuppression','autoGainControl','latency'].filter(k=>k in settings).map(k=>[k,(settings as Record<string,unknown>)[k]]));
   if(settings.channelCount&&settings.channelCount!==requested.channels)throw Error('浏览器未提供指定通道数，请调整设备设置后重试。');
   audio=new AudioContext({sampleRate:requested.sample_rate});await audio.resume();await audio.audioWorklet.addModule(new URL('recording-capture.js',document.baseURI).href);
   if(disposed)throw Error('录音页已关闭。');
   resetLive();config={...requested,sample_rate:audio.sampleRate,device_name:stream.getAudioTracks()[0].label,hostapi:'浏览器 AudioWorklet'};task=clone(project.tasks.find(t=>t.id===body.task_id)??null);blocks=[];frames=0;captureId=crypto.randomUUID();checkpointBlock=checkpointIndex=0;recoveryError='';recoveryQueue=Promise.resolve();
   if(body.op!=='probe_start')manager.beginCapture(captureId,config,task);
   ack=new Promise(resolve=>resolveAck=resolve);
   node=new AudioWorkletNode(audio,'ptb-recorder',{numberOfInputs:1,numberOfOutputs:1,channelCount:config.channels,channelCountMode:'explicit',outputChannelCount:[1],processorOptions:{limit:Math.floor((remaining-44)/(config.channels*4))}});
   node.port.onmessage=event=>{if(disposed||ended)return;if(event.data.done){ended=true;resolveAck?.();problem=problem||(event.data.limit?'已达到本页录音容量上限，采集自动停止。':'');void release();return;}const block=event.data.block as Float32Array[];if(block.length!==config.channels){problem='输入通道发生变化，已停止采集。';ended=true;resolveAck?.();void release();return;}frames+=block[0].length;blocks.push(block);checkpoint();};
   node.onprocessorerror=()=>{problem='采集发生错误，已保留此前收到的采样。';ended=true;resolveAck?.();void release();};
   stream.getAudioTracks().forEach(t=>{t.onended=()=>{problem='麦克风已断开，已保留此前采样。';ended=true;resolveAck?.();void release();};});
   input=audio.createMediaStreamSource(stream);input.connect(node);node.connect(audio.destination);probing=body.op==='probe_start';recording=!probing;
  }catch(e){await release();throw e;}
 }
 async function upload(){await manager.save();}
 async function stop(){
  resetLive();
  const probe=probing;if(!recording&&!probing)return {project:clone(project),probe,error:''};
  if(!ended){node?.port.postMessage('stop');let timer:ReturnType<typeof setTimeout>|undefined;await Promise.race([ack,new Promise<void>(resolve=>timer=setTimeout(()=>{problem='停止确认超时，保留此前收到的采样。';resolve();},2000))]);clearTimeout(timer);}
  await release();checkpoint(true);await recoveryQueue;recording=probing=false;
  let error=problem;
  if(!probe&&frames){const bytes=floatWave(blocks,config.sample_rate,config.channels),id=captureId,now=new Date().toISOString();
   const name=(task?.filename_stem||'录音').replace(/[\\/:*?"<>|]/g,'_')+'-'+id.slice(0,8)+'.wav';
   const m=meter(0,true),take:Take={id,started_at:now,config:clone(config),task_snapshot:clone(task),versions:[{id:crypto.randomUUID(),kind:'raw',frames,created_at:now,metadata:{sample_rate_origin:'AudioContext; hardware resampling may occur',browser_input:browserInput}}],head:0,undo:[],redo:[],status:problem?'interrupted':'complete',error:problem,quality:{raw_clip:m.raw_clip,near_full_scale:m.near_full_scale,peak:m.peak,rms:m.rms},frames};
   manager.add(take,new File([bytes],name,{type:'audio/wav'}),blocks);
   try{await upload();}catch(e){error='保存未完成，录音仍保留在本页。可重试保存或导出 WAV。'+(e instanceof Error?e.message:'');}
  }
  if(!probe&&!frames){error=error||'未采集到有效音频，未创建空录音条目。';delete manager.captures[captureId];}
  blocks=[];problem='';return {project:clone(project),probe,error,frames};
 }
 function download(file:File){const url=URL.createObjectURL(file),a=document.createElement('a');a.href=url;a.download=file.name;a.click();setTimeout(()=>URL.revokeObjectURL(url),10000);}
 async function request(body:Record<string,unknown>):Promise<unknown>{
  if(disposed)throw Error('录音页已关闭。');const op=body.op;
  if(op==='capabilities'){await manager.load();return {available:true,devices:await devices(),device_error:'',project:clone(project),local_only:false};}
  if(op==='devices')return devices();
  if(op==='open'){if(importFiles){await manager.import(importFiles);importFiles=undefined;}return clone(project);}
  if(op==='backup'){await manager.backup();return {};}
  if(op==='tasks'){project.tasks=clone(body.tasks as Task[]);project.revision++;await manager.save();return clone(project);}
  if(op==='start'||op==='probe_start'){if(body.task_id&&!project.tasks.some(t=>t.id===body.task_id))throw Error('所选录音任务不存在，请重新选择任务。');await start(body);return {config:clone(config)};}
  if(op==='stop')return stop();
  if(op==='save'){await upload();return clone(project);}
  if(op==='select_take'){const take=project.takes.find(t=>t.id===body.id);if(!take)throw Error('录音不存在。');project.selected[take.task_snapshot?.id||'__free__']=take.id;await manager.save();return clone(project);}
  if(op==='job_cancel'){await manager.cancel();return {state:'cancelling'};}
  if(op==='recover'){await manager.recover(String(body.id));return clone(project);}
  if(op==='status'){
   const tail=blocks.slice(-Math.ceil((config?.sample_rate??48000)*2/128));const tailFrames=tail.reduce((n,b)=>n+b[0].length,0);
   const view=preview(tail,config?.sample_rate??48000,Number(body.channel)||0,0,tailFrames);
   let display_error=recoveryError||manager.cleanupWarning;
   if(body.spectrum&&tailFrames&&(recording||probing))updateLive(tail,tailFrames,Number(body.channel)||0);
   else if(!body.spectrum)resetLive();
   const spectrum=liveSpectrum;
   if(liveError)display_error=liveError;
   const confirmed=probing?frames:manager.captures[captureId]?.chunks.reduce((n,c)=>n+c.frames,0)??0;
   return {...view,spectrum:body.spectrum?spectrum?.value??null:null,spectrum_start_frame:spectrum?.start,spectrum_window_frames:spectrum?.frames,start_frame:frames-tailFrames,frames,confirmed_frames:confirmed,queue_blocks:probing?0:Math.ceil(Math.max(0,frames-confirmed)/128),recording,probing,capture_active:recording||probing,error:problem,display_error,meter:meter(Number(body.gain_preview_db)||0),job:await manager.poll(),playback:{playing:!!player,frame:player&&playback?Math.round(playOffset+(playback.currentTime-playStart)*playRate):0,error:''}};
  }
  if(op==='play_stop'){stopPlay();return {};}
  const take=project.takes.find(t=>t.id===body.id);
  if(op==='export_start')return manager.export(body,take);
  if(!take)throw Error('请先选择一段录音。');
  if(['undo','redo','restore','version'].includes(String(op))){stopPlay();await manager.navigate(take,String(op),Number(body.index));return clone(project);}
  if(op==='edit'){stopPlay();await manager.edit(take,String(body.action),Number(body.start),Number(body.end));return clone(project);}
  if(op==='noise')return manager.setNoise(take,Number(body.start),Number(body.end),Number(body.channel));
  if(op==='process_start'){stopPlay();return manager.process(take,String(body.kind),body);}
  const index=body.version===undefined?take.head:Number(body.version),value=await manager.data(take,index),currentFrames=take.versions[index].frames;
  if(op==='preview'){const start=Number(body.start)||0,end=Number(body.end)||currentFrames,channel=Number(body.channel)||0,view=preview(value.blocks,take.config.sample_rate,channel,start,end,Number(body.width)||900);const asset=manager.assets[take.versions[index].id];if(body.spectrum&&asset)view.spectrum=await research.files.recording!.spectrum(asset,start,end,channel);return view;}
  if(op==='download'){download(value.file);return {};}
  if(op==='play'){
   stopPlay();playback=new AudioContext({sampleRate:take.config.sample_rate});
   try{const version=take.versions[index],source=body.variant==='raw'?await manager.data(take,0):value,selected=Math.max(0,Math.min(take.config.channels-1,Number(body.channel)||0)),count=source.blocks.reduce((n,b)=>n+b[0].length,0),buffer=playback.createBuffer(1,count,take.config.sample_rate);let at=0;for(const b of source.blocks){buffer.getChannelData(0).set(b[selected],at);at+=b[selected].length;}
    if(body.variant==='residual'){const parent=take.versions.findIndex(v=>v.id===version.metadata.parent);if(!['denoised','gain'].includes(version.kind)||parent<0||take.versions[parent].frames!==version.frames)throw Error('差分试听仅支持等帧长的降噪/增益版本与其直接来源。');const reference=await manager.data(take,parent);let at=0;for(const b of reference.blocks){for(let i=0;i<b[selected].length;i++)buffer.getChannelData(0)[at+i]=b[selected][i]-buffer.getChannelData(0)[at+i];at+=b[selected].length;}}
    if(body.device&&body.device!=='default'){const setSink=(playback as AudioContext&{setSinkId?:(id:string)=>Promise<void>}).setSinkId;if(!setSink)throw Error('当前浏览器不支持切换输出设备，请选系统默认输出。');await setSink.call(playback,String(body.device));}
    await playback.resume();player=playback.createBufferSource();player.buffer=buffer;const gain=playback.createGain();gain.gain.value=Number(body.volume);player.connect(gain);gain.connect(playback.destination);playStart=playback.currentTime;playOffset=Math.min(Number(body.start)||0,count);playRate=take.config.sample_rate;const active=player;player.onended=()=>{if(player===active){player=undefined;void playback?.close();playback=undefined;}};player.start(0,playOffset/playRate,(Math.min(Number(body.end)||count,count)-playOffset)/playRate);return {};
   }catch(e){stopPlay();throw e;}
  }
  throw Error('此操作的网页接口暂未接通，请使用桌面版。');
 }
 const queued=serialRequests(async<T>(body:unknown)=>await request(body as Record<string,unknown>) as T);
 return {kind:'browser',request:<T>(body:Record<string,unknown>)=>{
  // Stop the device immediately, even while a public HTTPS spectrum request
  // occupies the serial queue. Project finalization still retains its order.
  if(body.op==='stop'&&!ended)node?.port.postMessage('stop');
  return queued<T>(body);
 },choose:async purpose=>{if(purpose!=='open')return {grant:project.id,label:research.label};const files=await new Promise<File[]|null>(resolve=>{const input=document.createElement('input');input.type='file';input.webkitdirectory=true;input.multiple=true;input.onchange=()=>resolve(input.files?Array.from(input.files):null);input.addEventListener('cancel',()=>resolve(null),{once:true});input.click();});if(!files?.length)return null;importFiles=new Map(files.map(f=>[f.webkitRelativePath||f.name,f]));return {grant:project.id,label:'导入录音工程'};},dispose(){disposed=true;resetLive();stopPlay();void release();resolveAck?.();blocks=[];manager.close();}};
}
