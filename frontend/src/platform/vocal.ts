/** VTL web host: unchanged protocol, local Wasm preferred, native server fallback. */
import {BrowserVocal,type LocalVocalExecutor} from './browser-vocal.ts';
import {LocalAssets} from './local-assets.ts';
import {vocalProfileStore,parseVocalProfile,type VocalProfileStore} from './vocal-profile.ts';
export type VocalPort={request:(op:string,body?:any)=>Promise<any>;close:(forget?:boolean)=>Promise<void>};
type Transport=(method:string,path:string,body?:unknown)=>Promise<any>;
const bytes=(text:string)=>Uint8Array.from(atob(text),c=>c.charCodeAt(0));
function download(raw:BlobPart,name:string,type:string){const url=URL.createObjectURL(new Blob([raw],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),10000);}
async function openDocument(pack=false){
 return new Promise<File|null>(resolve=>{const input=document.createElement('input');input.type='file';input.accept=pack?'.ptb-poses.json,.json':'.ptb-vocal.json,.json';input.style.display='none';document.body.append(input);const finish=(file:File|null)=>{input.remove();resolve(file);};input.onchange=()=>finish(input.files?.[0]??null);input.oncancel=()=>finish(null);input.click();});
}

export class VocalPlayback {
 context:AudioContext|null=null;nodes=new Set<AudioBufferSourceNode>();epoch=0;active=false;completed=false;buffering=false;start=0;next=0;duration=0;positionAtStop=0;mode='idle';error:string|null=null;underflows=0;initialLead=1;refillLead=.85;
 segments:{time:number;offset:number;duration:number;rms?:number;peak?:number}[]=[];
 private pull:(body:any)=>Promise<any>;
 constructor(pull:(body:any)=>Promise<any>){this.pull=pull;}
 position(){if(!this.active||!this.context)return this.positionAtStop;const now=this.context.currentTime;while(this.segments.length>1&&this.segments[1]!.time<=now)this.segments.shift();const part=this.segments[0];return part?part.offset+Math.max(0,Math.min(part.duration,now-part.time)):0;}
 levels(){const now=this.context?.currentTime,part=this.active&&now!==undefined?this.segments.find(s=>s.time<=now&&now<s.time+s.duration):undefined;return {rms:part?.rms??0,peak:part?.peak??0};}
 async unlock(){this.context??=new AudioContext({sampleRate:48000});await this.context.resume();if(this.context.state!=='running')throw Error('请允许浏览器播放声音。');}
 async play(mode:string){
  this.stop();const epoch=++this.epoch;
  // A newly assigned server generation must never receive the previous clip's
  // completion/position while resume() is awaiting the audio device.
  this.completed=false;this.positionAtStop=0;this.mode=mode;this.error=null;this.underflows=0;this.duration=0;this.segments=[];this.buffering=true;
  try{await this.unlock();}catch(error){if(epoch===this.epoch){this.buffering=false;this.mode='idle';}throw error;}
  if(epoch!==this.epoch)return;
  this.active=true;this.start=this.context!.currentTime+.08;this.next=this.start;
  const pump=async()=>{
   try{
    const prepared:any[]=[];let preparedSeconds=0,finished=false;
    // Local PCM needs only a short lead. The server fallback retains enough
    // buffering for HTTPS latency; neither path computes the whole live sound.
    while(this.active&&epoch===this.epoch&&preparedSeconds<this.initialLead){
     const packet=await this.pull({position:0});if(epoch!==this.epoch)return;
     prepared.push(packet);preparedSeconds+=packet.samples/packet.sample_rate;
     if(packet.eof){finished=true;break;}
    }
    if(!this.active||epoch!==this.epoch)return;
    this.start=this.context!.currentTime+.08;this.next=this.start;
    const schedule=(packet:any)=>{
     const pcm=new Float32Array(bytes(packet.base64).buffer);if(!pcm.length)return;
     const buffer=this.context!.createBuffer(1,pcm.length,packet.sample_rate);buffer.copyToChannel(pcm,0);const node=this.context!.createBufferSource();node.buffer=buffer;node.connect(this.context!.destination);this.nodes.add(node);node.onended=()=>this.nodes.delete(node);
     if(this.next<this.context!.currentTime){const gap=this.context!.currentTime+.02-this.next;if(this.duration===0)this.start+=gap;else this.underflows++;this.next+=gap;}
     let sum=0,peak=0;for(const value of pcm){sum+=value*value;peak=Math.max(peak,Math.abs(value));}
     this.segments.push({time:this.next,offset:this.duration,duration:buffer.duration,rms:Math.sqrt(sum/pcm.length),peak});node.start(this.next);this.next+=buffer.duration;this.duration+=buffer.duration;
    };
    for(const packet of prepared)schedule(packet);
    if(!this.duration)throw Error('未收到可播放的音频，请重新测试发声。');
    this.buffering=false;
    while(this.active&&epoch===this.epoch){
     if(!finished){
      if(this.next-this.context!.currentTime>this.refillLead){await new Promise(r=>setTimeout(r,10));continue;}
      const packet=await this.pull({position:this.position()});if(epoch!==this.epoch)return;
      schedule(packet);finished=packet.eof;
     }
     if(finished){
      while(this.active&&epoch===this.epoch&&this.context!.currentTime<this.next)await new Promise(r=>setTimeout(r,20));
      if(epoch===this.epoch){this.positionAtStop=this.duration;this.active=false;this.completed=true;this.mode='idle';}
      return;
     }
    }
   }catch(e){if(epoch===this.epoch){this.stop();this.error=e instanceof Error?e.message:'音频传输失败';}}
  };void pump();
 }
 stop(){this.positionAtStop=this.position();++this.epoch;for(const node of this.nodes){node.onended=null;try{node.stop();}catch{}node.disconnect();}this.nodes.clear();this.active=false;this.buffering=false;this.mode='idle';}
 async close(){this.stop();await this.context?.close();this.context=null;}
}

export function webVocalPort(transport:Transport,project:string,storageKey:string,createLocal:()=>LocalVocalExecutor|undefined=()=>typeof Worker==='undefined'?undefined:new BrowserVocal(),storage:VocalProfileStore=vocalProfileStore(new LocalAssets(storageKey,project))):VocalPort {
 let session:Promise<{id:string}>|null=null,sessionId='',closed=false,playGeneration=0,streamGeneration=0,acknowledgedCompletion=0;const key='ptb.vocal.'+storageKey;
 let local:LocalVocalExecutor|undefined,selection:Promise<void>|undefined;
 let profile:any={},rawSaved:string|null=null,recoveryRaw:string|undefined,loaded=false,hydrated=false;
 let savedQueue:Promise<unknown>=Promise.resolve();
 const serialized=<T>(work:()=>Promise<T>)=>{const task=savedQueue.catch(()=>{}).then(work);savedQueue=task;return task;};
 const model=(value:any,speaker=value.speaker||'JD2')=>{
  value.models??={JD2:{presets:value.presets??[],favorites:value.favorites??['a','i','u','e','o','E','y','2','l','n','t'],sequence:value.sequence??{frames:[],pitch_curve:[]}}};
  return value.models[speaker]??=( {presets:[],favorites:['a','i','u','e','o','E','y','2','l','n','t'],sequence:{frames:[],pitch_curve:[]}} );
 };
 async function readProfile(){
  if(loaded)return;
  const saved=await storage.read();rawSaved=saved??null;
  let legacy:string|null=null;if(saved===undefined)try{legacy=sessionStorage.getItem(key);}catch{}
  recoveryRaw=saved??legacy??undefined;
  profile=recoveryRaw===undefined?{}:parseVocalProfile(recoveryRaw);loaded=true;
 }
 async function commit(value:any){const raw=JSON.stringify(value);await storage.write(rawSaved,raw);rawSaved=raw;profile=value;}
 async function restoreModel(value:any){
  if(value.library)await remote('library/save',value.library);
  else if(value.presets||value.favorites)await remote('library/save',{version:1,active:'default',sets:[{id:'default',name:'默认构型',presets:value.presets??[],favorites:value.favorites??['a','i','u','e','o','E','y','2','l','n','t']}]});
  if(value.sequence)await remote('keyframes',value.sequence);
 }
 async function hydrate(){
  await readProfile();let meta=await remote('meta');
  if(!hydrated){
   if(profile.speaker&&profile.speaker!==meta.speaker&&meta.speaker_options){await remote('speaker/select',{speaker:profile.speaker});meta=await remote('meta');}
   const saved=profile.models?.[profile.speaker||'JD2']??profile;
   try{await restoreModel(saved);if(profile.audio)await remote('audio/settings',profile.audio);}
   catch(error){throw Error((error instanceof Error?error.message:'保存记录无效')+'。原数据已保留，请导出恢复副本。');}
   if(rawSaved===null&&recoveryRaw!==undefined)await commit(profile);
   hydrated=true;
  }
  return meta;
 }
 const mutations=new Set(['keyframes','presets/save','favorites/save','audio/settings','library/save','library/edit','library/import','speaker/select']);
 async function persist(op:string,body:any){
  const meta=await hydrate();const next=structuredClone(profile),speaker=profile.speaker||'JD2';
  const previousLibrary=await remote('library/load'),previousSequence=await remote('keyframes/load');
  const previousAudio=profile.audio??meta.audio_settings??{volume:.8,gain_db:0,device:'browser'};let applied=false;
  try{
   const result=await remote(op,body);applied=true;
   if(op==='speaker/select'){next.speaker=result.speaker;await restoreModel(model(next,result.speaker));}
   const saved=model(next);saved.library=await remote('library/load');saved.sequence=await remote('keyframes/load');
   const current=saved.library.sets.find((x:any)=>x.id===saved.library.active);saved.presets=current.presets;saved.favorites=current.favorites;
   if(op==='audio/settings')next.audio={...profile.audio,...body,device:'browser'};
   await commit(next);return result;
  }catch(error){
   if(applied){
    try{
     if(op==='speaker/select')await remote('speaker/select',{speaker});
     await remote('library/save',previousLibrary);await remote('keyframes',previousSequence);
     if(op==='audio/settings'&&previousAudio)await remote('audio/settings',previousAudio);
    }catch{hydrated=false;throw Error('保存未完成且计算状态恢复失败，请导出草稿后重新打开工作区。');}
   }
   throw error;
  }
 }
 const route=(id:string)=>'vocal/sessions/'+id+'?project_id='+encodeURIComponent(project);
 async function server(op:string,body:any={}){
  if(closed)throw Error('声道工作区已关闭。');
  session??=transport('POST','vocal/sessions?project_id='+encodeURIComponent(project),{});
  const {id}=await session;sessionId=id;return transport('POST',route(id),{op,body});
 }
 async function remote(op:string,body:any={}){
  if(closed)throw Error('声道工作区已关闭。');
  selection??=(async()=>{
   const candidate=createLocal();if(!candidate)return;local=candidate;
   try{await candidate.request('meta');if(closed){candidate.close();return;}}
   catch(error){candidate.close();local=undefined;if(closed)throw error;console.info('浏览器声道运行组件未就绪，使用服务器计算。');}
  })();await selection;
  if(closed)throw Error('声道工作区已关闭。');
  // Existing WebCodecs packets retain the original server H.264/AAC encoder.
  // Geometry, acoustic synthesis, pictures, audio and monitoring stay local.
  return local&&!op.startsWith('video/')?local.request(op,body):server(op,body);
 }
 const player=new VocalPlayback(body=>remote('audio/pull',{...body,playback_generation:streamGeneration}));
 const playbackBody=()=>({position:player.position(),completed:player.completed,playback_generation:streamGeneration});
 const state=(s:any)=>({...s,active:player.active,buffering:player.buffering,mode:player.mode,audio_error:player.error||s.audio_error,audio:{...s.audio,...player.levels(),underflows:player.underflows,device:'浏览器音频输出'}});
 async function request(op:string,body:any={}){
  if(op==='profile/recovery'){await readProfile().catch(()=>{});if(recoveryRaw===undefined)throw Error('没有可导出的保存记录');download(recoveryRaw,'声道保存记录-恢复副本.json','application/json');return {saved:true};}
  const starts=['preview','animation/play','audio/test'].includes(op)||op==='live'&&body.active;
  const run=starts?++playGeneration:playGeneration;
  if(player.completed&&streamGeneration>acknowledgedCompletion&&['pose','preview','audio/test','audio/settings','animation/prepare','animation/play'].includes(op)){
   const completed=playbackBody();await remote('heartbeat',completed);acknowledgedCompletion=Math.max(acknowledgedCompletion,completed.playback_generation);
  }
  if(['preview','live','animation/play','audio/test'].includes(op)&&!(op==='live'&&!body.active))await player.unlock();
  if(op==='meta'){
   const meta=await serialized(hydrate);
   if(meta.execution==='browser-wasm'){player.initialLead=.24;player.refillLead=.2;}
   const devices=[{id:'browser',name:'浏览器默认输出设备'}];
   if('setSinkId' in AudioContext.prototype){
    const listed=await navigator.mediaDevices?.enumerateDevices().catch(()=>[])??[];
    for(const d of listed)if(d.kind==='audiooutput'&&d.deviceId&&d.deviceId!=='default')devices.push({id:d.deviceId,name:d.label||'音频输出设备'});
    if('selectAudioOutput' in (navigator.mediaDevices??{}))devices.push({id:'choose',name:'选择其他输出设备…'});
   }
   return {...meta,host:'web',output_devices:devices,audio_settings:{device:'browser',volume:profile.audio?.volume??.8,gain_db:profile.audio?.gain_db??0}};
  }
  if(op==='document/open'||op==='preset-pack/open'){
   const pack=op==='preset-pack/open',file=await openDocument(pack);if(!file)return {cancelled:true};if(file.size>8_000_000)throw Error('关键帧文件超过 8 MB');
   return {document:await remote(pack?'preset-pack/validate':'document/validate',JSON.parse((await file.text()).replace(/^\uFEFF/,''))),name:file.name};
  }
  if(op==='document/save'||op==='preset-pack/save'){
   const pack=op==='preset-pack/save',doc=await remote(pack?'preset-pack/create':'document/create',body),raw=JSON.stringify(doc,null,2);if(new TextEncoder().encode(raw).length>8_000_000)throw Error('关键帧文件超过 8 MB');
   const name=pack?'声道构型包.ptb-poses.json':'声道关键帧.ptb-vocal.json';download(raw,name,'application/json');return {saved:true,name};
  }
   if(op==='shutdown'){await close();return {closed:true};}
  if(op==='animation/stop'||op==='deactivate'||op==='live'&&!body.active){++playGeneration;player.stop();return remote(op,body);}
  if(['preview','animation/prepare','audio/settings'].includes(op))player.stop();
  if(op==='audio/settings'&&body.device){
   await player.unlock();let id=body.device;
   if(id==='choose'){
    const media=navigator.mediaDevices as MediaDevices&{selectAudioOutput:()=>Promise<MediaDeviceInfo>};
    id=(await media.selectAudioOutput()).deviceId;
   }
   if(id!=='browser'&&!(player.context as any).setSinkId)throw Error('当前浏览器不支持切换输出设备，请在系统声音设置中选择。');
   if((player.context as any).setSinkId)await (player.context as any).setSinkId(id==='browser'?'':id);
   body={...body,device:'browser'};
  }
  if(['heartbeat','status','animation/frame','audio/monitor'].includes(op))body={...body,...playbackBody()};
  if(op==='speaker/select'){++playGeneration;player.stop();}
  let result=mutations.has(op)?await serialized(()=>persist(op,body)):
   ['keyframes/load','presets/load','favorites/load','library/load'].includes(op)?await serialized(()=>remote(op,body)):await remote(op,body);
  if(closed)throw Error('声道工作区已关闭。');
  if(['heartbeat','status'].includes(op)&&body.completed===true)acknowledgedCompletion=Math.max(acknowledgedCompletion,body.playback_generation);
  if(typeof result.playback_generation==='number')streamGeneration=Math.max(streamGeneration,result.playback_generation);
  if(op==='video/finish'){download(bytes(result.base64),'声道动作.mp4','video/mp4');delete result.base64;return result;}
  if(starts){if(run===playGeneration)await player.play(op==='animation/play'?'animation':op==='audio/test'?'test':op==='live'?'live':'preview');else await remote('deactivate');return state(result);}
  if(op==='animation/frame')return {...result,active:player.active,completed:player.completed,audio_error:player.error};
  if(op==='heartbeat'||op==='status')return state(result);
  return result;
 }
 async function close(forget=false){
  // Closing a workspace never removes explicitly saved configurations.
  // The legacy record is retained for recovery after adoption.
  if(closed)return;await savedQueue.catch(()=>{});closed=true;++playGeneration;window.removeEventListener('pagehide',onPageHide);
  local?.close();
  const release=sessionId?transport('DELETE',route(sessionId)):session?.then(({id})=>transport('DELETE',route(id)));
  await player.close();try{await release;}catch{}
 }
 const onPageHide=()=>{void close();};window.addEventListener('pagehide',onPageHide);
 return {request,close};
}
