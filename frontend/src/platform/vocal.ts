/** VTL web host: same iframe protocol, server science and browser-only devices/files. */
export type VocalPort={request:(op:string,body?:any)=>Promise<any>;close:(forget?:boolean)=>Promise<void>};
type Transport=(method:string,path:string,body?:unknown)=>Promise<any>;
const bytes=(text:string)=>Uint8Array.from(atob(text),c=>c.charCodeAt(0));
function download(raw:BlobPart,name:string,type:string){const url=URL.createObjectURL(new Blob([raw],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),10000);}
async function openDocument(){
 return new Promise<File|null>(resolve=>{const input=document.createElement('input');input.type='file';input.accept='.ptb-vocal.json,.json';input.style.display='none';document.body.append(input);const finish=(file:File|null)=>{input.remove();resolve(file);};input.onchange=()=>finish(input.files?.[0]??null);input.oncancel=()=>finish(null);input.click();});
}

export class VocalPlayback {
 context:AudioContext|null=null;nodes=new Set<AudioBufferSourceNode>();epoch=0;active=false;completed=false;start=0;next=0;duration=0;positionAtStop=0;mode='idle';error:string|null=null;underflows=0;
 segments:{time:number;offset:number;duration:number;rms?:number;peak?:number}[]=[];
 private pull:(body:any)=>Promise<any>;
 constructor(pull:(body:any)=>Promise<any>){this.pull=pull;}
 position(){if(!this.active||!this.context)return this.positionAtStop;const now=this.context.currentTime;while(this.segments.length>1&&this.segments[1]!.time<=now)this.segments.shift();const part=this.segments[0];return part?part.offset+Math.max(0,Math.min(part.duration,now-part.time)):0;}
 levels(){const now=this.context?.currentTime,part=this.active&&now!==undefined?this.segments.find(s=>s.time<=now&&now<s.time+s.duration):undefined;return {rms:part?.rms??0,peak:part?.peak??0};}
 async unlock(){this.context??=new AudioContext({sampleRate:48000});await this.context.resume();if(this.context.state!=='running')throw Error('请允许浏览器播放声音。');}
 async play(mode:string){
  this.stop();await this.unlock();this.active=true;this.completed=false;this.mode=mode;this.error=null;this.underflows=0;this.duration=0;this.segments=[];this.start=this.context!.currentTime+.08;this.next=this.start;const epoch=++this.epoch;
  const pump=async()=>{
   try{
    const prepared:any[]=[];let preparedSeconds=0,finished=false;
    // Prepared clips are bounded to five seconds. Start with enough PCM in
    // hand to cover public HTTPS round trips before advancing the audio clock.
    while(this.active&&epoch===this.epoch&&preparedSeconds<1){
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
    while(this.active&&epoch===this.epoch){
     if(!finished){
      if(this.next-this.context!.currentTime>.85){await new Promise(r=>setTimeout(r,25));continue;}
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
 stop(){this.positionAtStop=this.position();++this.epoch;for(const node of this.nodes){node.onended=null;try{node.stop();}catch{}node.disconnect();}this.nodes.clear();this.active=false;this.mode='idle';}
 async close(){this.stop();await this.context?.close();this.context=null;}
}

export function webVocalPort(transport:Transport,project:string,storageKey:string):VocalPort {
 let session:Promise<{id:string}>|null=null,sessionId='',closed=false,playGeneration=0,streamGeneration=0;const key='ptb.vocal.'+storageKey;
 let profile:any={};try{profile=JSON.parse(sessionStorage.getItem(key)||'{}');}catch{}
 let savedQueue=Promise.resolve();
 const save=()=>{try{sessionStorage.setItem(key,JSON.stringify(profile));}catch{throw Error('浏览器保存空间不足，请导出关键帧文件后再继续。');}};
 const route=(id:string)=>'vocal/sessions/'+id+'?project_id='+encodeURIComponent(project);
 async function remote(op:string,body:any={}){
  if(closed)throw Error('声道工作区已关闭。');
  session??=transport('POST','vocal/sessions?project_id='+encodeURIComponent(project),{});
  const {id}=await session;sessionId=id;return transport('POST',route(id),{op,body});
 }
 const player=new VocalPlayback(body=>remote('audio/pull',{...body,playback_generation:streamGeneration}));
 const playbackBody=()=>({position:player.position(),completed:player.completed,playback_generation:streamGeneration});
 const state=(s:any)=>({...s,active:player.active,mode:player.mode,audio_error:player.error||s.audio_error,audio:{...s.audio,...player.levels(),underflows:player.underflows,device:'浏览器音频输出'}});
 async function request(op:string,body:any={}){
  const starts=['preview','animation/play','audio/test'].includes(op)||op==='live'&&body.active;
  const run=starts?++playGeneration:playGeneration;
  if(['preview','live','animation/play','audio/test'].includes(op)&&!(op==='live'&&!body.active))await player.unlock();
  if(op==='meta'){
   const meta=await remote(op);
   // Replay only this browser tab's authenticated project state, with core validation.
   if(profile.presets)await remote('presets/save',{presets:profile.presets});
   if(profile.favorites)await remote('favorites/save',{ids:profile.favorites});
   if(profile.sequence)await remote('keyframes',profile.sequence);
   if(profile.audio)await remote('audio/settings',profile.audio);
   const devices=[{id:'browser',name:'浏览器默认输出设备'}];
   if('setSinkId' in AudioContext.prototype){
    const listed=await navigator.mediaDevices?.enumerateDevices().catch(()=>[])??[];
    for(const d of listed)if(d.kind==='audiooutput'&&d.deviceId&&d.deviceId!=='default')devices.push({id:d.deviceId,name:d.label||'音频输出设备'});
    if('selectAudioOutput' in (navigator.mediaDevices??{}))devices.push({id:'choose',name:'选择其他输出设备…'});
   }
   return {...meta,host:'web',section_count:33,section_options:[33,65],output_devices:devices,audio_settings:{device:'browser',volume:profile.audio?.volume??.8,gain_db:profile.audio?.gain_db??0}};
  }
  if(op==='document/open'){
   const file=await openDocument();if(!file)return {cancelled:true};if(file.size>8_000_000)throw Error('关键帧文件超过 8 MB');
   return {document:await remote('document/validate',JSON.parse((await file.text()).replace(/^\uFEFF/,''))),name:file.name};
  }
  if(op==='document/save'){
   const doc=await remote('document/create',body),raw=JSON.stringify(doc,null,2);if(new TextEncoder().encode(raw).length>8_000_000)throw Error('关键帧文件超过 8 MB');
   download(raw,'声道关键帧.ptb-vocal.json','application/json');return {saved:true,name:'声道关键帧.ptb-vocal.json'};
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
  let result=await remote(op,body);
  if(closed)throw Error('声道工作区已关闭。');
  if(typeof result.playback_generation==='number')streamGeneration=Math.max(streamGeneration,result.playback_generation);
  if(op==='video/finish'){download(bytes(result.base64),'声道动作.mp4','video/mp4');delete result.base64;return result;}
  if(starts){if(run===playGeneration)await player.play(op==='animation/play'?'animation':op==='audio/test'?'test':op==='live'?'live':'preview');else await remote('deactivate');return state(result);}
  if(op==='animation/frame')return {...result,active:player.active,completed:player.completed,audio_error:player.error};
  if(op==='heartbeat'||op==='status')return state(result);
  if(['keyframes','presets/save','favorites/save','audio/settings'].includes(op)){
   const captured=structuredClone(body);savedQueue=savedQueue.then(()=>{
    if(op==='keyframes')profile.sequence=captured;
    if(op==='presets/save')profile.presets=result;
    if(op==='favorites/save')profile.favorites=result;
    if(op==='audio/settings')profile.audio={...profile.audio,...captured,device:'browser'};
    save();
   });await savedQueue;
  }
  return result;
 }
 async function close(forget=false){
  if(forget){try{sessionStorage.removeItem(key);}catch{}}
  if(closed)return;closed=true;++playGeneration;window.removeEventListener('pagehide',onPageHide);
  const release=sessionId?transport('DELETE',route(sessionId)):session?.then(({id})=>transport('DELETE',route(id)));
  await player.close();try{await release;}catch{}
 }
 const onPageHide=()=>{void close();};window.addEventListener('pagehide',onPageHide);
 return {request,close};
}
