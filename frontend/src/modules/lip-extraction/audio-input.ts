/** Device choice is a UI policy, never a change to Windows' default device. */
export function isLoopback(label:string){return /立体声混音|stereo\s*mix|what\s*u\s*hear|loopback|回环|系统声音/i.test(label);}
export function preferredMicrophone(devices:Pick<MediaDeviceInfo,'kind'|'deviceId'|'label'>[]){
 const inputs=devices.filter(d=>d.kind==='audioinput'&&!isLoopback(d.label));
 const microphones=inputs.filter(d=>/麦克风|话筒|microphone|\bmic\b|headset/i.test(d.label));
 return microphones.find(d=>!['default','communications',''].includes(d.deviceId))??microphones[0];
}
export function signalDb(samples:ArrayLike<number>){
 let peak=0,sum=0;for(let i=0;i<samples.length;i++){const v=samples[i];peak=Math.max(peak,Math.abs(v));sum+=v*v;}
 const db=(v:number)=>20*Math.log10(Math.max(1e-6,v));
 return {peakDb:db(peak),rmsDb:db(Math.sqrt(sum/Math.max(1,samples.length)))};
}
export interface InputLevel {label:string;deviceId:string;sampleRate:number|null;channelCount:number|null;muted:boolean;peakDb:number|null;maxPeakDb:number|null;rmsDb:number|null;lowSignal:boolean;monitorError:string;}
export function emptyInputLevel():InputLevel{return {label:'',deviceId:'',sampleRate:null,channelCount:null,muted:false,peakDb:null,maxPeakDb:null,rmsDb:null,lowSignal:false,monitorError:''};}

/** No connection to speakers and no stored PCM; the recorder keeps the original track. */
export class InputMeter {
 private context:AudioContext|null=null;
 private source:MediaStreamAudioSourceNode|null=null;
 private timer:ReturnType<typeof setInterval>|null=null;
 private closed=false;
 async start(track:MediaStreamTrack,level:InputLevel,changed:()=>void){
  const context=new AudioContext();this.context=context;
  const analyser=context.createAnalyser();analyser.fftSize=2048;
  this.source=context.createMediaStreamSource(new MediaStream([track]));this.source.connect(analyser);
  await context.resume();if(this.closed)return;
  const samples=new Float32Array(analyser.fftSize),started=performance.now();
  this.timer=setInterval(()=>{
   if(this.closed)return;
   if(context.state!=='running'){level.monitorError='输入电平监测暂停，请停止后试听录制核对声音。';changed();return;}
   analyser.getFloatTimeDomainData(samples);const value=signalDb(samples);
   level.peakDb=value.peakDb;level.rmsDb=value.rmsDb;level.maxPeakDb=Math.max(level.maxPeakDb??-120,value.peakDb);level.muted=track.muted;
   level.lowSignal=performance.now()-started>=3000&&level.maxPeakDb<-60;changed();
  },100);
 }
 close(){this.closed=true;if(this.timer)clearInterval(this.timer);this.timer=null;this.source?.disconnect();this.source=null;void this.context?.close().catch(()=>{});this.context=null;}
}
