import {LipInference,type Detection} from './inference.ts';
import {LandmarkStabilizer} from './stabilizer.ts';
import {lipMetrics,type Point} from './metrics.ts';
import {newStats,MAX_LOCAL_RECORDING_BYTES,MAX_LOCAL_RESULT_BYTES,MAX_RECORDING_SECONDS,verifyLocalBudget,type CaptureMode,type CapturePhase,type CaptureStats} from './state.ts';
export interface CapturedFrame {index:number;time_s:number;media_time_s:number;presentation_time_ms:number;capture_time_ms:number|null;detected:boolean;points:Point[]|null;metrics:Detection['metrics'];inference_ms:number;input_resolution:[number,number];video_presentation_number:number;callback_index:number;}
export interface CaptureSettings {camera:string;microphone:string;mode:CaptureMode;filter:boolean;cutoff:number;delegate:'CPU'|'GPU';requestedFps:number;}
export interface CaptureState {phase:CapturePhase;error:string;stats:CaptureStats;latest:CapturedFrame|null;dirty:boolean;analysisTruncated:boolean;trackSettings:MediaTrackSettings|null;support:Record<string,unknown>|null;}
const mimeOptions=['video/webm;codecs=vp9,opus','video/webm;codecs=vp8,opus','video/webm'];
export class LipCapture {
  readonly state:CaptureState={phase:'idle',error:'',stats:newStats(),latest:null,dirty:false,analysisTruncated:false,trackSettings:null,support:null};
  frames:CapturedFrame[]=[];
  previewFrame:ImageBitmap|null=null;
  chunks:Blob[]=[];
  chunkTimes:{timecode_ms:number;arrival_ms:number;bytes:number}[]=[];
  private stream:MediaStream|null=null;
  private recorder:MediaRecorder|null=null;
  private engine:LipInference|null=null;
  private filter:LandmarkStabilizer|null=null;
  private callback=0;
  private generation=0;
  private inFlight=false;
  private finalized=false;
  private encodingComplete=true;
  private settings:CaptureSettings|null=null;
  private firstMedia:number|null=null;
  private previousMedia=-Infinity;
  private previousPresented:number|null=null;
  private resultBytes=0;
  private stopPromise:Promise<void>|null=null;
  private resolveStop:(()=>void)|null=null;
  private timer:ReturnType<typeof setTimeout>|null=null;
  private changed:()=>void;
  private video:HTMLVideoElement;
  private acquire:(constraints:MediaStreamConstraints)=>Promise<MediaStream>;
  constructor(video:HTMLVideoElement,changed:()=>void,acquire=(c:MediaStreamConstraints)=>navigator.mediaDevices.getUserMedia(c)){
    this.video=video;this.changed=changed;this.acquire=acquire;
  }
  private publish(){this.changed();}
  async start(settings:CaptureSettings){
    if(!['idle','ready','failed'].includes(this.state.phase)||this.state.dirty)throw Error('请先保存或明确放弃当前录制');
    if(!this.video.requestVideoFrameCallback)throw Error('当前浏览器缺少逐帧时间回调，请使用支持的浏览器或桌面离线分析');
    const recording=settings.mode!=='preview';
    if(recording&&typeof MediaRecorder==='undefined')throw Error('当前平台不支持媒体编码，请使用桌面采集');
    if(!Number.isFinite(settings.cutoff)||settings.cutoff<1||settings.cutoff>240)throw Error('截止频率须为 1–240 Hz');
    if(!Number.isFinite(settings.requestedFps)||settings.requestedFps<1||settings.requestedFps>240)throw Error('请求帧率须为 1–240 fps');
    this.settings={...settings};this.generation++;const generation=this.generation;
    this.previewFrame?.close();this.previewFrame=null;
    this.frames=[];this.chunks=[];this.chunkTimes=[];this.resultBytes=0;this.firstMedia=null;this.previousMedia=-Infinity;this.previousPresented=null;this.finalized=false;this.encodingComplete=true;this.inFlight=false;
    Object.assign(this.state,{phase:'opening',error:'',stats:newStats(),latest:null,dirty:false,analysisTruncated:false,support:null});this.publish();
    let stage='检查本地存储';
    try{
      const memory=(performance as any).memory;
      if(recording&&memory&&memory.usedJSHeapSize>memory.jsHeapSizeLimit*.85)throw Error('浏览器可观测堆内存已接近上限，请先保存并关闭其他任务，或使用桌面磁盘录制。');
      this.state.support={storage_estimate:await navigator.storage?.estimate().catch(()=>null)??null,heap_limit:memory?.jsHeapSizeLimit??null,storage_note:'当前录制驻内存，浏览器存储配额不等于下载磁盘可用空间'};
      const constraints={video:{deviceId:settings.camera?{exact:settings.camera}:undefined,frameRate:{ideal:settings.requestedFps},width:{ideal:1280},height:{ideal:720}},audio:recording?{deviceId:settings.microphone?{exact:settings.microphone}:undefined,echoCancellation:false,noiseSuppression:false,autoGainControl:false}:false};
      stage='请求摄像头和麦克风';const stream=await this.acquire(constraints);
      if(generation!==this.generation){stream.getTracks().forEach(t=>t.stop());return;}
      this.stream=stream;this.state.trackSettings=stream.getVideoTracks()[0]?.getSettings()??null;
      for(const track of stream.getTracks())track.addEventListener('ended',()=>{if(['opening','previewing','recording'].includes(this.state.phase)){this.encodingComplete=false;this.state.error='设备已断开；正在收尾已采集数据。';void this.stop().catch(()=>{});}});
      this.video.srcObject=stream;this.video.muted=true;stage='启动视频预览';await this.video.play();
      if(generation!==this.generation)return;
      if(['preview','realtime'].includes(settings.mode)){
        stage='初始化唇形模型';this.engine=new LipInference();this.state.support={...this.state.support,...await this.engine.initialize(settings.delegate)};
        this.filter=settings.filter?new LandmarkStabilizer(settings.cutoff):null;
      }
      if(generation!==this.generation){this.release();return;}
      this.state.stats.started=performance.now();
      if(recording){
        const mimeType=mimeOptions.find(m=>MediaRecorder.isTypeSupported(m));if(!mimeType)throw Error('当前平台没有通过可用媒体格式检测');
        stage='初始化录制编码器';this.recorder=new MediaRecorder(stream,{mimeType,videoBitsPerSecond:6_000_000});
        this.stopPromise=new Promise(resolve=>this.resolveStop=resolve);
        this.recorder.ondataavailable=event=>{
          if(generation!==this.generation||!event.data.size)return;
          try{verifyLocalBudget(this.state.stats.encodedBytes,event.data.size);this.chunks.push(event.data);this.state.stats.encodedBytes+=event.data.size;
            this.chunkTimes.push({timecode_ms:event.timecode,arrival_ms:performance.now(),bytes:event.data.size});this.state.dirty=true;
          }catch(error){this.encodingComplete=false;this.state.error=String((error as Error).message)+' 最后一块超过预算，录制可能不完整。';void this.stop().catch(()=>{});}this.publish();
        };
        this.recorder.onerror=()=>{this.encodingComplete=false;this.state.error='媒体编码失败，已有片段保留但完整性未确认';void this.stop().catch(()=>{});};
        this.recorder.onstop=()=>{this.finalized=true;this.resolveStop?.();this.resolveStop=null;};
        stage='启动录制编码器';this.recorder.start(1000);this.state.phase='recording';
        this.timer=setTimeout(()=>{this.state.error='录制已达到 30 分钟保护上限，正在收尾。';void this.stop().catch(()=>{});},MAX_RECORDING_SECONDS*1000);
      }else this.state.phase='previewing';
      this.schedule(generation);this.publish();
    }catch(error){if(generation!==this.generation)return;const e=error as Error;this.state.support={...this.state.support,failure:{stage,name:e.name,message:e.message}};const detail=e.name==='NotAllowedError'?'未获得相机或麦克风权限，请允许本次采集后重试。':e.name==='NotFoundError'?'未找到所选设备，请连接设备并刷新列表。':e.name==='NotReadableError'?'无法读取设备，请检查它是否被其他程序占用。':e.name==='InvalidStateError'||e.message==='Invalid state'?'媒体组件拒绝了当前采集状态，请刷新设备后重试；录制尚未开始。':e.message;this.state.error=stage+'失败：'+detail;this.state.phase='failed';this.release();this.publish();throw Error(this.state.error);}
  }
  private schedule(generation:number){this.callback=this.video.requestVideoFrameCallback((now,metadata)=>{if(generation!==this.generation||!['previewing','recording'].includes(this.state.phase))return;this.schedule(generation);void this.frame(now,metadata,generation);});}
  private async frame(now:number,meta:VideoFrameCallbackMetadata,generation:number){
    if(meta.mediaTime<=this.previousMedia)return;
    this.previousMedia=meta.mediaTime;this.firstMedia??=meta.mediaTime;
    this.state.stats.presented++;const callbackIndex=this.state.stats.presented;this.state.stats.firstPresentedMs??=now;this.state.stats.lastPresentedMs=now;
    if(this.previousPresented!==null)this.state.stats.observedPresentationGaps+=Math.max(0,meta.presentedFrames-this.previousPresented-1);
    this.previousPresented=meta.presentedFrames;
    if(!this.engine||this.state.analysisTruncated){this.publish();return;}
    if(this.inFlight){this.state.stats.skippedInference++;this.publish();return;}
    this.inFlight=true;
    let display:ImageBitmap|null=null;
    try{
      const image=await createImageBitmap(this.video);
      if(generation!==this.generation||!this.engine){image.close();return;}
      // Overlay is rendered over the exact submitted image, never a newer
      // live video frame. Encoding continues independently from this display.
      try{display=await createImageBitmap(image);}catch(error){image.close();throw error;}
      const result=await this.engine.detect(image,meta.mediaTime*1000);
      if(generation!==this.generation||!['recording','previewing'].includes(this.state.phase))return;
      const points=result.points&&this.filter?this.filter.filter(result.points,meta.mediaTime):result.points;
      const row:CapturedFrame={index:this.state.stats.processed,time_s:meta.mediaTime-this.firstMedia,media_time_s:meta.mediaTime,presentation_time_ms:now,
        capture_time_ms:('captureTime' in meta&&typeof meta.captureTime==='number')?meta.captureTime:null,detected:!!points,points,metrics:points?lipMetrics(points):null,inference_ms:result.inference_ms,input_resolution:[result.width,result.height],video_presentation_number:meta.presentedFrames,callback_index:callbackIndex};
      this.state.stats.processed++;this.state.stats.detected+=points?1:0;this.state.stats.inferenceMs+=result.inference_ms;this.state.latest=row;
      this.previewFrame?.close();this.previewFrame=display;display=null;this.state.stats.firstOverlayMs??=performance.now();this.state.stats.lastOverlayMs=performance.now();
      if(this.settings?.mode==='realtime'){
        const size=new TextEncoder().encode(JSON.stringify(row)).byteLength;
        if(this.resultBytes+size>MAX_LOCAL_RESULT_BYTES){this.state.analysisTruncated=true;this.state.error='候选参数达到 32 MB 预算，已停止候选测量；原始录制继续，可在停止后离线分析。';}
        else{this.frames.push(row);this.resultBytes+=size;this.state.dirty=true;}
      }else{this.frames.push(row);if(this.frames.length>120)this.frames.shift();}
    }catch(error){if(generation===this.generation){this.state.error='候选推理失败：'+(error as Error).message;this.engine?.close();this.engine=null;}}
    finally{display?.close();if(generation===this.generation){this.inFlight=false;this.publish();}}
  }
  hidden(){if(this.state.phase==='recording'){this.state.stats.hiddenEvents++;this.state.error='页面进入后台，浏览器可能节流或暂停采集；实际缺帧情况请以录制文件离线解码核对。';this.publish();}}
  async stop(){
    if(['stopping','finalizing'].includes(this.state.phase)){await this.stopPromise;return;}
    if(!['opening','recording','previewing'].includes(this.state.phase))return;
    if(this.state.phase==='opening'){this.generation++;this.state.phase='idle';this.release();this.publish();return;}
    this.state.phase='stopping';this.publish();
    if(this.callback)this.video.cancelVideoFrameCallback(this.callback);
    if(this.timer)clearTimeout(this.timer);this.timer=null;
    if(this.recorder&&this.recorder.state!=='inactive')this.recorder.stop();
    this.state.phase='finalizing';this.publish();
    if(this.stopPromise){let timeout:ReturnType<typeof setTimeout>|undefined;
      try{await Promise.race([this.stopPromise,new Promise<never>((_,reject)=>{timeout=setTimeout(()=>reject(Error('编码器在 15 秒内未完成收尾；已有片段可另存，不能确认完整录制。')),15000);})]);}
      catch(error){this.encodingComplete=false;this.state.error=(error as Error).message;this.state.phase='failed';this.release();this.publish();throw error;}
      finally{if(timeout)clearTimeout(timeout);}
    }
    this.state.stats.stopped=performance.now();this.release();
    // ready means finalized IN MEMORY, never saved/downloaded.
    this.state.phase='ready';this.publish();
  }
  blob(){if(!['ready','failed'].includes(this.state.phase))throw Error('录制尚未收尾');if(!this.chunks.length)throw Error('没有已编码录制');return new Blob(this.chunks,{type:this.chunks[0].type});}
  metadata(){return {schema:'m05-capture/1',backend:['preview','realtime'].includes(this.settings?.mode??'')?'mediapipe-web/0.10.14/float16-1/candidate':'capture-only/1',inference_requested:['preview','realtime'].includes(this.settings?.mode??''),settings:this.settings,support:this.state.support,track_settings:this.state.trackSettings,
    model_smoothing:['preview','realtime'].includes(this.settings?.mode??'')?'VIDEO numFaces=1 internal smoothing':null,post_filter:['preview','realtime'].includes(this.settings?.mode??'')?(this.settings?.filter??false):null,stats:this.state.stats,chunk_times:this.chunkTimes,
    clock_mapping:{video:'requestVideoFrameCallback.mediaTime',presentation:'performance.now',audio:'encoded container PTS; verify offline',audio_video_drift_s:null},
    limits:{media_bytes:MAX_LOCAL_RECORDING_BYTES,result_bytes:MAX_LOCAL_RESULT_BYTES},analysis_truncated:this.state.analysisTruncated,
    display_strategy:'candidate overlay uses exact submitted frame; source callback count is separate',
    overlay_display_fps:this.state.stats.processed>1&&this.state.stats.lastOverlayMs!>this.state.stats.firstOverlayMs!?(this.state.stats.processed-1)*1000/(this.state.stats.lastOverlayMs!-this.state.stats.firstOverlayMs!):null,
    capture_fps:null,encoding_fps:null,source_presentation_fps:this.state.stats.presented>1&&this.state.stats.lastPresentedMs!>this.state.stats.firstPresentedMs!?(this.state.stats.presented-1)*1000/(this.state.stats.lastPresentedMs!-this.state.stats.firstPresentedMs!):null,
    inference_fps:this.state.stats.inferenceMs?this.state.stats.processed*1000/this.state.stats.inferenceMs:null,mirrored_measurements:false,
    warning:this.state.error||null,complete:this.finalized&&this.encodingComplete,recording_saved:false};}
  markSaved(){this.state.dirty=false;this.publish();}
  discard(){if(!['idle','ready','failed'].includes(this.state.phase))throw Error('请先停止录制');this.previewFrame?.close();this.previewFrame=null;this.frames=[];this.chunks=[];this.chunkTimes=[];this.state.dirty=false;this.state.phase='idle';this.publish();}
  private release(){this.resolveStop?.();this.engine?.close();this.engine=null;this.stream?.getTracks().forEach(t=>t.stop());this.stream=null;this.video.srcObject=null;this.video.pause();this.recorder=null;this.stopPromise=null;this.resolveStop=null;this.filter=null;}
  dispose(){this.generation++;if(this.callback)this.video.cancelVideoFrameCallback(this.callback);if(this.timer)clearTimeout(this.timer);if(this.recorder?.state!=='inactive')this.recorder?.stop();this.release();this.previewFrame?.close();this.previewFrame=null;}
}
