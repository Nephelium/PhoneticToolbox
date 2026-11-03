import {messageProperty,setMessageProperty,getMessageProperty} from '../module-localization.ts';
import {uiError,uiErrorMessage,uiMessage,joinMessages} from '../module-localization.ts';
import {tr,messageText,type MessageDescriptor} from '../../i18n/index.ts';
import {LipInference,type Detection} from './inference.ts';
import {LandmarkStabilizer} from './stabilizer.ts';
import {LiveFrameClock} from './live-frame-clock.ts';
import {lipMetrics,type Point} from './metrics.ts';
import {emptyInputLevel,InputMeter,preferredMicrophone,type InputLevel} from './audio-input.ts';
import {newStats,MAX_LOCAL_RECORDING_BYTES,MAX_LOCAL_RESULT_BYTES,MAX_RECORDING_SECONDS,verifyLocalBudget,type CaptureMode,type CapturePhase,type CaptureStats} from './state.ts';
export interface CapturedFrame {index:number;time_s:number;media_time_s:number;presentation_time_ms:number;capture_time_ms:number|null;detected:boolean;points:Point[]|null;metrics:Detection['metrics'];inference_ms:number;input_resolution:[number,number];video_presentation_number:number;callback_index:number;}
export interface CaptureSettings {camera:string;microphone:string;mode:CaptureMode;filter:boolean;cutoff:number;delegate:'CPU'|'GPU';requestedFps:number;preferMp4?:boolean;}
export interface CaptureState {phase:CapturePhase;error:string;stats:CaptureStats;latest:CapturedFrame|null;dirty:boolean;analysisTruncated:boolean;trackSettings:MediaTrackSettings|null;support:Record<string,unknown>|null;audio:InputLevel;}
const webmOptions=['video/webm;codecs=vp9,opus','video/webm;codecs=vp8,opus','video/webm'];
const mp4Options=['video/mp4;codecs=avc1.42001E,mp4a.40.2','video/mp4'];
export class LipCapture {
  readonly state:CaptureState=messageProperty<CaptureState,'error'>({phase:'idle',error:'',stats:newStats(),latest:null,dirty:false,analysisTruncated:false,trackSettings:null,support:null,audio:emptyInputLevel()},'error');
  frames:CapturedFrame[]=[];
  previewFrame:ImageBitmap|null=null;
  chunks:Blob[]=[];
  chunkTimes:{timecode_ms:number;arrival_ms:number;bytes:number}[]=[];
  private stream:MediaStream|null=null;
  private meter:InputMeter|null=null;
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
  private frameClock=new LiveFrameClock();
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
    if(!['idle','ready','failed'].includes(this.state.phase)||this.state.dirty)throw uiError(uiMessage("modules.lip-extraction.save_or_explicitly_discard_the_current_recording_first"));
    if(!this.video.requestVideoFrameCallback)throw uiError(uiMessage("modules.lip-extraction.this_browser_lacks_per_frame_timing_callbacks_use_a"));
    const recording=settings.mode!=='preview';
    if(recording&&typeof MediaRecorder==='undefined')throw uiError(uiMessage("modules.lip-extraction.media_encoding_is_unsupported_on_this_platform_use_desktop"));
    if(!Number.isFinite(settings.cutoff)||settings.cutoff<1||settings.cutoff>240)throw uiError(uiMessage("modules.lip-extraction.cutoff_frequency_must_be_1_240_hz"));
    if(!Number.isFinite(settings.requestedFps)||settings.requestedFps<1||settings.requestedFps>240)throw uiError(uiMessage("modules.lip-extraction.requested_frame_rate_must_be_1_240_fps"));
    this.settings={...settings};this.generation++;const generation=this.generation;
    this.previewFrame?.close();this.previewFrame=null;
    this.frames=[];this.chunks=[];this.chunkTimes=[];this.resultBytes=0;this.firstMedia=null;this.frameClock=new LiveFrameClock();this.previousPresented=null;this.finalized=false;this.encodingComplete=true;this.inFlight=false;
    this.stopPromise=null;this.resolveStop=null;
    Object.assign(this.state,{phase:'opening',error:'',stats:newStats(),latest:null,dirty:false,analysisTruncated:false,support:null,trackSettings:null,audio:emptyInputLevel()});this.publish();
    let stage=uiMessage("modules.lip-extraction.checking_local_storage");
    try{
      const memory=(performance as any).memory;
      if(recording&&memory&&memory.usedJSHeapSize>memory.jsHeapSizeLimit*.85)throw uiError(uiMessage("modules.lip-extraction.observable_browser_heap_memory_is_nearing_its_limit_save"));
      const storage=await navigator.storage?.estimate().catch(()=>null)??null;
      if(generation!==this.generation)return;
      this.state.support={storage_estimate:storage,heap_limit:memory?.jsHeapSizeLimit??null,storage_note:tr("modules.lip-extraction.this_recording_is_held_in_memory_check_free_space")};
      // Device labels may be hidden until permission is granted. Resolve again
      // after acquiring, before starting the recorder. Explicit choices stay exact.
      const before=recording&&!settings.microphone?await navigator.mediaDevices.enumerateDevices().catch(()=>[]):[];
      if(generation!==this.generation)return;
      const desired=settings.microphone||preferredMicrophone(before)?.deviceId||'';
      const audioConstraints=(id:string)=>({deviceId:id?{exact:id}:undefined,echoCancellation:false,noiseSuppression:false,autoGainControl:false});
      const constraints={video:{deviceId:settings.camera?{exact:settings.camera}:undefined,frameRate:{ideal:settings.requestedFps},width:{ideal:1280},height:{ideal:720}},audio:recording?audioConstraints(desired):false};
      stage=uiMessage("modules.lip-extraction.requesting_camera_and_microphone");const stream=await this.acquire(constraints);
      if(generation!==this.generation){stream.getTracks().forEach(t=>t.stop());return;}
      this.stream=stream;this.state.trackSettings=stream.getVideoTracks()[0]?.getSettings()??null;
      if(recording){
        const after=!settings.microphone?await navigator.mediaDevices.enumerateDevices().catch(()=>[]):[];
        if(generation!==this.generation)return;
        const preferred=preferredMicrophone(after),current=stream.getAudioTracks()[0];
        if(preferred&&preferred.deviceId!==desired&&preferred.deviceId!==current?.getSettings().deviceId){
          stage=uiMessage("modules.lip-extraction.switching_to_automatically_selected_microphone");const replacement=await this.acquire({video:false,audio:audioConstraints(preferred.deviceId)});
          if(generation!==this.generation){replacement.getTracks().forEach(t=>t.stop());return;}
          const track=replacement.getAudioTracks()[0];
          if(!track){replacement.getTracks().forEach(t=>t.stop());throw uiError(uiMessage("modules.lip-extraction.selected_microphone_returned_no_audio_track_select_an_input"));}
          stream.getAudioTracks().forEach(t=>{stream.removeTrack(t);t.stop();});stream.addTrack(track);
          replacement.getTracks().filter(t=>t!==track).forEach(t=>t.stop());
        }
        const track=stream.getAudioTracks()[0];
        if(!track||track.readyState!=='live'||!track.enabled)throw uiError(uiMessage("modules.lip-extraction.no_microphone_audio_track_is_available_select_audio_input"));
        const info=track.getSettings();Object.assign(this.state.audio,{label:track.label||tr("modules.lip-extraction.unnamed_audio_input"),deviceId:info.deviceId??'',sampleRate:info.sampleRate??null,channelCount:info.channelCount??null,muted:track.muted});
        const meter=new InputMeter();this.meter=meter;
        try{await meter.start(track,this.state.audio,()=>this.publish());}catch{meter.close();if(generation===this.generation){this.meter=null;setMessageProperty(this.state.audio,"monitorError",uiMessage("modules.lip-extraction.input_level_monitoring_is_unavailable_stop_and_listen_to"));}}
        if(generation!==this.generation)return;
      }
      for(const track of stream.getTracks())track.addEventListener('ended',()=>{if(generation===this.generation&&['opening','previewing','recording'].includes(this.state.phase)){this.encodingComplete=false;setMessageProperty(this.state,"error",uiMessage("modules.lip-extraction.device_disconnected_finalizing_captured_data"));void this.stop().catch(()=>{});}});
      this.video.srcObject=stream;this.video.muted=true;stage=uiMessage("modules.lip-extraction.starting_video_preview");await this.video.play();
      if(generation!==this.generation)return;
      if(['preview','realtime'].includes(settings.mode)){
        stage=uiMessage("modules.lip-extraction.initializing_lip_model");const engine=new LipInference();this.engine=engine;
        const support=await engine.initialize(settings.delegate);
        if(generation!==this.generation){engine.close();return;}
        this.state.support={...this.state.support,...support};
        this.filter=settings.filter?new LandmarkStabilizer(settings.cutoff):null;
      }
      if(generation!==this.generation)return;
      this.state.stats.started=performance.now();
      if(recording){
        // Desktop converts the internal Opus stream to verified MP4/WAV. Native
        // Chromium AAC recording can accumulate overlapping sample timestamps.
        const mimeType=(settings.preferMp4?mp4Options:[...webmOptions,...mp4Options]).find(m=>MediaRecorder.isTypeSupported(m));if(!mimeType)throw uiError(uiMessage("modules.lip-extraction.mp4_encoding_saving_is_unavailable_on_this_platform_use"));
        stage=uiMessage("modules.lip-extraction.initializing_recording_encoder");this.recorder=new MediaRecorder(stream,{mimeType,videoBitsPerSecond:6_000_000});
        this.stopPromise=new Promise(resolve=>this.resolveStop=resolve);
        this.recorder.ondataavailable=event=>{
          if(generation!==this.generation||!event.data.size)return;
          try{verifyLocalBudget(this.state.stats.encodedBytes,event.data.size);this.chunks.push(event.data);this.state.stats.encodedBytes+=event.data.size;
            this.chunkTimes.push({timecode_ms:event.timecode,arrival_ms:performance.now(),bytes:event.data.size});this.state.dirty=true;
          }catch(error){this.encodingComplete=false;setMessageProperty(this.state,"error",joinMessages(uiErrorMessage(error),uiMessage("modules.lip-extraction.the_final_chunk_exceeded_the_budget_recording_may_be")));void this.stop().catch(()=>{});}this.publish();
        };
        this.recorder.onerror=()=>{if(generation!==this.generation)return;this.encodingComplete=false;setMessageProperty(this.state,"error",uiMessage("modules.lip-extraction.media_encoding_failed_existing_segments_are_retained_but_completeness"));void this.stop().catch(()=>{});};
        this.recorder.onstop=()=>{if(generation!==this.generation)return;this.finalized=true;this.resolveStop?.();this.resolveStop=null;};
        stage=uiMessage("modules.lip-extraction.starting_recording_encoder");this.firstMedia=this.video.currentTime;this.recorder.start(1000);this.state.phase='recording';
        this.timer=setTimeout(()=>{setMessageProperty(this.state,"error",uiMessage("modules.lip-extraction.recording_reached_the_30_minute_protection_limit_finalizing"));void this.stop().catch(()=>{});},MAX_RECORDING_SECONDS*1000);
      }else this.state.phase='previewing';
      this.schedule(generation);this.publish();
    }catch(error){if(generation!==this.generation)return;const e=error as Error;this.state.support={...this.state.support,failure:{stage:messageText(stage),name:e.name,message:e.message}};const detail=e.name==='NotAllowedError'?uiMessage("modules.lip-extraction.camera_or_microphone_permission_was_not_granted_allow_this"):e.name==='NotFoundError'?uiMessage("modules.lip-extraction.selected_device_not_found_connect_it_and_refresh_the"):e.name==='NotReadableError'?uiMessage("modules.lip-extraction.cannot_read_device_check_whether_another_program_is_using"):e.name==='InvalidStateError'||e.message==='Invalid state'?uiMessage("modules.lip-extraction.the_media_component_rejected_the_capture_state_refresh_devices"):uiErrorMessage(e);setMessageProperty(this.state,"error",joinMessages(joinMessages(stage,uiMessage("modules.lip-extraction.failed")),detail));this.state.phase='failed';this.release();this.publish();throw uiError(getMessageProperty(this.state,'error') as MessageDescriptor);}
  }
  private schedule(generation:number){this.callback=this.video.requestVideoFrameCallback((now,metadata)=>{if(generation!==this.generation||!['previewing','recording'].includes(this.state.phase))return;this.schedule(generation);void this.frame(now,metadata,generation);});}
  private async frame(now:number,meta:VideoFrameCallbackMetadata,generation:number){
    const time=this.frameClock.next(now,meta.mediaTime,meta.presentedFrames);
    if(time===null)return;
    this.firstMedia??=time;
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
      const result=await this.engine.detect(image,time*1000);
      if(generation!==this.generation||!['recording','previewing'].includes(this.state.phase))return;
      const points=result.points&&this.filter?this.filter.filter(result.points,time):result.points;
      const row:CapturedFrame={index:this.state.stats.processed,time_s:time-this.firstMedia,media_time_s:meta.mediaTime,presentation_time_ms:now,
        capture_time_ms:('captureTime' in meta&&typeof meta.captureTime==='number')?meta.captureTime:null,detected:!!points,points,metrics:points?lipMetrics(points):null,inference_ms:result.inference_ms,input_resolution:[result.width,result.height],video_presentation_number:meta.presentedFrames,callback_index:callbackIndex};
      this.state.stats.processed++;this.state.stats.detected+=points?1:0;this.state.stats.inferenceMs+=result.inference_ms;this.state.latest=row;
      this.previewFrame?.close();this.previewFrame=display;display=null;this.state.stats.firstOverlayMs??=performance.now();this.state.stats.lastOverlayMs=performance.now();
      if(this.settings?.mode==='realtime'){
        const size=new TextEncoder().encode(JSON.stringify(row)).byteLength;
        if(this.resultBytes+size>MAX_LOCAL_RESULT_BYTES){this.state.analysisTruncated=true;setMessageProperty(this.state,"error",uiMessage("modules.lip-extraction.candidate_parameters_reached_the_32_mb_budget_candidate_measurement"));}
        else{this.frames.push(row);this.resultBytes+=size;this.state.dirty=true;}
      }else{this.frames.push(row);if(this.frames.length>120)this.frames.shift();}
    }catch(error){if(generation===this.generation&&['recording','previewing'].includes(this.state.phase)){setMessageProperty(this.state,"error",joinMessages(uiMessage("modules.lip-extraction.candidate_inference_failed"),uiErrorMessage(error)));this.engine?.close();this.engine=null;}}
    finally{display?.close();if(generation===this.generation){this.inFlight=false;this.publish();}}
  }
  hidden(){if(this.state.phase==='recording'){this.state.stats.hiddenEvents++;setMessageProperty(this.state,"error",uiMessage("modules.lip-extraction.the_page_is_in_the_background_the_browser_may"));this.publish();}}
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
      try{await Promise.race([this.stopPromise,new Promise<never>((_,reject)=>{timeout=setTimeout(()=>reject(Error(tr("modules.lip-extraction.encoder_did_not_finalize_within_15_seconds_existing_segments"))),15000);})]);}
      catch(error){this.encodingComplete=false;setMessageProperty(this.state,"error",uiErrorMessage(error));this.state.phase='failed';this.release();this.publish();throw error;}
      finally{if(timeout)clearTimeout(timeout);}
    }
    this.state.stats.stopped=performance.now();this.release();
    if(this.settings?.mode==='preview'){this.frames=[];this.state.latest=null;this.previewFrame?.close();this.previewFrame=null;this.state.dirty=false;}
    if(this.settings?.mode!=='preview'&&!this.chunks.length){this.encodingComplete=false;this.state.phase='failed';setMessageProperty(this.state,"error",uiMessage("modules.lip-extraction.recording_produced_no_valid_media_segments_it_may_have"));this.publish();throw uiError(getMessageProperty(this.state,'error') as MessageDescriptor);}
    // ready means finalized IN MEMORY, never saved/downloaded.
    this.state.phase='ready';this.publish();
  }
  blob(){if(!['ready','failed'].includes(this.state.phase))throw uiError(uiMessage("modules.lip-extraction.recording_has_not_finalized"));if(!this.chunks.length)throw uiError(uiMessage("modules.lip-extraction.no_encoded_recording"));return new Blob(this.chunks,{type:this.chunks[0].type});}
  metadata(){return {schema:'m05-capture/1',backend:['preview','realtime'].includes(this.settings?.mode??'')?'mediapipe-web/0.10.14/float16-1/candidate':'capture-only/1',inference_requested:['preview','realtime'].includes(this.settings?.mode??''),settings:this.settings,support:this.state.support,track_settings:this.state.trackSettings,audio_input:{...this.state.audio},
    model_smoothing:['preview','realtime'].includes(this.settings?.mode??'')?'VIDEO numFaces=1 internal smoothing':null,post_filter:['preview','realtime'].includes(this.settings?.mode??'')?(this.settings?.filter??false):null,stats:this.state.stats,chunk_times:this.chunkTimes,
    clock_mapping:{video:this.frameClock.estimated?'requestVideoFrameCallback callback observation (estimate)':'requestVideoFrameCallback.mediaTime',video_clock_estimated:this.frameClock.estimated,original_video_pts:'frames[].media_time_s',video_origin_s:this.firstMedia,candidate_time_base:'recording_start_estimate',origin_observation:'video.currentTime at MediaRecorder.start; asynchronous encoder origin is not exposed',origin_uncertainty_s:null,physical_sync_verified:false,presentation:'performance.now',audio:'encoded container PTS; inspect after recording',audio_video_drift_s:null},
    limits:{media_bytes:MAX_LOCAL_RECORDING_BYTES,result_bytes:MAX_LOCAL_RESULT_BYTES},analysis_truncated:this.state.analysisTruncated,
    display_strategy:'candidate overlay uses exact submitted frame; source callback count is separate',
    overlay_display_fps:this.state.stats.processed>1&&this.state.stats.lastOverlayMs!>this.state.stats.firstOverlayMs!?(this.state.stats.processed-1)*1000/(this.state.stats.lastOverlayMs!-this.state.stats.firstOverlayMs!):null,
    capture_fps:null,encoding_fps:null,source_presentation_fps:this.state.stats.presented>1&&this.state.stats.lastPresentedMs!>this.state.stats.firstPresentedMs!?(this.state.stats.presented-1)*1000/(this.state.stats.lastPresentedMs!-this.state.stats.firstPresentedMs!):null,
    inference_fps:this.state.stats.inferenceMs?this.state.stats.processed*1000/this.state.stats.inferenceMs:null,mirrored_measurements:false,
    warning:this.state.error||null,complete:this.finalized&&this.encodingComplete,recording_saved:false};}
  markSaved(){this.state.dirty=false;this.publish();}
  discard(){if(!['idle','ready','failed'].includes(this.state.phase))throw uiError(uiMessage("modules.lip-extraction.stop_recording_first"));this.generation++;this.previewFrame?.close();this.previewFrame=null;this.frames=[];this.chunks=[];this.chunkTimes=[];Object.assign(this.state,{dirty:false,phase:'idle',latest:null,error:'',stats:newStats()});this.publish();}
  private release(){this.meter?.close();this.meter=null;if(this.callback)this.video.cancelVideoFrameCallback(this.callback);this.callback=0;if(this.timer)clearTimeout(this.timer);this.timer=null;this.resolveStop?.();this.engine?.close();this.engine=null;this.stream?.getTracks().forEach(t=>t.stop());this.stream=null;this.video.srcObject=null;this.video.pause();this.recorder=null;this.stopPromise=null;this.resolveStop=null;this.filter=null;}
  dispose(){this.generation++;if(this.callback)this.video.cancelVideoFrameCallback(this.callback);if(this.timer)clearTimeout(this.timer);if(this.recorder?.state!=='inactive')this.recorder?.stop();this.release();this.previewFrame?.close();this.previewFrame=null;}
}
