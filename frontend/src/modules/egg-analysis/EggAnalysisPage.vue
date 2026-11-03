<script setup lang="ts">
import {eggJobTitle,isEggPreviewJob,type EggJobLabel} from './job-label.ts';
import {uiError,uiErrorMessage} from '../module-localization.ts';
import {uiMessage,joinMessages} from '../module-localization.ts';
import {localized} from '../module-localization.ts';
import {tr,messageRef,type MessageDescriptor} from '../../i18n/index.ts';
import {useWebSettings,settingsFields} from '../../state/web-settings.ts';
import ResearchUpload from '../../components/ResearchUpload.vue';
import ModuleFrame from '../../components/ModuleFrame.vue';
import ModuleWorkbench from '../../components/ModuleWorkbench.vue';
import {displayConfig} from './display.ts';
import ModuleToolbar from '../../components/ModuleToolbar.vue';
import ModuleStatus from '../../components/ModuleStatus.vue';
import {ref,computed,watch,onMounted,onUnmounted,markRaw} from 'vue';
import type {ResearchContext,ResearchFile,JobView,EggTaskConfig} from '../../platform/research.ts';
import type {AudioAsset} from '../../platform/types.ts';
import type {components} from '../../../../contracts/generated/api';
import EggInverseResult from './EggInverseResult.vue';
import {audioPreview} from '../../platform/research.ts';
import {decodeWav} from '../../platform/decode.ts';
import {workspace,host,type Workspace} from '../../state/workspace.ts';
import {stop} from '../../state/audio.ts';
import AudioTransport from '../../components/AudioTransport.vue';
import AppIcon from '../../components/AppIcon.vue';
import WaveformViewport from '../../components/WaveformViewport.vue';
import TaskPanel from '../../components/TaskPanel.vue';
import ModalDialog from '../../components/ModalDialog.vue';
import EggSignalPlots from './EggSignalPlots.vue';
import EggControls from './EggControls.vue';
import EggParameters from './EggParameters.vue';
import EggBatchPanel from './EggBatchPanel.vue';
import {exportFontSnapshot} from '../../state/fonts.ts';
import {preflightExportFonts} from './fonts.ts';
import {rangeAfterGesture,microAfterGesture} from './navigation.ts';
import {defaults,batchDefaults,signature,taskConfig,validate,validateParameters,errors,eggErrorMessage,inverseAudioFiles,type PreviewRecord} from './state.ts';
const props=defineProps<{context:ResearchContext;stateKey:string;active:boolean}>();const emit=defineEmits<{references:[];close:[]}>();
const {inverse_lp_order:savedOrder,...savedConfig}=host.projects.read<Partial<EggTaskConfig>&{inverse_lp_order?:number|string|null}>('egg.config.'+props.stateKey,{});
const wave=workspace(props.stateKey),config=ref<EggTaskConfig>({...defaults(),...savedConfig,f0_policy:'audio-f0/2',f0_min_hz:savedConfig.f0_min_hz??30,f0_max_hz:savedConfig.f0_max_hz??800,glottal_movement:false});
const autoDbEnabled=ref(host.projects.read<unknown>('egg.auto-db.'+props.stateKey,true)!==false);
let adoptingDb=false;
const savedBatchConfig:EggTaskConfig={...batchDefaults(),...host.projects.read<Partial<EggTaskConfig>>('egg.batch-config.'+props.stateKey,{}),f0_policy:'audio-f0/2'};delete savedBatchConfig.font;
const batchConfig=ref<EggTaskConfig>(savedBatchConfig);
const resultNames=ref<Record<string,string>>({}),resultError=messageRef(''),resultReadError=messageRef(''),resultLoading=ref(false),resultNotice=messageRef(''),savingResult=ref(false);
const showEggWave=ref(false);
const analysisAudio=ref<AudioAsset>();const inverseView=ref<components['schemas']['EggInverseData']>();
const sourceRate=ref(44100),longSource=ref(false);
const files=ref<ResearchFile[]>([]),source=ref<ResearchFile>(),sourceHash=ref(''),directory=ref(''),picker=ref<HTMLInputElement>();
const jobs=ref<JobView[]>([]),labels=ref<Record<string,string>>(host.projects.read('egg.jobs.'+props.stateKey,{}));
const jobLabelMeta=ref<Record<string,EggJobLabel>>(host.projects.read('egg.job-labels.'+props.stateKey,{}));
const jobTitle=(id:string,fallback=id.slice(0,8))=>eggJobTitle(jobLabelMeta.value[id],labels.value[id]??fallback);
const preview=ref<PreviewRecord>(),psd=ref(''),error=messageRef(''),notice=messageRef(''),loading=ref(false),submitting=ref(false),pending=ref('');
const batchError=messageRef(''),fontChecking=ref(false);let submittingBatchIds:string[]=[];
const batchOpen=ref(false),batchBusy=ref(false),help=ref(false),order=ref<number|string|null>(savedOrder??null),resultJob=ref<JobView>(),resultImages=ref<{name:string;url:string}[]>([]),inverseWaves=ref<{kind:'audio'|'if'|'egg';labelKey:string;state:Workspace}[]>([]),resultConfig=ref<EggTaskConfig>();
const tasks=computed(()=>props.context.files.tasks),stale=computed(()=>!!preview.value&&(signature(preview.value.config)!==signature(config.value)||!!source.value&&preview.value.input_sha256!==sourceHash.value));
// Keep the last complete snapshot visible with its own axes while a new one
// computes. It is never eligible for exporting as the current selection.
const displayed=computed(()=>preview.value?.preview);
const displayedConfig=computed(()=>displayConfig(config.value,preview.value?.config,wave.asset?.duration??0,sourceRate.value));
const playable=computed<Workspace>(()=>({...wave,asset:stale.value||!displayed.value?null:analysisAudio.value??null,channel:0}));
const duration=computed(()=>wave.asset?.duration??0);let disposed=false,epoch=0,viewEpoch=0,polling=false,batchEpoch=0;const updateQueued=ref(false);const urls=new Set<string>();const batchIds=ref<string[]>(host.projects.read('egg.batch.'+props.stateKey,[])),savingBatch=ref(false);
const completedBatch=computed(()=>jobs.value.filter(j=>batchIds.value.includes(j.id)&&j.state==='succeeded'));
const visibleJobs=computed(()=>jobs.value.filter(j=>!isEggPreviewJob(jobLabelMeta.value[j.id],labels.value[j.id])&&!(j.result_manifest?.kind==='managed_egg_files'&&j.result_manifest.files.some(f=>f.name==='egg_PSD.png'))));
let session:{id:string;file:ResearchFile}|undefined;
const previewBusy=ref(false);let inversePending:{id:string;epoch:number}|undefined;
function closeSession(){const old=session;session=undefined;if(old)void props.context.files.eggPreview?.close(old.file,old.id).catch(()=>{});}
let pendingSnapshot:{signature:string;hash:string;epoch:number}|undefined;
const jobViews=computed(()=>visibleJobs.value.map(j=>({id:j.id,title:jobTitle(j.id,`EGG · ${new Date(j.created_at*1000).toLocaleString()} · ${j.id.slice(0,8)}`),status:j.state,progress:j.progress,error:j.error_code?(errors[j.error_code]??j.error_code):undefined,canCancel:true,canRetry:true})));
function makeUrl(buffer:ArrayBuffer){const url=URL.createObjectURL(new Blob([buffer],{type:'image/png'}));urls.add(url);return url;}
function release(url:string){if(url){URL.revokeObjectURL(url);urls.delete(url);}}
function clearPreview(){clearTimeout(gestureTimer);stop();analysisAudio.value=undefined;preview.value=undefined;release(psd.value);psd.value='';}
function labelJob(job:JobView,label:string,metadata?:EggJobLabel){labels.value[job.id]=label;host.projects.write('egg.jobs.'+props.stateKey,labels.value);if(metadata){jobLabelMeta.value[job.id]=metadata;host.projects.write('egg.job-labels.'+props.stateKey,jobLabelMeta.value);}jobs.value=[job,...jobs.value.filter(j=>j.id!==job.id)];}
function save(){const okay=host.projects.write('egg.config.'+props.stateKey,{...config.value,inverse_lp_order:order.value});if(okay)wave.dirty=false;else error.value=uiMessage('modules.egg-analysis.failed_to_save_parameter_draft_check_local_storage');return okay;}
defineExpose({save});
watch(config,()=>{wave.dirty=true;stop();if(!adoptingDb)requestPreview();},{deep:true,flush:'sync'});
watch(autoDbEnabled,()=>{wave.dirty=true;if(!host.projects.write('egg.auto-db.'+props.stateKey,autoDbEnabled.value))error.value=uiMessage('modules.egg-analysis.automatic_db_is_active_but_local_settings_could_not');requestPreview();});
watch(order,()=>{wave.dirty=true;});
watch(()=>[wave.start,wave.end],([start,end])=>{if(longSource.value&&end-start>60){wave.end=Math.min(duration.value,start+60);return;}if(longSource.value){const visible=Math.min(60,duration.value/wave.zoom);if(start<wave.offset||end>wave.offset+visible)wave.offset=Math.max(0,Math.min(duration.value-visible,start-Math.max(0,visible-(end-start))/2));}if(start!==config.value.roi_start||end!==config.value.roi_end){config.value.roi_start=start;config.value.roi_end=end;config.value.micro_center=(start+end)/2;stop();}});
watch(()=>config.value.flip_channels,flip=>{if(wave.asset)wave.channel=wave.asset.channels.length===1?0:flip?0:1;});
watch(()=>props.active,active=>{if(!active)stop();else{void poll();if(updateQueued.value)requestPreview();}});
function setRange(start:number,end:number){config.value.roi_start=start;config.value.roi_end=end;config.value.micro_center=(start+end)/2;wave.start=start;wave.end=end;}
async function refresh(){try{const list=await props.context.files.list(directory.value||undefined);if(!disposed)files.value=list.filter(f=>f.kind==='audio');}catch(e){error.value=uiErrorMessage(e);}}
async function choose(){try{const grant=await props.context.files.choose?.('input');if(grant){directory.value=grant.id;await refresh();}}catch(e){error.value=uiErrorMessage(e);}}
function addFiles(event:Event){const input=event.target as HTMLInputElement;props.context.files.add?.([...input.files??[]]);input.value='';void refresh();}
async function load(file:ResearchFile){if(!file)return;const ticket=++epoch;closeSession();inversePending=undefined;closeResult();clearPreview();updateQueued.value=false;pending.value='';source.value=undefined;sourceHash.value='';wave.asset=null;error.value='';loading.value=true;
  try{
    const port=props.context.files.eggPreview;
    const opened=port?await port.open(file):undefined;
    if(disposed||ticket!==epoch){if(opened)void port!.close(file,opened.session_id);return;}
    if(opened)session={id:opened.session_id,file};
    const data=opened?.overview_base64?{asset:await decodeWav(binary(opened.overview_base64),file.name),sha256:opened.sha256}:await audioPreview(props.context.files,file);
    if(disposed||ticket!==epoch)return;
    if(opened&&opened.sha256!==data.sha256)throw uiError(uiMessage('modules.egg-analysis.audio_has_changed_select_it_again'));
    if(data.asset.channels.length!==2)throw Error(errors.egg_stereo_required);
    sourceRate.value=opened?.sample_rate_hz??data.asset.sampleRate;longSource.value=!!opened?.overview_base64;
    if(opened?.sample_count)data.asset.duration=opened.sample_count/sourceRate.value;
    validate({...taskConfig(config.value,'preview'),roi_start:0,roi_end:Math.min(.5,data.asset.duration)},data.asset.duration,sourceRate.value);
    config.value.f0_policy='audio-f0/2';source.value=file;sourceHash.value=data.sha256;wave.asset=markRaw(data.asset);wave.channel=config.value.flip_channels?0:1;wave.zoom=Math.max(1,data.asset.duration/60);wave.offset=0;setRange(0,Math.min(.5,data.asset.duration));notice.value=uiMessage('modules.egg-analysis.stereo_audio_loaded_analysing_automatically');
  }
  catch(e){if(ticket===epoch){closeSession();error.value=uiErrorMessage(e);}}finally{if(ticket===epoch){loading.value=false;if(source.value)requestPreview();}}}
async function readFile(job:JobView,name:string){if(job.result_manifest?.kind!=='managed_egg_files'||!tasks.value?.result)throw uiError(uiMessage('modules.egg-analysis.results_are_not_ready_to_read'));const f=job.result_manifest.files.find(f=>f.name===name);if(!f)throw uiError(uiMessage('modules.egg-analysis.result_files_are_incomplete'));return tasks.value.result(job.id,f.id,f.sha256);}
async function showPreview(job:JobView,restore=false,viewTicket?:number){const ticket=epoch;const expectation=pendingSnapshot;const meta=JSON.parse(new TextDecoder().decode(await readFile(job,'egg.ptb.json'))) as PreviewRecord;if(meta.config.mode!=='preview'||meta.preview?.schema_version!=='egg-preview/1')throw uiError(uiMessage('modules.egg-analysis.this_task_has_no_interactive_plot_snapshot'));
  if(!restore&&(!expectation||expectation.epoch!==ticket||expectation.signature!==signature(config.value)||expectation.hash!==sourceHash.value))return;
  // Normalized playback depends on source bytes and channel order, not ROI,
  // filters or event settings. Reuse the verified in-memory audio for this file
  // instead of transferring/decoding the full WAV on every local gesture.
  const previous=preview.value;
  const reusable=!restore&&previous?.input_sha256===meta.input_sha256&&previous.config.flip_channels===meta.config.flip_channels&&previous.sample_rate_hz===meta.sample_rate_hz&&previous.sample_count===meta.sample_count?analysisAudio.value:undefined;
  const [asset,png]=await Promise.all([reusable?Promise.resolve(reusable):readFile(job,'egg_AUDIO.wav').then(wav=>decodeWav(wav,'归一化分析音频.wav')),readFile(job,'egg_PSD.png')]);if(disposed||ticket!==epoch||(restore&&viewTicket!==viewEpoch))return;
  if(!restore&&(expectation?.signature!==signature(config.value)||expectation?.hash!==sourceHash.value))return;
  if(restore){source.value=undefined;sourceHash.value='';config.value={...meta.config,f0_min_hz:meta.config.f0_min_hz??(meta.config.f0_policy==='audio-f0/2'?30:75),f0_max_hz:meta.config.f0_max_hz??(meta.config.f0_policy==='audio-f0/2'?800:600)};wave.start=meta.config.roi_start??meta.selection.start_s;wave.end=meta.config.roi_end??meta.selection.end_s;notice.value=uiMessage('modules.egg-analysis.task_parameters_and_plots_restored_select_the_source_file');}
  clearPreview();preview.value=markRaw(meta);psd.value=makeUrl(png);analysisAudio.value=markRaw(asset);if(!restore)notice.value='';if(restore){wave.asset=markRaw(asset);wave.channel=0;wave.zoom=1;wave.offset=0;}
}
async function poll(){if(polling||disposed||!tasks.value?.eggJobs)return;polling=true;const ticket=epoch;
  try{const list=await tasks.value.eggJobs();if(disposed)return;jobs.value=list;const current=list.find(j=>j.id===pending.value);
    if(current&&!['queued','running','cancel_requested'].includes(current.state)){
      try{if(current.state==='succeeded')await showPreview(current);else error.value=errors[current.error_code??'']??(current.state==='cancelled'?tr('modules.egg-analysis.analysis_cancelled'):current.error_code??tr('modules.egg-analysis.analysis_did_not_complete'));}
      finally{if(pending.value===current.id)pending.value='';}}
    const inverse=list.find(j=>j.id===inversePending?.id);
    if(inverse&&inversePending?.epoch===epoch&&props.active&&!['queued','running','cancel_requested'].includes(inverse.state)){
      inversePending=undefined;
      if(inverse.state==='succeeded'){notice.value='';await view(inverse);}
      else error.value=errors[inverse.error_code??'']??tr('modules.egg-analysis.inverse_filtering_did_not_complete_check_processing_records');
    }}
  catch(e){if(!disposed&&ticket===epoch)error.value=uiErrorMessage(e);}finally{polling=false;if(updateQueued.value)requestPreview();}}
async function submit(mode:EggTaskConfig['mode'],whole=false){if(!source.value||!wave.asset||!tasks.value?.egg||submitting.value||pending.value)return;error.value='';notice.value='';submitting.value=true;const ticket=epoch;const file=source.value;
  try{let frozen=taskConfig(config.value,mode,order.value===''||order.value==null?null:Number(order.value));if(whole){frozen.roi_start=0;frozen.roi_end=null;}validate(frozen,duration.value,sourceRate.value);if(mode==='inverse'&&((frozen.roi_end??0)-(frozen.roi_start??0)>10||((frozen.roi_end??0)-(frozen.roi_start??0))*sourceRate.value>960000))throw Error(errors.egg_inverse_budget);
    const snapshot={signature:signature(config.value),hash:sourceHash.value,epoch:ticket};fontChecking.value=mode==='single';frozen=await preflightExportFonts(frozen,exportFontSnapshot(),tasks.value.eggFonts);fontChecking.value=false;if(disposed||ticket!==epoch)return;const job=await tasks.value.egg(file,frozen,crypto.randomUUID());labelJob(job,`${file.name} · ${{preview:'交互分析',single:'CSV + 三张图',batch:'批次',inverse:'逆滤波'}[mode??'single']}`,{file:file.name,mode:mode??'single'});
    if(disposed||ticket!==epoch)return;if(mode==='preview'){pendingSnapshot=snapshot;pending.value=job.id;}else if(mode==='inverse'){inversePending={id:job.id,epoch:ticket};notice.value=uiMessage('modules.egg-analysis.inverse_filtering_is_running_results_will_open_automatically');}else notice.value=uiMessage('modules.egg-analysis.export_task_submitted_view_and_save_it_from_processing');await poll();}
  catch(e){if(ticket===epoch)error.value=uiErrorMessage(e);}finally{submitting.value=false;fontChecking.value=false;if(updateQueued.value)requestPreview();}}
function requestPreview(){
  if(disposed||!source.value||!props.context.files.eggPreview)return;
  updateQueued.value=true;clearTimeout(gestureTimer);
  gestureTimer=setTimeout(()=>{
    if(disposed||!props.active||loading.value||previewBusy.value)return;
    void refreshPreview();
  },30);
}
function updateNow(){requestPreview();}
function binary(encoded:string){const value=atob(encoded),bytes=new Uint8Array(value.length);for(let i=0;i<value.length;i++)bytes[i]=value.charCodeAt(i);return bytes.buffer;}
async function refreshPreview(){
  const port=props.context.files.eggPreview,file=source.value;if(!port||!file||!wave.asset||previewBusy.value)return;
  const ticket=epoch,frozen=taskConfig(config.value,'preview'),automatic=autoDbEnabled.value;const expected=signature(frozen);
  const stillCurrent=()=>!disposed&&ticket===epoch&&automatic===autoDbEnabled.value&&expected===signature(taskConfig(config.value,'preview'));
  updateQueued.value=false;previewBusy.value=true;error.value='';
  try{
    validate(frozen,duration.value,sourceRate.value);
    if(!session){const opened=await port.open(file);if(disposed||ticket!==epoch){void port.close(file,opened.session_id).catch(()=>{});return;}if(opened.sha256!==sourceHash.value){void port.close(file,opened.session_id).catch(()=>{});throw uiError(uiMessage('modules.egg-analysis.audio_has_changed_select_it_again'));}session={id:opened.session_id,file};}
    let meta:components['schemas']['EggInteractiveResult'];
    try{meta=await port.update(file,session.id,frozen,{autoDb:automatic,needAudio:!analysisAudio.value});}
    catch(e){if(resultMessage(e)==='egg_preview_expired'){session=undefined;const opened=await port.open(file);if(disposed||ticket!==epoch){void port.close(file,opened.session_id).catch(()=>{});return;}session={id:opened.session_id,file};meta=await port.update(file,session.id,frozen,{autoDb:automatic,needAudio:!analysisAudio.value});}else throw e;}
    if(disposed||ticket!==epoch)return;
    if(meta.input_sha256!==sourceHash.value)throw uiError(uiMessage('modules.egg-analysis.audio_has_changed_select_it_again'));
    // A superseded viewport can still provide the one-time normalized audio.
    if(meta.audio_base64){const asset=await decodeWav(binary(meta.audio_base64),'归一化分析音频.wav');if(disposed||ticket!==epoch)return;asset.originSeconds=meta.audio_start_s??0;if(longSource.value)asset.duration=duration.value;if(meta.config.flip_channels===config.value.flip_channels)analysisAudio.value=markRaw(asset);}
    if(!stillCurrent())return;
    const range=automatic?meta.preview.suggested_db_range:null;
    // The existing PSD supplies its own display range. Repaint using that
    // range before accepting the snapshot, with the same stale-response fence
    // across both calls. No estimator settings or series are changed.
    if(range&&(frozen.spec_vmin!==range[0]||frozen.spec_vmax!==range[1])){
      if(meta.config.spec_vmin!==range[0]||meta.config.spec_vmax!==range[1])meta=await port.update(file,session.id,{...frozen,spec_vmin:range[0],spec_vmax:range[1]});
      if(!stillCurrent())return;
      if(meta.input_sha256!==sourceHash.value)throw uiError(uiMessage('modules.egg-analysis.audio_has_changed_select_it_again'));
      adoptingDb=true;
      try{config.value.spec_vmin=meta.config.spec_vmin;config.value.spec_vmax=meta.config.spec_vmax;}finally{adoptingDb=false;}
    }
    release(psd.value);psd.value=makeUrl(binary(meta.psd_base64));
    preview.value=markRaw({...meta,selection:{start_s:meta.selection.start_s,end_s:meta.selection.end_s}});notice.value='';
  }catch(e){if(!disposed&&ticket===epoch)error.value=resultMessage(e);}
  finally{previewBusy.value=false;if(updateQueued.value&&!disposed)requestPreview();}
}
function overviewSelection(start:number,end:number){
  setRange(start,end);if(end<=start)return;
  requestPreview();
}
function seek(time:number){if(!displayed.value||!source.value)return;config.value.micro_center=time;requestPreview();}
let gestureTimer:ReturnType<typeof setTimeout>|undefined;
function gesture(area:'main'|'micro',kind:'zoom'|'pan',value:number){
  if(!source.value||loading.value)return;
  const before=signature(config.value);
  if(area==='main'){const [a,b]=rangeAfterGesture(wave.start,wave.end,duration.value,kind,value);setRange(a,b);}
  else{const next=microAfterGesture(config.value.micro_center??(wave.start+wave.end)/2,config.value.micro_width_ms??50,duration.value,kind,value);config.value.micro_center=next.center;config.value.micro_width_ms=next.width;}
  if(before===signature(config.value))return;
  requestPreview();
}
async function cancel(id:string){if(id===pending.value){epoch++;pending.value='';pendingSnapshot=undefined;updateQueued.value=false;clearTimeout(gestureTimer);notice.value=uiMessage('modules.egg-analysis.automatic_updates_cancelled_click_update_analysis_to_retry');}try{await tasks.value?.cancelJob?.(id);await poll();}catch(e){error.value=uiErrorMessage(e);}}
async function retry(id:string){try{const job=await tasks.value!.retry(id,crypto.randomUUID());const prior=jobLabelMeta.value[id];labelJob(job,(labels.value[id]??'EGG')+' · 重试',prior?{...prior,retries:(prior.retries??0)+1}:undefined);if(batchIds.value.includes(id)){batchIds.value.push(job.id);host.projects.write('egg.batch.'+props.stateKey,batchIds.value);}await poll();}catch(e){error.value=uiErrorMessage(e);}}
async function batchSubmit(selected:ResearchFile[],settings:EggTaskConfig){
  if(!tasks.value?.egg||batchBusy.value||!selected.length)return;
  const ticket=++batchEpoch;batchBusy.value=true;batchError.value='';submittingBatchIds=[];let count=0;
  const failures:(string|MessageDescriptor)[]=[];let failureDisplay:string|MessageDescriptor='';
  try{
    const config=taskConfig(settings,'batch');validateParameters(config);
    const frozen=await preflightExportFonts(config,exportFontSnapshot(),tasks.value.eggFonts);
    const parameters={...config};delete parameters.font;
    if(disposed||ticket!==batchEpoch)return;
    for(const file of selected){
      if(disposed||ticket!==batchEpoch)break;
      try{
        const job=await tasks.value.egg(file,frozen,crypto.randomUUID());
        if(disposed||ticket!==batchEpoch){
          try{await tasks.value.cancelJob?.(job.id);if(!disposed)await poll();}
          catch(e){if(!disposed)error.value=joinMessages(uiMessage('modules.egg-analysis.failed_to_cancel_late_task'),resultMessage(e));}
          break;
        }
        submittingBatchIds.push(job.id);
        // Replace the previous batch only after a new task has been accepted.
        if(count===0){batchIds.value=[];batchConfig.value={...parameters};host.projects.write('egg.batch-config.'+props.stateKey,parameters);}
        batchIds.value.push(job.id);host.projects.write('egg.batch.'+props.stateKey,batchIds.value);
        labelJob(job,file.name+' · 批次',{file:file.name,mode:'batch'});count++;
      }catch(e){failures.push(joinMessages(file.name,'：',eggErrorMessage(e)));}
    }
    if(failures.length){failureDisplay=joinMessages(uiMessage('modules.egg-analysis.not_submitted'),...failures.flatMap((item,index)=>index?['；',item]:[item]));batchError.value=failureDisplay;}
  }catch(e){failureDisplay=uiErrorMessage(e);batchError.value=failureDisplay;}
  finally{
    batchBusy.value=false;
    if(!disposed&&ticket===batchEpoch&&count>0){batchOpen.value=false;notice.value=uiMessage('modules.egg-analysis.submitted_value_files_value',{v0:(count),v1:failureDisplay});await poll();}
  }
}
async function cancelBatch(){
  const ids=[...(batchBusy.value?submittingBatchIds:batchIds.value)];batchEpoch++;batchOpen.value=false;
  notice.value=batchBusy.value?uiMessage('modules.egg-analysis.further_submission_stopped_late_tasks_will_be_cancelled_after'):uiMessage('modules.egg-analysis.cancellation_requested_for_this_batch_completed_results_are_retained');
  for(const id of ids)await cancel(id);
}
function resultMessage(e:unknown){return eggErrorMessage(e);}
function currentResult(ticket:number,id:string){return !disposed&&ticket===viewEpoch&&resultJob.value?.id===id;}
function autoDb(){
  autoDbEnabled.value=!autoDbEnabled.value;
  if(autoDbEnabled.value){
    // An invalid manual range cannot be sent for PSD calculation. Reuse the
    // last valid display range until the current viewport supplies its own.
    const range=displayed.value?.suggested_db_range??[defaults().spec_vmin!,defaults().spec_vmax!];
    adoptingDb=true;
    try{[config.value.spec_vmin,config.value.spec_vmax]=range;}finally{adoptingDb=false;}
  }
}
function closeResult(){
  showEggWave.value=false;
  viewEpoch++;resultJob.value=undefined;resultConfig.value=undefined;inverseView.value=undefined;
  resultNames.value={};resultError.value='';resultReadError.value='';resultNotice.value='';resultLoading.value=false;savingResult.value=false;
  for(const image of resultImages.value)release(image.url);resultImages.value=[];inverseWaves.value=[];stop();
}
async function view(job:JobView){
  closeResult();const ticket=viewEpoch;error.value='';resultJob.value=job;resultLoading.value=true;let reading=tr('modules.egg-analysis.parameter_snapshot');
  try{
    const metadata=JSON.parse(new TextDecoder().decode(await readFile(job,'egg.ptb.json')));
    if(!currentResult(ticket,job.id))return;
    if(metadata.config.mode==='preview'){
      reading=tr('modules.egg-analysis.analysis_preview');epoch++;pending.value='';updateQueued.value=false;clearTimeout(gestureTimer);await showPreview(job,true,ticket);
      if(currentResult(ticket,job.id))closeResult();return;
    }
    resultNames.value=metadata.export_names??{};resultConfig.value=metadata.config;inverseView.value=metadata.inverse_view;
    if(job.result_manifest?.kind==='managed_egg_files')for(const f of [...job.result_manifest.files.filter(f=>f.name.endsWith('.png')),...inverseAudioFiles(job.result_manifest.files)]){
      reading=resultNames.value[f.name]??f.name;
      const raw=await readFile(job,f.name);if(!currentResult(ticket,job.id))return;
      if(f.name.endsWith('.png'))resultImages.value.push({name:reading,url:makeUrl(raw)});
      else{
        const asset=await decodeWav(raw,f.name);if(!currentResult(ticket,job.id))return;
        inverseWaves.value.push({kind:f.name==='egg_ORIG.wav'?'audio':'if',labelKey:f.name==='egg_ORIG.wav'?'modules.egg-analysis.normalized_analysis_audio':'modules.egg-analysis.if_estimate',state:{asset:markRaw(asset),start:0,end:asset.duration,channel:0,parameters:[],dirty:false,error:'',loading:false,zoom:1,offset:0}});
      }
    }
    if(inverseView.value?.full_egg_values?.length&&inverseView.value.sample_rate_hz){
      const values=inverseView.value.full_egg_values,rate=inverseView.value.sample_rate_hz;
      const asset:AudioAsset=localized(()=>({name:tr('modules.egg-analysis.filtered_egg_from_same_task'),sampleRate:rate,channels:[new Float32Array(values)],duration:values.length/rate,frames:values.length}));
      inverseWaves.value.push({kind:'egg',labelKey:'modules.egg-analysis.filtered_egg',state:{asset:markRaw(asset),start:0,end:asset.duration,channel:0,parameters:[],dirty:false,error:'',loading:false,zoom:1,offset:0}});
    }
  }catch(e){if(currentResult(ticket,job.id))resultReadError.value=uiMessage('modules.egg-analysis.failed_to_read_value_value',{v0:(reading),v1:(resultMessage(e))});}
  finally{if(currentResult(ticket,job.id))resultLoading.value=false;}
}
async function saveBatch(){if(savingBatch.value)return;savingBatch.value=true;error.value='';try{const grant=await props.context.files.choose?.('output');if(!grant||!tasks.value?.saveJob)return;const selected=[...completedBatch.value];let count=0;const failed:string[]=[];for(const job of selected){try{await tasks.value.saveJob(job.id,grant.id);count++;}catch{failed.push(labels.value[job.id]??job.id.slice(0,8));}}notice.value=uiMessage('modules.egg-analysis.saved_complete_results_for_value_batch_tasks_value',{v0:(count),v1:(failed.length?joinMessages(uiMessage('modules.egg-analysis.not_saved'),failed.join('、')):'')});}catch(e){error.value=uiErrorMessage(e);}finally{savingBatch.value=false;}}
async function saveResult(job:JobView){
  if(savingResult.value||resultLoading.value)return;const ticket=viewEpoch;
  savingResult.value=true;resultError.value='';resultNotice.value='';
  try{
    const grant=await props.context.files.choose?.('output');if(!currentResult(ticket,job.id))return;
    if(grant&&tasks.value?.saveJob){
      const result=await tasks.value.saveJob(job.id,grant.id);
      if(currentResult(ticket,job.id))resultNotice.value=uiMessage('modules.egg-analysis.saved_value_result_files',{v0:(result.count)});
    }else resultNotice.value=uiMessage('modules.egg-analysis.save_cancelled_computed_results_are_retained');
  }catch(e){if(currentResult(ticket,job.id))resultError.value=resultMessage(e);}
  finally{if(currentResult(ticket,job.id))savingResult.value=false;}
}
async function download(id:string,name:string){
  const ticket=viewEpoch,jobId=resultJob.value?.id;if(!jobId||resultLoading.value)return;
  try{await tasks.value?.download?.(id,resultNames.value[name]??name);}
  catch(e){if(currentResult(ticket,jobId))resultError.value=resultMessage(e);}
}
onMounted(()=>{if(props.context.files.kind!=='desktop')void refresh();void poll();});let lastPoll=0;const timer=setInterval(()=>{if((props.active!==false||pending.value||inversePending||jobs.value.some(j=>['queued','running','cancel_requested'].includes(j.state)))&&(pending.value||inversePending||Date.now()-lastPoll>=1500)){lastPoll=Date.now();void poll();}},250);
onUnmounted(()=>{disposed=true;closeSession();clearTimeout(gestureTimer);epoch++;viewEpoch++;batchEpoch++;clearInterval(timer);stop();for(const url of urls)URL.revokeObjectURL(url);});
useWebSettings(props.stateKey,{autoDbEnabled,order,...settingsFields(config,['signal_mode','flip_channels','micro_width_ms','gci_method','goi_method','peak_prominence','valley_prominence','auto_prominence','highpass_cutoff','lowpass_cutoff','spec_window_ms','spec_vmin','spec_vmax','keep_praat_f0','keep_gci_f0','keep_reaper_f0','f0_min_hz','f0_max_hz','silence_threshold','generate_images']),batchConfig,...settingsFields(wave,['showBoth','showSpectrogram'])});
</script>
<template><ModuleFrame unified fit :label="tr('modules.egg-analysis.egg_signal_analysis_workspace')" class="egg-page">
<template #toolbar><ModuleToolbar><ResearchUpload :files="context.files" accept=".wav" @uploaded="refresh"/><button class="primary" v-if="context.files.choose" @click="choose" :disabled="loading"><AppIcon name="folder"/>{{tr('modules.egg-analysis.open_audio_folder')}}</button><template v-if="context.files.add"><input ref="picker" type="file" accept=".wav" hidden multiple @change="addFiles"/><button class="primary" @click="picker?.click()">{{tr('modules.egg-analysis.import_wav')}}</button></template><button @click="refresh" :disabled="!directory&&context.files.kind==='desktop'">{{tr('modules.egg-analysis.refresh_files')}}</button><button @click="batchError='';batchOpen=true" :disabled="!files.length||!tasks?.egg||batchBusy">{{tr('modules.egg-analysis.batch_analysis')}}</button><button @click="config.flip_channels=!config.flip_channels" :disabled="!source">{{tr('modules.egg-analysis.swap_channels')}}</button><span class="egg-export-controls"><button class="primary" :disabled="!displayed||stale||!source||submitting||!!pending" @click="submit('single')">{{tr('modules.egg-analysis.save_csv_three_plots')}}</button><button v-if="longSource" class="primary" :disabled="!displayed||stale||!source||submitting" @click="submit('single',true)">{{tr('modules.egg-analysis.save_full_recording_csv_three_plots')}}</button><label :title="wave.asset?tr('modules.egg-analysis.v2_automatic_order_floor_fs_1000_6_current_order')+(Math.floor(sourceRate/1000)+6)+tr('modules.egg-analysis.manual_order_must_be_below_the_sample_count_of'):tr('modules.egg-analysis.v2_automatic_order_floor_fs_1000_6')"><span>{{tr('modules.egg-analysis.lp_order')}}</span><input v-model.number="order" :aria-label="tr('modules.egg-analysis.egg_lp_order')" type="number" min="1" max="256" :placeholder="wave.asset?tr('modules.egg-analysis.auto')+(Math.floor(sourceRate/1000)+6):tr('modules.egg-analysis.auto_52')"/></label><button class="primary" :disabled="!displayed||stale||!source||submitting||!!pending" @click="submit('inverse')">{{tr('modules.egg-analysis.inverse_filtered_if')}}</button></span><template #actions><button @click="save">{{tr('modules.egg-analysis.save_parameter_draft')}}</button><button @click="help=true">{{tr('modules.egg-analysis.help')}}</button><button @click="emit('references')"><AppIcon name="book"/>{{tr('modules.egg-analysis.methods_and_references')}}</button></template></ModuleToolbar></template>
<ModuleWorkbench unified :state-key="stateKey" :legacy-state-key="context.key" :left-label="tr('modules.egg-analysis.parameters_and_tasks')" :right-label="tr('modules.egg-analysis.tasks_and_records')" :left-width-alias="'--panel-right'"><div class="egg-current"><div class="egg-source"><span>{{config.flip_channels?tr('modules.egg-analysis.left_audio_right_egg'):tr('modules.egg-analysis.left_egg_right_audio')}}</span><small v-if="wave.asset">{{sourceRate}} Hz · {{duration.toFixed(3)}} s<span v-if="longSource"> {{tr('modules.egg-analysis.lightweight_overview_selection_playback_at_original_sample_rate_view')}}</span></small><span class="egg-live-status" role="status">{{loading?tr('modules.egg-analysis.reading_audio'):fontChecking?tr('modules.egg-analysis.checking_export_fonts'):submitting?tr('modules.egg-analysis.submitting_task'):previewBusy||updateQueued?tr('modules.egg-analysis.updating'):displayed?tr('modules.egg-analysis.live_preview'):''}}</span><span v-if="notice" role="status">{{notice}}</span></div>
<ModuleStatus v-if="error" kind="error" :message="error"/><button v-if="error&&source" @click="updateNow" :disabled="previewBusy">{{tr('modules.egg-analysis.retry_preview')}}</button><p v-if="!tasks?.egg" class="notice">{{tr('modules.egg-analysis.this_entry_supports_file_preview_only_use_a_desktop')}}</p>
</div><EggSignalPlots :data="displayed" :psd="psd" :config="displayedConfig" @seek="seek" @gesture="gesture"/>

<section class="egg-overview" :aria-label="tr('modules.egg-analysis.audio_overview')"><WaveformViewport :empty-message="tr('modules.egg-analysis.select_a_stereo_wav_to_show_its_overview')" :state="wave" :max-window-seconds="60" :track-height="64" normalize-display compact-overview click-moves-selection @selection-end="overviewSelection"/></section>

<template #left><section class="egg-side-controls"><div class="egg-source"><select :aria-label="tr('modules.egg-analysis.egg_audio_files')" :value="source?.id??''" :disabled="loading" @change="load(files.find(f=>f.id===($event.target as HTMLSelectElement).value)!)"><option value="" disabled>{{tr('modules.egg-analysis.select_stereo_wav')}}</option><option v-for="f in files" :key="f.id" :value="f.id">{{f.name}}</option></select><small v-if="wave.dirty">{{tr('modules.egg-analysis.parameters_not_saved')}}</small></div><div class="egg-filter"><label><input v-model="config.signal_mode" type="checkbox" true-value="filtered" false-value="raw"/>{{tr('modules.egg-analysis.filtering')}}</label><label>{{tr('modules.egg-analysis.high_pass_hz')}} <input v-model.number="config.highpass_cutoff" :aria-label="tr('modules.egg-analysis.egg_high_pass_frequency')" type="number" min="1" max="47999" step="1"/></label><label>{{tr('modules.egg-analysis.low_pass_hz')}} <input v-model.number="config.lowpass_cutoff" :aria-label="tr('modules.egg-analysis.egg_low_pass_frequency')" type="number" min="1" max="47999"/></label></div><section class="egg-controls"><EggControls v-model="config" :start="wave.start" :end="wave.end" :duration="duration" @range="setRange"/><EggParameters v-model="config" :auto-db-enabled="autoDbEnabled" :auto-db-empty="!!displayed&&!displayed.suggested_db_range" @auto-db="autoDb"/></section></section><details class="egg-history" open><summary>{{tr('modules.egg-analysis.processing_records')}} {{visibleJobs.length}} {{tr('modules.egg-analysis.tasks')}}</summary><p class="hint">{{tr('modules.egg-analysis.submitted_tasks_are_stored_in_the_current_project_after')}}</p><div class="egg-actions"><button v-if="batchBusy||batchIds.length" @click="cancelBatch">{{tr('modules.egg-analysis.cancel_this_batch')}}</button><button class="primary" v-if="tasks?.saveJob&&completedBatch.length" :disabled="savingBatch" @click="saveBatch">{{tr('modules.egg-analysis.saveBatchCount',{count:completedBatch.length})}}</button><button @click="poll">{{tr('modules.egg-analysis.refresh_records')}}</button></div><TaskPanel :tasks="jobViews" :empty-title="tr('modules.egg-analysis.no_egg_tasks_yet')" :empty-text="tr('modules.egg-analysis.csv_three_plot_inverse_filtering_and_batch_progress_appears')" @cancel="cancel" @retry="retry"/><div class="egg-result-links"><template v-for="job in visibleJobs.filter(j=>j.state==='succeeded')" :key="job.id"><button @click="view(job)">{{tr('modules.egg-analysis.view')}} {{jobTitle(job.id)}}</button></template></div></details></template></ModuleWorkbench><div class="global-transport egg-bottom-bar" :aria-label="tr('modules.egg-analysis.egg_time_selection_and_playback')"><AudioTransport :state="playable" :selection-state="wave" :active="active&&!resultJob"/></div><EggBatchPanel v-if="batchOpen" :files="files" :config="batchConfig" :busy="batchBusy" :error="batchError" @close="batchOpen=false" @submit="batchSubmit" @cancel="cancelBatch"/>
<ModalDialog v-if="help" :title="tr('modules.egg-analysis.egg_instructions')" wide @close="help=false"><p>{{tr('modules.egg-analysis.stereo_defaults_to_egg_on_the_left_and_audio')}}</p><p>{{tr('modules.egg-analysis.defaults_gci_slope_goi_scale_0_25_high_pass')}}</p><p>{{tr('modules.egg-analysis.saving_csv_three_plots_creates_a_complete_export_task')}}</p><p>{{tr('modules.egg-analysis.inverse_filtering_suits_stable_vowel_segments_and_is_limited')}}</p></ModalDialog>
<ModalDialog v-if="resultJob" class="egg-result-dialog" :title="tr('modules.egg-analysis.egg_task_results')" wide @close="closeResult"><p v-if="resultLoading" role="status">{{tr('modules.egg-analysis.reading_results')}}</p><p v-if="resultReadError" role="alert" class="error-banner">{{resultReadError}}</p><p v-if="resultError" role="alert" class="error-banner">{{resultError}}</p><p v-if="resultNotice" role="status">{{resultNotice}}</p><EggInverseResult v-if="inverseView" :data="inverseView"><template #actions><button v-if="resultReadError" :disabled="resultLoading||savingResult" @click="view(resultJob)">{{tr('modules.egg-analysis.read_results_again')}}</button><button class="primary" v-if="tasks?.saveJob" :disabled="savingResult||resultLoading||!resultConfig" @click="saveResult(resultJob)">{{tr('modules.egg-analysis.select_folder_to_save_complete_results')}}</button><template v-if="tasks?.download&&resultJob.result_manifest?.kind==='managed_egg_files'"><button v-for="f in resultJob.result_manifest.files" :key="f.id" :disabled="resultLoading||!resultConfig" @click="download(f.id,f.name)">{{tr('modules.egg-analysis.download')}} {{resultNames[f.name]??f.name}}</button></template><button @click="closeResult">{{tr('modules.egg-analysis.back_to_analysis')}}</button></template></EggInverseResult><div v-else class="egg-result-actions"><button v-if="resultReadError" :disabled="resultLoading||savingResult" @click="view(resultJob)">{{tr('modules.egg-analysis.read_results_again')}}</button><button class="primary" v-if="tasks?.saveJob" :disabled="savingResult||resultLoading||!resultConfig" @click="saveResult(resultJob)">{{tr('modules.egg-analysis.select_folder_to_save_complete_results')}}</button><template v-if="tasks?.download&&resultJob.result_manifest?.kind==='managed_egg_files'"><button v-for="f in resultJob.result_manifest.files" :key="f.id" :disabled="resultLoading||!resultConfig" @click="download(f.id,f.name)">{{tr('modules.egg-analysis.download')}} {{resultNames[f.name]??f.name}}</button></template><button @click="closeResult">{{tr('modules.egg-analysis.back_to_analysis')}}</button></div><p class="hint">{{tr('modules.egg-analysis.task')}} {{resultJob.id.slice(0,8)}} {{tr('modules.egg-analysis.parameters_and_results_come_from_this_computation_snapshot')}}</p><details v-if="resultConfig"><summary>{{tr('modules.egg-analysis.parameter_snapshot')}}</summary><pre>{{JSON.stringify(resultConfig,null,2)}}</pre></details><p v-if="resultConfig?.mode==='inverse'">{{tr('modules.egg-analysis.original_audio_is_the_normalized_analysis_segment_if_uses')}}</p><div v-if="inverseView" class="egg-wave-options"><strong>{{tr('modules.egg-analysis.waveform_analysis_across_selection')}}</strong><label><input v-model="showEggWave" type="checkbox" :disabled="!inverseWaves.some(item=>item.kind==='egg')"/>{{tr('modules.egg-analysis.show_egg_waveform')}}</label><small v-if="!inverseView.full_egg_values">{{tr('modules.egg-analysis.this_historical_task_did_not_save_full_interval_egg')}}</small></div><section v-for="(item,i) in inverseWaves.filter(item=>showEggWave||item.kind!=='egg')" :key="item.kind"><h3>{{tr(item.labelKey)}}</h3><WaveformViewport :state="item.state"/><AudioTransport v-if="item.kind!=='egg'" :state="item.state" :active="active" compact/></section><figure v-for="image in resultImages" :key="image.name"><figcaption>{{image.name}} {{tr('modules.egg-analysis.exported_plots')}}</figcaption><img :src="image.url" :alt="image.name" class="egg-export-image"/></figure></ModalDialog>
</ModuleFrame></template>
<style scoped>
.egg-result-dialog{width:min(1600px,96vw);max-width:96vw}.egg-live-status{min-width:90px;min-height:20px}.egg-page{display:flex;flex-direction:column;min-width:0;overflow:hidden;flex:1}.egg-heading{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap}.egg-heading h1{font-size:1.428571rem}.egg-actions,.egg-source,.egg-filter,.egg-filter label{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.egg-source select{flex:1;max-width:520px;min-width:170px}.egg-source>span{font-size:0.857143rem;color:var(--muted)}.egg-filter{font-size:0.857143rem}.egg-filter input[type=number]{width:67px;min-height:28px;padding:4px 6px}.egg-controls{background:var(--app);border:1px solid var(--border);border-radius:var(--radius);padding:8px 12px;display:grid;gap:8px}.egg-overview{border:1px solid var(--border);border-radius:var(--radius);padding:8px 12px}.egg-history{border-top:1px solid var(--border);padding:4px 0}.egg-history summary{cursor:pointer;font-weight:600}.egg-history :deep(.task-panel){max-height:280px;overflow:auto}.egg-result-links{display:flex;gap:6px;flex-wrap:wrap;max-height:160px;overflow:auto;margin-top:8px}.egg-result-links button{max-width:100%;overflow:hidden;text-overflow:ellipsis;display:block}.egg-export-image{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}figure{margin:12px 0}@media(max-width:700px){.egg-page{padding:10px}.egg-heading{align-items:flex-start}.egg-source select{max-width:100%;flex-basis:100%}}
.egg-side-controls{display:grid;gap:8px;min-width:0}.egg-source select{max-width:100%;min-width:0;width:100%;flex-basis:100%}.egg-source small{color:var(--muted)}.egg-side-controls .egg-source select{min-height:28px;padding:3px 6px;font-size:var(--control-size,12px)}
.egg-controls{padding:8px}.egg-filter{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,145px),1fr));gap:5px 10px;border:1px solid var(--border);border-radius:var(--radius);padding:8px}.egg-filter label{display:grid;grid-template-columns:minmax(0,1fr) minmax(64px,86px);gap:5px;align-items:center;font-size:var(--control-size,12px)}.egg-filter label:has(input[type=checkbox]){display:flex;grid-column:1/-1}.egg-filter input[type=number]{width:100%;min-height:28px;padding:3px 5px;line-height:1.25}
.egg-export-controls{display:flex;align-items:center;gap:var(--control-gap);flex-wrap:wrap}.egg-export-controls label{display:flex;align-items:center;gap:5px;font-size:var(--control-size,12px)}.egg-export-controls input{width:78px;min-height:28px;padding:3px 5px}.egg-page :deep(.workbench-left button){min-height:28px;padding:3px 7px;font-size:var(--control-size,12px)}.egg-page :deep(.workbench-left .task-panel){max-height:none}.egg-overview{padding:6px 8px}.egg-history{margin:0}.egg-result-actions,.egg-wave-options{display:flex;flex-wrap:wrap;gap:8px;align-items:center}.egg-wave-options{margin:12px 0}.egg-wave-options label{display:flex;align-items:center;gap:5px}
.egg-page :deep(.workbench-center){display:grid;grid-template-rows:auto minmax(480px,1fr) auto;gap:8px}.egg-current{display:grid;gap:6px}.egg-page :deep(.workbench-center>.egg-four-plots){min-height:0;flex-shrink:1}.egg-page :deep(.workbench-center>.egg-overview){margin:0}@media(max-width:980px){.egg-page :deep(.workbench-center){grid-template-rows:auto minmax(1120px,1fr) auto}}

.egg-page :deep(.module-workbench){flex:1;min-height:0;overflow:auto}.egg-bottom-bar{margin:0 calc(-1 * var(--module-padding));padding:8px var(--module-padding);flex:none}.egg-bottom-bar :deep(.selection-controls input){width:100px}.egg-bottom-bar :deep(.audio-transport){min-width:min(100%,280px)}.egg-page :deep(.module-toolbar){flex-shrink:0}
.egg-page :deep(.module-workbench > *){flex-shrink:0}
@container module (max-width:1000px){.egg-bottom-bar :deep(.transport-controls){flex-direction:column;align-items:stretch}.egg-bottom-bar :deep(.audio-transport){width:100%}}
</style>
