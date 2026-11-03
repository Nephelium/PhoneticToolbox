import {msg,tr,join,bindText,appendText,bindAttribute,textNode,setHTML,onLocaleChange,parameterLabel,uiError} from './i18n.mjs';
// Explicit phase controls share the ordinary keyframe list and saved poses.
const expanded=new Set();
const ms=value=>Number((value*1000).toFixed(3));
export function initialStopTiming(duration){
 const release=Math.min(.008,duration*.1),closure=Math.max(Math.min(.06,duration*.7),duration-release-1),aspiration=duration-closure-release;
 return {closure,release,aspiration,vot:release+aspiration};
}
export function editStopTiming(previous,key,milliseconds){
 const value=Number(milliseconds)/1000,t={...previous,[key]:value};
 if(String(milliseconds).trim()===''||!Number.isFinite(value))throw uiError(msg("modules.vocal-tract-frame.text.enter_valid_milliseconds_303"));
 if(key==='aspiration')t.vot=t.release+t.aspiration;
 else t.aspiration=Math.max(0,t.vot-t.release);
 const duration=t.closure+t.release+t.aspiration;
 if(t.closure<.01||t.closure>2.999||t.release<.001||t.release>.05||t.aspiration<0||t.aspiration>1||duration>3)throw uiError(msg("modules.vocal-tract-frame.text.closure_10_2999_ms_release_1_50_ms_aspiration_0_1000_ms_total_no_more_than_3_304"));
 if(t.vot < -t.closure)throw uiError(msg("modules.vocal-tract-frame.text.prevoicing_cannot_precede_closure_onset_lengthen_closure_or_shorten_negative_305"));
 return t;
}
function timingHelp(text){
 const details=document.createElement('details');details.className='stop-help';const summary=document.createElement('summary');bindText(summary,()=>msg("modules.vocal-tract-frame.text.timing_explanation_306"));const paragraph=document.createElement('p');paragraph.className='minor';bindText(paragraph,()=>text);details.append(summary,paragraph);return details;
}
export function stopEditor({frame,index,frames,busy,onChange}){
 const details=document.createElement('details');details.className='stop-timing';details.open=expanded.has(frame.frame_id);details.ontoggle=()=>{if(details.isConnected)details.open?expanded.add(frame.frame_id):expanded.delete(frame.frame_id);};
 const commit=()=>{if(details.open)expanded.add(frame.frame_id);onChange();};
 const summary=document.createElement('summary');bindText(summary,()=>frame.stop_timing?msg("modules.vocal-tract-frame.text.stop_timing_enabled_307"):msg("modules.vocal-tract-frame.text.stop_timing_308"));details.append(summary);
 const label=document.createElement('label'),toggle=document.createElement('input');toggle.type='checkbox';toggle.checked=!!frame.stop_timing;toggle.disabled=busy;label.append(toggle,textNode(()=>msg("modules.vocal-tract-frame.text.control_closure_release_and_voicing_onset_separately_309")));details.append(label);
 const error=document.createElement('p');error.setAttribute('role','status');error.className='stop-error';details.append(error);
 toggle.onchange=()=>{if(toggle.checked){let duration=frame.duration,count=0;while(frames[index+1+count]?.aspiration){duration+=frames[index+1+count].duration;count++;}if(duration>3){toggle.checked=false;bindText(error,()=>msg("modules.vocal-tract-frame.text.merging_exceeds_3_seconds_per_frame_shorten_the_legacy_aspiration_segment_fi_310"));return;}frames.splice(index+1,count);frame.duration=duration;frame.stop_timing=initialStopTiming(duration);}else{if(frame.duration<.05){toggle.checked=true;bindText(error,()=>msg("modules.vocal-tract-frame.text.normal_frames_require_at_least_50_ms_lengthen_closure_before_disabling_timin_311"));return;}delete frame.stop_timing;}commit();};
 if(!frame.stop_timing){details.append(timingHelp(msg("modules.vocal-tract-frame.text.the_following_configuration_is_the_release_target_enabling_merges_the_immedi_312")));return details;}
 for(const [key,title,min,max] of [['closure',msg("modules.vocal-tract-frame.text.closure_313"),10,2999],['release',msg("modules.vocal-tract-frame.text.release_314"),1,50],['aspiration',msg("modules.vocal-tract-frame.text.aspiration_140"),0,1000],['vot','VOT',-2999,1050]]){
  const row=document.createElement('label');bindText(row,()=>title);const input=document.createElement('input');Object.assign(input,{type:'number',step:'any',min,max,value:ms(frame.stop_timing[key]),disabled:busy});bindAttribute(input,"aria-label",()=>msg("modules.vocal-tract-frame.text.value_milliseconds_posture_value_315",{v0:()=>title,v1:()=>index+1}));row.append(input,textNode(()=>'ms'));details.append(row);
  input.onchange=()=>{try{const t=editStopTiming(frame.stop_timing,key,input.value);frame.stop_timing=t;frame.duration=t.closure+t.release+t.aspiration;commit();}catch(e){bindText(error,()=>e.message);input.value=ms(frame.stop_timing[key]);}};
 }
 const t=frame.stop_timing,timeline=document.createElement('p');timeline.className='stop-landmarks';bindText(timeline,()=>msg("modules.vocal-tract-frame.text.this_frame_release_value_ms_voicing_onset_value_ms_end_value_ms_316",{v0:()=>ms(t.closure),v1:()=>ms(t.closure+t.vot),v2:()=>ms(frame.duration)}));details.append(timeline);
 details.append(timingHelp(msg("modules.vocal-tract-frame.text.vot_is_voicing_onset_minus_release_time_negative_means_prevoicing_during_clo_317")));return details;
}
