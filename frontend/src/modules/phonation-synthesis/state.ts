import {uiError,uiErrorMessage,uiMessage,joinMessages} from '../module-localization.ts';
import {localized} from '../module-localization.ts';
import {tr} from '../../i18n/index.ts';
import type {components} from '../../../../contracts/generated/api';
export type Analysis=components['schemas']['M07Analysis'];
export type Generation=components['schemas']['M07Generation'];
export type Points=components['schemas']['M07Controls'];
export type Request=components['schemas']['M07Request'];
export const defaults=():Analysis=>({target_sample_rate:11025,f0_backend:'parselmouth',min_f0_hz:50,max_f0_hz:300,f0_frame_interval_ms:1,frame_length:128,frame_shift:32,lpc_order:20,preemphasis:.98,window_name:'hamming',negative_peak_threshold:-.005,pulse_inner_periods:.5,pulse_outer_periods:1.5,trim_silence:true,silence_threshold_db:-45,silence_padding_ms:8,voiced_margin_ms:30});
export const generationDefaults=():Generation=>({step_count:9,energy_match:true,normalize_to_source:true,output_peak_limit:.98});
export const labels:Partial<Record<keyof Analysis,string>>=localized(()=>({min_f0_hz:tr("modules.phonation-synthesis.minimum_f0_hz"),max_f0_hz:tr("modules.phonation-synthesis.maximum_f0_hz"),f0_frame_interval_ms:tr("modules.phonation-synthesis.f0_frame_interval_ms"),frame_length:tr("modules.phonation-synthesis.frame_length_samples"),frame_shift:tr("modules.phonation-synthesis.frame_step_samples"),lpc_order:tr("modules.phonation-synthesis.lpc_order"),preemphasis:tr("modules.phonation-synthesis.pre_emphasis"),negative_peak_threshold:tr("modules.phonation-synthesis.negative_peak_threshold"),pulse_inner_periods:tr("modules.phonation-synthesis.inner_pulse_window_periods"),pulse_outer_periods:tr("modules.phonation-synthesis.outer_pulse_window_periods"),silence_threshold_db:tr("modules.phonation-synthesis.silence_threshold_db"),silence_padding_ms:tr("modules.phonation-synthesis.silence_padding_ms"),voiced_margin_ms:tr("modules.phonation-synthesis.voiced_boundary_margin_ms")}));
export const kinds:Record<number,string>=localized(()=>({1:tr("modules.phonation-synthesis.phonation_type_only"),2:tr("modules.phonation-synthesis.f0_only"),3:tr("modules.phonation-synthesis.f0_and_phonation_type")}));
export const designs=()=>[false,true].flatMap(reverse=>[2,1,3].map(kind=>({reverse,kind:kind as 1|2|3})));
export const clone=<T>(value:T):T=>JSON.parse(JSON.stringify(value));
export function validate(a:Analysis,g:Generation){
 for(const [k,v] of Object.entries(a))if(typeof v==='number'&&!Number.isFinite(v))throw uiError(uiMessage("modules.phonation-synthesis.value_must_be_a_finite_number",{v0:(labels[k as keyof Analysis]??k)}));
 if(!(a.min_f0_hz>=20&&a.max_f0_hz<=1000&&a.min_f0_hz<a.max_f0_hz))throw uiError(uiMessage("modules.phonation-synthesis.minimum_f0_must_be_below_maximum_f0_within_20"));
 if(!Number.isInteger(a.frame_length)||!Number.isInteger(a.frame_shift)||!Number.isInteger(a.lpc_order)||a.frame_shift>=a.frame_length||a.lpc_order>=a.frame_length)throw uiError(uiMessage("modules.phonation-synthesis.frame_length_frame_step_and_lpc_order_must_be"));
 if(!(a.pulse_outer_periods>a.pulse_inner_periods))throw uiError(uiMessage("modules.phonation-synthesis.outer_pulse_window_must_exceed_inner_window"));
 if(!Number.isInteger(g.step_count)||g.step_count<2||g.step_count>50)throw uiError(uiMessage("modules.phonation-synthesis.continuum_steps_must_be_an_integer_from_2_to"));
 if(!Number.isFinite(g.output_peak_limit)||g.output_peak_limit<=0||g.output_peak_limit>1)throw uiError(uiMessage("modules.phonation-synthesis.output_peak_limit_must_be_greater_than_0_and"));
}
export function validatePoints(p:Points){
 if(p.axis.length<2||p.axis.length>200||p.source.length!==p.axis.length||p.target.length!==p.axis.length)throw uiError(uiMessage("modules.phonation-synthesis.control_point_counts_do_not_match"));
 if(p.axis.some((v,i)=>!Number.isFinite(v)||v<0||(i>0&&v<=p.axis[i-1])))throw uiError(uiMessage("modules.phonation-synthesis.control_point_times_must_be_finite_and_strictly_increasing"));
 for(const v of [p.source,p.target])if(v.some(n=>!Number.isFinite(n)||n<0||n>1000)||!v.some(n=>n>0))throw uiError(uiMessage("modules.phonation-synthesis.each_f0_trajectory_needs_at_least_one_valid_point"));
 return p;
}
export function table(source:number[],target:number[],count:number,mode:'normalize'|'onset'):Points{
 if(!Number.isInteger(count)||count<20||count>200)throw uiError(uiMessage("modules.phonation-synthesis.control_point_count_must_be_an_integer_from_20"));
 const bounds=(values:number[])=>{const indices=values.flatMap((v,i)=>Number.isFinite(v)&&v>0?[i]:[]);return indices.length?[indices[0],indices.at(-1)!]:[0,0]};
 const sb=bounds(source),tb=bounds(target),end=mode==='normalize'?100:Math.max(sb[1]-sb[0],tb[1]-tb[0],1),axis=Array.from({length:count},(_,i)=>i*end/(count-1));
 function sample(values:number[],b:number[]){const points=values.slice(b[0],b[1]+1).flatMap((v,i)=>v>0&&Number.isFinite(v)?[[mode==='normalize'?i*100/Math.max(1,b[1]-b[0]):i,v]]:[]);return axis.map(t=>{if(!points.length||t<points[0][0]||t>points.at(-1)![0])return 0;const right=points.findIndex(p=>p[0]>=t);if(right<=0)return Number(points[0][1].toFixed(3));const [x,y]=points[right-1],[xx,yy]=points[right];return Number((y+(yy-y)*(t-x)/(xx-x)).toFixed(3));});}
 return {axis,source:sample(source,sb),target:sample(target,tb)};
}
export const current=(ticket:number,epoch:number,revision:string,now:string)=>ticket===epoch&&revision===now;
