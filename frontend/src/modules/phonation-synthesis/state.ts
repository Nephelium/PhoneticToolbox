import type {components} from '../../../../contracts/generated/api';
export type Analysis=components['schemas']['M07Analysis'];
export type Generation=components['schemas']['M07Generation'];
export type Points=components['schemas']['M07Controls'];
export type Request=components['schemas']['M07Request'];
export const defaults=():Analysis=>({target_sample_rate:11025,f0_backend:'parselmouth',min_f0_hz:50,max_f0_hz:300,f0_frame_interval_ms:1,frame_length:128,frame_shift:32,lpc_order:20,preemphasis:.98,window_name:'hamming',negative_peak_threshold:-.005,pulse_inner_periods:.5,pulse_outer_periods:1.5,trim_silence:true,silence_threshold_db:-45,silence_padding_ms:8,voiced_margin_ms:30});
export const generationDefaults=():Generation=>({step_count:9,energy_match:true,normalize_to_source:true,output_peak_limit:.98});
export const labels:Partial<Record<keyof Analysis,string>>={min_f0_hz:'最低 F0 (Hz)',max_f0_hz:'最高 F0 (Hz)',f0_frame_interval_ms:'F0 帧间隔 (ms)',frame_length:'帧长 (采样点)',frame_shift:'帧移 (采样点)',lpc_order:'LPC 阶数',preemphasis:'预加重',negative_peak_threshold:'负峰阈值',pulse_inner_periods:'脉冲内窗 (周期)',pulse_outer_periods:'脉冲外窗 (周期)',silence_threshold_db:'静音阈值 (dB)',silence_padding_ms:'静音 padding (ms)',voiced_margin_ms:'有声边界余量 (ms)'};
export const kinds:Record<number,string>={1:'仅发声类型变化',2:'仅 F0 变化',3:'F0 与发声类型同时变化'};
export const designs=()=>[false,true].flatMap(reverse=>[2,1,3].map(kind=>({reverse,kind:kind as 1|2|3})));
export const clone=<T>(value:T):T=>JSON.parse(JSON.stringify(value));
export function validate(a:Analysis,g:Generation){
 for(const [k,v] of Object.entries(a))if(typeof v==='number'&&!Number.isFinite(v))throw Error(`${labels[k as keyof Analysis]??k}必须是有限数值`);
 if(!(a.min_f0_hz>=20&&a.max_f0_hz<=1000&&a.min_f0_hz<a.max_f0_hz))throw Error('最低 F0 必须小于最高 F0，范围为 20–1000 Hz');
 if(!Number.isInteger(a.frame_length)||!Number.isInteger(a.frame_shift)||!Number.isInteger(a.lpc_order)||a.frame_shift>=a.frame_length||a.lpc_order>=a.frame_length)throw Error('帧长、帧移、LPC 阶数必须为整数，帧移和阶数须小于帧长');
 if(!(a.pulse_outer_periods>a.pulse_inner_periods))throw Error('脉冲外窗必须大于内窗');
 if(!Number.isInteger(g.step_count)||g.step_count<2||g.step_count>50)throw Error('连续统步数须为 2–50 的整数');
 if(!Number.isFinite(g.output_peak_limit)||g.output_peak_limit<=0||g.output_peak_limit>1)throw Error('输出峰值限制须大于 0 且不超过 1');
}
export function validatePoints(p:Points){
 if(p.axis.length<2||p.axis.length>200||p.source.length!==p.axis.length||p.target.length!==p.axis.length)throw Error('控制点数量不一致');
 if(p.axis.some((v,i)=>!Number.isFinite(v)||v<0||(i>0&&v<=p.axis[i-1])))throw Error('控制点时间须为严格递增的有限数值');
 for(const v of [p.source,p.target])if(v.some(n=>!Number.isFinite(n)||n<0||n>1000)||!v.some(n=>n>0))throw Error('每份 F0 至少需要一个有效点，值须在 0–1000 Hz 内，空白表示无声');
 return p;
}
export function table(source:number[],target:number[],count:number,mode:'normalize'|'onset'):Points{
 if(!Number.isInteger(count)||count<20||count>200)throw Error('控制点数量须为 20–200 的整数');
 const bounds=(values:number[])=>{const indices=values.flatMap((v,i)=>Number.isFinite(v)&&v>0?[i]:[]);return indices.length?[indices[0],indices.at(-1)!]:[0,0]};
 const sb=bounds(source),tb=bounds(target),end=mode==='normalize'?100:Math.max(sb[1]-sb[0],tb[1]-tb[0],1),axis=Array.from({length:count},(_,i)=>i*end/(count-1));
 function sample(values:number[],b:number[]){const points=values.slice(b[0],b[1]+1).flatMap((v,i)=>v>0&&Number.isFinite(v)?[[mode==='normalize'?i*100/Math.max(1,b[1]-b[0]):i,v]]:[]);return axis.map(t=>{if(!points.length||t<points[0][0]||t>points.at(-1)![0])return 0;const right=points.findIndex(p=>p[0]>=t);if(right<=0)return Number(points[0][1].toFixed(3));const [x,y]=points[right-1],[xx,yy]=points[right];return Number((y+(yy-y)*(t-x)/(xx-x)).toFixed(3));});}
 return {axis,source:sample(source,sb),target:sample(target,tb)};
}
export const current=(ticket:number,epoch:number,revision:string,now:string)=>ticket===epoch&&revision===now;
