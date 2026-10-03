import type {LpcTaskConfig,LpcSpectrumData,FigureFontSnapshot,Tier} from '../../platform/research.ts';

export interface LpcDraft {order:string;freq_max_hz:string;amp_min_db:string;amp_max_db:string;dynamic_y:boolean}
export const defaults=():LpcDraft=>({order:'50',freq_max_hz:'8000',amp_min_db:'-5',amp_max_db:'35',dynamic_y:false});
function number(value:unknown,label:string,min:number,max:number,integer=false){
  if((typeof value!=='number'&&typeof value!=='string')||String(value).trim()==='')throw Error(`${label}不能为空。`);
  const n=Number(value);if(!Number.isFinite(n)||n<min||n>max||(integer&&!Number.isInteger(n)))throw Error(`${label}须为 ${min}–${max} 范围内的${integer?'整数':'数值'}。`);return n;
}
export function parameters(draft:LpcDraft){
  const order=number(draft.order,'LPC 阶数',1,200,true),freq_max_hz=number(draft.freq_max_hz,'频率上限',100,48000),amp_min_db=number(draft.amp_min_db,'幅度下限',-200,100),amp_max_db=number(draft.amp_max_db,'幅度上限',-200,100);
  if(amp_max_db<=amp_min_db)throw Error('幅度上限须大于下限。');
  if(typeof draft.dynamic_y!=='boolean')throw Error('动态纵轴设置无效。');
  return {order,freq_max_hz,amp_min_db,amp_max_db,dynamic_y:draft.dynamic_y};
}
export function restoreDraft(raw:unknown):LpcDraft {
  if(!raw||typeof raw!=='object')return defaults();
  const values=raw as LpcDraft;try{parameters(values);return {order:String(values.order),freq_max_hz:String(values.freq_max_hz),amp_min_db:String(values.amp_min_db),amp_max_db:String(values.amp_max_db),dynamic_y:values.dynamic_y};}catch{return defaults();}
}
export function visibleRange(duration:number,zoom:number,offset:number):[number,number]{const length=duration/Math.max(1,zoom),start=Math.max(0,Math.min(offset,duration-length));return [start,start+length];}
export function audioSelection(a:number,b:number,duration:number):[number,number]|null {
  if(![a,b,duration].every(Number.isFinite)||duration<=0)return null;
  const start=Math.max(0,Math.min(a,b)),end=Math.min(duration,Math.max(a,b));
  return end>start?[start,end]:null;
}
export function gridRangeStatus(tiers:Tier[],duration:number,sampleRate:number):'valid'|'blank-overhang'|'mismatch' {
  let blank=false;
  for(const tier of tiers)for(const interval of tier.intervals){
    if(interval.xmin<0||interval.xmax>duration+1/sampleRate){
      if(interval.text.trim())return 'mismatch';
      blank=true;
    }
  }
  return blank?'blank-overhang':'valid';
}
export function configuration(draft:LpcDraft,start:unknown,end:unknown,sampleRate:number,frames:number,tier:string|null,font:FigureFontSnapshot):LpcTaskConfig {
  const p=parameters(draft),duration=frames/sampleRate,roi_start=number(start,'选区起点',0,duration),roi_end=number(end,'选区终点',0,duration);
  const n=Math.trunc(roi_end*sampleRate)-Math.trunc(roi_start*sampleRate);
  if(roi_end<=roi_start)throw Error('选区终点须大于起点。');
  if(n<p.order+2)throw Error(`当前阶数至少需要 ${p.order+2} 个样本，请扩大选区或降低阶数。`);
  if(n>48000)throw Error(`选区包含 ${n.toLocaleString()} 个样本，单次上限为 48,000。请缩小选区或放大波形后分析可见范围。`);
  return {...p,roi_start,roi_end,tier_name:tier,font:{...font}};
}
export interface LpcResult {
  schema_version:'m04/1';config:LpcTaskConfig;input_name:string;input_sha256:string;textgrid_sha256:string|null;
  sample_rate_hz:number;sample_count:number;channels:number;label:string;tier_name:string|null;
  selection:{start_sample:number;end_sample:number;start_s:number;end_s:number;interval:'half-open'};
  spectrum:LpcSpectrumData;export_names:Record<string,string>;
}
export function parseResult(raw:ArrayBuffer):LpcResult {
  const r=JSON.parse(new TextDecoder().decode(raw)) as LpcResult,s=r.spectrum;
  if(r.schema_version!=='m04/1'||!s||s.frequencies_hz.length!==1024||s.magnitude_db.length!==1024||!s.frequencies_hz.every((f,i,a)=>Number.isFinite(f)&&(i===0?f===0:f>a[i-1]))||!s.magnitude_db.every(Number.isFinite)||!Number.isFinite(s.amp_min_db)||!Number.isFinite(s.amp_max_db)||s.amp_max_db<=s.amp_min_db)throw Error('LPC 结果格式或谱值无效。');
  if(!r.selection||!Number.isInteger(r.selection.start_sample)||!Number.isInteger(r.selection.end_sample)||r.selection.end_sample<=r.selection.start_sample||r.selection.interval!=='half-open'||!r.config||typeof r.input_name!=='string'||typeof r.input_sha256!=='string')throw Error('LPC 结果缺少有效时间与来源。');return r;
}
export function sameAnalysis(result:LpcResult,config:LpcTaskConfig,sha:string,gridSha:string|null){
  return result.input_sha256===sha&&result.textgrid_sha256===gridSha&&(['order','freq_max_hz','roi_start','roi_end','tier_name'] as const).every(k=>result.config[k]===config[k]);
}
export const jobError=(code:string)=>({lpc_roi_budget:'选区超过 48,000 样本，请缩小范围。',lpc_segment_too_short:'选区过短，请扩大范围或降低阶数。',lpc_solver_failed:'当前选区无法求解 LPC，请检查静音或奇异信号。',lpc_invalid_roi:'时间选区无效，请重新选择。',lpc_textgrid_range:'TextGrid 中有非空标签超出音频范围。请关联匹配的标注，或选择“不关联”后分析。',missing_or_invalid_tier:'TextGrid 层级无效，请重新关联。',lpc_runtime_unavailable:'LPC 计算环境不可用，请重启工作台。',lpc_runtime_mismatch:'LPC 计算环境版本不匹配。',lpc_input_budget:'音频超过 64 MB、800 万帧或 8 声道限制。',lpc_sample_rate:'采样率须在 8–96 kHz 范围内。',font_unavailable:'导出字体不可用，请在工作台设置中检查图表字体。',deadline_exceeded:'计算超时，请缩小选区后重试。',input_unavailable:'源文件已失效，请重新上传或选择。',quota_exceeded:'文件空间不足，请清理项目文件后重试。',invalid_audio:'音频无效或包含不支持的样本。'} as Record<string,string>)[code]??code;
