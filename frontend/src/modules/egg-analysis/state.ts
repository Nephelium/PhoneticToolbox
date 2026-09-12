import type {EggTaskConfig,EggPreviewData} from '../../platform/research.ts';
export const defaults=():EggTaskConfig=>({mode:'preview',signal_mode:'filtered',flip_channels:false,roi_start:0,roi_end:.5,micro_center:.25,micro_width_ms:50,gci_method:'slope',goi_method:'scale',peak_prominence:.01,valley_prominence:.01,auto_prominence:true,highpass_cutoff:25,lowpass_cutoff:1000,spec_window_ms:20,spec_vmin:-70,spec_vmax:-10,keep_praat_f0:false,keep_gci_f0:false,glottal_movement:false,silence_threshold:.01,generate_images:false,lp_order:null,export_policy:'sample-aligned/1'});
export function taskConfig(config:EggTaskConfig,mode:EggTaskConfig['mode'],order:number|null=null):EggTaskConfig {
  const result={...config,mode,font:undefined,lp_order:mode==='inverse'?order:null};
  if(mode!=='preview'){result.micro_center=null;result.micro_width_ms=50;}
  if(mode==='batch')Object.assign(result,{roi_start:0,roi_end:null,signal_mode:'filtered'});
  return result;
}
export const batchDefaults=():EggTaskConfig=>({...taskConfig(defaults(),'batch'),keep_praat_f0:true,keep_gci_f0:true});
export function signature(config:EggTaskConfig){const c={...defaults(),...config};delete c.font;return JSON.stringify(Object.keys(c).sort().map(k=>[k,c[k as keyof EggTaskConfig]]));}
export function validate(config:EggTaskConfig,duration:number,rate:number){
  if(config.mode==='preview'&&(!Number.isFinite(config.micro_width_ms)||config.micro_width_ms!<5||config.micro_width_ms!>5000))throw Error('微观窗口须在 5–5000 ms 范围内。');
  const end=config.roi_end??duration,start=config.roi_start??0;
  if(!Number.isFinite(start)||!Number.isFinite(end)||start<0||end<=start||end>duration)throw Error('分析选区须在音频范围内，且终点大于起点。');
  if(!Number.isFinite(config.highpass_cutoff)||!Number.isFinite(config.lowpass_cutoff)||config.highpass_cutoff!<=0||config.highpass_cutoff!>=config.lowpass_cutoff!||config.lowpass_cutoff!>=rate/2)throw Error('滤波频率须满足 0 < 高通 < 低通 < 采样率的一半。');
  if(!Number.isFinite(config.spec_vmin)||!Number.isFinite(config.spec_vmax)||config.spec_vmin!>=config.spec_vmax!)throw Error('语谱图 dB 下限必须小于上限。');
  if(duration>120||duration*rate>5_760_000)throw Error('当前 EGG 计算限 120 秒及 576 万帧，请先在参数估计中切分较长音频。总览可继续查看。');
}
export interface PreviewRecord {config:EggTaskConfig;input_sha256:string;sample_rate_hz:number;sample_count:number;preview:EggPreviewData;selection:{start_s:number;end_s:number}}
export const errors:Record<string,string>={quota_exceeded:'文件空间不足，请清理不需要的文件后重试。',asset_expired:'结果已到期，请重新计算。',egg_stereo_required:'EGG 需要双声道 WAV，默认左声道 EGG、右声道音频。',egg_input_budget:'超过 EGG 计算预算，请先切分为不超过 120 秒、576 万帧的音频。',egg_filter_failed:'滤波失败，请检查截止频率和片段长度。',egg_invalid_roi:'选区或微观中心超出音频范围。',egg_inverse_unavailable:'当前片段缺少足够的有效周期，无法进行简化 CP 逆滤波。',egg_inverse_budget:'逆滤波选区限 1 秒、48000 帧。',egg_runtime_unavailable:'EGG 独立科学运行环境尚未就绪。',analysis_resource_limit:'计算超过资源预算，请缩短音频或选区。',deadline_exceeded:'计算超时，请缩短音频后重试。',input_unavailable:'源文件已失效，请重新选择。',font_unavailable:'导出字体不可用，请在公共字体设置中检查。'};
