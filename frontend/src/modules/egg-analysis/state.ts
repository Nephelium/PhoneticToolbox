import type {EggTaskConfig,EggPreviewData} from '../../platform/research.ts';
export const defaults=():EggTaskConfig=>({mode:'preview',signal_mode:'filtered',flip_channels:false,roi_start:0,roi_end:.5,micro_center:.25,micro_width_ms:50,gci_method:'slope',goi_method:'scale',peak_prominence:.01,valley_prominence:.01,auto_prominence:true,highpass_cutoff:25,lowpass_cutoff:2000,spec_window_ms:20,spec_vmin:-70,spec_vmax:-10,keep_praat_f0:false,keep_gci_f0:false,keep_reaper_f0:false,f0_policy:'audio-f0/2',glottal_movement:false,silence_threshold:.01,generate_images:false,lp_order:null,export_policy:'sample-aligned/1'});
export function taskConfig(config:EggTaskConfig,mode:EggTaskConfig['mode'],order:number|null=null):EggTaskConfig {
  const result={...config,mode,font:undefined,lp_order:mode==='inverse'?order:null};
  if(mode!=='preview'){result.micro_center=null;result.micro_width_ms=50;}
  if(mode==='batch')Object.assign(result,{roi_start:0,roi_end:null,signal_mode:'filtered'});
  return result;
}
export const batchDefaults=():EggTaskConfig=>({...taskConfig(defaults(),'batch'),keep_praat_f0:true,keep_gci_f0:true});
export function signature(config:EggTaskConfig){const c={...defaults(),...config};delete c.font;return JSON.stringify(Object.keys(c).sort().map(k=>[k,c[k as keyof EggTaskConfig]]));}
// Parameter-only checks mirror M03/1. Per-file rate/ROI checks remain separate.
export function validateParameters(config:EggTaskConfig){
  const bounded=(value:unknown,min:number,max:number)=>typeof value==='number'&&Number.isFinite(value)&&value>=min&&value<=max;
  if(!bounded(config.highpass_cutoff,0,48000)||!bounded(config.lowpass_cutoff,0,48000)||config.highpass_cutoff!<=0||config.highpass_cutoff!>=config.lowpass_cutoff!||config.lowpass_cutoff!>=48000)throw Error('滤波频率须满足 0 < 高通 < 低通 < 48000 Hz；低通还须小于每个文件采样率的一半。');
  if(!bounded(config.silence_threshold,0,1))throw Error('静音阈值须在 0–1 范围内，不能为空。');
  if(!bounded(config.peak_prominence,0,10)||!bounded(config.valley_prominence,0,10))throw Error('峰、谷显著度须在 0–10 范围内。');
  if(!bounded(config.spec_window_ms,5,50))throw Error('谱窗须在 5–50 ms 范围内。');
  if(!bounded(config.spec_vmin,-160,20)||!bounded(config.spec_vmax,-160,20)||config.spec_vmin!>=config.spec_vmax!)throw Error('语谱图 dB 上下限须在 -160–20 范围内，且下限小于上限。');
}
export function validate(config:EggTaskConfig,duration:number,rate:number){
  validateParameters(config);
  if(config.mode==='inverse'&&config.lp_order!=null&&(!Number.isInteger(config.lp_order)||config.lp_order<1||config.lp_order>Math.min(256,Math.floor(.003*rate)-1)))throw Error(`LP 阶数须为 1–${Math.min(256,Math.floor(.003*rate)-1)} 的整数，低于当前采样率下 3 ms 窗的样本数。留空使用 V2 自动阶数。`);
  if(config.mode==='preview'&&(!Number.isFinite(config.micro_width_ms)||config.micro_width_ms!<5||config.micro_width_ms!>5000))throw Error('微观窗口须在 5–5000 ms 范围内。');
  const end=config.roi_end??duration,start=config.roi_start??0;
  if(config.mode==='inverse'&&(end-start>10||(end-start)*rate>960000))throw Error('逆滤波选区限 10 秒、960000 帧。');
  if(!Number.isFinite(start)||!Number.isFinite(end)||start<0||end<=start||end>duration)throw Error('分析选区须在音频范围内，且终点大于起点。');
  if(!Number.isFinite(config.highpass_cutoff)||!Number.isFinite(config.lowpass_cutoff)||config.highpass_cutoff!<=0||config.highpass_cutoff!>=config.lowpass_cutoff!||config.lowpass_cutoff!>=rate/2)throw Error('滤波频率须满足 0 < 高通 < 低通 < 采样率的一半。');
  if(!Number.isFinite(config.spec_vmin)||!Number.isFinite(config.spec_vmax)||config.spec_vmin!>=config.spec_vmax!)throw Error('语谱图 dB 下限必须小于上限。');
  if(duration>1800||rate<8000||rate>96000)throw Error('EGG 支持最长 30 分钟、8–96 kHz 的双声道 WAV。');
}
export interface PreviewRecord {config:EggTaskConfig;input_sha256:string;sample_rate_hz:number;sample_count:number;preview:EggPreviewData;selection:{start_s:number;end_s:number}}
export const errors:Record<string,string>={egg_reaper_unavailable:'REAPER 原生引擎未就绪，请使用已配置 REAPER 的桌面入口。',egg_reaper_failed:'REAPER 基频提取失败，可取消勾选后继续查看其他曲线。',egg_preview_failed:'实时预览未能完成，请重试预览。',egg_invalid_preview:'预览参数无效，请检查当前设置。',preview_timeout:'预览更新超时，已回收计算进程，请重试预览。',preview_platform_unverified:'当前平台尚未配置经过验证的 EGG 实时预览环境。',egg_runtime_mismatch:'EGG 科学环境与项目锁定版本不一致。',quota_exceeded:'文件空间不足，请清理不需要的文件后重试。',asset_expired:'结果已到期，请重新计算。',egg_stereo_required:'EGG 需要双声道 WAV，默认左声道 EGG、右声道音频。',egg_viewport_budget:'长录音实时视窗最多 60 秒，请缩小选区；完整导出可使用专用按钮。',egg_input_budget:'EGG 输入限 30 分钟、2 GB、8–96 kHz 双声道 WAV。',egg_filter_failed:'滤波失败，请检查截止频率和片段长度。',egg_invalid_roi:'选区或微观中心超出音频范围。',egg_inverse_unavailable:'当前片段缺少足够的有效周期，无法进行简化 CP 逆滤波。',egg_inverse_budget:'逆滤波选区限 10 秒、960000 帧。',egg_runtime_unavailable:'EGG 独立科学运行环境尚未就绪。',analysis_resource_limit:'计算超过资源预算，请缩短音频或选区。',deadline_exceeded:'计算超时，请缩短音频后重试。',input_unavailable:'源文件已失效，请重新选择。',font_unavailable:'导出字体不可用，请在公共字体设置中检查。'};

export function inverseAudioFiles<T extends {name:string}>(files:T[]){return ['egg_ORIG.wav','egg_IF.wav'].flatMap(name=>files.filter(file=>file.name===name));}
