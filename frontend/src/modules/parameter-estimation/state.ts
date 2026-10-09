import type { components } from '../../../../contracts/generated/api';
import type { Workspace } from '../../state/workspace.ts';
import type { ResearchFile,DirectoryGrant,Tier } from '../../platform/research.ts';
import schema from '../../../../contracts/schemas/acousticrequest.json' with {type:'json'};
export type Settings=components['schemas']['AcousticSettings'];
export type SettingKey=keyof Settings;
export type Extended=components['schemas']['AcousticExtended'];
export type EggSettings=components['schemas']['JointEggSettings'];
export const extendedDefaults=():Extended=>({revision:'bounded/1',max_duration_s:1800,audio_channel:null,egg:null,channel_overrides:{}});
export const eggDefaults=():EggSettings=>({egg_channel:0,storage:'aligned',smooth_ms:20,max_gap_ms:50,derived:false,highpass_cutoff:25,lowpass_cutoff:2000,gci_method:'slope',goi_method:'scale',auto_prominence:true,peak_prominence:.01,valley_prominence:.01,silence_threshold:.01});
type Rule={default:number|boolean;type:string;minimum?:number;exclusiveMinimum?:number;maximum?:number};
export const rules=schema.$defs.AcousticSettings.properties as Record<SettingKey,Rule>;
export const defaults=()=>Object.fromEntries(Object.entries(rules).map(([key,rule])=>[key,rule.default])) as Settings;
export const parameterKeys=[...schema.$defs.AcousticSelection.properties.keys.items.enum];
export const settingLabels:Record<SettingKey,[string,string]>={
 silence_threshold:['静音能量阈值','相对比例'],energy_window_ms:['能量窗口','ms'],frameshift_ms:['帧移','ms'],windowsize_ms:['分析窗口','ms'],
 smooth_win_size:['参数平滑窗口','帧'],lip_smooth_win_size:['唇形平滑窗口','帧，0为关闭'],only_voiced:['仅保留有声帧（基频判定）',''],
 n_periods:['频谱分析周期数','周期'],num_formants:['共振峰数量',''],max_formant:['共振峰上限','Hz'],
 min_f0:['最小基频','Hz'],max_f0:['最大基频','Hz'],reaper_hilbert:['REAPER Hilbert变换',''],reaper_no_highpass:['禁用REAPER高通滤波',''],
};
export function validateSettings(value:Settings):string {
 if(Object.keys(value).length!==14||Object.keys(value).some(k=>!(k in rules)))return '设置字段不完整或含未知字段。';
 for(const [key,rule] of Object.entries(rules)){const v=value[key as SettingKey];if(rule.type==='boolean'){if(typeof v!=='boolean')return '开关值无效。';continue;}
  if(typeof v!=='number'||!Number.isFinite(v)||(rule.type==='integer'&&!Number.isInteger(v))||(rule.minimum!==undefined&&v<rule.minimum)||(rule.exclusiveMinimum!==undefined&&v<=rule.exclusiveMinimum)||(rule.maximum!==undefined&&v>rule.maximum))return settingLabels[key as SettingKey][0]+'超出支持范围。';}
 return value.min_f0!>=value.max_f0!?'最小基频必须小于最大基频。':'';
}
export interface Association {textgrid:ResearchFile|null;lip:ResearchFile|null;legacy:ResearchFile|null;tiers:Tier[];gridHash:string;layer:number;error:string;lipError:string;manual:{textgrid:boolean;lip:boolean}}
export interface M01State {
 wave:Workspace;files:ResearchFile[];selected:string;marked:string[];associations:Record<string,Association>;
 input:DirectoryGrant|null;output:DirectoryGrant|null;associationDirectory:DirectoryGrant|null;lipDirectory:DirectoryGrant|null;sameDirectory:boolean;recursive:boolean;
 settings:Settings;settingsDraft:Settings;parameterDraft:string[];drawer:''|'parameters'|'settings';
 extended:Extended;channelOverrides:Record<string,number>;
 saved:string;loadVersion:number;listVersion:number;
}
export function createState(saved?:{parameters?:string[];settings?:Settings;extended?:Extended}):M01State {
 const settings=saved?.settings&&!validateSettings(saved.settings)?{...saved.settings}:defaults();
 const parameters=saved?.parameters?.length&&saved.parameters.every(k=>parameterKeys.includes(k as never))?[...new Set(saved.parameters)]:[...parameterKeys];
 const state:M01State={wave:{asset:null,start:0,end:0,channel:0,parameters,dirty:false,error:'',loading:false,zoom:1,offset:0},files:[],selected:'',marked:[],associations:{},input:null,output:null,associationDirectory:null,lipDirectory:null,sameDirectory:true,recursive:false,
   extended:{...extendedDefaults(),...saved?.extended,channel_overrides:{}},channelOverrides:{},settings,settingsDraft:{...settings},parameterDraft:[...parameters],drawer:'',saved:'',loadVersion:0,listVersion:0};state.saved=draftJson(state);return state;
}
export function draftJson(state:M01State){return JSON.stringify({parameters:state.wave.parameters,settings:state.settings,extended:state.extended});}
export function dirty(state:M01State){return draftJson(state)!==state.saved|| (state.drawer==='settings'&&JSON.stringify(state.settingsDraft)!==JSON.stringify(state.settings))||(state.drawer==='parameters'&&JSON.stringify(state.parameterDraft)!==JSON.stringify(state.wave.parameters));}
export function applyParameters(state:M01State,keys:string[]){if(!keys.length||new Set(keys).size!==keys.length||keys.some(k=>!parameterKeys.includes(k as never)))throw Error('请至少选择一个有效参数，且不能重复。');state.wave.parameters=[...keys];state.drawer='';state.wave.dirty=dirty(state);}
export function applySettings(state:M01State,value:Settings){const error=validateSettings(value);if(error)throw Error(error);state.settings={...value};state.drawer='';state.wave.dirty=dirty(state);}
export function association(state:M01State,id=state.selected):Association {return state.associations[id]??(state.associations[id]={textgrid:null,lip:null,legacy:null,tiers:[],gridHash:'',layer:0,error:'',lipError:'',manual:{textgrid:false,lip:false}});}
export function isLipFile(file:ResearchFile){return file.kind==='lip'||(file.kind==='lip_pickle'&&!/_timestamps\.pkl$/i.test(file.name));}
export function directoryResources(inputs:ResearchFile[],textgrids=inputs,lips=inputs){
 const values=[...inputs.filter(f=>f.kind!=='textgrid'&&!isLipFile(f)),
  ...textgrids.filter(f=>f.kind==='textgrid'||f.kind==='parameter'),...lips.filter(isLipFile)];
 return [...new Map(values.map(f=>[f.id,f])).values()];
}
export function resetAssociations(state:M01State,kind:'textgrid'|'lip'){
 for(const a of Object.values(state.associations)){
  a[kind]=null;a.manual[kind]=false;
  if(kind==='textgrid'){a.tiers=[];a.gridHash='';a.error='';}else a.lipError='';
 }
}
export function matchAssociation(audio:ResearchFile,files:ResearchFile[],kind:'textgrid'|'lip'){
 const stem=audio.name.replace(/\.wav$/i,'').toLowerCase();
 let matches=files.filter(f=>f.kind===kind&&f.name.toLowerCase()===stem+(kind==='textgrid'?'.textgrid':'.lip.json'));
 if(kind==='lip'&&!matches.length)matches=files.filter(f=>isLipFile(f)&&f.kind==='lip_pickle'&&f.name.toLowerCase()===stem+'.pkl');
 if(matches.length>1)throw Error('同名关联有多个候选，请为该音频明确选择。');return matches[0]??null;
}
export function associateLips(state:M01State,replaceManual=false){
 const result={matched:0,missing:0,ambiguous:0,skipped:0};
 for(const audio of state.files.filter(f=>f.kind==='audio')){
  const a=association(state,audio.id);if(a.manual.lip&&!replaceManual){result.skipped++;continue;}
  try{const file=matchAssociation(audio,state.files,'lip');a.lipError='';if(file){a.lip=file;a.manual.lip=false;result.matched++;}else{if(!a.manual.lip)a.lip=null;result.missing++;}}
  catch(e){a.lipError=e instanceof Error?e.message:'唇形关联失败。';result.ambiguous++;}
 }
 return result;
}
export function reconcile(state:M01State,files:ResearchFile[]){
 if(new Set(files.map(f=>f.id)).size!==files.length)throw Error('文件列表含重复标识。');
 state.files=files;
 state.marked=state.marked.filter(id=>files.some(f=>f.id===id&&f.kind==='audio'));
 if(!files.some(f=>f.kind==='audio'&&f.id===state.selected)){state.selected='';state.wave.asset=null;state.wave.start=0;state.wave.end=0;state.loadVersion++;}
 for(const id of Object.keys(state.associations)){if(!files.some(f=>f.id===id)){delete state.associations[id];continue;}const a=state.associations[id];for(const kind of ['textgrid','lip','legacy'] as const){const old=a[kind];if(old&&!files.some(f=>f.id===old.id&&(!f.sha256||!old.sha256||f.sha256===old.sha256))){a[kind]=null;if(kind==='textgrid'){a.tiers=[];a.gridHash='';}}}}
}
export function snapshot(state:M01State){if(!state.wave.parameters.length)throw Error('请选择输出参数。');const error=validateSettings(state.settings);if(error)throw Error(error);return {inputs:state.files.filter(f=>f.kind==='audio').map(f=>({...f})),parameters:[...state.wave.parameters],settings:{...state.settings}};}
export function effectiveOutput(state:M01State){return state.sameDirectory?state.input:state.output;}
