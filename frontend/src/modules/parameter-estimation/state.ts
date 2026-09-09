import type { components } from '../../../../contracts/generated/api';
import type { Workspace } from '../../state/workspace.ts';
import type { ResearchFile,DirectoryGrant,Tier } from '../../platform/research.ts';
import schema from '../../../../contracts/schemas/acousticrequest.json' with {type:'json'};
export type Settings=components['schemas']['AcousticSettings'];
export type SettingKey=keyof Settings;
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
export interface Association {textgrid:ResearchFile|null;lip:ResearchFile|null;tiers:Tier[];gridHash:string;layer:number;error:string;manual:{textgrid:boolean;lip:boolean}}
export interface M01State {
 wave:Workspace;files:ResearchFile[];selected:string;marked:string[];associations:Record<string,Association>;
 input:DirectoryGrant|null;output:DirectoryGrant|null;associationDirectory:DirectoryGrant|null;sameDirectory:boolean;
 settings:Settings;settingsDraft:Settings;parameterDraft:string[];drawer:''|'parameters'|'settings';
 saved:string;loadVersion:number;listVersion:number;
}
export function createState(saved?:{parameters?:string[];settings?:Settings}):M01State {
 const settings=saved?.settings&&!validateSettings(saved.settings)?{...saved.settings}:defaults();
 const parameters=saved?.parameters?.length&&saved.parameters.every(k=>parameterKeys.includes(k as never))?[...new Set(saved.parameters)]:[...parameterKeys];
 const state:M01State={wave:{asset:null,start:0,end:0,channel:0,parameters,dirty:false,error:'',loading:false,zoom:1,offset:0},files:[],selected:'',marked:[],associations:{},input:null,output:null,associationDirectory:null,sameDirectory:true,
   settings,settingsDraft:{...settings},parameterDraft:[...parameters],drawer:'',saved:'',loadVersion:0,listVersion:0};state.saved=draftJson(state);return state;
}
export function draftJson(state:M01State){return JSON.stringify({parameters:state.wave.parameters,settings:state.settings});}
export function dirty(state:M01State){return draftJson(state)!==state.saved|| (state.drawer==='settings'&&JSON.stringify(state.settingsDraft)!==JSON.stringify(state.settings))||(state.drawer==='parameters'&&JSON.stringify(state.parameterDraft)!==JSON.stringify(state.wave.parameters));}
export function applyParameters(state:M01State,keys:string[]){if(!keys.length||new Set(keys).size!==keys.length||keys.some(k=>!parameterKeys.includes(k as never)))throw Error('请至少选择一个有效参数，且不能重复。');state.wave.parameters=[...keys];state.drawer='';state.wave.dirty=dirty(state);}
export function applySettings(state:M01State,value:Settings){const error=validateSettings(value);if(error)throw Error(error);state.settings={...value};state.drawer='';state.wave.dirty=dirty(state);}
export function association(state:M01State,id=state.selected):Association {return state.associations[id]??(state.associations[id]={textgrid:null,lip:null,tiers:[],gridHash:'',layer:0,error:'',manual:{textgrid:false,lip:false}});}
export function matchAssociation(audio:ResearchFile,files:ResearchFile[],kind:'textgrid'|'lip'){
 const name=audio.name.replace(/\.wav$/i,'')+(kind==='textgrid'?'.textgrid':'.lip.json');const matches=files.filter(f=>f.kind===kind&&f.name.toLowerCase()===name.toLowerCase());
 if(matches.length>1)throw Error('同名关联有多个候选，请为该音频明确选择。');return matches[0]??null;
}
export function reconcile(state:M01State,files:ResearchFile[]){
 if(new Set(files.map(f=>f.id)).size!==files.length)throw Error('文件列表含重复标识。');
 state.files=files;
 state.marked=state.marked.filter(id=>files.some(f=>f.id===id&&f.kind==='audio'));
 if(!files.some(f=>f.kind==='audio'&&f.id===state.selected)){state.selected='';state.wave.asset=null;state.wave.start=0;state.wave.end=0;state.loadVersion++;}
 for(const id of Object.keys(state.associations)){if(!files.some(f=>f.id===id)){delete state.associations[id];continue;}const a=state.associations[id];for(const kind of ['textgrid','lip'] as const){const old=a[kind];if(old&&!files.some(f=>f.id===old.id&&f.sha256===old.sha256)){a[kind]=null;if(kind==='textgrid'){a.tiers=[];a.gridHash='';}}}}
}
export function snapshot(state:M01State){if(!state.wave.parameters.length)throw Error('请选择输出参数。');const error=validateSettings(state.settings);if(error)throw Error(error);return {inputs:state.files.filter(f=>f.kind==='audio').map(f=>({...f})),parameters:[...state.wave.parameters],settings:{...state.settings}};}
export function effectiveOutput(state:M01State){return state.sameDirectory?state.input:state.output;}
