import {uiError,uiErrorMessage,uiMessage,joinMessages,messageProperty,setMessageProperty} from '../module-localization.ts';
import {localized} from '../module-localization.ts';
import {tr,messageText,type MessageDescriptor} from '../../i18n/index.ts';
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
export const settingLabels:Record<SettingKey,[string,string]>=localized(()=>({
 silence_threshold:[tr("modules.parameter-estimation.silence_energy_threshold"),tr("modules.parameter-estimation.relative_ratio")],energy_window_ms:[tr("modules.parameter-estimation.energy_window"),'ms'],frameshift_ms:[tr("modules.parameter-estimation.frame_step"),'ms'],windowsize_ms:[tr("modules.parameter-estimation.analysis_window_220"),'ms'],
 smooth_win_size:[tr("modules.parameter-estimation.parameter_smoothing_window"),tr("modules.parameter-estimation.frames")],lip_smooth_win_size:[tr("modules.parameter-estimation.lip_smoothing_window"),tr("modules.parameter-estimation.frames_0_disables")],only_voiced:[tr("modules.parameter-estimation.keep_voiced_frames_only_by_fundamental_frequency"),''],
 n_periods:[tr("modules.parameter-estimation.spectral_analysis_periods"),tr("modules.parameter-estimation.periods")],num_formants:[tr("modules.parameter-estimation.number_of_formants"),''],max_formant:[tr("modules.parameter-estimation.maximum_formant"),'Hz'],
 min_f0:[tr("modules.parameter-estimation.minimum_fundamental_frequency"),'Hz'],max_f0:[tr("modules.parameter-estimation.maximum_fundamental_frequency"),'Hz'],reaper_hilbert:[tr("modules.parameter-estimation.reaper_hilbert_transform"),''],reaper_no_highpass:[tr("modules.parameter-estimation.disable_reaper_high_pass_filter"),''],
}));
const settingNameKeys:Record<SettingKey,string>={"silence_threshold":"modules.parameter-estimation.silence_energy_threshold","energy_window_ms":"modules.parameter-estimation.energy_window","frameshift_ms":"modules.parameter-estimation.frame_step","windowsize_ms":"modules.parameter-estimation.analysis_window_220","smooth_win_size":"modules.parameter-estimation.parameter_smoothing_window","lip_smooth_win_size":"modules.parameter-estimation.lip_smoothing_window","only_voiced":"modules.parameter-estimation.keep_voiced_frames_only_by_fundamental_frequency","n_periods":"modules.parameter-estimation.spectral_analysis_periods","num_formants":"modules.parameter-estimation.number_of_formants","max_formant":"modules.parameter-estimation.maximum_formant","min_f0":"modules.parameter-estimation.minimum_fundamental_frequency","max_f0":"modules.parameter-estimation.maximum_fundamental_frequency","reaper_hilbert":"modules.parameter-estimation.reaper_hilbert_transform","reaper_no_highpass":"modules.parameter-estimation.disable_reaper_high_pass_filter"};
export function validateSettingsMessage(value:Settings):string|MessageDescriptor {
 if(Object.keys(value).length!==14||Object.keys(value).some(k=>!(k in rules)))return uiMessage("modules.parameter-estimation.settings_are_incomplete_or_contain_unknown_fields");
 for(const [key,rule] of Object.entries(rules)){const v=value[key as SettingKey];if(rule.type==='boolean'){if(typeof v!=='boolean')return uiMessage("modules.parameter-estimation.invalid_switch_value");continue;}
  if(typeof v!=='number'||!Number.isFinite(v)||(rule.type==='integer'&&!Number.isInteger(v))||(rule.minimum!==undefined&&v<rule.minimum)||(rule.exclusiveMinimum!==undefined&&v<=rule.exclusiveMinimum)||(rule.maximum!==undefined&&v>rule.maximum))return joinMessages(uiMessage(settingNameKeys[key as SettingKey]),uiMessage("modules.parameter-estimation.is_outside_the_supported_range"));}
 return value.min_f0!>=value.max_f0!?uiMessage("modules.parameter-estimation.minimum_fundamental_frequency_must_be_below_maximum_fundamental_frequency"):'';
}
export const validateSettings=(value:Settings):string=>messageText(validateSettingsMessage(value));
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
export function applyParameters(state:M01State,keys:string[]){if(!keys.length||new Set(keys).size!==keys.length||keys.some(k=>!parameterKeys.includes(k as never)))throw uiError(uiMessage("modules.parameter-estimation.select_at_least_one_valid_parameter_without_duplicates"));state.wave.parameters=[...keys];state.drawer='';state.wave.dirty=dirty(state);}
export function applySettings(state:M01State,value:Settings){const error=validateSettingsMessage(value);if(error)throw uiError(error);state.settings={...value};state.drawer='';state.wave.dirty=dirty(state);}
export function association(state:M01State,id=state.selected):Association {const value=state.associations[id]??(state.associations[id]={textgrid:null,lip:null,legacy:null,tiers:[],gridHash:'',layer:0,error:'',lipError:'',manual:{textgrid:false,lip:false}});return messageProperty(messageProperty(value,'error'),'lipError');}
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
 if(matches.length>1)throw uiError(uiMessage("modules.parameter-estimation.multiple_same_name_associations_exist_select_one_explicitly_for"));return matches[0]??null;
}
export function associateLips(state:M01State,replaceManual=false){
 const result={matched:0,missing:0,ambiguous:0,skipped:0};
 for(const audio of state.files.filter(f=>f.kind==='audio')){
  const a=association(state,audio.id);if(a.manual.lip&&!replaceManual){result.skipped++;continue;}
  try{const file=matchAssociation(audio,state.files,'lip');a.lipError='';if(file){a.lip=file;a.manual.lip=false;result.matched++;}else{if(!a.manual.lip)a.lip=null;result.missing++;}}
  catch(e){setMessageProperty(a,"lipError",e instanceof Error?uiErrorMessage(e):uiMessage("modules.parameter-estimation.lip_association_failed"));result.ambiguous++;}
 }
 return result;
}
export function reconcile(state:M01State,files:ResearchFile[]){
 if(new Set(files.map(f=>f.id)).size!==files.length)throw uiError(uiMessage("modules.parameter-estimation.the_file_list_contains_duplicate_identifiers"));
 state.files=files;
 state.marked=state.marked.filter(id=>files.some(f=>f.id===id&&f.kind==='audio'));
 if(!files.some(f=>f.kind==='audio'&&f.id===state.selected)){state.selected='';state.wave.asset=null;state.wave.start=0;state.wave.end=0;state.loadVersion++;}
 for(const id of Object.keys(state.associations)){if(!files.some(f=>f.id===id)){delete state.associations[id];continue;}const a=state.associations[id];for(const kind of ['textgrid','lip','legacy'] as const){const old=a[kind];if(old&&!files.some(f=>f.id===old.id&&(!f.sha256||!old.sha256||f.sha256===old.sha256))){a[kind]=null;if(kind==='textgrid'){a.tiers=[];a.gridHash='';}}}}
}
export function snapshot(state:M01State){if(!state.wave.parameters.length)throw uiError(uiMessage("modules.parameter-estimation.select_output_parameters"));const error=validateSettingsMessage(state.settings);if(error)throw uiError(error);return {inputs:state.files.filter(f=>f.kind==='audio').map(f=>({...f})),parameters:[...state.wave.parameters],settings:{...state.settings}};}
export function effectiveOutput(state:M01State){return state.sameDirectory?state.input:state.output;}
