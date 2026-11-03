import {tr} from '../../i18n/index.ts';
export type EggJobMode='preview'|'single'|'batch'|'inverse';
export interface EggJobLabel {file:string;mode:EggJobMode;retries?:number}
const labelKeys:Record<EggJobMode,string>={preview:'modules.egg-analysis.interactive_analysis',single:'modules.egg-analysis.csv_three_plots',batch:'modules.egg-analysis.batch',inverse:'modules.egg-analysis.inverse_filtering'};
const valid=(value:EggJobLabel|undefined):value is EggJobLabel=>!!value&&typeof value.file==='string'&&Object.hasOwn(labelKeys,value.mode);
/** Keep historical user/file labels untouched; new labels carry stable mode metadata. */
export function eggJobTitle(metadata:EggJobLabel|undefined,fallback:string):string {
 if(!valid(metadata))return fallback;
 return metadata.file+' · '+tr(labelKeys[metadata.mode])+(metadata.retries?tr('modules.egg-analysis.retry')+(metadata.retries>1?' × '+metadata.retries:''):'');
}
export function isEggPreviewJob(metadata:EggJobLabel|undefined,legacyLabel:string|undefined):boolean {
 return valid(metadata)?metadata.mode==='preview':!!legacyLabel?.includes('交互分析');
}
