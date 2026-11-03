import {localized} from '../module-localization.ts';
import {tr} from '../../i18n/index.ts';
import type {Metadata} from '../../platform/m07.ts';

export type TaskSnapshot=Partial<Pick<Metadata,'reverse_direction'|'continuum_type'|'generation'>> & Pick<Metadata,'action'>;
export const taskStates=localized(()=>({queued:tr("modules.phonation-synthesis.queued"),running:tr("modules.phonation-synthesis.running"),cancel_requested:tr("modules.phonation-synthesis.cancelling"),cancelled:tr("modules.phonation-synthesis.cancelled"),succeeded:tr("modules.phonation-synthesis.completed"),failed:tr("modules.phonation-synthesis.failed"),interrupted:tr("modules.phonation-synthesis.interrupted")}));
const changes=localized(()=>({1:tr("modules.phonation-synthesis.phonation_type_only"),2:tr("modules.phonation-synthesis.f0_only"),3:tr("modules.phonation-synthesis.f0_and_phonation_type")}));
export function taskDescription(snapshot?:TaskSnapshot){
 if(!snapshot)return tr("modules.phonation-synthesis.phonation_continuum");
 if(snapshot.action==='analyze')return tr("modules.phonation-synthesis.extract_f0");
 if(snapshot.action==='apply')return tr("modules.phonation-synthesis.apply_f0_edits");
 return tr("modules.phonation-synthesis.value_value_value_steps",{v0:(snapshot.reverse_direction?tr('modules.phonation-synthesis.target_to_source'):tr('modules.phonation-synthesis.source_to_target')),v1:(changes[snapshot.continuum_type??2]),v2:(snapshot.generation?.step_count??9)});
}
export function taskTime(seconds:number){
 if(!Number.isFinite(seconds))return tr("modules.phonation-synthesis.time_not_recorded");
 const date=new Date(seconds*1000);if(!Number.isFinite(date.getTime()))return tr("modules.phonation-synthesis.time_not_recorded");
 const two=(value:number)=>String(value).padStart(2,'0');
 return `${date.getFullYear()}/${two(date.getMonth()+1)}/${two(date.getDate())}，${two(date.getHours())}:${two(date.getMinutes())}:${two(date.getSeconds())}`;
}
