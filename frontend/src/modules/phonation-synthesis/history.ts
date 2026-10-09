import type {Metadata} from '../../platform/m07.ts';

export type TaskSnapshot=Partial<Pick<Metadata,'reverse_direction'|'continuum_type'|'generation'>> & Pick<Metadata,'action'>;
export const taskStates={queued:'排队中',running:'运行中',cancel_requested:'正在取消',cancelled:'已取消',succeeded:'已完成',failed:'失败',interrupted:'已中断'};
const changes={1:'仅发声类型变化',2:'仅 F0 变化',3:'F0 与发声类型同时变化'};
export function taskDescription(snapshot?:TaskSnapshot){
 if(!snapshot)return '发声连续统';
 if(snapshot.action==='analyze')return '提取 F0';
 if(snapshot.action==='apply')return '应用 F0 编辑';
 return `${snapshot.reverse_direction?'目标到源':'源到目标'} · ${changes[snapshot.continuum_type??2]} · ${snapshot.generation?.step_count??9} 步`;
}
export function taskTime(seconds:number){
 if(!Number.isFinite(seconds))return '时间未记录';
 const date=new Date(seconds*1000);if(!Number.isFinite(date.getTime()))return '时间未记录';
 const two=(value:number)=>String(value).padStart(2,'0');
 return `${date.getFullYear()}/${two(date.getMonth()+1)}/${two(date.getDate())}，${two(date.getHours())}:${two(date.getMinutes())}:${two(date.getSeconds())}`;
}
