import type {Task,Take,Role,Project} from './types.ts';
export function projectSelection(project:Project|null,taskId:string,takeId:string,restoreLatest=false){
  if(!project)return {taskId:'',takeId:''};
  if(restoreLatest){const last=project.takes.at(-1);return {taskId:project.tasks.some(t=>t.id===last?.task_snapshot?.id)?last!.task_snapshot!.id:'',takeId:last?.id??''};}
  const current=project.takes.find(t=>t.id===takeId);
  if(current&&!taskId&&current.task_snapshot&&!project.tasks.some(t=>t.id===current.task_snapshot!.id))return {taskId,takeId};
  const matches=(take:Take)=>taskId?take.task_snapshot?.id===taskId:!take.task_snapshot;
  const candidates=project.takes.filter(matches),chosen=candidates.find(t=>t.id===takeId)??candidates.find(t=>t.id===project.selected[taskId||'__free__'])??candidates.at(-1);
  return {taskId,takeId:chosen?.id??''};
}
export function zoomTimeRange(range:[number,number],total:number,ratio:number,delta:number):[number,number]{
  const [start,end]=range,width=end-start,pivot=Math.max(0,Math.min(1,ratio));
  if(total<=0||width<=0||!Number.isFinite(delta)||!delta)return [...range];
  const length=Math.max(Math.min(64,total),Math.min(total,Math.round(width*(delta>0?1.25:.8))));
  const left=Math.max(0,Math.min(total-length,Math.round(start+width*pivot-length*pivot)));
  return [left,left+length];
}
export function channelName(index:number,roles:Role[]){
  const position=roles.length===1?'单声道':roles.length===2?(index===0?'左声道':'右声道'):`通道 ${index+1}`;
  return position+(roles[index]==='egg'?' · EGG':roles[index]==='other'?' · 其他':'');
}
export function newTask(index:number):Task{return {id:`T${String(index).padStart(3,'0')}`,title:'',prompt:'',filename_stem:`recording-${String(index).padStart(3,'0')}`,group:'',note:'',enabled:true,skipped:false};}
export function nextTask(tasks:Task[],id:string){const at=tasks.findIndex(t=>t.id===id);return tasks.slice(at+1).find(t=>t.enabled&&!t.skipped)?.id??'';}
export function normalizeSelection(a:number,b:number,total:number){const left=Math.max(0,Math.min(total,Math.round(Math.min(a,b)))),right=Math.max(left,Math.min(total,Math.round(Math.max(a,b))));return [left,right] as [number,number];}
export function setSelectionEndpoint(selection:[number,number],endpoint:0|1,value:number,total:number):[number,number]{
  if(!Number.isFinite(value))return [...selection];
  const frame=Math.max(0,Math.min(total,Math.round(value)));
  return endpoint===0?[frame,Math.max(frame,selection[1])]:[Math.min(selection[0],frame),frame];
}
export function currentFrames(take:Take|null){return take?.versions[take.head]?.frames??0;}
export function formatTime(frames:number,rate:number){return (frames/Math.max(rate,1)).toFixed(3)+' s';}
export function taskName(task:Task){return task.title||task.prompt.split('\n')[0]||task.id;}
