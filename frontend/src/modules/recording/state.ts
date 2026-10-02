import type {Task,Take} from './types.ts';
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
