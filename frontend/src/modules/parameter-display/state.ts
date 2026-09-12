import type { ParameterTable, ResearchFile } from '../../platform/research.ts';

export interface PlotGroup { id:number; title:string; parameters:string[] }
export function matchingTable(audio:ResearchFile,files:ResearchFile[]):ResearchFile|undefined {
  const stem=audio.name.replace(/\.wav$/i,'').toLowerCase();
  for(const suffix of ['.ptb.sqlite','.ptb.sqlite3','.xlsx']) {
    const found=files.filter(f=>f.kind==='parameter'&&f.name.toLowerCase()===stem+suffix);
    if(found.length===1)return found[0];
    if(found.length>1)return undefined;
  }
}
export function visibleParameters(names:string[],query:string,reaper:boolean,correction:boolean) {
  return names.filter(name=>name.toLowerCase().includes(query.trim().toLowerCase())&&(reaper||!name.toLowerCase().includes('rf0'))&&(correction||!name.includes('*')));
}
export function assignParameters(groups:PlotGroup[],selected:string[],target:number):PlotGroup[] {
  if(!groups.some(group=>group.id===target))throw Error('图窗不存在。');
  const unique=[...new Set(selected)];
  return groups.map(group=>({...group,parameters:[...group.parameters.filter(name=>!unique.includes(name)),...(group.id===target?unique:[])]}));
}
export function removeGroup(groups:PlotGroup[],id:number):PlotGroup[] {
  if(groups.length===1)return groups;
  const removed=groups.find(group=>group.id===id),remaining=groups.filter(group=>group.id!==id);
  if(!removed)return groups;
  remaining[0]={...remaining[0],parameters:[...new Set([...remaining[0].parameters,...removed.parameters])]};
  return remaining;
}
export interface Point {time:number;value:number|null}

export interface OverlayScale {min:number;max:number}
export interface OverlayCurve {name:string;axis:'left'|'right';points:Point[];mean:number}
// v2 _plot: cluster only when all means are nonzero and max/min > 50;
// values with mean absolute magnitude >100 then use the right axis.
// Statistics use original visible frames, never decimated screen vertices.
export function overlayPlot(table:ParameterTable,names:string[],start:number,end:number,width=900) {
  const ti=table.columns.indexOf('Time_s');
  const stats=names.filter(name=>table.kinds[table.columns.indexOf(name)]==='number').map(name=>{
    const ci=table.columns.indexOf(name);let count=0,sum=0,min=Infinity,max=-Infinity;
    for(const row of table.rows){const t=row[ti],v=row[ci];if(typeof t!=='number'||t<start||t>end||typeof v!=='number'||!Number.isFinite(v))continue;count++;sum+=Math.abs(v);min=Math.min(min,v);max=Math.max(max,v);}
    return {name,mean:count?sum/count:0,min,max,count};
  });
  const means=stats.map(s=>s.mean),smallest=means.length?Math.min(...means):0,largest=means.length?Math.max(...means):0;
  const dual=smallest>0&&largest/smallest>50;
  const curves:OverlayCurve[]=stats.map(s=>({name:s.name,mean:s.mean,axis:dual&&s.mean>100?'right':'left',points:series(table,s.name,start,end,width)}));
  const scale=(axis:'left'|'right'):OverlayScale=>{let min=Infinity,max=-Infinity;stats.forEach((s,i)=>{if(s.count&&curves[i].axis===axis){min=Math.min(min,s.min);max=Math.max(max,s.max);}});if(!Number.isFinite(min))return {min:0,max:1};const pad=min===max?Math.max(.5,Math.abs(min)*.05):(max-min)*.05;return {min:min-pad,max:max+pad};};
  return {curves,left:scale('left'),right:scale('right'),dual:curves.some(c=>c.axis==='right')};
}
// Preserve gaps explicitly, use original time coordinates, retain pixel minima/maxima.
export function series(table:ParameterTable,name:string,start:number,end:number,width=900):Point[] {
  const ti=table.columns.indexOf('Time_s'),ci=table.columns.indexOf(name);
  if(ti<0||ci<0||!(end>start))return [];
  const points:Point[]=[];let bucket:Point[]=[];let pixel=-1;
  const flush=()=>{if(!bucket.length)return;let lo=0,hi=0;for(let i=1;i<bucket.length;i++){if(bucket[i].value!<bucket[lo].value!)lo=i;if(bucket[i].value!>bucket[hi].value!)hi=i;}for(const i of [...new Set([0,lo,hi,bucket.length-1])].sort((a,b)=>a-b))points.push(bucket[i]);bucket=[];};
  for(const row of table.rows){const time=row[ti] as number;if(time<start||time>end)continue;const raw=row[ci];const value=typeof raw==='number'&&Number.isFinite(raw)?raw:null;
    if(value===null){flush();if(points.at(-1)?.value!==null)points.push({time,value:null});continue;}
    const next=Math.floor((time-start)/(end-start)*width);if(next!==pixel){flush();pixel=next;}bucket.push({time,value});}
  flush();return points;
}
export function annotationRuns(table:ParameterTable,name:string,start:number,end:number) {
  const ti=table.columns.indexOf('Time_s'),ci=table.columns.indexOf(name);if(ci<0||ti<0)return [];
  const runs:{start:number;end:number;text:string}[]=[];
  for(let i=0;i<table.rows.length;i++){const row=table.rows[i],left=row[ti] as number,right=(table.rows[i+1]?.[ti] as number|undefined)??end;if(left>end)break;if(right<start)continue;
    const text=String(row[ci]??'');const last=runs.at(-1);if(last&&last.text===text&&last.end===left)last.end=Math.min(end,right);else runs.push({start:Math.max(start,left),end:Math.min(end,right),text});}
  return runs;
}
