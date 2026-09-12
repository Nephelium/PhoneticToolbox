export interface TimedInterval {xmin:number;xmax:number;text:string}
export function intervalLayout(intervals:TimedInterval[],start:number,end:number){
  if(!Number.isFinite(start)||!Number.isFinite(end)||end<=start)return [];
  return intervals.flatMap((interval,index)=>{
    if(!Number.isFinite(interval.xmin)||!Number.isFinite(interval.xmax)||interval.xmax<=interval.xmin)return [];
    const left=Math.max(start,interval.xmin),right=Math.min(end,interval.xmax);
    return right>left?[{...interval,index,left:(left-start)/(end-start)*100,width:(right-left)/(end-start)*100}]:[];
  });
}
