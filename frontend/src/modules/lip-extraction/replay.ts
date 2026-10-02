import type {LipResult,LipRow,LipPort} from './port.ts';

export function atTime(rows:LipRow[],time:number){
 let lo=0,hi=rows.length;
 while(lo<hi){const mid=(lo+hi)>>>1;if(rows[mid].time_s<=time)lo=mid+1;else hi=mid;}
 return Math.max(0,lo-1);
}

/** Full frame playback uses at most three 90-frame pages, independently of plot samples. */
export class LipReplay {
 private pages=new Map<number,{rows:LipRow[];complete:boolean}>();
 private pending=new Map<number,Promise<{rows:LipRow[];complete:boolean}>>();
 private result:LipResult;
 private port:LipPort;
 constructor(result:LipResult,port:LipPort){this.result=result;this.port=port;}
 get count(){return this.result.metadata.timing?.decoded_frames??this.result.rows.length;}
 private async page(start:number){
  const existing=this.pages.get(start);if(existing)return existing;
  const pending=this.pending.get(start);if(pending)return pending;
  const request=this.port.replay!(this.result,start).then(value=>{
   this.pages.set(start,value);while(this.pages.size>3)this.pages.delete(this.pages.keys().next().value!);return value;
  }).finally(()=>this.pending.delete(start));
  this.pending.set(start,request);return request;
 }
 async frame(index:number){
  if(!this.port.replay)return this.result.rows[index];
  const start=Math.floor(Math.max(0,index)/90)*90,value=await this.page(start);
  return value.rows.find(r=>r.index===index);
 }
 async time(time:number):Promise<LipRow|undefined>{
  if(!this.port.replay)return this.result.rows[atTime(this.result.rows,time)];
  for(const value of this.pages.values())if(value.rows.length&&time>=value.rows[0].time_s&&(time<value.rows.at(-1)!.time_s||value.complete))return value.rows[atTime(value.rows,time)];
  const preview=this.result.rows,lower=atTime(preview,time),a=preview[lower],b=preview[lower+1];
  if(!a)return;
  const estimate=b?Math.floor(a.index+(b.index-a.index)*Math.max(0,Math.min(1,(time-a.time_s)/(b.time_s-a.time_s)))):a.index;
  let start=Math.floor(estimate/90)*90;
  for(let attempt=0;attempt<Math.ceil(this.count/90)+1;attempt++){
   const value=await this.page(start);if(!value.rows.length)return;
   if(time<value.rows[0].time_s&&start>0){start-=90;continue;}
   if(time>value.rows.at(-1)!.time_s&&!value.complete){
    const next=await this.page(start+90);
    if(next.rows.length&&time>=next.rows[0].time_s){start+=90;continue;}
   }
   return value.rows[atTime(value.rows,time)];
  }
 }
 prefetch(index:number){
  if(!this.port.replay)return;
  const next=(Math.floor(index/90)+1)*90;
  if(next<this.count)void this.page(next).catch(()=>{});
 }
}
