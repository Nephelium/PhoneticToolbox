export const ORGAN_PARAMS={tongue:['TCX','TCY','TTX','TTY','TBX','TBY','TS1','TS2','TS3'],lips:['LP','LD','JA'],velum:['VO','VS','HX','HY']};
export function moveControl(meta,baseline,name,dx,dy){
  const next=[...baseline];let limited=false;
  const get=k=>baseline[meta.parameters.findIndex(p=>p.name===k)],set=(key,value)=>{const i=meta.parameters.findIndex(p=>p.name===key),p=meta.parameters[i],v=Math.max(p.min,Math.min(p.max,value));limited ||= v!==value;next[i]=v;};
  for(const [handle,x,y] of [['tongue','TCX','TCY'],['tip','TTX','TTY'],['blade','TBX','TBY']])if(name===handle){set(x,get(x)+dx);set(y,get(y)+dy);}
  if(name.includes('lip')){set('LP',get('LP')+dx*.65);set('LD',get('LD')+dy*(name==='upper_lip'?2:-2));}
  if(name==='velum'){set('VO',get('VO')-dy*.6);set('VS',get('VS')+dx*.2);}
  if(name.startsWith('side')&&dy!==0){
    const key='TS'+name.slice(-1),height=Math.max(-.15,Math.min(.3,get(key)));
    // VTL saturates side elevation above 0.3 (and below -0.15). Dragging
    // edits visible elevation; the numerical slider retains bracing controls.
    const desired=height+dy*.3,value=Math.max(-.15,Math.min(.3,desired));
    limited ||= value!==desired;set(key,value);
  }
  return {params:next,limited};
}
const equal=(a,b)=>a.preset===b.preset&&(a.lip_width??1)===(b.lip_width??1)&&(a.f0??125)===(b.f0??125)&&JSON.stringify(a.source||null)===JSON.stringify(b.source||null)&&a.params.every((v,i)=>Math.abs(v-b.params[i])<1e-8);
const copy=s=>({...s,params:[...s.params],...(s.source?{source:{...s.source}}:{})});
export class PoseHistory{
  constructor(limit=80){this.limit=limit;this.past=[];this.future=[];this.start=null;}
  begin(state){if(!this.start)this.start=copy(state);}
  commit(state){if(this.start&&!equal(this.start,state)){this.past.push(this.start);this.past=this.past.slice(-this.limit);this.future=[];}this.start=null;}
  undo(state){this.commit(state);if(!this.past.length)return null;this.future.push(copy(state));return this.past.pop();}
  redo(state){this.commit(state);if(!this.future.length)return null;this.past.push(copy(state));return this.future.pop();}
}
