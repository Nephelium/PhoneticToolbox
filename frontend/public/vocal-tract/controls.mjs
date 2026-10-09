export const ORGAN_PARAMS={tongue:['TCX','TCY','TTX','TTY','TBX','TBY','TRX','TRY','HX','HY','TS1','TS2','TS3'],lips:['LP','LD','JA'],velum:['VO','VS','HX','HY']};
export function bladeSurfacePoint(state){
  // Native fractional station follows the blade segment through re-sampling
  // and hard-tissue contact. Interpolate the actual meridian, never TBX/TBY.
  const points=state.contours.tongue;
  const rib=Math.max(0,Math.min(32,state.tongue_blade_rib??21));
  const lo=Math.floor(rib),hi=Math.ceil(rib),t=rib-lo;
  return [0,1].map(k=>points[lo][k]*(1-t)+points[hi][k]*t);
}
export function moveControl(meta,baseline,name,dx,dy){
  const next=[...baseline];let limited=false;
  const get=k=>baseline[meta.parameters.findIndex(p=>p.name===k)],set=(key,value)=>{const i=meta.parameters.findIndex(p=>p.name===key),p=meta.parameters[i],v=Math.max(p.min,Math.min(p.max,value));limited ||= v!==value;next[i]=v;};
  // The flexible blade midpoint has 5/8 influence from its two interior
  // controls. Compensate that gain when dragging the surface attachment.
  if(name==='blade'){dx*=1.6;dy*=1.6;}
  for(const [handle,x,y] of [['tongue','TCX','TCY'],['tip','TTX','TTY'],['blade','TBX','TBY'],['root','TRX','TRY'],['hyoid','HX','HY']])if(name===handle){set(x,get(x)+dx);set(y,get(y)+dy);}
  if(name.includes('lip')){set('LP',get('LP')+dx*.65);set('LD',get('LD')+dy*(name==='upper_lip'?2:-2));}
  if(name==='velum'){set('VO',get('VO')-dy*.6);set('VS',get('VS')+dx*.2);}
  if(name.startsWith('side')&&dy!==0){
    const key='TS'+name.slice(-1);
    if(name==='side3'){
      // Invert the native elevation plus M10 lateral relief so the front edge
      // follows centimetres of pointer motion through its saturation regions.
      const p=get(key),height=p<-.15?-.5+(p+.15)*.6/.85:p<=.3?p/.3:1+(p-.3)*.3;
      const h=height+dy;set(key,h<-.5?-.15+(h+.5)*.85/.6:h<=1?h*.3:.3+(h-1)/.3);
    }else set(key,get(key)+dy*.6);
  }
  if(meta.posterior_limits&&['tongue','root'].includes(name)){
    const w=meta.posterior_limits,back=y=>w.x+(y-w.y)*w.slope;
    const index=k=>meta.parameters.findIndex(p=>p.name===k);
    if(name==='tongue')set('TCX',Math.max(next[index('TCX')],back(next[index('TCY')])+Math.hypot(w.rx,w.slope*w.ry)));
    else set('TRX',Math.max(next[index('TRX')],back(next[index('TRY')])));
  }
  return {params:next,limited};
}
const equal=(a,b)=>(a.larynx_height??0)===(b.larynx_height??0)&&(a.manual_root??false)===(b.manual_root??false)&&a.preset===b.preset&&(a.lip_width??1)===(b.lip_width??1)&&(a.f0??150)===(b.f0??150)&&JSON.stringify(a.source||null)===JSON.stringify(b.source||null)&&a.params.every((v,i)=>Math.abs(v-b.params[i])<1e-8);
const copy=s=>({...s,params:[...s.params],...(s.source?{source:{...s.source}}:{})});
export class PoseHistory{
  constructor(limit=80){this.limit=limit;this.past=[];this.future=[];this.start=null;}
  begin(state){if(!this.start)this.start=copy(state);}
  commit(state){if(this.start&&!equal(this.start,state)){this.past.push(this.start);this.past=this.past.slice(-this.limit);this.future=[];}this.start=null;}
  undo(state){this.commit(state);if(!this.past.length)return null;this.future.push(copy(state));return this.past.pop();}
  redo(state){this.commit(state);if(!this.future.length)return null;this.past.push(copy(state));return this.future.pop();}
}
