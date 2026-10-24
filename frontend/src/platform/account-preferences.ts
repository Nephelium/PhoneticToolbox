import {ref} from 'vue';

type Values=Record<string,any>;
type View={module:string;revision:number;values:Values;updated_at:string};
type Snapshot={modules:View[];allowed_fields:Record<string,string[]>};
const clone=<T>(value:T):T=>JSON.parse(JSON.stringify(value));
const object=(value:any):value is Values=>!!value&&typeof value==='object'&&!Array.isArray(value);
export function mergePreferences(base:Values,patch:Values):Values{return Object.fromEntries([...new Set([...Object.keys(base),...Object.keys(patch)])].map(key=>[key,Object.hasOwn(patch,key)?object(base[key])&&object(patch[key])?mergePreferences(base[key],patch[key]):clone(patch[key]):clone(base[key])]));}
export function preferenceChanges(before:Values,after:Values):Values{
 const equivalent=(a:any,b:any)=>JSON.stringify(a)===JSON.stringify(b)||(typeof a==='number'&&typeof b==='string'&&!!b.trim()&&Number(b)===a)||(typeof b==='number'&&typeof a==='string'&&!!a.trim()&&Number(a)===b);
 const patch:Values={};for(const [key,value]of Object.entries(after)){if(equivalent(before[key],value))continue;const next=object(before[key])&&object(value)?preferenceChanges(before[key],value):value;if(!object(next)||Object.keys(next).length)patch[key]=clone(next);}return patch;
}
export const accountSettingsError=ref('');
export class AccountPreferences {
 readonly views=new Map<string,View>();readonly pending=new Map<string,Values>();allowed:Record<string,string[]>={};
 private controller=new AbortController();private timer:ReturnType<typeof setTimeout>|undefined;private flight:Promise<void>|undefined;private closed=false;
 readonly owner:string;
 constructor(owner:string,privateCsrf:()=>string,privateFetch:typeof fetch=globalThis.fetch){this.owner=owner;this.csrf=privateCsrf;this.transport=(input,init)=>privateFetch(input,init);}
 private csrf:()=>string;private transport:typeof fetch;
 private pendingKey(){return 'ptb.v3.settings-pending.v2.'+encodeURIComponent(this.owner);}
 private record(){try{if(this.pending.size)sessionStorage.setItem(this.pendingKey(),JSON.stringify(Object.fromEntries(this.pending)));else sessionStorage.removeItem(this.pendingKey());}catch{/* Server writes continue; a failed request remains visible. */}}
 private async request(path='',body?:unknown,keepalive=false){
  const raw=body===undefined?undefined:JSON.stringify(body);
  const response=await this.transport('/api/v1/preferences'+path,{method:body===undefined?'GET':'PATCH',credentials:'same-origin',cache:'no-store',signal:this.controller.signal,
   headers:{'X-PTB-Account':this.owner,...(raw?{'Content-Type':'application/json','X-CSRF-Token':this.csrf()}: {})},body:raw,keepalive:keepalive&&!!raw&&new TextEncoder().encode(raw).length<60000});
  if(this.closed)throw Error('settings_account_closed');
  if(!response.ok){const value=await response.json().catch(()=>({}));throw Error(value.detail??'settings_service_unavailable');}
  return response.json();
 }
 private hydrate(snapshot:Snapshot){
  if(!snapshot||!Array.isArray(snapshot.modules)||!object(snapshot.allowed_fields))throw Error('settings_service_unavailable');
  this.allowed=snapshot.allowed_fields;this.views.clear();
  for(const view of snapshot.modules)this.views.set(view.module,{...view,values:mergePreferences(view.values,this.pending.get(view.module)??{})});
  for(const [module,patch]of this.pending)if(!this.views.has(module))this.views.set(module,{module,revision:0,values:clone(patch),updated_at:''});
 }
 async open(){
  this.hydrate(await this.request());
  let staged:Values={};try{staged=JSON.parse(sessionStorage.getItem(this.pendingKey())||'{}');}catch{}
  // Old account-scoped browser settings migrate only when the server has no
  // saved module. Anonymous/global browser values are never assigned to an account.
  for(const module of Object.keys(this.allowed))if(!this.views.has(module)){
   let legacy:Values={};try{legacy=JSON.parse(localStorage.getItem('ptb.v3.web-settings.v1.'+encodeURIComponent(this.owner)+'.'+module)||'{}')??{};
    if(module==='M10'){const frame=JSON.parse(localStorage.getItem('ptb.v3.web-settings.v1.'+encodeURIComponent(this.owner)+'.M10.frame')||'null');if(frame)legacy.frame=frame;}
    if(module==='APP'){const fonts=JSON.parse(localStorage.getItem('ptb.v3.fonts.v1.owner.'+encodeURIComponent(this.owner))||'null');if(fonts)legacy.fonts=fonts;}
   }catch{legacy={};}
   const filtered=Object.fromEntries(Object.entries(legacy).filter(([key])=>this.allowed[module]!.includes(key)));
   if(Object.keys(filtered).length)this.write(module,filtered);
  }
  for(const [module,value]of Object.entries(staged))if(this.allowed[module]&&object(value))this.write(module,value);
  accountSettingsError.value='';if(this.pending.size)await this.flush();return this;
 }
 read(module:string):Values{return clone(this.views.get(module)?.values??{});}
 write(module:string,values:Values){
  if(this.closed||!this.allowed[module]||Object.keys(values).some(key=>!this.allowed[module]!.includes(key)))return false;
  const previous=this.read(module),patch=preferenceChanges(previous,values);if(!Object.keys(patch).length)return true;
  this.views.set(module,{...(this.views.get(module)??{module,revision:0,updated_at:''}),values:mergePreferences(previous,patch)});
  this.pending.set(module,mergePreferences(this.pending.get(module)??{},patch));this.record();clearTimeout(this.timer);
  this.timer=setTimeout(()=>void this.flush().catch(()=>{}),200);return true;
 }
 async flush(keepalive=false):Promise<void>{
  clearTimeout(this.timer);if(this.flight)return this.flight;if(this.closed||!this.pending.size)return;
  this.flight=(async()=>{try{
   while(this.pending.size&&!this.closed){
    const [module,patch]=[...this.pending][0]!,sent=clone(patch);let saved:View|undefined;
    for(let attempt=0;attempt<4;attempt++){
     try{saved=await this.request('/'+module,{expected_revision:this.views.get(module)?.revision??0,changes:sent},keepalive);break;}
     catch(error){if((error as Error).message!=='settings_revision_conflict'||attempt===3)throw error;this.hydrate(await this.request());}
    }
    if(!saved)throw Error('settings_service_unavailable');
    const remaining=preferenceChanges(sent,this.pending.get(module)??{});
    if(Object.keys(remaining).length)this.pending.set(module,remaining);else this.pending.delete(module);
    this.views.set(module,{...saved,values:mergePreferences(saved.values,remaining)});this.record();accountSettingsError.value='';
   }
  }catch(error){if(!this.closed){accountSettingsError.value='设置尚未保存到服务器，请检查网络并重试。';this.timer=setTimeout(()=>void this.flush().catch(()=>{}),5000);}throw error;}
  finally{this.flight=undefined;}})();return this.flight;
 }
 close(){this.closed=true;clearTimeout(this.timer);this.controller.abort();this.record();}
}
let active:AccountPreferences|undefined;
export async function openAccountPreferences(owner:string,csrf:()=>string){
 closeAccountPreferences();const session=new AccountPreferences(owner,csrf);active=session;
 try{await session.open();if(active!==session)throw Error('settings_account_closed');return session;}
 catch(error){if(active===session)closeAccountPreferences();throw error;}
}
export function closeAccountPreferences(){active?.close();active=undefined;accountSettingsError.value='';}
export async function flushAccountPreferences(keepalive=false){await active?.flush(keepalive);}
export function accountPreferenceTarget(key:string){
 if(!active)return null;
 const module=/^web-settings\.v1\.([^.]*)\.(M\d{2})$/.exec(key);
 if(module)return decodeURIComponent(module[1]!)===active.owner?{session:active,module:module[2]!,field:null}:null;
 const layout=/^layout\.panels\.server:([^:]+):([^:]+)$/.exec(key);
 if(layout&&layout[1]===active.owner)return {session:active,module:/^M\d{2}$/.test(layout[2]!)?layout[2]!:'APP',field:'panelWidths'};
 if(key==='fonts.v1.owner.'+encodeURIComponent(active.owner))return {session:active,module:'APP',field:'fonts'};
 if(['theme','palette','collapsed','buttonAppearance','waveformAppearance','pageScale','backgroundAppearance','customPaletteColor','homeIconScale','automaticAudioGain'].includes(key))return {session:active,module:'APP',field:key};
 return null;
}
