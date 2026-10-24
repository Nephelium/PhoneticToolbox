/** Public, device-local notices. No account, project, telemetry or upload data. */
export const announcementsUrl='https://www.phonetictoolbox.com/announcements/v1.json';
export type AnnouncementPlatform='windows'|'macos'|'linux'|'web';
export interface Announcement {id:string;title:string;body:string;publishedAt:string;expiresAt?:string;minVersion?:string;maxVersion?:string;platforms?:AnnouncementPlatform[];action?:{label:string;url:string}}
export interface AnnouncementManifest {schema:'ptb-announcements/1';items:Announcement[]}
const limit=65_536,seenKey='ptb.v3.announcements.seen.v1',cacheKey='ptb.v3.announcements.feed.v1',leaseKey='ptb.v3.announcements.presenting.v1';
const idPattern=/^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$/;
const versionPattern=/^(0|[1-9]\d{0,3})\.(0|[1-9]\d{0,3})\.(0|[1-9]\d{0,3})$/;
const object=(v:unknown):v is Record<string,unknown>=>!!v&&typeof v==='object'&&!Array.isArray(v);
const text=(v:unknown,max:number)=>typeof v==='string'&&v.trim().length>0&&v.length<=max&&!/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/.test(v);
function date(v:unknown):v is string {
 if(typeof v!=='string'||!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?Z$/.test(v))return false;
 const value=new Date(v);return Number.isFinite(value.valueOf())&&value.toISOString()===v.replace(/(?<!\.\d{3})Z$/,'.000Z');
}
export function compareAnnouncementVersions(a:string,b:string){const x=a.split('.').map(Number),y=b.split('.').map(Number);return x[0]!-y[0]!||x[1]!-y[1]!||x[2]!-y[2]!;}
export function announcementActionUrl(value:unknown):string|undefined {
 if(typeof value!=='string'||value.length>2048||/[\s\\]/.test(value)||!/^https:\/\/www\.phonetictoolbox\.com(?:[/?#]|$)/i.test(value))return;
 try{const url=new URL(value),path=decodeURIComponent(url.pathname);
  if(url.protocol!=='https:'||url.hostname!=='www.phonetictoolbox.com'||url.port||url.username||url.password||url.search||/[\\\u0000-\u001f]/.test(path)||/\.(?:exe|msi|msix|zip|7z|dmg|pkg|app|apk|ipa|mobileconfig|bat|cmd|ps1|sh)(?:\/|$)/i.test(path))return;
  return url.href;
 }catch{return;}
}
export function parseAnnouncements(value:unknown):AnnouncementManifest {
 if(!object(value)||value.schema!=='ptb-announcements/1'||!Array.isArray(value.items)||value.items.length>20)throw Error('Invalid announcements manifest');
 const ids=new Set<string>(),items:Announcement[]=[];
 for(const raw of value.items){
  if(!object(raw)||typeof raw.id!=='string'||!idPattern.test(raw.id)||ids.has(raw.id)||!text(raw.title,160)||!text(raw.body,8000)||!date(raw.publishedAt))throw Error('Invalid announcement');
  const item:Announcement={id:raw.id,title:raw.title as string,body:raw.body as string,publishedAt:raw.publishedAt};ids.add(item.id);
  if(raw.expiresAt!==undefined){if(!date(raw.expiresAt)||Date.parse(raw.expiresAt)<=Date.parse(item.publishedAt))throw Error('Invalid announcement expiry');item.expiresAt=raw.expiresAt;}
  for(const field of ['minVersion','maxVersion'] as const)if(raw[field]!==undefined){if(typeof raw[field]!=='string'||!versionPattern.test(raw[field]))throw Error('Invalid announcement version');item[field]=raw[field];}
  if(item.minVersion&&item.maxVersion&&compareAnnouncementVersions(item.minVersion,item.maxVersion)>0)throw Error('Invalid announcement version range');
  if(raw.platforms!==undefined){if(!Array.isArray(raw.platforms)||!raw.platforms.length||raw.platforms.length>4||raw.platforms.some(p=>!['windows','macos','linux','web'].includes(p))||new Set(raw.platforms).size!==raw.platforms.length)throw Error('Invalid announcement platforms');item.platforms=[...raw.platforms];}
  if(raw.action!==undefined){if(!object(raw.action)||!text(raw.action.label,80))throw Error('Invalid announcement action');const url=announcementActionUrl(raw.action.url);if(!url)throw Error('Invalid announcement URL');item.action={label:raw.action.label as string,url};}
  items.push(item);
 }
 return {schema:'ptb-announcements/1',items};
}
export function announcementPlatform(desktop:boolean,agent:string):AnnouncementPlatform {return !desktop?'web':/Macintosh|Mac OS X/.test(agent)?'macos':/Windows/.test(agent)?'windows':'linux';}
interface Options {version:string;platform:AnnouncementPlatform;storage?:Pick<Storage,'getItem'|'setItem'|'removeItem'>;fetch?:typeof fetch;now?:()=>number;locks?:LockManager;timeoutMs?:number}
export class AnnouncementService {
 private transport:typeof fetch;private storage:Options['storage'];private now:()=>number;private options:Options;
 private request:Promise<void>|undefined;private controller=new AbortController();private listeners=new Set<()=>void>();private seen=new Set<string>();private feed:Announcement[]=[];private stopped=false;private persistenceAvailable=true;
 private owner=globalThis.crypto?.randomUUID?.()??Math.random().toString(36).slice(2);private claimed:string|undefined;private pending=false;private epoch=0;private releaseLock:(()=>void)|undefined;private leaseTimer:ReturnType<typeof setInterval>|undefined;
 ready=false;
 constructor(options:Options){
  this.options=options;this.storage=options.storage;this.transport=options.fetch??globalThis.fetch.bind(globalThis);this.now=options.now??Date.now;
  this.readSeen();const cached=this.read(cacheKey);
  if(cached!==null){try{if(cached.length<=limit&&new TextEncoder().encode(cached).length<=limit)this.feed=parseAnnouncements(JSON.parse(cached)).items;}catch{/* Invalid cache is silent. No production announcement is bundled. */}}
 }
 private read(key:string){try{return this.storage?.getItem(key)??null;}catch{return null;}}
 private write(key:string,value:string){try{this.storage?.setItem(key,value);return !!this.storage;}catch{return false;}}
 private readSeen(){try{const raw=this.read(seenKey);if(raw&&raw.length<=200_000){const ids=JSON.parse(raw);if(Array.isArray(ids))for(const id of ids)if(typeof id==='string'&&idPattern.test(id))this.seen.add(id);}}catch{}}
 private notify(){for(const listener of this.listeners)listener();}
 subscribe(listener:()=>void){this.listeners.add(listener);return ()=>{this.listeners.delete(listener);};}
 get items(){
  const now=this.now();return this.feed.filter(item=>Date.parse(item.publishedAt)<=now&&(!item.expiresAt||Date.parse(item.expiresAt)>now)&&(!item.minVersion||compareAnnouncementVersions(this.options.version,item.minVersion)>=0)&&(!item.maxVersion||compareAnnouncementVersions(this.options.version,item.maxVersion)<=0)&&(!item.platforms||item.platforms.includes(this.options.platform))).sort((a,b)=>Date.parse(b.publishedAt)-Date.parse(a.publishedAt));
 }
 isRead(id:string){this.readSeen();return this.seen.has(id);}
 get automaticAvailable(){return !!this.storage&&this.persistenceAvailable&&!this.stopped;}
 private persistSeen(){
  const value=JSON.stringify([...this.seen]);if(!this.write(seenKey,value)||this.read(seenKey)!==value){this.persistenceAvailable=false;this.notify();return false;}return true;
 }
 async load(){
  if(this.request)return this.request;
  this.request=(async()=>{
   let timer:ReturnType<typeof setTimeout>|undefined;
   try{
    const deadline=new Promise<never>((_,reject)=>{timer=setTimeout(()=>{this.controller.abort();reject(Error('Announcement timeout'));},this.options.timeoutMs??3000);});
    const manifest=await Promise.race([this.download(),deadline]);if(this.stopped)return;
    this.feed=manifest.items;this.write(cacheKey,JSON.stringify(manifest));
   }catch{/* Offline, invalid or timed-out feeds retain only a prior successful cache. */}
   finally{clearTimeout(timer);this.ready=true;if(!this.stopped)this.notify();}
  })();return this.request;
 }
 private async download(){
  const response=await this.transport(announcementsUrl,{method:'GET',credentials:'omit',referrerPolicy:'no-referrer',cache:'no-store',redirect:'error',signal:this.controller.signal});
  if(!response.ok||response.redirected||!/^application\/(?:[\w.+-]*\+)?json(?:;|$)/i.test(response.headers.get('content-type')??''))throw Error('Invalid announcement response');
  const length=response.headers.get('content-length');if(length!==null&&(!/^\d+$/.test(length)||Number(length)>limit))throw Error('Announcement too large');
  if(!response.body)throw Error('Missing announcement body');
  const reader=response.body.getReader(),chunks:Uint8Array[]=[];let size=0;
  try{while(true){this.controller.signal.throwIfAborted();const part=await reader.read();if(part.done)break;size+=part.value.byteLength;if(size>limit)throw Error('Announcement too large');chunks.push(part.value);}}
  catch(error){void reader.cancel().catch(()=>{});throw error;}finally{reader.releaseLock();}
  const raw=new Uint8Array(size);let offset=0;for(const part of chunks){raw.set(part,offset);offset+=part.byteLength;}
  return parseAnnouncements(JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(raw)));
 }
 acknowledge(ids:string[]){for(const id of ids)if(idPattern.test(id))this.seen.add(id);this.readSeen();this.persistSeen();if(this.claimed&&ids.includes(this.claimed))this.release();this.notify();}
 private acquireLease(){
  const raw=this.read(leaseKey);try{const lease=raw?JSON.parse(raw):null;if(lease&&lease.owner!==this.owner&&typeof lease.until==='number'&&lease.until>this.now()&&lease.until<this.now()+60_000)return false;}catch{}
  if(!this.write(leaseKey,JSON.stringify({owner:this.owner,until:this.now()+15_000}))){this.persistenceAvailable=false;this.notify();return false;}
  const confirm=this.read(leaseKey);try{if(!confirm||JSON.parse(confirm).owner!==this.owner){this.persistenceAvailable=false;this.notify();return false;}}catch{this.persistenceAvailable=false;this.notify();return false;}
  this.leaseTimer=setInterval(()=>this.write(leaseKey,JSON.stringify({owner:this.owner,until:this.now()+15_000})),5000);return true;
 }
 async claimNext():Promise<Announcement|undefined>{
  if(this.pending||this.claimed||!this.automaticAvailable)return;this.pending=true;const epoch=this.epoch;
  try{
   await this.load();this.readSeen();if(this.stopped||epoch!==this.epoch||!this.persistSeen())return;
   const claim=()=>{const item=this.items.find(item=>!this.isRead(item.id));if(!item||!this.acquireLease())return;this.claimed=item.id;return item;};
   if(!this.options.locks)return claim();
   return await new Promise<Announcement|undefined>(resolve=>{
    void this.options.locks!.request('ptb-announcements-auto',{ifAvailable:true},async lock=>{
     if(!lock||this.stopped||epoch!==this.epoch){resolve(undefined);return;}
     const item=claim();if(!item){resolve(undefined);return;}
     await new Promise<void>(release=>{this.releaseLock=release;resolve(item);});
    }).catch(()=>resolve(undefined));
   });
  }finally{this.pending=false;}
 }
 release(){
  this.epoch++;this.claimed=undefined;clearInterval(this.leaseTimer);this.leaseTimer=undefined;this.releaseLock?.();this.releaseLock=undefined;
  try{const raw=this.read(leaseKey);if(raw&&JSON.parse(raw).owner===this.owner)this.storage?.removeItem(leaseKey);}catch{}
 }
 dispose(){this.stopped=true;this.controller.abort();this.release();this.listeners.clear();}
}
let singleton:AnnouncementService|undefined;
export function announcements(version:string,platform:AnnouncementPlatform){
 if(!singleton){let storage:Storage|undefined;try{storage=localStorage;}catch{}
  singleton=new AnnouncementService({version,platform,storage,locks:globalThis.navigator?.locks});
  globalThis.addEventListener?.('pagehide',(event:PageTransitionEvent)=>{if(!event.persisted)singleton?.dispose();});
 }
 return singleton;
}
