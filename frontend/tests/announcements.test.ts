import test from 'node:test';
import assert from 'node:assert/strict';
import {AnnouncementService,announcementsUrl,parseAnnouncements,announcementActionUrl,announcementPlatform} from '../src/platform/announcements.ts';
const builtinAnnouncement={id:'ipa-ime-0-1-1-launched',title:'国际音标输入法上线啦',body:'现在可以在官网下载国际音标输入法，支持 Windows、macOS、iOS 和 Android。\n打开下载页查看各平台的安装方式与测试版说明。',publishedAt:'2026-10-24T00:00:00Z',action:{label:'去官网下载',url:'https://www.phonetictoolbox.com/#ipa-ime-download'}};
const now=()=>Date.parse('2026-10-24T15:00:00Z');
const item=(patch:Record<string,unknown>={})=>({...builtinAnnouncement,...patch});
const manifest=(items:unknown[]=[item()])=>({schema:'ptb-announcements/1',items});
class MemoryStorage {values=new Map<string,string>();getItem(k:string){return this.values.get(k)??null;}setItem(k:string,v:string){this.values.set(k,v);}removeItem(k:string){this.values.delete(k);}}
const response=(value:unknown)=>new Response(JSON.stringify(value),{headers:{'Content-Type':'application/json; charset=utf-8'}});
const service=(options:Record<string,any>={})=>new AnnouncementService({version:'3.1.0',platform:'windows',now,storage:new MemoryStorage(),fetch:async()=>response(manifest()),...options});

test('manifest rejects invalid schema, duplicate/unsafe IDs, limits, malformed dates, versions and platforms',()=>{
 for(const value of [{schema:'wrong',items:[]},manifest(Array.from({length:21},(_,i)=>item({id:'id-'+i}))),manifest([item(),item()]),...[
  {id:'../secret'},{id:'id.with.dot'},{id:''},{id:'a'.repeat(81)},{title:' '},{title:'a'.repeat(161)},{body:'a'.repeat(8001)},{body:'bad\u0000'},
  {publishedAt:'2026-02-30T00:00:00Z'},{publishedAt:'2026-10-24'},{publishedAt:'2026-10-24T00:00:00+00:00'},{expiresAt:'2026-10-23T00:00:00Z'},
  {minVersion:'3.1'},{minVersion:'v3.1.0'},{minVersion:'3.01.0'},{minVersion:'3.1.0-beta'},{minVersion:'4.0.0',maxVersion:'3.1.0'},{platforms:[]},{platforms:['windows','windows']},{platforms:['android']},
  {action:{label:' ',url:'https://www.phonetictoolbox.com/'}},{action:{label:'open',url:'javascript:alert(1)'}}
 ].map(patch=>manifest([item(patch)]))])assert.throws(()=>parseAnnouncements(value));
 assert.equal(parseAnnouncements(manifest([item({publishedAt:'2026-10-24T00:00:00.001Z',expiresAt:'2026-10-24T00:00:01Z'})])).items.length,1);
});
test('actions accept HTTPS official informational links and reject credentials, ports, scripts and install files',()=>{
 assert.equal(announcementActionUrl(builtinAnnouncement.action!.url),builtinAnnouncement.action!.url);
 for(const url of ['http://www.phonetictoolbox.com/','https://www.phonetictoolbox.com.evil.test/','https://phonetictoolbox.com/','https://user:pass@www.phonetictoolbox.com/','https://www.phonetictoolbox.com:444/','https://www.phonetictoolbox.com:443/','https://www.phonetictoolbox.com/?redirect=evil','javascript:alert(1)','file:///private','data:text/html,evil','https://www.phonetictoolbox.com/a.exe','https://www.phonetictoolbox.com/a.%7Aip','https://www.phonetictoolbox.com/a.apk','https://www.phonetictoolbox.com/a\\b'])assert.equal(announcementActionUrl(url),undefined,url);
 assert.equal(announcementPlatform(false,'Windows'),'web');assert.equal(announcementPlatform(true,'Macintosh'),'macos');assert.equal(announcementPlatform(true,'Windows'),'windows');
});
test('one request across repeated load/claims omits identity, cookies and referrer; same ID updates do not re-prompt',async()=>{
 const storage=new MemoryStorage();let calls=0;
 const s=service({storage,fetch:async(url:string,init:RequestInit)=>{calls++;assert.equal(url,announcementsUrl);assert.equal(init.method,'GET');assert.equal(init.credentials,'omit');assert.equal(init.redirect,'error');assert.equal(init.referrerPolicy,'no-referrer');assert.equal(init.cache,'no-store');assert.equal(init.body,undefined);return response(manifest());}});
 try{await Promise.all([s.load(),s.load(),s.load()]);assert.equal(calls,1);const claimed=await s.claimNext();assert.equal(claimed?.id,builtinAnnouncement.id);assert.equal(s.isRead(claimed!.id),false);assert.equal(await s.claimNext(),undefined);s.acknowledge([claimed!.id]);assert.equal(await s.claimNext(),undefined);assert.equal(calls,1);}finally{s.dispose();}
 const reopened=service({storage,fetch:async()=>response(manifest([item({body:'Updated text'})]))});
 try{await reopened.load();assert.equal(reopened.items[0]?.body,'Updated text');assert.equal(await reopened.claimNext(),undefined);assert.equal(reopened.items.length,1);}finally{reopened.dispose();}
});
test('successful remote manifest is authoritative, including withdrawal; first offline has no bundled announcement',async()=>{
 const storage=new MemoryStorage(),s=service({storage,fetch:async()=>response(manifest([]))});await s.load();assert.deepEqual(s.items,[]);s.dispose();
 const offline=service({storage,fetch:async()=>{throw Error('offline');}});try{await offline.load();assert.deepEqual(offline.items,[]);}finally{offline.dispose();}
 const first=service({fetch:async()=>{throw Error('first startup offline');}});try{await first.load();assert.deepEqual(first.items,[]);}finally{first.dispose();}
 const override=service({fetch:async()=>response(manifest([item({title:'Remote title'})]))});try{await override.load();assert.equal(override.items.length,1);assert.equal(override.items[0]?.title,'Remote title');}finally{override.dispose();}
});
test('new ID, publication, expiration, version and platform filtering are independent of project',async()=>{
 const s=service({fetch:async()=>response(manifest([item({id:'new-id'}),item({id:'future',publishedAt:'2027-01-01T00:00:00Z'}),item({id:'expired',expiresAt:'2026-10-24T10:00:00Z'}),item({id:'old-version',maxVersion:'3.0.9'}),item({id:'later-version',minVersion:'3.2.0'}),item({id:'only-mac',platforms:['macos']}),item({id:'inclusive',minVersion:'3.1.0',maxVersion:'3.1.0',platforms:['windows']})]))});
 try{await s.load();assert.deepEqual(s.items.map(i=>i.id),['new-id','inclusive']);s.acknowledge(['new-id']);assert.equal((await s.claimNext())?.id,'inclusive');}finally{s.dispose();}
});
test('bounded download rejects oversized header/stream, bad UTF8, HTML, redirects and invalid payload silently',async()=>{
 const responses=[()=>new Response('{}',{headers:{'Content-Type':'text/html'}}),()=>new Response('{}',{headers:{'Content-Type':'application/json','Content-Length':'65537'}}),()=>new Response(' '.repeat(65537),{headers:{'Content-Type':'application/json'}}),()=>new Response(new Uint8Array([0xff]),{headers:{'Content-Type':'application/json'}}),()=>response(manifest([item({action:{label:'bad',url:'https://evil.test/'}})])),()=>new Response('{bad',{headers:{'Content-Type':'application/json'}}),()=>Response.error()];
 for(const make of responses){const s=service({fetch:async()=>make()});try{await s.load();assert.equal(s.ready,true);assert.deepEqual(s.items,[]);}finally{s.dispose();}}
});
test('timeout aborts the single owned request and disposal cancels without publishing fallback',async()=>{
 let signal:AbortSignal|undefined,calls=0;const s=service({timeoutMs:15,fetch:async(_url:any,init:RequestInit)=>{calls++;signal=init.signal!;return new Promise(()=>{});}});
 try{await s.load();assert.equal(signal?.aborted,true);assert.deepEqual(s.items,[]);await s.load();assert.equal(calls,1);}finally{s.dispose();}
 let cancelled=false;const stopped=service({fetch:(_u:any,init:RequestInit)=>new Promise((_r,reject)=>{init.signal!.addEventListener('abort',()=>{cancelled=true;reject(Error('cancelled'));});})});const flight=stopped.load();stopped.dispose();await flight;assert.equal(cancelled,true);assert.deepEqual(stopped.items,[]);
});
test('device lease prevents overlapping automatic dialogs across instances and releases unacknowledged claims',async()=>{
 const storage=new MemoryStorage(),a=service({storage}),b=service({storage});
 try{await Promise.all([a.load(),b.load()]);const [one,two]=await Promise.all([a.claimNext(),b.claimNext()]);assert.equal([one,two].filter(Boolean).length,1);const owner=one?a:b,other=one?b:a;owner.release();assert.equal((await other.claimNext())?.id,builtinAnnouncement.id);other.acknowledge([builtinAnnouncement.id]);assert.equal(await owner.claimNext(),undefined);}finally{a.dispose();b.dispose();}
});
test('unavailable, readonly or non-persisting storage never claims automatically; center remains readable',async()=>{
 for(const storage of [undefined,{getItem(){throw Error('denied');},setItem(){throw Error('denied');},removeItem(){}},{getItem(){return null;},setItem(){},removeItem(){}}]){
  const s=service({storage});try{await s.load();assert.equal(s.items.length,1);assert.equal(await s.claimNext(),undefined);assert.equal(s.automaticAvailable,false);s.acknowledge([builtinAnnouncement.id]);assert.equal(s.isRead(builtinAnnouncement.id),true);}finally{s.dispose();}
 }
});
