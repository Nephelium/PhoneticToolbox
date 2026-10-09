import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
const file=path.resolve(process.argv[2]),plan=JSON.parse(await fs.readFile(file,'utf8'));
const cachePreparations=[];
async function observeCache(){
 try{const row=JSON.parse(await fs.readFile(path.join(plan.profile,'PhoneticToolbox/v3/startup-cache/last-launch.json'),'utf8'));if(!cachePreparations.some(v=>JSON.stringify(v)===JSON.stringify(row)))cachePreparations.push(row);}catch{}
}
let status;const deadline=Date.now()+600000;
while(Date.now()<deadline){
  await observeCache();
  try{status=JSON.parse(await fs.readFile(path.join(path.dirname(plan.request),'status.json'),'utf8'));}catch{}
  if(status?.state==='failed')throw Error(JSON.stringify(status));
  if(status?.state==='started')break;
  await new Promise(r=>setTimeout(r,1000));
}
assert.equal(status?.state,'started');
// Qt exposes page CDP targets, without Chrome's Browser-level target endpoint.
let target,lastError;
for(let n=0;n<120;n++){
  await observeCache();
  try{const list=await (await fetch(`http://127.0.0.1:${plan.port}/json`,{signal:AbortSignal.timeout(3000)})).json();target=list.find(p=>p.url.startsWith('ptbapp://app/'));if(target)break;}catch(e){lastError=e;}
  await new Promise(r=>setTimeout(r,1000));
}
assert(target,`New real process must expose its owned Qt page endpoint: ${lastError}`);
const endpoint=new URL(target.webSocketDebuggerUrl);assert.equal(endpoint.hostname,'127.0.0.1');assert.equal(endpoint.port,String(plan.port));
const socket=new WebSocket(endpoint),pending=new Map();let counter=0;
await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject;});
socket.onmessage=event=>{const r=JSON.parse(event.data);const item=pending.get(r.id);if(!item)return;pending.delete(r.id);clearTimeout(item.timer);r.error?item.reject(Error(JSON.stringify(r.error))):item.resolve(r.result);};
async function evaluate(expression){
 const id=++counter;const result=await new Promise((resolve,reject)=>{pending.set(id,{resolve,reject,timer:setTimeout(()=>{pending.delete(id);reject(Error('Owned Qt evaluation timed out'));},15000)});socket.send(JSON.stringify({id,method:'Runtime.evaluate',params:{expression,returnByValue:true,awaitPromise:true,userGesture:true}}));});
 if(result.exceptionDetails)throw Error(JSON.stringify(result.exceptionDetails));return result.result.value;
}
async function until(expression){const end=Date.now()+60000;while(Date.now()<end){if(await evaluate(expression))return;await new Promise(r=>setTimeout(r,150));}throw Error(`Owned Qt condition failed: ${expression}`);}
async function click(selector){await until(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});if(!e||e.disabled||!e.offsetParent)return false;e.click();return true})()`);}
async function button(name){await until(`(()=>{const e=[...document.querySelectorAll('button')].find(e=>e.offsetParent&&[${JSON.stringify(name)},${JSON.stringify(name+' *')}].includes(e.textContent.trim()));if(!e||e.disabled)return false;e.click();return true})()`);}
await button('首页');await until('!!document.querySelector(".home-page")');
if(plan.seedFresh){
 await button('设置');await button('深色');
 await until('document.documentElement.dataset.theme==="dark"');
 await click('.nav-item[title="汉字转国际音标"]');
 await until('!!document.querySelector("[aria-label=待转换汉字文本]")');
 await evaluate('(()=>{const e=document.querySelector("[aria-label=待转换汉字文本]");e.value="更新保存测试";e.dispatchEvent(new Event("input",{bubbles:true}));})()');
 await button('保存本机草稿');
 plan.preferences=await evaluate('Object.fromEntries(Object.keys(localStorage).filter(k=>k.startsWith("ptb.v3.")).map(k=>[k,localStorage.getItem(k)]))');
 await button('首页');await until('!!document.querySelector(".home-page")');
}
const theme=await evaluate('document.documentElement.dataset.theme');assert.equal(theme,'dark');
const now=await evaluate('Object.fromEntries(Object.keys(localStorage).filter(k=>k.startsWith("ptb.v3.")).map(k=>[k,localStorage.getItem(k)]))');
// Recents/tab bookkeeping may change at startup. Verify settings and the saved
// module draft from actual persistence, excluding only that UI bookkeeping.
const stable=Object.keys(plan.preferences).filter(k=>!/(recent|tabs|active|session|last)/i.test(k));
for(const key of stable)assert.equal(now[key],plan.preferences[key],key);
await click('.nav-item[title="汉字转国际音标"]');
await button('恢复本机草稿');
assert.equal(await evaluate('document.querySelector("[aria-label=待转换汉字文本]").value'),'更新保存测试');
const marker=await fs.readFile(plan.project_marker);assert.equal(createHash('sha256').update(marker).digest('hex'),plan.project_sha256);
// Check genuine software-only media through the actual frozen Qt scheme. Audio
// is muted by the owned QA process; successful decoding is not human listening.
await click('.nav-item[title="变速变调"]');
await button('帮助');
await until('!!document.querySelector(".manual-document[data-manual-chapter=m08]")');
const media=await evaluate(`(async()=>{
  const project=await (await fetch('./manual/project.json')).json();
  const descriptor=project.chapters.find(c=>c.id==='m08');
  const chapter=await (await fetch('./manual/'+descriptor.path)).json();
  const expected=[];const walk=n=>{if(n.type==='audio')expected.push(n.attrs.assetId);for(const c of n.content??[])walk(c);};walk(chapter.body);
  const players=[...document.querySelectorAll('.manual-document audio')];
  if(players.length!==expected.length)throw Error('Frozen chapter audio count mismatch');
  for(let i=0;i<players.length;i++){
    const asset=project.assets.find(a=>a.id===expected[i]);
    if(players[i].src!==new URL('./manual/'+asset.path,location.href).href)throw Error('Frozen chapter audio identity mismatch');
    await players[i].play();await new Promise(r=>setTimeout(r,100));
    if(players[i].readyState<2||!Number.isFinite(players[i].duration))throw Error('Audio decoding failed: '+expected[i]);
    players[i].pause();
  }
  const first=players[0],second=players.find(p=>p.src!==first?.src)??players[1];
  let exclusive=null;
  if(first)await first.play();
  if(second){await second.play();await new Promise(r=>setTimeout(r,200));if(!first.paused||second.paused)throw Error('Manual audio playback exclusivity failed');exclusive=true;}
  window.__ptbOwnedQAPlaying=second??first;
  return {players:players.length,firstDuration:first?.duration,secondDuration:second?.duration,exclusive,images:document.querySelectorAll('.manual-image-button').length};
})()`);
if(media.images){
 await click('.manual-image-button');
 await until('!!document.querySelector(".manual-image-overlay[role=dialog]")');
 await button('关闭图片');
}
await click('.nav-item[title="国际音标表Plus"]');
assert.equal(await evaluate('window.__ptbOwnedQAPlaying?.paused??true'),true);
await observeCache();
const report={success:true,scope:plan.scope,kind:plan.kind,newPid:status.pid,newTheme:theme,stableKeys:stable,restoredDraft:true,projectMarkerUnchanged:true,cachePreparations,...(plan.launchMode==='direct'?{launchState:status.state}:{helperState:status.state}),manualMedia:media,imageEnlargement:media.images>0?true:null,chapterDepartureStopsPlayback:media.players>0?true:null};
await fs.writeFile(path.join(plan.output,'relaunch-report.json'),JSON.stringify(report,null,2)+'\n','utf8');
console.log(JSON.stringify(report));
if(plan.closeAfter){
 socket.send(JSON.stringify({id:++counter,method:'Runtime.evaluate',params:{expression:'window.close()'}}));
 await new Promise(r=>setTimeout(r,2000));
}
socket.close();
