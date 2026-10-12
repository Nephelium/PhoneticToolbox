// Owned final EXE page via its explicitly enabled localhost Qt debug endpoint.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
const plan=JSON.parse(await fs.readFile(process.argv[2],'utf8'));
let target;
for(let n=0;n<120;n++){
 try{const list=await(await fetch(`http://127.0.0.1:${plan.port}/json`,{signal:AbortSignal.timeout(3000)})).json();target=list.find(p=>p.url.startsWith('ptbapp://app/'));if(target)break;}catch{}
 await new Promise(r=>setTimeout(r,1000));
}
assert(target,'Owned application page must start');
const endpoint=new URL(target.webSocketDebuggerUrl);assert.equal(endpoint.hostname,'127.0.0.1');assert.equal(endpoint.port,String(plan.port));
const socket=new WebSocket(endpoint),pending=new Map();let counter=0;
await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject;});
socket.onmessage=event=>{const r=JSON.parse(event.data);const job=pending.get(r.id);if(!job)return;pending.delete(r.id);clearTimeout(job.timer);r.error?job.reject(Error(JSON.stringify(r.error))):job.resolve(r.result);};
async function call(method,params){
 const id=++counter;return await new Promise((resolve,reject)=>{pending.set(id,{resolve,reject,timer:setTimeout(()=>{pending.delete(id);reject(Error('Qt evaluation timed out'));},15000)});socket.send(JSON.stringify({id,method,params}));});
}
async function evaluate(expression){
 const result=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true,userGesture:true});
 if(result.exceptionDetails)throw Error(JSON.stringify(result.exceptionDetails));return result.result.value;
}
async function until(expression){const end=Date.now()+60000;while(Date.now()<end){if(await evaluate(expression))return;await new Promise(r=>setTimeout(r,150));}throw Error(expression+'\n'+await evaluate('document.body.innerText.slice(-1800)'));}
async function click(selector){await until(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});if(!e||e.disabled||!e.offsetParent)return false;e.click();return true})()`);}
async function button(label){await until(`(()=>{const e=[...document.querySelectorAll('button')].find(e=>e.offsetParent&&e.textContent.trim()===${JSON.stringify(label)});if(!e||e.disabled)return false;e.click();return true})()`);}
async function select(selector,value){await evaluate(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});e.value=${JSON.stringify(value)};e.dispatchEvent(new Event('change',{bubbles:true}));})()`);}
const checks=[];
try{
 await click('.nav-item[title="语音学论文精读"]');
 await until('document.querySelectorAll(".paper-card").length===2');
 await until('[...document.querySelectorAll(".paper-card")].find(e=>e.textContent.includes("协同发音"))&&!document.querySelector(".paper-empty button")');
 await click('.paper-card');
 await until('document.querySelectorAll(".pdf-page").length===5&&!!document.querySelector(".pdf-text-line")');
 await button('中文译文');await until('document.querySelectorAll(".pdf-page").length===11&&!!document.querySelector(".pdf-text-line")');
 await button('论文解读');await until('document.querySelectorAll(".pdf-page").length===6&&!!document.querySelector(".pdf-text-line")');
 checks.push('Existing frozen EXE receives new catalogue and all three assets on application startup; no manual check needed; 5/11/6-page documents render');
 let pdf;
 if(plan.pdfMode){
  await button('我的 PDF 2');await until('document.querySelectorAll(".paper-card").length===2');
  await evaluate("[...document.querySelectorAll('.paper-card')].find(e=>e.textContent.includes('ExtIPA')).click();document.documentElement.dataset.theme='light'");
  await until('!!document.querySelector(".pdf-text-line")&&document.querySelector(".paper-sheet").complete');
  await evaluate('document.querySelector(".paper-viewport").scrollTop=document.querySelector(".pdf-page").offsetHeight*.43');
  await new Promise(r=>setTimeout(r,600));
  const capture=async name=>{const shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});const bytes=Buffer.from(shot.data,'base64');await fs.writeFile(path.join(path.dirname(plan.report),name+'.png'),bytes);return createHash('sha256').update(bytes).digest('hex');};
  const painted=await capture('extipa-with-text-layer');
  await evaluate("document.querySelector('.pdf-text-layer').style.visibility='hidden'");await new Promise(r=>setTimeout(r,300));
  const hidden=await capture('extipa-without-text-layer');
  await evaluate("document.querySelector('.pdf-text-layer').style.visibility=''");
  pdf={textLayerPaintIdentical:painted===hidden,initial:await evaluate("(()=>{const e=document.querySelector('.paper-sheet'),p=document.querySelector('.pdf-page');return {naturalWidth:e.naturalWidth,pageWidth:p.offsetWidth,dpr:devicePixelRatio}})()")};
  await select('[aria-label="阅读缩放"]','200');await new Promise(r=>setTimeout(r,900));
  pdf.zoom200=await evaluate("(()=>{const e=document.querySelector('.paper-sheet'),p=document.querySelector('.pdf-page');return {naturalWidth:e.naturalWidth,pageWidth:p.offsetWidth,dpr:devicePixelRatio}})()");
  await capture('extipa-200-percent');
  if(plan.pdfMode==='verify'){
   assert(pdf.textLayerPaintIdentical,'Selectable overlay must not paint extra glyphs');
   await until('document.querySelector(".paper-sheet").naturalWidth>=Math.ceil(document.querySelector(".pdf-page").offsetWidth*Math.max(1.25,devicePixelRatio))');
   const before=await evaluate('document.querySelector(".pdf-page").offsetWidth');
   const point=await evaluate("(()=>{const e=document.querySelector('.paper-viewport').getBoundingClientRect();return {x:e.left+e.width/2,y:e.top+e.height/2}})()");
   await call('Input.dispatchMouseEvent',{type:'mouseWheel',x:point.x,y:point.y,deltaX:0,deltaY:-120,modifiers:2});
   await until('document.querySelector(".pdf-page").offsetWidth>'+before);
   pdf.afterWheel=await evaluate('document.querySelector("[aria-label=阅读缩放]").value');
   assert.equal(await evaluate('document.documentElement.style.zoom'),'1');
   checks.push('Frozen extIPA selectable text paints no extra glyphs; 200% rerasterizes, actual Ctrl+wheel zooms PDF without changing application scale');
  }
 }
 if(plan.verifyDelete){
  const layout=`(()=>{const a=document.querySelector('.paper-library').getBoundingClientRect(),r=document.querySelector('.paper-reader').getBoundingClientRect();return [...document.querySelectorAll('.library-tools input,.library-tools select,.library-tools button')].every(e=>{const b=e.getBoundingClientRect();return b.width>0&&b.left>=a.left&&b.right<=a.right+1})&&a.right<=r.left+1&&document.querySelectorAll('.paper-card svg.paper-file-icon').length===document.querySelectorAll('.paper-card').length})()`;
  assert(await evaluate(layout));
  for(let n=0;n<3;n++){await button('我的 PDF 1');await until('document.querySelectorAll(".paper-card").length===1');assert(await evaluate(layout));await button('服务器推荐 2');await until('document.querySelectorAll(".paper-card").length===2');assert(await evaluate(layout));}
  await button('检查新论文');await until('document.body.innerText.includes("服务器目录共 2 篇，本次检查无需下载。")');
  await select('[aria-label="收藏夹筛选"]','协同发音');await until('document.querySelectorAll(".paper-card").length===1');
  await click('[aria-label="删除收藏夹"]');await button('取消');
  await until('!document.querySelector("dialog[open]")');assert.equal(await evaluate('document.querySelectorAll(".paper-card").length'),1);
  await click('[aria-label="删除收藏夹"]');await button('删除收藏夹并保留文件');
  // Both papers are uncollected now: the original TTS paper plus the removed
  // folder's new coarticulation paper. Check the exact resulting category.
  await until('document.body.innerText.includes("论文及批注保留")&&document.querySelectorAll(".paper-card").length===2&&document.querySelector("[aria-label=收藏夹筛选]").value===""');
  await until('!document.querySelector("dialog[open]")');
  assert(await evaluate('document.querySelector("[aria-label=删除收藏夹]").disabled'));
  await button('我的 PDF 1');await select('[aria-label="收藏夹筛选"]','构音研究');
  await until('document.querySelectorAll(".paper-card").length===1');
  await click('[aria-label="删除收藏夹"]');await button('删除收藏夹并保留文件');
  await until('document.querySelector("dialog[open]").textContent.includes("未移动文件")');
  await fs.unlink(path.join(plan.library,'说明.txt'));
  await button('删除收藏夹并保留文件');await until('!document.querySelector("dialog[open]")&&document.body.innerText.includes("已移到上一级")');
  assert(await evaluate('document.querySelector("[aria-label=删除收藏夹]").disabled'));
  await until('document.querySelectorAll(".pdf-page").length===5&&!!document.querySelector(".pdf-text-line")');
  const capture=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await fs.writeFile(path.join(path.dirname(plan.report),'frozen-paper-reading.png'),Buffer.from(capture.data,'base64'));
  checks.push('Frozen SVG file icons and sidebar controls stay within their column through three recommendation/personal source switches; actual viewport captured');
  checks.push('Frozen recommendation favourite cancellation/deletion preserves acquired files; real personal folder collision stops all moves, retry preserves PDF/non-PDF/nested folder and reader');
 }
 await fs.writeFile(plan.report,JSON.stringify({success:true,checks,pdf,deleteVerified:!!plan.verifyDelete},null,2));
}finally{
 socket.send(JSON.stringify({id:++counter,method:'Runtime.evaluate',params:{expression:'window.close()'}}));
 await new Promise(r=>setTimeout(r,1000));socket.close();
}
