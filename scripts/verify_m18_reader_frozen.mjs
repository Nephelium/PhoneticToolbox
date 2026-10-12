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
const shot=async name=>{await new Promise(r=>setTimeout(r,450));const v=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await fs.writeFile(path.join(path.dirname(plan.report),name+'.png'),Buffer.from(v.data,'base64'));};
try{
 await click('.nav-item[title="语音学论文精读"]');
 await until('!!document.querySelector(".paper-intro")');await click('.paper-intro');
 await until('document.querySelectorAll(".pdf-page").length===1&&document.querySelector(".paper-sheet")?.naturalWidth>0');
 assert.equal(await evaluate('document.querySelector(".auto-switch input").checked'),false);
 assert.equal(await evaluate('[...document.querySelectorAll(".paper-card")].filter(e=>!e.classList.contains("paper-intro")&&e.textContent.includes("协同发音"))[0]?.querySelector("small")?.textContent'),'↓');
 await shot('frozen-introduction');
 checks.push('Actual relocated EXE receives notice at startup with paper automatic acquisition disabled; new scientific paper stays undownloaded');
 await button('检查新论文');
 await until('document.body.innerText.includes("已获取 1 篇新论文")');
 await evaluate('[...document.querySelectorAll(".paper-card")].find(e=>e.textContent.includes("协同发音")).click()');
 await until('document.querySelectorAll(".pdf-page").length===5&&document.querySelector(".paper-sheet")?.naturalWidth>0');
 await button('中文译文');await until('document.querySelectorAll(".pdf-page").length===5&&document.querySelector(".paper-sheet")?.naturalWidth>0');
 await button('论文解读');await until('document.querySelectorAll(".pdf-page").length===6&&document.querySelector(".paper-sheet")?.naturalWidth>0');
 await button('原文');await until('document.querySelectorAll(".pdf-page").length===5');
 await click('.pdf-toolbar button[aria-controls]');await until('!!document.querySelector(".pdf-notes")');
 await button('添加本页笔记');
 await evaluate('(()=>{const e=document.querySelector("[aria-label=批注内容]");e.value="成品本页笔记";e.dispatchEvent(new Event("input",{bubbles:true}))})()');
 await button('保存批注');await until('!document.querySelector("dialog[open]")&&document.querySelector(".pdf-note-card p")?.textContent==="成品本页笔记"');
 await shot('frozen-page-note');
 await button('引用格式');await until('!!document.querySelector(".paper-citation")');
 assert(await evaluate('document.querySelector(".paper-citation").value.startsWith("FAN C, HENDERSON J M, MANNING C, et al.")'));
 assert(await evaluate('getComputedStyle(document.querySelector(".paper-citation")).fontFamily===getComputedStyle(document.querySelector("dialog[open]")).fontFamily'));
 await shot('frozen-citation');await button('关闭');
 checks.push('Actual frozen Qt renders 5/5/6 pages, creates and persists page note, citation uses app font and correct author order');
 await button('中文译文');await until('document.querySelectorAll(".pdf-page").length===5');
 await evaluate('[...document.querySelectorAll(".paper-card")].find(e=>e.textContent.includes("协同发音")).dispatchEvent(new MouseEvent("contextmenu",{bubbles:true,clientX:180,clientY:400}))');
 await click('[role=menuitem]');await button('删除本机副本');
 await until('document.querySelectorAll(".paper-card").length===2&&document.querySelectorAll(".pdf-page").length===1');
 await button('检查新论文');await until('document.body.innerText.includes("本次检查无需下载")');
 assert.equal(await evaluate('document.querySelectorAll(".paper-card").length'),2);
 await button('获取往期');await until('document.querySelectorAll(".archive-row:not(.obtained)").length===1&&!document.querySelector(".archive-row input").disabled');
 await click('.archive-row input');await button('获取已选 1 篇');await until('document.querySelectorAll(".archive-row:not(.obtained)").length===0');await button('关闭');
 await evaluate('[...document.querySelectorAll(".paper-card")].find(e=>e.textContent.includes("协同发音")).click()');
 await until('document.querySelectorAll(".pdf-page").length===5');await click('.pdf-toolbar button[aria-controls]');
 await until('document.querySelector(".pdf-note-card p")?.textContent==="成品本页笔记"');
 checks.push('Actual EXE deletes local recommendation, skips subsequent checks, explicitly reacquires from server and retains original note; deletion from translation returns to notice');
 await shot('frozen-reacquired');
 await fs.writeFile(plan.report,JSON.stringify({success:true,checks},null,2));
}finally{
 socket.send(JSON.stringify({id:++counter,method:'Runtime.evaluate',params:{expression:'window.close()'}}));
 await new Promise(r=>setTimeout(r,1000));socket.close();
}
