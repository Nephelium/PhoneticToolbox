import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import path from 'node:path';
const plan=JSON.parse(await fs.readFile(process.argv[2],'utf8'));
const pause=ms=>new Promise(r=>setTimeout(r,ms));
let target;
for(let n=0;n<90;n++){try{const list=await(await fetch('http://127.0.0.1:'+plan.port+'/json')).json();target=list.find(v=>v.url==='ptbapp://app/feedback-manager.html');if(target)break;}catch{}await pause(500);}
assert(target,'Frozen Qt page not available');
const socket=new WebSocket(target.webSocketDebuggerUrl),pending=new Map();let sequence=0;
await new Promise((r,j)=>{socket.onopen=r;socket.onerror=j;});
socket.onmessage=e=>{const v=JSON.parse(e.data),job=pending.get(v.id);if(!job)return;pending.delete(v.id);clearTimeout(job.timer);v.error?job.reject(Error(JSON.stringify(v.error))):job.resolve(v.result);};
async function cdp(method,params={}){const id=++sequence;return await new Promise((resolve,reject)=>{pending.set(id,{resolve,reject,timer:setTimeout(()=>{pending.delete(id);reject(Error(method+' timed out'));},15000)});socket.send(JSON.stringify({id,method,params}));});}
async function evaluate(expression){const v=await cdp('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true,userGesture:true});if(v.exceptionDetails)throw Error(JSON.stringify(v.exceptionDetails));return v.result.value;}
async function until(expression){const end=Date.now()+25000;while(Date.now()<end){if(await evaluate(expression))return;await pause(100);}throw Error('UI condition failed: '+expression+' '+await evaluate('document.body.innerText.slice(-1000)'));}
async function button(text){await until(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.offsetParent&&b.textContent.trim()===${JSON.stringify(text)});if(!b||b.disabled)return false;b.click();return true;})()`);}
async function fill(value){await evaluate(`(()=>{const e=document.querySelector('.thread-compose textarea');e.value=${JSON.stringify(value)};e.dispatchEvent(new Event('input',{bubbles:true}));})()`);}
async function control(operation,body={}){const r=await fetch(plan.control+'/'+operation,{method:'POST',body:JSON.stringify(body)});assert.equal(r.status,200);return r.json();}
async function shot(name){const result=await cdp('Page.captureScreenshot',{format:'png'});await fs.writeFile(path.join(plan.out,name+'.png'),Buffer.from(result.data,'base64'));}
await until('!!document.querySelector(".conversation-item")');
await evaluate('document.querySelector(".conversation-item").click()');
await until('document.querySelector(".thread-timeline")?.textContent.includes("Windows fixture feedback")');
await until('document.querySelector(".manager-author img")?.naturalWidth>0');
assert.equal(await evaluate('document.querySelector(".manager-author span").textContent'),'井韶子');
await shot('windows-light');
await fill('Frozen Windows reply');await button('发送回复');
await until('document.querySelector(".thread-timeline")?.textContent.includes("Frozen Windows reply")');
assert((await control('received')).text==='Frozen Windows reply');
await until('document.querySelector(".message-avatar.author-avatar")?.naturalWidth>0');
assert((await evaluate('document.querySelector(".thread-timeline").innerText')).includes('井韶子'));
const document=await cdp('DOM.getDocument');const input=await cdp('DOM.querySelector',{nodeId:document.root.nodeId,selector:'.image-picker input'});
await cdp('DOM.setFileInputFiles',{nodeId:input.nodeId,files:[plan.image]});
await until('!!document.querySelector(".draft-images img")');await fill('Frozen image reply');await button('发送回复');
await until('document.querySelector(".thread-timeline")?.textContent.includes("Frozen image reply")');
assert.equal((await control('received')).images,1);
await fill('Persistent frozen draft');await pause(300);
await evaluate('document.querySelector("button[aria-label=切换深色主题]").click()');await shot('windows-dark');
await cdp('Page.reload');await until('!!document.querySelector(".conversation-item")');await evaluate('document.querySelector(".conversation-item").click()');
await until('document.querySelector("textarea")?.value==="Persistent frozen draft"');assert.equal(await evaluate('document.documentElement.dataset.theme'),'dark');
await control('new-feedback');await until('document.querySelector(".thread-timeline")?.textContent.includes("Immediate server event")');
await button('设置');await button('生成配对码');await until('!!document.querySelector(".pairing-display strong")');
const code=await evaluate('document.querySelector(".pairing-display strong").textContent');assert.match(code,/^[0-9]{8}$/);assert.equal((await control('pair',{code})).paired,true);
await evaluate('document.querySelector("button[aria-label=关闭对话框]").click()');
await cdp('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});await pause(200);
assert.equal(await evaluate('getComputedStyle(document.querySelector(".conversation-sidebar")).display'),'none');
assert(await evaluate('document.documentElement.scrollWidth<=innerWidth'));
await shot('windows-mobile-layout');
await evaluate('document.querySelector(".back-button").click()');assert.equal(await evaluate('getComputedStyle(document.querySelector(".manager-conversation")).display'),'none');
await cdp('Emulation.clearDeviceMetricsOverride');
const windowTest=await control('window-lifecycle');assert(windowTest.trayHidden&&windowTest.reactivated&&windowTest.secondInstanceExited);
await button('设置');await button('退出工具');socket.close();
const report={success:true,finalExe:true,httpsFixture:true,reply:true,imageReply:true,draftRestart:true,themeRestart:true,instantSse:true,pairing:true,mobileLayout:true,...windowTest};
await fs.writeFile(path.join(plan.out,'windows-ui.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));
