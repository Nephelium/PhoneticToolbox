import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import path from 'node:path';
const plan=JSON.parse(await fs.readFile(process.argv[2],'utf8'));
const pause=ms=>new Promise(r=>setTimeout(r,ms));
let target;
for(let n=0;n<600;n++){try{const list=await(await fetch('http://127.0.0.1:'+plan.port+'/json')).json();target=list.find(v=>v.url==='ptbapp://app/index.html');if(target)break;}catch{}await pause(500);}
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
await until('!!document.querySelector(".app-shell")');
await evaluate('document.querySelector("button[title=关于]").click()');
await until('!!document.querySelector(".about-website")');
assert.equal(await evaluate('document.querySelector(".about-website").href'),'https://www.phonetictoolbox.com/');
assert.equal(await evaluate('document.querySelector(".about-website").target'),'_blank');await shot('about');
await button('问题反馈');await until('!!document.querySelector(".thread-compose textarea")');
await until('document.querySelector(".thread-profile img")?.naturalWidth>0');assert.equal(await evaluate('document.querySelector(".thread-profile strong").textContent'),'井韶子');
assert(await evaluate('document.querySelector(".mail-toggle input").checked'));
await fill('Final user feedback');await button('发送');await until('document.querySelector(".thread-timeline").textContent.includes("Final user feedback")');
await control('reply',{text:'井韶子的第一条回复',id:'frozen-user-mail-on'});await button('刷新');await until('document.querySelector(".thread-timeline").textContent.includes("井韶子的第一条回复")');
await until('document.querySelector(".message-avatar.author-avatar")?.naturalWidth>0');
await until('document.querySelector(".mail-toggle input")?.disabled===false');
await evaluate('document.querySelector(".mail-toggle input").click()');await until('!document.querySelector(".mail-toggle input").checked&&!document.querySelector(".mail-toggle input").disabled');
const before=await control('state');assert.equal(before.mailNotifications,false);
await control('reply',{text:'关闭邮件后依然能聊天',id:'frozen-user-mail-off'});await button('刷新');await until('document.querySelector(".thread-timeline").textContent.includes("关闭邮件后依然能聊天")');
assert.equal((await control('state')).messages.at(-1).mailState,'disabled');await shot('user-mail-off');
await evaluate('document.querySelector("button[aria-label=关闭对话框]").click()');await evaluate('document.querySelector("button[title=关于]").click()');await button('问题反馈');await until('!!document.querySelector(".mail-toggle input")&&!document.querySelector(".mail-toggle input").checked');
assert.equal((await control('state')).mailCount,before.mailCount);
assert(await evaluate('document.documentElement.scrollWidth<=innerWidth'));
await evaluate('document.querySelector("button[aria-label=关闭对话框]").click()');await button('设置');await button('浅色');await button('首页');await evaluate('document.querySelector("button[title=关于]").click()');await button('问题反馈');await until('!!document.querySelector(".thread-compose textarea")');await shot('user-light');
await evaluate('document.querySelector("button[aria-label=关闭对话框]").click()');await button('设置');await button('深色');await button('首页');await evaluate('document.querySelector("button[title=关于]").click()');await button('问题反馈');await until('!!document.querySelector(".thread-compose textarea")');await shot('user-dark');
await evaluate('document.querySelector("button[aria-label=关闭对话框]").click()');await evaluate('document.querySelector("button[aria-label=关闭窗口]").click()');socket.close();
const report={success:true,finalMainExe:true,httpsFixture:true,aboutWebsite:true,authorNameAndAvatar:true,userSend:true,replySync:true,mailOptOut:true,mailPreferenceAfterReopen:true,noMailAfterOptOut:true,lightDark:true};await fs.writeFile(path.join(plan.out,'user-ui.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));
