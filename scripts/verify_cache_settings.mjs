// Real frozen Qt cache controls with an isolated profile and private CDP port.
import {spawn} from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import net from 'node:net';
import assert from 'node:assert/strict';
const [exeArg,outArg,profileArg]=process.argv.slice(2);
const exe=path.resolve(exeArg),out=path.resolve(outArg),profile=path.resolve(profileArg??path.join(out,'profile'));
await fs.mkdir(out,{recursive:false});await fs.mkdir(profile,{recursive:true});
const cache=path.join(profile,'PhoneticToolbox/v3/startup-cache');
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
async function exists(p){try{await fs.access(p);return true;}catch{return false;}}
async function launch(label){
 const run=path.join(out,label);await fs.mkdir(run);await fs.mkdir(path.join(run,'temp'));
 const server=net.createServer();await new Promise(r=>server.listen(0,'127.0.0.1',r));const port=server.address().port;await new Promise(r=>server.close(r));
 const env={...process.env};for(const key of Object.keys(env))if(/^(PTB_|PYTHON|CONDA|_PYI)/.test(key)||['VIRTUAL_ENV','QT_PLUGIN_PATH','QT_QPA_PLATFORM_PLUGIN_PATH','QTWEBENGINEPROCESS_PATH','QTWEBENGINE_RESOURCES_PATH','QTWEBENGINE_LOCALES_PATH'].includes(key))delete env[key];
 Object.assign(env,{LOCALAPPDATA:profile,TEMP:path.join(run,'temp'),TMP:path.join(run,'temp'),PATH:`${process.env.SystemRoot}\\System32;${process.env.SystemRoot}`,QT_QPA_PLATFORM:'offscreen',QTWEBENGINE_REMOTE_DEBUGGING:`127.0.0.1:${port}`,QTWEBENGINE_CHROMIUM_FLAGS:'--mute-audio --disable-gpu --remote-allow-origins=*',PTB_OWNED_BOOTSTRAP_LOG:path.join(run,'bootstrap-error.log')});
 const updateRoot=path.join(profile,'PhoneticToolbox/v3/updates');await fs.mkdir(updateRoot,{recursive:true});await fs.writeFile(path.join(updateRoot,'state.json'),'{"autoCheck":false}');
 const child=spawn(exe,[],{cwd:run,env,windowsHide:true,stdio:['ignore','pipe','pipe']});current={child};const logs=[];child.stdout.on('data',b=>logs.push(b));child.stderr.on('data',b=>logs.push(b));
 const exit=new Promise(resolve=>child.once('exit',(code,signal)=>resolve({code,signal})));
 let target;for(let i=0;i<1200&&child.exitCode===null;i++){
  try{const tabs=await(await fetch(`http://127.0.0.1:${port}/json/list`,{signal:AbortSignal.timeout(200)})).json();target=tabs.find(t=>t.type==='page'&&t.url.startsWith('ptbapp://app/'));if(target)break;}catch{}await sleep(100);
 }
 if(!target)throw Error('Frozen Qt page failed to start: '+label);
 const socket=new WebSocket(target.webSocketDebuggerUrl);await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',reject,{once:true});});
 let next=0;const pending=new Map();socket.addEventListener('message',({data})=>{const v=JSON.parse(data);if(pending.has(v.id)){pending.get(v.id)(v);pending.delete(v.id);}});
 async function command(method,params){const id=++next;const promise=new Promise(resolve=>pending.set(id,resolve));socket.send(JSON.stringify({id,method,params}));const answer=await Promise.race([promise,sleep(10000).then(()=>{throw Error('Qt command timed out: '+method);})]);if(answer.error)throw Error(JSON.stringify(answer.error));return answer.result;}
 async function js(expression){const answer=await command('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true,userGesture:true});if(answer.exceptionDetails)throw Error(JSON.stringify(answer.exceptionDetails));return answer.result.value;}
 async function until(expression){for(let i=0;i<300;i++){if(await js(expression))return;await sleep(100);}throw Error(expression+'\n'+await js('document.body.innerText.slice(-1800)'));}
 async function button(text){await until(`(()=>{const e=[...document.querySelectorAll('button')].find(e=>e.offsetParent&&[${JSON.stringify(text)},${JSON.stringify(text+' *')}].includes(e.textContent.trim()));if(!e||e.disabled)return false;e.click();return true})()`);}
 async function screenshot(name){await fs.writeFile(path.join(out,name+'.png'),Buffer.from((await command('Page.captureScreenshot',{format:'png'})).data,'base64'));}
 async function finish(){socket.close();const result=await Promise.race([exit,sleep(60000).then(()=>null)]);assert.equal(result?.code,0);await fs.writeFile(path.join(run,'process.log'),Buffer.concat(logs));assert(!(await fs.readdir(path.join(run,'temp'))).some(n=>n.startsWith('_MEI')));return result;}
 return {child,js,until,button,screenshot,finish};
}
const report={success:false,scope:'real frozen Qt offscreen; scripted UI, private profile and owned cache; no physical devices',checks:[]};
let current;
try{
 current=await launch('controls');const {js,until,button,screenshot}=current;
 await until('!!document.querySelector(".home-page")');
 const marker=path.join(profile,'PhoneticToolbox/v3/research-marker.wav');await fs.writeFile(marker,'PRESERVE RESEARCH');
 const download=path.join(profile,'PhoneticToolbox/v3/updates/downloads/'+'d'.repeat(32));await fs.mkdir(download,{recursive:true});await fs.writeFile(path.join(download,'package.zip'),'OWNED UNUSED DOWNLOAD');
 await button('设置');await until('!!document.querySelector(".all-caches")');
 await button('浅色');await js('document.querySelector(".all-caches").scrollIntoView({block:"center"})');await sleep(300);await screenshot('cache-settings-light');
 await button('深色');await js('document.querySelector(".all-caches").scrollIntoView({block:"center"})');await sleep(300);await screenshot('cache-settings-dark');
 await button('清除全部缓存…');await until('!!document.querySelector("[aria-label=确认清除全部缓存]")');await button('取消');assert(await exists(cache));report.checks.push('explicit cleanup confirmation cancelled without deleting the cache');
 await until('(()=>{const e=document.querySelector(".nav-item[title=汉字转国际音标]");if(!e)return false;e.click();return true})()');await until('!!document.querySelector("[aria-label=待转换汉字文本]")');
 await js('(()=>{const e=document.querySelector("[aria-label=待转换汉字文本]");e.value="清理保留草稿";e.dispatchEvent(new Event("input",{bubbles:true}));})()');
 await button('设置');await button('清除全部缓存…');await button('清除全部缓存并退出');
 await until('document.body.innerText.includes("保存转换草稿？")');await button('取消关闭');
 assert(await exists(cache));assert(!await exists(path.join(cache,'clear-request.json')));report.checks.push('dirty-module cancellation preserves cache and editing');
 assert.equal(await js('document.querySelector("[aria-label=待转换汉字文本]").value'),'清理保留草稿');await button('保存本机草稿');
 await button('设置');await button('清除全部缓存…');await button('清除全部缓存并退出');
 await current.finish();current=null;
 assert(!await exists(cache));assert(!await exists(download));assert.equal(await fs.readFile(marker,'utf8'),'PRESERVE RESEARCH');report.checks.push('accepted safe exit removed persistent runtime and update cache, preserving research marker');
 current=await launch('reopen');await current.until('!!document.querySelector(".home-page")');
 assert.equal(await current.js('document.documentElement.dataset.theme'),'dark');
 await current.until('(()=>{const e=document.querySelector(".nav-item[title=汉字转国际音标]");if(!e)return false;e.click();return true})()');await current.button('恢复本机草稿');assert.equal(await current.js('document.querySelector("[aria-label=待转换汉字文本]").value'),'清理保留草稿');
 report.reopen=JSON.parse(await fs.readFile(path.join(cache,'last-launch.json'),'utf8'));assert.deepEqual(report.reopen.prepared,['science','host','apps']);
 report.checks.push('next launch rebuilt cache and restored saved theme and actual module draft');
 await current.js('window.close()');await current.finish();current=null;report.success=true;
}finally{
 if(current&&current.child.exitCode===null){const stop=spawn('taskkill.exe',['/PID',String(current.child.pid),'/T','/F'],{windowsHide:true,stdio:'ignore'});await new Promise(r=>stop.once('exit',r));}
 await fs.writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2));
}
console.log(JSON.stringify(report));
