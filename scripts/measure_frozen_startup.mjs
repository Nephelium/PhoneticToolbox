// P19: compare ordinary EXE startup with isolated profiles and no visible window.
// This measures application readiness, not physical display presentation time.
import {spawn} from 'node:child_process';
import {mkdir,readFile,writeFile,readdir} from 'node:fs/promises';
import path from 'node:path';
import net from 'node:net';

const [exeArg,outputArg,countArg='3',reuseProfile]=process.argv.slice(2);
if(!exeArg||!outputArg)throw new Error('Usage: node measure_frozen_startup.mjs <exe> <new-output> [count]');
const exe=path.resolve(exeArg),out=path.resolve(outputArg),count=Number(countArg);
if(!Number.isInteger(count)||count<1||count>5)throw new Error('Invalid repetition count');
await mkdir(out,{recursive:false});
const sharedProfile=reuseProfile?path.resolve(reuseProfile):null;
if(sharedProfile)await mkdir(sharedProfile,{recursive:true});
const sleep=ms=>new Promise(resolve=>{const timer=setTimeout(resolve,ms);if(ms>10000)timer.unref();});
async function port(){const server=net.createServer();await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));const value=server.address().port;await new Promise(resolve=>server.close(resolve));return value;}
async function connect(url){
  const socket=new WebSocket(url);await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',reject,{once:true});});
  let next=0;const calls=new Map();
  socket.addEventListener('message',({data})=>{const v=JSON.parse(data);if(calls.has(v.id)){calls.get(v.id)(v);calls.delete(v.id);}});
  return {socket,async evaluate(expression){const id=++next;const answer=new Promise(resolve=>calls.set(id,resolve));socket.send(JSON.stringify({id,method:'Runtime.evaluate',params:{expression,returnByValue:true}}));const v=await Promise.race([answer,sleep(2000).then(()=>null)]);return v?.result?.result?.value;}};
}
const rows=[];
for(let i=0;i<count;i++){
  const run=path.join(out,`run-${i+1}`);await mkdir(run);await mkdir(path.join(run,'temp'));await mkdir(path.join(run,'profile'));
  const debugPort=await port();
  const profile=sharedProfile??path.join(run,'profile');
  const env={...process.env,LOCALAPPDATA:profile,TEMP:path.join(run,'temp'),TMP:path.join(run,'temp'),
    PATH:`${process.env.SystemRoot}\\System32;${process.env.SystemRoot}`,QT_QPA_PLATFORM:'offscreen',
    QTWEBENGINE_REMOTE_DEBUGGING:`127.0.0.1:${debugPort}`,QTWEBENGINE_CHROMIUM_FLAGS:'--mute-audio --disable-gpu --remote-allow-origins=*'};
  for(const key of Object.keys(env))if(/^(PTB_|PYTHON|CONDA|_PYI)/.test(key)||['VIRTUAL_ENV','QT_PLUGIN_PATH','QT_QPA_PLATFORM_PLUGIN_PATH','QTWEBENGINEPROCESS_PATH','QTWEBENGINE_RESOURCES_PATH','QTWEBENGINE_LOCALES_PATH'].includes(key))delete env[key];
  env.PTB_OWNED_BOOTSTRAP_LOG=path.join(run,'bootstrap-error.log');
  const updates=path.join(profile,'PhoneticToolbox/v3/updates');await mkdir(updates,{recursive:true});await writeFile(path.join(updates,'state.json'),'{"autoCheck":false}');
  const start=performance.now(),child=spawn(exe,[],{cwd:run,env,windowsHide:true,stdio:['ignore','pipe','pipe']});
  const logs=[];child.stdout.on('data',b=>logs.push(b));child.stderr.on('data',b=>logs.push(b));
  const exit=new Promise(resolve=>child.once('exit',(code,signal)=>resolve({code,signal})));
  let connection,readyMs=null,debugMs=null,restoration=null,bundle=null;
  try{
    while(performance.now()-start<150000&&child.exitCode===null){
      if(!connection){
        try{const targets=await (await fetch(`http://127.0.0.1:${debugPort}/json/list`,{signal:AbortSignal.timeout(200)})).json();const page=targets.find(t=>t.type==='page'&&t.url.startsWith('ptbapp://app/'));if(page){connection=await connect(page.webSocketDebuggerUrl);debugMs=performance.now()-start;}}catch{}
      }
      if(connection){
        const ok=await connection.evaluate('!!document.querySelector(".home-page") && document.fonts.status==="loaded" && !!document.querySelector(".nav-item[title=汉字转国际音标]") && !!window.qt');
        if(ok){readyMs=performance.now()-start;break;}
      }
      await sleep(50);
    }
    if(readyMs===null)throw new Error('No responsive rendered home page before deadline');
    await connection.evaluate('document.querySelector(".nav-item[title=汉字转国际音标]").click()');
    let moduleReady=false;const moduleDeadline=performance.now()+30000;
    while(performance.now()<moduleDeadline){
      moduleReady=await connection.evaluate('!!document.querySelector(".mandarin-ipa-page") && !!document.querySelector("[aria-label=待转换汉字文本]") && document.querySelector("[aria-label=汉字字体]")?.options.length>1');
      if(moduleReady)break;await sleep(50);
    }
    if(!moduleReady)throw new Error('Home page could not open the native-font module');
    await connection.evaluate('(()=>{const e=document.querySelector("[aria-label=待转换汉字文本]");e.value="春江花月夜";e.dispatchEvent(new Event("input",{bubbles:true}));})()');
    await sleep(150);
    if(!await connection.evaluate('document.querySelectorAll(".m13-mapped").length===5'))throw new Error('Module did not respond to input');
    for(const name of await readdir(path.join(run,'temp'))){
      if(name.startsWith('_MEI')){
        bundle=path.join(run,'temp',name);
        try{restoration=JSON.parse(await readFile(path.join(bundle,'.ptb-host-ready.json'),'utf8'));}catch{}
      }
    }
    await connection.evaluate('window.close()');connection.socket.close();
    const finished=await Promise.race([exit,sleep(60000).then(()=>null)]);
    if(!finished)throw new Error('Owned application did not exit');
    await sleep(250);
    let persistentCache=null;
    try{persistentCache=JSON.parse(await readFile(path.join(profile,'PhoneticToolbox/v3/startup-cache/last-launch.json'),'utf8'));}catch{}
    rows.push({iteration:i+1,readyMs,debugMs,exit:finished,moduleReady,restoration,persistentCache,
      remainingTemp:await readdir(path.join(run,'temp')),bundle});
    await writeFile(path.join(run,'process.log'),Buffer.concat(logs));
    await writeFile(path.join(out,'results.json'),JSON.stringify({exe,sharedProfile,scope:'ordinary no-argument launch; new process every run; explicit shared profile for persistent warm tests, otherwise new profile; Qt offscreen with software rendering; OS cache not cleared; not visible-window latency',rows},null,2));
    if(finished.code!==0||rows.at(-1).remainingTemp.some(n=>n.startsWith('_MEI')))throw new Error('Application did not exit cleanly');
    console.log(JSON.stringify(rows.at(-1)));
  }catch(error){
    connection?.socket.close();
    if(child.exitCode===null){
      const stop=spawn('taskkill.exe',['/PID',String(child.pid),'/T','/F'],{windowsHide:true,stdio:'ignore'});
      await new Promise(resolve=>stop.once('exit',resolve));
    }
    await writeFile(path.join(run,'process.log'),Buffer.concat(logs));
    throw error;
  }
}
