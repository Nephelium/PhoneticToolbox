/** Profile navigation in an already-owned Qt process. Never closes the browser. */
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';

const [port,output,identity]=process.argv.slice(2);
if(!port||!output||!identity||!(/^(source-qt|final-exe:[a-f0-9]{64})$/.test(identity)))throw Error('Expected owned CDP port, evidence directory and source-qt or final-exe:<sha256>');
const out=path.resolve(output);await fs.mkdir(out,{recursive:true});
const wait=ms=>new Promise(resolve=>setTimeout(resolve,ms));
let socket,sequence=0,requests=0;const pending=new Map();
const report={success:false,identity,scope:'Owned Qt process; existing default GPU and window state; navigation CPU, resource reuse, actual Chromium screenshots; native pixels and final executable identity checked by its launcher',cases:[],errors:[]};
function call(method,params={}){return new Promise((resolve,reject)=>{const id=++sequence,timer=setTimeout(()=>{pending.delete(id);reject(Error(method+' timed out'));},15000);pending.set(id,{resolve,reject,timer});socket.send(JSON.stringify({id,method,params}));});}
async function evaluate(expression){report.lastEvaluation=expression;const value=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true,userGesture:true});if(value.exceptionDetails)throw Error(JSON.stringify(value.exceptionDetails));return value.result.value;}
async function until(code){for(let i=0;i<200;i++){if(await evaluate(code))return;await wait(100);}throw Error(code);}
async function click(selector){await until(`!!document.querySelector(${JSON.stringify(selector)})`);await evaluate(`document.querySelector(${JSON.stringify(selector)}).click();true`);await wait(80);}
async function measure(name,action){
 await evaluate('window.__readerLong=[]');const start=Date.now(),requestStart=requests;
 const before=Object.fromEntries((await call('Performance.getMetrics')).metrics.map(metric=>[metric.name,metric.value]));
 await action();await wait(120);
 const metrics=(await call('Performance.getMetrics')).metrics;
 const result={name,elapsedMs:Date.now()-start,requests:requests-requestStart,metrics:Object.fromEntries(metrics.filter(metric=>['TaskDuration','ScriptDuration','LayoutDuration','RecalcStyleDuration','LayoutCount','RecalcStyleCount','Nodes'].includes(metric.name)).map(metric=>[metric.name,metric.value-before[metric.name]])),longTasksMs:await evaluate('window.__readerLong')};
 report.cases.push(result);await fs.writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2)+'\n');return result;
}
try{
 let target;for(let i=0;i<180;i++){try{target=(await(await fetch(`http://127.0.0.1:${port}/json`)).json()).find(tab=>tab.url.startsWith('ptbapp:'));if(target)break;}catch{}await wait(100);}assert(target,'Owned Qt page unavailable');
 socket=new WebSocket(target.webSocketDebuggerUrl);await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject;});
 socket.onmessage=event=>{const value=JSON.parse(event.data);if(value.method==='Network.requestWillBeSent')requests++;if(value.method==='Inspector.targetCrashed')report.errors.push('targetCrashed');const entry=pending.get(value.id);if(!entry)return;pending.delete(value.id);clearTimeout(entry.timer);value.error?entry.reject(Error(JSON.stringify(value.error))):entry.resolve(value.result);};
 await until('!!document.querySelector("#tool-sidebar")');await call('Performance.enable');await call('Network.enable');
 await evaluate('window.__readerLong=[];window.__readerObserver=new PerformanceObserver(list=>window.__readerLong.push(...list.getEntries().map(entry=>entry.duration)));window.__readerObserver.observe({type:"longtask"});true');
 report.renderer=await evaluate('(()=>{const g=document.createElement("canvas").getContext("webgl"),e=g?.getExtension("WEBGL_debug_renderer_info");return {userAgent:navigator.userAgent,devicePixelRatio,renderer:e?g.getParameter(e.UNMASKED_RENDERER_WEBGL):null}})()');
 await click('#tool-sidebar button[title="设置"]');
 await evaluate('(()=>{const c=document.createElement("canvas");c.width=1200;c.height=800;const x=c.getContext("2d");for(let y=0;y<800;y+=40)for(let v=0;v<1200;v+=40){x.fillStyle=(v+y)%80?"#4477aa":"#eebbaa";x.fillRect(v,y,40,40)}c.toBlob(blob=>{const d=new DataTransfer();d.items.add(new File([blob],"reader-QA-checker.png",{type:"image/png"}));const e=document.querySelector("input[aria-label=上传背景图]");e.files=d.files;e.dispatchEvent(new Event("change",{bubbles:true}));});return true})()');
 await until('getComputedStyle(document.documentElement).getPropertyValue("--button-glass-image").includes("blob:")');await wait(400);
 for(const material of ['solid','glass']){
  await evaluate(`document.documentElement.dataset.buttonMaterial=${JSON.stringify(material)};true`);
  await click('#tool-sidebar button[title="国际音标表Plus"]');await until('document.querySelector(".m17-body")?.dataset.fontReady==="true"');
  await measure(material+'-systems-x9',async()=>{for(let i=0;i<9;i++)await click('.m17-tabs button:nth-child('+(i%3+1)+')');});
  await measure(material+'-m17-tab-switches-x12',async()=>{for(let i=0;i<6;i++){await click('#tab-home');await click('#tab-M17');}});
  await click('#tool-sidebar button[title="使用说明"]');await until('!!document.querySelector(".manual-document")');
  await click('[data-manual-chapter=m01]');await until('document.querySelector(".manual-document")?.dataset.manualChapter==="m01"');await wait(300);
  await evaluate('window.__readerDocument=document.querySelector(".manual-document");window.__readerMedia=[...window.__readerDocument.querySelectorAll("img,audio,video")];true');
  const switches=await measure(material+'-manual-tab-switches-x12',async()=>{for(let i=0;i<6;i++){await click('#tab-home');await click('#tab-help');}});
  assert.equal(switches.requests,0,'Unchanged tab switches read new resources');
  const outline=await measure(material+'-same-chapter-x6',async()=>{for(let i=0;i<6;i++)await click('[data-manual-section]');});
  assert.equal(outline.requests,0,'Same-chapter jumps read new resources');
  assert(await evaluate('window.__readerDocument===document.querySelector(".manual-document")&&window.__readerMedia.every(node=>node.isConnected)'),'Navigation replaced mounted chapter/media');
  const idle=await measure(material+'-idle',()=>wait(2000));assert.equal(idle.requests,0,'Idle reader read new resources');assert.equal(idle.metrics.LayoutCount,0,'Idle reader kept laying out');
  const shot=await call('Page.captureScreenshot',{format:'png'});await fs.writeFile(path.join(out,material+'-manual.png'),Buffer.from(shot.data,'base64'));
 }
 if(process.argv.includes('--geo-map')){
  await click('#tool-sidebar button[title="语言地图"]');await until('!!document.querySelector(".geo-map .ol-viewport")');
  await evaluate('window.__readerMap=document.querySelector(".geo-map .ol-viewport");true');
  await measure('M17-help-M19-x18',async()=>{for(let i=0;i<6;i++)for(const id of ['M17','help','M19'])await click('#tab-'+id);});
  assert(await evaluate('window.__readerMap===document.querySelector(".geo-map .ol-viewport")'),'Map remounted during navigation');
  report.windows=[];
  for(let i=0;i<4;i++){
   const state=await evaluate('[...document.querySelectorAll(".window-controls button")].find(e=>/最大化|还原/.test(e.getAttribute("aria-label")||""))?.getAttribute("aria-label")');
   assert(state,'Native maximize/restore control unavailable');
   await click('.window-controls button[aria-label="'+state+'"]');await wait(400);
   report.windows.push(await evaluate('({width:innerWidth,height:innerHeight,controls:[...document.querySelectorAll(".window-controls button")].map(e=>e.getAttribute("aria-label"))})'));
  }
  await click('[aria-label="关闭 语言地图"]');
  await evaluate('[...document.querySelectorAll("button")].find(e=>e.offsetParent&&e.textContent.trim()==="放弃草稿并关闭")?.click();true');
  await until('!document.querySelector(".geo-map")');report.mapClosed=true;
 }
 await click('#tab-M17');await click('.m17-tabs button:nth-child(2)');const shot=await call('Page.captureScreenshot',{format:'png'});await fs.writeFile(path.join(out,'extipa.png'),Buffer.from(shot.data,'base64'));
 assert.deepEqual(report.errors,[]);report.success=true;
}catch(error){report.error=String(error);process.exitCode=1;}finally{await fs.writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2)+'\n');socket?.close();console.log(JSON.stringify({success:report.success,identity,cases:report.cases.length,error:report.error,output:out}));}
