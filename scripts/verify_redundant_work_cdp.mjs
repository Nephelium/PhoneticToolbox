// Passive counters in an owned final-EXE Qt window. Qt exposes page-level CDP.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
const [port,output,mode]=process.argv.slice(2),out=path.resolve(output);
const keepOpen=process.argv.includes('--keep-open');await fs.mkdir(out,{recursive:true});
const wait=ms=>new Promise(resolve=>setTimeout(resolve,ms));
let socket;const pending=new Map();let sequence=0;
const report={success:false,mode,scope:'Actual final EXE; isolated user data/generated PDF; visible default GPU; passive native QWebChannel render counter'};
async function call(method,params={}){const id=++sequence;return new Promise((resolve,reject)=>{const timer=setTimeout(()=>{pending.delete(id);reject(Error(method+' timed out'));},30000);pending.set(id,{resolve,reject,timer});socket.send(JSON.stringify({id,method,params}));});}
async function evaluate(expression){const value=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true,userGesture:true});if(value.exceptionDetails)throw Error(JSON.stringify(value.exceptionDetails));return value.result.value;}
async function until(code){for(let i=0;i<180;i++){if(await evaluate(code))return;await wait(150);}throw Error(code);}
async function click(selector){await until(`!!document.querySelector(${JSON.stringify(selector)})`);await evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);}
try{
 let target;for(let i=0;i<180;i++){try{target=(await(await fetch('http://127.0.0.1:'+port+'/json')).json()).find(t=>t.url.startsWith('ptbapp:'));if(target)break;}catch{}await wait(500);}assert(target,'owned EXE debug endpoint unavailable');
 socket=new WebSocket(target.webSocketDebuggerUrl);await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject;});
 socket.onmessage=event=>{const value=JSON.parse(event.data),entry=pending.get(value.id);if(!entry)return;pending.delete(value.id);clearTimeout(entry.timer);value.error?entry.reject(Error(JSON.stringify(value.error))):entry.resolve(value.result);};
 await until('!!document.querySelector("#tool-sidebar")');
 await evaluate(`(()=>{window.__renders=[];const transport=qt.webChannelTransport,send=transport.send.bind(transport);transport.send=raw=>{try{const message=JSON.parse(raw);for(const value of message.args??[]){if(typeof value!=='string'||!value.startsWith('{'))continue;const body=JSON.parse(value);if(body.operation==='render')window.__renders.push({id:body.args.id,language:body.args.language,page:body.args.page,width:body.args.width,time:performance.now(),hidden:!document.querySelector('.paper-reading')?.offsetParent});}}catch{}return send(raw);};})()`);
 await click('#tool-sidebar button[title="语音学论文精读"]');await until('!!document.querySelector(".source-tabs")');
 await evaluate(`[...document.querySelectorAll('.source-tabs button')].find(e=>e.textContent.includes('我的 PDF')).click()`);
 await until(`[...document.querySelectorAll('.paper-list button')].some(e=>e.textContent.includes('Owned performance fixture'))`);
 await evaluate(`[...document.querySelectorAll('.paper-list button')].find(e=>e.textContent.includes('Owned performance fixture')).click()`);
 await until('document.querySelectorAll(".pdf-page img").length>=2');await wait(700);
 report.initial=await evaluate('window.__renders');assert(report.initial.length>0,'counter did not observe the real native bridge');
 if(mode==='clean'){
  const initialPages=new Map();for(const request of report.initial.filter(r=>r.page===1)){const key=request.id+':'+request.language;initialPages.set(key,(initialPages.get(key)??0)+1);}for(const [key,count] of initialPages)assert.equal(count,1,'initial page rasterized before the layout settled: '+key);
 }
 await evaluate('window.__renders=[]');
 for(let i=0;i<6;i++){await click('#tab-home');await wait(220);await click('#tab-M18');await wait(350);}
 await wait(500);report.switches=await evaluate('window.__renders');if(mode==='clean')assert.equal(report.switches.length,0,'unchanged final-EXE tab switches rasterized PDF pages');
 report.gpu=await evaluate(`(()=>{const c=document.createElement('canvas'),g=c.getContext('webgl'),e=g?.getExtension('WEBGL_debug_renderer_info');return e?g.getParameter(e.UNMASKED_RENDERER_WEBGL):null;})()`);
 const screenshot=await call('Page.captureScreenshot',{format:'png'});await fs.writeFile(path.join(out,'exe-pdf.png'),Buffer.from(screenshot.data,'base64'));
 if(mode==='clean'){
  await click('#tool-sidebar button[title="使用说明"]');await until('!!document.querySelector(".manual-document")');report.manual=[];
  for(const [theme,chapter] of [['light','m01'],['dark','m03']]){
   await evaluate(`document.documentElement.dataset.theme=${JSON.stringify(theme)}`);
   await click(`.manual-toc [data-manual-chapter="${chapter}"]`);
   await until(`document.querySelector('.manual-document')?.dataset.manualChapter===${JSON.stringify(chapter)}`);await wait(500);
   await evaluate(`window.__manualDocument=document.querySelector('.manual-document');window.__manualMedia=[...window.__manualDocument.querySelectorAll('img,audio,video')];window.__manualMutations=0;window.__manualObserver=new MutationObserver(records=>{window.__manualMutations+=records.filter(r=>r.type==='childList').length});window.__manualObserver.observe(window.__manualDocument,{childList:true,subtree:true})`);
   for(let i=0;i<6;i++){await click('.manual-toc [data-manual-section]');await wait(40);}
   const navigation=await evaluate(`({sameDocument:window.__manualDocument===document.querySelector('.manual-document'),mediaConnected:window.__manualMedia.every(e=>e.isConnected),childRebuilds:window.__manualMutations})`);
   await evaluate('window.__manualObserver.disconnect()');assert(navigation.sameDocument&&navigation.mediaConnected,'same chapter navigation replaced document or media');assert.equal(navigation.childRebuilds,0,'same chapter navigation rebuilt document children');
   await evaluate(`document.querySelector('.manual-toc [data-manual-section]').focus({focusVisible:true})`);
   await call('Input.dispatchKeyEvent',{type:'keyDown',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});await call('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});await wait(60);
   const focus=await evaluate(`(()=>{const e=document.activeElement,s=getComputedStyle(e);return {selected:e.matches('.manual-toc .active'),visible:e.matches(':focus-visible'),outline:s.outlineStyle,border:s.borderTopStyle,shadow:s.boxShadow,decoration:s.textDecorationLine}})()`);
   assert(focus.selected&&focus.visible&&focus.outline==='none'&&focus.border==='none'&&focus.shadow==='none'&&focus.decoration.includes('underline'),'manual keyboard focus regained a box: '+JSON.stringify(focus));
   report.manual.push({theme,chapter,...navigation,focus});const capture=await call('Page.captureScreenshot',{format:'png'});await fs.writeFile(path.join(out,`exe-manual-${theme}.png`),Buffer.from(capture.data,'base64'));
  }
 }
 if(keepOpen){await click('#tool-sidebar button[title="国际音标表Plus"]');await until('document.querySelector(".m17-body")?.dataset.loaded==="true"&&document.querySelector(".m17-body")?.dataset.fontReady==="true"');}report.success=true;
}catch(error){report.error=String(error);throw error;}
finally{await fs.writeFile(path.join(out,'cdp-report.json'),JSON.stringify(report,null,2)+'\n');if(socket?.readyState===1){if(!keepOpen)await evaluate('document.querySelector(".window-close")?.click()').catch(()=>{});socket.close();}console.log(JSON.stringify({success:report.success,initial:report.initial?.length,repeatedRenders:report.switches?.length,error:report.error}));}
