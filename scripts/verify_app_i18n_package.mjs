// Operate the real final EXE through its owned CDP page; never replace app state or rendered data.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
const [port,output,phase]=process.argv.slice(2),out=path.resolve(output),root=path.resolve(import.meta.dirname,'..');
const input=JSON.parse(await fs.readFile(path.join(out,'input.json'),'utf8'));
const rows={};for(const code of ['zh-CN','en','ja']){
 rows[code]={};for(const file of await fs.readdir(path.join(root,'frontend/src/i18n/locales',code)))if(file.endsWith('.json'))Object.assign(rows[code],JSON.parse(await fs.readFile(path.join(root,'frontend/src/i18n/locales',code,file),'utf8')));
 const native=JSON.parse(await fs.readFile(path.join(root,'desktop/src/ptb_desktop/locales',code+'.json'),'utf8'));Object.assign(rows[code],Object.fromEntries(Object.entries(native).map(([key,value])=>['native.'+key,value])));
}
const label=(key,code)=>{assert(key in rows[code],key);return rows[code][key];};
const wait=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const sampleText='春江花月夜\n银行女略';
let socket,sequence=0,closingRequested=false;const pending=new Map(),errors=[],languageRequests=[],languageScripts=[];
const report={success:false,phase,scope:'Actual final EXE/native bridge; visible default GPU; real language controls and ZIP input; no scientific or media state injection',modules:[],manuals:[],nativeDialogs:[]};
async function call(method,params={},timeout=30000){const id=++sequence;return new Promise((resolve,reject)=>{const timer=setTimeout(()=>{pending.delete(id);reject(Error(method+' timed out'));},timeout);pending.set(id,{resolve,reject,timer});socket.send(JSON.stringify({id,method,params}));});}
async function js(expression){report.lastEvaluation=expression;const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true,userGesture:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
async function until(expression,seconds=45){const end=Date.now()+seconds*1000;while(Date.now()<end){if(await js(expression))return;await wait(120);}throw Error(expression+'\n'+await js('document.body.innerText.slice(-3500)'));}
async function click(selector){await until(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});if(!e||e.disabled)return false;e.click();return true})()`);await wait(80);}
async function button(key,code){const text=label(key,code);await until(`(()=>{const e=[...document.querySelectorAll('button')].find(e=>e.offsetParent&&[${JSON.stringify(text)},${JSON.stringify(text+' *')}].includes(e.textContent.trim()));if(!e||e.disabled)return false;e.click();return true})()`);await wait(100);}
async function fill(selector,value){assert(await js(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});if(!e)return false;e.value=${JSON.stringify(value)};e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));return true})()`));await wait(100);}
async function language(code){await until('!!document.querySelector(".language-switch select:not(:disabled)")');await fill('.language-switch select',code);await until(`document.documentElement.lang===${JSON.stringify(code)}&&document.documentElement.dataset.fontLocale===${JSON.stringify(code)}&&!document.querySelector('.language-switch select:disabled')&&document.fonts.status==='loaded'`);}
async function shot(name){const s=await call('Page.captureScreenshot',{format:'png'});await fs.writeFile(path.join(out,phase+'-'+name+'.png'),Buffer.from(s.data,'base64'));}
const modules=[
 ['M01','parameter_estimation','.m01-page'],['M02','parameter_display','.m02-page'],['M03','egg_signal_analysis','.egg-page'],['M04','lpc_spectrum','.lpc-page'],['M05','lip_tracking','.lip-page'],['M06','acoustic_synthesis','.m06-page'],['M07','phonation_synthesis','.m07-page'],['M08','time_and_pitch_manipulation','.m08-page'],['M09','spectrogram_to_audio','.m09-page'],['M10','articulatory_synthesis','.vocal-page'],['M11','mfa_forced_alignment','.mfa-page'],['M12','textgrid_annotation','.annotation-page'],['M13','chinese_to_ipa_transcription','.mandarin-ipa-page'],['M14','phonological_analysis','.phonology-page'],['M15','perceptual_experiments','.perception-page'],['M16','recording_157','.recording-page'],['M17','ipa_chart_plus','.ipa-plus-page'],['M18','phonetics_paper_reader','.paper-reading'],['M19','language_maps','.language-atlas'],
];
async function open(id,code){const [,key,selector]=modules.find(row=>row[0]===id);assert.equal(await js(`document.querySelector('.nav-item[data-module-id=${id}]')?.title`),label('workbench.'+key,code));await click('.nav-item[data-module-id='+id+']');await until(`!!document.querySelector(${JSON.stringify(selector)})&&document.querySelector('#tab-${id}')?.getAttribute('aria-selected')==='true'`);}
async function pointer(selector){const point=await js(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});e.scrollIntoView({block:'nearest'});const r=e.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()`);await call('Input.dispatchMouseEvent',{type:'mousePressed',button:'left',clickCount:1,...point});await call('Input.dispatchMouseEvent',{type:'mouseReleased',button:'left',clickCount:1,...point});}
async function manual(code,install){
 await language(code);await button('workbench.user_guide',code);await until('!!document.querySelector(".manual-document")');
 if(install){
  await until(`document.querySelector('.manual-language-toolbar select')?.value===${JSON.stringify(code)}&&!!document.querySelector('.manual-language-toolbar input[type=file]:not(:disabled)')`);
  console.log(JSON.stringify({phase,manual:code,beforeImport:await js(`({selected:document.querySelector('.manual-language-toolbar select')?.value,toolbar:document.querySelector('.manual-language-toolbar')?.textContent})`)}));
  const document=await call('DOM.getDocument');const field=await call('DOM.querySelector',{nodeId:document.root.nodeId,selector:'.manual-language-toolbar input[type=file]'});
  await call('DOM.setFileInputFiles',{nodeId:field.nodeId,files:[input.packs[code].path]});await wait(250);
  console.log(JSON.stringify({phase,manual:code,afterInput:await js(`({selected:document.querySelector('.manual-language-toolbar select')?.value,files:[...(document.querySelector('.manual-language-toolbar input')?.files??[])].map(file=>file.name),disabled:document.querySelector('.manual-language-toolbar input')?.disabled,toolbar:document.querySelector('.manual-language-toolbar')?.textContent})`)}));
 }
 await until('document.querySelector(".manual-language-toolbar span:not([role=status])")?.textContent.includes('+JSON.stringify(label('manual.reader.pack.offline',code).split('{version}')[0])+')',180);
 const read=[];
 for(const [chapter,title]of Object.entries(input.packs[code].chapterTitles).filter(([chapter])=>chapter!=='web-introduction')){
  await click('.manual-toc [data-manual-chapter='+JSON.stringify(chapter)+']');
  await until(`document.querySelector('.manual-document')?.dataset.manualChapter===${JSON.stringify(chapter)}&&document.querySelector('.manual-document h1')?.textContent.includes(${JSON.stringify(title)})`);
  const expected=input.packs[code].chapterMedia[chapter];
  for(const kind of ['image','audio','video']){
   const figures='.manual-document figure.manual-media-'+kind;
   await until(`JSON.stringify([...document.querySelectorAll(${JSON.stringify(figures)})].map(e=>e.dataset.assetId))===${JSON.stringify(JSON.stringify(expected[kind]))}`);
   for(let i=0;i<expected[kind].length;i++){
    // Use the actual figure list, since other figure kinds may be interleaved.
    await js(`(()=>{const e=document.querySelectorAll(${JSON.stringify(figures)})[${i}],closed=[];for(let p=e.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS'&&!p.open)closed.push(p);for(const p of closed.reverse())p.querySelector('summary').click();e.scrollIntoView({block:'center'});})()`);
    await until(`(()=>{const e=document.querySelectorAll(${JSON.stringify(figures)})[${i}];return !e.querySelector('.manual-loading')&&!e.querySelector('.manual-media-error')&&!!e.querySelector(${JSON.stringify(kind==='image'?'img':kind)})})()`,60);
    if(kind==='image')await until(`(()=>{const e=document.querySelectorAll(${JSON.stringify(figures)})[${i}].querySelector('img');return e.complete&&e.naturalWidth>0})()`,60);
    else{
     await js(`document.querySelectorAll(${JSON.stringify(figures)})[${i}].querySelector(${JSON.stringify(kind)}).load()`);
     await until(`(()=>{const e=document.querySelectorAll(${JSON.stringify(figures)})[${i}].querySelector(${JSON.stringify(kind)});return !e.error&&e.readyState>=1&&Number.isFinite(e.duration)&&e.duration>0})()`,60);
    }
   }
  }
  read.push(await js(`({chapter:document.querySelector('.manual-document').dataset.manualChapter,title:document.querySelector('.manual-document h1').textContent,images:document.querySelectorAll('.manual-document figure.manual-media-image img').length,audio:document.querySelectorAll('.manual-document figure.manual-media-audio audio').length,video:document.querySelectorAll('.manual-document figure.manual-media-video video').length})`));
 }
 assert.equal(read.length,21);report.manuals.push({locale:code,installed:install,packChapters:22,desktopChapters:21,webOnlyChapter:'web-introduction',chapters:read});await shot('manual-'+code);
}
try{
 let target;for(let i=0;i<240;i++){try{target=(await(await fetch('http://127.0.0.1:'+port+'/json')).json()).find(t=>t.url.startsWith('ptbapp:'));if(target)break;}catch{}await wait(500);}assert(target,'owned final EXE unavailable');
 socket=new WebSocket(target.webSocketDebuggerUrl);await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject;});socket.onmessage=event=>{const v=JSON.parse(event.data),p=pending.get(v.id);if(p){pending.delete(v.id);clearTimeout(p.timer);v.error?p.reject(Error(JSON.stringify(v.error))):p.resolve(v.result);}else if(v.method==='Runtime.exceptionThrown')errors.push(v.params.exceptionDetails);else if(v.method==='Page.javascriptDialogOpening'){report.nativeDialogs.push({type:v.params.type,message:v.params.message,duringClose:closingRequested});if(closingRequested&&v.params.type==='beforeunload'){const entry=report.nativeDialogs.at(-1);void call('Page.handleJavaScriptDialog',{accept:true},1500).then(()=>{entry.accepted=true;}).catch(error=>{entry.error=String(error);});}}else if(v.method==='Debugger.scriptParsed'&&v.params.url&&/\/(?:en|ja)-[^/]+\.js$/.test(new URL(v.params.url).pathname))languageScripts.push({url:v.params.url,scriptId:v.params.scriptId});else if(v.method==='Network.requestWillBeSent'&&/\/(?:en|ja)-[^/]+\.js$/.test(new URL(v.params.request.url).pathname))languageRequests.push(v.params.request.url);};
 await call('Page.enable');await call('Debugger.enable');await call('Runtime.enable');await call('Performance.enable');await call('Network.enable');await until('!!document.querySelector(".home-page")');await js('performance.setResourceTimingBufferSize(10000)');
 assert.equal(await js('document.documentElement.lang'),phase==='initial'?'zh-CN':'ja');
 await js(`(()=>{window.__languageBusiness=[];window.__updateChecks=0;const t=qt.webChannelTransport,send=t.send.bind(t);t.send=raw=>{try{const packet=JSON.parse(raw);for(const arg of packet.args??[]){if(typeof arg!=='string'||!arg.startsWith('{'))continue;const body=JSON.parse(arg),op=body.op??body.operation;if(body.operation==='check')window.__updateChecks++;if(op&&/create|run|synthesize|analyse|analyze|upload/.test(op))window.__languageBusiness.push(op);}}catch{}return send(raw);};})()`);
 if(phase==='initial'){
  for(const code of ['zh-CN','en','ja']){await language(code);for(const [id,,selector]of modules){await open(id,code);if(id==='M10'){
   await until(`(()=>{const f=document.querySelector('.vocal-page iframe'),d=f?.contentDocument;return d?.body.dataset.engineState==='ready'&&d.body.dataset.posePending==='false'&&d.documentElement.lang===${JSON.stringify(code)}&&d.body.innerText.trim().length>0})()`,90);
   assert(await js(`(()=>{const f=document.querySelector('.vocal-page iframe');if(!window.__liveVocalFrame)window.__liveVocalFrame=f;return window.__liveVocalFrame===f})()`),'same live vocal iframe across languages');
  }else assert(await js(`document.querySelector(${JSON.stringify(selector)}).innerText.trim().length>0`),id);report.modules.push({locale:code,id});}await shot('modules-'+code);}
  await language('en');await open('M13','en');await fill('.m13-input-section textarea',sampleText);await until('document.querySelectorAll(".m13-mapped").length>0');
  await js('window.__originalInput=document.querySelector(".m13-input-section textarea");window.__originalConversion=[...document.querySelectorAll(".m13-mapped")].map(e=>e.dataset.value).join("|");window.__languageBusiness=[]');
  const before=await call('Performance.getMetrics');
  for(let i=0;i<15;i++){await language(['ja','zh-CN','en'][i%3]);assert(await js(`window.__originalInput===document.querySelector('.m13-input-section textarea')&&window.__originalInput.value===${JSON.stringify(sampleText)}&&window.__originalConversion===[...document.querySelectorAll('.m13-mapped')].map(e=>e.dataset.value).join('|')`));}
  assert.deepEqual(await js('window.__languageBusiness'),[]);report.performance={before:before.metrics,after:(await call('Performance.getMetrics')).metrics,businessRequests:await js('window.__languageBusiness')};
  await button('modules.mandarin-ipa.text.81a4b066ea77','en');
  await open('M01','en');await button('modules.parameter-estimation.edit_14_settings','en');
  await fill('dialog[open] input[aria-label='+JSON.stringify(label('modules.parameter-estimation.minimum_fundamental_frequency','en'))+']',900);await button('components.apply_settings','en');
  await until('!!document.querySelector("dialog[open] [role=alert]")');await js('window.__languageBusiness=[]');report.validation=[];
  for(const code of ['en','ja','zh-CN']){await language(code);const actual=await js('document.querySelector("dialog[open] [role=alert]").textContent.trim()');assert.equal(actual,label('modules.parameter-estimation.minimum_fundamental_frequency_must_be_below_maximum_fundamental_frequency',code));report.validation.push({locale:code,text:actual});}
  assert.deepEqual(await js('window.__languageBusiness'),[]);await button('components.cancel','zh-CN');
  report.updateOffline=[];
  for(const code of ['en','ja','zh-CN']){
   await language(code);await click('.sidebar button[title='+JSON.stringify(label('workbench.check_for_updates',code))+']');await until('!!document.querySelector(".update-actions .primary:not(:disabled)")');
   if(code==='en'){await js('window.__updateChecks=0');await click('.update-actions .primary');}
   await until('document.querySelectorAll(".source-results .source-error").length===2&&!document.querySelector(".update-actions .primary").disabled');
   const texts=await js('[...document.querySelectorAll(".source-results .source-error")].map(e=>e.textContent.trim())');
   assert(texts.every(value=>value===label('native.errors.updates.networkFailure',code)),JSON.stringify(texts));
   assert.equal(await js('window.__updateChecks'),1,'reopening an existing failed check in another language must not check again');
   report.updateOffline.push({locale:code,sourceErrors:texts});await shot('offline-update-'+code);await click('dialog[open]>.dialog-header>.icon-button');await until('!document.querySelector(".update-panel")');
  }
  report.fullscreen=[];
  for(const code of ['en','ja']){
   await language(code);await open('M18',code);await click('.source-tabs button:nth-child(2)');
   await until("[...document.querySelectorAll('.paper-list button')].some(e=>e.textContent.includes('Owned fullscreen fixture'))");
   await js("[...document.querySelectorAll('.paper-list button')].find(e=>e.textContent.includes('Owned fullscreen fixture')).click()");await until('!!document.querySelector(".pdf-page img")');
   const restore=label('workbench.restore_window',code);
   if(await js('document.querySelector(".window-controls button:nth-child(2)").getAttribute("aria-label")')!==restore)await click('.window-controls button:nth-child(2)');
   await until('document.querySelector(".window-controls button:nth-child(2)").getAttribute("aria-label")==='+JSON.stringify(restore));
   for(let round=0;round<3;round++){
    await pointer('.pdf-toolbar>.pdf-controls:nth-child(3) button');await until('!!document.fullscreenElement');await pointer('.pdf-toolbar>.pdf-controls:nth-child(3) button');await until('!document.fullscreenElement');
    for(let sample=0;sample<10;sample++){await wait(160);assert.equal(await js('document.querySelector(".window-controls button:nth-child(2)").getAttribute("aria-label")'),restore);}
    report.fullscreen.push({locale:code,round:round+1,stableAfterExitMs:1600});
   }
   await shot('reader-'+code);
  }
  await manual('en',true);await manual('ja',true);
 }else{await manual('en',false);await manual('ja',false);}
 const resources=await js("performance.getEntriesByType('resource').map(e=>e.name)");
 report.languageResources=resources.filter(name=>/\/(?:en|ja)-[^/]+\.js$/.test(new URL(name).pathname));report.languageNetworkRequests=languageRequests;
 report.languageScripts=languageScripts;
 if(phase==='initial'){
  report.languageCountEvidence=report.languageResources.length?'resourceTiming':'CDP Network.requestWillBeSent';
  const observed=report.languageResources.length?report.languageResources:languageRequests;
  assert.equal(observed.filter(s=>s.includes('/en-')).length,1,'English bundle fetched exactly once');assert.equal(observed.filter(s=>s.includes('/ja-')).length,1,'Japanese bundle fetched exactly once');
 }else{
  report.languageCountEvidence='Debugger.scriptParsed backfills the saved Japanese bundle loaded before attachment; Network records subsequent requests';
  for(const code of ['en','ja']){
   const scripts=languageScripts.filter(s=>s.url.includes('/'+code+'-'));
   assert.equal(new Set(scripts.map(s=>s.scriptId)).size,1,code+' compiled locale executes once in this process');
   assert(languageRequests.filter(s=>s.includes('/'+code+'-')).length<=1,code+' locale has no duplicate requests after attachment');
  }
 }
 report.gpu=await js("(()=>{const c=document.createElement('canvas'),g=c.getContext('webgl'),e=g?.getExtension('WEBGL_debug_renderer_info');return e?g.getParameter(e.UNMASKED_RENDERER_WEBGL):null})()");
 // Close the owned test tabs through their real controls. Saved M13 data and
 // installed manuals remain on disk; disposable unsaved test drafts use the
 // application's own discard action before the final native window closes.
 report.closedTabs=[];
 const tabs=await js("[...document.querySelectorAll('[role=tab][id^=tab-]')].map(e=>e.id).filter(id=>id!=='tab-home')");
 const discardLabels=['ja','en','zh-CN'].flatMap(code=>['workbench.discard_changes_and_close','workbench.discard_draft_and_close','workbench.discard_results_and_draft_then_close','workbench.discard_drawing_and_close','workbench.discard_font_edits_and_close'].map(key=>label(key,code)));
 for(const id of tabs){
  await js(`document.getElementById(${JSON.stringify(id)})?.closest('.tab-wrap')?.querySelector('.tab-close')?.click()`);
  const deadline=Date.now()+30000,actions=[];
  while(await js(`!!document.getElementById(${JSON.stringify(id)})`)&&Date.now()<deadline){
   const action=await js(`(()=>{const e=[...document.querySelectorAll('dialog[open] button')].find(e=>!e.disabled&&${JSON.stringify(discardLabels)}.includes(e.textContent.trim()));if(!e)return null;const text=e.textContent.trim();e.click();return text})()`);
   if(action)actions.push(action);await wait(100);
  }
  assert.equal(await js(`!!document.getElementById(${JSON.stringify(id)})`),false,'owned tab closes normally: '+id);report.closedTabs.push({id,actions});
 }
 assert.deepEqual(errors,[]);report.success=true;
}catch(error){report.error=String(error);report.imageFailureState=await js(`({visibility:document.visibilityState,viewport:[innerWidth,innerHeight],images:[...document.querySelectorAll('.manual-document figure.manual-media-image')].map(f=>{const e=f.querySelector('img'),r=e?.getBoundingClientRect();return {asset:f.dataset.assetId,src:e?.src,complete:e?.complete,naturalWidth:e?.naturalWidth,loading:e?.loading,width:e?.getAttribute('width'),height:e?.getAttribute('height'),rect:r?{x:r.x,y:r.y,width:r.width,height:r.height}:null,error:f.querySelector('.manual-media-error')?.textContent}})})`).catch(String);await shot('failure').catch(()=>{});throw error;}
finally{
 report.errors=errors;await fs.writeFile(path.join(out,phase+'-report.json'),JSON.stringify(report,null,2)+'\n');
 if(socket?.readyState===1){
  closingRequested=true;
  const closing=async expression=>{if(socket.readyState===1)await call('Runtime.evaluate',{expression,returnByValue:true,userGesture:true},1500).catch(()=>{});};
  await closing('document.querySelector(".window-close")?.click()');await wait(500);
  const discard=['ja','en','zh-CN'].flatMap(code=>['workbench.discard_changes_and_close','workbench.discard_draft_and_close','workbench.discard_results_and_draft_then_close','workbench.discard_drawing_and_close','workbench.discard_font_edits_and_close'].map(key=>label(key,code)));
  await closing(`(()=>{const e=[...document.querySelectorAll('dialog[open] button')].find(e=>${JSON.stringify(discard)}.includes(e.textContent.trim()));if(e)e.click()})()`);socket.close();
 }
 await fs.writeFile(path.join(out,phase+'-report.json'),JSON.stringify(report,null,2)+'\n');
 console.log(JSON.stringify({success:report.success,phase,modules:report.modules.length,manuals:report.manuals.length,error:report.error}));
}
