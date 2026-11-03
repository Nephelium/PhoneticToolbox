const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url'),{createHash}=require('node:crypto');
const root=path.resolve(__dirname,'../..');
async function main(){
 const spec=JSON.parse(await fs.readFile(process.argv[2],'utf8'));
 const report={success:false,completeCandidates:true,scope:'Already-open website; complete offline manual including help remount. No offline website startup claim.',errors:[],requests:[],locales:{}};
 let server,chrome,context,page;
 const sources=['frontend/src/app/HelpPage.vue','frontend/src/manual/ManualReader.vue','frontend/src/manual/shared-cache.ts','frontend/src/manual/image-cache.ts','frontend/src/manual/content.ts','frontend/src/platform/manual-packs.ts',...['zh-CN','en','ja'].map(locale=>'frontend/src/i18n/locales/'+locale+'/manual.json')];
 report.sources=Object.fromEntries(await Promise.all(sources.map(async file=>[file,createHash('sha256').update(await fs.readFile(path.join(root,file))).digest('hex')])));
 try{
  const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
  const harness=`<!doctype html><html><head><meta charset="utf-8"><style>html,body,#app{height:100%;margin:0}</style></head><body><div id="app"></div><script type="module">
   import {createApp,ref,nextTick,h} from 'vue';import HelpPage from '/src/app/HelpPage.vue';import {i18n,setLocale} from '/src/i18n/index.ts';import {manualPackStore} from '/src/platform/manual-packs.ts';import '/src/design/tokens.css';
   const shown=ref(true),store=manualPackStore(),reads={json:[],asset:[],created:0,revoked:0};const readJson=store.readJson.bind(store),readAsset=store.readAsset.bind(store);
   store.readJson=(locale,path)=>{reads.json.push(locale+':'+path);return readJson(locale,path)};store.readAsset=(locale,path)=>{reads.asset.push(locale+':'+path);return readAsset(locale,path)};
   const create=URL.createObjectURL.bind(URL),revoke=URL.revokeObjectURL.bind(URL);URL.createObjectURL=b=>{reads.created++;return create(b)};URL.revokeObjectURL=u=>{reads.revoked++;return revoke(u)};
   window.manualTest={setLocale,store,reads,show:async value=>{shown.value=value;await nextTick()}};
   createApp({setup:()=>()=>shown.value?h(HelpPage):null}).use(i18n).mount('#app');
  </script></body></html>`;
  server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{noDiscovery:true,include:['vue']},plugins:[{name:'complete-offline-manual',configureServer(vite){vite.middlewares.use(async(req,res,next)=>{
   const pathname=new URL(req.url,'http://localhost').pathname;
   if(pathname==='/complete-manual-test.html'){res.setHeader('Content-Type','text/html');res.end(await vite.transformIndexHtml(pathname,harness));return;}
   if(!pathname.startsWith('/manual/'))return next();
   report.requests.push(pathname);
   try{const file=path.resolve(spec.base,decodeURIComponent(pathname.slice('/manual/'.length)));if(!file.startsWith(path.resolve(spec.base)+path.sep))throw Error('path');
    const bytes=await fs.readFile(file);res.setHeader('Content-Length',bytes.length);res.setHeader('Content-Type',file.endsWith('.json')?'application/json':file.endsWith('.wav')?'audio/wav':file.endsWith('.mp4')?'video/mp4':'image/png');res.end(bytes);
   }catch(error){res.statusCode=404;res.end(String(error));}
  });}}]});await server.listen();const origin=server.resolvedUrls.local[0].replace(/\/$/,'');
  const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  // Chromium owns and removes its short temporary profile. Deep Windows paths
  // exceed the native CacheStorage path budget and are unsuitable test inputs.
  chrome=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--mute-audio']});context=await chrome.newContext({viewport:{width:1440,height:900}});page=await context.newPage();page.setDefaultTimeout(120000);
  page.on('pageerror',error=>report.errors.push(error.stack||error.message));page.on('requestfailed',request=>{(report.failedRequests??=[]).push({url:request.url(),error:request.failure()});});report.browser=chrome.version();
  await page.goto(origin+'/complete-manual-test.html');await page.locator('.manual-document h1').waitFor();
  assert(!report.requests.some(url=>url.endsWith('language-packs.json')));report.chineseStartupRequests=[...report.requests];
  for(const pack of spec.packs){
   const bytes=await fs.readFile(pack.path);assert.equal(createHash('sha256').update(bytes).digest('hex'),pack.sha256);
   await page.evaluate(locale=>window.manualTest.setLocale(locale),pack.locale);
   const chooser=page.waitForEvent('filechooser');await page.locator('.manual-language-toolbar button').last().click();await(await chooser).setFiles(pack.path);
   await page.waitForFunction(version=>{const input=document.querySelector('.manual-language-toolbar input');return document.querySelector('.manual-pack-error')||(input&&!input.disabled&&document.querySelector('.manual-language-toolbar')?.textContent.includes(version)&&!document.querySelector('.manual-start-state[role=status]'));},pack.version,{timeout:120000});
   assert.equal(await page.locator('.manual-pack-error').count(),0,await page.locator('.manual-language-toolbar').innerText());
   assert.equal(await page.locator('.manual-language-toolbar button').count(),1,'complete cache has no repair button');
   report.locales[pack.locale]={zip:{path:pack.path,sha256:pack.sha256,bytes:bytes.length},chapters:[],shared:[],overrides:[]};
  }
  const beforeOffline=report.requests.length;await context.setOffline(true);assert.equal(await page.evaluate(()=>navigator.onLine),false);
  for(const pack of spec.packs){
   await page.evaluate(locale=>window.manualTest.setLocale(locale),pack.locale);
   // Exercise a fresh HelpPage instance with no in-memory baseProject. The
   // failed network request must recover a verified, compatible cached index.
   await page.evaluate(()=>window.manualTest.show(false));await page.evaluate(()=>window.manualTest.show(true));
   await page.locator('.manual-document h1').waitFor();assert.equal(await page.locator('.manual-start-state[role=alert]').count(),0);
   const translated=await page.evaluate(locale=>window.manualTest.store.readJson(locale,'project.json'),pack.locale);
   const installed=await page.evaluate(async locale=>(await window.manualTest.store.list()).find(item=>item.locale===locale),pack.locale);
   assert.equal(translated.chapters.length,22);assert.equal(installed.sourceDigest,pack.sourceDigest);
   for(const chapter of translated.chapters){
    await page.locator('button[data-manual-chapter="'+chapter.id+'"]').click();
    await page.waitForFunction(chapter=>document.querySelector('.manual-document')?.getAttribute('data-manual-chapter')===chapter.id&&document.querySelector('.manual-document h1')?.textContent.includes(chapter.title),chapter,{timeout:15000});
    await page.waitForFunction(()=>!document.querySelector('.manual-document .manual-loading'));
    const media=await page.evaluate(async()=>{
     const result=[];
     for(const figure of document.querySelectorAll('.manual-document [data-asset-id]')){
      const element=figure.querySelector('img,audio,video');if(!element)throw Error('Media unavailable: '+figure.dataset.assetId);
      if(!element.src.startsWith('blob:'))throw Error('Non-offline source: '+element.src);
      const response=await fetch(element.src),bytes=new Uint8Array(await response.arrayBuffer());
      const sha=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');
      if(element.tagName==='IMG'){const bitmap=await createImageBitmap(new Blob([bytes]));bitmap.close();}
      if(element.tagName==='AUDIO'){const decoder=new AudioContext();try{const decoded=await decoder.decodeAudioData(bytes.buffer.slice(0));if(!decoded.length)throw Error('Empty decoded audio');}finally{await decoder.close();}}
      result.push({id:figure.dataset.assetId,kind:element.tagName.toLowerCase(),bytes:bytes.length,sha256:sha});
     }return result;
    });
    assert.equal(await page.locator('.manual-media-error').count(),0);
    for(const item of media){if(installed.assetOverrides[item.id])report.locales[pack.locale].overrides.push(item);else report.locales[pack.locale].shared.push(item);}
    report.locales[pack.locale].chapters.push({id:chapter.id,title:chapter.title,media:media.length});console.log(pack.locale+':'+chapter.id+' offline media='+media.length);
   }
   assert.equal(new Set(report.locales[pack.locale].shared.map(item=>item.id)).size,40);
   assert.equal(new Set(report.locales[pack.locale].overrides.map(item=>item.id)).size,343);
   await page.screenshot({path:path.join(spec.out,pack.locale+'-offline-complete.png')});
  }
  assert.equal(report.requests.length,beforeOffline,'fully disconnected reading cannot reach any HTTP resource');
  report.fonts=await page.evaluate(()=>[...document.fonts].filter(face=>face.family.includes('KaTeX')).map(face=>({family:face.family,status:face.status})));assert(!report.fonts.some(face=>face.status==='error'),'formula fonts must render offline');
  report.noNetworkAfterDisconnect=true;report.reads=await page.evaluate(()=>window.manualTest.reads);
  await page.evaluate(()=>window.manualTest.show(false));report.afterUnmount=await page.evaluate(()=>window.manualTest.reads);
  assert.equal(report.afterUnmount.created,report.afterUnmount.revoked,'all object URLs are released on closing help');
  for(const [file,digest]of Object.entries(report.sources))assert.equal(createHash('sha256').update(await fs.readFile(path.join(root,file))).digest('hex'),digest,'source changed during validation: '+file);
  assert.deepEqual(report.errors,[]);report.success=true;
 }catch(error){report.errors.push(error.stack||String(error));if(page){report.body=await page.locator('body').innerText().catch(()=>null);await page.screenshot({path:path.join(spec.out,'failure.png')}).catch(()=>{});}throw error;}
 finally{await context?.close();await chrome?.close();await server?.close();await fs.writeFile(path.join(spec.out,'report.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({success:report.success,report:path.join(spec.out,'report.json')}));}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
