// Actual frozen authenticated application. Only public UI/DOM/media are used.
const fs=require('node:fs/promises'),path=require('node:path'),os=require('node:os'),assert=require('node:assert/strict'),{createHash}=require('node:crypto');
const root=path.resolve(__dirname,'../..');
async function main(){
 const [origin,out,scratch,fixturePath]=process.argv.slice(2);assert.equal(path.dirname(scratch),out);
 assert(process.env.PTB_VERIFY_FRONTEND_DIST,'An explicit frozen frontend is required');
 const fixture=JSON.parse(await fs.readFile(fixturePath,'utf8'));
 const report={success:false,stage:'frozen-authenticated-Web-candidate',scope:'Already-open real website, login/home/help/local ZIP controls; all chapter/media offline plus real help close/reopen. No offline website navigation claim.',explicitCandidate:process.env.PTB_VERIFY_FRONTEND_DIST,sourceDigest:fixture.sourceDigest,testSources:fixture.testSources,locales:{},checks:[],errors:[],requests:[],failedRequests:[]};
 const catalogs={};for(const code of ['zh-CN','en','ja']){
  catalogs[code]={};for(const ns of ['account','workbench','components','manual'])Object.assign(catalogs[code],JSON.parse(await fs.readFile(path.join(root,'frontend/src/i18n/locales',code,ns+'.json'),'utf8')));
 }
 const {chromium}=require(path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 let chrome,context,page,offline=false,probingUrls=false;
 const seenUrls=new Set();
 const currentLocale=async()=>await page.locator('html').getAttribute('lang');
 const text=async key=>catalogs[await currentLocale()][key];
 const manual=()=>page.locator('.manual-help-page');
 async function changeUiLanguage(code){
  await page.waitForFunction(()=>!!document.querySelector('.language-switch select:not(:disabled)'));
  await page.locator('.language-switch select').first().selectOption(code);
  await page.waitForFunction(c=>document.documentElement.lang===c&&!document.querySelector('.language-switch select:disabled'),code);
  await page.waitForFunction(c=>document.documentElement.dataset.fontLocale===c&&document.fonts.status==='loaded',code);
 }
 async function openHelp(){
  await page.locator('.utility-grid').getByRole('button',{name:await text('workbench.user_guide'),exact:true}).click();
  await manual().waitFor({state:'visible'});
  await page.waitForFunction(()=>document.querySelector('#tab-help')?.getAttribute('aria-selected')==='true'&&!document.querySelector('.manual-start-state[role=status]'));
  assert.equal(await manual().locator('.manual-start-state[role=alert]').count(),0);
 }
 async function closeHelp(){
  await page.locator('.tab-wrap:has(#tab-help) .tab-close').click();
  await manual().waitFor({state:'detached'});assert.equal(await page.locator('#tab-home').getAttribute('aria-selected'),'true');
 }
 async function installed(pack){
  await page.waitForFunction(({version,offlineLabel})=>{
   const input=document.querySelector('.manual-language-toolbar input[type=file]'),toolbar=document.querySelector('.manual-language-toolbar');
   return !!document.querySelector('.manual-pack-error')||!!(input&&!input.disabled&&toolbar?.textContent.includes(version)&&toolbar.textContent.includes(offlineLabel)&&!document.querySelector('.manual-start-state[role=status]'));
  },{version:pack.version,offlineLabel:catalogs[pack.locale]['manual.reader.pack.offline'].split('{version}')[0].trim()},{timeout:120000});
  assert.equal(await manual().locator('.manual-pack-error').count(),0,await manual().innerText());
  assert.equal(await manual().locator('.manual-load-error').count(),0);
 }
 async function auditChapter(pack,chapter){
  await manual().locator('button[data-manual-chapter="'+chapter.id+'"]').click();
  await page.waitForFunction(({id,title})=>document.querySelector('.manual-document')?.getAttribute('data-manual-chapter')===id&&document.querySelector('.manual-document h1')?.textContent.includes(title),chapter,{timeout:30000});
  await page.waitForFunction(expected=>{
   const article=document.querySelector('.manual-document');return article&&article.querySelectorAll('figure[data-asset-id]').length===expected&&!article.querySelector('.manual-loading');
  },chapter.media.length,{timeout:30000});
  const actual=await page.evaluate(async()=>{
   const result=[];let decoder;
   try{
    for(const figure of document.querySelectorAll('.manual-document figure[data-asset-id]')){
     const element=figure.querySelector('img,audio,video');if(!element)throw Error('Missing media: '+figure.dataset.assetId);
     if(!element.src.startsWith('blob:'))throw Error('Media did not resolve offline: '+figure.dataset.assetId+' '+element.src);
     const response=await fetch(element.src);if(!response.ok)throw Error('Unreadable media blob: '+figure.dataset.assetId);
     const bytes=await response.arrayBuffer();const sha256=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');
     const kind=element.tagName.toLowerCase()==='img'?'image':element.tagName.toLowerCase();let decoded;
     if(kind==='image'){const bitmap=await createImageBitmap(new Blob([bytes]));decoded={width:bitmap.width,height:bitmap.height};bitmap.close();}
     if(kind==='audio'){decoder??=new AudioContext();const audio=await decoder.decodeAudioData(bytes.slice(0));decoded={channels:audio.numberOfChannels,sampleRate:audio.sampleRate,frames:audio.length};}
     if(kind==='video'){
      if(element.readyState<1)await new Promise((resolve,reject)=>{
       const timer=setTimeout(()=>finish(Error('Video metadata timeout')),15000);
       const done=()=>finish(),failed=()=>finish(Error('Video metadata could not be decoded'));
       function finish(error){clearTimeout(timer);element.removeEventListener('loadedmetadata',done);element.removeEventListener('error',failed);error?reject(error):resolve();}
       element.addEventListener('loadedmetadata',done);element.addEventListener('error',failed);element.preload='metadata';element.load();
      });
      if(!Number.isFinite(element.duration)||element.duration<=0||element.videoWidth<=0||element.videoHeight<=0)throw Error('Invalid video metadata');
      decoded={duration:element.duration,width:element.videoWidth,height:element.videoHeight};
     }
     result.push({id:figure.dataset.assetId,kind,sha256,bytes:bytes.byteLength,decoded,url:element.src});
    }
   }finally{await decoder?.close();}
   return result;
  });
  const identities=items=>items.map(item=>item.id+':'+item.kind).sort();
  assert.deepEqual(identities(actual),identities(chapter.media),'rendered media identity/multiplicity must match ZIP body: '+chapter.id);
  const payloadUrls=new Map();
  for(const item of actual){
   assert.equal(item.sha256,pack.assets[item.id].sha256,'real decoded payload digest: '+item.id);seenUrls.add(item.url);
   assert(item.bytes>0);const same=payloadUrls.get(item.sha256);if(same)assert.equal(item.url,same,'same payload in one chapter shares its object URL');payloadUrls.set(item.sha256,item.url);
  }
  assert.equal(await manual().locator('.manual-media-error').count(),0);
  console.log(pack.locale+':'+chapter.id+' offline media='+actual.length);
  return {id:chapter.id,title:chapter.title,media:actual.map(({url,...item})=>item)};
 }
 try{
  // Chrome owns a short temporary profile and removes it on close. Deep Windows
  // scratch/profile paths exceed CacheStorage's native path length budget.
  chrome=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--mute-audio']});context=await chrome.newContext({viewport:{width:1920,height:1080},locale:'zh-CN'});report.browser=chrome.version();
  await context.route('https://www.phonetictoolbox.com/announcements/v1.json',route=>route.fulfill({status:200,contentType:'application/json',body:JSON.stringify({schema:'ptb-announcements/1',items:[{id:'isolated-language-check',title:'Language QA notice',body:'测试公告原文保持。',publishedAt:'2026-10-01T00:00:00Z',platforms:['web']}]})}));
  page=await context.newPage();page.setDefaultTimeout(30000);page.on('pageerror',error=>report.errors.push(error.stack||error.message));page.on('dialog',dialog=>void dialog.accept());
  page.on('request',request=>{if(!probingUrls)report.requests.push({offline,method:request.method(),url:request.url()});});
  page.on('requestfailed',request=>{if(!probingUrls)report.failedRequests.push({offline,url:request.url(),failure:request.failure()});});
  await page.goto(origin+'/server/');await page.locator('input[autocomplete=username]').fill('synthetic');await page.locator('input[autocomplete=current-password]').fill('synthetic-password-only');
  await page.getByRole('button',{name:await text('account.log_in'),exact:true}).click();await page.locator('#tool-sidebar').waitFor();
  await page.locator('dialog[open] .announcements-panel').waitFor();assert((await page.locator('dialog[open] .announcements-panel').innerText()).includes('测试公告原文保持。'));
  await page.locator('dialog[open]').getByRole('button',{name:await text('components.got_it'),exact:true}).click();
  assert.equal(await page.locator('#tab-home').getAttribute('aria-selected'),'true');await openHelp();
  await manual().locator('.manual-document h1').waitFor();
  assert(!report.requests.some(request=>/\.zip(?:\?|$)|language-packs\.json/.test(request.url)),'Chinese startup must not download foreign resources');
  report.checks.push('Real synthetic account login, original notice acknowledgement and home-to-help controls on the explicit frozen candidate');
  for(const pack of fixture.packs){
   await changeUiLanguage(pack.locale);
   await manual().getByText(catalogs[pack.locale]['manual.reader.pack.unpublished'],{exact:true}).waitFor();
   assert.equal(await manual().getByRole('button',{name:catalogs[pack.locale]['manual.reader.pack.download'],exact:true}).count(),0);
   const chooser=page.waitForEvent('filechooser');await manual().getByRole('button',{name:catalogs[pack.locale]['manual.reader.pack.import'],exact:true}).click();await(await chooser).setFiles(pack.path);
   await installed(pack);assert.equal(await manual().locator('.manual-language-toolbar select').inputValue(),pack.locale);
   report.locales[pack.locale]={zip:{path:pack.path,sha256:pack.sha256,bytes:pack.bytes,version:pack.version},chapters:[]};
   await page.screenshot({path:path.join(out,pack.locale+'-installed.png')});
  }
  report.checks.push('Both actual complete local ZIPs install via the real picker while the production feed is unpublished; shared media is prepared by the product');
  offline=true;await context.setOffline(true);assert.equal(await page.evaluate(()=>navigator.onLine),false);
  for(const pack of fixture.packs){
   await changeUiLanguage(pack.locale);await closeHelp();await openHelp();await installed(pack);assert.equal(await manual().locator('.manual-language-toolbar select').inputValue(),pack.locale);
   assert.equal(await manual().locator('button[data-manual-chapter]').count(),22);
   for(const chapter of pack.chapters)report.locales[pack.locale].chapters.push(await auditChapter(pack,chapter));
   const media=report.locales[pack.locale].chapters.flatMap(chapter=>chapter.media),ids=new Set(media.map(item=>item.id));
   assert.deepEqual([...ids].sort(),Object.keys(pack.assets).sort());
   for(const [kind,count]of [['image',348],['audio',34],['video',1]])assert.equal(new Set(media.filter(item=>item.kind===kind).map(item=>item.id)).size,count);
   assert.equal([...ids].filter(id=>pack.assets[id].override).length,343);
   report.locales[pack.locale].counts={chapters:22,overrides:343,shared:40,audioDecoded:34,videoMetadata:1};
   await page.screenshot({path:path.join(out,pack.locale+'-offline-last-chapter.png')});
  }
  await closeHelp();probingUrls=true;
  const live=await page.evaluate(async urls=>{const result=[];for(const url of urls){try{await fetch(url);result.push(url);}catch{}}return result;},[...seenUrls]);
  assert.deepEqual(live,[],'closing the actual help tab must release all observed object URLs');report.urlLifetime={observed:seenUrls.size,unavailableAfterClose:seenUrls.size};
  report.checks.push('With the entire browser context offline, each language reopens actual help and renders every ZIP chapter and asset; images and 34 audio examples decode, video metadata loads, and payload hashes match independent ZIP/host expectations');
  const offlineManualHttp=report.requests.filter(request=>request.offline&&new URL(request.url).pathname.startsWith('/server/manual/'));
  assert(offlineManualHttp.every(request=>request.url.endsWith('/project.json')),'unread chapter/media must use installed storage, not HTTP fallback');
  report.offlineManualHttp=offlineManualHttp;assert.deepEqual(report.errors,[]);
  for(const [name,digest]of Object.entries(fixture.testSources))assert.equal(createHash('sha256').update(await fs.readFile(path.join(root,name))).digest('hex'),digest,'test source changed during verification');
  report.success=true;
 }catch(error){report.errors.push(error.stack||String(error));if(page){report.body=await page.locator('body').innerText().catch(()=>null);await page.screenshot({path:path.join(out,'failure.png')}).catch(()=>{});}throw error;}
 finally{await context?.close();await chrome?.close();await fs.writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({success:report.success,report:path.join(out,'report.json')}));}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
