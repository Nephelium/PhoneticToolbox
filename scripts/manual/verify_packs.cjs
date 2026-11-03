const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
const {createHash}=require('node:crypto');
const {createServer:createHttpServer}=require('node:http');
const root=path.resolve(__dirname,'../..');
async function main(){
  const fixture=JSON.parse(await fs.readFile(process.argv[2],'utf8')),report={fixtureOnly:true,success:false,checks:[],errors:[],requests:[],metrics:{}};
  let server,packServer,chrome,browser,page,offline=false;
  const write=()=>fs.writeFile(path.join(fixture.out,'report.json'),JSON.stringify(report,null,2)+'\n');
  try{
    const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
    const harness=`<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><style>html,body,#app{height:100%;margin:0}body{font-family:Arial,sans-serif}</style></head><body><div id="app"></div><script type="module">
      import {createApp} from 'vue';
      import HelpPage from '/src/app/HelpPage.vue';
      import {i18n,setLocale} from '/src/i18n/index.ts';
      import {manualPackStore} from '/src/platform/manual-packs.ts';
      import {ManualSharedCache} from '/src/manual/shared-cache.ts';
      import '/src/design/tokens.css';
      const store=manualPackStore(),reads={json:[],assets:[],created:0,revoked:0,installErrors:[]};
      const cacheInstall=ManualSharedCache.prototype.install;ManualSharedCache.prototype.install=async function(...args){try{return await cacheInstall.apply(this,args)}catch(error){reads.installErrors.push({message:error.message,stack:error.stack});throw error}};
      const install=store.install.bind(store);store.install=async spec=>{try{return await install(spec)}catch(error){reads.installErrors.push({message:error.message,cause:error.cause});throw error}};
      const json=store.readJson.bind(store),asset=store.readAsset.bind(store);
      store.readJson=(locale,path)=>{reads.json.push(locale+':'+path);return json(locale,path)};
      store.readAsset=(locale,path)=>{reads.assets.push(locale+':'+path);return asset(locale,path)};
      const create=URL.createObjectURL.bind(URL),revoke=URL.revokeObjectURL.bind(URL);
      URL.createObjectURL=blob=>{reads.created++;return create(blob)};URL.revokeObjectURL=url=>{reads.revoked++;return revoke(url)};
      window.manualTest={setLocale,store,reads};
      await setLocale(new URLSearchParams(location.search).get('lang')||'zh-CN');
      createApp(HelpPage).use(i18n).mount('#app');
    </script></body></html>`;
    server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{noDiscovery:true,include:['vue']},plugins:[{name:'isolated-manual-fixture',configureServer(vite){vite.middlewares.use(async(req,res,next)=>{
      const pathname=new URL(req.url,'http://localhost').pathname;
      try{
        if(pathname==='/manual-pack-fixture.html'){res.setHeader('Content-Type','text/html');res.end(await vite.transformIndexHtml(pathname,harness));return;}
        if(pathname.startsWith('/manual/')||/^\/(en|ja)-fixture\.zip$/.test(pathname)){
          report.requests.push(pathname);
          if(offline&&(pathname.endsWith('.zip')||pathname.endsWith('language-packs.json'))){res.statusCode=503;res.end('offline fixture');return;}
          const file=path.resolve(fixture.site,'.'+pathname);if(!file.startsWith(path.resolve(fixture.site)+path.sep)){res.statusCode=403;res.end();return;}
          const data=await fs.readFile(file);if(pathname.endsWith('.zip'))report.metrics.served={bytes:data.length,sha256:createHash('sha256').update(data).digest('hex'),expected:fixture.packs.find(item=>item.file===path.basename(pathname))?.sha256,method:req.method,statusBefore:res.statusCode};
          res.statusCode=200;res.setHeader('X-Manual-Fixture','owned-isolated-test');res.setHeader('Content-Type',pathname.endsWith('.json')?'application/json':pathname.endsWith('.zip')?'application/zip':'image/png');res.setHeader('Content-Length',data.length);res.end(data);return;
        }
      }catch(error){res.statusCode=500;res.end(String(error));return;}
      next();
    });}}]});
    await server.listen();const origin=server.resolvedUrls.local[0].replace(/\/$/,'');
    packServer=createHttpServer(async(req,res)=>{
      const pathname=new URL(req.url,'http://localhost').pathname;report.requests.push(pathname);
      res.setHeader('Access-Control-Allow-Origin','*');
      try{
        if(offline){res.statusCode=503;res.end('offline fixture');return;}
        const identity=fixture.packs.find(item=>'/download/'+item.locale===pathname);if(!identity){res.statusCode=404;res.end();return;}
        const data=await fs.readFile(path.join(fixture.site,identity.file));report.metrics.served={bytes:data.length,sha256:createHash('sha256').update(data).digest('hex'),expected:identity.sha256,method:req.method};
        res.writeHead(200,{'Content-Type':'application/octet-stream','Content-Length':data.length});res.end(data);
      }catch(error){res.statusCode=500;res.end(String(error));}
    });
    await new Promise(resolve=>packServer.listen(0,'127.0.0.1',resolve));const packOrigin='http://127.0.0.1:'+packServer.address().port;
    const packs=fixture.packs.map(item=>({...item,url:packOrigin+'/download/'+item.locale}));
    await fs.writeFile(path.join(fixture.site,'manual/language-packs.json'),JSON.stringify({schemaVersion:'ptb-manual-downloads/1',published:true,packs}));
    const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
    chrome=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--mute-audio']});browser=await chrome.newContext({viewport:{width:1440,height:900}});
    report.browser=browser.browser().version();page=await browser.newPage();page.on('pageerror',error=>report.errors.push(error.message));page.on('response',async response=>{if(response.url().includes('/download/'))report.metrics.browserResponse={status:response.status(),headers:response.headers(),bytes:(await response.body()).length};});
    await page.goto(origin+'/manual-pack-fixture.html');await page.locator('.manual-document h1').waitFor();
    assert.match(await page.locator('.manual-document h1').innerText(),/第一章/);
    assert.equal(report.requests.filter(item=>item.endsWith('.zip')).length,0);
    assert.equal(report.requests.filter(item=>item.endsWith('language-packs.json')).length,0);
    report.checks.push('Chinese startup fetches no language ZIP or download manifest');
    await page.evaluate(()=>window.manualTest.setLocale('en'));
    await page.getByRole('button',{name:'Download and install',exact:true}).waitFor();await page.getByRole('button',{name:'Download and install',exact:true}).click();
    await page.waitForFunction(()=>document.querySelector('.manual-document h1')?.textContent.includes('First chapter')||document.querySelector('.manual-pack-error'));
    assert.equal(await page.locator('.manual-pack-error').count(),0,await page.evaluate(()=>JSON.stringify(window.manualTest.reads.installErrors)));
    await page.waitForFunction(()=>document.querySelector('.manual-document img')?.src.startsWith('blob:'));
    await page.locator('.manual-search-label input').fill('fundamental frequency');
    await page.waitForFunction(()=>document.querySelectorAll('.manual-search-results button').length===2);
    await page.locator('.manual-search-label input').fill('');
    await page.screenshot({path:path.join(fixture.out,'en-installed.png')});
    await page.locator('[data-manual-chapter="second"]').click();await page.waitForFunction(()=>document.querySelector('.manual-document h1')?.textContent.includes('Second chapter'));
    await page.locator('[data-manual-chapter="getting-started"]').click();await page.waitForFunction(()=>document.querySelector('.manual-document h1')?.textContent.includes('First chapter'));
    const first=await page.evaluate(()=>({...window.manualTest.reads,json:[...window.manualTest.reads.json],assets:[...window.manualTest.reads.assets]}));
    assert.equal(first.json.filter(item=>item==='en:project.json').length,1);assert.equal(first.json.filter(item=>item==='en:chapters/getting-started.json').length,1);assert.equal(first.json.filter(item=>item==='en:chapters/second.json').length,1);assert.equal(first.assets.length,1);assert.equal(first.created,4);assert.equal(first.revoked,1);
    assert.equal(await page.locator('[data-asset-id=fixture-screen] img').getAttribute('src'),await page.locator('[data-asset-id=fixture-screen-copy] img').getAttribute('src'));
    const shared=await page.evaluate(async()=>{const img=document.querySelector('[data-asset-id=fixture-shared-image] img'),audio=document.querySelector('[data-asset-id=fixture-shared-audio] audio');const raw=await(await fetch(audio.src)).arrayBuffer();return {image:img.src,loaded:img.complete&&img.naturalWidth>0,audio:audio.src,audioBytes:raw.byteLength};});
    assert(shared.image.startsWith('blob:')&&shared.loaded);assert(shared.audio.startsWith('blob:')&&shared.audioBytes>100);assert(report.requests.some(item=>item.includes('/assets/packed/')&&item.endsWith('.webp')));assert(report.requests.some(item=>item.includes('/assets/packed/')&&item.endsWith('.mp3')));report.metrics.packedShared=shared;
    report.checks.push('English ZIP installs from actual HTTP into IndexedDB; translated full-text search, chapter cache, and one lazy screenshot read work');
    await page.evaluate(()=>window.manualTest.setLocale('ja'));await page.getByRole('button',{name:'ダウンロードしてインストール',exact:true}).waitFor();
    await page.getByRole('button',{name:'ダウンロードしてインストール',exact:true}).click();await page.waitForFunction(()=>document.querySelector('.manual-document h1')?.textContent.includes('最初の章'));
    await page.waitForFunction(()=>document.querySelector('.manual-document img')?.src.startsWith('blob:'));await page.screenshot({path:path.join(fixture.out,'ja-installed.png')});
    await page.evaluate(()=>window.manualTest.setLocale('en'));await page.waitForFunction(()=>document.querySelector('.manual-document h1')?.textContent.includes('First chapter'));
    const repeated=await page.evaluate(()=>({...window.manualTest.reads,json:[...window.manualTest.reads.json],assets:[...window.manualTest.reads.assets]}));
    assert.equal(repeated.json.filter(item=>item==='en:project.json').length,1);assert.equal(repeated.json.filter(item=>item==='en:chapters/getting-started.json').length,1);assert(repeated.revoked>=2);
    assert.equal(report.requests.filter(item=>item==='/download/en').length,1);assert.equal(report.requests.filter(item=>item==='/download/ja').length,1);
    report.checks.push('Japanese installs separately; repeated language changes reuse versioned indexes/chapters and revoke screenshot URLs');
    offline=true;await page.goto(origin+'/manual-pack-fixture.html?lang=en');await page.waitForFunction(()=>document.querySelector('.manual-document h1')?.textContent.includes('First chapter'));await page.waitForFunction(()=>document.querySelector('.manual-document img')?.src.startsWith('blob:'));
    assert.equal(report.requests.filter(item=>item==='/download/en').length,1);assert.equal(report.requests.filter(item=>item==='/download/ja').length,1);
    report.checks.push('After page reload, installed English text and screenshot read offline while ZIP/feed endpoints return 503');
    report.metrics={...report.metrics,first,repeated,offline:await page.evaluate(()=>window.manualTest.reads)};
    // The already-open page is now fully disconnected, including every shared
    // image/audio endpoint. The second chapter was not opened since this reload.
    await page.evaluate(()=>window.manualTest.setLocale('ja'));await page.waitForFunction(()=>document.querySelector('.manual-document h1')?.textContent.includes('最初の章'));
    await page.evaluate(()=>window.manualTest.setLocale('en'));await page.waitForFunction(()=>document.querySelector('.manual-document h1')?.textContent.includes('First chapter'));
    await browser.setOffline(true);
    await page.locator('[data-manual-chapter="second"]').click();await page.waitForFunction(()=>document.querySelector('.manual-document h1')?.textContent.includes('Second chapter'));
    await page.waitForFunction(()=>document.querySelector('[data-asset-id=fixture-shared-image] img')?.naturalWidth>0);
    const disconnected=await page.evaluate(async()=>{const audio=document.querySelector('[data-asset-id=fixture-shared-audio] audio');return {audio:audio.src,audioBytes:(await(await fetch(audio.src)).arrayBuffer()).byteLength,online:navigator.onLine};});
    assert.equal(disconnected.online,false);assert(disconnected.audio.startsWith('blob:')&&disconnected.audioBytes>100);
    await page.evaluate(()=>window.manualTest.setLocale('ja'));await page.waitForFunction(()=>document.querySelector('.manual-document h1')?.textContent.includes('次の章'));
    await page.screenshot({path:path.join(fixture.out,'ja-fully-disconnected.png')});
    report.metrics.fullyDisconnected=disconnected;report.checks.push('With the whole browser context offline, an unread chapter and Japanese switch load text plus shared WebP/MP3 from verified caches; no PWA/cold website startup is claimed');
    await browser.close();browser=undefined;
    // A second owned profile has no installed packs and no published feed. The
    // user-selected ZIP path must still work for this local-only delivery.
    offline=false;
    await fs.writeFile(path.join(fixture.site,'manual/language-packs.json'),JSON.stringify({schemaVersion:'ptb-manual-downloads/1',published:false,packs:[]}));
    const downloadsBeforeLocal=report.requests.filter(item=>item.startsWith('/download/')).length;
    browser=await chrome.newContext({viewport:{width:1440,height:900}});
    page=await browser.newPage();page.on('pageerror',error=>report.errors.push(error.message));
    await page.goto(origin+'/manual-pack-fixture.html?lang=en');
    await page.getByText('This manual language pack has not been published. The Chinese manual is shown.',{exact:true}).waitFor();
    assert.equal(await page.getByRole('button',{name:'Download and install',exact:true}).count(),0);
    const enChooser=page.waitForEvent('filechooser');
    await page.getByRole('button',{name:'Install a downloaded ZIP',exact:true}).click();
    await (await enChooser).setFiles(path.join(fixture.site,'en-fixture.zip'));
    await page.waitForFunction(()=>document.querySelector('.manual-document h1')?.textContent.includes('First chapter')||document.querySelector('.manual-pack-error'));
    assert.equal(await page.locator('.manual-pack-error').count(),0);await page.waitForFunction(()=>document.querySelector('.manual-document img')?.src.startsWith('blob:'));
    await page.screenshot({path:path.join(fixture.out,'en-unpublished-local-import.png')});
    await page.evaluate(()=>window.manualTest.setLocale('ja'));
    const jaChooser=page.waitForEvent('filechooser');
    await page.getByRole('button',{name:'ダウンロード済みZIPをインストール',exact:true}).click();
    await (await jaChooser).setFiles(path.join(fixture.site,'ja-fixture.zip'));
    await page.waitForFunction(()=>document.querySelector('.manual-document h1')?.textContent.includes('最初の章')||document.querySelector('.manual-pack-error'));
    assert.equal(await page.locator('.manual-pack-error').count(),0);await page.waitForFunction(()=>document.querySelector('.manual-document img')?.src.startsWith('blob:'));
    await page.evaluate(()=>window.manualTest.setLocale('en'));await page.waitForFunction(()=>document.querySelector('.manual-document h1')?.textContent.includes('First chapter'));
    assert.equal(report.requests.filter(item=>item.startsWith('/download/')).length,downloadsBeforeLocal);
    report.metrics.unpublishedLocalImport=await page.evaluate(()=>window.manualTest.reads);
    offline=true;await page.goto(origin+'/manual-pack-fixture.html?lang=ja');await page.waitForFunction(()=>document.querySelector('.manual-document h1')?.textContent.includes('最初の章'));await page.waitForFunction(()=>document.querySelector('.manual-document img')?.src.startsWith('blob:'));
    offline=false;await page.goto(origin+'/manual-pack-fixture.html?lang=en');await page.waitForFunction(()=>document.querySelector('.manual-document h1')?.textContent.includes('First chapter'));
    const badChooser=page.waitForEvent('filechooser');await page.getByRole('button',{name:'Install a downloaded ZIP',exact:true}).click();await(await badChooser).setFiles(path.join(fixture.site,'bad-pack.zip'));
    await page.locator('.manual-pack-error').waitFor();const errorEn=await page.locator('.manual-pack-error').innerText(),beforeErrors=report.requests.filter(item=>item.startsWith('/download/')).length;
    await page.evaluate(()=>window.manualTest.setLocale('ja'));await page.waitForFunction(()=>document.querySelector('.manual-pack-error')?.textContent.includes('パック'));
    const errorJa=await page.locator('.manual-pack-error').innerText();assert.notEqual(errorEn,errorJa);assert.equal(report.requests.filter(item=>item.startsWith('/download/')).length,beforeErrors);report.metrics.retainedError={en:errorEn,ja:errorJa};
    await page.evaluate(()=>window.manualTest.setLocale('en'));await page.waitForFunction(()=>document.querySelector('.manual-document h1')?.textContent.includes('First chapter'));
    const beforeUpdate=await page.evaluate(()=>window.manualTest.reads.json.filter(item=>item==='en:chapters/getting-started.json').length);
    const updateChooser=page.waitForEvent('filechooser');await page.getByRole('button',{name:'Install a downloaded ZIP',exact:true}).click();await(await updateChooser).setFiles(path.join(fixture.site,'en-update.zip'));
    await page.waitForFunction(()=>document.querySelector('.manual-document')?.textContent.includes('Fundamental frequency remains 100 Hz.'));
    assert((await page.evaluate(()=>window.manualTest.reads.json.filter(item=>item==='en:chapters/getting-started.json').length))>beforeUpdate);
    report.checks.push('Installing a translation-only new pack version replaces mounted chapter text despite unchanged Chinese source and manual version');
    // Corrupt only the owned MP3 cache. Reopening checks its real digest and
    // exposes a repair button; the explicit retry replaces this one payload.
    await page.evaluate(async()=>{const cache=await caches.open('phonetic-toolbox-manual-shared-v1');for(const key of await cache.keys()){const response=await cache.match(key);if(response.headers.get('Content-Type')?.includes('audio')){await cache.put(key,new Response('corrupt cached audio'));break;}}});
    const beforeRepair=report.requests.filter(item=>item.endsWith('.mp3')).length;
    await page.reload();await page.getByRole('button',{name:'Save shared media for offline reading',exact:true}).waitFor();
    await page.getByRole('button',{name:'Save shared media for offline reading',exact:true}).click();
    await page.waitForFunction(()=>document.querySelector('.manual-language-toolbar')?.textContent.includes('Installed for offline reading'));
    assert.equal(report.requests.filter(item=>item.endsWith('.mp3')).length,beforeRepair+1);
    await page.waitForFunction(()=>document.querySelector('[data-asset-id=fixture-shared-audio] audio')?.src.startsWith('blob:'));
    assert(await page.evaluate(async()=>{const audio=document.querySelector('[data-asset-id=fixture-shared-audio] audio');return (await(await fetch(audio.src)).arrayBuffer()).byteLength>100;}));
    report.checks.push('A real corrupted shared MP3 cache is detected after reopening; explicit UI repair re-downloads only that payload and restores blob reading');
    report.checks.push('Real WebP/MP3 host paths resolve by stable asset ID; duplicate PNG IDs perform one read and share one URL; leaving a chapter revokes and return reuses bytes; retained bad-ZIP error changes language without retrying installation');
    report.checks.push('With published:false and no downloads, explicit local English/Japanese ZIP installation, language switching, and offline reload succeed in a fresh IndexedDB profile');
    assert.deepEqual(report.errors,[]);report.success=true;
  }catch(error){report.errors.push(error.stack||String(error));if(page){report.body=await page.locator('body').innerText().catch(()=>null);report.metrics.failure=await page.evaluate(()=>window.manualTest?.reads).catch(()=>null);await page.screenshot({path:path.join(fixture.out,'failure.png')}).catch(()=>{});}throw error;}
  finally{await browser?.close();await chrome?.close();await server?.close();if(packServer)await new Promise((resolve,reject)=>packServer.close(error=>error?reject(error):resolve()));await write();console.log(JSON.stringify({success:report.success,fixtureOnly:true,report:path.join(fixture.out,'report.json')}));}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
