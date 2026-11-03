// Focused multilingual landing-page checks in a task-owned Chrome profile.
const fs=require('node:fs/promises'),path=require('node:path'),http=require('node:http');
const assert=require('node:assert/strict'),{spawnSync}=require('node:child_process'),{createHash}=require('node:crypto');
const root=path.resolve(__dirname,'..'),site=process.env.WEBSITE_DIRECTORY||path.join(root,'temporary-site');
const publicOrigin=process.env.WEBSITE_ORIGIN,readOnly=Boolean(publicOrigin||process.env.WEBSITE_READONLY);
const userAgent='Mozilla/5.0 PTB-Website-Verify';
const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
async function main(){
 const out=path.join(root,'output/validation/website','languages-'+Date.now()),scratch=path.join(out,'scratch');
 await fs.mkdir(scratch,{recursive:true});
 const write=(name,value)=>fs.writeFile(path.join(out,name),JSON.stringify(value,null,2)+'\n');
 const marker={schema:'ptb-test-run/1',owner_pid:process.pid,state:'active',started_at:new Date().toISOString(),recipe:'Multilingual website: shared media, static text, downloads, language/anchor navigation and owned Chrome; production checks are read-only'};
 await write('run.json',marker);
 let server,browser,page;const report={success:false,checks:[],errors:[],requests:[]};
 try{
  if(!publicOrigin){server=http.createServer(async(req,res)=>{
   try{
    const url=new URL(req.url,'http://localhost');
    if(url.pathname==='/site-stats-history.json'){const now=Math.floor(Date.now()/1000);return res.writeHead(200,{'Content-Type':'application/json'}).end(JSON.stringify({schema:1,points:[[now-3600,1200,50,20,19,1,2],[now,1234,56,23,20,2,3]]}));}
    if(url.pathname==='/site-stats.json')return res.writeHead(200,{'Content-Type':'application/json'}).end(JSON.stringify({schema:1,startedAt:'2026-10-12T00:00:00Z',updatedAt:new Date().toISOString(),pageViews:1234,webUses:23,downloads:{total:61,portable:40,installer:16,macos:2,ime:3},github:{count:20}}));
    if(url.pathname.startsWith('/api/'))return res.writeHead(400,{'Content-Type':'application/json'}).end(JSON.stringify({detail:'invalid_email'}));
    const pathname=process.env.WEBSITE_DIRECTORY?url.pathname.replace(/^\/site-releases\/r[0-9a-f]{16}\//,'/'):url.pathname;
    const file=path.resolve(site,'.'+decodeURIComponent(pathname)+(pathname.endsWith('/')?'index.html':''));
    assert(file.startsWith(site+path.sep));const data=await fs.readFile(file);
    const types={'.html':'text/html; charset=utf-8','.js':'application/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.svg':'image/svg+xml','.png':'image/png','.jpg':'image/jpeg','.ttf':'font/ttf','.woff2':'font/woff2','.wav':'audio/wav','.mp4':'video/mp4','.json':'application/json'};
    const headers={'Content-Type':types[path.extname(file)]||'application/octet-stream','Accept-Ranges':'bytes'};
    if(req.headers.range){const match=/^bytes=(\d+)-(\d*)$/.exec(req.headers.range);assert(match);const start=+match[1],end=match[2]?Math.min(+match[2],data.length-1):data.length-1;if(start>=data.length)return res.writeHead(416,{'Content-Range':`bytes */${data.length}`}).end();res.writeHead(206,{...headers,'Content-Range':`bytes ${start}-${end}/${data.length}`,'Content-Length':end-start+1}).end(data.subarray(start,end+1));}
    else res.writeHead(200,{...headers,'Content-Length':data.length}).end(data);
   }catch(error){res.writeHead(404).end();}
  });
  // This host can allocate low ephemeral ports that browsers intentionally ban.
  // Use an ordinary high loopback port; never disable browser port protection.
  for(let attempt=0;attempt<32;attempt++){
   try{await new Promise((resolve,reject)=>{const failed=error=>reject(error);server.once('error',failed);server.listen(20000+Math.floor(Math.random()*30000),'127.0.0.1',()=>{server.off('error',failed);resolve();});});break;}
   catch(error){if(!['EACCES','EADDRINUSE'].includes(error.code)||attempt===31)throw error;}
  }
  }
  const origin=publicOrigin||'http://127.0.0.1:'+server.address().port+'/';report.origin=origin;report.readOnly=readOnly;
  browser=await chromium.launchPersistentContext(path.join(scratch,'profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1600,height:1000},reducedMotion:'reduce',userAgent});
  report.browser=browser.browser().version();page=await browser.newPage();
  page.on('pageerror',error=>report.errors.push(error.message));
  if(readOnly)page.on('request',request=>{if(!['GET','HEAD'].includes(request.method()))report.errors.push('Unexpected write request: '+request.method()+' '+request.url());});
  page.on('response',response=>{if(response.status()>=400&&!response.url().includes('/api/v1/feedback/'))report.requests.push({url:response.url(),status:response.status()});});
  const translations=(await fs.readdir(path.join(root,'temporary-site/locales'))).filter(name=>name.endsWith('.json')).map(name=>name.slice(0,-5));
  async function chooseLanguage(language){await page.locator('.language-switch').click();await page.locator('.language-option[hreflang="'+language+'"]').click();}
  for(const language of ['zh',...translations]){
   const address=origin+(language==='zh'?'':language+'/');
   await page.goto(address);await page.evaluate(()=>document.fonts.ready);
   assert.equal(await page.locator('html').getAttribute('lang'),language==='zh'?'zh-CN':language);
   assert.equal(await page.locator('.language-switch').count(),1);
   assert.equal(await page.locator('.language-option').count(),translations.length+1);
   assert.equal(await page.locator('.download.primary').getAttribute('href'),'/releases/windows-x64/'+JSON.parse(await fs.readFile(path.join(site,'assets/downloads.json'),'utf8')).version+'/PhoneticToolbox-windows-x64.exe');
   assert.equal(await page.locator('.download.macos').getAttribute('href'),'/releases/macos-arm64/3.1.1/PhoneticToolbox-3.1.1-macos14-arm64-preview.zip');
   assert((await page.locator('.ime-platform').first().getAttribute('href')).includes('/0.1.7/'));
   assert.equal(await page.locator('[data-screen]').count(),34);
   assert.equal(await page.locator('[data-release-tab]').count(),9);
   for(const width of [1920,1600,1551,1550,1440,1280,1200,1100,1024,900,850,801,800,768,600,480,390,360,320]){
    await page.setViewportSize({width,height:1000});await page.evaluate(()=>scrollTo(0,0));await page.waitForTimeout(70);
    const layout=await page.evaluate(()=>{
     const rect=e=>{const r=e.getBoundingClientRect();return {x:r.x,y:r.y,right:r.right,bottom:r.bottom,width:r.width,height:r.height};};
     return {width:innerWidth,mode:document.querySelector('.site-header').dataset.layout,scroll:document.documentElement.scrollWidth,header:rect(document.querySelector('.site-header')),brand:rect(document.querySelector('.brand')),language:rect(document.querySelector('.language-switch')),github:rect(document.querySelector('.header-actions .header-github')),actions:rect(document.querySelector('.header-actions')),links:[...document.querySelectorAll('.site-header nav>a')].map(rect),downloads:[...document.querySelectorAll('.download,.web-entry')].map(rect)};
    });
    assert(layout.scroll<=width+1,JSON.stringify({language,width,layout}));
    for(const box of [layout.language,layout.github,...layout.links,...layout.downloads])assert(box.x>=-1&&box.right<=width+1&&box.width>0,JSON.stringify({language,width,box}));
    assert(layout.brand.right<=layout.actions.x+1,JSON.stringify({language,width,layout}));
    assert(Math.abs((layout.github.y+layout.github.height/2)-(layout.language.y+layout.language.height/2))<2,JSON.stringify({language,width,layout}));
    assert.equal(layout.links.length,8);
    if(layout.mode==='stacked')assert(layout.links.every(box=>box.y>=layout.actions.bottom),JSON.stringify({language,width,layout}));
    else assert(layout.links.every(box=>box.right<=layout.actions.x&&box.x>=layout.brand.right),JSON.stringify({language,width,layout}));
    if(width<=800)assert.equal(new Set(layout.links.map(box=>Math.round(box.y))).size,2);
    if(width>800)assert(Math.max(...layout.links.map(box=>box.y))<Math.min(...layout.links.map(box=>box.bottom)),JSON.stringify({language,width,layout}));
    if(width>800&&layout.mode==='stacked')assert(layout.links.every(box=>box.width<190),JSON.stringify({language,width,layout}));
    if(language==='ja'){
     const title=await page.locator('.ja-heading').evaluate(element=>{
      const copy=element.closest('.hero-copy').getBoundingClientRect();const spans=[...element.childNodes].filter(node=>node.nodeType===Node.TEXT_NODE||node.nodeName==='EM').map(node=>{const range=document.createRange();range.selectNodeContents(node);const rect=range.getBoundingClientRect();return {left:rect.left,right:rect.right,lines:range.getClientRects().length};});
      return {font:parseFloat(getComputedStyle(element).fontSize),copy:{left:copy.left,right:copy.right},spans};
     });
     assert(title.font>=40);assert(title.spans.length===2);
     assert(title.spans.every(span=>span.lines===1&&span.left>=title.copy.left-2&&span.right<=title.copy.right+2),JSON.stringify({width,title}));
    }
    if([1600,1200,1024,800,390,320].includes(width))await page.screenshot({path:path.join(out,language+'-home-'+width+'.png'),animations:'disabled'});
   }
   report.checks.push(language+': nineteen viewport widths, grouped GitHub/language actions, balanced eight-link navigation, download versions, 34 screenshots and nine releases');
   for(const width of [1200,800,390,320]){
    await page.setViewportSize({width,height:1000});await page.evaluate(()=>scrollTo(0,0));await page.locator('.language-switch').click();
    const menu=await page.locator('.language-options').boundingBox();assert(menu.x>=0&&menu.x+menu.width<=width+1,JSON.stringify({language,width,menu}));
    if(width===390)await page.screenshot({path:path.join(out,language+'-language-menu.png'),animations:'disabled'});
    await page.keyboard.press('Escape');
   }
  }
  if(readOnly){
   for(const from of ['zh-CN',...translations])for(const to of ['zh-CN',...translations]){
    await page.goto(origin+(from==='zh-CN'?'':from+'/')+'#methods');await chooseLanguage(to);
    await page.waitForFunction(language=>document.documentElement.lang===language&&!location.hash&&scrollY<2,to);
    assert.equal(new URL(page.url()).hash,'');assert((await page.evaluate(()=>scrollY))<2);
   }
   const defaultContext=await browser.browser().newContext({locale:'ja-JP',viewport:{width:1200,height:1000},userAgent});
   await defaultContext.addInitScript(()=>localStorage.setItem('ptb-website-language','en'));
   const defaultPage=await defaultContext.newPage();await defaultPage.goto(origin);await defaultPage.waitForTimeout(200);
   assert.equal(await defaultPage.locator('html').getAttribute('lang'),'zh-CN');assert.equal(new URL(defaultPage.url()).pathname,'/');
   await defaultContext.close();
   report.checks.push('Default root remains Simplified Chinese with Japanese browser locale and saved English preference; nine language selections clear the old section and start at the page top');
   await page.goto(origin+'ja/');await page.evaluate(()=>document.fonts.ready);
   const fonts=await page.locator('#next .next-comparison-heading p,#next .next-comparison-heading strong,.author-note p,.author-note strong').evaluateAll(elements=>elements.map(element=>getComputedStyle(element).fontFamily));
   assert(fonts.length>5&&fonts.every(font=>font.includes('Yu Gothic')&&!/Whisper|Playful|KuaiLe/.test(font)));
   await page.locator('#faq details').nth(1).locator('summary').click();assert((await page.locator('#faq').innerText()).includes('公開ベータテスト'));
   const audio=page.locator('audio').first();await audio.evaluate(element=>element.play());await page.waitForFunction(()=>document.querySelector('audio').currentTime>0);await audio.evaluate(element=>element.pause());
   for(const width of [1600,390]){await page.setViewportSize({width,height:1000});for(const section of ['a-small-note','next'])await page.locator('#'+section).screenshot({path:path.join(out,'ja-'+section+'-'+width+'.png'),animations:'disabled'});}
   report.checks.push('Japanese font consistency, current public-beta FAQ and real published WAV decoding/playback');
   for(const language of ['zh',...translations]){
    const context=await browser.browser().newContext({javaScriptEnabled:false,viewport:{width:390,height:844},userAgent}),np=await context.newPage();
    await np.goto(origin+(language==='zh'?'':language+'/'));
    assert.equal(await np.locator('html').getAttribute('lang'),language==='zh'?'zh-CN':language);
    await np.locator('.language-switch').click();assert(await np.locator('.language-option').first().isVisible());assert.equal(await np.locator('[data-screen]').count(),34);await context.close();
   }
   report.checks.push('All three published routes remain readable and the language dropdown works without JavaScript; no API writes or external feedback sent');
   assert.deepEqual(report.errors,[]);assert.deepEqual(report.requests,[]);report.success=true;return;
  }
  await page.setViewportSize({width:1600,height:1000});await page.goto(origin+'en/#methods');
  await page.waitForFunction(()=>document.querySelector('[data-stat="views"]').textContent==='1,234');
  const slider=page.locator('[data-chart="views"] input[type=range]');await slider.focus();await page.keyboard.press('Home');assert((await page.locator('[data-chart="views"] output').innerText()).length>0);await page.keyboard.press('End');
  report.checks.push('English number formatting and shared history-slider keyboard navigation use consistent local fixtures');
  await chooseLanguage('zh-CN');await page.waitForURL('**/index.html');assert.equal(await page.locator('html').getAttribute('lang'),'zh-CN');assert((await page.evaluate(()=>scrollY))<2);
  await chooseLanguage('en');await page.waitForURL('**/en/index.html');assert.equal(await page.locator('html').getAttribute('lang'),'en');assert((await page.evaluate(()=>scrollY))<2);
  await page.locator('.language-switch').focus();await page.keyboard.press('ArrowDown');assert(await page.locator('.language-picker').getAttribute('open')!==null);await page.keyboard.press('Escape');assert.equal(await page.locator('.language-picker').getAttribute('open'),null);
  report.checks.push('Language round-trip uses real static links, clears the old hash and starts at the top');
  const catalog=page.locator('.reference-catalog').first();await catalog.locator('summary').click();
  const search=catalog.locator('input[type=search]');await search.fill('no-result-phonetics-fixture');assert((await catalog.innerText()).includes('No matching sources'));
  await search.fill('');await catalog.locator('summary').click();
  await page.locator('[data-release-tab]').nth(1).click();assert.equal(await page.locator('[data-release-tab]').nth(1).getAttribute('aria-selected'),'true');
  await page.locator('#feedback-email').fill('research@example.invalid');await page.locator('#feedback-code-send').click();await page.waitForFunction(()=>document.getElementById('feedback-status').textContent.includes('valid email'));
  assert(await page.getByRole('button',{name:'Choose images',exact:true}).isVisible());
  assert.equal(await page.locator('#feedback-file-name').innerText(),'No images selected');
  report.checks.push('English citation empty state, release switching and invalid-email validation; no external email or feedback sent');
  await page.route('https://player.bilibili.com/**',route=>route.fulfill({status:200,contentType:'text/html',body:'<!doctype html><title>Local player fixture</title>'}));
  await page.locator('[data-tutorial="intro"]').click();assert((await page.locator('.tutorial-status').innerText()).includes('introduction'));
  assert((await page.locator('.tutorial-screen iframe').getAttribute('title')).includes('introduction'));
  report.checks.push('Tutorial selection creates the correct English player title; external player body is a local fixture');
  const audio=page.locator('audio').first();await audio.evaluate(element=>element.play());await page.waitForFunction(()=>document.querySelector('audio').currentTime>0);await audio.evaluate(element=>element.pause());
  assert(!(await page.locator('.audio-status').first().innerText()).match(/[\u3400-\u9fff]/));
  const preview=page.locator('[data-screen]').first();await preview.click();assert(await page.locator('#screen-viewer').isVisible());await page.keyboard.press('Escape');
  report.checks.push('Real WAV playback and English audio status; full screenshot viewer opens and closes');
  for(const width of [1600,390]){await page.setViewportSize({width,height:1000});for(const section of ['tutorials','lab','next','feedback']){await page.locator('#'+section).scrollIntoViewIfNeeded();await page.waitForTimeout(80);await page.locator('#'+section).screenshot({path:path.join(out,'en-'+section+'-'+width+'.png'),animations:'disabled'});}}
  const nojs=await browser.browser().newContext({javaScriptEnabled:false,viewport:{width:390,height:844}}),np=await nojs.newPage();await np.goto(origin+'en/');
  assert.equal(await np.locator('html').getAttribute('lang'),'en');assert((await np.locator('#faq').innerText()).includes('Portable or installer?'));assert(await np.locator('.language-switch').isVisible());assert.equal(await np.locator('[data-screen]').count(),34);await nojs.close();
  report.checks.push('Without JavaScript: English content, screenshots and language link remain available');
  if(translations.includes('ja')){
   for(const from of ['zh-CN',...translations])for(const to of ['zh-CN',...translations]){
    if(from===to)continue;
    await page.goto(origin+(from==='zh-CN'?'':from+'/')+'#methods');await chooseLanguage(to);
    await page.waitForFunction(language=>document.documentElement.lang===language&&!location.hash&&scrollY<2,to);assert.equal(new URL(page.url()).hash,'');
   }
   report.checks.push('All six directed language switches clear the old section and start at the top');
   await page.goto(origin+'ja/');await page.setViewportSize({width:1600,height:1000});
   const japaneseFonts=await page.locator('#next .next-comparison-heading p,#next .next-comparison-heading strong,.author-note p,.author-note strong').evaluateAll(elements=>elements.map(element=>getComputedStyle(element).fontFamily));
   assert(japaneseFonts.length>5);assert(japaneseFonts.every(font=>font.includes('Yu Gothic')&&!/Whisper|Playful|KuaiLe/.test(font)),JSON.stringify(japaneseFonts));
   await page.waitForFunction(()=>document.querySelector('[data-stat="views"]').textContent==='1,234');
   const jc=page.locator('.reference-catalog').first();await jc.locator('summary').click();await jc.locator('input[type=search]').fill('no-match-fixture');assert((await jc.innerText()).includes('該当する出典がありません'));await jc.locator('summary').click();
   await page.locator('#feedback-email').fill('research@example.invalid');await page.locator('#feedback-code-send').click();await page.waitForFunction(()=>document.getElementById('feedback-status').textContent.includes('有効なメールアドレス'));
   assert(await page.getByRole('button',{name:'画像を選択',exact:true}).isVisible());assert.equal(await page.locator('#feedback-file-name').innerText(),'画像は選択されていません');
   await page.locator('[data-tutorial="intro"]').click();assert((await page.locator('#tutorial-status').innerText()).includes('紹介'));
   const ja=page.locator('audio').first();await ja.evaluate(element=>element.play());await page.waitForFunction(()=>document.querySelector('audio').currentTime>0);await ja.evaluate(element=>element.pause());assert((await page.locator('.audio-status').first().innerText()).includes('一時停止'));
   await page.locator('[data-release-tab]').nth(1).click();assert((await page.locator('[role=tabpanel]:visible').innerText()).includes('言語地図'));
   for(const width of [1600,390]){await page.setViewportSize({width,height:1000});for(const section of ['tutorials','lab','next','feedback','a-small-note']){await page.locator('#'+section).scrollIntoViewIfNeeded();await page.locator('#'+section).screenshot({path:path.join(out,'ja-'+section+'-'+width+'.png'),animations:'disabled'});}await page.locator('.next-comparison-heading').screenshot({path:path.join(out,'ja-comparison-copy-'+width+'.png'),animations:'disabled'});}
   report.checks.push('Japanese live interaction: statistics, citation empty state, feedback error/attachment controls, tutorial title, WAV pause state and release notes');
   const context=await browser.browser().newContext({javaScriptEnabled:false,viewport:{width:390,height:844}}),jp=await context.newPage();await jp.goto(origin+'ja/');assert((await jp.locator('#faq').innerText()).includes('ポータブル版とインストーラー'));await jp.locator('.language-switch').click();assert(await jp.locator('.language-option[hreflang="en"]').isVisible());await context.close();
   report.checks.push('Without JavaScript: Japanese content and native language dropdown remain usable');
  }
  assert.deepEqual(report.errors,[]);assert.deepEqual(report.requests,[]);report.success=true;
 }catch(error){report.failure=String(error);await page?.screenshot({path:path.join(out,'failure.png')}).catch(()=>{});throw error;}
 finally{
  await browser?.close();if(server)await new Promise(resolve=>server.close(resolve));await write('report.json',report);
  const files=[];async function inventory(dir){for(const entry of await fs.readdir(dir,{withFileTypes:true})){assert(!entry.isSymbolicLink());const file=path.join(dir,entry.name);if(entry.isDirectory())await inventory(file);else{const data=await fs.readFile(file);files.push({path:path.relative(scratch,file),bytes:data.length,sha256:createHash('sha256').update(data).digest('hex')});}}}await inventory(scratch);
  await write('scratch-manifest.json',{schema:'ptb-test-scratch-files/1',files});await write('run.json',{...marker,state:'ready',test_outcome:report.success?'completed':'failed'});
  const cleanup=spawnSync('powershell.exe',['-NoProfile','-ExecutionPolicy','Bypass','-File',path.join(root,'scripts/cleanup_test_scratch.ps1'),'-RunDirectory',out],{windowsHide:true,encoding:'utf8'});assert.equal(cleanup.status,0,cleanup.stdout+cleanup.stderr);
  console.log(JSON.stringify({out,...report}));
 }
}
main().catch(error=>{console.error(error);process.exitCode=1});
