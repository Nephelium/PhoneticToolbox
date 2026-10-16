// Static landing page verification in a task-owned browser and local server.
const fs=require('node:fs/promises'),path=require('node:path'),http=require('node:http'),assert=require('node:assert/strict');
const {spawnSync}=require('node:child_process'),{createHash}=require('node:crypto');
const root=path.resolve(__dirname,'..'),site=path.resolve(process.env.WEBSITE_DIRECTORY||path.join(root,'temporary-site'));
async function main(){
 const out=path.join(root,'output/validation/website','verify-'+Date.now()),scratch=path.join(out,'scratch');await fs.mkdir(scratch,{recursive:true});
 const write=(name,data)=>fs.writeFile(path.join(out,name),JSON.stringify(data,null,2)+'\n');
 const marker={schema:'ptb-test-run/1',owner_pid:process.pid,state:'active',started_at:new Date().toISOString(),recipe:'Static website, task-owned HTTP server and isolated Chrome; screenshots, navigation, reduced motion, no JavaScript'};
 await write('run.json',marker);let server,browser,page;const report={success:false,checks:[],errors:[],requests:[]};
 try{
  server=http.createServer(async(req,res)=>{try{const url=new URL(req.url,'http://localhost');if(url.pathname==='/site-stats.json'){res.writeHead(200,{'Content-Type':'application/json'}).end(JSON.stringify({schema:1,startedAt:'2026-10-12T00:00:00Z',updatedAt:new Date().toISOString(),pageViews:1234,downloads:{total:56,portable:40,installer:16}}));return;}const relative=decodeURIComponent(url.pathname),file=path.resolve(site,'.'+relative+(relative.endsWith('/')?'index.html':''));if(!file.startsWith(site+path.sep)){res.writeHead(403).end();return;}const bytes=await fs.readFile(file);const mime={'.html':'text/html; charset=utf-8','.css':'text/css; charset=utf-8','.js':'application/javascript; charset=utf-8','.png':'image/png','.svg':'image/svg+xml','.ttf':'font/ttf','.txt':'text/plain; charset=utf-8','.json':'application/json','.wav':'audio/wav','.jpg':'image/jpeg'};res.writeHead(200,{'Content-Type':mime[path.extname(file)]||'application/octet-stream'}).end(bytes);}catch{res.writeHead(404).end();}});
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));const url=process.env.WEBSITE_ORIGIN||'http://127.0.0.1:'+server.address().port+'/';report.origin=url;
  const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  browser=await chromium.launchPersistentContext(path.join(scratch,'profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1600,height:1000},deviceScaleFactor:1,userAgent:'Mozilla/5.0 PTB-Website-Verify'});
  report.browser=browser.browser().version();page=await browser.newPage();page.on('pageerror',e=>report.errors.push(e.message));page.on('response',r=>{if(r.status()>=400)report.requests.push({url:r.url(),status:r.status()});});
  async function ready(){await page.evaluate(()=>document.fonts.ready);await page.waitForTimeout(250);}
  async function verifyDownloads(){
    const metadata=JSON.parse(await fs.readFile(path.join(site,'assets/downloads.json'),'utf8'));
    assert.equal(await page.locator('#download-state span').innerText(),metadata.version);
    assert(!(await page.locator('#download-state').innerText()).includes('当前下载：'));
    const files=['PhoneticToolbox-windows-x64.exe','PhoneticToolbox-'+metadata.version+'-windows-x64-setup.exe'];
    for(const [i,name] of files.entries()){
      const button=page.locator('.download').nth(i),relative='/releases/windows-x64/'+metadata.version+'/'+name;
      assert.equal(await button.getAttribute('href'),relative);
      const tooltip=page.locator('#download-tip-'+(i?'installer':'portable'));
      assert((await tooltip.textContent()).includes((metadata.files[name].size/1e6).toFixed(1)+' MB'));
      if(process.env.WEBSITE_ORIGIN){const response=await page.request.head(new URL(relative,url).href);assert.equal(response.status(),200);assert.equal(Number(response.headers()['content-length']),metadata.files[name].size);}
      await page.route('**'+relative,route=>route.fulfill({status:200,headers:{'content-type':'application/octet-stream','content-disposition':'attachment; filename="'+name+'"'},body:'download routing fixture'}));
      const pending=page.waitForEvent('download');await button.click();const download=await pending;assert.equal(download.suggestedFilename(),name);assert.equal(page.url(),url);await page.unroute('**'+relative);
    }
    for(const width of [1600,800,390,320]){
      await page.setViewportSize({width,height:1000});await page.evaluate(()=>scrollTo(0,0));await ready();
      await page.waitForFunction(()=>getComputedStyle(document.querySelector('.download-actions')).opacity==='1');
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
      await page.screenshot({path:path.join(out,'downloads-'+width+'.png'),animations:'disabled'});
    }
    await page.setViewportSize({width:1600,height:1000});report.checks.push(metadata.version+' header, both download URLs, attachment names, sizes and four viewport layouts match download metadata; public HEAD checks run when a public origin is supplied');
  }
  async function verifyReleases(){
    const snapshots=[JSON.parse(await fs.readFile(path.join(site,'assets/release.json'),'utf8'))];
    for(const name of await fs.readdir(path.join(site,'assets/releases')))if(name.endsWith('.json'))snapshots.push(JSON.parse(await fs.readFile(path.join(site,'assets/releases',name),'utf8')));
    snapshots.sort((a,b)=>b.tag_name.localeCompare(a.tag_name,undefined,{numeric:true}));
    const tabs=page.locator('.release-tabs [role="tab"]');assert.deepEqual(await tabs.allTextContents(),snapshots.map(r=>r.tag_name));
    assert.equal(await tabs.first().getAttribute('aria-selected'),'true');
    const latest=page.locator('#release-panel-0'),legacy=page.locator('#release-panel-'+(snapshots.length-1));
    assert.equal(await latest.evaluate(e=>e.hidden),false);assert.equal(await legacy.evaluate(e=>e.hidden),true);
    const snapshot=snapshots[0];
    const source=await fs.readFile(path.join(root,snapshot.source),'utf8');assert.equal(snapshot.body,source);assert.equal(snapshot.source_sha256,createHash('sha256').update(source).digest('hex'));
    const rendered=(await latest.locator('.release-content').innerText()).replace(/ ↗/g,'');
    for(const line of source.split(/\r?\n/).filter(l=>l.trim()&&!l.startsWith('|'))){
      const plain=line.replace(/^#+\s*|^-\s*/g,'').replace(/\[([^\]]+)\]\([^)]+\)/g,'$1').replace(/\*\*|`/g,'');
      assert(rendered.includes(plain),plain);
    }
    assert.equal(await latest.locator('.release-content h4').count(),source.split(/\r?\n/).filter(l=>l.startsWith('## ')).length);
    assert.equal(await latest.locator('.release-content li').count(),source.split(/\r?\n/).filter(l=>l.startsWith('- ')).length);
    assert.equal(await latest.locator('table tbody tr').count(),source.split(/\r?\n/).filter(l=>l.startsWith('|')).length?source.split(/\r?\n/).filter(l=>l.startsWith('|')).length-2:0);assert.equal(await latest.locator('.release-content strong').count(),(source.match(/\*\*[^*]+\*\*/g)||[]).length);
    for(let i=0;i<snapshots.length;i++){
      await tabs.nth(i).click();assert.equal(await tabs.nth(i).getAttribute('aria-selected'),'true');
      assert.equal(await page.locator('#release-panel-'+i).evaluate(e=>e.hidden),false);
      if(snapshots[i].source){const bytes=await fs.readFile(path.join(root,snapshots[i].source));assert.equal(snapshots[i].body,bytes.toString('utf8'));assert.equal(snapshots[i].source_sha256,createHash('sha256').update(bytes).digest('hex'));}
    }
    await tabs.last().click();assert.equal(await tabs.last().getAttribute('aria-selected'),'true');assert.equal(await latest.evaluate(e=>e.hidden),true);assert.match(await legacy.locator('.release-content').innerText(),/暂时还不行/);assert(await legacy.locator('.release-extra').evaluateAll(es=>es.every(e=>e.open)));
    await tabs.last().focus();await page.keyboard.press('Home');assert.equal(await tabs.first().getAttribute('aria-selected'),'true');assert.equal(await tabs.first().evaluate(e=>document.activeElement===e),true);
    await page.keyboard.press('ArrowRight');assert.equal(await tabs.nth(1).getAttribute('aria-selected'),'true');await page.keyboard.press('End');assert.equal(await tabs.last().getAttribute('aria-selected'),'true');await page.keyboard.press('ArrowRight');assert.equal(await tabs.first().getAttribute('aria-selected'),'true');await page.keyboard.press('ArrowLeft');assert.equal(await tabs.last().getAttribute('aria-selected'),'true');await page.keyboard.press('Home');
    for(const width of [1600,800,390,320]){
      await page.setViewportSize({width,height:1000});await latest.scrollIntoViewIfNeeded();await ready();
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
      assert(await tabs.evaluateAll(es=>es.every(e=>{const r=e.getBoundingClientRect();return r.left>=0&&r.right<=innerWidth+1})));
      await page.screenshot({path:path.join(out,'release-'+width+'.png'),animations:'disabled'});
    }
    await page.setViewportSize({width:1600,height:1000});
    report.checks.push(snapshot.tag_name+' is the default newest version; exact source body and hashes, Markdown paragraphs, bullets, headings, bold and tables preserved; mouse and keyboard switch all '+snapshots.length+' versions; four widths fit');
  }
  if(process.env.WEBSITE_DIVIDERS_ONLY==='1'){
    await page.goto(new URL('divider-preview.html',url).href);await ready();
    for(const width of [1440,390,320]){
      await page.setViewportSize({width,height:1100});await ready();
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
      const bounds=await page.locator('.preview-option').evaluateAll(es=>es.map(e=>{const r=e.getBoundingClientRect();return {left:r.left,right:r.right,svgWidth:e.querySelector('svg').getBoundingClientRect().width}}));
      assert(bounds.every(b=>b.left>=0&&b.right<=width&&b.svgWidth>0));
      await page.screenshot({path:path.join(out,'divider-options-'+width+'.png'),fullPage:true,animations:'disabled'});
    }
    await page.setViewportSize({width:1440,height:1100});
    for(const id of ['wave','paper','tape']){await page.locator('#'+id).screenshot({path:path.join(out,'divider-'+id+'.png'),animations:'disabled'});}
    await page.locator('.preview-nav a[href="#paper"]').focus();await page.keyboard.press('Enter');assert.equal(new URL(page.url()).hash,'#paper');
    assert.deepEqual(report.errors,[]);assert.deepEqual(report.requests,[]);report.checks.push('Three divider previews: 1440/390/320 responsive layout, SVGs within viewports, screenshots and keyboard section navigation');report.success=true;return;
  }
  if(process.env.WEBSITE_CHAPTERS_ONLY==='1'){
    await page.goto(url);await ready();await page.emulateMedia({reducedMotion:'reduce'});
    const expected=[['tutorials','wave'],['lab','wave'],['methods','paper'],['faq','paper'],['updates','tape'],['next','tape'],['feedback','tape'],['contact','tape'],['a-small-note','tape']];
    assert.deepEqual(await page.locator('.chapter-divider').evaluateAll(es=>es.map(e=>[e.nextElementSibling.id,e.dataset.style])),expected);
    for(const width of [1440,390,320]){
      await page.setViewportSize({width,height:1000});await ready();
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
      for(const style of ['wave','paper','tape']){
        const divider=page.locator('.chapter-divider[data-style="'+style+'"]').first();await divider.scrollIntoViewIfNeeded();await ready();
        const box=await divider.boundingBox();assert(box.x>=0&&box.x+box.width<=width);
        await page.screenshot({path:path.join(out,'chapter-'+style+'-'+width+'.png'),animations:'disabled'});
      }
    }
    assert.equal(await page.locator('#source,.guide-section,.reading-menu a[href="#source"]').count(),0);
    for(const selector of ['.header-github','.footer-github'])assert.equal(await page.locator(selector).getAttribute('href'),'https://github.com/Nephelium/PhoneticToolbox');
    for(const width of [1440,800,390,320]){await page.setViewportSize({width,height:1000});await page.locator('.site-footer').scrollIntoViewIfNeeded();await ready();assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));assert(await page.locator('.footer-github').isVisible());await page.locator('.site-footer').screenshot({path:path.join(out,'footer-'+width+'.png'),animations:'disabled'});}
    await page.locator('.reading-toggle').click();await page.locator('.reading-menu a[href="#methods"]').click();assert.equal(new URL(page.url()).hash,'#methods');
    const nojs=await browser.browser().newContext({javaScriptEnabled:false,viewport:{width:390,height:844}});const np=await nojs.newPage();await np.goto(url);assert.equal(await np.locator('.chapter-divider svg').count(),9);await nojs.close();
    assert.deepEqual(report.errors,[]);assert.deepEqual(report.requests,[]);report.checks.push('Nine chapter motifs assigned to their sections; all three styles inspected at 1440/390/320; no overflow, reading navigation and no-JS content pass');report.success=true;return;
  }
  if(process.env.WEBSITE_MOTION_ONLY==='1'){
    await page.goto(url);await ready();await page.emulateMedia({reducedMotion:'no-preference'});
    await page.setViewportSize({width:1440,height:1000});await page.locator('.hero-art').scrollIntoViewIfNeeded();await page.waitForTimeout(500);
    const floater=page.locator('.tile-sh'),tile=page.locator('.tile-sh .phonetic-tile');
    const ambient=[];for(let i=0;i<8;i++){ambient.push(await floater.evaluate(e=>new DOMMatrixReadOnly(getComputedStyle(e).transform).m42));await page.waitForTimeout(300);}
    assert(Math.max(...ambient)-Math.min(...ambient)>5,'Visible ambient movement');report.heroMotionRange=Math.max(...ambient)-Math.min(...ambient);
    const tileBox=await tile.boundingBox();await page.mouse.move(tileBox.x+tileBox.width/2,tileBox.y+tileBox.height/2);await page.waitForTimeout(650);assert(await tile.evaluate(e=>e.matches(':hover')));assert(parseFloat(await tile.evaluate(e=>getComputedStyle(e).scale))>1.03);assert(parseFloat(await tile.evaluate(e=>getComputedStyle(e).rotate))<-4);
    await page.screenshot({path:path.join(out,'hero-hover.png'),animations:'allow'});
    await page.mouse.move(0,0);await page.waitForTimeout(700);assert(['1','none'].includes(await tile.evaluate(e=>getComputedStyle(e).scale)));
    await page.locator('#methods').scrollIntoViewIfNeeded();await page.waitForTimeout(1000);assert.equal(await floater.evaluate(e=>getComputedStyle(e).animationPlayState),'paused');
    const card=page.locator('.reference-card').first();await card.hover();await page.waitForTimeout(500);assert(parseFloat(await card.evaluate(e=>getComputedStyle(e).scale))>1);
    const rows=page.locator('.faq-list details');assert(await rows.evaluateAll(es=>es.every(e=>e.classList.contains('reveal'))));
    for(const width of [1440,800,390,320]){
      await page.setViewportSize({width,height:1000});await page.goto(url);await ready();await page.locator('.hero-art').scrollIntoViewIfNeeded();await page.waitForTimeout(1100);
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await page.screenshot({path:path.join(out,'motion-'+width+'.png'),animations:'allow'});
    }
    await page.emulateMedia({reducedMotion:'reduce'});await ready();assert.equal(await floater.evaluate(e=>getComputedStyle(e).animationName),'none');assert.equal(await tile.evaluate(e=>getComputedStyle(e).scale),'none');
    assert(await page.locator('.reveal').evaluateAll(es=>es.every(e=>getComputedStyle(e).opacity==='1'&&['none','0px'].includes(getComputedStyle(e).translate))));
    const nojs=await browser.browser().newContext({javaScriptEnabled:false,viewport:{width:390,height:844},reducedMotion:'reduce'});const np=await nojs.newPage();await np.goto(url);assert(await np.locator('.reveal').evaluateAll(es=>es.every(e=>getComputedStyle(e).opacity==='1')));await nojs.close();
    assert.deepEqual(report.errors,[]);assert.deepEqual(report.requests,[]);report.checks.push('Visible hero float, tile hover/exit, offscreen pause, section stagger/card hover, four viewport layouts, runtime reduced-motion and no-JS static content');report.success=true;return;
  }
  await page.goto(url);await ready();
  if(process.env.WEBSITE_SPECTRUM_ONLY==='1'){
    const manifest=JSON.parse(await fs.readFile(path.join(site,'assets/audio/provenance.json'),'utf8'));
    const examples=JSON.parse(await fs.readFile(path.join(site,'assets/spectrogram-examples/provenance.json'),'utf8'));
    assert.equal(await page.locator('.reconstruction-example').count(),3);
    for(const row of manifest.files){
      assert.equal(createHash('sha256').update(await fs.readFile(path.join(site,'assets/audio',row.file))).digest('hex'),row.sha256);
    }
    for(const example of examples.cases){
      const name='spectrum-'+example.id,row=manifest.files.find(r=>r.file===name+'.wav');
      assert.equal(row.parameters.delivery,'exact_listening_export');
      assert.equal(row.sha256,row.parameters.source_output_sha256);assert.equal(row.sha256,example.output_sha256);
      assert.equal(row.sample_rate,24000);assert.equal(row.parameters.model.frontend.num_mels,80);
      const audio=page.locator('#audio-'+name);
      const source=(await audio.getAttribute('src')).replace(/^\/site-releases\/r[0-9a-f]{16}\//,'');
      assert.equal(source,'assets/audio/'+name+'.wav?v='+row.sha256.slice(0,12));
      const response=await page.request.get(await audio.evaluate(e=>e.src));assert.equal(response.status(),200);
      assert.equal(createHash('sha256').update(await response.body()).digest('hex'),row.sha256);
      await audio.evaluate(e=>e.load());await page.waitForFunction(id=>document.getElementById(id).readyState>=2,'audio-'+name);
      assert(Math.abs(await audio.evaluate(e=>e.duration)-row.duration)<0.001);
      await page.locator('[data-audio="audio-'+name+'"]').click();
      await page.waitForFunction(id=>!document.getElementById(id).paused,'audio-'+name);
      await page.waitForFunction(id=>document.getElementById(id).ended,'audio-'+name);
      assert.equal(await audio.evaluate(e=>e.error),null);
      assert.equal(await page.locator('[data-audio="audio-'+name+'"]').getAttribute('aria-pressed'),'false');
      await page.locator('#reconstruction-'+example.id).locator('..').locator('summary').click();
    }
    assert.match(await page.locator('.reconstruction-card').innerText(),/训练后 80 带 Mel 修复 → BigVGAN/);
    for(const width of [1280,390]){
      await page.setViewportSize({width,height:900});await page.locator('.reconstruction-card').scrollIntoViewIfNeeded();await ready();
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
      await page.screenshot({path:path.join(out,'spectrum-'+width+'.png'),animations:'disabled'});
    }
    assert.deepEqual(report.errors,[]);assert.deepEqual(report.requests,[]);
    report.checks.push('All audio SHA256 values match; three exact 80-band exports fetched and played to completion; cache versions, duration, method text and desktop/mobile layout pass');
    report.success=true;return;
  }
  if(process.env.WEBSITE_ORDER_ONLY==='1'){
    assert.equal(await page.locator('.more-features').count(),0);
    assert(await page.locator('#tutorials').evaluate(e=>e.previousElementSibling.classList.contains('chapter-divider')&&e.previousElementSibling.previousElementSibling.id==='background'));
    assert.equal(await page.locator('.more-dots span').count(),0);
    assert.equal(await page.locator('.more-lettering').count(),0);
    const cards=page.locator('#background-stack .stack-card');assert.equal(await cards.count(),4);
    const order=await cards.evaluateAll(es=>es.map(e=>({src:e.getAttribute('href'),z:Number(getComputedStyle(e).zIndex)})).sort((a,b)=>b.z-a.z).map(e=>'assets/'+e.src.split('assets/').pop()));
    assert.deepEqual(order,['home','settings','ipa','experiment'].map(k=>'assets/screens/custom-background-'+k+'-dark.png'));
    const inputs=JSON.parse(await fs.readFile(path.join(root,'output/validation/website/background-four-order-20261013/inputs.json'),'utf8'));
    for(const item of inputs.images)assert.equal(createHash('sha256').update(await fs.readFile(path.join(root,item.destination))).digest('hex'),item.sha256);
    for(const width of [1600,800,390,320]){
      await page.setViewportSize({width,height:1000});await page.locator('#background').scrollIntoViewIfNeeded();await ready();await page.waitForTimeout(1000);
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
      await page.waitForFunction(()=>[...document.querySelectorAll('#background-stack img')].every(e=>e.complete&&e.naturalWidth===2559&&e.naturalHeight===1527));
      assert(await cards.evaluateAll(es=>es.every(e=>e.querySelector('img').complete&&e.querySelector('img').naturalWidth===2559&&e.querySelector('img').naturalHeight===1527)));
      await page.evaluate(()=>document.activeElement.blur());
      for(const card of await cards.all()){
        await card.scrollIntoViewIfNeeded();await page.waitForTimeout(100);
        const point=await card.evaluate(e=>{const r=e.getBoundingClientRect();for(let y=Math.max(0,r.top)+4;y<Math.min(innerHeight,r.bottom)-4;y+=5)for(let x=Math.max(0,r.left)+4;x<Math.min(innerWidth,r.right)-4;x+=5){if(document.elementFromPoint(x,y)?.closest('[data-screen]')===e)return {x,y}}return null});
        assert(point,'All four background cards expose a clickable area at '+width);await page.mouse.click(point.x,point.y);
        assert.equal(await page.locator('#viewer-image').getAttribute('src'),await card.evaluate(e=>e.href));await page.keyboard.press('Escape');await page.evaluate(()=>document.activeElement.blur());
      }
      await page.screenshot({path:path.join(out,'background-'+width+'.png'),animations:'disabled'});
      await page.locator('.chapter-divider').first().scrollIntoViewIfNeeded();await page.waitForTimeout(1000);await page.screenshot({path:path.join(out,'more-'+width+'.png'),animations:'disabled'});
      report.checks.push(width+'px: background order 2,1,3,4; all four original viewers accessible; chapter divider separates background and tutorials; no overflow');
    }
    const nojs=await browser.browser().newContext({javaScriptEnabled:false,viewport:{width:390,height:844}});const np=await nojs.newPage();await np.goto(url);
    assert.equal(await np.locator('.more-features').count(),0);assert(await np.locator('#tutorials').evaluate(e=>e.previousElementSibling.classList.contains('chapter-divider')));await nojs.close();
    assert.deepEqual(report.errors,[]);assert.deepEqual(report.requests,[]);report.success=true;return;
  }
  if(process.env.WEBSITE_RELEASE_ONLY==='1'){
    await verifyDownloads();
    await verifyReleases();
    const nojs=await browser.browser().newContext({javaScriptEnabled:false,viewport:{width:390,height:844}});const np=await nojs.newPage();await np.goto(url);assert.equal(await np.locator('#release-panel-0').evaluate(e=>e.hidden),false);assert.match(await np.locator('#release-panel-0 .release-content').innerText(),/PhoneticToolbox 3\.0\.2 更新/);await nojs.close();
    report.checks.push('Without JavaScript the latest 3.0.2 notes remain visible');assert.equal(report.errors.length,0);assert.equal(report.requests.length,0);report.success=true;return;
  }
  if(process.env.WEBSITE_NAV_ONLY==='1'||process.env.WEBSITE_NEXT_ONLY==='1'||process.env.WEBSITE_TUTORIAL_ONLY==='1'){
    for(const width of [1920,1600,1440,1250,1100,800,390,320]){
      await page.setViewportSize({width,height:1000});await ready();await page.evaluate(()=>scrollTo(0,0));
      const links=page.locator('.site-header nav>a');assert.equal(await links.count(),9);
      const layout=await links.evaluateAll(es=>es.map(e=>{const r=e.getBoundingClientRect();const textRects=[...e.childNodes].filter(n=>n.nodeType===Node.TEXT_NODE&&n.textContent.trim()).flatMap(n=>{const range=document.createRange();range.selectNodeContents(n);return [...range.getClientRects()].map(q=>({x:q.x,right:q.right}))});return {textRects,text:e.childNodes[0]?.textContent,font:parseFloat(getComputedStyle(e).fontSize),x:r.x,right:r.right,width:r.width,scrollWidth:e.scrollWidth,clientWidth:e.clientWidth}}));
      assert(layout.every(r=>r.x>=0&&r.right<=width+1&&r.textRects.every(t=>t.x>=r.x-1&&t.right<=r.right+1)&&r.font>=(width>800?18:width>360?16:15)),JSON.stringify({width,layout}));
      assert.equal(await page.locator('.header-github').getAttribute('href'),'https://github.com/Nephelium/PhoneticToolbox');assert(await page.locator('.header-github svg').isVisible());
      if(width>800){
        await page.mouse.move(0,0);
        for(let i=0;i<await links.count();i++){
          await links.nth(i).focus();await page.waitForTimeout(220);
          const hint=links.nth(i).locator('.nav-tooltip');assert(await hint.isVisible());
          const b=await hint.evaluate(e=>{const r=e.getBoundingClientRect(),range=document.createRange();range.selectNodeContents(e);return {x:r.x,right:r.right,top:r.top,bottom:r.bottom,text:[...range.getClientRects()].map(t=>({x:t.x,right:t.right,top:t.top,bottom:t.bottom}))}});
          assert(b.x>=0&&b.right<=width&&b.text.every(t=>t.x>=b.x&&t.right<=b.right&&t.top>=b.top&&t.bottom<=b.bottom),JSON.stringify({width,index:i,hint:b}));
        }
        await links.nth(1).hover();await page.waitForTimeout(220);
      }
      await page.screenshot({path:path.join(out,'navigation-'+width+'.png'),animations:'disabled'});
      if(width===1600&&process.env.WEBSITE_NEXT_ONLY==='1'){await page.mouse.move(0,0);await page.evaluate(()=>document.activeElement?.blur());await ready();await page.locator('.site-header').screenshot({path:path.join(out,'next-header.png'),animations:'disabled'});}
      report.checks.push('Navigation '+width+'px: 9 links, readable font, repository icon/link, no overflow');
    }
    await page.locator('.header-github').focus();if(await page.evaluate(()=>innerWidth>800))assert.equal(await page.locator('#tip-github').evaluate(e=>getComputedStyle(e).visibility),'visible');
    if(process.env.WEBSITE_TUTORIAL_ONLY==='1'){
      await page.emulateMedia({reducedMotion:'reduce'});
      const expectedModules=[... (await fs.readFile(path.join(root,'README.md'),'utf8')).matchAll(/\| (?:分析与采集|合成与操控|标注与实验|文献与学习) \| \[([^\]]+)\]/g)].map(m=>m[1]);
      assert.equal(expectedModules.length,18);
      assert.deepEqual(await page.locator('.tutorial-channel:disabled').allTextContents(),expectedModules);
      assert.equal(await page.locator('.tutorial-channel:enabled').count(),1);
      assert((await page.locator('.tutorial-construction').innerText()).includes('正在施工中'));
      assert(await page.locator('#tutorials').evaluate(e=>e.nextElementSibling.classList.contains('chapter-divider')&&e.nextElementSibling.nextElementSibling.id==='lab'));
      assert.equal(await page.locator('.more-features,.tutorial-channel-label,.reading-caption').count(),0);
      assert.deepEqual(await page.locator('.chapter-divider').evaluateAll(es=>es.map(e=>e.nextElementSibling.id)),['tutorials','lab','methods','faq','updates','next','feedback','contact','a-small-note']);
      assert(await page.locator('.chapter-divider').evaluateAll(es=>es.every(e=>e.getAttribute('aria-hidden')==='true')));
      for(const width of [1600,1280,800,390,320]){
        await page.setViewportSize({width,height:1000});await page.goto(url);await ready();
        await page.locator('.site-header a[href="#tutorials"]').focus();await page.keyboard.press('Enter');
        await page.waitForFunction(()=>location.hash==='#tutorials');
        await page.locator('#tutorials').scrollIntoViewIfNeeded();await ready();
        assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
        const layout=await page.locator('.tutorial-layout').evaluate(e=>({columns:getComputedStyle(e).gridTemplateColumns.split(' ').length,boxes:[...e.querySelectorAll('.tutorial-tv,.tutorial-channel')].map(n=>{const r=n.getBoundingClientRect();return {x:r.x,right:r.right,fits:n.scrollWidth<=n.clientWidth+1}})}));
        assert.equal(layout.columns,width<=800?1:2);assert(layout.boxes.every(b=>b.x>=0&&b.right<=width&&b.fits),JSON.stringify({width,layout}));
        assert.equal(await page.locator('#tutorial-intro').evaluate(e=>getComputedStyle(e).backgroundColor),'rgb(255, 231, 85)');
        assert.equal(await page.locator('.tutorial-channel-grid').evaluate(e=>getComputedStyle(e).display),'flex');
        assert(await page.locator('.tutorial-channel:disabled').evaluateAll(es=>new Set(es.map(e=>Math.round(e.getBoundingClientRect().width))).size>3));
        const divider=page.locator('.chapter-divider').first();const minor=page.locator('.tool-divider').first();assert((await divider.boundingBox()).width>(await minor.boundingBox()).width);
        await divider.scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,'chapter-boundary-'+width+'.png'),animations:'disabled'});
        await page.locator('#tutorials').scrollIntoViewIfNeeded();
        assert.equal(await page.locator('#tutorial-screen iframe').count(),0);
        await page.locator('#tutorials').screenshot({path:path.join(out,'tutorial-'+width+'.png'),animations:'disabled'});
      }
      await page.locator('.reading-toggle').click();await page.locator('.reading-menu').screenshot({path:path.join(out,'reading-menu.png'),animations:'disabled'});await page.keyboard.press('Escape');
      await page.setViewportSize({width:1280,height:1000});
      const link=page.locator('.tutorial-actions a');assert.equal(await link.getAttribute('href'),'https://www.bilibili.com/video/BV1WWpr67EkX/');assert.equal(await link.getAttribute('target'),'_blank');
      await page.locator('#tutorial-intro').focus();await page.keyboard.press('Enter');
      const frame=page.locator('#tutorial-screen iframe');assert.equal(await frame.count(),1);assert((await frame.getAttribute('src')).includes('bvid=BV1WWpr67EkX'));assert(await frame.evaluate(e=>e.allowFullscreen));
      await page.locator('#tutorial-intro').click();assert.equal(await frame.count(),1);
      await page.locator('#tutorial-fullscreen').click();await page.waitForFunction(()=>document.fullscreenElement?.id==='tutorial-screen');await page.evaluate(()=>document.exitFullscreen());
      report.playerProbe={};
      try{await page.waitForFunction(()=>{const frame=document.querySelector('#tutorial-screen iframe');return !!frame;});const child=await frame.elementHandle();const embedded=await child.contentFrame();await embedded.waitForSelector('video',{timeout:20000});
        report.playerProbe.video=await embedded.locator('video').evaluate(e=>({readyState:e.readyState,duration:e.duration,paused:e.paused,error:e.error?.message}));
        await embedded.locator('video').evaluate(e=>e.play());await page.waitForTimeout(2000);
        report.playerProbe.playing=await embedded.locator('video').evaluate(e=>({currentTime:e.currentTime,paused:e.paused,readyState:e.readyState}));
      }catch(error){report.playerProbe.limitation=String(error).slice(0,450);}
      await page.locator('#tutorials').screenshot({path:path.join(out,'tutorial-player.png'),animations:'disabled'});
      const nojs=await browser.browser().newContext({javaScriptEnabled:false,viewport:{width:390,height:844}});const np=await nojs.newPage();await np.goto(url);
      assert(await np.locator('#tutorials').isVisible());assert.equal(await np.locator('.tutorial-channel:disabled').count(),18);assert(await np.locator('.tutorial-actions a').isVisible());await nojs.close();
      assert.deepEqual(report.errors,[]);assert(report.requests.every(r=>!r.url.startsWith(url)),'No failed local assets');
      report.checks.push('Tutorial: removed teaser/menu caption/channel heading; nine higher-level dividers; hand-drawn variable-width channels; all 18 module names match README, disabled modules, yellow intro, 5 responsive layouts, keyboard anchor/activation, one lazy official iframe, actual fullscreen entry/exit, Bilibili link, no-JS fallback; external media evidence recorded separately');
      report.success=true;return;
    }
    if(process.env.WEBSITE_NEXT_ONLY==='1'){
      const expected=['#tutorials','#lab','#methods','#faq','#updates','#next','#feedback','#contact','https://github.com/Nephelium/PhoneticToolbox'];
      assert.deepEqual(await page.locator('.site-header nav>a').evaluateAll(es=>es.map(e=>e.getAttribute('href'))),expected);
      assert.equal(await page.locator('.site-header a[href="#next"]').evaluate(e=>e.childNodes[0].textContent),'下一步…');
      assert(await page.locator('#next').evaluate(e=>e.previousElementSibling.previousElementSibling.id==='updates'&&e.nextElementSibling.nextElementSibling.id==='feedback'));
      assert(await page.locator('#methods').evaluate(e=>e.previousElementSibling.previousElementSibling.id==='lab'));
      assert.deepEqual(await page.locator('#next .upcoming-model').allTextContents(),['参数预测小模型','Griffin–Lim','条件化 NAFNet','BigVGAN']);
      const copy=await page.locator('#next').innerText();for(const text of ['韶子正在参考 Aoki','短时窗','模型训练进行中','相位','Mel 谱','处理前','处理后'])assert(copy.includes(text),text);
      assert.equal(await page.locator('.upcoming-reference').getAttribute('href'),'https://doi.org/10.1250/ast.44.186');
      assert.equal(await page.locator('#next img').count(),6);assert.equal(await page.locator('#next audio').count(),12);
      assert(await page.locator('[id]').evaluateAll(es=>new Set(es.map(e=>e.id)).size===es.length),'Unique DOM ids across repeated comparison audio');
      const comparison=JSON.parse(await fs.readFile(path.join(site,'assets/spectrogram-examples/comparison.json'),'utf8'));
      const metadata=JSON.parse(await fs.readFile(path.join(site,'assets/audio/provenance.json'),'utf8'));
      assert.deepEqual(comparison.pairs.map(p=>p.id),['smith','levantan','observo','lu2025-modal','lu2025-pressed','uploaded']);
      assert.equal(await page.locator('.next-example[data-case="uploaded"] h4').innerText(),'知乎例图 · 2 秒 / 6 kHz');assert((await page.locator('#next-lu-reference').innerText()).includes('Contribution of F0 and phonation to tone perception in the Zaiwa language.'));assert.equal(await page.locator('#next-lu-reference a').getAttribute('href'),'https://doi.org/10.1016/j.wocn.2025.101413');
      for(const pair of comparison.pairs){
        assert.equal(pair.before_defaults.phase_initialization,'random');assert.equal(pair.before_defaults.window_ms,10);assert.equal(pair.before_defaults.dynamic_range_db,60);assert.equal(pair.before_defaults.window,'Hann');
        const settings=page.locator('.next-comparison-settings');await settings.locator('summary').click();assert((await settings.innerText()).includes('随机相位初始化 Griffin–Lim'));await settings.locator('summary').click();const caseCard=page.locator('.next-example[data-case="'+pair.id+'"]');assert((await caseCard.locator('.sound-detail').first().innerText()).includes('统一默认 10 ms / 60 dB'));
        const image=page.locator('.next-example[data-case="'+pair.id+'"] img');
        const imageResponse=await page.request.get(await image.evaluate(e=>e.src));assert.equal(imageResponse.status(),200);assert.equal(createHash('sha256').update(await imageResponse.body()).digest('hex'),pair.display.image_sha256);
        if(pair.id.startsWith('lu2025-'))assert.equal(await caseCard.locator('figcaption a').getAttribute('href'),'https://doi.org/10.1016/j.wocn.2025.101413');
        const p=pair.parameter_prediction;
        assert(p.applied_to_reconstruction);assert.equal(p.checkpoint_sha256,pair.parameter_model_sha256);
        const text=await page.locator('.next-example[data-case="'+pair.id+'"] .next-parameters').innerText();
        assert(text.includes(p.window_ms.toFixed(2)+' ms')&&text.includes(p.dynamic_range_db.toFixed(2)+' dB'));
        if(p.window_rounding){assert.equal(pair.applied_window_ms,Math.floor(p.window_ms+0.5));assert(text.includes('计算采用 '+pair.applied_window_ms+' ms'));}
        for(const kind of ['before','after']){
          const name=pair[kind],row=metadata.files.find(r=>r.file===name),id='audio-next-'+name.replace('.wav','');
          assert.equal(row.sha256,pair[kind+'_sha256']);
          const audio=page.locator('#'+id),response=await page.request.get(await audio.evaluate(e=>e.src));assert.equal(response.status(),200);
          const volume=await audio.evaluate(e=>e.volume);assert(Math.abs(volume-pair.playback_volumes[name.replace('.wav','')])<1e-6);
          assert.equal(createHash('sha256').update(await response.body()).digest('hex'),row.sha256);
          await audio.evaluate(e=>e.load());await page.waitForFunction(id=>document.getElementById(id).readyState>=2,id);
          assert(Math.abs(await audio.evaluate(e=>e.duration)-row.duration)<.001);
          const button=page.locator('[data-audio="'+id+'"]');await button.click();
          await page.waitForFunction(id=>document.getElementById(id).ended,id);assert.equal(await audio.evaluate(e=>e.error),null);
          assert.equal(await button.getAttribute('aria-pressed'),'false');
          assert((await page.locator('#next .audio-status').innerText()).startsWith('听完了'));
        }
        const beforeRow=metadata.files.find(r=>r.file===pair.before),afterRow=metadata.files.find(r=>r.file===pair.after);
        assert(Math.abs(beforeRow.rms*pair.playback_volumes[pair.before.replace('.wav','')]-afterRow.rms*pair.playback_volumes[pair.after.replace('.wav','')])<5e-5);
      }
      const first=page.locator('#audio-next-spectrum-before-smith'),second=page.locator('#audio-next-spectrum-smith');
      await page.locator('[data-audio="audio-next-spectrum-before-smith"]').click();await page.locator('[data-audio="audio-next-spectrum-smith"]').click();assert(await first.evaluate(e=>e.paused));assert(!(await second.evaluate(e=>e.paused)));
      await page.locator('[data-audio="audio-vowel-a"]').click();await page.waitForFunction(()=>document.querySelector('#audio-next-spectrum-smith').paused&&document.querySelector('#next .audio-status').textContent.startsWith('已暂停'));
      assert(!(await page.locator('#audio-vowel-a').evaluate(e=>e.paused)));await page.locator('#audio-vowel-a').evaluate(e=>e.pause());
      await page.emulateMedia({reducedMotion:'reduce'});
      for(const width of [1600,1280,800,390,320]){
        await page.setViewportSize({width,height:1000});await page.goto(url);await ready();
        await page.locator('.site-header a[href="#next"]').focus();await page.keyboard.press('Enter');
        await page.waitForFunction(()=>location.hash==='#next');
        await page.waitForFunction(()=>{const section=document.querySelector('#next'),top=section.getBoundingClientRect().top;const inset=parseFloat(getComputedStyle(document.documentElement).scrollPaddingTop)+parseFloat(getComputedStyle(section).scrollMarginTop);return Math.abs(top-inset)<=2;});
        await page.evaluate(()=>document.activeElement?.blur());
        await page.locator('#next img').evaluateAll(es=>Promise.all(es.map(e=>{e.loading='eager';return e.decode();})));
        const layout=await page.locator('#next').evaluate(e=>({font:getComputedStyle(e.querySelector('.upcoming-paper')).fontFamily,overflow:document.documentElement.scrollWidth>innerWidth,boxes:[...e.querySelectorAll('.upcoming-paper,.upcoming-pipeline li,.next-example')].map(n=>{const r=n.getBoundingClientRect();return {left:r.left,right:r.right,textFits:[...n.querySelectorAll('p,h3,h4,cite,.upcoming-model,.sound-play')].every(t=>{const range=document.createRange();range.selectNodeContents(t);return [...range.getClientRects()].every(q=>q.left>=r.left&&q.right<=r.right)})}})}));
        assert(!layout.overflow,JSON.stringify({width,layout}));assert(layout.font.includes('Playful'));
        assert(layout.boxes.every(b=>b.left>=0&&b.right<=width+1&&b.textFits),JSON.stringify({width,layout}));
        assert.equal(await page.locator('#next .upcoming-pipeline').evaluate(e=>getComputedStyle(e).gridTemplateColumns.split(' ').length),width<=800?1:4);
        await page.locator('#next').screenshot({path:path.join(out,'next-'+width+'.png'),animations:'disabled'});
        const toggle=page.locator('.reading-toggle');await toggle.click();
        const link=page.locator('.reading-menu a[href="#next"]');assert(await link.isVisible());assert.equal(await link.getAttribute('aria-current'),'location');await page.keyboard.press('Escape');
      }
      assert(await page.evaluate(()=>document.fonts.check('20px Playful')));
      const cdp=await page.context().newCDPSession(page);await cdp.send('DOM.enable');await cdp.send('CSS.enable');
      const dom=await cdp.send('DOM.getDocument'),node=await cdp.send('DOM.querySelector',{nodeId:dom.root.nodeId,selector:'.upcoming-copy p'});
      const fonts=await cdp.send('CSS.getPlatformFontsForNode',{nodeId:node.nodeId});assert(fonts.fonts.some(f=>f.isCustomFont&&f.familyName==='ZCOOL KuaiLe'));report.upcomingFonts=fonts.fonts;await cdp.detach();
      const nojs=await browser.browser().newContext({javaScriptEnabled:false,reducedMotion:'reduce',viewport:{width:390,height:844}});const np=await nojs.newPage();await np.goto(url+'#next');await np.evaluate(()=>document.fonts.ready);
      assert(await np.locator('#next .upcoming-paper').isVisible());assert.equal(await np.locator('#next .reveal').evaluate(e=>getComputedStyle(e).opacity),'1');assert(await np.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
      assert.equal(await np.locator('#next audio[controls]').count(),12);
      await np.locator('#next').screenshot({path:path.join(out,'next-nojs-390.png')});await nojs.close();
      report.checks.push('Next step: revised navigation and DOM order, 18px desktop font, paper attribution, six figures/twelve matched audios fetched with SHA256 and played to completion, exclusive playback/local status, unique ids, keyboard/reading menu, five widths, actual cute font, reduced motion/no-JS');
    }
    assert.deepEqual(report.errors,[]);assert.deepEqual(report.requests,[]);report.success=true;return;
  }

  assert.equal(await page.locator('.site-header a[href="#guide"]').count(),0);
  assert.equal(await page.locator('.small-ipa').count(),0);
  assert.equal(await page.locator('.tool-divider').count(),15);
  assert.equal(await page.locator('.scroll-hint').count(),0);
  assert.deepEqual(await page.locator('nav[aria-label="主导航"] > a').evaluateAll(es=>es.map(e=>e.getAttribute('href'))),['#tutorials','#lab','#methods','#faq','#updates','#next','#feedback','#contact','https://github.com/Nephelium/PhoneticToolbox']);
  const heading=page.locator('.custom-heading'),actions=page.locator('.download-actions');
  const before=await heading.evaluate(e=>getComputedStyle(e).transform);
  await page.mouse.move(1170,500);await page.waitForTimeout(600);
  assert.notEqual(await heading.evaluate(e=>getComputedStyle(e).transform),before);
  assert.notEqual(await actions.evaluate(e=>getComputedStyle(e).transform),'none');
  await page.mouse.move(0,0);await page.waitForTimeout(600);
  report.checks.push('Hero heading and buttons track mouse gently; fifteen decorative separators; removed top guide link and extra IPA row');
  assert.match(await page.locator('.web-release').innerText(),/2026 年 10 月 31 日/);assert.equal(await page.getByText('用于语音学研究与教学',{exact:true}).count(),0);
  assert.equal(await page.locator('.github-button').getAttribute('href'),'https://github.com/Nephelium/PhoneticToolbox');
  assert.equal(await page.locator('.float-wrap img').evaluateAll(es=>es.map(e=>e.alt).join(',')),'[ʂ],[ɻ],[ɕ]');
  await verifyDownloads();
  await page.waitForTimeout(2900);await page.screenshot({path:path.join(out,'desktop-hero.png'),animations:'disabled'});
  const stage=page.locator('.home-stage');await stage.scrollIntoViewIfNeeded();await page.waitForTimeout(1300);assert.equal(await stage.evaluate(e=>getComputedStyle(e).opacity),'1');
  const angle1=await stage.evaluate(e=>e.style.getPropertyValue('--angle'));await page.mouse.wheel(0,200);await page.waitForTimeout(700);const angle2=await stage.evaluate(e=>e.style.getPropertyValue('--angle'));assert.notEqual(angle1,angle2);
  await page.screenshot({path:path.join(out,'scroll-home.png'),animations:'disabled'});
  await page.locator('.ipa-showcase').scrollIntoViewIfNeeded();await page.waitForTimeout(1300);await page.screenshot({path:path.join(out,'scroll-ipa.png'),animations:'disabled'});
  const opener=page.locator('#ipa-stack .stack-card').last();await opener.click();assert.equal(await page.locator('#screen-viewer').evaluate(e=>e.open),true);await page.keyboard.press('Escape');assert.equal(await page.locator('#screen-viewer').evaluate(e=>e.open),false);assert.equal(await opener.evaluate(e=>document.activeElement===e),true);
  for(const section of await page.locator('.reveal:visible').all()){await section.scrollIntoViewIfNeeded();await page.waitForTimeout(100);}
  await page.waitForTimeout(1200);assert.equal(await page.locator('.reveal:not(.is-visible):visible').count(),0);
  const sequence=await page.locator('.showcase').evaluateAll(es=>es.map(e=>e.querySelector('[data-screen]').dataset.caption.endsWith('浅色')?'light':'dark'));
  assert.deepEqual(sequence,[...Array.from({length:13},(_,i)=>i%2?'dark':'light'),'light','dark','dark']);
  assert.equal(await page.locator('.appearance-showcase,#appearance-heading').count(),0);
  assert(await page.locator('#background').evaluate(e=>e.previousElementSibling.classList.contains('tool-divider')&&e.previousElementSibling.previousElementSibling.id==='guide'&&e.nextElementSibling.classList.contains('chapter-divider')&&e.nextElementSibling.nextElementSibling.id==='tutorials'));
  const backgrounds=await page.locator('#background-stack .stack-card').evaluateAll(es=>es.map(e=>({src:e.href,z:Number(getComputedStyle(e).zIndex)})));
  assert.equal(backgrounds.length,4);assert(backgrounds.sort((a,b)=>b.z-a.z)[0].src.endsWith('custom-background-home-dark.png'));
  assert.equal(await page.locator('#background .section-label').innerText(),'自定义背景图');
  assert.equal(await page.getByText('素材与字体',{exact:true}).count(),0);
  assert.equal(await page.locator('.showcase figcaption > .palette-dot').count(),16);
  assert(await page.locator('[data-screen] img').evaluateAll(es=>es.every(e=>e.complete&&e.naturalWidth>0)));
  assert(await page.locator('[data-screen] img').evaluateAll(es=>es.every(e=>(e.src.includes('lip-nord')||e.src.includes('custom-background-'))?e.naturalWidth===2559&&e.naturalHeight===1527:e.naturalWidth===2560&&e.naturalHeight===1440)));
  report.checks.push('Twenty-eight real screenshots loaded; existing tool themes retained; four user background screenshots after manual in requested front-to-back order; palette overview removed');
  assert.equal(await page.locator('.expand-hint').count(),1);assert(await page.locator('.showcase').first().locator('.expand-hint').count());
  assert.equal(await page.locator('a[href="https://www.zhihu.com/people/jingshaozi"]').count(),1);
  assert.equal(await page.locator('.author-avatar').getAttribute('href'),'https://www.zhihu.com/people/jingshaozi');
  const article=page.locator('#spectrum .article-button');assert.equal(await article.count(),1);assert.equal(await article.getAttribute('href'),'https://www.zhihu.com/question/793458414/answer/4607998600');assert.equal(await article.getAttribute('target'),'_blank');assert.match(await article.getAttribute('rel'),/noopener/);assert.doesNotMatch(await article.innerText(),/https?:/);
  assert.equal(await page.locator('a[href="https://www.zhihu.com/question/793458414/answer/4607998600"]').count(),1);
  const articleTarget=await article.getAttribute('href');await page.context().route(articleTarget,r=>r.fulfill({status:200,contentType:'text/html',body:'<!doctype html><title>Article navigation target</title>'}));
  await article.scrollIntoViewIfNeeded();const popupReady=page.waitForEvent('popup');await article.click();const articlePage=await popupReady;await articlePage.waitForLoadState('domcontentloaded');assert.equal(articlePage.url(),articleTarget);await articlePage.close();await page.context().unroute(articleTarget);await page.screenshot({path:path.join(out,'spectrum-article-button.png'),animations:'disabled'});
  report.checks.push('Spectrogram article button opens the exact requested Zhihu answer in a new tab; URL is not printed as button text');

  assert.equal(await page.locator('#author-name').innerText(),'井韶子');assert.equal(await page.locator('#author-name').evaluate(e=>e.closest('a')),null);
  assert.equal(await page.locator('.note-promise').innerText(),'嗯……最重要的一点：本项目，以及基于本项目的任何修改版，均不得以盈利为目的，不得用于任何商业用途，也不得以任何形式向任何用户收取任何费用。');
  assert.equal(await page.locator('.note-wish').innerText(),'希望语音学的门槛，更低一些\n也希望各位，科研之路顺利。');assert.match(await page.locator('.note-body').last().innerText(),/版权声明、许可证和第三方声明/);
  await page.locator('.author-note').scrollIntoViewIfNeeded();await page.waitForTimeout(500);
  assert(await page.locator('.author-note img').evaluateAll(es=>es.every(e=>e.complete&&e.naturalWidth>0)));
  assert(await page.evaluate(()=>document.fonts.check('24px PhoneticWhisper')));
  assert.match(await page.locator('.note-body').last().innerText(),/请不要删掉/);
  await page.locator('.note-body').last().evaluate(e=>{e.innerHTML=e.innerHTML.replace('删掉','<span data-font-audit>删掉</span>')});
  const cdp=await page.context().newCDPSession(page);await cdp.send('DOM.enable');await cdp.send('CSS.enable');
  const documentNode=await cdp.send('DOM.getDocument');const auditNode=await cdp.send('DOM.querySelector',{nodeId:documentNode.root.nodeId,selector:'[data-font-audit]'});
  const actualFonts=await cdp.send('CSS.getPlatformFontsForNode',{nodeId:auditNode.nodeId});
  assert(actualFonts.fonts.some(f=>f.isCustomFont&&f.familyName==='Phonetic Whisper'&&f.glyphCount===2),JSON.stringify(actualFonts));report.deletedWordFonts=actualFonts.fonts;
  await page.locator('[data-font-audit]').evaluate(e=>e.replaceWith(document.createTextNode(e.textContent)));await cdp.detach();
  report.checks.push('Single original-image hint; author bubble and supplied media; only avatar links to the Zhihu profile; custom derived lettering loaded');
  assert.equal(await page.locator('.more-lettering').count(),0);
  assert(await page.locator('#tutorials').evaluate(e=>e.previousElementSibling.classList.contains('chapter-divider')&&e.previousElementSibling.previousElementSibling.id==='background'));
  for(const id of ['ipa-stack','transcription-stack','egg-stack','spectrum-stack','pitch-stack','phonology-stack','background-stack']){
   const cards=page.locator('#'+id+' .stack-card');assert.equal(await cards.count(),id==='background-stack'?4:['ipa-stack','spectrum-stack','phonology-stack'].includes(id)?3:2);
   const rotations=await cards.evaluateAll(es=>es.map(e=>{const m=new DOMMatrixReadOnly(getComputedStyle(e).transform);return Math.atan2(m.b,m.a)*180/Math.PI}));assert(rotations[0]>5&&rotations[1]<-3,JSON.stringify({id,rotations}));
   for(const card of await cards.all()){await page.evaluate(()=>document.activeElement.blur());await card.scrollIntoViewIfNeeded();await page.waitForTimeout(750);const point=await card.evaluate(e=>{const r=e.getBoundingClientRect();for(let y=Math.max(0,r.top)+8;y<Math.min(innerHeight,r.bottom)-8;y+=12)for(let x=Math.max(0,r.left)+8;x<Math.min(innerWidth,r.right)-8;x+=12){if(document.elementFromPoint(x,y)?.closest('[data-screen]')===e)return {x,y}}return null});assert(point,'Each overlapping card must expose a clickable region');await page.mouse.click(point.x,point.y);assert.equal(await page.locator('#viewer-image').getAttribute('src'),await card.evaluate(e=>e.href));await page.keyboard.press('Escape');}
  }
  assert.equal(await page.locator('.stack-tab').count(),0);report.checks.push('Each of nineteen overlapping cards has an exposed clickable region and opens its own original; no labels');
  assert.equal(await page.locator('#phonology .example-note').innerText(),'（声调仅为举例，并非真实调类）');
  const phonologyCards=await page.locator('#phonology-stack .stack-card').evaluateAll(es=>es.map(e=>({src:e.href,z:Number(getComputedStyle(e).zIndex),caption:e.dataset.caption})));
  assert.equal(phonologyCards.length,3);assert(phonologyCards.every(e=>e.caption.includes('声调仅为举例，并非真实调类')));assert(phonologyCards.sort((a,b)=>b.z-a.z)[0].src.endsWith('phonology-matrix-gruvbox-light.png'));
  const cardRotations=await page.locator('#phonology-stack .stack-card').evaluateAll(es=>es.map(e=>{const m=new DOMMatrixReadOnly(getComputedStyle(e).transform);return Math.atan2(m.b,m.a)*180/Math.PI}));assert(cardRotations[0]>5&&cardRotations[1]<-3&&cardRotations[2]>1,JSON.stringify(cardRotations));
  for(const id of ['phonation','alignment','annotation','lpc','phonology','background','lab','faq','updates','methods','feedback','contact','a-small-note']){await page.locator('#'+id).scrollIntoViewIfNeeded();await page.waitForTimeout(800);await page.screenshot({path:path.join(out,'scroll-'+id+'.png'),animations:'disabled'});}
  assert.equal(await page.locator('.site-footer').evaluate(e=>getComputedStyle(e).backgroundColor),'rgb(255, 233, 107)');
  await page.evaluate(()=>{document.activeElement.blur();window.scrollTo({top:0,behavior:'instant'});});await page.waitForTimeout(800);await page.screenshot({path:path.join(out,'desktop-full.png'),fullPage:true,animations:'disabled'});
  report.checks.push('Scroll reveal, perspective changes, image viewer open/Escape/focus return, yellow footer');
  // Server counters render aggregates only, with an honest unavailable state.
  await page.waitForFunction(()=>document.querySelector('[data-stat="views"]').textContent!=='—');
  assert.match(await page.locator('[data-stats-since]').innerText(),/北京时间 \d{2}:\d{2}:\d{2} 起计数 · 每分钟更新/);
  const statUrl=new URL('/site-stats.json',url).href;
  await page.route(statUrl,r=>r.fulfill({status:200,contentType:'application/json',body:'{"schema":1,"pageViews":-5}'}));
  await page.reload();await ready();await page.waitForFunction(()=>document.querySelector('[data-stats-since]').textContent.includes('暂时休息'));
  assert.equal(await page.locator('[data-stat="views"]').innerText(),'—');
  await page.unroute(statUrl);await page.reload();await ready();
  await page.locator('.site-stats').scrollIntoViewIfNeeded();await page.waitForTimeout(500);await page.screenshot({path:path.join(out,'site-stats.png'),animations:'disabled'});
  report.checks.push('Server aggregates render; invalid counters never become fake zeros');
  // Listening examples are real, decodable assets, mutually exclusive and user-triggered.
  const audioRows=await page.locator('.sound-option audio').evaluateAll(es=>es.map(e=>({id:e.id,src:e.src,paused:e.paused,autoplay:e.autoplay})));
  assert.equal(audioRows.length,34);assert(audioRows.every(x=>x.paused&&!x.autoplay));
  for(const row of audioRows){
   const response=await page.request.get(row.src);assert.equal(response.status(),200);const bytes=await response.body();assert.equal(bytes.toString('ascii',0,4),'RIFF');assert(bytes.length>10000);
  }
  await page.locator('#lab').scrollIntoViewIfNeeded();await page.waitForTimeout(1000);
  await page.locator('#audio-pitch-slower').evaluate(e=>e.playbackRate=0.25);
  await page.locator('[data-audio="audio-pitch-slower"]').click();
  await page.waitForFunction(()=>!document.getElementById('audio-pitch-slower').paused);
  await page.locator('#audio-vowel-i').evaluate(e=>e.playbackRate=0.25);
  await page.locator('[data-audio="audio-vowel-i"]').click();
  await page.waitForFunction(()=>!document.getElementById('audio-vowel-i').paused&&document.querySelector('[data-audio="audio-vowel-i"]').getAttribute('aria-pressed')==='true');
  assert(await page.locator('#audio-pitch-slower').evaluate(e=>e.paused));
  assert.equal(await page.locator('[data-audio="audio-vowel-i"]').getAttribute('aria-pressed'),'true');
  await page.locator('[data-audio="audio-vowel-i"]').click();assert(await page.locator('#audio-vowel-i').evaluate(e=>e.paused));
  await page.locator('[data-audio="audio-phonation-4"]').focus();await page.keyboard.press('Enter');
  await page.waitForFunction(()=>!document.getElementById('audio-phonation-4').paused);
  await page.waitForFunction(()=>document.getElementById('audio-phonation-4').ended);
  assert.equal(await page.locator('[data-audio="audio-phonation-4"]').getAttribute('aria-pressed'),'false');
  await page.locator('.faq-list summary').first().click();assert(await page.locator('.faq-list details').first().evaluate(e=>e.open));
  await verifyReleases();
  assert.match(await page.locator('[data-stats-since]').innerText(),/北京时间 \d{2}:\d{2}:\d{2} 起计数/);
  await page.locator('#feedback-module').fill('LPC 谱图');await page.locator('#feedback-text').fill('修改频率范围后如何保存？');
  assert.equal(await page.locator('#feedback a[href*="github.com"]').count(),0);
  assert(await page.locator('#feedback-email').isVisible());assert(await page.locator('#feedback-submit').isDisabled());
  assert(await page.locator('[data-audio="audio-phonation-endpoint-1"]').isVisible());assert(await page.locator('[data-audio="audio-phonation-endpoint-2"]').isVisible());
  assert.equal(await page.locator('.lab-recipe audio').count(),0);assert.match(await page.locator('.continuum-card').innerText(),/基频不同/);
  const audioManifest=JSON.parse(await fs.readFile(path.join(site,'assets/audio/provenance.json'),'utf8'));
  const phonation=audioManifest.files.filter(r=>r.file.startsWith('phonation-'));assert.equal(phonation.length,11);
  assert(phonation.every(r=>r.parameters.f0_measurement.backend==='parselmouth'));
  assert.equal(phonation.find(r=>r.file==='phonation-endpoint-1.wav').parameters.input.origin,'SCSC');assert.equal(phonation.find(r=>r.file==='phonation-endpoint-2.wav').parameters.input.origin,'SCSC');
  assert.equal(await page.locator('.audio-origin').innerText(),'原声 A：SCSC 录音 · 原声 B：SCSC 录音');
  const steps=phonation.filter(r=>!r.file.includes('endpoint'));assert.equal(steps.length,9);assert(steps.every(r=>r.parameters.kind==='f0_and_phonation'&&r.parameters.delivery==='user_export'&&r.sha256===r.parameters.source_output_sha256));
  assert.equal(await page.locator('.continuum .sound-option').count(),9);assert.match(await page.locator('.continuum-card').innerText(),/九步/);
  for(const [i,name] of ['dl_m02_0558_ma.wav','dl_m05_0558_ma.wav'].entries())assert.equal(phonation.find(r=>r.file===`phonation-endpoint-${i+1}.wav`).parameters.input.file,name);
  await page.locator('.continuum-card .lab-recipe summary').click();
  assert.match(await page.locator('.continuum-card .lab-recipe').innerText(),/Praat \/ Parselmouth/);assert.match(await page.locator('.continuum-card .lab-recipe').innerText(),/同时改变发声类型与基频/);assert.doesNotMatch(await page.locator('.continuum-card .lab-recipe').innerText(),/仅改变发声类型|保留 A 的基频轮廓/);
  for(let i=0;i<9;i++)assert((await page.locator('.continuum .sound-option').nth(i).innerText()).includes(Math.round(steps[i].median_f0)+' Hz'));
  for(const row of phonation){const response=await page.request.get(new URL(row.file,await page.locator('#audio-phonation-1').evaluate(e=>e.src)).href);assert.equal(createHash('sha256').update(await response.body()).digest('hex'),row.sha256);}
  report.checks.push('m02/m05 original endpoints and nine exact user-export files match their SHA256; Praat/Parselmouth recipe and both SCSC labels agree');

  assert.equal(await page.locator('.email-card').getAttribute('href'),'mailto:jinglw3@mail2.sysu.edu.cn');
  assert.equal(await page.locator('.qq-card').getAttribute('data-copy'),'698116028');
  await page.locator('.download.primary').hover();assert.equal(await page.locator('#download-tip-portable').evaluate(e=>getComputedStyle(e).visibility),'visible');
  await page.locator('.download.primary').focus();await page.keyboard.press('Tab');assert.equal(await page.locator('#download-tip-installer').evaluate(e=>getComputedStyle(e).visibility),'visible');
  report.checks.push('34 audio players sharing 31 WAVs; no autoplay; exclusive playback, pause/end, keyboard; FAQ/release, server feedback with email gate, visible original voices, public contact details, hover/focus download help');
  await page.locator('.reconstruction-card').scrollIntoViewIfNeeded();await page.waitForTimeout(800);
  assert.equal(await page.locator('.reconstruction-example').count(),3);
  for(const name of ['smith','levantan','observo']){
    await page.locator('[data-audio="audio-spectrum-'+name+'"]').click();
    await page.waitForFunction(id=>!document.getElementById(id).paused,'audio-spectrum-'+name);
    await page.waitForFunction(id=>document.getElementById(id).ended,'audio-spectrum-'+name);
  }
  await page.screenshot({path:path.join(out,'reconstruction-desktop.png'),animations:'disabled'});
  const dock=page.locator('.reading-dock'),menu=page.locator('.reading-menu');assert(await dock.isVisible());
  await page.getByRole('button',{name:'展开页面目录'}).click();assert(await menu.isVisible());
  await page.locator('.reading-menu a[href="#methods"]').click();await page.waitForTimeout(1200);assert(await menu.isHidden());
  assert.equal(await page.locator('.reading-menu a[href="#methods"]').getAttribute('aria-current'),'location');
  assert.equal(await page.evaluate(()=>document.activeElement.id),'methods');
  await page.getByRole('button',{name:'展开页面目录'}).click();await page.keyboard.press('Escape');assert(await menu.isHidden());
  assert.equal(await page.evaluate(()=>document.activeElement.className),'reading-toggle');
  await page.getByRole('button',{name:'展开页面目录'}).click();await page.screenshot({path:path.join(out,'reading-menu-desktop.png'),animations:'disabled'});
  await page.keyboard.press('Escape');await page.getByRole('button',{name:'返回顶部'}).click();await page.waitForFunction(()=>scrollY<2);assert(await dock.isHidden());
  await page.screenshot({path:path.join(out,'navigation-desktop.png'),animations:'disabled'});
  report.checks.push('Three attributed figure reconstructions play to completion; reading progress/section navigation, current marker, Escape/focus return and back-to-top pass');
  for(const width of [1440,800,390,320]){
   if(width===390){await page.setViewportSize({width,height:900});await page.locator('.reconstruction-card').scrollIntoViewIfNeeded();await page.waitForTimeout(700);await page.screenshot({path:path.join(out,'reconstruction-mobile.png'),animations:'disabled'});await page.locator('.reading-toggle').click();await page.screenshot({path:path.join(out,'reading-menu-mobile.png'),animations:'disabled'});assert(await page.locator('.reading-menu').evaluate(e=>{const r=e.getBoundingClientRect();return r.left>=0&&r.right<=innerWidth&&r.top>=0}));await page.keyboard.press('Escape');}

   await page.setViewportSize({width,height:width<800?844:1000});await page.goto(url);await ready();
   const bounds=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,buttons:[...document.querySelectorAll('.download')].map(e=>({left:e.getBoundingClientRect().left,right:e.getBoundingClientRect().right,scroll:e.scrollWidth,client:e.clientWidth}))}));assert(bounds.scroll<=width,JSON.stringify(bounds));for(const b of bounds.buttons){assert(b.left>=0&&b.right<=width,JSON.stringify(bounds));}
   assert.equal(await page.locator('nav[aria-label="主导航"] > a:visible').count(),9);
   await page.evaluate(()=>document.activeElement?.blur());
   await page.screenshot({path:path.join(out,'viewport-'+width+'.png'),animations:'disabled'});report.checks.push('Viewport '+width+': eight visible navigation links; no document or button overflow');
   assert(await page.locator('.article-button').evaluate(e=>{const r=e.getBoundingClientRect();return r.left>=0&&r.right<=innerWidth}));
   if(width===390){await page.locator('.author-note').scrollIntoViewIfNeeded();await page.waitForTimeout(1000);await page.locator('.author-note').screenshot({path:path.join(out,'mobile-author.png'),animations:'disabled'});}
   for(const id of ['lab','next','faq','updates','methods','feedback','contact']){
    await page.locator('#'+id).scrollIntoViewIfNeeded();await page.waitForTimeout(350);
    const r=await page.locator('#'+id).evaluate(e=>{const r=e.getBoundingClientRect();return {left:r.left,right:r.right,scroll:document.documentElement.scrollWidth}});assert(r.left>=0&&r.right<=width&&r.scroll<=width,JSON.stringify({id,width,r}));
    if(width===390){
     for(const child of await page.locator('#'+id+' .reveal:visible').all()){await child.scrollIntoViewIfNeeded();await page.waitForTimeout(100);}
     await page.waitForTimeout(1000);await page.evaluate(()=>document.activeElement?.blur());
     await page.locator('#'+id).screenshot({path:path.join(out,'mobile-'+id+'.png'),animations:'disabled'});
    }
   }
  }
  await page.emulateMedia({reducedMotion:'reduce'});await page.goto(url);await ready();assert.equal(await page.locator('.float-wrap').first().evaluate(e=>getComputedStyle(e).animationName),'none');assert.equal(await page.locator('.reveal').first().evaluate(e=>getComputedStyle(e).opacity),'1');report.checks.push('Reduced motion: no floating and content visible');
  const nojs=await browser.browser().newContext({javaScriptEnabled:false,viewport:{width:390,height:844}});const np=await nojs.newPage();await np.goto(url);assert.equal(await np.locator('.reveal').first().evaluate(e=>getComputedStyle(e).opacity),'1');assert.equal(await np.locator('[data-screen]').count(),26);await nojs.close();report.checks.push('No JavaScript: content and twenty-six image links remain available');
  await page.setViewportSize({width:1600,height:1000});await page.goto(url);await ready();
  const images=await page.locator('img:not(#viewer-image)').evaluateAll(es=>es.filter(e=>!e.loading||e.loading!=='lazy').map(e=>({src:e.getAttribute('src'),ok:e.complete&&e.naturalWidth>0})));assert(images.every(e=>e.ok),JSON.stringify(images));
  assert.deepEqual(report.errors,[]);assert.deepEqual(report.requests,[]);report.success=true;
 }catch(e){report.failure=String(e);if(page)await page.screenshot({path:path.join(out,'failure.png')}).catch(()=>{});throw e;}
 finally{
  await browser?.close();if(server)await new Promise(resolve=>server.close(resolve));await write('report.json',report);
  const files=[];async function inventory(dir){for(const entry of await fs.readdir(dir,{withFileTypes:true})){assert(!entry.isSymbolicLink());const file=path.join(dir,entry.name);if(entry.isDirectory())await inventory(file);else{const bytes=await fs.readFile(file);files.push({path:path.relative(scratch,file),bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex')});}}}await inventory(scratch);
  await write('scratch-manifest.json',{schema:'ptb-test-scratch-files/1',files});await write('run.json',{...marker,state:'ready',test_outcome:report.success?'completed':'failed'});
  const cleanup=spawnSync('powershell.exe',['-NoProfile','-ExecutionPolicy','Bypass','-File',path.join(root,'scripts/cleanup_test_scratch.ps1'),'-RunDirectory',out],{windowsHide:true,encoding:'utf8'});assert.equal(cleanup.status,0,cleanup.stdout+cleanup.stderr);console.log(JSON.stringify({out,...report}));
 }
}
main().catch(e=>{console.error(e);process.exitCode=1});
