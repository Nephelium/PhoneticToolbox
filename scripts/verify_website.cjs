// Static landing page verification in a task-owned browser and local server.
const fs=require('node:fs/promises'),path=require('node:path'),http=require('node:http'),assert=require('node:assert/strict');
const {spawnSync}=require('node:child_process'),{createHash}=require('node:crypto');
const root=path.resolve(__dirname,'..'),site=path.resolve(process.env.WEBSITE_DIRECTORY||path.join(root,'temporary-site'));
async function main(){
 const out=path.join(root,'output/validation/website','verify-'+Date.now()+'-'+process.pid),scratch=path.join(out,'scratch');await fs.mkdir(scratch,{recursive:true});
 const write=(name,data)=>fs.writeFile(path.join(out,name),JSON.stringify(data,null,2)+'\n');
 const marker={schema:'ptb-test-run/1',owner_pid:process.pid,state:'active',started_at:new Date().toISOString(),recipe:'Static website, task-owned HTTP server and isolated Chrome; screenshots, navigation, reduced motion, no JavaScript'};
 await write('run.json',marker);let server,browser,page;const report={success:false,checks:[],errors:[],requests:[]};
 try{
  server=http.createServer(async(req,res)=>{try{const url=new URL(req.url,'http://localhost');if(process.env.WEBSITE_STATS_DIRECTORY&&['/site-stats.json','/site-stats-history.json'].includes(url.pathname)){res.writeHead(200,{'Content-Type':'application/json'}).end(await fs.readFile(path.join(process.env.WEBSITE_STATS_DIRECTORY,url.pathname.slice(1))));return;}if(url.pathname==='/site-stats-history.json'){res.writeHead(200,{'Content-Type':'application/json'}).end(JSON.stringify({schema:1,points:[]}));return;}if(url.pathname==='/site-stats.json'){res.writeHead(200,{'Content-Type':'application/json'}).end(JSON.stringify({schema:1,startedAt:'2026-10-12T00:00:00Z',updatedAt:new Date().toISOString(),pageViews:1234,downloads:{total:56,portable:40,installer:16}}));return;}const relative=decodeURIComponent(url.pathname),file=path.resolve(site,'.'+relative+(relative.endsWith('/')?'index.html':''));if(!file.startsWith(site+path.sep)){res.writeHead(403).end();return;}const bytes=await fs.readFile(file);const mime={'.html':'text/html; charset=utf-8','.css':'text/css; charset=utf-8','.js':'application/javascript; charset=utf-8','.png':'image/png','.svg':'image/svg+xml','.ttf':'font/ttf','.txt':'text/plain; charset=utf-8','.json':'application/json','.wav':'audio/wav','.jpg':'image/jpeg'};res.writeHead(200,{'Content-Type':mime[path.extname(file)]||'application/octet-stream'}).end(bytes);}catch{res.writeHead(404).end();}});
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));const url=process.env.WEBSITE_ORIGIN||'http://127.0.0.1:'+server.address().port+'/';report.origin=url;
  const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  browser=await chromium.launchPersistentContext(path.join(scratch,'profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:process.env.WEBSITE_STATS_ONLY!=='1',viewport:{width:1600,height:1000},deviceScaleFactor:(process.env.WEBSITE_RIBBON_ONLY==='1'||process.env.WEBSITE_CODE_ONLY==='1')?3:1,userAgent:'Mozilla/5.0 PTB-Website-Verify'});
  report.browser=browser.browser().version();page=await browser.newPage();page.on('pageerror',e=>report.errors.push(e.message));page.on('response',r=>{if(r.status()>=400)report.requests.push({url:r.url(),status:r.status()});});
  const navigationTimeout=Number(process.env.WEBSITE_NAVIGATION_TIMEOUT_MS||30000);
  assert(Number.isInteger(navigationTimeout)&&navigationTimeout>=30000&&navigationTimeout<=120000);
  page.setDefaultNavigationTimeout(navigationTimeout);report.navigationTimeout=navigationTimeout;
  async function ready(){await page.evaluate(()=>document.fonts.ready);await page.waitForTimeout(250);}
  let publishedRelease;
  async function assertMacPanelUncovered(label){
    const result=await page.locator('#download-tip-macos').evaluate(panel=>{
      const r=panel.getBoundingClientRect(),blocked=[];
      for(let y=r.top+12;y<r.bottom-12;y+=16){
        for(const x of [r.left+12,r.left+r.width/2,r.right-12]){
          const hit=document.elementFromPoint(x,y);
          if(!hit||!panel.contains(hit))blocked.push({x,y,hit:hit?.className||hit?.tagName||null});
        }
      }
      return {rect:{x:r.x,y:r.y,width:r.width,height:r.height},blocked};
    });
    assert.equal(result.blocked.length,0,JSON.stringify({label,...result}));
    return result.rect;
  }
  async function verifyDownloads(){
    const metadata=JSON.parse(await fs.readFile(path.join(site,'assets/downloads.json'),'utf8'));
    if(process.env.WEBSITE_ORIGIN){
      const response=await page.request.get(new URL('/releases/windows-x64/stable/latest.json?website-check='+Date.now(),url).href);
      assert.equal(response.status(),200);publishedRelease=await response.json();assert.equal(publishedRelease.version,metadata.version);
      const installer=publishedRelease.packages.find(p=>p.kind==='installer'&&p.platform==='windows-x64');assert(installer);
      assert.equal(metadata.files[installer.name].size,installer.size);assert.equal(metadata.files[installer.name].sha256,installer.sha256);
      report.checks.push('Published website version and installer identity match the stable update channel');
    }
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
    if(metadata.macos){
      const info=metadata.macos,mac=page.getByRole('link',{name:'下载 macOS 测试版',exact:true});
      assert.equal(info.platform,'macos-arm64');assert.equal(info.channel,'preview');assert.equal(info.notarized,false);
      assert(info.size>0&&info.size<500000000);assert.match(info.sha256,/^[0-9a-f]{64}$/);
      assert.equal(await mac.getAttribute('href'),info.url);assert.equal(await mac.getAttribute('download'),info.name);
      const panel=page.locator('#download-tip-macos'),help=page.locator('.macos-help');
      assert((await panel.textContent()).includes((info.size/1e6).toFixed(2)+' MB'));
      assert((await page.locator('.platform-macos').innerText()).includes('Apple 芯片'));
      assert.equal(info.instructions,'/#macos-download');
      assert.equal(await page.getByRole('link',{name:'安装说明',exact:true}).count(),0);
      await page.evaluate(()=>document.activeElement?.blur());await page.mouse.move(0,0);await page.waitForTimeout(180);assert(!(await panel.isVisible()));
      await mac.hover();assert(await panel.isVisible());
      const copy=await panel.innerText();for(const text of [info.sha256,'系统设置 → 隐私与安全性 → 仍要打开','Apple 芯片（M 系列）','PhoneticToolbox.app'])assert(copy.includes(text));assert(!/几何|失败|尚未实测|MFA/.test(copy));assert.equal(await panel.locator('ol li').count(),3);
      const apple=panel.getByRole('link',{name:'Apple 官方说明 ↗'});assert.equal(await apple.getAttribute('href'),'https://support.apple.com/zh-cn/102445');await apple.hover();await page.waitForTimeout(200);assert(await panel.isVisible());
      assert(await apple.evaluate(e=>{const r=e.getBoundingClientRect(),hit=document.elementFromPoint(r.left+r.width/2,r.top+r.height/2);return hit===e||e.contains(hit);}));
      await page.keyboard.press('Escape');assert(!(await panel.isVisible()));await mac.focus();assert(await panel.isVisible());await page.keyboard.press('Escape');assert(!(await panel.isVisible()));
      if(process.env.WEBSITE_ORIGIN){
        const response=await page.request.head(new URL(info.url,url).href);assert.equal(response.status(),200);assert.equal(Number(response.headers()['content-length']),info.size);
      }
      await page.route('**'+info.url,route=>route.fulfill({status:200,headers:{'content-type':'application/zip','content-disposition':'attachment; filename="'+info.name+'"'},body:'Mac download routing fixture'}));
      const pending=page.waitForEvent('download');await mac.click();const downloaded=await pending;assert.equal(downloaded.suggestedFilename(),info.name);assert.equal(page.url(),url);await page.unroute('**'+info.url);
    }else{
      const mac=page.getByRole('button',{name:'下载 macOS，尚未上线',exact:true});
      assert(await mac.isDisabled());assert.equal(await mac.getAttribute('href'),null);
    }
    assert.equal(await page.locator('.download-actions a.download').count(),metadata.macos?3:2);
    assert(await page.getByRole('button',{name:'网页版，灰测中，即将开放',exact:true}).isDisabled());
    for(const width of [1920,1440,1280,1024,800,390,320]){
      await page.setViewportSize({width,height:1000});await page.evaluate(()=>scrollTo(0,0));await ready();
      await page.evaluate(()=>document.activeElement?.blur());await page.mouse.move(0,0);await page.waitForTimeout(220);
      await page.waitForFunction(()=>getComputedStyle(document.querySelector('.download-actions')).opacity==='1');
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
      const layout=await page.locator('.download-actions').evaluate(e=>{
        const rect=e=>{const r=e.getBoundingClientRect();return {left:r.left,right:r.right,top:r.top,bottom:r.bottom,width:r.width,height:r.height};};
        return {group:rect(e),windows:rect(e.querySelector('.platform-windows')),divider:rect(e.querySelector('.download-divider')),mac:rect(e.querySelector('.platform-macos')),buttons:[...e.querySelectorAll('.download')].map(b=>{const r=rect(b),range=document.createRange();range.selectNodeContents(b.firstChild);const text=range.getBoundingClientRect(),icon=rect(b.querySelector('svg'));return {...r,fits:text.left>=r.left+2&&text.right<=icon.left&&icon.right<=r.right-2};}),notes:[...e.querySelectorAll('.download-note')].map(n=>n.scrollWidth<=n.clientWidth+1)};
      });
      assert(layout.windows.right<layout.divider.left&&layout.divider.right<layout.mac.left,JSON.stringify({width,layout}));
      assert(layout.divider.height>0&&layout.divider.height<layout.buttons[0].height);
      assert(layout.buttons.every(b=>b.fits&&b.left>=layout.group.left-1&&b.right<=layout.group.right+1),JSON.stringify({width,layout}));
      assert(layout.notes.every(Boolean),JSON.stringify({width,layout}));
      assert(Math.abs((await page.locator('.web-entry').boundingBox()).width-layout.group.width)<1);
      await page.screenshot({path:path.join(out,'downloads-'+width+'.png'),animations:'disabled'});
      if(metadata.macos){
        await page.locator('.download.macos').hover();const panel=page.locator('#download-tip-macos');assert(await panel.isVisible());
        const box=await panel.boundingBox();assert(box.x>=0&&box.y>=0&&box.x+box.width<=width+1&&box.y+box.height<=1001,JSON.stringify({width,box}));
        await assertMacPanelUncovered('initial-'+width);
        await page.screenshot({path:path.join(out,'mac-hover-'+width+'.png'),animations:'disabled'});
        await page.mouse.move(0,0);await page.evaluate(()=>document.activeElement?.blur());await page.waitForTimeout(180);assert(!(await panel.isVisible()));
      }
    }
    if(metadata.macos){
      // These desktop widths place the ribbon under the downward panel.
      // On mobile the hero artwork separates them; mobile coverage is checked above.
      for(const width of [1440,1024,825]){
        await page.setViewportSize({width,height:1000});await page.evaluate(()=>scrollTo({top:0,behavior:'instant'}));await ready();
        await page.evaluate(()=>{const r=document.querySelector('.download.macos').getBoundingClientRect();scrollTo({top:scrollY+r.top-120,behavior:'instant'});});await ready();
        await page.locator('.download.macos').hover();
        const panel=await assertMacPanelUncovered('downward-'+width),button=await page.locator('.download.macos').boundingBox(),ribbon=await page.locator('.feature-ribbon').boundingBox();
        assert(panel.y>=button.y+button.height,JSON.stringify({width,panel,button}));
        const overlap={left:Math.max(panel.x,ribbon.x),right:Math.min(panel.x+panel.width,ribbon.x+ribbon.width),top:Math.max(panel.y,ribbon.y),bottom:Math.min(panel.y+panel.height,ribbon.y+ribbon.height)};
        assert(overlap.right>overlap.left&&overlap.bottom>overlap.top,JSON.stringify({width,panel,ribbon,overlap}));
        await page.screenshot({path:path.join(out,'mac-hover-down-'+width+'.png'),animations:'disabled'});
        await page.keyboard.press('Escape');assert(!(await page.locator('#download-tip-macos').isVisible()));
      }
      const touch=await browser.browser().newContext({viewport:{width:390,height:844},isMobile:true,hasTouch:true}),tp=await touch.newPage();await tp.goto(url,{waitUntil:'domcontentloaded'});const summary=tp.locator('.macos-help summary');await summary.tap();assert(await tp.locator('#download-tip-macos').isVisible());await summary.tap();assert(!(await tp.locator('#download-tip-macos').isVisible()));await touch.close();
    }
    await page.setViewportSize({width:1600,height:1000});report.checks.push(metadata.version+' Windows/Mac download URLs and filenames; Mac installation hover panel, pointer entry, Escape/focus and touch toggle; approved copy and clickable Apple link; seven widths without clipping or covered content; three downward panels overlap the ribbon with full-area hit checks; public HEAD checks run with a public origin');
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
    if(publishedRelease){assert.equal(snapshot.tag_name,'v'+publishedRelease.version);assert.equal(snapshot.body,publishedRelease.notes);}
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
  if(process.env.WEBSITE_CODE_ONLY==='1'){
    await page.goto(url);await ready();await page.emulateMedia({reducedMotion:'reduce'});report.inlineCode=[];
    for(const width of [1280,390,320]){
      await page.setViewportSize({width,height:900});await ready();
      const code=page.locator('.next-parameters code').first();await code.scrollIntoViewIfNeeded();
      const metric=await code.evaluate(e=>{
        const original=e.style.overflowX;
        function delta(overflow){
          e.style.overflowX=overflow;
          const inside=document.createElement('span'),outside=document.createElement('span');
          for(const n of [inside,outside]){n.setAttribute('aria-hidden','true');n.style.cssText='display:inline-block;width:0;height:0;padding:0;border:0;margin:0;vertical-align:baseline'}
          e.append(inside);e.after(outside);const d=inside.getBoundingClientRect().bottom-outside.getBoundingClientRect().bottom;inside.remove();outside.remove();return d;
        }
        const current=delta(original),scrolling=delta('auto'),visible=delta('visible');e.style.overflowX=original;
        const style=getComputedStyle(e);return {current,scrolling,visible,overflow:style.overflowX,nowrap:style.whiteSpace,fontRatio:parseFloat(style.fontSize)/parseFloat(getComputedStyle(e.parentElement).fontSize)};
      });
      report.inlineCode.push({width,...metric});
      assert(Math.abs(metric.current)<=.25,'Inline-code baseline differs from surrounding text: '+JSON.stringify(metric));
      assert(metric.scrolling < -2 && Math.abs(metric.visible)<=.25,'Overflow baseline reproduction');
      assert.equal(metric.nowrap,'nowrap');assert(Math.abs(metric.fontRatio-.92)<.001);assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
      await page.locator('.next-parameters').first().screenshot({path:path.join(out,'inline-code-'+width+'-3x.png'),animations:'disabled'});
    }
    assert.deepEqual(report.errors,[]);assert.deepEqual(report.requests,[]);report.checks.push('Numeric inline-code baseline agrees with surrounding text at three widths; original scrolling overflow reproduces raised glyphs; DPR3 screenshots');report.success=true;return;
  }
  if(process.env.WEBSITE_RIBBON_ONLY==='1'){
    await page.goto(url);await ready();await page.emulateMedia({reducedMotion:'reduce'});
    const ribbon=page.locator('.feature-ribbon');
    const names=['声学分析','语音合成','语音标注','感知实验','论文阅读','国际音标'];
    assert.deepEqual(await ribbon.locator(':scope > div > span:last-child').allTextContents(),names);
    assert.equal(await ribbon.locator('.ribbon-icon').count(),6);
    assert(await ribbon.locator('.ribbon-icon').evaluateAll(es=>es.every(e=>{const s=getComputedStyle(e);return s.filter==='none'&&s.transform==='none'&&s.backgroundImage.includes('feature-icons-3d-v1.png')})));
    const art=JSON.parse(await fs.readFile(path.join(site,'assets/feature-icons.provenance.json'),'utf8'));
    const artBytes=await fs.readFile(path.join(site,'assets',art.file));assert.equal(createHash('sha256').update(artBytes).digest('hex'),art.sha256);
    const artUrl=await ribbon.locator('.ribbon-icon').first().evaluate(e=>getComputedStyle(e).backgroundImage.match(/url\(["']?([^"')]+)["']?\)/)[1]);
    const response=await page.request.get(artUrl);assert.equal(response.status(),200);assert.deepEqual(await response.body(),artBytes);
    report.devicePixelRatio=await page.evaluate(()=>devicePixelRatio);assert.equal(report.devicePixelRatio,3);report.layouts=[];
    for(const [width,height] of [[1920,1080],[1440,900],[1280,800],[1024,768],[800,1000],[390,844],[320,740]]){
      await page.setViewportSize({width,height});await page.evaluate(()=>scrollTo(0,0));await ready();
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
      const bounds=await ribbon.boundingBox();
      const tiles=await ribbon.locator(':scope > div').evaluateAll(es=>es.map(e=>{const r=e.getBoundingClientRect();return {x:r.x,y:r.y,right:r.right,bottom:r.bottom}}));
      assert(tiles.every(t=>t.x>=0&&t.right<=width+1&&t.y>=bounds.y&&t.bottom<=bounds.y+bounds.height+1));
      if(width>=1024){assert(bounds.height<=110);assert(bounds.y+bounds.height+26<=height,JSON.stringify({width,height,bounds}));}
      report.layouts.push({width,height,bounds});
      if(width>=1024)await page.screenshot({path:path.join(out,'home-'+width+'.png'),animations:'disabled'});
      else{await ribbon.scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,'ribbon-'+width+'.png'),animations:'disabled'});}
      if(width===1280){await ribbon.screenshot({path:path.join(out,'ribbon-desktop-3x.png'),animations:'disabled'});await ribbon.locator('.ribbon-icon-1').screenshot({path:path.join(out,'synthesis-icon-3x.png'),animations:'disabled'});}
    }
    await page.setViewportSize({width:1280,height:800});
    const code=page.locator('.release-content code').filter({hasText:/^0\.1$/}).first();
    assert.equal(await code.count(),1);
    await code.scrollIntoViewIfNeeded();
    assert(await code.evaluate(e=>{const s=getComputedStyle(e);return s.whiteSpace==='nowrap'&&s.backgroundColor!=='rgba(0, 0, 0, 0)'&&parseFloat(s.fontSize)>=12&&e.getClientRects().length===1}));
    await page.screenshot({path:path.join(out,'release-inline-code.png'),animations:'disabled'});
    const nojs=await browser.newPage({javaScriptEnabled:false});await nojs.goto(url);assert.deepEqual(await nojs.locator('.feature-ribbon > div > span:last-child').allTextContents(),names);await nojs.close();
    assert.deepEqual(report.errors,[]);assert.deepEqual(report.requests,[]);
    report.checks.push('Six 3D miniatures from the original transparent 1536px sheet, HTTP bytes and SHA, no filters or transforms; seven viewport layouts, full desktop first-screen ribbon at 1024/1280/1440/1920; styled unbroken numeric inline code; no-JS content');report.success=true;return;
  }
  if(process.env.WEBSITE_PLATFORM_ONLY==='1'){
    if(process.env.WEBSITE_ORIGIN){
      await page.goto(url);await ready();await page.emulateMedia({reducedMotion:'reduce'});
      await page.waitForFunction(()=>document.querySelector('[data-chart="downloads"] .chart-line'));
      const response=await page.request.get(new URL('/site-stats.json',url).href);assert.equal(response.status(),200);
      const actual=await response.json(),windowsTotal=actual.downloads.portable+actual.downloads.installer;
      assert.equal(actual.downloads.total,windowsTotal+(actual.downloads.macos??0));
      if(actual.downloads.windows!==undefined)assert.equal(actual.downloads.windows,windowsTotal);
      assert((await page.locator('[data-chart="downloads"] output').innerText()).includes('Windows '));
      assert.equal(await page.locator('[data-stats-breakdown]').count(),0);
      report.publicStatistics={updatedAt:actual.updatedAt,windowsTotal,macosTotal:actual.downloads.macos??null,total:actual.downloads.total,selected:await page.locator('[data-chart="downloads"] output').innerText()};
      for(const width of [1440,390]){
        await page.setViewportSize({width,height:1000});await page.locator('.site-stats').scrollIntoViewIfNeeded();await ready();
        await page.locator('.site-stats').screenshot({path:path.join(out,'public-statistics-'+width+'.png'),animations:'disabled'});
      }
    }
    const t=Math.floor(Date.now()/1000),points=[[t-7200,7,250,0,33],[t-600,1230,265,8,33],[t,1260,269,9,33]];
    await page.route('**/site-stats.json',r=>r.fulfill({json:{schema:1,startedAt:new Date((t-7200)*1000).toISOString(),updatedAt:new Date(t*1000).toISOString(),pageViews:1260,downloads:{total:269,portable:120,installer:149},webUses:9,github:{count:33,updatedAt:new Date(t*1000).toISOString()}}}));
    await page.route('**/site-stats-history.json',r=>r.fulfill({json:{schema:1,points}}));
    await page.goto(url);await ready();await page.emulateMedia({reducedMotion:'reduce'});
    await verifyDownloads();
    const card=page.locator('[data-chart="downloads"]');
    await page.waitForFunction(()=>document.querySelector('[data-chart="downloads"] .chart-line'));
    assert.equal(await page.locator('[data-stats-breakdown]').count(),0);
    assert(!(await page.locator('.site-stats').innerText()).match(/免安装版|安装版/));
    assert.equal(await card.locator('.legend-windows').innerText(),'Windows');
    const macAvailable=!!JSON.parse(await fs.readFile(path.join(site,'assets/downloads.json'),'utf8')).macos;
    assert.equal(await card.locator('.legend-macos').innerText(),'macOS');
    assert((await card.locator('output').innerText()).includes('Windows 269 次'));
    await card.locator('input').focus();await page.keyboard.press('Home');
    assert((await card.locator('output').innerText()).includes('Windows 250 次'));
    await page.keyboard.press('End');assert((await card.locator('output').innerText()).includes('Windows 269 次'));
    for(const width of [1440,390,320]){
      await page.setViewportSize({width,height:1000});await page.locator('.site-stats').scrollIntoViewIfNeeded();await ready();
      await page.evaluate(()=>document.activeElement?.blur());
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
      await page.locator('.site-stats').screenshot({path:path.join(out,'statistics-'+width+'.png'),animations:'disabled'});
    }
    const nojs=await browser.browser().newContext({javaScriptEnabled:false,viewport:{width:390,height:844}}),np=await nojs.newPage();await np.emulateMedia({reducedMotion:'reduce'});await np.goto(url,{waitUntil:'load',timeout:navigationTimeout});await np.evaluate(()=>document.fonts.ready);await np.bringToFront();
    if(macAvailable){
      const info=JSON.parse(await fs.readFile(path.join(site,'assets/downloads.json'),'utf8')).macos;
      assert.equal(await np.locator('.download.macos').getAttribute('href'),info.url);
      assert.equal(await np.getByRole('link',{name:'安装说明',exact:true}).count(),0);
      await np.locator('.macos-help summary').scrollIntoViewIfNeeded();await np.locator('.macos-help summary').click();assert(await np.locator('#download-tip-macos').isVisible());assert((await np.locator('#download-tip-macos').innerText()).includes(info.sha256));
      await np.route('**'+info.url,r=>r.fulfill({status:200,headers:{'content-type':'application/zip','content-disposition':'attachment; filename="'+info.name+'"'},body:'No-JS Mac routing fixture'}));
      const pending=np.waitForEvent('download');await np.locator('.download.macos').click();const downloaded=await pending;assert.equal(downloaded.suggestedFilename(),info.name);await downloaded.cancel();
    }else assert(await np.locator('.download.macos').isDisabled());
    assert(await np.locator('.web-entry').isDisabled());
    assert.equal(await np.locator('.download-actions a.download').count(),macAvailable?3:2);assert.equal(await np.locator('[data-stats-breakdown]').count(),0);await nojs.close();
    report.checks.push('Windows aggregate 120+149=269 with original history and keyboard scrubbing; no invented Mac statistics; no package breakdown; desktop/mobile chart screenshots; Mac download and guide links work without JS');
    assert.deepEqual(report.errors,[]);assert.deepEqual(report.requests,[]);report.success=true;return;
  }
  if(process.env.WEBSITE_MACOS_STATS_ONLY==='1'){
    const t=Math.floor(Date.now()/1000),start=t-7200;
    let points=[[start,100,270,3,33,null],[t-3600,101,271,3,null,0],[t-1200,null,null,null,null,1],[t,102,274,3,33,2]];
    let totals={schema:1,startedAt:new Date(start*1000).toISOString(),updatedAt:new Date(t*1000).toISOString(),pageViews:102,downloads:{total:276,windows:274,portable:122,installer:152,macos:2},webUses:3,github:{count:33,updatedAt:new Date(t*1000).toISOString()}};
    if(process.env.WEBSITE_MACOS_STATS_LIVE!=='1'){
      await page.route('**/site-stats.json',r=>r.fulfill({json:totals}));
      await page.route('**/site-stats-history.json',r=>r.fulfill({json:{schema:1,points}}));
    }
    await page.goto(url);await ready();await page.emulateMedia({reducedMotion:'reduce'});
    const card=page.locator('[data-chart="downloads"]'),slider=card.locator('input'),output=card.locator('output');
    await page.waitForFunction(()=>document.querySelectorAll('[data-chart="downloads"] .chart-line').length===2);
    assert.equal(await card.locator('.legend-windows').innerText(),'Windows');assert.equal(await card.locator('.legend-macos').innerText(),'macOS');
    const colors=await card.locator('.chart-line').evaluateAll(es=>es.map(e=>getComputedStyle(e).stroke));assert.equal(new Set(colors).size,2);
    for(const platform of ['windows','macos']){
      const color=await card.locator('.chart-line[data-platform="'+platform+'"]').evaluate(e=>getComputedStyle(e).stroke);
      assert.equal(await card.locator('.chart-dot[data-platform="'+platform+'"]').evaluate(e=>getComputedStyle(e).fill),color);
    }
    const current=await page.request.get(new URL('/site-stats.json',url).href);assert.equal(current.status(),200);
    if(process.env.WEBSITE_MACOS_STATS_LIVE==='1'){
      const data=await current.json();assert.equal(data.downloads.total,data.downloads.windows+data.downloads.macos);
      assert((await output.innerText()).includes('macOS '));report.liveDownloads=data.downloads;
    }else assert.equal(await page.locator('[data-stat="downloads"]').innerText(),'276');
    for(const width of [1920,1440,1280,1024,800,390,320]){
      await page.setViewportSize({width,height:1000});await card.scrollIntoViewIfNeeded();await ready();
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
      const bounds=await output.boundingBox();assert(bounds.x>=0&&bounds.x+bounds.width<=width+1);
      await card.screenshot({path:path.join(out,'macos-downloads-'+width+'.png'),animations:'disabled'});
    }
    await slider.focus();await page.keyboard.press('Home');report.earliest=await output.innerText();
    assert(report.earliest.includes('macOS 暂无记录'),'Unknown historical Mac samples must remain missing');
    await page.keyboard.press('End');report.latest=await output.innerText();assert.match(report.latest,/Windows \d[\d,]* 次 · macOS \d[\d,]* 次/);
    for(const period of ['1','7','30','all'])await page.locator('[data-stats-range]').selectOption(period);
    await page.route('**/site-stats.json',r=>r.fulfill({json:totals}));await page.route('**/site-stats-history.json',r=>r.fulfill({json:{schema:1,points}}));
    async function refresh(){await page.evaluate(()=>document.dispatchEvent(new Event('visibilitychange')));await page.waitForTimeout(350);}
    await refresh();await page.waitForFunction(()=>document.querySelector('[data-stat="downloads"]').textContent==='276');
    await slider.fill(String(t-3600));assert((await output.innerText()).includes('macOS 0 次'));
    await slider.fill(String(t-1200));assert((await output.innerText()).includes('Windows 271 次 · macOS 1 次'));
    const marker=card.locator('.chart-dot[data-platform="macos"]'),x=Number(await marker.getAttribute('cx'));
    assert(Math.abs(x-(38+6000/7200*426))<1e-6);assert.equal(await slider.getAttribute('aria-valuetext'),await output.innerText());
    totals={...totals,downloads:{total:274,portable:122,installer:152}};points=points.filter(p=>p[2]!==null).map(p=>p.slice(0,5));await refresh();
    await page.waitForFunction(()=>document.querySelector('[data-stat="downloads"]').textContent==='274');assert.equal(await card.locator('.chart-line').count(),1);assert((await output.innerText()).includes('macOS 暂无记录'));
    totals={...totals,downloads:{total:999,portable:122,installer:152,macos:2}};await refresh();
    await page.waitForFunction(()=>document.querySelector('[data-stats-updated]').textContent.includes('暂时无法更新'));assert.equal(await page.locator('[data-stat="downloads"]').innerText(),'274');
    const nojs=await browser.browser().newContext({javaScriptEnabled:false});const np=await nojs.newPage();await np.goto(url);assert(await np.locator('.stats-caption noscript').isVisible());assert.equal(await np.locator('.legend-macos').innerText(),'macOS');await nojs.close();
    report.checks.push('Windows/macOS actual totals and independent colors; seven widths; shared timestamp slider and markers; unknown versus zero; sparse platform events; legacy five-column compatibility; invalid totals retain values; no JavaScript');
    assert.deepEqual(report.errors,[]);assert.deepEqual(report.requests,[]);report.success=true;return;
  }
  if(process.env.WEBSITE_STATS_AXIS_ONLY==='1'){
    const t=Math.floor(Date.now()/1000),start=t-3600;
    let points=[[start,1,240,0,33],[t-300,10,242,0,null],[t-120,20,243,0,null],[t-60,21,243,0,null],[t,22,243,0,33]];
    let end=t;
    await page.route('**/site-stats.json',r=>r.fulfill({json:{schema:1,startedAt:new Date(start*1000).toISOString(),updatedAt:new Date(end*1000).toISOString(),pageViews:points.at(-1)?.[1]??24,downloads:{total:243,portable:111,installer:132},webUses:0,github:{count:33,updatedAt:new Date(end*1000).toISOString()}}}));
    await page.route('**/site-stats-history.json',r=>r.fulfill({json:{schema:1,points}}));
    await page.goto(url);await ready();await page.emulateMedia({reducedMotion:'reduce'});
    const card=page.locator('[data-chart="downloads"]'),slider=card.locator('input'),dot=card.locator('.chart-dot');
    await page.waitForFunction(()=>document.querySelector('[data-chart="downloads"] .chart-line'));
    assert.equal(Number(await slider.getAttribute('min')),start,'Slider and plot must share elapsed-time scale, not sample indices');
    assert.equal(Number(await slider.getAttribute('max')),t);
    async function select(stamp){await slider.fill(String(stamp));}
    async function geometry(stamp,value){
      const x=38+(stamp-start)/(end-start)*426,y=153-value/243*125;
      assert(Math.abs(Number(await dot.getAttribute('cx'))-x)<1e-6);assert(Math.abs(Number(await dot.getAttribute('cy'))-y)<1e-6);
      assert.equal(Number(await slider.inputValue()),stamp);assert((await card.locator('output').innerText()).includes(value+' 次'));
    }
    await select(t-120);await geometry(t-120,243);
    await slider.focus();await page.keyboard.press('ArrowLeft');await geometry(t-300,242);
    await page.keyboard.press('ArrowRight');await geometry(t-120,243);
    await page.keyboard.press('End');await geometry(t,243);
    const plateau=await card.locator('.chart-line').getAttribute('d');assert(plateau.endsWith('H464V28'),'243 plateau stays flat to the newest timestamp');
    await select(t-300);end=t+60;points.push([end,23,243,0,null]);
    await page.evaluate(()=>document.dispatchEvent(new Event('visibilitychange')));
    await page.waitForFunction(stamp=>Number(document.querySelector('[data-chart="downloads"] input').max)===stamp,end);
    await geometry(t-300,242);
    await page.keyboard.press('End');await geometry(end,243);
    end=t+120;points.push([end,24,243,0,33]);await page.evaluate(()=>document.dispatchEvent(new Event('visibilitychange')));
    await page.waitForFunction(stamp=>Number(document.querySelector('[data-chart="downloads"] input').value)===stamp,end);await geometry(end,243);
    report.axis={start,end,irregular_sample:true,historical_selection_retained:true,latest_followed:true,plateau_flat:true};
    for(const width of [1440,390,320]){await page.setViewportSize({width,height:1000});await card.scrollIntoViewIfNeeded();await ready();assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await card.screenshot({path:path.join(out,'stats-time-axis-'+width+'.png'),animations:'disabled'});}
    points=[[end,24,243,0,33]];await page.evaluate(()=>document.dispatchEvent(new Event('visibilitychange')));
    await page.waitForFunction(()=>document.querySelector('[data-chart="downloads"] input').disabled);
    assert.equal(Number(await slider.inputValue()),end);assert.equal(Number(await dot.getAttribute('cx')),38);
    points=[];await page.evaluate(()=>document.dispatchEvent(new Event('visibilitychange')));await page.waitForFunction(()=>document.querySelector('[data-chart="downloads"] output').textContent==='暂无记录');assert.equal(await dot.count(),0);
    assert.deepEqual(report.errors,[]);assert.deepEqual(report.requests,[]);report.checks.push('Irregular time samples share slider/curve/selected-dot timestamps; keyboard jumps to adjacent real records; 243 plateau stays flat; refresh retains history or follows latest; three widths; single/empty history');report.success=true;return;
  }
  if(process.env.WEBSITE_STATS_ONLY==='1'){
    await page.goto(url);await ready();await page.emulateMedia({reducedMotion:'reduce'});
    await page.waitForFunction(()=>document.querySelector('[data-chart="views"] .chart-line'));
    assert.deepEqual(await page.locator('.stats-chart h3').allTextContents(),['页面浏览','下载发起','网页版使用','GitHub Star']);
    assert.equal(await page.locator('.stats-chart .chart-line').count(),4+(await page.locator('[data-chart="downloads"] .chart-line[data-platform="macos"]').count()));
    report.statsPalette=await page.locator('.stats-chart').evaluateAll(es=>es.map(e=>({key:e.dataset.chart,background:getComputedStyle(e).backgroundColor,line:getComputedStyle(e.querySelector('.chart-line')).stroke,marker:getComputedStyle(e.querySelector('.chart-dot')).fill,slider:getComputedStyle(e.querySelector('input')).accentColor})));assert.equal(new Set(report.statsPalette.map(p=>p.line)).size,4);assert.equal(new Set(report.statsPalette.map(p=>p.background)).size,4);assert(report.statsPalette.every(p=>p.line===p.marker&&p.line===p.slider));
    const timeline=page.locator('[data-chart="downloads"] input'),timeStart=Number(await timeline.getAttribute('min')),timeEnd=Number(await timeline.getAttribute('max'));
    assert(timeStart>1e9&&timeEnd>=timeStart,'Real history slider uses Unix seconds');
    await timeline.fill(String(Math.max(timeStart,timeEnd-3600)));const selectedTime=Number(await timeline.inputValue()),markerX=Number(await page.locator('[data-chart="downloads"] .chart-dot[data-platform="windows"]').getAttribute('cx'));
    assert(Math.abs(markerX-(38+(selectedTime-timeStart)/Math.max(1,timeEnd-timeStart)*426))<1e-6);
    report.statsTimeMapping={start:timeStart,end:timeEnd,selected:selectedTime,markerX,output:await page.locator('[data-chart="downloads"] output').innerText()};await timeline.focus();await page.keyboard.press('End');
    const webEntry=page.getByRole('button',{name:'网页版，灰测中，即将开放',exact:true});
    assert(await webEntry.isDisabled());assert.equal(await webEntry.getAttribute('href'),null);
    for(const width of [1920,1440,1280,1024,800,390,320]){
      await page.setViewportSize({width,height:1000});await page.evaluate(()=>scrollTo(0,0));await ready();
      assert(await webEntry.isVisible());assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
      const entryBounds=await webEntry.boundingBox(),downloadsBounds=await page.locator('.download-actions').boundingBox();
      assert(Math.abs(entryBounds.width-downloadsBounds.width)<1);assert(entryBounds.y<downloadsBounds.y);
      if(width>=1024){
        const rows=await page.locator('.site-header .brand,.site-header nav>a').evaluateAll(es=>es.map(e=>{const r=e.getBoundingClientRect();return r.top+r.height/2;}));
        assert(Math.max(...rows)-Math.min(...rows)<8,'Desktop navigation stays in one row');
      }
      await page.screenshot({path:path.join(out,'web-entry-'+width+'.png'),animations:'disabled'});
      await page.setViewportSize({width,height:1000});await page.locator('.site-stats').scrollIntoViewIfNeeded();await ready();
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
      await page.locator('.site-stats').screenshot({path:path.join(out,'statistics-'+width+'.png'),animations:'disabled'});
    }
    await page.setViewportSize({width:1440,height:1000});
    for(const period of ['1','7','30','all'])await page.locator('[data-stats-range]').selectOption(period);
    await page.locator('[data-chart="views"] input').fill(await page.locator('[data-chart="views"] input').getAttribute('min'));
    assert((await page.locator('[data-chart="views"] output').innerText()).includes('7 次'));
    const t=Math.floor(Date.now()/1000),totals={schema:1,startedAt:new Date((t-7200)*1000).toISOString(),updatedAt:new Date(t*1000).toISOString(),pageViews:1235,downloads:{total:227,portable:98,installer:129},webUses:2,github:{count:34,updatedAt:new Date(t*1000).toISOString()}};
    await page.route('**/site-stats.json',r=>r.fulfill({json:totals}));
    await page.route('**/site-stats-history.json',r=>r.fulfill({json:{schema:1,points:[[t-3600,1234,226,1,35],[t,1235,227,2,34]]}}));
    await page.evaluate(()=>document.dispatchEvent(new Event('visibilitychange')));
    await page.waitForFunction(()=>document.querySelector('[data-stat="views"]').textContent==='1,235');
    await page.waitForFunction(()=>document.querySelector('[data-chart="stars"] output').textContent.includes('34 个'));
    await page.locator('[data-chart="stars"] input').fill(await page.locator('[data-chart="stars"] input').getAttribute('min'));assert((await page.locator('[data-chart="stars"] output').innerText()).includes('35 个'));
    await page.unroute('**/site-stats.json');await page.route('**/site-stats.json',r=>r.fulfill({status:200,body:'{}',contentType:'application/json'}));
    await page.evaluate(()=>document.dispatchEvent(new Event('visibilitychange')));
    await page.waitForFunction(()=>document.querySelector('[data-stats-updated]').textContent.includes('暂时无法更新'));
    assert.equal(await page.locator('[data-stat="views"]').innerText(),'1,235');
    await page.reload();await page.waitForFunction(()=>document.querySelector('[data-stats-updated]').textContent.includes('暂时无法读取'));
    assert.equal(await page.locator('[data-stat="views"]').innerText(),'—');
    const nojs=await browser.browser().newContext({javaScriptEnabled:false});const np=await nojs.newPage();await np.goto(url);assert(await np.locator('.stats-caption noscript').isVisible());await nojs.close();
    report.checks.push('Four actual-history charts; seven widths 1920 through 320; prominent disabled web entry aligned above downloads; desktop header single row; four periods; keyboard slider; decreasing stars; refresh and retained values on failure; unavailable initial values; no-JavaScript explanation');
    assert.deepEqual(report.errors,[]);report.success=true;return;
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
    const webEntry=page.locator('.web-entry'),entryMotion=[];
    assert(await webEntry.isEnabled());assert.equal(await webEntry.getAttribute('href'),'/app/');
    for(let i=0;i<10;i++){entryMotion.push(await webEntry.evaluate(e=>new DOMMatrixReadOnly(getComputedStyle(e).transform).m42));await page.waitForTimeout(250);}
    report.webEntryMotionRange=Math.max(...entryMotion)-Math.min(...entryMotion);
    assert(report.webEntryMotionRange>.5&&report.webEntryMotionRange<=4.1,'Web entrance gently floats within four pixels');
    const tileBox=await tile.boundingBox();await page.mouse.move(tileBox.x+tileBox.width/2,tileBox.y+tileBox.height/2);await page.waitForTimeout(650);assert(await tile.evaluate(e=>e.matches(':hover')));assert(parseFloat(await tile.evaluate(e=>getComputedStyle(e).scale))>1.03);assert(parseFloat(await tile.evaluate(e=>getComputedStyle(e).rotate))<-4);
    await page.screenshot({path:path.join(out,'hero-hover.png'),animations:'allow'});
    await page.mouse.move(0,0);await page.waitForTimeout(700);assert(['1','none'].includes(await tile.evaluate(e=>getComputedStyle(e).scale)));
    await page.locator('#methods').scrollIntoViewIfNeeded();await page.waitForTimeout(1000);assert.equal(await floater.evaluate(e=>getComputedStyle(e).animationPlayState),'paused');
    assert.equal(await webEntry.evaluate(e=>getComputedStyle(e).animationPlayState),'paused');
    const card=page.locator('.reference-card').first();await card.hover();await page.waitForTimeout(500);assert(parseFloat(await card.evaluate(e=>getComputedStyle(e).scale))>1);
    const rows=page.locator('.faq-list details');assert(await rows.evaluateAll(es=>es.every(e=>e.classList.contains('reveal'))));
    for(const width of [1440,800,390,320]){
      await page.setViewportSize({width,height:1000});await page.goto(url);await ready();await page.locator('.hero-art').scrollIntoViewIfNeeded();await page.waitForTimeout(1100);
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await page.screenshot({path:path.join(out,'motion-'+width+'.png'),animations:'allow'});
    }
    await page.emulateMedia({reducedMotion:'reduce'});await ready();assert.equal(await floater.evaluate(e=>getComputedStyle(e).animationName),'none');assert.equal(await tile.evaluate(e=>getComputedStyle(e).scale),'none');
    assert.equal(await webEntry.evaluate(e=>getComputedStyle(e).animationName),'none');
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
    const version=JSON.parse(await fs.readFile(path.join(site,'assets/downloads.json'),'utf8')).version;
    const nojs=await browser.browser().newContext({javaScriptEnabled:false,viewport:{width:390,height:844}});const np=await nojs.newPage();await np.goto(url);assert.equal(await np.locator('#release-panel-0').evaluate(e=>e.hidden),false);assert((await np.locator('#release-panel-0 .release-content').innerText()).includes('PhoneticToolbox '+version+' 更新'));await nojs.close();
    report.checks.push('Without JavaScript the latest '+version+' notes remain visible');assert.equal(report.errors.length,0);assert.equal(report.requests.length,0);report.success=true;return;
  }
  if(process.env.WEBSITE_METHODS_ONLY==='1'){
    await page.emulateMedia({reducedMotion:'reduce'});
    const sources=JSON.parse(await fs.readFile(path.join(root,'frontend/src/generated/sources.json'),'utf8'));
    const catalogs=page.locator('.reference-catalog'),cards=page.locator('#methods .reference-card');
    assert.equal(await catalogs.count(),2);assert.equal(await cards.count(),4);assert(await cards.evaluateAll(es=>es.every(e=>!e.closest('details'))));
    assert(await catalogs.evaluateAll(es=>es.every(e=>!e.open)));assert.equal(await page.locator('.methods-note').count(),0);
    assert(!(await page.locator('#methods').textContent()).includes('写进论文时，记得'));
    assert(await page.locator('[id]').evaluateAll(es=>new Set(es.map(e=>e.id)).size===es.length));
    await catalogs.first().locator(':scope > summary').focus();await page.keyboard.press('Enter');
    assert(await catalogs.first().evaluate(e=>e.open));assert.equal(await catalogs.last().evaluate(e=>e.open),false);
    await catalogs.last().locator(':scope > summary').focus();await page.keyboard.press('Enter');
    assert(await catalogs.evaluateAll(es=>es.every(e=>e.open)));
    await browser.grantPermissions(['clipboard-read','clipboard-write'],{origin:new URL(url).origin});
    for(const [group,title] of [['academic','学术引用'],['software','软件与代码库引用']]){
      const catalog=page.locator('[data-reference-group="'+group+'"]'),expected=sources.filter(s=>s.acknowledgement_group===group);
      assert((await catalog.locator(':scope > summary').innerText()).includes(title+' · '+expected.length+' 条'));
      const rendered=await catalog.locator('.reference-source').evaluateAll(es=>es.map(e=>({id:e.dataset.sourceId,title:e.querySelector('h4').textContent,citation:e.querySelector('.reference-citation').textContent,links:[...e.querySelectorAll('a')].map(a=>a.getAttribute('href')),text:e.textContent,modules:e.dataset.modules.split(' ')})));
      assert.equal(rendered.length,expected.length);assert.equal(new Set(rendered.map(r=>r.id)).size,expected.length);
      for(const source of expected){const row=rendered.find(r=>r.id===source.id);assert(row);assert.equal(row.title,source.title);assert.equal(row.citation,source.citation||[source.authors,source.title].filter(Boolean).join(' · '));assert.deepEqual(row.links,Object.values(source.urls));assert.deepEqual(row.modules,source.modules.length?source.modules:['']);for(const key of ['reference_note','license','version','attribution','adaptation_note','adaptation_caution'])if(source[key])assert(row.text.includes(source[key]),source.id+': '+key);}
      const query=catalog.locator('.reference-query'),module=catalog.locator('.reference-module');
      await query.fill(group==='academic'?'Aoki':'NAFNet');
      assert(await catalog.locator('.reference-source:not([hidden])').count()>0);
      assert(await catalog.locator('.reference-source:not([hidden])').evaluateAll((es,term)=>es.every(e=>e.textContent.toLowerCase().includes(term)),group==='academic'?'aoki':'nafnet'));
      const other=page.locator('.reference-catalog:not([data-reference-group="'+group+'"])');assert.equal(await other.locator('.reference-source:not([hidden])').count(),sources.length-expected.length);
      await query.fill('');await module.selectOption('M09');assert.equal(await catalog.locator('.reference-source:not([hidden])').count(),expected.filter(s=>s.modules.includes('M09')).length);
      await query.fill('this-source-does-not-exist');assert.equal(await catalog.locator('.reference-source:not([hidden])').count(),0);assert.match(await catalog.locator('.reference-count').innerText(),/没有匹配/);
      assert.equal(await other.locator('.reference-source:not([hidden])').count(),sources.length-expected.length);
      await query.fill('');await module.selectOption('');assert.equal(await catalog.locator('.reference-source:not([hidden])').count(),expected.length);
      await catalog.locator('.reference-copy').first().click();assert.equal(await page.evaluate(()=>navigator.clipboard.readText()),rendered[0].citation);
    }
    for(const width of [1600,800,390,320]){
      await page.setViewportSize({width,height:1000});await ready();
      for(const group of ['academic','software']){
        const catalog=page.locator('[data-reference-group="'+group+'"]');await catalog.scrollIntoViewIfNeeded();
        assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));assert(await catalog.locator('.reference-results').evaluate(e=>e.scrollWidth<=e.clientWidth+1));
        await catalog.screenshot({path:path.join(out,'references-'+group+'-'+width+'.png'),animations:'disabled'});
      }
      await catalogs.evaluateAll(es=>es.forEach(e=>e.open=false));await ready();
      await page.locator('#methods').screenshot({path:path.join(out,'methods-'+width+'.png'),animations:'disabled'});
      await catalogs.evaluateAll(es=>es.forEach(e=>e.open=true));
    }
    const nojs=await browser.browser().newContext({javaScriptEnabled:false,viewport:{width:390,height:844}});const np=await nojs.newPage();await np.goto(url);
    assert(await np.locator('.reference-catalog').evaluateAll(es=>es.every(e=>!e.open)));
    for(const group of ['academic','software']){const catalog=np.locator('[data-reference-group="'+group+'"]');await catalog.locator(':scope > summary').click();assert(await catalog.locator('.reference-source').first().isVisible());assert.equal(await catalog.locator('.reference-source').count(),sources.filter(s=>s.acknowledgement_group===group).length);assert.equal(await catalog.locator('.reference-filters').isVisible(),false);}
    await nojs.close();
    report.checks.push('Four method cards stay visible; academic and software references are separate default-collapsed lists; every source, citation, URL, module and attribution matches the desktop group; independent search/module/empty state and clipboard, keyboard toggle, four widths and no-JS pass');
    assert.deepEqual(report.errors,[]);assert.deepEqual(report.requests,[]);report.success=true;return;
  }
  if(process.env.WEBSITE_NAV_ONLY==='1'||process.env.WEBSITE_NEXT_ONLY==='1'||process.env.WEBSITE_TUTORIAL_ONLY==='1'){
    for(const width of [1920,1600,1440,1250,1100,800,390,320]){
      await page.setViewportSize({width,height:1000});await ready();await page.evaluate(()=>scrollTo(0,0));
      const links=page.locator('.site-header nav>a');assert.equal(await links.count(),9);
      const layout=await links.evaluateAll(es=>es.map(e=>{const r=e.getBoundingClientRect();const textRects=[...e.childNodes].filter(n=>n.nodeType===Node.TEXT_NODE&&n.textContent.trim()).flatMap(n=>{const range=document.createRange();range.selectNodeContents(n);return [...range.getClientRects()].map(q=>({x:q.x,right:q.right}))});return {textRects,text:e.childNodes[0]?.textContent,font:parseFloat(getComputedStyle(e).fontSize),x:r.x,right:r.right,width:r.width,scrollWidth:e.scrollWidth,clientWidth:e.clientWidth}}));
      const navigationFont=width>=1024?Math.min(16,Math.max(13,width*.0108)):width>800?18:width>360?16:15;
      assert(layout.every(r=>r.x>=0&&r.right<=width+1&&r.textRects.every(t=>t.x>=r.x-1&&t.right<=r.right+1)&&Math.abs(r.font-navigationFont)<.05),JSON.stringify({width,navigationFont,layout}));
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
      assert.deepEqual(await page.locator('.tutorial-channel:not(#tutorial-intro)').allTextContents(),expectedModules);
      assert.deepEqual(await page.locator('.tutorial-channel:disabled').allTextContents(),expectedModules.filter(name=>name!=='语谱图转音频'));
      assert.equal(await page.locator('.tutorial-channel:enabled').count(),2);
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
        assert.equal(await page.locator('#tutorial-episodes').isVisible(),false);
        await page.locator('#tutorials').screenshot({path:path.join(out,'tutorial-'+width+'.png'),animations:'disabled'});
      }
      await page.locator('.reading-toggle').click();await page.locator('.reading-menu').screenshot({path:path.join(out,'reading-menu.png'),animations:'disabled'});await page.keyboard.press('Escape');
      await page.setViewportSize({width:1280,height:1000});
      const link=page.locator('.tutorial-actions a');assert.equal(await link.getAttribute('href'),'https://www.bilibili.com/video/BV1WWpr67EkX/');assert.equal(await link.getAttribute('target'),'_blank');
      await page.locator('#tutorial-intro').focus();await page.keyboard.press('Enter');
      const frame=page.locator('#tutorial-screen iframe');assert.equal(await frame.count(),1);assert((await frame.getAttribute('src')).includes('bvid=BV1WWpr67EkX'));assert(await frame.evaluate(e=>e.allowFullscreen));
      await page.locator('#tutorial-intro').click();assert.equal(await frame.count(),1);
      await page.locator('#tutorial-spectrogram').focus();await page.keyboard.press('Enter');
      assert.equal(await frame.count(),1);assert((await frame.getAttribute('src')).includes('bvid=BV1RuJm6gEWE'));
      assert.equal(await link.getAttribute('href'),'https://www.bilibili.com/video/BV1RuJm6gEWE/');assert.equal(await frame.getAttribute('title'),'语谱图转音频 · 第一集 · B 站视频');
      assert(await page.locator('#tutorial-episodes').isVisible());assert.equal(await page.locator('#tutorial-spectrogram').getAttribute('aria-pressed'),'true');assert.equal(await page.locator('#tutorial-intro').getAttribute('aria-pressed'),'false');
      assert.equal(await page.locator('#tutorial-episodes button:disabled').innerText(),'第二集 · 待更新');
      await frame.evaluate(e=>e.dataset.instance='retained');await page.locator('[data-tutorial-episode="spectrogram"]').click();assert.equal(await frame.getAttribute('data-instance'),'retained');
      await page.setViewportSize({width:320,height:1000});await ready();assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await page.locator('#tutorials').screenshot({path:path.join(out,'tutorial-episode-320.png'),animations:'disabled'});await page.setViewportSize({width:1280,height:1000});
      await page.locator('#tutorial-fullscreen').click();await page.waitForFunction(()=>document.fullscreenElement?.id==='tutorial-screen');await page.evaluate(()=>document.exitFullscreen());
      report.playerProbe={};
      try{await page.waitForFunction(()=>{const frame=document.querySelector('#tutorial-screen iframe');return !!frame;});const child=await frame.elementHandle();const embedded=await child.contentFrame();await embedded.waitForSelector('video',{timeout:20000});
        report.playerProbe.video=await embedded.locator('video').evaluate(e=>({readyState:e.readyState,duration:e.duration,paused:e.paused,error:e.error?.message}));
        await embedded.locator('video').evaluate(e=>e.play());await page.waitForTimeout(2000);
        report.playerProbe.playing=await embedded.locator('video').evaluate(e=>({currentTime:e.currentTime,paused:e.paused,readyState:e.readyState}));
      }catch(error){report.playerProbe.limitation=String(error).slice(0,450);}
      await page.locator('#tutorials').screenshot({path:path.join(out,'tutorial-player.png'),animations:'disabled'});
      await page.locator('#tutorial-intro').click();assert.equal(await frame.count(),1);assert((await frame.getAttribute('src')).includes('bvid=BV1WWpr67EkX'));assert.equal(await page.locator('#tutorial-episodes').isVisible(),false);assert.equal(await link.getAttribute('href'),'https://www.bilibili.com/video/BV1WWpr67EkX/');
      const nojs=await browser.browser().newContext({javaScriptEnabled:false,viewport:{width:390,height:844}});const np=await nojs.newPage();await np.goto(url);
      assert(await np.locator('#tutorials').isVisible());assert.equal(await np.locator('.tutorial-channel:disabled').count(),17);assert(await np.locator('.tutorial-actions a').isVisible());assert(await np.locator('noscript a[href="https://www.bilibili.com/video/BV1RuJm6gEWE/"]').isVisible());await nojs.close();
      assert.deepEqual(report.errors,[]);assert(report.requests.every(r=>!r.url.startsWith(url)),'No failed local assets');
      report.checks.push('Tutorial: all 18 module names match README, 17 unavailable modules, yellow intro and available spectrogram, 5 layouts, keyboard activation, first episode and disabled second episode, switching both directions replaces one lazy player and external link, repeated selection retains player, actual fullscreen, no-JS video links; external media evidence recorded separately');
      report.success=true;return;
    }
    if(process.env.WEBSITE_NEXT_ONLY==='1'){
      const expected=['#tutorials','#lab','#methods','#faq','#updates','#next','#feedback','#contact','https://github.com/Nephelium/PhoneticToolbox'];
      assert.deepEqual(await page.locator('.site-header nav>a').evaluateAll(es=>es.map(e=>e.getAttribute('href'))),expected);
      assert.equal(await page.locator('.site-header a[href="#next"]').evaluate(e=>e.childNodes[0].textContent),'下一步…');
      assert(await page.locator('#next').evaluate(e=>e.previousElementSibling.previousElementSibling.id==='updates'&&e.nextElementSibling.nextElementSibling.id==='feedback'));
      assert(await page.locator('#methods').evaluate(e=>e.previousElementSibling.previousElementSibling.id==='lab'));
      assert.deepEqual(await page.locator('#next .upcoming-model').allTextContents(),['窗宽与动态范围预测模型','Griffin–Lim · 动量 0.99','条件化 NAFNet','BigVGAN']);
      const copy=await page.locator('#next').innerText();for(const text of ['青木（Naofumi Aoki）','沿频率方向求和','峰值脉冲','相位初始化 Griffin–Lim','短时窗','模型训练进行中','Mel 谱','原始 GL','动量加速 GL','0.99'])assert(copy.includes(text),text);
      assert.equal(await page.locator('.upcoming-reference').first().getAttribute('href'),'https://doi.org/10.1250/ast.44.186');
      assert.equal(await page.locator('.upcoming-reference').nth(1).getAttribute('href'),'https://perraudin.info/publications/perraudin-note-002.pdf');
      assert.equal(await page.locator('.web-release').count(),0);assert(!(await page.locator('body').innerText()).includes('2026 年 10 月 31 日'));
      assert.equal(await page.locator('#methods [data-source-id="REF-FAST-GRIFFINLIM"]').count(),1);
      assert.equal(await page.locator('#next img').count(),6);assert.equal(await page.locator('#next audio').count(),24);assert.equal(await page.locator('#next .mini-wave').count(),0);assert.equal(await page.locator('#next .sound-play').count(),24);
      assert(await page.locator('[id]').evaluateAll(es=>new Set(es.map(e=>e.id)).size===es.length),'Unique DOM ids across repeated comparison audio');
      const comparison=JSON.parse(await fs.readFile(path.join(site,'assets/spectrogram-examples/comparison.json'),'utf8'));
      const metadata=JSON.parse(await fs.readFile(path.join(site,'assets/audio/provenance.json'),'utf8'));
      assert.deepEqual(comparison.pairs.map(p=>p.id),['smith','levantan','observo','lu2025-modal','lu2025-pressed','uploaded']);
      assert.equal(await page.locator('.next-example[data-case="uploaded"] h4').innerText(),'知乎例图 · 2 秒 / 6 kHz');assert((await page.locator('#next-lu-reference').innerText()).includes('Contribution of F0 and phonation to tone perception in the Zaiwa language.'));assert.equal(await page.locator('#next-lu-reference a').getAttribute('href'),'https://doi.org/10.1016/j.wocn.2025.101413');
      for(const pair of comparison.pairs){
        assert.equal(pair.before_defaults.phase_initialization,'random');assert.equal(pair.before_defaults.window_ms,10);assert.equal(pair.before_defaults.dynamic_range_db,60);assert.equal(pair.before_defaults.window,'Hann');
        assert.equal(pair.before_defaults.momentum,0);assert.equal(pair.before_defaults.iterations,32);assert.equal(pair.before_defaults.pulse_initialization,false);
        const settings=page.locator('.next-comparison-settings');await settings.locator('summary').click();assert((await settings.innerText()).includes('固定种子的随机相位'));assert((await settings.innerText()).includes('这组试听不使用青木脉冲初始化'));await settings.locator('summary').click();const caseCard=page.locator('.next-example[data-case="'+pair.id+'"]');assert((await caseCard.locator('.sound-detail').first().innerText()).includes('动量 0'));
        const image=page.locator('.next-example[data-case="'+pair.id+'"] img');
        const imageResponse=await page.request.get(await image.evaluate(e=>e.src));assert.equal(imageResponse.status(),200);assert.equal(createHash('sha256').update(await imageResponse.body()).digest('hex'),pair.display.image_sha256);
        if(pair.id.startsWith('lu2025-'))assert.equal(await caseCard.locator('figcaption a').getAttribute('href'),'https://doi.org/10.1016/j.wocn.2025.101413');
        assert.equal(pair.accelerated_settings.momentum,.99);assert(pair.parameter_prediction.applied_to_reconstruction);
        const text=await caseCard.locator('.next-parameters').innerText();
        assert(text.includes('10 ms')&&text.includes('60 dB')&&text.includes('32'));
        const fastRow=metadata.files.find(r=>r.file===pair.accelerated),neuralRow=metadata.files.find(r=>r.file===pair.after);
        assert.equal(neuralRow.parameters.vocoder_input_sha256,pair.predicted_sha256);assert.equal(neuralRow.parameters.vocoder_input_file,pair.predicted);assert.deepEqual(neuralRow.parameters.gl,pair.predicted_settings);assert.equal(pair.predicted_settings.iterations,32);assert.equal(pair.predicted_settings.momentum,.99);assert.equal(pair.predicted_settings.window_ms,Math.floor(pair.parameter_prediction.window_ms+.5));assert.equal(pair.predicted_settings.dynamic_range_db,pair.parameter_prediction.dynamic_range_db);assert.equal(pair.parameter_prediction.checkpoint_sha256,'7ac61f075259099e0c58e743818b2b61a1ec42d5a125eab6ef9fc9216182a86e');assert.equal(await caseCard.locator('.sound-play').count(),4);
        assert.equal(neuralRow.parameters.repair_model_step,6000);
        assert.equal(fastRow.parameters.pulse_initialization,false);assert.equal(fastRow.parameters.parameter_prediction,false);
        for(const kind of ['before','accelerated','predicted','after']){
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
      const first=page.locator('#audio-next-spectrum-before-32i-smith'),second=page.locator('#audio-next-spectrum-fast-32i-smith');
      await page.locator('[data-audio="audio-next-spectrum-before-32i-smith"]').click();await page.locator('[data-audio="audio-next-spectrum-fast-32i-smith"]').click();assert(await first.evaluate(e=>e.paused));assert(!(await second.evaluate(e=>e.paused)));
      await page.locator('[data-audio="audio-vowel-a"]').click();await page.waitForFunction(()=>document.querySelector('#audio-next-spectrum-fast-32i-smith').paused&&document.querySelector('#next .audio-status').textContent.startsWith('已暂停'));
      assert(!(await page.locator('#audio-vowel-a').evaluate(e=>e.paused)));await page.locator('#audio-vowel-a').evaluate(e=>e.pause());
      await page.emulateMedia({reducedMotion:'reduce'});
      for(const width of [1600,1280,800,390,320]){
        report.nextPhase={step:'keyboard-navigation',width};
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
      report.nextPhase={step:'no-javascript',width:390};
      const nojs=await browser.browser().newContext({javaScriptEnabled:false,reducedMotion:'reduce',viewport:{width:390,height:844}});const np=await nojs.newPage();const pendingNojs=new Set();np.on('request',r=>pendingNojs.add(r.url()));for(const event of ['requestfinished','requestfailed'])np.on(event,r=>pendingNojs.delete(r.url()));try{await np.goto(url+'#next');await np.evaluate(()=>document.fonts.ready);}catch(error){report.nojsPending=[...pendingNojs];throw error;}
      assert(await np.locator('#next .upcoming-paper').isVisible());assert.equal(await np.locator('#next .reveal').evaluate(e=>getComputedStyle(e).opacity),'1');assert(await np.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
      assert.equal(await np.locator('#next audio[controls]').count(),24);
      await np.locator('#next').screenshot({path:path.join(out,'next-nojs-390.png')});await nojs.close();
      report.checks.push('Next step: revised navigation and DOM order, current responsive navigation font, paper attribution, six figures/twenty-four audios with predicted parameter provenance fetched with SHA256 and played to completion; four buttons per case, no waveforms, exclusive playback/local status, unique ids, keyboard/reading menu, five widths, actual cute font, reduced motion/no-JS');
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
  assert.equal(await page.locator('.web-release').count(),0);assert.equal(await page.getByText('用于语音学研究与教学',{exact:true}).count(),0);
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
  assert.match(await page.locator('[data-stats-since]').innerText(),/从 .+ 开始记录/);
  const statUrl=new URL('/site-stats.json',url).href;
  await page.route(statUrl,r=>r.fulfill({status:200,contentType:'application/json',body:'{"schema":1,"pageViews":-5}'}));
  await page.reload();await ready();await page.waitForFunction(()=>document.querySelector('[data-stats-updated]').textContent.includes('暂时无法读取'));
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
  assert.match(await page.locator('[data-stats-since]').innerText(),/从 .+ 开始记录/);
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
 }catch(e){report.failure=String(e);if(page){report.failureState=await page.evaluate(()=>{const section=document.querySelector('#next');return {url:location.href,width:innerWidth,scrollY,nextTop:section?.getBoundingClientRect().top,scrollPadding:getComputedStyle(document.documentElement).scrollPaddingTop,scrollMargin:section&&getComputedStyle(section).scrollMarginTop,activeElement:document.activeElement?.outerHTML.slice(0,400),fonts:document.fonts.status}}).catch(()=>null);await page.screenshot({path:path.join(out,'failure.png')}).catch(()=>{});}throw e;}
 finally{
  await browser?.close();if(server)await new Promise(resolve=>server.close(resolve));await write('report.json',report);
  const files=[];async function inventory(dir){for(const entry of await fs.readdir(dir,{withFileTypes:true})){assert(!entry.isSymbolicLink());const file=path.join(dir,entry.name);if(entry.isDirectory())await inventory(file);else{const bytes=await fs.readFile(file);files.push({path:path.relative(scratch,file),bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex')});}}}await inventory(scratch);
  await write('scratch-manifest.json',{schema:'ptb-test-scratch-files/1',files});await write('run.json',{...marker,state:'ready',test_outcome:report.success?'completed':'failed'});
  const cleanup=spawnSync('powershell.exe',['-NoProfile','-ExecutionPolicy','Bypass','-File',path.join(root,'scripts/cleanup_test_scratch.ps1'),'-RunDirectory',out],{windowsHide:true,encoding:'utf8'});assert.equal(cleanup.status,0,cleanup.stdout+cleanup.stderr);console.log(JSON.stringify({out,...report}));
 }
}
main().catch(e=>{console.error(e);process.exitCode=1});
