const fs=require('node:fs/promises');
const path=require('node:path');
const assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..'),[out,scratch]=process.argv.slice(2);
(async()=>{
  assert(out&&scratch,'Use the owned verification runner');
  const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
  const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  const server=await createServer({root:path.join(root,'frontend'),cacheDir:path.join(scratch,'vite-cache'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{noDiscovery:true,include:['vue']}});
  const report={success:false,cases:[],errors:[],externalRequests:[],png:null};let browser;
  try{
    await server.listen();const url=server.resolvedUrls.local[0];
    browser=await chromium.launchPersistentContext(path.join(scratch,'chrome-profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1440,height:900},acceptDownloads:true});
    const page=await browser.newPage();page.on('pageerror',e=>report.errors.push(e.message));page.on('request',r=>{if(/^https?:/.test(r.url())&&!r.url().startsWith(url))report.externalRequests.push(r.url());});
    await page.goto(url);await page.locator('.home-page').waitFor();
    const open=()=>page.locator('.nav-item').and(page.getByRole('button',{name:'汉字转国际音标',exact:true})).click();
    const input=page.getByRole('textbox',{name:'待转换汉字文本'}),standard=page.getByRole('combobox',{name:'转换标准'});
    const groups=page.locator('.m13-output [data-erhua=true]');
    const rawValues=()=>page.locator('.m13-output .m13-mapped').evaluateAll(nodes=>nodes.map(n=>n.dataset.value));
    if(process.argv.includes('--manual-only')){
      await open();await page.getByRole('button',{name:'帮助',exact:true}).click();
      const heading=page.getByRole('heading',{name:/组合与拆分儿化韵/});await heading.waitFor();await heading.scrollIntoViewIfNeeded();
      const picture=page.locator('img[src*="m13-erhua-current-20261011"]');await picture.scrollIntoViewIfNeeded();
      await page.waitForFunction(()=>{const img=document.querySelector('img[src*="m13-erhua-current-20261011"]');return img?.complete&&img.naturalWidth===2560;});
      const manualText=await page.locator('.manual-document').innerText();assert(!manualText.includes('声调显示和拼音 ü'));assert(!manualText.includes('女略'));
      await page.evaluate(()=>document.fonts.ready);const font=await page.locator('.manual-ipa').first().evaluate(el=>({family:getComputedStyle(el).fontFamily,loaded:[...document.fonts].some(face=>face.family==='PTB-Doulos'&&face.status==='loaded')}));assert(font.family.includes('PTB-Doulos')&&font.loaded,JSON.stringify(font));
      const chapter=JSON.parse(await fs.readFile(path.join(root,'frontend/public/manual/chapters/m13.json'),'utf8')),project=JSON.parse(await fs.readFile(path.join(root,'frontend/public/manual/project.json'),'utf8'));
      const pictures=chapter.body.content.filter(n=>n.type==='image');assert.equal(pictures.length,18);
      for(const picture of pictures){const asset=project.assets.find(a=>a.id===picture.attrs.assetId);const img=page.locator('img[src*="'+asset.path.split('/').at(-1)+'"]');await img.scrollIntoViewIfNeeded();await img.evaluate(el=>el.decode());assert.equal(await img.evaluate(el=>el.naturalWidth),asset.width);if(asset.id!=='m13-export-kaiti-independent-colors'){assert.equal(asset.capture.dpr,1.5);assert.equal(asset.capture.pageZoom,1);assert.equal(asset.height,1600);}}
      await picture.scrollIntoViewIfNeeded();
      await page.screenshot({path:path.join(out,'manual-erhua-reader.png')});
      report.manual={headingVisible:true,currentScreenshotLoaded:true,imagesLoaded:pictures.length,dpr:1.5,pageZoom:1,ipaFont:font,removedPinyinSection:true,examplesUpdated:true};assert.deepEqual(report.errors,[]);assert.deepEqual(report.externalRequests,[]);report.success=true;
      console.log(JSON.stringify(report,null,2));return;
    }
    for(const theme of ['light','dark']){
      await page.locator('.utility-grid').getByRole('button',{name:'设置',exact:true}).click();
      await page.getByRole('button',{name:theme==='light'?'浅色':'深色',exact:true}).click();await open();
      for(const viewport of [{width:1440,height:900},{width:900,height:700}]){
        await page.setViewportSize(viewport);await input.fill('花儿女儿门儿\n儿子 花 儿');await standard.selectOption('UntPhesoca宽');
        await page.getByRole('button',{name:'全部拆分',exact:true}).evaluate(b=>{if(!b.disabled)b.click();});
        assert.equal(await groups.count(),0);
        await page.getByRole('button',{name:'自动组合儿化',exact:true}).click();assert.equal(await groups.count(),3);
        assert.equal(await groups.first().getAttribute('data-value'),'xwɐɚ˥');
        await groups.nth(1).click();await page.getByRole('button',{name:'拆分儿化',exact:true}).click();assert.equal(await groups.count(),2);
        await page.locator('.m13-output [data-index="2"]').click();await page.getByRole('button',{name:'组合儿化',exact:true}).click();assert.equal(await groups.count(),3);
        await groups.nth(1).click();await page.getByRole('button',{name:'拆分儿化',exact:true}).click();
        await standard.selectOption('UntPhesoca严');assert.equal(await groups.first().getAttribute('data-value'),'x̞ʷwɐɻ˕˥');
        await page.getByRole('button',{name:'显示声调',exact:true}).click();assert.equal(await groups.first().getAttribute('data-value'),'x̞ʷwɐɻ˕');
        await page.getByRole('button',{name:'显示声调',exact:true}).click();
        await input.fill('😀花儿女儿门儿\n儿子 花 儿');assert.equal(await groups.count(),2);assert.equal(await groups.first().getAttribute('data-index'),'1');
        await standard.selectOption('Standard Chinese (Beijing)');assert.equal(await groups.count(),0);
        assert((await rawValues()).includes('ɚ˧˥'));
        await input.fill('😀花儿女儿门儿\n儿子 花 儿行');
        const otherStandards=await standard.locator('option').evaluateAll(nodes=>nodes.map(n=>n.value).filter(v=>!v.startsWith('UntPhesoca')));
        for(const other of otherStandards){
          await standard.selectOption(other);
          assert.equal(await groups.count(),0);
          assert.equal(await page.locator('.m13-erhua').count(),0);
          assert.equal(await page.getByRole('button',{name:'自动组合儿化',exact:true}).count(),0);
          assert.equal(await page.getByRole('button',{name:'全部拆分',exact:true}).count(),0);
          await page.locator('.m13-output .m13-ambiguous').last().click();
          await page.getByRole('dialog',{name:'选择 行 的读音',exact:true}).waitFor();
          assert.equal(await page.locator('.m13-erhua-action').count(),0);
          await page.getByRole('button',{name:'关闭读音选择',exact:true}).click();
        }
        await input.fill('😀花儿女儿门儿\n儿子 花 儿');
        await standard.selectOption('UntPhesoca严');assert.equal(await groups.count(),2);
        const geometry=await page.locator('.m13-erhua').evaluate(el=>({overflow:el.scrollWidth-el.clientWidth,width:el.clientWidth}));assert(geometry.overflow<=1,JSON.stringify(geometry));
        await page.locator('.mandarin-ipa-page').screenshot({path:path.join(out,`erhua-${theme}-${viewport.width}.png`)});
        report.cases.push({theme,viewport,combineSplit:true,manualCombine:true,editMovesPair:true,otherStandardUnchanged:true,otherStandardControlsHidden:otherStandards.length,geometry});
      }
    }
    await page.getByRole('button',{name:/^保存本机草稿/}).click();await page.reload();await open();assert.equal(await input.inputValue(),'');
    await page.getByRole('button',{name:'恢复本机草稿',exact:true}).click();assert.equal(await input.inputValue(),'😀花儿女儿门儿\n儿子 花 儿');assert.equal(await groups.count(),2);
    await page.setViewportSize({width:1440,height:900});
    await page.evaluate(()=>{window.qaPaint=[];const original=CanvasRenderingContext2D.prototype.fillText;CanvasRenderingContext2D.prototype.fillText=function(text,...args){window.qaPaint.push(text);return original.call(this,text,...args);};});
    await browser.setOffline(true);const downloadWait=page.waitForEvent('download');await page.getByRole('button',{name:'保存为 PNG',exact:true}).click();const download=await downloadWait;
    const pngPath=path.join(out,'erhua-export.png');await download.saveAs(pngPath);const bytes=await fs.readFile(pngPath);assert.equal(bytes.subarray(1,4).toString(),'PNG');
    const paint=await page.evaluate(()=>window.qaPaint);assert(paint.includes('花儿'));assert(paint.includes('x̞ʷwɐɻ˕˥'));assert(paint.includes('女'));assert(paint.includes('儿'));
    report.png={bytes:bytes.length,width:bytes.readUInt32BE(16),height:bytes.readUInt32BE(20),joinedAndSplitTextVerified:true};await browser.setOffline(false);
    await page.locator('.utility-grid').getByRole('button',{name:'设置',exact:true}).click();await page.getByRole('button',{name:'浅色',exact:true}).click();await open();await page.setViewportSize({width:2560,height:1600});
    await input.fill('春江花月夜\n银行行走女略');await standard.selectOption('UntPhesoca严');await page.screenshot({path:path.join(out,'manual-unt-current.png')});
    await input.fill('花儿女儿门儿\n儿子 花 儿');await page.getByRole('button',{name:'自动组合儿化',exact:true}).click();await groups.nth(1).click();await page.getByRole('button',{name:'拆分儿化',exact:true}).click();
    await page.screenshot({path:path.join(out,'manual-erhua-current.png')});
    await groups.first().click();await page.screenshot({path:path.join(out,'manual-erhua-split.png')});await page.getByRole('button',{name:'关闭读音选择',exact:true}).click();
    await page.getByRole('button',{name:'全部拆分',exact:true}).click();assert.equal(await groups.count(),0);
    await standard.selectOption('Standard Chinese (Beijing)');await input.fill('');await page.screenshot({path:path.join(out,'manual-overview-current.png')});
    assert.deepEqual(report.errors,[]);assert.deepEqual(report.externalRequests,[]);report.success=true;
  }finally{if(browser)await browser.close();await server.close();await fs.writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2)+'\n');}
  console.log(JSON.stringify(report,null,2));
})().catch(e=>{console.error(e);process.exitCode=1;});
