// Actual Chrome/Vite M17 R2 font preference, expanded matrix and scroll regression.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{pathToFileURL}=require('node:url');
(async()=>{
 const root=path.resolve(__dirname,'../..'),out=path.join(root,'output/playwright/m17-ui-r2',new Date().toISOString().replace(/[:.]/g,'-'));await fs.mkdir(out,{recursive:true});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js'))),server=await createServer({root:path.join(root,'frontend'),optimizeDeps:{entries:['index.html']},server:{host:'127.0.0.1',port:0},logLevel:'warn'});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright')),browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,ignoreDefaultArgs:['--hide-scrollbars']}),context=await browser.newContext(),page=await context.newPage(),errors=[],layouts=[],preferences=[];
 const catalog=JSON.parse(await fs.readFile(path.join(root,'frontend/src/modules/ipa-plus/data/catalog.json'),'utf8')),views={ipa:['base','marks'],extipa:['base','marks','context','combinations'],voqs:['base']};
 const ready=()=>page.locator('.m17-body[data-loaded=true][data-font-ready=true]').waitFor();
 const sizeInput=page.getByRole('spinbutton',{name:'音标字号',exact:true});
 const setSize=async n=>{await sizeInput.fill(String(n));await sizeInput.press('Tab');};
 page.on('pageerror',e=>errors.push(e.message));
 try{
  await page.goto(server.resolvedUrls.local[0]+'#M17');await ready();assert.equal(await sizeInput.inputValue(),'26');assert.equal(await page.locator('.ipa-plus-page input[type=number]').count(),1);assert.equal(await page.locator('.m17-editor-toolbar input').count(),0);await page.getByLabel('悬停介绍',{exact:true}).uncheck();
  for(const size of [{width:1366,height:768},{width:1920,height:1080}])for(const theme of ['light','dark'])for(const font of [18,26,54]){
   await page.setViewportSize(size);await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);await setSize(font);
   for(const system of ['ipa','extipa','voqs']){
    await page.getByRole('button',{name:{ipa:'IPA',extipa:'extIPA',voqs:'VoQS'}[system],exact:true}).click();const covered=[];
    for(const view of views[system]){
     if(system!=='voqs')await page.locator(`[data-chart-view="${view}"]`).click();await page.locator('.m17-chart-viewport').evaluate(e=>{e.scrollTop=0;e.scrollLeft=0;});
     const metric=await page.evaluate(()=>{const v=document.querySelector('.m17-chart-viewport'),ed=document.querySelector('.m17-editor'),r=ed.getBoundingClientRect(),nodes=[...v.querySelectorAll('[data-symbol-id]')];return{width:v.clientWidth,height:v.clientHeight,scrollWidth:v.scrollWidth,scrollHeight:v.scrollHeight,editorSize:parseFloat(getComputedStyle(ed).fontSize),editorVisible:r.top>=0&&r.bottom<=innerHeight,ids:nodes.map(e=>e.dataset.symbolId),glyphSizes:nodes.map(e=>parseFloat(getComputedStyle(e.querySelector('.m17-ipa')).fontSize)),nameSizes:[...v.querySelectorAll('.m17-symbol-name,.m17-row-caption b')].map(e=>parseFloat(getComputedStyle(e).fontSize)),glyphOverflows:nodes.filter(e=>e.scrollWidth>e.clientWidth+2).map(e=>e.dataset.symbolId)};});
     assert.equal(metric.editorSize,font);assert(metric.editorVisible,`${system}/${view}/${font} editor visible`);assert(metric.glyphSizes.every(n=>n>=font-10&&n<=font-5),`${system}/${view}/${font} every table glyph follows top control`);assert(metric.nameSizes.every(n=>n<=13),'names remain compact');assert.deepEqual(metric.glyphOverflows,[],`${system}/${view}/${font} glyph button content clipped`);if(font===26)assert(metric.scrollWidth<=metric.width+2,'default width fits viewport');
     const targets=page.locator('.m17-chart-viewport [data-symbol-id]');for(const index of [0,Math.floor(metric.ids.length/2),metric.ids.length-1]){await targets.nth(index).scrollIntoViewIfNeeded();assert(await targets.nth(index).evaluate(e=>{const r=e.getBoundingClientRect(),v=e.closest('.m17-chart-viewport').getBoundingClientRect();return r.left>=v.left-1&&r.right<=v.right+1&&r.top>=v.top-1&&r.bottom<=v.bottom+1;}),`${system}/${view}/${font} scroll target ${index} reachable`);}
     covered.push(...metric.ids);layouts.push({size,theme,font,system,view,...metric});if(font===26||(font===54&&view==='base')){await page.locator('.m17-chart-viewport').evaluate(e=>{e.scrollTop=0;e.scrollLeft=0;});await page.screenshot({path:path.join(out,`${size.width}-${theme}-${system}-${view}-${font}.png`)});}
    }
    assert.deepEqual(covered.sort(),catalog.entries.filter(e=>e.system===system).map(e=>e.id).sort(),`${system} complete unique coverage at ${font}`);
   }
  }
  await setSize(99);assert.equal(await sizeInput.inputValue(),'54');await setSize(1);assert.equal(await sizeInput.inputValue(),'18');await setSize(28);await page.waitForFunction(()=>document.querySelector('.m17-save-state').textContent==='本机草稿已保存');await page.reload();await ready();assert.equal(await sizeInput.inputValue(),'28','explicit new 28 preference persists');
  for(const [stored,version,expected] of [[28,null,26],[34,null,34],[28,1,28]]){
   await page.evaluate(({stored,version})=>new Promise((resolve,reject)=>{const r=indexedDB.open('phonetic-toolbox-m17',1);r.onsuccess=()=>{const tx=r.result.transaction('drafts','readwrite'),s=tx.objectStore('drafts'),g=s.get('ipa-plus.v1.local:M17');g.onsuccess=()=>{const d={...g.result,textSize:stored};if(version===null)delete d.textSizeVersion;else d.textSizeVersion=version;s.put(d,'ipa-plus.v1.local:M17');};tx.oncomplete=()=>{r.result.close();resolve();};tx.onerror=reject;};}),{stored,version});await page.reload();await ready();assert.equal(await sizeInput.inputValue(),String(expected));preferences.push({stored,version,expected});
  }
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'report.json'),JSON.stringify({passed:true,layouts,preferences,errors,browser:await browser.version(),checks:['84 layouts: 2 sizes x 2 themes x 3 font settings x 7 chart partitions','625 unique entries retained at each size/theme/font setting; sampled edge/middle symbols scroll into view','all visible table symbols follow shared font control; editor fixed and labels retain compact sizes','fresh26, old28 migration, nondefault34 preserved, intentional new28 persists, numeric range clamped']},null,2));console.log(out);
 }catch(e){await fs.writeFile(path.join(out,'failure.txt'),String(e));await page.screenshot({path:path.join(out,'failure.png')}).catch(()=>{});throw e;}finally{await browser.close();await server.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
