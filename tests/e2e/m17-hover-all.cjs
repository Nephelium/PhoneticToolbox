// M17-HOVER-ALL-VIEW: independent Chrome profile, no user drafts or media.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url'),{spawnSync}=require('node:child_process');
const {createHash}=require('node:crypto');
(async()=>{
 const root=path.resolve(__dirname,'../..'),out=path.join(root,'output/validation/m17-hover-all',new Date().toISOString().replace(/[:.]/g,'-')),scratch=path.join(out,'scratch');
 await fs.mkdir(scratch,{recursive:true});
 const marker={schema:'ptb-test-run/1',owner_pid:process.pid,state:'active',recipe:'Isolated Chrome profile; catalogue-only input, no WAV or research files'};
 const write=(name,data)=>fs.writeFile(path.join(out,name),JSON.stringify(data,null,2)+'\n');
 await write('run.json',marker);
 let server,context,page,outcome='failed';const errors=[],layouts=[],charts=[];
 try{
  const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
  server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0},logLevel:'error'});await server.listen();
  const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  context=await chromium.launchPersistentContext(path.join(scratch,'chrome'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1366,height:768}});
  page=context.pages()[0];page.on('pageerror',error=>errors.push(error.message));
  const data=JSON.parse(await fs.readFile(path.join(root,'frontend/src/modules/ipa-plus/data/catalog.json'),'utf8'));
  const ready=()=>page.locator('.m17-body[data-loaded=true][data-font-ready=true]').waitFor();
  const editor=page.getByRole('textbox',{name:'国际音标文本',exact:true}),toggle=page.getByLabel('点击播放',{exact:true}),panel=page.locator('.m17-hover-panel');
  await page.goto(server.resolvedUrls.local[0]+'#M17');await ready();
  assert(await toggle.isDisabled());assert(!await toggle.isChecked());
  assert.equal(await page.locator('.m17-toggle-unavailable').getAttribute('title'),'演示内容正在创作中，将在后续更新中开放。');
  const unavailable=page.locator('.m17-toggle-unavailable');await unavailable.hover();const toggleBox=await unavailable.boundingBox();await page.mouse.click(toggleBox.x+toggleBox.width*.75,toggleBox.y+toggleBox.height/2);assert(!await toggle.isChecked());
  for(const size of [{width:1366,height:768},{width:1024,height:720}])for(const theme of ['light','dark'])for(const zoom of [1,1.5]){
   await page.setViewportSize(size);await page.evaluate(async({theme,zoom})=>{const {setPageScale}=await import('/src/state/pageZoom.ts');setPageScale(zoom*100);document.documentElement.dataset.theme=theme;},{theme,zoom});
   await editor.fill('保留的草稿');
   for(const system of ['ipa','extipa']){
    await page.getByRole('button',{name:system==='ipa'?'IPA':'extIPA',exact:true}).click();
    const tabs=page.locator('.m17-section-tabs button'),all=page.locator('.m17-section-tabs [data-chart-view=all]');
    assert.equal(await tabs.first().innerText(),'显示全部');await all.click();assert.equal(await all.getAttribute('aria-pressed'),'true');
    const expected=data.entries.filter(entry=>entry.system===system).map(entry=>entry.id).sort();
    assert.deepEqual((await page.locator('.m17-chart-viewport [data-symbol-id]').evaluateAll(nodes=>nodes.map(node=>node.dataset.symbolId))).sort(),expected);
    const view=page.locator('.m17-chart-viewport');await view.evaluate(node=>node.scrollTop=0);
    const scrollLayout=()=>view.evaluate(node=>{const rect=selector=>{const box=document.querySelector(selector).getBoundingClientRect();return{top:box.top,bottom:box.bottom};};return{scrollTop:node.scrollTop,viewport:rect('.m17-chart-viewport'),meta:rect('.m17-chart-meta'),tabs:rect('.m17-section-tabs'),editor:rect('.m17-editor')};});
    const beforeScroll=await scrollLayout();await view.hover();await page.mouse.wheel(0,600);
    await page.waitForFunction(()=>document.querySelector('.m17-chart-viewport').scrollTop>0);
    await page.waitForTimeout(250);const afterScroll=await scrollLayout(),scrolled=afterScroll.scrollTop-beforeScroll.scrollTop;
    for(const item of ['meta','tabs']){assert(Math.abs(beforeScroll[item].top-afterScroll[item].top-scrolled*zoom)<1,`${item} must travel with the table: ${JSON.stringify({beforeScroll,afterScroll,zoom})}`);assert(afterScroll[item].bottom<=afterScroll.viewport.top,`${item} must scroll out of the clipped viewport`);}
    assert(Math.abs(beforeScroll.editor.top-afterScroll.editor.top)<1,'Editor stays in place while chart controls scroll');
    if(size.width===1366&&zoom===1)await page.screenshot({path:path.join(out,`${system}-scrolled-${theme}.png`)});
    const last=page.locator('.m17-chart-viewport [data-symbol-id]').last();await last.scrollIntoViewIfNeeded();
    const metric=await view.evaluate(node=>{const editor=document.querySelector('.m17-editor').getBoundingClientRect(),last=node.querySelector('[data-symbol-id]:last-child');return{scrollTop:node.scrollTop,scrollHeight:node.scrollHeight,clientHeight:node.clientHeight,editorVisible:editor.top>=0&&editor.bottom<=innerHeight};});
    assert(metric.editorVisible);assert(metric.scrollHeight>metric.clientHeight);assert.equal(await editor.inputValue(),'保留的草稿');charts.push({size,theme,zoom,system,count:expected.length,...metric,beforeScroll,afterScroll});
    await page.locator('.m17-section-tabs [data-chart-view=base]').click();assert.equal(await page.locator('.m17-chart-grid').getAttribute('data-chart-view-active'),'base');await all.click();
    await view.evaluate(node=>node.scrollTop=0);
    if(size.width===1366&&zoom===1)await page.screenshot({path:path.join(out,`${system}-all-${theme}.png`)});
   }
   await page.getByRole('button',{name:'VoQS',exact:true}).click();
   const voqsView=page.locator('.m17-chart-viewport');await voqsView.evaluate(node=>node.scrollTop=0);
   assert.equal(await voqsView.locator('.m17-chart-meta').count(),1,'VoQS legend belongs to the clipped chart viewport');
   if(await voqsView.evaluate(node=>node.scrollHeight>node.clientHeight)){
    const metaBefore=await page.locator('.m17-chart-meta').boundingBox();await voqsView.hover();await page.mouse.wheel(0,600);await page.waitForFunction(()=>document.querySelector('.m17-chart-viewport').scrollTop>0);await page.waitForTimeout(250);const metaAfter=await page.locator('.m17-chart-meta').boundingBox(),voqsScroll=await voqsView.evaluate(node=>node.scrollTop);assert(Math.abs(metaBefore.y-metaAfter.y-voqsScroll*zoom)<1,'VoQS legend scrolls with its chart');await voqsView.evaluate(node=>node.scrollTop=0);
   }
   for(const id of ['voqs-scope-008','voqs-scope-009','voqs-larynx-002']){
    const button=page.locator(`[data-symbol-id="${id}"]`);await button.scrollIntoViewIfNeeded();await button.hover();await panel.waitFor();await page.waitForTimeout(80);
    const metric=await panel.evaluate(node=>{const box=node.getBoundingClientRect(),heading=node.querySelector('.m17-detail-heading').getBoundingClientRect(),name=node.querySelector('.m17-detail-heading>div').getBoundingClientRect(),text=node.querySelector('p').getBoundingClientRect();return{left:box.left,top:box.top,right:box.right,bottom:box.bottom,width:box.width,scrollWidth:node.scrollWidth,clientWidth:node.clientWidth,headingHeight:heading.height,nameWidth:name.width,nameRight:name.right,textTop:text.top,view:[innerWidth,innerHeight]};});
    assert(metric.scrollWidth<=metric.clientWidth+1,JSON.stringify(metric));assert(metric.headingHeight<180*zoom,JSON.stringify(metric));assert(metric.nameWidth>100*zoom,JSON.stringify(metric));assert(metric.nameRight<=metric.right+1);assert(metric.textTop-metric.top<220*zoom);assert(metric.left>=0&&metric.top>=0&&metric.right<=metric.view[0]+1&&metric.bottom<=metric.view[1]+1);
    layouts.push({size,theme,zoom,id,...metric});
    if(size.width===1366&&zoom===1&&id==='voqs-scope-009')await page.screenshot({path:path.join(out,`long-example-${theme}.png`)});
    await panel.hover();await page.waitForTimeout(400);assert(await panel.isVisible());await page.mouse.move(1,1);await panel.waitFor({state:'hidden'});
   }
  }
  await page.evaluate(async()=>{const {setPageScale}=await import('/src/state/pageZoom.ts');setPageScale(100);});await page.setViewportSize({width:1366,height:768});
  const long=page.locator('[data-symbol-id=voqs-scope-009]');await long.click({button:'right'});const detail=page.locator('.m17-detail-panel');await detail.waitFor();assert(await detail.evaluate(node=>node.scrollWidth<=node.clientWidth+1));
  await page.getByRole('button',{name:'收起',exact:true}).click();await editor.fill('甲乙');await editor.evaluate(node=>{node.focus();node.setSelectionRange(1,1);node.dispatchEvent(new Event('select'));});
  await long.click();assert.equal(await editor.inputValue(),'甲'+data.entries.find(entry=>entry.id==='voqs-scope-009').insertText+'乙');await page.getByRole('button',{name:'撤销',exact:true}).click();assert.equal(await editor.inputValue(),'甲乙');
  await page.locator('[data-symbol-id=voqs-larynx-002]').hover();await panel.waitFor();assert((await panel.innerText()).includes('VoQS: 音质符号 (2016中英双语版)[EB/OL]. (2020-08-30)[2026-10-11]'));await page.mouse.move(1,1);await panel.waitFor({state:'hidden'});
  await page.getByRole('button',{name:'IPA',exact:true}).click();const search=page.getByRole('textbox',{name:'查找符号、名称、码位或 CIN',exact:true});await search.fill('pʰ');const shortId=data.entries.find(entry=>entry.system==='ipa'&&entry.isExample&&entry.insertText==='pʰ').id;await page.locator(`[data-symbol-id="${shortId}"]`).hover();await panel.waitFor();assert.equal(await panel.locator('.m17-detail-heading-example').count(),0,'Short examples keep the original side-by-side heading');await search.fill('');
  await page.reload();await ready();assert(!await toggle.isChecked());assert(await toggle.isDisabled());
  for(const system of ['ipa','extipa']){await page.getByRole('button',{name:system==='ipa'?'IPA':'extIPA',exact:true}).click();assert.equal(await page.locator('.m17-chart-grid').getAttribute('data-chart-view-active'),'all');}
  assert.deepEqual(errors,[]);await write('report.json',{passed:true,charts,layouts,errors,browser:context.browser()?.version(),checks:['Disabled playback with hover message and label-click protection','All-first defaults after reload; all 351 IPA and 209 extIPA IDs appear once; real wheel scroll, end reached, editor and text retained; individual views remain usable','24 long/short hover layouts at two windows, two themes and two zooms; readable headings, no horizontal overflow, viewport bounds and panel entry','Long full detail wraps; exact Unicode insertion and undo; current VoQS access date displayed']});outcome='passed';console.log(out);
 }catch(error){await write('failure.json',{error:String(error),errors,charts,layouts});if(page)await page.screenshot({path:path.join(out,'failure.png')}).catch(()=>{});throw error;}
 finally{
  try{if(context)await context.close();if(server)await server.close();}
  finally{const files=[];async function inventory(directory){for(const item of await fs.readdir(directory,{withFileTypes:true})){const file=path.join(directory,item.name),stat=await fs.lstat(file);assert(!stat.isSymbolicLink(),'Linked scratch retained');if(stat.isDirectory())await inventory(file);else files.push({path:path.relative(scratch,file),bytes:stat.size,sha256:createHash('sha256').update(await fs.readFile(file)).digest('hex')});}}await inventory(scratch);await write('scratch-manifest.json',{schema:'ptb-test-scratch-files/1',recipe:marker.recipe,files});await write('run.json',{...marker,state:'ready',test_outcome:outcome});const result=spawnSync('powershell.exe',['-NoProfile','-ExecutionPolicy','Bypass','-File',path.join(root,'scripts/cleanup_test_scratch.ps1'),'-RunDirectory',out],{windowsHide:true,encoding:'utf8'});if(result.status!==0)throw Error(result.stderr||result.stdout||'Scratch cleanup failed');}
 }
})().catch(error=>{console.error(error);process.exitCode=1;});
