// Renderer counts/policy only. Trial2 geometry/content remains untouched.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..'),[out,scratch]=process.argv.slice(2);
(async()=>{
 const {preview}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const report={success:false,cases:[],errors:[],scope:'Actual AppShell/OL Canvas with read-only live-layer observer; real trial2 files; rendering policy/counts only, no content/coordinate review.'};let server,context,page;
 try{
  server=await preview({configFile:false,root:path.join(root,'frontend'),base:'./',build:{outDir:process.env.PTB_GEO_PERF_DIST},preview:{host:'127.0.0.1',port:0}});
  context=await chromium.launchPersistentContext(path.join(scratch,'profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1440,height:900},acceptDownloads:true});context.setDefaultTimeout(60000);
  await context.addInitScript(()=>{window.__ptbGeoPerf={stage:'renderer',reads:[]};Object.defineProperty(window,'showSaveFilePicker',{configurable:true,value:undefined});});
  page=await context.newPage();page.on('pageerror',e=>report.errors.push(e.message));await page.goto(server.resolvedUrls.local[0]);await page.locator('.home-page').waitFor();await page.waitForTimeout(3200);const announcement=page.getByRole('dialog').getByRole('button',{name:'知道了',exact:true});if(await announcement.isVisible())await announcement.click();await page.locator('.nav-item').and(page.getByRole('button',{name:'语言地图',exact:true})).click();const atlas=page.locator('.language-atlas');await atlas.getByLabel('选择语言地图资料夹',{exact:true}).setInputFiles(path.join(scratch,'geolinguistics-data'));await atlas.locator('.atlas-workbench').waitFor();await atlas.getByRole('tab',{name:'语言现象',exact:true}).click();
  const snapshot=()=>page.evaluate(()=>[...window.__ptbRendererEntries.values()].map(fn=>fn()).filter(m=>m.active));
  const sizes={province:34,city:370,district:2875};
  async function settle(){await page.waitForFunction(()=>!document.querySelector('.map-loading'));await page.waitForTimeout(400);}
  async function selectAuto(level){for(let i=0;i<12;i++){const m=(await snapshot())[0];if(m.level===level)return;await atlas.locator(m.zoom<(level==='province'?6:level==='city'?6:8)?'.ol-zoom-in':'.ol-zoom-out').first().click();await settle();}throw Error('Automatic level did not reach '+level);}
  async function inspect(theme,level,count){
   await settle();const live=await snapshot();assert.equal(live.length,count);
   for(const m of live){assert.equal(m.level,level);assert.deepEqual(m.layers.filter(l=>l.hasAdmin).map(l=>l.index),[0]);assert.equal(m.layers[0].features,sizes[level]);assert.equal(m.territory.length,13);for(const type of ['国界','争议','港澳']){const f=m.territory.filter(f=>f.boundaryType===type);assert.equal(f.length,1);assert(f[0].styles>0);}assert(!m.territory.some(f=>['省界','海洋'].includes(f.boundaryType)||f.role==='administrative-display-area'));assert.equal(m.territory.filter(f=>f.role==='boundary-symbol'&&f.styles>0).length,10);}
   const event=page.waitForEvent('download');await atlas.getByRole('button',{name:'SVG',exact:true}).click();await page.getByRole('dialog').getByRole('button',{name:'导出图片',exact:true}).click();const file=await event,save=path.join(scratch,`${theme}-${level}-${count}.svg`);await file.saveAs(save);const xml=await fs.readFile(save,'utf8');
   const svg=await page.evaluate(xml=>{const d=new DOMParser().parseFromString(xml,'image/svg+xml');return [...d.querySelectorAll('g[clip-path]')].map(g=>{const p=[...g.querySelectorAll('path')];return {paths:p.length,lines:p.filter(x=>x.getAttribute('d')&&!x.getAttribute('d').includes('Z')).length,symbols:p.filter(x=>x.getAttribute('fill')==='#526c78').length,admin:p.filter(x=>x.getAttribute('stroke')==='#9aa9aa').length,disputed:p.filter(x=>x.getAttribute('stroke-dasharray')==='4 3').length};});},xml);
   assert.equal(svg.length,count);for(const s of svg){assert.equal(s.paths,13+sizes[level]);assert.equal(s.lines,3);assert.equal(s.symbols,10);assert.equal(s.admin,sizes[level]);assert.equal(s.disputed,1);}
   await page.screenshot({path:path.join(out,`${theme}-${level}-${count}-canvas.png`)});report.cases.push({theme,level,count,live,svg});
  }
  await atlas.locator('.geography-toolbar label').filter({hasText:'边界'}).locator('select').selectOption('auto');
  // Establish the existing export status row before screen/export comparisons.
  for(const theme of ['light','dark']){await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);for(const level of ['province','city','district','province']){await selectAuto(level);await inspect(theme,level,1);}}
  for(const count of [2,4]){await atlas.locator('.map-heading label select').selectOption(String(count));await settle();await inspect('dark','province',count);}
  assert.deepEqual(report.errors,[]);report.success=true;
 }catch(e){report.error=e.stack;process.exitCode=1;}
 finally{await context?.close();await new Promise(resolve=>server?server.httpServer.close(resolve):resolve());await fs.writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2));}
})();
