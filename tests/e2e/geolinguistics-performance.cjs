/* QA observation only. Run scripts/verify_geolinguistics_performance.py. */
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..'),[out,scratch]=process.argv.slice(2);
(async()=>{
 const {preview}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const report={success:false,stages:[],errors:[],scope:'Actual AppShell/GeoPort/GeoMap, exact real runtime, one owned Chrome, normal production build with observer only',
  limitations:['Controlled no-concurrent-owned-QA session; one machine, one run, no population performance claim.',
   'readFeatures measures OL object-to-feature and projection together. It excludes VectorSource.addFeatures and Canvas rendering.',
   'Stage wall time includes UI events, parsing, projection, source insertion, rendering and an explicit 400ms quiet tail.',
   'JSON.parse timing records map FeatureCollections only; no altered results or replaced platform ports.',
   'SVG identity checks select the supported download fallback; native operating-system save picker is not exercised.',
   'After each measured stage, explicit CDP GC measures live JS heap; GC time is outside stage wall time. Read-only Page snapshot creates no new strong map references.']};
 let server,context,page,cdp;
 try{
  server=await preview({configFile:false,root:path.join(root,'frontend'),base:'./',build:{outDir:process.env.PTB_GEO_PERF_DIST||path.join(scratch,'frontend')},preview:{host:'127.0.0.1',port:0}});
  context=await chromium.launchPersistentContext(path.join(scratch,'profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1440,height:900}});
  context.setDefaultTimeout(90000);
  await context.addInitScript(()=>{
   Object.defineProperty(window,'showSaveFilePicker',{configurable:true,value:undefined});
   const p=window.__ptbGeoPerf={stage:'navigation',reads:[],parses:[],longtasks:[],frames:[],begins:[]};
   const parse=JSON.parse;
   JSON.parse=function(value,...rest){const map=typeof value==='string'&&value.slice(0,100).includes('"FeatureCollection"');if(!map)return parse.call(this,value,...rest);
    const began=performance.now(),result=parse.call(this,value,...rest);p.parses.push({stage:p.stage,start:began,ms:performance.now()-began,characters:value.length,features:result?.features?.length,firstId:result?.features?.[0]?.id});return result;};
   p.longtaskSupported=PerformanceObserver.supportedEntryTypes.includes('longtask');
   if(p.longtaskSupported)new PerformanceObserver(list=>{for(const e of list.getEntries())p.longtasks.push({start:e.startTime,ms:e.duration});}).observe({type:'longtask',buffered:true});
   let last;const frame=now=>{if(last!==undefined)p.frames.push({start:last,ms:now-last});last=now;requestAnimationFrame(frame);};requestAnimationFrame(frame);
  });
  page=await context.newPage();page.on('pageerror',e=>report.errors.push(e.message));
  cdp=await context.newCDPSession(page);await cdp.send('Performance.enable');
  await page.goto(server.resolvedUrls.local[0]);await page.locator('.home-page').waitFor();
  await page.locator('.nav-item').and(page.getByRole('button',{name:'语言地图',exact:true})).click();
  const atlas=page.locator('.language-atlas');await atlas.getByLabel('选择语言地图资料夹',{exact:true}).waitFor({state:'attached'});
  async function measure(name,action,count){
   const start=await page.evaluate(name=>{const p=window.__ptbGeoPerf;p.stage=name;const t=performance.now();p.begins.push({name,start:t});return t;},name);
   await action();
   await page.waitForFunction(count=>document.querySelectorAll('.geo-map').length===count&&document.querySelectorAll('.map-target canvas').length>=count&&!document.querySelector('.map-loading'),count);
   // Wait through the final canvas frame and a fixed observable quiet tail.
   await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
   await page.waitForTimeout(400);
   const stage=await page.evaluate(({name,start})=>{const p=window.__ptbGeoPerf,end=performance.now(),reads=p.reads.filter(v=>v.start>=start&&v.start<end),parses=p.parses.filter(v=>v.start>=start&&v.start<end),longtasks=p.longtasks.filter(v=>v.start>=start&&v.start<end),frames=p.frames.filter(v=>v.start>=start&&v.start<end);
    return {name,start,end,wallMs:end-start,quietTailMs:400,readFeatures:reads,parses,longtasks,maxFrameGapMs:Math.max(0,...frames.map(v=>v.ms)),readFeaturesTotalMs:reads.reduce((a,v)=>a+v.ms,0),parseTotalMs:parses.reduce((a,v)=>a+v.ms,0),longtaskTotalMs:longtasks.reduce((a,v)=>a+v.ms,0)};},{name,start});
   stage.mapRefs=await page.evaluate(()=>({domCount:document.querySelectorAll('.geo-map').length,...window.__ptbMapRefSnapshot()}));
   const before=await cdp.send('Runtime.getHeapUsage');await cdp.send('HeapProfiler.collectGarbage');const after=await cdp.send('Runtime.getHeapUsage');
   stage.heapBytes={beforeGc:before.usedSize,afterGc:after.usedSize,total:after.totalSize,embedder:after.embedderHeapUsedSize,backing:after.backingStorageSize};
   report.stages.push(stage);await fs.writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2));
  }
  await measure('first-import-and-map',async()=>{await atlas.getByLabel('选择语言地图资料夹',{exact:true}).setInputFiles(path.join(scratch,'geolinguistics-data'));await atlas.locator('.atlas-workbench').waitFor();},1);
  await atlas.getByRole('tab',{name:'语言现象',exact:true}).click();
  await atlas.getByLabel('检索语言特征',{exact:true}).fill('F120_P');await atlas.locator('.feature-list button').click();
  const count=atlas.locator('.map-heading label select'),level=atlas.locator('.geography-toolbar label').filter({hasText:'边界'}).locator('select');
  await measure('one-to-two-maps',()=>count.selectOption('2'),2);
  await measure('two-to-four-maps',()=>count.selectOption('4'),4);
  await measure('four-first-district',()=>level.selectOption('district'),4);
  for(let i=1;i<=2;i++){
   await measure('four-province-repeat-'+i,()=>level.selectOption('province'),4);
   await measure('four-district-repeat-'+i,()=>level.selectOption('district'),4);
  }
  await measure('four-to-one-map',()=>count.selectOption('1'),1);
  await measure('one-to-four-district-remount',()=>count.selectOption('4'),4);
  await measure('district-to-city',()=>level.selectOption('city'),4);
  await measure('city-to-district',()=>level.selectOption('district'),4);
  await measure('hide-and-reopen-m19',async()=>{await page.locator('.nav-item').and(page.getByRole('button',{name:'首页',exact:true})).click();await page.locator('.home-page').waitFor();await page.locator('.nav-item').and(page.getByRole('button',{name:'语言地图',exact:true})).click();},4);
  report.measurementsComplete=true;await fs.writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2));
  if(process.env.PTB_GEO_EXPECT_CACHED==='1'){
   const byName=Object.fromEntries(report.stages.map(s=>[s.name,s]));
   assert.equal(byName['four-first-district'].readFeatures.filter(r=>r.features===2875).length,1);
   for(const name of ['four-district-repeat-1','four-district-repeat-2','one-to-four-district-remount','city-to-district'])assert.equal(byName[name].readFeatures.filter(r=>r.features>0).length,0,name);
   assert.equal(byName['four-to-one-map'].mapRefs.entries.filter(v=>v?.present).length,1);
   for(const stage of report.stages)assert.equal(stage.mapRefs.entries.filter(v=>v?.present).length,stage.mapRefs.domCount,stage.name);
   async function triangleCount(name){
    const pending=page.waitForEvent('download');await atlas.getByRole('button',{name:'SVG',exact:true}).click();const downloaded=await pending;
    const file=path.join(scratch,name+'.svg');await downloaded.saveAs(file);const xml=await fs.readFile(file,'utf8');
    return page.evaluate(xml=>[...new DOMParser().parseFromString(xml,'image/svg+xml').querySelectorAll('g[clip-path] path')].filter(p=>p.getAttribute('fill')==='#edf0ed'&&(p.getAttribute('d').match(/M/g)||[]).length===1&&(p.getAttribute('d').match(/L/g)||[]).length===3&&p.getAttribute('d').endsWith('Z')).length,xml);
   }
   await level.selectOption('province');await page.waitForFunction(()=>!document.querySelector('.map-loading'));
   await atlas.getByRole('button',{name:'中国',exact:true}).click();await page.waitForTimeout(500);
   const original=await triangleCount('generation-original');
   await measure('dataset-generation-fixture',async()=>{await atlas.getByLabel('选择语言地图资料夹',{exact:true}).setInputFiles(path.join(scratch,'generation-fixture'));await page.waitForFunction(()=>window.__ptbGeoPerf.reads.some(r=>r.stage==='dataset-generation-fixture'&&r.features===128));},4);
   const replacement=await triangleCount('generation-replacement');assert.equal(replacement,original+4);
   await measure('dataset-original-restore',async()=>{await atlas.getByLabel('选择语言地图资料夹',{exact:true}).setInputFiles(path.join(scratch,'geolinguistics-data'));await page.waitForFunction(()=>window.__ptbGeoPerf.reads.some(r=>r.stage==='dataset-original-restore'&&r.features===127));},4);
   const restored=await triangleCount('generation-restored');assert.equal(restored,original);
   report.datasetGeneration={syntheticMarker:true,panels:4,boundary:'province',originalTriangles:original,replacementTriangles:replacement,restoredTriangles:restored};
  }
  report.observer=await page.evaluate(()=>({longtaskSupported:window.__ptbGeoPerf.longtaskSupported,reads:window.__ptbGeoPerf.reads.length,parses:window.__ptbGeoPerf.parses.length}));
  assert(report.observer.longtaskSupported);assert.deepEqual(report.errors,[]);
  for(const stage of report.stages)for(const read of stage.readFeatures){assert.equal(read.error,undefined);assert.equal(read.inputType,'object');assert.equal(read.dataProjection,'EPSG:4326');assert.equal(read.featureProjection,'EPSG:3857');}
  report.success=true;
 }catch(e){report.error=e.stack;process.exitCode=1;}
 finally{await context?.close();await new Promise(resolve=>server?server.httpServer.close(resolve):resolve());await fs.writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2));}
})();
