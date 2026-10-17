// Source component counters; no scientific computation or user media involved.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url'),{spawnSync}=require('node:child_process'),{createHash}=require('node:crypto');
const root=path.resolve(__dirname,'../..');
async function main(){
 const baseline=process.argv.includes('--baseline'),out=path.join(root,'output/validation/redundant-work',(baseline?'before-':'after-')+Date.now()),scratch=path.join(out,'scratch');await fs.mkdir(scratch,{recursive:true});
 const write=(name,value)=>fs.writeFile(path.join(out,name),JSON.stringify(value,null,2)+'\n');
 const marker={schema:'ptb-test-run/1',owner_pid:process.pid,state:'active',recipe:'Real Vue components; 12000-point tracks, 2x1200-bin waveform, controlled four-page PDF port; 12 hide/show changes, ten selection changes, real resize and new data'};
 await write('run.json',marker);let browser,server,page;const report={success:false,baseline,errors:[]};
 try{
  const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
  const functions=[['/ParameterFigure.vue','function path(curve:OverlayCurve){','parameters'],['/RecordingWave.vue','function points(bins:number[][]){','recording'],['/HistoryPlot.vue','function path(r:Result){','history'],['/PitchCurve.vue','function path(values:number[]){','pitch'],['/F0Comparison.vue','function path(curve:{axis:number[];values:number[]}){','f0']];
  server=await createServer({root:path.join(root,'frontend'),plugins:[{name:'owned-computation-counter',enforce:'pre',transform(code,id){for(const [file,signature,key] of functions)if(id.endsWith(file)){assert(code.includes(signature),file);return code.replace(signature,signature+`(window.__workAudit.counts.${key}=(window.__workAudit.counts.${key}||0)+1);`);}}}],server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{noDiscovery:true,include:['vue']}});await server.listen();
  const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  browser=await chromium.launchPersistentContext(path.join(scratch,'profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1440,height:1000},args:['--mute-audio']});page=await browser.newPage();page.on('pageerror',error=>report.errors.push(error.message));
  await page.goto(server.resolvedUrls.local[0]);
  await page.evaluate(async()=>{window.__workAudit={counts:{},renders:[],long:[]};new PerformanceObserver(list=>window.__workAudit.long.push(...list.getEntries().map(e=>e.duration))).observe({entryTypes:['longtask']});const {mountRedundantWork}=await import('/tests/fixtures/redundant-work.ts');window.__workAudit.ui=mountRedundantWork();});
  await page.waitForFunction(()=>document.querySelectorAll('[data-audit-pane=pdf] img').length>=3);await page.waitForTimeout(400);
  const state=()=>page.evaluate(()=>({counts:{...window.__workAudit.counts},renders:[...window.__workAudit.renders],long:[...window.__workAudit.long],parameters:document.querySelector('.parameter-curve')?.getAttribute('d'),recording:document.querySelector('.wave-line')?.getAttribute('points')}));
  report.initial=await state();
  if(!baseline){assert.equal(report.initial.renders.filter(r=>r.page===1).length,1,'first page must wait for actual layout instead of rasterizing the default width');assert.equal(new Set(report.initial.renders.map(r=>r.width)).size,1,'initial PDF requests must use the settled viewport width');}
  await page.evaluate(()=>{window.__workAudit.counts={};window.__workAudit.renders=[];window.__workAudit.long=[];});
  for(let i=0;i<6;i++){await page.evaluate(()=>window.__workAudit.ui.toggle(false));await page.waitForTimeout(190);await page.evaluate(()=>window.__workAudit.ui.toggle(true));await page.waitForTimeout(250);}
  report.toggles=await state();
  await page.evaluate(()=>{window.__workAudit.counts={};window.__workAudit.renders=[];window.__workAudit.long=[];});
  for(let i=0;i<10;i++)await page.evaluate(i=>window.__workAudit.ui.select(i),i);
  report.selections=await state();
  assert.equal(report.selections.parameters,report.initial.parameters);assert.equal(report.selections.recording,report.initial.recording);
  if(!baseline){assert.equal(report.toggles.renders.length,0);assert.equal(report.selections.counts.parameters||0,0);assert.equal(report.selections.counts.recording||0,0);assert.equal(report.selections.counts.f0||0,0);for(const key of ['parameters','recording','history','pitch','f0'])assert.equal(report.toggles.counts[key]||0,0,key);}
  await page.evaluate(()=>window.__workAudit.ui.resize(900));await page.waitForTimeout(350);report.resize=await state();assert.notEqual(report.resize.parameters,report.initial.parameters);assert(report.resize.renders.length>0,'actual width change refreshes PDF raster');
  await page.evaluate(()=>window.__workAudit.ui.newData());report.updated=await state();assert.notEqual(report.updated.parameters,report.resize.parameters);assert.notEqual(report.updated.recording,report.initial.recording);
  await page.screenshot({path:path.join(out,'components.png')});await page.evaluate(()=>window.__workAudit.ui.dispose());
  await page.evaluate(async()=>{const oldTimeout=window.setTimeout.bind(window),oldInterval=window.setInterval.bind(window);window.setTimeout=(fn,ms,...args)=>oldTimeout(fn,[600,1500,6000].includes(ms)?60:ms,...args);window.setInterval=(fn,ms,...args)=>oldInterval(fn,Math.min(ms,60),...args);window.__workAudit.polls={};const {mountPollingAudit}=await import('/tests/fixtures/redundant-work.ts');window.__workAudit.polling=mountPollingAudit();});
  await page.waitForTimeout(1800);report.pollingInitial=await page.evaluate(()=>({...window.__workAudit.polls}));
  await page.evaluate(async()=>{await window.__workAudit.polling.toggle(false);window.__workAudit.polls={};});await page.waitForTimeout(1800);report.hiddenIdle=await page.evaluate(()=>({...window.__workAudit.polls}));
  if(!baseline)assert.deepEqual(report.hiddenIdle,{});
  await page.evaluate(async()=>{window.__workAudit.polling.setLive(true);await window.__workAudit.polling.toggle(true);});await page.waitForTimeout(1800);
  await page.evaluate(async()=>{await window.__workAudit.polling.toggle(false);window.__workAudit.polls={};});await page.waitForTimeout(1800);report.hiddenRunning=await page.evaluate(()=>({...window.__workAudit.polls}));
  for(const key of ['egg','lpc','pitch','reconstruction','recording'])assert(report.hiddenRunning[key]>0,key);
  await page.evaluate(()=>window.__workAudit.polling.setLive(false));await page.waitForTimeout(1800);await page.evaluate(()=>window.__workAudit.polls={});await page.waitForTimeout(1800);report.hiddenCompleted=await page.evaluate(()=>({...window.__workAudit.polls}));
  if(!baseline)assert.deepEqual(report.hiddenCompleted,{});
  await page.evaluate(()=>window.__workAudit.polling.dispose());
  await page.evaluate(async()=>{const {mountMonitorAudit}=await import('/tests/fixtures/redundant-work.ts');window.__workAudit.monitor=await mountMonitorAudit();});await page.waitForTimeout(250);assert(await page.evaluate(()=>window.__workAudit.monitorCalls>0));
  await page.evaluate(()=>{window.__workAudit.monitor.toggle(false);window.__workAudit.monitorCalls=0;});await page.waitForTimeout(300);report.hiddenMonitorCalls=await page.evaluate(()=>window.__workAudit.monitorCalls);assert.equal(report.hiddenMonitorCalls,0);
  await page.evaluate(()=>window.__workAudit.monitor.toggle(true));await page.waitForTimeout(250);assert(await page.evaluate(()=>window.__workAudit.monitorCalls>0));await page.evaluate(()=>window.__workAudit.monitor.dispose());
  assert.deepEqual(report.errors,[]);report.success=true;
 }catch(error){report.error=String(error);throw error;}
 finally{
  await browser?.close();await server?.close();await write('report.json',report);const files=[];async function inventory(dir){for(const item of await fs.readdir(dir,{withFileTypes:true})){assert(!item.isSymbolicLink());const file=path.join(dir,item.name);if(item.isDirectory())await inventory(file);else{const bytes=await fs.readFile(file);files.push({path:path.relative(scratch,file),bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex')});}}}await inventory(scratch);await write('scratch-manifest.json',{schema:'ptb-test-scratch-files/1',files});await write('run.json',{...marker,state:'ready',test_outcome:report.success?'completed':'failed'});
  const cleanup=spawnSync('powershell.exe',['-NoProfile','-ExecutionPolicy','Bypass','-File',path.join(root,'scripts/cleanup_test_scratch.ps1'),'-RunDirectory',out],{windowsHide:true,encoding:'utf8'});assert.equal(cleanup.status,0,cleanup.stdout+cleanup.stderr);
  console.log(JSON.stringify({out,success:report.success,toggles:report.toggles&&{counts:report.toggles.counts,renders:report.toggles.renders.length},selections:report.selections?.counts,hiddenIdle:report.hiddenIdle,hiddenRunning:report.hiddenRunning,hiddenCompleted:report.hiddenCompleted,error:report.error}));
 }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
