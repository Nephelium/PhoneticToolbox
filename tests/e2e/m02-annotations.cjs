// M02: streamed and legacy frame labels, actual SVG rendering and exports.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url'),{spawnSync}=require('node:child_process'),{createHash}=require('node:crypto');
const root=path.resolve(__dirname,'../..');
async function main(){
 const out=path.join(root,'output/validation/m02-annotations','chrome-'+Date.now()),scratch=path.join(out,'scratch');await fs.mkdir(scratch,{recursive:true});
 const write=(name,data)=>fs.writeFile(path.join(out,name),JSON.stringify(data,null,2)+'\n');
 const marker={schema:'ptb-test-run/1',owner_pid:process.pid,state:'active',started_at:new Date().toISOString(),recipe:'In-memory 8000 Hz synthetic PCM; 4393 frames, labels shang4/yi1/blank/shang4; legacy and streamed browser ports, light/dark full/zoom/boundary, SVG/PNG exports'};await write('run.json',marker);
 const report={success:false,cases:[],errors:[]};let browser,server,page;
 try{
  const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{noDiscovery:true,include:['vue']}});await server.listen();
  const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  browser=await chromium.launchPersistentContext(path.join(scratch,'profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1440,height:1000},acceptDownloads:true,args:['--mute-audio']});report.browser=browser.browser().version();page=await browser.newPage();page.on('pageerror',e=>report.errors.push(e.message));
  for(const streamed of [false,true]){
   await page.goto(server.resolvedUrls.local[0]+'tests/m02-annotations.html?streamed='+Number(streamed));await page.getByRole('button',{name:'annotations.wav',exact:true}).click();
   for(const name of ['H1-H2','text_words'])await page.getByLabel(name,{exact:true}).check();
   const inspect=()=>page.evaluate(()=>{const groups=[document.querySelector('.m02-wave-annotations'),document.querySelector('.shared-plot-area')];return groups.map(g=>({labels:[...g.querySelectorAll('.ipa-text')].map(t=>[...t.childNodes].filter(n=>n.nodeType===3).map(n=>n.textContent).join('')),boundaries:[...g.querySelectorAll('path[stroke-dasharray]')].filter(p=>!p.classList.contains('parameter-curve')&&p.getAttribute('d').includes('V')).map(p=>p.getAttribute('d'))}));});
   for(const mode of ['light','dark'])for(const view of ['full','zoom','boundary']){
    await page.evaluate(({mode,view})=>{document.documentElement.dataset.theme=mode;window.qa.wave.zoom=view==='full'?1:window.qa.wave.asset.duration/(view==='zoom'?.05065:.1);window.qa.wave.offset=view==='full'?0:view==='zoom'?.64947:.7;},{mode,view});
    const expected=view==='full'?['shang4','yi1','shang4']:view==='zoom'?['shang4','yi1']:['yi1'];
    await page.waitForFunction(expected=>{const g=document.querySelector('.m02-wave-annotations');return JSON.stringify([...g.querySelectorAll('.ipa-text')].map(t=>[...t.childNodes].filter(n=>n.nodeType===3).map(n=>n.textContent).join('')))===JSON.stringify(expected);},expected);
    await page.waitForFunction(()=>!document.querySelector('.parameter-figure').getAttribute('aria-busy')||document.querySelector('.parameter-figure').getAttribute('aria-busy')==='false');
    const actual=await inspect();for(const g of actual){assert.deepEqual(g.labels,expected);assert.equal(g.boundaries.length,view==='full'?4:expected.length);}assert.equal(await page.locator('.parameter-curve').count(),1);
    const curve=page.locator('.parameter-curve');assert(Number(await curve.getAttribute('data-scale-min'))<11);assert(Number(await curve.getAttribute('data-scale-max'))>9);assert((await curve.getAttribute('d')).includes('L'));
    report.cases.push({streamed,mode,view,...{actual}});await page.locator('.signal-panel').screenshot({path:path.join(out,`${streamed?'streamed':'legacy'}-${mode}-${view}.png`)});
   }
   // Export the same zoomed view. SVG text and dense boundary paths are checked independently.
   await page.evaluate(()=>{window.qa.wave.offset=.64947;window.qa.wave.zoom=window.qa.wave.asset.duration/.05065;});await page.waitForFunction(()=>document.querySelectorAll('.m02-wave-annotations .ipa-text').length===2&&document.querySelector('.parameter-figure').getAttribute('aria-busy')==='false');
   await page.getByLabel('图窗 1图片格式',{exact:true}).selectOption('svg');let download=page.waitForEvent('download');await page.getByRole('button',{name:'保存当前图',exact:true}).click();let file=await download;const exported=path.join(scratch,`annotations-${streamed}.svg`);await file.saveAs(exported);const xml=await fs.readFile(exported,'utf8');assert.equal((xml.match(/>shang4<\/text>/g)??[]).length,1);assert.equal((xml.match(/>yi1<\/text>/g)??[]).length,1);
   download=page.waitForEvent('download');await page.getByRole('button',{name:'保存整幅 PNG',exact:true}).click();file=await download;const png=path.join(scratch,`annotations-${streamed}.png`);await file.saveAs(png);const bytes=await fs.readFile(png);assert.equal(bytes.subarray(1,4).toString(),'PNG');assert(bytes.length>1000);
  }
  assert.deepEqual(report.errors,[]);report.success=true;
 }catch(error){report.failure=String(error);if(page)await page.screenshot({path:path.join(out,'failure.png')}).catch(()=>{});throw error;}
 finally{
  await browser?.close();await server?.close();await write('report.json',report);const files=[];async function inventory(dir){for(const entry of await fs.readdir(dir,{withFileTypes:true})){assert(!entry.isSymbolicLink());const file=path.join(dir,entry.name);if(entry.isDirectory())await inventory(file);else try{const bytes=await fs.readFile(file);files.push({path:path.relative(scratch,file),bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex')});}catch(e){if(e.code!=='ENOENT')throw e;}}}await inventory(scratch);await write('scratch-manifest.json',{schema:'ptb-test-scratch-files/1',files});await write('run.json',{...marker,state:'ready',test_outcome:report.success?'completed':'failed'});const cleanup=spawnSync('powershell.exe',['-NoProfile','-ExecutionPolicy','Bypass','-File',path.join(root,'scripts/cleanup_test_scratch.ps1'),'-RunDirectory',out],{windowsHide:true,encoding:'utf8'});if(cleanup.status!==0)await write('cleanup.json',{status:'retained',reason:cleanup.stderr||cleanup.stdout});console.log(JSON.stringify({out,success:report.success,cases:report.cases.length,failure:report.failure}));
 }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
