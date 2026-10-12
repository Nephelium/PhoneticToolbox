// Actual M06 help, generated manual, registered media, and muted playback.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url'),{spawnSync}=require('node:child_process'),{createHash}=require('node:crypto');
const root=path.resolve(__dirname,'../..');
async function main(){
 const out=path.join(root,'output/validation/m06-manual','reader-'+Date.now()),scratch=path.join(out,'scratch');
 await fs.mkdir(scratch,{recursive:true});
 const write=(name,data)=>fs.writeFile(path.join(out,name),JSON.stringify(data,null,2)+'\n');
 const marker={schema:'ptb-test-run/1',owner_pid:process.pid,state:'active',started_at:new Date().toISOString(),recipe:'M06 actual Vue help and software manual; independent muted Chrome, 9 example media, light/dark and narrow layouts'};
 await write('run.json',marker);
 const report={success:false,checks:[],media:[],errors:[]};let server,browser;
 try{
  const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
  server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{noDiscovery:true,include:['vue']}});await server.listen();
  const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  browser=await chromium.launchPersistentContext(path.join(scratch,'profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--mute-audio'],viewport:{width:1600,height:1000}});
  const page=await browser.newPage();page.on('pageerror',e=>report.errors.push(e.message));
  await page.goto(server.resolvedUrls.local[0]);
  await page.locator('nav').getByRole('button',{name:'声学参数合成',exact:true}).click();
  await page.locator('.m06-page').getByRole('button',{name:'帮助',exact:true}).click();
  await page.locator('[data-manual-chapter="m06"][aria-current="page"]').waitFor();
  await page.locator('.manual-document').waitFor();
  assert.equal(await page.locator('.manual-media-audio').count(),9);
  const project=JSON.parse(await fs.readFile(path.join(root,'frontend/public/manual/project.json'),'utf8'));
  const figures=await page.locator('.manual-media').evaluateAll(elements=>elements.map(e=>({id:e.dataset.assetId,kind:e.querySelector('audio')?'audio':e.querySelector('img')?'image':'other'})));
  for(const figure of figures){
   const asset=project.assets.find(a=>a.id===figure.id);assert(asset,'Unknown manual media');
   const response=await page.request.get(server.resolvedUrls.local[0]+'manual/'+asset.path);assert.equal(response.status(),200);
   const digest=createHash('sha256').update(await response.body()).digest('hex');assert.equal(digest,asset.sha256);
   if(figure.kind==='audio'){
    const element=page.locator('[data-asset-id="'+figure.id+'"] audio');
    const result=await element.evaluate(async e=>{
     e.muted=true;e.load();
     await new Promise((resolve,reject)=>{if(e.readyState>=2)return resolve();e.addEventListener('canplay',resolve,{once:true});e.addEventListener('error',()=>reject(new Error('Audio decode failed')),{once:true});setTimeout(()=>reject(new Error('Audio readiness timeout')),10000);});
     const duration=e.duration;await e.play();
     await new Promise((resolve,reject)=>{e.addEventListener('ended',resolve,{once:true});setTimeout(()=>reject(new Error('Playback timeout')),10000);});
     return {duration,time:e.currentTime,ended:e.ended,src:e.currentSrc};
    });
    assert(result.ended);assert(Math.abs(result.duration-asset.duration)<.15);assert.equal(await page.locator('[data-asset-id="'+figure.id+'"] .manual-download').getAttribute('href'),await element.getAttribute('src'));
    report.media.push({id:figure.id,sha256:digest,...result});
   }
  }
  assert.equal(new Set(report.media.filter(m=>/^m06-[aiu]-modal-150/.test(m.id)).map(m=>m.sha256)).size,3);
  report.checks.push('M06 module help opens chapter 8; all referenced media return 200 with exact registered SHA-256; nine MP3s decode and play to end; three vowel files distinct; download URLs match players');
  for(const theme of ['浅色','深色']){
   await page.locator('#tool-sidebar').getByRole('button',{name:'设置',exact:true}).click();await page.getByRole('button',{name:theme,exact:true}).click();
   await page.locator('[id="tab-help"]').click();
   for(const [key,id] of [['syntax','m06-vowel-syntax-complex-light'],['fade100','m06-fade-100-100-light'],['fade30','m06-fade-30-80-light']]){
    const figure=page.locator('[data-asset-id="'+id+'"]');await figure.scrollIntoViewIfNeeded();
    await figure.locator('img').evaluate(e=>e.decode());await page.screenshot({path:path.join(out,key+'-'+theme+'.png')});
   }
  }
  await page.setViewportSize({width:960,height:760});
  await page.locator('[data-asset-id="m06-vowel-syntax-complex-light"]').scrollIntoViewIfNeeded();
  const layout=await page.locator('.manual-reading-scroll').evaluate(e=>({client:e.clientWidth,scroll:e.scrollWidth}));assert(layout.scroll<=layout.client+1);
  await page.screenshot({path:path.join(out,'narrow.png')});
  assert.equal(await page.locator('.manual-media-error,.manual-load-error').count(),0);assert.deepEqual(report.errors,[]);
  report.checks.push('Light/dark complex and fade figures displayed; 960×760 reading pane has no horizontal overflow or missing-media message');
  report.success=true;
 }catch(e){report.error=e.stack;throw e;}
 finally{
  await browser?.close();await server?.close();await write('report.json',report);marker.state='ready';marker.test_outcome=report.success?'completed':'failed';await write('run.json',marker);
  const inventory=async(dir,prefix='')=>{const rows=[];for(const e of await fs.readdir(dir,{withFileTypes:true})){const name=path.join(dir,e.name),rel=prefix+e.name;assert(!e.isSymbolicLink());if(e.isDirectory())rows.push(...await inventory(name,rel+'/'));else{const data=await fs.readFile(name);rows.push({path:rel,bytes:data.length,sha256:createHash('sha256').update(data).digest('hex')});}}return rows;};
  await write('scratch-manifest.json',{schema:'ptb-test-scratch-files/1',files:await inventory(scratch)});
  const cleanup=spawnSync('powershell.exe',['-NoProfile','-ExecutionPolicy','Bypass','-File',path.join(root,'scripts/cleanup_test_scratch.ps1'),'-RunDirectory',out],{encoding:'utf8',windowsHide:true});
  console.log(JSON.stringify({out,success:report.success,media:report.media.length,cleanup:cleanup.status}));assert.equal(cleanup.status,0,cleanup.stdout+cleanup.stderr);
 }
}
main().catch(e=>{console.error(e);process.exitCode=1;});
