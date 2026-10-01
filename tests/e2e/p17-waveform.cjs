// Natural input only. Isolated Chrome and Vite belong to this test process.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const out=path.join(root,'output/validation/p17/shared-waveform',String(Date.now()));await fs.mkdir(out,{recursive:true});
 const inventory=JSON.parse(await fs.readFile(path.join(root,'output/validation/p17/natural-inventory.json'),'utf8'));
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0},optimizeDeps:{entries:['tests/p17-waveform.html']}});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--mute-audio']});
 const page=await browser.newPage({viewport:{width:1920,height:1000}});const errors=[],checks=[],measurements=[];
 page.on('pageerror',e=>errors.push(e.message));
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/p17-waveform.html');
  for(const kind of ['short','medium','egg']){
   const fixture=inventory.selected[kind];if(!fixture)continue;
   await page.getByLabel('真实录音').setInputFiles(fixture.path);await page.getByLabel('读取状态').filter({hasText:'ready'}).waitFor();
   assert(await page.getByLabel('波形振幅刻度').count(),'default shared waveform must show amplitude axis');
   const loadMs=await page.evaluate(()=>window.__p17.loadMs);
   // Find a natural voiced region from the real samples, without making test audio.
   const center=await page.evaluate(()=>{const a=window.__p17.state.asset,c=a.channels[0];let best=0;for(let i=0;i<c.length;i+=10)if(Math.abs(c[i])>Math.abs(c[best]))best=i;return best/a.sampleRate;});
   for(const seconds of [.16,.08,.04,.01]){
    await page.evaluate(({seconds,center})=>{const s=window.__p17.state;s.zoom=s.asset.duration/seconds;s.offset=Math.max(0,Math.min(center-seconds/2,s.asset.duration-seconds));},{seconds,center});
    await page.evaluate(()=>new Promise(requestAnimationFrame));
    const result=await page.evaluate(()=>{const s=window.__p17.state,a=s.asset,el=document.querySelector('.wave-line'),limit=+document.querySelector('.amplitude-axis').dataset.limit;let peak=0;for(let i=Math.floor(s.offset*a.sampleRate);i<Math.min(a.frames,Math.ceil((s.offset+a.duration/s.zoom)*a.sampleRate));i++)peak=Math.max(peak,Math.abs(a.channels[0][i]));return {mode:el.dataset.mode,limit,peak,commands:(el.getAttribute('d').match(/[MLV]/g)||[]),width:document.querySelector('.wave-track svg').clientWidth};});
    assert.equal(result.limit,result.peak||1,'visible-window amplitude must equal true peak');
    if(seconds*fixture.sample_rate<=result.width*16)assert.equal(result.mode,'samples','detail must connect original samples');
    if(result.mode==='samples'){assert.equal(result.commands.filter(x=>x==='M').length,1);assert(!result.commands.includes('V'));}
    measurements.push({kind,seconds,loadMs,...result,commands:result.commands.length});
   }
   await page.screenshot({path:path.join(out,kind+'-detail.png')});checks.push(kind+' natural detail and amplitude');
  }
  const timings=[];
  for(let i=0;i<24;i++){
   const box=await page.locator('.wave-track svg').first().boundingBox();await page.mouse.move(box.x+box.width*.5,box.y+box.height*.5);
   const oldZoom=await page.evaluate(()=>window.__p17.state.zoom);
   await page.keyboard.down('Control');const start=performance.now();await page.mouse.wheel(0,i%2?-120:120);
   await page.waitForFunction(old=>window.__p17.state.zoom!==old,oldZoom);await page.evaluate(()=>new Promise(requestAnimationFrame));timings.push(performance.now()-start);await page.keyboard.up('Control');
  }
  await page.setViewportSize({width:3840,height:2080});await page.evaluate(()=>{const s=window.__p17.state;s.zoom=s.asset.duration/(50000/s.asset.sampleRate);s.offset=0;});await page.evaluate(()=>new Promise(requestAnimationFrame));
  const large=await page.locator('.wave-line').first().evaluate(el=>({mode:el.dataset.mode,points:(el.getAttribute('d').match(/[ML]/g)||[]).length,plotWidth:el.closest('svg').clientWidth}));assert.equal(large.mode,'samples');assert(large.points>=50000&&large.points<=65536);assert(large.plotWidth>3000);checks.push('4K detail uses original 50000 samples at native viewport density');await page.screenshot({path:path.join(out,'large-detail.png')});
  await page.evaluate(()=>document.documentElement.dataset.theme='dark');await page.waitForTimeout(180);await page.screenshot({path:path.join(out,'dark.png')});
  await page.setViewportSize({width:800,height:650});await page.screenshot({path:path.join(out,'narrow.png')});
  assert.deepEqual(errors,[]);timings.sort((a,b)=>a-b);
  const report={checks,measurements,timings,median:timings[12],p95:timings[22],max:timings[23],errors};
  await fs.writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({out,...report},null,2));
 }catch(error){await page.screenshot({path:path.join(out,'failed.png')});throw error;}
 finally{await browser.close();await server.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
