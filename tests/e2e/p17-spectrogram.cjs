const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{spawn}=require('node:child_process'),readline=require('node:readline'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const out=path.join(root,'output/validation/p17/shared-spectrogram',String(Date.now()));await fs.mkdir(out,{recursive:true});
 const worker=spawn(path.join(root,'.venv/m14/Scripts/python.exe'),['-X','utf8','scripts/p17_spectrogram_bridge.py'],{cwd:root,windowsHide:true,stdio:['pipe','pipe','inherit']});
 let id=0;const pending=new Map();readline.createInterface({input:worker.stdout}).on('line',line=>{const r=JSON.parse(line);pending.get(r.id)?.(r);pending.delete(r.id);});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0},optimizeDeps:{entries:['tests/p17-waveform.html']},plugins:[{name:'p17-natural-preview',configureServer(s){s.middlewares.use('/__p17spec',(req,res)=>{let body='';req.on('data',c=>body+=c);req.on('end',async()=>{const input=JSON.parse(body),current=++id;const result=await new Promise(resolve=>{pending.set(current,resolve);worker.stdin.write(JSON.stringify({...input,id:current})+'\n');});res.setHeader('Content-Type','application/json');res.end(JSON.stringify(result));});});}}]});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--mute-audio']});const page=await browser.newPage({viewport:{width:1920,height:1000}});
 const manifest=JSON.parse(await fs.readFile(path.join(root,'output/validation/p17/natural-inventory.json'),'utf8')),errors=[],timings=[],checks=[];page.on('pageerror',e=>errors.push(e.message));
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/p17-waveform.html?spectrogram=medium');await page.getByLabel('真实录音').setInputFiles(manifest.selected.medium.path);
  await page.locator('.spectrogram-canvas canvas').waitFor();
  const geometry=await page.evaluate(()=>['.wave-track svg','.spectrogram-canvas canvas'].map(s=>{const r=document.querySelector(s).getBoundingClientRect();return {left:r.left,right:r.right};}));
  assert(Math.abs(geometry[0].left-geometry[1].left)<=2&&Math.abs(geometry[0].right-geometry[1].right)<=2,'time areas must align');checks.push('waveform and spectrum share horizontal time geometry');
  await page.evaluate(()=>{const s=window.__p17.state;s.zoom=s.asset.duration/.5;s.offset=2;});await page.locator('.spectrogram-canvas canvas').waitFor();
  for(let i=0;i<20;i++){
   const before=await page.locator('.spectrogram-canvas canvas').getAttribute('aria-label');const b=await page.locator('.spectrogram-canvas canvas').boundingBox();
   await page.mouse.move(b.x+b.width*.5,b.y+b.height*.5);await page.keyboard.down('Control');const t=performance.now();await page.mouse.wheel(0,i%2?120:-120);await page.keyboard.up('Control');
   await page.waitForFunction(old=>{const e=document.querySelector('.spectrogram-canvas canvas');return e&&e.getAttribute('aria-label')!==old&&!document.querySelector('.spectrogram-empty');},before);
   timings.push(performance.now()-t);
  }
  checks.push('20 spectrum wheel gestures update shared waveform and final real pixels');
  const b=await page.locator('.spectrogram-canvas canvas').boundingBox();await page.mouse.move(b.x+b.width*.2,b.y+b.height*.5);await page.mouse.down();await page.mouse.move(b.x+b.width*.4,b.y+b.height*.5,{steps:5});await page.mouse.up();
  assert(await page.locator('.wave-selection').evaluate(e=>+e.getAttribute('width')>0));checks.push('spectrum drag updates common selection');
  await page.screenshot({path:path.join(out,'light.png')});await page.evaluate(()=>document.documentElement.dataset.theme='dark');await page.waitForTimeout(200);await page.screenshot({path:path.join(out,'dark.png')});
  await page.setViewportSize({width:800,height:650});await page.locator('.spectrogram-canvas canvas').waitFor();await page.screenshot({path:path.join(out,'narrow.png')});
  assert.deepEqual(errors,[]);const sorted=[...timings].sort((a,b)=>a-b),report={checks,timings,p95:sorted[18],maximum:sorted[19],geometry,errors};
  await fs.writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({out,...report},null,2));
 }catch(e){await page.screenshot({path:path.join(out,'failed.png')});throw e;}
 finally{await browser.close();await server.close();worker.stdin.write('{"op":"shutdown"}\n');await new Promise(resolve=>worker.once('exit',resolve));}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
