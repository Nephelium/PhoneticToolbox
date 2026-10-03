// M04-D: independent Chrome, real TaskBridge/HTTP/MKL child. No EXE or DDL.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{spawn}=require('node:child_process'),{createInterface}=require('node:readline'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const worker=spawn(path.join(root,'.venv/m09-ui/Scripts/python.exe'),['-X','utf8','scripts/m04_ui_bridge.py'],{cwd:root,windowsHide:true,env:{...process.env,PYTHONPATH:path.join(root,'backend/src')+';'+path.join(root,'desktop/src')+';'+path.join(root,'packages/phonetic_core/src')}});
 const pending=new Map();let counter=0,resolveReady,rejectReady;const ready=new Promise((r,j)=>{resolveReady=r;rejectReady=j;});
 createInterface({input:worker.stdout}).on('line',line=>{const v=JSON.parse(line);if(v.ready)resolveReady(v);else{const done=pending.get(v.id);pending.delete(v.id);done?.(v);}});worker.stderr.on('data',d=>process.stderr.write(d));worker.on('error',rejectReady);worker.once('exit',code=>rejectReady(Error('bridge exited '+code)));
 const {out}=await ready;const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0},optimizeDeps:{entries:['tests/m04-live.html']},plugins:[{name:'owned-m04-rpc',configureServer(s){s.middlewares.use('/__m04',(req,res)=>{let body='';req.on('data',d=>body+=d);req.on('end',()=>{const data=JSON.parse(body),id=++counter;pending.set(id,v=>{res.setHeader('Content-Type','application/json');res.end(JSON.stringify(v));});worker.stdin.write(JSON.stringify({...data,rpc_id:id})+'\n');});});}}]});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});const context=await browser.newContext({viewport:{width:1440,height:1000}}),page=await context.newPage(),checks=[],errors=[],requests=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(r.url().endsWith('/__m04'))requests.push(r.postDataJSON());});
 const click=name=>page.getByRole('button',{name,exact:true}).first().click();const plotted=()=>page.locator('.lpc-spectrum:visible svg').waitFor({timeout:60000});
 const range=async(a,b)=>{const t=page.locator('.lpc-transport');await t.getByRole('spinbutton',{name:/^起点/}).fill(String(a));await t.getByRole('spinbutton',{name:/^终点/}).fill(String(b));await t.getByRole('spinbutton',{name:/^终点/}).blur();};
 const source=async name=>{await page.getByLabel('LPC 音频文件').selectOption({label:name});await page.locator('.lpc-page .wave-track svg').first().waitFor();await page.waitForFunction(()=>!document.querySelector('.lpc-files').innerText.includes('正在读取音频'));};

 const result={checks,errors,requests,layouts:[]};
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/m04-live.html');await click('LPC 谱图');await click('打开 WAV 目录');await source('LPC ɑ̃˥.wav');
  await page.getByLabel('显示语谱图（Praat）').check();
  for(const [w,h] of [[1920,1080],[1600,800],[1440,900],[1280,720]]){
   await page.setViewportSize({width:w,height:h});
   for(let i=0;i<4;i++){
    await new Promise(r=>setTimeout(r,600));
    result.layouts.push(await page.evaluate(()=>({width:innerWidth,height:innerHeight,loading:!!document.querySelector('.spectrogram-empty'),canvas:!!document.querySelector('.spectrogram-canvas canvas'),wv:document.querySelector('.wave-viewport')?.getBoundingClientRect().width,scroll:document.querySelector('.workbench-center')?.scrollHeight})));
   }
  }
  await range(.2,.4);await page.locator('.lpc-page .spectrogram-canvas canvas').waitFor({timeout:45000});
  await page.screenshot({path:path.join(out,'diagnostic.png'),fullPage:true});
 }catch(e){result.failure=String(e.stack);await page.screenshot({path:path.join(out,'diagnostic-failed.png')});}
 finally{
  await fs.writeFile(path.join(out,'diagnostic.json'),JSON.stringify(result,null,2));console.log(JSON.stringify({out,layouts:result.layouts,spectrogram:requests.filter(r=>r.op==='spectrogram').map(r=>r.view),errors,failure:result.failure}));
  await browser.close();await server.close();worker.stdin.write(JSON.stringify({op:'shutdown',rpc_id:++counter})+'\n');await new Promise(r=>worker.once('exit',r));
 }
}
main().catch(e=>{console.error(e);process.exitCode=1;});
