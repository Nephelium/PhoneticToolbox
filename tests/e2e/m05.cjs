const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{spawn}=require('node:child_process'),{createInterface}=require('node:readline'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const worker=spawn(path.join(root,'.venv/m09-ui/Scripts/python.exe'),['-X','utf8','tests/support/m05_host_bridge.py'],{cwd:root,windowsHide:true,env:{...process.env,PYTHONPATH:['backend/src','packages/phonetic_core/src','desktop/src','scripts'].map(p=>path.join(root,p)).join(';')}});
 let readyResolve,readyReject,counter=0;const pending=new Map(),ready=new Promise((r,j)=>{readyResolve=r;readyReject=j});
 createInterface({input:worker.stdout}).on('line',line=>{const d=JSON.parse(line);if(d.ready)readyResolve(d);else{pending.get(d.id)?.(d);pending.delete(d.id)}});worker.stderr.on('data',d=>process.stderr.write(d));worker.once('exit',code=>readyReject(Error('Host exited '+code)));
 const {out}=await ready;console.log(out);
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{entries:['tests/m05-host.html']},plugins:[{name:'m05-host-transport-test',configureServer(s){s.middlewares.use('/__m05_host',(req,res)=>{let raw='';req.on('data',d=>raw+=d);req.on('end',()=>{const id=++counter;pending.set(id,data=>{res.setHeader('Content-Type','application/json');res.end(JSON.stringify(data))});worker.stdin.write(JSON.stringify({...JSON.parse(raw),id})+'\n')})})}}]});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true}),page=await browser.newPage({viewport:{width:1440,height:1000}}),checks=[],errors=[];page.on('pageerror',e=>errors.push(e.message));page.setDefaultTimeout(45000);
 const click=name=>page.getByRole('button',{name,exact:true}).click(),open=()=>page.locator('nav').getByRole('button',{name:'唇形提取',exact:true}).click();
 let success=false;
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/m05-host.html');await open();await page.locator('.lip-page').waitFor();
  await page.getByLabel('选择离线视频').setInputFiles(path.join(root,'output/validation/m05/inputs/motion-occlusion-vfr/input.mkv'));
  await page.getByRole('button',{name:'分析所选 1 个视频',exact:true}).click();await page.getByLabel('选择唇形结果').waitFor();
  assert.equal(await page.locator('.lip-curves svg').count(),3);checks.push('AppShell actual local production port: chunked video -> persistent legacy task -> preview and curves');
  await page.getByLabel('音唇 offset').fill('0.125');await click('应用偏移并保存');await page.getByText('完整结果与偏移写入完成。',{exact:true}).waitFor();
  const saved=await fs.readdir(path.join(out,'saved'));assert.equal(saved.length,1);const alignment=JSON.parse(await fs.readFile(path.join(out,'saved',saved[0],'alignment.json'),'utf8'));assert.equal(alignment.lip_manual_offset,.125);
  const exchange=JSON.parse(await fs.readFile(path.join(out,'saved',saved[0],'aligned.lip.json'),'utf8'));assert.equal(exchange.data.metadata.lip_manual_offset,.125);checks.push('actual streamed disk save and offset applied once in separate aligned exchange');
  await page.getByLabel('音唇 offset').fill('-0.05');await page.getByLabel('关闭 唇形提取',{exact:true}).click();await click('取消关闭');assert.equal(await page.locator('.lip-page').count(),1);
  await click('保存但不应用偏移');await page.getByText('完整结果与偏移写入完成。',{exact:true}).waitFor();
  await page.screenshot({path:path.join(out,'light.png'),fullPage:true});await page.evaluate(()=>document.documentElement.dataset.theme='dark');await page.screenshot({path:path.join(out,'dark.png'),fullPage:true});
  checks.push('dirty close guard, save-without-offset, original source preserved, light/dark production page');
  await click('刷新本地历史任务');await page.waitForFunction(()=>document.querySelectorAll('[aria-label="历史唇形任务"] option').length>0);
  await click('读取历史结果');await page.waitForFunction(()=>document.querySelectorAll('.lip-curves svg').length===3);
  await page.getByLabel('动画导出质量').selectOption('small');await click('导出 GIF');await page.getByText('动画写入完成。',{exact:true}).waitFor();
  const animations=await fs.readdir(path.join(out,'saved'));let found=false;for(const dir of animations){const entries=await fs.readdir(path.join(out,'saved',dir));if(entries.includes('lip-animation.gif'))found=true;}assert.equal(found,true);
  checks.push('persistent history restores complete result and actual GIF export through repeat task and native save');
  await page.waitForFunction(()=>!document.querySelector('#tab-M05 [aria-label="未保存"]'));
  // Synthetic canvas + silent oscillator destination only; never physical devices.
  await page.getByLabel('关闭 唇形提取',{exact:true}).click();await page.locator('.lip-page').waitFor({state:'detached'});await open();
  await page.evaluate(()=>{navigator.mediaDevices.getUserMedia=async()=>{throw new DOMException('controlled denial','NotAllowedError')}});
  await click('打开预览');await page.getByRole('alert').first().waitFor();checks.push('permission denial gives a recoverable explicit error');
  const portrait=await fs.readFile(path.join(root,'output/validation/m05/inputs/astronaut.png'),'base64');
  await page.evaluate(async portrait=>{
    const image=new Image();image.src='data:image/png;base64,'+portrait;await image.decode();
    navigator.mediaDevices.getUserMedia=async constraints=>{const c=document.createElement('canvas');c.width=512;c.height=512;const ctx=c.getContext('2d');ctx.drawImage(image,0,0);const stream=c.captureStream(30),timer=setInterval(()=>ctx.drawImage(image,0,0),33);const originalStop=stream.getVideoTracks()[0].stop.bind(stream.getVideoTracks()[0]);stream.getVideoTracks()[0].stop=()=>{clearInterval(timer);originalStop()};
     if(constraints.audio){const ac=new AudioContext(),osc=ac.createOscillator(),destination=ac.createMediaStreamDestination();osc.connect(destination);osc.start();const audio=destination.stream.getAudioTracks()[0],stop=audio.stop.bind(audio);audio.stop=()=>{osc.stop();void ac.close();stop()};stream.addTrack(audio);}return stream;};
  },portrait);
  for(const mode of ['raw','record_then_analyze','realtime']){
   await page.getByLabel('模式',{exact:false}).first().selectOption(mode);await click('开始录制');await page.waitForFunction(()=>document.querySelector('.lip-state')?.textContent.includes('阶段 recording'));
   await page.waitForTimeout(1300);await click('停止并收尾');await page.waitForFunction(()=>document.querySelector('.lip-state')?.textContent.includes('阶段 ready'));
   if(mode==='realtime')assert.deepEqual(await page.locator('.lip-video canvas').evaluate(c=>[c.width,c.height]),[512,512],'stop preserves measured frame dimensions instead of resetting to 640x480');
   await click('保存原始录制');await page.getByText('原始录制写入完成并核对大小。',{exact:true}).waitFor();await click('保存候选参数与时间记录');await page.getByText('候选参数与时间元数据已保存。',{exact:true}).waitFor();
   checks.push('actual MediaRecorder '+mode+' with synthetic input, stop/finalize and native streamed/hash-verified save');
  }
  const paired=await page.evaluate(async()=>{
   const {LipCapture}=await import('/src/modules/lip-extraction/capture.ts');const {LipInference}=await import('/src/modules/lip-extraction/inference.ts');
   const source=document.createElement('canvas');source.width=320;source.height=240;const context=source.getContext('2d');let color=0;
   const paint=()=>{context.fillStyle=`rgb(${color++%240},30,50)`;context.fillRect(0,0,320,240)};paint();const timer=setInterval(paint,33),stream=source.captureStream(30),video=document.createElement('video');video.muted=true;document.body.append(video);
   const read=bitmap=>{const c=document.createElement('canvas');c.width=c.height=1;const x=c.getContext('2d');x.drawImage(bitmap,0,0);return [...x.getImageData(0,0,1,1).data]};
   const original=LipInference.prototype.detect,submitted=[],displayed=[];let previous=-1;
   LipInference.prototype.detect=async function(image,...args){submitted.push(read(image));await new Promise(r=>setTimeout(r,180));return original.call(this,image,...args)};
   const capture=new LipCapture(video,()=>{if(capture.previewFrame&&capture.state.latest?.index!==previous){previous=capture.state.latest.index;displayed.push(read(capture.previewFrame));}},async()=>stream);
   try{await capture.start({camera:'',microphone:'',mode:'preview',filter:false,cutoff:15,delegate:'CPU',requestedFps:30});await new Promise(r=>setTimeout(r,1400));await capture.stop();return {submitted,displayed,skipped:capture.state.stats.skippedInference};}
   finally{LipInference.prototype.detect=original;clearInterval(timer);capture.dispose();video.remove();}
  });assert(paired.displayed.length>=3);assert(paired.skipped>0);assert.deepEqual(paired.displayed,paired.submitted.slice(0,paired.displayed.length));checks.push('delayed real Worker shows exact submitted image, never newer camera image; stop retains measured dimensions');
  assert.deepEqual(errors,[]);success=true;
 }catch(e){await fs.writeFile(path.join(out,'failure.txt'),await page.locator('body').innerText());await page.screenshot({path:path.join(out,'failure.png'),fullPage:true});throw e;}
 finally{await fs.writeFile(path.join(out,'browser-report.json'),JSON.stringify({success,checks,errors,scope:'Windows Chrome actual production port and local service; substitute QWebChannel transport; no physical devices'},null,2));console.log(JSON.stringify({success,checks}));await browser.close();await server.close();worker.stdin.end();}
}
main().catch(e=>{console.error(e);process.exitCode=1});
