const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{spawn}=require('node:child_process'),{createInterface}=require('node:readline'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const worker=spawn(path.join(root,'.venv/m09-ui/Scripts/python.exe'),['-X','utf8','tests/support/m05_host_bridge.py'],{cwd:root,windowsHide:true,env:{...process.env,PYTHONPATH:['backend/src','packages/phonetic_core/src','desktop/src'].map(p=>path.join(root,p)).join(';')}});
 let readyResolve,readyReject,counter=0;const pending=new Map(),ready=new Promise((r,j)=>{readyResolve=r;readyReject=j});
 createInterface({input:worker.stdout}).on('line',line=>{const d=JSON.parse(line);if(d.ready)readyResolve(d);else{pending.get(d.id)?.(d);pending.delete(d.id)}});worker.stderr.on('data',d=>process.stderr.write(d));worker.once('exit',code=>readyReject(Error('Host exited '+code)));
 const {out}=await ready;console.log(out);const operations=[];
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{entries:['tests/m05-host.html']},plugins:[{name:'m05-r2-test-transport',configureServer(s){s.middlewares.use('/__m05_host',(req,res)=>{let raw='';req.on('data',d=>raw+=d);req.on('end',()=>{const id=++counter,request=JSON.parse(raw);operations.push(request.body.op);if(request.body.op==='choose'&&cancelNext){cancelNext=false;res.end(JSON.stringify({ok:true,value:null}));return;}pending.set(id,data=>{res.setHeader('Content-Type','application/json');res.end(JSON.stringify(data))});worker.stdin.write(JSON.stringify({...request,id})+'\n')})})}}]});await server.listen();
 let cancelNext=false;
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true}),page=await browser.newPage({viewport:{width:1920,height:1080}}),report={success:false,physical_devices:false,checks:[],errors:[]};page.on('pageerror',e=>report.errors.push(e.message));page.setDefaultTimeout(45000);
 const click=name=>page.getByRole('button',{name,exact:true}).click(),phase=p=>page.waitForFunction(p=>document.querySelector('.lip-state')?.textContent.includes('阶段 '+p),p),mode=m=>page.getByLabel('模式',{exact:false}).first().selectOption(m);
 const saveName='保存录制（视频 + 音频 + 唇形）',anotherName='另存录制（视频 + 音频 + 唇形）';
 const dirs=async()=> (await fs.readdir(path.join(out,'saved'))).filter(n=>n.startsWith('M05-recording-'));
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/m05-host.html');await page.locator('nav').getByRole('button',{name:'唇形提取',exact:true}).click();
  await page.addScriptTag({path:path.join(root,'tests/support/m05_r2_capture.js')});
  const portrait=await fs.readFile(path.join(root,'output/validation/m05/inputs/astronaut.png'),'base64');await page.evaluate(p=>window.installM05R2Capture('data:image/png;base64,'+p),portrait);
  await mode('realtime');await click('开始录制');await phase('recording');
  await page.waitForFunction(()=>Number(document.querySelector('meter')?.value)>-40);
  await page.waitForFunction(()=>Number(document.querySelector('.lip-state')?.textContent.match(/检出 (\d+)/)?.[1])>5);
  await page.waitForTimeout(1300);
  const requests=await page.evaluate(()=>window.m05R2.requests);assert.equal(requests.length,2);assert.equal(requests[1].video,false);assert.equal(requests[1].audio.deviceId.exact,'mic');
  assert((await page.locator('.lip-input-level').innerText()).includes('Microphone USB'));report.checks.push('hidden labels: permission then audio-only switch from default stereo mix to actual microphone, live level');
  await click('停止并收尾');await phase('ready');await page.getByLabel('刚录制的视频与声音').waitFor();assert.equal(await page.getByRole('button',{name:saveName,exact:true}).isEnabled(),true);
  cancelNext=true;await click(saveName);await page.getByText('已取消保存，录制保留。',{exact:true}).waitFor();assert.equal((await dirs()).length,0);assert(await page.locator('#tab-M05 [aria-label="未保存"]').count());
  await click(saveName);await page.getByText('MP4、WAV 与采集记录已保存并核对。',{exact:true}).waitFor();
  const first=(await dirs())[0],directory=path.join(out,'saved',first),info=JSON.parse(await fs.readFile(path.join(directory,'recording-export.json'),'utf8'));
  assert.equal(operations.filter(op=>op==='m05_create').length,0);assert.equal(info.audio.present,true);assert(info.audio.signal.peak_dbfs>-30);assert.equal(info.audio.signal.low_signal,false);assert.equal(info.associated_exchange,'audio_recording.lip.json');assert(info.candidate_frames>5);
  const capture=JSON.parse(await fs.readFile(path.join(directory,'capture.m05-preview.json'),'utf8'));assert.equal(capture.audio_input.deviceId,'mic');assert.equal(capture.audio_input.label,'Microphone USB (QA)');
  const exchange=JSON.parse(await fs.readFile(path.join(directory,'audio_recording.lip.json'),'utf8'));assert.equal(exchange.data.metadata.source_status,'candidate');assert.equal(exchange.data.relative_times.values.length,info.candidate_frames);
  assert.equal(await page.locator('#tab-M05 [aria-label="未保存"]').count(),0);await click(anotherName);await page.waitForFunction(()=>!document.querySelector('.lip-page')?.getAttribute('aria-busy')?.includes('true'));assert.equal((await dirs()).length,2);
  report.checks.push('stop -> cancel/retry -> real MP4/WAV/auto-associated candidate save -> save another copy; zero analysis tasks');report.first_directory=directory;report.first_export=info;
  await click('可选：按 V2 模型重新分析');await page.getByText('正式分析完成，结果等待保存。',{exact:true}).waitFor();
  const slider=page.getByLabel('回放帧',{exact:true});await slider.fill(String(Math.floor(Number(await slider.getAttribute('max'))/2)));await slider.dispatchEvent('input');
  await page.waitForFunction(()=>{const c=document.querySelector('.lip-video canvas');return c&&c.getContext('2d').getImageData(0,0,c.width,c.height).data.some((v,i)=>i%4===3&&v>0)});
  const pixels=await page.locator('.lip-video canvas').evaluate(c=>{const d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;let left=c.width,top=c.height,right=0,bottom=0;for(let y=0;y<c.height;y++)for(let x=0;x<c.width;x++){let i=(y*c.width+x)*4;if(d[i+1]>180&&d[i]<80&&d[i+2]<80&&d[i+3]>0){left=Math.min(left,x);top=Math.min(top,y);right=Math.max(right,x);bottom=Math.max(bottom,y)}}return {left,top,right,bottom,width:c.width,height:c.height}});
  assert((pixels.bottom-pixels.top)/pixels.height>.75,JSON.stringify(pixels));assert(pixels.top>10&&pixels.bottom<pixels.height-10);assert(Math.abs((pixels.left+pixels.right)/2-pixels.width/2)<10);report.checks.push({enlarged_animation:pixels});
  for(const theme of ['light','dark']){await page.evaluate(t=>{document.documentElement.dataset.theme=t;document.documentElement.style.colorScheme=t},theme);await page.waitForTimeout(250);await page.screenshot({path:path.join(out,'m05-r2-'+theme+'.png'),fullPage:true});}
  await click('保存但不应用偏移');await page.getByText('完整结果与偏移写入完成。',{exact:true}).waitFor();
  await mode('raw');await click('开始录制');await phase('recording');await page.waitForTimeout(800);await click('停止并收尾');await phase('ready');
  await click('保存并开始下一次');await phase('recording');await page.waitForTimeout(800);await click('停止并收尾');await phase('ready');await click(saveName);await page.getByText('MP4、WAV 与采集记录已保存并核对。',{exact:true}).waitFor();report.checks.push('same-page save-and-start-next and raw recording');
  await page.getByLabel('麦克风',{exact:true}).selectOption('mix');await click('开始录制');await phase('recording');await page.waitForTimeout(350);assert((await page.locator('.lip-input-warning').innerText()).includes('立体声混音'));await click('停止并收尾');await phase('ready');await click(saveName);await page.getByText('MP4、WAV 与采集记录已保存并核对。',{exact:true}).waitFor();report.checks.push('explicit stereo mix remains selected with actionable warning');
  await page.getByLabel('麦克风',{exact:true}).selectOption('mic');await page.evaluate(()=>{window.m05R2.low=true});await click('开始录制');await phase('recording');await page.waitForFunction(()=>document.querySelector('.lip-input-warning')?.textContent.includes('输入信号很低'));await click('停止并收尾');await phase('ready');await click(saveName);await page.getByText('MP4、WAV 与采集记录已保存并核对。',{exact:true}).waitFor();
  report.checks.push('near-silent track triggers live low-signal warning and decoded WAV audit');
  await page.evaluate(()=>{window.m05R2.missing=true});await click('开始录制');await phase('failed');assert((await page.locator('[role="alert"]').first().innerText()).includes('音轨'));assert.equal(await page.getByRole('button',{name:saveName,exact:true}).isEnabled(),false);
  await page.evaluate(()=>{window.m05R2.missing=false;window.m05R2.deny=true});await click('开始录制');await phase('failed');assert((await page.locator('[role="alert"]').first().innerText()).includes('权限'));
  const tracks=await page.evaluate(()=>({created:window.m05R2.created,ended:window.m05R2.ended}));assert.equal(tracks.created,tracks.ended);report.checks.push({missing_track_and_denial_release_all_owned_tracks:tracks});
  report.recordings=await dirs();assert.deepEqual(report.errors,[]);report.success=true;
 }finally{await page.screenshot({path:path.join(out,'m05-r2-final.png'),fullPage:true}).catch(()=>{});report.final_text=await page.locator('body').innerText().catch(()=>'');await fs.writeFile(path.join(out,'m05-r2-report.json'),JSON.stringify(report,null,2));await browser.close();await server.close();worker.stdin.end();}
 console.log(JSON.stringify({out,success:report.success,checks:report.checks},null,2));
}
main().catch(e=>{console.error(e);process.exitCode=1});
