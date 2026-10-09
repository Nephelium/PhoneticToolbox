// M08-R1: real desktop adapter, persistent service and Praat; muted owned Chrome.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{spawn}=require('node:child_process'),{createInterface}=require('node:readline'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const worker=spawn(path.join(root,'.venv/m09-ui/Scripts/python.exe'),['-X','utf8','tests/support/m08_host_bridge.py'],{cwd:root,windowsHide:true,env:{...process.env,PYTHONPATH:['backend/src','packages/phonetic_core/src','desktop/src','scripts'].map(p=>path.join(root,p)).join(';')}});
 let readyResolve,readyReject,counter=0;const pending=new Map(),ready=new Promise((r,j)=>{readyResolve=r;readyReject=j});
 createInterface({input:worker.stdout}).on('line',line=>{const d=JSON.parse(line);if(d.ready)readyResolve(d);else{pending.get(d.id)?.(d);pending.delete(d.id)}});worker.stderr.on('data',d=>process.stderr.write(d));worker.once('exit',code=>readyReject(Error('Host exited '+code)));
 const {out}=await ready;console.log(out);const requests=[];
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{entries:['tests/m08-host.html']},plugins:[{name:'m08-r1-host',configureServer(s){s.middlewares.use('/__m08_host',(req,res)=>{let raw='';req.on('data',d=>raw+=d);req.on('end',()=>{const id=++counter,message=JSON.parse(raw);requests.push(message);pending.set(id,data=>{res.setHeader('Content-Type','application/json');res.end(JSON.stringify(data))});worker.stdin.write(JSON.stringify({...message,id})+'\n')})})}}]});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--mute-audio']}),page=await browser.newPage({viewport:{width:1800,height:1100}}),checks=[],layouts=[],errors=[];
 page.on('pageerror',e=>errors.push(e.message));page.setDefaultTimeout(45000);
 await page.addInitScript(()=>{window.starts=[];const create=AudioContext.prototype.createBufferSource;AudioContext.prototype.createBufferSource=function(){const node=create.call(this),start=node.start;node.start=function(when,offset,duration){window.starts.push({offset,duration,frames:this.buffer.length,rate:this.buffer.sampleRate,sample:this.buffer.getChannelData(0)[100]});return start.call(this,when,offset,duration);};return node;};});
 const click=name=>page.getByRole('button',{name,exact:true}).click(),idle=()=>page.waitForFunction(()=>document.querySelector('.m08-page')?.getAttribute('aria-busy')==='false');
 const count=n=>page.waitForFunction(n=>document.querySelectorAll('.history li').length===n,n);
 const paths=()=>page.locator('.history-plot>svg>path').evaluateAll(es=>es.map(e=>e.getAttribute('d')));
 const play=async name=>{const n=await page.evaluate(()=>starts.length);await click(name);await page.waitForFunction(n=>starts.length>n,n);return page.evaluate(()=>starts.at(-1));};
 async function drag(index,a,b){const el=page.locator('.wave-viewport').nth(index).locator('.wave-track>svg').first();await el.scrollIntoViewIfNeeded();const box=await el.boundingBox();await page.mouse.move(box.x+a*box.width,box.y+box.height*.5);await page.mouse.down();await page.mouse.move(box.x+b*box.width,box.y+box.height*.5,{steps:8});await page.mouse.up();}
 let success=false;
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/m08-host.html');await page.locator('nav').getByRole('button',{name:'变速变调',exact:true}).click();await click('打开音频目录');
  const select=page.getByLabel('M08 音频');await page.waitForFunction(()=>document.querySelector('[aria-label="M08 音频"]')?.options.length>1);
  const options=await select.locator('option').evaluateAll(es=>es.map(e=>({value:e.value,text:e.textContent})));await select.selectOption(options.find(o=>o.text.endsWith('.wav')).value);await idle();await page.locator('svg.m08-curve').waitFor();
  const before=await page.locator('.m08-curve .modified').getAttribute('d'),curve=await page.locator('.m08-curve').boundingBox();
  await page.keyboard.down('Shift');await page.mouse.move(curve.x+curve.width*.25,curve.y+curve.height*.35);await page.mouse.down();await page.mouse.move(curve.x+curve.width*.8,curve.y+curve.height*.55,{steps:24});await page.mouse.up();await page.keyboard.up('Shift');assert.notEqual(await page.locator('.m08-curve .modified').getAttribute('d'),before);
  await page.getByLabel('语速倍率').fill('0.8');await click('合成当前视野');await page.getByText('输出 1.250 s',{exact:false}).waitFor();await idle();await count(1);assert((await paths())[0].length>50);checks.push('hand-drawn F0 durable synthesis enters history without saving');
  assert(await page.getByText('批量改变基频与拐点',{exact:true}).evaluate(e=>e.parentElement.open));
  for(const name of ['保存并编号','添加拐点','清除拐点','批量生成并保存'])assert.equal(await page.getByRole('button',{name,exact:true}).count(),0);checks.push('expanded batch editor and removed obsolete buttons');
  await drag(0,.2,.6);const selection=()=>page.locator('.wave-selection').evaluateAll(es=>es.map(e=>[e.getAttribute('x'),e.getAttribute('width')]));const originalSelection=await selection();
  let started=await play('播放当前视野原音');assert(Math.abs(started.offset)<1e-6&&Math.abs(started.duration-1)<.002);assert.deepEqual(await selection(),originalSelection);
  started=await play('播放选区');assert(Math.abs(started.offset-.2)<.006&&Math.abs(started.duration-.4)<.006);await click('停止');
  started=await play('播放合成音');assert.equal(started.offset,0);assert(Math.abs(started.duration-1.25)<.002);assert.deepEqual(await selection(),originalSelection);
  started=await play('播放选区');assert(Math.abs(started.offset-.2)<.006&&Math.abs(started.duration-.4)<.006);await click('停止');
  await drag(1,.3,.7);started=await play('播放当前视野原音');assert.equal(started.offset,0);started=await play('播放选区');assert(Math.abs(started.offset-.375)<.006&&Math.abs(started.duration-.5)<.006);await click('停止');checks.push('real WebAudio direct source/result and mouse selection playback independent');
  started=await play('试听此版本');assert.equal(started.offset,0);assert(Math.abs(started.duration-1.25)<.002);await click('停止');checks.push('audition version loads and plays actual audio');
  await page.locator('.wave-viewport').first().getByRole('button',{name:'放大波形',exact:true}).click();const visible=await page.locator('.workbench-right .row>strong').first().innerText();const bounds=visible.match(/([0-9.]+)–([0-9.]+)/).slice(1).map(Number);started=await play('播放当前视野原音');assert(Math.abs(started.offset-bounds[0])<.002&&Math.abs(started.duration-(bounds[1]-bounds[0]))<.002);await click('停止');await page.locator('.wave-viewport').first().getByRole('button',{name:'适合窗口',exact:true}).click();checks.push('direct original playback follows the zoomed view independently of selection');
  await click('编辑拐点表');await page.getByLabel('频率 0',{exact:true}).fill('120,180');await page.getByLabel('频率 1',{exact:true}).fill('140,200');await page.getByLabel('连接 0',{exact:true}).selectOption('full');await page.getByLabel('连接 1',{exact:true}).selectOption('full');await click('保存更改');await click('批量生成');await count(5);await idle();assert.equal(await fs.readdir(path.join(out,'saved')).then(a=>a.length),0);assert.equal((await paths()).filter(p=>p.length>50).length,5);checks.push('four real linear combinations generated with no external save');
  const control=body=>page.evaluate(async body=>(await fetch('/__m08_host',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({channel:'test',body})})).json(),body);
  await control({output_mode:'cancel'});await click('批量保存');await click('保存所选音频');await idle();await page.getByText('已取消保存，生成结果保留',{exact:true}).waitFor();assert.equal((await fs.readdir(path.join(out,'saved'))).length,0);await click('取消');await control({});checks.push('native directory cancellation keeps results and produces no files');
  const historyRequest=requests.findLast(r=>r.body.action==='history');
  const historyData=await page.evaluate(async body=>(await(await fetch('/__m08_host',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({channel:'task',body})})).json()).value,historyRequest.body);
  const failedId=historyData[0].id;
  // Exercise web ZIP fallback against the same real generated WAV result reads.
  const zipDownload=page.waitForEvent('download');
  await page.evaluate(async values=>{
   const {m08Port}=await import('/src/platform/m08.ts');
   const source=values[0].source_id;const web=m08Port({project:'fixture',source:async()=>({asset_id:source,sha256:'fixture'}),request:async action=>{if(action==='history')return values;throw Error('unexpected mutation '+action)},job:async()=>{throw Error('unexpected job')},cancel:async()=>{},read:async(job,id)=>{const r=await(await fetch('/__m08_host',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({channel:'task',body:{op:'result',job,id}})})).json();if(!r.ok)throw Error(r.error);return Uint8Array.from(atob(r.value.base64),c=>c.charCodeAt(0)).buffer;}});
   const results=await web.history({id:'web-input',name:'input.wav',kind:'audio',size:1});await web.saveMany(results);
  },historyData);
  await(await zipDownload).saveAs(path.join(out,'m08-web-results.zip'));checks.push('web adapter downloads one ZIP containing actual generated WAVs');
  await control({fail_export:failedId});
  // A slow version load must not interrupt a later direct playback request.
  let release,continued;const gate=new Promise(resolve=>release=resolve),finished=new Promise(resolve=>continued=resolve);let delayed=false;
  await page.route('**/__m08_host',async route=>{const payload=route.request().postDataJSON();if(payload.body.op==='result'&&payload.body.id===failedId){delayed=true;await gate;await route.continue();continued();}else await route.continue();});
  await page.getByRole('button',{name:'试听此版本',exact:true}).first().click();for(let i=0;i<50&&!delayed;i++)await new Promise(r=>setTimeout(r,10));assert(delayed);
  started=await play('播放当前视野原音');const startsBefore=await page.evaluate(()=>starts.length);release();await finished;await page.unroute('**/__m08_host');await page.waitForTimeout(350);assert.equal(await page.evaluate(()=>starts.length),startsBefore);assert.equal(started.frames,16000);await click('停止');checks.push('late version response cannot override a later direct playback');
  const jobsBefore=requests.filter(r=>r.body.action==='create'||r.body.action==='save').length;
  await click('批量保存');await page.getByRole('dialog').waitFor();await click('保存所选音频');await idle();await page.getByText('已保存 4 个音频，1 个未保存',{exact:true}).waitFor();assert.equal(await page.getByRole('dialog').locator('input:checked').count(),1);assert.equal((await fs.readdir(path.join(out,'saved'))).length,4);
  await control({});await click('保存所选音频');await idle();await page.getByText('已保存 1 个音频',{exact:true}).waitFor();checks.push('partial save reports 4 successes and 1 failure, retries only the failed ID');const saved=await fs.readdir(path.join(out,'saved'));assert.equal(saved.length,5);await count(5);assert.equal(requests.filter(r=>r.body.action==='create'||r.body.action==='save').length,jobsBefore);checks.push('one directory choice exports five real WAVs without duplicate history or compute');
  await click('批量保存');await click('保存所选音频');await idle();assert.deepEqual(await fs.readdir(path.join(out,'saved')),saved);checks.push('repeat export reuses unchanged files');
  // Reload verifies persisted history without copying or silently recomputing.
  await click('保存编辑草稿');await page.reload();await page.locator('nav').getByRole('button',{name:'变速变调',exact:true}).click();await click('打开音频目录');await page.getByLabel('M08 音频').selectOption(options.find(o=>o.text.endsWith('.wav')).value);await idle();await count(5);checks.push('all generated histories survive page reload');
  const audition=page.getByRole('button',{name:'试听此版本',exact:true}).last();const auditionCount=await page.evaluate(()=>starts.length);await audition.click();await page.waitForFunction(n=>starts.length>n,auditionCount);await page.getByText('输出 1.000 s',{exact:false}).waitFor();await click('停止');
  const download=page.waitForEvent('download');await click('保存对比图');await(await download).saveAs(path.join(out,'comparison.png'));checks.push('history PNG exported after reload');
  for(const [width,height]of [[1800,1100],[1280,800],[900,760]]){await page.setViewportSize({width,height});await page.screenshot({path:path.join(out,`m08-r1-${width}.png`)});const box=await page.locator('.m08-page').evaluate(e=>({client:e.clientWidth,scroll:e.scrollWidth}));assert(box.scroll<=box.client+2,JSON.stringify(box));layouts.push({width,height,...box});}
  await page.setViewportSize({width:1800,height:1100});await page.evaluate(()=>document.documentElement.dataset.theme='dark');await page.waitForTimeout(300);await page.screenshot({path:path.join(out,'m08-r1-dark.png')});assert.deepEqual(errors,[]);
  await click('批量重命名');await page.getByLabel('新的共同前缀').fill('M08_R1_');await click('确认重命名');await idle();assert((await fs.readdir(path.join(out,'saved'))).every(n=>n.startsWith('M08_R1_')));await count(5);
  await click('删除本批次音频');await click('确认删除所选音频');await idle();await count(0);assert.equal((await fs.readdir(path.join(out,'saved'))).length,0);await page.waitForTimeout(1300);assert.equal(await page.locator('.history li').count(),0);assert(await page.getByRole('button',{name:'播放合成音',exact:true}).isDisabled());assert(!(await page.locator('.m08-page').innerText()).includes('任务操作失败'));assert((await fs.stat(path.join(out,'input','public ɑ̃˥.wav'))).size>1000);checks.push('explicit batch rename/delete clears current result and history after polling, original retained');success=true;
 }catch(e){await page.screenshot({path:path.join(out,'m08-r1-failed.png')});throw e;}
 finally{await fs.writeFile(path.join(out,'m08-r1-report.json'),JSON.stringify({success,checks,layouts,errors,requests},null,2));await browser.close();await server.close();worker.stdin.end();console.log(JSON.stringify({out,success,checks}));}
}
main().catch(e=>{console.error(e);process.exitCode=1});
