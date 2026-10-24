// Real native exports and browser downloads from the same six M07 result snapshots.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {spawn}=require('node:child_process'),{createInterface}=require('node:readline'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const worker=spawn(path.join(root,'.venv/m14/Scripts/python.exe'),['-E','-s','-B','-X','utf8','scripts/verify_m07_save_names.py'],{cwd:root,windowsHide:true});
 let counter=0,readyResolve,readyReject;const pending=new Map(),ready=new Promise((r,j)=>{readyResolve=r;readyReject=j;});
 const exited=new Promise(resolve=>worker.once('exit',resolve));
 createInterface({input:worker.stdout}).on('line',line=>{try{const data=JSON.parse(line);if(data.ready)readyResolve(data);else if(pending.has(data.id)){pending.get(data.id)(data);pending.delete(data.id);}}catch{process.stderr.write(line+'\n');}});
 worker.stderr.on('data',data=>process.stderr.write(data));worker.once('exit',code=>readyReject(Error('Export host exited '+code)));
 const rpc=(channel,body)=>new Promise((resolve,reject)=>{const id=++counter;pending.set(id,r=>r.ok?resolve(r.value):reject(Error(r.error)));worker.stdin.write(JSON.stringify({id,channel,body})+'\n');});
 let server,browser,out,report={success:false,checks:[],scope:'Windows Chrome with production desktop.ts/M07Port, real FileProvider/TaskBridge/LocalService/Praat; web ZIP save on the same actual results; no macOS, native Qt or final package'};
 try{
  const info=await ready;out=info.out;console.log(out);await fs.mkdir(path.join(info.scratch,'downloads'));
  const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
  server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{entries:['tests/m07-host.html']},plugins:[{name:'m07-save-names',configureServer(s){s.middlewares.use('/__m07_host',(req,res)=>{let raw='';req.on('data',d=>raw+=d);req.on('end',async()=>{try{const request=JSON.parse(raw);const value=await rpc(request.channel,request.body);res.setHeader('Content-Type','application/json');res.end(JSON.stringify({ok:true,value}));}catch(error){res.end(JSON.stringify({ok:false,error:error.message}));}});});}}]});await server.listen();
  const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--mute-audio']});
  const page=await browser.newPage({viewport:{width:1920,height:1080},acceptDownloads:true}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));page.setDefaultTimeout(60000);
  await page.goto(server.resolvedUrls.local[0]+'tests/m07-host.html');
  await page.locator('nav').getByRole('button',{name:'发声类型合成',exact:true}).click();
  const scope=page.getByLabel('发声类型连续统工作区',{exact:true}),click=name=>scope.getByRole('button',{name,exact:true}).click();
  const idle=()=>page.waitForFunction(()=>document.querySelector('.m07-page')?.getAttribute('aria-busy')==='false');
  await click('打开音频目录');await scope.getByRole('combobox',{name:'源音频',exact:true}).selectOption({label:'input0.wav'});await scope.getByRole('combobox',{name:'目标音频',exact:true}).selectOption({label:'input1.wav'});
  await click('提取 F0');await idle();await scope.getByText('分析完成，控制点尚未修改',{exact:true}).waitFor();
  await click('生成全部六组');await idle();assert.match(await scope.locator('.history-heading').innerText(),/6 组/);
  // Change current controls, then export the existing historical nine-step snapshots.
  await scope.getByLabel('连续统步数',{exact:true}).fill('4');
  const labels=['仅 F0 变化','仅发声类型变化','F0 与发声类型同时变化'],names=[];
  for(const direction of ['源到目标','目标到源'])for(const label of labels){
   const name=`${direction} · ${label} · 9 步`;names.push(name);
   await scope.locator('.history-select').filter({hasText:name}).click();
   assert.equal(await scope.locator('.result-row>strong').innerText(),name);
   const saved=page.waitForResponse(r=>r.url().includes('/__m07_host')&&r.request().postData()?.includes('m07_save'));
   await click('保存完整组');assert.equal((await (await saved).json()).ok,true);
  }
  report.checks.push('all six real nine-step native groups use direction/type/step names after current step count changes');
  await rpc('test',{op:'sentinel',folder:names[0]});
  await scope.locator('.history-select').filter({hasText:names[0]}).click();
  const repeated=page.waitForResponse(r=>r.url().includes('/__m07_host')&&r.request().postData()?.includes('m07_save'));
  await click('保存完整组');assert.equal((await (await repeated).json()).ok,true);
  const jobs=await rpc('task',{op:'m07_list'}),job=jobs.find(j=>j.state==='succeeded'),grant=await rpc('invoke',{op:'choose',purpose:'output'});
  for(const name of ['../escape','bad/name','bad\\name','bad:name','.','trailing.'])await assert.rejects(rpc('task',{op:'m07_save',job:job.id,directory:grant.id,name}),/结果组名称不正确/);
  report.checks.push('repeat save creates a separate numbered folder and preserves the existing edited file');
  await page.screenshot({path:path.join(out,'saved-light.png')});
  for(const name of names){
   const download=page.waitForEvent('download');
   await page.evaluate(async expected=>{
    const {m07Port}=await import('/src/platform/m07.ts');
    const request=async body=>{const r=await(await fetch('/__m07_host',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({channel:'task',body})})).json();if(!r.ok)throw Error(r.error);return r.value;};
    const port=m07Port({project:'test',read:async(job,id)=>{const r=await request({op:'result',job,id});return Uint8Array.from(atob(r.base64),c=>c.charCodeAt(0)).buffer;}});
    const jobs=await request({op:'m07_list'});
    for(const job of jobs.filter(j=>j.state==='succeeded')){const result=await port.result(job),m=result.metadata;const label={1:'仅发声类型变化',2:'仅 F0 变化',3:'F0 与发声类型同时变化'}[m.continuum_type];if(m.action==='generate'&&`${m.reverse_direction?'目标到源':'源到目标'} · ${label} · ${m.generation.step_count} 步`===expected){await port.save(result,'');return;}}
    throw Error('Historical result missing');
   },name);
   const artifact=await download;assert.equal(artifact.suggestedFilename(),name+'.zip');await artifact.saveAs(path.join(info.scratch,'downloads',artifact.suggestedFilename()));
  }
  report.checks.push('all six browser downloads use the same names and original verified result files');
  report.exports=await rpc('test',{op:'audit'});assert(report.exports.folders.some(f=>f.name===names[0]+'（2）'));
  await page.evaluate(()=>document.documentElement.dataset.theme='dark');await page.screenshot({path:path.join(out,'saved-dark.png')});
  assert.deepEqual(errors,[]);report.success=true;
 }catch(error){report.error=error.stack;throw error;}finally{
  if(out)await fs.writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2));
  if(browser)await browser.close();if(server)await server.close();worker.stdin.end();const code=await exited;assert.equal(code,0);
  if(out){
   const receipt=JSON.parse(await fs.readFile(path.join(out,'cleanup.json'),'utf8'));
   if(receipt.status!=='deleted'){
    const cleanup=spawn('pwsh.exe',['-NoProfile','-File',path.join(root,'scripts/cleanup_test_scratch.ps1'),'-RunDirectory',out],{cwd:root,windowsHide:true,stdio:'inherit'});
    const status=await new Promise((resolve,reject)=>{cleanup.once('error',reject);cleanup.once('exit',resolve);});assert.equal(status,0);
   }
   assert.equal(JSON.parse((await fs.readFile(path.join(out,'cleanup.json'),'utf8')).replace(/^\uFEFF/,'')).status,'deleted');
  }
 }
 console.log(JSON.stringify({success:report.success,checks:report.checks,out}));
}
main().catch(error=>{console.error(error);process.exitCode=1;});
