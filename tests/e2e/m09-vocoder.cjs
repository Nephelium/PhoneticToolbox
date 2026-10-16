// Actual Vue page with task-owned isolated Chrome. Scientific execution is separate.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url'),{spawnSync}=require('node:child_process'),{createHash}=require('node:crypto');
const root=path.resolve(__dirname,'../..'),headless=process.argv.includes('--headless');
async function main(){
 const out=path.join(root,'output/validation/m09-vocoder-ui','chrome-'+Date.now()),scratch=path.join(out,'scratch');await fs.mkdir(scratch,{recursive:true});
 const write=(n,v)=>fs.writeFile(path.join(out,n),JSON.stringify(v,null,2)+'\n');
 const marker={schema:'ptb-test-run/1',owner_pid:process.pid,state:'active',started_at:new Date().toISOString(),recipe:'real M09 Vue page; optional component absent/download/failure/ready, default preserved, local task snapshot; light/dark and narrow layout'};await write('run.json',marker);
 let server,browser,page;const report={success:false,headless,checks:[],errors:[]};
 try{
  const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
  server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{noDiscovery:true,include:['vue']}});await server.listen();
  const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  browser=await chromium.launchPersistentContext(path.join(scratch,'profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless,viewport:{width:1440,height:1000}});page=await browser.newPage();page.on('pageerror',e=>report.errors.push(e.message));
  for(const theme of ['light','dark']){
   await page.goto(server.resolvedUrls.local[0]+'tests/m09-vocoder.html');await page.evaluate(v=>document.documentElement.dataset.theme=v,theme);
   const choice=page.getByLabel('声码器（可选）',{exact:true});await choice.waitFor();assert.equal(await choice.inputValue(),'none');
   const option=await choice.locator('option[value="bigvgan"]').evaluate(e=>({disabled:e.disabled,html:e.outerHTML}));assert(option.disabled,JSON.stringify(option));
   await page.getByLabel('语谱图图片',{exact:true}).selectOption('synthetic-image');
   const predict=page.getByRole('button',{name:'预测窗长与 dB 范围',exact:true});
   assert(await predict.isDisabled());assert(await page.getByRole('button',{name:'开始重建',exact:true}).isDisabled());
   assert.equal(await page.getByRole('checkbox').count(),0);
   await page.getByLabel('时长（s）',{exact:true}).fill('0.5102');
   await page.getByLabel('频率终点（Hz）',{exact:true}).fill('5000');
   assert.equal(await page.evaluate(()=>qa.predictions||0),0,'Filling calibration must not trigger prediction');
   await predict.click();await page.getByText(/预测窗长 10.40/).waitFor();
   await page.getByLabel('窗长（ms）',{exact:true}).fill('5');
   await page.getByLabel('最小 dB（白）',{exact:true}).fill('-55');
   await page.getByLabel('最大 dB（黑）',{exact:true}).fill('-2');
   await page.getByLabel('Aoki 短时窗初始化',{exact:true}).selectOption('random');
   await page.getByRole('button',{name:'开始重建',exact:true}).click();
   assert.deepEqual(await page.evaluate(()=>{const c=qa.submitted.at(-1);return [c.win_length_ms,c.min_db,c.max_db,c.auto_parameters,c.reconstruction_profile,c.target_sr,c.phase_initialization]}),[5,-55,-2,false,'calibrated',0,'random']);
   assert.equal(await page.evaluate(()=>qa.downloads||0),0);
   const count=await page.evaluate(()=>qa.predictions);
   await page.getByLabel('时长（s）',{exact:true}).fill('1');await page.waitForTimeout(900);
   assert.equal(await page.evaluate(()=>qa.predictions),count);
   await predict.click();await page.getByText(/预测窗长 10.40/).waitFor();
   assert.equal(await page.getByLabel('窗长（ms）',{exact:true}).inputValue(),'10');
   await page.getByLabel('选择算法',{exact:true}).selectOption('neural_pipeline');
   assert(await page.getByRole('button',{name:'开始重建',exact:true}).isDisabled());
   assert(await predict.isEnabled());
   assert.equal(await page.getByLabel('选择算法',{exact:true}).locator('option:checked').textContent(),'Mel 谱 → NAFNet → BigVGAN');
   await page.evaluate(()=>qa.fail=true);await page.getByRole('button',{name:'下载频谱修复与声码器',exact:true}).click();
   await page.getByText('声码器下载或校验未完成，请检查网络和磁盘空间后重试。',{exact:true}).waitFor();
   await page.evaluate(()=>qa.fail=false);await page.getByRole('button',{name:'下载频谱修复与声码器',exact:true}).click();
   await page.getByText('已下载，可离线使用',{exact:true}).waitFor();
   await page.getByLabel('模型推理设备',{exact:true}).selectOption('cpu');
   await page.getByText(/CPU 推理可能很慢/).waitFor();
   assert(await page.getByLabel('模型推理设备',{exact:true}).locator('option[value=cuda]').evaluate(e=>e.disabled),JSON.stringify(await page.evaluate(()=>qa.status)));
   await page.getByRole('button',{name:'选择已有 Python 环境',exact:true}).click();
   await page.getByRole('button',{name:'检查兼容性',exact:true}).click();
   await page.getByText('检查通过，可使用 fixture GPU。',{exact:true}).waitFor();
   await page.getByLabel('Aoki 短时窗初始化',{exact:true}).selectOption('pulse');
   await page.getByLabel('迭代次数',{exact:true}).fill('256');
   await page.getByRole('button',{name:'开始重建',exact:true}).click();
   assert.deepEqual(await page.evaluate(()=>{const c=qa.submitted.at(-1);return [c.n_iter,c.inference_device,c.phase_initialization]}),[256,'cuda','pulse']);
   await page.getByLabel('迭代次数',{exact:true}).fill('257');assert(await page.getByRole('button',{name:'开始重建',exact:true}).isDisabled());
   await page.getByLabel('迭代次数',{exact:true}).fill('32');
   await page.getByLabel('频率轴',{exact:true}).selectOption('log');assert(await predict.isDisabled());
   await page.getByLabel('频率起点（Hz）',{exact:true}).fill('50');await predict.click();await page.getByText(/预测窗长 10.40/).waitFor();
   await page.getByRole('button',{name:'开始重建',exact:true}).click();assert.equal(await page.evaluate(()=>qa.submitted.at(-1).frequency_axis),'log');
   // A late response cannot overwrite a user edit.
   await page.evaluate(()=>qa.delay=500);await predict.click();await page.getByLabel('窗长（ms）',{exact:true}).fill('7');
   await page.waitForTimeout(650);assert.equal(await page.getByLabel('窗长（ms）',{exact:true}).inputValue(),'7');
   await page.getByRole('button',{name:'图片涂鸦重建',exact:true}).click();
   await page.getByRole('button',{name:'应用校正并绘图',exact:true}).click();
   const canvas=page.getByLabel('可缩放频谱画布',{exact:true});await canvas.waitFor();
   for(const size of [1,200]){
    await page.getByLabel('画笔直径（像素）',{exact:true}).fill(String(size));
    await canvas.click({position:{x:200,y:120}});
    await page.getByRole('button',{name:'生成涂鸦音频',exact:true}).click();
    assert.deepEqual(await page.evaluate(()=>{const s=qa.submitted.at(-1).strokes.at(-1);return [s.size,s.size_unit]}),[size,'px']);
    await page.getByRole('button',{name:'撤销笔迹',exact:true}).click();
   }
   await page.evaluate(()=>document.fonts.ready);await page.screenshot({path:path.join(out,theme+'.png'),animations:'disabled'});
   for(const width of [1280,800]){await page.setViewportSize({width,height:900});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));}
   report.checks.push(theme+' calibration before explicit prediction, manual edits and stale response, common Aoki selector, default 2*fmax output, CPU warning, GPU picker/probe choice, download retry, GL 256 limit, log axis and 1280/800 widths');
   await page.setViewportSize({width:1440,height:1000});
  }
  assert.deepEqual(report.errors,[]);report.success=true;
 }catch(e){report.failure=String(e);if(page)await page.screenshot({path:path.join(out,'failure.png')}).catch(()=>{});throw e;}
 finally{
  await browser?.close();await server?.close();await write('report.json',report);
  const files=[];async function inventory(d){for(const e of await fs.readdir(d,{withFileTypes:true})){assert(!e.isSymbolicLink());const p=path.join(d,e.name);if(e.isDirectory())await inventory(p);else{const b=await fs.readFile(p);files.push({path:path.relative(scratch,p),bytes:b.length,sha256:createHash('sha256').update(b).digest('hex')});}}}await inventory(scratch);await write('scratch-manifest.json',{schema:'ptb-test-scratch-files/1',files});marker.state='ready';marker.test_outcome=report.success?'completed':'failed';await write('run.json',marker);
  const cleanup=spawnSync('powershell.exe',['-NoProfile','-ExecutionPolicy','Bypass','-File',path.join(root,'scripts/cleanup_test_scratch.ps1'),'-RunDirectory',out],{encoding:'utf8',windowsHide:true});if(cleanup.status)throw Error('Owned browser scratch cleanup failed');
  console.log('REPORT='+path.join(out,'report.json'));
 }
}
main().catch(e=>{console.error(e);process.exitCode=1;});
