const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');const root=path.resolve(__dirname,'../..');
async function main(){
 const out=path.join(root,'output/validation/m01-r4','chrome-'+Date.now());await fs.mkdir(out,{recursive:true});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,hmr:false},optimizeDeps:{entries:['tests/m01-r4.html']}});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--mute-audio']});
 const page=await browser.newPage({viewport:{width:1800,height:1000}});const report={success:false,checks:[],layouts:[],errors:[]};page.on('pageerror',e=>report.errors.push(e.message));
 const click=name=>page.getByRole('button',{name,exact:true}).click();
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/m01-r4.html');await page.locator('.nav-item').filter({hasText:'参数估计'}).click();await click('选择音频目录');await page.locator('.m01-file-list .file-row').first().click();
  await page.getByText('联合计算 EGG',{exact:true}).click();await page.getByText('当前文件为单声道，将跳过 EGG。',{exact:true}).waitFor();
  await page.getByLabel('EGG参数保存方式').selectOption('cycles');assert.equal(await page.getByLabel('EGG平滑毫秒').count(),0);
  await page.getByLabel('EGG参数保存方式').selectOption('aligned');await page.getByLabel('EGG平滑毫秒').fill('0');
  await page.getByText('EGG 计算设置',{exact:true}).click();await page.getByText('计算 GCI F0 派生声学参数',{exact:true}).click();await click('开始全列表分析');
  await page.waitForFunction(()=>window.qa.submitted.length===1);const config=await page.evaluate(()=>window.qa.submitted[0].config);
  assert.equal(config.extended.max_duration_s,1800);assert.equal(config.extended.audio_channel,1);assert.equal(config.extended.egg.smooth_ms,0);assert.equal(config.extended.egg.derived,true);
  report.checks.push('Opt-in defaults, mono explanation, two table modes, smoothing off, GCI-derived switch and immutable 30min batch snapshot');
  const selectedButton=page.getByRole('button',{name:'分析选中文件',exact:true});
  assert(await selectedButton.isDisabled(),'Preview selection alone must not submit a selected-file batch');
  await page.getByLabel('选择音频 b.wav',{exact:true}).check();await selectedButton.click();
  await page.waitForFunction(()=>window.qa.submitted.length===2);
  assert.deepEqual(await page.evaluate(()=>window.qa.submitted[1].inputs.map(i=>i.audio.id)),['b']);
  assert.equal(await page.evaluate(()=>window.qa.submitted[1].inputs[0].textgrid.id),'gb');
  await page.getByLabel('选择音频 a.wav',{exact:true}).check();await selectedButton.click();
  await page.waitForFunction(()=>window.qa.submitted.length===3);
  assert.deepEqual(await page.evaluate(()=>window.qa.submitted[2].inputs.map(i=>i.audio.id)),['a','b']);
  await page.getByLabel('选择音频 a.wav',{exact:true}).uncheck();await click('开始全列表分析');
  await page.waitForFunction(()=>window.qa.submitted.length===4);
  assert.deepEqual(await page.evaluate(()=>window.qa.submitted[3].inputs.map(i=>i.audio.id)),['a','b']);
  await page.getByLabel('全选音频',{exact:true}).check();await page.getByLabel('全选音频',{exact:true}).uncheck();assert(await selectedButton.isDisabled());
  report.checks.push('M01-R5 no-selection disabled, single/multiple selected inputs preserve order/associations, full-list action remains independent');
  const heights=[];
  for(const [state,progress] of [['running',.4],['succeeded',1],['failed',.65],['interrupted',.4]]){
   await page.evaluate(({state,progress})=>{const b=window.qa.batch;b.summary.items[0]={index:0,state,progress,job_id:'job',error_code:state==='failed'?'m01_duration_limit':null};b.summary.closed=state!=='running';b.summary.complete=state==='succeeded';b.summary.counts.succeeded=state==='succeeded'?1:0;b.summary.counts.failed=state==='failed'?1:0;},{state,progress});
   await page.waitForFunction(({state,progress})=>document.querySelector('progress[aria-label="单个音频处理进度"]')?.value===progress&&document.querySelector('.m01-file-progress small')?.textContent.includes(state==='running'?'分段':state==='succeeded'?'完成':state==='failed'?'失败':'中断'),{state,progress},{timeout:12000});
   heights.push(await page.locator('.m01-file-progress').evaluate(e=>e.getBoundingClientRect().height));
  }
  assert(heights.every(v=>Math.abs(v-heights[0])<1));report.checks.push('Persistent single-file progress at running/success/failure/interruption retains exactly the same card height');
  for(const [width,height] of [[1800,1000],[1100,760],[650,800]]){
   await page.setViewportSize({width,height});const overflow=await page.locator('.m01-joint').evaluate(e=>e.scrollWidth>e.clientWidth+1);assert(!overflow);
   const allRect=await page.getByRole('button',{name:'开始全列表分析',exact:true}).boundingBox(),selectedRect=await selectedButton.boundingBox();
   assert(allRect&&selectedRect&&selectedRect.y>=allRect.y+allRect.height&&Math.abs(allRect.width-selectedRect.width)<1);report.layouts.push([width,height]);
  }
  for(const mode of ['light','dark']){await page.evaluate(mode=>document.documentElement.dataset.theme=mode,mode);await page.screenshot({path:path.join(out,'m01-selected-'+mode+'.png')});}
  await page.setViewportSize({width:1800,height:1000});await page.locator('.nav-item').filter({hasText:'参数显示'}).click();await click('选择音频目录');await click('a.wav');await page.getByLabel('CQ',{exact:true}).waitFor();
  for(const name of ['CQ','SQ','F0 - GCI'])await page.getByLabel(name,{exact:true}).check();await click('将 3 项分配到图窗');await page.waitForFunction(()=>document.querySelectorAll('.m02-page .parameter-curve[d*=M]').length===3);
  assert((await page.evaluate(()=>window.qa.views)).some(v=>v?.parameters.length===3));
  await page.waitForFunction(()=>!document.querySelector('.m02-page [aria-busy=true]'));
  const settledViews=await page.evaluate(()=>window.qa.views.length);await page.waitForTimeout(500);
  assert.equal(await page.evaluate(()=>window.qa.views.length),settledViews,'A completed view must not schedule itself again');
  await page.getByLabel('时间窗长度（秒）',{exact:true}).fill('0.1');await page.getByLabel('时间窗长度（秒）',{exact:true}).dispatchEvent('change');await page.waitForFunction(()=>window.qa.views.at(-1)?.end<=.101);
  await page.waitForFunction(()=>!document.querySelector('.m02-page [aria-busy=true]'));
  assert.equal(await page.getByRole('button',{name:'保存当前图',exact:true}).isDisabled(),false);
  report.checks.push('M02 dynamic EGG selection, separate native tracks, plotted curves and bounded requests follow zoom');
  for(const mode of ['light','dark']){await page.evaluate(mode=>document.documentElement.dataset.theme=mode,mode);await page.waitForTimeout(200);await page.screenshot({path:path.join(out,'m02-'+mode+'.png')});}
  assert.deepEqual(report.errors,[]);report.success=true;
 }catch(e){report.error=String(e);await page.screenshot({path:path.join(out,'failed.png')});throw e;}
 finally{await fs.writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2));console.log(out);await browser.close();await server.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
