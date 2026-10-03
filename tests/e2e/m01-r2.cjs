// UI capability fixture; audio comes exclusively from the authorized recording copy.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');const root=path.resolve(__dirname,'../..');
async function main(){
 const source=path.join(root,'output/validation/m01-m02-r1/real-qt-b2322971b8aa4e7b8b86e176e328536a/recursive/甲/15-范皓云-男-1.wav');
 const wav=await fs.readFile(source),out=path.join(root,'output/validation/m01-r2','chrome-'+Date.now());await fs.mkdir(out,{recursive:true});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,hmr:false},plugins:[{name:'authorized-recording',configureServer(s){s.middlewares.use('/__m01_r2_audio',(_req,res)=>{res.setHeader('Content-Type','audio/wav');res.end(wav);});}}],optimizeDeps:{entries:['tests/m01-r2.html']}});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--mute-audio']});
 const page=await browser.newPage({viewport:{width:1800,height:1000}});page.setDefaultTimeout(15000);
 const checks=[],errors=[],report={success:false,checks,errors,scope:'Actual Chrome; UI capability fixtures reuse one unchanged authorized recording; no scientific or task execution'};page.on('pageerror',e=>errors.push(e.message));
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/m01-r2.html');await page.locator('.nav-item').filter({hasText:'参数估计'}).click();
  await page.getByRole('button',{name:'选择音频目录',exact:true}).click();await page.locator('.m01-file-list .file-row').first().click();await page.locator('.m01-slicing').waitFor();
  assert.equal(await page.getByLabel('唇形关联',{exact:true}).inputValue(),'old-a');
  assert.equal(await page.locator('select[aria-label="唇形关联"] option[value="time-a"]').count(),0);
  assert.equal(await page.locator('.signal-panel .m01-tiers').count(),0);
  assert(await page.locator('.m01-slicing').evaluate(el=>!!el.closest('.workbench-left')&&el.previousElementSibling?.textContent.includes('编辑14项设置')));
  assert.equal(await page.getByText('查看完整标签列表与精确时间',{exact:true}).count(),0);assert.equal(await page.getByText('关联说明与旧文件兼容',{exact:true}).count(),0);
  await page.locator('.textgrid-interval').first().click();assert.equal(Number(await page.getByLabel(/^终点/).inputValue()),.4);
  checks.push('Leftmost TextGrid section follows analysis settings; obsolete sections removed; timeline selection and old PKL association work');
  await page.getByRole('button',{name:'选择输出参数',exact:true}).click();assert.equal(await page.locator('.parameter-grid input').count(),80);
  for(const [width,height,columns] of [[1800,1000,4],[1440,900,4],[700,760,2],[480,700,1]]){
   await page.setViewportSize({width,height});
   assert.equal(await page.locator('.parameter-grid').evaluate(el=>getComputedStyle(el).gridTemplateColumns.split(' ').length),columns);
   assert(await page.locator('dialog').evaluate(el=>el.scrollWidth<=el.clientWidth+1));
  }
  await page.setViewportSize({width:1800,height:1000});await page.screenshot({path:path.join(out,'parameters-four-columns.png')});
  await page.getByLabel('搜索参数',{exact:true}).fill('REAPER');assert.equal(await page.locator('.parameter-grid input').count(),1);
  await page.getByRole('button',{name:'全不选',exact:true}).click();assert(await page.getByRole('button',{name:'应用到草稿',exact:true}).isDisabled());
  await page.getByRole('button',{name:'取消',exact:true}).click();assert.equal(await page.locator('.parameter-count strong').innerText(),'80');
  checks.push('All 80 parameters in four desktop columns; two/one narrow columns; no dialog overflow; search/empty selection/cancel retained');
  await page.getByRole('button',{name:'关联唇形',exact:true}).click();await page.getByText(/唇形关联：2 个已匹配，1 个未找到/).waitFor();
  assert.equal(await page.getByLabel('唇形关联',{exact:true}).inputValue(),'new-a');
  await page.getByLabel('唇形关联',{exact:true}).selectOption('');
  await page.locator('.m01-file-list .file-row').nth(1).click();assert.equal(await page.getByLabel('唇形关联',{exact:true}).inputValue(),'new-b');
  await page.getByRole('button',{name:'刷新列表',exact:true}).click();await page.locator('.m01-file-list .file-row').first().click();assert.equal(await page.getByLabel('唇形关联',{exact:true}).inputValue(),'');
  await page.getByRole('button',{name:'开始全列表分析',exact:true}).click();await page.waitForFunction(()=>window.qa.submitted.length===1);
  const submitted=await page.evaluate(()=>window.qa.submitted[0].inputs);assert.equal(submitted.length,3);assert.equal(submitted[0].lip,null);assert.equal(submitted[1].lip.id,'new-b');
  checks.push('Bulk directory matching covers whole list; JSON preferred; cancelling only current audio survives refresh and submitted snapshot');
  await page.evaluate(()=>window.qa.cancel=true);await page.getByRole('button',{name:'关联唇形',exact:true}).click();assert.equal(await page.getByLabel('唇形关联',{exact:true}).inputValue(),'');
  await page.evaluate(()=>{window.qa.cancel=false;window.qa.duplicate=true;});await page.getByRole('button',{name:'关联唇形',exact:true}).click();await page.getByText(/1 个需要逐条确认/).waitFor();
  await page.locator('.m01-file-list .file-row').nth(2).click();await page.getByRole('alert').filter({hasText:'多个候选'}).waitFor();
  await page.getByLabel('唇形关联',{exact:true}).selectOption('dup1');assert.equal(await page.getByRole('alert').filter({hasText:'多个候选'}).count(),0);
  checks.push('Directory cancellation preserves choices; duplicate old records require per-audio selection and resolve explicitly');
  for(const theme of ['light','dark']){
   await page.evaluate(theme=>document.documentElement.dataset.theme=theme,theme);await page.screenshot({path:path.join(out,'m01-'+theme+'.png')});
   await page.getByRole('button',{name:'选择输出参数',exact:true}).click();await page.screenshot({path:path.join(out,'parameters-'+theme+'.png')});await page.getByRole('button',{name:'取消',exact:true}).click();
  }
  for(const height of [660,1000]){await page.setViewportSize({width:1440,height});assert(await page.locator('.m01-slicing').evaluate(el=>{const pane=el.closest('.workbench-left');pane.scrollTop=pane.scrollHeight;const r=el.getBoundingClientRect(),p=pane.getBoundingClientRect();return r.bottom<=p.bottom+1&&r.right<=p.right+1;}));}
  checks.push('Light/dark page and modal screenshots; slicing controls reachable inside left pane at short/tall windows');
  assert.deepEqual(errors,[]);report.success=true;
 }catch(error){report.error=String(error);await page.screenshot({path:path.join(out,'failed.png')});throw error;}
 finally{await fs.writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2));console.log(out);await browser.close();await server.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
