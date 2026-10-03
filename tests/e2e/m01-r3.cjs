const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');const root=path.resolve(__dirname,'../..');
async function main(){
 const out=path.join(root,'output/validation/m01-r3','chrome-'+Date.now());await fs.mkdir(out,{recursive:true});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,hmr:false},optimizeDeps:{entries:['tests/m01-r3.html']}});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--mute-audio']});
 const page=await browser.newPage({viewport:{width:1800,height:1000}});page.setDefaultTimeout(15000);
 const report={success:false,checks:[],layouts:[],errors:[],scope:'Actual Chrome with synthetic audio and directory/task capability fixtures'};page.on('pageerror',e=>report.errors.push(e.message));
 const click=name=>page.getByRole('button',{name,exact:true}).click();
 const directory=async(name,id)=>{await page.evaluate(id=>window.qa.nextDirectory=id,id);await click(name);await page.waitForFunction(()=>![...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='正在读取…'));};
 const row=index=>page.locator('.m01-file-list .file-row').nth(index);
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/m01-r3.html');await page.locator('.nav-item').filter({hasText:'参数估计'}).click();
  assert.equal(await page.getByRole('button',{name:'选择关联目录',exact:true}).count(),0);
  await directory('选择音频目录','input');await row(0).click();await page.locator('.m01-slicing').waitFor();
  assert.equal(await page.getByLabel('TextGrid关联',{exact:true}).inputValue(),'ga');assert.equal(await page.getByLabel('唇形关联',{exact:true}).inputValue(),'la');
  assert.deepEqual(await page.evaluate(()=>window.qa.choices),['input']);assert.equal(await page.getByText('同音频目录',{exact:true}).count(),2);
  report.checks.push('Two named directory buttons; default TextGrid and lip matches need only the audio grant');
  await directory('选择TextGrid目录','grids');assert.equal(await page.getByLabel('TextGrid关联',{exact:true}).inputValue(),'external-ga');assert.equal(await page.getByLabel('唇形关联',{exact:true}).inputValue(),'la');
  await directory('选择唇形目录','lips');assert.equal(await page.getByLabel('唇形关联',{exact:true}).inputValue(),'');
  await row(1).click();assert.equal(await page.getByLabel('TextGrid关联',{exact:true}).inputValue(),'external-gb');assert.equal(await page.getByLabel('唇形关联',{exact:true}).inputValue(),'external-lb');
  assert.equal(await page.locator('.m01-file-list .file-row').count(),2);assert.equal(await page.getByRole('alert').count(),0);
  report.checks.push('Independent overrides exclude wrong resource kinds and external audio; no default-directory duplicate ambiguity; missing override clears automatic old links');
  await page.evaluate(()=>window.qa.cancel=true);await click('选择TextGrid目录');await page.evaluate(()=>window.qa.cancel=false);assert.equal(await page.getByLabel('TextGrid关联',{exact:true}).inputValue(),'external-gb');
  await click('TextGrid使用音频目录');assert.equal(await page.getByLabel('TextGrid关联',{exact:true}).inputValue(),'gb');assert.equal(await page.getByLabel('唇形关联',{exact:true}).inputValue(),'external-lb');
  await click('唇形使用音频目录');assert.equal(await page.getByLabel('唇形关联',{exact:true}).inputValue(),'lb');
  await page.getByLabel('唇形关联',{exact:true}).selectOption('');await click('刷新列表');assert.equal(await page.getByLabel('唇形关联',{exact:true}).inputValue(),'');
  const choices=await page.evaluate(()=>window.qa.choices.length);await click('关联唇形');await page.getByText(/唇形关联：2 个已匹配/).waitFor();assert.equal(await page.getByLabel('唇形关联',{exact:true}).inputValue(),'lb');assert.equal(await page.evaluate(()=>window.qa.choices.length),choices);
  await directory('选择音频目录','input2');await row(0).click();await page.locator('.m01-slicing').waitFor();assert.equal(await page.getByLabel('TextGrid关联',{exact:true}).inputValue(),'2ga');assert.equal(await page.getByLabel('唇形关联',{exact:true}).inputValue(),'2la');
  report.checks.push('Cancel preserves paths; reset restores defaults; manual cancellation survives refresh; bulk relink uses current directory; defaults follow new audio folder');
  await click('开始全列表分析');await page.waitForFunction(()=>window.qa.submitted.length===1);assert.equal((await page.evaluate(()=>window.qa.submitted[0].inputs)).length,2);
  for(const [width,height] of [[1800,1000],[1440,900],[900,700],[650,700],[450,700]]){
   await page.setViewportSize({width,height});
   const bounds=await page.locator('.m01-directory-bar').evaluate(el=>{const r=el.getBoundingClientRect();return {width:r.width,scroll:el.scrollWidth,client:el.clientWidth,overflow:[...el.querySelectorAll('button')].filter(b=>b.getBoundingClientRect().right>r.right+1).map(b=>b.textContent)};});
   assert.equal(bounds.overflow.length,0);assert(bounds.scroll<=bounds.client+1);report.layouts.push({width,height,bounds});
  }
  await page.setViewportSize({width:1440,height:900});for(const theme of ['light','dark']){await page.evaluate(theme=>document.documentElement.dataset.theme=theme,theme);await page.waitForTimeout(200);await page.screenshot({path:path.join(out,'directories-'+theme+'.png')});}
  report.checks.push('Five toolbar widths including 450px; light/dark screenshots; batch uses all current audio');
  await page.goto(server.resolvedUrls.local[0]+'tests/m01-r3.html?web');await page.locator('.nav-item').filter({hasText:'参数估计'}).click();await page.locator('.m01-result-downloads').waitFor();
  assert.deepEqual(await page.locator('.m01-result-downloads button').allTextContents(),['a.xlsx','a.ptb.sqlite']);await click('a.xlsx');await page.waitForFunction(()=>window.qa.downloads.length===1);assert.equal((await page.evaluate(()=>window.qa.downloads[0])).name,'a.xlsx');
  report.checks.push('Web result list exposes XLSX and SQLite only and downloads with the source filename');assert.deepEqual(report.errors,[]);report.success=true;
 }catch(e){report.error=String(e);await page.screenshot({path:path.join(out,'failed.png')});throw e;}
 finally{await fs.writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2));console.log(out);await browser.close();await server.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
