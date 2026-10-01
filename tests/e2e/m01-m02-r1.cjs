// Owned Chrome / synthetic capability fixtures; native science covered separately.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');const root=path.resolve(__dirname,'../..');
async function main(){
 const out=path.join(root,'output/validation/m01-m02-r1','chrome-'+Date.now());await fs.mkdir(out,{recursive:true});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0},optimizeDeps:{entries:['tests/m01-m02-r1.html']}});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--mute-audio']});
 const page=await browser.newPage({viewport:{width:1440,height:900},acceptDownloads:true});page.setDefaultTimeout(15000);
 const checks=[],errors=[];page.on('pageerror',e=>errors.push(e.message));
 const report={success:false,checks,errors};
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/m01-m02-r1.html');
  const open=async name=>{await page.locator('.nav-item').filter({hasText:name}).click();};
  await open('参数估计');await page.getByRole('button',{name:'选择音频目录',exact:true}).click();await page.locator('.m01-file-row').count();
  await page.waitForFunction(()=>document.querySelectorAll('.m01-file-entry').length===400);
  await page.locator('.m01-file-list .file-row').first().click();await page.locator('.wave-track').first().waitFor();
  for(const height of [900,660]){
   await page.setViewportSize({width:1440,height});await page.waitForTimeout(150);
   const geometry=await page.locator('.workbench-columns').evaluate(grid=>{const panes=[...grid.querySelectorAll(':scope>.workbench-pane')],heights=panes.map(p=>p.clientHeight);const signal=panes[1].getBoundingClientRect().top;panes[0].scrollTop=1000;return {height:grid.clientHeight,heights,scrolled:panes[0].scrollTop,stable:panes[1].getBoundingClientRect().top===signal,shared:grid.classList.contains('shared-scroll')};});
   assert(geometry.scrolled>100&&geometry.stable&&!geometry.shared);assert(Math.max(...geometry.heights)-Math.min(...geometry.heights)<3);
  }
  checks.push('400 audio rows scroll independently at 900/660px heights, without inflating the workbench');
  await page.getByLabel('包含子文件夹',{exact:true}).check();await page.waitForFunction(()=>document.querySelectorAll('.m01-file-entry').length===401);
  await page.getByLabel('包含子文件夹',{exact:true}).uncheck();await page.waitForFunction(()=>document.querySelectorAll('.m01-file-entry').length===400);
  await page.waitForFunction(()=>window.qa.polls>=2);
  const observed=await page.evaluate(async()=>{const el=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='开始全列表分析'),records=[];for(let i=0;i<65;i++){records.push([el.getBoundingClientRect().y,el.disabled,document.querySelector('.task-operation-status').textContent]);await new Promise(r=>setTimeout(r,100));}return records;});
  assert(observed.every(r=>Math.abs(r[0]-observed[0][0])<.1&&!r[1]&&!r[2].includes('正在')));
  await page.getByRole('button',{name:'取消后续处理',exact:true}).click();await page.waitForFunction(()=>window.qa.cancelled===1);
  checks.push('Background polling leaves action geometry and enablement stable; cancel remains usable');
  await page.setViewportSize({width:1440,height:900});await page.screenshot({path:path.join(out,'m01-light.png')});
  await open('参数显示');await page.getByRole('button',{name:'选择音频目录',exact:true}).click();await page.waitForFunction(()=>document.querySelectorAll('.m02-files button').length===400);
  const geom=await page.locator('.m02-files').evaluate(el=>{el.scrollTop=5000;return {scroll:el.scrollTop,height:el.clientHeight,total:el.scrollHeight};});assert(geom.scroll>100&&geom.height<900&&geom.total>5000);
  await page.getByLabel('包含子文件夹',{exact:true}).check();await page.getByRole('button',{name:'子目录/音频000.wav',exact:true}).click();await page.locator('.empty-plot').waitFor();
  checks.push('M02 bounded file list, recursive toggle and relative-path parameter association');
  await page.locator('.m02-parameters input').nth(0).check();await page.getByRole('button',{name:'将 1 项分配到图窗',exact:true}).click();
  await page.getByRole('button',{name:'新建图窗',exact:true}).click();await page.getByRole('button',{name:'清空勾选',exact:true}).click();await page.locator('.m02-parameters input').nth(1).check();await page.getByRole('button',{name:'将 1 项分配到图窗',exact:true}).click();
  await page.getByRole('button',{name:'清空选定图窗',exact:true}).click();assert.equal(await page.locator('.parameter-curve').count(),1);assert.equal(await page.locator('.empty-plot').count(),1);
  await page.getByRole('button',{name:'删除选定图窗',exact:true}).click();assert.equal(await page.locator('.parameter-figure').count(),1);assert.equal(await page.locator('.parameter-curve').count(),1);
  const download=async (name)=>{const wait=page.waitForEvent('download');await page.getByRole('button',{name:'保存当前图',exact:true}).click();const file=await wait;assert(file.suggestedFilename().endsWith(name));await file.saveAs(path.join(out,'current'+name));return fs.readFile(path.join(out,'current'+name));};
  const png=await download('.png');assert.equal(png.subarray(1,4).toString(),'PNG');
  await page.getByLabel('图窗 1图片格式',{exact:true}).selectOption('svg');const svg=await download('.svg');assert(svg.toString().includes('<svg'));
  await page.evaluate(()=>document.documentElement.dataset.theme='dark');await page.getByLabel('图窗 1图片格式',{exact:true}).selectOption('png');await download('.png');await page.screenshot({path:path.join(out,'m02-dark.png')});
  await page.getByRole('button',{name:'删除选定图窗',exact:true}).click();assert.equal(await page.locator('.parameter-figure').count(),0);await page.getByRole('button',{name:'保存绘图配置',exact:true}).click();
  await page.reload();await open('参数显示');await page.getByRole('button',{name:'选择音频目录',exact:true}).click();await page.getByRole('button',{name:'音频000.wav',exact:true}).click();await page.getByText('暂无图窗，可在右侧新建图窗。',{exact:true}).waitFor();
  await page.getByRole('button',{name:'新建图窗',exact:true}).click();await page.locator('.empty-plot').waitFor();
  checks.push('Clear/delete/last deletion/reload/new plot; default PNG and explicit SVG downloaded in light/dark themes');
  for(const scale of [70,100,150]){await page.evaluate(async scale=>(await import('/src/state/pageZoom.ts')).setPageScale(scale),scale);await page.waitForTimeout(100);assert(await page.locator('.m02-files').evaluate(el=>el.clientHeight<1000&&el.scrollHeight>el.clientHeight));}
  await page.setViewportSize({width:600,height:700});await page.screenshot({path:path.join(out,'m02-narrow.png')});assert(await page.locator('.m02-files').evaluate(el=>el.clientHeight<=181));
  assert.deepEqual(errors,[]);report.success=true;
 }catch(e){report.error=String(e);await page.screenshot({path:path.join(out,'failed.png')});throw e;}
 finally{await fs.writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2));console.log(out);await browser.close();await server.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
