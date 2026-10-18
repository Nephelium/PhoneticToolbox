// Owned paths are supplied by verification_run; finally releases only this test.
const assert=require('node:assert/strict'),fs=require('node:fs/promises'),path=require('node:path');
const {pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..'),out=process.argv[2],scratch=process.argv[3];
if(!out||!scratch||path.resolve(scratch)!==path.join(path.resolve(out),'scratch'))throw Error('Expected owned verification_run paths');
async function main(){
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const server=await createServer({root:path.join(root,'frontend'),cacheDir:path.join(scratch,'vite-cache'),server:{host:'127.0.0.1',port:0},optimizeDeps:{entries:['tests/m16-workflow.html']}});
 let browser,page,success=false;const errors=[],checks=[];
 try{
  await server.listen();browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream']});
  page=await browser.newPage({viewport:{width:1440,height:1000}});page.on('pageerror',e=>errors.push(e.message));
  await page.goto(server.resolvedUrls.local[0]+'tests/m16-workflow.html');
  const scope=page.locator('.recording-page'),click=text=>scope.getByRole('button',{name:text,exact:true}).click(),idle=()=>page.waitForFunction(()=>document.querySelector('.recording-page')?.getAttribute('aria-busy')==='false');
  await scope.waitFor();await idle();await click('设备与录前检测');
  await scope.getByText('同声卡输入通道数',{exact:true}).locator('input').fill('2');await scope.getByLabel('显示语谱图',{exact:true}).uncheck();
  for(const id of ['T004','T005']){
   await click('＋ 新任务');await scope.getByLabel('任务编号',{exact:true}).fill(id);await scope.getByLabel('录音内容',{exact:true}).fill(id+' synthetic only');await click('完成编辑');
   await click('● 开始录音');await page.waitForFunction(()=>document.querySelector('.recording-indicator')||document.querySelector('.module-status-error'));assert.equal(await scope.locator('.module-status-error').count(),0);await scope.getByText('● 正在录音',{exact:true}).waitFor();await page.waitForTimeout(400);await click('■ 停止录音');await idle();
   await page.waitForFunction(expected=>window.__recordingQA.snapshot?.project.takes.at(-1)?.task_snapshot?.id===expected,id);
  }
  assert.deepEqual(await page.evaluate(()=>window.__recordingQA.snapshot.project.takes.map(t=>t.task_snapshot.id)),['T004','T005']);checks.push('Real AudioWorklet/fake Chromium microphone: new unsaved T005 never records into T004');
  const range=()=>scope.locator('.m16-time-axis').last().innerText();const before=await range();const plot=scope.locator('.m16-wave-lanes');await plot.hover();await page.keyboard.down('Control');await page.mouse.wheel(0,-120);await page.keyboard.up('Control');await page.waitForTimeout(350);assert.notEqual(await range(),before);await click('整段总览');checks.push('Native browser Ctrl-wheel changes audio time range');
  await click('＋ 新任务');await click('完成编辑');await click('保存 / 批量导出');const dialog=page.getByRole('dialog',{name:'保存和批量导出'});
  const modes=['all','versions','selected'];for(const mode of modes){
   if(mode!==modes[0])await click('保存 / 批量导出');await dialog.getByLabel('导出范围',{exact:true}).selectOption(mode);
   if(mode==='versions')assert(await dialog.getByLabel('导出版本',{exact:true}).isDisabled());
   const downloadPromise=page.waitForEvent('download');await dialog.getByRole('button',{name:'生成下载包',exact:true}).click();const download=await downloadPromise;await download.saveAs(path.join(scratch,mode+'.zip'));await idle();await scope.locator('.job-progress').waitFor({state:'hidden'});
  }
  assert.equal(await page.evaluate(()=>window.__recordingQA.requests.filter(r=>r.action==='export').length),6);checks.push('Three batch scopes work from an empty new task; history disables the single-version selector');
  await click('导入任务');const help=page.getByRole('dialog',{name:'录音任务表格格式'});const exampleDownload=page.waitForEvent('download');await help.getByRole('button',{name:'下载 XLSX 示例',exact:true}).click();const example=await exampleDownload;assert.equal(example.suggestedFilename(),'录音任务表格示例.xlsx');const examplePath=path.join(scratch,example.suggestedFilename());await example.saveAs(examplePath);
  const XLSX=await import(pathToFileURL(path.join(root,'frontend/src/modules/perception/vendor/xlsx.mjs'))),book=XLSX.read(await fs.readFile(examplePath),{type:'buffer'});assert.equal(book.Sheets['录音任务'].C1.v,'录音内容');assert.equal(book.Sheets['录音任务'].E3.v,2);await page.screenshot({path:path.join(out,'xlsx-example-dialog.png'),fullPage:true});await help.getByRole('button',{name:'关闭',exact:true}).click();
  await scope.locator('input[type=file]').setInputFiles(examplePath);const preview=page.getByRole('dialog',{name:'任务表格导入预览'});await preview.waitFor();assert((await preview.innerText()).includes('展开后 3 条任务'));await preview.getByRole('button',{name:'取消',exact:true}).click();checks.push('XLSX button downloads a real workbook, exact Chinese columns/numeric repeats, and reimports as 3 tasks');
  for(const theme of ['light','dark']){await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);await page.waitForTimeout(350);const slider=await scope.locator('.device-levels>label').first().boundingBox(),meters=await scope.locator('.input-meters').boundingBox();assert(meters.x>slider.x+slider.width&&Math.abs(meters.y-slider.y)<2);await page.screenshot({path:path.join(out,'browser-'+theme+'.png'),fullPage:true});}
  checks.push('Light/dark browser levels stay right of playback volume');assert.deepEqual(errors,[]);success=true;
 }catch(error){if(page){await fs.writeFile(path.join(out,'failure.txt'),await page.locator('body').innerText());await page.screenshot({path:path.join(out,'failure.png'),fullPage:true});console.error(await page.locator('.module-status-error').allTextContents());}throw error;
 }finally{await browser?.close();await server.close();await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success,scope:'Windows Chrome with real browser port and AudioWorklet, fake input and memory/export double; no server or physical device',checks,errors},null,2));}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
