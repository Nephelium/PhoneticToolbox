// Owned Chrome + synthetic geometry/storage; Qt checks exercise native PDF bytes separately.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');const root=path.resolve(__dirname,'../..');
async function main(){
 const [out,scratch]=process.argv.slice(2);assert(out&&scratch);let server,browser,page;const errors=[],checks=[];
 const report={success:false,errors,checks};
 try{
  const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
  server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{noDiscovery:true,include:['vue']}});await server.listen();
  const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  browser=await chromium.launchPersistentContext(path.join(scratch,'chrome-profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1440,height:1000}});
  page=await browser.newPage();page.on('pageerror',e=>errors.push(e.message));await page.goto(server.resolvedUrls.local[0]+'tests/m18-annotations.html');
  await page.locator('.pdf-text-line').first().waitFor();
  const highlight=page.getByRole('button',{name:'荧光笔',exact:true}),note=page.getByRole('button',{name:'添加批注',exact:true});
  async function selectLine(index=0){
   const b=await page.locator('[data-page="1"] .pdf-text-line').nth(index).boundingBox();
   await page.mouse.move(b.x+4,b.y+b.height/2);await page.mouse.down();await page.mouse.move(b.x+b.width-4,b.y+b.height/2,{steps:12});await page.mouse.up();
   await page.waitForFunction(()=>document.querySelector('.pdf-mark-tools').textContent.includes('已选择 1 页'));
  }
  async function selectBothPages(){
   await page.waitForFunction(()=>document.querySelector('[data-page="2"] .pdf-text-line'));
   await page.evaluate(()=>{const first=document.querySelector('[data-page="1"] .pdf-text-line').firstChild,last=document.querySelector('[data-page="2"] .pdf-text-line').firstChild;const r=document.createRange();r.setStart(first,0);r.setEnd(last,last.textContent.length-1);getSelection().removeAllRanges();getSelection().addRange(r);});
   await page.waitForFunction(()=>document.querySelector('.pdf-mark-tools').textContent.includes('已选择 2 页'));
  }
  await selectLine();await highlight.click();await page.waitForFunction(()=>window.m18Fixture.store.original.items.length===1,{},{timeout:3000});
  checks.push('Native pointer selection and immediate highlight');
  await selectLine();
  // Model engines/input methods that collapse the native range before toolbar activation.
  await page.evaluate(()=>document.querySelector('[aria-label="荧光笔颜色"]').addEventListener('pointerdown',()=>getSelection().removeAllRanges(),{once:true}));
  await page.getByLabel('荧光笔颜色',{exact:true}).click();await page.keyboard.press('ArrowDown');await page.keyboard.press('Enter');
  await page.waitForTimeout(150);
  report.afterColor=await page.locator('.pdf-mark-tools').innerText();
  await highlight.click({timeout:3000});await page.waitForFunction(()=>window.m18Fixture.store.original.items.length===2);
  assert.equal(await page.evaluate(()=>window.m18Fixture.store.original.items[1].color),'green');
  checks.push('Injected native-range collapse on toolbar focus survives colour change and creates a green highlight');
  const undo=page.getByRole('button',{name:'撤销标注',exact:true}),redo=page.getByRole('button',{name:'重做标注',exact:true});
  await page.keyboard.press('Control+z');await page.waitForFunction(()=>window.m18Fixture.store.original.items.length===1);
  await page.keyboard.press('Control+Shift+z');await page.waitForFunction(()=>window.m18Fixture.store.original.items.length===2);
  async function clickMark(){const b=await page.locator('.pdf-mark.highlight').last().boundingBox();await page.mouse.click(b.x+b.width/2,b.y+b.height/2);await page.locator('.pdf-mark.is-selected').first().waitFor();}
  await clickMark();await page.keyboard.press('Delete');await page.waitForFunction(()=>window.m18Fixture.store.original.items.length===1);
  await page.keyboard.press('Control+z');await page.waitForFunction(()=>window.m18Fixture.store.original.items.length===2);
  await page.keyboard.press('Control+Shift+z');await page.waitForFunction(()=>window.m18Fixture.store.original.items.length===1);
  await undo.click();await page.waitForFunction(()=>window.m18Fixture.store.original.items.length===2);
  await clickMark();await page.getByRole('button',{name:'删除荧光笔',exact:true}).click();await page.waitForFunction(()=>window.m18Fixture.store.original.items.length===1);
  await undo.click();await page.waitForFunction(()=>window.m18Fixture.store.original.items.length===2);
  checks.push('Ctrl+Z/Ctrl+Shift+Z and toolbar history persist highlights; native click selects an overlapping highlight for Delete/button deletion and restoration');
  await page.getByRole('button',{name:'全屏阅读',exact:true}).click();await page.waitForFunction(()=>!!document.fullscreenElement);
  await selectLine(1);await page.evaluate(()=>getSelection().removeAllRanges());await page.waitForTimeout(100);
  await note.click();await page.getByLabel('批注内容',{exact:true}).fill('选文字后记录疑问。');
  assert.equal(await page.locator('dialog blockquote').count(),0);assert.equal(await page.locator('.annotation-note-dialog').count(),1);
  report.noteDialogStyle=await page.locator('.annotation-note-dialog').evaluate(e=>({background:getComputedStyle(e).backgroundColor,color:getComputedStyle(e).color,width:e.getBoundingClientRect().width}));
  assert.equal(report.noteDialogStyle.background,'rgb(226, 243, 217)');assert(report.noteDialogStyle.width<=502);
  await page.screenshot({path:path.join(out,'fullscreen-note.png')});
  await page.getByRole('button',{name:'保存批注',exact:true}).click();await page.locator('.pdf-note-pin').waitFor();
  checks.push('Fullscreen selected-text note survives native-range collapse, opens and saves');
  await page.locator('.pdf-note-pin').click();await page.getByLabel('批注内容',{exact:true}).fill('修改后的便签内容');await page.getByRole('button',{name:'保存批注',exact:true}).click();
  await page.waitForFunction(()=>window.m18Fixture.store.original.items.some(x=>x.text==='修改后的便签内容'));
  await page.keyboard.press('Control+z');await page.waitForFunction(()=>window.m18Fixture.store.original.items.some(x=>x.text==='选文字后记录疑问。'));
  await page.keyboard.press('Control+Shift+z');await page.waitForFunction(()=>window.m18Fixture.store.original.items.some(x=>x.text==='修改后的便签内容'));
  await page.getByRole('button',{name:'批注 3',exact:true}).click();assert.equal(await page.locator('.pdf-note-card blockquote').count(),0);
  const noteCard=page.locator('.pdf-note-card').filter({hasText:'修改后的便签内容'});
  await noteCard.getByRole('button',{name:'删除',exact:true}).click();await page.waitForFunction(()=>window.m18Fixture.store.original.items.length===2);
  await undo.click();await page.waitForFunction(()=>window.m18Fixture.store.original.items.length===3);
  await redo.click();await page.waitForFunction(()=>window.m18Fixture.store.original.items.length===2);
  await undo.click();await page.waitForFunction(()=>window.m18Fixture.store.original.items.length===3);
  await page.locator('.pdf-note-pin').click();const textarea=page.getByLabel('批注内容',{exact:true});await textarea.press('End');await textarea.pressSequentially('临时');await textarea.press('Control+z');
  assert.equal(await page.evaluate(()=>window.m18Fixture.store.original.items.length),3);assert.equal(await page.evaluate(()=>window.m18Fixture.store.original.items.find(x=>x.kind==='note').text),'修改后的便签内容');
  await page.getByRole('button',{name:'取消',exact:true}).click();
  await page.evaluate(()=>window.m18Fixture.failSave=true);await undo.click();await page.getByRole('alert').filter({hasText:'保存失败'}).waitFor();
  assert.equal(await undo.isEnabled(),true);await page.evaluate(()=>window.m18Fixture.failSave=false);await undo.click();
  await page.waitForFunction(()=>window.m18Fixture.store.original.items.some(x=>x.text==='选文字后记录疑问。'));await redo.click();
  await page.waitForFunction(()=>window.m18Fixture.store.original.items.some(x=>x.text==='修改后的便签内容'));
  checks.push('Note edits/deletions undo and redo with native text editing isolated; failed undo can retry; sidebar and sticky editor hide source quotes');
  await page.getByRole('button',{name:'批注 3',exact:true}).click();
  await selectLine();await page.evaluate(()=>window.m18Fixture.failSave=true);await highlight.click();
  await page.getByRole('alert').filter({hasText:'保存失败'}).waitFor();assert.equal(await highlight.isEnabled(),true);
  await page.evaluate(()=>window.m18Fixture.failSave=false);await highlight.click();
  await page.waitForFunction(()=>window.m18Fixture.store.original.items.length===4);
  checks.push('Failed save keeps the selected quote for a successful retry');
  await selectLine();await page.mouse.click(30,180);assert.equal(await highlight.isDisabled(),true);
  await selectLine();await page.getByRole('button',{name:'下一页',exact:true}).click();assert.equal(await highlight.isDisabled(),true);
  await page.getByRole('button',{name:'上一页',exact:true}).click();
  await selectLine(1);await note.click();await page.getByRole('button',{name:'取消',exact:true}).click();
  await note.click();assert.equal(await page.locator('dialog blockquote').count(),0);assert.equal(await page.getByRole('button',{name:'保存批注',exact:true}).isDisabled(),true);await page.getByRole('button',{name:'取消',exact:true}).click();
  checks.push('Cancelling a note retains the selected quote for reopening');
  await selectLine();await page.getByRole('button',{name:'中文译文',exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('.pdf-controls .primary')?.textContent==='中文译文'&&document.querySelectorAll('.pdf-mark').length===0);
  assert.equal(await highlight.isDisabled(),true);await selectLine(1);await highlight.click();
  await page.waitForFunction(()=>window.m18Fixture.store.translation.items.length===1);
  await page.getByRole('button',{name:'论文解读',exact:true}).click();await page.waitForFunction(()=>document.querySelector('.pdf-controls .primary')?.textContent==='论文解读');
  await selectLine();await highlight.click();await page.waitForFunction(()=>window.m18Fixture.store.analysis.items.length===1);
  checks.push('Blank click/page navigation/document switch clear stale selection; original/translation/analysis marks remain isolated');
  await selectBothPages();await highlight.click();
  await page.waitForFunction(()=>window.m18Fixture.store.analysis.items.length===3);await undo.click();await page.waitForFunction(()=>window.m18Fixture.store.analysis.items.length===1);
  await redo.click();await page.waitForFunction(()=>window.m18Fixture.store.analysis.items.length===3);checks.push('A single two-page selection highlights both pages and undoes/redoes as one persisted edit');
  await selectBothPages();await page.evaluate(()=>window.m18Fixture.failPage2=true);await highlight.click();await page.getByRole('alert').filter({hasText:'第二页保存失败'}).waitFor();
  assert.equal(await page.evaluate(()=>window.m18Fixture.store.analysis.items.length),4);assert((await page.locator('.pdf-mark-tools').innerText()).includes('已选择 1 页'));
  await page.evaluate(()=>window.m18Fixture.failPage2=false);await highlight.click();await page.waitForFunction(()=>window.m18Fixture.store.analysis.items.length===5);
  await undo.click();await page.waitForFunction(()=>window.m18Fixture.store.analysis.items.length===4);await undo.click();await page.waitForFunction(()=>window.m18Fixture.store.analysis.items.length===3);
  checks.push('Partial cross-page save retains only the unsaved page for retry and records both acknowledged edits without duplicate highlights');
  await page.getByRole('button',{name:'批注 3',exact:true}).click();
  await page.getByRole('button',{name:'添加本页笔记',exact:true}).click();await page.getByLabel('批注内容',{exact:true}).fill('便签只显示阅读想法。');await page.getByRole('button',{name:'保存批注',exact:true}).click();
  for(const theme of ['light','dark']){await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);await page.screenshot({path:path.join(out,theme+'.png')});}
  await page.locator('.pdf-note-pin').click();
  for(const theme of ['light','dark']){await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);const palette=await page.locator('.annotation-note-dialog').evaluate(e=>getComputedStyle(e).backgroundColor);assert.equal(palette,theme==='light'?'rgb(226, 243, 217)':'rgb(41, 55, 43)');await page.screenshot({path:path.join(out,'sticky-'+theme+'.png')});}
  await page.getByRole('button',{name:'取消',exact:true}).click();
  await page.getByRole('button',{name:'退出全屏',exact:true}).click();
  assert.deepEqual(errors,[]);report.success=true;
 }catch(error){report.failure=error.message;if(page){report.ui=await page.locator('body').innerText();report.calls=await page.evaluate(()=>window.m18Fixture.calls);await page.screenshot({path:path.join(out,'failure.png')});}throw error;}finally{
  if(browser)await browser.close();if(server)await server.close();await fs.writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2));
 }
}
main().catch(e=>{console.error(e);process.exitCode=1;});
