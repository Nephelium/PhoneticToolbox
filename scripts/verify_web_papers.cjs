// Owned synthetic PostgreSQL/HTTP fixture or private Linux instance over SSH.
const fs=require('node:fs/promises'),path=require('node:path'),os=require('node:os'),assert=require('node:assert/strict');
(async()=>{
 const {chromium}=require(path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const origin=process.env.PTB_M18_ORIGIN,out=process.env.PTB_M18_OUTPUT,fixture=process.env.PTB_M18_PDF;
 assert(origin?.startsWith('http://127.0.0.1:'));assert(out&&fixture);await fs.mkdir(out,{recursive:true});
 const browser=await chromium.launch({headless:false,executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe'});
 const context=await browser.newContext({viewport:{width:1440,height:1000}}),page=await context.newPage(),errors=[],checks=[];
 await context.addCookies([{name:'ptb-dev-session',value:process.env.PTB_M18_COOKIE,url:origin,httpOnly:true,sameSite:'Strict'}]);
 page.setDefaultTimeout(90000);page.on('pageerror',e=>errors.push(e.message));
 const click=name=>page.getByRole('button',{name,exact:true}).click();
 try{
  await page.goto(origin+'/app/');await page.locator('#tool-sidebar').getByRole('button',{name:'语音学论文精读',exact:true}).click();
  await page.getByRole('tab',{name:/我的 PDF/}).click();await page.getByRole('button',{name:'导入 PDF',exact:true}).waitFor({state:'visible'});
  await page.waitForFunction(()=>!document.querySelector('.paper-actions button.primary')?.disabled);
  await click('导入 PDF');const chooser=page.waitForEvent('filechooser');await click('选择 PDF');await(await chooser).setFiles(fixture);
  await page.locator('.paper-sheet').first().waitFor();assert.equal(await page.locator('.paper-reading .module-seal-content[inert]').count(),0);checks.push('same workbench / no seal / import / real Qt PDF image');
  await click('本页文字');await page.locator('pre.paper-text').filter({hasText:'Paper reading fixture'}).waitFor();await click('本页文字');
  await page.getByLabel('下一页',{exact:true}).click();assert.equal(await page.getByLabel('页码',{exact:true}).inputValue(),'2');
  await page.locator('.paper-viewport').press('ArrowLeft');assert.equal(await page.getByLabel('页码',{exact:true}).inputValue(),'1');
  await page.getByLabel('阅读缩放').selectOption('125');await page.locator('.pdf-text-line').first().waitFor({state:'attached'});checks.push('continuous pages / text / page navigation / keyboard / zoom');
  await page.evaluate(()=>{const node=document.querySelector('.pdf-text-line')?.firstChild;if(!node)throw Error('missing real PDF text');const range=document.createRange();range.setStart(node,0);range.setEnd(node,5);const selection=getSelection();selection.removeAllRanges();selection.addRange(range);document.dispatchEvent(new Event('selectionchange'));});
  await page.waitForFunction(()=>[...document.querySelectorAll('button')].some(b=>b.textContent==='荧光笔'&&!b.disabled));await click('荧光笔');
  await page.getByRole('button',{name:'批注 1',exact:true}).click();await click('添加本页笔记');await page.getByLabel('批注内容',{exact:true}).fill('中文笔记：核对实验条件。');await click('保存批注');
  await page.getByText('中文笔记：核对实验条件。',{exact:true}).waitFor();checks.push('text selection / highlight / Chinese page note / persistence');
  await click('导出 PDF');await page.getByRole('checkbox',{name:/包含此文档/}).check();const download=page.waitForEvent('download');await click('选择位置并导出');const exported=await download;const raw=await fs.readFile(await exported.path());assert.equal(raw.subarray(0,5).toString(),'%PDF-');await fs.writeFile(process.env.PTB_M18_EXPORT,raw);await exported.delete();checks.push('annotated PDF downloaded through authenticated content endpoint');
  await page.reload();await page.locator('#tool-sidebar').getByRole('button',{name:'语音学论文精读',exact:true}).click();await page.getByRole('tab',{name:/我的 PDF/}).click();await page.locator('.paper-sheet').first().waitFor();if(await page.getByRole('button',{name:'批注 2',exact:true}).getAttribute('aria-expanded')!=='true')await page.getByRole('button',{name:'批注 2',exact:true}).click();await page.getByText('中文笔记：核对实验条件。',{exact:true}).waitFor();checks.push('refresh restores library and both annotations');
  await page.locator('.paper-card').filter({hasText:'测试论文'}).click({button:'right'});await page.getByRole('menuitem',{name:'移入回收站',exact:true}).click();await page.getByRole('dialog').getByRole('button',{name:'移入回收站',exact:true}).click();await click('回收站');await page.getByRole('dialog').getByRole('button',{name:'恢复',exact:true}).click();await page.getByRole('dialog').getByRole('button',{name:'关闭',exact:true}).click();await page.locator('.paper-sheet').first().waitFor();checks.push('context menu / confirmation / recycle / restore');
  for(const theme of ['light','dark']){await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);await page.screenshot({path:path.join(out,'reader-'+theme+'.png')});}
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'browser.json'),JSON.stringify({success:true,checks,errors},null,2));console.log(JSON.stringify(checks));
 }catch(e){await page.screenshot({path:path.join(out,'failure.png')});await fs.writeFile(path.join(out,'failure.txt'),await page.locator('body').innerText());throw e;}
 finally{await context.close();await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
