// Owned Chrome + simulated catalogue. This does not replace native PDF/filesystem evidence.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');const root=path.resolve(__dirname,'../..');
async function main(){
 const out=process.argv[2],scratch=process.argv[3];assert(out&&scratch);let server,browser,page;const checks=[],errors=[];
 try{
  const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
  server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{noDiscovery:true,include:['vue']}});await server.listen();
  const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  browser=await chromium.launchPersistentContext(path.join(scratch,'chrome-profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1440,height:900}});page=await browser.newPage();page.on('pageerror',e=>errors.push(e.message));
  await page.goto(server.resolvedUrls.local[0]+'tests/m18-r3.html');await page.getByRole('button',{name:'获取往期',exact:true}).click();
  const rows=page.locator('.archive-row:not(.obtained)');await page.waitForFunction(()=>document.querySelectorAll('.archive-row:not(.obtained)').length===3&&!document.querySelector('.archive-row input').disabled);
  await page.getByRole('button',{name:'09 月',exact:true}).click();assert.equal(await rows.count(),2);
  await page.getByLabel('全选当前筛选',{exact:true}).check();await page.getByRole('button',{name:/2025 年/}).click();assert.equal(await rows.count(),1);
  await rows.locator('input').check();await rows.locator('.archive-titles').hover();await page.locator('.abstract-tooltip').first().waitFor({state:'visible'});assert((await page.locator('.abstract-tooltip').first().innerText()).includes('English abstract d'));
  await page.getByRole('button',{name:'获取已选 3 篇',exact:true}).click();await page.waitForFunction(()=>document.querySelectorAll('.archive-row:not(.obtained)').length===0);
  assert.deepEqual(await page.evaluate(()=>window.m18Calls.find(c=>c.operation==='downloadSelected').args.ids),['c','b','d']);
  await page.getByLabel(/显示已获取/).check();assert.equal(await page.locator('.obtained input:disabled').count(),2);
  await page.getByRole('button',{name:'关闭',exact:true}).click();await page.getByRole('button',{name:'获取往期',exact:true}).click();await page.waitForFunction(()=>document.querySelectorAll('.archive-row:not(.obtained)').length===1&&!document.querySelector('.archive-row input').disabled);
  assert((await rows.innerText()).includes('中文题名 a'));checks.push('year/month selection, cross-period select-all, bilingual hover abstracts, exact selected IDs and next-open exclusion; acquired disabled');
  await page.screenshot({path:path.join(out,'archive-light.png')});await page.getByRole('button',{name:'关闭',exact:true}).click();
  await page.getByRole('button',{name:'6 小时',exact:true}).click();await page.getByLabel('自动检查间隔',{exact:true}).fill('12');await page.getByRole('button',{name:'关闭对话框',exact:true}).click();assert.equal(await page.getByRole('button',{name:'6 小时',exact:true}).count(),1);
  await page.getByRole('button',{name:'6 小时',exact:true}).click();assert.equal(await page.getByLabel('自动检查间隔',{exact:true}).inputValue(),'6');await page.getByLabel('自动检查间隔',{exact:true}).fill('12');await page.getByRole('button',{name:'保存间隔',exact:true}).click();await page.getByRole('button',{name:'12 小时',exact:true}).waitFor();
  await page.getByLabel('自动获取',{exact:true}).uncheck();assert.equal(await page.evaluate(()=>window.m18State.autoReceive),false);checks.push('interval dialog dismissal preserves saved interval; explicit save and disabling use native preference protocol');
  await page.getByRole('tab',{name:/我的 PDF/}).click();assert.equal(await page.locator('.paper-card').count(),0);await page.getByRole('tab',{name:/服务器推荐/}).click();assert.equal(await page.locator('.paper-card').count(),4);checks.push('personal/recommended separation and previously acquired historical papers remain in server library');
  for(const theme of ['light','dark'])for(const width of [1440,1100]){await page.setViewportSize({width,height:800});await page.evaluate(theme=>document.documentElement.dataset.theme=theme,theme);assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));await page.screenshot({path:path.join(out,`${width}-${theme}.png`)});}
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success:true,evidence:'simulated catalogue / isolated standalone Chrome',checks,errors},null,2));console.log(JSON.stringify({out,checks}));
 }finally{if(browser)await browser.close();if(server)await server.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
