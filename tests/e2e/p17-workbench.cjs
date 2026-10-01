// Layout-only data. No audio is generated or loaded here.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const out=path.join(root,'output/validation/p17/shared-workbench',String(Date.now()));await fs.mkdir(out,{recursive:true});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0},optimizeDeps:{entries:['tests/p17-workbench.html']}});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
 const page=await browser.newPage({viewport:{width:1920,height:1000}}),errors=[],checks=[];
 page.on('pageerror',e=>errors.push(e.message));
 const frame=()=>page.evaluate(()=>new Promise(requestAnimationFrame));
 const geometry=()=>page.evaluate(()=>Object.fromEntries(['module-frame','module-workbench','workbench-left','workbench-center','workbench-right'].map(c=>{const el=document.querySelector('.'+c);if(!el)return[c,null];const r=el.getBoundingClientRect();return[c,{x:r.x,y:r.y,width:r.width,height:r.height,scrollHeight:el.scrollHeight,clientHeight:el.clientHeight,scrollWidth:el.scrollWidth,clientWidth:el.clientWidth}];})));
 try{
  const url=server.resolvedUrls.local[0]+'tests/p17-workbench.html';await page.goto(url);await page.getByLabel('参数 0',{exact:true}).waitFor();await frame();
  let g=await geometry();assert(g['workbench-left'].x<g['workbench-center'].x&&g['workbench-center'].x<g['workbench-right'].x);assert(g['module-frame'].scrollHeight<=g['module-frame'].clientHeight+1);assert.equal(g['module-workbench'].y,4);checks.push({name:'three columns, top inset 4px, no page overflow',geometry:g});
  await page.getByLabel('参数 0',{exact:true}).fill('保留编辑');const before=g['workbench-center'].width;
  await page.getByRole('button',{name:'收起任务与记录'}).click();await frame();g=await geometry();assert(g['workbench-center'].width>before+200);assert.equal(await page.getByLabel('参数 0',{exact:true}).inputValue(),'保留编辑');assert.equal(await page.getByLabel('保存测试状态').isVisible(),false);
  await page.reload();await page.getByRole('button',{name:'展开任务与记录'}).waitFor();await page.getByRole('button',{name:'展开任务与记录'}).focus();await page.keyboard.press('Enter');await page.getByLabel('保存测试状态').waitFor();checks.push({name:'collapse preserves mounted state and persists; keyboard reopens'});
  const handle=page.getByRole('separator',{name:'输入与参数宽度'});await handle.focus();const old=+(await handle.getAttribute('aria-valuenow'));await page.keyboard.press('ArrowRight');await frame();assert.equal(+(await handle.getAttribute('aria-valuenow')),old+10);checks.push({name:'shared keyboard resize'});
  await page.screenshot({path:path.join(out,'wide.png')});await page.evaluate(()=>document.documentElement.dataset.theme='dark');await page.waitForTimeout(180);await page.screenshot({path:path.join(out,'dark.png')});
  const basePlot=await page.locator('.scientific-plot>svg').evaluate(el=>({height:el.clientHeight,font:getComputedStyle(el.querySelector('text')).fontSize}));
  for(const viewport of [{width:2560,height:1360},{width:3840,height:2080}]){
   await page.setViewportSize(viewport);await frame();await frame();
   const result=await page.locator('.scientific-plot>svg').evaluate(el=>({height:el.clientHeight,width:el.clientWidth,viewBox:el.viewBox.baseVal.height,font:getComputedStyle(el.querySelector('text')).fontSize}));
   assert(result.height>basePlot.height+(viewport.height-1000)*.9);assert.equal(result.height,result.viewBox);assert.equal(result.font,basePlot.font);g=await geometry();assert(g['module-frame'].scrollHeight<=g['module-frame'].clientHeight+1);
   checks.push({name:'larger viewport expands true plotting area while preserving font',viewport,plot:result});await page.screenshot({path:path.join(out,'large-'+viewport.width+'.png')});
  }
  await page.setViewportSize({width:1280,height:720});await frame();g=await geometry();assert(g['workbench-right'].y>g['workbench-center'].y);await page.getByLabel('保存测试状态').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,'small.png')});checks.push({name:'narrow window moves history below and keeps actions reachable',geometry:g});
  await page.setViewportSize({width:820,height:600});await frame();g=await geometry();assert(g['workbench-center'].y>g['workbench-left'].y);checks.push({name:'small single column',geometry:g});
  await page.evaluate(()=>window.__p17.noLeft.value=true);await frame();await page.getByLabel('中央操作').scrollIntoViewIfNeeded();await page.evaluate(()=>window.__p17.noRight.value=true);await frame();assert.equal(await page.locator('.workbench-right').count(),0);checks.push({name:'optional side panes remain usable'});
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'report.json'),JSON.stringify({checks,errors},null,2));console.log(JSON.stringify({out,checks:checks.length,errors}));
 }catch(error){await page.screenshot({path:path.join(out,'failed.png')});throw error;}
 finally{await browser.close();await server.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
