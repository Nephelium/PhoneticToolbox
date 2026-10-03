// M13 / P19: settings stay left in both layouts; owned Chrome geometry and interaction checks.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const out=path.join(root,'output/validation/m13-layout',String(Date.now()));await fs.mkdir(out,{recursive:true});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0},optimizeDeps:{entries:['tests/m13-live.html']}});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
 const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[],checks=[];
 page.on('pageerror',e=>errors.push(e.message));
 const rect=selector=>page.locator(selector).boundingBox();
 const idle=()=>page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
 async function popupCheck(){
  await idle();const p=await rect('.m13-variants'),a=await rect('.m13-ambiguous[aria-expanded=true]');
  const view=page.viewportSize();assert(p&&a);assert(Math.abs(p.width-p.height)<2,'square');
  assert(p.x>=0&&p.y>=0&&p.x+p.width<=view.width+1&&p.y+p.height<=view.height+1,JSON.stringify({p,a,view}));
  const dx=Math.max(a.x-p.x-p.width,p.x-a.x-a.width,0),dy=Math.max(a.y-p.y-p.height,p.y-a.y-a.height,0);
  assert(dx<=10&&dy<=10,'popup must remain adjacent to its character');return {p,a};
 }
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/m13-live.html');
  await page.getByLabel('待转换汉字文本').fill('春江花月夜\n春江潮水连海平，海上明月共潮生。银行行行');
  for(const theme of ['light','dark'])for(const scale of [0.7,1,1.5])for(const layout of ['左右排布','上下排布']){
   await page.evaluate(({theme,scale})=>{document.documentElement.dataset.theme=theme;document.documentElement.style.zoom=String(scale);document.documentElement.style.setProperty('--page-scale',String(scale));},{theme,scale});
   await page.getByLabel(layout,{exact:true}).check();await idle();
   const input=await rect('.m13-input-section'),result=await rect('.m13-result-section'),settings=await rect('.m13-settings-section');
   assert(settings.x+settings.width<=Math.min(input.x,result.x)+1,'controls stay left');
   if(layout==='上下排布'){assert(Math.abs(input.x-result.x)<1);assert(result.y>=input.y+input.height);}
   else assert(result.x>=input.x+input.width);
   assert.equal(await page.locator('.m13-settings-section input[type=range]').count(),4);
   assert.equal(await page.locator('.mandarin-ipa-page .module-toolbar button').count(),3);
   await page.locator('.m13-ambiguous').first().click();await popupCheck();
   await page.keyboard.press('Escape');assert.equal(await page.locator('.m13-variants').count(),0);
   assert(await page.locator('.m13-ambiguous').first().evaluate(e=>e===document.activeElement));
   checks.push({theme,scale,layout});
  }
  await page.evaluate(()=>{document.documentElement.style.zoom='1';document.documentElement.style.setProperty('--page-scale','1');});
  await page.getByLabel('上下排布',{exact:true}).check();await page.locator('.m13-ambiguous').first().click();await popupCheck();
  await page.screenshot({path:path.join(out,'stacked-dark-popup.png'),fullPage:true});
  await page.locator('.m13-variants > button').last().click();assert.equal(await page.locator('.m13-variants').count(),0);
  await page.locator('.m13-ambiguous').first().click();await page.getByLabel('待转换汉字文本').click();assert.equal(await page.locator('.m13-variants').count(),0);
  await page.locator('.m13-ambiguous').first().click();await page.evaluate(()=>{document.documentElement.style.zoom='1.5';document.documentElement.style.setProperty('--page-scale','1.5');});await popupCheck();await page.keyboard.press('Escape');
  await page.evaluate(()=>{document.documentElement.style.zoom='1';document.documentElement.style.setProperty('--page-scale','1');});
  await page.getByLabel('待转换汉字文本').fill('行\n'.repeat(30));await page.locator('.m13-ambiguous').nth(12).click();await popupCheck();
  await page.locator('.mandarin-ipa-page').evaluate(e=>e.scrollTop+=30);await idle();
  if(await page.locator('.m13-variants').count())await popupCheck();
  await page.keyboard.press('Escape');await page.getByLabel('待转换汉字文本').fill('银行花');
  await page.goto(server.resolvedUrls.local[0]);await page.locator('.nav-item[title="普通话转 IPA"]').click();
  await page.getByLabel('待转换汉字文本').fill('银行花');
  const collapse=page.getByRole('button',{name:'收起侧栏',exact:true});if(await collapse.count())await collapse.click();
  const icon=page.locator('.nav-item[title="普通话转 IPA"] .ipa-icon');assert(await icon.isVisible());assert((await icon.boundingBox()).width>0);
  await page.setViewportSize({width:800,height:700});assert(await icon.isVisible());
  await page.getByLabel('上下排布',{exact:true}).check();await idle();
  const narrow=await rect('.m13-settings-section'),left=await rect('.m13-result-section');assert(narrow.x+narrow.width<=left.x+1);
  await page.setViewportSize({width:1440,height:1000});await page.getByLabel('左右排布',{exact:true}).check();
  await page.evaluate(()=>document.documentElement.dataset.theme='light');await page.locator('.m13-ambiguous').first().click();await popupCheck();
  await page.screenshot({path:path.join(out,'side-light-collapsed-popup.png'),fullPage:true});
  await page.keyboard.press('Escape');
  const symbols=page.locator('.nav-item[title="国际音标 Plus"] svg path');
  assert.equal(await symbols.count(),1,'M17 uses a distinct symbol-keyboard SVG');
  assert.equal(await page.locator('.nav-item[title="普通话转 IPA"] .ipa-icon').textContent(),'æ');
  for(const layout of ['左右排布','上下排布']){
   await page.getByLabel(layout,{exact:true}).check();await idle();
   const handle=page.getByRole('separator',{name:'转换与排版宽度',exact:true});
   const before=(await rect('.m13-settings-section')).width;
   await handle.focus();await page.keyboard.press('ArrowRight');await idle();
   assert.equal(Math.round((await rect('.m13-settings-section')).width),Math.round(before+10),'left-boundary keyboard resize');
   const box=await handle.boundingBox();assert(box);
   await page.mouse.move(box.x+box.width/2,box.y+30);await page.mouse.down();await page.mouse.move(box.x+box.width/2+20,box.y+30,{steps:4});await page.mouse.up();await idle();
   assert.equal(Math.round((await rect('.m13-settings-section')).width),Math.round(before+30),'rightward drag widens left controls');
   assert.equal(await page.getByLabel('待转换汉字文本').inputValue(),'银行花');
   checks.push({layout,resize:'keyboard and drag keep left-side direction and text'});
  }
  const remembered=(await rect('.m13-settings-section')).width;
  await page.getByRole('button',{name:/保存本机草稿/}).click();
  await page.reload();await page.locator('.nav-item[title="普通话转 IPA"]').click();await idle();
  assert.equal(Math.round((await rect('.m13-settings-section')).width),Math.round(remembered),'width survives reload');
  assert(await page.getByLabel('上下排布',{exact:true}).isChecked());
  await page.getByRole('button',{name:'转换设置',exact:true}).click();await idle();assert.equal(await page.locator('.m13-settings-section:visible').count(),0);
  await page.getByRole('button',{name:'转换设置',exact:true}).click();await idle();
  assert.equal(Math.round((await rect('.m13-settings-section')).width),Math.round(remembered));
  checks.push({persistence:'layout, text, width and collapsed settings round trip',icons:'M13 æ / M17 keyboard'});
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'report.json'),JSON.stringify({checks,errors,extra:['select','outside click','Escape focus','live zoom','lower viewport edge','scroll','collapsed icon','narrow viewport']},null,2));console.log(out);
 }catch(e){await page.screenshot({path:path.join(out,'failure.png'),fullPage:true});console.error(out);throw e;}
 finally{await browser.close();await server.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
