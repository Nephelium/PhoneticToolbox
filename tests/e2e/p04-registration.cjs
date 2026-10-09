// Shared registration tests: the module implementation is owned by M13.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 let success=false;
 const out=path.join(root,'output/validation/p04-unify','registration-'+Date.now());await fs.mkdir(out,{recursive:true});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0},optimizeDeps:{entries:['tests/p04-preview.html']}});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true}),page=await browser.newPage({viewport:{width:1280,height:800}}),checks=[],errors=[],requests=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
 const open=title=>page.locator('nav').getByRole('button',{name:title,exact:true}).click();
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/p04-preview.html');await page.getByRole('heading',{name:'从这里开始'}).waitFor();assert(!requests.some(url=>url.includes('MandarinIpaPage.vue')||url.includes('ipa-data.json')));
  await open('汉字转国际音标');const mod=page.locator('.mandarin-ipa-page'),input=page.getByLabel('待转换汉字文本');await mod.waitFor();assert.equal(await page.locator('.global-transport').count(),0);assert.equal(await mod.locator('h1').count(),0);
  await input.fill('井井，秋叶 ɑ̃˥');assert.equal(await page.locator('#tab-M13 [aria-label=未保存]').count(),1);
  await open('LPC 谱图');await open('汉字转国际音标');assert.equal(await input.inputValue(),'井井，秋叶 ɑ̃˥');checks.push('M13 is lazy loaded only when opened, has no audio bar or duplicate heading, retains text and dirty marker across tabs');
  const close=()=>page.getByLabel('关闭 汉字转国际音标',{exact:true}).click(),dialog=page.getByRole('dialog',{name:'保存转换草稿？'});
  await close();await dialog.getByRole('button',{name:'取消关闭'}).click();assert.equal(await input.inputValue(),'井井，秋叶 ɑ̃˥');
  await page.evaluate(()=>{window.__set=Storage.prototype.setItem;Storage.prototype.setItem=function(k,v){if(k.includes('mandarin-ipa.v1.'))throw Error('controlled storage failure');return window.__set.call(this,k,v);};});
  await close();await dialog.getByRole('button',{name:'保存草稿并关闭'}).click();await dialog.getByRole('alert').filter({hasText:'保存失败'}).waitFor();await dialog.screenshot({path:path.join(out,'M13-close-failure.png')});assert.equal(await input.inputValue(),'井井，秋叶 ɑ̃˥');
  await page.evaluate(()=>Storage.prototype.setItem=window.__set);await dialog.getByRole('button',{name:'保存草稿并关闭'}).click();await mod.waitFor({state:'detached'});await open('汉字转国际音标');assert.equal(await input.inputValue(),'井井，秋叶 ɑ̃˥');assert.equal(await page.locator('#tab-M13 [aria-label=未保存]').count(),0);
  await input.fill('放弃的文本');await close();await dialog.getByRole('button',{name:'放弃草稿并关闭'}).click();await mod.waitFor({state:'detached'});await open('汉字转国际音标');assert.equal(await input.inputValue(),'井井，秋叶 ɑ̃˥');checks.push('M13 tab cancel, storage failure, save retry, clean reopen and discard restore the right draft');
  for(const [width,height] of [[1280,800],[1920,1080]])for(const theme of ['light','dark'])for(const scale of [70,100,150]){
   await page.setViewportSize({width,height});await page.evaluate(async({theme,scale})=>{document.documentElement.dataset.theme=theme;(await import('/src/state/pageZoom.ts')).setPageScale(scale);},{theme,scale});await page.waitForTimeout(150);const size=await mod.evaluate(el=>({width:el.clientWidth,scroll:el.scrollWidth}));assert(size.scroll<=size.width+2);await page.screenshot({path:path.join(out,`M13-${theme}-${width}-${scale}.png`),animations:'disabled'});
  }checks.push('M13 actual AppShell: 12 viewport/theme/page-scale combinations');assert.deepEqual(errors,[]);success=true;
 }catch(e){await page.screenshot({path:path.join(out,'failure.png')});throw e;}
 finally{await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success,checks,errors,scope:'Windows Chrome, real shared shell; controlled browser storage fault only'},null,2));console.log(out);await browser.close();await server.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
