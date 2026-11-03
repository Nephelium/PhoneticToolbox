// Development browser checks only. Final Windows and authenticated Web acceptance
// remain separate, with their own artifact identities and real host boundaries.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const out=path.join(root,'output/validation/app-i18n-browser',String(Date.now()));await fs.mkdir(out,{recursive:true});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0},cacheDir:path.join(out,'vite-cache'),optimizeDeps:{entries:['index.html']}});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
 const context=await browser.newContext({viewport:{width:1920,height:1080},locale:'ja-JP'}),page=await context.newPage(),errors=[],checks=[],requests=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('request',request=>requests.push(request.url()));
 const catalogs={};for(const lang of ['zh-CN','en','ja'])catalogs[lang]=JSON.parse(await fs.readFile(path.join(root,'frontend/src/i18n/locales',lang,'components.json'),'utf8'));
 const select=page.locator('.topbar .language-switch select');
 async function language(code){await select.selectOption(code);await page.waitForFunction(value=>document.documentElement.lang===value,code);await page.waitForFunction(()=>!document.querySelector('.topbar .language-switch select:disabled'));}
 async function layout(name){
  const value=await page.evaluate(()=>({width:innerWidth,documentWidth:document.documentElement.scrollWidth,body:document.body.innerText,buttons:[...document.querySelectorAll('.topbar button,.topbar select')].filter(e=>e.getBoundingClientRect().width).map(e=>{const r=e.getBoundingClientRect();return {x:r.x,right:r.right,width:r.width};})}));
  assert(value.documentWidth<=value.width+1,name+' document overflow');assert(!value.body.includes('[object Object]'),name+' object text');
  for(const item of value.buttons)assert(item.x>=-1&&item.right<=value.width+1,name+' inaccessible top control');
  checks.push({name,width:value.width,documentWidth:value.documentWidth});
 }
 try{
  await page.goto(server.resolvedUrls.local[0]);await select.waitFor();assert.equal(await page.locator('html').getAttribute('lang'),'zh-CN');checks.push({name:'Chinese default with Japanese browser locale'});
  const announcement=page.locator('dialog[open] .announcements-panel');
  if(await announcement.isVisible())await page.locator('dialog[open]').getByRole('button',{name:catalogs['zh-CN']['components.close_dialog'],exact:true}).click();
  await page.locator('.nav-item[data-module-id="M04"]').click();await page.locator('#tab-M04[aria-selected="true"]').waitFor();await page.locator('.lpc-page').waitFor({state:'visible'});
  const owner=page.locator('#main-content');await owner.evaluate(e=>e.dataset.i18nProbe='preserve-mounted-workspace');
  const loadedBefore=requests.filter(url=>url.includes('/i18n/generated/en.ts')).length;
  for(const code of ['en','ja','zh-CN','en','ja']){await language(code);assert.equal(await owner.getAttribute('data-i18n-probe'),'preserve-mounted-workspace');assert.equal(await page.locator('#tab-M04').getAttribute('aria-selected'),'true');}
  const englishLoads=requests.filter(url=>url.includes('/i18n/generated/en.ts')).length-loadedBefore;assert.equal(englishLoads,1);checks.push({name:'same workspace stays mounted; English catalog loaded once',englishLoads});
  const divider=page.locator('.lpc-page .panel-resize-handle:not([hidden])').first();await divider.waitFor();
  await divider.evaluate(e=>e.dataset.i18nProbe='same-divider');await divider.focus();await page.keyboard.press('ArrowRight');
  const width=await divider.getAttribute('aria-valuenow'),names=[];
  for(const code of ['en','ja','zh-CN']){
   await language(code);assert.equal(await divider.getAttribute('data-i18n-probe'),'same-divider');
   assert.equal(await divider.getAttribute('aria-valuenow'),width);names.push(await divider.getAttribute('aria-label'));
   assert.equal(await divider.getAttribute('title'),catalogs[code]['components.panel_resize_help']);
   assert.equal(await divider.getAttribute('aria-valuetext'),catalogs[code]['components.panel_width_pixels'].replace('{width}',width));
  }
  assert.equal(new Set(names).size,3);checks.push({name:'existing divider updates accessible labels and keeps its resized width',width});
  await page.locator('.utility-grid button').first().click();await page.locator('#settings-tab-language').click();await page.locator('#settings-panel-language .language-switch select').selectOption('en');await page.waitForFunction(()=>document.documentElement.lang==='en');assert.equal(await select.inputValue(),'en');checks.push({name:'settings and home controls share the same language'});
  for(const code of ['en','ja','zh-CN']){
   await language(code);await page.locator('#settings-tab-appearance').click();
   assert.equal(await page.locator('.theme-mode-cards button').first().getAttribute('aria-label'),catalogs[code]['components.follow_system']);
   for(const width of [820,1280,1920]){await page.setViewportSize({width,height:1000});await layout('appearance '+code);}
   await page.screenshot({path:path.join(out,code+'-settings.png')});
  }
  await language('ja');await page.reload();await select.waitFor();assert.equal(await select.inputValue(),'ja');checks.push({name:'explicit language restored after reload'});
  await page.waitForFunction(()=>document.documentElement.dataset.fontLocale==='ja');
  const font=await page.evaluate(()=>getComputedStyle(document.documentElement).getPropertyValue('--font'));
  assert(/Yu Gothic|Meiryo|Hiragino|Noto Sans (?:CJK )?JP|Source Han Sans JP/.test(font));checks.push({name:'Japanese default uses an installed Japanese font',font});
  await page.locator('.utility-grid button').first().click();await page.locator('#settings-tab-fonts').click();
  const size=page.locator('.font-settings input[type="number"]').first();await size.fill('16');
  for(const code of ['en','ja','zh-CN']){await language(code);assert.equal(await size.inputValue(),'16');}
  checks.push({name:'language switches preserve unapplied font edits'});
  assert.deepEqual(errors,[]);
  await fs.writeFile(path.join(out,'report.json'),JSON.stringify({stage:'development-preview',checks,errors,requests},null,2));console.log(JSON.stringify({out,checks:checks.length,errors}));
 }catch(error){await page.screenshot({path:path.join(out,'failed.png')});await fs.writeFile(path.join(out,'failed.json'),JSON.stringify({error:String(error),errors,checks},null,2));throw error;}
 finally{await context.close();await browser.close();await server.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
