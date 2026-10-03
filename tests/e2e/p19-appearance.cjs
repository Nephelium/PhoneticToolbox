// P19: owned headless Chrome + Vite, no user profile, database or devices.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const out=path.join(root,'output/validation/p19','chrome-'+Date.now());await fs.mkdir(out,{recursive:true});
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0}});await server.listen();
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
 const context=await browser.newContext({viewport:{width:1920,height:1080},colorScheme:'light'}),page=await context.newPage(),checks=[],errors=[],layouts=[];
 page.on('pageerror',e=>errors.push(e.message));
 const base=server.resolvedUrls.local[0];
 try{
  await page.goto(base);await page.getByRole('button',{name:'设置',exact:true}).click();
  assert.equal(await page.getByLabel('配色方案').inputValue(),'everforest');
  await page.getByText('默认使用 Everforest。也欢迎试试更多配色，选一款自己喜欢的，让工作台更合心意。',{exact:true}).waitFor();
  await page.waitForFunction(()=>document.documentElement.style.getPropertyValue('--font').includes('SimSun'));
  await page.evaluate(async()=>{await document.fonts.load('12px "JetBrains Mono"');await document.fonts.ready;});
  assert.equal(await page.getByLabel('中文字体',{exact:true}).inputValue(),'SimSun');assert.equal(await page.getByLabel('英文与数字字体',{exact:true}).inputValue(),'Times New Roman');assert.equal(await page.getByLabel('代码与等宽字体',{exact:true}).inputValue(),'JetBrains Mono');
  const loaded=await page.evaluate(()=>[...document.fonts].filter(f=>f.family==='JetBrains Mono').map(f=>f.status));assert(loaded.includes('loaded'));
  const width=await page.evaluate(()=>{const c=document.createElement('canvas').getContext('2d');c.font='14px "JetBrains Mono"';return [c.measureText('iiiiii').width,c.measureText('WWWWWW').width];});assert(Math.abs(width[0]-width[1])<.01);checks.push('new defaults and real bundled monospace font loaded');
  const ids=await page.getByLabel('配色方案').locator('option').evaluateAll(es=>es.map(e=>e.value));assert.equal(ids.length,29);assert(!ids.includes('ptb'));
  for(const id of ids)for(const mode of ['浅色','深色']){
   await page.getByLabel('配色方案').selectOption(id);await page.getByRole('button',{name:mode,exact:true}).click();
   const value=await page.evaluate(()=>{const s=getComputedStyle(document.documentElement);return {palette:document.documentElement.dataset.palette,mode:document.documentElement.dataset.theme,bg:s.getPropertyValue('--app').trim(),color:s.getPropertyValue('--text').trim(),option:getComputedStyle(document.querySelector('#palette-choice option')).backgroundColor};});
   assert.equal(value.palette,id);assert.equal(value.mode,mode==='浅色'?'light':'dark');assert.notEqual(value.bg,value.color);assert.notEqual(value.option,'rgba(0, 0, 0, 0)');
  }checks.push('all 29 palettes x 2 modes and opaque option surfaces');
  await page.getByLabel('配色方案').selectOption('everforest');await page.getByRole('button',{name:'跟随系统',exact:true}).click();await page.emulateMedia({colorScheme:'dark'});await page.waitForFunction(()=>document.documentElement.dataset.theme==='dark');
  assert.equal(await page.evaluate(()=>getComputedStyle(document.documentElement).getPropertyValue('--app').trim()),'#2d353b');await page.emulateMedia({colorScheme:'light'});await page.waitForFunction(()=>document.documentElement.dataset.theme==='light');checks.push('system changes select the matching Everforest variant');
  await page.getByLabel('中文字体',{exact:true}).fill('KaiTi');await page.getByLabel('配色方案').selectOption('catppuccin');await page.getByRole('button',{name:'深色',exact:true}).click();await page.getByRole('button',{name:'参数显示',exact:true}).click();await page.getByRole('tab',{name:'设置'}).click();assert.equal(await page.getByLabel('中文字体',{exact:true}).inputValue(),'KaiTi');
  await page.getByRole('button',{name:'关闭 设置',exact:true}).click();await page.getByRole('button',{name:'取消关闭',exact:true}).click();assert.equal(await page.getByLabel('中文字体',{exact:true}).inputValue(),'KaiTi');await page.getByRole('button',{name:'取消字体编辑',exact:true}).click();checks.push('theme and tab switching preserve unapplied font draft and close guard');
  await page.reload();await page.getByRole('button',{name:'设置',exact:true}).click();assert.equal(await page.getByLabel('配色方案').inputValue(),'catppuccin');await page.waitForFunction(()=>document.documentElement.dataset.theme==='dark');checks.push('palette and mode survive reload');
  for(const [saved,expected] of [['ptb','everforest'],['matrix','matrix']]){
   await page.evaluate(saved=>localStorage.setItem('ptb.v3.palette',JSON.stringify(saved)),saved);await page.reload();await page.getByRole('button',{name:'设置',exact:true}).click();
   assert.equal(await page.getByLabel('配色方案').inputValue(),expected);assert.equal(await page.evaluate(()=>document.documentElement.dataset.theme),'dark');
   assert.equal(await page.evaluate(()=>JSON.parse(localStorage.getItem('ptb.v3.palette'))),expected);
  }checks.push('legacy PTB migrates to Everforest; explicit Matrix and dark mode persist');
  for(const [w,h] of [[1920,1080],[1440,900],[1024,768],[600,900],[390,844]])for(const scale of [70,100,150]){
   await page.setViewportSize({width:w,height:h});await page.evaluate(async scale=>(await import('/src/state/pageZoom.ts')).setPageScale(scale),scale);
   await page.waitForTimeout(80);
   const bounds=await page.evaluate(()=>{const r=document.querySelector('.settings-columns'),f=document.querySelector('.font-settings');return {width:document.documentElement.clientWidth,scroll:document.documentElement.scrollWidth,formScroll:f.scrollWidth,formWidth:f.clientWidth,columns:getComputedStyle(r).gridTemplateColumns};});
   assert(bounds.scroll<=bounds.width+2,JSON.stringify({w,scale,bounds}));assert(bounds.formScroll<=bounds.formWidth+2,JSON.stringify({w,scale,bounds}));layouts.push({w,h,scale,...bounds});
  }checks.push('15 width/zoom combinations remain reachable without horizontal overflow');
  await page.setViewportSize({width:1920,height:1080});await page.evaluate(async()=>(await import('/src/state/pageZoom.ts')).setPageScale(100));await page.getByLabel('配色方案').selectOption('everforest');
  for(const [label,file] of [['浅色','settings-everforest-light.png'],['深色','settings-everforest-dark.png']]){await page.getByRole('button',{name:label,exact:true}).click();await page.waitForTimeout(300);await page.screenshot({path:path.join(out,file),fullPage:true});}
  await page.setViewportSize({width:390,height:844});await page.screenshot({path:path.join(out,'settings-narrow.png'),fullPage:true});
  // A fresh profile with legacy saved choices must not acquire the new defaults.
  await page.evaluate(async()=>{const f=await import('/src/state/fonts.ts');await f.setFonts({...f.preferences.value,zh:'KaiTi',latin:'Georgia',mono:'Consolas'});});await page.reload();await page.getByRole('button',{name:'设置',exact:true}).click();assert.equal(await page.getByLabel('中文字体',{exact:true}).inputValue(),'KaiTi');assert.equal(await page.getByLabel('代码与等宽字体',{exact:true}).inputValue(),'Consolas');checks.push('saved explicit user font choices preserved');
  const mono=page.getByLabel('代码与等宽字体',{exact:true});
  await mono.selectOption('__custom__');await page.getByLabel('自定义代码字体',{exact:true}).fill('Georgia');await page.getByRole('button',{name:'应用字体',exact:true}).click();await page.getByRole('status').filter({hasText:'字体已应用'}).waitFor();
  await page.reload();await page.getByRole('button',{name:'设置',exact:true}).click();assert.equal(await mono.inputValue(),'Georgia');assert.equal(await mono.evaluate(e=>e.tagName),'SELECT');
  assert.equal(await mono.locator('option').first().textContent(),'JetBrains Mono（内置）');
  await page.getByRole('button',{name:'预览字体',exact:true}).click();await page.getByRole('status').filter({hasText:'预览已更新'}).waitFor();
  assert((await page.locator('.font-code-preview').evaluate(e=>getComputedStyle(e).fontFamily)).includes('Georgia'));
  await mono.selectOption({label:'JetBrains Mono（内置）'});await page.getByRole('button',{name:'应用字体',exact:true}).click();await page.getByRole('status').filter({hasText:'字体已应用'}).waitFor();
  assert((await page.locator('.font-code-preview').evaluate(e=>getComputedStyle(e).fontFamily)).includes('JetBrains Mono'));
  await page.reload();await page.getByRole('button',{name:'设置',exact:true}).click();
  assert.equal(await mono.inputValue(),'JetBrains Mono');assert.equal(await page.getByLabel('中文字体',{exact:true}).inputValue(),'KaiTi');assert.equal(await page.getByLabel('英文与数字字体',{exact:true}).inputValue(),'Georgia');
  await page.waitForFunction(()=>getComputedStyle(document.querySelector('.font-code-preview')).fontFamily.includes('JetBrains Mono'));checks.push('custom Georgia saves; unfiltered native select chooses bundled Mono, previews, applies and survives reload');
  await mono.selectOption('__custom__');await page.getByLabel('自定义代码字体',{exact:true}).fill('P19 Missing Font');await page.getByRole('button',{name:'应用字体',exact:true}).click();await page.getByRole('status').filter({hasText:'不可用'}).waitFor();
  assert.equal(await page.evaluate(async()=>(await import('/src/state/fonts.ts')).preferences.value.mono),'JetBrains Mono');
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=document.documentElement.clientWidth+2));
  await page.getByRole('button',{name:'取消字体编辑',exact:true}).click();assert.equal(await mono.inputValue(),'JetBrains Mono');assert.equal(await page.getByLabel('自定义代码字体',{exact:true}).count(),0);checks.push('custom missing mono rejects, preserves applied font and cancels cleanly in narrow window');
  const missingContext=await browser.newContext();
  await missingContext.addInitScript(()=>{
   const Native=window.FontFace;
   window.FontFace=function(family,source,options){const face=new Native(family,source,options);if(/local\("(SimSun|Times New Roman)"\)/.test(source))Object.defineProperty(face,'load',{value:()=>Promise.reject(new Error('P19 controlled missing default font'))});return face;};
   window.FontFace.prototype=Native.prototype;
  });
  const missingPage=await missingContext.newPage();await missingPage.goto(base);await missingPage.getByRole('button',{name:'设置',exact:true}).click();await missingPage.getByRole('status').filter({hasText:'兼容字体'}).waitFor();
  const fallback=await missingPage.evaluate(async()=>{const f=await import('/src/state/fonts.ts');return f.fontPayload.value;});assert(fallback?.resolved?.zh&&fallback?.resolved?.latin);assert.equal(fallback.resolved.mono,'JetBrains Mono');await missingContext.close();checks.push('controlled missing SimSun/Times preserves a usable UI and explains fallback');
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success:true,checks,layouts,errors},null,2));console.log(out);
 }catch(e){console.error(out);console.error(await page.evaluate(()=>({fonts:[...document.fonts].map(f=>({name:f.family,status:f.status})),font:getComputedStyle(document.documentElement).getPropertyValue('--font'),text:document.querySelector('.font-settings')?.innerText})));await page.screenshot({path:path.join(out,'failed.png'),fullPage:true});throw e;}
 finally{await context.close();await browser.close();await server.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
