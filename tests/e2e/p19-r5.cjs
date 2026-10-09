// P19-R5, real Vue settings in an owned headless Chrome profile.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const out=path.join(root,'output/validation/p19-r5','chrome-'+Date.now());await fs.mkdir(out,{recursive:true});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:31090,strictPort:true,watch:null},optimizeDeps:{noDiscovery:true,include:['vue']}});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
 const page=await browser.newPage({viewport:{width:1440,height:1000},colorScheme:'light'}),checks=[],errors=[],layouts=[];
 page.on('pageerror',e=>errors.push(e.message));
 const open=()=>page.getByRole('combobox',{name:'配色方案',exact:true}).click();
 const picker=()=>page.locator('#palette-list');
 const option=id=>picker().locator('[data-palette-option="'+id+'"]');
 const choose=async id=>{await open();await option(id).click();};
 try{
  await page.goto(server.resolvedUrls.local[0]);await page.getByRole('button',{name:'设置',exact:true}).click();
  assert.equal(await page.evaluate(()=>document.documentElement.dataset.palette),'codex');
  assert.equal(await page.getByLabel('波形线颜色',{exact:true}).inputValue(),'theme');
  const input=page.getByLabel('英文与数字字体',{exact:true});assert.equal(await input.inputValue(),'Times New Roman');
  const names=()=>input.locator('option').evaluateAll(es=>es.map(e=>e.value));const full=await names();assert(full.includes('Georgia'));
  await input.selectOption('__custom__');await page.getByLabel('自定义英文与数字字体',{exact:true}).fill('');assert.deepEqual(await names(),full);
  await input.selectOption('Georgia');assert.equal(await input.inputValue(),'Georgia');await input.selectOption('Times New Roman');
  await input.focus();await input.press('ArrowDown');await input.selectOption('Times New Roman');
  await page.getByLabel('图表与导出跟随全局字体').uncheck();assert.equal(await page.getByLabel('图表中文字体',{exact:true}).inputValue(),'SimSun');assert.equal(await page.getByLabel('图表英文字体',{exact:true}).inputValue(),'Times New Roman');
  await page.getByLabel('图表英文字体',{exact:true}).selectOption('Georgia');
  await page.getByRole('button',{name:'应用字体',exact:true}).click();await page.getByRole('status').filter({hasText:'字体已应用'}).waitFor();
  await page.reload();await page.getByRole('button',{name:'设置',exact:true}).click();assert.equal(await page.getByLabel('图表英文字体',{exact:true}).inputValue(),'Georgia');
  checks.push('filled/empty font inputs expose the same complete list; keyboard, selection, figure defaults and actual apply/reload');
  await open();const ids=await picker().getByRole('option').evaluateAll(es=>es.map(e=>e.dataset.paletteOption));assert.equal(ids.length,29);await page.keyboard.press('Escape');
  for(const mode of ['light','dark']){
   await page.getByRole('button',{name:mode==='light'?'浅色':'深色',exact:true}).click();
   for(const id of ids){
    const before=await page.evaluate(()=>({id:document.documentElement.dataset.palette,saved:localStorage.getItem('ptb.v3.palette')}));
    await open();await option(id).hover();
    const mini=page.getByLabel('悬浮配色预览',{exact:true});assert.equal(await mini.getAttribute('data-preview-palette'),id);assert.equal(await mini.getAttribute('data-preview-mode'),mode);
    const preview=await mini.evaluate(e=>({app:getComputedStyle(e).getPropertyValue('--app').trim(),accent:getComputedStyle(e).getPropertyValue('--accent').trim(),bg:getComputedStyle(e).backgroundColor,wave:getComputedStyle(e.querySelector('path')).stroke}));
    assert(preview.app&&preview.accent);assert.equal(await page.evaluate(()=>document.documentElement.dataset.palette),before.id);assert.equal(await page.evaluate(()=>localStorage.getItem('ptb.v3.palette')),before.saved);
    assert.equal(await mini.locator('.preview-actions span').count(),3);assert.equal(await option(id).locator('.palette-aa').innerText(),'Aa');
    await option(id).click();assert.equal(await mini.count(),0);
    const rootColors=await page.evaluate(()=>{const s=getComputedStyle(document.documentElement);return {app:s.getPropertyValue('--app').trim(),accent:s.getPropertyValue('--accent').trim()};});assert.equal(rootColors.app,preview.app);assert.equal(rootColors.accent,preview.accent);
    const stroke=await page.locator('.wave-color-preview path').evaluate(el=>getComputedStyle(el).stroke);assert.equal(stroke,preview.wave);
   }
  }checks.push('29 palettes × 2 modes: Aa with background, full mini-window, hover does not persist or change theme, selection/preview/waveform agree');
  await choose('codex');await open();await page.keyboard.press('End');await page.keyboard.press('Enter');assert.equal(await page.evaluate(()=>document.documentElement.dataset.palette),'xcode');
  await open();await page.mouse.click(20,10);assert.equal(await picker().count(),0);checks.push('palette keyboard End/Enter and outside-click dismissal');
  for(const [width,height,scale]of [[1440,1000,100],[900,700,150],[390,844,100]]){
   await page.setViewportSize({width,height});await page.evaluate(async n=>(await import('/src/state/pageZoom.ts')).setPageScale(n),scale);
   await page.getByRole('combobox',{name:'配色方案',exact:true}).scrollIntoViewIfNeeded();await open();await option('catppuccin').hover();
   const geometry=await page.evaluate(()=>[...document.querySelectorAll('.palette-list,.palette-hover-preview')].map(e=>{const r=e.getBoundingClientRect();return {x:r.x,y:r.y,right:r.right,bottom:r.bottom,width:r.width,height:r.height};}));
   for(const g of geometry){assert(g.x>=-1&&g.y>=-1&&g.right<=width+1&&g.bottom<=height+1,JSON.stringify({width,height,scale,g}));}
   assert(await page.evaluate(()=>document.documentElement.scrollWidth<=document.documentElement.clientWidth+2));layouts.push({width,height,scale,geometry});
   await page.screenshot({path:path.join(out,`palette-${width}-${scale}.png`)});await page.keyboard.press('Escape');
  }checks.push('popup and mini-window within 1440/900/390 viewports, including 150% page zoom');
  await page.setViewportSize({width:1440,height:1000});await page.evaluate(async()=>{(await import('/src/state/pageZoom.ts')).setPageScale(100);localStorage.setItem('ptb.v3.fonts.v1.desktop',JSON.stringify({version:1,zh:'KaiTi',latin:'Georgia',mono:'Consolas',ipa:'Doulos SIL',figure:{follow:false,zh:'',latin:'',size:15}}));localStorage.setItem('ptb.v3.palette',JSON.stringify('everforest'));localStorage.setItem('ptb.v3.waveformAppearance',JSON.stringify({mode:'blue',custom:'#ff0088'}));});
  await page.reload();await page.getByRole('button',{name:'设置',exact:true}).click();assert.equal(await page.evaluate(()=>document.documentElement.dataset.palette),'everforest');assert.equal(await page.getByLabel('波形线颜色',{exact:true}).inputValue(),'blue');assert.equal(await page.getByLabel('中文字体',{exact:true}).inputValue(),'KaiTi');assert.equal(await page.getByLabel('图表中文字体',{exact:true}).inputValue(),'SimSun');assert.equal(await page.getByLabel('图表英文字体',{exact:true}).inputValue(),'Times New Roman');
  checks.push('explicit old palette/blue/UI fonts persist; empty historical figure fonts adopt SimSun/Times');
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success:true,checks,layouts,errors},null,2));console.log(JSON.stringify({out,checks}));
 }catch(e){await page.screenshot({path:path.join(out,'failed.png')});await fs.writeFile(path.join(out,'failed.json'),JSON.stringify({error:String(e),checks,errors},null,2));throw e;}
 finally{await browser.close();await server.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
