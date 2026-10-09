// Actual Vue app, owned Chrome profile, no scientific tasks or media devices.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const out=path.join(root,'output/validation/p19-r9','chrome-'+Date.now());await fs.mkdir(out,{recursive:true});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:31089,strictPort:true,watch:null},optimizeDeps:{noDiscovery:true,include:['vue']}});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
 const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[],checks=[],layouts=[];page.on('pageerror',e=>errors.push(e.message));
 const settings=()=>page.getByRole('button',{name:'设置',exact:true}).click();const field=name=>page.getByLabel(name,{exact:true});
 try{
  await page.goto(server.resolvedUrls.local[0]);await settings();await field('图表与导出跟随全局字体').uncheck();
  const labels=['中文字体','英文与数字字体','代码与等宽字体','图表中文字体','图表英文字体'];
  for(const label of labels){const control=field(label);assert.equal(await control.evaluate(e=>e.tagName),'SELECT');const values=await control.locator('option').evaluateAll(es=>es.map(e=>e.value));assert(values.includes('Georgia')||label==='代码与等宽字体');await control.selectOption('__custom__');const input=field(label==='代码与等宽字体'?'自定义代码字体':'自定义'+label);await input.fill('');assert.deepEqual(await control.locator('option').evaluateAll(es=>es.map(e=>e.value)),values);await input.fill('My Font');assert(await control.locator('option').evaluateAll(es=>es.some(e=>e.value==='My Font')));await control.selectOption(label.includes('中文')?'SimSun':label==='代码与等宽字体'?'JetBrains Mono':'Times New Roman');}
  checks.push('five native selects, complete unfiltered options with filled/empty/custom values');
  await field('正文基础字号').fill('18');await field('图表基础字号').fill('13');await page.getByRole('button',{name:'预览字体',exact:true}).click();await page.getByRole('status').filter({hasText:'预览已更新'}).waitFor();
  assert.equal(await page.locator('.font-preview').evaluate(e=>getComputedStyle(e).fontSize),'18px');assert.equal(await page.evaluate(()=>getComputedStyle(document.documentElement).fontSize),'14px');
  await page.getByRole('button',{name:'应用字体',exact:true}).click();await page.getByRole('status').filter({hasText:'字体已应用'}).waitFor();assert.equal(await page.evaluate(()=>getComputedStyle(document.documentElement).fontSize),'18px');assert.equal(await page.evaluate(()=>getComputedStyle(document.documentElement).getPropertyValue('--figure-size').trim()),'13px');
  await page.reload();await settings();assert.equal(await field('正文基础字号').inputValue(),'18');assert.equal(await field('图表基础字号').inputValue(),'13');
  const sizes=await page.evaluate(()=>({nav:getComputedStyle(document.querySelector('.nav-item')).fontSize,tab:getComputedStyle(document.querySelector('.tab-wrap>button')).fontSize}));assert.deepEqual(sizes,{nav:'18px',tab:'18px'});checks.push('body and plot sizes preview/apply/save/reload independently; navigation and tabs match');
  await page.getByRole('button',{name:'参数估计',exact:true}).click();await page.locator('.m01-page').waitFor();assert.equal(await page.locator('.m01-page').evaluate(e=>getComputedStyle(e).fontSize),'18px');
  // Use the real file-list class without importing a user WAV or launching work.
  assert.equal(await page.locator('.m01-file-list').evaluate(e=>{const row=document.createElement('div');row.className='file-row';row.innerHTML='<span>声学音频.wav</span>';e.append(row);const size=getComputedStyle(row.firstChild).fontSize;row.remove();return size;}),'18px');
  const moduleIds=['M03','M04','M05','M06','M07','M08','M09','M11','M12','M13','M14','M15','M16','M17'];
  const {modules}=await page.evaluate(async()=>({modules:(await import('/src/app/registry.ts')).modules}));assert.equal(new Set(modules.map(m=>m.icon)).size,17);
  for(const id of moduleIds){const m=modules.find(m=>m.id===id);await page.getByRole('button',{name:m.title,exact:true}).click();const font=await page.locator('main').evaluate(e=>{const child=[...e.querySelectorAll('.module-frame')].find(e=>e.offsetParent);return child?getComputedStyle(child).fontSize:getComputedStyle(e).fontSize;});assert.equal(font,'18px',id);}
  checks.push('module text baseline and real file-list styling follow 18px; all 17 module icons distinct');
  await settings();await field('正文基础字号').fill('14');await page.getByRole('button',{name:'应用字体',exact:true}).click();await page.getByRole('status').filter({hasText:'字体已应用'}).waitFor();
  for(const [width,height,scale,mode]of [[1440,1000,100,'light'],[1920,1080,100,'dark'],[900,800,150,'light'],[390,844,100,'dark']]){
   await page.setViewportSize({width,height});await page.evaluate(async n=>(await import('/src/state/pageZoom.ts')).setPageScale(n),scale);await page.getByRole('button',{name:mode==='light'?'浅色':'深色',exact:true}).click();
   const geometry=await page.evaluate(()=>({overflow:document.documentElement.scrollWidth>innerWidth+1,controls:[...document.querySelectorAll('.font-family-select select')].map(e=>{const r=e.getBoundingClientRect();return {left:r.left,right:r.right,width:r.width};})}));assert(!geometry.overflow,JSON.stringify(geometry));assert(geometry.controls.every(r=>r.width>100&&r.left>=0&&r.right<=width+1),JSON.stringify(geometry));layouts.push({width,height,scale,mode,geometry});await page.locator('main').evaluate(e=>e.scrollTop=0);await page.screenshot({path:path.join(out,`settings-${width}-${mode}.png`)});
  }
  await page.setViewportSize({width:1440,height:1000});await page.evaluate(async()=>(await import('/src/state/pageZoom.ts')).setPageScale(100));
  for(const width of [184,240,360]){await page.locator('.app-shell').evaluate((e,w)=>e.style.setProperty('--panel-left',w+'px'),width);const geometry=await page.locator('.sidebar-search-row').evaluate(e=>{const i=e.querySelector('input').getBoundingClientRect(),b=e.querySelector('button').getBoundingClientRect();return {inputWidth:i.width,inputRight:i.right,buttonLeft:b.left,inputY:i.y,buttonY:b.y};});assert(geometry.inputWidth>30&&geometry.inputRight<=geometry.buttonLeft&&Math.abs(geometry.inputY-geometry.buttonY)<3);layouts.push({sidebar:width,geometry});}
  await page.getByRole('button',{name:'收起侧栏',exact:true}).click();assert(await page.getByRole('button',{name:'展开侧栏',exact:true}).isVisible());await page.getByRole('button',{name:'展开侧栏',exact:true}).click();assert(await field('搜索工具').isVisible());checks.push('search/toggle share row across 184/240/360 widths; collapsed toggle reachable');
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success:true,checks,layouts,errors},null,2));console.log(JSON.stringify({out,checks}));
 }catch(e){await page.screenshot({path:path.join(out,'failed.png')});await fs.writeFile(path.join(out,'failed.json'),JSON.stringify({error:String(e),checks,layouts,errors},null,2));throw e;}
 finally{await browser.close();await server.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
