// P04-RESIZE / M02-DEFAULT. Owned Chrome, synthetic inputs, no scientific/DB claims.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');const root=path.resolve(__dirname,'../..');
async function main(){
 const out=path.join(root,'output/validation/p04-resize',String(Date.now()));await fs.mkdir(out,{recursive:true});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0},optimizeDeps:{entries:['tests/p04-resize.html']}});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--mute-audio']});
 const page=await browser.newPage({viewport:{width:1920,height:1080}});page.setDefaultTimeout(10000);
 await page.addInitScript(()=>{window.__layoutErrors=[];window.addEventListener('error',e=>window.__layoutErrors.push({message:e.message,tab:document.querySelector('[role=tab][aria-selected=true]')?.textContent,url:location.pathname}));});
 const errors=[],checks=[],matrix=[];page.on('pageerror',e=>errors.push(e.message));let success=false;
 const base=server.resolvedUrls.local[0],url=base+'tests/p04-resize.html';
 const open=async title=>{await page.locator('nav').getByRole('button',{name:title,exact:true}).click();await page.getByRole('tab',{name:title,exact:true}).waitFor();await page.waitForTimeout(130);};
 const size=async(selector)=>page.locator(selector).evaluate(el=>el.getBoundingClientRect().width/(document.documentElement.style.zoom||1));
 const drag=async(handle,dx)=>{await handle.scrollIntoViewIfNeeded();const b=await handle.boundingBox();assert(b,'visible separator');await page.mouse.move(b.x+b.width/2,b.y+Math.min(65,b.height/2));await page.mouse.down();await page.mouse.move(b.x+b.width/2+dx,b.y+Math.min(65,b.height/2),{steps:8});await page.mouse.up();await page.waitForTimeout(100);};
 const near=(a,b,label)=>assert(Math.abs(a-b)<3,`${label}: ${a} vs ${b}`);
 const snap=async name=>{await page.screenshot({path:path.join(out,name+'.png'),animations:'disabled'});};
 try{
  await page.goto(url);await open('参数估计');
  near(await size('.workbench-columns>.file-panel'),220,'M01 default left');near(await size('.workbench-columns>.parameter-summary'),250,'M01 default right');
  await drag(page.locator('.workbench-columns').getByRole('separator',{name:'音频列表宽度'}),65);
  await drag(page.locator('.workbench-columns').getByRole('separator',{name:'参数与结果宽度'}),-55);
  near(await size('.workbench-columns>.file-panel'),285,'M01 drag left');near(await size('.workbench-columns>.parameter-summary'),305,'M01 drag right');
  await snap('M01-light');await page.reload();await open('参数估计');near(await size('.workbench-columns>.file-panel'),285,'M01 reload');
  checks.push('M01 wider defaults, pointer drag both sides and reload persistence');
  await open('参数显示');await page.locator('.m02-files button').click();await page.locator('.empty-plot').waitFor();
  assert.equal(await page.locator('.parameter-curve').count(),0);assert.equal(await page.locator('.m02-parameters input:checked').count(),0);
  assert.equal(await page.locator('.m02-page input[type=range][aria-label*=列宽]').count(),0);
  assert.equal(await page.getByRole('tab',{name:'参数显示',exact:true}).getByLabel('未保存').count(),0);
  near(await size('.m02-columns>.file-panel'),220,'M02 independent');
  await page.locator('.m02-parameters label').filter({hasText:'F0 - Praat'}).count();
  await page.locator('.m02-parameters input').first().check();await page.getByRole('button',{name:'将 1 项分配到图窗',exact:true}).click();
  await page.locator('.parameter-curve').waitFor();await page.getByRole('button',{name:'保存绘图配置',exact:true}).click();
  await page.getByRole('button',{name:'关闭 参数显示',exact:true}).click();await open('参数显示');await page.locator('.m02-files button').click();await page.locator('.parameter-curve').waitFor();
  assert.equal(await page.locator('.parameter-curve').count(),1);
  await page.locator('.m02-parameters input').first().check();await page.getByRole('button',{name:'从图中移除勾选项',exact:true}).click();await page.getByRole('button',{name:'保存绘图配置',exact:true}).click();
  await page.reload();await open('参数显示');await page.locator('.m02-files button').click();await page.locator('.empty-plot').waitFor();
  checks.push('M02 first load empty; explicit saved curves and saved empty config both restored');
  for(const scale of [70,100,150]){
   await page.evaluate(async scale=>(await import('/src/state/pageZoom.ts')).setPageScale(scale),scale);await page.waitForTimeout(160);
   const before=await size('.m02-columns>.file-panel');await drag(page.locator('.m02-columns').getByRole('separator',{name:'音频列表宽度'}),21);
   near(await size('.m02-columns>.file-panel')-before,21/(scale/100),'zoom-adjusted drag '+scale);
  }
  await page.evaluate(async()=>{(await import('/src/state/pageZoom.ts')).setPageScale(100);document.documentElement.dataset.theme='dark';});await page.waitForTimeout(160);await snap('M02-dark');
  const savedWidth=await size('.m02-columns>.file-panel');await page.setViewportSize({width:800,height:650});await page.waitForTimeout(200);
  assert.equal(await page.locator('.m02-columns .panel-resize-handle:visible').count(),0);await snap('M02-narrow');
  await page.setViewportSize({width:1920,height:1080});await page.waitForTimeout(200);near(await size('.m02-columns>.file-panel'),savedWidth,'restore after narrow');
  const sep=page.locator('.m02-columns').getByRole('separator',{name:'音频列表宽度'});await sep.focus();await sep.press('ArrowRight');near(await size('.m02-columns>.file-panel'),savedWidth+10,'keyboard resize');
  await sep.press('Home');near(await size('.m02-columns>.file-panel'),180,'keyboard minimum');await sep.press('End');near(await size('.m02-columns>.file-panel'),520,'keyboard maximum');
  // Escape cancels an in-flight pointer resize without replacing the preference.
  const b=await sep.boundingBox();await page.mouse.move(b.x+4,b.y+40);await page.mouse.down();await page.mouse.move(b.x-60,b.y+40);await page.keyboard.press('Escape');await page.mouse.up();
  // Cancellation schedules a layout frame. Assert the rendered restoration,
  // rather than racing that frame with an immediate DOM read.
  await page.waitForFunction(()=>document.querySelector('.m02-columns [aria-label="音频列表宽度"]')?.getAttribute('aria-valuenow')==='520');
  near(await size('.m02-columns>.file-panel'),520,'pointer cancel');
  checks.push('M02 logical-pixel drag at 70/100/150%, narrow restore, keyboard limits and Escape cancel');
  const layouts=[['LPC 谱图','.lpc-layout',1],['唇形提取','.lip-columns',1],['语音合成','.editor-layout',1],['发声类型合成','.editor-layout',1],['语谱图转音频','.m09-columns',1],['MFA 自动标注','.mfa-columns',1],['TextGrid标注','.annotation-layout',2],['汉字转国际音标','.m13-workspace',2]];
  for(const [title,selector,count] of layouts){
   await open(title);const layout=page.locator(selector+':visible');await layout.waitFor();const handles=layout.locator('.panel-resize-handle:visible');assert.equal(await handles.count(),count,title);
   const h=handles.first(),before=Number(await h.getAttribute('aria-valuenow'));await drag(h,20);assert.notEqual(Number(await h.getAttribute('aria-valuenow')),before,title+' pointer boundary');await h.focus();await h.press('ArrowRight');await page.waitForTimeout(100);
   matrix.push({title,handles:count});
  }
  checks.push('All eight other layouts expose functional shared boundaries');
  for(const title of ['EGG 信号分析','变速变调','音系归纳','感知实验']){await open(title);assert.equal(await page.locator('main h1:visible').count(),0);}
  checks.push('Modules with no sidebars retain their existing content layout and omit redundant headings');
  for(const [title,selector] of [['参数估计','.m01-page'],['参数显示','.m02-page'],['LPC 谱图','.lpc-page'],['唇形提取','.lip-page'],['语音合成','section[aria-label="语音合成工作区"]'],['发声类型合成','section[aria-label="发声类型连续统工作区"]'],['语谱图转音频','.m09-page'],['MFA 自动标注','section[aria-label="MFA 自动标注工作区"]'],['TextGrid标注','.annotation-page'],['汉字转国际音标','.mandarin-ipa-page']]){
   await open(title);
   for(const [width,height,scale,theme] of [[1280,800,100,'light'],[1280,800,150,'dark'],[1920,1080,70,'light'],[800,650,100,'dark']]){
    await page.setViewportSize({width,height});await page.evaluate(async({scale,theme})=>{(await import('/src/state/pageZoom.ts')).setPageScale(scale);document.documentElement.dataset.theme=theme;},{scale,theme});await page.waitForTimeout(150);
    const geometry=await page.locator(selector).evaluate(el=>({width:el.clientWidth,scroll:el.scrollWidth,handles:[...el.querySelectorAll('.panel-resize-handle:not([hidden])')].map(h=>({width:Number(h.getAttribute('aria-valuenow')),min:Number(h.getAttribute('aria-valuemin'))}))}));
    if(title==='汉字转国际音标'&&geometry.scroll>geometry.width+2){await page.locator('.m13-settings-section').evaluate(el=>el.scrollIntoView({block:'nearest',inline:'end'}));assert(await page.locator(selector).evaluate(el=>el.scrollLeft>0&&getComputedStyle(el).overflowX==='auto'),'M13 keeps its separately requested right panel reachable by scrolling');}else assert(geometry.scroll<=geometry.width+2,title+' overflow '+JSON.stringify({width,scale,...geometry}));for(const h of geometry.handles)assert(h.width>=h.min,title+' minimum width');assert.equal(await page.locator(selector+' h1:visible').count(),0);
    matrix.push({title,width,height,scale,theme,...geometry});
    if(title==='参数估计'||title==='参数显示')await snap(`${title}-${width}-${scale}-${theme}`);
   }
   await page.setViewportSize({width:1920,height:1080});await page.evaluate(async()=>(await import('/src/state/pageZoom.ts')).setPageScale(100));await page.waitForTimeout(150);
  }
  checks.push('40 module/viewport/scale/theme geometry checks; M13 intentional scroll remains accessible, other pages fit; no duplicated module h1');
  await page.getByRole('button',{name:'设置',exact:true}).click();await page.getByRole('tab',{name:'设置',exact:true}).waitFor();assert.equal(await page.getByRole('dialog').count(),0);
  await page.getByLabel('中文字体',{exact:true}).fill('Missing-Font-For-Test');await page.getByRole('button',{name:'使用说明',exact:true}).click();
  assert.equal(await page.getByRole('dialog').count(),0);await page.getByRole('button',{name:'设置',exact:true}).click();assert.equal(await page.getByLabel('中文字体',{exact:true}).inputValue(),'Missing-Font-For-Test');
  assert.equal(await page.getByRole('tab',{name:/设置/}).count(),1);await page.getByRole('button',{name:'关闭 设置',exact:true}).click();
  const dialog=page.getByRole('dialog',{name:'应用字体设置？'});await dialog.waitFor();await dialog.getByRole('button',{name:'取消关闭'}).click();
  await page.getByRole('button',{name:'关闭 设置',exact:true}).click();await dialog.getByRole('button',{name:'应用字体并关闭'}).click();await dialog.getByRole('alert').waitFor();assert.equal(await page.getByLabel('中文字体',{exact:true}).inputValue(),'Missing-Font-For-Test');
  await dialog.getByRole('button',{name:'放弃字体编辑并关闭'}).click();await page.getByRole('button',{name:'设置',exact:true}).click();assert.notEqual(await page.getByLabel('中文字体',{exact:true}).inputValue(),'Missing-Font-For-Test');
  await page.getByLabel('配色主题').selectOption('light');await snap('settings-tab');await page.getByRole('button',{name:'使用说明',exact:true}).click();await snap('help-tab');
  assert.equal(await page.getByRole('tab',{name:'使用说明',exact:true}).count(),1);
  checks.push('Settings/help are unique tabs; draft retained across tabs; cancel/failure/discard closing paths preserve correct state');
  const nav=page.getByRole('separator',{name:'工具导航宽度'});await drag(nav,35);near(await size('.sidebar'),259,'navigation drag');await page.getByRole('button',{name:'收起侧栏',exact:true}).click();assert.equal(await nav.count(),0);await page.getByRole('button',{name:'展开侧栏',exact:true}).click();near(await size('.sidebar'),259,'navigation expand');
  await page.reload();near(await size('.sidebar'),259,'navigation reload');
  // Storage exceptions must leave the usable width and report failure.
  await page.evaluate(()=>{window.__setItem=Storage.prototype.setItem;Storage.prototype.setItem=function(){throw Error('controlled storage failure');};});
  await drag(page.getByRole('separator',{name:'工具导航宽度'}),10);await page.getByRole('status').filter({hasText:'栏宽未能保存'}).waitFor();near(await size('.sidebar'),269,'storage failure keeps width');
  await page.evaluate(()=>Storage.prototype.setItem=window.__setItem);checks.push('Navigation drag/collapse/reload; explicit storage failure keeps current width usable');
  await page.goto(url+'?vocal=1');await open('声道工作台');const vf=page.frameLocator('.vocal-page iframe');await vf.locator('.workspace .panel-resize-handle:visible').first().waitFor();
  await vf.locator('#columnLayout').selectOption('three');await page.waitForTimeout(250);assert.equal(await vf.locator('.panel-resize-handle:visible').count(),2);
  const vh=vf.getByRole('separator',{name:'操作面板宽度'}),vw=Number(await vh.getAttribute('aria-valuenow'));await vh.focus();await vh.press('ArrowLeft');await page.waitForTimeout(100);near(Number(await vh.getAttribute('aria-valuenow')),vw+10,'M10 shared handler');
  assert.equal(await vf.locator('.masthead h1').count(),0);assert.equal(await vf.locator('#shutdownButton').count(),0);await snap('M10-layout-only');
  checks.push('Actual M10 wrapper + iframe reuse controller in three-column mode; native engine deliberately unavailable');
  await page.waitForTimeout(1200);for(const frame of page.frames())errors.push(...await frame.evaluate(()=>window.__layoutErrors??[]));assert.deepEqual(errors,[]);success=true;
 }catch(error){await snap('failure');await fs.writeFile(path.join(out,'failure.txt'),String(error)+'\n'+await page.locator('body').innerText());throw error;}
 finally{await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success,checks,matrix,errors,platform:process.platform,scope:'Windows Chrome, synthetic inputs; no native scientific/device evidence'},null,2));console.log(out);await browser.close();await server.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
