// P18: owned headless Chrome; source UI only, no production database or capture.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
const modules=[['M01','参数估计'],['M02','参数显示'],['M03','EGG 信号分析'],['M04','LPC 谱图'],['M05','唇形提取'],['M06','语音合成'],['M07','发声类型合成'],['M08','变速变调'],['M09','语谱图转音频'],['M11','MFA 自动标注'],['M12','TextGrid标注'],['M13','汉字转国际音标'],['M14','音系归纳'],['M15','感知实验'],['M16','录音']];
async function main(){
 const out=path.join(root,'output/validation/p18/layout',String(Date.now()));await fs.mkdir(out,{recursive:true});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0}});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
 let currentId='';const page=await browser.newPage({viewport:{width:1920,height:1080}}),checks=[],errors=[];page.on('pageerror',e=>errors.push({id:currentId,message:e.message}));
 await page.addInitScript(()=>{window.__p18Errors=[];window.addEventListener('error',e=>window.__p18Errors.push(e.message));});
 const active=()=>page.locator('.module-frame:visible');
 const settle=async()=>{await page.evaluate(()=>document.fonts.ready);await page.waitForTimeout(150);};
 const geometry=()=>active().evaluate(frame=>{
  const rect=e=>{if(!e)return null;const r=e.getBoundingClientRect();return{x:r.x,y:r.y,right:r.right,bottom:r.bottom,width:r.width,height:r.height,clientWidth:e.clientWidth,scrollWidth:e.scrollWidth,clientHeight:e.clientHeight,scrollHeight:e.scrollHeight};};
  const find=s=>frame.querySelector(s),left=find('.workbench-left,.annotation-files,.m13-input-section,.recording-sidebar'),center=find('.workbench-center,.annotation-editor,.m13-result-section,.recording-main'),right=find('.workbench-right,.annotation-settings,.m13-settings-section');
  return{frame:rect(frame),left:rect(left),center:rect(center),right:rect(right),toolbar:rect(find('.module-toolbar')),controls:[...frame.querySelectorAll('.module-toolbar button')].filter(e=>e.offsetWidth).map(e=>({text:e.textContent.trim(),height:e.getBoundingClientRect().height})),overflow:[...frame.querySelectorAll('.workbench-left,.workbench-right,.annotation-files,.annotation-settings,.recording-sidebar')].filter(e=>e.offsetWidth).map(e=>({name:e.className,width:e.clientWidth,scroll:e.scrollWidth}))};
 });
 try{
  await page.goto(server.resolvedUrls.local[0]);
  for(const [id,name] of modules){currentId=id;console.log('P18',id);
   await page.locator('.nav-item').filter({hasText:name}).click();await active().waitFor();await settle();
   await page.setViewportSize({width:1920,height:1080});await page.evaluate(()=>{document.documentElement.style.zoom='1';document.documentElement.style.setProperty('--page-scale','1');document.documentElement.dataset.theme='light';});await settle();
   if(id==='M11'){assert.equal(await active().getByLabel('Beam',{exact:true}).count(),1);assert(await active().getByLabel('Beam',{exact:true}).isVisible());assert.equal(await active().getByLabel('Beam',{exact:true}).inputValue(),'10');}
   let g=await geometry();assert(g.toolbar,id+' toolbar');assert.equal(Math.round(g.left.width),300,id+' default left width');
   if(g.right)assert.equal(Math.round(g.right.width),300,id+' default right width');
   assert(g.left.x<g.center.x,id+' operations on left');if(g.right)assert(Math.abs(g.left.bottom-g.right.bottom)<2,id+' full-height side rails');
   assert(g.frame.scrollWidth<=g.frame.clientWidth+1,id+' no horizontal page overflow');
   const refs=active().locator('.module-toolbar-actions button').filter({hasText:/方法|来源/});assert.equal(await refs.count(),1,id+' references in common toolbar');
   await refs.click();await page.getByRole('dialog').waitFor();await page.keyboard.press('Escape');checks.push({id,name:'default panes and methods',geometry:g});
   await page.screenshot({path:path.join(out,id+'-light.png')});await page.evaluate(()=>document.documentElement.dataset.theme='dark');await settle();await page.screenshot({path:path.join(out,id+'-dark.png')});
   for(const [width,height,scale] of [[2560,1440,1],[1920,1080,.7],[1920,1080,1.5],[1280,720,1],[1000,720,1.5]]){
    await page.setViewportSize({width,height});await page.evaluate(scale=>{document.documentElement.style.zoom=String(scale);document.documentElement.style.setProperty('--page-scale',String(scale));},scale);await settle();g=await geometry();
    if(id==='M13')assert.equal(await active().evaluate(el=>getComputedStyle(el).overflowX),'auto','M13 intentionally retains independent horizontal access');
    else assert(g.frame.scrollWidth<=g.frame.clientWidth+1,id+' horizontal overflow '+[width,scale]);
    for(const p of g.overflow)assert(p.scroll<=p.width+1,id+' side overflow '+p.name+' '+[width,scale]);
    await refs.scrollIntoViewIfNeeded();assert(await refs.isVisible());checks.push({id,name:'viewport',viewport:[width,height,scale],geometry:g});
   }
   await page.evaluate(()=>{document.documentElement.style.zoom='1';document.documentElement.style.setProperty('--page-scale','1');});await page.setViewportSize({width:1920,height:1080});await settle();
  }
  // Old EGG shared the M01 layout key and persisted both entries, including
  // the absent left slot. Read the actual right width without mutating M01.
  const legacyKey='ptb.v3.layout.panels.local:M01.workbench';
  await page.evaluate(key=>localStorage.setItem(key,JSON.stringify({'--panel-left':250,'--panel-right':350})),legacyKey);await page.reload();await page.locator('.nav-item').filter({hasText:'EGG 信号分析'}).click();await settle();
  assert.equal(Math.round((await geometry()).left.width),350,'old right preference moves left');
  const handle=page.getByRole('separator',{name:'参数、试听与任务宽度'});await handle.focus();await page.keyboard.press('ArrowRight');await settle();assert.equal(Math.round((await geometry()).left.width),360);
  assert.deepEqual(await page.evaluate(key=>JSON.parse(localStorage.getItem(key)),legacyKey),{'--panel-left':250,'--panel-right':350},'M03 resizing preserves legacy M01 width');
  await page.reload();await page.locator('.nav-item').filter({hasText:'EGG 信号分析'}).click();await settle();assert.equal(Math.round((await geometry()).left.width),360,'moved preference persists after resize');checks.push({id:'M03',name:'old dual-key width moves left, new resize persists independently of M01'});
  await page.locator('.nav-item').filter({hasText:'汉字转国际音标'}).click();await settle();await page.getByLabel('上下排布').check();await settle();
  let setting=await page.locator('.m13-settings-section').boundingBox(),input=await page.locator('.m13-input-section').boundingBox();assert(setting.x<input.x,'stacked M13 settings on left');
  const m13handle=page.getByRole('separator',{name:'转换与排版宽度'});await m13handle.focus();await page.keyboard.press('ArrowRight');await settle();assert.equal(Math.round((await page.locator('.m13-settings-section').boundingBox()).width),310);checks.push({id:'M13',name:'stacked settings on left, keyboard resize'});
  assert.deepEqual(errors,[]);assert.deepEqual(await page.evaluate(()=>window.__p18Errors),[]);await fs.writeFile(path.join(out,'report.json'),JSON.stringify({checks,errors},null,2));console.log(JSON.stringify({out,checks:checks.length,errors}));
 }catch(error){await page.screenshot({path:path.join(out,'failed.png')});await fs.writeFile(path.join(out,'failed.json'),JSON.stringify({checks,errors,geometry:await geometry()},null,2));console.error(out);throw error;}
 finally{await browser.close();await server.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
