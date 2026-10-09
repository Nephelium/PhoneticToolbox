// Owned Chrome, source frontend, no backend mutation or synthetic audio.
const assert=require('node:assert/strict'),fs=require('node:fs/promises'),path=require('node:path'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const out=path.join(root,'output/validation/p17/layout-c',String(Date.now()));await fs.mkdir(out,{recursive:true});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0}});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
 const page=await browser.newPage({viewport:{width:1920,height:1000}}),results=[],errors=[];page.on('pageerror',e=>errors.push(e.message));
 try{await page.goto(server.resolvedUrls.local[0]);
 for(const [id,name] of [['M11','MFA 自动标注'],['M12','TextGrid标注'],['M13','汉字转国际音标'],['M14','音系归纳'],['M15','感知实验']]){
  await page.locator('.nav-item').filter({hasText:name}).click();await page.waitForTimeout(300);
  const active=page.locator('.module-frame:visible');await active.getByRole('button',{name:id==='M12'?'方法与引用':id==='M13'?'帮助与来源':'方法与来源',exact:true}).click();await page.getByRole('dialog').waitFor();await page.keyboard.press('Escape');results.push({id,methodDialog:true});
  const center=page.locator(id==='M12'?'.annotation-editor':id==='M13'?'.m13-result-section':'.workbench-center:visible');const before=await center.boundingBox();
  const toggle=id==='M12'?'标注设置':id==='M13'?'转换设置':id==='M11'?'收起任务与记录':id==='M14'?'收起任务与结果':'收起运行与记录';
  await page.getByRole('button',{name:toggle,exact:true}).click();await page.waitForTimeout(250);assert((await center.boundingBox()).width>before.width);
  await page.reload();await page.locator('.nav-item').filter({hasText:name}).click();await page.waitForTimeout(350);
  const expand=id==='M12'?'标注设置':id==='M13'?'转换设置':id==='M11'?'展开任务与记录':id==='M14'?'展开任务与结果':'展开运行与记录';
  assert((await center.boundingBox()).width>before.width);await page.getByRole('button',{name:expand,exact:true}).click();await page.waitForTimeout(250);results.push({id,collapseAndReload:true});
  for(const size of [{width:1920,height:1000},{width:2560,height:1360},{width:3840,height:2080},{width:1280,height:720}]){
   await page.setViewportSize(size);
   for(const scale of [1,1.5]){
    await page.evaluate(scale=>{document.documentElement.style.zoom=String(scale);document.documentElement.style.setProperty('--page-scale',String(scale));},scale);
    await page.waitForTimeout(250);
    const geometry=await page.evaluate(()=>({viewport:[innerWidth,innerHeight],dpr:devicePixelRatio,nodes:[...document.querySelectorAll('.workspace-content,.module-frame,.m13-workspace,.mfa-columns,.annotation-main')].filter(e=>e.getBoundingClientRect().width).map(e=>({class:e.className,client:[e.clientWidth,e.clientHeight],scroll:[e.scrollWidth,e.scrollHeight],rect:{y:e.getBoundingClientRect().y,bottom:e.getBoundingClientRect().bottom},overflow:getComputedStyle(e).overflow})),controls:[...document.querySelectorAll('.module-frame button,.module-frame input,.module-frame select')].filter(e=>e.getBoundingClientRect().width).map(e=>({text:(e.textContent||e.getAttribute('aria-label')||e.type).slice(0,60),bottom:e.getBoundingClientRect().bottom,disabled:e.disabled}))}));
    results.push({id,size,scale,geometry});
    if(scale===1)await page.screenshot({path:path.join(out,id+'-'+size.width+'-empty.png')});
   }
  }
  await page.evaluate(()=>{document.documentElement.style.zoom='1';document.documentElement.style.setProperty('--page-scale','1');});await page.setViewportSize({width:1920,height:1000});
 }
 assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'report.json'),JSON.stringify(results,null,2));console.log(out);
 }finally{await browser.close();await server.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
