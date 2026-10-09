// M07-R2: production host, owned synthetic files, footer geometry and public actions.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{spawn}=require('node:child_process'),{createInterface}=require('node:readline'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const worker=spawn(path.join(root,'.venv/m09-ui/Scripts/python.exe'),['-B','-X','utf8','tests/support/m07_host_bridge.py'],{cwd:root,windowsHide:true,env:{...process.env,PYTHONPATH:['backend/src','packages/phonetic_core/src','desktop/src','scripts'].map(p=>path.join(root,p)).join(';')}});
 let resolveReady,rejectReady,counter=0;const pending=new Map(),ready=new Promise((r,j)=>{resolveReady=r;rejectReady=j;});
 createInterface({input:worker.stdout}).on('line',line=>{const d=JSON.parse(line);if(d.ready)resolveReady(d);else{pending.get(d.id)?.(d);pending.delete(d.id);}});worker.stderr.on('data',d=>process.stderr.write(d));worker.once('exit',code=>rejectReady(Error('Host exited '+code)));
 const {out}=await ready;console.log(out);
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{entries:['tests/m07-host.html']},plugins:[{name:'m07-attribution-host',configureServer(s){s.middlewares.use('/__m07_host',(req,res)=>{let raw='';req.on('data',d=>raw+=d);req.on('end',()=>{const id=++counter;pending.set(id,data=>{res.setHeader('Content-Type','application/json');res.end(JSON.stringify(data));});worker.stdin.write(JSON.stringify({...JSON.parse(raw),id})+'\n');});});}}]});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--mute-audio']}),page=await browser.newPage({viewport:{width:1920,height:1080}});
 const scope=page.getByLabel('发声类型连续统工作区',{exact:true}),footer=scope.getByLabel('发声类型合成来源与致谢',{exact:true}),checks=[],layouts=[],errors=[];let success=false;
 page.on('pageerror',e=>errors.push(e.message));page.setDefaultTimeout(60000);
 const expected='Lu, Y., Liang, C., & Kong, J. (2025). Contribution of F0 and phonation to tone perception in the Zaiwa language. Journal of Phonetics, 110, 101413. https://doi.org/10.1016/j.wocn.2025.101413';
 const click=name=>scope.getByRole('button',{name,exact:true}).click();
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/m07-host.html');await page.locator('nav').getByRole('button',{name:'发声类型合成',exact:true}).click();await footer.waitFor();
  assert((await footer.innerText()).includes('陆尧、梁昌维、孔江平'));assert((await footer.innerText()).includes('北京大学语言学实验室'));assert((await footer.innerText()).includes('2026-09-10'));assert.equal(await footer.locator('.paper-citation').innerText(),expected);
  assert.equal(await footer.getByRole('link',{name:'论文',exact:true}).getAttribute('href'),'https://doi.org/10.1016/j.wocn.2025.101413');assert.equal(await footer.getByRole('link',{name:'原始仓库',exact:true}).getAttribute('href'),'https://github.com/Luyao2025/Contribution-of-F0-and-phonation-to-tone-perception-in-the-Zaiwa-language');
  await page.evaluate(async()=>{window.copied=[];const {installClipboardWriter}=await import('/src/platform/clipboard.ts');installClipboardWriter(async text=>window.copied.push(text));});
  await footer.getByRole('button',{name:'复制引用',exact:true}).click();assert.equal(await page.evaluate(()=>copied.at(-1)),expected);await footer.getByRole('status').filter({hasText:'引用已复制'}).waitFor();
  checks.push('visible attribution, permission date, full registered citation, exact DOI/repository links and clipboard port copy');
  await footer.getByRole('button',{name:'改写说明',exact:true}).click();const dialog=page.getByRole('dialog',{name:'发声类型合成改写说明',exact:true});await dialog.waitFor();assert((await dialog.innerText()).includes('REAPER 算法作为 PhoneticToolbox 新增的 F0 提取选项'));await dialog.getByRole('button',{name:'完整方法与来源',exact:true}).click();await dialog.waitFor({state:'detached'});await page.getByRole('tab',{name:'软件与代码来源',exact:false}).click();assert((await page.locator('.reference-row').filter({hasText:'载瓦语'}).innerText()).includes('（2026-09-10）作者邮件许可'));await page.locator('.dialog-header button').click();
  checks.push('adaptation explanation and complete module references use the same updated author permission');
  await click('打开音频目录');await scope.getByRole('combobox',{name:'源音频',exact:true}).selectOption({label:'input0.wav'});await scope.getByRole('combobox',{name:'目标音频',exact:true}).selectOption({label:'input1.wav'});await click('提取 F0');await page.waitForFunction(()=>document.querySelector('.m07-page').getAttribute('aria-busy')==='false');await scope.getByLabel('连续统步数',{exact:true}).fill('3');await click('生成当前');await page.waitForFunction(()=>document.querySelectorAll('.f0-plot path[data-curve=synthesis]').length===3);assert.equal(await scope.locator('.result-row').count(),1);
  for(const [width,height] of [[1920,1080],[1440,900],[1280,800],[960,720]])for(const theme of ['light','dark']){
   await page.setViewportSize({width,height});await page.evaluate(theme=>document.documentElement.dataset.theme=theme,theme);await page.waitForTimeout(1000);
   const geo=await scope.evaluate(e=>{const w=e.querySelector('.m07-workspace').getBoundingClientRect(),f=e.querySelector('.m07-attribution').getBoundingClientRect(),b=e.querySelector('.module-workbench').getBoundingClientRect();return {workspace:{x:w.x,y:w.y,width:w.width,bottom:w.bottom,height:w.height},footer:{x:f.x,y:f.y,width:f.width,bottom:f.bottom,height:f.height},workbenchWidth:b.width,screen:innerHeight,plots:[...e.querySelectorAll('.m07-plots>.module-section')].map(n=>{const r=n.getBoundingClientRect();return {bottom:r.bottom,x:r.x,y:r.y}})};});
   assert(geo.footer.y>=geo.workspace.bottom-1&&geo.footer.bottom<=geo.screen+1);assert(Math.abs(geo.footer.x-geo.workspace.x)<1&&Math.abs(geo.footer.width-geo.workspace.width)<1);
   if(width>=1440)assert(geo.plots.every(plot=>plot.bottom<=geo.footer.y),JSON.stringify(geo));
   const before=await footer.boundingBox();await scope.locator('.m07-workspace').evaluate(e=>e.scrollTop=e.scrollHeight);await scope.locator('.workbench-right-body').evaluate(e=>e.scrollTop=e.scrollHeight);const after=await footer.boundingBox();assert.equal(after.y,before.y);await scope.locator('.m07-workspace').evaluate(e=>e.scrollTop=0);await scope.locator('.workbench-right-body').evaluate(e=>e.scrollTop=0);
   await page.screenshot({path:path.join(out,`r2-${width}-${theme}.png`)});layouts.push({width,height,theme,...geo});
  }
  checks.push('eight light/dark layouts: footer spans all columns, stays visible during inner scrolling and does not overlap four plots at desktop sizes');
  await page.setViewportSize({width:1920,height:1080});await page.evaluate(()=>{document.documentElement.dataset.theme='light';document.documentElement.style.setProperty('--body-size','21px');});await page.waitForTimeout(1000);const large=await footer.boundingBox();assert(large.y+large.height<=1080);await page.screenshot({path:path.join(out,'r2-21px.png')});
  await page.evaluate(()=>document.documentElement.style.removeProperty('--body-size'));await page.locator('nav').getByRole('button',{name:'变速变调',exact:true}).click();assert.equal(await page.locator('.m07-attribution:visible').count(),0);checks.push('21 px base font remains visible; another module has no visible M07 footer');
  assert.deepEqual(errors,[]);success=true;
 }catch(e){await page.screenshot({path:path.join(out,'r2-failure.png')});throw e;}
 finally{await fs.writeFile(path.join(out,'r2-report.json'),JSON.stringify({success,checks,layouts,errors,scope:'Windows Chrome actual desktop adapter/host; owned short synthetic input; clipboard writer endpoint records only; large base font is not physical DPI'},null,2));console.log(JSON.stringify({out,success,checks}));await browser.close();await server.close();worker.stdin.end();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
