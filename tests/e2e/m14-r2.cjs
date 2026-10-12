const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{spawn,spawnSync}=require('node:child_process'),{createInterface}=require('node:readline'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const python=path.join(root,'.venv/m14/Scripts/python.exe'),worker=spawn(python,['-B','-X','utf8','scripts/verify_m14_r2_host.py'],{cwd:root,windowsHide:true,env:{...process.env,PYTHONPATH:['backend/src','packages/phonetic_core/src','desktop/src','scripts'].map(p=>path.join(root,p)).join(';')}});
 let readyResolve,readyReject,counter=0;const pending=new Map(),ready=new Promise((r,j)=>{readyResolve=r;readyReject=j});
 createInterface({input:worker.stdout}).on('line',line=>{if(!line.startsWith('{')){console.log(line);return;}const d=JSON.parse(line);if(d.ready)readyResolve(d);else{pending.get(d.id)?.(d);pending.delete(d.id)}});worker.stderr.on('data',d=>process.stderr.write(d));worker.once('exit',code=>readyReject(Error('Host exited '+code)));
 const {out:scratch}=await ready,out=path.dirname(scratch),inputs=path.join(scratch,'input');console.log(out);await fs.writeFile(path.join(inputs,'large.tsv'),largeTable(5452));await fs.writeFile(path.join(inputs,'dense.tsv'),largeTable(10000,true));
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:5207,strictPort:true},optimizeDeps:{entries:['tests/m14-host.html']},plugins:[{name:'m14-baseline',enforce:'pre',load(id){if(process.env.M14_BASELINE==='1'&&id.endsWith('/ResultPreview.vue'))return spawnSync('git',['show','HEAD:frontend/src/modules/phonology-induction/ResultPreview.vue'],{cwd:root,encoding:'utf8',windowsHide:true}).stdout;}},{name:'m14-owned-transport',configureServer(s){s.middlewares.use('/__m14_host',(req,res)=>{let raw='';req.on('data',d=>raw+=d);req.on('end',()=>{const id=++counter;pending.set(id,data=>{res.setHeader('Content-Type','application/json');res.end(JSON.stringify(data))});worker.stdin.write(JSON.stringify({...JSON.parse(raw),id})+'\n')})})}}]});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true}),page=await browser.newPage({viewport:{width:1920,height:1080}}),checks=[],errors=[],formats=[],layouts=[];page.on('pageerror',e=>errors.push(e.message));page.setDefaultTimeout(60000);
 const scope=page.getByLabel('音系归纳工作区',{exact:true}),click=name=>scope.getByRole('button',{name:name.replace(/DOCX|XLSX$/,''),exact:!/(DOCX|XLSX)$/.test(name)}).click(),idle=()=>page.waitForFunction(()=>document.querySelector('[aria-label="音系归纳工作区"]')?.getAttribute('aria-busy')==='false');
 const upload=async p=>{await scope.locator('input[type=file]').setInputFiles(p);await idle();};
 const reviewAbsent=async()=>assert.equal(await scope.getByText('导入与归并审阅',{exact:false}).count(),0);

 let success=false;const timings=[];
 const baseline=process.env.M14_BASELINE==='1';
 const capture=async name=>page.screenshot({path:path.join(out,name+'.png')});
 const ensureFits=async label=>{
  const geometry=await scope.evaluate((e,label)=>{const b=[...e.querySelectorAll('button')].find(b=>b.textContent.trim()===label),r=b.getBoundingClientRect(),c=e.querySelector('.workbench-center');return {bottom:r.bottom,height:innerHeight,top:r.top,scroll:c.scrollHeight,client:c.clientHeight,frame:e.scrollHeight,frameClient:e.clientHeight,width:e.scrollWidth,clientWidth:e.clientWidth};},label);
  assert(geometry.bottom<=geometry.height&&geometry.top>=0,JSON.stringify({label,geometry}));assert(geometry.scroll<=geometry.client+1,JSON.stringify({label,geometry}));assert(geometry.frame<=geometry.frameClient+1,JSON.stringify({label,geometry}));assert(geometry.width<=geometry.clientWidth+1,JSON.stringify({label,geometry}));return geometry;
 };
 const sample=async()=>{
  await click('声母 → 韵母DOCX');
  const ms=await page.evaluate(async()=>{let start=performance.now();[...document.querySelectorAll('.preview-tabs button')].find(e=>e.textContent.includes('二维声韵表')).click();await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));return performance.now()-start;});
  timings.push(ms);await scope.getByRole('table',{name:'二维声韵表',exact:true}).waitFor();await page.waitForFunction(()=>document.querySelectorAll('.matrix-scroll tbody tr').length>0);
 };
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/m14-host.html');await page.locator('nav').getByRole('button',{name:'音系归纳',exact:true}).click();await scope.waitFor();
  await upload(path.join(inputs,'large.tsv'));
  if(!baseline){
   for(const [width,height,scale] of [[800,500,1],[1366,768,1],[1920,1080,1],[2560,1440,1],[1920,1080,1.5]])for(const theme of ['light','dark']){
    await page.setViewportSize({width,height});await page.evaluate(({theme,scale})=>{document.documentElement.dataset.theme=theme;document.documentElement.style.zoom=scale;document.documentElement.style.setProperty('--page-scale',scale);},{theme,scale});layouts.push({width,height,scale,theme,step:'import',...await ensureFits('确认导入并继续')});await capture(`import-${width}-${scale}-${theme}`);
   }
   await page.evaluate(()=>{document.documentElement.style.zoom=1;document.documentElement.style.setProperty('--page-scale','1');});await page.setViewportSize({width:1920,height:1080});await page.evaluate(()=>document.documentElement.dataset.theme='light');
  }
  await click('确认导入并继续');await idle();assert((await scope.innerText()).includes('5452 条记录'));
  await click('保存调类设置并继续');await click('保存声韵设置并继续');
  if(!baseline){
   assert.equal(await scope.locator('.word-statistic').count(),3);assert.equal(await scope.getByRole('table',{name:'声母统计',exact:true}).locator('tr:first-child td').count(),2);await capture('initial-word');await click('韵母 → 声母DOCX');await capture('final-word');
   await scope.getByLabel('搜索字头或 IPA',{exact:true}).fill('末');await scope.locator('.preview-entry.focused').waitFor();await click('二维声韵表XLSX');await scope.locator('.matrix-sheet .preview-entry.focused').waitFor();await click('清空');
   while(await scope.getByRole('button',{name:'上一组韵母',exact:true}).isEnabled())await click('上一组韵母');while(await scope.getByRole('button',{name:'上一组声母',exact:true}).isEnabled())await click('上一组声母');await click('下一组韵母');await click('下一组声母');await page.waitForFunction(()=>document.querySelectorAll('.matrix-scroll tbody tr').length>0);await click('上一组韵母');await click('上一组声母');
  }
  for(let i=0;i<8;i++)await sample();await capture('matrix');
  if(!baseline){
   for(const [width,height] of [[1366,768],[1920,1080],[2560,1440]])for(const theme of ['light','dark']){
    await page.setViewportSize({width,height});await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);await click('2 · 调类与调值');layouts.push({width,height,theme,step:'tone',...await ensureFits('保存调类设置并继续')});await click('3 · 声韵排序归并');layouts.push({width,height,theme,step:'symbols',...await ensureFits('保存声韵设置并继续')});await click('4 · 结果');await capture(`matrix-${width}-${theme}`);
   }
   await page.setViewportSize({width:1920,height:1080});await page.evaluate(()=>document.documentElement.dataset.theme='light');await click('生成三份结果');await idle();assert((await scope.innerText()).includes('三份结果已完整生成'));await click('选择目录保存三份结果');await idle();assert.equal((await fs.readdir(path.join(scratch,'saved'))).length,3);await capture('result');
   await click('1 · 导入');await upload(path.join(inputs,'dense.tsv'));await click('确认导入并继续');await idle();await click('保存调类设置并继续');await click('保存声韵设置并继续');await click('二维声韵表XLSX');await scope.locator('.matrix-sheet .preview-entry').first().waitFor();assert.equal(await scope.locator('.matrix-sheet .preview-entry').count(),60);
   await scope.getByLabel('搜索字头或 IPA',{exact:true}).fill('末');await scope.locator('.matrix-sheet .preview-entry.focused').waitFor();assert((await scope.locator('.cell-pages').innerText()).includes('167 / 167'));await click('清空');await scope.locator('.cell-pages button').first().click();assert((await scope.locator('.cell-pages').innerText()).includes('166 / 167'));
   await scope.getByLabel('搜索字头或 IPA',{exact:true}).fill('ma55');await scope.locator('.matrix-sheet .preview-entry.focused').waitFor();const first=await scope.locator('.preview-entry.focused').getAttribute('data-index');await scope.getByRole('button',{name:'下一个命中',exact:true}).click();assert.notEqual(await scope.locator('.preview-entry.focused').getAttribute('data-index'),first);
   checks.push('10000-record dense cell is bounded to 60 entries, last-record search and cell pagination');
   await page.locator('#tool-sidebar').getByRole('button',{name:'使用说明',exact:true}).click();await page.locator('.manual-document').waitFor();await page.locator('.manual-toc button[data-manual-chapter="m14"]').click();await page.locator('.manual-document[data-manual-chapter="m14"]').waitFor();
   const manual=page.locator('.manual-document[data-manual-chapter="m14"]');assert((await manual.innerText()).includes('双栏统计表'));const figures=manual.locator('img[src*="m14-r2"]');assert.equal(await figures.count(),5);
   for(const figure of await figures.all()){await figure.scrollIntoViewIfNeeded();await figure.evaluate(img=>img.decode());assert(await figure.evaluate(img=>img.naturalWidth===2560&&img.naturalHeight===1600));}await page.screenshot({path:path.join(out,'manual-reader.png')});checks.push('actual manual reader opens M14, revised prose and all five new Qt figures decode');
  }
  assert.deepEqual(errors,[]);success=true;
 }catch(e){errors.push(e.stack);await capture('failure');throw e;}
 finally{await fs.writeFile(path.join(out,baseline?'baseline-report.json':'browser-report.json'),JSON.stringify({success,baseline,checks,layouts,errors,timings,scope:'Windows isolated Chrome with production adapter, real durable host/child and synthetic TSV; no user database migration'},null,2));console.log(JSON.stringify({success,baseline,layouts:layouts.length,timings}));await browser.close();await server.close();worker.stdin.end();await new Promise(r=>worker.once('exit',r));const cleanup=spawnSync('powershell.exe',['-NoProfile','-ExecutionPolicy','Bypass','-File',path.join(root,'scripts/cleanup_test_scratch.ps1'),'-RunDirectory',out],{cwd:root,windowsHide:true,encoding:'utf8'});console.log(cleanup.stdout);if(cleanup.status!==0)console.error('Scratch cleanup retained:',cleanup.stderr);}
}
function largeTable(n,dense=false){const initials=['m','p','pʰ','b','t','tʰ','d','k','kʰ','g','f','v','s','z','h','n','l','ts','tsʰ','tʃ','tʃʰ','ʃ','ʒ','ŋ'],finals=['a','i','u','e','o','ai','au','ia','ua','an','in','un','am','im','um','aŋ','iŋ','uŋ'];return '字头\tIPA\t备注\n'+Array.from({length:n},(_,i)=>(i===n-1?'末':String.fromCodePoint(0x4e00+i%3000))+'\t'+(dense?'ma':initials[i%initials.length]+finals[Math.floor(i/initials.length)%finals.length])+(dense?'55':i%2?'35':'55')+'\t'+(i%5===0?'例字备注':'')).join('\n');}
main().catch(e=>{console.error(e);process.exitCode=1});
