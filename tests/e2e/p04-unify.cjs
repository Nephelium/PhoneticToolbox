// P04-UNIFY: actual AppShell + owned M12 file bridge; no database or device claims.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {spawn}=require('node:child_process'),{createInterface}=require('node:readline'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..'),before=process.argv.includes('--before');
async function main(){
 let success=false;
 const out=path.join(root,'output/validation/p04-unify',`${before?'before':'after'}-${Date.now()}`);await fs.mkdir(out,{recursive:true});
 const worker=spawn(path.join(root,'.venv/m09-ui/Scripts/python.exe'),['-X','utf8','scripts/m12_ui_bridge.py'],{cwd:root,windowsHide:true,env:{...process.env,PYTHONPATH:['backend/src','desktop/src','packages/phonetic_core/src'].map(p=>path.join(root,p)).join(';')}});
 const pending=new Map();let n=0,resolveReady,rejectReady;const ready=new Promise((r,j)=>{resolveReady=r;rejectReady=j;});
 createInterface({input:worker.stdout}).on('line',line=>{const v=JSON.parse(line);if(v.ready)resolveReady(v);else{pending.get(v.id)?.(v);pending.delete(v.id);}});worker.stderr.on('data',d=>process.stderr.write(d));worker.once('error',rejectReady);worker.once('exit',code=>rejectReady(Error('Bridge exited '+code)));
 const bridge=await ready;const rpc=data=>new Promise(resolve=>{const id=++n;pending.set(id,resolve);worker.stdin.write(JSON.stringify({...data,rpc_id:id})+'\n');});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0},optimizeDeps:{entries:['tests/m12-live.html']},plugins:[{name:'p04-owned-rpc',configureServer(s){s.middlewares.use('/__m12',(req,res)=>{let body='';req.on('data',d=>body+=d);req.on('end',async()=>{res.setHeader('Content-Type','application/json');res.end(JSON.stringify(await rpc(JSON.parse(body))));});});}}]});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--mute-audio']}),page=await browser.newPage({viewport:{width:1280,height:800}}),errors=[],checks=[],matrix=[];
 page.on('pageerror',e=>errors.push(e.message));
 const pages=[['M01','参数估计','.m01-page'],['M03','EGG 信号分析','.egg-page'],['M04','LPC 谱图','.lpc-page'],['M09','语谱图转音频','.m09-page'],['M12','TextGrid标注','.annotation-page']];
 const snap=async name=>{await page.evaluate(()=>document.activeElement?.blur());await page.screenshot({path:path.join(out,name+'.png'),animations:'disabled'});};
 const appearance=async(theme,scale)=>{await page.evaluate(async({theme,scale})=>{document.documentElement.dataset.theme=theme;const m=await import('/src/state/pageZoom.ts');m.setPageScale(scale);},{theme,scale});await page.waitForTimeout(180);};
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/m12-live.html');
  for(const [id,title,selector] of pages){
   await page.locator('nav').getByRole('button',{name:title,exact:true}).click();await page.locator(selector).waitFor();
   await appearance('light',100);await snap(id+'-empty-light-1280-100');
   if(!before){assert.equal(await page.locator(selector+' h1').count(),0);assert.equal(await page.locator(selector).getByRole('button',{name:'关闭模块',exact:true}).count(),0);await page.locator(selector+' .module-toolbar').waitFor();
    for(const [width,height] of [[1280,800],[1920,1080]])for(const theme of ['light','dark'])for(const scale of [70,100,150]){
     await page.setViewportSize({width,height});await appearance(theme,scale);await page.locator('main').evaluate(el=>el.scrollTop=0);
     const info=await page.locator(selector).evaluate(el=>{const toolbar=el.querySelector('.module-toolbar'),r=toolbar.getBoundingClientRect();return {scroll:el.scrollWidth,width:el.clientWidth,toolbar:{x:r.x,right:r.right,y:r.y,bottom:r.bottom},buttons:[...toolbar.querySelectorAll('button')].map(b=>({text:b.textContent,rect:b.getBoundingClientRect().toJSON()}))};});
     assert(info.scroll<=info.width+2,`${id} content overflow ${JSON.stringify(info)}`);for(const b of info.buttons)assert(b.rect.x>=0&&b.rect.right<=width+2,`${id} button clipped: ${b.text}`);
     await snap(`${id}-empty-${theme}-${width}-${scale}`);matrix.push({id,width,height,theme,scale,...info});
    }
    checks.push(id+' heading removed, toolbar reachable across 12 viewport/theme/scale combinations');
   }
  }
  await page.setViewportSize({width:1280,height:800});await appearance('light',100);
  await page.getByRole('button',{name:'选择语料文件夹',exact:true}).click();await page.getByRole('button',{name:'audio_recording.wav',exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('.annotation-page')?.getAttribute('aria-busy')==='false'&&document.querySelector('.annotation-grid'));
  await snap('M12-loaded-light-1280-100');await appearance('dark',100);await snap('M12-loaded-dark-1280-100');
  checks.push('M12 real WAV/TextGrid/lip loaded through local file bridge');
  if(!before){
   const stateMatrix=async state=>{
    for(const [width,height] of [[1280,800],[1920,1080]])for(const theme of ['light','dark'])for(const scale of [70,100,150]){
     await page.setViewportSize({width,height});await appearance(theme,scale);await page.locator('main').evaluate(el=>el.scrollTop=0);
     const size=await page.locator('.annotation-page').evaluate(el=>({width:el.clientWidth,scroll:el.scrollWidth}));assert(size.scroll<=size.width+2,`${state} horizontal overflow`);
     await snap(`M12-${state}-${theme}-${width}-${scale}`);matrix.push({id:'M12',state,width,height,theme,scale,...size});
    }
   };
   await stateMatrix('loaded');
   let release,seen;const held=new Promise(r=>seen=r),gate=new Promise(r=>release=r);let intercepted=false;
   const delay=async route=>{if(!intercepted&&route.request().postDataJSON()?.op==='read'){intercepted=true;const response=await route.fetch();seen();await gate;await route.fulfill({response});}else await route.continue();};
   await page.route('**/__m12',delay);await page.getByRole('button',{name:'second.wav',exact:true}).click();await held;
   assert.equal(await page.locator('.annotation-page').getAttribute('aria-busy'),'true');await stateMatrix('loading');release();
   await page.waitForFunction(()=>document.querySelector('.annotation-page')?.getAttribute('aria-busy')==='false');await page.unroute('**/__m12',delay);
   const fail=async route=>{if(route.request().postDataJSON()?.op==='read')await route.fulfill({json:{error:'受控读取失败，请重新选择文件。'}});else await route.continue();};
   await page.route('**/__m12',fail);await page.getByRole('button',{name:'audio_recording.wav',exact:true}).click();await page.getByRole('alert').filter({hasText:'受控读取失败'}).waitFor();await stateMatrix('error');await page.unroute('**/__m12',fail);
   await page.getByRole('button',{name:'audio_recording.wav',exact:true}).click();await page.waitForFunction(()=>document.querySelector('.annotation-page')?.getAttribute('aria-busy')==='false'&&document.querySelector('.annotation-grid'));
   await page.setViewportSize({width:1280,height:800});await appearance('light',100);await page.getByLabel('唇形共同偏移毫秒').fill('21');await page.getByLabel('唇形共同偏移毫秒').press('Tab');
   await page.locator('nav').getByRole('button',{name:'LPC 谱图',exact:true}).click();await page.locator('nav').getByRole('button',{name:'TextGrid标注',exact:true}).click();assert.equal(await page.getByLabel('唇形共同偏移毫秒').inputValue(),'21');
   const failSave=async route=>{if(route.request().postDataJSON()?.op==='annotation_save')await route.fulfill({json:{error:'受控保存失败，编辑保留。'}});else await route.continue();};
   await page.route('**/__m12',failSave);await page.getByLabel('关闭 TextGrid标注',{exact:true}).click();const dialog=page.getByRole('dialog',{name:'保存标注修改？'});await dialog.getByRole('button',{name:'保存修改并关闭',exact:true}).click();await dialog.getByRole('alert').waitFor();assert.equal(await page.getByLabel('唇形共同偏移毫秒').inputValue(),'21');
   await appearance('dark',150);await dialog.screenshot({path:path.join(out,'M12-close-save-failure-dark-150.png'),animations:'disabled'});const rect=await dialog.boundingBox();assert(rect.x>=0&&rect.y>=0&&rect.x+rect.width<=1280&&rect.y+rect.height<=800);
   await dialog.getByRole('button',{name:'取消关闭'}).click();await page.unroute('**/__m12',failSave);await page.getByLabel('关闭 TextGrid标注',{exact:true}).click();await dialog.getByRole('button',{name:'保存修改并关闭',exact:true}).click();await page.locator('.annotation-page').waitFor({state:'detached'});
   const saved=(await rpc({op:'inspect'})).value;assert.equal(saved.lip_offset,.021);
   checks.push('M12 loaded/loading/error 36 combinations; controlled read/save failure recovery, tab retention, cancel close, successful close and real lip file reread');
  }
  assert.deepEqual(errors,[]);success=true;
 }catch(e){await snap('failure');await fs.writeFile(path.join(out,'failure.txt'),String(e)+'\n'+await page.locator('body').innerText());throw e;}
 finally{await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success,before,checks,matrix,errors,bridge,platform:process.platform,dpi:'Browser deviceScaleFactor=1; physical system DPI not changed or validated'},null,2));console.log(out);await browser.close();await server.close();worker.stdin.end();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
