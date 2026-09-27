// Actual browser File input / image decode / WAV decode; no fabricated analysis jobs.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 let success=false;
 const out=path.join(root,'output/validation/p04-unify','preview-'+Date.now());await fs.mkdir(out,{recursive:true});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0},optimizeDeps:{entries:['tests/p04-preview.html']}});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--mute-audio']}),checks=[],matrix=[];
 const wav=await fs.readFile(path.join(root,'frontend/src/assets/SYN-EGG-44100.wav'));
 try{
 for(const [id,title,selector] of [['M01','参数估计','.m01-page'],['M03','EGG 信号分析','.egg-page'],['M04','LPC 谱图','.lpc-page'],['M09','语谱图转音频','.m09-page']]){
  const page=await browser.newPage({viewport:{width:1280,height:800}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
  try{
   await page.goto(server.resolvedUrls.local[0]+'tests/p04-preview.html');await page.locator('nav').getByRole('button',{name:title,exact:true}).click();
   const mod=page.locator(selector);await mod.waitFor();let payload={name:'语音 ɑ̃˥.wav',mimeType:'audio/wav',buffer:wav};
   if(id==='M09'){const base64=await page.evaluate(()=>{const c=document.createElement('canvas');c.width=100;c.height=65;const x=c.getContext('2d');x.fillStyle='white';x.fillRect(0,0,100,65);x.fillStyle='black';x.fillRect(0,24,100,3);x.fillRect(0,44,100,3);return c.toDataURL('image/png').split(',')[1];});payload={name:'合成语谱图.png',mimeType:'image/png',buffer:Buffer.from(base64,'base64')};}
   await mod.locator('input[type=file]').first().setInputFiles(payload);
   const select=async()=>{if(id==='M01')await mod.locator('.file-row').filter({hasText:payload.name}).click();else await mod.getByLabel(id==='M03'?'EGG 音频文件':id==='M04'?'LPC 音频文件':'语谱图图片',{exact:true}).selectOption({label:payload.name});};
   const capture=async state=>{for(const [width,height] of [[1280,800],[1920,1080]])for(const theme of ['light','dark'])for(const scale of [70,100,150]){
    await page.setViewportSize({width,height});await page.evaluate(async({theme,scale})=>{document.documentElement.dataset.theme=theme;(await import('/src/state/pageZoom.ts')).setPageScale(scale);},{theme,scale});await page.waitForTimeout(140);await page.locator('main').evaluate(el=>el.scrollTop=0);
    const size=await mod.evaluate(el=>({width:el.clientWidth,scroll:el.scrollWidth}));assert(size.scroll<=size.width+2,`${id} ${state} overflow ${JSON.stringify(size)}`);
    await page.screenshot({path:path.join(out,`${id}-${state}-${theme}-${width}-${scale}.png`),animations:'disabled'});matrix.push({id,state,width,height,theme,scale,...size});
   }};
   await page.evaluate(()=>window.__p04.delay=true);await select();await page.waitForFunction(()=>window.__p04.waiting);await capture('loading');await page.evaluate(()=>{window.__p04.delay=false;window.__p04.release();});
   await mod.locator(id==='M09'?'.image-frame img':'.wave-track svg').first().waitFor();await capture('loaded');
   if(id==='M01'){
    await page.setViewportSize({width:1280,height:800});await page.evaluate(async()=>(await import('/src/state/pageZoom.ts')).setPageScale(100));
    await page.getByRole('button',{name:'选择输出参数',exact:true}).click();await page.getByRole('button',{name:'全不选',exact:true}).click();await page.locator('.parameter-grid input[value="pF0"]').check();await page.getByRole('button',{name:'应用到草稿',exact:true}).click();await page.locator('nav').getByRole('button',{name:'LPC 谱图',exact:true}).click();await page.locator('nav').getByRole('button',{name:title,exact:true}).click();assert.equal((await mod.locator('.parameter-count strong').innerText()).trim(),'1');checks.push('M01 parameter draft retained across tabs with real WAV');
   }
   await page.evaluate(()=>window.__p04.fail=true);await select();await mod.getByRole('alert').filter({hasText:'受控文件读取失败'}).waitFor();await capture('error');
   await page.evaluate(()=>window.__p04.fail=false);await select();await mod.locator(id==='M09'?'.image-frame img':'.wave-track svg').first().waitFor();assert.deepEqual(errors,[]);checks.push(id+' real file loading/loaded/error matrix and recovery; no scientific task claim');
  }catch(e){await page.screenshot({path:path.join(out,id+'-failure.png')});await fs.writeFile(path.join(out,id+'-failure.txt'),String(e)+'\n'+await page.locator('body').innerText());throw e;}finally{await page.close();}
 }
 success=true;
 }finally{await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success,checks,matrix,platform:process.platform,scope:'Real File reads, controlled read delay/error. No task provider, server DB or physical device assertions.'},null,2));console.log(out);await browser.close();await server.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
