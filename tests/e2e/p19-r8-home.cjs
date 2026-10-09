// P19-R8: actual shell navigation, renamed tools and home footer geometry.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
const expected=[['录音','唇形提取','参数估计','参数显示','EGG 信号分析','LPC 谱图'],['语音合成','声道工作台','发声类型合成','变速变调','语谱图转音频'],['国际音标表Plus','汉字转国际音标','MFA 自动标注','TextGrid标注','音系归纳','感知实验']];
async function main(){
 const out=path.join(root,'output/validation/p19-r8','chrome-'+Date.now());await fs.mkdir(out,{recursive:true});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:31084,watch:null}});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});const page=await browser.newPage({viewport:{width:1440,height:900}}),errors=[],checks=[],layouts=[];page.on('pageerror',e=>errors.push(e.message));
 await page.addInitScript(()=>{localStorage.setItem('ptb.v3.recent',JSON.stringify(['M17','M13','M12','M06','M07']));});
 try{
  await page.goto(server.resolvedUrls.local[0]);await page.locator('.home-footer').waitFor();
  const nav=await page.locator('.nav-group').evaluateAll(groups=>groups.map(g=>[...g.querySelectorAll('.nav-item>span:not(.ipa-icon)')].map(e=>e.textContent.trim())));assert.deepEqual(nav,expected);
  const cards=await page.locator('.tool-group').evaluateAll(groups=>groups.map(g=>[...g.querySelectorAll('button strong')].map(e=>e.textContent.trim())));assert.deepEqual(cards,expected);
  assert.deepEqual(await page.locator('.recent-list button').evaluateAll(es=>es.map(e=>{const copy=e.cloneNode(true);copy.querySelector('.ipa-icon')?.remove();return copy.textContent.trim();})),['国际音标表Plus','汉字转国际音标','TextGrid标注','语音合成','发声类型合成']);
  for(const [id,title]of [['M17','国际音标表Plus'],['M13','汉字转国际音标'],['M12','TextGrid标注']]){await page.locator('nav .nav-item').filter({hasText:title}).click();await page.locator('#tab-'+id).waitFor();assert((await page.locator('#tab-'+id).textContent()).includes(title));await page.getByRole('tab',{name:'首页',exact:true}).click();}
  checks.push('Sidebar and all-tool cards use the requested three orders; renamed tools retain M17/M13/M12 tab IDs, and old recent IDs display their new names');
  const fit=[[1440,900,100],[1680,1050,100],[1920,1080,100],[1920,1080,125],[1920,1080,150],[1920,900,100],[1366,768,100],[2560,1440,150],[3840,2160,150]];
  for(const [width,height,scale]of fit){await page.setViewportSize({width,height});await page.evaluate(async scale=>(await import('/src/state/pageZoom.ts')).setPageScale(scale),scale);await page.waitForTimeout(250);
   const geometry=await page.evaluate(()=>{const main=document.querySelector('#main-content'),footer=document.querySelector('.home-footer'),m=main.getBoundingClientRect(),f=footer.getBoundingClientRect(),row=document.querySelector('.tool-group>button').getBoundingClientRect();return {main:{top:m.top,bottom:m.bottom,clientHeight:main.clientHeight,scrollHeight:main.scrollHeight,scrollTop:main.scrollTop},footer:{top:f.top,bottom:f.bottom},rowHeight:row.height};});layouts.push({width,height,scale,...geometry});assert(geometry.footer.bottom<=geometry.main.bottom+1,JSON.stringify(layouts.at(-1)));assert(geometry.main.scrollHeight<=geometry.main.clientHeight+2,JSON.stringify(layouts.at(-1)));
   if((width===1920&&scale!==125)||width===1366)await page.screenshot({path:path.join(out,`home-${width}-${height}-${scale}.png`)});
  }
  checks.push('Home footer and attribution button fit without scrolling at 1440/1680/1920/1366/2560/3840 desktop viewports, including full-HD at 150% with five recent items');
  for(const [width,height]of [[900,700],[390,844],[1920,600]]){await page.setViewportSize({width,height});await page.evaluate(async()=>(await import('/src/state/pageZoom.ts')).setPageScale(100));await page.waitForTimeout(250);await page.locator('.home-footer button').scrollIntoViewIfNeeded();assert(await page.locator('.home-footer button').isVisible());await page.locator('.home-footer button').click();await page.getByRole('dialog').waitFor();await page.getByRole('dialog').getByRole('button',{name:/关闭/}).first().click();layouts.push({width,height,scope:'small viewport retains scrolling and accessible attribution'});}
  checks.push('Narrow and very short windows retain scrolling to all tools and a working attribution dialog');
  await page.setViewportSize({width:1920,height:1080});await page.evaluate(()=>{document.documentElement.dataset.theme='dark';document.querySelector('#main-content').scrollTop=0;});await page.waitForTimeout(250);await page.screenshot({path:path.join(out,'home-dark-1920.png')});
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success:true,checks,layouts,errors},null,2));console.log(JSON.stringify({out,checks,layouts}));
 }catch(e){await page.screenshot({path:path.join(out,'failed.png')});await fs.writeFile(path.join(out,'failed.json'),JSON.stringify({error:String(e),checks,layouts,errors},null,2));throw e;}
 finally{await browser.close();await server.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
