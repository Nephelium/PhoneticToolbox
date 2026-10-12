// P04-CUSTOM-APPEARANCE: owned browser/profile, no user media or external service.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url'),{spawnSync}=require('node:child_process'),{createHash}=require('node:crypto');
const root=path.resolve(__dirname,'../..');
async function main(){
 const out=path.join(root,'output/validation/custom-appearance','chrome-'+Date.now()),scratch=path.join(out,'scratch');await fs.mkdir(scratch,{recursive:true});
 const write=(name,data)=>fs.writeFile(path.join(out,name),JSON.stringify(data,null,2)+'\n');
 const marker={schema:'ptb-test-run/1',owner_pid:process.pid,state:'active',started_at:new Date().toISOString(),recipe:'Generated red/green PNG in memory; background upload/reload/removal; invalid input retention; paired custom palette; image sample; three highlight modes/two materials; all 18 module entrances; 12 responsive layouts'};
 await write('run.json',marker);const report={success:false,checks:[],cases:[],errors:[]};let server,browser,page;
 try{
  const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{noDiscovery:true,include:['vue']}});await server.listen();
  const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  browser=await chromium.launchPersistentContext(path.join(scratch,'profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--mute-audio'],viewport:{width:1440,height:900}});
  page=await browser.newPage();report.browser=browser.browser().version();page.on('pageerror',e=>report.errors.push(e.message));
  await page.goto(server.resolvedUrls.local[0]);const side=page.locator('#tool-sidebar');
  const settings=async()=>{await side.getByRole('button',{name:'设置',exact:true}).click();await page.locator('#settings-tab-appearance').click();};await settings();
  const fixture=await page.evaluate(()=>{const c=document.createElement('canvas');c.width=400;c.height=240;const x=c.getContext('2d');x.fillStyle='#ee2211';x.fillRect(0,0,200,240);x.fillStyle='#11bb66';x.fillRect(200,0,200,240);return c.toDataURL('image/png').split(',')[1];});
  const upload=page.locator('input[type=file][aria-label=上传背景图]');
  await upload.setInputFiles({name:'背景测试.png',mimeType:'image/png',buffer:Buffer.from(fixture,'base64')});
  await page.waitForFunction(()=>document.documentElement.dataset.backgroundImage==='on');assert.equal(await page.locator('.app-background-layer').count(),1);
  const image=await page.locator('.app-background-layer').evaluate(e=>getComputedStyle(e).backgroundImage);
  await page.getByRole('spinbutton',{name:'背景图高斯模糊值'}).fill('16');await page.getByRole('spinbutton',{name:'背景图高斯模糊值'}).press('Tab');
  await page.getByRole('spinbutton',{name:'背景图不透明度值'}).fill('65');await page.getByRole('spinbutton',{name:'背景图不透明度值'}).press('Tab');
  assert.deepEqual(await page.locator('.app-background-layer').evaluate(e=>({blur:e.style.filter,opacity:e.style.opacity})),{blur:'blur(16px)',opacity:'0.65'});
  assert.equal(await side.locator('input').evaluate(e=>getComputedStyle(e).backgroundColor),'rgba(0, 0, 0, 0)');
  assert.equal(await page.locator('.appearance-settings').evaluate(e=>getComputedStyle(e).backgroundColor),'rgba(0, 0, 0, 0)');
  assert.notEqual(await side.getByRole('button',{name:'首页',exact:true}).evaluate(e=>getComputedStyle(e).backgroundColor),'rgba(0, 0, 0, 0)');
  report.checks.push('Real file upload; single global layer; Gaussian blur/opacity applied; panels/search/text inputs transparent');
  await upload.setInputFiles({name:'坏图.png',mimeType:'image/png',buffer:Buffer.from('bad')});await page.getByRole('alert').filter({hasText:'无法读取'}).waitFor();
  assert.equal(await page.locator('.app-background-layer').evaluate(e=>getComputedStyle(e).backgroundImage),image);
  await upload.setInputFiles({name:'背景.svg',mimeType:'image/svg+xml',buffer:Buffer.from('<svg/>')});await page.getByRole('alert').filter({hasText:'支持 PNG'}).waitFor();
  assert.equal(await page.locator('.app-background-layer').evaluate(e=>getComputedStyle(e).backgroundImage),image);
  report.checks.push('Corrupt/disallowed files preserve the active background');
  const hex=page.getByRole('textbox',{name:'自定义色系 HEX 颜色'});await hex.fill('#abc');await hex.press('Tab');assert.equal(await hex.inputValue(),'#aabbcc');
  await page.waitForFunction(()=>document.documentElement.dataset.palette==='custom');
  await hex.fill('#123456');await hex.press('Tab');await page.waitForFunction(()=>document.documentElement.dataset.customPalette==='#123456');
  const before=await page.evaluate(()=>getComputedStyle(document.documentElement).getPropertyValue('--app'));
  await hex.fill('#oops');await hex.press('Tab');await page.getByRole('alert').filter({hasText:'请输入有效'}).waitFor();assert.equal(await page.evaluate(()=>document.documentElement.dataset.customPalette),'#123456');
  await page.getByRole('button',{name:'深色',exact:true}).click();assert.notEqual(await page.evaluate(()=>getComputedStyle(document.documentElement).getPropertyValue('--app')),before);
  await page.getByRole('button',{name:'从背景图取色',exact:true}).click();const sampler=page.locator('.background-sampler');await sampler.waitFor();await sampler.click({position:{x:10,y:10}});
  const sampled=await hex.inputValue();assert.match(sampled,/^#[0-9a-f]{6}$/);const channels=[1,3,5].map(i=>parseInt(sampled.slice(i,i+2),16));assert(channels[0]>200&&channels[1]<60&&channels[2]<60,sampled);
  report.checks.push('HEX shorthand/full/invalid values; custom palette light/dark; background image pixel sampling');
  await page.getByRole('combobox',{name:'按钮材质',exact:true}).selectOption('glass');
  for(const mode of ['auto','all','plain']){
   await page.getByRole('combobox',{name:'按钮高亮',exact:true}).selectOption(mode);
   await page.waitForTimeout(180);
   const glass=await page.getByRole('button',{name:'更换背景图',exact:true}).evaluate(e=>({bg:getComputedStyle(e).backgroundColor,filter:getComputedStyle(e).backdropFilter}));assert.match(glass.filter,/blur\(12px\)/);assert(glass.bg.includes('0.35'),glass.bg);report.cases.push({mode,glass});
  }
  await page.getByRole('combobox',{name:'按钮材质',exact:true}).selectOption('solid');assert.equal(await page.getByRole('button',{name:'更换背景图',exact:true}).evaluate(e=>getComputedStyle(e).backdropFilter),'none');
  await page.getByRole('combobox',{name:'按钮材质',exact:true}).selectOption('glass');
  await page.reload();await side.waitFor();await settings();assert.equal(await hex.inputValue(),sampled);assert.equal(await page.getByRole('combobox',{name:'按钮材质',exact:true}).inputValue(),'glass');
  await page.locator('#button-glass-opacity').fill('20');await page.locator('#button-glass-blur').fill('24');await page.waitForTimeout(180);
  const adjusted=await page.getByRole('button',{name:'更换背景图',exact:true}).evaluate(e=>({bg:getComputedStyle(e).backgroundColor,filter:getComputedStyle(e).backdropFilter}));assert(adjusted.bg.includes('0.2'),adjusted.bg);assert.match(adjusted.filter,/blur\(24px\)/);
  await page.reload();await side.waitFor();await settings();assert.equal(await page.locator('#button-glass-opacity').inputValue(),'20');assert.equal(await page.locator('#button-glass-blur').inputValue(),'24');
  report.checks.push('Independent glass opacity/blur changes material and persists after reload');
  assert.equal(await page.getByRole('spinbutton',{name:'背景图高斯模糊值'}).inputValue(),'16');assert.equal(await page.getByRole('spinbutton',{name:'背景图不透明度值'}).inputValue(),'65');await page.waitForFunction(()=>document.documentElement.dataset.backgroundImage==='on');
  report.checks.push('Glass transparency and blur across all three highlight modes; solid restoration; reload restores image, controls, custom colour and material');
  const modules=await page.evaluate(async()=>{const m=await import('/src/app/registry.ts');return m.modules.map(v=>({id:v.id,title:v.title}));});
  for(const m of modules){await side.getByRole('button',{name:m.title,exact:true}).click();await page.waitForTimeout(180);assert.equal(await page.locator('#tab-'+m.id).getAttribute('aria-selected'),'true');
   assert.equal(await page.locator('main').evaluate(e=>getComputedStyle(e).backgroundColor),'rgba(0, 0, 0, 0)');
   const plots=await page.locator('main').locator('canvas:visible,svg[role=img]:visible,svg[tabindex]:visible').evaluateAll(nodes=>nodes.map(e=>({tag:e.tagName,background:getComputedStyle(e).backgroundColor})));assert(plots.every(p=>p.background!=='rgba(0, 0, 0, 0)'),JSON.stringify({m,plots}));report.cases.push({module:m.id,plots:plots.length});
  }
  report.checks.push('All 18 module entrances retain the global layer; visible plot/canvas surfaces stay opaque (M10 desktop renderer checked separately)');
  await side.getByRole('button',{name:'使用说明',exact:true}).click();await page.locator('.manual-document').first().waitFor();assert.equal(await page.locator('main').evaluate(e=>getComputedStyle(e).backgroundColor),'rgba(0, 0, 0, 0)');await settings();
  for(const theme of ['light','dark'])for(const width of [1440,800])for(const zoom of [70,100,150]){
   await page.setViewportSize({width,height:900});await page.getByRole('button',{name:theme==='light'?'浅色':'深色',exact:true}).click();await page.evaluate(async zoom=>{const {setPageScale}=await import('/src/state/pageZoom.ts');setPageScale(zoom);},zoom);
   const overflow=await page.locator('.settings-panel:visible').evaluate(e=>e.scrollWidth-e.clientWidth);assert(overflow<=1,JSON.stringify({theme,width,zoom,overflow}));report.cases.push({theme,width,zoom,overflow});
  }
  await page.setViewportSize({width:1440,height:900});await page.evaluate(async()=>{const {setPageScale}=await import('/src/state/pageZoom.ts');setPageScale(100);});
  await page.getByRole('button',{name:'浅色',exact:true}).click();await page.screenshot({path:path.join(out,'appearance-light.png')});await page.getByRole('button',{name:'深色',exact:true}).click();await page.screenshot({path:path.join(out,'appearance-dark.png')});
  report.checks.push('Manual page uses global background; 12 light/dark/wide/narrow/zoom layouts fit');
  await page.getByRole('button',{name:'移除背景图',exact:true}).click();await page.waitForFunction(()=>document.documentElement.dataset.backgroundImage==='off');await page.reload();await side.waitFor();assert.equal(await page.evaluate(()=>document.documentElement.dataset.backgroundImage),'off');
  report.checks.push('Remove persists across reload and restores original surfaces');assert.deepEqual(report.errors,[]);report.success=true;
 }catch(error){report.failure=String(error);if(page)await page.screenshot({path:path.join(out,'failure.png')}).catch(()=>{});throw error;}
 finally{
  await browser?.close();await server?.close();await write('report.json',report);
  const files=[];async function inventory(dir){for(const entry of await fs.readdir(dir,{withFileTypes:true})){assert(!entry.isSymbolicLink());const file=path.join(dir,entry.name);if(entry.isDirectory())await inventory(file);else try{const bytes=await fs.readFile(file);files.push({path:path.relative(scratch,file),bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex')});}catch(e){if(e.code!=='ENOENT')throw e;}}}await inventory(scratch);await write('scratch-manifest.json',{schema:'ptb-test-scratch-files/1',files});await write('run.json',{...marker,state:'ready',test_outcome:report.success?'completed':'failed'});
  const cleanup=spawnSync('powershell.exe',['-NoProfile','-ExecutionPolicy','Bypass','-File',path.join(root,'scripts/cleanup_test_scratch.ps1'),'-RunDirectory',out],{windowsHide:true,encoding:'utf8'});if(cleanup.status!==0)await write('cleanup.json',{status:'retained',reason:cleanup.stderr||cleanup.stdout||'Cleanup exited '+cleanup.status});
  console.log(JSON.stringify({out,success:report.success,checks:report.checks.length,cases:report.cases.length,failure:report.failure}));
 }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
