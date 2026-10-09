// M17-R3: owned Chrome; only an isolated author content file is written.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{pathToFileURL}=require('node:url');
(async()=>{
 const root=path.resolve(__dirname,'../..'),out=path.join(root,'output/playwright/m17-r3',new Date().toISOString().replace(/[:.]/g,'-'));await fs.mkdir(out,{recursive:true});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const {m17AuthorTool}=await import(pathToFileURL(path.join(root,'frontend/tools/m17-author-server.mjs')));
 const original=await fs.readFile(path.join(root,'frontend/src/modules/ipa-plus/data/symbol-content.json'),'utf8'),target=path.join(out,'symbol-content.json');await fs.writeFile(target,original);
 const author=m17AuthorTool({contentPath:target}),server=await createServer({root:path.join(root,'frontend'),plugins:[author.plugin],optimizeDeps:{entries:['index.html']},server:{host:'127.0.0.1',port:0},logLevel:'error'});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright')),browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true}),page=await browser.newPage({viewport:{width:1366,height:768}}),errors=[],layouts=[],checks=[];
 const url=server.resolvedUrls.local[0],catalog=JSON.parse(await fs.readFile(path.join(root,'frontend/src/modules/ipa-plus/data/catalog.json'),'utf8'));
 let mediaServer;
 const symbol=value=>catalog.entries.find(e=>e.system==='ipa'&&e.insertText===value&&!e.isExample).id;
 const button=value=>page.locator(`[data-symbol-id="${symbol(value)}"]`),panel=page.locator('.m17-hover-panel');
 page.on('pageerror',e=>errors.push(e.message));
 const closeDetail=async()=>{if(await page.getByRole('button',{name:'收起',exact:true}).count())await page.getByRole('button',{name:'收起',exact:true}).click();};
 try{
  await page.goto(url+'#M17');await page.locator('.m17-body[data-loaded=true][data-font-ready=true]').waitFor();
  const editor=page.getByRole('textbox',{name:'国际音标文本',exact:true}),toggle=page.getByLabel('点击播放',{exact:true});
  if(!process.argv.includes('--media-only')){
  assert(!await toggle.isChecked());await editor.fill('甲𝼆乙');await editor.evaluate(e=>{e.focus();e.setSelectionRange(1,3);e.dispatchEvent(new Event('select'));});
  await button('p').click();assert.equal(await editor.inputValue(),'甲p乙');await page.getByRole('button',{name:'撤销',exact:true}).click();assert.equal(await editor.inputValue(),'甲𝼆乙');
  await toggle.check();const before=await editor.evaluate(e=>({text:e.value,start:e.selectionStart,end:e.selectionEnd}));
  await button('p').click();await page.getByText('此音标尚未添加演示内容。',{exact:true}).waitFor();assert.deepEqual(await editor.evaluate(e=>({text:e.value,start:e.selectionStart,end:e.selectionEnd})),before);assert.equal(await page.getByRole('button',{name:'重做',exact:true}).isEnabled(),true);
  await button('p').focus();await page.keyboard.press('Enter');assert.deepEqual(await editor.evaluate(e=>({text:e.value,start:e.selectionStart,end:e.selectionEnd})),before);
  await closeDetail();await toggle.uncheck();await button('b').click();assert.equal(await editor.inputValue(),'甲b乙');checks.push('default insertion, replacement, undo; play-only pointer/Enter preserve text, selection and redo; missing material and return to input');
  for(const size of [{width:1366,height:768},{width:1920,height:1080},{width:1024,height:720}])for(const theme of ['light','dark'])for(const zoom of [1,1.5]){
   await page.setViewportSize(size);await page.evaluate(({theme,zoom})=>{document.documentElement.dataset.theme=theme;document.documentElement.style.zoom=String(zoom);document.documentElement.style.setProperty('--page-scale',zoom);},{theme,zoom});
   for(const value of ['p','ʔ','h','u']){
    const b=button(value);await b.scrollIntoViewIfNeeded();await b.hover();await panel.waitFor();await page.waitForTimeout(80);
    const metric=await b.evaluate(e=>{const a=e.getBoundingClientRect(),p=document.querySelector('.m17-hover-panel').getBoundingClientRect();return {anchor:{left:a.left,top:a.top,right:a.right,bottom:a.bottom},panel:{left:p.left,top:p.top,right:p.right,bottom:p.bottom},view:[innerWidth,innerHeight]};});
    const a=metric.anchor,p=metric.panel;assert(p.right<=a.left||p.left>=a.right||p.bottom<=a.top||p.top>=a.bottom,JSON.stringify({size,theme,zoom,value,...metric}));assert(p.left>=-1&&p.top>=-1&&p.right<=metric.view[0]+1&&p.bottom<=metric.view[1]+1);
    const pointer={x:(a.left+a.right)/2,y:(a.top+a.bottom)/2};assert.equal(await page.evaluate(({x,y})=>document.elementFromPoint(x,y)?.closest('[data-symbol-id]')?.dataset.symbolId,pointer),symbol(value));
    assert(!/图4|用户|自主概述|构音和转写原则|中文名称优先依据/.test(await panel.innerText()));assert(!await panel.getByText('国际语音学会编，江荻译（2008）· 国际语音学会手册：国际音标使用指南',{exact:true}).count());
    layouts.push({size,theme,zoom,value,...metric});if(value==='h')await page.screenshot({path:path.join(out,`${size.width}-${theme}-${zoom}-near-edge.png`)});
    await panel.hover();await page.waitForTimeout(450);assert(await panel.isVisible(),'enter panel for reading');await page.mouse.move(1,1);await panel.waitFor({state:'hidden'});
   }
  }
  await page.evaluate(()=>{document.documentElement.style.zoom='1';document.documentElement.style.setProperty('--page-scale','1');});await page.setViewportSize({width:1366,height:768});
  await page.getByLabel('悬停介绍',{exact:true}).uncheck();await button('p').hover();await page.waitForTimeout(500);assert(!await panel.count());await page.getByLabel('悬停介绍',{exact:true}).check();await button('p').hover();await panel.waitFor();
  await page.mouse.move(1,1);await page.waitForTimeout(50);await button('p').hover();await page.waitForTimeout(450);assert(await panel.isVisible(),'quick re-entry retains open tooltip');
  await page.mouse.move(1,1);await panel.waitFor({state:'hidden'});await button('p').hover();await page.waitForTimeout(80);await page.mouse.move(1,1);await page.waitForTimeout(50);await button('p').hover();await panel.waitFor();
  await page.getByRole('textbox',{name:'查找符号、名称、码位或 CIN',exact:true}).fill('pʰ');const searchButton=page.locator('.m17-search-results [data-symbol-id]').first();await searchButton.hover();await panel.waitFor();assert((await panel.innerText()).includes('组合输入示例'));await searchButton.click({button:'right'});await page.locator('.m17-detail-panel').waitFor();await closeDetail();await searchButton.focus();await page.keyboard.press('Alt+Enter');await page.locator('.m17-detail-panel').waitFor();await closeDetail();checks.push('48 near-pointer layouts: symbol and pointer unobstructed, panel reachable, short content and chart citation, search/right-click/Alt+Enter and toggle');
  // Actual owner page write/read with auth, conflict and invalid paths, isolated file.
  const owner=await browser.newPage();owner.on('pageerror',e=>errors.push(e.message));await owner.goto(url+'__m17_author#'+author.capability);await owner.locator('#title').getByText('p · 清双唇爆发音',{exact:true}).waitFor();
  assert.equal(await owner.locator('#entries option').count(),625);await owner.locator('textarea[name=descriptionZh]').fill('独立维护测试说明。');await owner.locator('textarea[name=notesZh]').fill('第一行\n<em>作为纯文本保留</em>');await owner.getByRole('button',{name:'保存此音标',exact:true}).click();await owner.getByText('已保存到源码内容文件。',{exact:true}).waitFor();
  const saved=JSON.parse(await fs.readFile(target,'utf8'));assert.equal(saved.entries[symbol('p')].descriptionZh,'独立维护测试说明。');assert.equal(saved.entries[symbol('p')].notesZh,'第一行\n<em>作为纯文本保留</em>');assert.equal(await owner.locator('#preview em').count(),0);await owner.screenshot({path:path.join(out,'owner-editor.png')});
  const current=await fetch(url+'__m17_author/content',{headers:{'X-M17-Session':author.capability}}).then(r=>r.json());
  for(const [headers,body,status] of [[{},undefined,403],[{'X-M17-Session':author.capability,Origin:'https://example.invalid'},undefined,403],[{'X-M17-Session':author.capability,'Content-Type':'application/json'},{id:symbol('p'),revision:'old',content:{notesZh:'覆盖'}},409],[{'X-M17-Session':author.capability,'Content-Type':'application/json'},{id:symbol('p'),revision:current.revision,content:{media:{audio:'../private.wav'}}},400]]){
   const r=await fetch(url+'__m17_author/content',{method:body?'PUT':'GET',headers,body:body?JSON.stringify(body):undefined});assert.equal(r.status,status);
  }
  await owner.goto(url+'__m17_author#'+author.capability);await owner.locator('#title').getByText('p · 清双唇爆发音',{exact:true}).waitFor();assert.equal(await owner.locator('textarea[name=descriptionZh]').inputValue(),'独立维护测试说明。');
  const stale=await browser.newPage();await stale.goto(url+'__m17_author#'+author.capability);await stale.locator('textarea[name=descriptionZh]').fill('另一个窗口编辑');
  await owner.locator('textarea[name=notesZh]').fill('新的补充内容');await owner.getByRole('button',{name:'保存此音标',exact:true}).click();await owner.getByText('已保存到源码内容文件。',{exact:true}).waitFor();await stale.getByRole('button',{name:'保存此音标',exact:true}).click();await stale.getByText('内容已在另一窗口改变。请重新打开后合并，当前编辑保留。',{exact:true}).waitFor();assert.equal(await stale.locator('textarea[name=descriptionZh]').inputValue(),'另一个窗口编辑');
  checks.push('owner 625 choices, actual UTF-8 source save, reopen, plain text preview, missing auth/cross-origin/path rejection and stale-window conflict without data overwrite');
  }
  // Configure only the test page response; the real source remains untouched.
  const fixture={version:1,entries:{[symbol('p')]:{notesZh:'发行内容测试',media:{audio:'m17-media/audio.wav'}},[symbol('b')]:{media:{video:'m17-media/video.webm'}},[symbol('t')]:{media:{audio:'m17-media/audio.wav',video:'m17-media/video.webm'}},[symbol('d')]:{media:{animation:{renderer:'test-motion',version:1,config:{}}}},[symbol('k')]:{media:{audio:'m17-media/missing.wav'}}}};
  mediaServer=await createServer({root:path.join(root,'frontend'),plugins:[{name:'m17-fixture-only',enforce:'pre',load(id){if(id.split('?')[0].replaceAll('\\','/').endsWith('/ipa-plus/data/symbol-content.json'))return JSON.stringify(fixture);}}],optimizeDeps:{entries:['index.html']},server:{host:'127.0.0.1',port:0},logLevel:'error'});await mediaServer.listen();
  const wav=Buffer.alloc(44+16000);wav.write('RIFF',0);wav.writeUInt32LE(wav.length-8,4);wav.write('WAVEfmt ',8);wav.writeUInt32LE(16,16);wav.writeUInt16LE(1,20);wav.writeUInt16LE(1,22);wav.writeUInt32LE(16000,24);wav.writeUInt32LE(32000,28);wav.writeUInt16LE(2,32);wav.writeUInt16LE(16,34);wav.write('data',36);wav.writeUInt32LE(wav.length-44,40);
  for(let i=0;i<8000;i++)wav.writeInt16LE(Math.round(1000*Math.sin(i*2*Math.PI*220/16000)),44+i*2);
  const {execFileSync}=require('node:child_process'),ffmpeg=execFileSync('powershell.exe',['-NoProfile','-Command','(Get-Command ffmpeg -ErrorAction Stop).Source'],{encoding:'utf8',windowsHide:true}).trim();
  execFileSync(ffmpeg,['-hide_banner','-loglevel','error','-f','lavfi','-i','color=c=0x426b48:s=160x90:r=20','-t','0.7','-an','-c:v','libvpx',path.join(out,'demo.webm')],{windowsHide:true});const videoBytes=await fs.readFile(path.join(out,'demo.webm'));
  await page.route('**/m17-media/audio.wav',route=>route.fulfill({contentType:'audio/wav',body:wav}));await page.route('**/m17-media/video.webm',route=>route.fulfill({contentType:'video/webm',body:Buffer.from(videoBytes)}));await page.route('**/m17-media/missing.wav',route=>route.fulfill({status:404,body:''}));
  await page.goto(mediaServer.resolvedUrls.local[0]+'#M17');await page.locator('.m17-body[data-loaded=true][data-font-ready=true]').waitFor();await editor.fill('甲𝼆乙');await page.getByLabel('点击播放',{exact:true}).check();const preserved=await editor.inputValue();
  for(const value of ['p','b','t']){await closeDetail();await button(value).click();const selector=value==='p'?'audio':'video';await page.waitForFunction(selector=>{const e=document.querySelector('.m17-playback '+selector);return e?.currentTime>0&&e.readyState>=2;},selector);assert.equal(await editor.inputValue(),preserved);if(value==='t')assert(await page.locator('.m17-playback video').evaluate(e=>e.muted));}
  await page.evaluate(async()=>{const {registerSymbolAnimation}=await import('/src/modules/ipa-plus/playback.ts');window.motionStop=0;window.motionAbort=0;registerSymbolAnimation('test-motion',({container,signal})=>{container.textContent='实时接口验证';signal.addEventListener('abort',()=>window.motionAbort++);return ()=>window.motionStop++;});});
  await closeDetail();await button('d').click();await page.getByText('实时接口验证',{exact:true}).waitFor();await closeDetail();assert.equal(await page.evaluate(()=>window.motionStop),1);assert.equal(await page.evaluate(()=>window.motionAbort),1);
  await button('k').click();await page.getByText('演示素材无法播放，请检查素材文件。',{exact:true}).waitFor();assert.equal(await editor.inputValue(),preserved);await closeDetail();
  checks.push('decoded WAV audio, decoded WebM video, combined audio/video, missing-file feedback, real renderer registration and cleanup; test fixtures only, no physical listening or scientific animation');
  assert.equal(await fs.readFile(path.join(root,'frontend/src/modules/ipa-plus/data/symbol-content.json'),'utf8'),original);assert.deepEqual(errors,[]);
  await fs.writeFile(path.join(out,'report.json'),JSON.stringify({passed:true,scope:'Windows headless Chrome, source runtime, synthetic media and isolated owner source',layouts,checks,errors},null,2));console.log(out);
 }catch(error){await page.screenshot({path:path.join(out,'failure.png')});await fs.writeFile(path.join(out,'failure.json'),JSON.stringify({error:String(error),layouts,checks,errors},null,2));throw error;}
 finally{await browser.close();await mediaServer?.close();await server.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
