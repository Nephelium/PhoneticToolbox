// Real workbench UI with a synthetic updater channel. No real update is downloaded or applied.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {createHash,randomUUID} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {createServer} from 'vite';
const frontend=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),repo=path.dirname(frontend);
const preview=process.argv.includes('--preview');
const notesOption=process.argv.indexOf('--notes-file');
const actualNotes=notesOption<0?null:await fs.readFile(path.resolve(process.argv[notesOption+1]),'utf8');
const notesIdentity=actualNotes===null?null:{path:path.resolve(process.argv[notesOption+1]),sha256:createHash('sha256').update(actualNotes).digest('hex')};
const folder=path.join(repo,'output','validation','updates-ui',`${preview?'preview':'check'}-${randomUUID()}`),scratch=path.join(folder,'scratch');
for(let cursor=folder;;cursor=path.dirname(cursor)){
  const stat=await fs.lstat(cursor).catch(error=>{if(error.code!=='ENOENT')throw error;});
  if(stat?.isSymbolicLink())throw Error(`Linked test directory: ${cursor}`);
  if(cursor===path.dirname(cursor))break;
}
await fs.mkdir(scratch,{recursive:true});
const write=(name,value)=>fs.writeFile(path.join(folder,name),JSON.stringify(value,null,2)+'\n');
const marker={schema:'ptb-test-run/1',owner_pid:process.pid,state:'active',started_at:new Date().toISOString(),recipe:'Actual App.vue, shared ModalDialog and UpdatePanel with a synthetic updater channel; generated HTML/Vite cache/owned Chrome profile only; no real release network, downloads, installers or user files'};
await write('run.json',marker);
await fs.writeFile(path.join(scratch,'index.html'),`<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><title>更新弹窗预览（演示数据）</title><div id="app"></div><script type="module" src="/@fs/${path.join(scratch,'main.ts').split(path.sep).join('/')}"></script></html>`);
await fs.writeFile(path.join(scratch,'main.ts'),`
import {createApp} from 'vue';
import App from '/src/App.vue';
import {initializePlatform} from '/src/platform/desktop.ts';
import {installPageZoom} from '/src/state/pageZoom.ts';
import {installUpdatesChannel} from '/src/platform/updates.ts';
installPageZoom();await initializePlatform();
const requests=[],cancelled=[],prefs={source:'auto',channel:'stable',autoCheck:true,currentVersion:'3.0.0',packageKind:'portable',checkIntervalHours:6,applyAvailable:true};
const callbacks=()=>{const values=new Set();return{connect:fn=>values.add(fn),disconnect:fn=>values.delete(fn),emit:(id,value)=>values.forEach(fn=>fn(id,JSON.stringify(value)))}};
const ready=callbacks(),progress=callbacks();
const fixture={mode:'available',requests,cancelled,preferences:prefs,offline:()=>installUpdatesChannel(undefined)};window.fixture=fixture;
const release={id:'owned-release',version:'3.0.1',source:'server',notes:[
 '- **更新窗口：** 更新说明分条展示，重点加粗，较长的内容可以在卡片内滚动阅读。',
 '- **版本对照更清楚：** 当前版本和新版本一起显示，方便查看这次更新。',
 '- **关于页按钮统一高亮：** 官方网站、问题反馈、开源与学术致谢、内置字体版权与许可使用一致的按钮样式。',
 '- **确认后下载：** 阅读说明后选择下载更新，确认之后才开始下载。',
 '- **完整校验：** 下载完成后校验更新包，通过后可以退出并更新。',
 '- **退出前保存：** 更新时仍会检查正在进行的工作和未保存内容。',
 '- **更新来源：** 保留自动选择、国内服务器与 GitHub，来源状态可以展开查看。',
 '- **外观跟随设置：** 更新窗口随应用的浅色、深色、配色和字号显示。'
].join('\\n'),publishedAt:'2026-10-12',packages:[{kind:'portable',name:'PhoneticToolbox-windows-x64-portable.zip',size:104857600,sha256:'a'.repeat(64),source:'server'}]};fixture.release=release;
installUpdatesChannel({ready,progress,cancel:id=>cancelled.push(id),request(id,payload){const {operation,args}=JSON.parse(payload);requests.push({id,operation,args});
 const reply=value=>{if(!cancelled.includes(id))ready.emit(id,{ok:true,value})};
 if(operation==='preferences')reply({...prefs});
 if(operation==='configure'){Object.assign(prefs,args);reply({...prefs});}
 if(operation==='acknowledge')reply({acknowledged:true});
 if(operation==='check')setTimeout(()=>reply({status:fixture.mode==='incomplete'?'incomplete':'available',currentVersion:prefs.currentVersion,channel:prefs.channel,checkedAt:1000000,region:{source:prefs.source==='auto'?'system':'manual',country:prefs.source==='auto'?'CN':null,label:prefs.source==='auto'?'系统地区线索':'手动选择',preferredSource:prefs.source==='github'?'github':'server'},preferredSource:prefs.source==='github'?'github':'server',sources:{server:fixture.mode==='incomplete'?{status:'error',message:'连接失败或超时，请稍后重试。'}:{status:'checked',version:release.version,count:1},github:{status:'error',message:'发布来源返回 HTTP 404。'}},candidate:fixture.mode==='incomplete'?null:release,shouldPrompt:!args.manual,packageKind:'portable'}),fixture.mode==='slow'?250:15);
 if(operation==='download'){const mode=fixture.mode;let received=0;const timer=setInterval(()=>{received+=26214400;if(cancelled.includes(id)){clearInterval(timer);return;}progress.emit(id,{phase:'downloading',source:'server',received,total:104857600});if(received===104857600){clearInterval(timer);setTimeout(()=>{if(cancelled.includes(id))return;if(mode==='checksum-error'){ready.emit(id,{ok:false,error:{code:'CHECKSUM_MISMATCH',message:'更新包校验失败，请重新下载。'}});return;}progress.emit(id,{phase:'verified',source:'server',received,total:104857600});reply({downloadId:'owned-download',name:release.packages[0].name,size:104857600,sha256:'a'.repeat(64),source:'server',kind:'portable',verified:true,applyAvailable:true});},250);}},mode==='download-slow'?150:60);}
 if(operation==='apply')reply({started:true});
}});
createApp(App).mount('#app');
`);
let server,context,page;
const checks=[],errors=[];let success=false;
try{
 server=await createServer({root:frontend,cacheDir:path.join(scratch,'vite-cache'),optimizeDeps:{noDiscovery:true,include:['vue']},server:{host:'127.0.0.1',port:0,watch:null,fs:{allow:[repo]}},logLevel:'error'});await server.listen();
 const runtime=process.env.PTB_PLAYWRIGHT??path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
 const {chromium}=await import(pathToFileURL(runtime));
 context=await chromium.launchPersistentContext(path.join(scratch,'chrome-profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:!preview,viewport:{width:1440,height:900},args:['--no-first-run','--mute-audio']});
 page=context.pages()[0]||await context.newPage();page.on('pageerror',error=>errors.push(error.message));
 await page.addInitScript(()=>{localStorage.setItem('ptb.v3.theme',JSON.stringify('dark'));localStorage.setItem('ptb.v3.palette',JSON.stringify('everforest'));});
 const url=`http://127.0.0.1:${server.httpServer.address().port}/@fs/${path.join(scratch,'index.html').split(path.sep).join('/')}`;
 await page.goto(url);await page.locator('.tool-group').last().waitFor();await page.getByLabel('新版本提示').waitFor();
 assert.equal(await page.evaluate(()=>window.fixture.requests.filter(r=>r.operation==='check').length),1);
 assert.equal(await page.evaluate(()=>window.fixture.requests.filter(r=>r.operation==='acknowledge').length),1);
 await page.getByRole('button',{name:'查看更新',exact:true}).click();
 const dialog=page.getByRole('dialog',{name:'发现新版本',exact:true});await dialog.waitFor();
 await dialog.getByText('当前版本：v3.0.0').waitFor();assert.equal(await dialog.locator('.release-notes li').count(),8);assert.equal(await dialog.locator('.release-notes strong').count(),8);
 assert.equal(await dialog.getByRole('button',{name:'退出并更新',exact:true}).count(),0);
 assert.equal(await page.evaluate(()=>window.fixture.requests.filter(r=>r.operation==='download').length),0);
 checks.push({name:'actual AppShell opens the shared update dialog with version comparison, eight list items and eight bold topics; startup never downloads'});
 if(actualNotes!==null){
   await page.evaluate(notes=>{window.fixture.release.notes=notes;window.fixture.release.version='3.0.3';},actualNotes);
   await dialog.locator('summary').click();await dialog.getByRole('button',{name:'检查更新',exact:true}).click();
   await dialog.locator('.release-markdown strong').getByText('PhoneticToolbox 3.0.3 更新啦！',{exact:true}).waitFor();
   await dialog.locator('summary').click();
   // Expected structure comes from the author's current Markdown, including
   // revisions made during release review; do not bind a writer to old counts.
   assert.equal(await dialog.locator('.release-markdown h2').count(),actualNotes.match(/^## /gm)?.length??0);
   assert.equal(await dialog.locator('.release-markdown li').count(),actualNotes.match(/^- /gm)?.length??0);
   assert.equal(await dialog.locator('.release-markdown script,.release-markdown iframe,.release-markdown img').count(),0);
   for(const link of await dialog.locator('.release-markdown a').all())assert.equal(await link.getAttribute('rel'),'noopener noreferrer');
   for(const theme of ['light','dark'])for(const [width,height,scale,font] of [[1440,900,100,14],[1280,800,150,18],[560,760,100,18]]){
     await page.setViewportSize({width,height});await page.evaluate(async({theme,scale,font})=>{const {setPageScale}=await import('/src/state/pageZoom.ts');setPageScale(scale);document.documentElement.dataset.theme=theme;document.documentElement.style.fontSize=font+'px';},{theme,scale,font});
     await page.evaluate(()=>document.fonts.ready);
     const layout=await dialog.evaluate(el=>{const notes=el.querySelector('.release-notes'),action=el.querySelector('.update-footer .primary'),box=el.getBoundingClientRect(),button=action.getBoundingClientRect();return{overflow:el.scrollWidth>el.clientWidth+1,notesScroll:notes.scrollHeight>notes.clientHeight,buttonVisible:button.top>=0&&button.bottom<=innerHeight,dialogVisible:box.top>=0&&box.bottom<=innerHeight};});
     assert.equal(layout.overflow,false);assert.equal(layout.notesScroll,true);assert.equal(layout.buttonVisible,true);assert.equal(layout.dialogVisible,true);
     await dialog.screenshot({path:path.join(folder,`notes-${theme}-${width}-${scale}.png`)});
     await dialog.locator('.release-notes').evaluate(el=>el.scrollTop=el.scrollHeight);
     await dialog.screenshot({path:path.join(folder,`notes-end-${theme}-${width}-${scale}.png`)});
     await dialog.locator('.release-notes').evaluate(el=>el.scrollTop=0);
     checks.push({name:'actual 3.0.3 notes render safely with headings, lists, link, scroll and visible actions',theme,width,height,scale,font,layout});
   }
   assert.equal(await page.evaluate(()=>window.fixture.requests.filter(r=>r.operation==='download'||r.operation==='apply').length),0);
   assert.deepEqual(errors,[]);success=true;
 }else if(preview){
   await dialog.screenshot({path:path.join(folder,'preview.png')});await write('preview.json',{url,browser:context.browser().version(),fixture:'synthetic 3.0.1; not a published version; no real downloads or application updates',owner_pid:process.pid});
   console.log(JSON.stringify({preview:true,url,output:folder,owner_pid:process.pid}));success=true;
   await new Promise(resolve=>context.once('close',resolve));
 }else{
   for(const theme of ['light','dark'])for(const [width,height,scale,font] of [[1440,900,100,14],[1280,800,150,18],[560,760,100,18]]){
     await page.setViewportSize({width,height});await page.evaluate(async({theme,scale,font})=>{const {setPageScale}=await import('/src/state/pageZoom.ts');setPageScale(scale);document.documentElement.dataset.theme=theme;document.documentElement.style.fontSize=font+'px';},{theme,scale,font});
     await page.evaluate(()=>document.fonts.ready);
     const layout=await dialog.evaluate(el=>{const notes=el.querySelector('.release-notes'),action=el.querySelector('.update-footer .primary'),box=el.getBoundingClientRect(),button=action.getBoundingClientRect();return{overflow:el.scrollWidth>el.clientWidth+1,notesScroll:notes.scrollHeight>notes.clientHeight,buttonVisible:button.top>=0&&button.bottom<=innerHeight,dialogVisible:box.top>=0&&box.bottom<=innerHeight,settingsClosed:!el.querySelector('.update-options').open};});
     assert.equal(layout.overflow,false);assert.equal(layout.buttonVisible,true);assert.equal(layout.dialogVisible,true);assert.equal(layout.settingsClosed,true);
     await dialog.screenshot({path:path.join(folder,`update-${theme}-${width}-${scale}.png`)});checks.push({name:'theme, large font, page scale and narrow layout',theme,width,height,scale,font,layout});
   }
   await page.setViewportSize({width:1440,height:900});await page.evaluate(async()=>{const {setPageScale}=await import('/src/state/pageZoom.ts');setPageScale(100);document.documentElement.style.fontSize='14px';});
   await dialog.locator('summary').click();assert.ok((await dialog.locator('.sources').innerText()).includes('HTTP 404'));assert.ok((await dialog.locator('.source-results').innerText()).includes('系统地区线索'));checks.push({name:'two-source statuses and region information remain available in expandable settings'});await dialog.locator('summary').click();
   await page.evaluate(()=>{window.fixture.baseNotes=window.fixture.release.notes;window.fixture.release.notes='## 更多格式\n\n- **重点**、*斜体*、~~旧说明~~与 `版本号`。\n  - 次级列表。\n\n1. 保存工作。\n2. 确认更新。\n\n> 操作提示可以单独引用。\n\n[官方网站](https://www.phonetictoolbox.com/)\n\n```text\n更新说明的代码块\n```\n\n| 功能 | 变化 |\n| --- | --- |\n| 更新窗口 | 分条展示 |\n\n---';});
   await dialog.locator('summary').click();await dialog.getByRole('button',{name:'检查更新',exact:true}).click();await dialog.locator('.release-notes h2').getByText('更多格式',{exact:true}).waitFor();
   for(const selector of ['strong','em','s','code','ol','blockquote','pre','table','hr'])assert.ok(await dialog.locator('.release-notes '+selector).count(),selector);
   assert.equal(await dialog.locator('.release-notes ul').count(),2);const link=dialog.locator('.release-notes a');assert.equal(await link.getAttribute('target'),'_blank');assert.equal(await link.getAttribute('rel'),'noopener noreferrer');
   await dialog.screenshot({path:path.join(folder,'update-markdown.png')});checks.push({name:'real heading, emphasis, strike-through, inline and fenced code, nested and numbered lists, quote, external link, table and rule render'});
   await page.evaluate(()=>window.fixture.release.notes=window.fixture.baseNotes);
   await page.evaluate(()=>{window.fixture.baseNotes=window.fixture.release.notes;window.fixture.release.notes+='\n- **文字检查：** <img src=x onerror=alert(1)> [链接](javascript:alert(1))';});
   await dialog.locator('summary').click();await dialog.getByRole('button',{name:'检查更新',exact:true}).click();await dialog.locator('.release-notes').getByText(/文字检查/).waitFor();assert.equal(await dialog.locator('.release-notes img,.release-notes a,.release-notes script').count(),0);assert.ok((await dialog.locator('.release-notes').innerText()).includes('<img src=x'));checks.push({name:'release HTML and links remain literal text'});
   await page.evaluate(()=>{window.fixture.release.notes=window.fixture.baseNotes;window.fixture.mode='download-slow';});
   await dialog.getByRole('button',{name:'下载更新',exact:true}).click();const confirm=page.getByRole('dialog',{name:'下载新版本？',exact:true});await confirm.waitFor();assert.equal(await page.evaluate(()=>window.fixture.requests.filter(r=>r.operation==='download').length),0);await confirm.getByRole('button',{name:'暂不下载',exact:true}).click();assert.equal(await page.evaluate(()=>window.fixture.requests.filter(r=>r.operation==='download').length),0);
   await dialog.getByRole('button',{name:'下载更新',exact:true}).click();await page.getByRole('button',{name:'确认下载',exact:true}).click();await dialog.getByRole('button',{name:'取消',exact:true}).click();await dialog.getByRole('alert').waitFor();assert.equal(await dialog.getByRole('button',{name:'退出并更新',exact:true}).count(),0);checks.push({name:'cancelled confirmation never downloads; cancelled download never enables apply'});
   await page.evaluate(()=>window.fixture.mode='checksum-error');await dialog.getByRole('button',{name:'下载更新',exact:true}).click();await page.getByRole('button',{name:'确认下载',exact:true}).click();await dialog.getByRole('alert').getByText('更新包校验失败，请重新下载。',{exact:true}).waitFor();assert.equal(await dialog.getByRole('button',{name:'退出并更新',exact:true}).count(),0);assert.equal(await dialog.locator('.download-progress').count(),0);checks.push({name:'native checksum error clears progress and keeps exit update unavailable'});
   await page.evaluate(()=>window.fixture.mode='available');await dialog.getByRole('button',{name:'下载更新',exact:true}).click();await page.getByRole('button',{name:'确认下载',exact:true}).click();await dialog.getByText('正在从国内服务器下载更新 100%…',{exact:true}).waitFor();assert.equal(await dialog.getByRole('button',{name:'退出并更新',exact:true}).count(),0);await dialog.getByRole('heading',{name:'更新包已下载并通过校验',exact:true}).waitFor();await dialog.screenshot({path:path.join(folder,'update-verified.png')});
   assert.deepEqual(await page.evaluate(()=>window.fixture.requests.filter(r=>r.operation==='download').at(-1).args),{releaseId:'owned-release',packageKind:'portable',confirmed:true});checks.push({name:'100 percent received is insufficient; verified response enables exit update using native opaque identifier'});
   await dialog.getByRole('button',{name:'退出并更新',exact:true}).click();assert.equal(await page.evaluate(()=>window.fixture.requests.filter(r=>r.operation==='apply').length),0);await page.getByRole('button',{name:'暂不更新',exact:true}).click();assert.equal(await page.evaluate(()=>window.fixture.requests.filter(r=>r.operation==='apply').length),0);await dialog.getByRole('button',{name:'退出并更新',exact:true}).click();await page.getByRole('button',{name:'确认退出并更新',exact:true}).click();assert.deepEqual(await page.evaluate(()=>window.fixture.requests.find(r=>r.operation==='apply').args),{downloadId:'owned-download',confirmed:true});checks.push({name:'exit update keeps its separate confirmation'});
   await dialog.locator('summary').click();await dialog.getByLabel('下载来源').selectOption('github');await page.waitForFunction(()=>window.fixture.preferences.source==='github');assert.equal(await page.getByRole('dialog',{name:'检查更新',exact:true}).count(),1);await page.getByLabel('启动时检查更新').uncheck();await page.waitForFunction(()=>window.fixture.preferences.autoCheck===false);checks.push({name:'preferences remain editable and clear stale candidate and outer title'});
   await page.evaluate(()=>window.fixture.mode='incomplete');await page.getByRole('button',{name:'检查更新',exact:true}).last().click();await page.getByRole('heading',{name:'检查未完整，请查看各来源状态'}).waitFor();assert.equal(await page.getByRole('heading',{name:'当前已是最新版本'}).count(),0);assert.equal(await page.locator('.source-error').count(),2);checks.push({name:'two-source failure remains unknown, with both errors'});
   await page.evaluate(()=>window.fixture.mode='slow');await page.getByRole('dialog').getByRole('button',{name:'检查更新',exact:true}).click();await page.getByRole('button',{name:'取消',exact:true}).click();await page.getByRole('alert').waitFor();assert.ok((await page.getByRole('alert').innerText()).includes('取消'));await page.waitForTimeout(300);checks.push({name:'cancelled query ignores late result'});
   await page.evaluate(()=>window.fixture.offline());await page.getByText('在线更新在桌面版提供。网页工具可继续离线使用。',{exact:true}).waitFor();await page.getByRole('dialog').getByRole('button',{name:'关闭对话框',exact:true}).click();
   await page.locator('#tool-sidebar').getByRole('button',{name:'关于',exact:true}).click();const about=page.getByRole('dialog',{name:'关于 PhoneticToolbox',exact:true});await about.waitFor();assert.equal(await about.locator('.about-actions>.primary').count(),4);
   for(const theme of ['light','dark']){
     await page.evaluate(theme=>document.documentElement.dataset.theme=theme,theme);
     await page.waitForFunction(()=>{const styles=[...document.querySelector('.about-actions').children].map(el=>getComputedStyle(el));return styles.every(css=>css.backgroundColor===styles[0].backgroundColor&&css.color===styles[0].color);},undefined,{timeout:5000});
     const colors=await about.locator('.about-actions').evaluate(el=>[...el.children].map(button=>{const css=getComputedStyle(button);return{background:css.backgroundColor,color:css.color};}));assert.ok(colors.every(c=>c.background===colors[0].background&&c.color===colors[0].color));await about.screenshot({path:path.join(folder,`about-${theme}.png`)});checks.push({name:'all four actual about actions share highlighted background and text',theme,colors});
   }
   assert.deepEqual(errors,[]);success=true;
 }
}catch(error){await page?.screenshot({path:path.join(folder,'failure.png')}).catch(()=>{});await write('failure.json',{error:error.stack,checks,errors});throw error;}
finally{
 await context?.close();await server?.close();await write('report.json',{success,checks,errors,notesIdentity,fixture:'native updater protocol stand-in; no actual release download, checksum calculation or installer executed'});
 const files=[],removedByBrowser=[];
 async function visit(directory){
   for(const entry of await fs.readdir(directory,{withFileTypes:true})){
     const file=path.join(directory,entry.name);
     try{
       const stat=await fs.lstat(file);if(stat.isSymbolicLink())throw Error(`Linked scratch: ${file}`);
       if(stat.isDirectory())await visit(file);
       else files.push({path:path.relative(scratch,file),bytes:stat.size,sha256:createHash('sha256').update(await fs.readFile(file)).digest('hex')});
     }catch(error){
       // Chrome may remove its WAL file while its last window finishes closing.
       if(error.code!=='ENOENT')throw error;
       removedByBrowser.push(path.relative(scratch,file));
     }
   }
 }
 let cleanupStatus;
 try{
   await visit(scratch);await write('scratch-manifest.json',{schema:'ptb-test-scratch-files/1',files,removedByBrowser});await write('run.json',{...marker,state:'ready',test_outcome:success?'completed':'failed'});
   const cleanup=spawnSync('powershell.exe',['-NoProfile','-ExecutionPolicy','Bypass','-File',path.join(repo,'scripts/cleanup_test_scratch.ps1'),'-RunDirectory',folder],{windowsHide:true,encoding:'utf8'});cleanupStatus=cleanup.status;
   if(cleanup.status!==0){console.error(cleanup.stderr||cleanup.stdout);process.exitCode=1;}
 }catch(error){await write('cleanup.json',{status:'retained',path:scratch,reason:String(error)});process.exitCode=1;cleanupStatus=1;}
 console.log(JSON.stringify({passed:checks.length,success,cleanup:cleanupStatus,report:path.join(folder,'report.json')}));
}
