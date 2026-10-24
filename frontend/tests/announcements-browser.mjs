// Actual AppShell + ModalDialog, owned browser/profile, controlled public feed.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {randomUUID,createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {createServer} from 'vite';
const frontend=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),repo=path.dirname(frontend);
const out=path.join(repo,'output/validation/announcements','chrome-'+randomUUID()),scratch=path.join(out,'scratch');
await fs.mkdir(scratch,{recursive:true});
const write=(name,value)=>fs.writeFile(path.join(out,name),JSON.stringify(value,null,2)+'\n');
const marker={schema:'ptb-test-run/1',owner_pid:process.pid,state:'active',started_at:new Date().toISOString(),recipe:'Actual AppShell/ModalDialog, controlled HTTPS feed via owned Chrome routes; synthetic capture lease, no physical recording/experiment or user projects'};
await write('run.json',marker);
await fs.writeFile(path.join(scratch,'index.html'),`<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><title>公告自动验证</title><div id="app"></div><script type="module" src="/@fs/${path.join(scratch,'main.ts').split(path.sep).join('/')}"></script></html>`);
await fs.writeFile(path.join(scratch,'main.ts'),`
import {createApp} from 'vue';import AppShell from '/src/app/AppShell.vue';import '/src/design/tokens.css';
import {initializePlatform} from '/src/platform/desktop.ts';import {initializeIcons} from '/src/state/app-icon.ts';import {initializeBackground} from '/src/state/background.ts';
import {claimCapture,releaseCapture} from '/src/platform/capture-lease.ts';import {previewFiles} from '/src/platform/research.ts';
await initializePlatform();await Promise.all([initializeIcons(),initializeBackground()]);
if(location.search.includes('fresh')){for(const key of Object.keys(localStorage))if(key.startsWith('ptb.v3.announcements.'))localStorage.removeItem(key);}
if(location.search.includes('capture'))claimCapture('M16');
let app=createApp(AppShell),project=0;app.mount('#app');
window.owned={stopCapture:()=>releaseCapture('M16'),switchProject:()=>{app.unmount();app=createApp(AppShell,{research:{key:'owned:project'+(++project)+':M01',label:'独立测试工程',files:previewFiles()}});app.mount('#app');},ready:true};
`);
const sample={id:'ipa-ime-0-1-1-launched',title:'国际音标输入法上线啦',body:'现在可以在官网下载国际音标输入法，支持 Windows、macOS、iOS 和 Android。\n打开下载页查看各平台的安装方式与测试版说明。',publishedAt:'2026-10-24T00:00:00Z',action:{label:'去官网下载',url:'https://www.phonetictoolbox.com/#ipa-ime-download'}};
const url='https://www.phonetictoolbox.com/announcements/v1.json';
let server,context,page,mode='empty',items=[],held,success=false;const checks=[],errors=[],requests=[];
const check=(name,extra={})=>checks.push({name,...extra});
try{
 server=await createServer({root:frontend,cacheDir:path.join(scratch,'vite-cache'),optimizeDeps:{noDiscovery:true,include:['vue']},server:{host:'127.0.0.1',port:0,watch:null,fs:{allow:[repo]}},logLevel:'error'});await server.listen();
 const runtime=process.env.PTB_PLAYWRIGHT??path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs');
 const {chromium}=await import(pathToFileURL(runtime));
 context=await chromium.launchPersistentContext(path.join(scratch,'chrome-profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1100,height:800},args:['--no-first-run','--mute-audio']});
 const routePublic=async route=>{
  const request=route.request();if(request.url()!==url){if(request.isNavigationRequest()&&request.url()==='https://www.phonetictoolbox.com/')await route.fulfill({contentType:'text/html',body:'<!doctype html><title>Owned official link target</title><p>Controlled navigation target</p>'});else await route.abort();return;}
  requests.push({url:request.url(),method:request.method(),body:request.postData(),headers:await request.allHeaders(),mode});
  if(mode==='offline'){await route.abort('internetdisconnected');return;}
  if(mode==='hold')await new Promise(resolve=>held=resolve);
  if(mode==='timeout')await new Promise(resolve=>setTimeout(resolve,3500));
  try{await route.fulfill({status:200,contentType:'application/json',body:mode==='bad'?JSON.stringify({schema:'ptb-announcements/1',items:[{...sample,action:{label:'危险',url:'javascript:alert(1)'}}]}):mode==='oversize'?' '.repeat(65537):JSON.stringify({schema:'ptb-announcements/1',items:mode==='empty'?[]:items})});}catch{/* The 3-second client deadline may already have cancelled its owned request. */}
 };await context.route('https://**/*',routePublic);
 await context.addInitScript(()=>{
  const Base=window.MutationObserver,active=new Set();window.announcementObservers=()=>active.size;
  window.MutationObserver=class extends Base{observe(target,options){if(target===document.body&&options?.attributeFilter?.includes('aria-modal'))active.add(this);return super.observe(target,options);}disconnect(){active.delete(this);return super.disconnect();}};
  if(location.search.includes('storage-denied')){const set=Storage.prototype.setItem;Storage.prototype.setItem=function(key,value){if(key.startsWith('ptb.v3.announcements.'))throw new DOMException('Owned readonly fixture','QuotaExceededError');return set.call(this,key,value);};}
 });
 page=await context.newPage();page.on('pageerror',error=>errors.push(String(error)));
 const base=()=>`http://127.0.0.1:${server.httpServer.address().port}/@fs/${path.join(scratch,'index.html').split(path.sep).join('/')}`;
 const navigate=async(query='',hash='')=>{await page.goto(base()+query+hash);await page.waitForFunction(()=>window.owned?.ready);await page.locator('.announcement-entry').waitFor();};
 const dialog=()=>page.getByRole('dialog',{name:'公告',exact:true});
 const noAuto=async(ms=700)=>{await page.waitForTimeout(ms);assert.equal(await dialog().count(),0);};
 await navigate('?fresh');await noAuto();assert.equal(await page.evaluate(()=>window.announcementObservers()),0);await page.locator('.announcement-entry').click();await dialog().getByText('暂时没有公告。',{exact:true}).waitFor();await page.keyboard.press('Escape');await page.waitForFunction(()=>document.activeElement?.classList.contains('announcement-entry'));check('empty authoritative feed remains silent with zero announcement observers; discoverable center and Escape return focus');
 mode='valid';items=[sample];await navigate('?fresh');await dialog().getByRole('heading',{name:sample.title}).waitFor();
 assert.equal(await dialog().locator('.announcement-body').innerText(),sample.body);assert.equal(await dialog().locator('a').getAttribute('href'),sample.action.url);
 const themeColors=[];for(const theme of ['light','dark']){await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);await page.waitForTimeout(250);themeColors.push({theme,...await dialog().locator('.dialog-actions button').evaluate(e=>({background:getComputedStyle(e).backgroundColor,color:getComputedStyle(e).color}))});await page.screenshot({path:path.join(out,'automatic-'+theme+'.png')});}
 await page.setViewportSize({width:420,height:800});await page.screenshot({path:path.join(out,'automatic-narrow.png')});const rect=await dialog().boundingBox();assert(rect.x>=0&&rect.x+rect.width<=421);await page.keyboard.press('Tab');assert(await page.evaluate(()=>!!document.activeElement?.closest('dialog[open]')));await page.setViewportSize({width:1100,height:800});
 await dialog().getByRole('button',{name:'知道了',exact:true}).click();await noAuto();assert.equal(await page.evaluate(()=>window.announcementObservers()),0);check('automatic plain-text notice, light/dark/narrow layout, focus remains in modal; acknowledged idle has zero observers',{themeColors});
 const callsBefore=requests.length;await page.evaluate(()=>window.owned.switchProject());await noAuto();await page.evaluate(()=>window.owned.switchProject());await noAuto();assert.equal(requests.length,callsBefore);check('project remounts preserve device read state and reuse one process request');
 items=[{...sample,body:'相同 ID 的更新内容 <script>window.bad=true</script>'}];await navigate();await noAuto();await page.locator('.announcement-entry').click();await dialog().getByText(items[0].body,{exact:true}).waitFor();assert.equal(await page.evaluate(()=>window.bad),undefined);await page.keyboard.press('Escape');check('same ID content updates remain readable without re-prompt; HTML is inert text');
 items=[{...sample,id:'new-notice',title:'新的测试公告'}];await navigate();await dialog().getByRole('heading',{name:'新的测试公告'}).waitFor();const popupPromise=context.waitForEvent('page');await dialog().getByRole('link',{name:'去官网下载'}).click();const popup=await popupPromise;await popup.waitForLoadState().catch(()=>{});assert.equal(popup.url(),sample.action.url);await popup.close();await noAuto();check('new ID prompts; explicit HTTPS action marks only that announcement read');
 mode='hold';items=[{...sample,id:'avoid-dialog'}];await navigate('?fresh');await page.locator('#tool-sidebar button[title="关于"]').click();await page.getByRole('dialog',{name:'关于 PhoneticToolbox',exact:true}).waitFor();held();await page.waitForFunction(()=>window.announcementObservers()===1);assert.equal(await dialog().count(),0);await page.keyboard.press('Escape');await dialog().waitFor();assert.equal(await page.evaluate(()=>window.announcementObservers()),0);await page.keyboard.press('Escape');check('only waiting for another application modal temporarily enables the announcement observer');
 mode='valid';items=[{...sample,id:'capture-deferred'}];await navigate('?fresh&capture');await noAuto();await page.evaluate(()=>window.owned.stopCapture());await dialog().waitFor();await page.keyboard.press('Escape');check('active real capture lease suppresses auto notice until released (synthetic recording, no device)');
 items=[{...sample,id:'experiment-deferred'}];await navigate('?fresh','#M15');await page.locator('#tab-M15').waitFor();await noAuto();assert.equal(await page.evaluate(()=>window.announcementObservers()),0);await page.locator('#tab-home').click();await dialog().waitFor();await page.keyboard.press('Escape');check('perception/tool page has zero announcement observers and is never interrupted; deferred notice appears on home');
 items=[{...sample,id:'persisted-page'}];await navigate('?fresh');await dialog().waitFor();const beforePersisted=requests.length;await page.evaluate(()=>window.dispatchEvent(new PageTransitionEvent('pagehide',{persisted:true})));await noAuto(100);await page.evaluate(()=>window.dispatchEvent(new PageTransitionEvent('pageshow',{persisted:true})));await dialog().waitFor();assert.equal(requests.length,beforePersisted);await page.keyboard.press('Escape');check('actual persisted pagehide/pageshow events release the modal and restore the live singleton without another request');
 items=[{...sample,id:'readonly-store'}];await navigate('?fresh&storage-denied');await noAuto();assert.equal(await page.evaluate(()=>window.announcementObservers()),0);await page.locator('.announcement-entry').click();await dialog().getByRole('heading',{name:sample.title}).waitFor();await page.keyboard.press('Escape');check('failed device persistence suppresses automatic prompts while manual center remains readable');
 mode='valid';items=[{...sample,id:'cross-tab'}];await navigate('?fresh');await dialog().waitFor();const second=await context.newPage();await second.goto(base());await second.waitForFunction(()=>window.owned?.ready);await second.waitForTimeout(700);assert.equal(await second.getByRole('dialog',{name:'公告',exact:true}).count(),0);await page.keyboard.press('Escape');await second.waitForTimeout(700);assert.equal(await second.getByRole('dialog',{name:'公告',exact:true}).count(),0);await second.close();check('shared Web Lock/device lease and persisted IDs prevent two-tab simultaneous prompts');
 for(const failure of ['offline','timeout','bad','oversize']){mode=failure;await navigate('?fresh');await noAuto(failure==='timeout'?3800:700);await page.locator('.announcement-entry').click();await dialog().getByText('暂时没有公告。',{exact:true}).waitFor();await page.keyboard.press('Escape');check('first startup '+failure+' stays silent and the workbench remains usable');}
 mode='valid';items=[{...sample,id:'cached-offline'}];await navigate('?fresh');await dialog().waitFor();await page.keyboard.press('Escape');mode='offline';await navigate();await noAuto();await page.locator('.announcement-entry').click();await dialog().getByRole('heading',{name:sample.title}).waitFor();await page.keyboard.press('Escape');check('successful feed/read IDs persist across reload and remain manually readable offline');
 mode='empty';await navigate();await noAuto();mode='offline';await navigate();await noAuto();await page.locator('.announcement-entry').click();await dialog().getByText('暂时没有公告。',{exact:true}).waitFor();await page.keyboard.press('Escape');check('successful empty feed withdraws cached content and stays withdrawn offline');
 mode='valid';items=[{...sample,id:'browser-reopened'}];await navigate('?fresh');await dialog().waitFor();await page.keyboard.press('Escape');await context.close();mode='offline';context=await chromium.launchPersistentContext(path.join(scratch,'chrome-profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1100,height:800},args:['--no-first-run','--mute-audio']});await context.route('https://**/*',routePublic);page=await context.newPage();page.on('pageerror',error=>errors.push(String(error)));await navigate();await noAuto();await page.locator('.announcement-entry').click();await dialog().getByRole('heading',{name:sample.title}).waitFor();await page.keyboard.press('Escape');check('closing and reopening the actual persistent Chrome process preserves cache and device read IDs');
 assert(requests.every(r=>r.method==='GET'&&r.body===null&&!r.headers.cookie&&!r.headers.referer));assert.deepEqual(errors,[]);check('fixed public GET contains no cookies, referrer or project/account request body');success=true;
}catch(error){errors.push(String(error.stack??error));process.exitCode=1;try{await page?.screenshot({path:path.join(out,'failure.png')});}catch{}}
finally{
 await context?.close();await server?.close();
 const sourceFiles=['src/platform/announcements.ts','src/components/AnnouncementsPanel.vue','src/app/AppShell.vue'];const identities=[];for(const name of sourceFiles){const bytes=await fs.readFile(path.join(frontend,name));identities.push({path:'frontend/'+name,sha256:createHash('sha256').update(bytes).digest('hex')});}
 await write('report.json',{success,checks,errors,requests,sourceFiles:identities,scope:marker.recipe});
 const files=[],removedByBrowser=[];async function visit(folder){for(const entry of await fs.readdir(folder,{withFileTypes:true})){const file=path.join(folder,entry.name);try{const stat=await fs.lstat(file);assert(!stat.isSymbolicLink());if(stat.isDirectory())await visit(file);else{const bytes=await fs.readFile(file);files.push({path:path.relative(scratch,file),bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex')});}}catch(error){if(error.code!=='ENOENT')throw error;removedByBrowser.push(path.relative(scratch,file));}}}
 try{await visit(scratch);await write('scratch-manifest.json',{schema:'ptb-test-scratch-files/1',files,removedByBrowser});await write('run.json',{...marker,state:'ready',test_outcome:success?'completed':'failed'});const cleanup=spawnSync('powershell.exe',['-NoProfile','-ExecutionPolicy','Bypass','-File',path.join(repo,'scripts/cleanup_test_scratch.ps1'),'-RunDirectory',out],{windowsHide:true,encoding:'utf8'});if(cleanup.status!==0){process.exitCode=1;console.error(cleanup.stderr||cleanup.stdout);}}
 catch(error){await write('cleanup.json',{status:'retained',path:scratch,reason:String(error)});process.exitCode=1;}
 console.log(JSON.stringify({success,checks:checks.length,report:path.join(out,'report.json')}));
}
