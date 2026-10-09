const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'..'),out=path.join(root,'output/validation/p19-r17/browser');
(async()=>{
 await fs.mkdir(out,{recursive:true});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{noDiscovery:true,include:['vue']}});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
 const page=await browser.newPage({viewport:{width:1440,height:1000}}),report={success:false,navigation:[],modules:[],errors:[]};
 page.on('pageerror',e=>report.errors.push(e.message));
 const project=JSON.parse(await fs.readFile(path.join(root,'frontend/public/manual/project.json'),'utf8'));
 const titles=['参数估计','参数显示','EGG 信号分析','LPC 谱图','唇形提取','声学参数合成','发声类型合成','变速变调','语谱图转音频','生理参数合成','MFA 自动标注','TextGrid标注','汉字转国际音标','音系归纳','感知实验','录音','国际音标表Plus'];
 const open=title=>page.locator('.nav-item').and(page.getByRole('button',{name:title,exact:true})).click();
 const style=locator=>locator.evaluate(e=>{const s=getComputedStyle(e);return {outline:s.outlineStyle,width:s.outlineWidth,shadow:s.boxShadow,decoration:s.textDecorationLine,background:s.backgroundColor,color:s.color,focus:e.matches(':focus-visible')};});
 try{
  await page.goto(server.resolvedUrls.local[0]);await page.locator('.home-page').waitFor();await page.evaluate(()=>document.fonts.ready);
  assert.equal(await page.locator('.module-manual-entry').count(),0);
  for(const mode of ['light','dark'])for(const buttons of ['auto','all','plain']){
   await open('EGG 信号分析');
   await page.evaluate(({mode,buttons})=>{document.documentElement.dataset.theme=mode;document.documentElement.dataset.palette='codex';document.documentElement.dataset.buttonStyle=buttons;},{mode,buttons});
   const nav=page.locator('.nav-item.selected');await nav.focus();const navStyle=await style(nav);assert.equal(navStyle.outline,'none');assert(navStyle.decoration.includes('underline'));assert(navStyle.focus);
   const tab=page.getByRole('tab',{name:'EGG 信号分析',exact:true});await tab.focus();const tabStyle=await style(tab);assert.equal(tabStyle.outline,'none');assert.equal(tabStyle.shadow,'none');assert(tabStyle.decoration.includes('underline'));
   const close=page.getByRole('button',{name:'关闭 EGG 信号分析',exact:true});await close.focus();const closeStyle=await style(close);assert.equal(closeStyle.outline,'none');assert.equal(closeStyle.shadow,'none');
   // The selection still has its single original border and bottom line.
   const selected=await nav.evaluate(e=>{const s=getComputedStyle(e);return {left:s.borderLeftWidth,border:s.borderTopWidth,color:s.borderTopColor,bg:s.backgroundColor};});assert.equal(selected.left,'3px');assert.equal(selected.border,'1px');
   const wrap=await tab.locator('..').evaluate(e=>getComputedStyle(e).boxShadow);assert.notEqual(wrap,'none');
   await page.locator('main').getByRole('button',{name:'帮助',exact:true}).click();
   await page.locator('.manual-chapter-header h1').filter({hasText:project.chapters.find(c=>c.id==='m03').title}).waitFor();
   await page.getByRole('button',{name:'返回 EGG 信号分析',exact:true}).click();
   await open('EGG 信号分析');await page.screenshot({path:path.join(out,`navigation-${mode}-${buttons}.png`)});
   report.navigation.push({mode,buttons,navStyle,tabStyle,closeStyle,selected});
  }
  await page.evaluate(()=>{document.documentElement.dataset.theme='light';document.documentElement.dataset.buttonStyle='auto';});
  for(const [i,title]of titles.entries()){
   await open(title);
   const help=page.locator('main').getByRole('button',{name:'帮助',exact:true}).filter({visible:true});await help.waitFor();assert.equal(await help.count(),1,title);
   assert.equal(await page.locator('.topbar').getByRole('button',{name:/模块使用说明|帮助/}).count(),0);
   const method=page.locator('main').getByRole('button',{name:'方法与引用',exact:true}).filter({visible:true}).first();
   const hg=await help.boundingBox(),mg=await method.boundingBox();assert(hg&&mg&&Math.abs(hg.y-mg.y)<2,title+' help adjacent to references');
   await help.click();const id='m'+String(i+1).padStart(2,'0'),chapter=project.chapters.find(c=>c.id===id);assert(chapter);
   await page.locator('.manual-chapter-header h1').filter({hasText:chapter.title}).waitFor();
   await page.getByRole('button',{name:'返回 '+title,exact:true}).click();await help.waitFor();
   report.modules.push({id,title,chapter:chapter.title,helpCount:1,insidePage:true,returned:true});
  }
  await open('参数显示');const buttons=page.locator('.module-toolbar-actions').filter({visible:true});await buttons.screenshot({path:path.join(out,'module-help-placement.png')});
  assert.deepEqual(report.errors,[]);report.success=true;
 }catch(e){report.error=String(e.stack??e);await page.screenshot({path:path.join(out,'failure.png')});throw e;}
 finally{await fs.writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2));await browser.close();await server.close();}
 console.log(JSON.stringify({success:report.success,navigation:report.navigation.length,modules:report.modules.length,out}));
})().catch(e=>{console.error(e);process.exitCode=1;});
