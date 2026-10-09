const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'..'),out=path.join(root,'output/validation/p19-r15');
(async()=>{
 await fs.mkdir(out,{recursive:true});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{noDiscovery:true,include:['vue']}});
 await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
 const page=await browser.newPage({viewport:{width:1440,height:1000}}),report={success:false,groups:[],modules:[],errors:[]};
 page.on('pageerror',e=>report.errors.push(e.message));
 const sources=JSON.parse(await fs.readFile(path.join(root,'frontend/src/generated/sources.json'),'utf8'));
 const titles=['参数估计','参数显示','EGG 信号分析','LPC 谱图','唇形提取','声学参数合成','发声类型合成','变速变调','语谱图转音频','生理参数合成','MFA 自动标注','TextGrid标注','汉字转国际音标','音系归纳','感知实验','录音','国际音标表Plus'];
 const close=()=>page.locator('dialog[open]').getByRole('button',{name:'关闭对话框',exact:true}).click();
 const openGlobal=async()=>{await page.locator('.utility-grid').getByRole('button',{name:'关于',exact:true}).click();await page.locator('dialog[open]').getByRole('button',{name:'开源与学术致谢',exact:true}).click();};
 const group=async id=>{await page.getByRole('tab',{name:id==='academic'?/^语音学与语言学学术来源/:/^软件与代码来源/}).click();};
 try{
  await page.goto(server.resolvedUrls.local[0]);await page.locator('.home-page').waitFor();await page.evaluate(()=>document.fonts.ready);
  await page.evaluate(()=>{window.qaCopied='';Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:async t=>{window.qaCopied=t;}}});});
  for(const mode of ['light','dark']){
   await page.locator('.utility-grid').getByRole('button',{name:'设置',exact:true}).click();await page.getByRole('button',{name:mode==='light'?'浅色':'深色',exact:true}).click();
   assert.equal(await page.locator('.settings-page-heading').count(),0);
   const settings=page.locator('.settings-page');assert(!(await settings.innerText()).includes('让工作台更适合自己的阅读习惯'));
   await settings.screenshot({path:path.join(out,'settings-'+mode+'.png')});
   await openGlobal();
   const scope=page.getByRole('combobox',{name:'按模块查看致谢'}),search=page.getByRole('textbox',{name:'搜索来源'}),rows=page.locator('dialog[open] .reference-row');
   assert.equal(await scope.locator('option').count(),18);
   for(const [i,title]of titles.entries()){
    const id='M'+String(i+1).padStart(2,'0');await scope.selectOption(id);
    for(const g of ['academic','software']){await group(g);assert.equal(await rows.count(),sources.filter(s=>s.modules.includes(id)&&s.acknowledgement_group===g).length,title+' '+g);}
   }
   await scope.selectOption('M10');await group('academic');await search.fill('Oliveira');assert.equal(await rows.count(),1);assert.equal(await rows.getByRole('link',{name:'pdf',exact:true}).getAttribute('href'),'https://www.isca-archive.org/interspeech_2012/oliveira12b_interspeech.pdf');
   await rows.getByRole('button',{name:'复制引用'}).click();assert.equal(await page.evaluate(()=>window.qaCopied),sources.find(s=>s.id==='REF-M10-OLIVEIRA-2012-MRI').citation);
   assert.equal(await rows.locator('small').count(),2);await search.fill('Jordan');assert.equal(await rows.count(),1);assert((await rows.innerText()).includes('270–274'));
   await scope.selectOption('M07');await search.fill('载瓦');await group('software');assert.equal(await rows.count(),1);assert((await rows.innerText()).includes('2026-09-10'));assert((await rows.innerText()).includes('北京大学语言学实验室'));await rows.locator('summary').click();assert((await rows.innerText()).includes('MATLAB'));
   await rows.screenshot({path:path.join(out,'global-author-'+mode+'.png')});await search.fill('');await scope.selectOption('');
   for(const g of ['academic','software']){await group(g);assert.equal(await rows.count(),sources.filter(s=>s.acknowledgement_group===g).length);}
   report.groups.push({mode,total:sources.length,modules:17,missingMRI:0,permissionAndAdaptation:true});await close();
  }
  for(const [i,title]of titles.entries()){
   await page.locator('.nav-item').and(page.getByRole('button',{name:title,exact:true})).click();
   const method=page.locator('main').getByRole('button',{name:'方法与引用',exact:true}).filter({visible:true}).first();await method.waitFor();assert.equal(await method.locator('svg').count(),1,title);
   const help=page.locator('main').getByRole('button',{name:'帮助',exact:true}).filter({visible:true});
   if(await help.count()){
    await help.first().click();const dialog=page.locator('dialog[open]');if(await dialog.count())await close();else await help.first().click();
   }
   await method.click();await page.locator('dialog[open] .reference-groups').waitFor();
   assert.equal(await page.getByRole('combobox',{name:'按模块查看致谢'}).count(),0,title);
   let count=0;for(const g of ['academic','software']){await group(g);count+=await page.locator('dialog[open] .reference-row').count();}
   const id='M'+String(i+1).padStart(2,'0');assert.equal(count,sources.filter(s=>s.modules.includes(id)).length,title);
   report.modules.push({id,title,methodIcon:true,referenceCount:count,helpCount:await help.count()});await close();
  }
  for(const viewport of [{width:900,height:700},{width:1440,height:1000}]){
   await page.setViewportSize(viewport);await page.locator('.utility-grid').getByRole('button',{name:'设置',exact:true}).click();
   await page.locator('.settings-page').screenshot({path:path.join(out,`settings-${viewport.width}.png`)});
   const geometry=await page.locator('.settings-columns').evaluate(e=>({left:e.getBoundingClientRect().left,right:e.getBoundingClientRect().right,viewport:innerWidth}));assert(geometry.right<=geometry.viewport+1,JSON.stringify(geometry));
  }
  assert.deepEqual(report.errors,[]);report.success=true;
 }catch(e){report.error=String(e);await page.screenshot({path:path.join(out,'ui-failure.png')});throw e;}
 finally{await fs.writeFile(path.join(out,'ui-report.json'),JSON.stringify(report,null,2));await browser.close();await server.close();}
 console.log(JSON.stringify(report));
})().catch(e=>{console.error(e);process.exitCode=1;});
