const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..'),[out,scratch]=process.argv.slice(2);
(async()=>{
 assert(out&&scratch,'Use verify_source_citations.py');
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const sources=JSON.parse(await fs.readFile(path.join(root,'frontend/src/generated/sources.json'),'utf8'));
 const expected=sources.filter(s=>s.acknowledgement_group==='academic');
 const report={success:false,cases:[],errors:[],externalRequests:[]};
 const server=await createServer({root:path.join(root,'frontend'),cacheDir:path.join(scratch,'vite-cache'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{noDiscovery:true,include:['vue']}});
 let browser;
 try{
  await server.listen();browser=await chromium.launchPersistentContext(path.join(scratch,'chrome-profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1440,height:900}});
  await browser.addInitScript(()=>{window.qaCopied='';Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:async text=>{window.qaCopied=text;}}});});
  const page=await browser.newPage();page.on('pageerror',e=>report.errors.push(e.message));
  page.on('request',req=>{if(/^https?:/.test(req.url())&&!req.url().startsWith(server.resolvedUrls.local[0]))report.externalRequests.push(req.url());});
  await page.goto(server.resolvedUrls.local[0]);await page.locator('.home-page').waitFor();
  const rows=page.locator('dialog[open] .reference-row'),close=()=>page.locator('dialog[open]').getByRole('button',{name:'关闭对话框',exact:true}).click();
  for(const mode of ['light','dark']){
   await page.locator('.utility-grid').getByRole('button',{name:'设置',exact:true}).click();await page.getByRole('button',{name:mode==='light'?'浅色':'深色',exact:true}).click();
   for(const width of [1440,900]){
    await page.setViewportSize({width,height:width===1440?900:700});
    await page.locator('.utility-grid').getByRole('button',{name:'关于',exact:true}).click();await page.locator('dialog[open]').getByRole('button',{name:'开源与学术致谢',exact:true}).click();
    await page.getByRole('tab',{name:/^语音学与语言学学术来源/}).click();assert.equal(await rows.count(),48);
    for(const source of expected){
     const row=rows.filter({has:page.getByRole('heading',{name:source.title,exact:true})});assert.equal(await row.count(),1,source.id);
     assert.equal(await row.locator('.reference-citation').textContent(),source.citation,source.id);
     await row.getByRole('button',{name:'复制引用',exact:true}).click();assert.equal(await page.evaluate(()=>window.qaCopied),source.citation,source.id+' clipboard');
    }
    const geometry=await page.locator('dialog[open]').evaluate(dialog=>{const body=dialog.querySelector('.dialog-body'),rect=dialog.getBoundingClientRect();return{left:rect.left,right:rect.right,viewport:innerWidth,overflow:body.scrollWidth-body.clientWidth};});assert(geometry.left>=-1&&geometry.right<=geometry.viewport+1&&geometry.overflow<=1,JSON.stringify(geometry));
    await page.getByRole('combobox',{name:'按模块查看致谢'}).selectOption('M17');await page.getByRole('tab',{name:/^软件与代码来源/}).click();assert(!((await rows.allTextContents()).join(' ').includes('ipa.cin')));
    const app=sources.find(s=>s.id==='M17-IOS-IPA-PHONETICS-REFERENCE'),appRow=rows.filter({has:page.getByRole('heading',{name:app.title,exact:true})});assert.equal(await appRow.count(),1);
    await appRow.getByRole('button',{name:'复制引用',exact:true}).click();assert.equal(await page.evaluate(()=>window.qaCopied),app.citation);
    const interactive=sources.find(s=>s.id==='M17-INTERACTIVE-LAYOUT-REFERENCE'),interactiveRow=rows.filter({has:page.getByRole('heading',{name:interactive.title,exact:true})});assert.equal(await interactiveRow.locator('.reference-citation').evaluate(el=>getComputedStyle(el).whiteSpace),'pre-line');
    await page.getByRole('textbox',{name:'搜索来源'}).fill('ipa.cin');assert.equal(await rows.count(),0);await page.getByRole('textbox',{name:'搜索来源'}).fill('');await close();
    await page.locator('.nav-item').and(page.getByRole('button',{name:'国际音标表Plus',exact:true})).click();await page.locator('.ipa-plus-page').getByRole('button',{name:'方法与引用',exact:true}).click();await page.getByRole('tab',{name:/^语音学与语言学学术来源/}).click();assert.equal(await rows.count(),expected.filter(s=>s.modules.includes('M17')).length);
    await page.locator('dialog[open]').screenshot({path:path.join(out,'citations-'+mode+'-'+width+'.png')});await close();
    report.cases.push({mode,width,academicCopies:48,privateCinHidden:true,appCopy:true,multiCitationNewlines:true,moduleAndGlobalMatch:true,geometry});
   }
  }
  await page.setViewportSize({width:1440,height:900});
  await page.locator('#tool-sidebar').getByRole('button',{name:'使用说明',exact:true}).click();await page.locator('.manual-document').waitFor();
  await page.locator('.manual-toc button[data-manual-chapter="m17"]').click();await page.locator('.manual-document[data-manual-chapter="m17"]').waitFor();
  const manual=JSON.parse(await fs.readFile(path.join(root,'manual/chapters/m17.json'),'utf8'));
  const bibliography=[];let bibliographySection=false;
  const nodeText=node=>(node.text||'')+(node.content||[]).map(nodeText).join('');
  const collect=node=>{if(node.type==='citation')bibliography.push(node.attrs);for(const child of node.content||[])collect(child);};
  for(const block of manual.body.content){if(block.type==='heading')bibliographySection=nodeText(block)==='参考资料';if(bibliographySection)collect(block);}
  let complete=0;
  for(const entry of bibliography){const source=sources.find(s=>s.id===entry.referenceId);if(!source?.citation)continue;const reference=page.locator('.manual-document .manual-citation a').filter({hasText:source.citation});assert.equal(await reference.count(),1,entry.referenceId+' manual');assert.equal(await reference.textContent(),source.citation);complete++;}
  assert.equal(complete,11);await page.locator('.manual-toc button[data-manual-section="m17-sources"]').click();await page.locator('.manual-toc button[data-manual-subsection="m17-references"]').click();
  const manualGeometry=await page.locator('.manual-reading-scroll').evaluate(el=>({overflow:el.scrollWidth-el.clientWidth}));assert(manualGeometry.overflow<=1,JSON.stringify(manualGeometry));
  await page.locator('.manual-reader').screenshot({path:path.join(out,'manual-citations.png')});report.manual={completeReferences:complete,geometry:manualGeometry};
  assert.deepEqual(report.errors,[]);assert.deepEqual(report.externalRequests,[]);report.success=true;
 }finally{if(browser)await browser.close();await server.close();await fs.writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2)+'\n');}
 console.log(JSON.stringify(report,null,2));
})().catch(e=>{console.error(e);process.exitCode=1;});
