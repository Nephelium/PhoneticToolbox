const assert=require('node:assert/strict'),fs=require('node:fs/promises'),path=require('node:path');
module.exports=async({page,scope,click,idle,out,checks})=>{
 const picker=scope.getByRole('combobox',{name:'音频文件',exact:true}),extract=scope.getByRole('button',{name:'提取参数',exact:true});
 const creations=[];const listener=req=>{if(req.url().includes('/__m06_host')&&req.method()==='POST'){const b=req.postDataJSON()?.body;if(b?.op==='m06_create')creations.push(b.body.action);}};page.on('request',listener);
 await page.setViewportSize({width:1920,height:1000});
 assert.equal(await scope.getByLabel('合成方法',{exact:true}).count(),0);
 assert.deepEqual(await scope.getByLabel('F0 提取算法').locator('option').evaluateAll(e=>e.map(x=>x.value)),['praat_cc','praat_ac','reaper']);
 await picker.selectOption('');assert(await extract.isDisabled());
 await click('打开音频目录');
 const before=await scope.locator('.curve-editor path.curve').getAttribute('d'),duration=await scope.getByLabel('总时长',{exact:true}).inputValue();
 await picker.selectOption({label:'source.wav'});await scope.getByText('音频已加载，点击提取参数开始提取。',{exact:true}).waitFor();
 assert.equal(await page.getByRole('dialog').count(),0);assert(!await extract.isDisabled());assert.deepEqual(creations,[]);
 assert.equal(await scope.getByLabel('总时长',{exact:true}).inputValue(),duration);
 // Different display durations may change the SVG coordinates; compare exported parameter snapshots instead below.
 assert(before);await click('播放选区');await click('停止');
 await click('提取参数');await idle();assert((await scope.innerText()).includes('参数提取完成'));assert.deepEqual(creations,['extract']);
 const snapshot=async(name)=>{const download=page.waitForEvent('download');await click('导出参数');const d=await download;const file=path.join(out,name+'.csv');await d.saveAs(file);return fs.readFile(file,'utf8');};
 const original=await snapshot('r5-before-reload');await picker.selectOption('');assert(await extract.isDisabled());
 await picker.selectOption({label:'source.wav'});await scope.getByText('音频已加载，点击提取参数开始提取。',{exact:true}).waitFor();assert.equal(await snapshot('r5-after-reload'),original);assert.deepEqual(creations,['extract']);
 checks.push('R5 source picker is first left section; load/clear/reload never extracts or modifies parameters; only explicit extraction submits job');
 const fileId=await picker.inputValue();
 // Delay the read, clear the selection, and ensure the old preview cannot return.
 let delayed=false;
 await page.route('**/__m06_host',async route=>{const b=route.request().postDataJSON()?.body;if(b?.op==='read'&&b.id===fileId){delayed=true;await new Promise(r=>setTimeout(r,500));}await route.continue();});
 await picker.selectOption('');await picker.selectOption(fileId);assert(await extract.isDisabled());await picker.selectOption('');await page.waitForTimeout(800);assert(await extract.isDisabled());assert.equal(await picker.inputValue(),'');assert.equal(await scope.getByRole('button',{name:'源音频',exact:true}).count(),0);assert(delayed);await page.unroute('**/__m06_host');
 await page.route('**/__m06_host',async route=>{const b=route.request().postDataJSON()?.body;if(b?.op==='read')await route.fulfill({contentType:'application/json',body:JSON.stringify({ok:false,error:'受控音频读取失败'})});else await route.continue();});
 await picker.selectOption(fileId);await scope.getByText('受控音频读取失败',{exact:true}).waitFor();assert(await extract.isDisabled());assert.equal(await picker.inputValue(),'');await page.unroute('**/__m06_host');
 checks.push('R5 loading/error/late-read ownership: disabled extraction and no stale source after clearing or failure');
 await picker.selectOption(fileId);await scope.getByText('音频已加载，点击提取参数开始提取。',{exact:true}).waitFor();
 await click('元音规则');const dialog=page.getByRole('dialog');
 await page.context().grantPermissions(['clipboard-read','clipboard-write']);
 for(const symbol of ['i','ɯ','ɤ','ə']){await dialog.getByRole('button',{name:symbol,exact:true}).click();await dialog.getByText('已复制 '+symbol,{exact:true}).waitFor();assert.equal(await page.evaluate(()=>navigator.clipboard.readText()),symbol);}
 await page.evaluate(()=>{window.__m06Clipboard=navigator.clipboard.writeText.bind(navigator.clipboard);navigator.clipboard.writeText=()=>Promise.reject(new DOMException('denied','NotAllowedError'));});
 await dialog.getByRole('button',{name:'i',exact:true}).click();await dialog.getByText('复制失败，请选中音标后按 Ctrl+C。',{exact:true}).waitFor();assert(!await scope.innerText().then(t=>t.includes('Write permission denied')));
 await page.evaluate(()=>navigator.clipboard.writeText=window.__m06Clipboard);
 const rows=await dialog.locator('tbody tr').evaluateAll(es=>es.map(e=>e.getBoundingClientRect().height));assert(Math.max(...rows)<=36,JSON.stringify(rows));
 await page.screenshot({path:path.join(out,'r5-vowels-light.png'),fullPage:true});await dialog.getByRole('button',{name:'关闭对话框',exact:true}).click();
 checks.push('R5 compact vowels: four IPA clipboard roundtrips, denied browser clipboard error inside dialog, row height <=36px');
 const layouts=[];
 for(const [width,height] of [[1920,1000],[1440,900],[900,700]])for(const theme of ['light','dark']){
  await page.setViewportSize({width,height});await click('设置');await click(theme==='light'?'浅色':'深色');await page.locator('nav').getByRole('button',{name:'语音合成',exact:true}).click();await page.waitForTimeout(150);
  const m=await scope.evaluate(r=>{const b=e=>{const q=e.getBoundingClientRect();return {x:q.x,y:q.y,w:q.width,b:q.bottom}};return {first:r.querySelector('.workbench-left').firstElementChild.classList.contains('source-section'),source:b(r.querySelector('.source-section')),root:b(r),bar:b(r.querySelector('.m06-transport')),overflow:r.scrollWidth-r.clientWidth};});
  assert(m.first&&m.overflow<=1&&m.bar.b<=m.root.b+1,JSON.stringify(m));layouts.push({width,height,theme,...m});await page.screenshot({path:path.join(out,`r5-${width}-${theme}.png`),fullPage:true});
 }
 await fs.writeFile(path.join(out,'r5-layouts.json'),JSON.stringify(layouts,null,2));
 await page.setViewportSize({width:1920,height:1000});await click('元音规则');await page.screenshot({path:path.join(out,'r5-vowels-dark.png'),fullPage:true});await page.getByRole('dialog').getByRole('button',{name:'关闭对话框',exact:true}).click();
 await click('合成音频');await idle();assert((await scope.innerText()).includes('合成完成'));await click('导出音频');await scope.getByText('已导出合成结果',{exact:true}).waitFor();
 checks.push('R5 six real-theme layouts and actual Klatt synthesis / native result save');page.off('request',listener);
};
