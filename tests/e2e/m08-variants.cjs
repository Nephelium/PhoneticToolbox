// Real Chrome and production desktop adapter; only native-dialog choices are injected.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{spawn}=require('node:child_process'),{createInterface}=require('node:readline'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const worker=spawn(path.join(root,'.venv/m14/Scripts/python.exe'),['-B','-X','utf8','scripts/verify_m08_variants.py','--host'],{cwd:root,windowsHide:true});
 let resolveReady,rejectReady,counter=0;const pending=new Map(),ready=new Promise((r,j)=>{resolveReady=r;rejectReady=j});
 createInterface({input:worker.stdout}).on('line',line=>{try{const d=JSON.parse(line);if(d.ready)resolveReady(d);else{pending.get(d.id)?.(d);pending.delete(d.id)}}catch{console.log(line)}});worker.stderr.on('data',d=>process.stderr.write(d));worker.once('exit',code=>rejectReady(Error('Host exited '+code)));
 const {out,scratch}=await ready;console.log(out);const requests=[],report={success:false,checks:[],errors:[]};
 const rpc=(channel,body)=>new Promise(resolve=>{const id=++counter;pending.set(id,resolve);worker.stdin.write(JSON.stringify({id,channel,body})+'\n')});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),cacheDir:path.join(scratch,'vite'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{entries:['tests/m08-host.html']},plugins:[{name:'m08-owned-host',configureServer(s){s.middlewares.use('/__m08_host',(req,res)=>{let raw='';req.on('data',d=>raw+=d);req.on('end',async()=>{const message=JSON.parse(raw);requests.push(message);res.setHeader('Content-Type','application/json');res.end(JSON.stringify(await rpc(message.channel,message.body)))})})}}]});
 await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launchPersistentContext(path.join(scratch,'chrome'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1800,height:1100},args:['--mute-audio']}),page=await browser.newPage();
 page.on('pageerror',e=>report.errors.push(e.message));page.setDefaultTimeout(60000);
 const click=name=>page.getByRole('button',{name,exact:true}).click(),fill=(label,value)=>page.getByLabel(label,{exact:true}).fill(value),idle=()=>page.waitForFunction(()=>document.querySelector('.m08-page')?.getAttribute('aria-busy')==='false');
 const cancelModal=async()=>{const m=page.getByRole('dialog');if(await m.count())await m.getByRole('button',{name:'取消',exact:true}).click()};
 async function verifyManual(){
  await click('使用说明');await page.locator('.manual-reader').getByText('单文件和文件夹处理都支持多个语速与多个音高值',{exact:false}).first().waitFor();
  await page.locator('.manual-reader').getByText('选择横向 · Hz 或纵向 · 秒',{exact:false}).first().waitFor();await page.locator('.manual-reader').getByText('该选项每次打开默认不选',{exact:false}).first().waitFor();
  const image=page.locator('.manual-reader img[alt*="每份与总版本数"]');await image.scrollIntoViewIfNeeded();await image.evaluate(e=>e.decode());await page.waitForTimeout(350);await image.screenshot({path:path.join(out,'manual-image.png')});await page.screenshot({path:path.join(out,'manual-batch.png')});report.checks.push('actual manual reader opens revised M08 and decodes/paints the new 2560px Qt batch screenshot');
 }
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/m08-host.html');await page.locator('nav').getByRole('button',{name:'变速变调',exact:true}).click();
  if(process.argv.includes('--manual-only')){await verifyManual();assert.deepEqual(report.errors,[]);report.success=true;return;}
  await click('打开音频目录');
  const select=page.getByLabel('变速变调音频');await page.waitForFunction(()=>document.querySelector('[aria-label="变速变调音频"]')?.options.length===3);
  const opts=await select.locator('option').evaluateAll(es=>es.filter(o=>o.value).map(o=>({id:o.value,name:o.textContent})));await select.selectOption(opts[0].id);await page.locator('svg.m08-curve').waitFor();await idle();
  await fill('参考线 Hz','200');await click('添加参考线');
  await page.getByLabel('参考线方向').selectOption('time');await fill('参考线 秒','0.5');await click('添加参考线');
  const vertical=page.locator('svg.m08-curve .time-reference');assert.equal(await vertical.count(),1);
  let geometry=await vertical.evaluate(e=>{const s=e.ownerSVGElement,b=e.getBBox();return {x:b.x,w:s.viewBox.baseVal.width,h:b.height}});assert(Math.abs(geometry.x-(60+(geometry.w-80)*.5))<.01);assert(geometry.h>100);
  const curve=page.locator('svg.m08-curve'),rect=await curve.boundingBox();await page.mouse.move(rect.x+rect.width*.6,rect.y+rect.height*.5);await page.keyboard.down('Control');await page.mouse.wheel(0,-100);await page.keyboard.up('Control');await page.waitForTimeout(150);
  assert.notEqual(await vertical.getAttribute('d'),`M${geometry.x},20V${geometry.h+20}`);
  await fill('参考线 秒','-1');await click('添加参考线');await page.getByRole('alert').filter({hasText:'参考线时间不能小于 0 秒'}).waitFor();assert.equal(await vertical.count(),1);
  await fill('参考线 秒','0.5');await click('保存编辑草稿');
  await page.reload();await page.locator('nav').getByRole('button',{name:'变速变调',exact:true}).click();await click('打开音频目录');await select.selectOption(opts[0].id);await curve.waitFor();await idle();assert.equal(await vertical.count(),1);assert.equal(await curve.getByText('200 Hz',{exact:true}).count(),1);
  for(const theme of ['light','dark']){await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);await page.screenshot({path:path.join(out,`references-${theme}.png`)});}
  await click('清除参考线');assert.equal(await curve.locator('.reference').count(),0);
  report.checks.push('horizontal Hz and vertical seconds coexist; exact x coordinate, zoom, negative-time rejection, draft reload and clear; light/dark captures');
  await fill('语速倍率','0.8, 1.2');await fill('音高倍率','1, 1.2');await page.getByText('每份音频 4 个版本',{exact:true}).waitFor();
  await click('保存编辑草稿');await click('合成整段');await page.getByRole('dialog',{name:'批量保存',exact:true}).waitFor();assert.equal(await page.getByRole('dialog').locator('.file-list input[type=checkbox]').count(),4);
  let creates=requests.filter(r=>r.body.action==='create'&&r.body.body.config.action==='synthesize');assert.deepEqual(creates.map(r=>[r.body.body.config.speed,r.body.body.config.pitch_ratio,r.body.body.config.pitch_hz]),[[.8,1,0],[.8,1.2,0],[1.2,1,0],[1.2,1.2,0]]);
  const hz=creates[1].body.body.config.modified_f0.filter(x=>x>0);assert(Math.abs(hz[Math.floor(hz.length/2)]-180)<1);report.checks.push('single-file 2 speeds x 2 ratios creates four real edited-F0 outputs and prompts bulk save');await cancelModal();
  // Batch input values appear in the batch page; inactive ratio cannot combine with Hz.
  await click('批量变速变调');await fill('语速倍率','0.8, 1.2');await page.getByLabel('音高调整方式').selectOption('hz');assert.equal(await page.getByLabel('音高倍率',{exact:true}).count(),0);await fill('音高偏移 Hz','-20, 20');await click('全选音频');
  await page.getByText('每份音频 4 个版本，所选 2 份共 8 个',{exact:true}).waitFor();await click('开始批量处理');await page.getByRole('dialog',{name:'批量保存',exact:true}).waitFor();assert.equal(await page.getByRole('dialog').locator('.file-list input[type=checkbox]').count(),8);report.checks.push('batch page exposes speed and exclusive Hz mode, all two-source/eight results auto-open save');
  await rpc('test',{output_mode:'cancel'});await click('保存所选音频');await page.getByText('已取消保存，生成结果保留',{exact:true}).waitFor();assert.equal((await fs.readdir(path.join(scratch,'saved'))).length,0);await cancelModal();
  await click('保存最近批次');await rpc('test',{});await click('保存所选音频');await page.getByRole('dialog').waitFor({state:'hidden'});await idle();
  let audit=(await rpc('test',{})).value;assert.equal(audit.choices,2);assert.equal(audit.audit.length,8);
  for(const a of audit.audit){const speed=Number(/speed-([\d.]+)/.exec(a.name)[1]),shift=Number(/pitch-([+-][\d.]+)Hz/.exec(a.name)[1]),base=a.name.startsWith('原音 二')?220:150;assert(Math.abs(a.duration-1/speed)<.001,JSON.stringify(a));assert(Math.abs(a.f0-base-shift)<1,JSON.stringify(a));}
  report.checks.push('cancel preserves all outputs; retry selects one native directory for all eight WAVs; actual durations and F0 match 150/220Hz with +/-20Hz');
  await click('保存最近批次');await click('保存所选音频');await page.getByRole('dialog').waitFor({state:'hidden'});audit=(await rpc('test',{})).value;assert.equal(audit.audit.length,8);report.checks.push('repeat save reuses unchanged existing copies without duplication');
  const failedId=requests.findLast(r=>r.body.op==='m08_export').body.id;await rpc('test',{fail_export:failedId});await click('保存最近批次');await click('保存所选音频');await page.getByRole('dialog').getByRole('alert').filter({hasText:'M08 test disk unavailable'}).waitFor();assert.equal(await page.getByRole('dialog').locator('input:checked').count(),1);await rpc('test',{});await click('保存所选音频');await page.getByRole('dialog').waitFor({state:'hidden'});report.checks.push('one actual export failure retains only its checkbox for targeted retry');
  await click('保存最近批次');const numbering=page.getByRole('checkbox',{name:'在文件名前添加序号（1_、2_、3_…）'});assert.equal(await numbering.isChecked(),false);await numbering.check();
  const originalNames=await page.getByRole('dialog').locator('.file-list label span').allTextContents();
  await rpc('test',{fail_export:failedId});await click('保存所选音频');await page.getByRole('dialog').getByRole('alert').filter({hasText:'M08 test disk unavailable'}).waitFor();assert.equal(await page.getByRole('dialog').locator('.file-list input:checked').count(),1);
  await rpc('test',{});await click('保存所选音频');await page.getByRole('dialog').waitFor({state:'hidden'});await idle();
  audit=(await rpc('test',{})).value;assert.equal(audit.audit.length,16);
  for(let i=0;i<originalNames.length;i++){const plain=audit.audit.find(r=>r.name===originalNames[i]),numbered=audit.audit.find(r=>r.name===`${i+1}_${originalNames[i]}`);assert(plain&&numbered);assert.equal(plain.sha256,numbered.sha256);}
  await click('保存最近批次');assert.equal(await numbering.isChecked(),false);assert.deepEqual(await page.getByRole('dialog').locator('.file-list label span').allTextContents(),originalNames);await cancelModal();
  report.checks.push('numbering defaults off on every open; opt-in adds 1_ through 8_; partial retry preserves index, bytes and history names');
  // Result from a different source must open that source, respecting current dirty draft.
  const target=page.locator('.task-row').filter({hasText:opts[1].name}).last().locator('.task-result').first();const targetName=await target.locator('span').first().innerText();await target.click();
  await page.getByRole('dialog',{name:'保留当前编辑'}).waitFor();await click('放弃并切换');await idle();await page.waitForFunction(id=>document.querySelector('[aria-label="变速变调音频"]').value===id,opts[1].id);
  await page.waitForFunction(name=>[...document.querySelectorAll('.task-result[aria-pressed=true]')].some(e=>e.textContent.includes(name)),targetName);assert(await page.locator('.single').isVisible());
  const other=page.locator('.task-row').filter({hasText:opts[0].name}).first().locator('.task-result').first();await other.click();await idle();await page.waitForFunction(id=>document.querySelector('[aria-label="变速变调音频"]').value===id,opts[0].id);report.checks.push('task versions switch between source files and outputs, with save/discard guard for dirty F0');
  for(const width of [1800,1280,900])for(const theme of ['light','dark']){await page.setViewportSize({width,height:900});await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);await page.waitForTimeout(200);const box=await page.locator('.m08-page').evaluate(e=>({w:e.clientWidth,sw:e.scrollWidth}));assert(box.sw<=box.w+2,JSON.stringify(box));await page.screenshot({path:path.join(out,`${width}-${theme}.png`)});}
  report.checks.push('six narrow/wide light/dark layouts have no horizontal overflow');
  await page.reload();await page.locator('nav').getByRole('button',{name:'变速变调',exact:true}).click();await click('打开音频目录');await page.locator('.task-result').first().waitFor();await page.locator('.task-row').filter({hasText:opts[1].name}).last().locator('.task-result').first().click();await idle();await page.waitForFunction(()=>!!document.querySelector('.task-result[aria-pressed=true]'));assert.equal(await select.locator('option:checked').innerText(),opts[1].name);report.checks.push('reloaded historical task locates original file using name plus verified SHA-256 and switches correctly');
  await click('批量变速变调');await fill('语速倍率','0');const prior=requests.filter(r=>r.body.action==='create').length;await click('开始批量处理');await page.getByRole('alert').filter({hasText:'语速倍率必须大于 0'}).waitFor();assert.equal(requests.filter(r=>r.body.action==='create').length,prior);report.checks.push('invalid parameters fail before submitting any task');
  // Exercise the production web exporter with the same real service result bytes.
  const download=page.waitForEvent('download');
  const exported=await page.evaluate(async()=>{
   const {m08Port}=await import('/src/platform/m08.ts');
   const rpc=async body=>{const response=await fetch('/__m08_host',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({channel:'task',body})});const result=await response.json();if(!result.ok)throw Error(result.error);return result.value;};
   const port=m08Port({project:'00000000-0000-4000-8000-000000000001',source:async()=>{throw Error('existing results only')},request:(action,body)=>rpc({op:'m08_call',action,body}),job:async()=>{throw Error('no copy job')},cancel:async()=>{},read:async(job,id)=>{const result=await rpc({op:'result',job,id});return Uint8Array.from(atob(result.base64),c=>c.charCodeAt(0)).buffer;}});
   const selected=(await port.jobs()).filter(j=>j.config?.action==='transform').flatMap(j=>j.results);const picker=window.showSaveFilePicker;window.showSaveFilePicker=undefined;
   try{return await port.saveMany(selected);}finally{window.showSaveFilePicker=picker;}
  });
  const downloaded=await download,targetZip=path.join(scratch,'web-variants.zip');await downloaded.saveAs(targetZip);const raw=await fs.readFile(targetZip),{crc32}=require('node:zlib');let position=0;const entries=[];
  while(raw.readUInt32LE(position)===0x04034b50){const size=raw.readUInt32LE(position+18),n=raw.readUInt16LE(position+26),extra=raw.readUInt16LE(position+28),name=raw.subarray(position+30,position+30+n).toString('utf8'),start=position+30+n+extra,bytes=raw.subarray(start,start+size);assert.equal(raw.readUInt32LE(position+14),crc32(bytes));assert.equal(bytes.subarray(0,4).toString(),'RIFF');entries.push(name);position=start+size;}
  assert.equal(entries.length,8);assert.deepEqual(entries,[...exported.saved.map(r=>r.name)]);report.checks.push('production web ZIP exporter downloads eight real WAVs across two sources with Unicode names and valid CRC; creates no copy tasks');
  await verifyManual();
  assert.deepEqual(report.errors,[]);report.success=true;
 }catch(e){report.failure=String(e.stack);await page.screenshot({path:path.join(out,'failed.png')});throw e;}
 finally{await fs.writeFile(path.join(out,'report.json'),JSON.stringify({...report,requests},null,2));await browser.close();await server.close();worker.stdin.end();await new Promise(r=>worker.once('exit',r));console.log(JSON.stringify({out,...report}));}
}
main().catch(e=>{console.error(e);process.exitCode=1});
