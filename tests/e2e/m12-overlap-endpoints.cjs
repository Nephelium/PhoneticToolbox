// M12: real shared Vue/Chrome events with an in-memory synthetic file port.
// No research files, databases or native directory permissions are used.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..'),out=path.join(root,'output/validation/m12-overlap-endpoints');
async function main(){
 await fs.mkdir(out,{recursive:true});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const end=2.000063,frames=32001,wav=Buffer.alloc(44+frames*2);
 wav.write('RIFF');wav.writeUInt32LE(wav.length-8,4);wav.write('WAVEfmt ',8);wav.writeUInt32LE(16,16);wav.writeUInt16LE(1,20);wav.writeUInt16LE(1,22);wav.writeUInt32LE(16000,24);wav.writeUInt32LE(32000,28);wav.writeUInt16LE(2,32);wav.writeUInt16LE(16,34);wav.write('data',36);wav.writeUInt32LE(wav.length-44,40);
 for(let i=0;i<frames;i++)wav.writeInt16LE(Math.round(6000*Math.sin(2*Math.PI*220*i/16000)),44+i*2);
 const grid=(low,high,label)=>Buffer.from(`"ooTextFile short" "TextGrid" ${low} ${high} <exists> 2 "IntervalTier" "words" ${low} ${high} 1 ${low} ${high} "${label}" "IntervalTier" "phones" ${low} ${high} 1 ${low} ${high} "${label}"`);
 const buffers=new Map([['端点.wav',wav],['端点.TextGrid',grid(0,end,'')],['复用.wav',wav],['复用.TextGrid',grid(0,end,'原标注')]]),saved=new Map(),targets=new Map();let failNextSave=false;
 const sha=buffer=>crypto.createHash('sha256').update(buffer).digest('hex');
 const file=name=>({id:name,name,kind:name.endsWith('.wav')?'audio':'textgrid',size:buffers.get(name).length,sha256:sha(buffers.get(name))});
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0},plugins:[{name:'m12-synthetic-files',configureServer(s){s.middlewares.use('/__m12',(req,res)=>{let body='';req.on('data',d=>body+=d);req.on('end',()=>{try{const q=JSON.parse(body);let value;
  if(q.op==='choose')value={id:'fixture',label:'合成标注检查',purpose:'input'};
  else if(q.op==='annotation_scan')value=[...buffers.keys()].map(file);
  else if(q.op==='read'){const b=buffers.get(q.id);value={base64:b.toString('base64'),sha256:sha(b)};}
  else if(q.op==='annotation_target'){const name=q.id.replace(/\.wav$/i,'')+q.suffix+'.TextGrid';targets.set(name,name);value={id:name,name,sha256:buffers.has(name)?sha(buffers.get(name)):null};}
  else if(q.op==='annotation_save'){if(failNextSave){failNextSave=false;throw Error('合成写入失败，编辑仍保留。');}assert(targets.has(q.target));const b=Buffer.from(q.text);buffers.set(q.target,b);saved.set(q.target,q.text);value={file:file(q.target),name:q.target,sha256:sha(b)};}
  else throw Error('Unexpected '+q.op);res.setHeader('Content-Type','application/json');res.end(JSON.stringify({value}));
 }catch(e){res.end(JSON.stringify({error:e.message}));}});});}}]});
 let browser,page;const checks=[],errors=[];
 try{
  await server.listen();browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});page=await browser.newPage({viewport:{width:1600,height:1100}});page.setDefaultTimeout(10000);page.on('pageerror',e=>errors.push(e.message));
  const click=name=>page.getByRole('button',{name,exact:true}).first().click();
  const ready=()=>page.waitForFunction(()=>document.querySelector('.annotation-grid')&&document.querySelector('.annotation-page')?.getAttribute('aria-busy')==='false');
  const choose=async name=>{await page.locator('.annotation-file-list button').filter({hasText:name}).click();await ready();};
  const double=async(selector,side)=>{const p=page.locator(selector).first();await p.scrollIntoViewIfNeeded();const r=await p.boundingBox();await page.mouse.dblclick(r.x+(side==='left'?1.5:r.width-1.5),r.y+r.height*(selector.includes('annotation-grid')?.22:.5),{delay:60});};
  const save=async name=>{await page.getByRole('button',{name:/^保存 TextGrid/}).first().click();await page.waitForFunction(()=>document.querySelector('.annotation-page')?.getAttribute('aria-busy')==='false');assert(saved.has(name));return saved.get(name);};
  const undo=async()=>{await page.locator('.annotation-grid').focus();await page.keyboard.press('Control+z');await page.keyboard.press('Escape');};
  await page.goto(server.resolvedUrls.local[0]+'tests/m12-live.html');await click('TextGrid标注');await click('打开音频目录');await choose('端点.wav');
  for(const sequence of [false,true]){
   if(sequence){await click('粘贴词表');await page.getByLabel('拼音词表',{exact:true}).fill('a1');await click('应用词表');await page.getByRole('checkbox',{name:'自动生成音节和音素'}).check();}
   for(const selector of ['.annotation-plots .wave-track svg','.spectrum-wrap canvas','.annotation-grid']){
    console.log('endpoint',sequence,selector);
    await double(selector,'left');await page.getByRole('status').filter({hasText:'起点 0.000000 s'}).waitFor();await double(selector,'right');
    await page.getByLabel('编辑选中标注文本').fill(sequence?'a1':'首尾 æ');await page.getByLabel('编辑选中标注文本').press('Enter');
    const text=await save('端点_自动保存.TextGrid');assert.match(text,new RegExp('xmin = 0\\.0\\s+xmax = 2\\.000063\\s+text = "'+(sequence?'a1':'首尾 æ')+'"'));
    assert.equal(await page.getByRole('alert').count(),0);checks.push(`${sequence?'sequence':'manual'}: ${selector}, actual first/last pixels create exact 0..2.000063 interval and save`);await undo();
   }
  }
  await choose('复用.wav');await page.getByRole('checkbox',{name:'自动生成音节和音素'}).uncheck();
  for(const mode of ['inside','outside','before','after']){
   await page.locator('input[type=file][accept=".TextGrid,.textgrid"]').setInputFiles({name:'范围不同.TextGrid',mimeType:'text/plain',buffer:grid(.25,1.75,'参考 æ')});
   await page.getByLabel('参考复用模式').selectOption(mode);await page.getByLabel('参考起点',{exact:true}).fill('.5');if(['inside','outside'].includes(mode))await page.getByLabel('参考终点',{exact:true}).fill('1.5');await click('复用参考标注');
   assert.equal(await page.getByRole('alert').count(),0);const text=await save('复用_自动保存.TextGrid');assert.match(text,/text = "参考 æ"/);assert.match(text,/text = "原标注"/);checks.push(`${mode}: different reference domain imported, saved, outside coverage preserved`);await undo();
  }
  await page.locator('input[type=file][accept=".TextGrid,.textgrid"]').setInputFiles({name:'更长参考.TextGrid',mimeType:'text/plain',buffer:grid(0,3,'更长参考')});
  await page.getByLabel('参考复用模式').selectOption('inside');await page.getByLabel('参考起点',{exact:true}).fill('0');await page.getByLabel('参考终点',{exact:true}).fill('9');await click('复用参考标注');let text=await save('复用_自动保存.TextGrid');assert(!text.includes('xmax = 3'));assert.match(text,/xmax = 2\.000063/);checks.push('longer reference clipped to current file end');await undo();
  await page.locator('input[type=file][accept=".TextGrid,.textgrid"]').setInputFiles({name:'无交集.TextGrid',mimeType:'text/plain',buffer:grid(3,4,'无交集')});await click('复用参考标注');await page.getByRole('alert').filter({hasText:'没有重叠'}).waitFor();checks.push('no overlap shows actionable error without replacement');
  const selectWord=async()=>{const p=page.locator('.annotation-grid');await p.scrollIntoViewIfNeeded();const r=await p.boundingBox();await page.mouse.click(r.x+r.width*.4,r.y+r.height*.2);};
  await choose('端点.wav');await selectWord();await page.getByLabel('编辑选中标注文本').fill('切换保留A');await page.getByLabel('编辑选中标注文本').press('Enter');
  await choose('复用.wav');assert.match(saved.get('端点_自动保存.TextGrid'),/切换保留A/);await choose('端点.wav');await selectWord();assert.equal(await page.getByLabel('编辑选中标注文本').inputValue(),'切换保留A');checks.push('switch A to B awaits autosave; returning A restores latest edit instead of original');
  await page.getByLabel('编辑选中标注文本').fill('尚未按Enter');await choose('复用.wav');assert.match(saved.get('端点_自动保存.TextGrid'),/尚未按Enter/);await choose('端点.wav');await selectWord();assert.equal(await page.getByLabel('编辑选中标注文本').inputValue(),'尚未按Enter');checks.push('switch commits the input field even without pressing Enter');
  await page.getByLabel('编辑选中标注文本').fill('失败保留B');await page.getByLabel('编辑选中标注文本').press('Enter');failNextSave=true;await page.locator('.annotation-file-list button').filter({hasText:'复用.wav'}).click();await page.getByRole('alert').filter({hasText:'写入失败'}).waitFor();assert.equal(await page.getByLabel('编辑选中标注文本').inputValue(),'失败保留B');assert.equal(await page.locator('.annotation-file-list button.selected').innerText(),'端点.wav');
  await choose('复用.wav');await choose('端点.wav');await selectWord();assert.equal(await page.getByLabel('编辑选中标注文本').inputValue(),'失败保留B');checks.push('failed TextGrid save blocks navigation, keeps text; retry and return restore the edit');
  await page.reload();await click('TextGrid标注');await click('打开音频目录');await choose('端点.wav');await selectWord();assert.equal(await page.getByLabel('编辑选中标注文本').inputValue(),'失败保留B');checks.push('IndexedDB history restores current edit across page reload');
  const versions=page.getByLabel('标注历史版本');await versions.selectOption({label:await versions.locator('option').filter({hasText:'原始标注'}).innerText()});await click('恢复此版本');await ready();assert(!saved.get('端点_自动保存.TextGrid').includes('失败保留B'));assert.match(await versions.locator('option').first().innerText(),/恢复版本/);checks.push('original version restores via a new saved version, previous edits retained');
  await selectWord();await page.getByLabel('编辑选中标注文本').fill('删除这个标注');await page.getByLabel('编辑选中标注文本').press('Enter');await page.locator('.annotation-grid').focus();await page.keyboard.press('Delete');await save('端点_自动保存.TextGrid');assert(!saved.get('端点_自动保存.TextGrid').includes('删除这个标注'));checks.push('Delete key removes selected annotation and saved readback is empty');
  for(const theme of ['light','dark']){await page.evaluate(async mode=>{const {paletteTokens}=await import('/src/design/themes.ts');document.documentElement.dataset.theme=mode;for(const[k,v]of Object.entries(paletteTokens('nord',mode)))document.documentElement.style.setProperty(k,v);},theme);await page.screenshot({path:path.join(out,theme+'.png')});}
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'report.json'),JSON.stringify({scope:'Windows isolated Chrome, actual Vue pointer events and in-memory synthetic file port; no native directory chain',fixture:{frames,sampleRate:16000,audioEnd:frames/16000,gridEnd:end},checks,errors,cleanup:'WAV/TextGrid only in memory; browser and Vite closed in finally'},null,2));console.log(JSON.stringify({checks,errors},null,2));
 }catch(e){await page?.screenshot({path:path.join(out,'failure.png')});await fs.writeFile(path.join(out,'failure.txt'),String(e)+'\n'+(page?await page.locator('body').innerText():''));throw e;}
 finally{await browser?.close();await server.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
