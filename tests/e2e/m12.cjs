// M12: project-owned isolated Chrome with real file capabilities, no EXE/DDL.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{spawn}=require('node:child_process'),{createInterface}=require('node:readline'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const worker=spawn(path.join(root,'.venv/m09-ui/Scripts/python.exe'),['-X','utf8','scripts/m12_ui_bridge.py'],{cwd:root,windowsHide:true,env:{...process.env,PYTHONPATH:['backend/src','desktop/src','packages/phonetic_core/src'].map(p=>path.join(root,p)).join(';')}});
 const pending=new Map();let counter=0,readyResolve,readyReject;
 const ready=new Promise((r,j)=>{readyResolve=r;readyReject=j;});
 createInterface({input:worker.stdout}).on('line',line=>{const v=JSON.parse(line);if(v.ready)readyResolve(v);else{pending.get(v.id)?.(v);pending.delete(v.id);}});worker.stderr.on('data',d=>process.stderr.write(d));worker.on('error',readyReject);worker.once('exit',code=>readyReject(Error('RPC exited '+code)));
 const {out}=await ready;const rpc=data=>new Promise(resolve=>{const id=++counter;pending.set(id,resolve);worker.stdin.write(JSON.stringify({...data,rpc_id:id})+'\n');});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:46212},optimizeDeps:{entries:['tests/m12-live.html']},plugins:[{name:'m12-owned-rpc',configureServer(s){s.middlewares.use('/__m12',(req,res)=>{let body='';req.on('data',d=>body+=d);req.on('end',async()=>{const value=await rpc(JSON.parse(body));res.setHeader('Content-Type','application/json');res.end(JSON.stringify(value));});});}}]});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--mute-audio']}),context=await browser.newContext({viewport:{width:1600,height:1100},acceptDownloads:true}),page=await context.newPage(),checks=[],errors=[];
 await page.clock.install();
 page.on('pageerror',e=>errors.push(e.message));
 const click=name=>page.getByRole('button',{name,exact:true}).first().click();
 const loaded=()=>page.waitForFunction(()=>document.querySelector('.annotation-page')?.getAttribute('aria-busy')==='false'&&document.querySelector('.annotation-grid'));
 const gridClick=async(time,row=0)=>{const box=await page.locator('.annotation-grid').boundingBox();await page.mouse.click(box.x+box.width*time/2,box.y+box.height*(row===0?.22:.73));};
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/m12-live.html');await click('语音标注对齐');await click('选择语料文件夹');await page.locator('.annotation-file-list button').first().waitFor();
  await page.locator('.annotation-file-list button').filter({hasText:'audio_recording.wav'}).click();await loaded();
  assert.equal(await page.getByLabel('唇形共同偏移毫秒').inputValue(),'7');assert.equal(await page.getByLabel('词层名',{exact:true}).inputValue(),'words');
  await page.screenshot({path:path.join(out,'light-loaded.png'),fullPage:true});checks.push('actual WAV/TextGrid/PKL/lab load, default names, shared wave, original lip offset');
  await require('./m12-gestures.cjs')({page,click,loaded,out,checks});
  await gridClick(.3);await page.getByLabel('编辑选中标注文本').fill('井井 æ');await page.getByLabel('编辑选中标注文本').press('Enter');await click('保存 TextGrid *');await loaded();
  let data=(await rpc({op:'inspect'})).value;assert(data.grids['audio_recording_自动保存.TextGrid'].tiers[0].intervals.some(i=>i.text==='井井 æ'));assert.equal(data.grids['audio_recording_自动保存.TextGrid'].tiers[2].points[0].mark,'事件 ʔ');checks.push('Chinese/IPA real edit/save reread, point tier preserved');
  const gridHash=data.hashes['audio_recording_自动保存.TextGrid'];await page.getByLabel('唇形共同偏移毫秒').fill('-13');await page.getByLabel('唇形共同偏移毫秒').press('Tab');await click('保存唇偏 *');await loaded();data=(await rpc({op:'inspect'})).value;assert.equal(data.lip_offset,-.013);assert.equal(data.hashes['audio_recording_自动保存.TextGrid'],gridHash);checks.push('negative lip offset independently persisted without changing TextGrid');
  await page.locator('input[type=file][accept=".dict,.txt"]').setInputFiles(path.join(out,'custom.dict'));await page.getByRole('status').filter({hasText:'custom.dict'}).waitFor();
  await page.getByLabel('搜索词层文本').fill('ba2');await page.getByLabel('替换文本').fill('井井');await click('全部替换');await page.getByRole('dialog').waitFor();await click('替换全部');await click('撤销');checks.push('dictionary upload, search/replace confirmation and undo');
  await page.locator('input[type=file][accept=".TextGrid,.textgrid"]').setInputFiles(path.join(out,'reference.TextGrid'));await page.getByRole('status').filter({hasText:'reference.TextGrid'}).waitFor();
  for(const mode of ['inside','outside','before','after']){await page.getByLabel('参考复用模式').selectOption(mode);await page.getByLabel('参考起点').fill('.5');if(['inside','outside'].includes(mode))await page.getByLabel('参考终点').fill('1.5');await click('复用参考标注');assert(!(await page.getByRole('alert').count()));await click('撤销');}checks.push('all four reference modes with point tiers, each undoable');
  await page.getByLabel('强度内收毫秒').fill('-10');await click('强度贴合');await page.getByRole('status').filter({hasText:'强度贴合'}).waitFor();checks.push('negative outward intensity fit accepted');
  await click('参数显示');await click('语音标注对齐');assert.equal(await page.getByLabel('唇形共同偏移毫秒').inputValue(),'-13');assert.equal(await page.getByLabel('参考复用模式').inputValue(),'after');checks.push('module tabs retain document, resource and lip state');
  await page.getByLabel('TextGrid 保存后缀').fill('');await click('保存 TextGrid *');await page.getByRole('dialog').filter({hasText:'确认覆盖原始标注'}).waitFor();await click('取消');await page.getByLabel('TextGrid 保存后缀').fill('_自动保存');await click('保存 TextGrid *');await loaded();checks.push('empty suffix actual target preview and overwrite confirmation');
  await page.getByLabel('配色主题').selectOption('dark');await page.screenshot({path:path.join(out,'dark-edited.png'),fullPage:true});await page.setViewportSize({width:800,height:720});await page.screenshot({path:path.join(out,'small-dark.png'),fullPage:true});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));checks.push('dark theme and small viewport no document horizontal overflow');
  await page.setViewportSize({width:1600,height:1100});await page.getByLabel('搜索词层文本').fill('ba2');await page.getByLabel('替换文本').fill('conflict');await click('替换');await rpc({op:'external_change'});await click('保存 TextGrid *');await page.getByRole('alert').waitFor();assert(await page.getByRole('alert').filter({hasText:/变化|修改/}).count());assert(await page.getByRole('button',{name:'保存 TextGrid *',exact:true}).count());checks.push('external file modification refuses overwrite and keeps unsaved editor');
  await click('关闭模块');await page.getByRole('dialog').filter({hasText:'保存标注修改'}).waitFor();await click('保存修改并关闭');await page.getByRole('dialog').filter({hasText:'标注或唇偏保存失败'}).waitFor();await click('取消关闭');assert.equal(await page.locator('.annotation-page').count(),1);checks.push('failed close-and-save retains module and offers cancel without losing edits');
  await click('关闭模块');await click('放弃修改并关闭');
  const natural=(await rpc({op:'natural'}));assert(!natural.error,natural.error);
  await click('语音标注对齐');await click('选择语料文件夹');
  for(const item of natural.value){
   await page.locator('.annotation-file-list button').filter({hasText:item.case+'.wav'}).click();await loaded();
   await page.getByLabel('词层名',{exact:true}).selectOption(item.word);await page.getByLabel('音素层名',{exact:true}).selectOption(item.phone);
   await page.locator('.annotation-grid').scrollIntoViewIfNeeded();const box=await page.locator('.annotation-grid').boundingBox();await page.mouse.click(box.x+box.width*item.time/3.2,box.y+box.height*.22);await page.getByLabel('编辑选中标注文本').fill('M12验证 æ');await page.getByLabel('编辑选中标注文本').press('Enter');await click('保存 TextGrid *');await loaded();assert.equal(await page.getByRole('alert').count(),0);
  }
  const naturalResult=await rpc({op:'natural_verify'});assert(!naturalResult.error,naturalResult.error);assert.equal(naturalResult.value.length,2);checks.push('two previously authorized natural recordings load/edit/save via real UI, custom tier names and original hashes unchanged');
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'report.json'),JSON.stringify({checks,errors},null,2));console.log(JSON.stringify({out,checks},null,2));
 }catch(error){await page.screenshot({path:path.join(out,'failure.png'),fullPage:true});await fs.writeFile(path.join(out,'failure.txt'),String(error)+'\n'+await page.locator('body').innerText()+'\nErrors:'+JSON.stringify(errors));console.error('M12 evidence:',out);throw error;}
 finally{await browser.close();await server.close();worker.stdin.end();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
