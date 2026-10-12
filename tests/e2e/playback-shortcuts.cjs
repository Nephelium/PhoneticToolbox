// Real Vue pages, browser File/WAV/TextGrid decoding and muted WebAudio.
// Synthetic data verifies scheduling and focus; physical hearing is not asserted.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{pathToFileURL}=require('node:url');
const {createHash}=require('node:crypto'),{spawnSync}=require('node:child_process');
const root=path.resolve(__dirname,'../..'),before=process.argv.includes('--before');
async function main(){
 const out=path.join(root,'output/validation/playback-shortcuts',(before?'before-':'after-')+Date.now()),scratch=path.join(out,'scratch');
 await fs.mkdir(scratch,{recursive:true});const write=(name,value)=>fs.writeFile(path.join(out,name),JSON.stringify(value,null,2)+'\n');
 const marker={schema:'ptb-test-run/1',owner_pid:process.pid,state:'active',recipe:'Public stereo WAV and generated TextGrid, actual M01/M02/M04/M07/M08 pages with synthetic data, shared multi-audio keyboard focus and audio modal; muted independent Chrome',started_at:new Date().toISOString()};await write('run.json',marker);
 const report={success:false,before,checks:[],errors:[]};let server,browser;
 try{
  const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
  server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{noDiscovery:true,include:['vue']}});await server.listen();
  const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  browser=await chromium.launchPersistentContext(path.join(scratch,'profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--mute-audio'],viewport:{width:1600,height:1000}});
  report.browser=browser.browser().version();const page=await browser.newPage();page.on('pageerror',e=>report.errors.push(e.message));
  await page.addInitScript(()=>{window.starts=[];const start=AudioBufferSourceNode.prototype.start;AudioBufferSourceNode.prototype.start=function(...args){window.starts.push({args,rate:this.buffer.sampleRate,sample:this.buffer.getChannelData(0)[10]});return start.apply(this,args);};});
  const url=server.resolvedUrls.local[0],wav=await fs.readFile(path.join(root,'frontend/src/assets/SYN-EGG-44100.wav'));
  async function dragWave(index,left=.2,right=.7){const svg=page.locator('.wave-viewport').nth(index).locator('.wave-track>svg').first();await svg.scrollIntoViewIfNeeded();const bounds=await svg.boundingBox();assert(bounds);await page.mouse.move(bounds.x+bounds.width*left,bounds.y+bounds.height/2);await page.mouse.down();await page.mouse.move(bounds.x+bounds.width*right,bounds.y+bounds.height/2,{steps:6});await page.mouse.up();}
  async function stopsWithSpace(){const count=await page.evaluate(()=>starts.length),reads=await page.evaluate(()=>qa.reads.value);assert.equal(await page.evaluate(()=>qa.playback.playing),true);await page.keyboard.press('Space');assert.equal(await page.evaluate(()=>qa.playback.playing),false);assert.equal(await page.evaluate(()=>starts.length),count);assert.equal(await page.evaluate(()=>qa.reads.value),reads);}
  const grid=`File type = "ooTextFile"\nObject class = "TextGrid"\nxmin = 0\nxmax = 0.8\ntiers? <exists>\nsize = 1\nitem []:\n item [1]:\n  class = "IntervalTier"\n  name = "words"\n  xmin = 0\n  xmax = 0.8\n  intervals: size = 3\n  intervals [1]:\n   xmin = 0\n   xmax = 0.2\n   text = "a"\n  intervals [2]:\n   xmin = 0.2\n   xmax = 0.6\n   text = "b"\n  intervals [3]:\n   xmin = 0.6\n   xmax = 0.8\n   text = "c"\n`;
  for(const [id,title,scope] of [['M01','参数估计','.m01-page'],['M04','LPC 谱图','.lpc-page']]){
   await page.goto(url+'tests/playback-shortcuts.html?mode=shell');await page.locator('nav').getByRole('button',{name:title,exact:true}).click();
   const module=page.locator(scope);await module.locator('input[type=file]').setInputFiles([{name:'space.wav',mimeType:'audio/wav',buffer:wav},{name:'space.TextGrid',mimeType:'text/plain',buffer:Buffer.from(grid)}]);
   if(id==='M01')await module.locator('.file-row').filter({hasText:'space.wav'}).click();else await module.getByLabel('LPC 音频文件',{exact:true}).selectOption({label:'space.wav'});
   const interval=module.locator('.textgrid-interval').nth(1);await interval.click();assert.equal(await interval.getAttribute('data-xmin'),'0.2');assert.equal(await interval.getAttribute('data-xmax'),'0.6');
   await page.keyboard.press('Space');await page.waitForTimeout(60);
   if(before){assert.equal(await page.evaluate(()=>starts.length),0);assert.equal(await page.evaluate(()=>document.activeElement.className),'textgrid-interval selected');await page.getByRole('button',{name:'播放选区',exact:true}).click();}
   await page.waitForFunction(()=>starts.length===1);const first=await page.evaluate(()=>starts[0]);assert.equal(first.args[1],.2);assert(Math.abs(first.args[2]-.4)<1e-12);assert.equal(first.rate,44100);
   if(!before){await page.keyboard.press('Space');assert.equal(await page.getByRole('button',{name:'暂停',exact:true}).count(),0);assert.equal(await page.evaluate(()=>starts.length),1);}
   report.checks.push(id+(before?': reproduced interval button Space failure; transport schedules exact .2–.6 s':': interval click + Space schedules exact .2–.6 s once and second Space stops'));
   await page.screenshot({path:path.join(out,id+'.png')});
  }
  await page.goto(url+'tests/playback-shortcuts.html?mode=m02');await page.locator('.m02-files button').click();await page.locator('.m02-waveform-area svg').first().waitFor();
  await page.getByLabel('F0',{exact:true}).check();await page.getByRole('button',{name:'将 1 项分配到图窗',exact:true}).click();
  const chart=page.locator('.parameter-chart');await chart.waitFor();await page.getByRole('button',{name:'保存绘图配置',exact:true}).focus();
  const box=await chart.boundingBox();await page.mouse.move(box.x+box.width*.3,box.y+box.height*.5);await page.mouse.down();await page.mouse.move(box.x+box.width*.6,box.y+box.height*.5,{steps:6});await page.mouse.up();
  const range=await page.evaluate(()=>[qa.wave.start,qa.wave.end]);assert(range[1]>range[0]);await page.keyboard.press('Space');await page.waitForTimeout(60);
  if(before){assert.equal(await page.evaluate(()=>starts.length),0);report.checks.push('M02: parameter chart selection retains prior button focus and Space fails');}
  else{await page.waitForFunction(()=>starts.length===1);const played=await page.evaluate(()=>starts[0]);assert.equal(played.args[1],range[0]);assert.equal(played.args[2],range[1]-range[0]);await page.keyboard.press('Space');report.checks.push('M02: parameter plot transfers focus and Space schedules its exact selection once');}
  if(!before){
   await page.goto(url+'tests/playback-shortcuts.html?mode=m07');
   await page.getByRole('combobox',{name:'源音频',exact:true}).selectOption('wave');await page.getByRole('combobox',{name:'目标音频',exact:true}).selectOption('wave');await page.waitForFunction(()=>qa.m07States.slice(0,2).every(s=>s.asset));
   for(const index of [0,1,0]){await dragWave(index);const selected=await page.evaluate(i=>[qa.m07States[i].start,qa.m07States[i].end],index),count=await page.evaluate(()=>starts.length);await page.keyboard.press('Space');await page.waitForFunction(n=>starts.length===n+1,count);const played=await page.evaluate(()=>starts.at(-1));assert.equal(played.args[1],selected[0]);assert.equal(played.args[2],selected[1]-selected[0]);assert.equal(await page.evaluate(i=>qa.isCurrentAudio(qa.m07States[i].asset,0),index),true);await stopsWithSpace();}
   await page.getByRole('button',{name:'刷新任务历史',exact:true}).click();await page.locator('.result-audios button').first().waitFor();await page.waitForFunction(()=>qa.m07States[2].asset);
   const resultCount=await page.evaluate(()=>starts.length);await page.locator('.result-audios button').first().click();await page.waitForFunction(n=>starts.length===n+1,resultCount);assert.equal(await page.evaluate(()=>qa.isCurrentAudio(qa.m07States[2].asset,0)),true);assert.deepEqual(await page.evaluate(()=>starts.at(-1).args.slice(1)),[0,3]);await stopsWithSpace();await page.keyboard.press('Space');await page.waitForFunction(n=>starts.length===n+2,resultCount);await stopsWithSpace();
   report.checks.push('M07: source/target/source drag selects exact audio; result audition button Space stops and replays without reloading or duplicate starts');await page.screenshot({path:path.join(out,'M07.png')});
   await page.goto(url+'tests/playback-shortcuts.html?mode=m08');await page.getByRole('combobox',{name:'变速变调音频',exact:true}).selectOption('wave');await page.waitForFunction(()=>qa.m08Source.asset&&document.querySelector('.m08-curve')?.tagName==='svg');
   for(const name of ['播放当前视野原音','试听此版本','播放合成音','播放当前视野原音']){const count=await page.evaluate(()=>starts.length);await page.getByRole('button',{name,exact:true}).click();await page.waitForFunction(n=>starts.length===n+1,count);assert.deepEqual(await page.evaluate(()=>starts.at(-1).args.slice(1)),[0,3]);assert.equal(await page.evaluate(()=>qa.isCurrentAudio(qa.m08Source.asset,0)),name==='播放当前视野原音');await stopsWithSpace();}
   for(const index of [0,1,0]){await dragWave(index);const count=await page.evaluate(()=>starts.length);await page.keyboard.press('Space');await page.waitForFunction(n=>starts.length===n+1,count);const played=await page.evaluate(()=>starts.at(-1));assert(played.args[1]>0&&played.args[2]<3);assert.equal(await page.evaluate(()=>qa.isCurrentAudio(qa.m08Source.asset,0)),index===0);await stopsWithSpace();}
   report.checks.push('M08: visible source, history and result auditions stop with Space without reload; source/result/source drag still routes selected audio');await page.screenshot({path:path.join(out,'M08.png')});
   await page.goto(url+'tests/playback-shortcuts.html?mode=modal');await page.getByRole('button',{name:'Open audio dialog',exact:true}).click();await page.locator('dialog[open]').waitFor();await dragWave(1);await page.keyboard.press('Space');await page.waitForFunction(()=>starts.length===1);assert.equal(await page.evaluate(()=>qa.isCurrentAudio(qa.states[1].asset,0)),true);await stopsWithSpace();
   await page.locator('dialog').getByRole('button',{name:'全部',exact:true}).click();await page.keyboard.press('Space');await page.waitForFunction(()=>starts.length===2);assert.deepEqual(await page.evaluate(()=>starts.at(-1).args.slice(1)),[0,3]);await stopsWithSpace();await page.getByRole('button',{name:'Native dialog action',exact:true}).focus();await page.keyboard.press('Space');assert.equal(await page.evaluate(()=>qa.nativeClicks.value),1);assert.equal(await page.evaluate(()=>starts.length),2);
   report.checks.push('Audio result modal: own selected audio plays/stops with Space; background transports remain silent and dialog buttons keep native action');
   await page.goto(url+'tests/playback-shortcuts.html');await page.locator('.textgrid-interval').last().waitFor();
   for(const [track,start,end] of [[0,1,2.5],[1,0,1],[0,2.5,3]]){
    await page.locator(`[data-track="${track}"] .textgrid-interval`).filter({hasText:start===1?'b':start===0?'a':'c'}).click();const count=await page.evaluate(()=>starts.length);
    await page.keyboard.press('Space');await page.waitForFunction(n=>starts.length===n+1,count);const last=await page.evaluate(()=>starts.at(-1));assert.equal(last.args[1],start);assert.equal(last.args[2],end-start);
    const ranges=await page.evaluate(()=>qa.states.map(s=>[s.start,s.end]));assert.deepEqual(ranges[1-track],[0,0]);await page.keyboard.press('Space');assert.equal(await page.evaluate(()=>qa.playback.playing),false);
   }
   report.checks.push('Shared multi-audio: TextGrid buttons transfer sole selection ownership A/B/A, exact selection playback once and stop');
   const count=await page.evaluate(()=>starts.length);
   for(const label of ['Text input','Text area','Native select','Checkbox','Range','Editable text']){await page.getByLabel(label,{exact:true}).focus();await page.keyboard.press('Space');assert.equal(await page.evaluate(()=>starts.length),count,label);}
   await page.getByRole('button',{name:'Native action',exact:true}).focus();await page.keyboard.press('Space');assert.equal(await page.evaluate(()=>qa.nativeClicks.value),1);assert.equal(await page.evaluate(()=>starts.length),count);
   await page.getByRole('link',{name:'Native link',exact:true}).focus();await page.keyboard.press('Space');assert.equal(await page.evaluate(()=>starts.length),count);
   await page.getByText('Native summary',{exact:true}).focus();await page.keyboard.press('Space');assert.equal(await page.locator('details').evaluate(e=>e.open),true);assert.equal(await page.evaluate(()=>starts.length),count);
   const selected=page.locator('[data-track="0"] .textgrid-interval').last();await selected.focus();
   for(const key of ['Control+Space','Alt+Space','Meta+Space']){await page.keyboard.press(key);assert.equal(await page.evaluate(()=>starts.length),count);}
   await selected.dispatchEvent('keydown',{key:' ',code:'Space',repeat:true});await selected.dispatchEvent('keydown',{key:' ',code:'Space',isComposing:true});assert.equal(await page.evaluate(()=>starts.length),count);
   await page.evaluate(()=>qa.modal.value=true);await selected.focus();await page.keyboard.press('Space');assert.equal(await page.evaluate(()=>starts.length),count);await page.evaluate(()=>{qa.modal.value=false;qa.active.value=false;});await selected.focus();await page.keyboard.press('Space');assert.equal(await page.evaluate(()=>starts.length),count);
   report.checks.push('Focus guards: inputs, textarea, select, checkbox, range, contenteditable, native button/link/summary, modifiers, repeat, IME, dialog and inactive module preserve playback');
  }
  assert.deepEqual(report.errors,[]);report.success=true;
 }catch(error){report.failure=String(error);throw error;}
 finally{
  if(browser)await browser.close();if(server)await server.close();await write('report.json',report);
  const files=[];async function inventory(directory){for(const entry of await fs.readdir(directory,{withFileTypes:true})){const file=path.join(directory,entry.name),stat=await fs.lstat(file);if(stat.isSymbolicLink())throw Error('Linked scratch');if(stat.isDirectory())await inventory(file);else files.push({path:path.relative(scratch,file),bytes:stat.size,sha256:createHash('sha256').update(await fs.readFile(file)).digest('hex')});}}
  await inventory(scratch);await write('scratch-manifest.json',{schema:'ptb-test-scratch-files/1',files});await write('run.json',{...marker,state:'ready',test_outcome:report.success?'completed':'failed'});
  const cleanup=spawnSync('powershell.exe',['-NoProfile','-ExecutionPolicy','Bypass','-File',path.join(root,'scripts/cleanup_test_scratch.ps1'),'-RunDirectory',out],{windowsHide:true,encoding:'utf8'});if(cleanup.status!==0)throw Error(cleanup.stderr||cleanup.stdout);
  console.log(JSON.stringify({success:report.success,before,checks:report.checks.length,output:out,failure:report.failure}));
 }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
