// Actual M07/M08 Vue pages and muted WebAudio in owned headless Chrome.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const out=path.join(root,'output/validation/audio-selection','chrome-'+Date.now());await fs.mkdir(out,{recursive:true});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const plugin={name:'selection-fixture',configureServer(server){server.middlewares.use(async(req,res,next)=>{
  if(!req.url?.startsWith('/selection-fixture.html'))return next();
  res.setHeader('content-type','text/html; charset=utf-8');res.end(await server.transformIndexHtml('/selection-fixture.html',`<!doctype html><html><head><link rel="stylesheet" href="/interaction-selection.css"><script src="/interaction-selection.js" defer></script></head><body><div id="fixture" style="height:100vh"></div><script type="module">
import {createApp,h} from 'vue';import '/src/design/tokens.css';
import M07 from '/src/modules/phonation-synthesis/PhonationSynthesisPage.vue';
import M08 from '/src/modules/pitch-manipulation/PitchManipulationPage.vue';
import {workspace} from '/src/state/workspace.ts';import {playback,isCurrentAudio} from '/src/state/audio.ts';
function wav(hz){const rate=8000,n=16000,b=new ArrayBuffer(44+n*2),v=new DataView(b);const text=(o,s)=>[...s].forEach((c,i)=>v.setUint8(o+i,c.charCodeAt(0)));text(0,'RIFF');v.setUint32(4,b.byteLength-8,true);text(8,'WAVEfmt ');v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,rate,true);v.setUint32(28,rate*2,true);v.setUint16(32,2,true);v.setUint16(34,16,true);text(36,'data');v.setUint32(40,n*2,true);for(let i=0;i<n;i++)v.setInt16(44+i*2,Math.round(12000*Math.sin(2*Math.PI*hz*i/rate)),true);return b;}
const names=['source','target'],files=names.map((id,i)=>({id,name:id+'.wav',kind:'audio',size:32044,sha256:id}));
const buffers={source:wav(220),target:wav(440),result:wav(880)};
const context={label:'Synthetic UI fixture',files:{kind:'browser',list:async()=>files,read:async file=>({buffer:buffers[file.id].slice(0),sha256:file.id})}};
let readyResult=null;const m08={preview:async file=>({sha256:file.id,wav:buffers[file.id].slice(0),times:[0,.5,1,1.5,2],original_f0:[150,150,150,150,150]}),jobs:async()=>readyResult?[{id:'job1',state:'succeeded',source_id:'source',results:[readyResult]}]:[],history:async()=>readyResult?[readyResult]:[],audio:async()=>buffers.result.slice(0),submit:async(file,config)=>{readyResult={id:'result',name:'result.wav',source_id:file.id,start:config.start,end:config.end??2,config,times:[0,1,2],original_f0:[150,200,150]};return {id:'job1',state:'queued',source_id:file.id,results:[]};}};
const mode=new URLSearchParams(location.search).get('module')||'m07',key='selection-'+mode;
window.fixture={context,key,workspace,playback,isCurrentAudio,buffers,getStates:()=>mode==='m07'?[workspace(key+'.source'),workspace(key+'.target'),workspace(key)]:[workspace(key)]};
window.starts=[];const original=AudioContext.prototype.createBufferSource;AudioContext.prototype.createBufferSource=function(){const node=original.call(this),start=node.start;node.start=function(when,offset,duration){window.starts.push({offset,duration,sample:this.buffer.getChannelData(0)[10]});return start.call(this,when,offset,duration);};return node;};
createApp({render:()=>h(mode==='m07'?M07:M08,{context,stateKey:key,active:true,...(mode==='m08'?{port:m08}:{})})}).mount('#fixture');
</script></body></html>`));
 });}};
 const server=await createServer({root:path.join(root,'frontend'),plugins:[plugin],server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{noDiscovery:true,include:['vue']}});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--mute-audio']});
 const page=await browser.newPage({viewport:{width:1600,height:1000}}),checks=[],layouts=[],errors=[];page.on('pageerror',e=>errors.push(e.message));
 const url=server.resolvedUrls.local[0];
 const waves=()=>page.locator('.wave-viewport');
 const geometry=loc=>loc.evaluate(e=>{const r=e.getBoundingClientRect();return {width:r.width,height:r.height};});
 async function drag(index,a=.1,b=.6){const svg=waves().nth(index).locator('.wave-track>svg').first();await svg.scrollIntoViewIfNeeded();const box=await svg.boundingBox();assert(box);await page.mouse.move(box.x+box.width*a,box.y+box.height/2);await page.mouse.down();await page.mouse.move(box.x+box.width*b,box.y+box.height/2,{steps:8});await page.mouse.up();}
 const selection=()=>page.evaluate(()=>[...document.querySelectorAll('.wave-viewport')].map(e=>({active:e.dataset.selectionActive,rect:+(e.querySelector('.wave-selection')?.getAttribute('width')??0)})));
 const play=()=>page.keyboard.press('Space');
 try{
  await page.goto(url+'selection-fixture.html?module=m07');await page.getByRole('combobox',{name:'源音频',exact:true}).waitFor();
  const empty=await geometry(waves().first());layouts.push({emptyNodes:await waves().first().evaluate(e=>[...e.querySelectorAll('*')].map(x=>({tag:x.tagName,cls:x.getAttribute('class'),height:x.getBoundingClientRect().height,grid:getComputedStyle(x).gridTemplateRows}))) });assert((await geometry(waves().first().locator('svg'))).height>=90);
  await page.getByRole('combobox',{name:'源音频',exact:true}).selectOption('source');await page.getByRole('combobox',{name:'目标音频',exact:true}).selectOption('target');await page.waitForFunction(()=>fixture.getStates().slice(0,2).every(s=>s.asset));
  const live=await geometry(waves().first());layouts.push({liveNodes:await waves().first().evaluate(e=>[...e.querySelectorAll('*')].map(x=>({tag:x.tagName,cls:x.getAttribute('class'),height:x.getBoundingClientRect().height,grid:getComputedStyle(x).gridTemplateRows}))) });assert(Math.abs(live.height-empty.height)<2,JSON.stringify({empty,live}));layouts.push({module:'M07',empty,live});
  assert.deepEqual(await page.evaluate(()=>fixture.getStates().map(s=>[s.start,s.end])),[[0,0],[0,0],[0,0]]);await page.locator('h2').first().click();await play();assert.equal(await page.evaluate(()=>starts.length),0);
  await drag(0);await play();await page.waitForFunction(()=>fixture.playback.playing);assert.equal(await page.evaluate(()=>starts.length),1);
  assert.equal(await page.evaluate(()=>fixture.isCurrentAudio(fixture.getStates()[0].asset,0)),true);assert.deepEqual((await selection()).map(e=>e.rect>0),[true,false,false]);
  await play();assert.equal(await page.evaluate(()=>fixture.playback.playing),false);assert.equal(await page.evaluate(()=>starts.length),1);
  await drag(1,.2,.7);await play();await page.waitForFunction(()=>fixture.playback.playing);assert.equal(await page.evaluate(()=>starts.length),2);assert.equal(await page.evaluate(()=>fixture.isCurrentAudio(fixture.getStates()[1].asset,0)),true);assert.deepEqual((await selection()).map(e=>e.rect>0),[false,true,false]);
  await play();await drag(0,.3,.8);await play();await page.waitForFunction(()=>fixture.playback.playing);assert.equal(await page.evaluate(()=>starts.length),3);assert.equal(await page.evaluate(()=>fixture.isCurrentAudio(fixture.getStates()[0].asset,0)),true);await play();
  checks.push('M07 initial empty selection, real pointer source/target/source ownership, exactly one WebAudio start per Space and stop on second Space');
  await page.evaluate(async()=>{const {decodeWav}=await import('/src/platform/decode.ts');const s=fixture.getStates()[2];s.asset=await decodeWav(fixture.buffers.result.slice(0),'generated.wav');s.start=0;s.end=2;document.querySelector('.task-results').open=true;});
  await page.waitForFunction(()=>document.querySelectorAll('.wave-viewport').length===3);assert.equal(await page.evaluate(()=>fixture.getStates()[2].end),0);
  await drag(2);await play();await page.waitForFunction(()=>fixture.playback.playing);assert.equal(await page.evaluate(()=>fixture.isCurrentAudio(fixture.getStates()[2].asset,0)),true);assert.deepEqual((await selection()).map(e=>e.rect>0),[false,false,true]);await play();
  checks.push('M07 third generated-equivalent audio loads without full selection and takes exclusive ownership when dragged');
  await page.screenshot({path:path.join(out,'m07-selected.png')});
  await page.goto(url+'selection-fixture.html?module=m08');await page.locator('.m08-page').waitFor();
  const m08empty=await geometry(waves().first());const curveEmpty=await geometry(page.locator('.m08-curve'));
  await page.locator('.m08-page select').first().selectOption('source');await page.waitForFunction(()=>fixture.getStates()[0].asset&&document.querySelector('.m08-curve')?.tagName==='svg');
  const m08live=await geometry(waves().first()),curveLive=await geometry(page.locator('.m08-curve'));assert(Math.abs(m08live.height-m08empty.height)<2);assert(Math.abs(curveLive.height-curveEmpty.height)<2);layouts.push({module:'M08',empty:m08empty,live:m08live,curveEmpty,curveLive});
  assert.equal(await page.evaluate(()=>fixture.getStates()[0].end),0);await page.getByRole('button',{name:'合成整段',exact:true}).click();await page.waitForFunction(()=>document.querySelectorAll('.wave-line').length===2);
  assert.deepEqual((await selection()).map(e=>e.rect>0),[false,false]);
  await drag(0,.1,.6);await play();await page.waitForFunction(()=>fixture.playback.playing);const originalStart=await page.evaluate(()=>starts.at(-1));assert.equal(await page.evaluate(()=>fixture.isCurrentAudio(fixture.getStates()[0].asset,0)),true);await play();
  await drag(1,.2,.7);await play();await page.waitForFunction(()=>fixture.playback.playing);const resultStart=await page.evaluate(()=>starts.at(-1));assert.notEqual(originalStart.sample,resultStart.sample);assert.deepEqual((await selection()).map(e=>e.rect>0),[false,true]);await play();
  await drag(0,.3,.8);await page.getByRole('button',{name:'播放选区',exact:true}).click();await page.waitForFunction(()=>fixture.playback.playing);assert.equal(await page.evaluate(()=>fixture.isCurrentAudio(fixture.getStates()[0].asset,0)),true);
  const ranges=await page.evaluate(()=>starts.map(x=>[x.offset,x.duration]));for(const [offset,duration]of ranges){assert(offset>0&&duration<2,JSON.stringify(ranges));}
  checks.push('M08 original/result/original gestures route keyboard and button to exact chosen ROI, independent of current synthesis view');
  await page.screenshot({path:path.join(out,'m08-selected.png')});
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success:true,checks,layouts,errors},null,2));console.log(JSON.stringify({out,checks,layouts}));
 }catch(e){await page.screenshot({path:path.join(out,'failed.png')});await fs.writeFile(path.join(out,'failed.json'),JSON.stringify({error:String(e),checks,layouts,errors},null,2));throw e;}
 finally{await browser.close();await server.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
