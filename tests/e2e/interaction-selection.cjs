// P19-R6: real native Chrome selection with actual chart components; no user data.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const out=path.join(root,'output/validation/interaction-selection','chrome-'+Date.now());await fs.mkdir(out,{recursive:true});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const plugin={name:'interaction-selection-fixture',configureServer(server){server.middlewares.use(async(req,res,next)=>{
  if(!req.url?.startsWith('/interaction-fixture.html'))return next();
  res.setHeader('content-type','text/html; charset=utf-8');res.end(await server.transformIndexHtml('/interaction-fixture.html',`<!doctype html><html><head><link rel="stylesheet" href="/interaction-selection.css"><script src="/interaction-selection.js" defer></script></head><body><div id="fixture"></div><script type="module">
import {createApp,h,reactive,markRaw} from 'vue';import '/src/design/tokens.css';
import PitchCurve from '/src/modules/pitch-manipulation/PitchCurve.vue';import Waveform from '/src/components/WaveformViewport.vue';import Plot from '/src/components/ScientificPlot.vue';
const data=Float32Array.from({length:16000},(_,i)=>.2*Math.sin(2*Math.PI*220*i/8000));
const wave=reactive({asset:markRaw({name:'public-sine.wav',sampleRate:8000,channels:[data],duration:2,frames:16000}),start:0,end:0,channel:0,parameters:[],dirty:false,error:'',loading:false,zoom:1,offset:0});
const state=reactive({curve:[150,150,150,150,150],pan:0,plotPan:0,clicks:0,canvasMoves:0});window.fixture={state,wave};
const text=(id)=>h('p',{id,style:'font:16px monospace;width:max-content;max-width:100%;padding:4px;margin:10px 0'},'Ordinary text remains selectable by dragging here.');
createApp({render:()=>h('main',{style:'width:700px;margin:16px'},[
 text('above'),h(PitchCurve,{times:[0,.5,1,1.5,2],original:[150,150,150,150,150],modified:state.curve,start:0,end:2,min:50,max:300,references:[],editable:true,onEdit:v=>state.curve=v,onPan:v=>state.pan+=v}),text('below'),
 h('div',{style:'display:flex;gap:8px'},[h('button',{id:'action',onClick:()=>state.clicks++},[h('span','Action button'),h('svg',{viewBox:'0 0 20 20'},[h('text',{x:0,y:15},'B')])]),h('button',{id:'second'},'Second button'),h('button',{disabled:true,id:'disabled'},'Disabled button')]),
 h('details',{id:'details'},[h('summary',{id:'summary'},'Toggle details'),h('p','Details text')]),
 h('input',{id:'input',value:'Editable text remains selectable'}),h('textarea',{id:'textarea'},'Textarea text remains selectable'),h('div',{id:'editable',contenteditable:'true'},'Contenteditable text remains selectable'),h('input',{id:'native-range',type:'range',min:0,max:100,value:10}),h('input',{id:'native-check',type:'checkbox'}),h('div',{id:'empty-plot',role:'img',style:'height:70px'},'Empty plot placeholder'),
 h(Waveform,{state:wave}),h(Plot,{title:'Public synthetic plot',x:[0,2],y:[0,200],height:180,interactive:true,livePan:true,traces:[{label:'F0',times:[0,1,2],values:[120,150,130],color:'#a33'}],onPan:v=>state.plotPan+=v}),
 h('canvas',{id:'canvas',width:600,height:120,style:'display:block;border:1px solid grey',onPointermove:e=>{if(e.buttons===1)state.canvasMoves++}}),text('bottom')
 ])}).mount('#fixture');
</script></body></html>`));
 });}};
 const server=await createServer({root:path.join(root,'frontend'),plugins:[plugin],server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{noDiscovery:true,include:['vue']}});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--mute-audio']});
 const page=await browser.newPage({viewport:{width:1000,height:1500}}),checks=[],errors=[];page.on('pageerror',e=>errors.push(e.message));
 const url=server.resolvedUrls.local[0];
 const noSelection=async()=>assert.equal(await page.evaluate(()=>getSelection().toString()),'');
 async function textDrag(id){const loc=page.locator(id);await loc.scrollIntoViewIfNeeded();const b=await loc.boundingBox();await page.mouse.move(b.x+6,b.y+b.height/2);await page.mouse.down();await page.mouse.move(b.x+b.width-8,b.y+b.height/2,{steps:10});await page.mouse.up();assert((await page.evaluate(()=>getSelection().toString())).includes('text remains selectable'));}
 async function dragChart(selector,modifier,leave=false){const loc=page.locator(selector);await loc.scrollIntoViewIfNeeded();const b=await loc.boundingBox();await page.evaluate(()=>{const r=document.createRange();r.selectNodeContents(document.querySelector('#above'));getSelection().removeAllRanges();getSelection().addRange(r);});
  if(modifier)await page.keyboard.down(modifier);await page.mouse.move(b.x+b.width*.3,b.y+b.height*.4);await page.mouse.down();assert.equal(await page.evaluate(()=>document.documentElement.hasAttribute('data-ptb-selection-lock')),true);
  await page.mouse.move(b.x+b.width*.6,b.y+b.height*.55,{steps:9});if(leave)await page.mouse.move(b.x+b.width*.6,b.y+b.height+35,{steps:5});await noSelection();await page.mouse.up();if(modifier)await page.keyboard.up(modifier);await noSelection();assert.equal(await page.evaluate(()=>document.documentElement.hasAttribute('data-ptb-selection-lock')),false);
 }
 try{
  await page.goto(url+'interaction-fixture.html');await page.locator('.m08-curve').waitFor();
  await textDrag('#above');await dragChart('.m08-curve','Shift',true);assert((await page.evaluate(()=>fixture.state.curve)).some(v=>v!==150));
  await dragChart('.m08-curve','Control');assert.deepEqual(await page.evaluate(()=>fixture.state.curve),[150,150,150,150,150]);await dragChart('.m08-curve');assert.notEqual(await page.evaluate(()=>fixture.state.pan),0);
  await textDrag('#below');checks.push('Actual M08 Shift draw, Ctrl restore and ordinary pan still work; old text ranges and leaving the plot cannot select adjacent text');
  const button=page.locator('#action');await button.click();await button.dblclick();await noSelection();assert.equal(await page.evaluate(()=>fixture.state.clicks),3);
  await button.focus();await page.keyboard.press('Enter');await page.keyboard.press('Space');assert.equal(await page.evaluate(()=>fixture.state.clicks),5);
  await page.locator('#disabled').click({force:true});await noSelection();await page.locator('#summary').click();assert.equal(await page.locator('#details').evaluate(e=>e.open),true);await page.locator('#summary').dblclick();await noSelection();
  await dragChart('#action',null,true);await textDrag('#below');checks.push('Buttons, nested icons, disabled buttons and summary single/double-click do not select text; clicks, keyboard activation and details toggling are retained');
  await dragChart('.wave-track>svg',null,true);assert(await page.evaluate(()=>fixture.wave.end>fixture.wave.start));assert.equal(await page.evaluate(()=>document.activeElement.tagName.toLowerCase()),'svg');
  await dragChart('.scientific-plot>svg','Shift',true);assert.notEqual(await page.evaluate(()=>fixture.state.plotPan),0);await dragChart('#canvas','Shift',true);assert((await page.evaluate(()=>fixture.state.canvasMoves))>0);await textDrag('#bottom');
  checks.push('Shared waveform, scientific plot and canvas drags are protected, with working selection, focus and pan; text selection resumes immediately after release');
  for(const id of ['#input','#textarea']){await page.locator(id).focus();await page.keyboard.press('Control+A');assert((await page.locator(id).evaluate(e=>e.selectionEnd-e.selectionStart))>5);}
  await page.locator('#editable').focus();await page.keyboard.press('Control+A');assert((await page.evaluate(()=>getSelection().toString())).includes('Contenteditable'));
  await page.locator('#input').fill('A user can still type');assert.equal(await page.locator('#input').inputValue(),'A user can still type');
  const rangeBox=await page.locator('#native-range').boundingBox();await page.mouse.move(rangeBox.x+rangeBox.width*.2,rangeBox.y+rangeBox.height/2);await page.mouse.down();await page.mouse.move(rangeBox.x+rangeBox.width*.8,rangeBox.y+rangeBox.height/2,{steps:8});await page.mouse.up();assert(+await page.locator('#native-range').inputValue()>65);await page.locator('#native-check').check();assert.equal(await page.locator('#native-check').isChecked(),true);await dragChart('#empty-plot','Shift',true);
  await page.locator('.m08-curve').scrollIntoViewIfNeeded();const interrupted=await page.locator('.m08-curve').boundingBox();await page.mouse.move(interrupted.x+100,interrupted.y+80);await page.mouse.down();assert.equal(await page.evaluate(()=>document.documentElement.hasAttribute('data-ptb-selection-lock')),true);await page.evaluate(()=>window.dispatchEvent(new Event('blur')));assert.equal(await page.evaluate(()=>document.documentElement.hasAttribute('data-ptb-selection-lock')),false);await page.mouse.up();
  checks.push('Inputs, textarea and contenteditable retain selection/editing, native range/checkbox work, empty graph is protected, and interrupted gestures release the lock');
  const vocal=await (await page.request.get(url+'vocal-tract/index.html')).text();assert(vocal.includes('../interaction-selection.css')&&vocal.includes('../interaction-selection.js'));
  const frameHtml='<html><head><link rel="stylesheet" href="/interaction-selection.css"><script src="/interaction-selection.js" defer></script></head><body><p id="text">Ordinary text remains selectable by dragging here.</p><button id="button">Vocal tract button</button><canvas id="plot" width="500" height="150" style="border:1px solid grey"></canvas></body></html>';
  await page.route('**/interaction-frame.html',route=>route.fulfill({contentType:'text/html',body:frameHtml}));await page.evaluate(()=>{const f=document.createElement('iframe');f.src='/interaction-frame.html';f.style='width:650px;height:300px';document.body.prepend(f);});
  const frame=page.frameLocator('iframe');await frame.locator('#button').dblclick();assert.equal(await frame.locator('body').evaluate(()=>getSelection().toString()),'');
  const cb=await frame.locator('#plot').boundingBox();await page.mouse.move(cb.x+100,cb.y+50);await page.keyboard.down('Shift');await page.mouse.down();await page.mouse.move(cb.x+300,cb.y+190,{steps:8});await page.mouse.up();await page.keyboard.up('Shift');assert.equal(await frame.locator('body').evaluate(()=>getSelection().toString()),'');
  const tb=await frame.locator('#text').boundingBox();await page.mouse.move(tb.x+2,tb.y+8);await page.mouse.down();await page.mouse.move(tb.x+260,tb.y+8,{steps:8});await page.mouse.up();assert((await frame.locator('body').evaluate(()=>getSelection().toString())).includes('text remains selectable'));
  checks.push('M10 standalone entry loads the same resources; iframe canvas/buttons are protected while its ordinary text remains selectable');
  await page.evaluate(()=>{document.documentElement.dataset.theme='dark';document.documentElement.style.zoom='1.5';});await dragChart('.m08-curve','Shift',true);await page.locator('#action').dblclick();await noSelection();await textDrag('#below');await page.screenshot({path:path.join(out,'dark-150.png')});
  checks.push('Dark theme and 150% page zoom retain protected chart/button gestures and ordinary text dragging');
  await page.evaluate(()=>{document.documentElement.dataset.theme='light';document.documentElement.style.zoom='1';});await page.screenshot({path:path.join(out,'selection.png')});assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success:true,checks,errors},null,2));console.log(JSON.stringify({out,checks}));
 }catch(e){await page.screenshot({path:path.join(out,'failed.png')});await fs.writeFile(path.join(out,'failed.json'),JSON.stringify({error:String(e),checks,errors},null,2));throw e;}
 finally{await browser.close();await server.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
