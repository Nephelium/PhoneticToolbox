// Actual Vue/native controls in owned Chrome, synthetic values only.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const out=path.join(root,'output/validation/time-precision','chrome-'+Date.now());await fs.mkdir(out,{recursive:true});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const native=(await fs.readFile(path.join(root,'frontend/public/vocal-tract/index.html'),'utf8')).replace(/<script\b[^>]*>[\s\S]*?<\/script>/g,'');
 const fixturePlugin={name:'time-input-fixtures',configureServer(server){server.middlewares.use(async(req,res,next)=>{
  const name=req.url?.split('?')[0];
  if(name==='/time-input-fixture.html'){res.setHeader('content-type','text/html; charset=utf-8');res.end(await server.transformIndexHtml(name,`<!doctype html><html><body><div id="fixture"></div><script type="module">
import {createApp,h,reactive,withDirectives,vModelText} from 'vue';
import {vTimePrecision} from '/src/design/time-precision.ts';
import AudioTransport from '/src/components/AudioTransport.vue';
import EggControls from '/src/modules/egg-analysis/EggControls.vue';
import EggParameters from '/src/modules/egg-analysis/EggParameters.vue';
import SettingsDrawer from '/src/components/SettingsDrawer.vue';
import {defaults as eggDefaults} from '/src/modules/egg-analysis/state.ts';
import {defaults as settingsDefaults} from '/src/modules/parameter-estimation/state.ts';
const wave=reactive({asset:{duration:50,sampleRate:44100,channels:[new Float32Array(1)]},start:14.985544858,end:16.364117232,channel:0,parameters:[],dirty:false,error:'',loading:false,zoom:1,offset:0});
const config=reactive({...eggDefaults(),micro_width_ms:156.92141,spec_window_ms:20.234567});
const state=reactive({seconds:'1.23456789',frequency:'200.123456789',unit:'s',showSettings:true}),settings=reactive({...settingsDefaults(),energy_window_ms:20.1234567});
window.fixture={wave,config,state,settings};
createApp({render(){return h('main',[
h(AudioTransport,{state:wave,active:false}),
h(EggControls,{modelValue:config,start:wave.start,end:wave.end,duration:50,onRange:(a,b)=>{wave.start=a;wave.end=b;}}),
h(EggParameters,{modelValue:config}),
withDirectives(h('input',{'aria-label':'Text time','onUpdate:modelValue':v=>state.seconds=v}),[[vTimePrecision,state.unit],[vModelText,state.seconds]]),
withDirectives(h('input',{'aria-label':'Frequency','onUpdate:modelValue':v=>state.frequency=v}),[[vTimePrecision,undefined],[vModelText,state.frequency]]),
state.showSettings?h(SettingsDrawer,{draft:settings,onClose:()=>state.showSettings=false,onDraft:v=>Object.assign(settings,v)}):null]);}}).mount('#fixture');
</script></body></html>`));}
  else if(name==='/time-keyframes.html'){res.setHeader('content-type','text/html; charset=utf-8');res.end(native);}else next();
 });}};
 const server=await createServer({root:path.join(root,'frontend'),plugins:[fixturePlugin],server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{noDiscovery:true,include:['vue']}});
 await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
 const page=await browser.newPage({viewport:{width:1440,height:1000}}),checks=[],errors=[];page.on('pageerror',e=>errors.push(e.message));
 const url=server.resolvedUrls.local[0];
 try{
  await page.goto(url+'time-input-fixture.html');await page.waitForFunction(()=>!!window.fixture&&!!document.querySelector('[data-time-unit]'));
  const range=page.locator('.selection-controls input'),micro=page.getByLabel('EGG 微观窗口'),text=page.getByLabel('Text time');
  assert.deepEqual(await range.evaluateAll(es=>es.map(e=>e.value)),['14.98554','16.36412']);assert.equal(await micro.inputValue(),'156.92');assert.equal(await page.getByLabel('EGG 选区时长').inputValue(),'1.37857');assert.equal(await text.inputValue(),'1.23457');
  assert.equal(await page.getByLabel('能量窗口',{exact:true}).inputValue(),'20.12');assert.equal(await page.getByLabel('Frequency').inputValue(),'200.123456789');
  assert.deepEqual(await page.evaluate(()=>[fixture.wave.start,fixture.wave.end,fixture.config.micro_width_ms,fixture.state.seconds]),[14.985544858,16.364117232,156.92141,'1.23456789']);
  checks.push('shared transport/EGG/text/schema controls cap displayed decimals; source state retains exact values; Hz untouched');
  await page.getByRole('button',{name:'取消',exact:true}).click();
  await range.first().fill('14.123456789');assert.equal(await range.first().inputValue(),'14.12346');await range.first().blur();assert.equal(await page.evaluate(()=>fixture.wave.start),14.12346);
  await micro.fill('155.987654');await micro.blur();assert.equal(await page.evaluate(()=>fixture.config.micro_width_ms),155.99);
  await text.fill('1.987654321');await text.blur();assert.equal(await page.evaluate(()=>fixture.state.seconds),'1.98765');
  await text.fill('');assert.equal(await page.evaluate(()=>fixture.state.seconds),'');
  await text.pressSequentially('1.23000');assert.equal(await text.inputValue(),'1.23000');await text.blur();assert.equal(await text.inputValue(),'1.23');
  await page.getByLabel('Frequency').fill('300.987654321');assert.equal(await page.evaluate(()=>fixture.state.frequency),'300.987654321');
  checks.push('editing/paste/change limits number/string models; decimal typing and clearing work; unrelated numeric edits retain precision');
  await micro.focus();await page.evaluate(()=>{fixture.config.micro_width_ms=123.456789;fixture.wave.start=13.987654321;fixture.wave.end=15.543219876;});
  await page.waitForFunction(()=>document.querySelector('[aria-label="EGG 微观窗口"]').value==='123.46');assert.equal(await page.evaluate(()=>fixture.config.micro_width_ms),123.456789);
  await micro.blur();assert.equal(await page.evaluate(()=>fixture.config.micro_width_ms),123.456789);
  await page.evaluate(()=>{fixture.state.unit='ms';fixture.state.seconds='1.23456789';});await page.waitForFunction(()=>document.querySelector('[aria-label="Text time"]').value==='1.23');
  checks.push('programmatic drag-equivalent updates while focused and unit changes format without mutating exact state');
  await page.screenshot({path:path.join(out,'time-inputs.png')});
  await page.goto(url+'time-keyframes.html');
  await page.evaluate(async()=>{
   const {setupKeyframes}=await import('/vocal-tract/keyframes.js');
   const frames=[{name:'synthetic',duration:1.23456789,f0:150,lip_width:1,params:[],source:{mode:'voiced'}}];window.frames=frames;window.saved=[];
   setupKeyframes({initialFrames:frames,getPose:()=>frames[0],applyPose:()=>{},post:async(op,body)=>{window.saved.push({op,body});return new Response(JSON.stringify({}));},setBusy:()=>{},isReady:()=>true,isSilent:true,keepVowel:()=>true,viewer:{}});
   document.getElementById('panel-motion').hidden=false;
  });
  const duration=page.locator('#keyframeList input[data-time-unit="s"]');assert.equal(await duration.inputValue(),'1.23457');assert.equal(await page.evaluate(()=>frames[0].duration),1.23456789);
  await duration.fill('1.87654321');await duration.blur();assert.equal(await page.evaluate(()=>frames[0].duration),1.87654);
  checks.push('native M10 keyframe seconds preserve imported precision until explicit user edit');
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success:true,checks,errors},null,2));console.log(JSON.stringify({out,checks}));
 }catch(e){await page.screenshot({path:path.join(out,'failed.png')});await fs.writeFile(path.join(out,'failed.json'),JSON.stringify({error:String(e),checks,errors},null,2));throw e;}
 finally{await browser.close();await server.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
