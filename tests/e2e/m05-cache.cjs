const path=require('node:path'),fs=require('node:fs/promises'),{pathToFileURL}=require('node:url'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..');
(async()=>{
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{entries:['tests/m05-probe.html']}});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true}),context=await browser.newContext(),page=await context.newPage(),report={scope:'Windows Chrome fresh profile / development HTTP cache; not production PWA or physical GPU evidence'};
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/m05-probe.html');
  report.before=await page.evaluate(()=>performance.getEntriesByType('resource').filter(r=>r.name.includes('mediapipe-0.10.14')).length);assert.equal(report.before,0);
  for(const key of ['cold','warm'])report[key]=await page.evaluate(async()=>{window.Engine=(await import('/src/modules/lip-extraction/inference.ts')).LipInference;window.engine?.close();window.engine=new window.Engine();return window.engine.initialize('CPU')});
  await context.setOffline(true);
  report.initialized_worker_offline=await page.evaluate(async()=>{const canvas=new OffscreenCanvas(64,64);canvas.getContext('2d').fillRect(0,0,64,64);const image=await createImageBitmap(canvas);return window.engine.detect(image,0)});
  assert.equal(report.initialized_worker_offline.time_ms,0);
  report.new_worker_offline=await page.evaluate(async()=>{const e=new window.Engine();try{return {success:true,support:await e.initialize('CPU')}}catch(error){return {success:false,error:String(error)}}finally{e.close()}});
  report.storage=await page.evaluate(()=>navigator.storage.estimate());
 }finally{await fs.writeFile(path.join(root,'output/validation/m05/cache-report.json'),JSON.stringify(report,null,2));await browser.close();await server.close();}
 console.log(JSON.stringify(report));
})().catch(e=>{console.error(e);process.exitCode=1});
