/* Actual Chrome, owned Vite middleware serving only the authorized public fixture tree. */
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..'),out=path.join(root,'output/validation/m05');
(async()=>{
 const {createServer}=await import(require('node:url').pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')).href);
 const manifest=JSON.parse(await fs.readFile(path.join(out,'inputs/manifest.json'),'utf8'));
 const allowed=new Map(manifest.cases.flatMap(c=>c.frames.map(f=>[`/${c.name}/${f.file}`,path.join(out,'inputs',c.name,f.file)])));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0},optimizeDeps:{entries:['tests/m05-probe.html']},plugins:[{name:'m05-public-fixtures',configureServer(s){s.middlewares.use('/__m05fixture',async(req,res)=>{const file=allowed.get(req.url);if(!file){res.statusCode=404;res.end();return;}res.setHeader('Content-Type','image/png');res.end(await fs.readFile(file));});}}]});
 await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
 const page=await browser.newPage(),requests=[],errors=[],report={platform:'Windows Chrome',probes:[],requests,errors};
 page.on('request',r=>requests.push(r.url()));page.on('pageerror',e=>errors.push(e.message));
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/m05-probe.html');await page.waitForFunction(()=>typeof window.m05Probe==='function');
  report.lazy_before=requests.filter(x=>x.includes('mediapipe-0.10.14'));assert.equal(report.lazy_before.length,0);
  for(const delegate of ['CPU','GPU'])for(const mode of ['IMAGE','VIDEO'])for(const c of manifest.cases){
   try{const result=await page.evaluate(async({frames,delegate,mode})=>window.m05Probe(frames,delegate,mode),{frames:c.frames.map(f=>({url:`/__m05fixture/${c.name}/${f.file}`,time_s:f.time_s})),delegate,mode});report.probes.push({case:c.name,delegate,mode,success:true,...result});}
   catch(error){report.probes.push({case:c.name,delegate,mode,success:false,error:String(error)});}
  }
  report.external_requests=requests.filter(x=>!x.startsWith(server.resolvedUrls.local[0]));
  assert.equal(report.external_requests.length,0);assert.equal(errors.length,0);
  await page.screenshot({path:path.join(out,'chrome-feasibility.png')});
 }finally{await fs.writeFile(path.join(out,'chrome-feasibility.json'),JSON.stringify(report));await browser.close();await server.close();}
 console.log(JSON.stringify(report.probes.map(({case:c,delegate,mode,success,error,results,support,heartbeat_max_ms})=>({case:c,delegate,mode,success,error,frames:results?.length,detected:results?.filter(x=>x.points).length,init_ms:support?.init_ms,heartbeat_max_ms})),null,2));
})().catch(e=>{console.error(e);process.exitCode=1;});
