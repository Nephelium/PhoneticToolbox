/* Actual GeoMap/OpenLayers + actual M19 SVG/PNG export in owned Chrome.
 * Geometry/port are synthetic; no geographic-completeness or science claim.
 */
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {createHash,randomUUID}=require('node:crypto'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..'),frontend=path.join(root,'frontend'),sha=bytes=>createHash('sha256').update(bytes).digest('hex');

async function pixels(count){
 const exports=window.__paintExports,svg=await exports.at(-2).blob.text(),png=exports.at(-1).blob;
 const xml=new DOMParser().parseFromString(svg,'image/svg+xml');if(xml.querySelector('parsererror'))throw Error('Invalid exported SVG');
 const rootSvg=xml.documentElement,panels=count===1?[rootSvg]:Array.from(rootSvg.children).filter(child=>child.localName==='svg');
 const url=URL.createObjectURL(png),image=new Image();image.src=url;await image.decode();
 const raster=document.createElement('canvas');raster.width=image.naturalWidth;raster.height=image.naturalHeight;const pngContext=raster.getContext('2d');pngContext.drawImage(image,0,0);URL.revokeObjectURL(url);
 const ratio=image.naturalWidth/Number(rootSvg.getAttribute('width')),report=[];
 const box=d=>{const values=(d.match(/[-+]?(?:\d*\.?\d+)(?:e[-+]?\d+)?/gi)||[]).map(Number),xs=values.filter((_,i)=>i%2===0),ys=values.filter((_,i)=>i%2);return {x:(Math.min(...xs)+Math.max(...xs))/2,y:(Math.min(...ys)+Math.max(...ys))/2,minX:Math.min(...xs),maxX:Math.max(...xs)};};
 const same=(a,b)=>a.length===4&&a.every((n,i)=>Math.abs(n-b[i])<=1);
 const require=(condition,message)=>{if(!condition)throw Error(message);};
 const sea=[229,239,242,255],ordinary=[248,248,241,255],symbol=[82,108,120,255];
 for(let i=0;i<panels.length;i++){
  const panel=panels[i],paths=Array.from(panel.querySelector('g[clip-path]').querySelectorAll('path'));
  require(paths.length===9,`Panel ${i}: expected four active admin faces and five context paths, got ${paths.length}`);
  const raw=paths.map((p,order)=>({order,d:p.getAttribute('d'),fill:p.getAttribute('fill'),stroke:p.getAttribute('stroke'),dash:p.getAttribute('stroke-dasharray')}));
  const faces=raw.filter(p=>p.fill==='#f8f8f1'),open=raw.filter(p=>p.fill==='none'&&p.stroke==='#74878c');
  require(faces.length===4&&open.length===2,`Panel ${i}: active administrative faces or collection line missing`);
  require(!raw.some(p=>p.fill==='#edf0ed'||p.stroke==='#bcc7c9'),`Panel ${i}: obsolete world land remains`);
  const symbols=raw.find(p=>p.fill==='#526c78'),disputed=raw.find(p=>p.dash==='4 3'),legacy=raw.find(p=>p.fill==='none'&&p.stroke==='#526c78'&&!p.dash);
  require(symbols&&disputed&&legacy,`Panel ${i}: independent context missing`);
  const attrs=[...faces,...open,symbols,disputed,legacy],areaOrders=faces.map(p=>p.order),lineOrders=[...open,disputed,legacy].map(p=>p.order);
  require(Math.max(...areaOrders)<Math.min(...lineOrders)&&Math.max(...lineOrders)<symbols.order,`Panel ${i}: active land/line/symbol painter priority differs`);
  const pieces=symbols.d.match(/M[^M]+Z/g)||[];require(pieces.length===10,`Panel ${i}: ten symbol components were not preserved`);
  const target=document.querySelectorAll('.map-target')[i],w=target.clientWidth,h=target.clientHeight,scale=w/1200;
  const canvas=document.createElement('canvas');canvas.width=w;canvas.height=h;const ctx=canvas.getContext('2d');ctx.fillStyle='#e5eff2';ctx.fillRect(0,0,w,h);
  for(const layer of target.querySelectorAll('.ol-layer canvas')){const transform=layer.style.transform.match(/^matrix\(([^)]+)\)$/);ctx.setTransform(...(transform?transform[1].split(',').map(Number):[1,0,0,1,0,0]));ctx.globalAlpha=Number(layer.parentElement.style.opacity||1);ctx.drawImage(layer,0,0);}ctx.setTransform(1,0,0,1,0,0);ctx.globalAlpha=1;
  const px=(context,x,y)=>Array.from(context.getImageData(Math.round(x),Math.round(y),1,1).data);
  const offset=[Number(panel.getAttribute('x')||0),Number(panel.getAttribute('y')||0)];
  const sample=p=>({canvas:px(ctx,p.x*scale,p.y*scale),png:px(pngContext,(offset[0]+p.x)*ratio,(offset[1]+p.y)*ratio)});
  const samples=[];
  // The thin overlap face is crossed by the disputed line; sample the other
  // three opaque faces and both open-line triangle interiors independently.
  const height=p=>{const ys=(p.d.match(/-?[\d.]+/g)||[]).map(Number).filter((_,i)=>i%2);return Math.max(...ys)-Math.min(...ys);};
  const smallest=Math.min(...faces.map(height));
  for(const p of [...faces.filter(p=>height(p)>smallest+1),...open]){const result=sample(box(p.d)),color=p.fill==='none'?sea:ordinary;require(same(result.canvas,color)&&same(result.png,color),`Panel ${i} path ${p.order}: Canvas ${result.canvas} / PNG ${result.png}, expected ${color}`);samples.push({path:p.order,...result});}
  for(const [n,piece] of pieces.entries()){const result=sample(box(piece));require(same(result.canvas,symbol)&&same(result.png,symbol),`Panel ${i} symbol ${n}: Canvas ${result.canvas} / PNG ${result.png}`);}
  const a=box(pieces[0]),b=box(pieces[1]),gap=sample({x:(a.maxX+b.minX)/2,y:a.y});require(same(gap.canvas,sea)&&same(gap.png,sea),`Panel ${i}: area appeared between boundary symbols`);
  const ink=(context,x,y)=>{for(let dy=-2;dy<=2;dy++){const p=px(context,x,y+dy);if(p.slice(0,3).reduce((sum,n,j)=>sum+Math.abs(n-sea[j]),0)>45)return true;}return false;};
  const lineCoverage=(index,context,factor,xOffset=0,yOffset=0)=>{const b=box(attrs[index].d),bits=[];for(let x=Math.ceil((b.minX+xOffset)*factor)+2;x<Math.floor((b.maxX+xOffset)*factor)-2;x++)bits.push(ink(context,x,(b.y+yOffset)*factor));return {ink:bits.filter(Boolean).length,total:bits.length,changes:bits.slice(1).filter((v,j)=>v!==bits[j]).length};};
  const disputedCanvas=lineCoverage(7,ctx,scale),disputedPng=lineCoverage(7,pngContext,ratio,offset[0],offset[1]);
  for(const result of [disputedCanvas,disputedPng])require(result.ink>0&&result.ink<result.total&&result.changes>=3,`Panel ${i}: disputed line lacks visible separated dashes`);
  const legacyPieces=attrs[8].d.match(/M[^M]+/g)||[];require(legacyPieces.length===3,`Panel ${i}: legacy separated segments were joined`);
  for(const piece of legacyPieces){const b=box(piece),bits=[];for(let x=Math.ceil(b.minX*scale)+2;x<Math.floor(b.maxX*scale)-2;x++)bits.push(ink(ctx,x,b.y*scale));require(bits.length>2&&bits.filter(Boolean).length/bits.length>=.9,`Panel ${i}: legacy solid segment received invented dashes`);}
  report.push({panel:i,paths:attrs.map(({d,...rest})=>({...rest,subpaths:(d.match(/M/g)||[]).length})),painterPriority:{areaOrders,lineOrders,symbolOrder:symbols.order,disputeOverOpaqueFace:true},samples,symbolComponents:pieces.length,symbolGap:gap,disputedCanvas,disputedPng});
 }
 const ids=Array.from(xml.querySelectorAll('[id]'),el=>el.id);require(new Set(ids).size===ids.length,'Multi-map SVG clip IDs collide');
 require(image.naturalWidth===Number(rootSvg.getAttribute('width'))*2&&image.naturalHeight===Number(rootSvg.getAttribute('height'))*2,'Actual PNG export changed dimensions relative to its SVG');
 return {panels:report,svgSize:[Number(rootSvg.getAttribute('width')),Number(rootSvg.getAttribute('height'))],pngSize:[image.naturalWidth,image.naturalHeight],clipIds:ids};
}

(async()=>{
 const [requestedOut,requestedScratch]=process.argv.slice(2);
 const base=path.join(root,'output/validation/geolinguistics-paint');await fs.mkdir(base,{recursive:true});const out=requestedOut||path.join(base,'browser-'+randomUUID().replaceAll('-','')),scratch=requestedScratch||path.join(out,'scratch');await fs.mkdir(scratch,{recursive:true});
 const report={success:false,checks:[],errors:[],cases:[],sourceIdentity:{},limitations:['Synthetic geometry and test-only data/save port; actual Page, GeoMap/OpenLayers, SVG composition, font embedding and PNG export run unchanged. No geographical-completeness, official-data rendering or final-package claim.']};let server,context,page,stage='build';
 try{
  const vite=await import(pathToFileURL(path.join(frontend,'node_modules/vite/dist/node/index.js'))),vue=(await import(pathToFileURL(path.join(frontend,'node_modules/@vitejs/plugin-vue/dist/index.mjs')))).default,{chromium}=require(process.env.PTB_PLAYWRIGHT_PATH||path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  for(const name of ['frontend/src/modules/geolinguistics/GeolinguisticsPage.vue','frontend/src/modules/geolinguistics/GeoMap.vue','frontend/src/modules/geolinguistics/map-paint.ts','frontend/src/modules/geolinguistics/export.ts','frontend/src/modules/geolinguistics/attribution.ts','frontend/tests/fixtures/geolinguistics-paint-port.ts'])report.sourceIdentity[name]=sha(await fs.readFile(path.join(root,name)));
  const dist=path.join(scratch,'dist');await vite.build({configFile:false,root:frontend,base:'./',publicDir:false,cacheDir:path.join(scratch,'vite-cache'),logLevel:'error',plugins:[vue(),{name:'owned-paint-fixture-port',enforce:'pre',resolveId(source,importer){if(source==='../../platform/geolinguistics.ts'&&importer?.split('?')[0].endsWith('GeolinguisticsPage.vue'))return path.join(frontend,'tests/fixtures/geolinguistics-paint-port.ts');}}],build:{outDir:dist,target:'es2022',rolldownOptions:{input:path.join(frontend,'tests/geolinguistics-paint.html')}}});
  server=await vite.preview({configFile:false,root:frontend,build:{outDir:dist},preview:{host:'127.0.0.1',port:0},logLevel:'error'});context=await chromium.launchPersistentContext(path.join(scratch,'profile'),{executablePath:process.env.PTB_CHROME_PATH||'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1600,height:1100},deviceScaleFactor:1});
  page=await context.newPage();page.setDefaultTimeout(20000);page.on('pageerror',error=>report.errors.push({stage,message:error.message}));await page.goto(server.resolvedUrls.local[0]+'tests/geolinguistics-paint.html');await page.locator('.atlas-workbench').waitFor();await page.getByRole('tab',{name:'语言现象',exact:true}).click();await page.getByRole('button',{name:'收起表格',exact:true}).click();
  // Let the real export status row appear before comparing subsequent SVG/PNG
  // dimensions; its first insertion legitimately resizes the map viewport.
  await page.locator('.geography-toolbar').getByRole('button',{name:'SVG',exact:true}).click();await page.getByRole('dialog').getByRole('button',{name:'导出图片',exact:true}).click();await page.waitForFunction(()=>window.__paintExports.length===1);await page.waitForTimeout(300);
  for(const count of [2,4])for(const theme of ['light','dark']){
   stage=`${count}-maps-${theme}`;await page.locator('.map-heading select').selectOption(String(count));await page.getByLabel('省级地区',{exact:true}).selectOption('province:fixture');await page.waitForTimeout(500);await page.waitForFunction(()=>document.querySelectorAll('.map-loading').length===0);assert.equal(await page.locator('.geo-map').count(),count);await page.evaluate(theme=>document.documentElement.dataset.theme=theme,theme);await page.waitForTimeout(250);
   for(const format of ['SVG','PNG']){const n=await page.evaluate(()=>window.__paintExports.length);await page.locator('.geography-toolbar').getByRole('button',{name:format,exact:true}).click();await page.getByRole('dialog').getByRole('button',{name:'导出图片',exact:true}).click();await page.waitForFunction(n=>window.__paintExports.length>n,n);}
   const exportedFiles=await page.evaluate(async()=>Promise.all(window.__paintExports.slice(-2).map(async({name,blob})=>({name,type:blob.type,data:await new Promise(resolve=>{const reader=new FileReader();reader.onload=()=>resolve(String(reader.result).split(',')[1]);reader.readAsDataURL(blob);})}))));
   const files=[];for(const file of exportedFiles){const bytes=Buffer.from(file.data,'base64'),name=stage+(file.type==='image/png'?'.png':'.svg');await fs.writeFile(path.join(out,name),bytes);files.push({name,bytes:bytes.length,sha256:sha(bytes)});}const result=await page.evaluate(pixels,count);await page.screenshot({path:path.join(out,stage+'-canvas.png')});report.cases.push({count,theme,...result,files});report.checks.push(stage+': actual Canvas and actual SVG→PNG preserve sea, polygon/collection fills, ten distinct dark symbols, disputed dashes and solid legacy segments.');
  }
  assert.deepEqual(report.errors,[]);for(const [name,expected] of Object.entries(report.sourceIdentity))assert.equal(sha(await fs.readFile(path.join(root,name))),expected,'Input changed during paint acceptance: '+name);report.success=true;
 }catch(error){report.failure={stage,message:error.message,stack:error.stack};process.exitCode=1;if(page)await page.screenshot({path:path.join(out,'failure.png')}).catch(()=>{});}
 finally{if(context)await context.close();if(server)await new Promise(resolve=>server.httpServer.close(resolve));report.cleanup={state:'pending',path:scratch,ownedProcessesStopped:true};await fs.writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({out,success:report.success,cases:report.cases.length,failure:report.failure,cleanup:report.cleanup}));}
})().catch(error=>{console.error(error);process.exitCode=1;});
