// Independent actual M01/M12 page regression. Natural recordings only, owned bridges.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const {spawn}=require('node:child_process'),{createInterface}=require('node:readline'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..'),out=path.join(root,'output/validation/p17/shared-m04-r2',String(Date.now()));
const pause=page=>page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
async function bridge(module){
 const script=module==='M01'?'p17_m01_m05_bridge.py':'p17_m12_bridge.py';
 const worker=spawn(path.join(root,'.venv',module==='M01'?'m14':'m09-ui','Scripts/python.exe'),['-X','utf8','scripts/'+script],{cwd:root,windowsHide:true,env:{...process.env,PYTHONPATH:['backend/src','desktop/src','packages/phonetic_core/src'].map(p=>path.join(root,p)).join(';')}});
 const pending=new Map();let counter=0,resolveReady,rejectReady;const ready=new Promise((r,j)=>{resolveReady=r;rejectReady=j;});
 createInterface({input:worker.stdout}).on('line',line=>{const v=JSON.parse(line);if(v.ready)resolveReady(v);else{pending.get(v.id)?.(v);pending.delete(v.id);}});worker.stderr.on('data',d=>process.stderr.write(d));worker.on('error',rejectReady);worker.once('exit',c=>rejectReady(Error('bridge exited '+c)));
 const info=await ready,rpc=data=>new Promise(resolve=>{const id=++counter;pending.set(id,resolve);worker.stdin.write(JSON.stringify({...data,rpc_id:id})+'\n');});
 return {info,rpc,close:async()=>{const exited=new Promise(r=>worker.once('exit',r));if(module==='M01')worker.stdin.write(JSON.stringify({op:'shutdown',rpc_id:++counter})+'\n');else worker.stdin.end();await exited;}};
}
async function main(){
 await fs.mkdir(out,{recursive:true});const report={out,measurements:[],errors:[],sourceHashes:{},bridges:[]};
 for(const file of ['frontend/src/components/WaveformViewport.vue','frontend/src/components/SpectrogramViewport.vue','frontend/src/components/TextGridTimeline.vue','frontend/src/modules/annotation/AnnotationTracks.vue'])report.sourceHashes[file]=crypto.createHash('sha256').update(await fs.readFile(path.join(root,file))).digest('hex');
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--mute-audio']});
 try{for(const module of (process.env.P17_ALIGNMENT_FINAL?['M01']:['M01','M12'])){
  const b=await bridge(module);report.bridges.push({module,...b.info});let latestSpectrum;
  const endpoint=module==='M01'?'/__p17a':'/__m12',harness=module==='M01'?'p17-m01-m05.html':'p17-m12-live.html';
  const server=await createServer({root:path.join(root,'frontend'),cacheDir:path.join(out,'vite-'+module),server:{host:'127.0.0.1',port:0,hmr:false},optimizeDeps:{entries:['tests/'+harness]},plugins:[{name:'owned-alignment',configureServer(s){s.middlewares.use(endpoint,(req,res)=>{let raw='';req.on('data',d=>raw+=d);req.on('end',async()=>{const body=JSON.parse(raw),value=await b.rpc(body);if(body.op==='spectrogram'&&value.value)latestSpectrum={request:body.view,data:value.value};res.setHeader('Content-Type','application/json');res.end(JSON.stringify(value));});});}}]});await server.listen();
  const page=await browser.newPage({viewport:{width:1920,height:1000}});page.on('pageerror',e=>report.errors.push({module,message:e.message}));
  try{
   await page.goto(server.resolvedUrls.local[0]+'tests/'+harness);
   let interval;
   if(module==='M01'){
    await page.getByRole('button',{name:'参数估计',exact:true}).click();await page.getByRole('button',{name:'选择音频目录',exact:true}).click();await page.locator('.m01-file-list .file-row').filter({hasText:b.info.short}).click();await page.locator('.textgrid-interval').first().waitFor();await page.getByLabel('显示语谱图（Praat）').check();await page.locator('.spectrogram-canvas canvas').waitFor({timeout:90000});
    interval=await page.locator('.textgrid-tier').first().locator('.textgrid-interval').evaluateAll(nodes=>{const n=nodes.find(n=>Number(n.dataset.xmin)>0&&n.getBoundingClientRect().width>25)||nodes[0];return {xmin:Number(n.dataset.xmin),xmax:Number(n.dataset.xmax)};});
   }else{
    await page.getByRole('button',{name:'TextGrid标注',exact:true}).click();await page.getByRole('button',{name:'选择语料文件夹',exact:true}).click();await page.locator('.annotation-file-list button').filter({hasText:b.info.audio}).click();await page.waitForFunction(()=>document.querySelector('.annotation-page')?.getAttribute('aria-busy')==='false'&&document.querySelector('.annotation-grid'));
    const tiers=b.info.document.tiers.filter(t=>t.intervals);await page.getByLabel('词层名',{exact:true}).selectOption(tiers[0].name);await page.getByLabel('音素层名',{exact:true}).selectOption(tiers[1].name);interval=tiers[0].intervals.find(i=>i.xmin>0&&i.text&&i.xmax-i.xmin>.05)||tiers[0].intervals[1];await page.getByLabel('标注可视时长').fill(String(b.info.document.xmax));await page.getByLabel('标注可视时长').press('Tab');
   }
   for(const viewport of (process.env.P17_ALIGNMENT_FINAL?[{width:2560,height:1360}]:[{width:1920,height:1000},{width:2560,height:1360}]))for(const fontSize of (process.env.P17_ALIGNMENT_FINAL?[12]:[12,24])){
    await page.setViewportSize(viewport);await page.evaluate(async size=>{const f=await import('/src/state/fonts.ts');await f.setFonts({...f.preferences.value,figure:{...f.preferences.value.figure,size}},false);},fontSize);await pause(page);await page.waitForTimeout(250);
    if(module==='M01')await page.locator('.textgrid-tier').first().locator(`.textgrid-interval[data-xmin="${interval.xmin}"]`).click();
    else{const box=await page.locator('.annotation-grid').boundingBox();await page.mouse.click(box.x+box.width*(interval.xmin+interval.xmax)/2/b.info.document.xmax,box.y+box.height*.22);}
    await pause(page);if(module==='M01')await page.locator('.spectrogram-canvas canvas').waitFor({timeout:90000});
    const m=await page.evaluate(({module,interval})=>{
     const rect=e=>{const r=e.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height,right:r.right};};
     const wave=document.querySelector('.wave-track svg'),wr=rect(wave),selected=rect(wave.querySelector('.wave-selection')),start=+wave.dataset.start,end=+wave.dataset.end;
     const track=document.querySelector('.wave-track'),inset=wr.x-track.getBoundingClientRect().x;
     const spectrum=document.querySelector(module==='M01'?'.spectrogram-canvas canvas':'.spectrum-wrap canvas'),sr=rect(spectrum);
     const annotation=document.querySelector(module==='M01'?'.textgrid-lane':'.annotation-grid'),ar=rect(annotation);
     const chosen=module==='M01'?rect(document.querySelector('.textgrid-interval.selected')):rect(document.querySelector('.grid-selection'));
     const spectrumSelection=module==='M12'?rect(document.querySelector('.spectrum-wrap .linked-selection')):null;
     const point=(r,t)=>r.x+(t-start)/(end-start)*r.width;
     const positions=[interval.xmin,interval.xmax].map(t=>({t,wave:point(wr,t),spectrum:point(sr,t),annotation:point(ar,t)}));
     // M12 actual canvas boundary pixels, independently sampled near source TextGrid start.
     let boundaryPixels=null;if(module==='M12'){
      const c=annotation,ctx=c.getContext('2d'),x=(interval.xmin-start)/(end-start)*c.width,y=Math.floor(c.height*.12),pixels=[];
      for(let dx=-3;dx<=3;dx++){const p=ctx.getImageData(Math.max(0,Math.round(x)+dx),y,1,1).data;pixels.push([...p]);}
      const teal=getComputedStyle(document.documentElement).getPropertyValue('--teal').trim();boundaryPixels={x,y,pixels,teal};
     }
     return {wave:wr,spectrum:sr,annotation:ar,selected,chosen,spectrumSelection,inset,start,end,positions,boundaryPixels,alignmentWrappers:document.querySelectorAll('.wave-viewport>.wave-aligned-layer').length,font:getComputedStyle(document.querySelector('.amplitude-axis')).fontSize};
    },{module,interval});
    for(const p of m.positions){assert(Math.abs(p.wave-p.spectrum)<1.1,`${module} wave-spectrum at ${p.t}: ${JSON.stringify(p)}`);assert(Math.abs(p.wave-p.annotation)<1.1,`${module} wave-grid at ${p.t}: ${JSON.stringify(p)}`);}
    assert(Math.abs(m.selected.x-m.chosen.x)<1.1,`${module} actual selection start differs`);assert(Math.abs(m.selected.right-m.chosen.right)<1.1,`${module} actual selection end differs`);
    if(module==='M12'){
     assert.equal(m.alignmentWrappers,0,'M12 external tracks must not acquire slot inset');assert(Math.abs(m.selected.x-m.spectrumSelection.x)<1.1);assert(Math.abs(m.selected.right-m.spectrumSelection.right)<1.1);
     assert(m.boundaryPixels.pixels.some(p=>p[1]>p[0]*1.2&&p[1]>p[2]*.9&&p[3]>100),'actual canvas boundary must contain teal stroke at real TextGrid time');
    }else{
     assert.equal(m.alignmentWrappers,2);assert(Math.abs(m.inset-64)<.1,'M01 default 64px axis inset once');
     // Check the actual Praat raster at source frame/bin centers, not just canvas boxes.
     m.raster=await page.evaluate(({data,request})=>{const c=document.querySelector('.spectrogram-canvas canvas'),ctx=c.getContext('2d'),raw=atob(data.pixels_base64),samples=[];for(const fx of [.2,.5,.8])for(const fy of [.25,.6]){const ix=Math.floor(data.width*fx),iy=Math.floor(data.height*fy),t=data.x1+ix*data.dx,frequency=data.y1+iy*data.dy,x=(t-request.start)/(request.end-request.start)*c.width,y=(data.frequency_max-frequency)/data.frequency_max*c.height;const actual=ctx.getImageData(Math.floor(x),Math.floor(y),1,1).data[0],expected=raw.charCodeAt(iy*data.width+ix);samples.push({t,ix,iy,actual,expected});}return samples;},latestSpectrum);
     assert(m.raster.every(p=>Math.abs(p.actual-p.expected)<=2),'Praat source pixels must occupy their mapped time positions');
    }
    report.measurements.push({module,viewport,fontSize,interval,...m});await page.screenshot({path:path.join(out,`${module}-${viewport.width}-font${fontSize}.png`),fullPage:true});
   }
   if(process.env.P17_ALIGNMENT_FINAL&&module==='M01'){
    report.finalTicks=[];for(const mode of ['normal','micro']){
     if(mode==='micro'){for(let n=0;n<7;n++)await page.getByRole('button',{name:'放大波形',exact:true}).click();await pause(page);await page.locator('.spectrogram-canvas canvas').waitFor({timeout:90000});}
     const axes=await page.evaluate(()=>['.wave-track .time-axis','.spectrogram-time-axis','.textgrid-timeline .time-axis'].map(selector=>({selector,ticks:[...document.querySelector(selector).children].map(e=>({text:e.textContent.trim(),x:e.getBoundingClientRect().x,right:e.getBoundingClientRect().right}))})));
     for(const axis of axes.slice(1))for(let i=0;i<5;i++){assert.equal(axis.ticks[i].text,axes[0].ticks[i].text);assert(Math.abs(axis.ticks[i].x-axes[0].ticks[i].x)<.1);}
     if(mode==='micro')assert(axes[0].ticks.every(t=>/^\d+\.\d{4}( s)?$/.test(t.text)));
     report.finalTicks.push({mode,axes});await page.screenshot({path:path.join(out,'M01-final-ticks-'+mode+'.png'),fullPage:true});
    }
   }
   if(module==='M12'){const inspected=(await b.rpc({op:'inspect'})).value;assert(inspected.originals_unchanged);report.bridges.at(-1).originals_unchanged=inspected.originals_unchanged;}
  }catch(e){await page.screenshot({path:path.join(out,module+'-failure.png'),fullPage:true});throw e;}
  finally{await page.close();await server.close();await b.close();if(module==='M01'){const check=JSON.parse(await fs.readFile(path.join(b.info.out,'originals-unchanged.json'),'utf8'));assert(check.unchanged);report.bridges.at(-1).originals_unchanged=check.unchanged;}}
 }
 assert.deepEqual(report.errors,[]);report.status='passed';
 }catch(e){report.failure=String(e.stack);throw e;}finally{await browser.close();await fs.writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({out,status:report.status,failure:report.failure,measurements:report.measurements.length},null,2));}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
