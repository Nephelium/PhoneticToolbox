/* Real product build and real read-only runtime data. Overlay observes OL only. */
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url'),{createHash}=require('node:crypto');
const {execFileSync}=require('node:child_process');
const root=path.resolve(__dirname,'../..'),frontend=path.join(root,'frontend');
const [out,scratch,source,mode]=process.argv.slice(2),baseline=mode==='baseline';
const sha=b=>createHash('sha256').update(b).digest('hex');
(async()=>{
 const report={success:false,mode,cases:[],performance:[],errors:[],scope:'Windows owned Chrome; real trial2 input, actual AppShell/OL/exports, observed production build. One-machine timings; no final EXE, Mac or geographic content claim.'};
 let server,context,page,stage='build';
 try{
  const vite=await import(pathToFileURL(path.join(frontend,'node_modules/vite/dist/node/index.js')));
  const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  const manifest=JSON.parse(await fs.readFile(path.join(source,'data-manifest.json'),'utf8'));
  const identity=async()=>Object.fromEntries(await Promise.all(manifest.files.map(async f=>[f.path,sha(await fs.readFile(path.join(source,f.path)))])));
  report.inputs=await identity();
  const loaded=await vite.loadConfigFromFile({command:'build',mode:'production'},path.join(frontend,'vite.config.ts'));
  const baselineRef=baseline?process.env.PTB_GEO_BASELINE_REF:undefined,baselineSources=new Map();
  if(baselineRef){assert.match(baselineRef,/^[0-9a-f]{40}$/);for(const name of ['GeoMap.vue','GeolinguisticsPage.vue','map-paint.ts','projected-base.ts','attribution.ts']){const relative='frontend/src/modules/geolinguistics/'+name,raw=execFileSync('git',['show',baselineRef+':'+relative],{cwd:root});baselineSources.set(path.join(root,relative).replaceAll('\\','/'),raw.toString('utf8'));}report.baselineRef=baselineRef;report.baselineSources=Object.fromEntries([...baselineSources].map(([name,code])=>[path.relative(root,name),sha(Buffer.from(code))]));}
  const frozenBaseline={name:'owned-previous-map-source',enforce:'pre',load(id){return baselineSources.get(id.replaceAll('\\','/'));}};
  const observer={name:'owned-map-interaction-observer',enforce:'pre',transform(code,id){
   if(!id.replaceAll('\\','/').endsWith('/src/modules/geolinguistics/GeoMap.vue'))return;
   return {code:code.replace('</script>',String.raw`
const __qaKey={};
onMounted(()=>{
 const metrics={calls:0,styles:new Set<unknown>()};
 for(const layer of layers){const style=layer.getStyleFunction();if(style)layer.setStyle((f,r)=>{const result=style(f,r);metrics.calls++;for(const s of Array.isArray(result)?result:result?[result]:[])if(metrics.styles.size<1000)metrics.styles.add(s);return result;});}
 ((window as any).__geoQa??=new Map()).set(__qaKey,{map,layers,metrics,exportSvg,level:currentLevel});
});
onBeforeUnmount(()=>{(window as any).__geoQa.delete(__qaKey);});
</script>`),map:null};
  }};
  const dist=path.join(scratch,'dist');
  await vite.build({...loaded.config,configFile:false,root:frontend,logLevel:'error',cacheDir:path.join(scratch,'vite-cache'),plugins:[frozenBaseline,...loaded.config.plugins,observer],build:{...loaded.config.build,outDir:dist}});
  server=await vite.preview({configFile:false,root:frontend,build:{outDir:dist},preview:{host:'127.0.0.1',port:0},logLevel:'error'});
  context=await chromium.launchPersistentContext(path.join(scratch,'profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1600,height:1000},acceptDownloads:true});
  await context.addInitScript(()=>Object.defineProperty(window,'showSaveFilePicker',{configurable:true,value:undefined}));
  page=await context.newPage();page.setDefaultTimeout(60000);page.on('pageerror',e=>report.errors.push({stage,message:e.message}));
  await page.goto(server.resolvedUrls.local[0]);await page.locator('.home-page').waitFor();await page.waitForTimeout(3200);
  const announcement=page.getByRole('dialog').getByRole('button',{name:'知道了',exact:true});if(await announcement.isVisible())await announcement.click();
  stage='map-import';
  await page.locator('.nav-item').and(page.getByRole('button',{name:'语言地图',exact:true})).click();
  const atlas=page.locator('.language-atlas');await atlas.getByLabel('选择语言地图资料夹',{exact:true}).setInputFiles(source);
  await atlas.locator('.atlas-workbench').waitFor();
  const settle=async()=>{await page.waitForFunction(()=>!document.querySelector('.map-loading'));await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));};
  await settle();await atlas.getByRole('button',{name:'收起表格',exact:true}).click();
  const boundary=atlas.locator('.geography-toolbar label').filter({hasText:'边界'}).locator('select');await boundary.selectOption('auto');
  async function view(zoom){await page.evaluate(zoom=>{const q=[...window.__geoQa.values()][0];q.map.getView().setCenter([12690321,2500000]);q.map.getView().setZoom(zoom);},zoom);await page.waitForTimeout(180);await settle();}
  if(mode==='native-options'){const cdp=await context.newCDPSession(page);await atlas.getByRole('button',{name:'展开表格',exact:true}).click();await require('./geolinguistics-browser-options.cjs')({page,cdp,out,report});assert.deepEqual(await identity(),report.inputs);assert.deepEqual(report.errors,[]);report.success=true;return;}
  if(mode==='options'){await require('./geolinguistics-options.cjs')({page,atlas,boundary,settle,view,out,report});assert.deepEqual(await identity(),report.inputs);assert.deepEqual(report.errors,[]);report.success=true;return;}
  async function inspect(theme,zoom,level){
   await view(zoom);
   const state=await page.evaluate(()=>[...window.__geoQa.values()].map(q=>({level:q.level(),zoom:q.map.getView().getZoom(),layers:q.layers.map(l=>({features:l.getSource().getFeatures().length,visible:l.getVisible(),zIndex:l.getZIndex()??0,styles:l.getSource().getFeatures().flatMap(f=>{const s=l.getStyleFunction()?.(f,q.map.getView().getResolution());return (Array.isArray(s)?s:s?[s]:[]).map(s=>({type:(s.getGeometry()||f.getGeometry()).getType(),fill:s.getFill()?.getColor(),stroke:s.getStroke()?.getColor(),role:f.get('renderRole'),boundaryType:f.get('boundaryType')}));})}))})));
   const xml=await page.evaluate(()=>[...window.__geoQa.values()][0].exportSvg('QA',[],''));
   const svg=await page.evaluate(xml=>[...new DOMParser().parseFromString(xml,'image/svg+xml').querySelectorAll('g[clip-path] path')].map(p=>({fill:p.getAttribute('fill'),stroke:p.getAttribute('stroke')})),xml);
   if(!baseline){for(const m of state){assert.equal(m.level,level);const visible=m.layers.filter(l=>l.visible).flatMap(l=>l.styles);assert.equal(m.layers.length,5);assert.equal(m.layers[0].features,{province:34,city:370,district:2875}[level]);assert(!visible.some(s=>s.stroke==='#bcc7c9'||s.role==='administrative-display-area')); assert(visible.filter(s=>s.type.includes('Polygon')&&s.role!=='boundary-symbol').every(s=>s.fill&&s.fill!=='none'));assert.equal(visible.filter(s=>s.boundaryType==='省界').length,0);assert.equal(visible.filter(s=>s.role==='boundary-symbol').length,10);}assert.equal(svg.length,{province:34,city:370,district:2875}[level]+14);assert(!svg.some(p=>p.fill==='#edf0ed'));}
   report.cases.push({theme,zoom,level,state,svgPaths:svg.length});
   await page.screenshot({path:path.join(out,`${theme}-${level}-${zoom}.png`)});
  }
  for(const theme of ['light','dark']){await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);for(const [zoom,level] of [[5.5,'province'],[6.5,'city'],[8.5,'district'],[10.5,'district']])await inspect(theme,zoom,level);}
  if(!baseline){assert(await atlas.getByText('彩色小点表示当前层级的单一分类，蓝灰色大点表示同一地点兼有多类。分类数量按完整父级路径计数。',{exact:true}).isVisible());await atlas.getByRole('button',{name:'全屏地图',exact:true}).click();await page.waitForFunction(()=>!!document.fullscreenElement);await page.getByRole('button',{name:'退出全屏',exact:true}).click();await page.waitForFunction(()=>!document.fullscreenElement);report.cases.push({classificationFullscreen:true});}
  await atlas.getByRole('tab',{name:'语言现象',exact:true}).click();
  await atlas.getByLabel('检索语言特征',{exact:true}).fill('F120_P');await atlas.locator('.feature-list button').click();
  const panels=atlas.locator('.map-heading label select');
  const cdp=await context.newCDPSession(page);await cdp.send('Performance.enable');
  await cdp.send('Emulation.setCPUThrottlingRate',{rate:4});report.cpuThrottle=4;
  for(const count of [1,4]){
   await panels.selectOption(String(count));await view(8.5);
   await page.evaluate(()=>{for(const q of window.__geoQa.values()){q.metrics.calls=0;q.metrics.styles.clear();}});
   const begin=await cdp.send('Performance.getMetrics');
   const frames=await page.evaluate(async()=>{const first=[...window.__geoQa.values()][0],v=first.map.getView(),start=v.getCenter().slice(),frames=[];v.beginInteraction();let last=performance.now();for(let i=0;i<24;i++){v.setCenter([start[0]+Math.sin(i/3)*40000,start[1]+Math.cos(i/3)*20000]);await new Promise(r=>requestAnimationFrame(r));const now=performance.now();frames.push(now-last);last=now;}v.endInteraction(0);return frames;});
   await settle();const rect=await atlas.locator('.map-target').first().boundingBox();for(let i=0;i<30;i++)await page.mouse.move(rect.x+20+i*(rect.width-40)/30,rect.y+rect.height*.5+Math.sin(i)*30);
   const end=await cdp.send('Performance.getMetrics'),get=(m,k)=>m.metrics.find(x=>x.name===k).value;
   const styles=await page.evaluate(()=>[...window.__geoQa.values()].map(q=>({calls:q.metrics.calls,uniqueStyles:q.metrics.styles.size})));
   report.performance.push({count,frames,taskMs:(get(end,'TaskDuration')-get(begin,'TaskDuration'))*1000,styles});
  }
  report.nativeGestures=[];
  await page.evaluate(()=>{window.__gestureFrames=[];window.__gestureTasks=[];window.__gestureActive=false;let previous=performance.now();const frame=now=>{if(window.__gestureActive)window.__gestureFrames.push(now-Math.max(previous,window.__gestureStart));previous=now;requestAnimationFrame(frame);};requestAnimationFrame(frame);new PerformanceObserver(list=>{if(window.__gestureActive)for(const item of list.getEntries())window.__gestureTasks.push(item.duration);}).observe({type:'longtask',buffered:false});});
  for(const count of [1,4])for(const kind of ['drag','wheel'])for(let trial=0;trial<3;trial++){
   await panels.selectOption(String(count));await boundary.selectOption('district');await view(8.5);await page.waitForTimeout(350);const rect=await atlas.locator('.map-target').first().boundingBox(),x=rect.x+rect.width*.45,y=rect.y+rect.height*.5;
   await page.mouse.move(x,y);await page.evaluate(()=>{window.__gestureFrames=[];window.__gestureTasks=[];for(const q of window.__geoQa.values()){q.metrics.calls=0;q.metrics.styles.clear();}window.__gestureStart=performance.now();window.__gestureActive=true;});const begin=await cdp.send('Performance.getMetrics');
   if(kind==='drag'){await page.mouse.down();for(let step=1;step<=30;step++){await page.mouse.move(x+Math.sin(step/6)*rect.width*.15,y+Math.sin(step/4)*rect.height*.1);await page.waitForTimeout(16);}await page.mouse.up();}
   else{for(const delta of [-90,-90,-90,-90,90,90,90,90]){await page.mouse.wheel(0,delta);await page.waitForTimeout(90);}}
   await page.waitForTimeout(500);await settle();const end=await cdp.send('Performance.getMetrics');const metrics=await page.evaluate(()=>{window.__gestureActive=false;const f=window.__gestureFrames.sort((a,b)=>a-b);return {frames:f.length,p95FrameMs:f[Math.floor(f.length*.95)],maxFrameMs:f.at(-1),longtasks:window.__gestureTasks.length,longtaskMs:window.__gestureTasks.reduce((a,b)=>a+b,0),levels:[...window.__geoQa.values()].map(q=>q.level())};});
   assert(metrics.frames>0);assert(metrics.levels.every(level=>level==='district'));report.nativeGestures.push({count,kind,trial,...metrics,taskMs:(end.metrics.find(v=>v.name==='TaskDuration').value-begin.metrics.find(v=>v.name==='TaskDuration').value)*1000});
  }
  await cdp.send('Emulation.setCPUThrottlingRate',{rate:1});
  await panels.selectOption('1');await boundary.selectOption('auto');
  if(!baseline){
   for(const level of ['province','city','district']){await boundary.selectOption(level);for(const zoom of [5.5,10.5])await inspect('manual-override',zoom,level);}
   await boundary.selectOption('city');await view(7.7);const r=await atlas.locator('.map-target').first().boundingBox();await page.mouse.move(r.x+r.width*.5,r.y+r.height*.5);await page.mouse.wheel(0,-600);await page.waitForTimeout(700);await settle();assert(await page.evaluate(()=>[...window.__geoQa.values()].every(q=>q.level()==='city'&&q.layers[0].getSource().getFeatures().length===370)));report.cases.push({manualCitySurvivesNativeWheel:true});await boundary.selectOption('auto');await inspect('auto-resumed',8.5,'district');
   for(const count of [1,2,4]){await panels.selectOption(String(count));await settle();
    const before=await page.evaluate(()=>[...window.__geoQa.values()].map(q=>({center:q.map.getView().getCenter(),zoom:q.map.getView().getZoom()})));
    await atlas.getByRole('button',{name:'全屏地图',exact:true}).click();await page.waitForFunction(()=>document.fullscreenElement?.classList.contains('atlas-center'));await settle();
    assert.equal(await page.locator(':fullscreen .geo-map').count(),count);await page.screenshot({path:path.join(out,`fullscreen-${count}.png`)});
    await page.getByRole('button',{name:'地图来源',exact:true}).first().click();const popup=page.locator(':fullscreen .map-attribution-details:not([hidden])').first();assert(await popup.isVisible());assert.notEqual(await popup.evaluate(e=>getComputedStyle(e).backgroundColor),'rgba(0, 0, 0, 0)');
    if(count===2){await page.keyboard.press('Escape');assert.equal(await page.locator(':fullscreen .map-attribution-details:not([hidden])').count(),0);await page.keyboard.press('Escape');}else await page.getByRole('button',{name:'退出全屏',exact:true}).click();
    await page.waitForFunction(()=>!document.fullscreenElement);await settle();
    assert.deepEqual(await page.evaluate(()=>[...window.__geoQa.values()].map(q=>({center:q.map.getView().getCenter(),zoom:q.map.getView().getZoom()}))),before);
    report.cases.push({fullscreen:count,exit:count===2?'Escape':'button',viewPreserved:true});
   }
   await atlas.getByRole('button',{name:'全屏地图',exact:true}).click();await page.waitForFunction(()=>!!document.fullscreenElement);
   await page.evaluate(()=>document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true})));
   await page.waitForFunction(()=>!document.fullscreenElement);
   report.cases.push({fullscreenKeyHandler:true});
   await panels.selectOption('1');await view(8.5);const target=await atlas.locator('.map-target').first().boundingBox();
   const region=await page.evaluate(()=>{const q=[...window.__geoQa.values()][0],[w,h]=q.map.getSize();for(const f of q.layers[0].getSource().getFeatures()){const g=f.getGeometry(),cs=g.getInteriorPoint?[g.getInteriorPoint().getCoordinates()]:g.getInteriorPoints().getCoordinates();for(const c of cs){const p=q.map.getPixelFromCoordinate(c);if(p[0]>w*.2&&p[0]<w*.8&&p[1]>h*.2&&p[1]<h*.8)return {id:f.getId(),pixel:p};}}});
   assert(region,'Visible administrative interior needed for actual image-layer hit test');await page.mouse.click(target.x+region.pixel[0],target.y+region.pixel[1]);await page.waitForFunction(id=>[...document.querySelectorAll('.geography-toolbar select')].some(s=>s.value===id),region.id);report.cases.push({actualRegionClick:region.id});
   await page.mouse.move(target.x+target.width*.5,target.y+target.height*.5);await page.mouse.down();await page.mouse.move(target.x+target.width*.7,target.y+target.height*.55,{steps:24});await page.mouse.up();await settle();
   await page.mouse.wheel(0,-600);await page.waitForTimeout(600);await settle();await page.mouse.wheel(0,600);await page.waitForTimeout(600);await settle();
   report.cases.push({nativeMouseDragAndWheel:true,responsive:await page.getByRole('button',{name:'全屏地图',exact:true}).isEnabled()});
   await boundary.selectOption('province');await atlas.getByRole('button',{name:'中国',exact:true}).click();await settle();await atlas.getByRole('button',{name:'展开表格',exact:true}).click();
   for(const theme of ['light','dark'])for(const wallpaper of ['off','on']){
    await page.evaluate(({theme,wallpaper})=>{document.documentElement.dataset.theme=theme;document.documentElement.dataset.backgroundImage=wallpaper;document.querySelector('.table-scroll').scrollTop=0;},{theme,wallpaper});
    const scroll=atlas.locator('.table-scroll'),rect=await scroll.boundingBox();
    const before=await scroll.locator('thead').boundingBox();await page.mouse.move(rect.x+rect.width/2,rect.y+rect.height*.75);await page.mouse.wheel(0,170);await page.waitForTimeout(250);
    const after=await scroll.locator('thead').boundingBox(),result=await scroll.evaluate(e=>{const head=e.querySelector('thead'),r=head.getBoundingClientRect(),th=e.querySelector('th');return {scrollTop:e.scrollTop,headerBackground:getComputedStyle(head).backgroundColor,cellBackground:getComputedStyle(th).backgroundColor,hit:head.contains(document.elementFromPoint(r.x+20,r.y+r.height/2))};});
    assert(result.scrollTop>0);assert(Math.abs(before.y-after.y)<1);assert(!['transparent','rgba(0, 0, 0, 0)'].includes(result.headerBackground));assert(result.hit);
    await page.screenshot({path:path.join(out,`table-${theme}-${wallpaper}.png`)});report.cases.push({tableScroll:true,theme,wallpaper,...result});
    await page.getByRole('button',{name:'地图来源',exact:true}).first().click();const popup=page.locator('.map-attribution-details:not([hidden])').first(),paint=await popup.evaluate(e=>({background:getComputedStyle(e).backgroundColor,text:getComputedStyle(e).color,surface:e.hasAttribute('data-background-surface')}));
    assert(paint.surface);assert(paint.background.startsWith('rgb('),'Source popup must have an opaque theme surface');assert.notEqual(paint.background,paint.text);await page.screenshot({path:path.join(out,`sources-${theme}-${wallpaper}.png`)});await page.getByRole('button',{name:'地图来源',exact:true}).first().click();report.cases.push({sourcePopup:true,theme,wallpaper,...paint});
    await atlas.locator('.atlas-table tbody .text-button').first().click();await settle();await page.waitForTimeout(350);const mapRect=await atlas.locator('.map-target').first().boundingBox();await page.mouse.move(mapRect.x+mapRect.width/2,mapRect.y+mapRect.height/2);const tip=atlas.locator('.map-hover');await tip.waitFor();const hoverPaint=await tip.evaluate(e=>({background:getComputedStyle(e).backgroundColor,text:getComputedStyle(e).color,surface:e.hasAttribute('data-background-surface')}));assert(hoverPaint.surface&&hoverPaint.background.startsWith('rgb('));assert.notEqual(hoverPaint.background,hoverPaint.text);await page.screenshot({path:path.join(out,`hover-${theme}-${wallpaper}.png`)});report.cases.push({hoverPopup:true,theme,wallpaper,...hoverPaint});await page.mouse.move(1,1);
   }
   await page.evaluate(()=>delete document.documentElement.dataset.backgroundImage);
   await require('./geolinguistics-options.cjs')({page,atlas,boundary,settle,view,out,report});
   assert(report.performance.every(s=>s.styles.every(m=>m.uniqueStyles<1000)),'Styles should be reused between frames');
  }
  assert.deepEqual(await identity(),report.inputs);assert.deepEqual(report.errors,[]);report.success=true;
 }catch(e){report.error={stage,message:e.message,stack:e.stack};process.exitCode=1;if(page)await page.screenshot({path:path.join(out,'failure.png')}).catch(()=>{});}
 finally{await context?.close();await new Promise(r=>server?server.httpServer.close(r):r());await fs.writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({out,success:report.success,error:report.error,performance:report.performance}));}
})();
