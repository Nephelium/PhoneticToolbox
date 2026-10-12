// Real M07 layout with deterministic synthetic display data, no scientific jobs.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'..'),out=path.join(root,'output/validation/m07-resize-20261013');
async function main(){
 await fs.mkdir(out,{recursive:true});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{entries:['tests/p04-resize.html']}});
 await server.listen();
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,ignoreDefaultArgs:['--hide-scrollbars'],args:['--mute-audio']}),page=await browser.newPage({viewport:{width:1706,height:1018}}),errors=[],layouts=[];
 page.on('pageerror',e=>errors.push(e.message));
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/p04-resize.html');
  if(process.argv.includes('--kaiti'))await page.evaluate(async()=>{const {setFonts,preferences}=await import('/src/state/fonts.ts');await setFonts({...preferences.value,zh:'KaiTi',bodySize:15,figure:{follow:false,zh:'SimSun',latin:'Times New Roman',size:12}},false);localStorage.setItem('ptb.v3.layout.panels.local:M07.workbench',JSON.stringify({'--panel-left':294,'--panel-right':357}));});
  await page.evaluate(async()=>{
   const {host}=await import('/src/state/workspace.ts');const {defaults,generationDefaults}=await import('/src/modules/phonation-synthesis/state.ts');
   const points={axis:Array.from({length:51},(_,i)=>i*2),source:Array.from({length:51},(_,i)=>140+15*Math.sin(i/8)),target:Array.from({length:51},(_,i)=>95+10*Math.cos(i/8))};
   host.projects.write('m07.draft.resize-test:M07',{analysis:defaults(),generation:generationDefaults(),alignment:'normalize',count:51,kind:3,reverse:false,points,pending:false});
  });
  await page.evaluate(()=>document.querySelector('[aria-label=工具导航宽度]').dispatchEvent(new KeyboardEvent('keydown',{key:'Home',bubbles:true})));
  await page.locator('nav').getByRole('button',{name:'发声类型合成',exact:true}).click();await page.locator('.m07-page').waitFor();await page.waitForTimeout(700);
  await page.evaluate(async()=>{
   const {states,assignAsset}=await import('/src/state/workspace.ts');
   for(const key of Object.keys(states).filter(k=>k.includes('M07'))){const rate=11025,frames=6615;assignAsset(key,{id:key,name:'step09.wav',sampleRate:rate,frames,duration:frames/rate,channels:[Float32Array.from({length:frames},(_,i)=>.5*Math.sin(i*2*Math.PI*140/rate)*Math.sin(Math.PI*i/frames))]});}
  });
  const sample=()=>page.evaluate(async()=>{
   const selectors=['.m07-workspace','.module-workbench','.workbench-center','.workbench-right','.m07-plots','.source-plot','.target-plot','.main-f0','.f0-surface','.wave-viewport','.wave-track svg'];
   const frames=[];for(let i=0;i<36;i++){await new Promise(requestAnimationFrame);frames.push(selectors.map(s=>{const e=document.querySelector('.m07-page '+s),r=e.getBoundingClientRect();return [r.x,r.y,r.width,r.height,e.clientWidth,e.clientHeight,e.scrollHeight].map(n=>Math.round(n*100)/100);}));}return {selectors,frames};
  });
  const gutter=await page.evaluate(async()=>{
   const workspace=document.querySelector('.m07-workspace'),workbench=workspace.querySelector('.module-workbench'),center=workbench.querySelector('.workbench-center'),plots=center.querySelector('.m07-plots');
   const settle=async()=>{for(let i=0;i<8;i++)await new Promise(requestAnimationFrame);};
   const geometry=()=>[workbench,center,workbench.querySelector('.workbench-right')].map(e=>{const r=e.getBoundingClientRect();return [r.x,r.width]});
   await settle();const outerBefore=geometry();workbench.style.minHeight=(workspace.clientHeight+20)+'px';await settle();const outerAfter=geometry();workbench.style.minHeight='';await settle();const centerBefore=geometry();const plotBefore=plots.getBoundingClientRect().width;plots.style.minHeight=(center.clientHeight+20)+'px';await settle();const plotAfter=plots.getBoundingClientRect().width;const centerAfter=geometry();plots.style.minHeight='';await settle();return {outerBefore,outerAfter,centerBefore,centerAfter,plotBefore,plotAfter};
  });
  const scan=process.argv.includes('--scan');
  const sizes=scan?Array.from({length:60},(_,i)=>[1400+i*8,900+Math.floor(i/10)*20]):[[1706,1018],[1920,1080],[1440,900],[1280,800],[960,720],[800,500],[2559,1527]];
  if(process.argv.includes('--dense'))sizes.splice(0,sizes.length,...Array.from({length:100},(_,i)=>[1650+i*2,1018]));
  const fontMatrix=process.argv.includes('--font-matrix');
  const fonts=fontMatrix?['SimSun','Microsoft YaHei','KaiTi'].flatMap(zh=>[10,12,14,15,18,20,24].map(bodySize=>({zh,bodySize}))):[null];
  if(fontMatrix)sizes.splice(0,sizes.length,[1440,900],[1706,1018],[1920,1080]);
  for(const font of fonts){
  if(font)await page.evaluate(async font=>{const {setFonts,preferences}=await import('/src/state/fonts.ts');await setFonts({...preferences.value,...font},false);await document.fonts.ready;},font);
  for(const theme of scan?['light']:['light','dark'])for(const scale of scan||fontMatrix?[100]:[70,100,150])for(const [width,height] of sizes){
   await page.evaluate(({theme,scale})=>{document.documentElement.dataset.theme=theme;document.documentElement.style.zoom=String(scale/100);document.documentElement.style.setProperty('--page-scale',String(scale/100));},{theme,scale});
   await page.setViewportSize({width,height});await page.waitForTimeout(300);const data=await sample();
   const variants=[...new Set(data.frames.slice(12).map(f=>JSON.stringify(f)))];layouts.push({width,height,theme,scale,font,variants:variants.length,...data});
   if(variants.length>1){console.log('UNSTABLE',width,height,variants.length);await page.screenshot({path:path.join(out,'unstable.png')});if(scan)break;}
  }
  if(fontMatrix)console.log('font checked',font.zh,font.bodySize);
  }
  await page.setViewportSize({width:1706,height:1018});await page.evaluate(()=>{document.documentElement.style.zoom='1';document.documentElement.style.setProperty('--page-scale','1');});
  await page.getByRole('button',{name:'首页',exact:true}).first().click();await page.getByRole('button',{name:'开源与学术致谢',exact:true}).click();
  const dialog=page.getByRole('dialog',{name:'开源与学术致谢',exact:true});
  for(const title of ['The Voice Source in Speech Production: Data, Analysis and Models','VoiceSauce: A program for voice analysis','发声态研究的相关问题与VoiceSauce的使用'])assert.equal(await dialog.getByRole('heading',{name:title,exact:true}).count(),1,title);
  await dialog.getByRole('tab',{name:/软件与代码来源/}).click();assert.equal(await dialog.getByRole('heading',{name:'VoiceVista 录音模块',exact:true}).count(),0);await page.screenshot({path:path.join(out,'acknowledgements.png')});
  await fs.writeFile(path.join(out,fontMatrix?'font-matrix-report.json':scan?'before-scan.json':'browser-report.json'),JSON.stringify({errors,gutter,layouts,acknowledgements:true},null,2));
  console.log(JSON.stringify({layouts:layouts.length,stable:layouts.every(r=>r.variants===1),errors,acknowledgements:true}));
  if(!scan){assert.equal(errors.length,0,errors.join('\n'));assert.deepEqual(gutter.outerBefore,gutter.outerAfter,'workspace scrollbar must not move columns');assert.equal(gutter.plotBefore,gutter.plotAfter,'center scrollbar must not change plot width');assert(layouts.every(r=>r.variants===1),'M07 must settle after resizing');await page.screenshot({path:path.join(out,'browser-final.png')});}
 }finally{await browser.close();await server.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
