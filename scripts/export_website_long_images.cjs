// Capture the actual public page as three adjacent native screenshot clips.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {createHash}=require('node:crypto'),{spawnSync}=require('node:child_process');
const root=path.resolve(__dirname,'..');
async function main(){
 const out=path.join(root,'output/validation/website','long-images-'+Date.now()),scratch=path.join(out,'scratch');await fs.mkdir(scratch,{recursive:true});
 const write=(name,data)=>fs.writeFile(path.join(out,name),JSON.stringify(data,null,2)+'\n');
 const marker={schema:'ptb-test-run/1',owner_pid:process.pid,state:'active',started_at:new Date().toISOString(),recipe:'Three consecutive public homepage screenshots; whitespace boundaries, no workbook or audio input'};
 await write('run.json',marker);const report={success:false,url:'https://www.phonetictoolbox.com/',width:1600,files:[],errors:[]};let browser;
 try{
  const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  browser=await chromium.launchPersistentContext(path.join(scratch,'profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1600,height:1000},deviceScaleFactor:1,userAgent:'Mozilla/5.0 PTB-Website-Verify'});
  const page=await browser.newPage();page.on('pageerror',e=>report.errors.push(e.message));
  await page.goto(report.url);await page.evaluate(()=>document.fonts.ready);
  for(const section of await page.locator('main > section,.site-stats,.site-footer').all()){await section.scrollIntoViewIfNeeded();await page.waitForTimeout(130);}
  for(const img of await page.locator('img[loading="lazy"]').all()){await img.scrollIntoViewIfNeeded();await img.evaluate(e=>e.decode());}
  await page.evaluate(()=>{document.querySelectorAll('.reveal').forEach(e=>e.classList.add('is-visible'));document.activeElement?.blur();scrollTo(0,0);});
  await page.mouse.move(1598,1);await page.waitForTimeout(1400);
  await page.addStyleTag({content:'*,*::before,*::after{animation-play-state:paused!important;transition:none!important}html{scroll-behavior:auto!important}.reading-dock{display:none!important}'});
  await page.evaluate(()=>document.querySelectorAll('.screen-stage').forEach(e=>e.style.setProperty('--angle','.22')));
  assert.equal(await page.locator('[data-screen]').count(),26);
  assert(await page.locator('[data-screen] img').evaluateAll(es=>es.every(e=>e.complete&&e.naturalWidth)));
  await page.waitForFunction(()=>!document.querySelector('[data-stats-since]').textContent.includes('读取中'));
  const layout=await page.evaluate(()=>{
   const height=document.documentElement.scrollHeight;
   const rect=e=>{const r=e.getBoundingClientRect();return {top:r.top+scrollY,bottom:r.bottom+scrollY}};
   // Protect complete functional sections, screenshots, headings and decorations.
   const protectedRects=[...document.querySelectorAll('.hero,.feature-ribbon,.showcase-copy,.screen-stage,.section-heading,.tool-divider,.more-features,.content-section,.guide-section,.author-note,.site-stats,.site-footer,.site-header')].map(rect);
   // Protect rotated screenshots and shadows; decorative background color may cross a seam.
   protectedRects.push(...[...document.querySelectorAll('.screen-link')].map(e=>{const r=rect(e);return {top:r.top-24,bottom:r.bottom+24}}));
   const blocked=protectedRects.filter(r=>r.bottom>r.top).sort((a,b)=>a.top-b.top),merged=[];
   for(const r of blocked){const last=merged.at(-1);if(last&&r.top<=last.bottom)last.bottom=Math.max(last.bottom,r.bottom);else merged.push({...r});}
   const gaps=[];for(let i=1;i<merged.length;i++){const a=Math.ceil(merged[i-1].bottom),b=Math.floor(merged[i].top);if(b-a>=24)gaps.push({start:a,end:b,y:Math.round((a+b)/2)});}
   return {height,gaps,sections:[...document.querySelectorAll('main > section')].map(e=>({id:e.id,...rect(e)}))};
  });
  const total=layout.height;let best;
  for(const a of layout.gaps)for(const b of layout.gaps){if(b.y<=a.y)continue;const sizes=[a.y,b.y-a.y,total-b.y];if(Math.min(...sizes)<total*.2)continue;const cost=sizes.reduce((s,h)=>s+(h-total/3)**2,0);if(!best||cost<best.cost)best={a,b,cost,sizes};}
  assert(best,'Two safe whitespace gaps required');report.layout=layout;report.boundaries=[0,best.a.y,best.b.y,total];report.whitespace=[best.a,best.b];
  for(let i=0;i<3;i++){
   const name='官网长图-'+String(i+1).padStart(2,'0')+'.png',y=report.boundaries[i],height=report.boundaries[i+1]-y;
   const file=path.join(out,name);await page.screenshot({path:file,fullPage:true,clip:{x:0,y,width:1600,height},animations:'disabled',caret:'hide',timeout:60000});
   const bytes=await fs.readFile(file);assert.equal(bytes.readUInt32BE(16),1600);assert.equal(bytes.readUInt32BE(20),height);report.files.push({file,width:1600,height,y,bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex')});
  }
  // Evidence views around both seams; no seams intersect protected content.
  for(const [i,gap] of [best.a,best.b].entries())await page.screenshot({path:path.join(out,'分界检查-'+(i+1)+'.png'),fullPage:true,clip:{x:0,y:gap.y-200,width:1600,height:400},animations:'disabled'});
  assert.deepEqual(report.errors,[]);report.success=true;
 }finally{
  await browser?.close();await write('report.json',report);
  const files=[];async function inventory(dir){for(const entry of await fs.readdir(dir,{withFileTypes:true})){assert(!entry.isSymbolicLink());const file=path.join(dir,entry.name);if(entry.isDirectory())await inventory(file);else{const b=await fs.readFile(file);files.push({path:path.relative(scratch,file),bytes:b.length,sha256:createHash('sha256').update(b).digest('hex')});}}}await inventory(scratch);
  await write('scratch-manifest.json',{schema:'ptb-test-scratch-files/1',files});await write('run.json',{...marker,state:'ready',test_outcome:report.success?'completed':'failed'});
  const cleanup=spawnSync('powershell.exe',['-NoProfile','-ExecutionPolicy','Bypass','-File',path.join(root,'scripts/cleanup_test_scratch.ps1'),'-RunDirectory',out],{windowsHide:true,encoding:'utf8'});assert.equal(cleanup.status,0,cleanup.stdout+cleanup.stderr);
  console.log(JSON.stringify({out,success:report.success,files:report.files,boundaries:report.boundaries}));
 }
}
main().catch(e=>{console.error(e);process.exitCode=1});
