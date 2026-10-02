// R1: real host result, actual sidebar controls, no data mocked or rewritten.
const assert=require('node:assert/strict'),fs=require('node:fs/promises'),path=require('node:path');
module.exports=async({page,scope,click,out,checks})=>{
 const measurements=[];
 const graph=scope.locator('.curve-editor svg');
 const settle=()=>page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(()=>requestAnimationFrame(r)))));
 async function measure(tag){
  await settle();
  const m=await scope.evaluate((root,tag)=>{
   const svg=root.querySelector('.curve-editor svg'),r=svg.getBoundingClientRect(),matrix=svg.getScreenCTM(),point=svg.createSVGPoint();
   const px=x=>{point.x=x;point.y=0;return point.matrixTransform(matrix).x;};
   const lower=root.querySelector('.wave-track svg')??root.querySelector('.spectrum-plot'),b=lower.getBoundingClientRect();
   const selectors=['.module-workbench','.workbench-left','.workbench-center','.workbench-right-body','.curve-section','.preview-section'];
   const overflow=selectors.map(s=>{const e=root.querySelector(s);return {s,w:e.clientWidth,h:e.clientHeight,sw:e.scrollWidth,sh:e.scrollHeight}});
   const coords=p=>{const svg=p.ownerSVGElement,pt=svg.createSVGPoint();const v=p.getAttribute('d').match(/^M([\d.e+-]+)/);pt.x=Number(v[1]);pt.y=0;return pt.matrixTransform(svg.getScreenCTM()).x;};
   return {tag,upper:[px(72),px(svg.viewBox.baseVal.width-24)],lower:[b.left,b.right],upperRange:[svg.dataset.start,svg.dataset.end],lowerRange:[lower.dataset.start,lower.dataset.end],upperBounds:[r.top,r.bottom],lowerBounds:[b.top,b.bottom],page:[root.clientHeight,root.scrollHeight,root.clientWidth,root.scrollWidth],overflow,upperMarkers:[...root.querySelectorAll('.curve-editor .boundary')].map(coords),lowerMarkers:[...root.querySelectorAll('.vowel-boundary,.spectrum-plot .boundary')].map(coords)};
  },tag);
  measurements.push(m);
  for(let i=0;i<2;i++)assert(Math.abs(m.upper[i]-m.lower[i])<=1.1,tag+' plot edges '+JSON.stringify(m));
  assert.deepEqual(m.upperRange,m.lowerRange,tag+' time range');
  assert(m.page[1]<=m.page[0]+1&&m.page[3]<=m.page[2]+1,tag+' whole-page overflow');
  for(const c of m.overflow)if(c.w&&c.h)assert(c.sw<=c.w+1&&c.sh<=c.h+1,tag+' column/plot overflow '+JSON.stringify(c));
  assert(m.upperBounds[1]<m.lowerBounds[0],tag+' overlap');
  assert.equal(m.lowerMarkers.length,m.upperMarkers.length,tag+' marker count');
  m.upperMarkers.forEach((v,i)=>assert(Math.abs(v-m.lowerMarkers[i])<=1.1,tag+' marker alignment'));
 }
 assert(await scope.locator('.workbench-right .ipa-input').isVisible());
 assert.equal(await scope.locator('.module-toolbar').getByLabel('发声类型预设').count(),1);
 assert.equal(await scope.getByRole('button',{name:'导出实际合成参数',exact:true}).count(),0);
 // Keep old 0.6 s audio. The new 1.2 s timeline must leave its second half empty.
 await scope.getByLabel('总时长',{exact:true}).fill('1.2');await click('应用时长');await settle();
 assert.equal(await graph.getAttribute('data-end'),'1.2');
 assert.equal(await scope.locator('.wave-track svg').getAttribute('data-end'),'1.2');
 const occupied=await scope.locator('.wave-line').evaluate(p=>p.getBoundingClientRect().width/p.ownerSVGElement.getBoundingClientRect().width);
 assert(occupied>.45&&occupied<.55,'old WAV is not stretched '+occupied);
 for(const theme of ['light','dark']){
  await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);
  for(const [width,height] of [[1920,1000],[1920,1080],[2560,1440],[3840,2160]]){
   await page.setViewportSize({width,height});await settle();
   for(const [left,right] of [['Home','Home'],['End','End'],['Home','End'],['End','Home']]){
    const l=scope.getByRole('separator',{name:'合成参数宽度',exact:true}),r=scope.getByRole('separator',{name:'合成与任务宽度',exact:true});
    await l.press(left);await r.press(right);await settle();
    for(const mode of ['波形','语谱图']){
     await click(mode);await measure(`${theme}-${width}x${height}-${left}-${right}-${mode}`);
     await click('收起合成与任务');await measure(`${theme}-${width}x${height}-${left}-${right}-${mode}-collapsed`);await click('展开合成与任务');
    }
   }
  }
 }
 await page.setViewportSize({width:1920,height:1000});await page.evaluate(()=>document.documentElement.dataset.theme='light');
 await scope.getByRole('separator',{name:'合成参数宽度',exact:true}).press('Home');await scope.getByRole('separator',{name:'合成与任务宽度',exact:true}).press('Home');
 await click('波形');await scope.getByLabel('总时长',{exact:true}).fill('0.6');await click('应用时长');
 // Real pointer resize, intermediate width.
 const handle=await scope.getByRole('separator',{name:'合成参数宽度',exact:true}).boundingBox();await page.mouse.move(handle.x+2,handle.y+80);await page.mouse.down();await page.mouse.move(handle.x+102,handle.y+80,{steps:6});await page.mouse.up();await measure('pointer-intermediate');
 // Editing F2 must never mutate the visible F1/F3 references.
 await scope.getByRole('button',{name:'F2 Hz',exact:true}).click();await settle();assert.equal(await graph.locator('.reference-curve').count(),4);
 const refs=await graph.locator('.reference-curve').evaluateAll(a=>a.map(e=>[e.dataset.formant,e.getAttribute('d')]));
 const old=await graph.locator('.curve').getAttribute('d'),b=await graph.boundingBox();
 await page.keyboard.down('Shift');await page.mouse.move(b.x+b.width*.4,b.y+b.height*.4);await page.mouse.down();await page.mouse.move(b.x+b.width*.6,b.y+b.height*.6,{steps:4});await page.mouse.up();await page.keyboard.up('Shift');
 assert.notEqual(await graph.locator('.curve').getAttribute('d'),old);assert.deepEqual(await graph.locator('.reference-curve').evaluateAll(a=>a.map(e=>[e.dataset.formant,e.getAttribute('d')])),refs);
 await page.keyboard.down('Control');await graph.hover();await page.mouse.wheel(0,-100);await page.keyboard.up('Control');await settle();
 const ranges=await scope.evaluate(r=>[r.querySelector('.curve-editor svg').dataset,r.querySelector('.wave-track svg').dataset].map(d=>[d.start,d.end]));assert.deepEqual(ranges[0],ranges[1]);
 await click('重置范围');await page.screenshot({path:path.join(out,'r1-light.png'),fullPage:true});await page.evaluate(()=>document.documentElement.dataset.theme='dark');await click('语谱图');await page.screenshot({path:path.join(out,'r1-dark.png'),fullPage:true});
 await page.evaluate(()=>document.documentElement.dataset.theme='light');await click('波形');await scope.getByRole('button',{name:'F0 Hz',exact:true}).click();
 // Unapplied duration survives parameter selection and applying another control.
 await scope.getByLabel('总时长',{exact:true}).fill('0.8');await scope.getByRole('button',{name:'F1 Hz',exact:true}).click();assert.equal(await scope.getByLabel('总时长',{exact:true}).inputValue(),'0.8');await click('应用覆盖');assert.equal(await scope.getByLabel('总时长',{exact:true}).inputValue(),'0.8');
 await click('导出参数');await scope.getByRole('alert').filter({hasText:'请先应用时长'}).waitFor();await scope.getByLabel('总时长',{exact:true}).fill('0.6');await click('应用时长');await scope.getByRole('button',{name:'F0 Hz',exact:true}).click();
 // Public component with the default time domain, outside the M06 page.
 const shared=await page.context().browser().newPage({viewport:{width:1440,height:900}});
 await shared.goto(new URL('p17-waveform.html',page.url()).href);
 await shared.getByLabel('真实录音').setInputFiles(path.resolve(__dirname,'../fixtures/m06/source.wav'));
 await shared.getByLabel('读取状态').filter({hasText:'ready'}).waitFor();
 const defaultRange=await shared.locator('.wave-track svg').evaluate(e=>({start:+e.dataset.start,end:+e.dataset.end,duration:window.__p17.state.asset.duration}));
 assert.equal(defaultRange.start,0);assert.equal(defaultRange.end,defaultRange.duration);
 await shared.getByRole('button',{name:'放大波形',exact:true}).click();
 assert.equal(+(await shared.locator('.wave-track svg').getAttribute('data-end')),defaultRange.duration/2);
 await shared.getByRole('button',{name:'适合窗口',exact:true}).click();
 assert.equal(+(await shared.locator('.wave-track svg').getAttribute('data-end')),defaultRange.duration);
 await fs.writeFile(path.join(out,'r1-shared-wave.json'),JSON.stringify(defaultRange,null,2));
 await fs.writeFile(path.join(out,'r1-layout.json'),JSON.stringify(measurements,null,2));checks.push(`R1 ${measurements.length} layout/axis/boundary combinations, real sidebar drag, duration without stretching, read-only formants, pending input/export guard; shared default range and zoom`);
};
