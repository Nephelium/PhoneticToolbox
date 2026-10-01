// P17-M04-R2: actual module, natural TextGrid intervals and real Praat preview.
// The caller owns the browser, corpus bridge and output directory.
module.exports=async({page,report,out})=>{
 const assert=require('node:assert/strict'),path=require('node:path');
 const click=name=>page.getByRole('button',{name,exact:true}).first().click();
 const settled=()=>page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
 const check=async(label)=>{
  await settled();
  await page.waitForFunction(()=>!!document.querySelector('.lpc-page .spectrogram-canvas canvas')&&!document.querySelector('.lpc-page .spectrogram-empty'),null,{timeout:30000});
  const value=await page.evaluate(()=>{
   const root=document.querySelector('.lpc-page'),wave=root.querySelector('.wave-track svg'),lane=root.querySelector('.textgrid-lane'),canvas=root.querySelector('.spectrogram-canvas canvas');
   const box=e=>{const b=e.getBoundingClientRect();return {left:b.left,right:b.right,width:b.width};};
   const selected=root.querySelector('.textgrid-interval.selected'),selection=root.querySelector('.wave-selection'),specSelection=root.querySelector('.spectrogram-selection');
   return {wave:box(wave),lane:box(lane),canvas:box(canvas),interval:selected?box(selected):null,waveSelection:box(selection),specSelection:specSelection?box(specSelection):null,start:+wave.dataset.start,end:+wave.dataset.end,selectedStart:+root.querySelector('[aria-label="LPC 选区起点"]').value,selectedEnd:+root.querySelector('[aria-label="LPC 选区终点"]').value,axes:[...root.querySelectorAll('.wave-track .time-axis,.spectrogram-time-axis,.textgrid-timeline>.time-axis')].map(box)};
  });
  // SVG and spectrogram have a one-pixel frame; the TextGrid has horizontal borders only.
  const near=(a,b,name)=>assert(Math.abs(a-b)<=1.05,`${label} ${name}: ${a} vs ${b}`);
  for(const edge of ['left','right']){
   near(value.wave[edge],value.lane[edge],'wave/TextGrid '+edge);
   near(value.wave[edge],value.canvas[edge],'wave/Praat '+edge);
   for(const axis of value.axes)near(value.wave[edge],axis[edge],'time axis '+edge);
   if(value.interval)near(value.waveSelection[edge],value.interval[edge],'natural interval boundary '+edge);
   assert(value.specSelection,'spectrogram selection is visible');
   near(value.waveSelection[edge],value.specSelection[edge],'linked selection '+edge);
  }
  report.alignment??=[];report.alignment.push({label,...value});
 };
 await page.getByLabel('显示语谱图（Praat）').check();
 await page.locator('.spectrogram-canvas canvas').waitFor({timeout:30000});
 await click('适合窗口');
 const interval=await page.locator('.textgrid-interval').evaluateAll(items=>items.map(e=>({a:+e.dataset.xmin,b:+e.dataset.xmax,text:e.textContent})).filter(e=>e.b>e.a&&e.text.trim()).sort((a,b)=>(a.b-a.a)-(b.b-b.a))[0]);
 assert(interval,'Natural TextGrid must include a labelled interval');
 for(const [width,height] of [[1920,1000],[2560,1360],[3840,2080]]){
  await page.setViewportSize({width,height});
  for(const zoom of [1,8,16]){
   await click('适合窗口');
   for(let z=1;z<zoom;z*=2)await page.getByLabel('放大波形',{exact:true}).click();
   if(zoom>1){
    await page.getByLabel('平移波形时间窗').evaluate((e,a)=>{const duration=+e.max/(1-1/(a.zoom)),window=duration/a.zoom;e.value=String(Math.max(0,Math.min(+e.max,(a.interval.a+a.interval.b)/2-window/2)));e.dispatchEvent(new Event('input',{bubbles:true}));},{zoom,interval});
   }
   await settled();
   const selected=page.locator(`.textgrid-interval[data-xmin="${interval.a}"][data-xmax="${interval.b}"]`);
   await selected.click();
   await page.locator('.spectrogram-canvas canvas').waitFor({timeout:30000});
   await check(`${width}x${height} zoom ${zoom}`);
  }
  await page.screenshot({path:path.join(out,'aligned-'+width+'.png'),fullPage:true});
 }
 await page.setViewportSize({width:1920,height:1000});
 await page.locator('.lpc-page').evaluate(e=>{e.style.setProperty('--wave-axis-width','88px');e.style.setProperty('--figure-size','18px');});
 await check('larger figure font and amplitude-axis gutter');
 await page.screenshot({path:path.join(out,'aligned-large-font.png'),fullPage:true});
 await page.locator('.lpc-page').evaluate(e=>{e.style.removeProperty('--wave-axis-width');e.style.removeProperty('--figure-size');});
 // A gesture made on the Praat canvas must resolve to the same absolute time in the waveform.
 const canvas=page.locator('.spectrogram-canvas canvas');await canvas.scrollIntoViewIfNeeded();const bounds=await canvas.boundingBox();
 await page.mouse.move(bounds.x+bounds.width*.2,bounds.y+bounds.height*.5);await page.mouse.down();await page.mouse.move(bounds.x+bounds.width*.7,bounds.y+bounds.height*.5,{steps:8});await page.mouse.up();
 await check('Praat drag synchronizes waveform selection');
 await page.mouse.move(bounds.x+bounds.width*.5,bounds.y+bounds.height*.5);await page.keyboard.down('Control');await page.mouse.wheel(0,-120);await page.keyboard.up('Control');
 await page.waitForFunction(previous=>+document.querySelector('.lpc-page .wave-track svg').dataset.end-+document.querySelector('.lpc-page .wave-track svg').dataset.start<previous,report.alignment.at(-1).end-report.alignment.at(-1).start);
 await page.locator('.spectrogram-canvas canvas').waitFor({timeout:30000});await check('Praat Ctrl-wheel zoom preserves time geometry');
 await page.getByLabel('显示语谱图（Praat）').uncheck();await click('适合窗口');
 await page.getByLabel('LPC 选区起点').fill('.2');await page.getByLabel('LPC 选区终点').fill('.4');
 report.checks.push('R2 12 actual-module time-geometry groups: natural TextGrid selection, Praat drag/zoom, 1080p/1440p/4K usable windows and wider amplitude gutter');
};
