module.exports=async({page,click,loaded,out,rpc,checks})=>{
 const assert=require('node:assert/strict'),path=require('node:path');
 await rpc({op:'r3_setup'});await click('TextGrid标注');await click('选择语料文件夹');
 const choose=async name=>{await page.locator('.annotation-file-list button').filter({hasText:name}).click();await loaded();};
 await choose('split.wav');
 const view=async span=>{await page.getByLabel('标注可视时长').fill(String(span));await page.getByLabel('标注可视时长').press('Tab');await page.clock.runFor(40);};await view(2);
 const plot=kind=>page.locator(kind==='wave'?'.annotation-plots .wave-track svg':kind==='spec'?'.spectrum-wrap canvas':'.annotation-grid').first();
 const point=async(time,kind='grid',row=0)=>{const p=plot(kind);await p.scrollIntoViewIfNeeded();await page.clock.runFor(30);const box=await p.boundingBox(),wave=plot('wave'),a=Number(await wave.getAttribute('data-start')),b=Number(await wave.getAttribute('data-end'));return {x:box.x+box.width*(time-a)/(b-a),y:box.y+box.height*(kind==='grid'?(row===0?.22:.73):.5)};};
 const tap=async(time,kind='grid',row=0,double=false)=>{const p=await point(time,kind,row);await page.mouse[double?'dblclick':'click'](p.x,p.y);await page.clock.runFor(40);};
 const drag=async(a,b,kind='grid',ctrl=false,row=0)=>{const from=await point(a,kind,row),to=await point(b,kind,row);await page.mouse.move(from.x,from.y);if(ctrl)await page.keyboard.down('Control');await page.mouse.down();await page.mouse.move(to.x,to.y,{steps:8});await page.mouse.up();if(ctrl)await page.keyboard.up('Control');await page.clock.runFor(40);};
 const save=async()=>{await plot('grid').focus();await page.keyboard.press('Control+s');await loaded();await page.getByRole('status').filter({hasText:'已保存：'}).waitFor();return (await rpc({op:'inspect'})).value.grids['split_自动保存.TextGrid'];};
 const undo=async()=>{await plot('grid').focus();await page.keyboard.press('Control+z');await page.clock.runFor(40);};
 const near=(a,b)=>assert(Math.abs(a-b)<.004,`${a} != ${b}`);
 const checkRange=async()=>{const r=await page.getByLabel('当前标注选区').evaluate(e=>[Number(e.dataset.start),Number(e.dataset.end)]);assert.equal(Number(await page.getByLabel('强度起点',{exact:true}).inputValue()),r[0]);assert.equal(Number(await page.getByLabel('强度终点',{exact:true}).inputValue()),r[1]);return r;};
 assert.equal(await page.getByRole('button',{name:'框选标注',exact:true}).count(),0);assert.equal(await page.locator('.overview-controls').count(),0);assert.equal(await page.locator('.selection-readout').count(),0);
 await tap(.3);assert.deepEqual(await checkRange(),[.2,.8]);await drag(.31,.63,'wave');let range=await checkRange();near(range[0],.31);near(range[1],.63);
 await drag(.64,.33,'spec');range=await checkRange();near(range[0],.33);near(range[1],.64);
 await page.getByLabel('强度起点',{exact:true}).fill('0.345678');await page.getByLabel('强度起点',{exact:true}).press('Tab');assert.equal((await checkRange())[0],.345678);
 await page.getByLabel('强度终点',{exact:true}).fill('0.765432');await page.getByLabel('强度终点',{exact:true}).press('Tab');assert.equal((await checkRange())[1],.765432);
 checks.push('removed requested three rows; grid/wave/spectrum and precise intensity fields share one selection');
 await drag(.8,.9,'wave');let d=await save();near(d.tiers[0].intervals[1].xmax,.9);near(d.tiers[1].intervals[1].xmax,.9);await undo();
 await drag(.8,.9,'wave',true);d=await save();let words=d.tiers[0].intervals;near(words.find(i=>i.text==='ba1').xmax,.8);near(words.find(i=>i.text==='ba2').xmin,.9);assert(words.some(i=>!i.text&&Math.abs(i.xmin-.8)<.004&&Math.abs(i.xmax-.9)<.004));assert(d.tiers[1].intervals.some(i=>!i.text&&Math.abs(i.xmin-.8)<.004&&Math.abs(i.xmax-.9)<.004));
 await plot('wave').focus();await page.keyboard.press('Backspace');d=await save();assert.equal(d.tiers[0].intervals.length,4);await undo();await undo();
 await drag(.8,.7,'grid',true);d=await save();near(d.tiers[0].intervals.find(i=>i.text==='ba1').xmax,.7);near(d.tiers[0].intervals.find(i=>i.text==='ba2').xmin,.8);await undo();
 await drag(.8,.9,'grid',true,1);d=await save();assert.equal(d.tiers[0].intervals.length,4);assert.equal(d.tiers[1].intervals.length,5);await undo();
 checks.push('wave ordinary/shared and Ctrl/split boundary drags, grid splits both directions, independent phone split, Backspace and one-step undo');
 for(const [a,b,row] of [['wave','spec',0],['spec','grid',1],['grid','wave',1]]){
  await tap(1.55,a,row,true);await tap(1.8,b,row,true);await page.getByLabel('编辑选中标注文本').fill('跨窗标注');await page.getByLabel('编辑选中标注文本').press('Enter');d=await save();const item=d.tiers[0].intervals.find(i=>i.text==='跨窗标注');assert(item);near(item.xmin,1.55);near(item.xmax,1.8);await undo();await undo();
  // Undo creation restores its pending first endpoint. Clear it before testing
  // a new independent pair; wave/canvas pixel rounding must not decide this.
  assert(await page.locator('.sequence-hint').innerText().then(text=>text.includes('待定起点')));
  await page.keyboard.press('Escape');await page.clock.runFor(40);
  assert.equal(await page.getByRole('button',{name:'取消起点（Esc）',exact:true}).count(),0);
 }
 checks.push('manual double-click endpoints work across wave, spectrum and either annotation row, with saved Chinese labels and undo');
 await click('粘贴词表');await page.getByLabel('拼音词表',{exact:true}).fill('ba1 ba2 ba3');await click('应用词表');await page.locator('.sequence-toggle input').check();await tap(1.55,'spec',0,true);await tap(1.8,'wave',0,true);d=await save();assert(d.tiers[0].intervals.some(i=>i.text==='ba3'));await undo();await page.locator('.sequence-toggle input').uncheck();
 checks.push('sequence mode creates next syllable from cross-window endpoints without consuming extra entries');
 const images=[];for(const ms of ['5','10','20','50','100','0']){await page.getByLabel('语谱窗长',{exact:true}).selectOption(ms);await page.clock.runFor(50);images.push(await plot('spec').evaluate(e=>e.toDataURL()));}assert.equal(new Set(images).size,6);await page.getByLabel('语谱窗长',{exact:true}).selectOption('20');
 checks.push('all millisecond and legacy window choices redraw distinct real spectra');
 await click('设置');for(let i=0;i<5;i++)await page.getByLabel('放大页面',{exact:true}).click();await page.getByRole('dialog').getByRole('button',{name:'关闭对话框'}).click();await drag(.8,.9,'grid');d=await save();near(d.tiers[0].intervals[1].xmax,.9);await undo();await click('设置');await click('恢复 100%');await page.getByRole('dialog').getByRole('button',{name:'关闭对话框'}).click();
 checks.push('150% page zoom preserves real pointer boundary coordinates');
 await tap(.3);await page.getByLabel('编辑选中标注文本').fill('恢复副本');await page.getByLabel('编辑选中标注文本').press('Enter');await save();await choose('second.wav');await choose('split.wav');assert((await page.getByRole('status').innerText()).includes('已加载：split.TextGrid'));await tap(.3);assert.equal(await page.getByLabel('编辑选中标注文本').inputValue(),'ba1');
 const recovery=page.getByLabel('当前 TextGrid',{exact:true});await recovery.selectOption({label:'split_自动保存.TextGrid'});await loaded();await tap(.3);assert.equal(await page.getByLabel('编辑选中标注文本').inputValue(),'恢复副本');
 await page.getByLabel('TextGrid 保存后缀',{exact:true}).fill('');await plot('grid').focus();await page.keyboard.press('Control+s');await page.getByRole('dialog').waitFor();await click('确认保存');await loaded();await page.getByRole('dialog').waitFor({state:'hidden'});await choose('second.wav');await choose('split.wav');await tap(.3);assert.equal(await page.getByLabel('编辑选中标注文本').inputValue(),'恢复副本');assert((await page.getByRole('status').innerText()).includes('已加载：split.TextGrid'));
 const files=(await rpc({op:'inspect'})).value.files;assert(files.includes('split.TextGrid')&&files.includes('split_自动保存.TextGrid'));
 checks.push('reopening prefers original even with a newer recovery; recovery remains selectable; confirmed manual overwrite reopens saved original');
 await plot('grid').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,'r5-light.png'),fullPage:true});await page.evaluate(()=>document.documentElement.dataset.theme='dark');await page.clock.runFor(40);await page.screenshot({path:path.join(out,'r5-dark.png'),fullPage:true});await page.setViewportSize({width:850,height:850});await page.clock.runFor(40);await plot('grid').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,'r5-small.png'),fullPage:true});
 checks.push('light/dark and narrow-window screenshots captured');
};
