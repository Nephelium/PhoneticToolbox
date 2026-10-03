const assert=require('node:assert/strict'),fs=require('node:fs/promises'),path=require('node:path');
module.exports=async({page,scope,click,out,checks})=>{
 const idle=()=>page.waitForFunction(()=>document.querySelector('.m06-page')?.getAttribute('aria-busy')==='false');
 const snapshot=async(tag)=>{
  const event=page.waitForEvent('download');await click('导出参数');const d=await event;
  const file=path.join(out,'r3-'+tag+'.csv');await d.saveAs(file);
  const line=(await fs.readFile(file,'utf8')).split(/\r?\n/).find(l=>l.startsWith('"__PTB_CONFIG__"'));
  return JSON.parse(line.match(/^"__PTB_CONFIG__","0","(.*)","false"$/)[1].replaceAll('""','"'));
 };
 const bar=scope.locator('.m06-transport');
 assert.equal(await scope.locator('.transport-controls').count(),1);
 assert.equal(await scope.locator('.preview-section .transport-controls').count(),0);
 for(const method of ['praat_cc','praat_ac','reaper']){
  await scope.getByLabel('F0 提取算法').selectOption(method);await click('提取参数');await idle();
  assert((await scope.innerText()).includes('参数提取完成'),await scope.innerText());
  assert.equal((await snapshot(method)).f0_method,method);
 }
 await click('源音频');
 const points=bar.locator('.selection-controls input');await points.nth(0).fill('0.1');await points.nth(0).press('Tab');await points.nth(1).fill('0.3');await points.nth(1).press('Tab');
 assert.equal(await bar.locator('.playback-seek input').getAttribute('min'),'0.1');
 assert.equal(await bar.locator('.playback-seek input').getAttribute('max'),'0.3');
 await click('合成结果');assert.equal(await points.nth(0).inputValue(),'0');
 await click('源音频');assert.equal(await points.nth(0).inputValue(),'0.1');
 await bar.getByRole('button',{name:'播放选区',exact:true}).click();await bar.getByRole('button',{name:'停止',exact:true}).click();
 await bar.getByLabel('播放音量').fill('0.4');assert((await bar.innerText()).includes('40%'));
 await click('全部');assert.equal(await points.nth(0).inputValue(),'0');
 await click('语谱图');
 const layouts=[];
 for(const [width,height] of [[1920,1000],[1440,900],[1100,800],[900,700]])for(const theme of ['light','dark']){
  await page.setViewportSize({width,height});await page.evaluate(t=>document.documentElement.dataset.theme=t,theme);
  await page.waitForTimeout(120);
  const m=await scope.evaluate(root=>{const b=e=>{const r=e.getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height,b:r.bottom}};
   const label=root.querySelector('.preview-controls label'),sel=label.querySelector('select'),walker=document.createRange();walker.selectNodeContents(label);walker.setEnd(label,1);
   return {bar:b(root.querySelector('.m06-transport')),work:b(root.querySelector('.module-workbench')),left:b(root.querySelector('.workbench-left')),center:b(root.querySelector('.workbench-center')),right:b(root.querySelector('.workbench-right')),root:b(root),label:b(label),select:b(sel),text:b(walker),rules:b([...root.querySelectorAll('button')].find(e=>e.textContent==='元音规则')),generate:b([...root.querySelectorAll('button')].find(e=>e.textContent==='生成元音'))};});
  assert(m.bar.y>=m.work.b-1&&m.bar.b<=m.root.b+1,JSON.stringify(m));
  assert(Math.abs(m.bar.x-m.work.x)<1&&Math.abs(m.bar.w-m.work.w)<1);
  assert(Math.abs(m.text.y+m.text.h/2-m.select.y-m.select.h/2)<3,JSON.stringify(m));
  assert(m.rules.b<=m.generate.y+1);
  if(m.work.w<=1060)assert(m.right.y>=Math.max(m.left.b,m.center.b)-1,JSON.stringify(m));
  layouts.push({width,height,theme,...m});
  if(width===1920||width===900)await page.screenshot({path:path.join(out,`r3-${width}-${theme}.png`),fullPage:true});
 }
 await page.setViewportSize({width:1920,height:1000});await page.evaluate(()=>document.documentElement.dataset.theme='light');
 await fs.writeFile(path.join(out,'r3-controls.json'),JSON.stringify({success:true,layouts},null,2));
 checks.push('R3: actual CC/AC/native REAPER extraction, CSV persistence, shared whole-module transport, per-source selection, volume, inline window label and rule order at 4 sizes × 2 themes');
};
