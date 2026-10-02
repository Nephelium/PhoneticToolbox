const assert=require('node:assert/strict'),fs=require('node:fs/promises'),path=require('node:path');
module.exports=async({page,scope,click,out,checks})=>{
 let id=0;const rows=[];
 async function snapshot(){
  const event=page.waitForEvent('download');await click('导出参数');const d=await event;
  const file=path.join(out,`r2-${++id}.csv`);await d.saveAs(file);
  const line=(await fs.readFile(file,'utf8')).split(/\r?\n/).find(l=>l.startsWith('"__PTB_CONFIG__"'));
  const encoded=line.match(/^"__PTB_CONFIG__","0","(.*)","false"$/)[1].replaceAll('""','"');
  return JSON.parse(encoded);
 }
 async function apply(name){await scope.getByLabel('发声类型预设').selectOption(name);await click('应用预设');await click('应用并覆盖');await page.waitForFunction(()=>!document.querySelector('[role=dialog]'));}
 const near=(a,b)=>assert(Math.abs(a-b)<1e-7,`${a} vs ${b}`);
 await click('F0 Hz');await scope.getByLabel('曲线覆盖',{exact:true}).fill('100,160,120');await click('应用覆盖');
 const base=await snapshot();const svg=scope.getByRole('img',{name:'F0 参数曲线',exact:true});
 const initial=await svg.locator('path.curve').getAttribute('d');
 await apply('假声');const high=await snapshot();assert(high.f0_transform.offset_hz>0);assert.equal(high.curves.F0.override,null);
 assert.notEqual(await svg.locator('path.curve').getAttribute('d'),initial);
 assert.equal(await scope.getByLabel('F0 下限',{exact:true}).inputValue(),String(high.f0_range[0]));
 // Plot x stays fixed; each y coordinate uses the actual shifted Hz and new axis.
 const geometry=await svg.evaluate(s=>({height:s.viewBox.baseVal.height,path:s.querySelector('path.curve').getAttribute('d')}));
 const xy=[...geometry.path.matchAll(/[ML]([^, ]+),([^ ]+)/g)].map(m=>[Number(m[1]),Number(m[2])]);
 high.curves.F0.points.forEach((p,i)=>near(xy[i][1],geometry.height-40-(p[1]-high.f0_range[0])/(high.f0_range[1]-high.f0_range[0])*(geometry.height-70)));
 await apply('假声');assert.deepEqual((await snapshot()).curves.F0,high.curves.F0);
 // Real Shift-draw on the transformed contour must remain editable.
 const box=await svg.boundingBox();await page.keyboard.down('Shift');await page.mouse.move(box.x+box.width*.4,box.y+box.height*.45);await page.mouse.down();await page.mouse.move(box.x+box.width*.55,box.y+box.height*.5,{steps:3});await page.mouse.up();await page.keyboard.up('Shift');
 const edited=await snapshot();assert.notDeepEqual(edited.curves.F0,high.curves.F0);
 await apply('气声');const restored=await snapshot();assert.equal(restored.f0_transform.offset_hz,0);
 edited.curves.F0.points.forEach(([t,v],i)=>{near(restored.curves.F0.points[i][0],t);near(restored.curves.F0.points[i][1],v-high.f0_transform.offset_hz);});
 await apply('嘎裂');const low=await snapshot();assert(low.f0_transform.offset_hz<0);await apply('嘎裂');assert.deepEqual((await snapshot()).curves.F0,low.curves.F0);
 await apply('耳语');const whisper=await snapshot();assert.equal(whisper.curves.AV.override,0);assert.equal(whisper.curves.AH.override,40);
 low.curves.F0.points.forEach(([,v],i)=>near(whisper.curves.F0.points[i][1],v-low.f0_transform.offset_hz));
 await click('AV dB');const av=scope.getByRole('img',{name:'AV 参数曲线',exact:true});assert((await av.textContent()).includes('80.0'));assert((await av.textContent()).includes('0.0'));
 await click('F0 Hz');await apply('假声');const saved=await snapshot();await click('保存参数草稿');
 await page.getByLabel('关闭 语音合成',{exact:true}).click();await scope.waitFor({state:'detached'});
 await page.locator('nav').getByRole('button',{name:'语音合成',exact:true}).click();await scope.waitFor();
 const reopened=await snapshot();assert.deepEqual(reopened.f0_transform,saved.f0_transform);assert.deepEqual(reopened.curves.F0,saved.curves.F0);
 await apply('常态浊声');assert.equal((await snapshot()).f0_transform.offset_hz,0);
 // Reject obsolete scale without replacing current data.
 const current=await snapshot();await scope.locator('input[accept=".csv,.json"]').setInputFiles({name:'old.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify({...current,schema_version:'m06/1'}))});
 assert((await scope.innerText()).includes('旧版 AV'));assert.deepEqual(await snapshot(),current);
 await page.screenshot({path:path.join(out,'r2-f0.png'),fullPage:true});
 await fs.writeFile(path.join(out,'r2-presets.json'),JSON.stringify({success:true,base,high,edited,restored,low,whisper,reopened},null,2));
 checks.push('R2 real UI: 0–80 AV, F0 constant translation, chart Hz, editable shifted path, noncumulative presets, neutral restoration, draft reopen and old-scale rejection');
};
