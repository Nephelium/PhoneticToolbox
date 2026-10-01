module.exports=async(page,out,prefix,selectors)=>{
 const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),values=[];
 for(const size of [{width:1920,height:1000},{width:2560,height:1360},{width:3840,height:2080}]){
  await page.setViewportSize(size);await page.waitForTimeout(250);await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
  const geometry={};for(const selector of selectors)geometry[selector]=await page.locator(selector).filter({visible:true}).first().evaluate(e=>({width:e.clientWidth,height:e.clientHeight,scroll:e.scrollHeight,font:getComputedStyle(e).fontSize}));values.push({size,geometry});await page.screenshot({path:path.join(out,prefix+'-'+size.width+'.png')});
 }
 const samples=[];for(let i=0;i<20;i++){const begin=Date.now();if(selectors[0].includes('m13'))await page.getByLabel('待转换汉字文本').fill(i%2?'银行花，普通话国际音标。':'银行花，普通话。');else if(selectors[0].includes('phonology'))await page.getByLabel('跳过首行（表头）').click();else await page.getByLabel('随机种子').fill(String(42+i));await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));samples.push(Date.now()-begin);}await fs.writeFile(path.join(out,'interaction-timing.json'),JSON.stringify({scope:'20 source-frontend parameter/text changes to two animation frames; includes automation input overhead; not cold load or scientific task completion',samples},null,2));
 const area=selectors[0];assert(values[2].geometry[area].width>values[0].geometry[area].width);assert(values[2].geometry[area].height>values[0].geometry[area].height);assert.equal(values[2].geometry[area].font,values[0].geometry[area].font);await fs.writeFile(path.join(out,prefix+'-sizes.json'),JSON.stringify(values,null,2));
};
