const assert=require('node:assert/strict'),path=require('node:path');
module.exports=async({page,cdp,out,report})=>{
 let counter=0;const js=expression=>page.evaluate(expression),pause=ms=>page.waitForTimeout(ms);
 const click=async label=>{assert(await js(`(()=>{const e=[...document.querySelectorAll('button')].find(e=>e.offsetParent&&!e.disabled&&(e.textContent.trim()===${JSON.stringify(label)}||e.getAttribute('aria-label')===${JSON.stringify(label)}));if(!e)return false;e.click();return true})()`),'Missing '+label);await pause(150);};
 const fill=async(selector,value,event='input')=>{assert(await js(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});if(!e)return false;e.value=${JSON.stringify(value)};e.dispatchEvent(new Event(e.tagName==='SELECT'?'change':${JSON.stringify(event)},{bubbles:true}));return true})()`));await pause(150);};
 const save=async(label,ext,configure)=>{const pending=page.waitForEvent('download');await click(label);if(['SVG','PNG'].includes(label)){await page.getByRole('dialog').waitFor();if(configure)await configure();await click('导出图片');}const download=await pending,file=path.join(out,'browser-options-'+(++counter)+ext);await download.saveAs(file);await pause(150);return file;};
 await require('./geolinguistics-native-options.cjs')({call:(method,params)=>cdp.send(method,params),js,click,fill,pause,save,out,report});
};
