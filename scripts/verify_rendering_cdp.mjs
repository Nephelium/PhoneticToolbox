/** Owned visible final EXE only. No GPU flags/emulation and no browser shutdown. */
import fs from 'node:fs';import path from 'node:path';
const [port,output]=process.argv.slice(2);if(!port||!output)throw Error('port and owned evidence directory required');
const out=path.resolve(output);fs.mkdirSync(path.join(out,'frames'),{recursive:true});
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
async function connect(url){
 const socket=new WebSocket(url);await new Promise((r,j)=>{socket.onopen=r;socket.onerror=j});let id=0;const pending=new Map(),listeners=new Map();
 socket.onmessage=e=>{const v=JSON.parse(e.data);if(v.method)listeners.get(v.method)?.(v.params);const p=pending.get(v.id);if(!p)return;pending.delete(v.id);clearTimeout(p.timer);v.error?p.reject(Error(JSON.stringify(v.error))):p.resolve(v.result)};
 const call=(method,params={})=>{const n=++id;return new Promise((resolve,reject)=>{pending.set(n,{resolve,reject,timer:setTimeout(()=>reject(Error(method+' timed out')),30000)});socket.send(JSON.stringify({id:n,method,params}))})};
 const evaluate=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true,userGesture:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
 return {call,evaluate,on:(name,fn)=>listeners.set(name,fn),close:()=>socket.close()};
}
let tabs;for(let i=0;i<180;i++){try{tabs=await(await fetch(`http://127.0.0.1:${port}/json`)).json();if(tabs.some(t=>t.url.startsWith('ptbapp:')))break}catch{}await sleep(500)}
const p=await connect(tabs.find(t=>t.url.startsWith('ptbapp:')).webSocketDebuggerUrl);
const until=async code=>{for(let i=0;i<180;i++){if(await p.evaluate(code))return;await sleep(150)}throw Error(code)};
const button=async name=>until(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.offsetParent&&(b.getAttribute('aria-label')||b.textContent.trim())===${JSON.stringify(name)});if(!b||b.disabled)return false;b.click();return true})()`);
const report={success:false,scope:'Visible final EXE; normal/maximized native windows; default GPU; CDP screenshots plus separate native screen inspection',frames:[],layers:[],errors:[]};
let layers=[];
try{
 await until('!!document.querySelector("#tool-sidebar")');
 await p.evaluate(`localStorage.setItem('ptb.v3.theme','"light"');localStorage.setItem('ptb.v3.palette','"catppuccin"');localStorage.setItem('ptb.v3.buttonAppearance',JSON.stringify({mode:'auto',effects:true,material:'glass',glassOpacity:35,glassBlur:12}));localStorage.setItem('ptb.v3.fonts.v1.desktop',JSON.stringify({version:1,zh:'KaiTi',latin:'Times New Roman',bodySize:15,figure:{follow:false,zh:'SimSun',latin:'Times New Roman',size:12}}));location.reload()`);
 await sleep(900);await button('设置');await until('!!document.querySelector("input[aria-label=上传背景图]")');
 const uploadName='render-test-'+Date.now()+'.png';
 const priorBitmap=await p.evaluate(`getComputedStyle(document.documentElement).getPropertyValue('--button-glass-image')`);
 await p.evaluate(`(()=>{const c=document.createElement('canvas');c.width=1800;c.height=1000;const x=c.getContext('2d');for(let y=0;y<1000;y+=40)for(let v=0;v<1800;v+=40){x.fillStyle=(v+y)%80?'#4477aa':'#eebbaa';x.fillRect(v,y,40,40);}c.toBlob(blob=>{const d=new DataTransfer();d.items.add(new File([blob],${JSON.stringify(uploadName)},{type:'image/png'}));const e=document.querySelector('input[aria-label="上传背景图"]');e.files=d.files;e.dispatchEvent(new Event('change',{bubbles:true}));});})()`);
 await until(`getComputedStyle(document.documentElement).getPropertyValue('--button-glass-image').includes('blob:')&&getComputedStyle(document.documentElement).getPropertyValue('--button-glass-image')!==${JSON.stringify(priorBitmap)}`);
 await until(`document.body.innerText.includes(${JSON.stringify(uploadName)})`);
 // The upload label changes when IndexedDB commit/show finishes. Allow its
 // one debounced rasterization to complete before checking for feedback.
 await sleep(500);
 const imageA=await p.evaluate(`getComputedStyle(document.documentElement).getPropertyValue('--button-glass-image')`);await sleep(700);
 if(imageA!==await p.evaluate(`getComputedStyle(document.documentElement).getPropertyValue('--button-glass-image')`))throw Error('Wallpaper must settle without a render feedback loop');
 // Exercise real blur controls and compare decoded image pixels, not just URLs.
 const bitmapStats=()=>p.evaluate(`(async()=>{const u=JSON.parse(getComputedStyle(document.documentElement).getPropertyValue('--button-glass-image').trim().slice(4,-1)),i=new Image();i.src=u;await i.decode();const c=document.createElement('canvas');c.width=i.width;c.height=i.height;const x=c.getContext('2d');x.drawImage(i,0,0);const d=x.getImageData(0,0,c.width,c.height).data;let energy=0;for(let k=4;k<d.length;k+=4)energy+=Math.abs(d[k]-d[k-4]);return {energy,width:c.width,height:c.height}})()`);
 for(const blur of [0,40]){
  const before=await p.evaluate(`getComputedStyle(document.documentElement).getPropertyValue('--button-glass-image')`);
  await p.evaluate(`(()=>{const e=document.querySelector('#button-glass-blur');e.value=${blur};e.dispatchEvent(new Event('input',{bubbles:true}));})()`);
  await until(`getComputedStyle(document.documentElement).getPropertyValue('--button-glass-image')!==${JSON.stringify(before)}`);report['blur'+blur]=await bitmapStats();
 }
 if(!(report.blur40.energy<report.blur0.energy))throw Error('Blur must reduce image edge energy');
 await p.evaluate(`(()=>{const e=document.querySelector('#button-glass-blur');e.value=12;e.dispatchEvent(new Event('input',{bubbles:true}));})()`);
 await button('国际音标表Plus');await until("document.querySelector('.m17-chart-viewport')?.scrollHeight>700");
 const version=await(await fetch(`http://127.0.0.1:${port}/json/version`)).json(),browser=await connect(version.webSocketDebuggerUrl);
 try{report.gpu=await browser.call('SystemInfo.getInfo')}finally{browser.close()}
 p.on('LayerTree.layerTreeDidChange',e=>{layers=e.layers||[]});await p.call('LayerTree.enable');
 p.on('Inspector.targetCrashed',()=>report.errors.push('targetCrashed'));
 for(const state of ['normal','maximized']){
  if(state==='maximized')await button('最大化窗口');await sleep(500);
  for(const theme of ['light','dark'])for(const font of ['SimSun','Microsoft YaHei','KaiTi'])for(const size of [10,12,14,15,18,20,24]){
   await p.evaluate(`document.documentElement.dataset.theme=${JSON.stringify(theme)};document.documentElement.style.setProperty('--font',${JSON.stringify('"Times New Roman","'+font+'"')});document.documentElement.style.setProperty('--body-size','${size}px')`);await sleep(180);
   for(const scroll of [350,0]){
    await p.evaluate(`document.querySelector('.m17-chart-viewport').scrollTop=${scroll}`);await sleep(65);
    const snapshot=await p.evaluate(`(()=>{const s=document.querySelector('.m17-chart-viewport'),clip=s.getBoundingClientRect();const boxes=[...document.querySelectorAll('.nav-item,.m17-chart-viewport button')].filter(e=>{const r=e.getBoundingClientRect();return e.offsetParent&&r.top>100&&r.bottom<innerHeight-100&&(!e.closest('.m17-chart-viewport')||(r.top>clip.top+35&&r.bottom<clip.bottom))}).filter((e,i,a)=>e.matches('.nav-item')?a.filter(x=>x.matches('.nav-item')).indexOf(e)<6:a.filter(x=>!x.matches('.nav-item')).indexOf(e)<6).map(e=>{const r=e.getBoundingClientRect(),c=document.createElement('canvas').getContext('2d');c.fillStyle=getComputedStyle(e.querySelector('.m17-ipa')||e).color;c.fillRect(0,0,1,1);return {text:e.textContent,box:[r.x,r.y,r.width,r.height],color:[...c.getImageData(0,0,1,1).data].slice(0,3)}});const nav=document.querySelector('.nav-item.selected');nav.focus({focusVisible:true});const c=getComputedStyle(nav);return {viewport:[innerWidth,innerHeight,devicePixelRatio],scroll:s.scrollTop,boxes,nav:{border:c.borderColor,outline:c.outlineStyle,shadow:c.boxShadow},filtered:[...document.querySelectorAll('*')].filter(e=>getComputedStyle(e).backdropFilter!=='none').length}})()`);
    if(snapshot.filtered||snapshot.nav.border!=='rgba(0, 0, 0, 0)'||snapshot.nav.outline!=='none'||snapshot.nav.shadow!=='none'||(scroll>0&&snapshot.scroll<=0))throw Error(JSON.stringify(snapshot));
    const file=`frames/${state}-${theme}-${font.replaceAll(' ','_')}-${size}-${scroll}.png`;
    fs.writeFileSync(path.join(out,file),Buffer.from((await p.call('Page.captureScreenshot',{format:'png'})).data,'base64'));
    report.frames.push({state,theme,font,size,file,...snapshot});
   }
  }
  report.layers.push({state,count:layers.length});console.log(state+' visible font/scroll matrix completed');
 }
 // Real wheel input, including the asynchronous scrolling path.
 for(let i=0;i<48;i++)await p.call('Input.dispatchMouseEvent',{type:'mouseWheel',x:900,y:420,deltaX:0,deltaY:i%12<6?140:-140});
 await sleep(500);fs.writeFileSync(path.join(out,'wheel-final.png'),Buffer.from((await p.call('Page.captureScreenshot',{format:'png'})).data,'base64'));
 if(report.errors.length)throw Error(JSON.stringify(report.errors));report.success=true;
}finally{fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2));p.close()}
console.log(JSON.stringify({success:report.success,frames:report.frames.length,layers:report.layers}));
