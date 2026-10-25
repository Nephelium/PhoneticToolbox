// Connect only to an owned Qt EXE. No browser is launched by this regression.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const [port,out]=process.argv.slice(2),pause=ms=>new Promise(r=>setTimeout(r,ms));
const report={success:false,scope:'Actual Windows EXE/Qt, default GPU, isolated state; theme/background/count matrix and rendered screenshot pixels',cases:[],errors:[]};
let socket,sequence=0;const pending=new Map();
async function call(method,params={}){return new Promise((resolve,reject)=>{const id=++sequence,timer=setTimeout(()=>{pending.delete(id);reject(Error(method+' timeout'));},60000);pending.set(id,{resolve,reject,timer});socket.send(JSON.stringify({id,method,params}));});}
async function js(expression){const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true,userGesture:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
async function until(expression){for(let i=0;i<400;i++){if(await js(expression))return;await pause(150);}throw Error(expression);}
async function select(selector,value){await js(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});if(!e)throw Error('Missing select');e.value=${JSON.stringify(value)};e.dispatchEvent(new Event('change',{bubbles:true}));})()`);}
async function image(name){const r=await call('Page.captureScreenshot',{format:'png'});await fs.writeFile(path.join(out,name),Buffer.from(r.data,'base64'));}
async function inspect(evaluate=js){return evaluate(`(()=>{
 const maps=[...document.querySelectorAll('.geo-map')].filter(e=>e.offsetParent),first=maps[0],canvases=[...first.querySelectorAll('.ol-layers canvas')];
 const layers=canvases.map(c=>{const r=c.getBoundingClientRect();return {c,r,d:c.getContext('2d').getImageData(0,0,c.width,c.height).data};}),samples=[];
 const base=layers[0],land=(v,i)=>v[i]===248&&v[i+1]===248&&v[i+2]===241&&v[i+3]===255;
 const stepX=Math.max(4,Math.floor(base.c.width/50)),stepY=Math.max(4,Math.floor(base.c.height/35));
 for(let y=12;y<base.c.height-12&&samples.length<12;y+=stepY)for(let x=12;x<base.c.width-12&&samples.length<12;x+=stepX){
  const at=(a,b)=>(b*base.c.width+a)*4;if(![at(x,y),at(x-3,y),at(x+3,y),at(x,y-3),at(x,y+3)].every(i=>land(base.d,i)))continue;
  const sx=base.r.x+x/base.c.width*base.r.width,sy=base.r.y+y/base.c.height*base.r.height;
  if(document.elementsFromPoint(sx,sy).some(e=>e.closest('.ol-control,.map-instructions,.map-attributions,.map-hover')))continue;
  const covered=layers.slice(1).some(l=>{const px=Math.floor((sx-l.r.x)/l.r.width*l.c.width),py=Math.floor((sy-l.r.y)/l.r.height*l.c.height);return px>=0&&py>=0&&px<l.c.width&&py<l.c.height&&l.d[(py*l.c.width+px)*4+3]>0;});
  if(!covered)samples.push({x:Math.round(sx*devicePixelRatio),y:Math.round(sy*devicePixelRatio),rgb:[248,248,241]});
 }
 const scale=first.querySelector('.ol-scale-line'),ink=scale.querySelector('.ol-scale-line-inner');
 return {count:maps.length,visible:document.visibilityState,canvasCount:canvases.length,canvasSize:[base.c.width,base.c.height],sampleGrid:[stepX,stepY],backgrounds:maps.flatMap(m=>[...m.querySelectorAll('.ol-layers canvas')].map(c=>getComputedStyle(c).backgroundColor)),surface:getComputedStyle(first).backgroundColor,scale:{background:getComputedStyle(scale).backgroundColor,ink:getComputedStyle(ink).color},samples};
})()`);}
function scaleContrast(scale){const luminance=value=>{const c=value.match(/[\d.]+/g).slice(0,3).map(Number).map(v=>v/255).map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4);return c[0]*.2126+c[1]*.7152+c[2]*.0722;};const a=luminance(scale.background),b=luminance(scale.ink);return (Math.max(a,b)+.05)/(Math.min(a,b)+.05);}
async function verifyMatrix({call,js,pause,image,report,select,until,directory}){
 report.seedPhase='real-file-upload';
 const fixture=await js("(()=>{const c=document.createElement('canvas');c.width=640;c.height=360;const p=c.getContext('2d');p.fillStyle='#d5e6f4';p.fillRect(0,0,640,360);p.fillStyle='#ebc8b6';p.beginPath();p.arc(440,180,160,0,Math.PI*2);p.fill();return c.toDataURL('image/png').split(',')[1]})()");
 const fixturePath=path.resolve(directory||out,'QA-wallpaper.png');await fs.writeFile(fixturePath,Buffer.from(fixture,'base64'));
 await js('[...document.querySelectorAll("#tool-sidebar button")].find(e=>e.textContent.trim()==="设置").click()');await until('!!document.querySelector("#settings-tab-appearance")');await js('document.querySelector("#settings-tab-appearance").click()');
 const doc=await call('DOM.getDocument'),input=await call('DOM.querySelector',{nodeId:doc.root.nodeId,selector:'input[type=file][aria-label="上传背景图"]'});assert(input.nodeId);await call('DOM.setFileInputFiles',{nodeId:input.nodeId,files:[fixturePath]});
 await until('document.documentElement.dataset.backgroundImage==="on"&&document.querySelector(".background-name")?.textContent.includes("QA-wallpaper.png")');
 report.wallpaper=await js('({image:document.querySelector(".app-background-layer").style.backgroundImage,name:document.querySelector(".background-name").textContent,persisted:true,via:"actual Settings file input"})');await image('wallpaper-settings.png');
 await js('document.querySelector("#tool-sidebar button[title=语言地图]").click()');await until('!!document.querySelector(".atlas-workbench")');await js('[...document.querySelectorAll(".mode-tabs button")].find(e=>e.textContent.trim()==="语言现象").click()');await select('.geography-toolbar label select','district');
 for(const count of [1,2,4]){await select('.map-heading label select',String(count));await until(`document.querySelectorAll('.geo-map').length===${count}&&!document.querySelector('.map-loading')`);await pause(700);
  for(const theme of ['light','dark'])for(const wallpaper of [false,true]){
   await select('[aria-label="省级地区"]','province:430000');await pause(500);
   await js(`document.documentElement.dataset.theme=${JSON.stringify(theme)};document.documentElement.dataset.backgroundImage=${JSON.stringify(wallpaper?'on':'off')}`);
   await js(`document.querySelector('.app-background-layer').style.visibility=${JSON.stringify(wallpaper?'visible':'hidden')}`);
   const row={count,theme,wallpaper,states:[]};report.cases.push(row);
   const p=await js('(()=>{document.querySelector(".map-target").scrollIntoView({block:"center"});const r=document.querySelector(".map-target").getBoundingClientRect();return {x:r.x+r.width*.5,y:r.y+r.height*.5,w:r.width,h:r.height}})()');
   for(let n=0;n<4;n++){
    if(n===2){await call('Input.dispatchMouseEvent',{type:'mousePressed',button:'left',buttons:1,clickCount:1,x:p.x,y:p.y});for(let i=1;i<=12;i++)await call('Input.dispatchMouseEvent',{type:'mouseMoved',button:'left',buttons:1,x:p.x+p.w*.08*i/12,y:p.y+p.h*.035*i/12});await call('Input.dispatchMouseEvent',{type:'mouseReleased',button:'left',buttons:0,clickCount:1,x:p.x+p.w*.08,y:p.y+p.h*.035});}
    else await call('Input.dispatchMouseEvent',{type:'mouseWheel',x:p.x,y:p.y,deltaX:0,deltaY:n===1?160:-160});
    await pause(40);const animation=await js('({canvasCount:document.querySelectorAll(".geo-map .ol-layers canvas").length/document.querySelectorAll(".geo-map").length,backgrounds:[...document.querySelectorAll(".geo-map .ol-layers canvas")].map(c=>getComputedStyle(c).backgroundColor)})');assert(animation.backgrounds.every(v=>v==='rgba(0, 0, 0, 0)'));await pause(610);const state=await inspect(js);state.animation=animation;row.states.push(state);assert.equal(state.count,count);assert.equal(state.visible,'visible');assert(state.backgrounds.every(v=>v==='rgba(0, 0, 0, 0)'),`Opaque map Canvas: ${state.backgrounds}`);assert(state.samples.length>=8,'Expected independent land samples');
    state.scaleContrast=scaleContrast(state.scale);assert(state.scaleContrast>=4.5,`Unreadable map scale: ${JSON.stringify(state.scale)}`);
    if(n===0||n===2){state.screenshot=`map-${count}-${theme}-${wallpaper?'wallpaper':'plain'}-${n}.png`;await image(state.screenshot);}
   }
  }
 }
 assert(report.cases.some(c=>c.wallpaper&&c.states.some(s=>s.canvasCount>1||s.animation?.canvasCount>1)),'Regression must exercise separate canvases');assert.deepEqual(report.errors,[]);
 await select('.map-heading label select','1');
 await select('[aria-label="省级地区"]','province:330000');
 await until('!!document.querySelector("[aria-label=下一级地区]")');await select('[aria-label="下一级地区"]','city:330200');
 await until('!!document.querySelector("[aria-label=县级地区]")');await select('[aria-label="县级地区"]','district:330225');
 await until('!document.querySelector(".map-loading")');await pause(700);
 report.coastline={primary:'district',extra:await js('[...document.querySelectorAll(".boundary-options input:checked")].map(e=>e.value)')};
 assert.deepEqual(report.coastline.extra,[]);await image('xiangshan-district.png');
 await select('.geography-toolbar label select','province');await pause(700);await image('xiangshan-province.png');
 await select('.geography-toolbar label select','district');await pause(700);await image('xiangshan-district-restored.png');
}
module.exports={verifyMatrix};
if(require.main===module)(async()=>{try{
 let target;for(let i=0;i<480;i++){try{target=(await(await fetch('http://127.0.0.1:'+port+'/json')).json()).find(t=>t.url.startsWith('ptbapp:'));if(target)break;}catch{}await pause(500);}assert(target,'EXE Qt endpoint unavailable');
 socket=new WebSocket(target.webSocketDebuggerUrl);await new Promise((r,j)=>{socket.onopen=r;socket.onerror=j;});socket.onmessage=e=>{const v=JSON.parse(e.data),p=pending.get(v.id);if(!p){if(v.method==='Runtime.exceptionThrown')report.errors.push(v.params.exceptionDetails);return;}pending.delete(v.id);clearTimeout(p.timer);v.error?p.reject(Error(JSON.stringify(v.error))):p.resolve(v.result);};await call('Runtime.enable');
 await until('!!document.querySelector("#tool-sidebar")');await pause(3200);await js('[...document.querySelectorAll("dialog button")].find(e=>e.textContent.trim()==="知道了")?.click()');
 await js('document.querySelector("#tool-sidebar button[title=语言地图]").click()');await until('!!document.querySelector(".atlas-workbench")');await js('[...document.querySelectorAll(".mode-tabs button")].find(e=>e.textContent.trim()==="语言现象").click()');
 await select('[aria-label="省级地区"]','province:430000');await select('.geography-toolbar label select','district');await until('!document.querySelector(".map-loading")');await pause(700);
 report.environment=await js('({ua:navigator.userAgent,dpr:devicePixelRatio,width:innerWidth,height:innerHeight,protocol:location.protocol})');
 await verifyMatrix({call,js,pause,image,report,select,until});
 report.success=true;
}catch(e){report.error=e.stack;process.exitCode=1;}finally{await fs.writeFile(path.join(out,'background-report.json'),JSON.stringify(report,null,2));if(socket?.readyState===1)socket.close();console.log(JSON.stringify({success:report.success,cases:report.cases.length,error:report.error}));}})();
