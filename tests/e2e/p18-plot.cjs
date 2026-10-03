// P18: analytic geometry fixture, not generated audio or scientific validation.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const out=path.join(root,'output/validation/p18/plot',String(Date.now()));await fs.mkdir(out,{recursive:true});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0},optimizeDeps:{entries:['tests/p18-plot.html']}});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
 const page=await browser.newPage({viewport:{width:1920,height:1080}}),errors=[],checks=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.addInitScript(()=>{window.__errors=[];window.addEventListener('error',e=>window.__errors.push(e.message));});
 const settle=()=>page.waitForTimeout(150);
 const geometry=()=>page.locator('.scientific-plot>svg').evaluate(svg=>{
  const r=svg.getBoundingClientRect(),clip=svg.querySelector('clipPath rect');return{width:svg.clientWidth,height:svg.clientHeight,vw:svg.viewBox.baseVal.width,vh:svg.viewBox.baseVal.height,font:getComputedStyle(svg.querySelector('text')).fontSize,box:{x:r.x,y:r.y,width:r.width,height:r.height},plot:{x:+clip.getAttribute('x'),y:+clip.getAttribute('y'),width:+clip.getAttribute('width'),height:+clip.getAttribute('height')}};
 });
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/p18-plot.html');await page.getByRole('img').waitFor();await settle();
  for(const [width,height,scale] of [[1920,1080,1],[2560,1440,1],[1920,1080,.7],[1920,1080,1.5],[1280,720,1]]){
   await page.setViewportSize({width,height});await page.evaluate(scale=>{document.documentElement.style.zoom=String(scale);document.documentElement.style.setProperty('--page-scale',String(scale));},scale);await settle();
   const a=await geometry();await settle();const b=await geometry();assert.deepEqual(a,b,'dimensions settle');assert.equal(b.height,b.vh);assert.equal(b.width,b.vw);assert.equal(b.font,'12px');
   // Independently map 75% of [0,1] via SVG clip extent, then verify callback and marker.
   const clientX=b.box.x+(b.plot.x+b.plot.width*.75)/b.vw*b.box.width,clientY=b.box.y+(b.plot.y+b.plot.height*.5)/b.vh*b.box.height;
   await page.mouse.click(clientX,clientY);await settle();
   assert(Math.abs(await page.evaluate(()=>window.__p18Plot.marker.value)-.75)<.002,'click maps to 0.75 seconds');
   const marker=await page.locator('svg g[clip-path]>line').last().getAttribute('x1');assert(Math.abs(+marker-(b.plot.x+b.plot.width*.75))<2);
   await page.mouse.move(clientX,clientY);await page.keyboard.down('Control');await page.mouse.wheel(0,-100);await page.keyboard.up('Control');
   await page.getByRole('img').focus();await page.keyboard.press('ArrowRight');await settle();
   const events=await page.evaluate(()=>window.__p18Plot.events);assert(events.some(e=>e.kind==='zoom'&&e.value===.9));assert(events.some(e=>e.kind==='pan'&&Math.abs(e.value+.1)<1e-9));
   checks.push({name:'settled dimensions, font, seek marker, wheel and keyboard coordinates',viewport:[width,height,scale],geometry:b});
  }
  await page.screenshot({path:path.join(out,'plot.png')});assert.deepEqual(errors,[]);assert.deepEqual(await page.evaluate(()=>window.__errors),[]);
  await fs.writeFile(path.join(out,'report.json'),JSON.stringify({checks,errors},null,2));console.log(JSON.stringify({out,checks:checks.length,errors}));
 }catch(e){await page.screenshot({path:path.join(out,'failed.png')});throw e;}
 finally{await browser.close();await server.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
