// Owned headless Chrome: waveform display/preferences/export, synthetic input only.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
const rgb=hex=>'rgb('+[1,3,5].map(i=>parseInt(hex.slice(i,i+2),16)).join(', ')+')';
async function main(){
 const out=path.join(root,'output/validation/p19-r4','chrome-'+Date.now());await fs.mkdir(out,{recursive:true});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,watch:null}});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
 const page=await browser.newPage({viewport:{width:1920,height:1080}}),checks=[],errors=[];page.on('pageerror',e=>errors.push(e.message));
 try{
  await page.goto(server.resolvedUrls.local[0]);await page.getByRole('button',{name:'设置',exact:true}).click();
  const choice=page.getByLabel('波形线颜色',{exact:true}),stroke=()=>page.locator('.wave-color-preview path').evaluate(el=>getComputedStyle(el).stroke);
  assert.equal(await choice.inputValue(),'blue');await choice.selectOption('theme');
  const palettes=await page.getByLabel('配色方案').locator('option').evaluateAll(es=>es.map(e=>e.value));
  for(const palette of palettes)for(const theme of ['light','dark']){
   await page.getByLabel('配色方案').selectOption(palette);await page.getByRole('button',{name:theme==='light'?'浅色':'深色',exact:true}).click();
   const a=await page.evaluate(()=>{const p=document.createElement('span');document.body.append(p);p.style.color='var(--accent)';const c=getComputedStyle(p).color;p.remove();return c;});assert.equal(await stroke(),a);
  }checks.push('29 palettes × 2 modes: theme-follow preview uses resolved accent immediately');
  await choice.selectOption('custom');const hex=page.getByLabel('波形线 HEX 颜色');await hex.fill('#D24');await hex.blur();assert.equal(await hex.inputValue(),'#dd2244');assert.equal(await stroke(),'rgb(221, 34, 68)');
  await hex.fill('bad');await hex.blur();assert(await page.getByRole('status').filter({hasText:'有效的 HEX'}).isVisible());assert.equal(await stroke(),'rgb(221, 34, 68)');
  await page.getByLabel('波形线颜色色盘').evaluate(el=>{el.value='#b52f91';el.dispatchEvent(new Event('input',{bubbles:true}));});assert.equal(await stroke(),'rgb(181, 47, 145)');
  await page.getByLabel('配色方案').selectOption('everforest');await page.getByRole('button',{name:'深色',exact:true}).click();assert.equal(await stroke(),'rgb(181, 47, 145)');
  await choice.selectOption('theme');await choice.selectOption('custom');assert.equal(await hex.inputValue(),'#b52f91');await page.reload();await page.getByRole('button',{name:'设置',exact:true}).click();assert.equal(await choice.inputValue(),'custom');assert.equal(await hex.inputValue(),'#b52f91');checks.push('color picker, short HEX normalization, invalid input rejection, custom fixed across theme change, retained choice and reload persistence');
  await page.screenshot({path:path.join(out,'settings-custom-dark.png')});
  for(const [width,height,scale] of [[1024,768,150],[390,844,100]]){
   await page.setViewportSize({width,height});await page.evaluate(async scale=>(await import('/src/state/pageZoom.ts')).setPageScale(scale),scale);await hex.scrollIntoViewIfNeeded();assert(await hex.isVisible());assert(await page.evaluate(()=>document.documentElement.scrollWidth<=document.documentElement.clientWidth+2));
  }checks.push('custom controls accessible at narrow width and 150% scale');
  await page.setViewportSize({width:1440,height:1000});await page.goto(server.resolvedUrls.local[0]+'tests/waveform-color.html');await page.waitForFunction(()=>window.qa&&document.querySelector('.wave-line')&&window.qa.monitorColors.length);
  const shapes=()=>page.evaluate(()=>[...document.querySelectorAll('.wave-line,.scientific-trace')].map(el=>({d:el.getAttribute('d'),points:el.getAttribute('points'),transform:el.getAttribute('transform')})));
  const before=await shapes();
  const waveSelectors=['#common .wave-line','#recording .wave-line','#audio-egg-inverse-wave .scientific-trace'];
  for(const mode of ['blue','theme','custom'])for(const theme of ['light','dark']){
   await page.evaluate(async({mode,theme})=>{document.documentElement.dataset.theme=theme;const {paletteTokens}=await import('/src/design/themes.ts');for(const [k,v]of Object.entries(paletteTokens('everforest',theme)))document.documentElement.style.setProperty(k,v);window.qa.set({mode,custom:'#b52f91'});window.qa.redraw();},{mode,theme});
   const expected=await page.evaluate(()=>{const el=document.createElement('span');el.style.color='var(--waveform-color)';document.body.append(el);const value=getComputedStyle(el).color;el.remove();return value;});
   for(const selector of waveSelectors)assert.equal(await page.locator(selector).first().evaluate(el=>getComputedStyle(el).stroke),expected,selector);
   const monitor=await page.evaluate(()=>window.qa.monitorColors.at(-1));const expectedHex=await page.evaluate(()=>getComputedStyle(document.documentElement).getPropertyValue('--waveform-color').trim());assert.equal(monitor,expectedHex);
   const result=await page.evaluate(()=>{const read=svg=>new DOMParser().parseFromString(svg,'image/svg+xml');const s=read(window.qa.scientific().text),w=read(window.qa.wholeSvg().text);return {audio:[...s.querySelectorAll('.scientific-trace[data-export-color="var(--waveform-color)"]')].map(el=>el.style.stroke),other:[...s.querySelectorAll('.scientific-trace[data-export-color]:not([data-export-color="var(--waveform-color)"])')].map(el=>[el.style.stroke,el.dataset.exportColor]),whole:w.querySelector('.wave-line').style.stroke};});
   assert(result.audio.length);for(const c of result.audio)assert.equal(c,rgb(mode==='blue'?'#174b82':expectedHex));assert.equal(result.whole,rgb(mode==='blue'?'#245ab5':expectedHex));for(const [color,exportColor] of result.other)assert.equal(color,rgb(exportColor));
   assert.deepEqual(await shapes(),before);
  }checks.push('shared waveform, M16 recording and M03 inverse audio: 3 modes × 2 themes; M10 canvas redraw; wave geometry unchanged; M02/M03 SVG export follows chosen color and retains legacy blue print and other curve colors');
  await page.evaluate(()=>window.qa.set({mode:'custom',custom:'#b52f91'}));
  for(const kind of ['whole','scientific']){const bytes=await page.evaluate(async kind=>Array.from(new Uint8Array(await (await window.qa[kind+'Png']()).arrayBuffer())),kind);await fs.writeFile(path.join(out,kind+'-custom.png'),Buffer.from(bytes));}
  checks.push('M02 whole figure and M03 scientific PNG exports generated for custom color pixel/DPI readback');
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success:true,checks,errors},null,2));console.log(JSON.stringify({out,checks}));
 }catch(e){await page.screenshot({path:path.join(out,'failed.png')});await fs.writeFile(path.join(out,'failed.json'),JSON.stringify({error:String(e),checks,errors},null,2));throw e;}
 finally{await browser.close();await server.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
