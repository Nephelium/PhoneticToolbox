// Owned Vite and headless Chrome. No DB, personal browser or global installation.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const out=path.join(root,'output/validation/fonts','chrome-'+Date.now());await fs.mkdir(out,{recursive:true});
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0}});await server.listen();
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
 const context=await browser.newContext({viewport:{width:1440,height:1000},acceptDownloads:true}),page=await context.newPage(),checks=[],errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 try{
  const base=server.resolvedUrls.local[0];await page.goto(base);await page.getByRole('button',{name:'设置',exact:true}).click();
  await page.getByLabel('中文字体',{exact:true}).fill('SimSun');await page.getByLabel('英文与数字字体',{exact:true}).fill('Times New Roman');await page.getByLabel('代码与等宽字体',{exact:true}).fill('Consolas');
  assert.equal(await page.getByLabel('IPA 字体').getAttribute('readonly'),'');
  await page.getByRole('button',{name:'应用字体',exact:true}).click();await page.getByRole('status').filter({hasText:'字体已应用'}).waitFor();
  const selected=await page.evaluate(async()=>{const f=await import('/src/state/fonts.ts');return {p:f.preferences.value,payload:f.fontPayload.value};});
  assert.equal(selected.p.ipa,'Doulos SIL');assert.equal(selected.payload.resolved.zh,'SimSun');assert.equal(selected.payload.resolved.latin,'Times New Roman');
  await page.screenshot({path:path.join(out,'settings-light.png'),fullPage:true});checks.push('settings apply: SimSun / Times New Roman / Consolas, IPA fixed');
  await page.getByLabel('英文与数字字体',{exact:true}).fill('PTB unavailable 123');await page.getByRole('button',{name:'应用字体',exact:true}).click();await page.getByRole('status').filter({hasText:'不可用'}).waitFor();
  assert.equal(await page.evaluate(async()=> (await import('/src/state/fonts.ts')).preferences.value.latin),'Times New Roman');checks.push('missing font rejected without changing active preferences');
  await page.getByRole('button',{name:'取消字体编辑'}).click();await page.reload();await page.getByRole('button',{name:'设置',exact:true}).click();
  await page.waitForFunction(()=>document.documentElement.style.getPropertyValue('--font').includes('SimSun'));
  assert.equal(await page.getByLabel('英文与数字字体',{exact:true}).inputValue(),'Times New Roman');checks.push('preferences survive reload');
  await page.evaluate(()=>document.documentElement.dataset.theme='dark');await page.setViewportSize({width:390,height:900});await page.screenshot({path:path.join(out,'settings-dark-narrow.png'),fullPage:true});
  assert(await page.evaluate(()=>document.documentElement.scrollWidth)<=391);checks.push('dark narrow settings fit viewport');
  await page.setViewportSize({width:1440,height:1000});await page.goto(base+'tests/m02-export.html');await page.getByRole('button',{name:'元音ɑ̃.wav',exact:true}).click();await page.locator('.parameter-curve').first().waitFor();
  async function config(zh,latin){await page.evaluate(async({zh,latin})=>{const f=await import('/src/state/fonts.ts'),p=(await import('/src/design/fonts.ts')).defaults();p.zh=zh;p.latin=latin;await f.setFonts(p);},{zh,latin});}
  async function save(name,button){const waiting=page.waitForEvent('download');await page.locator('.parameter-figure').first().getByRole('button',{name:button,exact:true}).click();const d=await waiting;await d.saveAs(path.join(out,name));}
  await config('SimSun','Times New Roman');await save('simsun-times.png','保存整幅 PNG');await save('simsun-times.svg','保存当前图');
  await config('KaiTi','Arial');await save('kaiti-arial.png','保存整幅 PNG');
  const svg=await fs.readFile(path.join(out,'simsun-times.svg'),'utf8');assert(svg.includes('data:font/ttf;base64,'));assert(svg.includes('SIL OPEN FONT LICENSE'));assert(svg.includes('PTB-Doulos'));
  assert.notDeepEqual(await fs.readFile(path.join(out,'simsun-times.png')),await fs.readFile(path.join(out,'kaiti-arial.png')));checks.push('actual M02 PNG changes fonts, editable SVG carries Doulos and license');
  const ipa=await page.evaluate(async()=>{
   const {paintSvg}=await import('/src/design/svg-fonts.ts');const canvas=document.createElement('canvas');canvas.width=400;canvas.height=100;
   await paintSvg(canvas,'<svg xmlns="http://www.w3.org/2000/svg" width="400" height="100" viewBox="0 0 400 100"><rect width="400" height="100" fill="white"/><text x="10" y="60" style="font-family:PTB-Doulos;font-size:30px;fill:black">aː tʰ ɕ ŋ ə ã n̩</text></svg>',400,100);
   const expected=document.createElement('canvas');expected.width=400;expected.height=100;const ctx=expected.getContext('2d');ctx.fillStyle='white';ctx.fillRect(0,0,400,100);ctx.fillStyle='black';ctx.font='30px PTB-Doulos';ctx.fillText('aː tʰ ɕ ŋ ə ã n̩',10,60);
   return {actual:canvas.toDataURL(),expected:expected.toDataURL()};
  });assert.equal(ipa.actual,ipa.expected);await fs.writeFile(path.join(out,'ipa-reference.png'),Buffer.from(ipa.actual.split(',')[1],'base64'));checks.push('IPA raster exactly matches independent Doulos Canvas reference');
  await page.evaluate(async()=>{const f=await import('/src/state/fonts.ts');await f.selectFontOwner('font-test-alice');const p=(await import('/src/design/fonts.ts')).defaults();p.zh='KaiTi';await f.setFonts(p);await f.selectFontOwner('font-test-bob');if(f.preferences.value.zh==='KaiTi')throw Error('owner preferences leaked');await f.selectFontOwner('font-test-alice');if(f.preferences.value.zh!=='KaiTi')throw Error('owner preference lost');});checks.push('owner switching isolates and restores fonts');
  await page.goto(base);await page.getByRole('button',{name:'设置',exact:true}).click();
  await page.getByRole('checkbox',{name:'图表与导出跟随全局字体'}).uncheck();await page.getByLabel('图表中文字体',{exact:true}).fill('KaiTi');await page.getByLabel('图表英文字体',{exact:true}).fill('Arial');await page.getByLabel('图表基础字号').fill('24');await page.getByRole('button',{name:'应用字体',exact:true}).click();await page.getByRole('status').filter({hasText:'字体已应用'}).waitFor();
  let snapshot=await page.evaluate(async()=> (await import('/src/state/fonts.ts')).exportFontSnapshot());assert.equal(snapshot.zh,'KaiTi');assert.equal(snapshot.latin,'Arial');assert.equal(snapshot.size_px,24);checks.push('independent figure fonts and 24px snapshot via real settings');
  await page.getByRole('button',{name:'恢复默认',exact:true}).click();await page.getByRole('button',{name:'取消字体编辑',exact:true}).click();assert.equal(await page.getByLabel('图表基础字号').inputValue(),'24');await page.getByRole('button',{name:'恢复默认',exact:true}).click();await page.getByRole('button',{name:'应用字体',exact:true}).click();await page.getByRole('status').filter({hasText:'字体已应用'}).waitFor();assert.equal(await page.getByLabel('图表基础字号').inputValue(),'12');checks.push('restore default remains draft until apply, cancel retains active state');
  await page.goto(base+'tests/m02-export.html');await page.getByRole('button',{name:'元音ɑ̃.wav',exact:true}).click();await page.locator('.parameter-curve').first().waitFor();
  await page.evaluate(async()=>{const f=await import('/src/state/fonts.ts'),p=(await import('/src/design/fonts.ts')).defaults();p.zh='SimSun';p.latin='Times New Roman';p.figure.size=24;await f.setFonts(p);});await page.setViewportSize({width:390,height:900});await save('large-font.png','保存整幅 PNG');await page.screenshot({path:path.join(out,'large-font-narrow.png'),fullPage:true});assert(await page.evaluate(()=>document.documentElement.scrollWidth)<=391);checks.push('24px chart in narrow viewport and real PNG');
  const printSizes=await page.evaluate(async()=>{const chart=document.querySelector('.parameter-chart'),{wholeFigureSvg}=await import('/src/modules/parameter-display/export.ts');const result=wholeFigureSvg({chart,waveform:document.querySelector('.m02-page'),title:'字体核验',start:0,end:1,plotLeft:Number(chart.querySelector('.left-tick').getAttribute('x'))+9,plotRight:Number(chart.querySelector('.right-tick').getAttribute('x'))-9});const tree=new DOMParser().parseFromString(result.text,'image/svg+xml');return {axis:tree.querySelector('.left-tick').style.fontSize,ipa:tree.querySelector('.ipa-text').style.fontSize,legend:tree.querySelector('.curve-legend text').style.fontSize};});assert.deepEqual(printSizes,{axis:'24px',ipa:'24px',legend:'24px'});checks.push('whole PNG snapshot has identical axis, IPA and legend font sizes');
  await page.goto(base+'tests/font-textgrid.html');await page.locator('.textgrid-interval .ipa-text').waitFor();await page.waitForFunction(()=>document.documentElement.style.getPropertyValue('--font').length>0);assert((await page.locator('.textgrid-interval .ipa-text').first().evaluate(e=>getComputedStyle(e).fontFamily)).includes('PTB-Doulos'));checks.push('real TextGrid component fixes complete ASCII and extended IPA label');
  assert.deepEqual(errors,[]);await page.screenshot({path:path.join(out,'m02.png'),fullPage:true});await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success:true,checks,errors},null,2));console.log(out);
 }catch(e){console.error(out);await page.screenshot({path:path.join(out,'failed.png'),fullPage:true});throw e;}
 finally{await context.close();await browser.close();await server.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
