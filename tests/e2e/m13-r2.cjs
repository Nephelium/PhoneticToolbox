// M13-R2: owned Chrome, local mapping/formatting and actual downloaded PNGs.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const out=path.join(root,'output/validation/m13-r2','chrome-'+Date.now());await fs.mkdir(out,{recursive:true});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0},optimizeDeps:{entries:['tests/m13-live.html']}});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
 const context=await browser.newContext({viewport:{width:1440,height:900},acceptDownloads:true}),page=await context.newPage();
 const report={success:false,checks:[],layouts:[],pngs:[],errors:[],external:[]};
 page.on('pageerror',e=>report.errors.push(e.message));page.on('request',r=>{const u=new URL(r.url());if(!['127.0.0.1','localhost'].includes(u.hostname)&&!['data:','blob:'].includes(u.protocol))report.external.push(u.href);});
 const input=()=>page.getByLabel('待转换汉字文本'),standard=()=>page.getByLabel('转换标准');
 const idle=()=>page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
 const values=()=>page.locator('.m13-mapped').evaluateAll(es=>es.map(e=>e.dataset.value));
 const shot=name=>page.screenshot({path:path.join(out,name+'.png'),fullPage:true,animations:'disabled'});
 async function png(name,expected){
  await page.evaluate(()=>{window.paint=[];window.originalPaint??=CanvasRenderingContext2D.prototype.fillText;CanvasRenderingContext2D.prototype.fillText=function(text,...args){paint.push({text,font:this.font,color:this.fillStyle});return originalPaint.call(this,text,...args);};});
  const downloadReady=page.waitForEvent('download');await page.getByRole('button',{name:'保存为 PNG',exact:true}).click();
  const file=path.join(out,name+'.png');await (await downloadReady).saveAs(file);
  const paint=await page.evaluate(()=>window.paint);const ipa=paint.filter(x=>x.font.includes('PTB-Doulos'));assert.deepEqual(ipa.map(x=>x.text),expected);
  const hanzi=paint.find(x=>x.text==='女');assert(hanzi);assert(ipa.every(x=>x.color==='#2345ab'));assert.equal(hanzi.color,'#ab4523');assert(hanzi.font.includes('KaiTi'));assert(!ipa.some(x=>x.font.includes('KaiTi')));
  const bytes=await fs.readFile(file);assert.equal(bytes.subarray(1,4).toString(),'PNG');
  const pixels=await page.evaluate(async data=>{const im=new Image();im.src='data:image/png;base64,'+data;await im.decode();const c=document.createElement('canvas');c.width=im.width;c.height=im.height;const cx=c.getContext('2d');cx.drawImage(im,0,0);const p=cx.getImageData(0,0,c.width,c.height).data;let blue=0,brown=0;for(let i=0;i<p.length;i+=4){if(p[i]===35&&p[i+1]===69&&p[i+2]===171)blue++;if(p[i]===171&&p[i+1]===69&&p[i+2]===35)brown++;}return {width:c.width,height:c.height,blue,brown};},bytes.toString('base64'));
  assert(pixels.blue>10&&pixels.brown>10);report.pngs.push({name,paint,pixels});
 }
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/m13-live.html');await input().waitFor();
  await page.evaluate(()=>{localStorage.setItem('ptb.v3.mandarin-ipa.v1.m13-live',JSON.stringify({version:1,text:'旧文本银行',standard:'UntPhesoca严',selectedVariants:{'行_4':1},hanziSize:32}));localStorage.setItem('ptb.v3.m13-settings-open:m13-live','false');});
  await page.reload();await input().waitFor();assert.equal(await input().inputValue(),'');assert.equal(await standard().inputValue(),'Standard Chinese (Beijing)');assert(await page.locator('.m13-settings-section').isVisible());assert.equal(await page.getByRole('button',{name:'转换设置',exact:true}).count(),0);
  await page.getByRole('button',{name:'恢复本机草稿',exact:true}).click();assert.equal(await input().inputValue(),'旧文本银行');assert.equal(await standard().inputValue(),'UntPhesoca严');assert.equal((await values()).at(-1),'x̞ɑ̟ŋ̚˧˥');
  await page.reload();await input().waitFor();assert.equal(await input().inputValue(),'');assert.equal(await standard().inputValue(),'Standard Chinese (Beijing)');report.checks.push('old collapsed preference ignored; empty Beijing startup; old draft preserved and explicitly restored');
  await input().fill('妈女略吗');assert.deepEqual(await values(),['ma˥','ny˨˩˧','lye˥˩','ma']);
  await page.getByRole('button',{name:'显示声调',exact:true}).click();assert.deepEqual(await values(),['ma','ny','lye','ma']);
  await standard().selectOption('汉语拼音');assert.deepEqual(await values(),['ma','nü','lüe','ma']);await page.getByRole('button',{name:'显示声调',exact:true}).click();assert.deepEqual(await values(),['mā','nǚ','lüè','ma']);
  await standard().selectOption('Standard Chinese (Beijing)严');await page.getByRole('button',{name:'显示声调',exact:true}).click();assert.equal((await values())[0],'ma̠');report.checks.push('tones independently toggle in IPA and pinyin; ü and segmental diacritics preserved');
  await page.getByLabel('音标颜色',{exact:true}).fill('#2345ab');await page.getByLabel('汉字颜色',{exact:true}).fill('#ab4523');await page.getByLabel('汉字字体',{exact:true}).selectOption('KaiTi');
  const styles=await page.locator('.m13-mapped').first().evaluate(e=>({ipa:getComputedStyle(e.querySelector('.m13-ipa')).fontFamily,hanzi:getComputedStyle(e.querySelector('.m13-hanzi')).fontFamily,ic:getComputedStyle(e.querySelector('.m13-ipa')).color,hc:getComputedStyle(e.querySelector('.m13-hanzi')).color}));assert(styles.ipa.includes('PTB-Doulos'));assert(!styles.ipa.includes('KaiTi'));assert(styles.hanzi.includes('KaiTi'));assert.equal(styles.ic,'rgb(35, 69, 171)');assert.equal(styles.hc,'rgb(171, 69, 35)');
  await context.setOffline(true);await png('custom-hidden-tones',['ma̠','ny','lye̞','ma̠']);await page.getByRole('button',{name:'显示声调',exact:true}).click();await png('custom-visible-tones',['ma̠˥','ny˨˩˧','lye̞˥˩','ma̠']);await context.setOffline(false);report.checks.push('both offline PNGs have exact custom colors and Hanzi font; fixed Doulos IPA; visible/hidden tones match preview');
  await page.getByLabel('汉字字体',{exact:true}).selectOption('__custom__');await page.getByLabel('自定义汉字字体').fill('PTB unavailable font 8675309');await page.getByRole('button',{name:'保存为 PNG',exact:true}).click();await page.getByRole('alert').filter({hasText:'在当前设备不可用'}).waitFor();assert.equal(await input().inputValue(),'妈女略吗');await page.getByLabel('汉字字体',{exact:true}).selectOption('KaiTi');report.checks.push('unavailable custom font gives recoverable explicit export error');
  await input().fill('银行');await standard().selectOption('Standard Chinese (Beijing)严');await page.locator('.m13-ambiguous[data-index="1"]').click();await page.getByRole('dialog',{name:'选择 行 的读音'}).getByRole('button',{name:/hang2.*xa/}).click();assert.equal((await values())[1],'xa̝ŋ˧˥');
  await input().fill('😀银行');assert.equal((await values())[1],'xa̝ŋ˧˥');await input().fill('新行');assert.equal((await values())[1],'xa̝ŋ˧˥');await input().fill('妈');await input().fill('行');assert.equal((await values())[0],'ɕiŋ˧˥');report.checks.push('choices follow Unicode prefix/suffix edits; unrelated replacements clear stale choices');
  await page.getByRole('button',{name:/保存本机草稿/}).click();await page.reload();await input().waitFor();await page.getByLabel('仅音标').check();
  assert.equal(await page.getByLabel('音标字号').inputValue(),'28');
  await input().fill('春江花月夜\n银行，女略。');await page.getByLabel('字音同显').check();
  for(const theme of ['light','dark'])for(const [width,height,scale] of [[1440,900,1],[1100,700,1],[1920,1080,1.5]])for(const layout of ['左右排布','上下排布']){
   await page.setViewportSize({width,height});await page.evaluate(({theme,scale})=>{document.documentElement.dataset.theme=theme;document.documentElement.style.zoom=String(scale);document.documentElement.style.setProperty('--page-scale',String(scale));},{theme,scale});await page.getByLabel(layout,{exact:true}).check();await idle();
   const geometry=await page.evaluate(()=>{const rect=s=>{const r=document.querySelector(s).getBoundingClientRect();return {left:r.left,top:r.top,right:r.right,bottom:r.bottom};},settings=document.querySelector('.m13-settings-section');return {settings:rect('.m13-settings-section'),input:rect('.m13-input-section'),output:rect('.m13-result-section'),scroll:settings.scrollHeight,client:settings.clientHeight};});
   assert(geometry.settings.right<=geometry.input.left+1);if(layout==='左右排布')assert(geometry.input.right<=geometry.output.left+1);else assert(geometry.input.bottom<=geometry.output.top+1);
   if(width===1440&&layout==='左右排布')assert(geometry.scroll<=geometry.client+2,'normal controls fit without excess scroll');
   await page.getByLabel('汉字字体',{exact:true}).scrollIntoViewIfNeeded();assert(await page.getByLabel('汉字字体',{exact:true}).isVisible());
   const caption=await page.locator('.font-family-select>span').boundingBox();assert(caption.width>70&&caption.height<40,'font label keeps a horizontal line');
   report.layouts.push({theme,width,height,scale,layout,geometry});await shot(`${theme}-${width}-${scale}-${layout==='左右排布'?'side':'stacked'}`);
  }
  await page.setViewportSize({width:1440,height:900});await page.evaluate(()=>{document.documentElement.style.zoom='1';document.documentElement.style.setProperty('--page-scale','1');document.documentElement.dataset.theme='dark';});await page.getByLabel('左右排布').check();await page.getByRole('button',{name:'跟随主题',exact:true}).click();
  await page.evaluate(()=>{window.paint=[];const original=CanvasRenderingContext2D.prototype.fillText;CanvasRenderingContext2D.prototype.fillText=function(text,...args){paint.push({text,font:this.font,color:this.fillStyle});return original.call(this,text,...args);};});const defaultReady=page.waitForEvent('download');await page.getByRole('button',{name:'保存为 PNG',exact:true}).click();await (await defaultReady).saveAs(path.join(out,'default-dark-white-background.png'));const defaultPaint=await page.evaluate(()=>window.paint);assert.equal(defaultPaint.find(p=>p.text==='女').color,'#182332');report.checks.push('default ink remains readable on the white PNG background in dark mode');
  assert.deepEqual(report.errors,[]);assert.deepEqual(report.external,[]);report.success=true;
 }catch(e){report.error=String(e);await shot('failure');throw e;}
 finally{await fs.writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2));console.log(out);await browser.close();await server.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
