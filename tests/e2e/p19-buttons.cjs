// P19-R3: owned headless Chrome + existing Vite, no dialogs, tasks or devices.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
const modules=[['M01','参数估计'],['M02','参数显示'],['M03','EGG 信号分析'],['M04','LPC 谱图'],['M05','唇形提取'],['M06','语音合成'],['M07','发声类型合成'],['M08','变速变调'],['M09','语谱图转音频'],['M11','MFA 自动标注'],['M12','TextGrid标注'],['M13','汉字转国际音标'],['M14','音系归纳'],['M15','感知实验'],['M16','录音'],['M17','国际音标表Plus']];
async function main(){
 const out=path.join(root,'output/validation/p19-r3','chrome-'+Date.now());await fs.mkdir(out,{recursive:true});
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,watch:null}});await server.listen();
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
 const page=await browser.newPage({viewport:{width:1920,height:1080}}),errors=[],checks=[],matrix=[];
 page.on('pageerror',e=>errors.push(e.message));
 const active=()=>page.locator('.module-frame:visible');
 const set=async(mode,effects)=>{await page.evaluate(async({mode,effects})=>(await import('/src/state/buttons.ts')).setButtonAppearance({mode,effects}),{mode,effects});await page.waitForTimeout(160);};
 const audit=async()=>active().evaluate(el=>{
  const css=getComputedStyle(document.documentElement),probe=document.createElement('span');el.append(probe);
  const color=token=>{probe.style.color=css.getPropertyValue(token);return getComputedStyle(probe).color;};
  const accent=color('--accent'),panel=color('--panel'),text=color('--text'),onAccent=color('--on-accent');probe.remove();
  return {accent,panel,text,onAccent,buttons:[...el.querySelectorAll('button')].filter(b=>b.checkVisibility({contentVisibilityAuto:true,visibilityProperty:true})).map(b=>{const s=getComputedStyle(b),r=b.getBoundingClientRect();return {label:b.textContent.trim().slice(0,60),primary:b.classList.contains('primary'),disabled:b.disabled,bg:s.backgroundColor,color:s.color,shadow:s.boxShadow,filter:s.filter,transition:s.transitionDuration,outline:s.outlineStyle,rect:[r.x,r.y,r.width,r.height]};})};
 });
 try{
  await page.goto(server.resolvedUrls.local[0]);await page.getByRole('button',{name:'设置',exact:true}).click();
  assert.equal(await page.getByLabel('按钮高亮').inputValue(),'auto');assert(await page.getByLabel('按钮阴影与悬停光效').isChecked());
  await page.getByLabel('按钮高亮').selectOption('plain');await page.getByLabel('按钮阴影与悬停光效').uncheck();await page.reload();await page.getByRole('button',{name:'设置',exact:true}).click();
  assert.equal(await page.getByLabel('按钮高亮').inputValue(),'plain');assert.equal(await page.getByLabel('按钮阴影与悬停光效').isChecked(),false);checks.push('settings controls immediately apply and persist both preferences after reload');
  for(const [id,name] of modules){
   await page.locator('.nav-item').filter({hasText:name}).click();await active().waitFor();
   await active().evaluate(el=>el.querySelectorAll('details').forEach(d=>d.open=true));
   const profiles=[];
   for(const theme of ['light','dark']){
    await page.evaluate(theme=>document.documentElement.dataset.theme=theme,theme);
    for(const mode of ['auto','all','plain'])for(const effects of [true,false]){
     await set(mode,effects);const a=await audit();assert(a.buttons.length,id+' has buttons');
     for(const b of a.buttons){
      const details=JSON.stringify({id,theme,mode,effects,b});
      if(mode==='all'||(mode==='auto'&&b.primary)){assert.equal(b.bg,a.accent,details);assert.equal(b.color,a.onAccent,details);}
      if(mode==='plain'){assert.equal(b.bg,a.panel,details);assert.equal(b.color,a.text,details);}
      if(!effects||b.disabled)assert.equal(b.shadow,'none',details);else assert.notEqual(b.shadow,'none',details);
      if(!effects){assert.equal(b.filter,'none',details);assert.equal(b.transition,'0s',details);}
     }
     profiles.push({theme,mode,effects,count:a.buttons.length,primary:a.buttons.filter(b=>b.primary).length});
    }
   }
   matrix.push({id,profiles});console.log('P19-R3',id,'12 mode/effect/theme combinations');
  }checks.push('16 module pages x 12 combinations: all visible buttons, pure fill and text, effects and disabled state');
  await page.locator('.nav-item').filter({hasText:'语音合成'}).click();await set('auto',true);
  await page.evaluate(()=>document.documentElement.dataset.theme='dark');await page.waitForTimeout(200);await page.screenshot({path:path.join(out,'m06-auto-dark.png')});
  await page.getByRole('button',{name:'设置',exact:true}).click();
  const palettes=await page.getByLabel('配色方案').locator('option').evaluateAll(es=>es.map(e=>e.value));
  for(const palette of palettes)for(const theme of ['light','dark']){
   await page.getByLabel('配色方案').selectOption(palette);await page.getByRole('button',{name:theme==='light'?'浅色':'深色',exact:true}).click();
   for(const mode of ['auto','all','plain']){
    await set(mode,false);
    const result=await page.locator('.appearance-settings').evaluate((el,mode)=>{
     const b=el.querySelector('button'),css=getComputedStyle(document.documentElement),s=getComputedStyle(b),probe=document.createElement('span');el.append(probe);probe.style.color=css.getPropertyValue(mode==='all'?'--accent':'--panel');const bg=getComputedStyle(probe).color;probe.remove();return {bg:s.backgroundColor,expected:bg,shadow:s.boxShadow};
    },mode);
    if(mode!=='auto')assert.equal(result.bg,result.expected,JSON.stringify({palette,theme,mode,result}));assert.equal(result.shadow,'none');
   }
  }checks.push('29 palettes x 2 themes x 3 modes: pure fill and effects disabled');
  await page.getByLabel('配色方案').selectOption('everforest');await page.getByRole('button',{name:'深色',exact:true}).click();
  await set('auto',true);await page.screenshot({path:path.join(out,'settings-auto-dark.png')});
  for(const [width,height,scale] of [[1920,1080,100],[1024,768,150],[390,844,100]]){
   await page.setViewportSize({width,height});await page.evaluate(async scale=>(await import('/src/state/pageZoom.ts')).setPageScale(scale),scale);
   await page.getByLabel('按钮高亮').scrollIntoViewIfNeeded();assert(await page.getByLabel('按钮高亮').isVisible());assert(await page.evaluate(()=>document.documentElement.scrollWidth<=document.documentElement.clientWidth+2));
  }checks.push('settings accessible at desktop, narrow and 150% zoom');
  await page.setViewportSize({width:1920,height:1080});await page.evaluate(async()=>(await import('/src/state/pageZoom.ts')).setPageScale(100));
  await page.locator('.nav-item').filter({hasText:'国际音标表Plus'}).click();await set('plain',false);
  const selected=active().locator('button[aria-pressed=true]').first();assert.equal(await selected.evaluate(b=>getComputedStyle(b).outlineStyle),'solid');await page.keyboard.press('Tab');await selected.focus();assert(await selected.evaluate(b=>b.matches(':focus-visible')));assert.equal(await selected.evaluate(b=>getComputedStyle(b).outlineWidth),'2px');
  await page.emulateMedia({reducedMotion:'reduce'});await set('all',true);assert.equal(await selected.evaluate(b=>getComputedStyle(b).transitionDuration),'0s');checks.push('plain mode keeps selected outline and keyboard focus; reduced motion disables transitions');
  await page.screenshot({path:path.join(out,'m17-all-dark.png')});
  await page.evaluate(()=>localStorage.setItem('ptb.v3.buttonAppearance',JSON.stringify({mode:'broken',effects:false})));await page.reload();await page.getByRole('button',{name:'设置',exact:true}).click();assert.equal(await page.getByLabel('按钮高亮').inputValue(),'auto');assert.equal(await page.getByLabel('按钮阴影与悬停光效').isChecked(),false);checks.push('corrupt mode recovers independently of valid effect preference');
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success:true,checks,matrix,errors},null,2));console.log(JSON.stringify({out,checks}));
 }catch(error){await page.screenshot({path:path.join(out,'failed.png')});await fs.writeFile(path.join(out,'failed.json'),JSON.stringify({error:String(error),checks,matrix,errors},null,2));console.error(out);throw error;}
 finally{await browser.close();await server.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
