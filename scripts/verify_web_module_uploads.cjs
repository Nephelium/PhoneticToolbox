const fs=require('node:fs/promises'),path=require('node:path'),os=require('node:os'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..'),out=path.join(root,'output/validation/web-launch/isolated'),origin='http://127.0.0.1:18765';
(async()=>{
 const account=JSON.parse(await fs.readFile(path.join(out,'client/module-qa.json'),'utf8'));
 const {chromium}=require(path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const context=await chromium.launchPersistentContext(path.join(out,'module-ui-browser'),{headless:false,executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',viewport:{width:1440,height:1000}});
 const page=await context.newPage(),errors=[],checks=[];page.setDefaultTimeout(30000);page.on('pageerror',e=>errors.push(e.message));
 let session;
 try{
  await page.goto(origin+'/app/',{timeout:120000});await page.getByLabel('登录名').fill(account.username);await page.getByLabel('密码',{exact:true}).fill(account.password);await page.getByRole('button',{name:'登录',exact:true}).click();await page.locator('#tool-sidebar').waitFor();
  session=await(await context.request.get(origin+'/api/v1/auth/me')).json();
  const api=async route=>(await context.request.get(origin+'/api/v1/'+route,{headers:{'X-PTB-Account':session.user.id}})).json();
  const project=(await api('projects')).projects[0].id;
  const wav=Buffer.alloc(44+8820);wav.write('RIFF');wav.writeUInt32LE(wav.length-8,4);wav.write('WAVEfmt ',8);wav.writeUInt32LE(16,16);wav.writeUInt16LE(1,20);wav.writeUInt16LE(1,22);wav.writeUInt32LE(44100,24);wav.writeUInt32LE(88200,28);wav.writeUInt16LE(2,32);wav.writeUInt16LE(16,34);wav.write('data',36);wav.writeUInt32LE(8820,40);for(let i=0;i<4410;i++)wav.writeInt16LE(Math.round(12000*Math.sin(i*2*Math.PI*200/44100)),44+i*2);
  const png=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jP2kAAAAASUVORK5CYII=','base64');
  const names=['参数估计','参数显示','EGG 信号分析','LPC 谱图','声学参数合成','发声类型合成','变速变调','语谱图转音频','TextGrid标注'];
  for(const [i,name] of names.entries()){
   await page.locator('#tool-sidebar').getByRole('button',{name,exact:true}).click();
   const control=page.locator('.research-upload:visible');await control.getByRole('button',{name:'上传文件',exact:true}).waitFor();
   const filename='module-'+i+(i===7?'.png':'.wav');
   await control.locator('input[type=file]').first().setInputFiles({name:filename,mimeType:i===7?'image/png':'audio/wav',buffer:i===7?png:wav});
   await control.getByText('已上传 1 个文件',{exact:true}).waitFor();
   assert((await api('assets?project_id='+project)).assets.some(a=>a.name===filename&&a.state==='ready'));
   checks.push(name+' module upload');console.log('UPLOAD_OK',name);
  }
  await page.getByRole('button',{name:'容量与扩容',exact:true}).click();assert.equal(await page.getByLabel('申请总额度（MB）').inputValue(),'1000');await page.getByRole('dialog').getByRole('button',{name:'关闭对话框',exact:true}).click();checks.push('capacity-default-1000');
  await page.locator('#tool-sidebar').getByRole('button',{name:'生理参数合成',exact:true}).click();await page.getByText('网页版暂未接入生理参数合成，当前请在桌面应用中使用。',{exact:true}).waitFor();checks.push('m10-accurate-availability');
  for(const theme of ['dark','light']){
   await page.evaluate(theme=>{document.documentElement.dataset.theme=theme;},theme);
   await page.getByRole('button',{name:'文件与空间',exact:true}).click();const dialog=page.getByRole('dialog',{name:'文件与空间',exact:true});
   await dialog.locator('.storage-meter').waitFor();await dialog.getByText('module-0.wav',{exact:true}).waitFor();assert.equal(await dialog.locator('.storage-files li').count(),9);
   assert.equal(await dialog.evaluate(e=>getComputedStyle(e).getPropertyValue('--panel').trim()),await page.evaluate(()=>getComputedStyle(document.documentElement).getPropertyValue('--panel').trim()));
   await page.getByText('补充上传文件',{exact:true}).click();await dialog.getByText('单文件上传上限 100.0 MB。',{exact:false}).waitFor();await page.screenshot({path:path.join(out,'module-files-'+theme+'.png')});
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
   await dialog.getByRole('button',{name:'关闭对话框',exact:true}).click();
  }
  await page.setViewportSize({width:390,height:844});await page.getByRole('button',{name:'文件与空间',exact:true}).click();await page.getByRole('dialog').locator('.storage-meter').waitFor();await page.getByRole('dialog').getByText('module-0.wav',{exact:true}).waitFor();await page.screenshot({path:path.join(out,'module-files-mobile.png')});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);checks.push('shared-theme-dark-light-and-mobile');
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'module-ui-verification.json'),JSON.stringify({success:true,checks,errors},null,2));console.log('MODULE_UI_PASS');
 }catch(e){await page.screenshot({path:path.join(out,'module-ui-failure.png')});throw e;}
 finally{
  if(session)await context.request.post(origin+'/api/v1/auth/logout',{headers:{Origin:origin,'X-CSRF-Token':session.csrf_token}}).catch(()=>{});
  await context.close();
 }
})().catch(e=>{console.error(e);process.exitCode=1;});
