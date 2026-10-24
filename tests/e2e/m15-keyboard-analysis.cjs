// Real production AppShell, owned browser profile, synthetic WAV and questionnaire.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
function wav(i){const rate=16000,n=3200,b=Buffer.alloc(44+n*2);b.write('RIFF');b.writeUInt32LE(b.length-8,4);b.write('WAVEfmt ',8);b.writeUInt32LE(16,16);b.writeUInt16LE(1,20);b.writeUInt16LE(1,22);b.writeUInt32LE(rate,24);b.writeUInt32LE(rate*2,28);b.writeUInt16LE(2,32);b.writeUInt16LE(16,34);b.write('data',36);b.writeUInt32LE(n*2,40);for(let k=0;k<n;k++)b.writeInt16LE(Math.round(1500*Math.sin(2*Math.PI*(220+i*30)*k/rate)),44+k*2);return b;}
async function main(){
 const out=process.env.M15_QA_OUT,scratch=process.env.M15_QA_SCRATCH,engine=process.env.M15_QA_BROWSER??'chrome';assert(out&&scratch,'Run through verify_m15_keyboard_analysis.py');
 const {preview}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const server=await preview({root:path.join(root,'frontend'),preview:{host:'127.0.0.1',port:4177}});
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 let context,page;const checks=[],errors=[],downloads=[];const report={success:false,checks,errors,engine,recipe:'4 mono PCM16 WAV, 16000 Hz, 0.2 s, 220+i*30 Hz, 1500 peak; separate synthetic participants; no physical hearing/timing claim'};
 try{
  context=await chromium.launchPersistentContext(path.join(scratch,'browser'),{executablePath:engine==='edge'?'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe':'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1440,height:900},acceptDownloads:true,args:['--mute-audio']});page=context.pages()[0];
  page.on('pageerror',e=>errors.push(e.message));page.on('dialog',d=>d.accept());page.on('download',d=>downloads.push(d));
  const button=name=>page.getByRole('button',{name,exact:true}),tab=name=>page.getByRole('tab',{name,exact:true});
  const idle=()=>page.waitForFunction(()=>document.querySelector('.perception-page')?.getAttribute('aria-busy')==='false');
  const phase=p=>page.locator(`.m15-run[data-phase="${p}"]`).waitFor({timeout:15000});
  async function downloadFile(action,name){const event=page.waitForEvent('download');await action();const d=await event,file=path.join(scratch,name);await d.saveAs(file);await idle();return file;}
  async function prepare(){await button('预检与试音').click();await idle();assert((await page.locator('body').innerText()).includes('预检通过'));await page.getByLabel('已听见试音，设备正确').check();await page.getByLabel('已暂停其他录制、播放及重型任务').check();}
  async function participant(name){
   await button('填写被试信息').click();await page.getByLabel('姓名/编号',{exact:true}).waitFor();
   assert.equal(await page.getByLabel('姓名/编号',{exact:true}).inputValue(),'');
   // From this point through the last answer, the participant uses only the keyboard.
   await page.keyboard.type(name);await page.keyboard.press('Tab');await page.keyboard.press('ArrowDown');await page.keyboard.press('Space');await page.keyboard.press('Tab');await page.keyboard.press('Tab');await page.keyboard.press('Enter');await phase('intro');
   assert((await page.locator('.m15-presentation').innerText()).includes('按空格或回车开始实验'));
   await page.keyboard.press('Enter');await phase('interstitial');
  }
  await page.goto(server.resolvedUrls.local[0]);await page.locator('.nav-item').filter({hasText:'感知实验'}).click();await idle();
  await page.getByLabel('导入刺激 X',{exact:true}).setInputFiles([1,2,3,4].map(i=>({name:`trial-${i}.wav`,mimeType:'audio/wav',buffer:wav(i)})));await idle();
  await tab('问卷').click();const q=page.locator('.m15-design-panel .m15-group').nth(1);await q.getByLabel('题目',{exact:true}).fill('英语水平');await page.getByLabel('英语水平 选项 1',{exact:true}).fill('CET-4');await page.getByLabel('英语水平 选项 1',{exact:true}).press('End');await page.getByLabel('英语水平 选项 1',{exact:true}).pressSequentially(',CET-6');assert.equal(await page.getByLabel('英语水平 选项 1',{exact:true}).inputValue(),'CET-4,CET-6');await page.getByLabel('英语水平 选项 2',{exact:true}).fill('托福，雅思');await page.getByLabel('英语水平 添加选项',{exact:true}).click();await page.getByLabel('英语水平 选项 3',{exact:true}).fill('其他');await page.getByLabel('英语水平 移除选项 3',{exact:true}).click();assert.equal(await page.getByLabel('英语水平 选项 3',{exact:true}).count(),0);await page.screenshot({path:path.join(out,'questionnaire-options.png'),fullPage:true});checks.push('independent option rows preserve typed English comma and Chinese comma; add/remove');
  await tab('参数').click();await page.getByLabel('推进方式',{exact:true}).selectOption('manual');await page.getByLabel('自动播放',{exact:true}).uncheck();await page.getByLabel('提示音',{exact:true}).uncheck();await page.getByLabel('作答页提示语',{exact:true}).fill('请判断 X 的类别，F 或 J。');await page.getByLabel('显示作答页提示语',{exact:true}).check();
  await tab('序列').click();for(let i=1;i<=4;i++)await page.getByLabel(`试次 ${i} 标准答案`,{exact:true}).selectOption(i===4?'':'j');
  const configFile=await downloadFile(()=>button('导出项目 JSON').click(),'design.json');const config=JSON.parse(await fs.readFile(configFile,'utf8'));assert.equal(config.trials[0].correctKey,'j');assert.equal(config.config.showResponsePrompt,true);assert.deepEqual(config.questionnaire[1].options,['CET-4,CET-6','托福，雅思']);
  await prepare();await participant('P01');
  for(let i=0;i<4;i++){
   assert((await page.locator('.m15-presentation').innerText()).includes('按空格或回车进入试次'));
   await page.keyboard.press('Enter');await phase('ready');assert((await page.locator('.m15-presentation').innerText()).includes('按空格或回车播放当前试次'));
   await page.keyboard.press('Space');await phase('playing');assert.equal(await page.locator('.m15-response-prompt').innerText(),'请判断 X 的类别，F 或 J。');await page.keyboard.press('j');
   await phase('responding');if(i===0)await page.screenshot({path:path.join(out,'response-prompt.png'),fullPage:true});await page.keyboard.press(i===1?'f':'j');await phase(i===3?'completed':'interstitial');
  }
  await idle();const resultFile=await downloadFile(()=>button('导出结果').click(),'p01.json'),s=JSON.parse(await fs.readFile(resultFile,'utf8'));assert.equal(s.attempts.length,4);assert.deepEqual(s.attempts.map(a=>a.key),['j','f','j','j']);assert(s.attempts.every(a=>a.status==='completed'&&a.rtMs>=0));assert.equal(s.answers.q2,'托福，雅思');
  await button('确认导出文件已保存').click();await idle();await button('结束实验').click();await idle();checks.push('keyboard-only questionnaire, intro, enter trial, manual playback and answers; playing keys ignored; four distinct completed responses');
  // A second session has no correct key snapshot; later analysis can explicitly rescore it.
  await tab('序列').click();for(let i=1;i<=4;i++)await page.getByLabel(`试次 ${i} 标准答案`,{exact:true}).selectOption('');await prepare();await participant('P02');await page.keyboard.press('Enter');await phase('ready');await page.keyboard.press('Space');await phase('responding');await page.keyboard.press('j');await phase('interstitial');
  // Leave through native button keyboard activation, preserving unfinished trials.
  await button('结束实验').focus();await page.keyboard.press('Enter');await idle();await page.getByLabel('实验范式',{exact:true}).waitFor();
  await tab('序列').click();for(let i=1;i<=3;i++)await page.getByLabel(`试次 ${i} 标准答案`,{exact:true}).selectOption('j');await tab('分析').click();await idle();
  let rows=page.locator('.analysis-table tbody tr');assert.equal(await rows.count(),2);const p01=rows.filter({hasText:'P01'}),p02=rows.filter({hasText:'P02'});assert((await p01.innerText()).includes('66.67%'));assert((await p02.innerText()).includes('未计分'));await page.getByLabel('标准答案来源',{exact:true}).selectOption('current');assert((await p02.innerText()).includes('100.00%'));assert((await p01.innerText()).includes('66.67%'));
  const analysisFile=await downloadFile(()=>button('导出被试分析').click(),'analysis.xlsx');const XLSX=await import(pathToFileURL(path.join(root,'frontend/src/modules/perception/vendor/xlsx.mjs'))),wb=XLSX.read(await fs.readFile(analysisFile));assert.deepEqual(wb.SheetNames,['Participants','Scoring']);const summaries=XLSX.utils.sheet_to_json(wb.Sheets.Participants),one=summaries.find(a=>a.Participant==='P01');assert.equal(one.Correct,2);assert.equal(one.Scored_Trials,3);assert.equal(one.Accuracy_0_to_1,2/3);assert.equal(one.RT_N,4);const measured=s.attempts.map(a=>a.rtMs).sort((a,b)=>a-b);assert(Math.abs(one.Mean_RT_ms-measured.reduce((a,b)=>a+b,0)/4)<1e-9);assert(Math.abs(one.Q1_RT_ms-(measured[0]*.25+measured[1]*.75))<1e-9);await page.getByLabel('分析格式',{exact:true}).selectOption('csv');const csv=await downloadFile(()=>button('导出被试分析').click(),'analysis.csv');assert((await fs.readFile(csv,'utf8')).includes('P02'));checks.push('current-project two participant sessions; ungraded and early-ended records; explicit immutable rescoring; actual XLSX/CSV readback and independently checked RT quartile/mean');
  for(const theme of ['light','dark'])for(const width of [1440,900]){
   await page.setViewportSize({width,height:900});await page.evaluate(theme=>document.documentElement.dataset.theme=theme,theme);await page.screenshot({path:path.join(out,`analysis-${theme}-${width}.png`),fullPage:true});assert(await button('导出被试分析').isVisible());const box=await page.locator('.analysis-table').boundingBox();assert(box.x>=0&&box.x+box.width<=width+1);
  }
  await tab('参数').click();await page.getByLabel('显示作答页提示语',{exact:true}).uncheck();await prepare();await participant('P03');await page.keyboard.press('Enter');await phase('ready');await page.keyboard.press('Space');await phase('playing');assert.equal(await page.locator('.m15-response-prompt').count(),0);await phase('responding');await page.keyboard.press('Escape');await phase('paused');await idle();checks.push('prompt visibility off and Esc pause; shallow/deep themes and narrow analysis scroll container');
  // Verify new manual content through the same shipped reader resources.
  const chapter=await page.request.get(server.resolvedUrls.local[0]+'manual/chapters/m15.json');assert.equal(chapter.status(),200);const text=await chapter.text();assert(text.includes('查看被试成绩分析')&&text.includes('按空格或回车开始'));await button('帮助').click();const manual=page.getByLabel('说明书正文',{exact:true});await manual.getByRole('heading',{name:/查看被试成绩分析/}).waitFor();await manual.getByRole('heading',{name:/查看被试成绩分析/}).evaluate(e=>e.scrollIntoView({block:'start',behavior:'instant'}));await page.waitForFunction(()=>{const h=[...document.querySelectorAll('.manual-reading-scroll h3')].find(e=>e.textContent.includes('查看被试成绩分析'));const top=h?.getBoundingClientRect().top;return top>=80&&top<300;});await page.screenshot({path:path.join(out,'manual-analysis.png'),fullPage:true});assert((await manual.innerText()).includes('Participants'));checks.push('shipped manual opens from module help and renders new analysis section');
  assert.deepEqual(errors,[]);report.success=true;report.browser=await context.browser().version();report.downloads=downloads.length;report.rtExample=one;
 }catch(e){report.error=String(e);report.stack=e.stack;if(page){await page.screenshot({path:path.join(out,'failure.png'),fullPage:true}).catch(()=>{});report.body=await page.locator('body').innerText().catch(()=>null);}throw e;}
 finally{await fs.writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2));await context?.close();await new Promise(resolve=>server.httpServer.close(resolve));}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
