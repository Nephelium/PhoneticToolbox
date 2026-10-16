// Real current application views for the public website. No private recordings or user profile.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url'),{spawnSync}=require('node:child_process'),{createHash}=require('node:crypto');
const root=path.resolve(__dirname,'..');
async function main(){
 const out=path.join(root,'output/validation/website','capture-'+Date.now()),scratch=path.join(out,'scratch');
 const assets=path.join(root,'temporary-site/assets/screens');await fs.mkdir(scratch,{recursive:true});await fs.mkdir(assets,{recursive:true});
 const write=(name,data)=>fs.writeFile(path.join(out,name),JSON.stringify(data,null,2)+'\n');
 const marker={schema:'ptb-test-run/1',owner_pid:process.pid,state:'active',started_at:new Date().toISOString(),recipe:'Current Vue source; isolated browser; home, IPA chart, Mandarin conversion and appearance settings; no private input, camera or microphone'};
 await write('run.json',marker);let server,browser,page;const report={success:false,captures:[],errors:[]};
 try{
  const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
  server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{noDiscovery:true,include:['vue']}});await server.listen();
  const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  browser=await chromium.launchPersistentContext(path.join(scratch,'profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--mute-audio'],viewport:{width:2560,height:1440},deviceScaleFactor:1});
  page=await browser.newPage();page.on('pageerror',e=>report.errors.push(e.message));report.browser=browser.browser().version();
  await page.goto(server.resolvedUrls.local[0]);await page.locator('#tool-sidebar').waitFor();
  async function palette(name,mode){
   await page.locator('#tool-sidebar').getByRole('button',{name:'设置',exact:true}).click();await page.getByRole('tab',{name:'外观',exact:true}).click();
   await page.getByRole('button',{name:mode==='light'?'浅色':'深色',exact:true}).click();
   await page.getByRole('combobox',{name:'配色方案',exact:true}).click();await page.locator('[data-palette-option="'+name+'"]').click();
   await page.waitForFunction(({name,mode})=>document.documentElement.dataset.palette===name&&document.documentElement.dataset.theme===mode,{name,mode});
   while(await page.getByLabel('页面缩放比例',{exact:true}).innerText()!=='150%')await page.getByRole('button',{name:'放大页面',exact:true}).click();
   report.page_scale_percent=150;
  }
  async function capture(file,module,paletteName,mode,input=null){
   await page.evaluate(()=>{document.activeElement?.blur();document.querySelectorAll('textarea').forEach(e=>e.scrollTop=0);return document.fonts.ready});await page.waitForTimeout(500);const viewport=page.viewportSize();await page.mouse.move(viewport.width-2,viewport.height-2);
   await page.screenshot({path:path.join(assets,file),animations:'disabled'});
   const bytes=await fs.readFile(path.join(assets,file));report.captures.push({file,module,palette:paletteName,mode,...page.viewportSize(),page_scale_percent:report.page_scale_percent||100,input,sha256:createHash('sha256').update(bytes).digest('hex'),bytes:bytes.length,source:'Current Vue application in isolated Chrome, unmodified whole viewport'});
   console.log('Captured '+file);
  }
  if(process.argv.includes('--stacks-only')||process.argv.includes('--poem-only')){
   if(!process.argv.includes('--poem-only')){
   await palette('catppuccin','dark');
   assert.equal(await page.getByLabel('页面缩放比例',{exact:true}).innerText(),'150%');
   report.page_scale_percent=150;
   await page.locator('#tool-sidebar').getByRole('button',{name:'国际音标表Plus',exact:true}).click();
   for(const [tab,file] of [['IPA','ipa-catppuccin-dark.png'],['extIPA','extipa-catppuccin-dark.png'],['VoQS','voqs-catppuccin-dark.png']]){
    await page.getByRole('button',{name:tab,exact:true}).click();await capture(file,'M17','catppuccin','dark');
   }
   }
   await palette('rose-pine','light');await page.locator('#tool-sidebar').getByRole('button',{name:'汉字转国际音标',exact:true}).click();
   // Public-domain poem, checked against Wikisource oldid 9685993; simplified characters.
   const poem=`春江潮水连海平，海上明月共潮生。
滟滟随波千万里，何处春江无月明？
江流宛转绕芳甸，月照花林皆似霰。
空里流霜不觉飞，汀上白沙看不见。
江天一色无纤尘，皎皎空中孤月轮。
江畔何人初见月，江月何年初照人？
人生代代无穷已，江月年年望相似。
不知江月待何人？但见长江送流水。
白云一片去悠悠，青枫浦上不胜愁。
谁家今夜扁舟子，何处相思明月楼？
可怜楼上月徘徊，应照离人妆镜台。
玉户帘中卷不去，捣衣砧上拂还来。
此时相望不相闻，愿逐月华流照君。
鸿雁长飞光不度，鱼龙潜跃水成文。
昨夜闲潭梦落花，可怜春半不还家。
江水流春去欲尽，江潭落月复西斜。
斜月沉沉藏海雾，碣石潇湘无限路。
不知乘月几人归，落月摇情满江树。`;
   const verseLines=poem.replace(/([，。？])\n?/g,'$1\n').trim();
   await page.getByRole('textbox',{name:'待转换汉字文本'}).fill(verseLines);await page.getByRole('combobox',{name:'转换标准'}).selectOption('UntPhesoca严');
   await page.getByRole('button',{name:'等宽排版',exact:true}).click();
   // Keep normal 24 px Hanzi / 16 px IPA and 150% app zoom; do not shrink to fit the poem.
   await page.getByRole('slider',{name:'行距',exact:true}).fill('12');
   const divider=page.getByRole('separator',{name:'汉字输入宽度',exact:true});await divider.press('Home');for(let i=0;i<6;i++)await divider.press('ArrowRight');
   await page.getByRole('button',{name:'收起侧栏',exact:true}).click();
   await page.locator('.m13-token').last().waitFor();await page.waitForTimeout(800);
   assert.equal(await page.getByRole('button',{name:'等宽排版',exact:true}).getAttribute('aria-pressed'),'true');
   assert.equal((poem.match(/[\u4e00-\u9fff]/g)||[]).length,252);
   assert(verseLines.split('\n').every(line=>/^[\u4e00-\u9fff]{7}[，。？]$/.test(line)));
   const rows=await page.locator('.m13-paragraph').evaluateAll(rows=>rows.map(row=>[...row.querySelectorAll('.m13-token')].map(cell=>Math.round(cell.getBoundingClientRect().top))));
   assert(rows.every(row=>row.length===8&&new Set(row).size===1),'Seven Hanzi and punctuation must remain on one output row');
   // Context readings manually checked for the entire poem. Standard Mandarin
   // citation tones are retained (the application does not apply tone sandhi).
   const expected={处:['chu',4],转:['zhuan',3],似:['si',4],空:['kong',1],觉:['jue',2],沙:['sha',1],看:['kan',4],见:['jian',4],色:['se',4],纤:['xian',1],中:['zhong',1],相:['xiang',1],待:['dai',4],长:['chang',2],片:['pian',4],扁:['pian',1],子:['zi',3],可:['ke',3],徊:['huai',2],应:['ying',1],卷:['juan',3],还:['huan',2],时:['shi',2],华:['hua',2],度:['du',4],落:['luo',4],尽:['jin',4],藏:['cang',2],石:['shi',2],乘:['cheng',2],几:['ji',3],树:['shu',4]};
   const raw=JSON.parse(await fs.readFile(path.join(root,'frontend/src/modules/mandarin-ipa/ipa-data.json'),'utf8'));
   const choices={},audit=[];
   for(const [index,char] of [...verseLines].entries()){
    const entries=raw.rows.filter(r=>r[0]===char);if(!entries.length)continue;
    const reading=expected[char]||[entries[0][2],entries[0][1]],pick=entries.findIndex(r=>r[2]===reading[0]&&r[1]===reading[1]);assert(pick>=0);
    choices[char+'_'+index]=pick;audit.push({index,char,pinyin:reading[0],tone:reading[1],corrected:pick!==0});
    if(pick!==0){
     await page.locator('.m13-token[data-index="'+index+'"]').click();
     await page.locator('.m13-variants button[aria-pressed]').nth(pick).click();
    }
   }
   report.poemReadingAudit={basis:'Contemporary Standard Mandarin citation readings; no archaic rhyme substitution or tone sandhi',sources:['https://www.zdic.net/hans/扁舟興盡','https://www.zdic.net/hans/不胜','https://ctext.org/text.pl?if=en&node=127868&remap=gb'],tokens:audit};
   for(const [standard,file] of [['Standard Chinese (Beijing)','mandarin-rose-pine-light.png'],['UntPhesoca严','mandarin-unt-poem-light.png']]){
    await page.getByRole('combobox',{name:'转换标准'}).selectOption(standard);
    const actual=await page.locator('.m13-mapped').evaluateAll(es=>es.map(e=>({index:Number(e.dataset.index),value:e.dataset.value})));
    const expectedTokens=await page.evaluate(async({text,standard,choices})=>{const {convertText}=await import('/src/modules/mandarin-ipa/state.ts');return convertText(text,standard,choices).filter(t=>t.kind==='mapped').map(t=>({index:t.index,value:t.value}));},{text:verseLines,standard,choices});
    assert.deepEqual(actual,expectedTokens);assert.equal(actual.length,252);
    await page.evaluate(()=>{document.querySelector('.m13-output').scrollTop=0;document.querySelectorAll('textarea').forEach(e=>e.scrollTop=0)});
    await capture(file,'M13','rose-pine','light',verseLines);
   }
  }else if(process.argv.includes('--branding-only')){
   await palette('everforest','light');await page.locator('#tool-sidebar').getByRole('button',{name:'首页',exact:true}).click();await capture('home-everforest-light.png','home','everforest','light');
   await palette('tokyo-night','dark');await capture('appearance-tokyo-night-dark.png','settings','tokyo-night','dark');
   await palette('dracula','light');await page.getByRole('tab',{name:'图标',exact:true}).click();await page.waitForFunction(()=>document.querySelectorAll('.builtin-icon-options .icon-choice').length===31);await capture('icons-dracula-light.png','settings-icons','dracula','light');
   await palette('temple','dark');await page.locator('#tool-sidebar').getByRole('button',{name:'使用说明',exact:true}).click();await page.locator('.manual-reader').waitFor();await page.waitForTimeout(1200);await capture('manual-temple-dark.png','manual','temple','dark');
  }else if(process.argv.includes('--manual-only')){
   await palette('temple','light');await page.locator('#tool-sidebar').getByRole('button',{name:'使用说明',exact:true}).click();await page.locator('.manual-reader').waitFor();await page.waitForTimeout(1200);await capture('manual-temple-light.png','manual','temple','light');
  }else{
  await palette('everforest','light');await page.locator('#tool-sidebar').getByRole('button',{name:'首页',exact:true}).click();await capture('home-everforest-light.png','home','everforest','light');
  await palette('catppuccin','dark');await page.locator('#tool-sidebar').getByRole('button',{name:'国际音标表Plus',exact:true}).click();await page.locator('.ipa-plus-page').waitFor({timeout:10000}).catch(()=>page.locator('main').waitFor());
  await capture('ipa-catppuccin-dark.png','M17','catppuccin','dark');
  await palette('rose-pine','light');await page.locator('#tool-sidebar').getByRole('button',{name:'汉字转国际音标',exact:true}).click();
  const input='语音学\n声母　韵母　声调\n舌尖　舌面　舌根\n普通话　实验语音学';await page.getByRole('textbox',{name:'待转换汉字文本'}).fill(input);await page.locator('.m13-token').first().waitFor();
  await capture('mandarin-rose-pine-light.png','M13','rose-pine','light',input);
  await palette('tokyo-night','dark');await capture('appearance-tokyo-night-dark.png','settings','tokyo-night','dark');
  }
  assert.deepEqual(report.errors,[]);report.success=true;
  const old=JSON.parse(await fs.readFile(path.join(assets,'provenance.json'),'utf8').catch(()=>'{}'));const fresh=new Set(report.captures.map(x=>x.file));
  await fs.writeFile(path.join(assets,'provenance.json'),JSON.stringify({schema:'ptb-website-captures/1',captured_at:new Date().toISOString(),browser:report.browser,scope:'Current Chinese UI; no product results reconstructed; original images not cropped or recolored; website applies CSS perspective only',captures:[...(old.captures||[]).filter(x=>!fresh.has(x.file)),...report.captures]},null,2)+'\n');
 }catch(e){report.failure=String(e);if(page)await page.screenshot({path:path.join(out,'failure.png')}).catch(()=>{});throw e;}
 finally{
  await browser?.close();await server?.close();await write('report.json',report);
  const files=[];async function inventory(dir){for(const entry of await fs.readdir(dir,{withFileTypes:true})){assert(!entry.isSymbolicLink());const file=path.join(dir,entry.name);if(entry.isDirectory())await inventory(file);else{const bytes=await fs.readFile(file);files.push({path:path.relative(scratch,file),bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex')});}}}await inventory(scratch);
  await write('scratch-manifest.json',{schema:'ptb-test-scratch-files/1',files});await write('run.json',{...marker,state:'ready',test_outcome:report.success?'completed':'failed'});
  const cleanup=spawnSync('powershell.exe',['-NoProfile','-ExecutionPolicy','Bypass','-File',path.join(root,'scripts/cleanup_test_scratch.ps1'),'-RunDirectory',out],{windowsHide:true,encoding:'utf8'});
  assert.equal(cleanup.status,0,cleanup.stdout+cleanup.stderr);console.log(JSON.stringify({out,...report}));
 }
}
main().catch(e=>{console.error(e);process.exitCode=1});
