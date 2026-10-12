const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..'),[out,scratch]=process.argv.slice(2);
(async()=>{
  assert(out&&scratch,'Use the owned verification runner');
  const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
  const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  const server=await createServer({root:path.join(root,'frontend'),cacheDir:path.join(scratch,'vite-cache'),server:{host:'127.0.0.1',port:0,watch:null},optimizeDeps:{noDiscovery:true,include:['vue']}});
  const report={success:false,cases:[],pngs:[],errors:[],externalRequests:[]};let browser;
  try{
    await server.listen();const url=server.resolvedUrls.local[0];
    browser=await chromium.launchPersistentContext(path.join(scratch,'chrome-profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1440,height:900},acceptDownloads:true});
    const page=await browser.newPage();page.on('pageerror',e=>report.errors.push(e.message));page.on('request',r=>{if(/^https?:/.test(r.url())&&!r.url().startsWith(url))report.externalRequests.push(r.url());});
    await page.goto(url);await page.locator('.home-page').waitFor();
    const open=()=>page.locator('.nav-item').and(page.getByRole('button',{name:'汉字转国际音标',exact:true})).click();
    await open();const input=page.getByRole('textbox',{name:'待转换汉字文本'}),standard=page.getByRole('combobox',{name:'转换标准'}),equal=page.getByRole('button',{name:'等宽排版',exact:true});
    const idle=()=>page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
    const range=async(label,value)=>{await page.getByRole('slider',{name:label,exact:true}).evaluate((el,v)=>{el.value=String(v);el.dispatchEvent(new Event('input',{bubbles:true}));},value);await idle();};
    const values=()=>page.locator('.m13-mapped').evaluateAll(nodes=>nodes.map(n=>n.dataset.value));
    const widths=()=>page.locator('.m13-token').evaluateAll(nodes=>nodes.map(n=>n.getBoundingClientRect().width));
    async function uniform(){
      await idle();const model=await page.locator('.m13-output').evaluate(area=>{
        const zoom=parseFloat(getComputedStyle(document.documentElement).zoom)||1,cells=[...area.querySelectorAll('.m13-token')];
        const widths=cells.map(el=>el.getBoundingClientRect().width/zoom);
        const natural=Math.max(...cells.map(el=>Math.max(...[...el.children].map(c=>c.getBoundingClientRect().width/zoom))+8));
        return {min:Math.min(...widths),max:Math.max(...widths),natural};
      });
      assert(model.max-model.min<.1,JSON.stringify(model));assert(model.min>=model.natural-.2,JSON.stringify(model));assert(model.max-model.natural<1.2,JSON.stringify(model));return model;
    }
    async function pitch(){return page.locator('.m13-paragraph').first().evaluate(el=>{
      const zoom=parseFloat(getComputedStyle(document.documentElement).zoom)||1,ys=[...new Set([...el.children].map(n=>Math.round(n.getBoundingClientRect().top/zoom*100)/100))];return {pitch:ys[1]-ys[0],rows:ys.length,height:el.firstElementChild.getBoundingClientRect().height/zoom};
    });}
    const text='春江潮水连海平，海上明月共潮生。'.repeat(3)+'\n银行行走花儿\n\n😀春字？';
    for(const theme of ['light','dark'])for(const viewport of [{width:1440,height:900},{width:900,height:700}])for(const scale of [1,1.5]){
      await page.setViewportSize(viewport);await page.evaluate(({theme,scale})=>{document.documentElement.dataset.theme=theme;document.documentElement.style.zoom=String(scale);document.documentElement.style.setProperty('--page-scale',String(scale));},{theme,scale});
      await input.fill(text);await standard.selectOption('Standard Chinese (Beijing)');await range('汉字字号',24);await range('音标字号',16);
      const inputLayout=await page.locator('.m13-input-section').evaluate(section=>{
        const input=section.querySelector('textarea').getBoundingClientRect(),count=section.querySelector('.m13-count').getBoundingClientRect(),title=section.querySelector('h2').getBoundingClientRect(),result=document.querySelector('.m13-output').getBoundingClientRect();
        const left=section.getBoundingClientRect().width,right=document.querySelector('.m13-result-section').getBoundingClientRect().width;
        return {countInHeading:!!section.querySelector('.module-section-heading .m13-count'),count:section.querySelector('.m13-count').textContent,countLeft:count.left,titleRight:title.right,countBottom:count.bottom,inputTop:input.top,bottomDifference:input.bottom-result.bottom,leftWidth:left,rightWidth:right};
      });
      assert(inputLayout.countInHeading);assert.equal(inputLayout.count,[...text].length.toLocaleString()+' 个字符');assert(inputLayout.countLeft>=inputLayout.titleRight);assert(inputLayout.countBottom<=inputLayout.inputTop);assert(Math.abs(inputLayout.bottomDifference)<.1,JSON.stringify(inputLayout));
      assert(Math.abs(inputLayout.leftWidth-inputLayout.rightWidth)<1.5,JSON.stringify(inputLayout));
      if(await equal.getAttribute('aria-pressed')==='true')await equal.click();await idle();
      const unequal=await widths();assert(Math.max(...unequal)-Math.min(...unequal)>5);
      const original=await values();await equal.click();const initial=await uniform();assert.deepEqual(await values(),original);
      await range('音标字号',40);const largeIpa=await uniform();assert(largeIpa.max>initial.max+10);
      await range('音标字号',12);await range('汉字字号',64);const largeHanzi=await uniform();assert(largeHanzi.max>initial.max+10);
      await range('汉字字号',24);await range('音标字号',20);
      await range('行距',8);const tight=await pitch();await range('行距',30);const loose=await pitch();assert(tight.rows>=2&&loose.rows>=2);assert(Math.abs(loose.pitch-tight.pitch-52.8)<.3,JSON.stringify({tight,loose}));
      const arrow=await page.locator('.m13-ambiguous').first().evaluate(el=>({bottom:getComputedStyle(el,'::after').bottom,content:getComputedStyle(el,'::after').content}));assert.equal(arrow.bottom,'0px');assert(arrow.content.includes('▼'));
      await page.getByLabel('仅音标',{exact:true}).check();await uniform();await range('行距',8);const ipaTight=await pitch();await range('行距',30);const ipaLoose=await pitch();assert(Math.abs(ipaLoose.pitch-ipaTight.pitch-44)<.3);
      await page.getByLabel('字音同显',{exact:true}).check();await range('行距',18);
      assert.equal(await page.locator('.m13-paragraph').count(),4);assert.equal(await page.locator('.m13-paragraph').nth(2).locator('.m13-token').count(),0);
      await standard.selectOption('UntPhesoca严');await page.getByRole('button',{name:'自动组合儿化',exact:true}).click();await uniform();assert.equal(await page.locator('[data-erhua=true]').count(),1);
      assert((await page.locator('.m13-erhua p').textContent()).endsWith('自动组合后请核对独立的儿字。'));
      await page.getByLabel('上下排布',{exact:true}).check();await uniform();await page.getByLabel('左右排布',{exact:true}).check();
      report.cases.push({theme,viewport,scale,initial,largeIpa,largeHanzi,tight,loose,ipaTight,ipaLoose,inputLayout,paragraphs:4,erhua:true});
    }
    await page.evaluate(()=>{document.documentElement.dataset.theme='light';document.documentElement.style.zoom='1';document.documentElement.style.setProperty('--page-scale','1');});await page.setViewportSize({width:1440,height:900});
    const divider=page.getByRole('separator',{name:'汉字输入宽度',exact:true}),beforeDrag=await page.locator('.m13-input-section').evaluate(el=>el.getBoundingClientRect().width);
    await divider.press('ArrowRight');await idle();const manualWidth=await page.locator('.m13-input-section').evaluate(el=>el.getBoundingClientRect().width);assert(manualWidth>beforeDrag+9);report.manualWidth=manualWidth;
    await input.fill('春江花月夜，银行行走\n春江花月夜，银行行走');await standard.selectOption('Standard Chinese (Beijing)');await range('汉字字号',26);await range('音标字号',20);await uniform();
    await page.evaluate(()=>{window.qaPaint=[];const original=CanvasRenderingContext2D.prototype.fillText;CanvasRenderingContext2D.prototype.fillText=function(text,x,y,...args){window.qaPaint.push({text,x,y,font:this.font});return original.call(this,text,x,y,...args);};});
    await browser.setOffline(true);
    for(const spacing of [8,30]){
      await range('行距',spacing);await page.evaluate(()=>window.qaPaint=[]);const downloadReady=page.waitForEvent('download');await page.getByRole('button',{name:'保存为 PNG',exact:true}).click();const download=await downloadReady;
      const file=path.join(out,'equal-'+spacing+'.png');await download.saveAs(file);const bytes=await fs.readFile(file);assert.equal(bytes.subarray(1,4).toString(),'PNG');
      const paint=await page.evaluate(()=>window.qaPaint);assert(!paint.some(p=>p.text==='▼'));assert(paint.some(p=>p.text==='行'));
      const chars=paint.filter(p=>p.font.startsWith('26px ')||p.font.includes('26px '));const firstRow=chars.filter(p=>p.y===chars[0].y);assert(firstRow.length>=3);
      const advances=firstRow.slice(1).map((p,i)=>p.x-firstRow[i].x);assert(Math.max(...advances)-Math.min(...advances)<.1);
      const ys=[...new Set(chars.map(p=>p.y))];report.pngs.push({spacing,width:bytes.readUInt32BE(16),height:bytes.readUInt32BE(20),bytes:bytes.length,pitch:ys[1]-ys[0],advance:advances[0],triangleAbsent:true});
    }
    assert(Math.abs(report.pngs[1].pitch-report.pngs[0].pitch-57.2)<.1);
    await browser.setOffline(false);await range('行距',18);await page.getByRole('button',{name:/^保存本机草稿/}).click();await page.reload();await open();assert.equal(await input.inputValue(),'');assert.equal(await equal.getAttribute('aria-pressed'),'true');await page.getByRole('button',{name:'恢复本机草稿',exact:true}).click();assert.equal(await input.inputValue(),'春江花月夜，银行行走\n春江花月夜，银行行走');await uniform();
    assert(Math.abs(await page.locator('.m13-input-section').evaluate(el=>el.getBoundingClientRect().width)-manualWidth)<1.1);
    await page.screenshot({path:path.join(out,'equal-width-light.png')});assert.deepEqual(report.errors,[]);assert.deepEqual(report.externalRequests,[]);report.success=true;
  }finally{if(browser)await browser.close();await server.close();await fs.writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2)+'\n');}
  console.log(JSON.stringify(report,null,2));
})().catch(e=>{console.error(e);process.exitCode=1;});
