const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..'),[out,scratch]=process.argv.slice(2);
(async()=>{
  assert(out&&scratch,'Use owned runner');
  const {preview}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
  const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  const server=await preview({root:path.join(root,'frontend'),preview:{host:'127.0.0.1',port:0}}),url=server.resolvedUrls.local[0];
  const report={success:false,checks:[],errors:[],uploads:[],screenshots:[]};let context,page;
  const launch=async()=>{
    context=await chromium.launchPersistentContext(path.join(scratch,'profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1440,height:900},acceptDownloads:true});
    page=await context.newPage();page.on('pageerror',e=>report.errors.push(e.message));page.on('request',r=>{if(['POST','PUT','PATCH'].includes(r.method()))report.uploads.push(r.url());});
    await page.goto(url);await page.locator('.home-page').waitFor();await page.locator('.nav-item').and(page.getByRole('button',{name:'汉字转国际音标',exact:true})).click();
    await page.waitForFunction(()=>!!document.querySelector('[aria-label="转换标准"]')&&!document.querySelector('[aria-label="转换标准"]').disabled);
  };
  const values=()=>page.locator('.m13-mapped').evaluateAll(nodes=>nodes.map(n=>n.dataset.value).join('|'));
  async function waitValues(expected){await page.waitForFunction(expected=>[...document.querySelectorAll('.m13-mapped')].map(n=>n.dataset.value).join('|')===expected,expected);}
  async function importer(file,kind='hanzi',name){
    await page.getByRole('button',{name:'导入自定义字表',exact:true}).click();const dialog=page.getByRole('dialog',{name:'导入自定义字表',exact:true});
    await dialog.getByRole('combobox',{name:'字表对照类型'}).selectOption(kind);await dialog.getByLabel('选择字表文件',{exact:true}).setInputFiles(path.join(scratch,file));
    await page.waitForFunction(()=>!document.querySelector('[aria-label="选择字表文件"]').disabled);if(name)await dialog.getByRole('textbox',{name:'字表名称'}).fill(name);return dialog;
  }
  try{
    await launch();const input=()=>page.getByRole('textbox',{name:'待转换汉字文本'});
    await input().fill('行𠮷妈，A😀');let dialog=await importer('invalid.csv','hanzi','错误表');assert.match(await dialog.innerText(),/第 2 行/);assert(await dialog.getByRole('button',{name:'保存字表并使用'}).isDisabled());await dialog.getByRole('button',{name:'取消',exact:true}).click();
    dialog=await importer('hanzi.csv','hanzi','调查字音');assert.equal(await dialog.getByRole('table',{name:'导入结果预览'}).locator('tbody tr').count(),3);await dialog.getByRole('button',{name:'保存字表并使用'}).click();await dialog.waitFor({state:'hidden'});await waitValues('hɑŋ˧˥|kiʔ|?');
    await page.locator('.m13-ambiguous').click();await page.getByRole('dialog',{name:'选择 行 的读音'}).getByRole('button',{name:'读音 2'}).click();await waitValues('ɕiŋ˧˥|kiʔ|?');
    report.checks.push('CSV BOM/header/validation, named save, direct Unicode mapping, polyphone picker, missing/punctuation handling');
    dialog=await importer('hanzi.csv','hanzi','调查字音');await dialog.getByRole('button',{name:'保存字表并使用'}).click();await dialog.getByRole('alert').filter({hasText:'同名'}).waitFor();assert.equal(await page.locator('[aria-label="转换标准"] optgroup option').count(),1);await dialog.getByRole('button',{name:'取消',exact:true}).click();
    dialog=await importer('shared.xlsx','hanzi','工作表字音');assert.match(await dialog.innerText(),/mɑ˥/);await dialog.getByRole('combobox',{name:'字表工作表'}).selectOption('1');await dialog.getByRole('table',{name:'导入结果预览'}).locator('tbody').getByText('nɯ˩',{exact:true}).first().waitFor();await page.screenshot({path:path.join(out,'import-light.png')});report.screenshots.push('import-light.png');
    await dialog.getByRole('button',{name:'保存字表并使用'}).click();await dialog.waitFor({state:'hidden'});await input().fill('女妈');await waitValues('nɯ˩|?');
    dialog=await importer('stored.xlsx','hanzi','存储模式');await dialog.getByRole('table',{name:'导入结果预览'}).locator('tbody').getByText('mɑ˥',{exact:true}).first().waitFor();await dialog.getByRole('button',{name:'取消',exact:true}).click();
    report.checks.push('XLSX DEFLATE/stored, shared/inline strings, workbook sheet switching, duplicate-name refusal');
    dialog=await importer('syllables.tsv','syllable','拼音方案');await dialog.getByRole('button',{name:'保存字表并使用'}).click();await dialog.waitFor({state:'hidden'});await input().fill('妈麻马骂吗女');await waitValues('ma˥|mɑ|mɑ|mɑ|mə|ny˨˩˦');
    await page.getByRole('button',{name:'显示声调',exact:true}).click();await waitValues('ma|mɑ|mɑ|mɑ|mə|ny');await page.getByRole('button',{name:'显示声调',exact:true}).click();
    await page.getByRole('button',{name:/^保存本机草稿/}).click();
    await page.evaluate(()=>{window.__paint=[];const original=CanvasRenderingContext2D.prototype.fillText;CanvasRenderingContext2D.prototype.fillText=function(text,...args){window.__paint.push(text);return original.call(this,text,...args);};});
    const downloadEvent=page.waitForEvent('download');await page.getByRole('button',{name:'保存为 PNG',exact:true}).click();await page.getByRole('dialog',{name:'PNG 导出设置',exact:true}).getByRole('button',{name:'导出 PNG',exact:true}).click();const download=await downloadEvent;await download.saveAs(path.join(scratch,'result.png'));
    const paint=await page.evaluate(()=>window.__paint);assert(paint.includes('拼音方案'));assert(paint.includes('ma˥'));assert(paint.includes('ny˨˩˦'));assert.equal((await fs.readFile(path.join(scratch,'result.png'))).subarray(1,4).toString(),'PNG');report.checks.push('Pinyin tone/ü/neutral matching, display tone, named PNG and explicit draft save');
    await context.close();context=null;await launch();assert.equal(await input().inputValue(),'');assert.equal(await page.locator('[aria-label="转换标准"] option:checked').innerText(),'拼音方案');assert.equal(await page.locator('[aria-label="转换标准"] optgroup option').count(),3);
    await page.getByRole('button',{name:'恢复本机草稿',exact:true}).click();await waitValues('ma˥|mɑ|mɑ|mɑ|mə|ny˨˩˦');
    report.checks.push('Full browser restart reopens persisted named library and last custom selection; old text restores separately');
    for(const theme of ['light','dark']){
      await page.evaluate(theme=>document.documentElement.dataset.theme=theme,theme);await page.setViewportSize(theme==='light'?{width:1440,height:900}:{width:900,height:700});
      await page.screenshot({path:path.join(out,`result-${theme}.png`)});report.screenshots.push(`result-${theme}.png`);
      dialog=await importer('syllables.tsv','syllable','第二套');assert(await dialog.getByRole('button',{name:'保存字表并使用'}).isVisible());
      assert(await dialog.evaluate(el=>el.getBoundingClientRect().bottom<=innerHeight+1));await dialog.getByRole('button',{name:'取消',exact:true}).click();
    }
    await page.setViewportSize({width:1440,height:900});await page.evaluate(()=>document.documentElement.dataset.theme='light');
    await page.getByRole('combobox',{name:'转换标准'}).selectOption({label:'调查字音'});await waitValues('?|?|?|?|?|?');
    const originalId=await page.getByRole('combobox',{name:'转换标准'}).inputValue();await input().fill('行');await page.locator('.m13-ambiguous').click();await page.getByRole('dialog',{name:'选择 行 的读音'}).getByRole('button',{name:'读音 2'}).click();await waitValues('ɕiŋ˧˥');await page.getByRole('button',{name:/^保存本机草稿/}).click();
    const replacement=async(file)=>{await page.getByRole('button',{name:'替换字表',exact:true}).click();const dialog=page.getByRole('dialog',{name:'替换自定义字表',exact:true});await dialog.getByLabel('选择字表文件',{exact:true}).setInputFiles(path.join(scratch,file));await page.waitForFunction(()=>!document.querySelector('[aria-label="选择字表文件"]').disabled);return dialog;};
    dialog=await replacement('invalid.csv');assert(await dialog.getByRole('button',{name:'替换字表并使用'}).isDisabled());await dialog.getByRole('button',{name:'取消',exact:true}).click();await waitValues('ɕiŋ˧˥');
    dialog=await replacement('replacement.csv');assert.equal(await dialog.getByRole('textbox',{name:'字表名称'}).inputValue(),'调查字音');await dialog.getByRole('textbox',{name:'字表名称'}).fill('拼音方案');await dialog.getByRole('button',{name:'替换字表并使用'}).click();await dialog.getByRole('alert').filter({hasText:'同名'}).waitFor();assert.equal(await page.locator('[aria-label="转换标准"] optgroup option').count(),3);
    await dialog.getByRole('textbox',{name:'字表名称'}).fill('调查字音新版');await page.screenshot({path:path.join(out,'replace-light.png')});report.screenshots.push('replace-light.png');await dialog.getByRole('button',{name:'替换字表并使用'}).click();await dialog.waitFor({state:'hidden'});await waitValues('a˥');assert.equal(await page.getByRole('combobox',{name:'转换标准'}).inputValue(),originalId);assert.equal(await page.locator('[aria-label="转换标准"] optgroup option').count(),3);
    await context.close();context=null;await launch();assert.equal(await page.getByRole('combobox',{name:'转换标准'}).inputValue(),originalId);assert.equal(await page.locator('[aria-label="转换标准"] option:checked').innerText(),'调查字音新版');
    await page.getByRole('button',{name:'恢复本机草稿',exact:true}).click();await waitValues('a˥');assert.match(await page.locator('.m13-settings-section').innerText(),/字表已替换/);assert.equal(await input().inputValue(),'行');await page.getByRole('button',{name:/^保存本机草稿/}).click();
    report.checks.push('Replacement preview/cancel, invalid/duplicate-name refusal, same identity/count, rename and restart; stale polyphone indices cleared while draft text survives');
    await page.getByRole('button',{name:'删除字表',exact:true}).click();dialog=page.getByRole('dialog',{name:'删除自定义字表',exact:true});assert.match(await dialog.innerText(),/调查字音新版/);await dialog.getByRole('button',{name:'取消',exact:true}).click();await waitValues('a˥');
    await page.evaluate(()=>document.documentElement.dataset.theme='dark');await page.getByRole('button',{name:'删除字表',exact:true}).click();dialog=page.getByRole('dialog',{name:'删除自定义字表',exact:true});await page.screenshot({path:path.join(out,'delete-dark.png')});report.screenshots.push('delete-dark.png');await dialog.getByRole('button',{name:'删除字表',exact:true}).click();await dialog.waitFor({state:'hidden'});assert.equal(await input().inputValue(),'行');assert.equal(await page.getByRole('combobox',{name:'转换标准'}).inputValue(),'Standard Chinese (Beijing)');assert.equal(await page.locator('[aria-label="转换标准"] optgroup option').count(),2);assert.equal(await page.getByRole('button',{name:'删除字表',exact:true}).count(),0);
    await context.close();context=null;await launch();assert.equal(await page.locator('[aria-label="转换标准"] optgroup option').count(),2);assert.equal(await page.locator(`[aria-label="转换标准"] option[value="${originalId}"]`).count(),0);await page.getByRole('button',{name:'恢复本机草稿',exact:true}).click();assert.equal(await input().inputValue(),'行');assert.match(await page.locator('.m13-settings-section').innerText(),/已删除或不可用/);assert.equal(await page.getByRole('combobox',{name:'转换标准'}).inputValue(),'Standard Chinese (Beijing)');
    report.checks.push('Delete confirmation/cancel, selected-table fallback with input intact, other tables retained, full browser restart and deleted-table draft recovery');
    dialog=await importer('wide.tsv','hanzi','任意列调值');
    assert.equal(await dialog.getByLabel('数据起始行',{exact:true}).inputValue(),'4');
    await dialog.getByLabel('汉字或音节列',{exact:true}).selectOption('0');assert(await dialog.getByRole('button',{name:'保存字表并使用'}).isDisabled());
    await dialog.getByLabel('汉字或音节列',{exact:true}).selectOption('3');await dialog.getByLabel('音标列',{exact:true}).selectOption('3');assert(await dialog.getByRole('button',{name:'保存字表并使用'}).isDisabled());
    await dialog.getByLabel('音标列',{exact:true}).selectOption('1');await dialog.getByLabel('数据起始行',{exact:true}).fill('3');assert.match(await dialog.getByRole('alert').innerText(),/第 3 行/);
    await dialog.getByLabel('数据起始行',{exact:true}).fill('4');await dialog.getByLabel('音标声调表示方式',{exact:true}).selectOption('values');
    const preview=dialog.getByRole('table',{name:'导入结果预览'});await preview.getByText('ma˨˩˦',{exact:true}).waitFor();assert.match(await dialog.getByRole('table',{name:'原始字表预览'}).innerText(),/备注丙/);
    await page.screenshot({path:path.join(out,'columns-tone-values.png')});report.screenshots.push('columns-tone-values.png');
    await dialog.getByRole('button',{name:'保存字表并使用'}).click();await dialog.waitFor({state:'hidden'});await input().fill('妈麻马');await waitValues('ma˥˥|ma˧˥|ma˨˩˦');await page.getByRole('button',{name:'显示声调',exact:true}).click();await waitValues('ma|ma|ma');await page.getByRole('button',{name:'显示声调',exact:true}).click();
    report.checks.push('Arbitrary key/IPA columns and exact first row, extra notes ignored, original row preview, numeric/superscript/fullwidth values become IPA tone letters');
    dialog=await importer('classes.tsv','hanzi','调类字表');await dialog.getByLabel('音标声调表示方式',{exact:true}).selectOption('classes');
    await dialog.getByRole('button',{name:'保存字表并使用'}).click();await dialog.waitFor({state:'hidden'});await waitValues('ma¹|ma2|ma３');
    await page.getByRole('button',{name:'声调映射',exact:true}).click();dialog=page.getByRole('dialog',{name:'字表声调映射',exact:true});
    await dialog.getByLabel('调类显示方式',{exact:true}).selectOption('values');assert(await dialog.getByRole('button',{name:'保存声调映射'}).isDisabled());
    await dialog.getByLabel('调类 1 的调值',{exact:true}).fill('33');await dialog.getByLabel('调类 2 的调值',{exact:true}).fill('6');await dialog.getByLabel('调类 3 的调值',{exact:true}).fill('214');assert(await dialog.getByRole('button',{name:'保存声调映射'}).isDisabled());
    await dialog.getByLabel('调类 2 的调值',{exact:true}).fill('21');await dialog.getByRole('table',{name:'声调结果预览'}).getByText('ma˧˧',{exact:true}).waitFor();
    await page.evaluate(()=>document.documentElement.dataset.theme='dark');await page.setViewportSize({width:900,height:700});await page.screenshot({path:path.join(out,'tone-classes-dark.png')});report.screenshots.push('tone-classes-dark.png');
    await dialog.getByRole('button',{name:'取消',exact:true}).click();await waitValues('ma¹|ma2|ma３');
    await page.getByRole('button',{name:'声调映射',exact:true}).click();dialog=page.getByRole('dialog',{name:'字表声调映射',exact:true});assert.equal(await dialog.getByLabel('调类显示方式',{exact:true}).inputValue(),'original');await dialog.getByLabel('调类显示方式',{exact:true}).selectOption('values');
    for(const [key,value] of [['1','33'],['2','21'],['3','214']])await dialog.getByLabel('调类 '+key+' 的调值',{exact:true}).fill(value);
    await dialog.getByRole('button',{name:'保存声调映射'}).click();await dialog.waitFor({state:'hidden'});await waitValues('ma˧˧|ma˨˩|ma˨˩˦');await page.getByRole('button',{name:/^保存本机草稿/}).click();
    await context.close();context=null;await launch();await page.getByRole('button',{name:'恢复本机草稿',exact:true}).click();await waitValues('ma˧˧|ma˨˩|ma˨˩˦');
    await page.getByRole('button',{name:'声调映射',exact:true}).click();dialog=page.getByRole('dialog',{name:'字表声调映射',exact:true});assert.equal(await dialog.getByLabel('调类 1 的调值',{exact:true}).inputValue(),'33');await dialog.getByLabel('调类显示方式',{exact:true}).selectOption('original');await dialog.getByRole('table',{name:'声调结果预览'}).getByText('ma¹',{exact:true}).last().waitFor();await dialog.getByRole('button',{name:'保存声调映射'}).click();await dialog.waitFor({state:'hidden'});await waitValues('ma¹|ma2|ma３');
    report.checks.push('Editable tone classes preserve original labels, validate custom maps, cancel safely, survive a full browser restart and restore original IPA without reimport');
    await page.getByRole('button',{name:'帮助',exact:true}).click();await page.getByRole('heading',{name:/导入并保存自己的对照表/}).waitFor();report.checks.push('Light/dark and narrow import controls, real module help and generated custom-table section');
    await page.addInitScript(()=>{IDBFactory.prototype.open=()=>{throw Error('owned storage denial');};});
    await page.reload();await page.locator('.home-page').waitFor();await page.locator('.nav-item').and(page.getByRole('button',{name:'汉字转国际音标',exact:true})).click();
    await page.getByRole('button',{name:'重新载入已存字表',exact:true}).waitFor();await page.getByRole('combobox',{name:'转换标准'}).selectOption('汉语拼音');await input().fill('妈');await waitValues('mā');
    report.checks.push('Denied IndexedDB shows retry while built-in conversion and standard selection remain usable');
    assert.deepEqual(report.errors,[]);assert.deepEqual(report.uploads,[]);report.success=true;console.log(JSON.stringify(report,null,2));
  }catch(error){report.failure=String(error.stack||error);if(page)await page.screenshot({path:path.join(out,'failure.png')}).catch(()=>{});throw error;}
  finally{if(context)await context.close();await server.close();await fs.writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2));}
})().catch(error=>{console.error(error);process.exitCode=1;});
