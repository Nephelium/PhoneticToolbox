module.exports=async({page,click,audio,out,checks})=>{
 const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
 await page.getByLabel('筛选标注文件').fill('不存在P17');assert.equal(await page.locator('.annotation-file-list button').count(),0);await page.getByLabel('筛选标注文件').fill('');await click('扫描');assert.equal(await page.locator('.annotation-file-list button').count(),1);
 await page.getByLabel('音量',{exact:true}).fill('0.35');assert.equal(await page.getByLabel('音量',{exact:true}).inputValue(),'0.35');await page.getByLabel('音量',{exact:true}).fill('0.7');
 for(const [button,file] of [['上传词典',path.join(__dirname,'../../frontend/src/modules/annotation/default.dict')],['上传词表','C:/Users/13680/Desktop/project/音频数据/creak/老年组 15人/1.凌静梅/00141低_梯_题.lab']]){const chosen=page.waitForEvent('filechooser');await click(button);await(await chosen).setFiles(file);await page.waitForTimeout(100);}
 await page.locator('.sequence-toggle input').check();await page.getByLabel('下一音节',{exact:true}).selectOption('1');await page.getByLabel('下一音节',{exact:true}).selectOption('0');await page.locator('.sequence-toggle input').uncheck();await click('清除词表');
 await page.getByLabel('搜索词层文本').fill('P17');await click('上一个');await click('下一个');await page.getByLabel('替换文本').fill('P17 单次');await click('替换');await page.locator('.annotation-grid').focus();await page.keyboard.press('Control+z');
 const waiting=page.waitForEvent('download');await click('下载当前 TextGrid');const file=path.join(out,'downloaded.TextGrid');await(await waiting).saveAs(file);assert((await fs.readFile(file,'utf8')).includes('P17 井井 æ'));
 const ref=page.waitForEvent('filechooser');await click('选择参考 TextGrid');await(await ref).setFiles(path.join(out,'inputs',audio.replace(/\.wav$/i,'.TextGrid')));await page.waitForTimeout(100);await click('清除参考');
 checks.push('filter zero-match/clear, actual rescan, volume, both resource chooser buttons and original LAB/default dictionary import, next syllable, previous/next match, single replace/undo, actual TextGrid download, reference chooser');
};
