module.exports=async({page,click,audio,out,rpc,checks})=>{
 const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
 const input=path.join(out,'inputs'),original=path.join(input,audio),grid=path.join(input,audio.replace(/\.wav$/i,'.TextGrid'));
 // These are byte-for-byte copies of the already authorized recording, not generated audio.
 await fs.copyFile(original,path.join(input,'P17-空白.wav'));
 await fs.copyFile(original,path.join(input,'P17-歧义.wav'));
 for(const suffix of ['_A','_B'])await fs.copyFile(grid,path.join(input,'P17-歧义'+suffix+'.TextGrid'));
 await click('扫描');await page.locator('.annotation-file-list button[title="P17-空白.wav"]').click();await page.getByRole('button',{name:'创建标注层',exact:true}).waitFor();await click('创建标注层');await page.getByLabel('新建音节层名',{exact:true}).fill('真实副本音节');await page.getByLabel('新建音素层名',{exact:true}).fill('真实副本音素');await click('创建层');await page.getByRole('button',{name:/^保存 TextGrid/}).click();await page.getByText('已保存：P17-空白_自动保存.TextGrid',{exact:true}).waitFor();
 const saved=(await rpc({op:'inspect'})).value;assert.equal(saved.grids['P17-空白_自动保存.TextGrid'].tiers.length,2);assert(saved.originals_unchanged);
 await page.locator('.annotation-file-list button[title="P17-歧义.wav"]').click();await page.getByLabel('选择关联标注文件').selectOption({label:'P17-歧义_B.TextGrid'});await click('打开标注');await page.getByText('已加载：P17-歧义_B.TextGrid',{exact:true}).waitFor();await page.getByLabel('当前 TextGrid',{exact:true}).selectOption({label:'P17-歧义_A.TextGrid'});await page.getByText('已加载：P17-歧义_A.TextGrid',{exact:true}).waitFor();
 checks.push('byte-identical real recording copies: no-TextGrid explicit two-layer creation/save/readback; ambiguous paired grids require explicit B selection, current-grid selector switches to A; originals unchanged');
};
