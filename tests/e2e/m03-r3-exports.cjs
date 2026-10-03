// Print-only changes in shared M02 / M08 exports. Owned Chrome; display fixtures.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..');
async function main(){
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const out=path.join(root,'output/validation/m03-r3-exports','chrome-'+Date.now());await fs.mkdir(out,{recursive:true});
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0},optimizeDeps:{noDiscovery:true,include:['vue']}});await server.listen();
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true}),page=await browser.newPage({viewport:{width:1440,height:1000}}),checks=[],errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/m02-export.html');await page.getByRole('button',{name:'元音ɑ̃.wav',exact:true}).click();await page.locator('.empty-plot').waitFor();await page.getByRole('button',{name:'全选可见参数',exact:true}).click();await page.getByRole('button',{name:/^将 .* 项分配到图窗$/}).click();await page.getByRole('button',{name:'清空勾选',exact:true}).click();await page.locator('.parameter-curve').first().waitFor();
  for(const theme of ['light','dark']){
   await page.evaluate(theme=>document.documentElement.dataset.theme=theme,theme);
   const r=await page.evaluate(async()=>{
    const serial=XMLSerializer.prototype.serializeToString,xml=[];XMLSerializer.prototype.serializeToString=function(node){const text=serial.call(this,node);xml.push(text);return text;};
    try{
     const {currentFigurePng}=await import('/src/modules/parameter-display/export.ts'),blob=await currentFigurePng(document.querySelector('.parameter-chart'));
     return {xml:xml.find(x=>x.includes('text-anchor="middle"')),bytes:Array.from(new Uint8Array(await blob.arrayBuffer()))};
    }finally{XMLSerializer.prototype.serializeToString=serial;}
   });
   assert(r.xml.includes('text-anchor="middle"'));assert(r.xml.includes('#182332'));assert(r.xml.includes('ɑ'));await fs.writeFile(path.join(out,`m02-${theme}.png`),Buffer.from(r.bytes));await fs.writeFile(path.join(out,`m02-${theme}.svg`),r.xml);checks.push(`M02 ${theme}: real current-figure PNG, neutral centered title and IPA retained`);
  }
  await page.goto(server.resolvedUrls.local[0]+'tests/m03-r3-exports.html');await page.locator('.history-plot').waitFor();await page.waitForTimeout(100);
  const r=await page.evaluate(async()=>{
   const serial=XMLSerializer.prototype.serializeToString,xml=[];XMLSerializer.prototype.serializeToString=function(node){const text=serial.call(this,node);xml.push(text);return text;};
   try{const blob=await window.exportHistory();return {xml:xml.find(x=>x.includes('F0 历史对比')),bytes:Array.from(new Uint8Array(await blob.arrayBuffer()))};}
   finally{XMLSerializer.prototype.serializeToString=serial;}
  });
  assert(r.xml.includes('F0 历史对比'));assert(!r.xml.includes('条已保存音频'));assert(r.xml.includes('元音ɑ̃-5.wav'));assert(r.xml.includes('6 3'));assert(r.xml.includes('stroke: rgb(100, 116, 139)')||r.xml.includes('stroke: #64748b'));await fs.writeFile(path.join(out,'m08-history.png'),Buffer.from(r.bytes));await fs.writeFile(path.join(out,'m08-history.svg'),r.xml);checks.push('M08 real history PNG: centered neutral heading, filename mapping with swatches, axis independent of trace colors');
  assert.deepEqual(errors,[]);await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success:true,checks,errors},null,2));console.log(out);
 }finally{await browser.close();await server.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
