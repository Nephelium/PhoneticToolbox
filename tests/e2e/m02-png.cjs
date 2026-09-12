// Owned Vite + independent Chrome; no DB, account or public network operations.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
const {crc32}=require('node:zlib');
const root=path.resolve(__dirname,'../..');
async function main(){
 const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
 const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
 const out=path.join(root,'output/validation/m02-png','chrome-'+Date.now());await fs.mkdir(out,{recursive:true});
 const server=await createServer({root:path.join(root,'frontend'),server:{host:'127.0.0.1',port:0}});await server.listen();
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
 const context=await browser.newContext({viewport:{width:1440,height:1100},acceptDownloads:true});const page=await context.newPage(),checks=[];
 try{
  await page.goto(server.resolvedUrls.local[0]+'tests/m02-export.html');
  await page.getByRole('button',{name:'元音ɑ̃.wav',exact:true}).click();await page.locator('.parameter-curve').first().waitFor();
  assert.equal(await page.locator('.right-tick').count(),5);
  async function save(name,figure=0){const waiting=page.waitForEvent('download');await page.locator('.parameter-figure').nth(figure).getByRole('button',{name:'保存整幅 PNG',exact:true}).click();const download=await waiting;assert(download.suggestedFilename().endsWith('.png'));await download.saveAs(path.join(out,name));}
  await save('light.png');await page.evaluate(()=>document.documentElement.dataset.theme='dark');await page.getByLabel('显示两个声道').check();await save('dark-stereo.png');
  await page.getByRole('button',{name:'新建图窗',exact:true}).click();assert(await page.locator('.parameter-figure').nth(1).getByRole('button',{name:'保存整幅 PNG'}).isDisabled());
  await page.locator('.m02-parameters input').nth(0).check();await page.getByRole('button',{name:'将 1 项分配到图窗',exact:true}).click();await save('figure-2.png',1);
  await page.getByLabel('时间窗长度（秒）',{exact:true}).fill('0.5');await page.getByLabel('时间窗长度（秒）',{exact:true}).blur();
  await page.getByLabel('时间窗起点（秒）',{exact:true}).fill('0.2');await page.getByLabel('时间窗起点（秒）',{exact:true}).blur();
  await save('zoom.png',1);
  const snapshot=await page.evaluate(async()=>{
    const {wholeFigureSvg}=await import('/src/modules/parameter-display/export.ts');
    const chart=document.querySelectorAll('.parameter-chart')[1],waveform=document.querySelector('.wave-viewport');
    const result=wholeFigureSvg({chart,waveform,title:'中文 / ɑ̃˥',start:.2,end:.7,plotLeft:76,plotRight:chart.viewBox.baseVal.width-22});
    const doc=new DOMParser().parseFromString(result.text,'image/svg+xml');
    const nested=[...doc.documentElement.children].filter(e=>e.tagName==='svg');
    return {text:result.text,width:result.width,height:result.height,tracks:nested.map(e=>({x:e.getAttribute('x'),width:e.getAttribute('width'),hasCurve:!!e.querySelector('.parameter-curve')}))};
  });
  assert.equal(snapshot.tracks.length,3);assert.equal(snapshot.tracks[0].x,'76');assert.equal(snapshot.tracks[0].width,snapshot.tracks[1].width);assert(snapshot.tracks[2].hasCurve);assert(snapshot.text.includes('0.200'));assert(snapshot.text.includes('0.700'));assert(snapshot.text.includes('ɑ̃˥'));await fs.writeFile(path.join(out,'snapshot.svg'),snapshot.text);
  checks.push('real PNG downloads: light/dark, stereo, dual axis, second group, zoom; SVG snapshot times and aligned tracks');
  // Fault injection: active spectrogram cannot disappear silently from export.
  await page.evaluate(()=>{const e=document.createElement('section');e.className='spectrogram-view';e.innerHTML='<p role="status">loading fixture</p>';document.querySelector('.wave-viewport').append(e);});
  await page.locator('.parameter-figure').nth(1).getByRole('button',{name:'保存整幅 PNG'}).click();await page.getByRole('alert').filter({hasText:'语谱图尚未就绪'}).waitFor();
  await page.evaluate(()=>document.querySelector('.spectrogram-view').remove());checks.push('pending spectrogram explicitly rejects incomplete export');
  await page.setViewportSize({width:390,height:900});await page.locator('.parameter-figure').nth(1).scrollIntoViewIfNeeded();assert(await page.evaluate(()=>document.documentElement.scrollWidth)<=391);await save('narrow.png',1);
  await page.screenshot({path:path.join(out,'page.png'),fullPage:true});
  const images={};
  for(const name of ['light.png','dark-stereo.png','figure-2.png','zoom.png','narrow.png']){
   const bytes=await fs.readFile(path.join(out,name));let physical;
   for(let i=8;i<bytes.length;){const n=bytes.readUInt32BE(i),type=bytes.toString('ascii',i+4,i+8);assert.equal(crc32(bytes.subarray(i+4,i+8+n)),bytes.readUInt32BE(i+8+n));if(type==='pHYs')physical=[bytes.readUInt32BE(i+8),bytes.readUInt32BE(i+12),bytes[i+16]];i+=12+n;}
   assert.deepEqual(physical,[11811,11811,1]);
   const decoded=await page.evaluate(async data=>{const image=new Image();image.src=data;await image.decode();const canvas=document.createElement('canvas');canvas.width=image.width;canvas.height=image.height;const ctx=canvas.getContext('2d');ctx.drawImage(image,0,0);return {width:image.width,height:image.height,corner:[...ctx.getImageData(0,0,1,1).data]};},'data:image/png;base64,'+bytes.toString('base64'));
   assert(decoded.width>1000&&decoded.height>1800);assert.deepEqual(decoded.corner,[255,255,255,255]);images[name]={...decoded,dpi:299.9994};
  }
  assert(images['dark-stereo.png'].height>images['light.png'].height);
  await fs.writeFile(path.join(out,'image-readback.json'),JSON.stringify(images,null,2));checks.push('five PNG files independently decoded with all chunk CRCs, 300 dpi and white pixels verified');
  await fs.writeFile(path.join(out,'report.json'),JSON.stringify({success:true,checks,output:out},null,2));console.log(out);
 }catch(e){await page.screenshot({path:path.join(out,'failed.png'),fullPage:true});console.error(out);throw e;}
 finally{await context.close();await browser.close();await server.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
