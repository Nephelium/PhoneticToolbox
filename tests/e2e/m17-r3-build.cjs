// Verify static publication of curated content; no source file is modified.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict'),{pathToFileURL}=require('node:url'),{execFileSync}=require('node:child_process');
(async()=>{
 const root=path.resolve(__dirname,'../..'),out=path.join(root,'output/validation/m17-r3','build-'+Date.now());await fs.mkdir(out,{recursive:true});
 const {build,preview}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js'))),catalog=JSON.parse(await fs.readFile(path.join(root,'frontend/src/modules/ipa-plus/data/catalog.json'),'utf8'));
 const id=value=>catalog.entries.find(e=>e.system==='ipa'&&e.insertText===value&&!e.isExample).id;
 const content={version:1,entries:{[id('p')]:{notesZh:'源码内容分发验证',media:{audio:'m17-media/demo.wav'}},[id('b')]:{media:{video:'m17-media/demo.webm'}},[id('t')]:{media:{audio:'m17-media/demo.wav',video:'m17-media/demo.webm'}}}};
 const dist=path.join(out,'frontend');
 await build({root:path.join(root,'frontend'),plugins:[{name:'m17-build-fixture-only',enforce:'pre',load(file){if(file.split('?')[0].replaceAll('\\','/').endsWith('/ipa-plus/data/symbol-content.json'))return JSON.stringify(content);}}],build:{outDir:dist,emptyOutDir:false},logLevel:'error'});
 const assets=[];async function scan(dir){for(const item of await fs.readdir(dir,{withFileTypes:true})){const file=path.join(dir,item.name);if(item.isDirectory())await scan(file);else if(/\.(js|html)$/.test(item.name))assets.push(await fs.readFile(file,'utf8'));}}await scan(dist);
 assert(assets.some(x=>x.includes('源码内容分发验证')));assert(!assets.some(x=>/__m17_author|X-M17-Session|m17-author-server|音标内容维护/.test(x)),'ordinary build excludes owner editor and write API');
 const media=path.join(dist,'m17-media');await fs.mkdir(media,{recursive:true});
 const wav=Buffer.alloc(44+64000);wav.write('RIFF',0);wav.writeUInt32LE(wav.length-8,4);wav.write('WAVEfmt ',8);wav.writeUInt32LE(16,16);wav.writeUInt16LE(1,20);wav.writeUInt16LE(1,22);wav.writeUInt32LE(16000,24);wav.writeUInt32LE(32000,28);wav.writeUInt16LE(2,32);wav.writeUInt16LE(16,34);wav.write('data',36);wav.writeUInt32LE(wav.length-44,40);await fs.writeFile(path.join(media,'demo.wav'),wav);
 const ffmpeg=execFileSync('powershell.exe',['-NoProfile','-Command','(Get-Command ffmpeg -ErrorAction Stop).Source'],{encoding:'utf8',windowsHide:true}).trim();execFileSync(ffmpeg,['-hide_banner','-loglevel','error','-f','lavfi','-i','color=c=0x426b48:s=160x90:r=20','-t','2','-an','-c:v','libvpx',path.join(media,'demo.webm')],{windowsHide:true});
 const server=await preview({root:path.join(root,'frontend'),build:{outDir:dist},preview:{host:'127.0.0.1',port:0},logLevel:'error'}),{chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright')),browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true}),page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
 try{
  await page.goto(server.resolvedUrls.local[0]+'#M17');await page.locator('.m17-body[data-loaded=true][data-font-ready=true]').waitFor();await page.getByLabel('点击播放',{exact:true}).check();
  for(const value of ['p','b','t']){if(await page.getByRole('button',{name:'收起',exact:true}).count())await page.getByRole('button',{name:'收起',exact:true}).click();await page.locator(`[data-symbol-id="${id(value)}"]`).click();await page.waitForFunction(()=>[...document.querySelectorAll('.m17-playback audio,.m17-playback video')].every(e=>e.currentTime>0&&e.readyState>=2));}
  assert.equal((await fetch(server.resolvedUrls.local[0]+'__m17_author/content',{method:'PUT',headers:{'Content-Type':'application/json'},body:'{}'})).status,404);assert.deepEqual(errors,[]);
  await fs.writeFile(path.join(out,'report.json'),JSON.stringify({passed:true,dist,checks:['curated source content included in static production build','owner editor and write API absent from normal build, PUT 404','bundled WAV/WebM and combined playback decoded from real static server'],errors},null,2));console.log(out);
 }finally{await browser.close();await new Promise(r=>server.httpServer.close(r));}
})().catch(e=>{console.error(e);process.exitCode=1;});
