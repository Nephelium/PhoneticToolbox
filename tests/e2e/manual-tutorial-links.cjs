// Focused, read-only manual/website verification. Third-party iframe requests
// are stubbed: this verifies links and player lifecycle, not Bilibili playback.
const fs=require('node:fs/promises'),path=require('node:path'),http=require('node:http'),assert=require('node:assert/strict'),vm=require('node:vm'),{createHash}=require('node:crypto');
const root=path.resolve(__dirname,'../..'),[out,scratch,beforeFile]=process.argv.slice(2),site=path.join(root,'temporary-site');
const specific={m01:'BV1NcJd6UEZv',m02:'BV11cJd6UESc',m03:'BV1KFJd6JEvg',m04:'BV17mnc6QExx',m05:'BV1NsnJ6nE2Z',m07:'BV1g1ns63EXQ',m08:'BV1QCns6kEqY',m09:'BV17mnc6QEYY',m11:'BV1g1ns63EyT',m12:'BV17mnc6QEJN',m13:'BV13Cns6kExb',m14:'BV17mnc6QEJf',m15:'BV1Bmnc6QEjn',m16:'BV1xsnJ6nES2',m17:'BV17mnc6QEYL',m18:'BV1Bmnc6QELa'};
const pending={m06:{key:'speech-synthesis',title:'声学参数合成'},m10:{key:'vocal-tract',title:'生理参数合成'},m19:{key:'geolinguistics',title:'语言地图'}};
const text=node=>(node.text??'')+(node.content??[]).map(text).join('');
const hrefs=node=>[...(node.marks??[]).filter(m=>m.type==='link').map(m=>m.attrs.href),...(node.content??[]).flatMap(hrefs)];
async function main(){
 assert(out&&scratch&&beforeFile,'Use scripts/verify_manual_tutorial_links.py');
 const report={success:false,checks:[],errors:[],chapters:[],scope:'Local manual source + actual website JS in owned Chrome; inert third-party player fixture; no publishing'};
 let server,context,page;
 try{
  const before=JSON.parse(await fs.readFile(beforeFile,'utf8')),project=JSON.parse(await fs.readFile(path.join(root,'manual/project.json'),'utf8'));
  const inputNames=['manual/project.json','temporary-site/site.js','temporary-site/index.html','scripts/build_website_expansion.py',...project.chapters.map(chapter=>'manual/'+chapter.path)];
  const hashes=async()=>Object.fromEntries(await Promise.all(inputNames.map(async name=>[name,createHash('sha256').update(await fs.readFile(path.join(root,name))).digest('hex')])));
  report.inputSha256=await hashes();report.baselineSha256=createHash('sha256').update(await fs.readFile(beforeFile)).digest('hex');
  const script=await fs.readFile(path.join(site,'site.js'),'utf8'),match=script.match(/const videos\s*=\s*({[\s\S]*?\n\s*});/);assert(match,'tutorial videos declaration');
  const videos=vm.runInNewContext('('+match[1]+')'),byBvid=new Map(Object.entries(videos).map(([key,video])=>[video.bvid,key]));
  assert.equal(Object.keys(videos).length,18);assert.equal(byBvid.size,18);
  for(const bvid of [...Object.values(specific),'BV1WWpr67EkX','BV1RuJm6gEWE'])assert(byBvid.has(bvid),'Published tutorial missing: '+bvid);
  assert.equal(project.chapters.length,22);assert.equal(Object.keys(before.chapters).length,22);
  for(const chapter of project.chapters){
   const current=JSON.parse(await fs.readFile(path.join(root,'manual',chapter.path),'utf8')),first=current.body.content[0],links=hrefs(first),plain=text(first);
   assert.equal(first.type,'paragraph',chapter.id+' tutorial must be first paragraph');assert.equal(first.attrs.id,chapter.id+'-video-tutorial');assert.equal(links.length,1);
   const url=new URL(links[0]);assert.equal(url.origin,'https://www.phonetictoolbox.com');assert.equal(url.pathname,'/');assert.equal(url.search,'');
   if(specific[chapter.id])assert.equal(url.hash,'#tutorial-'+byBvid.get(specific[chapter.id]),chapter.id+' functional mapping');
   else if(pending[chapter.id]){assert.equal(url.hash,'#tutorial-'+pending[chapter.id].key);assert(!Object.hasOwn(videos,pending[chapter.id].key),'Unpublished chapter must not invent a video');}
   else{assert(['web-introduction','getting-started','settings'].includes(chapter.id));assert.equal(url.hash,'#tutorial-intro');assert.match(plain,/介绍/);}
   const stripped=structuredClone(current);stripped.body.content.shift();assert.deepEqual(stripped,before.chapters[chapter.id],chapter.id+' existing content/metadata changed');
   report.chapters.push({id:chapter.id,href:links[0],text:plain});
  }
  const added=project.searchIndex.filter(row=>row.targetId?.endsWith('-video-tutorial'));
  assert.equal(added.length,22);assert.deepEqual(project.searchIndex.filter(row=>!row.targetId?.endsWith('-video-tutorial')),before.searchIndex);
  report.checks.push('All 22 opening paragraphs use correct short links: 16 published, 3 introduction, 3 chapter-specific pending destinations; original chapter objects and all original search entries preserved');
  const html=await fs.readFile(path.join(site,'index.html'),'utf8'),nojs=html.match(/<noscript><p>前往 B 站观看：[\s\S]*?<\/noscript>/)?.[0];assert(nojs);
  for(const bvid of byBvid.keys())assert(nojs.includes('https://www.bilibili.com/video/'+bvid+'/'),'noscript '+bvid);
  server=http.createServer(async(req,res)=>{try{const url=new URL(req.url,'http://localhost');if(url.pathname==='/site-stats.json'||url.pathname==='/site-stats-history.json'){res.setHeader('Content-Type','application/json');res.end(JSON.stringify(url.pathname.includes('history')?{schema:1,points:[]}:{schema:1,pageViews:0,downloads:{total:0}}));return;}const file=path.resolve(site,'.'+decodeURIComponent(url.pathname)+(url.pathname.endsWith('/')?'index.html':''));if(!file.startsWith(site+path.sep)){res.writeHead(403).end();return;}const raw=await fs.readFile(file),mime={'.html':'text/html','.js':'application/javascript','.css':'text/css','.json':'application/json','.svg':'image/svg+xml','.png':'image/png','.jpg':'image/jpeg','.ttf':'font/ttf'};res.setHeader('Content-Type',mime[path.extname(file)]||'application/octet-stream');res.end(raw);}catch{res.writeHead(404).end();}});
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));const origin='http://127.0.0.1:'+server.address().port;
  const {chromium}=require(path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  context=await chromium.launchPersistentContext(path.join(scratch,'chrome-profile'),{executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--mute-audio'],viewport:{width:1440,height:1000},reducedMotion:'reduce'});
  await context.route('https://player.bilibili.com/**',route=>route.fulfill({status:200,contentType:'text/html',body:'<!doctype html><title>Owned inert player fixture</title>'}));
  await context.addInitScript(()=>{window.__tutorialCreated=0;new MutationObserver(rows=>{for(const row of rows)if(row.target.id==='tutorial-screen')for(const node of row.addedNodes)if(node.tagName==='IFRAME')window.__tutorialCreated++;}).observe(document,{subtree:true,childList:true});});
  page=await context.newPage();page.on('pageerror',error=>report.errors.push(error.message));page.setDefaultTimeout(15000);
  const ready=()=>page.waitForFunction(()=>document.getElementById('tutorial-intro')&&document.readyState==='complete');
  const fresh=async hash=>{await page.goto('about:blank');await page.goto(origin+'/'+hash);};
  await page.goto(origin);await ready();assert.equal(await page.locator('#tutorial-screen iframe').count(),0);assert.equal(await page.evaluate(()=>window.__tutorialCreated),0);
  async function check(key){
   await page.waitForFunction(bvid=>document.querySelector('#tutorial-screen iframe')?.src.includes('bvid='+bvid),videos[key].bvid);
   const frames=page.locator('#tutorial-screen iframe');assert.equal(await frames.count(),1);const url=new URL(await frames.getAttribute('src'));assert.equal(url.hostname,'player.bilibili.com');assert.equal(url.searchParams.get('autoplay'),'0');assert.equal(url.searchParams.get('page'),'1');
   assert.equal(await page.locator('.tutorial-actions a').getAttribute('href'),'https://www.bilibili.com/video/'+videos[key].bvid+'/');
   const parent=key==='spectrogram-special'?'spectrogram':key;assert.equal(await page.locator('[data-tutorial="'+parent+'"]').getAttribute('aria-pressed'),'true');
  }
  for(const key of Object.keys(videos)){
   await fresh('#tutorial-'+key);await check(key);assert.equal(await page.evaluate(()=>window.__tutorialCreated),1,'initial hash duplicated iframe: '+key);
   await page.evaluate(()=>window.__tutorialFrame=document.querySelector('#tutorial-screen iframe'));
   const selector=key==='spectrogram-special'?'[data-tutorial-episode="spectrogram-special"]':'[data-tutorial="'+key+'"]';await page.locator(selector).click();
   assert(await page.evaluate(()=>window.__tutorialFrame===document.querySelector('#tutorial-screen iframe')),'repeat click reloads '+key);assert.equal(await page.evaluate(()=>window.__tutorialCreated),1);
  }
  let created=1;for(const key of Object.keys(videos)){const previous=await page.evaluate(()=>location.hash);await page.evaluate(hash=>location.hash=hash,'#tutorial-'+key);await check(key);if(previous!=='#tutorial-'+key)created++;assert.equal(await page.evaluate(()=>window.__tutorialCreated),created,'hash navigation duplicated iframe: '+key);}
  await page.screenshot({path:path.join(out,'tutorial-linked-desktop.png'),fullPage:false});
  for(const {key,title} of Object.values(pending)){
   await fresh('#tutorial-'+key);await ready();await page.locator('#tutorial-screen').getByText('教程更新中。',{exact:true}).waitFor();assert((await page.locator('#tutorial-screen').innerText()).includes(title));assert.equal(await page.locator('#tutorial-screen iframe').count(),0);assert.equal(await page.evaluate(()=>window.__tutorialCreated),0);
   await page.evaluate(()=>location.hash='#tutorial-intro');await check('intro');assert.equal(await page.evaluate(()=>window.__tutorialCreated),1);
   await page.evaluate(hash=>location.hash=hash,'#tutorial-'+key);await page.locator('#tutorial-screen').getByText('教程更新中。',{exact:true}).waitFor();assert.equal(await page.locator('#tutorial-screen iframe').count(),0);assert.equal(await page.evaluate(()=>window.__tutorialCreated),1);
   assert.equal(await page.locator('[data-tutorial="'+key+'"]').getAttribute('aria-pressed'),'true');assert.equal(await page.locator('[data-tutorial="'+key+'"]').isDisabled(),true);assert.equal(await page.locator('.tutorial-actions a').isVisible(),false);
  }
  for(const hash of ['#tutorials','#tutorial-missing','#tutorial-constructor']){await fresh(hash);await ready();assert.equal(await page.locator('#tutorial-screen iframe').count(),0);assert.equal(await page.evaluate(()=>window.__tutorialCreated),0);}
  await page.setViewportSize({width:390,height:844});for(const bvid of ['BV1g1ns63EXQ','BV1g1ns63EyT','BV13Cns6kExb','BV1QCns6kEqY']){const key=byBvid.get(bvid);await fresh('#tutorial-'+key);await check(key);assert.equal(await page.evaluate(()=>window.__tutorialCreated),1);}
  await page.locator('#tutorial-screen').screenshot({path:path.join(out,'tutorial-linked-mobile.png')});assert.deepEqual(report.errors,[]);
  for(const {key,title} of Object.values(pending)){await fresh('#tutorial-'+key);await ready();await page.locator('#tutorial-screen').getByText('教程更新中。',{exact:true}).waitFor();assert((await page.locator('#tutorial-screen').innerText()).includes(title));assert.equal(await page.locator('[data-tutorial="'+key+'"]').isDisabled(),true);assert.equal(await page.locator('#tutorial-screen iframe').count(),0);}
  assert.deepEqual(await hashes(),report.inputSha256,'Product inputs changed during read-only validation');report.checks.push('All 18 direct hashes and hash changes create one iframe with autoplay=0; repeats keep the same iframe; 3 pending destinations show chapter title/update notice, remove prior player and hide old video link; unknown/prototype hashes create none; 4 new tutorials also work at mobile width');report.success=true;
 }catch(error){report.failure=String(error);if(page)await page.screenshot({path:path.join(out,'failure.png')}).catch(()=>{});throw error;}
 finally{await fs.writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2)+'\n');if(context)await context.close();if(server)await new Promise(resolve=>server.close(resolve));}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
