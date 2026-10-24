/* Real M19 SFC in independent Chrome; official administrative-tree sample only.
 * Storage is real, transport/map geometries/survey records are test fixtures.
 * Run: node tests/e2e/geolinguistics-regions.cjs
 */
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {createHash,randomUUID}=require('node:crypto'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..'),frontend=path.join(root,'frontend');
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
(async()=>{
 const base=path.join(root,'output/validation/geolinguistics-regions');await fs.mkdir(base,{recursive:true});
 const out=path.join(base,'browser-'+randomUUID().replaceAll('-','')),scratch=path.join(out,'scratch');await fs.mkdir(scratch,{recursive:true});
 const report={success:false,checks:[],errors:[],screenshots:[],sourceIdentity:{},limitations:['Actual M19 SFC and GeoMap; test-only port supplies a 42-node official GeoAtlas sample, empty geometry and synthetic survey rows. This does not verify geographical completeness, 930-site science, native devices or final packages.'],environment:{platform:process.platform,browser:'independent Chrome profile',storage:'real isolated IndexedDB'}};
 let server,context,page,stage='build';const check=text=>report.checks.push(text);
 try{
  const vite=await import(pathToFileURL(path.join(frontend,'node_modules/vite/dist/node/index.js'))),vue=(await import(pathToFileURL(path.join(frontend,'node_modules/@vitejs/plugin-vue/dist/index.mjs')))).default;
  const {chromium}=require(process.env.PTB_PLAYWRIGHT_PATH||path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  for(const name of ['frontend/src/modules/geolinguistics/GeolinguisticsPage.vue','frontend/src/modules/geolinguistics/model.ts','frontend/tests/fixtures/geolinguistics-regions.ts'])report.sourceIdentity[name]=sha(await fs.readFile(path.join(root,name)));
  const dist=path.join(scratch,'dist');
  await vite.build({configFile:false,root:frontend,base:'./',publicDir:false,cacheDir:path.join(scratch,'vite-cache'),logLevel:'error',plugins:[vue(),{name:'owned-region-fixture-port',enforce:'pre',resolveId(source,importer){if(source==='../../platform/geolinguistics.ts'&&importer?.split('?')[0].endsWith('GeolinguisticsPage.vue'))return path.join(frontend,'tests/fixtures/geolinguistics-region-port.ts');}}],build:{outDir:dist,target:'es2022',rolldownOptions:{input:path.join(frontend,'tests/geolinguistics-regions.html')}}});
  server=await vite.preview({configFile:false,root:frontend,build:{outDir:dist},preview:{host:'127.0.0.1',port:0},logLevel:'error'});
  context=await chromium.launchPersistentContext(path.join(scratch,'profile'),{executablePath:process.env.PTB_CHROME_PATH||'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,viewport:{width:1440,height:960}});
  page=await context.newPage();page.setDefaultTimeout(15000);page.on('pageerror',error=>report.errors.push({stage,message:error.message}));
  await page.goto(server.resolvedUrls.local[0]+'tests/geolinguistics-regions.html');await page.locator('.atlas-workbench').waitFor();
  const province=()=>page.getByLabel('省级地区',{exact:true}),next=()=>page.getByLabel('下一级地区',{exact:true}),county=()=>page.getByLabel('县级地区',{exact:true});
  const values=locator=>locator.locator('option').evaluateAll(options=>options.map(o=>o.value).filter(Boolean));
  const oneRecord=()=>page.waitForFunction(()=>document.querySelector('.table-toolbar span')?.textContent.trim()==='1 条记录');
  stage='direct-districts';
  for(const [p,d] of [['110000','110101'],['120000','120101'],['310000','310101'],['500000','500101'],['810000','810001'],['820000','820001']]){
   await province().selectOption('province:'+p);assert.deepEqual(await values(next()),['district:'+d]);await next().selectOption('district:'+d);await oneRecord();assert.equal(await next().inputValue(),'district:'+d);assert.equal(await county().count(),0);
  }check('Six municipality/SAR provinces expose their actual direct districts; selection filters to one record without inventing a city.');
  stage='source-city-leaves';
  for(const [p,c] of [['410000','419001'],['420000','429004'],['420000','429005'],['420000','429006'],['420000','429021'],['460000','469001'],['460000','469024'],['650000','659001'],['650000','659010'],['440000','441900'],['440000','442000'],['460000','460400'],['620000','620200']]){
   await province().selectOption('province:'+p);assert((await values(next())).includes('city:'+c));await next().selectOption('city:'+c);await oneRecord();assert.equal(await county().count(),0);await page.getByRole('status').filter({hasText:'整体查看，未提供下一级地区'}).waitFor();
  }check('Source-city terminal entities, including province-administered counties and four childless cities, remain selectable/filterable with no empty third selector.');
  stage='normal-three-level';await province().selectOption('province:440000');await next().selectOption('city:440100');assert.deepEqual(await values(county()),['district:440103']);await county().selectOption('district:440103');await oneRecord();
  assert.equal(await next().inputValue(),'city:440100');await county().selectOption('');assert.equal(await county().inputValue(),'');await next().selectOption('');assert.equal(await county().count(),0);check('Ordinary province/city/district paths and clearing a county/city retain the correct parent scope.');
  stage='province-only';await province().selectOption('province:710000');await oneRecord();assert.equal(await next().count(),0);assert.equal(await county().count(),0);await page.getByRole('status').filter({hasText:'台湾省整体查看'}).waitFor();assert.equal(await page.getByRole('alert').count(),0);
  for(const theme of ['light','dark']){await page.evaluate(theme=>document.documentElement.dataset.theme=theme,theme);await page.waitForTimeout(250);const file='province-only-'+theme+'.png';await page.screenshot({path:path.join(out,file)});report.screenshots.push(file);const size=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth}));assert(size.scroll<=size.width+1);}check('Province-only Taiwan clearly remains available at province scope in light/dark themes without a load failure or page overflow.');
  stage='national-reset';await province().selectOption('');assert.equal(await next().count(),0);assert.equal(await county().count(),0);assert.equal(await page.getByRole('button',{name:'世界',exact:true}).count(),0);check('National option clears lower selectors; China entry remains and no world button is introduced.');
  async function exported(){const before=await page.evaluate(()=>window.__regionExports.length);await page.getByRole('button',{name:'导出工程',exact:true}).click();await page.waitForFunction(n=>window.__regionExports.length>n,before);return page.evaluate(()=>JSON.parse(window.__regionExports.at(-1).text));}
  const old=await exported();old.name='直属地区旧工程';old.settings.region='district:110101';old.edits={'S1:F1_P':'kept'};old.classEdits={};old.customSites=[{id:'user:kept',name:'中性调查点',lon:113,lat:23,admin:null,value:'甲',note:'保留'}];old.drawings={type:'FeatureCollection',features:[{type:'Feature',geometry:{type:'LineString',coordinates:[[113,23],[114,24]]}}]};
  stage='old-project-import';await page.locator('.atlas-toolbar input[accept=".json"]').setInputFiles({name:'neutral-region.ptb-map.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(old))});await page.getByRole('dialog').getByRole('button',{name:'打开工程',exact:true}).click();await oneRecord();assert.equal(await province().inputValue(),'province:110000');assert.equal(await next().inputValue(),'district:110101');
  const imported=await exported();for(const field of ['edits','classEdits','customSites','drawings'])assert.deepEqual(imported[field],old[field]);assert.equal(imported.settings.region,old.settings.region);await page.waitForTimeout(650);await page.reload();await page.locator('.atlas-workbench').waitFor();await oneRecord();assert.equal(await next().inputValue(),'district:110101');check('Actual project import/export and independent IndexedDB reload preserve a direct-district selection and all research edits/drawings.');
  stage='final';assert.deepEqual(report.errors,[]);assert.equal(await page.getByRole('alert').count(),0);report.success=true;
 }catch(error){report.failure={stage,message:error.message,stack:error.stack};process.exitCode=1;if(page)await page.screenshot({path:path.join(out,'failure.png')}).catch(()=>{});}
 finally{if(context)await context.close();if(server)await new Promise(resolve=>server.httpServer.close(resolve));report.cleanup={state:'pending',path:scratch,ownedProcessesStopped:true};await fs.writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({out,success:report.success,checks:report.checks.length,failure:report.failure,cleanup:report.cleanup}));}
})().catch(error=>{console.error(error);process.exitCode=1;});
