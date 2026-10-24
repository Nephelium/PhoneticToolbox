/* M19: production AppShell and real directory input. Run only with the owned Python runner. */
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {createHash}=require('node:crypto'),{pathToFileURL}=require('node:url');
const root=path.resolve(__dirname,'../..'),[out,scratch]=process.argv.slice(2);
const sha=b=>createHash('sha256').update(b).digest('hex');
function csvRows(text){
  const rows=[];let row=[],cell='',quoted=false;
  text=text.replace(/^\uFEFF/,'');
  for(let i=0;i<text.length;i++){
    const c=text[i];
    if(c==='"'){if(quoted&&text[i+1]==='"'){cell+='"';i++;}else quoted=!quoted;}
    else if(c===','&&!quoted){row.push(cell);cell='';}
    else if((c==='\n'||c==='\r')&&!quoted){if(c==='\r'&&text[i+1]==='\n')i++;row.push(cell);if(row.some(Boolean))rows.push(row);row=[];cell='';}
    else cell+=c;
  }
  if(cell||row.length){row.push(cell);rows.push(row);}return rows;
}
(async()=>{
  assert(out&&scratch,'Use scripts/verify_geolinguistics_browser.py');
  const {preview}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
  const {chromium}=require(process.env.PTB_PLAYWRIGHT_PATH||path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'));
  const report={success:false,checks:[],errors:[],requests:[],fileReads:[],exports:[],screenshots:[],
    limitations:['Native operating-system save picker is not exercised: the supported download fallback is selected.',
      'Account/workspace scope switching is not exercised in this local AppShell preview.'],
    environment:{platform:process.platform,frontend:'existing production dist',browser:'independent Chrome persistent profile'}};
  let server,context,page,stage='startup';
  const check=description=>report.checks.push(description);
  const atlas=()=>page.locator('.language-atlas');
  const screenshot=async name=>{await page.screenshot({path:path.join(out,name)});report.screenshots.push(name);};
  async function openModule(){
    await page.locator('.home-page').waitFor();
    await page.locator('.nav-item').and(page.getByRole('button',{name:'语言地图',exact:true})).click();
    await atlas().getByLabel('选择语言地图资料夹',{exact:true}).waitFor({state:'attached'});
    await page.waitForFunction(()=>{const el=document.querySelector('[aria-label="选择语言地图资料夹"]');return el&&!el.disabled;});
  }
  async function download(button,name){
    const event=page.waitForEvent('download',{timeout:90000});await button.click();
    const file=await event;assert.equal(await file.failure(),null);
    const target=path.join(scratch,name);await file.saveAs(target);const bytes=await fs.readFile(target);
    report.exports.push({name,suggestedFilename:file.suggestedFilename(),bytes:bytes.length,sha256:sha(bytes)});return bytes;
  }
  async function recordFileReads(){report.fileReads.push(...await page.evaluate(()=>window.__geoFileReads||[]));}
  async function waitCount(n){await page.waitForFunction(n=>[...document.querySelectorAll('.table-toolbar span')].some(el=>el.textContent.trim()===n+' 个地点'),n);}
  async function choose120(){
    await atlas().getByRole('tab',{name:'语言现象',exact:true}).click();
    await atlas().getByLabel('检索语言特征',{exact:true}).fill('120');
    await atlas().locator('.feature-list button').filter({hasText:'阳声韵尾的演变'}).click();
    await atlas().locator('.map-heading strong').filter({hasText:'阳声韵尾的演变'}).waitFor();
  }
  try{
    server=await preview({root:path.join(root,'frontend'),build:{outDir:process.env.PTB_GEO_DIST||path.join(root,'frontend/dist')},preview:{host:'127.0.0.1',port:0}});
    const url=server.resolvedUrls.local[0];report.environment.url=url;
    context=await chromium.launchPersistentContext(path.join(scratch,'profile'),{
      executablePath:process.env.PTB_CHROME_PATH||'C:/Program Files/Google/Chrome/Application/chrome.exe',
      headless:process.env.PTB_GEO_HEADED!=='1',viewport:{width:1440,height:900},acceptDownloads:true});
    context.setDefaultTimeout(30000);
    // Observe the actual byte-read methods; all implementations still run unchanged.
    await context.addInitScript(()=>{
      window.__geoFileReads=[];
      const observe=(file,method)=>{if(file instanceof File)window.__geoFileReads.push({name:file.name,path:file.webkitRelativePath,method});};
      for(const name of ['text','arrayBuffer','stream','slice']){
        const original=Blob.prototype[name];if(!original)continue;
        Blob.prototype[name]=function(...args){observe(this,name);return original.apply(this,args);};
      }
      for(const name of ['readAsArrayBuffer','readAsText','readAsDataURL','readAsBinaryString']){
        const original=FileReader.prototype[name];if(!original)continue;
        FileReader.prototype[name]=function(file,...args){observe(file,name);return original.call(this,file,...args);};
      }
      Object.defineProperty(window,'showSaveFilePicker',{configurable:true,value:undefined});
    });
    context.on('request',r=>report.requests.push({stage,url:r.url(),method:r.method(),type:r.resourceType(),bodyBytes:r.postDataBuffer()?.length||0}));
    context.on('page',p=>p.on('pageerror',e=>report.errors.push({stage,message:e.message})));
    page=await context.newPage();await page.goto(url);await openModule();
    assert.equal(await atlas().locator('.atlas-workbench').count(),0,'Fresh profile must require local data');
    check('M19 is reachable from production navigation; fresh browser requests local data');

    if(process.env.PTB_GEO_DOWNLOAD_ONLY==='1'){
      report.limitations.push('Official data endpoint is a controlled HTTPS response stand-in backed by the exact private runtime files; actual production HTTPS/CORS is verified after reviewed publication.');
      let mode='corrupt';const remote='https://www.phonetictoolbox.com/app/geolinguistics/';
      await context.route(remote+'**',async route=>{
        const name=route.request().url().slice(remote.length);assert.equal(route.request().method(),'GET');assert(!route.request().postData());
        if(mode==='offline')return route.abort();
        const allowed=['data-manifest.json',...JSON.parse(await fs.readFile(path.join(scratch,'geolinguistics-data/data-manifest.json'),'utf8')).files.map(v=>v.path)];assert(allowed.includes(name));
        let body=await fs.readFile(path.join(scratch,'geolinguistics-data',name));
        if(mode==='corrupt'&&name==='catalog.json'){body=Buffer.from(body);body[0]=91;}
        if(mode==='cancel'&&name==='observations.json')await new Promise(resolve=>setTimeout(resolve,1000));
        await route.fulfill({status:200,headers:{'access-control-allow-origin':'*','content-type':name.endsWith('.json')?'application/json':'application/octet-stream'},body}).catch(error=>{if(mode!=='cancel')throw error;});
      });
      stage='download-corruption';await atlas().locator('.atlas-empty').getByRole('button',{name:'下载资料',exact:true}).click();
      await atlas().getByRole('alert').filter({hasText:'校验未通过'}).waitFor();assert.equal(await atlas().locator('.atlas-workbench').count(),0);
      check('Fresh empty state offers download; same-length corrupted server data is rejected without committing a partial set');
      stage='download-cancel';mode='cancel';await atlas().locator('.atlas-empty').getByRole('button',{name:'下载资料',exact:true}).click();
      await atlas().getByRole('button',{name:'取消下载',exact:true}).click();await atlas().getByRole('status').filter({hasText:'下载已取消'}).waitFor();
      assert.equal(await atlas().locator('.atlas-workbench').count(),0);check('User cancellation ends the in-flight download and leaves a recoverable empty state');
      stage='download-success';mode='good';await atlas().locator('.atlas-empty').getByRole('button',{name:'下载资料',exact:true}).click();
      await atlas().locator('.atlas-workbench').waitFor({timeout:90000});await choose120();await waitCount(930);await screenshot('downloaded-real-data.png');
      const start=report.requests.length;mode='offline';await page.reload();await openModule();await atlas().locator('.atlas-workbench').waitFor();await choose120();await waitCount(930);
      assert.equal(report.requests.slice(start).filter(r=>r.url.startsWith(remote)).length,0);check('Complete real 16-file data automatically renders 930 sites and survives reload with the server offline; no repeat download');
      for(const theme of ['light','dark']){await page.evaluate(theme=>document.documentElement.dataset.theme=theme,theme);await screenshot(`download-${theme}.png`);}
      assert.deepEqual(report.requests.filter(r=>!['GET','HEAD'].includes(r.method)||r.bodyBytes),[]);assert.deepEqual(report.errors,[]);
      check('Actual download workflow makes read-only requests and uploads no research data');report.success=true;return;
    }

    stage='directory-import';
    await atlas().getByLabel('选择语言地图资料夹',{exact:true}).setInputFiles(path.join(scratch,'geolinguistics-data'));
    await atlas().locator('.atlas-workbench').waitFor();
    await atlas().getByRole('status').filter({hasText:'资料已载入并保存在本机'}).waitFor();
    await choose120();assert.equal(await atlas().getByLabel('分类层级',{exact:true}).inputValue(),'broad');await waitCount(930);
    assert.match(await atlas().locator('.atlas-footnote').innerText(),/930 调查点 · 510 特征/);
    assert.equal(await atlas().locator('.legend button').count(),6);
    assert.match(await atlas().locator('.category-coverage').innerText(),/已对应 6 个大类/);
    assert.match(await atlas().locator('.legend').innerText(),/无鼻尾/);
    await screenshot('f120-broad-light.png');
    await recordFileReads();
    assert(report.fileReads.some(x=>x.name==='data-manifest.json'));
    assert.equal(new Set(report.fileReads.filter(x=>x.method==='arrayBuffer').map(x=>x.path)).size,16);
    assert(!report.fileReads.some(x=>x.name==='unlisted-private-book.pdf'));
    check('Actual directory picker imports 16 manifest-listed files; 930 sites/510 features; default broad F120 has six verified groups; unlisted synthetic PDF is not read');

    stage='china-new-project-and-regions';
    assert.equal(await atlas().getByRole('button',{name:'世界',exact:true}).count(),0);
    const projectSnapshot=async name=>JSON.parse((await download(atlas().getByRole('button',{name:'导出工程',exact:true}),name+'.ptb-map.json')).toString('utf8'));
    const assertChinaView=project=>{assert(Math.abs(project.view.center[0]-104)<1);assert(project.view.center[1]>20&&project.view.center[1]<40);assert(project.view.zoom>1&&project.view.zoom<6);};
    await page.waitForTimeout(450);const newProject=await projectSnapshot('new-china');assertChinaView(newProject);assert.equal(newProject.settings.region,'');
    const admin=JSON.parse(await fs.readFile(path.join(scratch,'geolinguistics-data/map-index.json'),'utf8'));
    const datav=admin.some(a=>a.source?.startsWith('datav-'));
    const province=admin.find(a=>a.level==='province'&&a.name==='广东省');assert(province);
    const city=admin.find(a=>a.level==='city'&&a.parent===province.id&&admin.some(c=>c.level==='district'&&c.parent===a.id));assert(city);
    const county=admin.find(a=>a.level==='district'&&a.parent===city.id);assert(county);
    report.chinaRegions=[];
    for(const [label,region] of [['省级地区',province],['下一级地区',city],['县级地区',county]]){
      await atlas().getByLabel(label,{exact:true}).selectOption(region.id);await page.waitForTimeout(450);
      const p=await projectSnapshot('region-'+region.level);assert.equal(p.settings.region,region.id);
      const [w,s,e,n]=region.extent;assert(Math.abs(p.view.center[0]-(w+e)/2)<.05);assert(p.view.center[1]>=s&&p.view.center[1]<=n);assert(p.view.zoom>newProject.view.zoom);
      report.chinaRegions.push({id:region.id,name:region.name,level:region.level,view:p.view});
    }
    await screenshot('china-county.png');
    await atlas().getByLabel('省级地区',{exact:true}).selectOption('');await page.waitForTimeout(450);
    const nationwide=await projectSnapshot('nationwide-selector');assertChinaView(nationwide);assert.equal(nationwide.settings.region,'');
    assert.equal(await atlas().getByLabel('下一级地区',{exact:true}).count(),0);assert.equal(await atlas().getByLabel('县级地区',{exact:true}).count(),0);
    await waitCount(930);check('Fresh project opens China without a world entry; province/city/district select and fit their scopes; nationwide selector resets view and clears only the geographic filter');

    if(datav&&process.env.PTB_GEO_GEOGRAPHY_AUDIT==='1'){
      stage='datav-real-regions-and-south-fit';report.datavRegions=[];
      const frame=async(p,extent)=>{
        const size=await atlas().locator('.map-target').first().evaluate(el=>[el.clientWidth,el.clientHeight]);
        const radius=6378137,resolution=156543.03392804097/2**p.view.zoom,merc=([lon,lat])=>[radius*lon*Math.PI/180,radius*Math.log(Math.tan(Math.PI/4+lat*Math.PI/360))],center=merc(p.view.center);
        const corners=[[extent[0],extent[1]],[extent[2],extent[3]]].map(merc).map(v=>[(v[0]-center[0])/resolution+size[0]/2,(center[1]-v[1])/resolution+size[1]/2]);
        for(const [x,y]of corners)assert(x>=0&&x<=size[0]&&y>=0&&y<=size[1],JSON.stringify({view:p.view,size,corners,extent}));return {view:p.view,size,corners,extent};
      };
      const hainan=admin.find(a=>a.id==='province:460000'),taiwan=admin.find(a=>a.id==='province:710000');assert(hainan.extent[1]<4);
      await atlas().getByRole('button',{name:'中国',exact:true}).click();await page.waitForTimeout(500);report.datavChinaFrame=await frame(await projectSnapshot('datav-china-fit'),[hainan.extent[0],hainan.extent[1],hainan.extent[2],hainan.extent[3]]);
      await atlas().getByLabel('省级地区',{exact:true}).selectOption(hainan.id);await page.waitForTimeout(500);report.datavHainanFrame=await frame(await projectSnapshot('datav-hainan-fit'),hainan.extent);await screenshot('datav-hainan-zengmu-fit.png');
      for(const [p,id]of [['province:110000','district:110101'],['province:810000','district:810001'],['province:410000','city:419001'],['province:420000','city:429004'],['province:460000','city:469024'],['province:650000','city:659010'],['province:440000','city:441900']]){
        const region=admin.find(a=>a.id===id);assert(region);await atlas().getByLabel('省级地区',{exact:true}).selectOption(p);await atlas().getByLabel('下一级地区',{exact:true}).selectOption(id);await page.waitForTimeout(300);assert.equal(await atlas().getByLabel('县级地区',{exact:true}).count(),0);const saved=await projectSnapshot('datav-direct-'+region.code);assert.equal(saved.settings.region,id);report.datavRegions.push({id,level:region.level,parent:region.parent,view:saved.view});
      }
      await atlas().getByLabel('省级地区',{exact:true}).selectOption(taiwan.id);await page.waitForTimeout(500);assert.equal(await atlas().getByLabel('下一级地区',{exact:true}).count(),0);assert.equal(await atlas().getByLabel('县级地区',{exact:true}).count(),0);await atlas().getByRole('status').filter({hasText:'台湾省整体查看'}).waitFor();report.datavTaiwanFrame=await frame(await projectSnapshot('datav-taiwan-fit'),taiwan.extent);
      await atlas().locator('.geography-toolbar label').filter({hasText:'边界'}).locator('select').selectOption('province');await page.waitForFunction(()=>!document.querySelector('.map-loading'));await page.waitForTimeout(350);
      const xml=(await download(atlas().getByRole('button',{name:'SVG',exact:true}),'datav-taiwan-province.svg')).toString('utf8');
      const geojson=await Promise.all(['world','territory','province'].map(async name=>JSON.parse(await fs.readFile(path.join(scratch,'geolinguistics-data/maps/'+name+'.json'),'utf8'))));
      const flatten=g=>g.type==='GeometryCollection'?g.geometries.flatMap(flatten):[g];const types=geojson.flatMap(layer=>layer.features.filter(f=>f.properties?.boundaryType!=='省界').flatMap(f=>flatten(f.geometry).map(g=>g.type)));
      const inspected=await page.evaluate(({xml,types})=>{const doc=new DOMParser().parseFromString(xml,'image/svg+xml'),paths=[...doc.querySelector('g[clip-path]').querySelectorAll('path')];return {paths:paths.length,fills:paths.map(p=>p.getAttribute('fill')),subpaths:paths.map(p=>(p.getAttribute('d').match(/M/g)||[]).length),text:[...doc.querySelectorAll('text')].map(p=>p.textContent).join('')};},{xml,types});
      assert.equal(inspected.paths,types.length);
      // VectorSource.getFeatures() returns R-tree/random order, not GeoJSON order.
      const taiwanFeature=geojson[2].features.find(f=>f.id===taiwan.id);assert(taiwanFeature);const taiwanParts=taiwanFeature.geometry.coordinates;assert.equal(taiwanParts.length,14);
      const rings=taiwanParts.reduce((n,part)=>n+part.length,0),vertices=taiwanParts.reduce((n,part)=>n+part.reduce((a,ring)=>a+ring.length,0),0);
      const geometryCheck=await page.evaluate(({xml,rings,vertices})=>{const doc=new DOMParser().parseFromString(xml,'image/svg+xml'),paths=[...doc.querySelector('g[clip-path]').querySelectorAll('path')],frame=doc.querySelector('clipPath rect');const lines=paths.filter(p=>p.getAttribute('d')&&!p.getAttribute('d').includes('Z')),areas=paths.filter(p=>p.getAttribute('d').includes('Z')),matches=paths.filter(p=>(p.getAttribute('d').match(/M/g)||[]).length===rings&&(p.getAttribute('d').match(/[ML]/g)||[]).length===vertices);return {lines:lines.length,areaCount:areas.length,lineFills:lines.map(p=>p.getAttribute('fill')),taiwanMatches:matches.length,taiwanWithin:matches.every(p=>{const ns=p.getAttribute('d').match(/-?[\d.]+/g).map(Number);return ns.every((v,i)=>v>=0&&v<=+frame.getAttribute(i%2?'height':'width'));})};},{xml,rings,vertices});
      const priority=await page.evaluate(xml=>{const paths=[...new DOMParser().parseFromString(xml,'image/svg+xml').querySelector('g[clip-path]').querySelectorAll('path')],select=predicate=>paths.flatMap((p,i)=>predicate(p)?[i]:[]);return {faces:select(p=>p.getAttribute('fill')==='#f8f8f1'),lines:select(p=>p.getAttribute('d')&&!p.getAttribute('d').includes('Z')),symbols:select(p=>p.getAttribute('fill')==='#526c78'),disputed:select(p=>p.getAttribute('stroke-dasharray')==='4 3')};},xml);
      assert.equal(priority.faces.length,1);assert.equal(priority.lines.length,5);assert.equal(priority.symbols.length,10);assert.equal(priority.disputed.length,1);assert(Math.max(...priority.faces)<Math.min(...priority.lines)&&Math.max(...priority.lines)<Math.min(...priority.symbols));report.datavPainterPriority=priority;
      assert.equal(geometryCheck.lines,types.filter(t=>['LineString','MultiLineString'].includes(t)).length);assert.equal(geometryCheck.areaCount,types.filter(t=>['Polygon','MultiPolygon'].includes(t)).length);for(const fill of geometryCheck.lineFills)assert.equal(fill,'none');assert.equal(geometryCheck.taiwanMatches,1);assert(geometryCheck.taiwanWithin);assert(inspected.text.includes('DataV GeoAtlas / 高德开放平台'));report.datavTaiwanParts=14;report.datavOpenLinePaths=geometryCheck.lines;report.datavGeometryCheck=geometryCheck;await screenshot('datav-taiwan-province.png');
      await atlas().locator('.geography-toolbar label').filter({hasText:'边界'}).locator('select').selectOption('auto');await atlas().getByLabel('省级地区',{exact:true}).selectOption('');await page.waitForTimeout(500);await waitCount(930);check('Reviewed real GeoAtlas data fits China/Hainan south to Zengmu, preserves direct city/district leaves, exposes Taiwan at province scope with exactly 14 SVG components, and actual open boundary paths never receive area fill');
    }

    if(process.env.PTB_GEO_LAYOUT_ONLY==='1'){
      stage='equal-map-panels';
      await atlas().locator('.volume-tabs').getByRole('button',{name:'语法',exact:true}).click();
      await atlas().getByLabel('检索语言特征',{exact:true}).fill('F86_S');await atlas().locator('.feature-list button').click();
      const longTitle=await atlas().locator('.map-heading strong').innerText();assert(longTitle.length>30);
      assert.equal(await atlas().locator('.map-panel-heading').count(),0,'Single map keeps only its existing heading');
      report.layouts=[];
      for(const theme of ['light','dark'])for(const count of [2,4]){
        await page.evaluate(theme=>document.documentElement.dataset.theme=theme,theme);
        await atlas().locator('.map-heading label select').selectOption(String(count));
        await atlas().getByLabel('对照图 2',{exact:true}).selectOption('F89_S');
        await page.waitForFunction(count=>document.querySelectorAll('.map-cell canvas').length>=count,count);await page.waitForTimeout(450);
        const sizes=await atlas().locator('.map-cell').evaluateAll(cells=>cells.map(cell=>{
          const rect=selector=>{const r=cell.querySelector(selector).getBoundingClientRect();return {width:r.width,height:r.height,bottom:r.bottom};};
          return {gridBottom:cell.parentElement.getBoundingClientRect().bottom,heading:rect('.map-panel-heading'),map:rect('.map-target'),title:cell.querySelector('.map-panel-title,select').getAttribute('title')};
        }));
        assert.equal(sizes.length,count);for(const row of sizes){assert(row.heading.height>=30);assert(Math.abs(row.heading.height-sizes[0].heading.height)<1);assert(Math.abs(row.map.width-sizes[0].map.width)<=1);assert(Math.abs(row.map.height-sizes[0].map.height)<=1);assert(row.map.height>=100);assert(row.map.bottom<=row.gridBottom+1);}
        assert(sizes[0].title.includes(longTitle));assert.equal(await atlas().locator('.map-heading strong').innerText(),'语言现象对照');
        const svg=(await download(atlas().getByRole('button',{name:'SVG',exact:true}),`layout-${theme}-${count}.svg`)).toString('utf8');
        const geometry=await page.evaluate(svg=>{const doc=new DOMParser().parseFromString(svg,'image/svg+xml'),root=doc.documentElement;return {width:+root.getAttribute('width'),height:+root.getAttribute('height'),panels:[...root.children].filter(e=>e.tagName==='svg').map(e=>({width:+e.getAttribute('width'),height:+e.getAttribute('height'),viewBox:e.getAttribute('viewBox')})),frames:[...doc.querySelectorAll('clipPath rect')].map(e=>({width:+e.getAttribute('width'),height:+e.getAttribute('height')})),text:[...doc.querySelectorAll('text')].map(e=>e.textContent).join('')};},svg);
        assert.equal(geometry.panels.length,count);assert.equal(geometry.frames.length,count);
        for(const panel of geometry.panels){assert.deepEqual(panel,geometry.panels[0]);assert.equal(panel.viewBox,`0 0 ${panel.width} ${panel.height}`);}
        for(const frame of geometry.frames)assert.deepEqual(frame,geometry.frames[0]);assert(geometry.text.includes(longTitle));
        const png=await download(atlas().getByRole('button',{name:'PNG',exact:true}),`layout-${theme}-${count}.png`);
        assert.equal(png.readUInt32BE(16),geometry.width*2);assert.equal(png.readUInt32BE(20),geometry.height*2);
        report.layouts.push({theme,count,sizes,svg:geometry,png:{width:png.readUInt32BE(16),height:png.readUInt32BE(20)}});
        await screenshot(`equal-panels-${theme}-${count}.png`);
      }
      stage='compact-map-sources';
      await page.setViewportSize({width:1100,height:800});
      const sourceButtons=atlas().getByRole('button',{name:'地图来源',exact:true});
      assert.equal(await sourceButtons.count(),4);
      for(const button of await sourceButtons.all())assert.equal(await button.getAttribute('aria-expanded'),'false');
      const credits=atlas().locator('.map-osm-credit');assert.equal(await credits.count(),datav?0:4);
      for(const credit of await credits.all()){assert(await credit.isVisible());assert.equal(await credit.getAttribute('href'),'https://www.openstreetmap.org/copyright');}
      report.sourceControls=await atlas().locator('.geo-map').evaluateAll(maps=>maps.map(map=>{
        const panel=map.getBoundingClientRect(),bar=map.querySelector('.map-attributions').getBoundingClientRect();
        return {map:{width:panel.width,height:panel.height},bar:{width:bar.width,height:bar.height},within:bar.left>=panel.left&&bar.right<=panel.right+1&&bar.top>=panel.top&&bar.bottom<=panel.bottom};
      }));
      for(const row of report.sourceControls){assert(row.within);assert(row.bar.height<=62,'Collapsed sources must not become the former four-line block');}
      await sourceButtons.nth(0).focus();await page.keyboard.press('Enter');assert.equal(await sourceButtons.nth(0).getAttribute('aria-expanded'),'true');
      for(let i=1;i<4;i++)assert.equal(await sourceButtons.nth(i).getAttribute('aria-expanded'),'false');
      const details=page.locator(`[id="${await sourceButtons.nth(0).getAttribute('aria-controls')}"]`);assert(await details.isVisible());
      const expectedSources=datav?['DataV GeoAtlas / 高德开放平台','内政部国土测绘中心','Natural Earth']:['© OpenStreetMap contributors','内政部国土测绘中心','© 香港特别行政区政府','Huang 等','Natural Earth'];
      const primaryUrl=datav?'https://help.aliyun.com/zh/datav/datav-7-0/user-guide/datav-geoatlas-widgets/':'https://www.openstreetmap.org/copyright';
      assert.equal(await details.locator('a').count(),expectedSources.length);for(const source of expectedSources)assert((await details.innerText()).includes(source),source);
      if(datav){assert.match(await details.innerText(),/仅供学习和交流/);assert(!/ODbL|DATA.GOV.HK/.test(await details.innerText()));}else{assert.match(await details.innerText(),/ODbL 1.0/);assert.match(await details.innerText(),/DATA.GOV.HK/);}
      const expandedBounds=await details.evaluate(list=>{const a=list.getBoundingClientRect();return a.left>=0&&a.right<=innerWidth&&a.top>=0&&a.bottom<=innerHeight;});assert(expandedBounds);
      await screenshot('map-sources-small-expanded-dark.png');
      await details.locator('a').first().focus();await page.keyboard.press('Escape');assert.equal(await sourceButtons.nth(0).getAttribute('aria-expanded'),'false');assert(await sourceButtons.nth(0).evaluate(button=>button===document.activeElement));
      await context.route(primaryUrl,route=>route.fulfill({contentType:'text/html',body:'<title>Controlled source link check</title>'}));
      await sourceButtons.nth(1).focus();await page.keyboard.press('Space');assert.equal(await sourceButtons.nth(1).getAttribute('aria-expanded'),'true');
      const secondDetails=page.locator(`[id="${await sourceButtons.nth(1).getAttribute('aria-controls')}"]`);
      const detailsOpened=context.waitForEvent('page');await secondDetails.locator('a').first().click();const detailsSourcePage=await detailsOpened;await detailsSourcePage.waitForURL(primaryUrl);await detailsSourcePage.close();
      await sourceButtons.nth(0).click();assert.equal(await sourceButtons.nth(0).getAttribute('aria-expanded'),'true');assert.equal(await sourceButtons.nth(1).getAttribute('aria-expanded'),'true');
      await secondDetails.locator('a').first().focus();await page.keyboard.press('Escape');assert.equal(await sourceButtons.nth(1).getAttribute('aria-expanded'),'false');assert.equal(await sourceButtons.nth(0).getAttribute('aria-expanded'),'true');
      await details.locator('a').first().focus();await page.keyboard.press('Escape');assert.equal(await sourceButtons.nth(0).getAttribute('aria-expanded'),'false');
      if(!datav){const opened=context.waitForEvent('page');await credits.first().click();const sourcePage=await opened;await sourcePage.waitForURL(primaryUrl);await sourcePage.close();}
      for(const theme of ['light','dark']){await page.evaluate(value=>document.documentElement.dataset.theme=value,theme);await screenshot(`map-sources-small-collapsed-${theme}.png`);}
      const creditedSvg=(await download(atlas().getByRole('button',{name:'SVG',exact:true}),'collapsed-sources-four.svg')).toString('utf8');
      for(const text of datav?['DataV GeoAtlas / 高德开放平台','仅供学习和交流',primaryUrl,'内政部国土测绘中心','Natural Earth']:['© OpenStreetMap contributors','ODbL 1.0',primaryUrl,'内政部国土测绘中心','© 香港特别行政区政府','Natural Earth'])assert(creditedSvg.includes(text),text);
      check('Small four-map sources start collapsed and accurately credit the loaded data; controls are independent, Enter/Space/Escape and focus work, full source/licence links remain accessible, collapsed SVG keeps complete attribution');
      await page.setViewportSize({width:1440,height:900});
      await atlas().locator('.map-heading label select').selectOption('1');assert.equal(await atlas().locator('.map-panel-heading').count(),0);assert.equal(await atlas().locator('.map-heading strong').innerText(),longTitle);
      assert.deepEqual(report.errors,[]);check('Light/dark double and four maps have equal title bars and drawing rectangles, long titles remain available, SVG panels are padded without scaling and PNG has matching dimensions; single-map heading unchanged');
      report.success=true;return;
    }

    stage='detail-filter-edit';
    await atlas().getByLabel('语言现象范围',{exact:true}).selectOption('all');
    await atlas().getByLabel('检索语言特征',{exact:true}).fill('');
    for(const [volume,total] of [['语音',205],['词汇',203],['语法',102]]){
      await atlas().locator('.volume-tabs').getByRole('button',{name:volume,exact:true}).click();
      assert.equal(await atlas().locator('.feature-list button').count(),total);
    }
    await atlas().locator('.volume-tabs').getByRole('button',{name:'语音',exact:true}).click();await choose120();
    check('All-features browsing exposes 205 phonology, 203 lexicon and 102 grammar features, totaling 510');
    await atlas().getByLabel('分类层级',{exact:true}).selectOption('fine');
    assert((await atlas().locator('.legend button').count())>6);
    assert.match(await atlas().locator('.category-coverage').innerText(),/细类释义/);
    assert((await atlas().locator('.table-scroll .category-code').count())>0);
    await screenshot('f120-fine.png');
    await atlas().getByLabel('分类层级',{exact:true}).selectOption('broad');
    assert.equal(await atlas().locator('.table-scroll .category-code').count(),0);

    // Personal records not in the source dictionary remain visible as locations,
    // never as a synthetic grey linguistic class. Source cells remain intact.
    await atlas().getByLabel('检索调查点',{exact:true}).fill('S49');await waitCount(1);
    await atlas().locator('.table-scroll').getByRole('button',{name:'北京',exact:true}).click();
    const originalBeijing=(await atlas().locator('.raw-value').textContent()).trim();
    await atlas().getByRole('button',{name:'编辑当前特征',exact:true}).click();
    let unknownDialog=page.getByRole('dialog').filter({hasText:'类别／有序变体'});
    await unknownDialog.getByLabel('类别／有序变体',{exact:true}).fill('QA1#QA2');
    await unknownDialog.getByRole('button',{name:'保存校订',exact:true}).click();await unknownDialog.waitFor({state:'hidden'});
    assert.equal(await atlas().locator('.legend button').count(),0);
    assert.match(await atlas().locator('.pending-records').innerText(),/1 个地点/);
    await atlas().getByRole('button',{name:'查看待核地点',exact:true}).click();await waitCount(1);
    const unknownCsv=csvRows((await download(atlas().getByRole('button',{name:'导出表格',exact:true}),'f120-unmapped-record.csv')).toString('utf8'));
    assert.equal(unknownCsv.length,2);assert.equal(unknownCsv[1][5],'QA1#QA2');
    assert.equal(unknownCsv[1][6],'此记录的大类尚待对应');assert.equal(unknownCsv[1][7],originalBeijing);
    const unknownSvg=(await download(atlas().getByRole('button',{name:'SVG',exact:true}),'f120-unmapped-record.svg')).toString('utf8');
    assert.match(unknownSvg,/1 个地点含待核记录/);assert.doesNotMatch([...unknownSvg.matchAll(/<text\b[^>]*>(.*?)<\/text>/gs)].map(m=>m[1]).join(''),/尚未对应的大类|QA1|QA2/);
    assert.match(unknownSvg,/<circle[^>]*fill="#fff"[^>]*stroke="#172c40"/);
    await atlas().getByRole('button',{name:'编辑当前特征',exact:true}).click();
    unknownDialog=page.getByRole('dialog').filter({hasText:'类别／有序变体'});
    await unknownDialog.getByLabel('类别／有序变体',{exact:true}).fill('A1#QA1#QA2');
    await unknownDialog.getByRole('button',{name:'保存校订',exact:true}).click();await unknownDialog.waitFor({state:'hidden'});
    assert.equal(await atlas().locator('.legend button').count(),1);
    assert.match(await atlas().locator('.legend').innerText(),/有\[-m\].*1 点/s);
    assert.match(await atlas().locator('.pending-records').innerText(),/1 个地点/);
    const mixedSvg=(await download(atlas().getByRole('button',{name:'SVG',exact:true}),'f120-mixed-record.svg')).toString('utf8');
    assert.match(mixedSvg,/stroke-dasharray="2 2"/);assert.match(mixedSvg,/有\[-m\] · 1 个地点/);
    assert.doesNotMatch(await atlas().locator('.table-scroll tbody').innerText(),/QA1|QA2/);
    await screenshot('unmapped-locations-without-grey-category.png');
    await atlas().getByRole('button',{name:'恢复原值',exact:true}).click();await waitCount(0);
    await atlas().getByRole('button',{name:'全部类别',exact:true}).click();
    await atlas().getByLabel('检索调查点',{exact:true}).fill('');await waitCount(930);
    check('Unmapped and mixed records retain raw values, locations and pending-site counts; no synthetic grey legend category; hollow/dashed markers and SVG/CSV remain consistent');

    await atlas().locator('.legend button').filter({hasText:'无鼻尾'}).click();await waitCount(4);
    const names=await atlas().locator('.table-scroll tbody tr td:nth-child(2)').allTextContents();
    assert.deepEqual(names.map(s=>s.trim()).sort(),['歙县','绩溪','辰溪湘_熊首山','汤溪旧_岩下'].sort());
    const filtered=csvRows((await download(atlas().getByRole('button',{name:'导出表格',exact:true}),'f120-no-nasal.csv')).toString('utf8'));
    assert.equal(filtered.length,5);assert.deepEqual(filtered.slice(1).map(r=>r[5]).sort(),['F1','F2','F3','F3']);
    assert(filtered.slice(1).every(r=>r[6]==='无鼻尾'&&r.at(-1)==='大类'));
    await atlas().getByRole('button',{name:'清空',exact:true}).click();
    await atlas().getByRole('checkbox',{name:'选择 绩溪',exact:true}).check();
    await atlas().getByRole('checkbox',{name:'选择 歙县',exact:true}).check();
    await atlas().getByRole('button',{name:'比较 2–8 个地点',exact:true}).click();
    const comparison=page.getByRole('dialog',{name:'调查点对照',exact:true});
    await comparison.waitFor();assert.equal(await comparison.locator('.category-code').count(),0);
    assert.match(await comparison.locator('tr').filter({hasText:'阳声韵尾的演变'}).innerText(),/无鼻尾.*无鼻尾/s);
    await comparison.getByRole('button',{name:'关闭对话框',exact:true}).click();
    check('Broad point-comparison cells show meanings without raw atomic-code subtitles');
    await atlas().locator('.table-scroll').getByRole('button',{name:'绩溪',exact:true}).click();
    assert((await atlas().locator('.legend').boundingBox()).height>=160,'Selecting a site must not collapse the six-group legend');
    await atlas().getByRole('button',{name:'编辑当前特征',exact:true}).click();
    const dialog=page.getByRole('dialog').filter({hasText:'类别／有序变体'});
    await dialog.getByLabel('类别／有序变体',{exact:true}).fill('A3#B1');
    await dialog.getByRole('button',{name:'保存校订',exact:true}).click();await dialog.waitFor({state:'hidden'});
    await waitCount(3);await atlas().getByRole('button',{name:'全部类别',exact:true}).click();await waitCount(930);
    await atlas().getByLabel('检索调查点',{exact:true}).fill('绩溪');await waitCount(1);
    assert.match(await atlas().locator('.table-scroll tbody').innerText(),/有\[-m\].*只有\[-n -ŋ\]/s);
    assert.doesNotMatch(await atlas().locator('.table-scroll tbody').innerText(),/A3#B1/);
    await atlas().getByLabel('检索调查点',{exact:true}).fill('');await waitCount(930);
    check('Broad/fine toggle, F120 no-nasal group filter and exact four sites, filtered CSV, point search and real observation edit');

    stage='exports';
    const csv=csvRows((await download(atlas().getByRole('button',{name:'导出表格',exact:true}),'f120-edited.csv')).toString('utf8'));
    assert.equal(csv.length,931);const edited=csv.find(r=>r[0]==='S42');assert(edited);
    assert.equal(edited[5],'A3#B1');assert.equal(edited[7],'F1');assert.equal(edited.at(-1),'大类');
    const svg=(await download(atlas().getByRole('button',{name:'SVG',exact:true}),'f120-edited.svg')).toString('utf8');
    assert.match(svg,/<svg/);assert.match(svg,/阳声韵尾的演变/);assert.match(svg,/无鼻尾/);assert.match(svg,/大类分布/);assert.match(svg,/Zenodo 10697975/);
    assert.doesNotMatch(svg,/[（(]大类\s+[A-Z][）)]/);
    const png=await download(atlas().getByRole('button',{name:'PNG',exact:true}),'f120-edited.png');
    assert.equal(png.subarray(0,8).toString('hex'),'89504e470d0a1a0a');
    const dimensions={width:png.readUInt32BE(16),height:png.readUInt32BE(20)};assert(dimensions.width>300&&dimensions.height>200);report.png=dimensions;
    const imagePage=await context.newPage();
    await imagePage.setContent('<img style="max-width:100%;height:auto" src="data:image/png;base64,'+png.toString('base64')+'">');
    await imagePage.locator('img').evaluate(img=>img.decode());
    report.png.distinctSampleColors=await imagePage.locator('img').evaluate(img=>{const c=document.createElement('canvas');c.width=80;c.height=80;const ctx=c.getContext('2d');ctx.drawImage(img,0,0,80,80);const d=ctx.getImageData(0,0,80,80).data,s=new Set();for(let i=0;i<d.length;i+=4)s.add(`${d[i]},${d[i+1]},${d[i+2]},${d[i+3]}`);return s.size;});
    assert(report.png.distinctSampleColors>20,'PNG must contain rendered map content');
    await imagePage.screenshot({path:path.join(out,'png-export-preview.png'),fullPage:true});report.screenshots.push('png-export-preview.png');await imagePage.close();
    const project=JSON.parse((await download(atlas().getByRole('button',{name:'导出工程',exact:true}),'edited.ptb-map.json')).toString('utf8'));
    assert.equal(project.edits['S42:F120_P'],'A3#B1');
    check('Real CSV/SVG/PNG/project downloads: 930 records, edit and original value retained, source and broad legend in SVG, decoded nonblank PNG');

    stage='persistence';
    await page.locator('#main-content').focus();await page.keyboard.press('Control+s');
    await page.waitForFunction(()=>new Promise((resolve,reject)=>{
      const request=indexedDB.open('ptb-research-files-v1',1);request.onerror=()=>reject(request.error);
      request.onsuccess=()=>{const db=request.result,read=db.transaction('values').objectStore('values').get('local\u0000geolinguistics\u0000M19');
        read.onsuccess=()=>{const p=read.result?.value?JSON.parse(read.result.value):null;db.close();resolve(p?.edits?.['S42:F120_P']==='A3#B1'&&p.featureDetail==='broad');};
        read.onerror=()=>{db.close();reject(read.error);};};
    }));
    await recordFileReads();await page.reload();await openModule();await atlas().locator('.atlas-workbench').waitFor();
    assert.equal(await atlas().getByLabel('分类层级',{exact:true}).inputValue(),'broad');
    assert.match(await atlas().locator('.map-heading strong').innerText(),/阳声韵尾/);await waitCount(930);
    await atlas().getByLabel('检索调查点',{exact:true}).fill('绩溪');await waitCount(1);
    assert.match(await atlas().locator('.table-scroll tbody').innerText(),/有\[-m\].*只有\[-n -ŋ\]/s);
    assert.doesNotMatch(await atlas().locator('.table-scroll tbody').innerText(),/A3#B1/);
    await atlas().getByLabel('检索调查点',{exact:true}).fill('');await waitCount(930);
    assert(!report.requests.some(r=>r.stage==='persistence'&&/\/geolinguistics\//.test(r.url)),'Saved data must restore locally without fetching the runtime data');
    check('Reload restores the selected local resource set and observation edit without reimport or runtime-data network requests');

    stage='resource-refresh-live-project';
    await atlas().getByLabel('选择语言地图资料夹',{exact:true}).setInputFiles(path.join(scratch,'geolinguistics-data'));
    await page.waitForFunction(()=>!document.querySelector('[aria-label="选择语言地图资料夹"]').disabled);
    await waitCount(930);
    const refreshed=JSON.parse((await download(atlas().getByRole('button',{name:'导出工程',exact:true}),'after-resource-refresh.ptb-map.json')).toString('utf8'));
    assert.equal(refreshed.edits['S42:F120_P'],'A3#B1');
    check('Replacing the resource folder keeps the live observation edit and current feature');

    stage='damaged-directory';
    await atlas().getByLabel('选择语言地图资料夹',{exact:true}).setInputFiles(path.join(scratch,'damaged-data'));
    await atlas().getByRole('alert').filter({hasText:'资料校验未通过：catalog.json'}).waitFor();await waitCount(930);
    assert.match(await atlas().locator('.map-heading strong').innerText(),/阳声韵尾/);
    await screenshot('damaged-directory-preserves-current.png');
    await atlas().getByRole('alert').getByRole('button',{name:'收起',exact:true}).click();
    const retained=JSON.parse((await download(atlas().getByRole('button',{name:'导出工程',exact:true}),'after-rejection.ptb-map.json')).toString('utf8'));
    assert.equal(retained.edits['S42:F120_P'],'A3#B1');
    await recordFileReads();await page.reload();await openModule();await atlas().locator('.atlas-workbench').waitFor();await waitCount(930);
    assert.equal(await atlas().locator('.legend button').count(),6);
    check('Same-size SHA-corrupted directory is rejected; current map, edits and persisted prior data survive rejection and reload');

    stage='comparison-pages';
    await atlas().getByRole('tab',{name:'方言相似性',exact:true}).click();
    await atlas().locator('.mds-plot circle').first().waitFor();
    await atlas().locator('.table-scroll tbody tr').first().click();
    await atlas().locator('.nearby button').last().click();
    await comparison.waitFor();
    await comparison.getByLabel('对照地点 1',{exact:true}).selectOption('S49');
    await comparison.getByLabel('对照地点 2',{exact:true}).selectOption('S151');
    await comparison.getByLabel('比较项目范围',{exact:true}).selectOption('P');
    await comparison.getByLabel('比较记录关系',{exact:true}).selectOption('');
    assert.match(await comparison.locator('.comparison-summary').innerText(),/^205 项/);
    assert.equal(await comparison.locator('.comparison-scroll tbody tr').count(),60);
    const firstRows=await comparison.locator('.comparison-scroll tbody').innerText();
    for(let i=0;i<3;i++)await comparison.getByRole('button',{name:'地点对照下一页',exact:true}).click();
    assert.equal(await comparison.locator('.comparison-scroll tbody tr').count(),25);
    assert(await comparison.getByRole('button',{name:'地点对照下一页',exact:true}).isDisabled());
    await comparison.getByLabel('比较记录关系',{exact:true}).selectOption('same');
    assert(await comparison.getByRole('button',{name:'地点对照上一页',exact:true}).isDisabled());
    await comparison.getByLabel('比较记录关系',{exact:true}).selectOption('');
    assert.equal(await comparison.locator('.comparison-scroll tbody').innerText(),firstRows);
    const allComparisons=csvRows((await download(comparison.getByRole('button',{name:'导出对照表',exact:true}),'comparison-all-filtered.csv')).toString('utf8'));
    assert.equal(allComparisons.length,206,'Export retains all 205 filtered rows despite 60-row visible page');
    await screenshot('comparison-pages.png');
    await comparison.getByRole('button',{name:'关闭对话框',exact:true}).click();
    await choose120();
    check('Comparison pagination, last page and filter reset; export retains every filtered record');

    stage='paper-main-restore-import';
    await atlas().getByRole('tab',{name:'方言相似性',exact:true}).click();
    const matrixSelect=atlas().locator('.atlas-catalog label').filter({hasText:/^距离方案/}).locator('select');assert.equal(await matrixSelect.inputValue(),'main');
    await atlas().locator('.table-scroll tbody tr').first().click();
    await page.waitForFunction(()=>document.querySelectorAll('.nearby').length===15);
    await atlas().locator('.map-target').first().focus();await page.keyboard.press('Control+s');
    await page.waitForFunction(()=>new Promise((resolve,reject)=>{
      const request=indexedDB.open('ptb-research-files-v1',1);request.onerror=()=>reject(request.error);
      request.onsuccess=()=>{const db=request.result,r=db.transaction('values').objectStore('values').get('local\u0000geolinguistics\u0000M19');r.onsuccess=()=>{const p=r.result?.value?JSON.parse(r.result.value):null;db.close();resolve(p?.mode==='paper'&&p.settings?.matrixName==='main'&&!!p.settings?.selectedSite);};r.onerror=()=>{db.close();reject(r.error);};};
    }));
    await page.reload();await openModule();await page.waitForFunction(()=>document.querySelectorAll('.nearby').length===15);
    assert.equal(await atlas().getByRole('tab',{name:'方言相似性',exact:true}).getAttribute('aria-selected'),'true');
    const paperProject=await download(atlas().getByRole('button',{name:'导出工程',exact:true}),'paper-main.ptb-map.json');
    const paperSaved=JSON.parse(paperProject.toString('utf8'));assert.equal(paperSaved.mode,'paper');assert.equal(paperSaved.settings.matrixName,'main');assert.equal(paperSaved.edits['S42:F120_P'],'A3#B1');
    await atlas().locator('input[type=file][accept=".json"]').setInputFiles({name:'paper-main.ptb-map.json',mimeType:'application/json',buffer:paperProject});
    await page.getByRole('dialog',{name:'打开地图工程',exact:true}).getByRole('button',{name:'打开工程',exact:true}).click();
    await page.waitForFunction(()=>document.querySelectorAll('.nearby').length===15);
    await screenshot('paper-main-restored.png');
    check('Saving, reloading and reimporting paper/main immediately restores 15 neighbours without changing the matrix selector; observation edit survives');

    stage='name-history-and-draw-exit';
    await atlas().getByRole('tab',{name:'我的研究地图',exact:true}).click();
    const nameInput=atlas().getByLabel('工程名称',{exact:true}),originalName=await nameInput.inputValue();
    await nameInput.fill('回归调查 tʂʰ ã n̩');await nameInput.blur();
    await atlas().getByRole('button',{name:'撤销',exact:true}).click();assert.equal(await nameInput.inputValue(),originalName);
    await atlas().locator('.map-target').first().focus();await page.keyboard.press('Control+Shift+Z');assert.equal(await nameInput.inputValue(),'回归调查 tʂʰ ã n̩');
    await atlas().getByRole('button',{name:'在地图上添加地点',exact:true}).click();
    await choose120();await atlas().locator('.table-scroll tbody tr').first().click();
    assert.equal(await page.getByRole('dialog',{name:'新增调查点',exact:true}).count(),0);
    check('Project rename checkpoints support undo and real Ctrl+Shift+Z; leaving custom drawing returns to feature selection');

    stage='legacy-world-restore-and-multiple-maps';
    await atlas().getByRole('button',{name:'中国',exact:true}).click();await page.waitForTimeout(400);
    const legacyWorld=await projectSnapshot('legacy-world-base');legacyWorld.view={center:[10,20],zoom:2};
    legacyWorld.customSites.push({id:'user:legacy-world',name:'旧工程保留点',lon:113,lat:23,admin:null,value:'原类',note:'恢复保留'});
    legacyWorld.classEdits['qa-legacy-row']=['','','','','','','保留大区','保留区','保留片',''];
    legacyWorld.edits['S1:F1_P']='QA保留';legacyWorld.drawings={type:'FeatureCollection',features:[{type:'Feature',geometry:{type:'LineString',coordinates:[[113,23],[114,24]]}}]};
    const legacyBytes=Buffer.from(JSON.stringify(legacyWorld));
    await atlas().locator('input[type=file][accept=".json"]').setInputFiles({name:'legacy-world.ptb-map.json',mimeType:'application/json',buffer:legacyBytes});
    await page.getByRole('dialog',{name:'打开地图工程',exact:true}).getByRole('button',{name:'打开工程',exact:true}).click();await page.waitForTimeout(450);
    const imported=await projectSnapshot('legacy-world-imported');assertChinaView(imported);
    for(const key of ['edits','classEdits','labels','customSites','drawings','selected','bookmarks'])assert.deepEqual(imported[key],legacyWorld[key],key+' retained on import');
    // Close the live page after its autosave, then seed only this owned profile's
    // saved project to exercise the initial restore path with the same old file.
    await page.waitForTimeout(500);const seedPage=await context.newPage();await seedPage.goto(url);await page.close();page=seedPage;
    await page.evaluate(raw=>new Promise((resolve,reject)=>{const request=indexedDB.open('ptb-research-files-v1',1);request.onerror=()=>reject(request.error);request.onsuccess=()=>{const db=request.result,tx=db.transaction('values','readwrite');tx.objectStore('values').put({key:'local\u0000geolinguistics\u0000M19',value:raw});tx.oncomplete=()=>{db.close();resolve();};tx.onerror=()=>{db.close();reject(tx.error);};};}),legacyBytes.toString('utf8'));
    await page.reload();await openModule();await atlas().locator('.atlas-workbench').waitFor();await page.waitForTimeout(450);
    const restored=await projectSnapshot('legacy-world-reloaded');assertChinaView(restored);
    for(const key of ['edits','classEdits','labels','customSites','drawings','selected','bookmarks'])assert.deepEqual(restored[key],legacyWorld[key],key+' retained on reload');
    await screenshot('legacy-world-restored-china.png');
    await atlas().locator('.map-heading label select').selectOption('2');
    await page.waitForFunction(()=>document.querySelectorAll('.map-cell canvas').length>=2);await page.waitForTimeout(700);
    const chinaProject=await projectSnapshot('china-two-panels');assertChinaView(chinaProject);
    assert(Math.abs(chinaProject.view.center[0]-restored.view.center[0])<1e-6);assert(Math.abs(chinaProject.view.center[1]-restored.view.center[1])<1e-6);assert(Math.abs(chinaProject.view.zoom-restored.view.zoom)<1e-6);
    await atlas().locator('.map-heading label select').selectOption('4');await page.waitForFunction(()=>document.querySelectorAll('.map-cell canvas').length>=4);
    const workspace=await atlas().locator('.atlas-workbench').evaluate(el=>['.atlas-catalog','.atlas-center','.atlas-details'].map(selector=>{const r=el.querySelector(selector).getBoundingClientRect();return {width:r.width,height:r.height};}));
    assert(workspace.every(r=>r.width>=200&&r.height>200));
    await screenshot('china-four-maps-light.png');
    const parallelSvg=(await download(atlas().getByRole('button',{name:'SVG',exact:true}),'china-four-maps.svg')).toString('utf8');
    assert.equal((parallelSvg.match(/<clipPath/g)||[]).length,4);assert.match(parallelSvg,/PTB-Map-IPA/);assert.match(parallelSvg,/base64,/);
    assert(![...parallelSvg.matchAll(/<text[^>]*>([^<]*)/g)].some(m=>/^[\u0303\u0329]/u.test(m[1])));
    check('Legacy world project import and initial restore return to China and preserve observations, classifications, custom points and drawings; China view survives second/fourth maps; four-panel SVG embeds IPA font');
    await atlas().locator('.map-heading label select').selectOption('1');await atlas().getByRole('button',{name:'中国',exact:true}).click();

    stage='themes-help';
    await page.waitForTimeout(500);
    await page.evaluate(()=>document.documentElement.dataset.theme='dark');await page.setViewportSize({width:1100,height:800});
    await screenshot('f120-broad-dark.png');
    await page.setViewportSize({width:1440,height:900});await page.evaluate(()=>document.documentElement.dataset.theme='light');
    await atlas().getByRole('button',{name:'使用方法',exact:true}).click();
    // Shared contextual help may route this button straight to the complete manual.
    const shortHelp=page.getByRole('dialog',{name:'语言地图使用方法',exact:true});
    if(await shortHelp.isVisible())await shortHelp.getByRole('button',{name:'打开完整使用说明',exact:true}).click();
    await page.locator('.manual-reader:visible').getByRole('heading',{name:/语言地图/}).first().waitFor();
    const manualReader=page.locator('.manual-reader:visible');
    assert.match(await manualReader.innerText(),/点击中国回到全国总览/);assert.doesNotMatch(await manualReader.innerText(),/点击世界或中国/);
    assert.equal(await manualReader.getByRole('link',{name:'观看本章视频',exact:true}).getAttribute('href'),'https://www.phonetictoolbox.com/#tutorial-geolinguistics');
    await screenshot('m19-manual.png');check('Light/dark actual map screenshots and M19 complete manual reachable from module help');
    await recordFileReads();
    assert(!report.fileReads.some(r=>r.name==='unlisted-private-book.pdf'));
    assert.deepEqual(report.requests.filter(r=>!['GET','HEAD'].includes(r.method)||r.bodyBytes),[],'No upload or other network write');
    assert.deepEqual(report.requests.filter(r=>/^https?:/.test(r.url)&&new URL(r.url).origin!==new URL(url).origin),[],'No external data requests');
    assert.deepEqual(report.errors,[]);check('All observed network requests are same-origin read-only; no local research bytes uploaded');
    report.success=true;
  }catch(error){report.failure=String(error.stack||error);report.failedStage=stage;if(page){await screenshot('failure.png').catch(()=>{});report.failureText=await page.locator('body').innerText().catch(()=>'');}throw error;}
  finally{
    try{if(context)await context.close();}finally{if(server)await server.close();await fs.writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2));}
    console.log(JSON.stringify({success:report.success,checks:report.checks,failedStage:report.failedStage},null,2));
  }
})().catch(error=>{console.error(error);process.exitCode=1;});
