import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import ts from 'typescript';
import {ref,computed} from 'vue';
import {emptyProject,validateProject,defaultSettings,isLegacyWorldView,classValues,classFilterMatches,unclassifiedClassFilter,inRegion,diversity,observation,isObservationEdited,volumeNames} from '../src/modules/geolinguistics/model.ts';
import {displayCounts,displayText,displayColor,broadPendingCount,isBroadCategory} from '../src/modules/geolinguistics/groups.ts';
import {wrapMapText,composeMapSvgs} from '../src/modules/geolinguistics/export.ts';
import {mapPaint,mapPaintPriority,contextBoundaryVisible} from '../src/modules/geolinguistics/map-paint.ts';

// Execute the functions and state declarations from the actual SFC source. Only
// browser / storage / map boundaries are faked; no workflow is reimplemented.
const source=(name:string)=>readFileSync(new URL(`../src/modules/geolinguistics/${name}.vue`,import.meta.url),'utf8');
function component(name:string,names:string[],env:Record<string,any>,state:string[]=[],expose:string[]=[]){
 env={mapPaint,mapPaintPriority,contextBoundaryVisible,projectedBaseFeatures:(value:any)=>Array.isArray(value)?value:[value],...env};
 const script=source(name).match(/<script setup lang="ts">([\s\S]*?)<\/script>/)![1];
 const file=ts.createSourceFile(name+'.ts',script,ts.ScriptTarget.Latest,true,ts.ScriptKind.TS);
 const functions=new Map<string,string>(),variables=new Map<string,string>();
 for(const node of file.statements){
  if(ts.isFunctionDeclaration(node)&&node.name)functions.set(node.name.text,node.getText(file));
  if(ts.isVariableStatement(node))for(const declaration of node.declarationList.declarations){
   if(ts.isIdentifier(declaration.name))variables.set(declaration.name.text,`let ${declaration.getText(file)};`);
  }
 }
 const required=(items:Map<string,string>,key:string)=>{assert(items.has(key),`Missing ${name}.${key}`);return items.get(key)!;};
 const body=[...state.map(key=>required(variables,key)),...names.map(key=>required(functions,key)),`return {${[...names,...expose].join(',')}};`].join('\n');
 const js=ts.transpileModule(body,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.None}}).outputText;
 return new Function(...Object.keys(env),js)(...Object.values(env));
}
function deferred<T>(){let resolve!:(value:T)=>void,reject!:(error:Error)=>void;const promise=new Promise<T>((yes,no)=>{resolve=yes;reject=no;});return {promise,resolve,reject};}
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const input=()=>({target:{files:[{name:'manifest.json'}],value:'selected'}});

function pageEnv(){
 const env={project:ref(emptyProject()),resources:ref<any>(),ready:ref(true),loading:ref(false),saving:ref(false),matrix:ref<any>(),matrixBusy:ref(false),matrixName:ref('main'),compareOpen:ref(false),controller:new AbortController(),error:ref(''),errorRetry:ref(''),notice:ref(''),fit:ref<any>(),props:{stateKey:'account:project:M19',active:true},restoreSettings:()=>{},validateProject,emit:()=>{},storage:{value:async()=>null,compare:async()=>{}},geoPort:{load:async()=>({version:'new'}),useDirectory:async()=>{},matrix:async()=>new Float64Array([0])}};
 return {...env,emptyProject,isLegacyWorldView,reportError:(message:string,retry='')=>{env.error.value=message;env.errorRetry.value=retry;}};
}

test('changing resources preserves edits arriving during save, history and the current scoped draft',async()=>{
 const env=pageEnv(),writing=deferred<void>(),writes:any[]=[],past=ref(['previous']);
 env.project.value.edits['S1:F1_P']='A1';
 env.storage.compare=async(...args:any[])=>{writes.push(args);await writing.promise;};
 env.storage.value=async()=>{throw Error('resource refresh must not reopen an old project snapshot');};
 const api=component('GeolinguisticsPage',['chooseData','replaceData','save','loadMatrix'],{...env,past,load:()=>{throw Error('live project replaced');}},['matrixRequest','saved','saveQueue','timer']);
 const project=env.project.value,action=api.chooseData(input());await tick();
 assert.equal(writes[0][0],env.props.stateKey);assert.equal(JSON.parse(writes[0][2]).edits['S1:F1_P'],'A1');
 env.project.value.edits['S1:F1_P']='B2';past.value.push('new edit');writing.resolve();await action;
 assert.equal(env.project.value,project);assert.equal(env.project.value.edits['S1:F1_P'],'B2');assert.deepEqual(past.value,['previous','new edit']);
 assert.equal(env.resources.value.version,'new');assert.equal(env.loading.value,false);
 await api.save();assert.equal(JSON.parse(writes.at(-1)[2]).edits['S1:F1_P'],'B2');
});

test('failed save prevents directory replacement and retains the draft',async()=>{
 const env=pageEnv();let imports=0;env.geoPort.useDirectory=async()=>{imports++;};env.storage.compare=async()=>{throw Error('conflict');};
 const api=component('GeolinguisticsPage',['chooseData','replaceData','save','loadMatrix'],{...env,load:()=>{}},['matrixRequest','saved','saveQueue','timer']);
 env.project.value.name='unsaved';await api.chooseData(input());assert.equal(imports,0);assert.equal(env.project.value.name,'unsaved');assert.match(env.error.value,/conflict/);assert.equal(env.loading.value,false);
});

test('restoring paper/main explicitly loads its matrix even when the setting did not change',async()=>{
 const env=pageEnv();env.ready.value=false;const stored=emptyProject();stored.mode='paper';let calls=0;
 env.storage.value=async()=>JSON.stringify(stored);env.geoPort.matrix=async(name:string)=>{calls++;assert.equal(name,'main');return new Float64Array([0,.4,.4,0]);};
 const api=component('GeolinguisticsPage',['load','loadMatrix','restoreView'],env,['saved','lastUndo','matrixRequest']);
 assert.equal(await api.load(),true);assert.equal(calls,1);assert.deepEqual([...env.matrix.value],[0,.4,.4,0]);assert.equal(env.ready.value,true);
});

test('importing paper/main reloads the matrix and clears a previous fit and draw interaction',async()=>{
 const env=pageEnv(),pendingProject=ref(emptyProject()),drawMode=ref('Polygon'),modal=ref('import-project');pendingProject.value.mode='paper';env.fit.value={extent:[1,2,3,4],nonce:1};let checkpoints=0,calls=0;
 env.geoPort.matrix=async()=>{calls++;return new Float64Array([0]);};
 const api=component('GeolinguisticsPage',['applyProject','loadMatrix','restoreView'],{...env,pendingProject,drawMode,modal,checkpoint:()=>{checkpoints++;}},['matrixRequest']);
 api.applyProject();await tick();assert.equal(calls,1);assert.equal(checkpoints,1);assert.equal(drawMode.value,'');assert.equal(env.fit.value,undefined);assert.equal(modal.value,'');
});

test('a stale matrix request cannot replace the newly selected matrix or its error status',async()=>{
 const env=pageEnv(),first=deferred<Float64Array>(),second=deferred<Float64Array>();env.geoPort.matrix=(name:string)=>name==='main'?first.promise:second.promise;
 const api=component('GeolinguisticsPage',['loadMatrix'],env,['matrixRequest']);const a=api.loadMatrix();env.matrixName.value='type-0-1';const b=api.loadMatrix();second.resolve(new Float64Array([2]));await b;first.reject(Error('obsolete'));await a;
 assert.deepEqual([...env.matrix.value],[2]);assert.equal(env.error.value,'');assert.equal(env.matrixBusy.value,false);
});

test('refreshing resources invalidates an in-flight matrix from the old directory',async()=>{
 const env=pageEnv(),old=deferred<Float64Array>();let replaced=false;env.project.value.mode='paper';env.geoPort.useDirectory=async()=>{replaced=true;};env.geoPort.matrix=()=>replaced?Promise.resolve(new Float64Array([9])):old.promise;
 const api=component('GeolinguisticsPage',['chooseData','replaceData','loadMatrix'],{...env,save:async()=>true,load:()=>{}},['matrixRequest']);
 const a=api.loadMatrix();await api.chooseData(input());old.resolve(new Float64Array([1]));await a;assert.deepEqual([...env.matrix.value],[9]);
});

test('switching out of a research map exits drawing and restores normal site selection',()=>{
 const env=pageEnv(),drawMode=ref('point'),category=ref('A'),tablePage=ref(5);let matrices=0;
 const api=component('GeolinguisticsPage',['changeMode'],{...env,drawMode,category,tablePage,loadMatrix:()=>{matrices++;}});
 for(const mode of ['features','classification','paper']){drawMode.value='Polygon';api.changeMode(mode);assert.equal(drawMode.value,'');assert.equal(env.project.value.mode,mode);assert.equal(category.value,'');assert.equal(tablePage.value,0);}assert.equal(matrices,1);
 assert.match(source('GeolinguisticsPage'),/:draw-mode="project.mode==='custom'&&index===0\?drawMode:''"/);
});

test('national reset and province/city/district choices fit their scopes; completed moves consume the fit',()=>{
 const env=pageEnv(),region=ref('province:1'),tablePage=ref(5),admins=new Map(['province','city','district'].map((level,i)=>[level+':1',{extent:[100+i,20+i,110-i,30-i]}]));
 const api=component('GeolinguisticsPage',['china','chooseRegion','setView'],{...env,region,tablePage,adminIndex:ref(admins)});
 for(const id of admins.keys()){api.chooseRegion(id);assert.equal(region.value,id);assert.deepEqual(env.fit.value.extent,admins.get(id)!.extent);assert.equal(tablePage.value,0);}
 api.chooseRegion('');assert.equal(region.value,'');assert.deepEqual(env.fit.value.extent,[73,3,135,54]);
 region.value='province:1';api.china();assert.equal(region.value,'');assert.deepEqual(env.fit.value.extent,[73,3,135,54]);
 env.fit.value={extent:[100,20,110,30],nonce:2};api.setView({center:[105,25],zoom:7});assert.equal(env.fit.value,undefined);assert.deepEqual(env.project.value.view,{center:[105,25],zoom:7});
 api.setView({center:[-30,40],zoom:2});assert.deepEqual(env.project.value.view,{center:[-30,40],zoom:2},'Manual pan/zoom remains unrestricted');
});

function legacyWorldProject(){
 const p=emptyProject();p.view={center:[10+1e-9,20-1e-9],zoom:2};p.edits={'S1:F1_P':'A1#B1'};p.classEdits={row:['','','','','','','甲大区','乙区','丙片','']};p.labels={F1_P:{A1:'个人释义'}};p.customSites=[{id:'user:kept',name:'自建点',lon:113,lat:23,admin:null,value:'甲类',note:'保留'}];p.drawings={type:'FeatureCollection',features:[{type:'Feature',geometry:{type:'LineString',coordinates:[[113,23],[114,24]]}}]};p.selected=['S1'];p.settings={...defaultSettings(),classFilters:['甲大区','','',''],mapCount:4};return p;
}

test('loading a legacy world project restores China while preserving all research fields and the original CAS value',async()=>{
 const env=pageEnv(),original=legacyWorldProject(),raw=JSON.stringify(original),writes:any[]=[];env.storage.value=async()=>raw;env.storage.compare=async(...args:any[])=>{writes.push(args);};
 const api=component('GeolinguisticsPage',['load','restoreView','save'],{...env,loadMatrix:async()=>{}},['saved','lastUndo','saveQueue','timer']);
 assert.equal(await api.load(),true);assert.deepEqual(env.fit.value.extent,[73,3,135,54]);assert.deepEqual(env.project.value.view,emptyProject().view);
 assert.deepEqual({...env.project.value,view:original.view},original);await api.save();assert.equal(writes[0][1],raw);assert.deepEqual(JSON.parse(writes[0][2]).drawings,original.drawings);
});

test('importing a legacy world project restores China without losing observations, classifications or drawings',()=>{
 const env=pageEnv(),original=legacyWorldProject(),pendingProject=ref(validateProject(original));
 const api=component('GeolinguisticsPage',['applyProject','restoreView'],{...env,pendingProject,drawMode:ref('Polygon'),modal:ref('import-project'),checkpoint:()=>{},loadMatrix:()=>{}});
 api.applyProject();assert.deepEqual(env.fit.value.extent,[73,3,135,54]);assert.deepEqual({...env.project.value,view:original.view},original);
 for(const view of [{center:[113,23],zoom:8},{center:[-30,40],zoom:2}] as const){env.project.value.view={center:[...view.center],zoom:view.zoom};api.restoreView();assert.equal(env.fit.value,undefined);assert.deepEqual(env.project.value.view,view);}
 assert.equal(isLegacyWorldView({center:[10,20],zoom:2}),true);assert.equal(isLegacyWorldView({center:[10,20],zoom:7}),false);
});

test('name and observation changes undo/redo through actual component history and uppercase Shift+Z',()=>{
 const env=pageEnv(),past=ref<string[]>([]),future=ref<string[]>([]);
 const api=component('GeolinguisticsPage',['checkpoint','restoreHistory','undo','redo','keydown'],{...env,past,future,save:()=>{}},['replaying','lastUndo']);
 const initial=env.project.value.name;env.project.value.name='海岛调查';api.checkpoint();env.project.value.edits['S1:F1_P']='B1';api.checkpoint();
 let prevented=0;const key=(key:string,shiftKey=false,editable=false)=>({key,shiftKey,ctrlKey:true,metaKey:false,target:{matches:()=>editable},preventDefault:()=>{prevented++;}});
 api.keydown(key('z'));assert.equal(env.project.value.edits['S1:F1_P'],undefined);api.keydown(key('Z',true));assert.equal(env.project.value.edits['S1:F1_P'],'B1');
 api.undo();api.undo();assert.equal(env.project.value.name,initial);api.redo();assert.equal(env.project.value.name,'海岛调查');
 api.keydown(key('z',false,true));assert.equal(env.project.value.name,'海岛调查');assert.equal(prevented,2);
 env.project.value.name='新分支';api.checkpoint();assert.deepEqual(future.value,[]);
 assert.match(source('GeolinguisticsPage'),/v-model="project.name" @change="checkpoint"/);
});

function mapEnv(){
 const requests:{name:string;job:ReturnType<typeof deferred<any>>}[]=[],displayed:any[]=[];
 const env={props:{level:'province',port:{map:(name:string)=>{const job=deferred<any>();requests.push({name,job});return job.promise;}}},map:{getView:()=>({getZoom:()=>4})},loading:ref(false),admin:{clear:()=>{displayed.length=0;},addFeatures:(value:any[])=>displayed.push(...value)},read:(value:any)=>value,emit:(...args:any[])=>events.push(args)};
 const events:any[]=[];const api=component('GeoMap',['currentLevel','syncLevel'],env,['disposed','levelEpoch','lastMapError','lastLevel','requestedLevel']);return {env,api,requests,displayed,events};
}

test('province → slow city → province never applies the stale city boundary',async()=>{
 const h=mapEnv(),first=h.api.syncLevel();h.requests[0].job.resolve('province');await first;
 h.env.props.level='city';const city=h.api.syncLevel();h.env.props.level='province';await h.api.syncLevel();assert.equal(h.env.loading.value,false);
 h.requests[1].job.resolve('city');await city;assert.deepEqual(h.displayed,['province']);assert.equal(h.requests.length,2);
});

test('boundary requests deduplicate the active target and ignore stale errors while allowing retries',async()=>{
 const h=mapEnv(),a=h.api.syncLevel();await h.api.syncLevel();assert.equal(h.requests.length,1);
 h.env.props.level='city';const b=h.api.syncLevel();h.requests[1].job.resolve('city');await b;h.requests[0].job.reject(Error('stale'));await a;assert.deepEqual(h.events,[]);assert.deepEqual(h.displayed,['city']);
 h.env.props.level='district';const c=h.api.syncLevel();h.requests[2].job.reject(Error('retry me'));await c;assert.deepEqual(h.events,[['error','retry me']]);
 const retry=h.api.syncLevel();assert.equal(h.requests.length,4);h.requests[3].job.resolve('district');await retry;assert.deepEqual(h.displayed,['district']);
});

test('actual SVG export keeps research sketch dashes, pending-site styling and escaped labels',()=>{
 const empty={getFeatures:()=>[]},geometry={type:'LineString',coordinates:[[1,2],[3,4]]},polygon={type:'Polygon',coordinates:[[[1,2],[3,2],[3,4],[1,2]]]};
 const env={map:{getView:()=>({getZoom:()=>4}),getSize:()=>[600,300],getPixelFromCoordinate:(p:number[])=>p},props:{level:'province',points:[{id:'1',name:'甲<&',lon:2,lat:3,color:'#123456',incomplete:true,selected:false}],labels:true},disposed:false,baseReady:true,loading:ref(false),lastLevel:'province',requestedLevel:'province',document:{createElement:()=>({getContext:()=>({font:'',measureText:(s:string)=>({width:s.length*10})})})},wrapMapText,background:empty,territory:empty,admin:empty,sketch:{getFeatures:()=>[geometry,polygon].map(g=>({getGeometry:()=>g,getId:()=>'',get:()=>undefined}))},format:{writeGeometryObject:(g:any)=>g},fromLonLat:(p:number[])=>p,target:ref(undefined)};
 const api=component('GeoMap',['currentLevel','exportSvg'],env,['escape']);const svg=api.exportSvg('调查图',[{label:'[ã] <甲>',color:'#123456'}],'来源 & 记录');
 assert.match(svg,/stroke-dasharray="6 4"/);assert.match(svg,/d="M2\.00,4\.00L6\.00,8\.00" fill="none"/);assert.match(svg,/fill="#397da525"/);assert.match(svg,/stroke-dasharray="2 2"/);assert.match(svg,/甲&lt;&amp;/);assert.match(svg,/来源 &amp; 记录/);assert.match(svg,/width="1200"/);
});


test('multi-map exports pad unequal title/legend footers without scaling equal map frames',()=>{
 const panel=(height:number,title:string)=>`<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="${height}" viewBox="0 0 1200 ${height}"><clipPath id="frame"><rect width="1200" height="600"/></clipPath><g clip-path="url(#frame)"><path d="M10,20L30,40"/></g><text>${title}</text></svg>`;
 const short=panel(700,'短标题'),long=panel(850,'长标题 [ã n̩]');
 for(const count of [2,4]){
  const svg=composeMapSvgs(Array.from({length:count},(_,i)=>i%2?long:short));
  const inner=[...svg.matchAll(/<svg xmlns[^>]*>/g)];assert.equal(inner.length,count+1);
  for(const match of inner.slice(1)){assert.match(match[0],/width="1200" height="850" viewBox="0 0 1200 850"/);}
  assert.equal((svg.match(/<rect width="1200" height="600"\/>/g)??[]).length,count);
  assert.equal((svg.match(/M10,20L30,40/g)??[]).length,count);
  assert.match(svg,new RegExp(`height="${count===2?850:1720}"`));
 }
 assert.equal(composeMapSvgs([short]),short);
});

function exportMapEnv(){
 const empty={getFeatures:()=>[],clear:()=>{},addFeatures:()=>{}};
 return {map:{getView:()=>({getZoom:()=>4}),getSize:()=>[600,300],getPixelFromCoordinate:(p:number[])=>p},props:{level:'province',points:[],port:{map:async(_name:string):Promise<any>=>[]}},baseReady:true,loading:ref(false),document:{createElement:()=>({getContext:()=>({measureText:(s:string)=>({width:s.length*8})})})},wrapMapText,background:empty,territory:empty,admin:empty,sketch:empty,format:{writeGeometryObject:(g:any)=>g},read:(value:any)=>value,emit:()=>{},target:ref(undefined)};
}

test('exports block unfinished or failed layers and a real reload restores the requested layer',async()=>{
 const env=exportMapEnv(),requests:{name:string;job:ReturnType<typeof deferred<any>>}[]=[];
 env.props.port.map=(name:string)=>{const job=deferred<any>();requests.push({name,job});return job.promise;};
 const api=component('GeoMap',['currentLevel','syncLevel','reload','exportSvg'],env,['disposed','levelEpoch','lastMapError','lastLevel','requestedLevel','escape']);
 const first=api.syncLevel();assert.throws(()=>api.exportSvg('图',[],'来源'),/尚未完整载入/);
 requests[0].job.resolve([]);await first;assert.match(api.exportSvg('图',[],'来源'),/<svg/);
 env.props.level='district';
 // A changed selector must block before its watcher even starts the next request.
 assert.throws(()=>api.exportSvg('图',[],'来源'),/尚未完整载入/);
 const slow=api.syncLevel();assert.throws(()=>api.exportSvg('图',[],'来源'),/尚未完整载入/);
 requests[1].job.reject(Error('县界读取失败'));await slow;
 assert.throws(()=>api.exportSvg('图',[],'来源'),/尚未完整载入/);
 const retry=api.reload();assert.equal(requests[2].name,'district');requests[2].job.resolve([]);await retry;
 assert.match(api.exportSvg('图',[],'来源'),/<svg/);
 const failed=api.reload();requests[3].job.reject(Error('仍然读取失败'));
 await assert.rejects(failed,/仍然读取失败/);assert.throws(()=>api.exportSvg('图',[],'来源'),/尚未完整载入/);
});

test('a failed base layer cannot export and reload actually retries both base sources before the administrative layer',async()=>{
 const env=exportMapEnv(),names:string[]=[];let fail=true;
 env.props.port.map=async(name:string)=>{names.push(name);if(name==='territory'&&fail)throw Error('国土轮廓读取失败');return [];};
 const {baseReady:_,...inputs}=env;
 const api=component('GeoMap',['currentLevel','syncLevel','loadBase','reload','exportSvg'],inputs,['disposed','baseReady','levelEpoch','lastMapError','lastLevel','requestedLevel','escape']);
 await assert.rejects(api.loadBase(),/国土轮廓读取失败/);assert.throws(()=>api.exportSvg('图',[],'来源'),/尚未完整载入/);
 fail=false;await api.reload();
 assert.deepEqual(names,['world','territory','world','territory','province']);assert.match(api.exportSvg('图',[],'来源'),/<svg/);
});

test('selecting an unclassified path matches only its empty layer and retains legacy wildcard projects',()=>{
 const rows=[{id:'missing',admin:null,values:['','','','','','','大区','','甲片','']},{id:'classified',admin:null,values:['','','','','','','大区','乙区','甲片','']}];
 const project=ref(emptyProject()),classFilters=ref(['','','','']);
 const env={computed,originalClassRows:ref(rows),project,classFilters,classSearch:ref(''),regionState:ref({effective:''}),selectedAdmin:ref(undefined),adminIndex:ref(new Map()),modal:ref('paths'),tablePage:ref(0),classValues,classFilterMatches,unclassifiedClassFilter,inRegion};
 const api=component('GeolinguisticsPage',['selectPath','filterClass'],env,['classRows','classOptions'],['classRows','classOptions']);
 assert.equal(api.classRows.value.length,2);assert.equal(diversity(rows as any,project.value,3).length,2);
 api.selectPath('大区 / 未分 / 甲片');assert.deepEqual(api.classRows.value.map((r:any)=>r.id),['missing']);
 assert.equal(classFilters.value[1],unclassifiedClassFilter);assert.ok(api.classOptions.value[1].includes(unclassifiedClassFilter));
 project.value.settings={...defaultSettings(),classFilters:classFilters.value};
 const restored=validateProject(JSON.parse(JSON.stringify(project.value)));assert.equal(restored.settings!.classFilters[1],unclassifiedClassFilter);
 api.filterClass(1,'');assert.equal(api.classRows.value.length,2);
 const legacy=validateProject({...emptyProject(),settings:defaultSettings()});assert.deepEqual(legacy.settings!.classFilters,['','','','']);
});

test('personal observations label map and CSV exports while retaining the original author and survey attribution',async()=>{
 const feature={id:'F1_P',volume:'P',number:1,page:7,title:'测试现象',categories:['A1','B1'],labels:{},groups:[{id:'a',label:'甲类',codes:['A1'],page:7,labelKind:'source'},{id:'b',label:'乙类',codes:['B1'],page:7,labelKind:'source'}]};
 const data=ref({sites:[{id:'S1',name:'点甲',lon:110,lat:30}],observations:{F1_P:{values:['A1'],raw:{}}}}),project=ref(emptyProject());project.value.mode='features';
 const captured:any[]=[],files:Blob[]=[];
 const env={data,project,feature:ref(feature),featureDetail:ref('broad'),visibleSites:ref([{site:data.value.sites[0],index:0}]),baseSites:ref([{site:data.value.sites[0],index:0}]),mapFeatures:ref([feature]),legend:ref([]),maps:ref([{exportSvg:(title:string,legend:any,source:string)=>{captured.push({title,legend,source});return '<svg width="1200" height="900"/>';}}]),sheet:ref('combined'),paperMode:ref('rgb'),observation,isObservationEdited,displayCounts,displayText,displayColor,broadPendingCount,isBroadCategory,volumeNames,composeMapSvgs,document:{fonts:{load:async()=>{},ready:Promise.resolve()}},ipaFontUrl:'test-font',ipaFontLicense:'font license',fetch:async()=>({ok:true,blob:async()=>new Blob(['font'])}),FileReader:class{result='data:font/ttf;base64,AA==';onload=()=>{};readAsDataURL(){this.onload();}},exportFile:async(_name:string,blob:Blob)=>{files.push(blob);},error:ref(''),csvText:(rows:any)=>JSON.stringify(rows),valueLabel:()=>''};
 const api=component('GeolinguisticsPage',['legendRecords','personalObservationCount','mapHandle','exportMap','exportTable'],env);
 await api.exportMap('svg');assert.equal(api.personalObservationCount(feature),0);assert.ok(!captured[0].source.includes('含个人校订'));
 project.value.edits['S1:F1_P']='B1';await api.exportMap('svg');assert.equal(api.personalObservationCount(feature),1);
 assert.match(captured[1].legend[0].label,/乙类/);assert.match(captured[1].source,/含个人校订（1 个地点）/);assert.match(captured[1].source,/Huang.*2024/);assert.match(captured[1].source,/调查 2000–2008/);
 api.exportTable();assert.match(await files.at(-1)!.text(),/Huang et al\. 2024.*含个人校订/);
 project.value.edits['S1:F1_P']='A1';assert.equal(api.personalObservationCount(feature),0);
 assert.match(source('GeolinguisticsPage'),/personalObservationCount\(feature\).*含个人校订/);
});

test('retry routes map, save and initial-load failures to their actual recovery and preserves a failed map error',async()=>{
 const {error:_,errorRetry:__,reportError:___,...env}=pageEnv();let reloads=0,saves=0,loads=0,fail=true;
 const api=component('GeolinguisticsPage',['mapHandle','reportError','retryError'],{...env,ref,computed,mapFeatures:ref([{},{}]),maps:ref([{reload:async()=>{reloads++;if(fail)throw Error('边界仍然读取失败');}},{reload:async()=>{reloads++;}}]),save:async()=>{saves++;},load:async()=>{loads++;}},['errorRetry','errorText','error'],['error','errorRetry']);
 env.project.value.mode='features';api.reportError('边界读取失败','map');await api.retryError();assert.equal(reloads,2);assert.equal(api.error.value,'边界仍然读取失败');assert.equal(api.errorRetry.value,'map');
 fail=false;await api.retryError();assert.equal(reloads,4);assert.equal(api.error.value,'');assert.equal(api.errorRetry.value,'');
 api.reportError('保存失败','save');await api.retryError();assert.equal(saves,1);assert.equal(loads,0);
 api.reportError('载入失败','load');await api.retryError();assert.equal(loads,1);
 api.error.value='导出失败';assert.equal(api.errorRetry.value,'');await api.retryError();assert.equal(reloads,4);assert.equal(saves,1);assert.equal(loads,1);
});

test('successful background saves preserve map errors and failed saves retain a real save retry action',async()=>{
 const env=pageEnv();env.error.value='地图读取失败';env.errorRetry.value='map';
 const api=component('GeolinguisticsPage',['save'],env,['saved','saveQueue','timer']);await api.save();assert.equal(env.error.value,'地图读取失败');assert.equal(env.errorRetry.value,'map');
 env.project.value.name='新名称';env.storage.compare=async()=>{throw Error('写入失败');};assert.equal(await api.save(),false);assert.equal(env.errorRetry.value,'save');assert.match(env.error.value,/写入失败/);
 env.storage.compare=async()=>{};assert.equal(await api.save(),true);assert.equal(env.error.value,'');
});

test('an actual initial resource failure retries load rather than saving an unavailable project',async()=>{
 const {error:_,errorRetry:__,reportError:___,...env}=pageEnv();let loads=0,saves=0;
 env.ready.value=false;env.geoPort.load=async()=>{loads++;if(loads===1)throw Error('读取资料失败');return {version:'recovered'};};
 const api=component('GeolinguisticsPage',['mapHandle','reportError','retryError','load','restoreView'],{...env,ref,computed,save:async()=>{saves++;},loadMatrix:async()=>{},maps:ref([]),mapFeatures:ref([])},['errorRetry','errorText','error','saved','lastUndo'],['error','errorRetry']);
 assert.equal(await api.load(),false);assert.equal(api.errorRetry.value,'load');assert.equal(env.ready.value,false);
 await api.retryError();assert.equal(loads,2);assert.equal(saves,0);assert.equal(env.ready.value,true);assert.equal(env.resources.value.version,'recovered');assert.equal(api.error.value,'');assert.equal(api.errorRetry.value,'');
});
