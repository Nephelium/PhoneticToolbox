import test from 'node:test';import assert from 'node:assert/strict';
import {compareFeatures,hasCompleteMeanings,categoryCounts,diversity,emptyProject,defaultSettings,inRegion,validateProject,nearest,csvText,observation,type ClassificationRow,type Admin,type Resources} from '../src/modules/geolinguistics/model.ts';
import {composeMapSvgs,dimensionColor,wrapMapText} from '../src/modules/geolinguistics/export.ts';
import {categoryText,observationText,type AtlasFeature} from '../src/modules/geolinguistics/model.ts';

test('source meanings stay feature-local and ordered variants keep IPA slashes intact',()=>{
 const feature={id:'F120_P',labels:{B1:'咸[n]；宕[ŋ]',B2:'咸[n]；宕[ŋ/n]'}} as AtlasFeature;
 const project=emptyProject();
 assert.equal(observationText('B2#B1',feature,project),'1. 咸[n]；宕[ŋ/n]；2. 咸[n]；宕[ŋ]');
 assert.equal(categoryText('A8',feature,project),'A8 · 释义待核');
 assert.equal(categoryText('B1',{...feature,id:'F1_L',labels:{}},project),'B1 · 释义待核');
 project.labels.F120_P={B1:'[a\u0303 n\u0329 ŋ ʔ]'};
 assert.equal(categoryText('B1',feature,project,true),'[a\u0303 n\u0329 ŋ ʔ]（B1）');
 project.labels.F120_P.B1='';assert.equal(categoryText('B1',feature,project),'咸[n]；宕[ŋ]');
 assert.equal(observationText(null,feature,project),'原值待核');
});
test('map legend wrapping never detaches combining IPA marks and preserves all text',()=>{
 const text='咸[a\u0303]；山[n\u0329]；宕[ŋ/n]';
 const lines=wrapMapText(text,2,s=>Array.from(s).length);
 assert.equal(lines.join(''),text);
 assert(!lines.some(line=>/^\p{M}/u.test(line)));
 assert(lines.includes('a\u0303'));assert(lines.includes('n\u0329'));
 const counted=wrapMapText('咸[n]；宕[ŋ/n]（B2） · 121 个地点',10,s=>s.length);
 assert(counted.some(line=>line.includes('121')));
 assert(counted.some(line=>line.includes('B2')));
});
test('parallel variants count sites once per category and retain missing separately',()=>{assert.deepEqual(categoryCounts(['A1#B2','A1#A1',null]),[{code:'A1',count:2},{code:'B2',count:1},{code:'待核',count:1}]);});
test('same leaf name under different dialect parents has distinct identity; duplicate rows do not inflate category count',()=>{const row=(id:string,area:string)=>({id,values:['','','','','','','大区',area,'湖广片','']} as ClassificationRow);const result=diversity([row('1','甲区'),row('2','乙区'),row('3','甲区')],emptyProject(),3);assert.equal(result.length,2);assert.equal(result[0].rows.length,2);});
test('user corrections are overlays, raw classification rows remain intact',()=>{const row={id:'combined:2',values:['','','','','','','甲','乙','丙','']} as ClassificationRow;const p=emptyProject();p.classEdits[row.id]=['','','','','','','甲','乙','丁',''];assert.equal(diversity([row],p,3)[0].path,'甲 / 乙 / 丁');assert.equal(row.values[8],'丙');});
test('administrative levels are namespaced and recursive filtering is bounded',()=>{const index=new Map([['province:1',{id:'province:1',parent:'100000'}],['city:1',{id:'city:1',parent:'province:1'}],['district:2',{id:'district:2',parent:'city:1'}],['loop',{id:'loop',parent:'loop'}]] as [string,Admin][]);assert(inRegion('district:2','province:1',index));assert(!inRegion('loop','province:1',index));assert(!inRegion(null,'province:1',index));assert(inRegion(null,'',index));});
test('original distance values and tie order survive lookup',()=>{const a=new Float64Array([0,.210830999,.1,.210830999,0,.2,.1,.2,0]);assert.deepEqual(nearest(a,0,2),[{index:2,distance:.1},{index:1,distance:.210830999}]);assert.throws(()=>nearest(a,3,2));});
test('project round trip preserves Unicode IPA and rejects incompatible or invalid coordinates',()=>{const p=emptyProject();p.customSites=[{id:'user:1',name:'珠海',lon:113.5,lat:22.3,admin:null,value:'ŋ̍˨˩',note:'文白异读'}];assert.deepEqual(validateProject(JSON.parse(JSON.stringify(p))),p);assert.throws(()=>validateProject({...p,schema:'other'}));assert.throws(()=>validateProject({...p,customSites:[{...p.customSites[0],lat:NaN}]}));assert.throws(()=>validateProject({...p,classEdits:{x:['missing']}}));});
test('observation correction does not change shared source values',()=>{const data={sites:[{id:'S1'}],observations:{F1_P:{values:['A1#A2'],raw:{}}}} as unknown as Resources;const p=emptyProject();p.edits['S1:F1_P']='B1';assert.equal(observation(data,p,'F1_P',0),'B1');assert.equal(data.observations.F1_P.values[0],'A1#A2');});
test('spreadsheet exports quote commas, newlines, quotes and formula prefixes',()=>{assert.equal(csvText([['=SUM(A1)','a,b','x"y','m\nn']]),'\uFEFF"\'=SUM(A1)","a,b","x""y","m\nn"');});

test('project filters round trip and malformed drawings or IPA labels fail before rendering',()=>{const p=emptyProject();p.settings={...defaultSettings(),region:'province:430000',mapCount:4};assert.deepEqual(validateProject(p).settings,p.settings);assert.throws(()=>validateProject({...p,settings:{...p.settings,mapCount:99}}));assert.throws(()=>validateProject({...p,labels:{F1_P:{A1:17}}}));assert.throws(()=>validateProject({...p,drawings:{type:'FeatureCollection',features:[{type:'Feature',geometry:{type:'Polygon',coordinates:[[[0,0],[1,0],[1,1],[3,3]]]}}]}}));});

test('multi-map exports retain every panel with separate clipping paths',()=>{const panel='<svg width="400" height="300"><defs><clipPath id="frame"/></defs><g clip-path="url(#frame)">[tʂʰ n̩]</g></svg>';const svg=composeMapSvgs([panel,panel,panel,panel]);assert(svg.includes('width="820" height="620"'));for(let i=0;i<4;i++){assert(svg.includes(`id="frame${i}"`));assert(svg.includes(`url(#frame${i})`));}assert.equal((svg.match(/\[tʂʰ n̩\]/g)??[]).length,4);});

test('MDS sign and zero retain a fixed full-dataset colour centre',()=>{assert.equal(dimensionColor(0,[-2,1]),'#ffffff');assert.equal(dimensionColor(-2,[-2,1]),'#ff0000');assert.equal(dimensionColor(1,[-2,1]),'#8080ff');});

test('new projects start from readable county classifications and old projects preserve their mode and edits',()=>{
 const fresh=emptyProject();assert.equal(fresh.mode,'classification');assert.equal(fresh.atlasView,undefined);
 const old={...fresh,mode:'features' as const,edits:{'S1:F1_P':'A1'},atlasView:undefined,bookmarks:undefined};
 const restored=validateProject(JSON.parse(JSON.stringify(old)));assert.equal(restored.mode,'features');assert.equal(restored.edits['S1:F1_P'],'A1');
 const saved={...fresh,atlasView:'split' as const,bookmarks:['F120_P','F1_L']};assert.deepEqual(validateProject(saved),saved);
 assert.throws(()=>validateProject({...fresh,bookmarks:['F1_P','F1_P']}));assert.throws(()=>validateProject({...fresh,atlasView:'other'}));
});
test('only complete source meanings qualify; personal labels and stale review totals cannot promote a feature',()=>{
 const f={id:'F1_P',categories:['A','B'],labels:{A:'甲',B:'乙'}} as AtlasFeature;
 assert(hasCompleteMeanings(f));assert(!hasCompleteMeanings({...f,labels:{A:'甲'},categoryReview:{verified:2,total:2}}));
 assert(!hasCompleteMeanings({...f,labels:{A:'甲',B:' '}}));assert(!hasCompleteMeanings({...f,categories:[]}));
 const p={...emptyProject(),featureScope:'all' as const};assert.equal(validateProject(p).featureScope,'all');
 assert.throws(()=>validateProject({...p,featureScope:'bad'}));
});
test('comparisons distinguish missing values, retain variant order, and use personal overlays without changing source distances or observations',()=>{
 const features=['F1_P','F2_P','F3_P','F4_P','F5_P'].map(id=>({id} as AtlasFeature));
 const values=[['A','A'],['A#B','B#A'],[null,null],[null,'B'],['A','B']];
 const data={sites:[{id:'S1'},{id:'S2'}],catalog:{features},observations:Object.fromEntries(features.map((f,i)=>[f.id,{values:values[i]}]))} as unknown as Resources;
 const p=emptyProject();assert.deepEqual(compareFeatures(data,p,[]),[]);
 assert.deepEqual(compareFeatures(data,p,[0,1]).map(r=>r.status),['same','different','missing','missing','different']);
 p.edits['S2:F5_P']='A';assert.equal(compareFeatures(data,p,[0,1])[4].status,'same');assert.equal(data.observations.F5_P.values[1],'B');
});
