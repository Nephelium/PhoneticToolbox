import test from 'node:test';
import assert from 'node:assert/strict';
import {reactive,computed} from 'vue';
import {displayCodes,displayCounts,displayValue,displayColor,completeBroadMeanings,hasBroadMeanings,isBroadCategory,broadPendingCount,pointAppearance} from '../src/modules/geolinguistics/groups.ts';
import {emptyProject,validateProject,observation,type Resources,type AtlasFeature} from '../src/modules/geolinguistics/model.ts';
import {geoDataNames,readGeoDirectory} from '../src/platform/geolinguistics-directory.ts';

const feature:AtlasFeature={id:'F120_P',volume:'P',number:120,page:456,title:'测试',titleReviewed:true,legend:'',categories:['A1','A2','A3','B1'],labels:{A1:'[a\u0303 n\u0329]'},groups:[{id:'nasal',label:'有[-m]',codes:['A1','A2'],page:456,labelKind:'source'},{id:'n',label:'只有[-n -ŋ]',codes:['B1'],page:456,labelKind:'source'}]};
test('broad membership is explicit and does not extend a prefix or a user fine label',()=>{
 const p=emptyProject();p.labels.F120_P={A3:'个人细类'};
 assert.deepEqual(displayCodes('A1#A2#A3',feature,'broad'),['@group:nasal','@unmapped-broad']);
 assert.equal(displayValue('A3',feature,p,'broad'),'此记录的大类尚待对应');
 assert.equal(displayValue('A3',feature,p,'fine'),'个人细类');
 assert(hasBroadMeanings(feature));assert(!completeBroadMeanings(feature));
});
test('broad site counts deduplicate within a group and retain mixed and missing observations',()=>{
 const records=[{value:'A1#A2'},{value:'B1#A1'},{value:'A3'},{value:null}];
 assert.deepEqual(displayCounts(records,feature,'broad'),[{code:'@group:nasal',count:2},{code:'@group:n',count:1},{code:'@unmapped-broad',count:1},{code:'待核',count:1}]);
 assert.deepEqual(records,[{value:'A1#A2'},{value:'B1#A1'},{value:'A3'},{value:null}]);
 assert.equal(displayValue('A1#A2',feature,emptyProject(),'broad'),'有[-m]');
 assert.equal(displayValue('A1',feature,emptyProject(),'fine'),'[a\u0303 n\u0329]');
 assert.equal(displayColor('A3',feature,'broad'),'#9099a2');
});
test('known map correspondence conflicts remain explicit in maps, counts and values',()=>{
 const f={...feature,groupExceptions:{S1:{code:'A1',reason:'原表与原图对应待核'}}};
 assert.equal(displayValue('A1',f,emptyProject(),'broad','S1'),'原图对应待核');
 assert.deepEqual(displayCounts([{value:'A1',siteId:'S1'},{value:'A1',siteId:'S2'}],f,'broad'),[{code:'@group:nasal',count:1},{code:'@source-conflict',count:1}]);
 assert.deepEqual(displayCodes('B1',f,'broad','S1'),['@group:n']);
 assert.deepEqual(displayCodes('A1#B1',f,'broad','S1'),['@source-conflict','@group:n']);
 assert.equal(displayValue('A1',f,emptyProject(),'fine','S1'),'[a\u0303 n\u0329]');
});
test('optional display layer survives roundtrip without migrating older projects',()=>{
 assert.equal(validateProject(emptyProject()).featureDetail,'broad');
 const legacy={...emptyProject(),featureDetail:undefined};assert.equal(validateProject(legacy).featureDetail,'fine');
 const p={...emptyProject(),featureDetail:'broad' as const};assert.deepEqual(validateProject(p),p);
 assert.throws(()=>validateProject({...p,featureDetail:'prefix'}));
});
test('two conflicting atoms at one site stay unresolved without hiding a valid third atom',()=>{
 const f={...feature,groupExceptions:{S1:[{code:'A1',reason:'第一项原图冲突'},{code:'A2',reason:'第二项原图冲突'}]}};
 assert.deepEqual(displayCodes('A1#A2',f,'broad','S1'),['@source-conflict']);
 assert.equal(pointAppearance('A1#A2',f,'broad','S1').unclassified,true);
 assert.deepEqual(displayCodes('A1#A2#B1',f,'broad','S1'),['@source-conflict','@group:n']);
 assert.equal(broadPendingCount([{value:'A1#A2',siteId:'S1'}],f),1);
 assert.deepEqual(displayCodes('A1#A2',f,'broad','S2'),['@group:nasal']);
 assert.deepEqual(displayCodes('A1#A2',f,'fine','S1'),['A1','A2']);
});
async function fixture(){
 const entries:{path:string;bytes:number;sha256:string}[]=[],files:File[]=[];
 for(const path of geoDataNames){const file=new File(['{}'],path.split('/').at(-1)!);Object.defineProperty(file,'webkitRelativePath',{value:'data/'+path});files.push(file);entries.push({path,bytes:file.size,sha256:[...new Uint8Array(await crypto.subtle.digest('SHA-256',await file.arrayBuffer()))].map(n=>n.toString(16).padStart(2,'0')).join('')});}
 const manifest=new File([JSON.stringify({schema:'ptb-geolinguistics-data/1',version:'m19-local-20261022',profile:'interactive-data',files:entries})],'data-manifest.json');Object.defineProperty(manifest,'webkitRelativePath',{value:'data/data-manifest.json'});return [...files,manifest];
}
test('folder importer verifies exact manifest members and ignores extra book files without reading them',async()=>{
 const files=await fixture();const book=new File(['private'],'book.pdf');book.arrayBuffer=async()=>{throw Error('must not read book');};files.push(book);
 assert.equal((await readGeoDirectory(files)).length,16);
 await assert.rejects(()=>readGeoDirectory(files.filter(f=>f.name!=='sites.json')),/缺失/);
 const bad=new File(['[]'],'sites.json');Object.defineProperty(bad,'webkitRelativePath',{value:'data/sites.json'});
 await assert.rejects(()=>readGeoDirectory(files.map(f=>f.name==='sites.json'?bad:f)),/校验/);
});

test('adding and restoring a personal observation immediately updates reactive broad counts',()=>{
 const data={sites:[{id:'S1'}],observations:{F120_P:{values:['B1']}}} as unknown as Resources;
 const project=reactive(emptyProject());
 const counts=computed(()=>displayCounts([{value:observation(data,project,'F120_P',0),siteId:'S1'}],feature,'broad'));
 assert.deepEqual(counts.value,[{code:'@group:n',count:1}]);
 project.edits['S1:F120_P']='A1#A2';
 assert.deepEqual(counts.value,[{code:'@group:nasal',count:1}]);
 delete project.edits['S1:F120_P'];
 assert.deepEqual(counts.value,[{code:'@group:n',count:1}]);
});

test('unmatched record status deduplicates atoms and never becomes a linguistic class',()=>{
 assert.deepEqual(displayCodes('A3#Z1#Z2',feature,'broad'),['@unmapped-broad']);
 assert.deepEqual(displayCodes('A3#Z1#Z2',feature,'fine'),['A3','Z1','Z2']);
 assert.deepEqual(displayCounts([{value:'A3#Z1'},{value:'Z2'}],feature,'broad'),[{code:'@unmapped-broad',count:2}]);
});

 test('unmapped points retain their position without a grey legend category; mixed records retain known membership',()=>{
 const records=[{value:'Z1#Z2'},{value:'A1#Z1'},{value:null},{value:'B1'}];
 assert.equal(broadPendingCount(records,feature),3);
 assert.deepEqual(displayCounts(records,feature,'broad').filter(v=>isBroadCategory(v.code)),[{code:'@group:nasal',count:1},{code:'@group:n',count:1}]);
 assert.equal(pointAppearance('Z1#Z2',feature,'broad').unclassified,true);
 assert.equal(pointAppearance('A1#Z1',feature,'broad').unclassified,false);
 assert.equal(pointAppearance('Z1#A1',feature,'broad').color,pointAppearance('A1',feature,'broad').color);
 assert.equal(pointAppearance('A1#Z1',feature,'broad').incomplete,true);
 assert.equal(pointAppearance('A1#Z1',feature,'broad').multi,false);
 assert.equal(pointAppearance('A1#B1#Z1',feature,'broad').multi,true);
 assert.equal(pointAppearance('A1#B1',feature,'broad').incomplete,false);
});
