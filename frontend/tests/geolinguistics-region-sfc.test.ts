import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {parse,compileScript} from '@vue/compiler-sfc';
import ts from 'typescript';
import * as vue from 'vue';
import * as model from '../src/modules/geolinguistics/model.ts';
import * as groups from '../src/modules/geolinguistics/groups.ts';
import {geoAtlasAdmins,geoAtlasResources} from './fixtures/geolinguistics-regions.ts';

// Execute the full, actual SFC setup. Only storage, file transport and mounted
// browser listeners are synthetic; this is not a physical-device/map audit.
function page(stored?:model.Project){
 const source=readFileSync(new URL('../src/modules/geolinguistics/GeolinguisticsPage.vue',import.meta.url),'utf8');
 const compiled=compileScript(parse(source).descriptor,{id:'region-test',inlineTemplate:false});
 const js=ts.transpileModule(compiled.content,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText;
 const module={exports:{} as any},cleanup:(()=>void)[]=[],scope=vue.effectScope(),noop=()=>{};
 const require=(name:string)=>{
  if(name==='vue')return {...vue,onMounted:noop,onBeforeUnmount:(fn:()=>void)=>cleanup.push(fn)};
  if(name==='./model.ts')return model;
  if(name==='./groups.ts')return groups;
  if(name==='../../platform/local-assets.ts')return {LocalAssets:class{async value(){return stored?JSON.stringify(stored):null;}async compare(){}}};
  if(name==='../../platform/geolinguistics.ts')return {createGeoPort:()=>({load:async()=>geoAtlasResources()})};
  return {default:{}};
 };
 new Function('require','module','exports','window','document',js)(require,module,module.exports,{addEventListener:noop,removeEventListener:noop},{fonts:{load:async()=>[],ready:Promise.resolve()}});
 const m=scope.run(()=>module.exports.default.setup({active:true,stateKey:'owned-region-test'},{emit:noop,expose:noop}));
 return {m,async load(){assert.equal(await m.load(),true,m.error.value);},close(){for(const fn of cleanup)fn();scope.stop();}};
}
const ids=(rows:model.Admin[])=>rows.map(a=>a.id);

test('actual SFC lists districts directly below municipalities and SAR provinces without inserting a city',async()=>{
 const f=page();try{await f.load();const m=f.m;
  for(const [province,district] of [['110000','110101'],['120000','120101'],['310000','310101'],['500000','500101'],['810000','810001'],['820000','820001']]){
   m.chooseRegion('province:'+province);assert.deepEqual(ids(m.nextRegions.value),['district:'+district]);
   m.chooseRegion('district:'+district);assert.equal(m.provinceId.value,'province:'+province);assert.equal(m.nextRegionId.value,'district:'+district);
   assert.deepEqual(m.subregions.value,[]);assert.equal(m.count.value,1);assert.deepEqual(m.fit.value.extent,m.adminIndex.value.get('district:'+district).extent);
  }
 }finally{f.close();}
});

test('actual SFC keeps source-city terminal entities selectable alongside ordinary cities',async()=>{
 const f=page();try{await f.load();const m=f.m;
  for(const [province,leaf] of [['410000','419001'],['420000','429004'],['420000','429021'],['460000','469024'],['650000','659010'],['440000','441900'],['440000','442000'],['460000','460400'],['620000','620200']]){
   m.chooseRegion('province:'+province);assert(ids(m.nextRegions.value).includes('city:'+leaf));
   m.chooseRegion('city:'+leaf);assert.equal(m.nextRegionId.value,'city:'+leaf);assert.deepEqual(m.subregions.value,[]);assert.equal(m.count.value,1);
   assert.equal(m.adminIndex.value.get('city:'+leaf).level,'city');assert.match(m.regionAvailability.value,/整体查看/);
  }
 }finally{f.close();}
});

test('actual SFC retains normal three-level paths, province-only scopes and deliberate national reset',async()=>{
 const f=page();try{await f.load();const m=f.m;
  m.chooseRegion('province:440000');assert.deepEqual(ids(m.nextRegions.value),['city:440100','city:441900','city:442000']);
  m.chooseRegion('city:440100');assert.deepEqual(ids(m.subregions.value),['district:440103']);assert.equal(m.subregionId.value,'');assert.equal(m.regionAvailability.value,'');
  m.chooseRegion('district:440103');assert.equal(m.nextRegionId.value,'city:440100');assert.equal(m.subregionId.value,'district:440103');assert.equal(m.count.value,1);
  m.chooseRegion('province:710000');assert.deepEqual(m.nextRegions.value,[]);assert.match(m.regionAvailability.value,/台湾省.*整体查看/);assert.equal(m.count.value,1);assert.equal(m.error.value,'');
  m.chooseRegion('');assert.equal(m.provinceId.value,'');assert.equal(m.nextRegionId.value,'');assert.equal(m.subregionId.value,'');assert.equal(m.regionAvailability.value,'');assert.deepEqual(m.fit.value.extent,[73,3,135,54]);
 }finally{f.close();}
});

test('actual SFC restores direct districts, ordinary counties and unavailable historical IDs without rewriting research fields',async()=>{
 for(const region of ['district:110101','district:440103','city:469024','province:710000','district:500112']){
  const original=model.emptyProject();original.settings={...model.defaultSettings(),region};original.edits={'S1:F1_P':'kept'};original.customSites=[{id:'user:kept',name:'自建点',lon:113,lat:23,value:'甲',note:'保留'}];
  const f=page(original);try{await f.load();const m=f.m;assert.equal(m.region.value,region);assert.deepEqual(m.project.value.edits,original.edits);assert.deepEqual(m.project.value.customSites,original.customSites);
   assert.equal(m.regionState.value.missing,region==='district:500112');assert.equal(m.error.value,'');
   if(region==='district:110101')assert.equal(m.nextRegionId.value,region);
   if(region==='district:440103')assert.equal(m.subregionId.value,region);
  }finally{f.close();}
 }
});

test('fixture retains every sampled official level and direct parent after selector use',async()=>{
 const original=geoAtlasAdmins(),f=page();try{await f.load();for(const a of original)f.m.chooseRegion(a.id);assert.deepEqual(f.m.resources.value.admin,original);}finally{f.close();}
});

test('actual page clears null map slots and trims four-to-one-to-four without render tracking',async()=>{
 const f=page();try{await f.load();const m=f.m,handles=Array.from({length:4},()=>({reload:async()=>{},exportSvg:()=>'<svg/>'}));
  let tracked=0;const runner=vue.effect(()=>{void m.maps.value;tracked++;});
  for(let i=0;i<4;i++)m.setMapRef(i,handles[i]);assert.equal(m.maps.value.length,4);assert.equal(vue.isReactive(m.maps.value),false);assert.strictEqual(m.mapHandle(0),handles[0]);
  m.setMapRef(1,null);assert.throws(()=>m.mapHandle(1),/地图尚未载入/);assert.equal(m.maps.value.length,4);
  m.setMapRef(3,null);m.setMapRef(2,null);assert.equal(m.maps.value.length,1);assert.strictEqual(m.mapHandle(0),handles[0]);
  for(let i=1;i<4;i++)m.setMapRef(i,handles[i]);assert.equal(m.maps.value.length,4);assert.equal(tracked,1);vue.stop(runner);
 }finally{f.close();assert.equal(f.m.maps.value.length,0);}
});

test('actual page map retry and export report missing handles rather than use removed panels',async()=>{
 const f=page();try{await f.load();const m=f.m;let retries=0;m.project.value.mode='custom';
  m.errorRetry.value='map';await m.retryError();assert.match(m.error.value,/地图尚未载入/);assert.equal(m.errorRetry.value,'map');
  m.setMapRef(0,{reload:async()=>{retries++;},exportSvg:()=>'<svg/>'});await m.retryError();assert.equal(retries,1);assert.equal(m.error.value,'');
  m.setMapRef(0,null);await m.exportMap('svg');assert.match(m.error.value,/地图尚未载入/);assert.equal(retries,1);
 }finally{f.close();}
});
