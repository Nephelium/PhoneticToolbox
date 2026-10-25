import {describe,it} from 'node:test';
import assert from 'node:assert/strict';
import {mapAttributions,mapAttributionText} from '../src/modules/geolinguistics/attribution.ts';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {parse,compileScript} from '@vue/compiler-sfc';
import ts from 'typescript';
import {createSSRApp,h} from 'vue';
import {renderToString} from '@vue/server-renderer';

function geoMap(){
 const file=new URL('../src/modules/geolinguistics/GeoMap.vue',import.meta.url),require=createRequire(file);
 const {descriptor}=parse(readFileSync(file,'utf8')),compiled=compileScript(descriptor,{id:'geo-map-attribution',inlineTemplate:true});
 const js=ts.transpileModule(compiled.content,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText;
 const module={exports:{} as any};new Function('require','module','exports',js)((name:string)=>name.endsWith('.css')?{}:require(name),module,module.exports);
 return module.exports.default;
}
const GeoMap=geoMap();
const mapProps=(attributions:ReturnType<typeof mapAttributions>)=>({port:{} as any,points:[],view:{center:[105,35],zoom:4},active:true,level:'province',attributions});

describe('boundary provenance',()=>{
 it('credits GeoAtlas study data and the selected island source without stale outline credits',()=>{
  const items=mapAttributions([{source:'datav-areas-v3-inferred-gcj-wgs84',supplementalSources:['nlsc-taiping']}]);
  assert.deepEqual(items.map(item=>item.id),['datav','nlsc']);
  const text=mapAttributionText(items);
  assert.ok(text.includes('仅供学习和交流'));assert.ok(text.includes('太平岛实际岛面'));
  assert.ok(!text.includes('MIT'));assert.ok(!text.includes('Huang'));assert.ok(!text.includes('OpenStreetMap'));
 });
 it('credits the loaded OSM and official regional data without claiming the old source',()=>{
  const items=mapAttributions([{source:'osm-20261022'},{source:'nlsc-official'},{source:'hk-had-official'},{source:'osm-20261022'}]);
  assert.deepEqual(items.map(item=>item.id),['osm','nlsc','hk-had','territory']);
  const text=mapAttributionText(items);
  assert.ok(text.includes('© OpenStreetMap contributors（ODbL 1.0） https://www.openstreetmap.org/copyright'));
  assert.ok(text.includes('© 香港特别行政区政府 · 民政事务总署 / DATA.GOV.HK'));
  assert.equal(items.find(item=>item.id==='hk-had')?.shortLabel,'© 香港政府');
  assert.equal(items.find(item=>item.id==='nlsc')?.shortLabel,'国土测绘中心');
  assert.ok(text.includes('县市 2020、乡镇 2023，含玛家三和补充面'));
  assert.ok(text.includes('2025-05-26'));
  assert.ok(!text.includes('GaryBikini'));
 });
 it('keeps legacy local-map attribution accurate',()=>{
  const items=mapAttributions([{source:'gary-20240206-gcj-iterative'},{source:'nlsc'}]);
  assert.ok(items.some(item=>item.id==='gary'));
  assert.ok(!items.some(item=>item.id==='osm'));
  assert.equal(items.find(item=>item.id==='gary')?.license,undefined);
 });
 it('renders a collapsed accessible source control while OSM remains visibly credited',async()=>{
  const items=mapAttributions([{source:'osm-20261022'},{source:'nlsc-official'},{source:'hk-had-official'}]);
  const context:any={},html=await renderToString(createSSRApp({render:()=>h(GeoMap,mapProps(items))}),context),details=context.teleports.body;
  assert.match(html,/<a[^>]+class="map-osm-credit"[^>]+href="https:\/\/www.openstreetmap.org\/copyright"[^>]*>© OpenStreetMap<\/a>/);
  assert.match(html,/aria-label="地图来源" aria-expanded="false" aria-controls="([^"]+)"/);
  assert.match(details,/<ul[^>]+class="map-attribution-details"[^>]* hidden[^>]+aria-label="地图来源与许可"/);
  assert.match(details,/<ul[^>]+data-background-surface/);
  for(const item of items){assert(details.includes(item.url));assert(details.includes(item.label));if(item.license)assert(details.includes(item.license));}
 });
 it('four controls have independent identities and legacy sources do not falsely credit OSM',async()=>{
  const props=mapProps(mapAttributions([{source:'gary-20240206-gcj-iterative'}]));
  const context:any={},html=await renderToString(createSSRApp({render:()=>h('div',Array.from({length:4},()=>h(GeoMap,props)))}),context);
  const ids=[...html.matchAll(/aria-controls="([^"]+)"/g)].map(match=>match[1]);
  assert.equal(ids.length,4);assert.equal(new Set(ids).size,4);assert(!html.includes('map-osm-credit'));assert(context.teleports.body.includes('GaryBikini'));
 });
});
