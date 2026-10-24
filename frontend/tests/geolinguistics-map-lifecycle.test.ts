import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {parse,compileScript} from '@vue/compiler-sfc';
import ts from 'typescript';
import * as vue from 'vue';
import Collection from 'ol/Collection.js';
import GeoJSON from 'ol/format/GeoJSON.js';
import * as projection from '../src/modules/geolinguistics/projected-base.ts';

const noop=()=>{},polygon=(x=110)=>({type:'Polygon',coordinates:[[[x,20],[x+1,20],[x+1,21],[x,21],[x,20]]]});
const feature=(id:string,geometry:any,properties={})=>({type:'Feature',id,geometry,properties});
const collection=(features:any[])=>({type:'FeatureCollection',features});
function data(){return {world:collection([feature('world',polygon())]),province:collection([feature('province:1',polygon(),{name:'province'})]),territory:collection([
 feature('area',polygon(),{renderRole:'administrative-display-area'}),feature('line',{type:'MultiLineString',coordinates:[[[110,20],[111,21]]]},{renderRole:'disputed-line'}),feature('symbol',polygon(111),{renderRole:'boundary-symbol'}),
 feature('mixed',{type:'GeometryCollection',geometries:[polygon(112),{type:'LineString',coordinates:[[112,20],[113,21]]}]})])};}
class OwnedItem{disposed=false;on(_type:string,_fn:unknown){}dispose(){this.disposed=true;}}
class FakeMap{
 interactions=new Collection([new OwnedItem()]);controls:Collection<OwnedItem>;disposed=false;layers:any[];view:any;
 constructor(options:any){this.layers=options.layers;this.view=options.view;this.controls=options.controls;}
 getView(){return this.view;}getSize(){return [200,200];}getPixelFromCoordinate(c:number[]){return [c[0]/1000,200-c[1]/1000];}
 on(){}updateSize(){}setTarget(){}setLayers(layers:any[]){this.layers=layers;}getInteractions(){return this.interactions;}getControls(){return this.controls;}
 addInteraction(item:any){this.interactions.push(item);}removeInteraction(item:any){this.interactions.remove(item);}
 dispose(){this.disposed=true;this.interactions.clear();this.controls.clear();}
}
const file=new URL('../src/modules/geolinguistics/GeoMap.vue',import.meta.url),nativeRequire=createRequire(file);
const compiled=compileScript(parse(readFileSync(file,'utf8')).descriptor,{id:'map-lifecycle-test',inlineTemplate:false});
const js=ts.transpileModule(compiled.content,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText;
// Actual SFC, OL Feature/Geometry/Source/Layer, format/cache/style and watchers.
// Only DOM/map host, controls, gestures and observer are synthetic, not a renderer test.
function mount(port:any,drawings?:object){
 const mounted:(()=>Promise<unknown>)[]=[],unmount:(()=>void)[]=[],scope=vue.effectScope(),module={exports:{} as any};let disconnected=false,closed=false;
 const require=(name:string)=>{
  if(name==='vue')return {...vue,onMounted:(fn:()=>Promise<unknown>)=>mounted.push(fn),onBeforeUnmount:(fn:()=>void)=>unmount.push(fn),useId:()=>String(Math.random())};
  if(name==='ol/Map.js')return {default:FakeMap};
  if(name==='ol/control.js')return {defaults:()=>new Collection([new OwnedItem()]),ScaleLine:OwnedItem};
  if(name==='ol/interaction/DragBox.js'||name==='ol/interaction/Draw.js')return {default:OwnedItem};
  if(name==='./projected-base.ts')return projection;
  return name.endsWith('.css')?{}:nativeRequire(name);
 };
 const document={createElement:()=>({getContext:()=>({font:'',measureText:(s:string)=>({width:s.length*8})})})};
 new Function('require','module','exports','window','document','ResizeObserver',js)(require,module,module.exports,{addEventListener:noop,removeEventListener:noop},document,class{observe(){}disconnect(){disconnected=true;}});
 const props=vue.reactive({port,points:[{id:'S1',lon:110,lat:20,name:'point',color:'#123456'}],view:{center:[105,35],zoom:4},active:true,level:'province',drawings,polygonColors:{'province:1':'#123456'}});
 const m=scope.run(()=>module.exports.default.setup(props,{emit:noop,expose:noop}));
 return {m,props,async start(){await Promise.all(mounted.map(fn=>fn()));},close(){if(closed)return;closed=true;for(const fn of unmount)fn();scope.stop();},disconnected:()=>disconnected};
}

test('actual GeoMap four-to-one-to-four shares only base/admin and clears owner listeners',async t=>{
 const read=t.mock.method(GeoJSON.prototype,'readFeatures'),raw=data(),port={map:async(name:keyof typeof raw)=>raw[name]},drawings=collection([feature('sketch',polygon())]);
 const maps=Array.from({length:4},()=>mount(port,drawings));
 try{await Promise.all(maps.map(m=>m.start()));const admin=maps[0].m.admin.getFeatures()[0],background=maps[0].m.background.getFeatures()[0];
  assert.equal(read.mock.callCount(),7);assert(maps.every(m=>m.m.admin.getFeatures()[0]===admin));assert(maps.every(m=>m.m.background.getFeatures()[0]===background));
  assert.equal(admin.getListeners('change').length,4);assert.notStrictEqual(maps[0].m.points.getFeatures()[0],maps[1].m.points.getFeatures()[0]);
  const sketch0=maps[0].m.sketch.getFeatures()[0],sketch1=maps[1].m.sketch.getFeatures()[0],before=sketch1.getGeometry().getCoordinates();
  assert.notStrictEqual(sketch0,sketch1);sketch0.getGeometry().translate(100,200);assert.deepEqual(sketch1.getGeometry().getCoordinates(),before);
  maps[1].props.polygonColors={'province:1':'#abcdef'};await vue.nextTick();
  const style=(m:any)=>m.layers[2].getStyleFunction()(admin,1).getStroke().getColor();
  assert.equal(style(maps[0].m),'#123456');assert.equal(style(maps[1].m),'#abcdef');assert.equal(admin.getStyle(),null);
  for(const map of maps.slice(1))map.close();assert.equal(admin.getListeners('change').length,1);assert.equal(maps[0].m.admin.getFeatures().length,1);
  const extra=Array.from({length:3},()=>mount(port));maps.push(...extra);await Promise.all(extra.map(m=>m.start()));
  assert.equal(read.mock.callCount(),7);assert.equal(admin.getListeners('change').length,4);
  for(const map of maps)map.close();assert.equal(admin.getListeners('change'),undefined);assert.equal(background.getListeners('change'),undefined);
  for(const map of maps){assert.equal(map.m.admin.getFeatures().length,0);assert.equal(map.m.admin.disposed,true);assert.equal(map.m.layers[2].getSource(),null);assert(map.disconnected());}
  assert.equal(admin.disposed,false);
 }finally{for(const map of maps)map.close();}
});

test('late base reply after actual SFC teardown cannot attach or project features',async t=>{
 const read=t.mock.method(GeoJSON.prototype,'readFeatures');let resolve!:(v:object)=>void;const wait=new Promise<object>(r=>resolve=r);
 const map=mount({map:()=>wait});const pending=map.start();map.close();resolve(collection([feature('late',polygon())]));await pending;
 assert.equal(read.mock.callCount(),0);assert.equal(map.m.background.getFeatures().length,0);assert.equal(map.m.territory.getFeatures().length,0);assert.equal(map.m.admin.getFeatures().length,0);
});

test('actual SVG random getter order paints all territory areas before linework and symbols',async t=>{
 const raw=data(),map=mount({map:async(name:keyof typeof raw)=>raw[name]});
 try{await map.start();const features=map.m.territory.getFeatures(),byId=new Map(features.map((f:any)=>[f.getId(),f]));
  t.mock.method(map.m.territory,'getFeatures',()=>['symbol','line','mixed','area'].map(id=>byId.get(id)));
  const svg=map.m.exportSvg('test',[],'source'),paths=[...svg.matchAll(/<path [^>]*fill="([^"]+)"[^>]*>/g)].map(m=>m[1]);
  assert.deepEqual(paths,['#edf0ed','#f8f8f1','#f8f8f1','none','none','#526c78','#12345638']);
  const styles=features.flatMap((f:any)=>map.m.layers[1].getStyleFunction()(f,1));
  assert.deepEqual(styles.map((s:any)=>s.getZIndex()).sort(),[0,0,1,1,2]);
 }finally{map.close();}
});
