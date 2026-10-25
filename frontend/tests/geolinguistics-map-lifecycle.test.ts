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
import {defaultMapExportOptions} from '../src/modules/geolinguistics/map-options.ts';

const noop=()=>{},polygon=(x=110)=>({type:'Polygon',coordinates:[[[x,20],[x+1,20],[x+1,21],[x,21],[x,20]]]});
const feature=(id:string,geometry:any,properties={})=>({type:'Feature',id,geometry,properties});
const collection=(features:any[])=>({type:'FeatureCollection',features});
function data(){return {world:collection([feature('world',polygon())]),province:collection([feature('province:1',polygon(),{name:'province'})]),territory:collection([
 feature('area',polygon(),{renderRole:'administrative-display-area'}),feature('line',{type:'MultiLineString',coordinates:[[[110,20],[111,21]]]},{renderRole:'disputed-line'}),feature('symbol',polygon(111),{renderRole:'boundary-symbol'}),
 feature('mixed',{type:'GeometryCollection',geometries:[polygon(112),{type:'LineString',coordinates:[[112,20],[113,21]]}]})])};}
class OwnedItem{disposed=false;on(_type:string,_fn:unknown){}dispose(){this.disposed=true;}}
class FakeMap{
 events=new Map<string,(event:any)=>void>();
 interactions=new Collection([new OwnedItem()]);controls:Collection<OwnedItem>;disposed=false;layers:any[];view:any;
 constructor(options:any){this.layers=options.layers;this.view=options.view;this.controls=options.controls;}
 getView(){return this.view;}getSize(){return [200,200];}getPixelFromCoordinate(c:number[]){return [c[0]/1000,200-c[1]/1000];}
 on(type:string,fn:(event:any)=>void){this.events.set(type,fn);}updateSize(){}setTarget(){}setLayers(layers:any[]){this.layers=layers;}getInteractions(){return this.interactions;}getControls(){return this.controls;}
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
 const frames=new Map<number,FrameRequestCallback>();let nextFrame=0;
 const emitted:{name:string;value:unknown}[]=[];
 const require=(name:string)=>{
  if(name==='vue')return {...vue,onMounted:(fn:()=>Promise<unknown>)=>mounted.push(fn),onBeforeUnmount:(fn:()=>void)=>unmount.push(fn),useId:()=>String(Math.random())};
  if(name==='ol/Map.js')return {default:FakeMap};
  if(name==='ol/control.js')return {defaults:()=>new Collection([new OwnedItem()]),ScaleLine:OwnedItem};
  if(name==='ol/interaction/DragBox.js'||name==='ol/interaction/Draw.js')return {default:OwnedItem};
  if(name==='./projected-base.ts')return projection;
  return name.endsWith('.css')?{}:nativeRequire(name);
 };
 const document={addEventListener:noop,removeEventListener:noop,createElement:()=>({getContext:()=>({font:'',measureText:(s:string)=>({width:s.length*8})})})};
 new Function('require','module','exports','window','document','ResizeObserver','requestAnimationFrame','cancelAnimationFrame',js)(require,module,module.exports,{addEventListener:noop,removeEventListener:noop},document,class{observe(){}disconnect(){disconnected=true;}},(fn:FrameRequestCallback)=>{frames.set(++nextFrame,fn);return nextFrame;},(id:number)=>frames.delete(id));
 const props=vue.reactive({port,points:[{id:'S1',lon:110,lat:20,name:'point',color:'#123456'}],view:{center:[105,35],zoom:4},active:true,level:'province',drawings,polygonColors:{'province:1':'#123456'}});
 const m=scope.run(()=>module.exports.default.setup(props,{emit:(name:string,value:unknown)=>emitted.push({name,value}),expose:noop}));
 return {m,props,frames,emitted,tick(){const callbacks=[...frames.values()];frames.clear();for(const fn of callbacks)fn(0);},async start(){await Promise.all(mounted.map(fn=>fn()));},close(){if(closed)return;closed=true;for(const fn of unmount)fn();scope.stop();},disconnected:()=>disconnected};
}

test('actual clicks retain pixel tolerances and point-before-region selection with only the intended layer',async()=>{
 const raw=data(),map=mount({map:async(name:keyof typeof raw)=>raw[name]});
 try{
  await map.start();const host=map.m.map as any,click=host.events.get('singleclick'),queries:any[]=[];let pointHit=true;
  host.forEachFeatureAtPixel=(_pixel:unknown,callback:any,options:any)=>{
   queries.push(options);
   for(const layer of [map.m.territoryLayer,map.m.sketchLayer])assert.equal(options.layerFilter(layer),false);
   if(options.layerFilter(map.m.pointLayer)){if(pointHit)return callback(map.m.points.getFeatures()[0]);}
   else{assert.equal(options.layerFilter(map.m.adminLayer),true);return callback(map.m.admin.getFeatures()[0]);}
  };
  click({pixel:[50,50]});assert.equal(queries.length,1);assert.equal(queries[0].hitTolerance,6);assert.deepEqual(map.emitted.at(-1),{name:'site',value:'S1'});
  pointHit=false;queries.length=0;click({pixel:[50,50]});assert.equal(queries.length,2);assert.equal(queries[1].hitTolerance,1);assert.deepEqual(map.emitted.at(-1),{name:'region',value:'province:1'});
 }finally{map.close();}
});

test('actual hover coalesces moves, wraps world coordinates and cancels when hidden or removed',async()=>{
 const raw=data(),map=mount({map:async(name:keyof typeof raw)=>raw[name]});
 try{
  await map.start();const host=map.m.map as FakeMap,move=host.events.get('pointermove')!,coordinate=map.m.points.getFeatures()[0].getGeometry().getCoordinates();
  move({coordinate,dragging:false});move({coordinate:[1e8,1e8],dragging:false});assert.equal(map.frames.size,1);map.tick();assert.equal(map.m.hover.value,'');
  const worldWidth=host.view.getProjection().getExtent()[2]-host.view.getProjection().getExtent()[0];
  move({coordinate:[coordinate[0]+worldWidth,coordinate[1]],dragging:false});map.tick();assert.equal(map.m.hover.value,'point');
  move({coordinate,dragging:false});map.props.active=false;await vue.nextTick();assert.equal(map.frames.size,0);assert.equal(map.m.hover.value,'');
  map.props.active=true;await vue.nextTick();move({coordinate,dragging:false});move({coordinate,dragging:true});assert.equal(map.frames.size,0);
  move({coordinate,dragging:false});assert.equal(map.frames.size,1);map.close();assert.equal(map.frames.size,0);
 }finally{map.close();}
});

test('actual GeoMap four-to-one-to-four shares only base/admin and clears owner listeners',async t=>{
 const read=t.mock.method(GeoJSON.prototype,'readFeatures'),raw=data(),port={map:async(name:keyof typeof raw)=>raw[name]},drawings=collection([feature('sketch',polygon())]);
 const maps=Array.from({length:4},()=>mount(port,drawings));
 try{await Promise.all(maps.map(m=>m.start()));const admin=maps[0].m.admin.getFeatures()[0],context=maps[0].m.territory.getFeatures()[0];
  assert.equal(read.mock.callCount(),6);assert(maps.every(m=>m.m.admin.getFeatures()[0]===admin));assert(maps.every(m=>m.m.territory.getFeatures()[0]===context));
  assert.equal(admin.getListeners('change').length,4);assert.notStrictEqual(maps[0].m.points.getFeatures()[0],maps[1].m.points.getFeatures()[0]);
  const sketch0=maps[0].m.sketch.getFeatures()[0],sketch1=maps[1].m.sketch.getFeatures()[0],before=sketch1.getGeometry().getCoordinates();
  assert.notStrictEqual(sketch0,sketch1);sketch0.getGeometry().translate(100,200);assert.deepEqual(sketch1.getGeometry().getCoordinates(),before);
  maps[1].props.polygonColors={'province:1':'#abcdef'};await vue.nextTick();
  const style=(m:any)=>m.adminLayer.getStyleFunction()(admin,1).getStroke().getColor();
  assert.equal(style(maps[0].m),'#123456');assert.equal(style(maps[1].m),'#abcdef');assert.equal(admin.getStyle(),null);
  for(const map of maps.slice(1))map.close();assert.equal(admin.getListeners('change').length,1);assert.equal(maps[0].m.admin.getFeatures().length,1);
  const extra=Array.from({length:3},()=>mount(port));maps.push(...extra);await Promise.all(extra.map(m=>m.start()));
  assert.equal(read.mock.callCount(),6);assert.equal(admin.getListeners('change').length,4);
  for(const map of maps)map.close();assert.equal(admin.getListeners('change'),undefined);assert.equal(context.getListeners('change'),undefined);
  for(const map of maps){assert.equal(map.m.admin.getFeatures().length,0);assert.equal(map.m.admin.disposed,true);assert.equal(map.m.adminLayer.getSource(),null);assert(map.disconnected());}
  assert.equal(admin.disposed,false);
 }finally{for(const map of maps)map.close();}
});

test('late base reply after actual SFC teardown cannot attach or project features',async t=>{
 const read=t.mock.method(GeoJSON.prototype,'readFeatures');let resolve!:(v:object)=>void;const wait=new Promise<object>(r=>resolve=r);
 const map=mount({map:()=>wait});const pending=map.start();map.close();resolve(collection([feature('late',polygon())]));await pending;
 assert.equal(read.mock.callCount(),0);assert.equal(map.m.territory.getFeatures().length,0);assert.equal(map.m.admin.getFeatures().length,0);
});

test('actual SVG uses only active land and paints independent context lines before symbols',async t=>{
 const raw=data(),map=mount({map:async(name:keyof typeof raw)=>raw[name]});
 try{await map.start();const features=map.m.territory.getFeatures(),byId=new Map(features.map((f:any)=>[f.getId(),f]));
  t.mock.method(map.m.territory,'getFeatures',()=>['symbol','line','mixed'].map(id=>byId.get(id)));
  const svg=map.m.exportSvg('test',[],'source'),paths=[...svg.matchAll(/<path [^>]*fill="([^"]+)"[^>]*>/g)].map(m=>m[1]);
  assert.deepEqual(paths,[map.m.adminLayer.getStyleFunction()(map.m.admin.getFeatures()[0],1).getFill().getColor(),'none','none','#526c78']);
  const styles=features.flatMap((f:any)=>map.m.layers[1].getStyleFunction()(f,1));
  assert.deepEqual(styles.map((s:any)=>s.getZIndex()).sort(),[1,1,2]);
 }finally{map.close();}
});

test('actual multilevel outlines have no fills, retain one primary, export independent choices and release listeners',async()=>{
 const raw={...data(),city:collection([feature('city:1',polygon())]),district:collection([feature('district:1',polygon())])},calls:string[]=[];
 const map=mount({map:async(name:keyof typeof raw)=>{calls.push(name);return raw[name];}});
 try{await map.start();map.props.level='district';(map.props as any).extraBoundaries=['province','city'];await vue.nextTick();await map.m.syncLevel();await new Promise(resolve=>setImmediate(resolve));
  assert.equal(map.m.lastLevel,'district');assert.equal(map.m.admin.getFeatures().length,1);assert.equal(map.m.boundaries.getFeatures().length,2);
  const strokes=map.m.boundaries.getFeatures().map((f:any)=>map.m.boundaryLayer.getStyleFunction()(f,1));assert(strokes.every((s:any)=>s.getFill()===null));
  assert.deepEqual(strokes.map((s:any)=>s.getStroke().getWidth()).sort(),[1.3,2.2]);assert.equal(map.m.adminLayer.getStyleFunction()(map.m.admin.getFeatures()[0],1).getStroke().getWidth(),.6);
  const options=defaultMapExportOptions(['province','city','district']);await map.m.prepareExport(options);const svg=map.m.exportSvg('图',[{label:'VERY_LONG_LEGEND',color:'#123456'}],'source',options);
  assert.equal((svg.match(/fill="#f8f8f1"/g)??[]).length,1,'one primary land fill');assert.equal((svg.match(/data-boundary=/g)??[]).length,3);assert(!svg.includes('VERY_LONG_LEGEND'));assert(!svg.includes('data-place-name='));
  options.boundaries=[];options.scale=false;options.names='all';options.legend=true;const names=map.m.exportSvg('图',[{label:'legend enabled',color:'#123456'}],'source',options);assert(!names.includes('data-boundary='));assert(names.includes('data-place-name="S1"'));assert(names.includes('legend enabled'));
  options.names='selected';assert(!map.m.exportSvg('图',[],'source',options).includes('data-place-name='));(map.props.points[0] as any).selected=true;assert(map.m.exportSvg('图',[],'source',options).includes('data-place-name="S1"'));
  const before=calls.length;for(let i=0;i<3;i++){map.m.publishView();await map.m.syncLevel();}assert.equal(calls.length,before,'pan does not reload boundary data');
  const held=map.m.boundaries.getFeatures();map.close();assert(held.every((f:any)=>!f.getListeners('change')));assert.equal(map.m.boundaries.disposed,true);
 }finally{map.close();}
});
