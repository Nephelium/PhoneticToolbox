import test from 'node:test';
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import GeoJSON from 'ol/format/GeoJSON.js';
import VectorSource from 'ol/source/Vector.js';
import {fromLonLat} from 'ol/proj.js';
import {projectedBaseFeatures} from '../src/modules/geolinguistics/projected-base.ts';

const collection=(offset=0)=>({type:'FeatureCollection',features:[{type:'Feature',id:'district:same',properties:{name:'same'},geometry:{type:'Polygon',coordinates:[[[110+offset,20],[111+offset,20],[111+offset,21],[110+offset,21],[110+offset,20]]]}}]});
const listeners=(f:any)=>(f.getListeners('change')??[]).length;

test('same immutable object projects once across four sources and reattaches without projection',t=>{
 const read=t.mock.method(GeoJSON.prototype,'readFeatures'),raw=collection(),before=structuredClone(raw);
 const sources=Array.from({length:4},()=>new VectorSource({features:[...projectedBaseFeatures(raw)]}));
 const feature=sources[0].getFeatures()[0];assert.equal(read.mock.callCount(),1);
 assert(sources.every(s=>s.getFeatureById('district:same')===feature));assert.equal(listeners(feature),4);
 assert.deepEqual((feature.getGeometry() as any).getCoordinates()[0][0],fromLonLat([110,20]));
 sources[3].clear(true);sources[2].clear();sources[1].removeFeature(feature);
 assert.equal(listeners(feature),1);assert.equal(sources[0].getFeatures().length,1);
 for(const source of sources.slice(1))source.addFeatures([...projectedBaseFeatures(raw)]);
 assert.equal(read.mock.callCount(),1);assert.equal(listeners(feature),4);assert.deepEqual(raw,before);
 for(const source of sources){source.clear(true);source.dispose();}
 assert.equal(listeners(feature),0);assert.equal(feature.disposed,false);
});

test('new JSON generation with identical IDs gets independent projected geometry',()=>{
 const old=projectedBaseFeatures(collection()),next=projectedBaseFeatures(collection(5));
 assert.notStrictEqual(old,next);assert.notStrictEqual(old[0],next[0]);
 assert.equal(old[0].getId(),next[0].getId());
 assert.notDeepEqual(old[0].getGeometry()!.getExtent(),next[0].getGeometry()!.getExtent());
});

test('failed conversion does not cache a partial result',()=>{
 const raw:any={type:'FeatureCollection',features:[{type:'Feature',geometry:{type:'unsupported',coordinates:[0,0]}}]};
 assert.throws(()=>projectedBaseFeatures(raw));raw.features=collection().features;
 assert.equal(projectedBaseFeatures(raw).length,1);
});

test('projected cache does not keep old raw keys, feature arrays or detached sources alive',()=>{
 const helper=new URL('../src/modules/geolinguistics/projected-base.ts',import.meta.url).href;
 const vector=new URL('../node_modules/ol/source/Vector.js',import.meta.url).href;
 const script=`import assert from 'node:assert/strict';import {projectedBaseFeatures} from ${JSON.stringify(helper)};import Source from ${JSON.stringify(vector)};
 function released(){const raw=${JSON.stringify(collection())};const features=projectedBaseFeatures(raw);const source=new Source({features:[...features]});source.clear(true);source.dispose();return [new WeakRef(raw),new WeakRef(features),new WeakRef(features[0]),new WeakRef(source)];}
 const weak=released();let collected=false;for(let i=0;i<40;i++){await new Promise(r=>setTimeout(r,5));globalThis.gc();await new Promise(r=>setTimeout(r,5));if(weak.every(w=>w.deref()===undefined)){collected=true;break;}}assert(collected,'Old cache key/value/source still retained after explicit GC');`;
 const child=spawnSync(process.execPath,['--expose-gc','--input-type=module'],{input:script,encoding:'utf8',timeout:10_000});
 assert.equal(child.status,0,child.stderr||String(child.error));
});
