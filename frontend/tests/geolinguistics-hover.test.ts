import {it} from 'node:test';
import assert from 'node:assert/strict';
import Feature from 'ol/Feature.js';
import Point from 'ol/geom/Point.js';
import Polygon from 'ol/geom/Polygon.js';
import LineString from 'ol/geom/LineString.js';
import GeometryCollection from 'ol/geom/GeometryCollection.js';
import type Geometry from 'ol/geom/Geometry.js';
import VectorSource from 'ol/source/Vector.js';
import {mapHoverName,type HoverLayer} from '../src/modules/geolinguistics/map-hover.ts';

const source=(features:Feature<Geometry>[])=>new VectorSource({features});
const point=(name:string,x:number,y:number,extra={})=>new Feature({geometry:new Point([x,y]),datum:{name,...extra}});
const area=(name:string,rings:number[][][])=>new Feature({name,geometry:new Polygon(rings)});
const square=(a:number,b:number)=>[[a,a],[b,a],[b,b],[a,b],[a,a]];
const hover=(layers:HoverLayer[],coordinate:number[],resolution=1,labels=false)=>mapHoverName(layers,coordinate,resolution,labels,name=>name.length*12,120);

it('finds nearby symbols with pixel tolerance at each map resolution',()=>{
 const points=source([point('调查点',100,200)]),layers:HoverLayer[]=[{source:points,kind:'points'}];
 assert.equal(hover(layers,[108,200]),'调查点');
 assert.equal(hover(layers,[109,200]),'');
 assert.equal(hover(layers,[116,200],2),'调查点');
 assert.equal(hover(layers,[118,200],2),'');
 assert.equal(hover(layers,[100,200],0),'');
 assert.equal(hover(layers,[100,200],NaN),'');
});

it('keeps selected and last-painted coincident symbols above other markers',()=>{
 const selected=point('选中',100,200,{selected:true}),later=point('普通',100,200);
 const layers:HoverLayer[]=[{source:source([selected,later]),kind:'points'}];
 assert.equal(hover(layers,[100,200]),'选中');
 selected.set('datum',{name:'原选中'});
 assert.equal(hover(layers,[100,200]),'普通');
});

it('includes visible labels above points and stops after labels are hidden',()=>{
 const layers:HoverLayer[]=[{source:source([point('很长的调查点名称',100,200)]),kind:'points'}];
 assert.equal(hover(layers,[140,213],1,true),'很长的调查点名称');
 assert.equal(hover(layers,[140,213],1,false),'');
 assert.equal(hover(layers,[140,187],1,true),'');
});

it('respects holes, source order, line tolerance and hidden province context',()=>{
 const layers:HoverLayer[]=[
  {source:source([area('行政区',[square(0,100),square(30,70)])]),kind:'admin'},
  {source:source([new Feature({name:'旧省界',boundaryType:'省界',geometry:new LineString([[0,50],[100,50]])})]),kind:'territory'},
  {source:source([area('背景',[square(-100,200)])]),kind:'background'}
 ];
 assert.equal(hover(layers,[10,10]),'行政区');
 assert.equal(hover(layers,[50,50]),'背景');
 assert.equal(hover(layers,[32,50]),'行政区');
 assert.equal(hover(layers,[400,400]),'');
 const lines:HoverLayer[]=[{source:source([new Feature({name:'国界',geometry:new LineString([[0,0],[100,0]])})]),kind:'territory'}];
 assert.equal(hover(lines,[50,3]),'国界');assert.equal(hover(lines,[50,5]),'');
});

it('queries the spatial index without scanning the full layer or touching a canvas',t=>{
 const points=source(Array.from({length:2000},(_,i)=>point('点'+i,i*100,0)));
 t.mock.method(points,'getFeatures',()=>{throw Error('Full-layer scans are forbidden during hover');});
 assert.equal(hover([{source:points,kind:'points'}],[100000,0]),'点1000');
});

it('does not trace the nearest boundary when the pointer is already inside an area',t=>{
 const geometry=new Polygon([square(0,100)]);
 t.mock.method(geometry,'getClosestPoint',()=>{throw Error('Interior hover must not trace the boundary');});
 assert.equal(hover([{source:source([new Feature({name:'行政区',geometry})]),kind:'admin'}],[50,50]),'行政区');
});

it('reads visible collection members without cloning their coordinates on each hover',t=>{
 const line=new LineString([[0,0],[100,0]]),hidden=new Polygon([square(20,80)]);
 const collection=new GeometryCollection([new GeometryCollection([line]),hidden]);
 t.mock.method(line,'clone',()=>{throw Error('Hover must not copy boundary coordinates');});
 const layers:HoverLayer[]=[{source:source([new Feature({name:'边界',geometry:collection})]),kind:'territory'}];
 assert.equal(hover(layers,[50,3]),'边界');
 assert.equal(hover(layers,[50,50]),'');
 line.setCoordinates([[0,50],[100,50]]);
 assert.equal(hover(layers,[50,3]),'');
 assert.equal(hover(layers,[50,53]),'边界');
});
