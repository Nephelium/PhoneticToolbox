import type {GeoPort} from '../../src/platform/geolinguistics.ts';
import {geoAtlasResources} from './geolinguistics-regions.ts';

// Synthetic geometry around 105E/30N, isolated from official boundaries and
// survey records. The real M19 page, OpenLayers styles and export code execute.
const coordinate=(x:number,y:number)=>[105+x,30+y];
const polygon=(x:number,y:number,w:number,h:number)=>[[coordinate(x,y),coordinate(x+w,y),coordinate(x+w,y+h),coordinate(x,y+h),coordinate(x,y)]];
const feature=(id:string,type:string,coordinates:unknown,properties:Record<string,string>={})=>({type:'Feature',id,properties,geometry:{type,coordinates}});
const collection=(features:object[])=>({type:'FeatureCollection',features});
export const paintWorld=collection([
 feature('land','Polygon',polygon(-9,-5,2,2)),
 feature('open-background','MultiLineString',[[coordinate(5,3),coordinate(8,3),coordinate(6.5,5)]]),
]);
export const paintTerritory=collection([
 feature('ordinary','Polygon',polygon(-8,-3,4,4)),
 feature('open-territory','MultiLineString',[[coordinate(0,0),coordinate(4,0),coordinate(2,4)]]),
 feature('symbols','MultiPolygon',Array.from({length:10},(_,i)=>polygon(-4+i*.9,-4.5,.3,.3)),{renderRole:'boundary-symbol'}),
 feature('dispute','MultiLineString',[[coordinate(-8,2),coordinate(-4,2)]],{renderRole:'disputed-line'}),
 feature('legacy','MultiLineString',[[coordinate(0,-2),coordinate(1,-2)],[coordinate(1.5,-2),coordinate(2.5,-2)],[coordinate(3,-2),coordinate(4,-2)]],{line:'dash'}),
 {type:'Feature',id:'mixed-collection',properties:{},geometry:{type:'GeometryCollection',geometries:[{type:'Polygon',coordinates:polygon(-8,3,3,2)},{type:'LineString',coordinates:[coordinate(-3,3),coordinate(-1,3),coordinate(-2,5)]}]}},
 // Deliberately last in the source: priority must keep dispute dashes visible
 // across this opaque display face in both actual Canvas and exported SVG.
 feature('overlap-dispute-face','Polygon',polygon(-8,1.7,4,.6)),
]);
export const paintAdmin=collection([
 feature('land','Polygon',polygon(-9,-5,2,2)),
 feature('ordinary','Polygon',polygon(-8,-3,4,4)),
 feature('collection-area','Polygon',polygon(-8,3,3,2)),
 feature('overlap-dispute-face','Polygon',polygon(-8,1.7,4,.6)),
]);
export function createGeoPort():GeoPort{
 const resources=geoAtlasResources();
 resources.admin=[{id:'province:fixture',code:'fixture',name:'合成绘制范围',level:'province',parent:'',province:'province:fixture',center:[105,30],extent:[95,24,115,37],source:'datav-qa-fixture'}];
 resources.sites=[{id:'fixture:offscreen',name:'合成调查点',lon:125,lat:10,admin:null}];resources.classification.sheets[0].rows=[];
 resources.observations.F1_P={values:['A1'],raw:{}};resources.paper.points=[{mds:[0,0,0],sharp:[],heat:[]}];
 return {load:async()=>resources,map:async name=>name==='world'?paintWorld:name==='territory'?paintTerritory:paintAdmin,matrix:async()=>new Float64Array(),useDirectory:async()=>{throw Error('Fixture does not import research data.');},download:async()=>{throw Error('Fixture does not download research data.');},save:async(name,blob)=>{(window as any).__paintExports.push({name,blob});return 'requested';}};
}
