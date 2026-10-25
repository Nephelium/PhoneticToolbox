import type Feature from 'ol/Feature.js';
import type Geometry from 'ol/geom/Geometry.js';
import type VectorSource from 'ol/source/Vector.js';
import {getUid} from 'ol/util.js';
import GeometryCollection from 'ol/geom/GeometryCollection.js';
import {contextBoundaryVisible,contextGeometryVisible} from './map-paint.ts';
import type {MapPoint} from './model.ts';

type Source=VectorSource<Feature<Geometry>>;
type Position=number[];
export interface HoverLayer {source:Source;kind:'points'|'sketch'|'admin'|'territory'|'background'}
function visibleContext(geometry:Geometry,role?:string):Geometry[]{
 // Read collection members without cloning every boundary coordinate on hover.
 if(geometry instanceof GeometryCollection)return geometry.getGeometriesArray().flatMap(g=>visibleContext(g,role));
 return contextGeometryVisible(geometry.getType(),role)?[geometry]:[];
}

/** Query the existing spatial indexes, without painting/readback on pointermove.
 * Sources are supplied topmost first. Survey geometry and displayed data stay
 * unchanged; pixel tolerance is converted using the current view resolution.
 */
export function mapHoverName(layers:readonly HoverLayer[],coordinate:Position,resolution:number,labels:boolean,labelWidth:(name:string)=>number,maxLabelWidth=0):string{
 if(!Number.isFinite(resolution)||resolution<=0)return '';
 const tolerance=4;
 for(const {source,kind} of layers){
  // Labels extend above their marker. Query only when labels are displayed.
  const xRadius=kind==='points'&&labels?Math.max(12,maxLabelWidth/2+6):12;
  const yRadius=kind==='points'&&labels?26:12;
  const candidates=source.getFeaturesInExtent([coordinate[0]-xRadius*resolution,coordinate[1]-yRadius*resolution,coordinate[0]+xRadius*resolution,coordinate[1]+yRadius*resolution]);
  let best:{name:string;distance:number;priority:number;order:number}|undefined;
  for(const feature of candidates){
   if(kind==='territory'&&!contextBoundaryVisible(feature.get('boundaryType')))continue;
   const datum=feature.get('datum') as MapPoint|undefined,name=datum?.name??feature.get('name');
   const geometry=feature.getGeometry();if(!name||!geometry)continue;
   let distance:number;
   if(kind==='points'&&datum){
    const center=geometry.getClosestPoint(coordinate),dx=(coordinate[0]-center[0])/resolution,dy=(coordinate[1]-center[1])/resolution;
    distance=Math.max(0,Math.hypot(dx,dy)-(datum.multi?5.5:4)-(datum.selected||datum.incomplete?0.75:0.5));
    if(labels){
     // Match the centred 12px label, -13px offset and 3px white outline.
     const labelDistance=Math.hypot(Math.max(0,Math.abs(dx)-labelWidth(name)/2-1.5),Math.max(0,Math.abs(dy-13)-7.5));
     distance=Math.min(distance,labelDistance);
    }
   }else{
    distance=Infinity;
    for(const g of kind==='territory'?visibleContext(geometry,feature.get('renderRole')):[geometry]){
     if(g.intersectsCoordinate(coordinate)){distance=0;break;}
     const closest=g.getClosestPoint(coordinate),d=Math.hypot(closest[0]-coordinate[0],closest[1]-coordinate[1])/resolution;
     distance=Math.min(distance,d);
    }
   }
   const priority=datum?.selected?2:1,order=Number(getUid(feature));
   if(distance<=tolerance&&(!best||distance<best.distance||distance===best.distance&&(priority>best.priority||priority===best.priority&&order>best.order)))best={name,distance,priority,order};
  }
  if(best)return best.name;
 }
 return '';
}
