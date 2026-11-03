import Feature from 'ol/Feature.js';
import Point from 'ol/geom/Point.js';
import VectorSource from 'ol/source/Vector.js';
import {fromLonLat} from 'ol/proj.js';
import type {MapPoint} from './model.ts';

/** Stable site IDs own geometry; display-only changes never reproject it. */
export function syncMapPointFeatures(source:VectorSource,rows:MapPoint[]){
 const retained=new Set(rows.map(row=>row.id));let projected=0,created=0;
 for(const feature of source.getFeatures())if(!retained.has(String(feature.getId())))source.removeFeature(feature);
 for(const row of rows){
  let feature=source.getFeatureById(row.id);
  if(!feature){feature=new Feature({geometry:new Point(fromLonLat([row.lon,row.lat])),datum:row});feature.setId(row.id);source.addFeature(feature);projected++;created++;}
  else{const prior=feature.get('datum') as MapPoint|undefined;if(!prior||prior.lon!==row.lon||prior.lat!==row.lat){feature.setGeometry(new Point(fromLonLat([row.lon,row.lat])));projected++;}feature.set('datum',row);}
 }
 return {created,projected};
}
