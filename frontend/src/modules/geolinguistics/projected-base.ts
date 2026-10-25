import GeoJSON from 'ol/format/GeoJSON.js';
import Feature from 'ol/Feature.js';
import type Geometry from 'ol/geom/Geometry.js';
import {contextBoundaryVisible,contextGeometryVisible} from './map-paint.ts';

const format=new GeoJSON();
const projected=new WeakMap<object,readonly Feature<Geometry>[]>();
const projectedContext=new WeakMap<object,readonly Feature<Geometry>[]>();
const projectedOutlines=new WeakMap<object,Map<string,readonly Feature<Geometry>[]>>();

/** Shared immutable base/admin data only. Each owner must clear its own Source
 * on removal; OL objects retain mutable internal caches and must not be frozen.
 * A new dataset's JSON object gets a new entry even when all feature IDs match.
 */
export function projectedBaseFeatures(data:object):readonly Feature<Geometry>[] {
 const cached=projected.get(data);if(cached)return cached;
 const features=format.readFeatures(data,{dataProjection:'EPSG:4326',featureProjection:'EPSG:3857'});
 projected.set(data,features);return features;
}

/** Municipality/province fallback IDs occur at several scales. Prefix outline
 * IDs by scale, sharing immutable geometries instead of projecting or cloning
 * them again. Owners detach Source listeners just as for the primary faces. */
export function projectedOutlineFeatures(data:object,level:string):readonly Feature<Geometry>[] {
 let levels=projectedOutlines.get(data);if(!levels){levels=new Map();projectedOutlines.set(data,levels);}
 let features=levels.get(level);if(!features){features=projectedBaseFeatures(data).map(original=>{const feature=new Feature({geometry:original.getGeometry(),boundaryLevel:level});feature.setId(level+'|'+original.getId());return feature;});levels.set(level,features);}return features;
}

/** Skip obsolete country faces before projection, leaving the input untouched. */
export function projectedContextFeatures(data:object):readonly Feature<Geometry>[] {
 const cached=projectedContext.get(data);if(cached)return cached;
 const visible=(g:any,role?:string):any=>{
  if(!g)return null;
  if(g.type==='GeometryCollection'){const geometries=g.geometries.map((child:any)=>visible(child,role)).filter(Boolean);return geometries.length?{...g,geometries}:null;}
  return contextGeometryVisible(g.type,role)?g:null;
 };
 const features=(data as {features:any[]}).features.flatMap(f=>{
  if(!contextBoundaryVisible(f.properties?.boundaryType))return [];
  const geometry=visible(f.geometry,f.properties?.renderRole);return geometry?[{...f,geometry}]:[];
 });
 const result=format.readFeatures({type:'FeatureCollection',features},{dataProjection:'EPSG:4326',featureProjection:'EPSG:3857'});
 projectedContext.set(data,result);return result;
}
