import {localized} from '../module-localization.ts';
import {tr} from '../../i18n/index.ts';
export type BoundaryLevel='province'|'city'|'district';
export type BoundaryMode=BoundaryLevel|'auto';
export const boundaryLevels:BoundaryLevel[]=['province','city','district'];
export const boundaryNames:Record<BoundaryLevel,string>=localized(()=>({province:tr("modules.geolinguistics.province_boundaries"),city:tr("modules.geolinguistics.city_boundaries"),district:tr("modules.geolinguistics.county_boundaries")}));
export interface MapExportOptions {legend:boolean;scale:boolean;names:'none'|'all'|'selected';boundaries:BoundaryLevel[]}
export function activeBoundary(level:BoundaryMode,zoom:number):BoundaryLevel{return level==='auto'?(zoom<6?'province':zoom<8?'city':'district'):level;}
export function currentBoundaries(level:BoundaryMode,zoom:number,extra:readonly BoundaryLevel[]=[]):BoundaryLevel[]{const primary=activeBoundary(level,zoom);return boundaryLevels.filter(value=>value===primary||extra.includes(value));}
/** Outline hierarchy is shared by the live map and vector/raster exports. */
export function boundaryPaint(level:BoundaryLevel){return {width:level==='province'?2.2:level==='city'?1.3:.6,color:level==='province'?'#4b6068':level==='city'?'#71858b':'#9aa9aa'};}
export function defaultMapExportOptions(boundaries:readonly BoundaryLevel[]):MapExportOptions{return {legend:false,scale:true,names:'none',boundaries:[...boundaries]};}
