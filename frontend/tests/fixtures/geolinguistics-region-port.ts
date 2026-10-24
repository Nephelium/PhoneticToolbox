import type {GeoPort} from '../../src/platform/geolinguistics.ts';
import {geoAtlasResources} from './geolinguistics-regions.ts';

// Selected only by the owned browser runner's Vite plugin. Production ports and
// the real page/map components remain unchanged. No source survey is loaded.
export function createGeoPort():GeoPort{
 return {load:async()=>geoAtlasResources(),map:async()=>({type:'FeatureCollection',features:[]}),matrix:async()=>new Float64Array(),useDirectory:async()=>{throw Error('Fixture does not import research data.');},download:async()=>{throw Error('Fixture does not download research data.');},save:async(name,blob)=>{(window as any).__regionExports.push({name,text:await blob.text()});return 'requested';}};
}
