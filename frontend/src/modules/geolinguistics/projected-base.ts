import GeoJSON from 'ol/format/GeoJSON.js';
import type Feature from 'ol/Feature.js';
import type Geometry from 'ol/geom/Geometry.js';

const format=new GeoJSON();
const projected=new WeakMap<object,readonly Feature<Geometry>[]>();

/** Shared immutable base/admin data only. Each owner must clear its own Source
 * on removal; OL objects retain mutable internal caches and must not be frozen.
 * A new dataset's JSON object gets a new entry even when all feature IDs match.
 */
export function projectedBaseFeatures(data:object):readonly Feature<Geometry>[] {
 const cached=projected.get(data);if(cached)return cached;
 const features=format.readFeatures(data,{dataProjection:'EPSG:4326',featureProjection:'EPSG:3857'});
 projected.set(data,features);return features;
}
