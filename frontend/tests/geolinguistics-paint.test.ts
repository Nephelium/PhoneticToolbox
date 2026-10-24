import {it} from 'node:test';
import assert from 'node:assert/strict';
import {mapPaint,mapPaintPriority,contextBoundaryVisible} from '../src/modules/geolinguistics/map-paint.ts';

it('uses the active administrative layer for province lines while retaining national context',()=>{
 assert.equal(contextBoundaryVisible('省界'),false);
 for(const kind of ['国界','争议','港澳','海洋',undefined])assert.equal(contextBoundaryVisible(kind),true);
});

it('never fills open boundary or drawing lines in any layer',()=>{
 for(const layer of [0,1,2,3])for(const type of ['LineString','MultiLineString']){
  assert.equal(mapPaint(layer,type,'#2468ac').fill,'none');
 }
});

it('keeps boundary symbols distinct from administrative display areas',()=>{
 const symbol=mapPaint(1,'MultiPolygon',undefined,'boundary-symbol');
 const outline=mapPaint(1,'MultiPolygon',undefined,'administrative-display-area');
 assert.equal(symbol.fill,'#526c78');assert.equal(outline.fill,'#f8f8f1');
 assert.equal(mapPaint(1,'MultiLineString',undefined,'boundary-symbol').fill,'none');
});

it('does not turn separate legacy dash components into an invented dashed stroke',()=>{
 const legacy=mapPaint(1,'LineString',undefined,undefined,'dash');
 assert.equal(legacy.fill,'none');assert.equal(legacy.dash,undefined);
 const disputed=mapPaint(1,'MultiLineString',undefined,'disputed-line');
 assert.equal(disputed.fill,'none');assert.deepEqual(disputed.dash,[4,3]);
});

it('preserves polygon colours and drawn polygon fill',()=>{
 assert.equal(mapPaint(2,'Polygon','#2468ac').fill,'#2468ac38');
 assert.equal(mapPaint(2,'Polygon').fill,'none');
 assert.equal(mapPaint(3,'Polygon').fill,'#397da525');
 assert.deepEqual(mapPaint(3,'Polygon').dash,[6,4]);
});

it('orders territory area before actual and disputed lines, then independent symbols',()=>{
 assert.equal(mapPaintPriority('MultiPolygon','administrative-display-area'),0);
 for(const type of ['LineString','MultiLineString'])assert.equal(mapPaintPriority(type),1);
 assert.equal(mapPaintPriority('Polygon','disputed-line'),1);
 assert.equal(mapPaintPriority('MultiPolygon','boundary-symbol'),2);
});
