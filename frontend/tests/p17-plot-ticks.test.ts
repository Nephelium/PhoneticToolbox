import test from 'node:test';import assert from 'node:assert/strict';
import {plotTickLabel} from '../src/platform/plotTicks.ts';
test('P17 faint positive and negative axes retain nonzero ticks',()=>{
 assert.equal(plotTickLabel(.00004,.00002),'4.00e-5');
 assert.equal(plotTickLabel(-.00002,.00002),'-2.00e-5');
 assert.equal(plotTickLabel(0,.00002),'0');
});
test('P17 microscopic time labels remain distinct without changing coordinates',()=>{
 const values=[44.00403,44.00404,44.00405];
 assert.equal(new Set(values.map(v=>plotTickLabel(v,.00001))).size,3);
 assert.equal(plotTickLabel(2500,1250),'2500');
 assert.equal(plotTickLabel(Number.NaN),'');
});
