import test from 'node:test';
import assert from 'node:assert/strict';
import {editStopTiming,initialStopTiming} from '../public/vocal-tract/stop-editor.js';

test('VOT and aspiration share one post-release interval, including signed edits',()=>{
 const original=initialStopTiming(.2);
 const negative=editStopTiming(original,'vot',-30);
 assert.equal(negative.aspiration,0);assert.equal(negative.vot,-.03);
 assert.equal(negative.closure+negative.release+negative.aspiration,.068);
 const positive=editStopTiming(negative,'vot',50);
 assert.equal(positive.aspiration,.042);assert.equal(Math.round((positive.closure+positive.release+positive.aspiration)*1e6),110000);
 const aspiration=editStopTiming(positive,'aspiration',70);
 assert.equal(Math.round(aspiration.vot*1e6),78000);
 assert.equal(Math.round(editStopTiming(aspiration,'release',20).aspiration*1e6),58000);
 assert.equal(editStopTiming(negative,'vot',0).aspiration,0);
 assert.deepEqual(original,initialStopTiming(.2));
});
test('invalid edits cannot silently clamp VOT; fractional values retain precision',()=>{
 const t={closure:.06,release:.008,aspiration:0,vot:-.04};
 assert.throws(()=>editStopTiming(t,'closure',30),/前浊/);
 for(const value of ['',NaN,Infinity])assert.throws(()=>editStopTiming(t,'vot',value),/有效/);
 assert.equal(editStopTiming(t,'vot',-25.125).vot,-.025125);
 assert.throws(()=>editStopTiming(t,'closure',10),/前浊/);
});
