import test from 'node:test';
import assert from 'node:assert/strict';
import {moveControl,PoseHistory,surfacePoint} from '../public/vocal-tract/controls.mjs';
import {framePitch,frameIntervals,silentAt} from '../public/vocal-tract/keyframes.js';

test('local tongue offsets have independent limits and undo snapshots',()=>{
  const baseline:any=[1,2];baseline.tongue_shape=[0,0,.1,.2];
  const meta={parameters:[],tongue_shape_limits:[[-.5,.5],[-.3,.8],[-.5,.5],[-.4,.4]]};
  const moved=moveControl(meta,baseline,'surface',.2,-.9);
  assert.deepEqual(moved.params,[1,2]);assert.deepEqual(moved.tongue_shape,[.2,-.3,.1,.2]);assert.ok(moved.limited);
  assert.deepEqual(baseline.tongue_shape,[0,0,.1,.2]);
  const history=new PoseHistory(),a={params:[1,2],tongue_shape:[0,0,0,0]},b={...a,tongue_shape:[.2,0,0,0]};
  history.begin(a);history.commit(b);a.tongue_shape[0]=1;
  assert.deepEqual(history.undo(b).tongue_shape,[0,0,0,0]);
});
test('surface handle lies on the exported native material station',()=>{
  assert.deepEqual(surfacePoint({contours:{tongue:[[0,0],[2,3],[6,7]]},tongue_surface_rib:1.5}),[4,5]);
});
test('aspiration timeline and F0 retain one continuous surrounding transition',()=>{
  const frames=[{f0:100,duration:.1},{aspiration:true,duration:.01},{f0:200,duration:.1}];
  assert.equal(frameIntervals(frames)[1].name,'ʰ');
  assert.equal(frameIntervals([{aspiration:true,duration:.01,name:'短送气'}])[0].name,'短送气');
  assert.equal(framePitch(frames,.055),150);
  assert.ok(silentAt(frames,.105/.21));assert.ok(!silentAt(frames,.12/.21));
  assert.equal(framePitch([{aspiration:true,duration:.1}],0),150);
});
