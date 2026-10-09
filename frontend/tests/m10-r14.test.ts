import test from 'node:test';
import assert from 'node:assert/strict';
import {bladeSurfacePoint,moveControl} from '../public/vocal-tract/controls.mjs';
import {oralSectionBoundary} from '../public/vocal-tract/geometry.mjs';

test('M10-R14 blade attachment follows the actual clipped tissue segment',()=>{
  const contour=Array.from({length:37},(_,i)=>[i,i*i]);
  const s={contours:{tongue:contour},tongue_blade_rib:17.25,limited:Array(19).fill(100)};
  assert.deepEqual(bladeSurfacePoint(s),[17.25,297.75]);
  contour[18]=[18,300];assert.deepEqual(bladeSurfacePoint(s),[17.25,291.75]);
  const meta={parameters:[{name:'TBX',min:-5,max:5},{name:'TBY',min:-5,max:5}]};
  assert.deepEqual(moveControl(meta,[0,0],'blade',.25,-.5).params,[.4,-.8]);
  assert.deepEqual(moveControl(meta,[0,0],'blade',0,0).params,[0,0]);
});
test('M10-R14 sagittal cavity uses cover/lip surfaces without tube interpolation',()=>{
  const c={upper_cover:[[0,0],[1,1]],lower_cover:[[0,-2],[1,-2]],upper_lip:[[1,1],[2,1],[3,1],[4,1],[5,1],[6,20]],lower_lip:[[1,-2],[2,-2],[3,-2],[4,-2],[5,-2],[6,-20]]};
  const copy=structuredClone(c),outline=oralSectionBoundary({contours:c});
  assert.equal(outline.length,13);assert.ok(outline.every(p=>Math.abs(p[1])<=2));
  assert.deepEqual(c,copy);
});
