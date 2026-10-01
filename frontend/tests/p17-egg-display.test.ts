import {test} from 'node:test';
import assert from 'node:assert/strict';
import {visibleAmplitude,displayConfig} from '../src/modules/egg-analysis/display.ts';
import {defaults} from '../src/modules/egg-analysis/state.ts';
test('M03 visible amplitude excludes out-of-window peaks and retains quiet scale',()=>{
  assert.deepEqual(visibleAmplitude([-2,0,1,3],[.7,.002,-.004,.6],0,1),[-.00432,.00432]);
  assert.deepEqual(visibleAmplitude([0,1],[null,NaN],0,1),[-1,1]);
  assert.deepEqual(visibleAmplitude([0,1],[0,0],0,1),[-1,1]);
});

test('M03 invalid typing retains valid axes without changing the draft or snapshot',()=>{
 const previous=defaults();
 for(const invalid of [0,NaN,Infinity]){const draft={...previous,micro_width_ms:invalid};assert.equal(displayConfig(draft,previous,77,44100),previous);assert(Object.is(draft.micro_width_ms,invalid));}
 const inverted={...previous,spec_vmin:0,spec_vmax:-10};assert.equal(displayConfig(inverted,previous,77,44100),previous);
 const valid={...previous,micro_width_ms:100};assert.deepEqual(displayConfig(valid,previous,77,44100),valid);
});
