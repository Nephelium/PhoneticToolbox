import {test} from 'node:test';
import assert from 'node:assert/strict';
import {inverseLayouts,inversePanels,peakNormalized} from '../src/modules/egg-analysis/inverse-plots.ts';
import {defaults,taskConfig,validate} from '../src/modules/egg-analysis/state.ts';
import type {components} from '../../contracts/generated/api';
import {visibleF0Range} from '../src/modules/egg-analysis/display.ts';
const data:components['schemas']['EggInverseData']={frequencies_hz:[0,100,4000],audio_db:[0,20,-10],inverse_db:[10,5,-20],egg_db:[-5,30,-30],relative_times_s:[-.01,0,.01],audio_values:[-2,0,4],inverse_values:[0,10,20],egg_values:[-100,0,50]};

test('M03-R3 F0 axis follows visible voiced values across sources, retaining outliers',()=>{
  const trace={times:[0,1,2,3,4,5,6],values:[2000,35,45,600,0,null,NaN]};
  assert.deepEqual(visibleF0Range([trace],1,2),[30,50]);
  assert.deepEqual(visibleF0Range([trace],3,3),[588,612]);
  assert.deepEqual(visibleF0Range([trace,{times:[1.5],values:[900]}],1,3),[0,987]);
  assert.equal(visibleF0Range([trace],4,6),undefined);
  assert.equal(visibleF0Range([],0,6),undefined);
});
test('M03-R3 all four layouts include each signal once per domain with 6/4/4/2 panels',()=>{
  inverseLayouts.forEach((layout,i)=>{const plots=inversePanels(data,layout.id);assert.equal(plots.length,[6,4,4,2][i]);for(const domain of ['spectrum','wave'])assert.deepEqual(plots.filter(p=>p.id.endsWith(domain)).flatMap(p=>p.traces.map(t=>t.label)).sort(),['EGG','IF','音频'].sort());});
});
test('M03-R3 pair overlays retain raw amplitudes, distinct styles and correct independent axes',()=>{
  for(const layout of ['audio-if','egg-if'] as const){const p=inversePanels(data,layout)[1];assert.equal(p.traces[1].right,true);assert.deepEqual(p.traces[1].values,data.inverse_values);assert(p.right![0]<0&&p.right![1]>20);assert.notEqual(p.traces[0].dash,p.traces[1].dash);assert.notEqual(p.traces[0].exportColor,p.traces[1].exportColor);}
});
test('M03-R3 three-signal normalization is explicit, isolated, and handles silence',()=>{
  const before=JSON.stringify(data),plot=inversePanels(data,'combined')[1];assert.match(plot.title,/各自峰值归一化/);assert.deepEqual(plot.traces.map(t=>t.values),[[-.5,0,1],[-1,0,.5],[0,.5,1]]);assert.equal(plot.right,undefined);assert.equal(JSON.stringify(data),before);assert.deepEqual(peakNormalized([0,0]),[0,0]);assert.equal(inversePanels(data,'combined')[0].traces[0].values,data.audio_db);
});
test('M03-R3 LP order rejects impossible 3 ms windows before submitting a job',()=>{
  for(const order of [0,132,256,1.5,NaN])assert.throws(()=>validate(taskConfig(defaults(),'inverse',order),1,44100),/LP 阶数/);
  for(const order of [null,1,50,131])assert.doesNotThrow(()=>validate(taskConfig(defaults(),'inverse',order),1,44100));
});
