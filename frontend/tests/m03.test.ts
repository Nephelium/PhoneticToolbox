import {test} from 'node:test';import assert from 'node:assert/strict';
import {defaults,taskConfig,validate,signature} from '../src/modules/egg-analysis/state.ts';
import {batchDefaults} from '../src/modules/egg-analysis/state.ts';
import {validateParameters} from '../src/modules/egg-analysis/state.ts';
import {rangeAfterGesture,microAfterGesture} from '../src/modules/egg-analysis/navigation.ts';
test('single display switches and batch defaults follow independent V2 dialogs',()=>{const page=defaults();assert.equal(page.keep_praat_f0,false);assert.equal(page.keep_gci_f0,false);const batch=batchDefaults();assert.equal(batch.keep_praat_f0,true);assert.equal(batch.keep_gci_f0,true);assert.equal(batch.generate_images,false);assert.equal(batch.roi_end,null);assert.equal(batch.highpass_cutoff,25);});
test('plot gestures preserve duration at file boundaries and clamp micro view',()=>{assert.deepEqual(rangeAfterGesture(.2,.7,1,'pan',.4),[0,.49999999999999994]);const [a,b]=rangeAfterGesture(.2,.7,1,'pan',-1);assert.equal(b,1);assert.ok(Math.abs(a-.5)<1e-12);assert.deepEqual(rangeAfterGesture(.2,.7,1,'zoom',10),[0,1]);assert.deepEqual(microAfterGesture(.1,50,1,'pan',200),{center:0,width:50});assert.deepEqual(microAfterGesture(.1,5000,1,'zoom',1.1),{center:.1,width:5000});assert.deepEqual(microAfterGesture(.1,5,1,'zoom',.9),{center:.1,width:5});});
test('M03 R2 keeps V2 event defaults and uses the requested 2000 Hz lowpass',()=>{const c=defaults();assert.equal(c.gci_method,'slope');assert.equal(c.goi_method,'scale');assert.equal(c.highpass_cutoff,25);assert.equal(c.lowpass_cutoff,2000);assert.equal(batchDefaults().lowpass_cutoff,2000);assert.equal(c.micro_width_ms,50);});
test('batch snapshot removes ROI, raw and micro settings without mutating page',()=>{const c={...defaults(),roi_start:.1,roi_end:.2,signal_mode:'raw' as const,micro_center:.15,micro_width_ms:200};const b=taskConfig(c,'batch');assert.equal(b.roi_start,0);assert.equal(b.roi_end,null);assert.equal(b.signal_mode,'filtered');assert.equal(b.micro_center,null);assert.equal(c.roi_start,.1);assert.equal(c.signal_mode,'raw');});
test('inverse order never leaks into ordinary exports',()=>{assert.equal(taskConfig(defaults(),'single',12).lp_order,null);assert.equal(taskConfig(defaults(),'inverse',12).lp_order,12);});
test('stale signature detects every scientific setting and ignores font snapshot',()=>{const c=defaults();for(const [key,value] of Object.entries(c)){if(key==='font')continue;assert.notEqual(signature(c),signature({...c,[key]:typeof value==='number'?value+1:typeof value==='boolean'?!value:'changed'}));}assert.equal(signature(c),signature({...c,font:{schema_version:'font/1',ipa:'Doulos SIL',zh:'SimSun',latin:'Arial',size_px:12}}));});
test('sample-rate and selection budgets reject invalid views',()=>{assert.doesNotThrow(()=>validate(defaults(),.8,44100));assert.throws(()=>validate({...defaults(),roi_end:NaN},.8,44100));assert.throws(()=>validate(defaults(),121,44100));assert.throws(()=>validate({...defaults(),lowpass_cutoff:4000},.8,8000));assert.throws(()=>validate({...defaults(),roi_start:.5,roi_end:.4},.8,44100));});

test('micro numeric input shares wheel limits and rejects non-finite widths',()=>{for(const width of [5,5000])assert.doesNotThrow(()=>validate({...defaults(),micro_width_ms:width},6.4,44100));for(const width of [4.99,5000.01,NaN,Infinity])assert.throws(()=>validate({...defaults(),micro_width_ms:width},6.4,44100));});

import {inverseAudioFiles} from '../src/modules/egg-analysis/state.ts';
test('IF player roles follow explicit file names despite sorted manifest order',()=>{const files=[{name:'egg_IF.wav',id:'if'},{name:'egg.ptb.json',id:'meta'},{name:'egg_ORIG.wav',id:'original'}];assert.deepEqual(inverseAudioFiles(files).map(f=>f.id),['original','if']);assert.equal(files[0].id,'if');});

test('M03/1 batch parameter errors reject blank and non-finite inputs before submission',()=>{
  for(const value of ['',NaN,Infinity,-.01,1.01])assert.throws(()=>validateParameters({...batchDefaults(),silence_threshold:value as number}),/静音阈值/);
  for(const value of ['',0,2000,48000])assert.throws(()=>validateParameters({...batchDefaults(),highpass_cutoff:value as number}),/高通/);
  for(const value of [0,1])assert.doesNotThrow(()=>validateParameters({...batchDefaults(),silence_threshold:value}));
  assert.doesNotThrow(()=>validateParameters({...batchDefaults(),highpass_cutoff:.5}));
});
test('shared parameter limits preserve contract endpoints without pretending to know file sample rate',()=>{
  for(const changes of [{peak_prominence:0,valley_prominence:10},{spec_window_ms:5},{spec_window_ms:50},{spec_vmin:-160,spec_vmax:20},{highpass_cutoff:25,lowpass_cutoff:47000}])assert.doesNotThrow(()=>validateParameters({...batchDefaults(),...changes}));
  for(const changes of [{peak_prominence:10.1},{valley_prominence:-1},{spec_window_ms:4},{spec_window_ms:51},{spec_vmin:-161},{spec_vmax:21},{spec_vmin:-10,spec_vmax:-10},{lowpass_cutoff:48000}])assert.throws(()=>validateParameters({...batchDefaults(),...changes}));
  assert.throws(()=>validate({...defaults(),lowpass_cutoff:47000},.8,44100),/采样率/);
});
