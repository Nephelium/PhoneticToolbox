import test from 'node:test';
import assert from 'node:assert/strict';
import type {ReconstructionConfig} from '../src/platform/research.ts';
import {resultConfigMatches} from '../src/modules/spectrogram-to-audio/result-config.ts';
import {setLocale,tr} from '../src/i18n/index.ts';

const defaults:ReconstructionConfig={mode:'image',invert_image:false,vocoder:'none',algorithm:'griffin_lim',reconstruction_profile:'legacy',inference_device:'auto',phase_initialization:'auto',frequency_axis:'linear',auto_parameters:false,channel:0,audio_fft:1024,dynamic_range:60,strokes:[],time_start:0,time_end:1,freq_start:0,freq_end:11025,min_db:-30,max_db:0,win_length_ms:10,n_iter:32,target_sr:0,seed:0,corners:null};
test('result hint accepts native optional stroke defaults and object key order without mutating snapshots',()=>{
 const request:ReconstructionConfig={...defaults,mode:'image_draw',strokes:[{color:0,size:12,size_unit:'px',opacity:1,points:[{x:.2,y:.65},{x:.65,y:.3}]}]};
 const native={...request,strokes:[{points:[{y:.65,x:.2},{y:.3,x:.65}],opacity:1,frequency_fraction:null,canvas_height:null,canvas_width:null,size_unit:'px',size:12,color:0}]} as ReconstructionConfig;
 const before=[JSON.stringify(request),JSON.stringify(native)];
 assert.equal(resultConfigMatches(request,native,defaults),true);
 assert.deepEqual([JSON.stringify(request),JSON.stringify(native)],before);
 const legacy={...request,strokes:[{color:0,size:12,opacity:1,points:[{x:.2,y:.65}]}]};
 assert.equal(resultConfigMatches(legacy,{...legacy,strokes:[{...legacy.strokes[0],size_unit:'legacy'}]},defaults),true);
});
test('result hint still detects exact coordinate, stroke order, canvas and calibration changes',()=>{
 const config:ReconstructionConfig={...defaults,mode:'audio_draw',strokes:[{color:0,size:12,size_unit:'px',canvas_width:43,canvas_height:513,frequency_fraction:.625,opacity:1,points:[{x:.2,y:.65}]},{color:255,size:20,opacity:.5,points:[{x:.8,y:.1}]}]};
 for(const mutate of [
  (value:ReconstructionConfig)=>{value.strokes![0].points[0].x+=1e-14;},
  (value:ReconstructionConfig)=>{value.strokes!.reverse();},
  (value:ReconstructionConfig)=>{value.strokes![0].canvas_height=514;},
  (value:ReconstructionConfig)=>{value.strokes![0].opacity=.9;},
  (value:ReconstructionConfig)=>{value.time_end=1.00001;},
 ]){const other=structuredClone(config);mutate(other);assert.equal(resultConfigMatches(config,other,defaults),false);}
});
test('drawing labels use complete translated count and dimensions with unchanged values',async()=>{
 try{
  await setLocale('en');assert.equal(tr('modules.spectrogram-to-audio.strokeCount',{count:1}),'Strokes: 1');assert.equal(tr('modules.spectrogram-to-audio.imageBrushHint',{width:658,height:520}),'Diameter uses original-image pixels (658 × 520 px). ');
  await setLocale('ja');assert.equal(tr('modules.spectrogram-to-audio.strokeCount',{count:1}),'筆跡数：1');
 }finally{await setLocale('zh-CN');}
});
