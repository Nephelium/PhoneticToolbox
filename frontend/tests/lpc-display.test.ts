import test from 'node:test';import assert from 'node:assert/strict';
import {axisLimits} from '../src/modules/lpc-spectrum/display.ts';
import {spectrumSvg} from '../src/modules/lpc-spectrum/export-scene.ts';
import {sameAnalysis,type LpcResult} from '../src/modules/lpc-spectrum/state.ts';

const fixture=()=>({config:{order:50,freq_max_hz:8000,amp_min_db:-5,amp_max_db:35,dynamic_y:false,roi_start:.1,roi_end:.2,tier_name:'phones',font:{zh:'SimSun',latin:'Times New Roman',size_px:12}},input_name:'sound.wav',input_sha256:'audio',textgrid_sha256:'grid',label:'ɑ̃<&',selection:{start_s:.1,end_s:.2},spectrum:{frequencies_hz:[0,2000,8000,9000],magnitude_db:[-12,42,51,100],amp_min_db:-5,amp_max_db:35}} as LpcResult);

test('M04 R2 dynamic bounds include the frequency endpoint and exclude outside values; fixed limits round-trip',()=>{
 const r=fixture(),before=JSON.stringify(r);
 assert.deepEqual(axisLimits(r,true,'',''),[-17,56]);
 assert.deepEqual(axisLimits(r,false,'-5','35'),[-5,35]);
 assert.deepEqual(axisLimits(r,false,'-20','80'),[-20,80]);
 assert.equal(JSON.stringify(r),before);
 for(const [a,b] of [['',35],[-5,''],[NaN,35],[5,5],[40,35],[-201,35],[-5,101]])assert.throws(()=>axisLimits(r,false,a,b));
});
test('M04 R2 display edits do not mark the spectrum stale; analysis and source edits still do',()=>{
 const r=fixture();assert(sameAnalysis(r,{...r.config,dynamic_y:true,amp_min_db:-20,amp_max_db:80},'audio','grid'));
 for(const c of [{order:40},{freq_max_hz:4000},{roi_end:.3},{tier_name:'words'}])assert(!sameAnalysis(r,{...r.config,...c},'audio','grid'));
 assert(!sameAnalysis(r,r.config,'other','grid'));
});
test('M04 R2 export keeps full frequency range, selected bounds, IPA label and immutable task values',()=>{
 const r=fixture(),before=JSON.stringify(r),fixed=spectrumSvg(r,[-5,35]),dynamic=spectrumSvg(r,[-17,56]);
 assert.equal(fixed.width,768);assert.equal(fixed.height,432);
 for(const s of [fixed.text,dynamic.text]){assert(s.includes('8000'));assert(s.includes('ɑ̃&lt;&amp;'));assert(s.includes('stroke="black"'));assert(s.includes('fill="white"'));assert(s.includes('PTB-Doulos'));}
 assert(fixed.text.includes('y-axis -5 to 35 dB'));assert(dynamic.text.includes('y-axis -17 to 56 dB'));assert.notEqual(fixed.text,dynamic.text);assert.equal(JSON.stringify(r),before);
});
