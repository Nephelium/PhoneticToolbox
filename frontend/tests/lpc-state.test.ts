import test from 'node:test';import assert from 'node:assert/strict';
import {defaults,parameters,restoreDraft,configuration,visibleRange,sameAnalysis,audioSelection,gridRangeStatus,type LpcResult} from '../src/modules/lpc-spectrum/state.ts';
const font={schema_version:'font/1' as const,zh:'Microsoft YaHei',latin:'Segoe UI',ipa:'Doulos SIL' as const,size_px:14};
test('M04 R1 annotation selection intersects the WAV without inventing out-of-range samples',()=>{
 assert.deepEqual(audioSelection(1.8,9,3.85),[1.8,3.85]);
 assert.deepEqual(audioSelection(9,1.8,3.85),[1.8,3.85]);
 assert.deepEqual(audioSelection(-1,.5,3.85),[0,.5]);
 for(const bounds of [[4,9],[1,1],[NaN,2],[-2,-1]])assert.equal(audioSelection(bounds[0],bounds[1],3.85),null);
});
test('M04 R1 blank overhang is explicit; nonblank out-of-range annotations remain invalid',()=>{
 const tiers=[{name:'phones',intervals:[{xmin:0,xmax:.2,text:'ɑ̃˥'},{xmin:.2,xmax:9,text:'  '}]}];
 assert.equal(gridRangeStatus(tiers,3.85,44100),'blank-overhang');
 tiers[0].intervals[1].text='word';assert.equal(gridRangeStatus(tiers,3.85,44100),'mismatch');
 tiers[0].intervals[1].xmax=3.85+1/44100;assert.equal(gridRangeStatus(tiers,3.85,44100),'valid');
 tiers[0].intervals[0].xmin=-.01;assert.equal(gridRangeStatus(tiers,3.85,44100),'mismatch');
});
test('M04 A06 rejects empty/nonfinite drafts and preserves scientific defaults',()=>{
 assert.deepEqual(parameters(defaults()),{order:50,freq_max_hz:8000,amp_min_db:-5,amp_max_db:35,dynamic_y:false});
 for(const value of ['', 'NaN','Infinity','1.5','201'])assert.throws(()=>parameters({...defaults(),order:value}));
 assert.throws(()=>parameters({...defaults(),amp_min_db:'35'}));assert.deepEqual(restoreDraft({order:99}),defaults());
});
test('M04 A09-A11 time window remains seconds, half-open samples bound the budget',()=>{
 assert.deepEqual(visibleRange(10,20,9.9),[9.5,10]);
 const c=configuration(defaults(),.100019,.200019,48000,480000,null,font);assert.equal(c.roi_start,.100019);assert.equal(c.roi_end,.200019);
 assert.throws(()=>configuration(defaults(),0,1.0001,48000,96000,null,font),/48,000/);
 assert.throws(()=>configuration(defaults(),0,51/48000,48000,96000,null,font),/52/);
 assert.equal(configuration(defaults(),0,1,48000,96000,null,font).roi_end,1);
});
test('M04 A15 changed parameter, source or tier marks previous results stale',()=>{
 const config=configuration(defaults(),.1,.2,48000,96000,'phones',font);
 const r={config,input_sha256:'audio',textgrid_sha256:'grid'} as LpcResult;
 assert(sameAnalysis(r,config,'audio','grid'));assert(!sameAnalysis(r,{...config,order:40},'audio','grid'));assert(!sameAnalysis(r,config,'new','grid'));assert(!sameAnalysis(r,config,'audio',null));
});
