import test from 'node:test';import assert from 'node:assert/strict';
import {defaults,parameters,restoreDraft,configuration,visibleRange,sameAnalysis,type LpcResult} from '../src/modules/lpc-spectrum/state.ts';
const font={schema_version:'font/1' as const,zh:'Microsoft YaHei',latin:'Segoe UI',ipa:'Doulos SIL' as const,size_px:14};
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
