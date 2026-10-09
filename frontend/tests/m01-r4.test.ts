import test from 'node:test';
import assert from 'node:assert/strict';
import {series,overlayPlot} from '../src/modules/parameter-display/state.ts';
import {createState,draftJson,eggDefaults} from '../src/modules/parameter-estimation/state.ts';
test('M01 joint settings persist without per-file capabilities and preserve old settings',()=>{
 const state=createState();assert.equal(state.extended.max_duration_s,1800);assert.equal(state.extended.egg,null);
 state.extended.egg=eggDefaults();state.extended.audio_channel=1;state.channelOverrides['private-file-capability']=0;
 const raw=draftJson(state);assert(!raw.includes('private-file-capability'));
 const next=createState(JSON.parse(raw));assert.equal(next.extended.egg?.storage,'aligned');assert.equal(next.extended.audio_channel,1);
});
test('M02 native cycles retain their own times and scaling uses full-window statistics',()=>{
 const table={schema_version:'m02/1' as const,sha256:'a'.repeat(64),columns:['Time_s','CQ','F0 - GCI'],kinds:['number','number','number'] as ('number'|'text')[],rows:[],streamed:true,
 tracks:{CQ:[[.011,.4],[.021,null],[.031,.6]] as [number,number|null][],'F0 - GCI':[[.016,100],[.026,120]] as [number,number|null][]},
 stats:{CQ:{count:100,mean:.5,min:.4,max:.6},'F0 - GCI':{count:100,mean:110,min:100,max:120}}};
 assert.deepEqual(series(table,'CQ',0,1),[{time:.011,value:.4},{time:.021,value:null},{time:.031,value:.6}]);
 assert.equal(series(table,'F0 - GCI',0,1)[0].time,.016);
 const chart=overlayPlot(table,['CQ','F0 - GCI'],0,1);assert.equal(chart.curves[1].axis,'right');assert.equal(chart.curves[0].mean,.5);
});
