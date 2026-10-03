import {test} from 'node:test';import assert from 'node:assert/strict';
import {createState,applyParameters,applySettings,dirty,defaults,validateSettings,matchAssociation,associateLips,isLipFile,directoryResources,resetAssociations,reconcile,snapshot,effectiveOutput,association,parameterKeys} from '../src/modules/parameter-estimation/state.ts';
import type {ResearchFile} from '../src/platform/research.ts';
const audio:ResearchFile={id:'a',name:'声调.wav',kind:'audio',size:10};
test('M01 all 80 keys and 14 defaults; empty selection is draft only',()=>{const s=createState();assert.equal(parameterKeys.length,80);assert.equal(Object.keys(defaults()).length,14);s.drawer='parameters';s.parameterDraft=[];assert(dirty(s));assert.throws(()=>applyParameters(s,[]));assert.equal(s.wave.parameters.length,80);s.drawer='';assert(!dirty(s));applyParameters(s,['pF0']);assert(dirty(s));});
test('settings apply/cancel cannot modify a previously captured job snapshot',()=>{const s=createState();s.files=[audio];const captured=snapshot(s);s.drawer='settings';s.settingsDraft.min_f0=70;assert(dirty(s));assert.equal(s.settings.min_f0,30);s.drawer='';assert(!dirty(s));applySettings(s,{...s.settings,min_f0:75});assert.equal(captured.settings.min_f0,30);assert.equal(s.settings.min_f0,75);});
test('public settings bounds and min/max are enforced before apply',()=>{for(const patch of [{max_formant:0},{min_f0:900,max_f0:100},{windowsize_ms:NaN},{only_voiced:1},{num_formants:3.5}])assert(validateSettings({...defaults(),...patch} as never));});
test('whole list, audition selection, and layer remain separate',()=>{const s=createState();s.files=[audio,{...audio,id:'b',name:'b.wav'}];s.selected='a';association(s).layer=2;assert.equal(snapshot(s).inputs.length,2);assert.equal(s.selected,'a');assert.equal(association(s).layer,2);});
test('association selection is case insensitive but rejects duplicate candidates',()=>{const grid:ResearchFile={id:'g',name:'声调.TextGrid',kind:'textgrid',size:12};assert.equal(matchAssociation(audio,[grid],'textgrid')?.id,'g');assert.throws(()=>matchAssociation(audio,[grid,{...grid,id:'g2'}],'textgrid'));assert.equal(matchAssociation(audio,[],'lip'),null);});
test('refresh removes vanished audio/association and invalidates pending preview',()=>{const s=createState();s.selected='a';s.files=[audio];association(s).textgrid={id:'g',name:'a.TextGrid',kind:'textgrid',size:12};association(s).gridHash='old';reconcile(s,[audio]);assert.equal(association(s).textgrid,null);assert.equal(association(s).gridHash,'');reconcile(s,[]);assert.equal(s.selected,'');assert.equal(s.loadVersion,1);});
test('independent projects retain draft and same-directory tracks input',()=>{const a=createState(),b=createState();applyParameters(a,['rF0']);assert.equal(b.wave.parameters.length,80);a.input={id:'input',purpose:'input',label:'语料'};a.output={id:'output',purpose:'output',label:'结果'};assert.equal(effectiveOutput(a)?.id,'input');a.sameDirectory=false;assert.equal(effectiveOutput(a)?.id,'output');});
test('native refresh retains a read association; changed server hash or missing legacy input clears it',()=>{
 const s=createState();s.selected=audio.id;const g:ResearchFile={id:'g',name:'声调.TextGrid',kind:'textgrid',size:12};
 association(s).textgrid={...g,sha256:'a'.repeat(64)};association(s).manual.textgrid=true;
 const legacy:ResearchFile={id:'old',name:'声调.xlsx',kind:'parameter',size:300};association(s).legacy=legacy;
 reconcile(s,[audio,g,legacy]);assert.equal(association(s).textgrid?.sha256,'a'.repeat(64));assert.equal(association(s).legacy?.id,'old');
 reconcile(s,[audio,{...g,sha256:'b'.repeat(64)}]);assert.equal(association(s).textgrid,null);assert.equal(association(s).legacy,null);
});
test('M01-R2 same-name lip prefers JSON, accepts case-insensitive PKL and excludes timestamps',()=>{
 const old:ResearchFile={id:'old',name:'声调.PKL',kind:'lip_pickle',size:20};
 const json:ResearchFile={id:'json',name:'声调.lip.json',kind:'lip',size:20};
 assert.equal(matchAssociation(audio,[old],'lip')?.id,'old');
 assert.equal(matchAssociation(audio,[old,json],'lip')?.id,'json');
 assert.equal(isLipFile({...old,name:'声调_timestamps.PKL'}),false);
 assert.equal(matchAssociation({...audio,name:'声调_timestamps.wav'},[{...old,name:'声调_timestamps.PKL'}],'lip'),null);
 assert.throws(()=>matchAssociation(audio,[old,{...old,id:'old2'}],'lip'));
 assert.throws(()=>matchAssociation(audio,[json,{...json,id:'json2'},old],'lip'));
});
test('M01-R2 recursive lip matching keeps distinct relative paths',()=>{
 const a={...audio,name:'甲/声调.wav'},b={...audio,id:'b',name:'乙/声调.wav'};
 const lips:ResearchFile[]=[{id:'l1',name:'甲/声调.pkl',kind:'lip_pickle',size:20},{id:'l2',name:'乙/声调.lip.json',kind:'lip',size:20}];
 assert.equal(matchAssociation(a,lips,'lip')?.id,'l1');assert.equal(matchAssociation(b,lips,'lip')?.id,'l2');
 assert.equal(matchAssociation(audio,lips,'lip'),null);
});
test('M01-R2 batch covers all audio and later manual choices survive refresh',()=>{
 const s=createState(),b={...audio,id:'b',name:'b.wav'};
 const lip:ResearchFile={id:'l1',name:'声调.pkl',kind:'lip_pickle',size:20},other:ResearchFile={id:'l2',name:'b.lip.json',kind:'lip',size:20};
 s.files=[audio,b,lip,other];s.selected=audio.id;s.marked=[audio.id];
 assert.deepEqual(associateLips(s,true),{matched:2,missing:0,ambiguous:0,skipped:0});
 association(s).lip=null;association(s).manual.lip=true;
 reconcile(s,[...s.files]);associateLips(s);
 assert.equal(association(s).lip,null);assert.equal(association(s,b.id).lip?.id,other.id);
 associateLips(s,true);assert.equal(association(s).lip?.id,lip.id);
});
test('M01-R2 missing matches preserve explicit choices and lip errors do not clear TextGrid errors',()=>{
 const s=createState(),lip:ResearchFile={id:'custom',name:'自定义.lip.json',kind:'lip',size:20};s.files=[audio,lip];
 const a=association(s,audio.id);a.lip=lip;a.manual.lip=true;a.error='TextGrid读取失败';
 assert.equal(associateLips(s,true).missing,1);assert.equal(a.lip?.id,lip.id);assert(a.manual.lip);assert.equal(a.error,'TextGrid读取失败');
 s.files.push({...lip,id:'d1',name:'声调.lip.json'},{...lip,id:'d2',name:'声调.lip.json'});
 assert.equal(associateLips(s,true).ambiguous,1);assert(a.lipError);assert.equal(a.error,'TextGrid读取失败');
});
test('M01-R3 default associations follow audio; explicit sources override only their resource kind',()=>{
 const grid:ResearchFile={id:'g',name:'声调.TextGrid',kind:'textgrid',size:10},lip:ResearchFile={id:'l',name:'声调.pkl',kind:'lip_pickle',size:10};
 const old:ResearchFile={id:'p',name:'声调.xlsx',kind:'parameter',size:10},inputs=[audio,grid,lip,old];
 assert.deepEqual(new Set(directoryResources(inputs).map(f=>f.id)),new Set(['a','g','l','p']));
 const separateGrid={...grid,id:'external-grid'},separateLip={...lip,id:'external-lip'};
 const files=directoryResources(inputs,[separateGrid,{...audio,id:'other-audio'}],[separateLip,{...grid,id:'wrong-kind'}]);
 assert.equal(matchAssociation(audio,files,'textgrid')?.id,'external-grid');assert.equal(matchAssociation(audio,files,'lip')?.id,'external-lip');
 assert(files.some(f=>f.id==='p'));assert(!files.some(f=>f.id==='other-audio'||f.id==='wrong-kind'));
 assert.equal(matchAssociation(audio,directoryResources(inputs,[],undefined),'textgrid'),null);
 assert.equal(matchAssociation(audio,directoryResources(inputs,undefined,[]),'lip'),null);
});
test('M01-R3 new directory resets its links and keeps the other directory and manual choices',()=>{
 const s=createState();s.files=[audio];const a=association(s,audio.id);
 a.textgrid={...audio,id:'grid',kind:'textgrid'};a.tiers=[{name:'word',intervals:[]}];a.gridHash='old';a.manual.textgrid=true;a.error='old error';
 a.lip={...audio,id:'lip',kind:'lip'};a.manual.lip=true;a.lipError='lip error';
 resetAssociations(s,'textgrid');assert.equal(a.textgrid,null);assert.equal(a.gridHash,'');assert.equal(a.tiers.length,0);assert(!a.manual.textgrid);assert.equal(a.error,'');
 assert.equal(a.lip?.id,'lip');assert(a.manual.lip);assert.equal(a.lipError,'lip error');
 resetAssociations(s,'lip');assert.equal(a.lip,null);assert(!a.manual.lip);assert.equal(a.lipError,'');
});
