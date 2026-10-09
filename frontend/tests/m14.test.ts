import {test} from 'node:test';
import assert from 'node:assert/strict';
import {select,move,swap,merge,resolve,undoMerge,entries,groupEntries,orderedEntries,searchEntries,pairKey} from '../src/modules/phonology-induction/state.ts';
test('M14 Ctrl/Shift selection retains empty-final identity',()=>{assert.deepEqual(select(['','a','i'],[''],'','i',false,true).selected,['','a','i']);assert.deepEqual(select(['a','i'],['a'],'a','i',true,false).selected,['a','i']);});
test('M14 group drag preserves selected relative order; tone drag swaps',()=>{assert.deepEqual(move(['p','t','k','m'],['t','m'],'p'),['t','m','p','k']);assert.deepEqual(swap(['55','35','214'],'55','214'),['214','35','55']);});
test('M14 merge chain and empty final do not mutate source text',()=>{const s={tone_map:{},tone_order:[],initial_order:['p','t','k'],final_order:['','a'],initial_map:{},final_map:{}};merge(s,'initial','p','t');merge(s,'initial','t','k');merge(s,'final','','a');assert.equal(resolve('p',s.initial_map),'k');assert.equal(resolve('',s.final_map),'a');assert.throws(()=>resolve('p',{p:'t',t:'p'}));});
test('M14 R1 preview preserves merged tone order, duplicates and both document directions',()=>{
 const settings={tone_map:{'35':'阳','55':'阴'},tone_order:['35','55'],initial_order:['m','ts'],final_order:['a','ã'],initial_map:{'p':'m'},final_map:{}};
 const rows=[{character:'妈',ipa:'ma55',note:'甲',initial:'m',final:'a',tone_value:'55'},{character:'麻',ipa:'pa35',note:'乙',initial:'p',final:'a',tone_value:'35'},{character:'妈',ipa:'ma55',note:'甲',initial:'m',final:'a',tone_value:'55'},{character:'鼻',ipa:'tsã35',note:'丙',initial:'ts',final:'ã',tone_value:'35'}];
 const original=JSON.stringify(rows),mapped=entries(rows,settings),grouped=groupEntries(mapped,settings);
 assert.deepEqual(grouped.get(pairKey('m','a'))!.map(r=>r.index),[1,0,2]);
 assert.deepEqual(orderedEntries(grouped,settings,'initial').map(r=>r.index),[1,0,2,3]);
 assert.deepEqual(orderedEntries(grouped,settings,'final').map(r=>r.index),[1,0,2,3]);
 assert.deepEqual(searchEntries(mapped,'ã','ipa'),[3]);assert.deepEqual(searchEntries(mapped,'ma35','ipa'),[1]);assert.deepEqual(searchEntries(mapped,'妈','character'),[0,2]);assert.equal(JSON.stringify(rows),original);
});
test('M14 R1 search covers records beyond 200 and undone chains remain resolvable',()=>{
 const settings={tone_map:{'55':'阴'},tone_order:['55'],initial_order:['p','t','k'],final_order:['a'],initial_map:{},final_map:{}};
 merge(settings,'initial','p','t');merge(settings,'initial','t','k');undoMerge(settings,'initial','t',['p','t','k']);assert.equal(resolve('p',settings.initial_map),'t');assert.deepEqual(settings.initial_order,['t','k']);
 const mapped=entries(Array.from({length:320},(_,i)=>({character:i===319?'罕':'妈',ipa:'pa55',note:'',initial:'p',final:'a',tone_value:'55'})),settings);
 assert.deepEqual(searchEntries(mapped,'罕','all'),[319]);assert.equal(mapped.length,320);
});
