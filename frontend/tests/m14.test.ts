import {test} from 'node:test';
import assert from 'node:assert/strict';
import {select,move,swap,merge,resolve} from '../src/modules/phonology-induction/state.ts';
test('M14 Ctrl/Shift selection retains empty-final identity',()=>{assert.deepEqual(select(['','a','i'],[''],'','i',false,true).selected,['','a','i']);assert.deepEqual(select(['a','i'],['a'],'a','i',true,false).selected,['a','i']);});
test('M14 group drag preserves selected relative order; tone drag swaps',()=>{assert.deepEqual(move(['p','t','k','m'],['t','m'],'p'),['t','m','p','k']);assert.deepEqual(swap(['55','35','214'],'55','214'),['214','35','55']);});
test('M14 merge chain and empty final do not mutate source text',()=>{const s={tone_map:{},tone_order:[],initial_order:['p','t','k'],final_order:['','a'],initial_map:{},final_map:{}};merge(s,'initial','p','t');merge(s,'initial','t','k');merge(s,'final','','a');assert.equal(resolve('p',s.initial_map),'k');assert.equal(resolve('',s.final_map),'a');assert.throws(()=>resolve('p',{p:'t',t:'p'}));});
