import {test} from 'node:test';
import assert from 'node:assert/strict';
import {syncNumberInput} from '../src/design/numeric-input.ts';

test('committed numeric controls write corrected values and leave equal values alone',()=>{
 for(const [typed,value,expected,writes] of [['-10',0,'0',1],['26.4',26,'26',1],['',0,'0',1],['1.25',1.25,'1.25',0]] as const){
  let text:string=typed,count=0;const target={get value(){return text;},set value(v:string){text=v;count++;}};
  syncNumberInput({target} as unknown as Event,value);assert.equal(text,expected);assert.equal(count,writes);
 }
});
