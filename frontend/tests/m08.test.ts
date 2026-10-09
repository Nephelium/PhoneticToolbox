import test from 'node:test';import assert from 'node:assert/strict';
import {validate,importF0,editCurve,renamePlan,visible,finite} from '../src/modules/pitch-manipulation/state.ts';
test('M08 four-mode counts including 216-way manual example',()=>{assert.equal(validate([0,.5,1].map(time=>({time,freqs:[100,120,140,160,180,200],mode:'full'})),0,1),216);assert.equal(validate([{time:0,freqs:[200,100],mode:'reverse'},{time:1,freqs:[100,200],mode:'order'}],0,1),2);assert.equal(validate([{time:0,freqs:[200,100],mode:'constant'},{time:1,freqs:[100,200],mode:'full'}],0,1),2);});
test('M08-R2 explicit numeric commit rejects empty and incomplete input without accepting zero',()=>{
 for(const value of ['', '  ', '-', '.', '1e', 'NaN', 'Infinity', null, undefined])assert.throws(()=>finite(value,'时间'));
 assert.equal(finite('0.3','时间'),.3);assert.equal(finite('-2.5','音高偏移'),-2.5);assert.equal(finite('0.','时间'),0);
});
test('M08 rejects invalid diagonal, out of view and duplicate knots',()=>{assert.throws(()=>validate([{time:0,freqs:[100],mode:'order'},{time:1,freqs:[100,200],mode:'reverse'}],0,1));assert.throws(()=>validate([{time:0,freqs:[100],mode:'constant'},{time:0,freqs:[200],mode:'constant'}],0,1));assert.throws(()=>validate([{time:0,freqs:[100],mode:'constant'},{time:2,freqs:[200],mode:'constant'}],0,1));});
test('M08 import ignores time column and retains view/gaps',()=>{assert.deepEqual(importF0([0,1,2,3,4],[0,100,100,100,0],0,4,'9 200\n10 300'),[0,200,250,300,0]);assert.throws(()=>importF0([0,1,2],[100,0,100],0,2,'100'));assert.throws(()=>importF0([0],[100],0,1,'NaN'));});
test('M08 stroke and restore retain unvoiced frames',()=>{const original=[100,0,120,130];const next=editCurve(original,[...original],3,200,{index:0,value:140});assert.deepEqual(next.curve,[140,0,180,200]);assert.deepEqual(editCurve(original,next.curve,0,0,{index:3,value:200},true).curve,original);assert.deepEqual(original,[100,0,120,130]);});
test('M08 rename has explicit scope and no collision/path escape',()=>{const files=[{id:'a',name:'x_1.wav'},{id:'b',name:'x_2.wav'}];assert.deepEqual(renamePlan(files,'new_',[]),[{id:'a',name:'new_1.wav'},{id:'b',name:'new_2.wav'}]);assert.throws(()=>renamePlan(files,'../',[]));assert.throws(()=>renamePlan(files,'new_',['NEW_1.wav']));});
test('M08 current view clamped independently of full duration',()=>{assert.deepEqual(visible(10,2,8),[5,10]);assert.deepEqual(visible(10,1,8),[0,10]);});

test('M08 reverse-direction stroke keeps the pointer endpoint and original mask',()=>{const original=[100,0,120,130];assert.deepEqual(editCurve(original,[...original],0,140,{index:3,value:200}).curve,[140,0,180,200]);});

test('M08-R1 rename keeps WAV extension for a single or repeated generated name',()=>{
 assert.deepEqual(renamePlan([{id:'a',name:'old.wav'}],'新音频',[]),[{id:'a',name:'新音频.wav'}]);
 assert.deepEqual(renamePlan([{id:'a',name:'old.wav'},{id:'b',name:'old.wav'}],'new',[]),[{id:'a',name:'new.wav'},{id:'b',name:'new (2).wav'}]);
 assert.throws(()=>renamePlan([{id:'a',name:'old.wav'}],'new',['new.wav']));
});
