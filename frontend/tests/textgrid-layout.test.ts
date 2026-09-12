import {test} from 'node:test';
import assert from 'node:assert/strict';
import {intervalLayout} from '../src/components/interval-layout.ts';
test('M01 TextGrid intervals use original proportions, including short and empty spans',()=>{
 const source=[{xmin:0,xmax:.1,text:''},{xmin:.1,xmax:.101,text:'IPA ɕ'},{xmin:.101,xmax:1,text:'long'}];
 const layout=intervalLayout(source,0,1);assert.equal(layout[0].width,10);assert.equal(layout[1].left,10);assert(Math.abs(layout[1].width-.1)<1e-12);assert.equal(layout[2].xmin,.101);assert.equal(source[0].text,'');
});
test('M01 TextGrid view clips display only, retaining full selection boundaries',()=>{
 const layout=intervalLayout([{xmin:0,xmax:1,text:'first'},{xmin:1,xmax:3,text:'second'},{xmin:3,xmax:4,text:'outside'}],.5,2.5);
 assert.deepEqual(layout.map(i=>[i.left,i.width,i.xmin,i.xmax]),[[0,25,0,1],[25,75,1,3]]);
 assert.deepEqual(intervalLayout([],1,1),[]);
});
