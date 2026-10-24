import test from 'node:test';
import assert from 'node:assert/strict';
import {AnnotationHistory} from '../src/modules/paper-reading/annotation-history.ts';
import type {PaperAnnotation} from '../src/platform/papers.ts';
const mark=(id:string,text='',page=1):PaperAnnotation=>({id,page,kind:text?'note':'highlight',color:'yellow',text,quote:'Original passage',rects:[[.1,.2,.3,.02]]});

test('annotation create/edit/delete round-trips through undo/redo without losing anchors or colours',async()=>{
 const history=new AnnotationHistory(),stored=new Map<string,PaperAnnotation>();
 const write=async(item:PaperAnnotation,remove:boolean)=>{if(remove)stored.delete(item.id);else stored.set(item.id,structuredClone(item));return true;};
 const first=mark('a','First note'),edited={...first,text:'Revised note',color:'pink' as const};
 await write(first,false);history.record('添加批注',[{before:null,after:first}]);
 await write(edited,false);history.record('编辑批注',[{before:first,after:edited}]);
 await write(edited,true);history.record('删除批注',[{before:edited,after:null}]);
 await history.travel('undo',write);assert.deepEqual(stored.get('a'),edited);
 await history.travel('undo',write);assert.deepEqual(stored.get('a'),first);
 await history.travel('undo',write);assert.equal(stored.size,0);assert.equal(history.canUndo,false);
 for(let i=0;i<3;i++)await history.travel('redo',write);
 assert.equal(stored.size,0);assert.equal(history.canRedo,false);
});

test('multi-page highlighting is one edit and uses defensive snapshots',async()=>{
 const history=new AnnotationHistory(),one=mark('a'),two=mark('b','',2),writes:unknown[]=[];
 history.record('荧光笔',[{before:null,after:one},{before:null,after:two}]);one.rects[0][0]=.9;
 const write=async(item:PaperAnnotation,remove:boolean)=>{writes.push([item.id,item.rects[0][0],remove]);return true;};
 await history.travel('undo',write);assert.deepEqual(writes,[['b',.1,true],['a',.1,true]]);assert.equal(history.canUndo,false);
 await history.travel('redo',write);assert.deepEqual(writes.slice(2),[['a',.1,false],['b',.1,false]]);
});

test('failed undo keeps its cursor; partial writes split history according to acknowledged changes',async()=>{
 const history=new AnnotationHistory(),calls:string[]=[];
 history.record('荧光笔',[{before:null,after:mark('a')},{before:null,after:mark('b')}]);
 assert.deepEqual(await history.travel('undo',async()=>false),{changed:0,complete:false});
 assert.equal(history.canUndo,true);assert.equal(history.canRedo,false);
 const result=await history.travel('undo',async(item)=>{calls.push(item.id);return item.id==='b';});
 assert.deepEqual(result,{changed:1,complete:false});assert.deepEqual(calls,['b','a']);
 await history.travel('undo',async(item)=>{assert.equal(item.id,'a');return true;});
 const redone:string[]=[];for(let i=0;i<2;i++)await history.travel('redo',async(item)=>{redone.push(item.id);return true;});
 assert.deepEqual(redone,['a','b']);
});

test('new edits invalidate redo, unchanged saves do not, and separate documents never share history',async()=>{
 const one=new AnnotationHistory(2),two=new AnnotationHistory(),a=mark('a');
 one.record('A',[{before:null,after:a}]);await one.travel('undo',async()=>true);
 one.record('unchanged',[{before:a,after:structuredClone(a)}]);assert.equal(one.canRedo,true);
 one.record('B',[{before:null,after:mark('b')}]);assert.equal(one.canRedo,false);assert.equal(two.canUndo,false);
 one.record('C',[{before:null,after:mark('c')}]);one.record('D',[{before:null,after:mark('d')}]);
 await one.travel('undo',async()=>true);await one.travel('undo',async()=>true);assert.equal(one.canUndo,false);
 one.clear();assert.equal(one.canRedo,false);
});

test('partial redo keeps acknowledged writes reversible while retaining the uncompleted remainder',async()=>{
 const history=new AnnotationHistory();history.record('two pages',[{before:null,after:mark('a')},{before:null,after:mark('b','',2)}]);
 await history.travel('undo',async()=>true);
 assert.deepEqual(await history.travel('redo',async(item)=>item.id==='a'),{changed:1,complete:false});
 assert.equal(history.canUndo,true);assert.equal(history.canRedo,true);
 await history.travel('undo',async(item,remove)=>{assert.equal(item.id,'a');assert.equal(remove,true);return true;});
 const ids:string[]=[];for(let i=0;i<2;i++)await history.travel('redo',async(item)=>{ids.push(item.id);return true;});assert.deepEqual(ids,['a','b']);
});
