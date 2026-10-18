import test from 'node:test';import assert from 'node:assert/strict';
import {AnnotationHistory,type AnnotationView} from '../src/modules/annotation/history.ts';
import {MemoryAssetDatabase} from '../src/platform/local-assets.ts';
import {createEditor} from '../src/modules/annotation/editor.mjs';
import {serializeGrid} from '../src/modules/annotation/format.ts';
const grid=(text:string)=>serializeGrid({xmin:0,xmax:2,tiers:[{name:'words',intervals:[{xmin:0,xmax:2,text}]}]});
const view:AnnotationView={offset:.5,zoom:2,channel:0,word:'words',phone:'phones',lab:['a1','b2'],labName:'词表',sequence:true,index:1};
test('M12 latest confirmed edit and view survive a new history instance; original is recoverable',async()=>{
 const db=new MemoryAssetDatabase(),a=new AnnotationHistory('owner','project',db),s=await a.open('audio','sha',grid('原始'));
 await a.save(s,grid('编辑A'),'auto',view);await a.save(s,grid('编辑B'),'auto',view);
 const b=new AnnotationHistory('owner','project',db),restored=await b.open('audio','sha',grid('原始'));
 assert.equal(b.head(restored).text,grid('编辑B'));assert.deepEqual(restored.value.view,view);assert.equal(restored.value.versions[0].text,grid('原始'));
 await b.save(restored,restored.value.versions[0].text,'restored',view);assert.equal(restored.value.versions.at(-1)!.kind,'restored');assert.equal(restored.value.versions.length,4);
});
test('M12 existing autosave is adopted separately from original and identical saves are deduplicated',async()=>{
 const a=new AnnotationHistory('owner','project',new MemoryAssetDatabase()),s=await a.open('audio','sha',grid('原始'),grid('旧自动保存'));
 assert.equal(a.head(s).kind,'recovered');await a.save(s,grid('旧自动保存'),'auto',view);await a.save(s,grid('旧自动保存'),'auto',view);assert.equal(s.value.versions.length,2);
});
test('M12 history separates account, project, file identity and changed audio bytes',async()=>{
 const db=new MemoryAssetDatabase(),a=new AnnotationHistory('owner','project',db),s=await a.open('audio','sha',grid('原始'));await a.save(s,grid('私有编辑'),'auto',view);
 for(const [owner,project,id,sha]of [['other','project','audio','sha'],['owner','other','audio','sha'],['owner','project','other','sha'],['owner','project','audio','changed']]){
  const b=new AnnotationHistory(owner,project,db),other=await b.open(id,sha,grid('原始'));assert.equal(b.head(other).text,grid('原始'));
 }
});
test('M12 stale history and failed writes retain the previous confirmed state',async()=>{
 class Failing extends MemoryAssetDatabase{fail=false;override async compare(k:string,e:unknown,v:unknown){if(this.fail)throw Error('quota');await super.compare(k,e,v);}}
 const db=new Failing(),a=new AnnotationHistory('owner','project',db),b=new AnnotationHistory('owner','project',db);
 const s=await a.open('audio','sha',grid('原始')),stale=await b.open('audio','sha',grid('原始'));await a.save(s,grid('最新'),'auto',view);
 await assert.rejects(b.save(stale,grid('陈旧'),'auto',view),/另一窗口/);assert.equal(b.head(stale).text,grid('原始'));
 db.fail=true;await assert.rejects(a.save(s,grid('空间不足'),'auto',view),/空间不足/);assert.equal(a.head(s).text,grid('最新'));
});
test('M12 history retention preserves original plus the most recent fifty saved contents',async()=>{
 const a=new AnnotationHistory('owner','project',new MemoryAssetDatabase()),s=await a.open('audio','sha',grid('原始'));
 for(let i=1;i<=55;i++)await a.save(s,grid(String(i)),'auto',view);
 assert.equal(s.value.versions.length,51);assert.equal(s.value.versions[0].text,grid('原始'));assert.equal(s.value.versions[1].text,grid('6'));assert.equal(a.head(s).text,grid('55'));
});
function editor(){const e=createEditor();e.state.textgrid={xmin:0,xmax:2,tiers:[{name:'words',intervals:[{xmin:0,xmax:1,text:'a'},{xmin:1,xmax:2,text:'b'}]},{name:'phones',intervals:[{xmin:0,xmax:.5,text:'a'},{xmin:.5,xmax:1,text:'i'},{xmin:1,xmax:2,text:'b'}]}]};return e;}
test('M12 deleting a word edge also removes the matching phone edge, keeps internal phone edges and undoes once',()=>{
 const e=editor(),before=serializeGrid(e.state.textgrid!);e.state.selectedBoundary={tier:'words',time:1};e.deleteSelectedBoundary();
 assert.deepEqual(e.wordTier()!.intervals,[{xmin:0,xmax:2,text:'a b'}]);assert.deepEqual(e.phoneTier()!.intervals,[{xmin:0,xmax:.5,text:'a'},{xmin:.5,xmax:2,text:'i b'}]);assert.equal(e.state.undoStack.length,1);e.undo();assert.equal(serializeGrid(e.state.textgrid!),before);
});
test('M12 deleting an outer domain gives recovery guidance without changing data or undo',()=>{
 const e=editor(),before=serializeGrid(e.state.textgrid!);for(const time of [0,2]){e.state.selectedBoundary={tier:'words',time};assert.throws(()=>e.deleteSelectedBoundary(),/内部边界/);assert.equal(e.state.undoStack.length,0);assert.equal(serializeGrid(e.state.textgrid!),before);}
});
test('M12 deletion clips crossing phones, preserves the outside, while cut still rejects partial phones',()=>{
 const e=editor();e.phoneTier()!.intervals=[{xmin:0,xmax:1.5,text:'跨界'},{xmin:1.5,xmax:2,text:'余下'}];e.state.selected={tier:'words',index:0};const before=serializeGrid(e.state.textgrid!);
 assert.throws(()=>e.copyAnnotation(true),/跨越/);e.deleteAnnotation();assert.deepEqual(e.phoneTier()!.intervals,[{xmin:0,xmax:1,text:''},{xmin:1,xmax:1.5,text:'跨界'},{xmin:1.5,xmax:2,text:'余下'}]);e.undo();assert.equal(serializeGrid(e.state.textgrid!),before);
});
