import test from 'node:test';
import assert from 'node:assert/strict';
import {annotationShortcut} from '../src/modules/annotation/shortcuts.ts';
const key=(value:string)=>({key:value,ctrlKey:false,metaKey:false,altKey:false,shiftKey:false,isComposing:false});
test('annotation commands accept Control and Command with the same behavior',()=>{
  for(const modifier of ['ctrlKey','metaKey'])for(const [letter,command] of Object.entries({c:'copy',x:'cut',v:'paste',z:'undo',s:'save'})){
    assert.equal(annotationShortcut({...key(letter),[modifier]:true}),command);
  }
});
test('plain text, IME composition and other modified commands are left alone',()=>{
  assert.equal(annotationShortcut(key('s')),undefined);
  for(const modifier of ['altKey','shiftKey','isComposing'])assert.equal(annotationShortcut({...key('z'),metaKey:true,[modifier]:true}),undefined);
  assert.equal(annotationShortcut({...key('p'),metaKey:true}),undefined);
});
