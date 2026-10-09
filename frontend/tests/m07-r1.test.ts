import test from 'node:test';
import assert from 'node:assert/strict';
import {taskDescription,taskTime} from '../src/modules/phonation-synthesis/history.ts';
import {generationDefaults} from '../src/modules/phonation-synthesis/state.ts';

test('M07-R1 task descriptions identify the saved design and action',()=>{
 assert.equal(taskDescription({action:'generate',continuum_type:3,reverse_direction:true,generation:generationDefaults()}),'目标到源 · F0 与发声类型同时变化 · 9 步');
 assert.equal(taskDescription({action:'analyze'}),'提取 F0');
 assert.equal(taskDescription({action:'apply'}),'应用 F0 编辑');
});
test('M07-R1 uses task creation time rather than refresh time',()=>{
 const seconds=new Date(2026,9,4,18,56,0).getTime()/1000;
 assert.equal(taskTime(seconds),'2026/10/04，18:56:00');
 assert.equal(taskTime(NaN),'时间未记录');
});
