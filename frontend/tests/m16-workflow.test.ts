import {test} from 'node:test';
import assert from 'node:assert/strict';
import {projectSelection,zoomTimeRange,newTask} from '../src/modules/recording/state.ts';
import type {Project,Take} from '../src/modules/recording/types.ts';

function project():Project{
 const old={id:'old',task_snapshot:newTask(4)} as Take;
 return {schema_version:'ptb-recording/1',id:'p',label:'test',revision:1,tasks:[newTask(4),newTask(5)],takes:[old],selected:{T004:'old'},recoveries:[]};
}
test('saving a new task never selects an earlier task take',()=>{
 const p=project();assert.deepEqual(projectSelection(p,'T005',''),{taskId:'T005',takeId:''});
 assert.deepEqual(projectSelection(p,'T005','old'),{taskId:'T005',takeId:''});
});
test('free recording stays free after task list save; opening may restore the latest take',()=>{
 const p=project();assert.deepEqual(projectSelection(p,'',''),{taskId:'',takeId:''});
 assert.deepEqual(projectSelection(p,'','',true),{taskId:'T004',takeId:'old'});
});
test('project update retains chosen historical take and selects a take from the current task only',()=>{
 const p=project();p.takes.push({id:'new',task_snapshot:newTask(5)} as Take);
 assert.deepEqual(projectSelection(p,'T005',''),{taskId:'T005',takeId:'new'});
 assert.deepEqual(projectSelection(p,'T004','old'),{taskId:'T004',takeId:'old'});
});
test('history from a deleted task remains selected while its versions update',()=>{
 const p=project();p.tasks=[];assert.deepEqual(projectSelection(p,'','old'),{taskId:'',takeId:'old'});
 assert.deepEqual(projectSelection(p,'',''),{taskId:'',takeId:''});
});
test('wheel zoom anchors the cursor, stays inside audio and returns to full view',()=>{
 assert.deepEqual(zoomTimeRange([0,1000],1000,.75,-120),[150,950]);
 assert.deepEqual(zoomTimeRange([150,950],1000,.75,120),[0,1000]);
 assert.deepEqual(zoomTimeRange([0,100],1000,0,120),[0,125]);
 assert.deepEqual(zoomTimeRange([900,1000],1000,1,120),[875,1000]);
 assert.deepEqual(zoomTimeRange([0,64],1000,.5,-1),[0,64]);
 assert.deepEqual(zoomTimeRange([0,16],16,.5,-1),[0,16]);
});
