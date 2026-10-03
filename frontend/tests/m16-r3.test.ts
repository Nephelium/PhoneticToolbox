import test from 'node:test';
import assert from 'node:assert/strict';
import {channelName} from '../src/modules/recording/state.ts';
import {csvRows,autoMapping,importRows,exampleCSV} from '../src/modules/recording/task-import.ts';

test('M16 downloaded example imports with Chinese headings and expands repeats',()=>{
 const rows=csvRows(exampleCSV.replace(/^\uFEFF/,'')),result=importRows(rows,autoMapping(rows[0]));
 assert.deepEqual(result.errors,[]);assert.equal(result.tasks.length,3);
 assert.equal(result.tasks[0].prompt,'请读：春天来了');
 assert.deepEqual(result.tasks.map(t=>t.id),['T001','T002_1','T002_2']);
});
test('M16 channel names retain physical order and explicit EGG roles',()=>{
 assert.equal(channelName(0,['microphone','microphone']),'左声道');
 assert.equal(channelName(1,['microphone','microphone']),'右声道');
 assert.equal(channelName(1,['microphone','egg']),'右声道 · EGG');
 assert.equal(channelName(0,['microphone']),'单声道');
 assert.equal(channelName(2,['microphone','egg','other']),'通道 3 · 其他');
});
