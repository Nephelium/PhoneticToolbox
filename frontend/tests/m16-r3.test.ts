import test from 'node:test';
import assert from 'node:assert/strict';
import {channelName} from '../src/modules/recording/state.ts';
import {autoMapping,importRows,exampleWorkbook,readTable} from '../src/modules/recording/task-import.ts';
import * as XLSX from '../src/modules/perception/vendor/xlsx.mjs';

test('M16 downloaded XLSX example reimports with Chinese headings and expands numeric repeats',async()=>{
 const bytes=exampleWorkbook();assert.equal(new DataView(bytes).getUint32(0,true),0x04034b50);
 const workbook=XLSX.read(bytes,{type:'array'});assert.deepEqual(workbook.SheetNames,['录音任务']);
 assert.equal(workbook.Sheets['录音任务'].A2.t,'s');assert.equal(workbook.Sheets['录音任务'].E3.t,'n');assert.equal(workbook.Sheets['录音任务'].E3.v,2);
 const table=await readTable(new File([bytes],'录音任务表格示例.xlsx')),rows=table.sheets[0].rows,result=importRows(rows,autoMapping(rows[0]));
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
