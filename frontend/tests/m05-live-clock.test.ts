import test from 'node:test';
import assert from 'node:assert/strict';
import {LiveFrameClock} from '../src/modules/lip-extraction/live-frame-clock.ts';

test('M05 live clock preserves valid media PTS exactly, independently of callback delay',()=>{
 const clock=new LiveFrameClock();
 assert.equal(clock.next(500,1.25,1),1.25);
 assert.equal(clock.next(900,1.3,2),1.3);
 assert.equal(clock.next(1200,1.75,5),1.75);
 assert.equal(clock.estimated,false);
});
test('M05 live zero PTS advances inference and filter time for each new presented frame',()=>{
 const clock=new LiveFrameClock();
 assert.equal(clock.next(100,0,1),0);
 assert.equal(clock.next(140,0,2),.04);
 assert.equal(clock.next(180,0,3),.08);
 assert.equal(clock.estimated,true);
});
test('M05 live clock rejects repeated frame identifiers even if media PTS changes',()=>{
 const clock=new LiveFrameClock();
 assert.equal(clock.next(100,0,1),0);
 assert.equal(clock.next(150,.05,1),null);
 assert.equal(clock.next(180,0,2),.08);
});
test('M05 estimated live clock remains continuous when a stalled media clock recovers',()=>{
 const clock=new LiveFrameClock();
 assert.equal(clock.next(100,2,1),2);
 assert.equal(clock.next(150,2,2),2.05);
 assert.equal(clock.next(200,30,3),2.1);
 assert.equal(clock.estimated,true);
});
test('M05 backwards live PTS uses observed elapsed time without altering the source PTS',()=>{
 const clock=new LiveFrameClock(),raw=[1,.5,.6];
 assert.deepEqual(raw.map((time,i)=>clock.next(100+i*250,time,i+1)),[1,1.25,1.5]);
 assert.deepEqual(raw,[1,.5,.6]);
});
test('M05 estimated live clock never invents an interval for a repeated callback time',()=>{
 const clock=new LiveFrameClock();
 assert.equal(clock.next(100,0,1),0);
 assert.equal(clock.next(100,0,2),null);
 assert.equal(clock.next(140,0,3),.04);
});
test('M05 live clock can mark missing source PTS as an estimate',()=>{
 const clock=new LiveFrameClock();
 assert.equal(clock.next(100,NaN,1),0);
 assert.equal(clock.next(150,NaN,2),.05);
 assert.equal(clock.estimated,true);
});
