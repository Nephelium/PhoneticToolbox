import test from 'node:test';
import assert from 'node:assert/strict';
import {installModelPinch} from '../public/vocal-tract/model-pinch.mjs';
function setup(){
 const surface=new EventTarget() as EventTarget&{setPointerCapture:(id:number)=>void};surface.setPointerCapture=()=>{};
 const calls:{kind:string;point?:any}[]=[],release=installModelPinch(surface,{begin:(point:any)=>calls.push({kind:'begin',point}),move:(point:any)=>calls.push({kind:'move',point}),end:()=>calls.push({kind:'end'})});
 const send=(type:string,id:number,x:number,y=0,pointerType='touch')=>{const event=new Event(type,{cancelable:true});Object.assign(event,{pointerId:id,clientX:x,clientY:y,pointerType});surface.dispatchEvent(event);return event;};
 return {send,calls,release};
}
test('head pinch starts at two touches and uses distance ratio and moving midpoint',()=>{
 const {send,calls,release}=setup();assert.equal(send('pointerdown',1,0).defaultPrevented,false);assert.equal(calls.length,0);
 assert(send('pointerdown',2,100).defaultPrevented);assert.equal(calls[0]!.point.clientX,50);
 assert(send('pointermove',2,200).defaultPrevented);assert.equal(calls[1]!.point.scale,2);assert.equal(calls[1]!.point.clientX,100);
 send('pointerup',2,200);const n=calls.length;assert(send('pointermove',1,20).defaultPrevented);assert.equal(calls.length,n,'remaining finger must not resume an organ edit');
 send('pointerup',1,20);assert.equal(calls.at(-1)!.kind,'end');assert.equal(send('pointerdown',3,10).defaultPrevented,false);release();
});
test('cancellation and capture loss release the gesture; mouse input remains unchanged',()=>{
 const {send,calls,release}=setup();assert.equal(send('pointerdown',9,0,0,'mouse').defaultPrevented,false);
 send('pointerdown',1,0);send('pointerdown',2,100);send('pointercancel',2,100);assert(send('pointermove',1,30).defaultPrevented);send('lostpointercapture',1,30);assert.equal(calls.at(-1)!.kind,'end');
 const n=calls.length;send('pointermove',2,200);assert.equal(calls.length,n);release();assert.equal(send('pointerdown',3,0).defaultPrevented,false);
});
test('3D orbit receives pointer endings without receiving pinch moves',()=>{
 const surface=new EventTarget() as EventTarget&{setPointerCapture:(id:number)=>void};surface.setPointerCapture=()=>{};let ended=0,down=0,moves=0;
 const release=installModelPinch(surface,{begin:()=>{},move:()=>{},propagateEnd:true,end:()=>ended++});
 surface.addEventListener('pointerdown',()=>down++);surface.addEventListener('pointermove',()=>moves++);surface.addEventListener('pointerup',()=>ended++);
 for(const [type,id] of [['pointerdown',1],['pointerdown',2],['pointermove',1],['pointerup',1],['pointerup',2]] as const){const event=new Event(type,{cancelable:true});Object.assign(event,{pointerType:'touch',pointerId:id,clientX:id*30,clientY:0});surface.dispatchEvent(event);}
 assert.equal(down,1);assert.equal(moves,0);assert.equal(ended,3);release();
});
