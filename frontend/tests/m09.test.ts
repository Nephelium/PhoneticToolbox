import {test} from 'node:test';import assert from 'node:assert/strict';
import {imageDimensions} from '../src/modules/spectrogram-to-audio/image-header.ts';
import {pixelStroke} from '../src/modules/spectrogram-to-audio/pixel-stroke.ts';
function png(width:number,height:number){const b=new Uint8Array(24);b.set([137,80,78,71,13,10,26,10],0);const v=new DataView(b.buffer);v.setUint32(12,0x49484452);v.setUint32(16,width);v.setUint32(20,height);return b.buffer;}
test('M09 image header bounds before renderer decode',()=>{assert.deepEqual(imageDimensions(png(1920,1080)),{width:1920,height:1080});assert.throws(()=>imageDimensions(png(100000,100000)));assert.throws(()=>imageDimensions(png(1,1)));assert.throws(()=>imageDimensions(new ArrayBuffer(0)));});
test('M09 JPEG SOF and malformed segment bounds',()=>{const raw=new Uint8Array([255,216,255,192,0,8,8,0,65,0,100,0]);assert.deepEqual(imageDimensions(raw.buffer),{width:100,height:65});raw[5]=255;assert.throws(()=>imageDimensions(raw.buffer));});
test('M09 displayed pixel brushes have the requested diameter without antialias expansion',()=>{
 for(const size of [1,2,3,12,22,74,199,200]){
  const p=pixelStroke(401,401,{color:0,opacity:1,size,size_unit:'px',points:[{x:.5,y:.5}]});
  const points=[...p.mask.entries()].filter(([,value])=>value).map(([i])=>[i%p.width,Math.floor(i/p.width)]);
  assert.equal(Math.max(...points.map(p=>p[0]))-Math.min(...points.map(p=>p[0]))+1,size);
  assert.equal(Math.max(...points.map(p=>p[1]))-Math.min(...points.map(p=>p[1]))+1,size);
  if(size===1)assert.equal(points.length,1);
 }
});
test('M09 one-pixel line, repeated points and tie-to-even quantization',()=>{
 const mark={color:0 as const,opacity:1,size:1,size_unit:'px' as const,points:[{x:0,y:.5},{x:1,y:.5}]};
 assert.equal(pixelStroke(9,9,mark).mask.reduce((a,b)=>a+b),9);
 assert.deepEqual(pixelStroke(9,9,{...mark,points:[{x:.5,y:.5}]}),pixelStroke(9,9,{...mark,points:Array(10).fill({x:.5,y:.5})}));
 assert.equal(pixelStroke(6,6,{...mark,points:[{x:.5,y:.5}]}).left,2);
});
