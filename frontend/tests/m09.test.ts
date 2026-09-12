import {test} from 'node:test';import assert from 'node:assert/strict';
import {imageDimensions} from '../src/modules/spectrogram-to-audio/image-header.ts';
function png(width:number,height:number){const b=new Uint8Array(24);b.set([137,80,78,71,13,10,26,10],0);const v=new DataView(b.buffer);v.setUint32(12,0x49484452);v.setUint32(16,width);v.setUint32(20,height);return b.buffer;}
test('M09 image header bounds before renderer decode',()=>{assert.deepEqual(imageDimensions(png(1920,1080)),{width:1920,height:1080});assert.throws(()=>imageDimensions(png(100000,100000)));assert.throws(()=>imageDimensions(png(1,1)));assert.throws(()=>imageDimensions(new ArrayBuffer(0)));});
test('M09 JPEG SOF and malformed segment bounds',()=>{const raw=new Uint8Array([255,216,255,192,0,8,8,0,65,0,100,0]);assert.deepEqual(imageDimensions(raw.buffer),{width:100,height:65});raw[5]=255;assert.throws(()=>imageDimensions(raw.buffer));});
