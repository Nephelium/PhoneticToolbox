import {test} from 'node:test';
import assert from 'node:assert/strict';
import {crc32} from 'node:zlib';
import {png300dpi,exportSize} from '../src/modules/parameter-display/png.ts';

const pixel=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=','base64');
function chunks(bytes:Uint8Array){
  const b=Buffer.from(bytes),result=[];
  for(let i=8;i<b.length;){const n=b.readUInt32BE(i);result.push({type:b.toString('ascii',i+4,i+8),data:b.subarray(i+8,i+8+n),crc:b.readUInt32BE(i+8+n)});i+=12+n;}
  return result;
}
test('M02-F05 PNG 300 dpi preserves compressed pixels and has an independently checked pHYs CRC',()=>{
  const original=chunks(pixel),result=chunks(png300dpi(pixel));
  assert.deepEqual(result.filter(c=>c.type!=='pHYs'),original);
  const physical=result.find(c=>c.type==='pHYs')!;
  assert.equal(physical.data.readUInt32BE(0),11811);assert.equal(physical.data.readUInt32BE(4),11811);assert.equal(physical.data[8],1);
  assert.equal(physical.crc,crc32(Buffer.concat([Buffer.from('pHYs'),physical.data])));
  assert(result.findIndex(c=>c.type==='pHYs')<result.findIndex(c=>c.type==='IDAT'));
  assert.deepEqual(png300dpi(png300dpi(pixel)),png300dpi(pixel));
});
test('M02-F05 rejects truncated PNG and bounds 300 dpi allocation without silent downsampling',()=>{
  assert.throws(()=>png300dpi(pixel.subarray(0,29)));
  assert.throws(()=>png300dpi(new Uint8Array(40)));
  assert.deepEqual(exportSize(640,800),{width:2000,height:2500});
  for(const [w,h] of [[0,500],[NaN,500],[640,Infinity],[6000,6000]])assert.throws(()=>exportSize(w,h));
});
