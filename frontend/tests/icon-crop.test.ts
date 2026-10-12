import {test} from 'node:test';
import assert from 'node:assert/strict';
import {initialIconCrop,moveIconCrop,resizeIconCrop,removeIconBackground} from '../src/design/icon-crop.ts';
test('default icon crop is centred square; alternative ratios keep image bounds',()=>{
 assert.deepEqual(initialIconCrop(400,200),{x:100,y:0,width:200,height:200});
 assert.deepEqual(initialIconCrop(200,400),{x:0,y:100,width:200,height:200});
 assert.deepEqual(initialIconCrop(400,200,null),{x:0,y:0,width:400,height:200});
 const c=initialIconCrop(400,200,4/3);assert.equal(c.width/c.height,4/3);
 assert.deepEqual(moveIconCrop({x:10,y:20,width:50,height:60},-100,500,200,200),{x:0,y:140,width:50,height:60});
 const r=resizeIconCrop({x:100,y:10,width:20,height:20},300,300,200,100,1);assert.equal(r.width,90);assert.equal(r.height,90);
 assert.deepEqual(resizeIconCrop({x:0,y:0,width:20,height:20},20,30,200,200,1),{x:0,y:0,width:30,height:30});
 const free=resizeIconCrop({x:10,y:10,width:50,height:50},70,25,200,200,null);assert.equal(free.width,70);assert.equal(free.height,25);
});
test('solid background key removes edge-connected colour but keeps enclosed matching details',()=>{
 const pixels=new Uint8ClampedArray(7*7*4).fill(255);
 for(let y=1;y<=5;y++)for(let x=1;x<=5;x++){if(x===1||x===5||y===1||y===5){const p=(y*7+x)*4;pixels[p]=pixels[p+1]=pixels[p+2]=0;}}
 removeIconBackground(pixels,7,7,[255,255,255],0);
 assert.equal(pixels[3],0);assert.equal(pixels[(3*7+3)*4+3],255);assert.equal(pixels[(1*7+1)*4+3],255);
});
test('colour key traverses transparent letterboxing, supports sampled colours and tolerance',()=>{
 const pixels=new Uint8ClampedArray(4*4*4);
 for(let y=1;y<3;y++)for(let x=0;x<4;x++){const p=(y*4+x)*4;pixels.set([18,102,201,255],p);}
 pixels.set([200,10,10,255],(1*4+2)*4);
 removeIconBackground(pixels,4,4,[20,100,200],2);
 assert.equal(pixels[(1*4+0)*4+3],0);assert.equal(pixels[(1*4+2)*4+3],255);
 const exact=new Uint8ClampedArray([254,255,255,255]);removeIconBackground(exact,1,1,[255,255,255],0);assert.equal(exact[3],255);
});
