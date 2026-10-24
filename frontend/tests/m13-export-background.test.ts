import test from 'node:test';
import assert from 'node:assert/strict';
import {backgroundColor,blurPixels,coverImage,pngBackgroundDefaults} from '../src/modules/mandarin-ipa/export-background.ts';
test('PNG defaults copy the current image/effects or current theme colour without mutating either',()=>{
  const current={url:'blob:owned',name:'当前.png',blur:13,opacity:46,color:'#123456'};
  const options=pngBackgroundDefaults(current);assert.deepEqual(options,{mode:'image',imageUrl:'blob:owned',imageName:'当前.png',blur:13,opacity:46,color:'#123456',baseColor:'#123456'});
  options.blur=0;assert.equal(current.blur,13);
  assert.equal(pngBackgroundDefaults({...current,url:''}).mode,'color');
  assert.equal(backgroundColor('rgb(18, 52, 86)'),'#123456');assert.equal(backgroundColor('#abc'),'#aabbcc');assert.equal(backgroundColor('invalid'),'#ffffff');
});
test('PNG cover preserves image proportions and includes three blur radii of overscan',()=>{
  assert.deepEqual(coverImage(400,200,100,100,10),{x:-30,y:-130,width:460,height:460});
  assert.deepEqual(coverImage(200,400,100,50,0),{x:-300,y:0,width:800,height:400});
});
test('Gaussian fallback preserves constant opaque colours, symmetric edges and unblurred input',()=>{
  const constant=new Uint8ClampedArray(7*9*4);for(let i=0;i<constant.length;i+=4)constant.set([37,82,129,255],i);
  assert.deepEqual(blurPixels(constant,7,9,8),constant);
  const impulse=new Uint8ClampedArray(9*9*4);for(let i=3;i<impulse.length;i+=4)impulse[i]=255;impulse.set([255,255,255,255],(4*9+4)*4);
  const blurred=blurPixels(impulse,9,9,1.5);assert(blurred[(4*9+4)*4]!>0&&blurred[(4*9+4)*4]!<255);
  for(let y=0;y<9;y++)for(let x=0;x<9;x++){
    assert.equal(blurred[(y*9+x)*4],blurred[(y*9+8-x)*4]);assert.equal(blurred[(y*9+x)*4],blurred[((8-y)*9+x)*4]);assert.equal(blurred[(y*9+x)*4+3],255);
  }
  assert.deepEqual(blurPixels(impulse,9,9,0),impulse);assert.equal(impulse[(4*9+4)*4],255);
});
