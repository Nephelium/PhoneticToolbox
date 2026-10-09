import test from 'node:test';
import assert from 'node:assert/strict';
import {playingStep} from '../src/modules/phonation-synthesis/f0.ts';

test('M07 group follows PCM sample boundaries, including seek into later steps',()=>{
 const rate=11025,perStep=7321,frames=perStep*9;
 assert.equal(playingStep(0,rate,frames,9),'step01');
 assert.equal(playingStep((perStep-1)/rate,rate,frames,9),'step01');
 assert.equal(playingStep(perStep/rate,rate,frames,9),'step02');
 assert.equal(playingStep(perStep*8/rate,rate,frames,9),'step09');
 assert.equal(playingStep((frames-1)/rate,rate,frames,9),'step09');
 assert.equal(playingStep(frames/rate,rate,frames,9),'');
});

test('M07 does not invent playback segments for invalid or unequal PCM groups',()=>{
 for(const args of [[NaN,11025,900,9],[-1,11025,900,9],[0,0,900,9],[0,11025,0,9],[0,11025,901,9],[0,11025,900,1]]){
  assert.equal(playingStep(...args as [number,number,number,number]),'');
 }
});
