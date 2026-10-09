import {test} from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';
import {parseWav,selection,envelope} from '../src/platform/wav.ts';import {modules,groups} from '../src/app/registry.ts';
const bytes=readFileSync(new URL('../src/assets/SYN-EGG-44100.wav',import.meta.url));
const buffer=()=>bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength) as ArrayBuffer;
test('new modules preserve all original workspace IDs and remain in visible groups',()=>{
 const original=['参数估计','参数显示','EGG 信号分析','LPC 谱图','唇形提取','声学参数合成','发声类型合成','变速变调','语谱图转音频','生理参数合成','MFA 自动标注','TextGrid标注','汉字转国际音标','音系归纳','感知实验'];
 original.forEach((title,i)=>{const m=modules.find(m=>m.id===`M${String(i+1).padStart(2,'0')}`);assert.equal(m?.title,title);assert.equal(m?.group,Math.floor(i/5));});
 assert.equal(new Set(modules.map(m=>m.id)).size,18);assert.equal(groups.length,4);
 assert.equal(modules.find(m=>m.id==='M18')?.title,'语音学论文精读');assert.equal(modules.find(m=>m.id==='M18')?.group,3);
 assert.equal(modules.find(m=>m.id==='M16')?.group,0);assert.equal(modules.find(m=>m.id==='M17')?.group,2);
 for(const module of modules)assert.ok(groups[module.group],`${module.id} must have a visible navigation group`);
});
test('real public stereo fixture retains samples, timing and channel order',()=>{
 const a=parseWav(buffer(),'fixture');assert.equal(a.sampleRate,44100);assert.equal(a.frames,35280);assert.equal(a.duration,.8);assert.equal(a.channels.length,2);
 const phase=2*Math.PI*120/44100;
 assert.equal(a.channels[0][1],Math.round((.5*Math.sin(phase)+.1*Math.sin(3*phase))*32767)/32768);
 assert.notEqual(a.channels[0][1],a.channels[1][1]);
});
test('invalid and truncated inputs cannot become fabricated waveforms',()=>{
 assert.throws(()=>parseWav(new ArrayBuffer(50),'bad'),/RIFF/);
 assert.throws(()=>parseWav(buffer().slice(0,100),'truncated'),/不完整/);
 const b=buffer();new DataView(b).setUint16(32,1,true);assert.throws(()=>parseWav(b,'bad block'),/音频头/);
});
test('FLOAT stereo reads distinct signed samples without channel averaging',()=>{
 const b=new ArrayBuffer(60),v=new DataView(b);const str=(p:number,s:string)=>[...s].forEach((c,i)=>v.setUint8(p+i,c.charCodeAt(0)));
 str(0,'RIFF');v.setUint32(4,52,true);str(8,'WAVEfmt ');v.setUint32(16,16,true);v.setUint16(20,3,true);v.setUint16(22,2,true);v.setUint32(24,44100,true);v.setUint32(28,352800,true);v.setUint16(32,8,true);v.setUint16(34,32,true);str(36,'data');v.setUint32(40,16,true);[.25,-.5,.75,-1].forEach((n,i)=>v.setFloat32(44+i*4,n,true));
 const a=parseWav(b,'float');assert.deepEqual([...a.channels[0]],[.25,.75]);assert.deepEqual([...a.channels[1]],[-.5,-1]);v.setFloat32(44,NaN,true);assert.throws(()=>parseWav(b,'nan'),/非有限/);
});
test('selection clamps to the actual signal and peak envelope keeps impulses',()=>{
 assert.deepEqual(selection(9,-1,.8),[0,.8]);assert.deepEqual(selection(NaN,.5,.8),[0,.5]);
 assert.deepEqual(envelope(new Float32Array([0,0,1,0,-.5,0]),0,6,2),[[0,1],[-.5,0]]);
});

import {positionSelection} from '../src/platform/wav.ts';
test('overview positioning preserves selection length and fits the end of the actual file',()=>{
 assert.deepEqual(positionSelection(.2,.5,.8),[.2,.7]);
 const [start,end]=positionSelection(.79,.5,.8);assert(Math.abs(start-.3)<1e-12);assert.equal(end,.8);
 assert.deepEqual(positionSelection(-1,.5,.8),[0,.5]);assert.deepEqual(positionSelection(2,1,.8),[0,.8]);
});
