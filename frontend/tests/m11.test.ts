import {test} from 'node:test';
import assert from 'node:assert/strict';
import {defaults,beamChanged,validate,pairFiles} from '../src/modules/mfa/state.ts';
const file=(name:string,text='a')=>new File([text],name);
test('M11 retains distinct GUI beam linkage and submit normalization',()=>{
 assert.deepEqual(defaults(),{beam:100,retry_beam:400});
 assert.deepEqual(beamChanged({beam:10,retry_beam:40},20),{beam:20,retry_beam:80});
 assert.equal(beamChanged({beam:20,retry_beam:80},10).retry_beam,80);
 assert.equal(validate({beam:10,retry_beam:11}).retry_beam,11);
 assert.equal(validate({beam:10,retry_beam:10}).retry_beam,40);
 assert.throws(()=>validate({beam:NaN,retry_beam:40}));
});
test('M11 explicit transcription source permits coexisting files and rejects missing selection',()=>{
 const files=[file('中文.wav'),file('中文.lab','lab'),file('中文.TextGrid','textgrid'),file('中文.txt','txt')];
 assert.equal(pairFiles(files,'.lab')[0].transcript.name,'中文.lab');
 assert.equal(pairFiles(files,'.TextGrid')[0].transcript.name,'中文.TextGrid');
 assert.equal(pairFiles(files,'.txt')[0].transcript.name,'中文.txt');
 assert.throws(()=>pairFiles(files),/选择转写来源/);
 assert.throws(()=>pairFiles([file('a.wav'),file('a.lab')],'.TextGrid'),/TextGrid/);
 assert.throws(()=>pairFiles([file('a.wav'),file('a.lab',''),file('a.TextGrid')],'.lab'),/为空/);
});
test('M11 requires existing uniquely paired text; never offers ASR',()=>{
 assert.throws(()=>pairFiles([file('中文.wav')]),/唯一同名/);
 assert.throws(()=>pairFiles([file('中文.wav'),file('中文.lab'),file('中文.txt')]),/唯一同名/);
 assert.throws(()=>pairFiles([file('中文.wav'),file('中文.lab','')]),/为空/);
 assert.equal(pairFiles([file('中文 空格.WAV'),file('中文 空格.TextGrid')])[0].name,'中文 空格.WAV');
});
test('M11 duplicate audio and overlong encoded paths fail before upload',()=>{
 assert.throws(()=>pairFiles([file('a.wav'),file('A.WAV'),file('a.lab'),file('A.lab')]),/重复/);
 const name='中文'.repeat(30);assert.throws(()=>pairFiles([file(name+'.wav'),file(name+'.lab')]),/编码预算/);
});
test('M11 selected-source budget counts only files that are uploaded',()=>{
 const wav=file('a.wav'),lab=file('a.lab'),grid=file('a.TextGrid');
 Object.defineProperty(wav,'size',{value:60_000_000});Object.defineProperty(grid,'size',{value:8_000_000});
 assert.equal(pairFiles([wav,lab,grid],'.lab').length,1);
 assert.throws(()=>pairFiles([wav,lab,grid],'.TextGrid'),/64 MB/);
});
