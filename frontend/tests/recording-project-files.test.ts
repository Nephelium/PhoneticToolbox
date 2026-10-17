import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readRecordingProject,recordingHash} from '../src/platform/recording-project-files.ts';
async function fixture(){
 const raw=new ArrayBuffer(32),v=new DataView(raw);for(let i=0;i<8;i++)v.setFloat32(i*4,(i-4)/8,true);
 const span={file:'raw/one.f32',start:0,end:4,frames:4,channels:2,sha256:await recordingHash(raw)};
 const project={schema_version:'ptb-recording/1',id:'project',revision:1,tasks:[],selected:{__free__:'take'},recoveries:[],takes:[{id:'take',config:{channels:2,sample_rate:48000,roles:['microphone','egg']},head:1,undo:[0],redo:[],versions:[{id:'raw',kind:'raw',metadata:{},spans:[span]},{id:'edit',kind:'edited',parent:'raw',metadata:{},spans:[{...span,start:1,end:3},{...span,start:0,end:1}]}]}]};
 const files=new Map<string,File>([['folder/project.json',new File([JSON.stringify(project)],'project.json')],['folder/raw/one.f32',new File([raw],'one.f32')]]);return {project,files};
}
test('desktop f32 EDL import preserves exact all-channel frames and parent history',async()=>{
 const {files}=await fixture(),result=await readRecordingProject(files),take=result.project.takes[0];assert.equal(take.head,1);assert.equal(take.versions[1].frames,3);assert.equal(take.versions[1].metadata.parent,'raw');const value=result.values.get('edit')!;
 assert.deepEqual(Array.from(value.blocks[0][0]),[-.25,0,-.5]);assert.deepEqual(Array.from(value.blocks[0][1]),[-.125,.125,-.375]);assert.equal(new DataView(await value.file.arrayBuffer()).getUint32(24,true),48000);
});
test('desktop import rejects duplicate version identities before cache publication',async()=>{
 const {files,project}=await fixture();project.takes[0].versions[1].id='raw';files.set('folder/project.json',new File([JSON.stringify(project)],'project.json'));await assert.rejects(readRecordingProject(files),/标识重复/);
});
test('desktop import rejects changed hashes and traversal before accepting samples',async()=>{
 const {files,project}=await fixture();files.set('folder/raw/one.f32',new File([new Uint8Array(32)],'one.f32'));await assert.rejects(readRecordingProject(files),/摘要/);
 project.takes[0].versions[0].spans[0].file='../outside.f32';files.set('folder/project.json',new File([JSON.stringify(project)],'project.json'));await assert.rejects(readRecordingProject(files),/路径/);
});
