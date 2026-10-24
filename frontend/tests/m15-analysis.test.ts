import test from 'node:test';
import assert from 'node:assert/strict';
import {defaults,generate,copy,shuffle,validate,type Asset,type Attempt,type Session} from '../src/modules/perception/model.ts';
import {summarizeSession,scoreAttempt,analysisRows,analysisScoringRows} from '../src/modules/perception/analysis.ts';
import {projectJSON,parseProject,sequenceRows,readSequence,workbook,resultRows,analysisBlob} from '../src/modules/perception/formats.ts';
import * as XLSX from '../src/modules/perception/vendor/xlsx.mjs';

function fixture(){
 const p=defaults();p.questionnaire=[];p.assets=[1,2,3,4].map(i=>({id:String(i),hash:String(i).repeat(64),name:`${i}.wav`,path:`${i}.wav`,group:'X',kind:'audio',size:44,mime:'audio/wav',originalSampleRate:48000} as Asset));p.trials=generate(p);p.trials.forEach(t=>t.correctKey='j');
 const attempts=[100,200,300,400].map((rtMs,trialIndex)=>({id:String(trialIndex),trialIndex,attempt:1,status:'completed',key:trialIndex===1?'f':'j',rtMs,startedAt:'2026-10-19T00:00:00Z',finishedAt:'2026-10-19T00:00:01Z',roles:{X:p.assets[trialIndex]}} as Attempt));
 return {schema:'ptb-m15-session/1',id:'s',participantId:'p',project:p,answers:{},createdAt:'2026-10-19T00:00:00Z',status:'completed',attempts} as Session;
}
test('M15 analysis known R7 quartiles, mean, sample SD and accuracy denominator',()=>{
 const s=fixture(),a=summarizeSession(s);assert.equal(a.mean,250);assert.equal(a.median,250);assert.equal(a.q1,175);assert.equal(a.q3,325);assert.equal(a.iqr,150);assert.equal(a.min,100);assert.equal(a.max,400);assert(Math.abs(a.sd!-Math.sqrt(50000/3))<1e-12);assert.equal(a.accuracy,.75);assert.equal(a.scored,4);
});
test('M15 analysis retains zero; excludes missing/negative/nonfinite RT without dropping scored answers',()=>{
 const s=fixture();s.attempts.forEach((a,i)=>a.rtMs=[0,null,-1,Infinity][i]);const a=summarizeSession(s);assert.equal(a.rtN,1);assert.equal(a.missingRT,3);assert.equal(a.mean,0);assert.equal(a.q1,0);assert.equal(a.q3,0);assert.equal(a.sd,null);assert.equal(a.scored,4);
 s.attempts=[];const empty=summarizeSession(s);assert.equal(empty.mean,null);assert.equal(empty.accuracy,null);assert.equal(empty.unanswered,4);
});
test('M15 analysis deduplicates last completed trial and separately counts interruption, invalid and unscored',()=>{
 const s=fixture();s.attempts=[s.attempts[0],{...s.attempts[0],id:'retry',rtMs:500,key:'f'},{...s.attempts[1],id:'interrupted',status:'interrupted',rtMs:12},{...s.attempts[2],id:'invalid',status:'invalid'},s.attempts[3]];delete s.project.trials[3].correctKey;
 const a=summarizeSession(s);assert.equal(a.completed,2);assert.equal(a.mean,450);assert.equal(a.duplicateCompleted,1);assert.equal(a.interrupted,1);assert.equal(a.invalid,1);assert.equal(a.unanswered,2);assert.equal(a.scored,1);assert.equal(a.correct,0);assert.equal(a.unscored,1);assert.equal(a.accuracy,0);
 assert.equal(analysisScoringRows([s]).filter(row=>row.at(-1)==='TRUE').length,2);
});
test('M15 current answer key rescoring is explicit, immutable, and matches trial identity/hash/original keys',()=>{
 const s=fixture();s.project.trials.forEach(t=>delete t.correctKey);const current=copy(s.project);current.trials.forEach(t=>t.correctKey='j');const before=JSON.stringify(s);
 assert.equal(summarizeSession(s).accuracy,null);assert.equal(summarizeSession(s,current).accuracy,.75);assert.equal(JSON.stringify(s),before);
 current.trials[0].stimuli.X!.hash='f'.repeat(64);current.trials[1].correctKey='z';current.trials[2].id='other';assert.equal(summarizeSession(s,current).scored,1);
 current.id='other-project';assert.equal(summarizeSession(s,current).accuracy,null);
});
test('M15 optional new settings and standard answers roundtrip JSON/XLSX and stay with shuffled/regenerated trials',()=>{
 const s=fixture(),p=s.project;p.config.responsePrompt='请判断，a̠';p.config.showResponsePrompt=true;p.trials[0].correctKey=' ';p.config.validKeys.push({key:' ',label:'空格'});
 assert.deepEqual(parseProject(projectJSON(p)),p);const restored=readSequence(workbook(sequenceRows(p)),p);assert.equal(restored[0].correctKey,' ');assert.equal(restored[1].correctKey,'j');assert.equal(generate(p)[0].correctKey,' ');
 const out=shuffle(p,123);assert(out.trials.every(t=>t.correctKey===p.trials.find(old=>old.id===t.id)?.correctKey));
 delete p.config.responsePrompt;delete p.config.showResponsePrompt;p.trials.forEach(t=>delete t.correctKey);assert.deepEqual(parseProject(projectJSON(p)),p);
});
test('M15 questionnaire punctuation is content and blank/trimmed duplicate options block preflight',()=>{
 const p=fixture().project;p.questionnaire=[{id:'q',label:'水平',type:'radio',required:true,options:['CET-4,CET-6','托福，雅思']}];assert.deepEqual(validate(p),[]);p.questionnaire[0].options!.push('');assert(validate(p).some(e=>e.includes('选项')));p.questionnaire[0].options=['A',' A '];assert(validate(p).some(e=>e.includes('选项')));
 p.trials[0].correctKey='z';assert(validate(p).some(e=>e.includes('标准答案')));assert.throws(()=>parseProject(JSON.stringify({...p,config:{...p.config,showResponsePrompt:'yes'}})));
});
test('M15 result correctness columns append without moving old columns; analysis workbook includes scoring provenance',async()=>{
 const s=fixture(),rows=resultRows(s);assert.equal(rows[0][0],'Participant_ID');assert.deepEqual(rows[0].slice(-3),['Correct_Key','Is_Correct','Playback_Volume']);assert.deepEqual(rows[2].slice(-3),['j','FALSE',1]);
 assert.equal(scoreAttempt(s,{...s.attempts[0],status:'interrupted'}).correct,null);
 const wb=XLSX.read(await analysisBlob([s],undefined,'xlsx').arrayBuffer(),{type:'array'});assert.deepEqual(wb.SheetNames,['Participants','Scoring']);assert.equal(XLSX.utils.sheet_to_json(wb.Sheets.Participants)[0].Mean_RT_ms,250);assert.equal(XLSX.utils.sheet_to_json(wb.Sheets.Scoring).length,4);
 const summary=analysisRows([s]);assert.equal(summary[0].length,summary[1].length);assert.equal((await analysisBlob([s],undefined,'csv').text()).includes('Analysis_Method'),true);
});
