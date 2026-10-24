import {keys,roles,type Attempt,type Project,type Session} from './model.ts';

export const ANALYSIS_METHOD='completed-last-per-trial; finite-nonnegative-RT; quantile-R7; sample-SD/v1';
export type AnswerSource='session'|'current';

export function participantLabel(s:Session){
 const q=s.project.questionnaire.find(q=>q.type==='text'&&/姓名|编号/.test(q.label))??s.project.questionnaire.find(q=>q.type==='text');
 const answer=q?s.answers[q.id]:undefined;
 return typeof answer==='string'&&answer.trim()?answer:s.participantId.slice(0,8);
}

export function scoreAttempt(s:Session,a:Attempt,current?:Project){
 const original=s.project.trials[a.trialIndex];
 const trial=current?current.trials.find(t=>t.id===original?.id):original;
 const matches=!current||current.id===s.project.id&&current.paradigm===s.project.paradigm&&roles(s.project.paradigm).every(r=>{
  const hash=original?.stimuli[r]?.hash;
  return !!hash&&hash===trial?.stimuli[r]?.hash&&(!a.roles[r]||a.roles[r]?.hash===hash);
 });
 const key=matches?trial?.correctKey?.toLowerCase():undefined;
 const correctKey=key&&keys(s.project,a.trialIndex).some(k=>k.key.toLowerCase()===key)?key:null;
 return {correctKey,correct:a.status==='completed'&&a.key!==null&&correctKey!==null?a.key.toLowerCase()===correctKey:null};
}

// R quantile(type=7): zero-based position (n-1)*p, linear interpolation.
function quantile(sorted:number[],p:number){
 if(!sorted.length)return null;
 const position=(sorted.length-1)*p,lo=Math.floor(position),fraction=position-lo;
 return sorted[lo]+fraction*(sorted[Math.min(lo+1,sorted.length-1)]-sorted[lo]);
}

export function summarizeSession(s:Session,current?:Project){
 const completed=new Map<number,Attempt>();
 for(const a of s.attempts)if(a.status==='completed'&&Number.isInteger(a.trialIndex)&&a.trialIndex>=0&&a.trialIndex<s.project.trials.length)completed.set(a.trialIndex,a);
 const answers=[...completed.values()],rt=answers.map(a=>a.rtMs).filter((v):v is number=>typeof v==='number'&&Number.isFinite(v)&&v>=0).sort((a,b)=>a-b);
 const n=rt.length,mean=n?rt.reduce((sum,v)=>sum+v,0)/n:null,q1=quantile(rt,.25),q3=quantile(rt,.75);
 const scores=answers.map(a=>scoreAttempt(s,a,current)),scored=scores.filter(a=>a.correct!==null),correct=scored.filter(a=>a.correct).length;
 return {
  sessionId:s.id,participantId:s.participantId,participant:participantLabel(s),createdAt:s.createdAt,status:s.status,
  total:s.project.trials.length,completed:answers.length,unanswered:s.project.trials.length-answers.length,
  interrupted:s.attempts.filter(a=>a.status==='interrupted').length,invalid:s.attempts.filter(a=>a.status==='invalid').length,
  duplicateCompleted:s.attempts.filter(a=>a.status==='completed').length-answers.length,
  rtN:n,missingRT:answers.length-n,mean,median:quantile(rt,.5),q1,q3,iqr:q1===null||q3===null?null:q3-q1,
  sd:n>1?Math.sqrt(rt.reduce((sum,v)=>sum+(v-mean!)**2,0)/(n-1)):null,min:n?rt[0]:null,max:n?rt[n-1]:null,
  scored:scored.length,correct,incorrect:scored.length-correct,unscored:answers.length-scored.length,accuracy:scored.length?correct/scored.length:null,
  answerSource:current?'current':'session',method:ANALYSIS_METHOD,
 };
}

export function analysisRows(sessions:Session[],current?:Project){
 const fields=['participant','participantId','sessionId','createdAt','status','total','completed','unanswered','interrupted','invalid','duplicateCompleted','rtN','missingRT','mean','median','q1','q3','iqr','sd','min','max','scored','correct','incorrect','unscored','accuracy','answerSource','method'] as const;
 return [
  ['Participant','Participant_UUID','Session_ID','Created_At','Status','Total_Trials','Completed_Trials','Unanswered_Trials','Interrupted_Attempts','Invalid_Attempts','Duplicate_Completed_Attempts','RT_N','Missing_RT','Mean_RT_ms','Median_RT_ms','Q1_RT_ms','Q3_RT_ms','IQR_RT_ms','Sample_SD_RT_ms','Min_RT_ms','Max_RT_ms','Scored_Trials','Correct','Incorrect','Unscored_Completed','Accuracy_0_to_1','Answer_Source','Analysis_Method'],
  ...sessions.map(s=>{const summary=summarizeSession(s,current);return fields.map(k=>summary[k]??'');}),
 ];
}

export function analysisScoringRows(sessions:Session[],current?:Project){
 return [
  ['Participant_UUID','Session_ID','Trial_ID','Trial_Index','Attempt_ID','Attempt_Status','Response_Key','RT_ms','Correct_Key','Is_Correct','Answer_Source','Included_In_Summary'],
  ...sessions.flatMap(s=>{
   const last=new Map<number,string>();for(const a of s.attempts)if(a.status==='completed')last.set(a.trialIndex,a.id);
   return s.attempts.map(a=>{const score=scoreAttempt(s,a,current);return [s.participantId,s.id,s.project.trials[a.trialIndex]?.id??'',a.trialIndex+1,a.id,a.status,a.key??'',a.rtMs??'',score.correctKey??'',score.correct===null?'':score.correct?'TRUE':'FALSE',current?'current':'session',last.get(a.trialIndex)===a.id?'TRUE':'FALSE'];});
  }),
 ];
}
