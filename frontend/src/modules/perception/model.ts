// M15 method/client-v1. Legacy source semantics are mapped in M15-source-map.md.
export type Paradigm='X'|'AX'|'ABX'|'AXB';
export type Role='A'|'B'|'X';
export type MediaKind='audio'|'image'|'text';
export interface Asset {id:string;hash:string;name:string;path:string;group:Role;kind:MediaKind;size:number;mime:string;originalSampleRate:number|null;decodedSampleRate?:number;channels?:number;duration?:number;decodedBytes?:number}
export interface Stimulus {id:string|null;name:string;hash:string|null}
export interface Trial {id:string;stimuli:Partial<Record<Role,Stimulus>>;instruction:string}
export interface Key {key:string;label:string}
export interface Range {start:number;end:number}
export interface KeyRange extends Range {id:string;keys:Key[];notice:string}
export interface Question {id:string;type:'text'|'radio'|'checkbox';label:string;required:boolean;options?:string[]}
export interface Config {title:string;introText:string;introLargeText:boolean;interTrialText:string;showInterTrialText:boolean;largeText:boolean;advanceMode:'auto'|'manual';autoPlay:boolean;showFilename:boolean;interTrialInterval:number;isi:number;useBeep:boolean;validKeys:Key[];keyRanges:KeyRange[];shuffleRanges:Range[]}
export interface Randomization {algorithm:'mulberry32-v1';seed:number;ranges:Range[];before:string[];after:string[]}
export interface Project {schema:'ptb-m15-project/1';id:string;paradigm:Paradigm;config:Config;assets:Asset[];trials:Trial[];questionnaire:Question[];randomizations:Randomization[]}
export type Answers=Record<string,string|string[]>;
export interface Timing {timeOrigin:number;preparedPerfMs:number;responseOpenPerfMs:number|null;keyEventPerfMs:number|null;keyHandledPerfMs:number|null;visualFramePerfMs:number|null;audioMapping:{audioSeconds:number;performanceMs:number;bracketMs:number};planned: {role:Role;startAudioSeconds:number;endAudioSeconds:number;scheduledAtAudioSeconds:number;scheduledAtPerfMs:number;scheduleLateByMs:number;outputStartEstimatePerfMs:number|null}[];observedEnded:{role:Role;performanceMs:number;audioSeconds:number}[];endDetectedPerfMs:number|null;responseDelayFromMappedEndMs:number|null;outputTimestamp:{contextTime:number;performanceTime:number}|null}
export interface Attempt {id:string;trialIndex:number;attempt:number;status:'running'|'completed'|'interrupted'|'invalid';roles:Partial<Record<Role,Asset>>;order:Role[];key:string|null;keyLabel:string|null;rtMs:number|null;rtDefinition:'response-open-to-handler-performance-ms/v1';timing:Timing|null;reason:string|null;startedAt:string;finishedAt:string|null}
export interface Session {schema:'ptb-m15-session/1';id:string;participantId:string;project:Project;answers:Answers;createdAt:string;revision:number;nextIndex:number;status:'ready'|'running'|'paused'|'completed'|'ended';attempts:Attempt[];events:{kind:string;at:string;perfMs:number;timeOrigin:number;detail:string}[];environment:Record<string,unknown>;exportRequestedRevision:number|null;exportConfirmedRevision:number|null}
export const LIMITS={files:2000,trials:10000,fileBytes:64*1024*1024,totalBytes:512*1024*1024,decodedBytes:128*1024*1024,textBytes:1024*1024,projectBytes:16*1024*1024};
export const uid=()=>crypto.randomUUID();
export const copy=<T>(v:T):T=>JSON.parse(JSON.stringify(v));
export function defaults():Project {return {schema:'ptb-m15-project/1',id:uid(),paradigm:'X',assets:[],trials:[],randomizations:[],config:{title:'听觉感知实验',introText:'接下来您将听到一组音频，请仔细聆听并根据直觉做出判断。',introLargeText:true,interTrialText:'请您判断xxx',showInterTrialText:false,largeText:true,advanceMode:'auto',autoPlay:true,showFilename:false,interTrialInterval:1000,isi:500,useBeep:true,validKeys:[{key:'f',label:'错误 (F)'},{key:'j',label:'正确 (J)'}],keyRanges:[],shuffleRanges:[]},questionnaire:[{id:'q1',type:'text',label:'姓名/编号',required:true},{id:'q2',type:'radio',label:'性别',options:['男','女'],required:true}]};}
export function roles(p:Paradigm):Role[]{return ({X:['X'],AX:['A','X'],ABX:['A','B','X'],AXB:['A','X','B']} as Record<Paradigm,Role[]>)[p];}
export function keyRange(p:Project,index:number){return p.config.keyRanges.find(r=>index>=r.start-1&&index<=r.end-1);}
export function keys(p:Project,index:number){const r=keyRange(p,index);return r?.keys.some(k=>k.key)?r.keys:p.config.validKeys;}
export function validRange(r:Range,length:number){return Number.isInteger(r.start)&&Number.isInteger(r.end)&&r.start>=1&&r.end>=r.start&&r.end<=length;}
export function validate(p:Project,requireAssets=true):string[]{
 const errors:string[]=[];
 if(p.schema!=='ptb-m15-project/1'||!roles(p.paradigm))return ['项目版本或范式无效'];
 if(!p.trials.length||p.trials.length>LIMITS.trials)errors.push('试次数须为 1–10000');
 for(const key of ['isi','interTrialInterval'] as const)if(!Number.isFinite(p.config[key])||p.config[key]<0||p.config[key]>3600000)errors.push(`${key} 须为 0–3600000 毫秒`);
 if(!['auto','manual'].includes(p.config.advanceMode))errors.push('推进模式无效');
 const checkKeys=(ks:Key[],label:string)=>{const seen=new Set<string>();if(!ks.length)errors.push(`${label}没有有效按键`);for(const k of ks){const v=k.key.toLowerCase();if(!/^[a-z0-9 ]$/.test(v)||seen.has(v))errors.push(`${label}按键须为不重复的英文字母、数字或空格`);seen.add(v);}};
 checkKeys(p.config.validKeys,'全局');p.config.keyRanges.forEach((r,i)=>{if(!validRange(r,p.trials.length))errors.push(`按键范围 ${i+1} 非法或越界`);if(r.keys.some(k=>k.key))checkKeys(r.keys,`分段 ${i+1}`);});
 p.config.shuffleRanges.forEach((r,i)=>{if(!validRange(r,p.trials.length))errors.push(`洗牌范围 ${i+1} 非法或越界`);});
 if(p.assets.length>LIMITS.files||p.assets.reduce((n,a)=>n+a.size,0)>LIMITS.totalBytes)errors.push('素材数量或总容量超预算');
 if(new Set(p.assets.map(a=>a.id)).size!==p.assets.length||new Set(p.trials.map(t=>t.id)).size!==p.trials.length)errors.push('素材或试次 ID 重复');
 p.questionnaire.forEach(q=>{if(!q.label.trim()||!['text','radio','checkbox'].includes(q.type))errors.push('问卷题目无效');if(q.type!=='text'&&(!q.options?.length||new Set(q.options).size!==q.options.length))errors.push(`问卷 ${q.label} 选项为空或重复`);});
 if(new Set(p.questionnaire.map(q=>q.id)).size!==p.questionnaire.length)errors.push('问卷 ID 重复');
 p.trials.forEach((t,i)=>roles(p.paradigm).forEach(r=>{const ref=t.stimuli[r],a=p.assets.find(a=>a.id===ref?.id);if(requireAssets&&(!a||!ref?.hash||a.hash!==ref.hash))errors.push(`试次 ${i+1} ${r} 缺素材或未明确关联`);if(a&&p.paradigm!=='X'&&a.kind!=='audio')errors.push(`试次 ${i+1} ${r}：复杂范式仅支持音频`);}));
 return errors;
}
export function validateAnswers(p:Project,a:Answers){for(const q of p.questionnaire){const v=a[q.id];if(q.required&&(!v||(Array.isArray(v)?v.length===0:!v.trim())))throw Error(`请填写：${q.label}`);if(v&&q.type!=='text'){const values=Array.isArray(v)?v:[v];if(values.some(x=>!q.options?.includes(x))||(q.type==='radio'&&Array.isArray(v)))throw Error(`问卷答案无效：${q.label}`);}}}
export function refFor(a:Asset):Stimulus{return {id:a.id,name:a.name,hash:a.hash};}
export function generate(p:Project):Trial[]{const groups={A:p.assets.filter(a=>a.group==='A'),B:p.assets.filter(a=>a.group==='B'),X:p.assets.filter(a=>a.group==='X')};return groups.X.map((x,i)=>{const old=p.trials.find(t=>t.stimuli.X?.id===x.id);return {id:old?.id??uid(),instruction:old?.instruction??'',stimuli:Object.fromEntries(roles(p.paradigm).map(r=>[r,groups[r][i]?refFor(groups[r][i]):{id:null,name:'',hash:null}]))};});}
export function shuffle(p:Project,seed:number,ranges:Range[]=p.config.shuffleRanges):Project {
 if(!Number.isInteger(seed)||seed<0||seed>0xffffffff)throw Error('种子须为 0–4294967295 的整数');
 if(ranges.some(r=>!validRange(r,p.trials.length)))throw Error('洗牌范围非法或越界');
 const out=copy(p),before=out.trials.map(t=>t.id);let state=seed>>>0;
 const random=()=>{state=(state+0x6D2B79F5)>>>0;let t=state;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};
 for(const r of ranges.length?ranges:[{start:1,end:out.trials.length}])for(let i=r.end-1;i>r.start-1;i--){const j=r.start-1+Math.floor(random()*(i-r.start+2));[out.trials[i],out.trials[j]]=[out.trials[j],out.trials[i]];}
 out.randomizations.push({algorithm:'mulberry32-v1',seed,ranges:copy(ranges),before,after:out.trials.map(t=>t.id)});return out;
}
export function isCleanKey(e:Pick<KeyboardEvent,'repeat'|'isComposing'|'ctrlKey'|'altKey'|'metaKey'|'shiftKey'|'key'>){return !e.repeat&&!e.isComposing&&!e.ctrlKey&&!e.altKey&&!e.metaKey&&!e.shiftKey&&e.key!=='Process'&&e.key!=='Dead'&&!['Control','Alt','Shift','Meta'].includes(e.key);}
export function kindFor(file:{name:string;type:string}):MediaKind {const ext=file.name.toLowerCase().split('.').pop();if(ext==='txt'||file.type==='text/plain')return 'text';if(['jpg','jpeg','png','gif','bmp','webp','svg'].includes(ext??'')||file.type.startsWith('image/'))return 'image';if(['mp3','wav','ogg','m4a','flac','aac','wma','mp4'].includes(ext??'')||file.type.startsWith('audio/'))return 'audio';throw Error(`不支持的素材类型：${file.name}`);}
export function relink(p:Project){for(const t of p.trials)for(const r of roles(p.paradigm)){const ref=t.stimuli[r];if(!ref)continue;const matches=ref.hash?p.assets.filter(a=>a.hash===ref.hash):[];if(matches.length===1)t.stimuli[r]=refFor(matches[0]);}return p;}
export function newSession(project:Project,answers:Answers):Session {validateAnswers(project,answers);const errors=validate(project);if(errors.length)throw Error(errors.join('\n'));return {schema:'ptb-m15-session/1',id:uid(),participantId:uid(),project:copy(project),answers:copy(answers),createdAt:new Date().toISOString(),revision:0,nextIndex:0,status:'ready',attempts:[],events:[],environment:{userAgent:navigator.userAgent,timeOrigin:performance.timeOrigin,method:'m15-client/1',rtClock:'performance.now',physicalTimingMeasured:false},exportRequestedRevision:null,exportConfirmedRevision:null};}
export function recover(s:Session):Session{const out=copy(s);for(const a of out.attempts)if(a.status==='running'){a.status='interrupted';a.reason='page-recovery';a.finishedAt=new Date().toISOString();a.rtMs=null;}if(out.status==='running')out.status='paused';out.events.push({kind:'recovery',at:new Date().toISOString(),perfMs:performance.now(),timeOrigin:performance.timeOrigin,detail:'只恢复已提交边界；不继续旧 RT'});return out;}
