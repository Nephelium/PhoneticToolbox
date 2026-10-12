import rules from './unt-rules.json' with {type:'json'};
import {normalizeReading as untReading} from './reading-repairs.ts';
export {normalizeReading as untReading} from './reading-repairs.ts';

export const untVersion=Object.freeze({version:rules.version,url:rules.url,accessDate:rules.access_date,revisionHistoryThrough:rules.revision_history_through,articleTextSha256:rules.article_text_sha256,selection:rules.selection});
export const isUnt=(standard:string)=>standard==='UntPhesoca宽'||standard==='UntPhesoca严';
const initials=rules.initials as Record<string,string[]>;
const finals=rules.finals as Record<string,string[]>;
const tones=['','˥','˧˥','˨˩˦','˥˩'];
const yFinals:Record<string,string>={yi:'i',ya:'ia',yo:'io',ye:'ie',yao:'iao',you:'iu',yan:'ian',yin:'in',yang:'iang',ying:'ing',yong:'iong',yu:'ü',yue:'üe',yuan:'üan',yun:'ün'};
const wFinals:Record<string,string>={wu:'u',wa:'ua',wo:'uo',wai:'uai',wei:'ui',wan:'uan',wen:'un',wang:'uang',weng:'ueng'};
// Shared reading corrections precede the standard-specific Unt rendering.
export function untSyllable(pinyin:string,tone:number,char=''){
  let p=untReading(pinyin,tone,char).pinyin.toLowerCase().replace(/v/g,'ü');
  if(!p)return null;
  // Numeric suffixes are historical malformed reading keys, with no reliable tone row.
  if(/\d/.test(p)||p==='ashu'||p==='be')return null;
  let initial='',final=yFinals[p]??wFinals[p]??p;
  if(!yFinals[p]&&!wFinals[p]&&!['er','m','n','ng'].includes(p)){
    initial=/^(zh|ch|sh|[bpmfdtnlgkhjqxrzcs])/u.exec(p)?.[0]??'';
    final=p.slice(initial.length);
  }
  if('jqx'.includes(initial)&&initial&&final.startsWith('u'))final='ü'+final.slice(1);
  if(final==='i'&&['z','c','s'].includes(initial))final='ï1';
  if(final==='i'&&['zh','ch','sh','r'].includes(initial))final='ï2';
  if(final==='u'&&initial==='f')final='fu';
  const coronal=['d','t','n','l','zh','ch','sh','r','z','c','s'].includes(initial);
  if(final==='ui')final=coronal?'ui2':'ui1';
  if(final==='un')final=coronal?'un2':'un1';
  if(final==='er')final=tone===4?'er2':'er1';
  return finals[final]&&(!initial||initials[initial])?{initial,final}:null;
}

export function untTranscription(pinyin:string,tone:number,narrow:boolean,char='',erhua=false):string|null{
  ({pinyin,tone}=untReading(pinyin,tone,char));
  const parsed=untSyllable(pinyin,tone,char);if(!parsed)return null;
  const {initial,final}=parsed,column=(erhua?2:0)+(narrow?1:0);
  if(erhua&&(rules.unsupported_erhua.includes(final)||finals[final].length<4))return null;
  let onset=initial?initials[initial][narrow?1:0]:'',rime=finals[final][column];
  if(!erhua&&tone===0&&['的','地','得','么','麼'].includes(char)&&['de','me'].includes(pinyin))rime='ə';
  if(narrow){
    if(rime.startsWith('◌ʷ')){rime=rime.slice(2);if(initial)onset+='ʷ';}
    else if(initial&&rime.startsWith('w'))onset+='ʷ';
    else if(initial&&rime.startsWith('ɥ'))onset+='ʷ';
    else if(initial&&!['j','q','x','t','d','n'].includes(initial)&&rime.startsWith('j'))onset+='ʲ';
    if(!initial&&!erhua){
      if(final==='i')rime='iᶨ';
      if(final==='u')rime='uᵝ';
      if(final==='ü')rime='yᶨ';
    }
  }
  const toneMark=narrow&&tone===3?'˨˩˧':tones[tone]??'';
  return onset+rime+toneMark;
}
