import {LocalAssets,type AssetDatabase} from '../../platform/local-assets.ts';
import {parseGrid} from './format.ts';

export type HistoryKind='original'|'recovered'|'auto'|'manual'|'restored';
export interface AnnotationView {offset:number;zoom:number;channel:number;word:string;phone:string;extra?:string;lab:string[];labName:string;sequence:boolean;index:number}
export interface AnnotationVersion {id:string;time:number;kind:HistoryKind;text:string;fileName?:string}
interface Journal {schema:1;audioSha:string;versions:AnnotationVersion[];head:string;view?:AnnotationView}
export interface AnnotationHistorySession {key:string;raw:string|null;value:Journal}
export class AnnotationHistory {
 private local:LocalAssets;
 constructor(owner:string,project:string,database?:AssetDatabase){this.local=new LocalAssets(owner,'annotation-history:'+project,database);}
 async open(identity:string,audioSha:string,original:string,working=original):Promise<AnnotationHistorySession>{
  const key='audio:'+JSON.stringify([identity,audioSha]),raw=await this.local.value<string>(key)??null;
  if(raw!==null){
   let value:Journal;try{value=JSON.parse(raw);}catch{throw Error('标注历史无法读取，请保留当前编辑并导出 TextGrid。');}
   if(value.schema!==1||value.audioSha!==audioSha||!Array.isArray(value.versions)||!value.versions.length||!value.versions.some(v=>v.id===value.head))throw Error('标注历史不完整，请保留当前编辑并导出 TextGrid。');
   for(const v of value.versions)parseGrid(v.text,true);
   return {key,raw,value};
  }
  parseGrid(original,true);parseGrid(working,true);
  const first=this.version(original,'original'),versions=[first];if(working!==original)versions.push(this.version(working,'recovered'));
  return {key,raw:null,value:{schema:1,audioSha,versions,head:versions.at(-1)!.id}};
 }
 head(session:AnnotationHistorySession){return session.value.versions.find(v=>v.id===session.value.head)!;}
 private version(text:string,kind:HistoryKind,fileName?:string):AnnotationVersion{return {id:crypto.randomUUID(),time:Date.now(),kind,text,fileName};}
 async save(session:AnnotationHistorySession,text:string,kind:HistoryKind,view:AnnotationView,fileName?:string){
  parseGrid(text,true);const previous=session.value,head=this.head(session);
  let versions=previous.versions;
  const entry=head.text===text?head:this.version(text,kind,fileName);
  if(entry!==head)versions=[versions[0],...versions.slice(1),entry].filter((v,i,a)=>i===0||i>=a.length-50);
  const value:Journal={...previous,versions,head:entry.id,view:structuredClone(view)},raw=JSON.stringify(value);
  try{await this.local.compare(session.key,session.raw,raw);}catch{throw Error('标注历史未能保存，可能已在另一窗口更新或本机空间不足。当前编辑仍保留，请重试或导出 TextGrid。');}
  session.raw=raw;session.value=value;
 }
}
