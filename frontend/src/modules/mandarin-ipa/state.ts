import rawData from './ipa-data.json' with {type:'json'};
import {validFamily} from '../../design/fonts.ts';
import {isUnt,untReading,untTranscription} from './unt.ts';
import {syllableKey,type CustomTable,type TableLibrary} from './custom-tables.ts';
import {transcribeTone} from './tones.ts';

export const mappingStandards=[
  'Standard Chinese (Beijing)',
  'Standard Chinese (Beijing)严',
  '胡裕树《现代汉语》',
  '黄伯荣、廖序东《现代汉语》',
  '钱乃荣《现代汉语》',
  '吴宗济',
  '赵元任《汉语口语语法》',
  '《汉语方音字汇》',
  'UntPhesoca宽',
  'UntPhesoca严',
] as const;
export const standards=[...mappingStandards,'汉语拼音'] as const;
export type MappingStandard=typeof mappingStandards[number];
export type IpaStandard=typeof standards[number]|CustomTable['id'];
export type DisplayMode='paired'|'ipa-only';
export type PageLayout='side-by-side'|'stacked';

type CompactRow=Array<string|number|null>;
interface RawData {
  schema_version:string;source_path:string;source_sha256:string;row_count:number;
  unique_character_count:number;duplicate_character_count:number;max_variants:number;
  columns:string[];rows:CompactRow[];
}
export interface IpaEntry {tone:number;pinyin:string;values:readonly (string|null)[];customValue?:string;customBareValue?:string}
export interface MappedToken {kind:'mapped';index:number;char:string;value:string;entry:IpaEntry;variants:readonly IpaEntry[];selectedVariant:number;erhua?:boolean;baseChar?:string;canCombine?:boolean}
export interface LiteralToken {kind:'literal';index:number;char:string;newline:boolean}
export type ConversionToken=MappedToken|LiteralToken;
export interface VariantOption {index:number;pinyin:string;tone:number;toneLabel:string;value:string}
export interface MandarinIpaDraft {
  version:1;text:string;standard:IpaStandard;display:DisplayMode;layout:PageLayout;
  hanziSize:number;ipaSize:number;ipaSizeUserSet:boolean;gap:number;lineHeight:number;
  bold:boolean;italic:boolean;underline:boolean;selectedVariants:Record<string,number>;
  showTone:boolean;ipaColor:string;hanziColor:string;hanziFont:string;erhuaPairs:Record<string,boolean>;equalWidth:boolean;
  tableRevision?:number;
}

const data=rawData as RawData;
if(data.schema_version!=='m13-ipa-data/1'||data.row_count!==data.rows.length||data.columns.length!==13)throw Error('汉字转国际音标的映射资源不完整。');
const standardColumn=new Map<MappingStandard,number>(mappingStandards.map((name,index)=>[name,index]));
const index=new Map<string,IpaEntry[]>();
for(const row of data.rows){
  const char=String(row[0]??''),entry:IpaEntry={tone:Number(row[1]),pinyin:String(row[2]??''),values:row.slice(3).map(value=>value==null?null:String(value))};
  const entries=index.get(char);if(entries)entries.push(entry);else index.set(char,[entry]);
}

export const dataInventory=Object.freeze({
  schemaVersion:data.schema_version,sourcePath:data.source_path,sourceSha256:data.source_sha256,
  rows:data.row_count,uniqueCharacters:data.unique_character_count,
  duplicateCharacters:data.duplicate_character_count,maxVariants:data.max_variants,
});

const toneMarks:Record<string,readonly string[]>={
  a:['ā','á','ǎ','à','a'],e:['ē','é','ě','è','e'],i:['ī','í','ǐ','ì','i'],
  o:['ō','ó','ǒ','ò','o'],u:['ū','ú','ǔ','ù','u'],'ü':['ǖ','ǘ','ǚ','ǜ','ü'],v:['ǖ','ǘ','ǚ','ǜ','ü'],
};

export function addToneMark(pinyin:string,tone:number){
  if(!pinyin||tone<1||tone>5||tone===0)return pinyin;
  const lower=pinyin.toLowerCase();let target=-1;
  for(let i=0;i<lower.length;i++)if(lower[i]==='a'||lower[i]==='e'){target=i;break;}
  if(target<0){const ou=lower.indexOf('ou');if(ou>=0)target=ou;}
  if(target<0)for(let i=lower.length-1;i>=0;i--)if('aeiouüv'.includes(lower[i])){target=i;break;}
  if(target<0)return pinyin;const mark=toneMarks[lower[target]]?.[tone-1];
  return mark?pinyin.slice(0,target)+mark+pinyin.slice(target+1):pinyin;
}

export function displayEntry(entry:IpaEntry,standard:IpaStandard,showTone=true){
  if(standard.startsWith('custom:')){const value=entry.customValue??'?';return showTone?value:entry.customBareValue??value.replace(/[˥˦˧˨˩]+$/u,'');}
  if(standard==='汉语拼音'){const pinyin=entry.pinyin.replace(/v/g,'ü');return showTone&&entry.tone!==0?addToneMark(pinyin,entry.tone):pinyin;}
  const value=entry.values[standardColumn.get(standard as MappingStandard)!]||'?';return showTone?value:value.replace(/[˥˦˧˨˩]+$/u,'');
}
const customIndexes=new WeakMap<CustomTable,Map<string,IpaEntry[]>>();
function customIndex(table:CustomTable){
  const cached=customIndexes.get(table);if(cached)return cached;
  const lookup=new Map<string,string[]>();for(const [key,value] of table.rows){const values=lookup.get(key)??[];values.push(value);lookup.set(key,values);}
  const converted=new Map<string,IpaEntry[]>();
  const transcription=(value:string)=>({customValue:transcribeTone(value,table.tones),customBareValue:transcribeTone(value,table.tones,false)});
  if(table.kind==='hanzi')for(const [char,values] of lookup)converted.set(char,values.map(value=>({tone:0,pinyin:'',values:[],...transcription(value)})));
  else for(const [char,entries] of index)converted.set(char,entries.flatMap(entry=>{
    const base=syllableKey(entry.pinyin),values=lookup.get(base+entry.tone)||lookup.get(base)||['?'];
    return values.map(value=>({...entry,...transcription(value)}));
  }));
  customIndexes.set(table,converted);return converted;
}
export const variantKey=(char:string,position:number)=>`${char}_${position}`;
export function convertText(text:string,standard:IpaStandard,selectedVariants:Readonly<Record<string,number>>={},showTone=true,erhuaPairs:Readonly<Record<string,boolean>>={},table?:CustomTable):ConversionToken[]{
  const result:ConversionToken[]=[];
  const custom=standard.startsWith('custom:'),lookup=custom&&table?.id===standard?customIndex(table):custom?new Map<string,IpaEntry[]>():index;
  [...text].forEach((char,position)=>{
    const variants=lookup.get(char)??(custom&&/^\p{Script=Han}$/u.test(char)?[{tone:0,pinyin:'',values:[],customValue:'?'}]:undefined);
    if(!variants?.length){result.push({kind:'literal',index:position,char,newline:char==='\n'});return;}
    const requested=selectedVariants[variantKey(char,position)],selected=Number.isInteger(requested)&&requested>=0&&requested<variants.length?requested:0;
    const entry=variants[selected];result.push({kind:'mapped',index:position,char,value:displayEntry(entry,standard,showTone),entry,variants,selectedVariant:selected});
  });
  if(!isUnt(standard))return result;
  const combined:ConversionToken[]=[];
  for(let i=0;i<result.length;i++){
    const token=result[i],next=result[i+1];
    if(token.kind==='mapped'&&next?.kind==='mapped'&&['儿','兒'].includes(next.char)&&next.entry.pinyin==='er'&&next.entry.tone===2){
      const value=untTranscription(token.entry.pinyin,token.entry.tone,standard==='UntPhesoca严',token.char,true);
      if(value){
        token.canCombine=true;
        if(erhuaPairs[variantKey(token.char,token.index)]){
          combined.push({...token,char:token.char+next.char,baseChar:token.char,erhua:true,value:showTone?value:value.replace(/[˥˦˧˨˩]+$/u,'')});i++;continue;
        }
      }
    }
    combined.push(token);
  }
  return combined;
}
export function variantsFor(token:MappedToken,standard:IpaStandard,showTone=true):VariantOption[]{
  return token.variants.map((entry,index)=>{
    const erhua=token.erhua?untTranscription(entry.pinyin,entry.tone,standard==='UntPhesoca严',token.baseChar,true):null;
    const value=erhua?(showTone?erhua:erhua.replace(/[˥˦˧˨˩]+$/u,'')):displayEntry(entry,standard,showTone);
    const reading=isUnt(standard)?untReading(entry.pinyin,entry.tone,token.baseChar??token.char):entry;
    return {index,pinyin:reading.pinyin,tone:reading.tone,toneLabel:reading.tone===0?'轻声':String(reading.tone),value};
  });
}
export function autoErhuaPairs(text:string,standard:IpaStandard,choices:Readonly<Record<string,number>>={}){
  const pairs:Record<string,boolean>={};
  const tokens=convertText(text,standard,choices);
  for(let i=0;i<tokens.length;i++){
    const token=tokens[i];if(token.kind==='mapped'&&token.canCombine){pairs[variantKey(token.char,token.index)]=true;i++;}
  }
  return pairs;
}
export function moveErhuaPairs(previous:string,next:string,pairs:Readonly<Record<string,boolean>>){
  const before=[...previous],after=[...next];let prefix=0,suffix=0;
  while(prefix<Math.min(before.length,after.length)&&before[prefix]===after[prefix])prefix++;
  while(suffix<Math.min(before.length,after.length)-prefix&&before[before.length-1-suffix]===after[after.length-1-suffix])suffix++;
  const moved:Record<string,boolean>={};
  for(let old=0;old<before.length-1;old++){
    if(!pairs[variantKey(before[old],old)])continue;
    const position=old+1<prefix?old:old>=before.length-suffix?old+after.length-before.length:-1;
    if(position>=0&&before[old]===after[position]&&before[old+1]===after[position+1])moved[variantKey(after[position],position)]=true;
  }
  return moved;
}
export function plainOutput(tokens:readonly ConversionToken[]){return tokens.map(token=>token.kind==='mapped'?token.value:token.char).join('');}
export function effectiveIpaSize(draft:Pick<MandarinIpaDraft,'display'|'ipaSize'|'ipaSizeUserSet'>){return draft.display==='ipa-only'&&!draft.ipaSizeUserSet?28:draft.ipaSize;}

export function createDraft():MandarinIpaDraft{return {version:1,text:'',standard:'Standard Chinese (Beijing)',display:'paired',layout:'side-by-side',hanziSize:24,ipaSize:16,ipaSizeUserSet:false,gap:0,lineHeight:1.8,bold:false,italic:false,underline:false,selectedVariants:{},showTone:true,ipaColor:'',hanziColor:'',hanziFont:'',erhuaPairs:{},equalWidth:false};}
export function createPageDraft(saved:unknown):MandarinIpaDraft{const draft={...restoreDraft(saved),text:'',standard:createDraft().standard,selectedVariants:{},erhuaPairs:{}};delete draft.tableRevision;return draft;}
// Keep choices attached to unchanged Unicode characters before/after an edit.
export function moveVariantChoices(previous:string,next:string,choices:Readonly<Record<string,number>>){
  const before=[...previous],after=[...next];let prefix=0,suffix=0;
  while(prefix<Math.min(before.length,after.length)&&before[prefix]===after[prefix])prefix++;
  while(suffix<Math.min(before.length,after.length)-prefix&&before[before.length-1-suffix]===after[after.length-1-suffix])suffix++;
  const moved:Record<string,number>={};
  for(let old=0;old<before.length;old++){
    const choice=choices[variantKey(before[old],old)];if(choice===undefined)continue;
    const position=old<prefix?old:old>=before.length-suffix?old+after.length-before.length:-1;
    if(position>=0)moved[variantKey(after[position],position)]=choice;
  }
  return moved;
}
export function snapshotDraft(draft:MandarinIpaDraft):MandarinIpaDraft{return {...draft,selectedVariants:{...draft.selectedVariants},erhuaPairs:{...draft.erhuaPairs}};}
const number=(value:unknown,fallback:number,min:number,max:number)=>typeof value==='number'&&Number.isFinite(value)&&value>=min&&value<=max?value:fallback;
export function restoreDraft(value:unknown):MandarinIpaDraft{
  const draft=createDraft();if(!value||typeof value!=='object'||Array.isArray(value))return draft;const saved=value as Record<string,unknown>;if(saved.version!==1)return draft;
  if(typeof saved.text==='string')draft.text=saved.text;
  if(typeof saved.standard==='string'&&(standards.some(s=>s===saved.standard)||/^custom:[\w-]+$/.test(saved.standard)))draft.standard=saved.standard as IpaStandard;
  if(Number.isSafeInteger(saved.tableRevision)&&Number(saved.tableRevision)>=1)draft.tableRevision=Number(saved.tableRevision);
  if(saved.display==='paired'||saved.display==='ipa-only')draft.display=saved.display;
  if(saved.layout==='side-by-side'||saved.layout==='stacked')draft.layout=saved.layout;
  draft.hanziSize=number(saved.hanziSize,draft.hanziSize,16,72);draft.ipaSize=number(saved.ipaSize,draft.ipaSize,12,48);
  draft.gap=number(saved.gap,draft.gap,-12,20);draft.lineHeight=number(saved.lineHeight,draft.lineHeight,.8,3);
  for(const key of ['ipaSizeUserSet','bold','italic','underline','showTone','equalWidth'] as const)if(typeof saved[key]==='boolean')draft[key]=saved[key];
  for(const key of ['ipaColor','hanziColor'] as const)if(typeof saved[key]==='string'&&/^#[\da-f]{6}$/i.test(saved[key]))draft[key]=saved[key];
  if(validFamily(saved.hanziFont))draft.hanziFont=saved.hanziFont.trim();
  if(saved.selectedVariants&&typeof saved.selectedVariants==='object'&&!Array.isArray(saved.selectedVariants))for(const [key,entry] of Object.entries(saved.selectedVariants))if(/^._\d+$/u.test(key)&&Number.isInteger(entry)&&Number(entry)>=0&&Number(entry)<50000)draft.selectedVariants[key]=Number(entry);
  if(saved.erhuaPairs&&typeof saved.erhuaPairs==='object'&&!Array.isArray(saved.erhuaPairs)){
    const chars=[...draft.text];
    for(const [key,enabled] of Object.entries(saved.erhuaPairs)){
      const match=/^(.+)_(\d+)$/u.exec(key);if(!match||enabled!==true)continue;
      const position=Number(match[2]);
      if(chars[position]===match[1]&&['儿','兒'].includes(chars[position+1]))draft.erhuaPairs[key]=true;
    }
  }
  return draft;
}
export function restoreTableDraft(value:unknown,library:TableLibrary,fallback:IpaStandard=createDraft().standard){
  const draft=restoreDraft(value);let notice='';
  if(draft.standard.startsWith('custom:')){
    const table=library.tables.find(t=>t.id===draft.standard);
    if(!table){
      draft.standard=standards.some(s=>s===fallback)||library.tables.some(t=>t.id===fallback)?fallback:createDraft().standard;
      draft.selectedVariants={};draft.erhuaPairs={};delete draft.tableRevision;
      const current=library.tables.find(t=>t.id===draft.standard);if(current)draft.tableRevision=current.revision??1;
      notice='草稿使用的字表已删除或不可用，文本和排版已恢复，请重新选择字表并核对读音。';
    }else{
      if((draft.tableRevision??1)!==(table.revision??1)){draft.selectedVariants={};draft.erhuaPairs={};notice='字表已替换，草稿文本和排版已恢复，请重新核对多音字读音。';}
      draft.tableRevision=table.revision??1;
    }
  }
  return {draft,notice};
}
