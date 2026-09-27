import rawData from './ipa-data.json' with {type:'json'};

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
export type IpaStandard=typeof standards[number];
export type DisplayMode='paired'|'ipa-only';
export type PageLayout='side-by-side'|'stacked';

type CompactRow=Array<string|number|null>;
interface RawData {
  schema_version:string;source_path:string;source_sha256:string;row_count:number;
  unique_character_count:number;duplicate_character_count:number;max_variants:number;
  columns:string[];rows:CompactRow[];
}
export interface IpaEntry {tone:number;pinyin:string;values:readonly (string|null)[]}
export interface MappedToken {kind:'mapped';index:number;char:string;value:string;entry:IpaEntry;variants:readonly IpaEntry[];selectedVariant:number}
export interface LiteralToken {kind:'literal';index:number;char:string;newline:boolean}
export type ConversionToken=MappedToken|LiteralToken;
export interface VariantOption {index:number;pinyin:string;tone:number;toneLabel:string;value:string}
export interface MandarinIpaDraft {
  version:1;text:string;standard:IpaStandard;display:DisplayMode;layout:PageLayout;
  hanziSize:number;ipaSize:number;ipaSizeUserSet:boolean;gap:number;lineHeight:number;
  bold:boolean;italic:boolean;underline:boolean;selectedVariants:Record<string,number>;
}

const data=rawData as RawData;
if(data.schema_version!=='m13-ipa-data/1'||data.row_count!==data.rows.length||data.columns.length!==13)throw Error('M13 IPA 映射资源不完整。');
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

export function displayEntry(entry:IpaEntry,standard:IpaStandard){
  if(standard==='汉语拼音')return entry.tone===0?entry.pinyin:addToneMark(entry.pinyin,entry.tone);
  const value=entry.values[standardColumn.get(standard)!];return value||'?';
}
export const variantKey=(char:string,position:number)=>`${char}_${position}`;
export function convertText(text:string,standard:IpaStandard,selectedVariants:Readonly<Record<string,number>>={}):ConversionToken[]{
  const result:ConversionToken[]=[];
  [...text].forEach((char,position)=>{
    const variants=index.get(char);
    if(!variants?.length){result.push({kind:'literal',index:position,char,newline:char==='\n'});return;}
    const requested=selectedVariants[variantKey(char,position)],selected=Number.isInteger(requested)&&requested>=0&&requested<variants.length?requested:0;
    const entry=variants[selected];result.push({kind:'mapped',index:position,char,value:displayEntry(entry,standard),entry,variants,selectedVariant:selected});
  });
  return result;
}
export function variantsFor(token:MappedToken,standard:IpaStandard):VariantOption[]{
  return token.variants.map((entry,index)=>({index,pinyin:entry.pinyin,tone:entry.tone,toneLabel:entry.tone===0?'轻声':String(entry.tone),value:displayEntry(entry,standard)}));
}
export function plainOutput(tokens:readonly ConversionToken[]){return tokens.map(token=>token.kind==='mapped'?token.value:token.char).join('');}
export function effectiveIpaSize(draft:Pick<MandarinIpaDraft,'display'|'ipaSize'|'ipaSizeUserSet'>){return draft.display==='ipa-only'&&!draft.ipaSizeUserSet?28:draft.ipaSize;}

export function createDraft():MandarinIpaDraft{return {version:1,text:'',standard:'Standard Chinese (Beijing)严',display:'paired',layout:'side-by-side',hanziSize:24,ipaSize:16,ipaSizeUserSet:false,gap:0,lineHeight:1.8,bold:false,italic:false,underline:false,selectedVariants:{}};}
export function snapshotDraft(draft:MandarinIpaDraft):MandarinIpaDraft{return {...draft,selectedVariants:{...draft.selectedVariants}};}
const number=(value:unknown,fallback:number,min:number,max:number)=>typeof value==='number'&&Number.isFinite(value)&&value>=min&&value<=max?value:fallback;
export function restoreDraft(value:unknown):MandarinIpaDraft{
  const draft=createDraft();if(!value||typeof value!=='object'||Array.isArray(value))return draft;const saved=value as Record<string,unknown>;if(saved.version!==1)return draft;
  if(typeof saved.text==='string')draft.text=saved.text;
  if(typeof saved.standard==='string'&&standards.includes(saved.standard as IpaStandard))draft.standard=saved.standard as IpaStandard;
  if(saved.display==='paired'||saved.display==='ipa-only')draft.display=saved.display;
  if(saved.layout==='side-by-side'||saved.layout==='stacked')draft.layout=saved.layout;
  draft.hanziSize=number(saved.hanziSize,draft.hanziSize,16,72);draft.ipaSize=number(saved.ipaSize,draft.ipaSize,12,48);
  draft.gap=number(saved.gap,draft.gap,-12,20);draft.lineHeight=number(saved.lineHeight,draft.lineHeight,.8,3);
  for(const key of ['ipaSizeUserSet','bold','italic','underline'] as const)if(typeof saved[key]==='boolean')draft[key]=saved[key];
  if(saved.selectedVariants&&typeof saved.selectedVariants==='object'&&!Array.isArray(saved.selectedVariants))for(const [key,entry] of Object.entries(saved.selectedVariants))if(/^._\d+$/u.test(key)&&Number.isInteger(entry)&&Number(entry)>=0&&Number(entry)<=6)draft.selectedVariants[key]=Number(entry);
  return draft;
}
