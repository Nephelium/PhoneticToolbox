import {LocalAssets,type AssetDatabase} from '../../platform/local-assets.ts';
import {validateToneSettings,tonePreview,type ToneSettings} from './tones.ts';

export type TableKind='hanzi'|'syllable';
export interface CustomTable {id:`custom:${string}`;name:string;kind:TableKind;rows:[string,string][];revision?:number;tones?:ToneSettings}
export interface TableLibrary {version:1;tables:CustomTable[];selected:CustomTable['id']|null}
export const emptyLibrary=():TableLibrary=>({version:1,tables:[],selected:null});
const maxRows=50000,maxBytes=8_000_000;

/** Numbered and marked Mandarin pinyin share a key; an unmarked syllable is a fallback. */
export function syllableKey(value:string){
  let text=value.trim().toLowerCase().replace(/u:/g,'ü').replace(/v/g,'ü'),tone:number|undefined;
  const numbered=/[0-5]$/.exec(text);if(numbered){tone=Number(numbered[0])%5;text=text.slice(0,-1);}
  let marked:number|undefined;
  text=text.normalize('NFD').replace(/[\u0304\u0301\u030c\u0300]/g,mark=>{
    const next=['\u0304','\u0301','\u030c','\u0300'].indexOf(mark)+1;
    if(marked!==undefined&&marked!==next)throw Error('同一音节含有不同声调。');marked=next;return '';
  }).normalize('NFC');
  if(!/^[a-züê]+$/.test(text)||tone!==undefined&&marked!==undefined&&tone!==marked)throw Error('请使用普通话拼音，例如 ma、ma1、mā 或 nü3。');
  tone??=marked;return text+(tone===undefined?'':tone);
}

export function parseDelimited(text:string,delimiter= text.split(/\r?\n/,1)[0]!.includes('\t')?'\t':','){
  text=text.replace(/^\uFEFF/,'');const rows:string[][]=[];let row:string[]=[],cell='',quoted=false,closed=false;
  for(let i=0;i<text.length;i++){
    const ch=text[i]!;
    if(ch==='"'){
      if(quoted&&text[i+1]==='"'){cell+='"';i++;}
      else if(quoted){quoted=false;closed=true;}
      else if(!cell&&!closed)quoted=true;else throw Error('表格引号格式有误，请检查 CSV 文件。');
    }else if(!quoted&&(ch===delimiter||ch==='\r'||ch==='\n')){
      row.push(cell);cell='';closed=false;
      if(ch!==delimiter){if(ch==='\r'&&text[i+1]==='\n')i++;rows.push(row);row=[];}
    }else{if(closed&&!/\s/.test(ch))throw Error('CSV 引号后有多余内容。');if(!closed)cell+=ch;}
    if(rows.length>maxRows+1)throw Error('字表最多支持 50,000 行。');
  }
  if(quoted)throw Error('CSV 引号未闭合。');if(cell||row.length||closed){row.push(cell);rows.push(row);}return rows;
}

export interface ImportColumns {keyColumn:number;ipaColumn:number;startRow:number}
const keyHeader=/^(汉字|字|字头|音节|拼音|hanzi|character|syllable|pinyin)$/i,ipaHeader=/^(国际音标|音标|ipa)$/i;
export function suggestColumns(input:readonly (readonly string[])[]):ImportColumns{
  for(let i=0;i<Math.min(input.length,20);i++){
    const cells=input[i]!.map(c=>c.trim()),keyColumn=cells.findIndex(c=>keyHeader.test(c)),ipaColumn=cells.findIndex(c=>ipaHeader.test(c));
    if(keyColumn>=0&&ipaColumn>=0)return {keyColumn,ipaColumn,startRow:i+2};
  }
  return {keyColumn:0,ipaColumn:1,startRow:1};
}
export function columnLabel(index:number){let label='';for(let n=index+1;n>0;n=Math.floor((n-1)/26))label=String.fromCharCode(65+(n-1)%26)+label;return label;}
export function tableRows(input:readonly (readonly string[])[],kind:TableKind,columns?:ImportColumns){
  if(columns&&(!Number.isInteger(columns.keyColumn)||!Number.isInteger(columns.ipaColumn)||columns.keyColumn<0||columns.ipaColumn<0||columns.keyColumn===columns.ipaColumn))throw Error('请选择不同的汉字／音节列和音标列。');
  if(columns&&(!Number.isInteger(columns.startRow)||columns.startRow<1||columns.startRow>input.length))throw Error('数据起始行须在表格范围内。');
  const rows:[string,string][]=[];const seen=new Set<string>();let first=true;
  for(let i=columns?columns.startRow-1:0;i<input.length;i++){
    const raw=input[i]!,cells=(columns?[raw[columns.keyColumn]??'',raw[columns.ipaColumn]??'']:raw).map(value=>value.trim());if(cells.every(value=>!value))continue;
    if(first&&!columns){first=false;if(keyHeader.test(cells[0]??'')&&ipaHeader.test(cells[1]??''))continue;}
    if(cells.length<2||cells.slice(2).some(Boolean)||!cells[0]||!cells[1])throw Error(`第 ${i+1} 行需要两列：${kind==='hanzi'?'汉字':'拼音音节'}和国际音标。`);
    let key=cells[0];
    if(kind==='hanzi'){if(!/^\p{Script=Han}$/u.test(key))throw Error(`第 ${i+1} 行汉字列须为单个汉字。`);}
    else try{key=syllableKey(key);}catch(error){throw Error(`第 ${i+1} 行：${(error as Error).message}`);}
    if(cells[1].length>256||/[\u0000-\u001f\u007f]/.test(cells[1]))throw Error(`第 ${i+1} 行音标过长或含有控制字符。`);
    const pair=JSON.stringify([key,cells[1]]);if(!seen.has(pair)){rows.push([key,cells[1]]);seen.add(pair);}
    if(rows.length>maxRows)throw Error('字表最多支持 50,000 行。');
  }
  if(!rows.length)throw Error('字表没有可用的对照记录。');return rows;
}
export function tableName(value:string){const name=value.trim();if(!name||[...name].length>60||/[\u0000-\u001f\u007f]/.test(name))throw Error('请填写 1–60 个字符的字表名称。');return name;}
export function restoreLibrary(raw:unknown):TableLibrary{
  if(raw==null)return emptyLibrary();
  try{
    const value=JSON.parse(String(raw)) as TableLibrary;
    if(value.version!==1||!Array.isArray(value.tables)||value.tables.length>32)throw Error();
    const ids=new Set<string>(),names=new Set<string>();
    for(const table of value.tables){
      if(!/^custom:[\w-]+$/.test(table.id)||ids.has(table.id)||!['hanzi','syllable'].includes(table.kind)||tableName(table.name)!==table.name||names.has(table.name)||!Array.isArray(table.rows)||table.rows.some(r=>!Array.isArray(r)||r.length!==2||r.some(c=>typeof c!=='string')))throw Error();
      table.rows=tableRows(table.rows,table.kind);ids.add(table.id);names.add(table.name);
      if(table.revision!==undefined&&(!Number.isSafeInteger(table.revision)||table.revision<1))throw Error();
      if(table.tones!==undefined){table.tones=validateToneSettings(table.tones);tonePreview(table.rows,table.tones);}
    }
    if(value.selected!==null&&!ids.has(value.selected))throw Error();return value;
  }catch{throw Error('已保存的自定义字表无法读取，请保留原文件并重新打开页面重试。');}
}
export function addTable(library:TableLibrary,table:CustomTable):TableLibrary{
  const name=tableName(table.name);if(library.tables.some(t=>t.name===name))throw Error('已有同名字表，请换一个名称。');
  if(library.tables.length>=32)throw Error('当前最多保存 32 套自定义字表。');
  const added={...table,name,rows:tableRows(table.rows,table.kind),revision:1};
  if(table.tones!==undefined){added.tones=validateToneSettings(table.tones);tonePreview(added.rows,added.tones);}
  return {...library,tables:[...library.tables,added],selected:table.id};
}
export function replaceTable(library:TableLibrary,id:CustomTable['id'],table:Pick<CustomTable,'name'|'kind'|'rows'|'tones'>):TableLibrary{
  const previous=library.tables.find(t=>t.id===id);if(!previous)throw Error('字表已不存在，请重新载入已存字表。');
  const name=tableName(table.name);if(library.tables.some(t=>t.id!==id&&t.name===name))throw Error('已有同名字表，请换一个名称。');
  const replacement:CustomTable={id,name,kind:table.kind,rows:tableRows(table.rows,table.kind),revision:(previous.revision??1)+1};
  if(table.tones!==undefined){replacement.tones=validateToneSettings(table.tones);tonePreview(replacement.rows,replacement.tones);}
  return {...library,tables:library.tables.map(t=>t.id===id?replacement:t),selected:id};
}
export function removeTable(library:TableLibrary,id:CustomTable['id']):TableLibrary{
  if(!library.tables.some(t=>t.id===id))throw Error('字表已不存在，请重新载入已存字表。');
  return {...library,tables:library.tables.filter(t=>t.id!==id),selected:library.selected===id?null:library.selected};
}
export class CustomTableStore {
  private assets:LocalAssets;private raw:string|null=null;
  constructor(scope:string,database?:AssetDatabase){const owner=/^server:([^:]+):/.exec(scope)?.[1]??'desktop';this.assets=new LocalAssets(owner,'m13-custom-tables',database);}
  async load(){const raw=await this.assets.value<string>('library-v1')??null;const library=restoreLibrary(raw);this.raw=raw;return library;}
  async save(library:TableLibrary){
    const raw=JSON.stringify(library);if(new TextEncoder().encode(raw).length>maxBytes)throw Error('自定义字表总量超过 8 MB，请缩小字表。');
    try{await this.assets.compare('library-v1',this.raw,raw);}catch{throw Error('字表保存失败，可能已在另一窗口更新或本机空间不足。请重新载入已存字表后重试。');}this.raw=raw;
  }
}
