import {uiErrorMessage} from '../module-localization.ts';
import {uiError as displayError,uiMessage as displayMessage} from '../module-localization.ts';
import {tr} from '../../i18n/index.ts';
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
    if(marked!==undefined&&marked!==next)throw displayError(displayMessage('modules.mandarin-ipa.text.e37d634a7884'));marked=next;return '';
  }).normalize('NFC');
  if(!/^[a-züê]+$/.test(text)||tone!==undefined&&marked!==undefined&&tone!==marked)throw displayError(displayMessage('modules.mandarin-ipa.text.0b0871172f43'));
  tone??=marked;return text+(tone===undefined?'':tone);
}

export function parseDelimited(text:string,delimiter= text.split(/\r?\n/,1)[0]!.includes('\t')?'\t':','){
  text=text.replace(/^\uFEFF/,'');const rows:string[][]=[];let row:string[]=[],cell='',quoted=false,closed=false;
  for(let i=0;i<text.length;i++){
    const ch=text[i]!;
    if(ch==='"'){
      if(quoted&&text[i+1]==='"'){cell+='"';i++;}
      else if(quoted){quoted=false;closed=true;}
      else if(!cell&&!closed)quoted=true;else throw displayError(displayMessage('modules.mandarin-ipa.text.0f336109ebd1'));
    }else if(!quoted&&(ch===delimiter||ch==='\r'||ch==='\n')){
      row.push(cell);cell='';closed=false;
      if(ch!==delimiter){if(ch==='\r'&&text[i+1]==='\n')i++;rows.push(row);row=[];}
    }else{if(closed&&!/\s/.test(ch))throw displayError(displayMessage('modules.mandarin-ipa.text.e64fcc03a291'));if(!closed)cell+=ch;}
    if(rows.length>maxRows+1)throw displayError(displayMessage('modules.mandarin-ipa.text.e75c2bcbd15d'));
  }
  if(quoted)throw displayError(displayMessage('modules.mandarin-ipa.text.bde0ba8cf0d4'));if(cell||row.length||closed){row.push(cell);rows.push(row);}return rows;
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
  if(columns&&(!Number.isInteger(columns.keyColumn)||!Number.isInteger(columns.ipaColumn)||columns.keyColumn<0||columns.ipaColumn<0||columns.keyColumn===columns.ipaColumn))throw displayError(displayMessage('modules.mandarin-ipa.text.6093403ac4a8'));
  if(columns&&(!Number.isInteger(columns.startRow)||columns.startRow<1||columns.startRow>input.length))throw displayError(displayMessage('modules.mandarin-ipa.text.83103111355e'));
  const rows:[string,string][]=[];const seen=new Set<string>();let first=true;
  for(let i=columns?columns.startRow-1:0;i<input.length;i++){
    const raw=input[i]!,cells=(columns?[raw[columns.keyColumn]??'',raw[columns.ipaColumn]??'']:raw).map(value=>value.trim());if(cells.every(value=>!value))continue;
    if(first&&!columns){first=false;if(keyHeader.test(cells[0]??'')&&ipaHeader.test(cells[1]??''))continue;}
    if(cells.length<2||cells.slice(2).some(Boolean)||!cells[0]||!cells[1])throw displayError(displayMessage('modules.mandarin-ipa.text.3ca8022f1e35',{v0:i+1,v1:kind==='hanzi'?tr('modules.mandarin-ipa.text.00e5d5601ce3'):tr('modules.mandarin-ipa.text.68da7a474f1f')}));
    let key=cells[0];
    if(kind==='hanzi'){if(!/^\p{Script=Han}$/u.test(key))throw displayError(displayMessage('modules.mandarin-ipa.text.7d5a56e6026e',{v0:i+1}));}
    else try{key=syllableKey(key);}catch(error){throw displayError(displayMessage('modules.mandarin-ipa.text.640f0bc5148e',{v0:i+1,v1:(error as Error).message}));}
    if(cells[1].length>256||/[\u0000-\u001f\u007f]/.test(cells[1]))throw displayError(displayMessage('modules.mandarin-ipa.text.bdb8f0531b22',{v0:i+1}));
    const pair=JSON.stringify([key,cells[1]]);if(!seen.has(pair)){rows.push([key,cells[1]]);seen.add(pair);}
    if(rows.length>maxRows)throw displayError(displayMessage('modules.mandarin-ipa.text.e75c2bcbd15d'));
  }
  if(!rows.length)throw displayError(displayMessage('modules.mandarin-ipa.text.693d538e1b90'));return rows;
}
export function tableName(value:string){const name=value.trim();if(!name||[...name].length>60||/[\u0000-\u001f\u007f]/.test(name))throw displayError(displayMessage('modules.mandarin-ipa.text.c82067c76154'));return name;}
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
  }catch{throw displayError(displayMessage('modules.mandarin-ipa.text.90856f67d6bb'));}
}
export function addTable(library:TableLibrary,table:CustomTable):TableLibrary{
  const name=tableName(table.name);if(library.tables.some(t=>t.name===name))throw displayError(displayMessage('modules.mandarin-ipa.text.ca7013e353e5'));
  if(library.tables.length>=32)throw displayError(displayMessage('modules.mandarin-ipa.text.de0565c86c7d'));
  const added={...table,name,rows:tableRows(table.rows,table.kind),revision:1};
  if(table.tones!==undefined){added.tones=validateToneSettings(table.tones);tonePreview(added.rows,added.tones);}
  return {...library,tables:[...library.tables,added],selected:table.id};
}
export function replaceTable(library:TableLibrary,id:CustomTable['id'],table:Pick<CustomTable,'name'|'kind'|'rows'|'tones'>):TableLibrary{
  const previous=library.tables.find(t=>t.id===id);if(!previous)throw displayError(displayMessage('modules.mandarin-ipa.text.ee9da353381b'));
  const name=tableName(table.name);if(library.tables.some(t=>t.id!==id&&t.name===name))throw displayError(displayMessage('modules.mandarin-ipa.text.ca7013e353e5'));
  const replacement:CustomTable={id,name,kind:table.kind,rows:tableRows(table.rows,table.kind),revision:(previous.revision??1)+1};
  if(table.tones!==undefined){replacement.tones=validateToneSettings(table.tones);tonePreview(replacement.rows,replacement.tones);}
  return {...library,tables:library.tables.map(t=>t.id===id?replacement:t),selected:id};
}
export function removeTable(library:TableLibrary,id:CustomTable['id']):TableLibrary{
  if(!library.tables.some(t=>t.id===id))throw displayError(displayMessage('modules.mandarin-ipa.text.ee9da353381b'));
  return {...library,tables:library.tables.filter(t=>t.id!==id),selected:library.selected===id?null:library.selected};
}
export class CustomTableStore {
  private assets:LocalAssets;private raw:string|null=null;
  constructor(scope:string,database?:AssetDatabase){const owner=/^server:([^:]+):/.exec(scope)?.[1]??'desktop';this.assets=new LocalAssets(owner,'m13-custom-tables',database);}
  async load(){const raw=await this.assets.value<string>('library-v1')??null;const library=restoreLibrary(raw);this.raw=raw;return library;}
  async save(library:TableLibrary){
    const raw=JSON.stringify(library);if(new TextEncoder().encode(raw).length>maxBytes)throw displayError(displayMessage('modules.mandarin-ipa.text.411b9436b960'));
    try{await this.assets.compare('library-v1',this.raw,raw);}catch{throw displayError(displayMessage('modules.mandarin-ipa.text.32f4898dd7c3'));}this.raw=raw;
  }
}

/** Recovery depends on stable error identity, never a translated sentence. */
export function tableErrorNeedsReload(cause:unknown):boolean{const value=uiErrorMessage(cause);return typeof value!=='string'&&['modules.mandarin-ipa.text.ee9da353381b','modules.mandarin-ipa.text.32f4898dd7c3'].includes(value.key);}
