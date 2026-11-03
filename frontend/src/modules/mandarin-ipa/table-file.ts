import {uiError as displayError,uiMessage as displayMessage} from '../module-localization.ts';
import {tr} from '../../i18n/index.ts';
import {parseDelimited} from './custom-tables.ts';

const limit=8_000_000,expandedLimit=24_000_000;
/** Bound expansion before handing XLSX to the existing locked client reader. */
function workbookBudget(raw:ArrayBuffer){
  const view=new DataView(raw);let end=-1;
  for(let p=raw.byteLength-22;p>=Math.max(0,raw.byteLength-65557);p--)if(view.getUint32(p,true)===0x06054b50&&p+22+view.getUint16(p+20,true)===raw.byteLength){end=p;break;}
  if(end<0)throw displayError(displayMessage('modules.mandarin-ipa.text.082077c9b644'));
  const count=view.getUint16(end+10,true);if(count>2048||view.getUint16(end+4,true)||view.getUint16(end+6,true))throw displayError(displayMessage('modules.mandarin-ipa.text.ede9b25a36ea'));
  let p=view.getUint32(end+16,true),total=0;
  for(let i=0;i<count;i++){
    if(p+46>raw.byteLength||view.getUint32(p,true)!==0x02014b50)throw displayError(displayMessage('modules.mandarin-ipa.text.a018420030fe'));
    const length=view.getUint16(p+28,true),extra=view.getUint16(p+30,true),comment=view.getUint16(p+32,true);
    if(p+46+length+extra+comment>end||view.getUint16(p+8,true)&1)throw displayError(displayMessage('modules.mandarin-ipa.text.736c8c44bae5'));
    total+=view.getUint32(p+24,true);if(total>expandedLimit)throw displayError(displayMessage('modules.mandarin-ipa.text.41c34e58892e'));
    p+=46+length+extra+comment;
  }
}
export async function readTableFile(file:File){
  if(!file.size||file.size>limit)throw displayError(displayMessage('modules.mandarin-ipa.text.ed3da2bd7918'));
  if(/\.xlsx$/i.test(file.name)){
    const raw=await file.arrayBuffer();workbookBudget(raw);
    const XLSX=await import('../perception/vendor/xlsx.mjs');
    let book:any;try{book=XLSX.read(raw,{type:'array',sheetRows:50002,cellFormula:true,cellHTML:false,bookVBA:false});}catch{throw displayError(displayMessage('modules.mandarin-ipa.text.2dae059391e8'));}
    const names=book.SheetNames as string[];if(!names.length)throw displayError(displayMessage('modules.mandarin-ipa.text.458e6138e66b'));
    return {names,async rows(index:number){
      const name=names[index],sheet=name&&book.Sheets[name];if(!sheet)throw displayError(displayMessage('modules.mandarin-ipa.text.26ad3e811f72'));
      const range=XLSX.utils.decode_range(sheet['!fullref']??sheet['!ref']??'A1');
      if(range.e.r>50000||range.e.c>255)throw displayError(displayMessage('modules.mandarin-ipa.text.88aa187a977e'));
      for(const [key,value] of Object.entries(sheet))if(!key.startsWith('!')&&(value as {f?:string}).f)throw displayError(displayMessage('modules.mandarin-ipa.text.750268498baf'));
      return (XLSX.utils.sheet_to_json(sheet,{header:1,range:{s:{r:0,c:0},e:range.e},blankrows:true,defval:'',raw:false}) as unknown[][]).map(row=>row.map(value=>String(value??'')));
    }};
  }
  if(!/\.(csv|tsv|txt)$/i.test(file.name))throw displayError(displayMessage('modules.mandarin-ipa.text.a494a4195c16'));
  let text:string;try{text=new TextDecoder('utf-8',{fatal:true}).decode(await file.arrayBuffer());}catch{throw displayError(displayMessage('modules.mandarin-ipa.text.405283cddbb4'));}
  const rows=parseDelimited(text,/\.(tsv|txt)$/i.test(file.name)?'\t':undefined);return {names:[file.name],async rows(){return rows;}};
}
