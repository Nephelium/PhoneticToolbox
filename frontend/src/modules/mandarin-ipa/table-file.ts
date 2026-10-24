import {parseDelimited} from './custom-tables.ts';

const limit=8_000_000,expandedLimit=24_000_000;
/** Bound expansion before handing XLSX to the existing locked client reader. */
function workbookBudget(raw:ArrayBuffer){
  const view=new DataView(raw);let end=-1;
  for(let p=raw.byteLength-22;p>=Math.max(0,raw.byteLength-65557);p--)if(view.getUint32(p,true)===0x06054b50&&p+22+view.getUint16(p+20,true)===raw.byteLength){end=p;break;}
  if(end<0)throw Error('无法读取 XLSX，请选择有效的 Excel 工作簿。');
  const count=view.getUint16(end+10,true);if(count>2048||view.getUint16(end+4,true)||view.getUint16(end+6,true))throw Error('XLSX 结构超出支持范围。');
  let p=view.getUint32(end+16,true),total=0;
  for(let i=0;i<count;i++){
    if(p+46>raw.byteLength||view.getUint32(p,true)!==0x02014b50)throw Error('XLSX 文件目录损坏。');
    const length=view.getUint16(p+28,true),extra=view.getUint16(p+30,true),comment=view.getUint16(p+32,true);
    if(p+46+length+extra+comment>end||view.getUint16(p+8,true)&1)throw Error('不支持加密或损坏的 XLSX。');
    total+=view.getUint32(p+24,true);if(total>expandedLimit)throw Error('XLSX 解压内容过大，请只保留需要的两列字表。');
    p+=46+length+extra+comment;
  }
}
export async function readTableFile(file:File){
  if(!file.size||file.size>limit)throw Error('请选择非空且不超过 8 MB 的字表文件。');
  if(/\.xlsx$/i.test(file.name)){
    const raw=await file.arrayBuffer();workbookBudget(raw);
    const XLSX=await import('../perception/vendor/xlsx.mjs');
    let book:any;try{book=XLSX.read(raw,{type:'array',sheetRows:50002,cellFormula:true,cellHTML:false,bookVBA:false});}catch{throw Error('XLSX 内容损坏，请重新保存表格。');}
    const names=book.SheetNames as string[];if(!names.length)throw Error('XLSX 没有工作表。');
    return {names,async rows(index:number){
      const name=names[index],sheet=name&&book.Sheets[name];if(!sheet)throw Error('请选择工作表。');
      const range=XLSX.utils.decode_range(sheet['!fullref']??sheet['!ref']??'A1');
      if(range.e.r>50000||range.e.c>255)throw Error('字表最多支持 50,000 条记录，请只保留需要的两列。');
      for(const [key,value] of Object.entries(sheet))if(!key.startsWith('!')&&(value as {f?:string}).f)throw Error('字表包含公式，请先复制并粘贴为值。');
      return (XLSX.utils.sheet_to_json(sheet,{header:1,range:{s:{r:0,c:0},e:range.e},blankrows:true,defval:'',raw:false}) as unknown[][]).map(row=>row.map(value=>String(value??'')));
    }};
  }
  if(!/\.(csv|tsv|txt)$/i.test(file.name))throw Error('请选择 XLSX、CSV、TSV 或制表符 TXT 文件。');
  let text:string;try{text=new TextDecoder('utf-8',{fatal:true}).decode(await file.arrayBuffer());}catch{throw Error('文本字表请保存为 UTF-8 编码后再导入。');}
  const rows=parseDelimited(text,/\.(tsv|txt)$/i.test(file.name)?'\t':undefined);return {names:[file.name],async rows(){return rows;}};
}
