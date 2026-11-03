import {uiError as displayError,uiMessage as displayMessage} from '../module-localization.ts';
import {localized} from '../module-localization.ts';
import {tr} from '../../i18n/index.ts';
import * as XLSX from '../perception/vendor/xlsx.mjs';
import type {Task} from './types.ts';
export const fields=['task_id','title','prompt','filename_stem','repeat_count','group','note','enabled'] as const;
export type Field=typeof fields[number];
export const fieldLabels:Record<Field,string>=localized(()=>({task_id:tr('modules.recording.text.160691af2acc'),title:tr('modules.recording.text.2479560deb33'),prompt:tr('modules.recording.text.0ff35b949bd8'),filename_stem:tr('modules.recording.text.618fd86c8e85'),repeat_count:tr('modules.recording.text.afb2bbd126b5'),group:tr('modules.recording.text.399ee6934772'),note:tr('modules.recording.text.daede9881787'),enabled:tr('modules.recording.text.f4f0ead1116b')}));
export function exampleWorkbook():ArrayBuffer{
 const sheet=XLSX.utils.aoa_to_sheet([
  ['任务编号','任务名称','录音内容','文件名','次数','分组','备注','启用'],
  ['T001','示例一','请读：春天来了','example-001',1,'练习','','是'],
  ['T002','示例二','请读：今天是晴天','example-002',2,'练习','自然语速','是']
 ]);
 sheet['!cols']=[14,14,30,20,10,14,24,10].map(wch=>({wch}));
 sheet['!autofilter']={ref:'A1:H3'};
 const workbook=XLSX.utils.book_new();XLSX.utils.book_append_sheet(workbook,sheet,'录音任务');
 return XLSX.write(workbook,{type:'array',bookType:'xlsx',compression:true});
}
export type Table={sheets:{name:string;rows:string[][]}[];template?:Task[]};
const aliases:Record<Field,string[]>={task_id:['task_id','任务编号','id'],title:['title','任务名称','名称'],prompt:['prompt','文本','录音内容','内容'],filename_stem:['filename_stem','文件名'],repeat_count:['repeat_count','次数'],group:['group','分组'],note:['note','备注'],enabled:['enabled','启用']};
export function csvRows(text:string,delimiter=','):string[][]{
 if(text.length>16*1024*1024)throw displayError(displayMessage('modules.recording.text.7256b95dc377'));
 const rows:string[][]=[];let row:string[]=[],cell='',quoted=false;
 for(let i=0;i<text.length;i++){const c=text[i];if(c==='"'){if(quoted&&text[i+1]==='"'){cell+='"';i++;}else if(quoted||cell==='')quoted=!quoted;else throw displayError(displayMessage('modules.recording.text.dceca969d823'));}else if(!quoted&&c===delimiter){row.push(cell);cell='';}else if(!quoted&&(c==='\n'||c==='\r')){if(c==='\r'&&text[i+1]==='\n')i++;row.push(cell);rows.push(row);row=[];cell='';}else cell+=c;}
 if(quoted)throw displayError(displayMessage('modules.recording.text.52ae49282710'));if(cell||row.length){row.push(cell);rows.push(row);}if(rows.length>10001||rows.some(r=>r.length>51||r.some(c=>c.length>20000)))throw displayError(displayMessage('modules.recording.text.4c68a76fa6da'));return rows.filter(r=>r.some(v=>v.trim()));
}
function zipBudget(bytes:ArrayBuffer){const v=new DataView(bytes);for(let i=Math.max(0,v.byteLength-65557);i+22<=v.byteLength;i++)if(v.getUint32(i,true)===0x06054b50){const entries=v.getUint16(i+10,true);let offset=v.getUint32(i+16,true),total=0;if(entries>2000)throw displayError(displayMessage('modules.recording.text.5dde29d3a9da'));for(let n=0;n<entries;n++){if(offset+46>v.byteLength||v.getUint32(offset,true)!==0x02014b50)throw displayError(displayMessage('modules.recording.text.253bfd66e3c2'));total+=v.getUint32(offset+24,true);if(total>64*1024*1024)throw displayError(displayMessage('modules.recording.text.cd7ea4f07f2d'));offset+=46+v.getUint16(offset+28,true)+v.getUint16(offset+30,true)+v.getUint16(offset+32,true);}return;}throw displayError(displayMessage('modules.recording.text.7df2315c4c1e'));}
export async function readTable(file:File):Promise<Table>{
 if(file.size>16*1024*1024)throw displayError(displayMessage('modules.recording.text.7256b95dc377'));const bytes=await file.arrayBuffer();
 if(/\.json$/i.test(file.name)){
  const value=JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(bytes));if(value.schema_version!=='ptb-recording-tasks/1'||!Array.isArray(value.tasks)||value.tasks.length>10000)throw displayError(displayMessage('modules.recording.text.dcc1e097fcdc'));
  const ids=new Set<string>(),stems=new Set<string>();for(const t of value.tasks){if(!t||typeof t.id!=='string'||!t.id||ids.has(t.id))throw displayError(displayMessage('modules.recording.text.fefe90c312f6'));ids.add(t.id);for(const field of ['title','prompt','filename_stem','group','note'])if(typeof t[field]!=='string'||t[field].length>20000)throw displayError(displayMessage('modules.recording.text.3b943f98f80a'));if(!safeStem(t.filename_stem)||stems.has(t.filename_stem.toLowerCase()))throw displayError(displayMessage('modules.recording.text.90226a7f48ae'));stems.add(t.filename_stem.toLowerCase());if(typeof t.enabled!=='boolean'||typeof t.skipped!=='boolean')throw displayError(displayMessage('modules.recording.text.e2123e11dfa4'));}
  return {sheets:[{name:file.name,rows:[]}],template:value.tasks};
 }
 if(/\.xlsx$/i.test(file.name)){zipBudget(bytes);const wb=XLSX.read(bytes,{type:'array',cellFormula:true,cellHTML:false,bookVBA:false,sheetRows:10002});return {sheets:wb.SheetNames.map((name:string)=>{const sheet=wb.Sheets[name];const range=XLSX.utils.decode_range(sheet['!fullref']??sheet['!ref']??'A1');if(range.e.r>10000||range.e.c>50)throw displayError(displayMessage('modules.recording.text.c86267863802'));for(const [key,value] of Object.entries(sheet))if(!key.startsWith('!')&&(value as {f?:string}).f)throw displayError(displayMessage('modules.recording.text.9c7c4501bcea',{v0:name,v1:key}));const rows=(XLSX.utils.sheet_to_json(sheet,{header:1,defval:'',raw:false}) as unknown[][]).map(row=>row.map(v=>String(v)));if(rows.some(row=>row.some(cell=>cell.length>20000)))throw displayError(displayMessage('modules.recording.text.938f9bd15c91'));return {name,rows};})};}
 const text=new TextDecoder('utf-8',{fatal:true}).decode(bytes).replace(/^\uFEFF/,'');return {sheets:[{name:file.name,rows:csvRows(text,/\.tsv$/i.test(file.name)?'\t':',')}]};
}
export function autoMapping(header:string[]):Record<Field,number>{return Object.fromEntries(fields.map(field=>[field,header.findIndex(h=>aliases[field].includes(h.trim().toLowerCase()))])) as Record<Field,number>;}
export function safeStem(text:string){return !!text&&!/[<>:"/\\|?*\x00-\x1f]/.test(text)&&!/[ .]$/.test(text)&&text.length<=100&&!/^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)/i.test(text);}
export function importRows(rows:string[][],mapping:Record<Field,number>):{tasks:Task[];errors:string[]}{
 const tasks:Task[]=[],errors:string[]=[],ids=new Set<string>(),stems=new Set<string>();let expansionExceeded=false;
 if(mapping.prompt<0&&mapping.title<0)return {tasks,errors:[tr('modules.recording.text.05a76d9ad2a8')]};
 if(rows.length>10001)return {tasks,errors:[tr('modules.recording.text.b5872686f38d')]};
 rows.slice(1).forEach((row,index)=>{if(expansionExceeded)return;if(!row.some(x=>x.trim()))return;if(row.length>51||row.some(c=>c.length>20000)){errors.push(tr('modules.recording.text.f0ae57deb23f',{v0:index+2}));return;}const get=(f:Field)=>mapping[f]>=0?(row[mapping[f]]??'').trim():'';const repeats=Number(get('repeat_count')||1),baseId=get('task_id')||`T${String(index+1).padStart(3,'0')}`,baseStem=get('filename_stem')||`recording-${String(index+1).padStart(3,'0')}`;
 if(!Number.isInteger(repeats)||repeats<1||repeats>100){errors.push(tr('modules.recording.text.b554e996311f',{v0:index+2}));return;}
 if(!safeStem(baseStem)){errors.push(tr('modules.recording.text.74c55360f227',{v0:index+2}));return;}
 const enabled=get('enabled').toLowerCase();if(enabled&&!['true','false','1','0','是','否','启用','禁用'].includes(enabled)){errors.push(tr('modules.recording.text.91efac216882',{v0:index+2}));return;}
 for(let n=1;n<=repeats;n++){if(tasks.length>=10000){expansionExceeded=true;errors.push(tr('modules.recording.text.d267c3569622'));break;}const suffix=repeats>1?`_${n}`:'',id=baseId+suffix,stem=baseStem+suffix;if(ids.has(id)){errors.push(tr('modules.recording.text.8879fc69bea4',{v0:index+2,v1:id}));continue;}if(stems.has(stem.toLowerCase())){errors.push(tr('modules.recording.text.1f09e80cad9f',{v0:index+2,v1:stem}));continue;}ids.add(id);stems.add(stem.toLowerCase());tasks.push({id,title:get('title'),prompt:get('prompt'),filename_stem:stem,group:get('group'),note:get('note'),enabled:!['false','0','否','禁用'].includes(enabled),skipped:false});}
 });if(tasks.length>10000)errors.push(tr('modules.recording.text.d267c3569622'));return {tasks,errors};
}
export function taskCSV(tasks:Task[]){const cell=(v:unknown)=>{let s=String(v??'');if(/^[=+@\-\t\r]/.test(s))s="'"+s;return '"'+s.replace(/"/g,'""')+'"';};return '\uFEFF'+[['task_id','title','prompt','filename_stem','repeat_count','group','note','enabled'],...tasks.map(t=>[t.id,t.title,t.prompt,t.filename_stem,1,t.group,t.note,t.enabled])].map(r=>r.map(cell).join(',')).join('\r\n');}
