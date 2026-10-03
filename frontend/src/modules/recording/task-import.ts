import * as XLSX from '../perception/vendor/xlsx.mjs';
import type {Task} from './types.ts';
export const fields=['task_id','title','prompt','filename_stem','repeat_count','group','note','enabled'] as const;
export type Field=typeof fields[number];
export const fieldLabels:Record<Field,string>={task_id:'任务编号',title:'任务名称',prompt:'录音内容',filename_stem:'导出文件名（不含扩展名）',repeat_count:'重复次数',group:'分组',note:'备注',enabled:'启用'};
export const exampleCSV='\uFEFF任务编号,任务名称,录音内容,文件名,次数,分组,备注,启用\r\nT001,示例一,请读：春天来了,example-001,1,练习,,是\r\nT002,示例二,请读：今天是晴天,example-002,2,练习,自然语速,是\r\n';
export type Table={sheets:{name:string;rows:string[][]}[];template?:Task[]};
const aliases:Record<Field,string[]>={task_id:['task_id','任务编号','id'],title:['title','任务名称','名称'],prompt:['prompt','文本','录音内容','内容'],filename_stem:['filename_stem','文件名'],repeat_count:['repeat_count','次数'],group:['group','分组'],note:['note','备注'],enabled:['enabled','启用']};
export function csvRows(text:string,delimiter=','):string[][]{
 if(text.length>16*1024*1024)throw Error('表格超过 16 MiB');
 const rows:string[][]=[];let row:string[]=[],cell='',quoted=false;
 for(let i=0;i<text.length;i++){const c=text[i];if(c==='"'){if(quoted&&text[i+1]==='"'){cell+='"';i++;}else if(quoted||cell==='')quoted=!quoted;else throw Error('CSV 引号位置无效');}else if(!quoted&&c===delimiter){row.push(cell);cell='';}else if(!quoted&&(c==='\n'||c==='\r')){if(c==='\r'&&text[i+1]==='\n')i++;row.push(cell);rows.push(row);row=[];cell='';}else cell+=c;}
 if(quoted)throw Error('CSV 引号未闭合');if(cell||row.length){row.push(cell);rows.push(row);}if(rows.length>10001||rows.some(r=>r.length>51||r.some(c=>c.length>20000)))throw Error('CSV 超过 10000 数据行、51 列或单格 20000 字符');return rows.filter(r=>r.some(v=>v.trim()));
}
function zipBudget(bytes:ArrayBuffer){const v=new DataView(bytes);for(let i=Math.max(0,v.byteLength-65557);i+22<=v.byteLength;i++)if(v.getUint32(i,true)===0x06054b50){const entries=v.getUint16(i+10,true);let offset=v.getUint32(i+16,true),total=0;if(entries>2000)throw Error('XLSX 条目超过预算');for(let n=0;n<entries;n++){if(offset+46>v.byteLength||v.getUint32(offset,true)!==0x02014b50)throw Error('XLSX ZIP 损坏');total+=v.getUint32(offset+24,true);if(total>64*1024*1024)throw Error('XLSX 解压超过 64 MiB');offset+=46+v.getUint16(offset+28,true)+v.getUint16(offset+30,true)+v.getUint16(offset+32,true);}return;}throw Error('无效 XLSX');}
export async function readTable(file:File):Promise<Table>{
 if(file.size>16*1024*1024)throw Error('表格超过 16 MiB');const bytes=await file.arrayBuffer();
 if(/\.json$/i.test(file.name)){
  const value=JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(bytes));if(value.schema_version!=='ptb-recording-tasks/1'||!Array.isArray(value.tasks)||value.tasks.length>10000)throw Error('任务模板版本或条数无效');
  const ids=new Set<string>(),stems=new Set<string>();for(const t of value.tasks){if(!t||typeof t.id!=='string'||!t.id||ids.has(t.id))throw Error('模板任务编号为空或重复');ids.add(t.id);for(const field of ['title','prompt','filename_stem','group','note'])if(typeof t[field]!=='string'||t[field].length>20000)throw Error('模板文本字段无效');if(!safeStem(t.filename_stem)||stems.has(t.filename_stem.toLowerCase()))throw Error('模板文件名不安全或重复');stems.add(t.filename_stem.toLowerCase());if(typeof t.enabled!=='boolean'||typeof t.skipped!=='boolean')throw Error('模板任务状态无效');}
  return {sheets:[{name:file.name,rows:[]}],template:value.tasks};
 }
 if(/\.xlsx$/i.test(file.name)){zipBudget(bytes);const wb=XLSX.read(bytes,{type:'array',cellFormula:true,cellHTML:false,bookVBA:false,sheetRows:10002});return {sheets:wb.SheetNames.map((name:string)=>{const sheet=wb.Sheets[name];const range=XLSX.utils.decode_range(sheet['!fullref']??sheet['!ref']??'A1');if(range.e.r>10000||range.e.c>50)throw Error('每张表最多 10000 行和 51 列');for(const [key,value] of Object.entries(sheet))if(!key.startsWith('!')&&(value as {f?:string}).f)throw Error(`${name} ${key} 含公式，请转为文字后导入`);const rows=(XLSX.utils.sheet_to_json(sheet,{header:1,defval:'',raw:false}) as unknown[][]).map(row=>row.map(v=>String(v)));if(rows.some(row=>row.some(cell=>cell.length>20000)))throw Error('单格文字超过 20000 字符');return {name,rows};})};}
 const text=new TextDecoder('utf-8',{fatal:true}).decode(bytes).replace(/^\uFEFF/,'');return {sheets:[{name:file.name,rows:csvRows(text,/\.tsv$/i.test(file.name)?'\t':',')}]};
}
export function autoMapping(header:string[]):Record<Field,number>{return Object.fromEntries(fields.map(field=>[field,header.findIndex(h=>aliases[field].includes(h.trim().toLowerCase()))])) as Record<Field,number>;}
export function safeStem(text:string){return !!text&&!/[<>:"/\\|?*\x00-\x1f]/.test(text)&&!/[ .]$/.test(text)&&text.length<=100&&!/^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)/i.test(text);}
export function importRows(rows:string[][],mapping:Record<Field,number>):{tasks:Task[];errors:string[]}{
 const tasks:Task[]=[],errors:string[]=[],ids=new Set<string>(),stems=new Set<string>();
 if(mapping.prompt<0&&mapping.title<0)return {tasks,errors:['至少映射录音内容或任务名称列']};
 if(rows.length>10001)return {tasks,errors:['表格超过 10000 数据行']};
 rows.slice(1).forEach((row,index)=>{if(errors.some(e=>e==='展开后任务超过 10000 条'))return;if(!row.some(x=>x.trim()))return;if(row.length>51||row.some(c=>c.length>20000)){errors.push(`第 ${index+2} 行：列数或单格文字超过预算`);return;}const get=(f:Field)=>mapping[f]>=0?(row[mapping[f]]??'').trim():'';const repeats=Number(get('repeat_count')||1),baseId=get('task_id')||`T${String(index+1).padStart(3,'0')}`,baseStem=get('filename_stem')||`recording-${String(index+1).padStart(3,'0')}`;
 if(!Number.isInteger(repeats)||repeats<1||repeats>100){errors.push(`第 ${index+2} 行：次数须为 1–100 整数`);return;}
 if(!safeStem(baseStem)){errors.push(`第 ${index+2} 行：文件名不安全`);return;}
 const enabled=get('enabled').toLowerCase();if(enabled&&!['true','false','1','0','是','否','启用','禁用'].includes(enabled)){errors.push(`第 ${index+2} 行：启用值无效`);return;}
 for(let n=1;n<=repeats;n++){if(tasks.length>=10000){errors.push('展开后任务超过 10000 条');break;}const suffix=repeats>1?`_${n}`:'',id=baseId+suffix,stem=baseStem+suffix;if(ids.has(id)){errors.push(`第 ${index+2} 行：重复任务编号 ${id}`);continue;}if(stems.has(stem.toLowerCase())){errors.push(`第 ${index+2} 行：文件名冲突 ${stem}`);continue;}ids.add(id);stems.add(stem.toLowerCase());tasks.push({id,title:get('title'),prompt:get('prompt'),filename_stem:stem,group:get('group'),note:get('note'),enabled:!['false','0','否','禁用'].includes(enabled),skipped:false});}
 });if(tasks.length>10000)errors.push('展开后任务超过 10000 条');return {tasks,errors};
}
export function taskCSV(tasks:Task[]){const cell=(v:unknown)=>{let s=String(v??'');if(/^[=+@\-\t\r]/.test(s))s="'"+s;return '"'+s.replace(/"/g,'""')+'"';};return '\uFEFF'+[['task_id','title','prompt','filename_stem','repeat_count','group','note','enabled'],...tasks.map(t=>[t.id,t.title,t.prompt,t.filename_stem,1,t.group,t.note,t.enabled])].map(r=>r.map(cell).join(',')).join('\r\n');}
