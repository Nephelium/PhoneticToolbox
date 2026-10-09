import type {Config,Mode,Point} from './port.ts';
export const modes:{value:Mode;label:string}[]=[{value:'full',label:'全连接'},{value:'order',label:'顺序'},{value:'reverse',label:'逆序'},{value:'constant',label:'常量'}];
export const defaults=():Config=>({action:'synthesize',start:0,end:null,speed:1,pitch_ratio:1,pitch_hz:0,modified_f0:null,points:[],offset:false});
export function finite(value:unknown,label:string){if(value===null||value===undefined||!String(value).trim()||!Number.isFinite(Number(value)))throw Error(label+'必须为有限数字');return Number(value);}
export function frequencies(text:string){const values=text.trim().split(/[,，;；\s]+/).filter(Boolean).map(x=>finite(x,'频率'));if(!values.length)throw Error('频率列表不能为空');return values;}
export function validate(points:Point[],start:number,end:number){
 if(points.length<2)throw Error('至少需要起点和终点');let diagonal:number|undefined,count=1;
 for(let i=0;i<points.length;i++){const p=points[i];finite(p.time,'时间');if(p.time<start||p.time>end||(i&&p.time<=points[i-1].time))throw Error('拐点时间须递增，且位于起止时间及当前视野内');if(!p.freqs.length||p.freqs.some(x=>!Number.isFinite(x)))throw Error('频率列表不能为空或包含非有限值');if(!modes.some(m=>m.value===p.mode))throw Error('未知连接方式');if(p.mode==='full')count*=p.freqs.length;if(p.mode==='order'||p.mode==='reverse'){if(diagonal!==undefined&&diagonal!==p.freqs.length)throw Error('顺序/逆序的列表长度必须一致');diagonal=p.freqs.length;}}
 count*=diagonal??1;if(count>256)throw Error('本次超过 256 个结果，请拆分频率列表');return count;
}
export function importF0(times:number[],modified:number[],start:number,end:number,text:string){
 const values=text.replace(/[,，;；]/g,' ').split(/\r?\n/).filter(x=>x.trim()).map(line=>{const parts=line.trim().split(/\s+/);return finite(parts.length>1?parts[1]:parts[0],'导入基频');});
 if(!values.length)throw Error('基频序列不能为空');const view=times.map((t,i)=>t>=start&&t<=end?i:-1).filter(i=>i>=0),voiced=view.filter(i=>modified[i]>0);
 if(!voiced.length)throw Error('当前视野没有有效基频');if(voiced.some((n,i)=>i>0&&n!==voiced[i-1]+1))throw Error('当前视野包含中断的基频，请缩放至一段连续曲线');
 const result=[...modified];for(let i=0;i<voiced.length;i++){const x=voiced.length===1?0:i/(voiced.length-1)*(values.length-1),a=Math.floor(x),b=Math.min(values.length-1,a+1);result[voiced[i]]=values[a]+(values[b]-values[a])*(x-a);}return result;
}
export function editCurve(original:number[],modified:number[],index:number,value:number,last?:{index:number;value:number},restore=false){
 const result=[...modified];if(!restore&&original[index]<=0)return {curve:result,last};const lo=Math.min(index,last?.index??index),hi=Math.max(index,last?.index??index);
 for(let i=lo;i<=hi;i++){if(restore)result[i]=original[i];else if(original[i]>0)result[i]=last&&hi>lo?last.value+(value-last.value)*(i-last.index)/(index-last.index):value;}
 return {curve:result,last:{index,value}};
}
export function renamePrefix(files:{name:string}[]){let common=files[0]?.name??'';for(const file of files)while(!file.name.startsWith(common))common=common.slice(0,-1);return common.replace(/\.wav$/i,'');}
export function renamePlan(files:{id:string;name:string}[],prefix:string,existing:string[]){
 if(!files.length)throw Error('请先选择本批次结果');const common=renamePrefix(files);
 if(!common)throw Error('所选文件没有共同前缀');if(!prefix.trim()||/[\x00-\x1f/\\:<>"|?*]/.test(prefix))throw Error('前缀包含无效文件名字符');
 const seen=new Set<string>();return files.map(f=>{const base=prefix+f.name.slice(common.length);let name=base,index=2;while(seen.has(name.toLowerCase()))name=base.replace(/\.wav$/i,` (${index++}).wav`);const key=name.toLowerCase();if(!/\.wav$/i.test(name)||name.length>180)throw Error('名称必须保留 .wav，且不超过 180 字符');if(existing.some(n=>n.toLowerCase()===key))throw Error('重复名称，未执行重命名');seen.add(key);return {id:f.id,name};});
}
export function visible(duration:number,zoom:number,offset:number):[number,number]{const span=duration/Math.max(1,zoom),left=Math.max(0,Math.min(offset,duration-span));return [left,left+span];}
export function errorText(error:unknown){const text=error instanceof Error?error.message:String(error);const labels:Record<string,string>={m08_audio_decode_failed:'音频无法解码，请检查文件格式与内容。',m08_praat_error:'Praat 无法处理此音频或频率配置，请检查区间与 F0。',m08_input_budget:'音频超过本次输入预算，请先明确选择较短材料。',m08_output_budget:'预计结果超过本次输出预算，请拆分任务。',m08_cancelled:'任务已取消。',m08_diagonal_length:'顺序/逆序列表长度必须一致。',m08_invalid_range:'时间范围超出音频。',m08_input_changed:'原文件已变化，请重新读取。',m08_curve_length:'F0 帧数与源音频不一致，请重新读取。'};return labels[text]??text;}
