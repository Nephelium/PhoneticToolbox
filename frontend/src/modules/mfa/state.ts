import type {MfaConfig} from './port.ts';
export const defaults=():MfaConfig=>({beam:100,retry_beam:400});
export type TranscriptSource='auto'|'.lab'|'.txt'|'.TextGrid';
export function beamChanged(config:MfaConfig,beam:number):MfaConfig {
 if(!Number.isInteger(beam)||beam<1||beam>10000)throw Error('Beam 需为 1–10000 的整数。');
 return {beam,retry_beam:Math.max(config.retry_beam,beam*4)};
}
export function validate(config:MfaConfig):MfaConfig {
 if(!Number.isInteger(config.beam)||config.beam<1||config.beam>10000||!Number.isInteger(config.retry_beam)||config.retry_beam<1||config.retry_beam>40000)throw Error('请检查 Beam / Retry beam 的整数范围。');
 return {...config,retry_beam:config.retry_beam<=config.beam?config.beam*4:config.retry_beam};
}
export function pairFiles(files:File[],source:TranscriptSource='auto') {
 const name=(f:File)=>f.webkitRelativePath||f.name;
 const audio=files.filter(f=>/\.wav$/i.test(f.name));
 if(!audio.length)throw Error('请选择 WAV 及对应的已有转写文件。');
 if(audio.length>100)throw Error('单任务最多 100 份音频。');
 const names=new Set<string>();
 const pairs=audio.map(wav=>{
  const path=name(wav),base=path.replace(/\.wav$/i,'');
  if(new TextEncoder().encode(path).length>110)throw Error('相对路径超过当前中文编码预算。');
  if(names.has(path.toLowerCase()))throw Error('音频名称重复，请分目录导入。');names.add(path.toLowerCase());
  const texts=files.filter(f=>name(f).replace(/\.(lab|txt|textgrid)$/i,'')===base&&/\.(lab|txt|textgrid)$/i.test(f.name)&&(source==='auto'||f.name.toLowerCase().endsWith(source.toLowerCase())));
  if(texts.length!==1)throw Error(source==='auto'?`${path} 需要唯一同名 LAB、TXT 或 TextGrid 转写；有多种格式时，请选择转写来源后重新导入。`:`${path} 需要一份同名 ${source.slice(1)} 转写，请检查所选格式后重新导入。`);
  if(!texts[0].size)throw Error(`${path} 的转写为空。`);
  return {audio:wav,transcript:texts[0],name:path};
 });
 if(pairs.reduce((sum,p)=>sum+p.audio.size+p.transcript.size,0)>64_000_000)throw Error('任务输入合计超过 64 MB。');
 return pairs;
}
