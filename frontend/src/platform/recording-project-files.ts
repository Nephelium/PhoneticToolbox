/** Desktop project.json/f32 interchange, without paths outside selected Files. */
import type {Project} from '../modules/recording/types.ts';
import {floatWave} from './recording-wav.ts';
export async function recordingHash(raw:ArrayBuffer){return [...new Uint8Array(await crypto.subtle.digest('SHA-256',raw))].map(v=>v.toString(16).padStart(2,'0')).join('');}
export async function readRecordingProject(files:Map<string,File>){
 const manifests=[...files.keys()].filter(p=>/(^|\/)project\.json$/.test(p));if(manifests.length!==1)throw Error('请选择只包含一个录音工程的文件夹。');
 const manifest=manifests[0],prefix=manifest.slice(0,-'project.json'.length),file=files.get(manifest)!;if(file.size>32_000_000)throw Error('工程清单超过32 MB。');const project=JSON.parse(await file.text()) as Project&{takes:any[]};
 if(project.schema_version!=='ptb-recording/1'||!Array.isArray(project.takes)||!Array.isArray(project.tasks)||project.takes.length>10000||project.tasks.length>10000)throw Error('录音工程版本或结构不支持。');
 const safeId=(id:unknown)=>typeof id==='string'&&id.length>0&&id.length<=200&&!['__proto__','prototype','constructor'].includes(id)&&!/[\\/:\x00-\x1f]/.test(id);
 const takeIds=new Set<string>(),versionIds=new Set<string>(),taskIds=new Set<string>();
 for(const task of project.tasks){if(!task||!safeId(task.id)||taskIds.has(task.id)||typeof task.filename_stem!=='string')throw Error('工程任务标识重复或无效。');taskIds.add(task.id);}
 if(!project.selected||typeof project.selected!=='object'||Array.isArray(project.selected))throw Error('工程选用记录无效。');
 const values=new Map<string,{file:File;blocks:Float32Array[][]}>(),verified=new Map<string,ArrayBuffer>();let total=0;
 for(const take of project.takes){const cfg=take.config;if(!cfg||!Number.isInteger(cfg.channels)||cfg.channels<1||cfg.channels>8||!Array.isArray(cfg.roles)||cfg.roles.length!==cfg.channels||!Number.isInteger(cfg.sample_rate)||cfg.sample_rate<8000||cfg.sample_rate>96000||!Array.isArray(take.versions)||!take.versions.length||take.versions.length>10000||!Number.isInteger(take.head)||take.head<0||take.head>=take.versions.length)throw Error('工程通道或版本结构无效。');
  if(!safeId(take.id)||takeIds.has(take.id)||cfg.roles.some((r:string)=>!['microphone','egg','other'].includes(r))||!Array.isArray(take.undo)||!Array.isArray(take.redo)||[...take.undo,...take.redo].some(i=>!Number.isInteger(i)||i<0||i>=take.versions.length))throw Error('工程录音标识或历史索引无效。');takeIds.add(take.id);
  for(const version of take.versions){if(!safeId(version.id)||versionIds.has(version.id))throw Error('工程版本标识重复或无效。');versionIds.add(version.id);}
  for(const version of take.versions){if(!Array.isArray(version.spans)||!version.spans.length)throw Error('工程音频片段缺失。');const frames=version.spans.reduce((n:number,s:any)=>n+s.end-s.start,0);if(!Number.isInteger(frames)||frames<=0||frames*cfg.channels*4+44>100_000_000)throw Error('工程版本为空或超过100 MB。');const channels=Array.from({length:cfg.channels},()=>new Float32Array(frames));let at=0;
   for(const span of version.spans){if(typeof span.file!=='string'||span.file.startsWith('/')||span.file.includes('\\')||span.file.includes(':')||span.file.split('/').includes('..')||!Number.isInteger(span.frames)||!Number.isInteger(span.start)||!Number.isInteger(span.end)||span.start<0||span.end<span.start||span.end>span.frames||span.channels!==cfg.channels)throw Error('工程片段路径或帧索引无效。');let raw=verified.get(span.file);if(!raw){const input=files.get(prefix+span.file);if(!input||input.size!==span.frames*cfg.channels*4||input.size>100_000_000)throw Error('工程音频块缺失或大小不符。');raw=await input.arrayBuffer();if(await recordingHash(raw)!==span.sha256)throw Error('工程音频块摘要不符，拒绝导入。');verified.set(span.file,raw);}const view=new DataView(raw);for(let f=span.start;f<span.end;f++)for(let c=0;c<cfg.channels;c++){const value=view.getFloat32((f*cfg.channels+c)*4,true);if(!Number.isFinite(value))throw Error('工程含非有限采样。');channels[c][at+f-span.start]=value;}at+=span.end-span.start;}
   const wave=floatWave([channels],cfg.sample_rate,cfg.channels);total+=wave.byteLength;if(total>500_000_000)throw Error('本次工程导入超过500 MB，请分项目导入。');version.frames=frames;version.metadata={...version.metadata,...(version.parent?{parent:version.parent}:{})};delete version.spans;values.set(version.id,{file:new File([wave],'工程录音-'+version.id.slice(0,12)+'.wav',{type:'audio/wav'}),blocks:[channels]});
  }
 }
 for(const [task,id] of Object.entries(project.selected))if(!safeId(task)||typeof id!=='string'||!takeIds.has(id))throw Error('工程选用录音不存在。');
 project.recoveries=[];
 return {project,values};
}
