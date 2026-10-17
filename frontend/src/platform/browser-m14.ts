import type {M14Port,Source,Options,Result} from '../modules/phonology-induction/port.ts';

type Output={files:{name:string;data:string}[]};
type Ports={upload(file:File,signal:AbortSignal):Promise<Source>;read(source:Source,signal:AbortSignal):Promise<ArrayBuffer>;verify(source:Source,signal:AbortSignal):Promise<void>;compute(raw:ArrayBuffer,args:unknown,signal:AbortSignal):Promise<Output|null>;save(name:string,raw:ArrayBuffer,signal:AbortSignal):Promise<{id:string;sha256:string;size_bytes:number}>};
/** Same reader/rules/renderer in a Worker; saved results use normal managed assets. */
export function browserM14(remote:M14Port,ports:Ports):M14Port {
 async function run(source:Source,options:Options,action:string,signal:AbortSignal,extra={}){
  signal.throwIfAborted();const raw=await ports.read(source,signal);
  const output=await ports.compute(raw,{name:source.name,config:{action,...options,...extra}},signal);
  signal.throwIfAborted();await ports.verify(source,signal);signal.throwIfAborted();return output;
 }
 function preview(output:Output){if(output.files.length!==1||output.files[0].name!=='m14-preview.json')throw Error('音系归纳预览不完整。');return JSON.parse(new TextDecoder().decode(bytes(output.files[0].data)));}
 function bytes(data:string){return Uint8Array.from(atob(data),c=>c.charCodeAt(0));}
 async function upload(file:File,signal:AbortSignal){if(!file.size||file.size>2_000_000)throw Error('调查表须为非空文件且不超过2 MB。');signal.throwIfAborted();const source=await ports.upload(file,signal);signal.throwIfAborted();return source;}
 return {
  async inspect(file,options,signal){const source=await upload(file,signal);return {source,inspection:await this.inspectSource(source,options,signal)};},
  async inspectSource(source,options,signal){const result=await run(source,options,'inspect',signal);return result?preview(result):remote.inspectSource(source,options,signal);},
  async analyze(source,options,signal){const result=await run(source,options,'preview',signal);if(!result)return remote.analyze(source,options,signal);const value=preview(result);if(value.input_sha256!==source.sha256)throw Error('调查表校验值不一致。');return value;},
  async import(file,options,signal){const source=await upload(file,signal);return {source,preview:await this.analyze(source,options,signal)};},
  async generate(source,options,settings,font,signal){const output=await run(source,options,'export',signal,{settings,font});if(!output)return remote.generate(source,options,settings,font,signal);
   const names=output.files.map(f=>f.name);if(names.length!==3||new Set(names).size!==3||names.filter(n=>n.endsWith('.docx')).length!==2||names.filter(n=>n.endsWith('.xlsx')).length!==1)throw Error('音系归纳导出文件不完整。');
   const job=crypto.randomUUID(),files:Result['files']=[];let size=0;
   for(const item of output.files){signal.throwIfAborted();const raw=bytes(item.data);size+=raw.byteLength;if(size>16_000_000)throw Error('导出文件超过16 MB。');const asset=await ports.save(job+'-'+item.name,raw.buffer,signal);files.push({...asset,name:item.name});}
   signal.throwIfAborted();await ports.verify(source,signal);return {job,files};
  },
  save:result=>remote.save(result),download:(result,id)=>remote.download(result,id)
 };
}
