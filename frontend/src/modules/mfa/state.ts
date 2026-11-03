import {uiError as displayError,uiMessage as displayMessage} from '../module-localization.ts';
import {tr} from '../../i18n/index.ts';
import type {MfaConfig} from './port.ts';
export const defaults=():MfaConfig=>({beam:100,retry_beam:400});
export type TranscriptSource='auto'|'.lab'|'.txt'|'.TextGrid';
export function beamChanged(config:MfaConfig,beam:number):MfaConfig {
 if(!Number.isInteger(beam)||beam<1||beam>10000)throw displayError(displayMessage('modules.mfa.text.b83762d047fd'));
 return {beam,retry_beam:Math.max(config.retry_beam,beam*4)};
}
export function validate(config:MfaConfig):MfaConfig {
 if(!Number.isInteger(config.beam)||config.beam<1||config.beam>10000||!Number.isInteger(config.retry_beam)||config.retry_beam<1||config.retry_beam>40000)throw displayError(displayMessage('modules.mfa.text.0d94cdd34415'));
 return {...config,retry_beam:config.retry_beam<=config.beam?config.beam*4:config.retry_beam};
}
export function pairFiles(files:File[],source:TranscriptSource='auto') {
 const name=(f:File)=>f.webkitRelativePath||f.name;
 const audio=files.filter(f=>/\.wav$/i.test(f.name));
 if(!audio.length)throw displayError(displayMessage('modules.mfa.text.c2c18268f91e'));
 if(audio.length>100)throw displayError(displayMessage('modules.mfa.text.7a5642e6bcdd'));
 const names=new Set<string>();
 const pairs=audio.map(wav=>{
  const path=name(wav),base=path.replace(/\.wav$/i,'');
  if(new TextEncoder().encode(path).length>110)throw displayError(displayMessage('modules.mfa.text.19daa7e3f93f'));
  if(names.has(path.toLowerCase()))throw displayError(displayMessage('modules.mfa.text.4c5c9df32ba6'));names.add(path.toLowerCase());
  const texts=files.filter(f=>name(f).replace(/\.(lab|txt|textgrid)$/i,'')===base&&/\.(lab|txt|textgrid)$/i.test(f.name)&&(source==='auto'||f.name.toLowerCase().endsWith(source.toLowerCase())));
  if(texts.length!==1)throw displayError(source==='auto'?displayMessage('modules.mfa.text.0b7088e71f5c',{v0:path}):displayMessage('modules.mfa.text.77a4e3a7cf7e',{v0:path,v1:source.slice(1)}));
  if(!texts[0].size)throw displayError(displayMessage('modules.mfa.text.b8ac98f83b47',{v0:path}));
  return {audio:wav,transcript:texts[0],name:path};
 });
 if(pairs.reduce((sum,p)=>sum+p.audio.size+p.transcript.size,0)>64_000_000)throw displayError(displayMessage('modules.mfa.text.db03474ec0c0'));
 return pairs;
}
