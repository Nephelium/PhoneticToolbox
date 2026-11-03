import {message,type MessageDescriptor} from '../i18n/index.ts';
import messages from '../i18n/locales/zh-CN/manual.json' with {type:'json'};

/** Keep the message identity through asynchronous install failures. */
export function manualPackError(reason:unknown):MessageDescriptor{
 const code=reason instanceof Error?reason.message:typeof reason==='string'?reason:'';
 const key=code.startsWith('manual.pack.')?'manual.reader.'+code.slice('manual.'.length):'';
 return message(Object.hasOwn(messages,key)?key:'manual.reader.pack.failed');
}
