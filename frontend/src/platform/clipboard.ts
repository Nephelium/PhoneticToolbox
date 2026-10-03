// Desktop writes go through the owned Qt host. No clipboard read capability is exposed.
type ClipboardWriter=(text:string)=>Promise<void>;
let nativeWriter:ClipboardWriter|undefined;

export function installClipboardWriter(writer:ClipboardWriter|undefined){nativeWriter=writer;}

export async function copyText(text:string):Promise<void>{
  if(nativeWriter){await nativeWriter(text);return;}
  if(!globalThis.navigator?.clipboard?.writeText)throw Error('剪贴板不可用，请选中文字后按 Ctrl+C。');
  await navigator.clipboard.writeText(text);
}
