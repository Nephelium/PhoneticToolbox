import {uiMessage,uiError,localized} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
// Desktop writes go through the owned Qt host. No clipboard read capability is exposed.
type ClipboardWriter=(text:string)=>Promise<void>;
let nativeWriter:ClipboardWriter|undefined;

export function installClipboardWriter(writer:ClipboardWriter|undefined){nativeWriter=writer;}

export async function copyText(text:string):Promise<void>{
  if(nativeWriter){await nativeWriter(text);return;}
  if(!globalThis.navigator?.clipboard?.writeText)throw uiError(uiMessage('platform.the_clipboard_is_unavailable_select_the_text_and_press'));
  await navigator.clipboard.writeText(text);
}
