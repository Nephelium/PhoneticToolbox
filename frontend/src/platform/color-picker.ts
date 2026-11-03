import {uiMessage,uiError,localized} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
let nativePicker:((initial:string)=>Promise<string|null>)|undefined;
export function installColorPicker(picker:(initial:string)=>Promise<string|null>){nativePicker=picker;}
interface EyeDropperInstance {open():Promise<{sRGBHex:string}>}
const eyeDropper=()=> (window as unknown as {EyeDropper?:new()=>EyeDropperInstance}).EyeDropper;
export function screenPickerAvailable(){return !!nativePicker||!!eyeDropper();}
export async function pickScreenColor(initial:string):Promise<string|null>{
 if(nativePicker)return nativePicker(initial);
 const Picker=eyeDropper();if(!Picker)throw uiError(uiMessage('platform.use_the_background_image_or_color_picker_to_choose'));
 try{return (await new Picker().open()).sRGBHex;}catch(error){if(error instanceof DOMException&&error.name==='AbortError')return null;throw error;}
}
