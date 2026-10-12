let nativePicker:((initial:string)=>Promise<string|null>)|undefined;
export function installColorPicker(picker:(initial:string)=>Promise<string|null>){nativePicker=picker;}
interface EyeDropperInstance {open():Promise<{sRGBHex:string}>}
const eyeDropper=()=> (window as unknown as {EyeDropper?:new()=>EyeDropperInstance}).EyeDropper;
export function screenPickerAvailable(){return !!nativePicker||!!eyeDropper();}
export async function pickScreenColor(initial:string):Promise<string|null>{
 if(nativePicker)return nativePicker(initial);
 const Picker=eyeDropper();if(!Picker)throw Error('当前环境请用背景图取色或色盘选择颜色。');
 try{return (await new Picker().open()).sRGBHex;}catch(error){if(error instanceof DOMException&&error.name==='AbortError')return null;throw error;}
}
