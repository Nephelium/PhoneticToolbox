export type ButtonMode='auto'|'all'|'plain';
export type ButtonMaterial='solid'|'glass';
export interface ButtonAppearance {mode:ButtonMode;effects:boolean;material:ButtonMaterial;glassOpacity:number;glassBlur:number}
export function normalizeButtons(value:unknown):ButtonAppearance {
 const saved=value&&typeof value==='object'&&!Array.isArray(value)?value as Record<string,unknown>:{};
 const bounded=(v:unknown,fallback:number,max:number)=>typeof v==='number'&&Number.isFinite(v)?Math.max(0,Math.min(max,v)):fallback;
 return {mode:saved.mode==='all'||saved.mode==='plain'?saved.mode:'auto',effects:typeof saved.effects==='boolean'?saved.effects:true,material:saved.material==='glass'?'glass':'solid',glassOpacity:bounded(saved.glassOpacity,35,100),glassBlur:bounded(saved.glassBlur,12,40)};
}
