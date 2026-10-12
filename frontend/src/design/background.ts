export interface BackgroundAppearance {blur:number;opacity:number}
export function normalizeBackground(value:unknown):BackgroundAppearance {
 const saved=value&&typeof value==='object'?value as Record<string,unknown>:{};
 const bounded=(v:unknown,fallback:number,max:number)=>typeof v==='number'&&Number.isFinite(v)?Math.max(0,Math.min(max,v)):fallback;
 return {blur:bounded(saved.blur,8,40),opacity:bounded(saved.opacity,25,100)};
}
