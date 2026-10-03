export type ButtonMode='auto'|'all'|'plain';
export interface ButtonAppearance {mode:ButtonMode;effects:boolean}
export function normalizeButtons(value:unknown):ButtonAppearance {
 const saved=value&&typeof value==='object'&&!Array.isArray(value)?value as Record<string,unknown>:{};
 return {mode:saved.mode==='all'||saved.mode==='plain'?saved.mode:'auto',effects:typeof saved.effects==='boolean'?saved.effects:true};
}
