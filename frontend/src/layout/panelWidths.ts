export interface PanelLimit { min:number; max:number; initial:number }
export function panelWidth(value:unknown,limit:PanelLimit):number {
  return Math.round(Math.max(limit.min,Math.min(limit.max,typeof value==='number'&&Number.isFinite(value)?value:limit.initial)));
}
// Shrinking the window must not overwrite the user's preferred widths.
export function fitPanels(wanted:number[],minimum:number[],available:number):number[] {
  const extra=wanted.reduce((sum,n,i)=>sum+n-minimum[i],0);
  const room=Math.max(0,available-minimum.reduce((a,b)=>a+b,0));
  return wanted.map((n,i)=>minimum[i]+(extra>room&&extra>0?(n-minimum[i])*room/extra:n-minimum[i]));
}
export function layoutKey(key:string):string {
  // Keep account isolation, share widths between that account's projects.
  const parts=key.split(':');
  return 'layout.panels.'+(parts[0]==='server'?`server:${parts[1]}:`:'local:')+parts.at(-1);
}
