export interface Rect {left:number;top:number;right:number;bottom:number}
export interface HoverTarget {entry:import('./types.ts').SymbolEntry;target:HTMLElement;pointer:{x:number;y:number}}

// Work in viewport pixels, then the caller converts to its scaled container.
export function placeHover(anchor:Rect,pointer:{x:number;y:number},bounds:Rect,scale=1,naturalHeight=520*scale){
 const gap=12*scale,margin=8*scale,minWidth=240*scale;
 const area={left:bounds.left+margin,top:bounds.top+margin,right:bounds.right-margin,bottom:bounds.bottom-margin};
 const avoid={left:Math.min(anchor.left,pointer.x-8*scale),right:Math.max(anchor.right,pointer.x+8*scale),top:Math.min(anchor.top,pointer.y-8*scale),bottom:Math.max(anchor.bottom,pointer.y+8*scale)};
 const desiredWidth=Math.min(420*scale,area.right-area.left),desiredHeight=Math.min(naturalHeight,520*scale,area.bottom-area.top);
 const right=area.right-avoid.right-gap,left=avoid.left-gap-area.left;
 let x:number,y:number,width=desiredWidth,height=desiredHeight;
 const clamp=(v:number,lo:number,hi:number)=>Math.max(lo,Math.min(v,hi));
 if(Math.max(right,left)>=Math.min(minWidth,desiredWidth)){
  const useRight=right>=Math.min(desiredWidth,minWidth)||right>=left;
  width=Math.min(desiredWidth,useRight?right:left);x=useRight?avoid.right+gap:avoid.left-gap-width;
  y=clamp(pointer.y-24*scale,area.top,area.bottom-height);
 }else{
  const below=area.bottom-avoid.bottom-gap,above=avoid.top-gap-area.top,useBelow=below>=Math.min(200*scale,desiredHeight)||below>=above;
  height=Math.min(desiredHeight,Math.max(1,useBelow?below:above));
  x=clamp(pointer.x-24*scale,area.left,area.right-width);y=useBelow?avoid.bottom+gap:avoid.top-gap-height;
 }
 return {left:x,top:y,width,maxHeight:height};
}
