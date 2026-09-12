// Display navigation only. Scientific calculation stays in the owned worker.
export function rangeAfterGesture(start:number,end:number,duration:number,kind:'zoom'|'pan',value:number):[number,number] {
  const span=Math.min(duration,Math.max(Math.min(.01,duration),kind==='zoom'?(end-start)*value:end-start));
  const wanted=kind==='zoom'?(start+end-span)/2:start-value;
  const first=Math.max(0,Math.min(duration-span,wanted));
  return [first,first+span];
}
export function microAfterGesture(center:number,width:number,duration:number,kind:'zoom'|'pan',value:number){
  return {center:Math.max(0,Math.min(duration,kind==='pan'?center-value/1000:center)),
    width:Math.max(5,Math.min(5000,kind==='zoom'?width*value:width))};
}
