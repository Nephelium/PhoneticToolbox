// Live time view, expressed relative to the newest sample so playback follows
// the same window after zooming. Both plots use this single transform.
export function zoomMonitor(view,delta,fraction){
  const anchor=Math.max(0,Math.min(1,fraction)),span=Math.max(.001,Math.min(12,view.span*Math.exp(Math.max(-500,Math.min(500,delta))*.002)));
  return {span,lag:Math.max(0,Math.min(12-span,view.lag+(view.span-span)*(1-anchor)))};
}
export function monitorTimeLabel(time,span){
  const digits=Math.max(1,Math.min(6,Math.ceil(-Math.log10(span/4))+1));
  return time.toFixed(digits);
}
export function waveformLimit(peak,automatic=true){
  if(!automatic||!Number.isFinite(peak)||peak<=0)return 1;
  const value=Math.max(1e-6,peak*1.1),power=10**Math.floor(Math.log10(value));
  return Math.min(1,[1,2,5,10].find(n=>n*power>=value)*power);
}
export function monitorRange(view,end){
  const start=Math.max(0,end-view.lag-view.span);
  return {start,end:start+view.span,seconds:view.span};
}
