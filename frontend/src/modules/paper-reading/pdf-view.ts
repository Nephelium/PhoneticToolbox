// Raster pixels follow both the monitor and the application's CSS page scale.
export const MIN_PDF_ZOOM=25,MAX_PDF_ZOOM=300;
export function rasterWidth(cssWidth:number,deviceRatio:number,uiPercent=100){
 const density=Math.max(1.25,(Number.isFinite(deviceRatio)?deviceRatio:1)*uiPercent/100);
 return Math.min(4096,Math.max(400,Math.ceil(cssWidth*density)));
}
export function wheelZoom(percent:number,delta:number,mode:number,viewportHeight:number){
 const pixels=delta*(mode===1?16:mode===2?viewportHeight:1);
 return Math.max(MIN_PDF_ZOOM,Math.min(MAX_PDF_ZOOM,Math.round(percent*Math.exp(-Math.max(-120,Math.min(120,pixels))*.0015))));
}
