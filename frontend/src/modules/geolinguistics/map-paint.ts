export interface MapPaint {fill:string;stroke:string;width:number;dash?:number[]}

/** Administrative layers provide the internal boundaries for the current scale. */
export function contextBoundaryVisible(boundaryType?:string):boolean{return boundaryType!=='省界';}

/** Territory area precedes linework; boundary symbols stay above both. */
export function mapPaintPriority(type:string,role?:string):number{
 if(role==='boundary-symbol')return 2;
 return type==='LineString'||type==='MultiLineString'||role==='boundary-line'||role==='disputed-line'?1:0;
}

/** Open linework never receives area fill, including in SVG raster exports. */
export function mapPaint(layer:number,type:string,color?:string,role?:string,line?:string):MapPaint{
 const area=type==='Polygon'||type==='MultiPolygon';
 if(layer===0)return {fill:area?'#edf0ed':'none',stroke:'#bcc7c9',width:.6};
 if(layer===1){
  if(role==='boundary-symbol')return {fill:area?'#526c78':'none',stroke:'#526c78',width:.7};
  if(role==='disputed-line')return {fill:'none',stroke:'#526c78',width:1.2,dash:[4,3]};
  if(line==='dash')return {fill:'none',stroke:'#526c78',width:1.2};
  return {fill:area?'#f8f8f1':'none',stroke:'#74878c',width:.8};
 }
 if(layer===2)return {fill:area&&color?color+'38':'none',stroke:color??'#9aa9aa',width:color?1.2:.6};
 return {fill:area?'#397da525':'none',stroke:'#28577b',width:2,dash:[6,4]};
}
