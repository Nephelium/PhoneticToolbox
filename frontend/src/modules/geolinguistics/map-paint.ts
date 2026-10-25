export interface MapPaint {fill:string;stroke:string;width:number;dash?:number[]}

/** Administrative layers provide the internal boundaries for the current scale. */
export function contextBoundaryVisible(boundaryType?:string):boolean{return boundaryType!=='省界';}

/** Only the active administrative scale supplies land and coastline. */
export function contextGeometryVisible(type:string,role?:string):boolean{
 return type!=='Polygon'&&type!=='MultiPolygon'||role==='boundary-symbol';
}

/** Flatten the historical 0x38 overlay onto neutral land, so another base
 * polygon cannot show through the current scale's administrative faces. */
export function administrativeFill(color?:string):string{
 const land='#f8f8f1';if(!color)return land;
 return '#'+[1,3,5].map(i=>Math.round(parseInt(color.slice(i,i+2),16)*56/255+parseInt(land.slice(i,i+2),16)*199/255).toString(16).padStart(2,'0')).join('');
}

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
 if(layer===2)return {fill:area?administrativeFill(color):'none',stroke:color??'#9aa9aa',width:color?1.2:.6};
 return {fill:area?'#397da525':'none',stroke:'#28577b',width:2,dash:[6,4]};
}
