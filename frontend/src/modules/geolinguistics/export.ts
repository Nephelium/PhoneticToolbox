/** Assemble the actual map panels without losing their individual legends. */
export function wrapMapText(text:string,width:number,measure:(text:string)=>number):string[]{
 const lines:string[]=[];let line='';
 // A combining nasalization, syllabicity or tone mark must stay with its base.
 const graphemes=new Intl.Segmenter('zh',{granularity:'grapheme'});
 const words=new Intl.Segmenter('zh',{granularity:'word'}).segment(text);
 const segments=Array.from(words,({segment})=>measure(segment)>width?Array.from(graphemes.segment(segment),g=>g.segment):[segment]).flat();
 for(const segment of segments){
  if(segment==='\n'){lines.push(line);line='';continue;}
  if(line&&measure(line+segment)>width){lines.push(line);line='';}
  line+=segment;
 }
 lines.push(line);return lines;
}

export function composeMapSvgs(panels:string[]):string{
 if(!panels.length||panels.length>4)throw Error('请选择一到四幅地图。');
 if(panels.length===1)return panels[0];
 const sizes=panels.map(svg=>{const size=svg.match(/<svg[^>]+width="([\d.]+)" height="([\d.]+)"/);if(!size)throw Error('地图尺寸无效。');return [+size[1],+size[2]];});
 const cellWidth=Math.max(...sizes.map(v=>v[0])),cellHeight=Math.max(...sizes.map(v=>v[1])),cols=2,rows=Math.ceil(panels.length/cols),gap=20;
 return `<svg xmlns="http://www.w3.org/2000/svg" width="${cols*cellWidth+gap}" height="${rows*cellHeight+(rows-1)*gap}"><rect width="100%" height="100%" fill="white"/>`+panels.map((svg,i)=>svg.replace(/<svg\b([^>]*)>/,(_,attributes:string)=>{
  // Pad the panel viewport rather than scaling its map to a longer neighbour's footer.
  const retained=attributes.replace(/\b(?:width|height|viewBox|x|y)="[^"]*"/g,'').trim();
  return `<svg ${retained} x="${(i%cols)*(cellWidth+gap)}" y="${Math.floor(i/cols)*(cellHeight+gap)}" width="${cellWidth}" height="${cellHeight}" viewBox="0 0 ${cellWidth} ${cellHeight}">`;
 }).replaceAll('id="frame"',`id="frame${i}"`).replaceAll('url(#frame)',`url(#frame${i})`)).join('')+'</svg>';
}

/** Zero-centred red / white / blue, with a fixed full-dataset symmetric range. */
export function dimensionColor(value:number,range:number[]){
 const t=Math.max(-1,Math.min(1,value/(Math.max(Math.abs(range[0]),Math.abs(range[1]))||1)));
 const c=Math.round(255*(1-Math.abs(t))).toString(16).padStart(2,'0');
 return t<0?'#ff'+c+c:'#'+c+c+'ff';
}
